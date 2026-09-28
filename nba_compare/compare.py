"""
Turns game-level rows into a stat block per span, then assembles
N-way comparisons across spans (players, years, or mixed).
"""
from __future__ import annotations
from collections.abc import Mapping
import pandas as pd
from .data import NBADataStore, COUNTING_STATS, TEAM_BOX_STATS
from .models import PlayerSpan, DuoSpan
from . import playoffs as _playoffs
from . import percentiles as _percentiles

# Stats we compute per-game standard deviation / coefficient-of-variation for,
# as a rough "consistency" read: how much a player's game-to-game output swings.
# Kept in sync with table.STAT_DEFS's *_CV%/*_Floor rows -- add a stat in both places.
CONSISTENCY_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGM", "FTM"]

# 1979-80, the first season with a 3-point line (SEASON is the start year).
FIRST_THREE_POINT_SEASON = 1979


def _any_estimated(rows: pd.DataFrame) -> bool:
    """
    True if any of these rows had team box-score values rebuilt from summed
    player lines instead of taken from the official team logs (see
    data._fill_missing_team_stats) -- what the comparison table marks with
    an asterisk.

    Always asked of the SUBSET a stat was actually computed from, never the
    whole span: a pre-1977 row can carry TEAM_EST from a filled PTS while
    still having no turnovers and so being dropped from every possession
    calc, and flagging a number as estimated on the strength of a row that
    didn't feed into it would be a lie in the other direction.
    """
    if "TEAM_EST" not in rows.columns:
        return False
    return bool(rows["TEAM_EST"].fillna(False).astype(bool).any())


def _compute_per_100(games_with_team: pd.DataFrame) -> dict | None:
    """
    Every counting stat per 100 team possessions the player was on the floor
    for -- Basketball-Reference's "Per 100 Poss" definition.

    A box score doesn't record how many possessions a player was out there
    for, so it's the standard estimate: take the team's possessions over
    these games and prorate by the share of the team's floor time he
    occupied.

        team_poss   = FGA - OREB + TOV + 0.44*FTA   (summed over the games)
        player_poss = player_MIN * team_poss / (team_MIN / 5)
        per_100     = 100 * stat / player_poss

    team_MIN/5 turns the team's ~240 player-minutes into minutes of game
    clock, so team_poss / (team_MIN/5) is possessions per game minute --
    this team's ACTUAL measured pace over these exact games, not a league
    constant and not an era assumption. The only assumption is that the team
    ran at its full-game pace while this player was on the floor, which is
    the same one Basketball-Reference makes, and it's the whole reason
    per-100 says something per-36 can't: per-36 is blind to pace, so it
    reads a 1962 possession torrent and a 1999 rock fight as the same
    playing time.

    Totals are summed over the SAME subset of games as the possessions, so a
    span where half the games lack team data reports the honest per-100 rate
    of the half that has it, rather than full-span production divided by
    half-span possessions. Returns None when no game has usable team
    columns -- every season before 1977, where turnovers simply weren't
    recorded and there is no possession estimate to be made from any source.
    """
    valid = games_with_team.dropna(subset=["TEAM_MIN", "TEAM_FGA", "TEAM_FTA", "TEAM_TOV", "TEAM_OREB"])
    if valid.empty:
        return None

    team_poss = (valid["TEAM_FGA"] - valid["TEAM_OREB"] + valid["TEAM_TOV"]
                 + 0.44 * valid["TEAM_FTA"]).sum()
    team_game_min = valid["TEAM_MIN"].sum() / 5
    player_min = valid["MIN_NUM"].sum()
    if not team_poss or not team_game_min or not player_min:
        return None

    player_poss = player_min * team_poss / team_game_min
    if not player_poss:
        return None

    n = len(valid)
    return {
        "per_100": {c: 100 * valid[c].sum() / player_poss for c in COUNTING_STATS},
        "player_poss": player_poss,
        "player_poss_per_game": player_poss / n,
        "games_with_poss_data": n,
        "games_missing_poss_data": len(games_with_team) - n,
        "estimated": _any_estimated(valid),
    }


def _compute_usage(games_with_team: pd.DataFrame) -> dict | None:
    """
    Usage% aggregated across the whole span the same way Basketball-Reference
    does it for multi-game spans: sum the raw components across all games,
    then apply the formula once -- NOT an average of per-game percentages,
    which would overweight garbage-time/low-minute games.

    USG% = 100 * (player_FGA + .44*player_FTA + player_TOV) * (team_MIN/5)
                 / (player_MIN * (team_FGA + .44*team_FTA + team_TOV))

    Returns None if no games have matching team data (e.g. team parquet
    missing that GAME_ID/TEAM_ID combo).
    """
    valid = games_with_team.dropna(subset=["TEAM_MIN", "TEAM_FGA", "TEAM_FTA", "TEAM_TOV"])
    if valid.empty:
        return None

    player_min = valid["MIN_NUM"].sum()
    player_usage_events = (valid["FGA"] + 0.44 * valid["FTA"] + valid["TOV"]).sum()
    team_min = valid["TEAM_MIN"].sum()
    team_usage_events = (valid["TEAM_FGA"] + 0.44 * valid["TEAM_FTA"] + valid["TEAM_TOV"]).sum()

    usg_pct = None
    if player_min and team_usage_events:
        usg_pct = 100 * (player_usage_events * (team_min / 5)) / (player_min * team_usage_events)

    # For a DUO's combined games (see NBADataStore.games_together), MIN_NUM/
    # FGA/FTA/TOV above are the two players' SUMMED totals -- fine for every
    # other stat here (linear), but USG% has minutes in the denominator, so
    # summing minutes first biases the ratio instead of just adding the two
    # players' own shares. games_together() stashes each player's own
    # MIN_NUM/FGA/FTA/TOV (over the same games) under A_*/B_* columns
    # specifically so it can be computed correctly per player and added --
    # which IS how USG% combines (it's each player's share of team plays,
    # and shares simply add).
    if "A_MIN_NUM" in valid.columns:
        usg_a = _single_usg_pct(valid, "A")
        usg_b = _single_usg_pct(valid, "B")
        if usg_a is not None and usg_b is not None:
            usg_pct = usg_a + usg_b

    # MIN% -- share of the team's total floor time this player occupied.
    # Same shape of calc as USG% but simpler: no possession-event estimate,
    # just player minutes over (team minutes / 5), aggregated across the span.
    min_pct = 100 * player_min / (team_min / 5) if team_min else None

    n = len(valid)
    return {
        "usg_pct": usg_pct,
        "usage_per_game": player_usage_events / n if n else None,  # raw plays used (FGA+.44FTA+TOV), PER GAME
        "total_usage": player_usage_events,  # full-span total, kept for reference/custom formulas
        "min_pct": min_pct,
        "games_with_team_data": n,
        "games_missing_team_data": len(games_with_team) - n,
        "estimated": _any_estimated(valid),
    }


def _single_usg_pct(valid: pd.DataFrame, prefix: str) -> float | None:
    """USG% for one half of a duo, from that player's own A_*/B_* minutes
    and events (see NBADataStore.games_together) against the shared
    TEAM_* totals -- same formula as the single-player case in
    _compute_usage, just scoped to one player's own numbers."""
    player_min = valid[f"{prefix}_MIN_NUM"].sum()
    player_events = (valid[f"{prefix}_FGA"] + 0.44 * valid[f"{prefix}_FTA"] + valid[f"{prefix}_TOV"]).sum()
    team_min = valid["TEAM_MIN"].sum()
    team_events = (valid["TEAM_FGA"] + 0.44 * valid["TEAM_FTA"] + valid["TEAM_TOV"]).sum()
    if not player_min or not team_events:
        return None
    return 100 * (player_events * (team_min / 5)) / (player_min * team_events)


def _compute_team_context(games_with_team: pd.DataFrame, game_minutes: float = 48) -> dict | None:
    """
    Team-level context for the games in this span: scoring pace, offensive
    rating, and defensive rating. Possessions are a standard single-team
    estimate (FGA - OREB + TOV + .44*FTA) applied separately to the team's
    own box line (for ORtg) and the opponent's box line (for DRtg) --
    not a full two-team pace formula, but the right idea for each side.

    ORtg = 100 * team points / team possessions
    DRtg = 100 * opponent points / opponent possessions (points allowed per
           100 opponent possessions -- lower is better defense)

    Pace uses the standard NBA formula (both sides' possessions, normalized
    to one regulation game -- game_minutes, 48 in the NBA and 40 in the
    WNBA -- via team minutes played, which is why it needs TEAM_MIN):
      Pace = game_minutes * ((team_poss + opp_poss) / (2 * (team_MIN / 5)))

    Margin of victory (MOV) is the plain, un-pace-adjusted average scoring
    margin (team points - opponent points, per game) -- distinct from Net
    Rtg, which is the same idea but normalized to per-100-possessions so
    it's comparable across different paces. MOV only needs TEAM_PTS/
    OPP_PTS, so it's computed off its own (looser) subset rather than
    piggybacking on the fuller ORtg/DRtg possession-column requirements.
    """
    valid = games_with_team.dropna(subset=["TEAM_PTS", "TEAM_FGA", "TEAM_OREB", "TEAM_FTA", "TEAM_TOV"])
    if valid.empty:
        return None
    n = len(valid)
    team_poss = (valid["TEAM_FGA"] - valid["TEAM_OREB"] + valid["TEAM_TOV"] + 0.44 * valid["TEAM_FTA"]).sum()
    team_pts = valid["TEAM_PTS"].sum()

    result = {
        "team_pts_per_game": team_pts / n,
        "team_poss_per_game": team_poss / n,
        "team_ortg": 100 * team_pts / team_poss if team_poss else None,
        "team_drtg": None,
        "team_net_rtg": None,
        "team_pace": None,
        "team_mov": None,
    }

    valid_opp = games_with_team.dropna(subset=["OPP_PTS", "OPP_FGA", "OPP_OREB", "OPP_FTA", "OPP_TOV"])
    if not valid_opp.empty:
        opp_poss = (valid_opp["OPP_FGA"] - valid_opp["OPP_OREB"] + valid_opp["OPP_TOV"] + 0.44 * valid_opp["OPP_FTA"]).sum()
        opp_pts = valid_opp["OPP_PTS"].sum()
        result["team_drtg"] = 100 * opp_pts / opp_poss if opp_poss else None

    if result["team_ortg"] is not None and result["team_drtg"] is not None:
        result["team_net_rtg"] = result["team_ortg"] - result["team_drtg"]

    valid_mov = games_with_team.dropna(subset=["TEAM_PTS", "OPP_PTS"])
    if not valid_mov.empty:
        result["team_mov"] = (valid_mov["TEAM_PTS"] - valid_mov["OPP_PTS"]).mean()

    # Whether these team numbers rest on rebuilt box-score lines. Read off
    # `valid` (the possession subset), not valid_mov -- TEAM_PTS/OPP_PTS are
    # present in the official logs for every season back to 1946, so MOV is
    # never an estimate even when everything around it is, and table.py
    # deliberately leaves that row unmarked.
    result["estimated"] = _any_estimated(valid)

    pace_cols = ["TEAM_MIN", "TEAM_FGA", "TEAM_OREB", "TEAM_FTA", "TEAM_TOV",
                 "OPP_FGA", "OPP_OREB", "OPP_FTA", "OPP_TOV"]
    valid_pace = games_with_team.dropna(subset=pace_cols)
    if not valid_pace.empty:
        team_poss_p = (valid_pace["TEAM_FGA"] - valid_pace["TEAM_OREB"] + valid_pace["TEAM_TOV"]
                       + 0.44 * valid_pace["TEAM_FTA"]).sum()
        opp_poss_p = (valid_pace["OPP_FGA"] - valid_pace["OPP_OREB"] + valid_pace["OPP_TOV"]
                      + 0.44 * valid_pace["OPP_FTA"]).sum()
        team_min_p = valid_pace["TEAM_MIN"].sum()
        if team_min_p:
            result["team_pace"] = game_minutes * ((team_poss_p + opp_poss_p) / (2 * (team_min_p / 5)))

    return result


def _compute_team_box(games_with_team: pd.DataFrame) -> dict | None:
    """
    The team's and the opponent's full box score, per game, over the games
    in this span: {"TEAM_MIN": ..., "TEAM_FGM": ..., "OPP_REB": ..., ...}.
    Raw material for the Team/Opp rows and for formulas like AST% or TRB%
    that need the whole floor's numbers, not just the player's.

    Each column is averaged over the games that actually have it, not the
    whole span -- a 1983 span keeps its official team PTS for every game
    even where the rebuilt FGM covers only some of them. "estimated" is
    per column for the same reason: a column is marked only if a rebuilt
    value went into THAT column's average.
    """
    if "TEAM_MIN" not in games_with_team.columns:
        return None
    # No 3-point line before 1979-80: those logs leave 3PM/3PA blank, but the
    # true count is zero -- and a blank Opp 3PA would wipe out BLK% (which
    # needs opponent 2-point tries) for the 1973-79 seasons that DO have
    # blocks. Player 3PM/3PA already come out as 0 there, since they're sums.
    if "SEASON" in games_with_team.columns:
        pre_three = games_with_team["SEASON"] < FIRST_THREE_POINT_SEASON
        if pre_three.any():
            games_with_team = games_with_team.copy()
            for col in ("TEAM_FG3M", "TEAM_FG3A", "OPP_FG3M", "OPP_FG3A"):
                games_with_team.loc[pre_three, col] = games_with_team.loc[pre_three, col].fillna(0)
    result, estimated = {}, {}
    for col in ["TEAM_MIN"] + [f"{p}_{c}" for p in ("TEAM", "OPP") for c in TEAM_BOX_STATS]:
        vals = games_with_team[col].dropna()
        if col == "TEAM_MIN":
            # Much of the 1960s logs record team minutes as 0 -- missing,
            # not a real value, and averaging it in would sink the mean.
            vals = vals[vals > 0]
        result[col] = vals.mean() if not vals.empty else None
        est_col = f"{col}_EST"
        estimated[col] = (
            bool(games_with_team.loc[vals.index, est_col].fillna(False).astype(bool).any())
            if est_col in games_with_team.columns and not vals.empty else False
        )
    result["estimated"] = estimated
    return result


def _stat_with_floor(vals: pd.Series, floor_q: float = 0.10) -> dict:
    """
    mean/std/cv_pct plus a "floor" = the floor_q percentile of the game log
    (default 10th percentile). Not the true minimum -- one fluky game (early
    exit, garbage-time DNP-ish line) shouldn't define "what to expect on a
    bad night." The 10th percentile means "about 1 game in 10 is this bad
    or worse," which is a much more usable idea of a realistic floor.
    """
    vals = vals.dropna()
    if vals.empty:
        return {"mean": None, "std": None, "cv_pct": None, "floor": None}
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
    cv_pct = (std / mean * 100) if mean else None
    floor = vals.quantile(floor_q)
    return {"mean": mean, "std": std, "cv_pct": cv_pct, "floor": floor}


def _compute_consistency(games: pd.DataFrame) -> dict:
    """
    Per-game std dev and coefficient of variation (std/mean, as a %) for a
    handful of counting stats, PLUS three derived per-game series:
    TSA (true shot attempts = FGA + .44*FTA, i.e. usage without turnovers),
    USG_EVENTS (TSA + TOV, the full usage-event count), and TS_PCT (true
    shooting %, game by game -- NOT the same number as the span-aggregate
    TS% shown elsewhere, since this is the variability of the per-game rate,
    not a totals-based aggregate).

    CV lets you compare consistency across players/spans regardless of
    scoring level -- a 20% CV means the same relative swing whether someone
    averages 12 or 30 a game. CV is not computed for stats that can be
    ~zero or swing negative (see plus_minus_std in _stat_block instead).

    Each stat also gets a "floor" -- see _stat_with_floor.
    """
    result = {stat: _stat_with_floor(games[stat]) for stat in CONSISTENCY_STATS}

    tsa_game = games["FGA"] + 0.44 * games["FTA"]
    usage_game = tsa_game + games["TOV"]
    ts_pct_game = (games["PTS"] / (2 * tsa_game)).replace([float("inf"), float("-inf")], None)

    for key, vals in [("TSA", tsa_game), ("USG_EVENTS", usage_game), ("TS_PCT", ts_pct_game),
                       ("MIN", games["MIN_NUM"])]:
        result[key] = _stat_with_floor(vals)

    return result


def _stat_block(games: pd.DataFrame, game_minutes: float = 48) -> dict | None:
    """One span's stats for one season-type (regular or playoffs). None if no
    games played. game_minutes is the league's regulation game length (see
    NBADataStore.game_minutes), which pace is normalized to."""
    if games.empty:
        return None

    gp = len(games)
    minutes = games["MIN_NUM"].sum()
    minutes_per_game = minutes / gp
    totals = {c: games[c].sum() for c in COUNTING_STATS}

    per_game = {c: totals[c] / gp for c in COUNTING_STATS}
    per_36 = {c: (totals[c] / minutes) * 36 if minutes else 0.0 for c in COUNTING_STATS}

    fga, fgm, fg3m, fta, ftm = totals["FGA"], totals["FGM"], totals["FG3M"], totals["FTA"], totals["FTM"]
    shooting = {
        "FG_PCT": fgm / fga if fga else None,
        "FG3_PCT": totals["FG3M"] / totals["FG3A"] if totals["FG3A"] else None,
        "FT_PCT": ftm / fta if fta else None,
        "EFG_PCT": (fgm + 0.5 * fg3m) / fga if fga else None,
        # true shooting: points per shooting possession
        "TS_PCT": totals["PTS"] / (2 * (fga + 0.44 * fta)) if (fga or fta) else None,
    }

    wins = int((games["WL"] == "W").sum()) if "WL" in games.columns else None
    losses = int((games["WL"] == "L").sum()) if "WL" in games.columns else None

    # True shot attempts = FGA + .44*FTA ("usage without the turnovers").
    # Computed from totals, not averaged per-game -- same result since it's
    # linear, and doesn't need team data the way usage%/USG Vol does.
    tsa_per_game = (totals["FGA"] + 0.44 * totals["FTA"]) / gp

    # Plus-minus can be ~zero or negative, which makes CV% (std/mean) blow up
    # or flip sign nonsensically -- so this is a raw std dev, not a %.
    plus_minus_std = games["PLUS_MINUS"].std(ddof=1) if "PLUS_MINUS" in games.columns and gp > 1 else None

    # usage% and team context need TEAM_* columns -- only present if this df
    # came from games_with_team_context(). Fall back to None otherwise.
    usage = _compute_usage(games) if "TEAM_MIN" in games.columns else None
    team = _compute_team_context(games, game_minutes) if "TEAM_PTS" in games.columns else None
    possessions = _compute_per_100(games) if "TEAM_MIN" in games.columns else None
    team_box = _compute_team_box(games)

    return {
        "games": gp,
        "minutes_total": minutes,
        "minutes_per_game": minutes_per_game,
        "wins": wins,
        "losses": losses,
        "totals": totals,
        "per_game": per_game,
        "per_36": per_36,
        # Per-100 is None (not zeros) whenever no game in the span has the
        # team possession columns -- i.e. anything before 1977. Callers must
        # treat it as optional the way they already do "usage"/"team".
        "per_100": possessions["per_100"] if possessions else None,
        "possessions": possessions,
        "shooting": shooting,
        "tsa_per_game": tsa_per_game,
        "usage": usage,
        "team": team,
        "team_box": team_box,
        "consistency": _compute_consistency(games),
        "plus_minus_per_game": games["PLUS_MINUS"].mean() if "PLUS_MINUS" in games.columns else None,
        "plus_minus_std": plus_minus_std,
    }


def _relative_shooting(store: NBADataStore, games: pd.DataFrame, shooting: dict, season_type: str) -> dict:
    """
    Each shooting rate in `shooting` minus the games-weighted league
    average for the same season(s) (see percentiles.span_league_average)
    -- e.g. rTS_PCT = +0.032 means 3.2 percentage points better than
    league average across these exact seasons. Weighted by how many games
    THIS span actually has in each season (from `games` itself), so a
    Duo's relative shooting is weighted toward the seasons/games they
    actually shared, not the league's own game counts.

    Unlike percentiles, this is meaningful for a Duo too -- it's plain
    subtraction against a league baseline, not a rank against a
    single-player distribution.
    """
    stats = _percentiles.RELATIVE_SHOOTING_STATS
    if games.empty or "SEASON" not in games.columns:
        return {stat: None for stat in stats}
    games_by_season = games.groupby("SEASON").size().to_dict()
    league = _percentiles.span_league_average(store, games_by_season, season_type)
    return {
        stat: (shooting[stat] - league[stat])
        if shooting.get(stat) is not None and league.get(stat) is not None else None
        for stat in stats
    }


def _regular_season_seed(store: NBADataStore, games: pd.DataFrame) -> float | None:
    """
    Average approximate conference seed (see playoffs.estimate_conference_seed)
    across every season in `games`, using whichever team the most of that
    season's games belong to (handles a mid-season trade the same way
    playoffs.compute_series_records() does -- pick the dominant team for
    the season, don't try to average across teams). Regular season only --
    a team's win%-rank standing exists whether or not this player (or duo)
    made the playoffs that year. Seasons where the seed can't be
    determined (conference/data missing) are excluded rather than counted
    as 0, and the whole thing is None if there's nothing to average.
    """
    if games.empty or "TEAM_ABBREVIATION" not in games.columns:
        return None
    seeds = []
    for season, g in games.groupby("SEASON"):
        team_abbr = g["TEAM_ABBREVIATION"].value_counts().idxmax()
        info = _playoffs.estimate_conference_seed(store, team_abbr, int(season))
        if info is not None:
            seeds.append(info["estimated_seed"])
    return sum(seeds) / len(seeds) if seeds else None


def aggregate_span(span: PlayerSpan, store: NBADataStore) -> dict:
    """
    A head-to-head span (span.vs_player_ids set) is built from only the
    games against that player, and differs in two ways:
    - no "percentiles" -- those rank whole seasons against the league and
      say nothing about these particular games, so they'd read as if they
      were head-to-head numbers when they aren't.
    - series records keep only the series actually played against the
      opponent, with W/L from those games. The team's other series that
      postseason would otherwise show up as "(DNP)" and count as series
      missed to injury.
    """
    h2h = bool(span.vs_player_ids)

    def games(season_type: str) -> pd.DataFrame:
        if h2h:
            return store.games_head_to_head(span.player_ids, span.vs_player_ids, span.seasons, season_type)
        return store.games_with_team_context(span.player_id, span.seasons, season_type)

    result = {"span": span, "label": span.label, "regular": None, "playoffs": None}
    if span.include_regular:
        reg_games = games("regular")
        result["regular"] = _stat_block(reg_games, store.game_minutes)
        if result["regular"] is not None:
            if not h2h:
                result["regular"]["percentiles"] = _percentiles.span_percentiles(store, span, "regular")
            result["regular"]["avg_seed"] = _regular_season_seed(store, reg_games)
            result["regular"]["relative_shooting"] = _relative_shooting(
                store, reg_games, result["regular"]["shooting"], "regular"
            )
    if span.include_playoffs:
        po_games = games("playoffs")
        result["playoffs"] = _stat_block(po_games, store.game_minutes)
        if result["playoffs"] is not None:
            result["playoffs"]["relative_shooting"] = _relative_shooting(
                store, po_games, result["playoffs"]["shooting"], "playoffs"
            )
            if h2h:
                records = _playoffs.compute_series_records(store, po_games, wl_from_player_games=True)
                records = [r for r in records if r["player_played"]]
            else:
                result["playoffs"]["percentiles"] = _percentiles.span_percentiles(store, span, "playoffs")
                raw_playoff_games = store.games(span.player_id, span.seasons, "playoffs")
                records = _playoffs.compute_series_records(store, raw_playoff_games)
            result["playoffs"]["series_records"] = records
            result["playoffs"]["depth"] = _playoffs.depth_summary(records)
    return result


def aggregate_duo_span(duo: DuoSpan, store: NBADataStore) -> dict:
    """
    Same shape of result as aggregate_span(), but built from the two
    players' COMBINED numbers for games they shared as teammates (see
    NBADataStore.games_together) instead of one player's games. No
    "percentiles" key -- league percentiles are a single-player-distribution
    concept and a duo's combined per-game value would trivially read near
    the 100th percentile, so it's omitted rather than shown misleadingly
    (table.py's getters already treat a missing percentiles key as "--").

    A head-to-head duo (duo.vs_player_ids set) keeps only the games against
    those player(s), and its series records only the series played against
    them -- see aggregate_span for why.
    """
    h2h = bool(duo.vs_player_ids)

    def games(season_type: str) -> pd.DataFrame:
        if h2h:
            return store.games_head_to_head(duo.player_ids, duo.vs_player_ids, duo.seasons, season_type)
        return store.games_together(duo.player_a_id, duo.player_b_id, duo.seasons, season_type)

    result = {"span": duo, "label": duo.label, "regular": None, "playoffs": None}
    if duo.include_regular:
        reg_games = games("regular")
        result["regular"] = _stat_block(reg_games, store.game_minutes)
        if result["regular"] is not None:
            result["regular"]["avg_seed"] = _regular_season_seed(store, reg_games)
            result["regular"]["relative_shooting"] = _relative_shooting(
                store, reg_games, result["regular"]["shooting"], "regular"
            )
    if duo.include_playoffs:
        po_games = games("playoffs")
        result["playoffs"] = _stat_block(po_games, store.game_minutes)
        if result["playoffs"] is not None:
            result["playoffs"]["relative_shooting"] = _relative_shooting(
                store, po_games, result["playoffs"]["shooting"], "playoffs"
            )
            records = _playoffs.compute_series_records(store, po_games, wl_from_player_games=True)
            if h2h:
                records = [r for r in records if r["player_played"]]
            result["playoffs"]["series_records"] = records
            result["playoffs"]["depth"] = _playoffs.depth_summary(records)
    return result


def compare_spans(
    spans: list[PlayerSpan | DuoSpan], store: NBADataStore | Mapping[str, NBADataStore]
) -> "ComparisonResult":
    """
    store: one NBADataStore, or {league: store} when the spans may come from
    more than one league -- each span is then read from the store for its
    own span.league, so its percentiles, league averages and playoff
    structure are measured against its own league.
    """
    def store_for(span):
        return store[span.league] if isinstance(store, Mapping) else store

    aggregates = [
        aggregate_duo_span(s, store_for(s)) if isinstance(s, DuoSpan) else aggregate_span(s, store_for(s))
        for s in spans
    ]
    return ComparisonResult(aggregates)


class ComparisonResult:
    """
    Wraps the raw per-span aggregates and exposes convenient tabular views.
    """

    def __init__(self, aggregates: list[dict]):
        self.aggregates = aggregates

    def wide_table(self, season_type: str = "regular", stat_group: str = "per_game") -> pd.DataFrame:
        """
        One row per stat, one column per span. season_type: 'regular'|'playoffs'.
        stat_group: 'per_game'|'per_36'|'per_100'|'totals'|'shooting'.

        An empty column means either no games in the span or no data for
        that group -- 'per_100' is legitimately absent for pre-1977 spans
        (see _compute_per_100), so it's read with .get() rather than [].
        """
        cols = {}
        for agg in self.aggregates:
            block = agg[season_type]
            cols[agg["label"]] = (block.get(stat_group) or {}) if block else {}
        return pd.DataFrame(cols)

    def long_table(self) -> pd.DataFrame:
        """
        Tidy long-format table across BOTH season types and
        per_game/per_36/per_100, for plotting.
        Columns: span, season_type, stat_group, stat, value

        Spans with no per-100 data (pre-1977) contribute no per_100 rows at
        all rather than rows of None, so a plot over this frame simply has
        nothing to draw for them instead of a line at zero.
        """
        rows = []
        for agg in self.aggregates:
            for season_type in ("regular", "playoffs"):
                block = agg[season_type]
                if not block:
                    continue
                for stat_group in ("per_game", "per_36", "per_100", "shooting"):
                    for stat, value in (block.get(stat_group) or {}).items():
                        rows.append({
                            "span": agg["label"],
                            "season_type": season_type,
                            "stat_group": stat_group,
                            "stat": stat,
                            "value": value,
                        })
                rows.append({
                    "span": agg["label"], "season_type": season_type,
                    "stat_group": "meta", "stat": "games", "value": block["games"],
                })
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Quick games/wins snapshot per span, both season types side by side."""
        rows = []
        for agg in self.aggregates:
            row = {"span": agg["label"]}
            for season_type in ("regular", "playoffs"):
                block = agg[season_type]
                prefix = "RS" if season_type == "regular" else "PO"
                row[f"{prefix}_GP"] = block["games"] if block else 0
                row[f"{prefix}_PPG"] = round(block["per_game"]["PTS"], 1) if block else None
                row[f"{prefix}_TS%"] = round(block["shooting"]["TS_PCT"], 3) if block and block["shooting"]["TS_PCT"] else None
            rows.append(row)
        return pd.DataFrame(rows)