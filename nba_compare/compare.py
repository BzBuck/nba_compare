"""
Turns game-level rows into a stat block per span, then assembles
N-way comparisons across spans (players, years, or mixed).
"""
from __future__ import annotations
import pandas as pd
from .data import NBADataStore, COUNTING_STATS
from .models import PlayerSpan, DuoSpan
from . import playoffs as _playoffs
from . import percentiles as _percentiles

# Stats we compute per-game standard deviation / coefficient-of-variation for,
# as a rough "consistency" read: how much a player's game-to-game output swings.
# Kept in sync with table.STAT_DEFS's *_CV%/*_Floor rows -- add a stat in both places.
CONSISTENCY_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGM", "FTM"]


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


def _compute_team_context(games_with_team: pd.DataFrame) -> dict | None:
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
    to a 48-minute game via team minutes played -- this is why it needed
    TEAM_MIN, which wasn't wired into anything until now):
      Pace = 48 * ((team_poss + opp_poss) / (2 * (team_MIN / 5)))

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
            result["team_pace"] = 48 * ((team_poss_p + opp_poss_p) / (2 * (team_min_p / 5)))

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


def _stat_block(games: pd.DataFrame) -> dict | None:
    """One span's stats for one season-type (regular or playoffs). None if no games played."""
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
    team = _compute_team_context(games) if "TEAM_PTS" in games.columns else None

    return {
        "games": gp,
        "minutes_total": minutes,
        "minutes_per_game": minutes_per_game,
        "wins": wins,
        "losses": losses,
        "totals": totals,
        "per_game": per_game,
        "per_36": per_36,
        "shooting": shooting,
        "tsa_per_game": tsa_per_game,
        "usage": usage,
        "team": team,
        "consistency": _compute_consistency(games),
        "plus_minus_per_game": games["PLUS_MINUS"].mean() if "PLUS_MINUS" in games.columns else None,
        "plus_minus_std": plus_minus_std,
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
    result = {"span": span, "label": span.label, "regular": None, "playoffs": None}
    if span.include_regular:
        reg_games = store.games_with_team_context(span.player_id, span.seasons, "regular")
        result["regular"] = _stat_block(reg_games)
        if result["regular"] is not None:
            result["regular"]["percentiles"] = _percentiles.span_percentiles(store, span, "regular")
            result["regular"]["avg_seed"] = _regular_season_seed(store, reg_games)
    if span.include_playoffs:
        result["playoffs"] = _stat_block(
            store.games_with_team_context(span.player_id, span.seasons, "playoffs")
        )
        if result["playoffs"] is not None:
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
    """
    result = {"span": duo, "label": duo.label, "regular": None, "playoffs": None}
    if duo.include_regular:
        reg_games = store.games_together(duo.player_a_id, duo.player_b_id, duo.seasons, "regular")
        result["regular"] = _stat_block(reg_games)
        if result["regular"] is not None:
            result["regular"]["avg_seed"] = _regular_season_seed(store, reg_games)
    if duo.include_playoffs:
        games = store.games_together(duo.player_a_id, duo.player_b_id, duo.seasons, "playoffs")
        result["playoffs"] = _stat_block(games)
        if result["playoffs"] is not None:
            records = _playoffs.compute_series_records(store, games, wl_from_player_games=True)
            result["playoffs"]["series_records"] = records
            result["playoffs"]["depth"] = _playoffs.depth_summary(records)
    return result


def compare_spans(spans: list[PlayerSpan | DuoSpan], store: NBADataStore) -> "ComparisonResult":
    aggregates = [
        aggregate_duo_span(s, store) if isinstance(s, DuoSpan) else aggregate_span(s, store)
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
        stat_group: 'per_game'|'per_36'|'totals'|'shooting'.
        """
        cols = {}
        for agg in self.aggregates:
            block = agg[season_type]
            cols[agg["label"]] = block[stat_group] if block else {}
        return pd.DataFrame(cols)

    def long_table(self) -> pd.DataFrame:
        """
        Tidy long-format table across BOTH season types and per_game/per_36, for plotting.
        Columns: span, season_type, stat_group, stat, value
        """
        rows = []
        for agg in self.aggregates:
            for season_type in ("regular", "playoffs"):
                block = agg[season_type]
                if not block:
                    continue
                for stat_group in ("per_game", "per_36", "shooting"):
                    for stat, value in block[stat_group].items():
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