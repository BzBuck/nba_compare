"""
Playoff series identification, round/depth tracking, and championship
detection -- derived from game logs, not an external standings source.

How series/rounds are identified:
- Opponent per game comes from the MATCHUP field ("OWN @ OPP" or
  "OWN vs. OPP" -- the NBA stats API always puts the opponent last).
- A "series" is a consecutive run of games against the same opponent for
  one team in one season. Since a team never faces two different opponents
  interleaved within a single postseason, this reliably separates series
  even without an explicit round/series ID in the source data.
- "Round number" is just that team's Nth series chronologically that
  postseason -- correct regardless of how many rounds existed that era.

How championship detection avoids hardcoding "round 4 = Finals":
- league_round_depth() looks at ALL teams' playoff series that season
  (from the team-level parquet, not one player) and finds the actual
  deepest round reached league-wide. A team is champion only if they
  reached that deepest round AND won their final series. This stays
  correct even in historical formats with a different number of rounds,
  instead of assuming the modern 4-round structure.

Seed is a different story -- there's no standings data available, so
estimate_conference_seed() is a clearly-flagged APPROXIMATION (regular
season win% rank within a hardcoded conference table), not a real seed.
See its docstring for exactly what it doesn't account for.
"""
from __future__ import annotations
import pandas as pd
from .data import NBADataStore

STANDARD_ROUND_LABELS = {1: "First Round", 2: "Conf Semis", 3: "Conf Finals", 4: "Finals"}

# Conference mapping including common historical abbreviation aliases
# (relocations/renames). Conference assignment has been very stable since
# the merger era, but pre-1970s/division-based formats aren't represented
# here -- round LABELS and champion detection still work in those eras
# (they don't depend on this table), only estimate_conference_seed() does.
EASTERN_TEAMS = {
    "ATL", "BOS", "BKN", "NJN", "NJA", "CHA", "CHH", "CHO", "BOB", "CHI", "CLE",
    "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS", "WSB",
}
WESTERN_TEAMS = {
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "VAN", "MIN", "NOP",
    "NOH", "NOK", "OKC", "SEA", "PHX", "POR", "SAC", "SAS", "UTA",
}


def _opponent_from_matchup(matchup: str) -> str:
    """NBA API MATCHUP convention: 'OWN @ OPP' or 'OWN vs. OPP' -- opponent
    is always the last whitespace-separated token."""
    return str(matchup).strip().split()[-1]


def _home_court_from_matchup(matchup: str) -> str | None:
    """
    'vs.' means the team named first hosted the game; '@' means they were
    the visitor. Whoever hosted Game 1 of a series has home court
    advantage for the whole series (that's what the term means) -- this
    needs no external schedule/standings data, just the same MATCHUP field
    already used for opponent detection, read from the series' first game.
    """
    m = str(matchup)
    if " vs. " in m or " vs " in m:
        return "Home"
    if " @ " in m:
        return "Away"
    return None


def _label_round(round_num: int, total_rounds: int) -> str:
    if total_rounds == 4:
        return STANDARD_ROUND_LABELS.get(round_num, f"Round {round_num}")
    return "Finals" if round_num == total_rounds else f"Round {round_num}"


def _identify_series_for_group(games: pd.DataFrame) -> pd.DataFrame:
    """games: playoff rows for ONE (season, team) combo. Adds OPPONENT,
    SERIES_ID (local to this group), and ROUND columns."""
    games = games.copy()
    games["OPPONENT"] = games["MATCHUP"].map(_opponent_from_matchup)
    games = games.sort_values("GAME_DATE")

    change = (games["OPPONENT"] != games["OPPONENT"].shift()).cumsum()
    games["SERIES_ID"] = change

    series_order = games.drop_duplicates("SERIES_ID").sort_values("GAME_DATE")
    round_map = {sid: i + 1 for i, sid in enumerate(series_order["SERIES_ID"])}
    games["ROUND"] = games["SERIES_ID"].map(round_map)
    return games


def identify_series(playoff_games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds OPPONENT, SERIES_ID (globally unique across the result), and ROUND
    columns to playoff game rows, inferred from THIS dataframe's own rows.

    CAVEAT: if playoff_games is one player's games, this mis-numbers
    rounds for a player who missed an entire round to injury (their
    games silently skip straight from Round 1 to what looks like
    "Round 2" but was actually Round 3), and drops championship credit
    entirely if they missed the clinching series. compute_series_records()
    below avoids this by using team-level data to establish the true
    round structure first -- prefer that for anything player-facing. This
    function is kept for cases where you deliberately want rounds inferred
    from a specific dataframe's own games (e.g. it's also how
    league_round_depth() analyzes team-level data, where the same-game
    assumption holds since a team's own log has no "missed games").
    """
    if playoff_games.empty:
        return playoff_games.assign(OPPONENT=None, SERIES_ID=None, ROUND=None)

    parts, offset = [], 0
    for _, group in playoff_games.groupby(["SEASON", "TEAM_ID"], sort=True):
        g = _identify_series_for_group(group)
        g["SERIES_ID"] = g["SERIES_ID"] + offset
        offset = int(g["SERIES_ID"].max()) + 1
        parts.append(g)
    return pd.concat(parts).sort_index()


def team_series_structure(store: NBADataStore, team_id: int, season: int) -> pd.DataFrame:
    """The TEAM's full round-by-round series structure that postseason,
    from team-level game logs -- the authoritative source for round
    numbers/opponents/results, since a team's own log doesn't have gaps
    the way one player's log can when they miss games to injury."""
    team_games = store.team_games_for_season(season, "playoffs")
    team_games = team_games[team_games["TEAM_ID"] == team_id]
    if team_games.empty:
        return pd.DataFrame()
    return _identify_series_for_group(team_games)


def identify_series_for_player(store: NBADataStore, player_games: pd.DataFrame) -> list[dict]:
    """
    For each (season, team) the player appears for at all, gets the TEAM's
    actual series/round structure (see team_series_structure -- NOT
    inferred from the player's own games), then matches the player's own
    games into each series by GAME_ID. A round where the player has zero
    matching games still produces an entry (with team_games populated but
    player_games empty), so:
    - later rounds don't get mis-numbered just because an earlier round is
      missing from the player's personal log, and
    - championship/round-reached credit is based on what the TEAM did,
      not gated on the player having box score rows in that specific series.

    Known limitation: this requires the player to have appeared in at
    least one game for that team THAT POSTSEASON to even be attributed any
    of that run (there's no roster data here, only game logs) -- a player
    who missed the entire postseason for a team can't be picked up this
    way. A player who suited up for part of the run but missed a round
    (or the clinching series) to injury is exactly the case this fixes.

    Returns a list of {season, team_id, round_num, opponent, team_games,
    player_games} dicts, one per series.
    """
    if player_games.empty:
        return []

    out = []
    for (season, team_id), pg in player_games.groupby(["SEASON", "TEAM_ID"], sort=True):
        team_series = team_series_structure(store, int(team_id), int(season))
        if team_series.empty:
            continue  # no team-level data available for this team/season -- can't verify, skip
        for series_id, team_g in team_series.groupby("SERIES_ID"):
            round_num = int(team_g["ROUND"].iloc[0])
            opponent = team_g["OPPONENT"].iloc[0]
            player_g = pg[pg["GAME_ID"].isin(team_g["GAME_ID"])]
            out.append({
                "season": int(season), "team_id": int(team_id), "round_num": round_num,
                "opponent": opponent, "team_games": team_g, "player_games": player_g,
            })
    return out


def _empty_series_stat_block() -> dict:
    """Placeholder stats for a series the player didn't appear in at all
    (missed to injury) -- GP=0, everything else None/0 rather than
    omitted, so SERIES_COLUMN_DEFS getters don't need special-casing."""
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "PF", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]
    return {
        "gp": 0,
        "per_game": {c: None for c in stat_cols},
        "shooting": {k: None for k in ("FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "TS_PCT")},
        "totals": {c: 0 for c in stat_cols},
    }


def league_round_depth(store: NBADataStore, season: int) -> pd.DataFrame:
    """
    Rounds played and final-series result for EVERY team that made the
    playoffs in `season`, using team-level data. Used to find that
    season's true deepest round and champion without assuming a fixed
    4-round structure.
    """
    team_games = store.team_games_for_season(season, "playoffs")
    if team_games.empty:
        return pd.DataFrame(columns=["TEAM_ID", "rounds_played", "won_final_series"])

    rows = []
    for team_id, group in team_games.groupby("TEAM_ID"):
        g = _identify_series_for_group(group)
        max_round = int(g["ROUND"].max())
        final = g[g["ROUND"] == max_round]
        wins, losses = int((final["WL"] == "W").sum()), int((final["WL"] == "L").sum())
        rows.append({"TEAM_ID": team_id, "rounds_played": max_round, "won_final_series": wins > losses})
    return pd.DataFrame(rows)


def season_champion_team_id(store: NBADataStore, season: int) -> int | None:
    """None if the data for that season is missing/ambiguous (won't guess)."""
    depth = league_round_depth(store, season)
    if depth.empty:
        return None
    max_round = depth["rounds_played"].max()
    champs = depth[(depth["rounds_played"] == max_round) & depth["won_final_series"]]
    if len(champs) != 1:
        return None
    return int(champs.iloc[0]["TEAM_ID"])


def _series_stat_block(g: pd.DataFrame) -> dict:
    """Per-series box score numbers -- same shape of calc as compare.py's
    _stat_block, but series-scoped and without team/usage context (not
    meaningful at single-series granularity)."""
    gp = len(g)
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "PF", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]
    totals = {c: g[c].sum() for c in stat_cols if c in g.columns}
    per_game = {c: (totals[c] / gp if gp else None) for c in totals}
    fga, fgm, fg3m, fta, ftm = (totals.get(k, 0) for k in ("FGA", "FGM", "FG3M", "FTA", "FTM"))
    shooting = {
        "FG_PCT": fgm / fga if fga else None,
        "FG3_PCT": totals.get("FG3M", 0) / totals["FG3A"] if totals.get("FG3A") else None,
        "FT_PCT": ftm / fta if fta else None,
        "EFG_PCT": (fgm + 0.5 * fg3m) / fga if fga else None,
        "TS_PCT": totals.get("PTS", 0) / (2 * (fga + 0.44 * fta)) if (fga or fta) else None,
    }
    return {"gp": gp, "per_game": per_game, "shooting": shooting, "totals": totals}


def aggregate_series_group(records: list[dict]) -> dict | None:
    """
    Combines multiple series' raw totals into one aggregate stat block --
    for comparing a span's overall performance in one ROUND across however
    many times they reached it (a span can have multiple Finals runs,
    etc.). Sums totals and recomputes rates from those sums, same
    sum-then-divide approach used everywhere else in this project, rather
    than naively averaging each series' rate (which would overweight
    short series).

    Returns the same shape as a series record's relevant fields (w, l,
    stats.gp/per_game/shooting) so SERIES_COLUMN_DEFS getters work
    unmodified on it -- minus home_court/seed, which don't aggregate
    meaningfully across multiple series.
    """
    if not records:
        return None
    combined_totals: dict[str, float] = {}
    for r in records:
        for k, v in r["stats"]["totals"].items():
            combined_totals[k] = combined_totals.get(k, 0) + v
    gp = sum(r["stats"]["gp"] for r in records)
    per_game = {k: (v / gp if gp else None) for k, v in combined_totals.items()}
    fga, fgm, fg3m, fta, ftm = (combined_totals.get(k, 0) for k in ("FGA", "FGM", "FG3M", "FTA", "FTM"))
    shooting = {
        "FG_PCT": fgm / fga if fga else None,
        "FG3_PCT": combined_totals.get("FG3M", 0) / combined_totals["FG3A"] if combined_totals.get("FG3A") else None,
        "FT_PCT": ftm / fta if fta else None,
        "EFG_PCT": (fgm + 0.5 * fg3m) / fga if fga else None,
        "TS_PCT": combined_totals.get("PTS", 0) / (2 * (fga + 0.44 * fta)) if (fga or fta) else None,
    }
    return {
        "w": sum(r["w"] for r in records),
        "l": sum(r["l"] for r in records),
        "stats": {"gp": gp, "per_game": per_game, "shooting": shooting, "totals": combined_totals},
        "series_count": len(records),
    }


_ROUND_ORDER_HINTS = {"First Round": 1, "Conf Semis": 2, "Conf Finals": 3, "Finals": 999}


def _round_sort_key(label: str) -> tuple:
    """Finals always sorts last; standard/generic round labels sort by
    depth; anything unrecognized sorts after, alphabetically."""
    if label in _ROUND_ORDER_HINTS:
        return (0, _ROUND_ORDER_HINTS[label])
    if label.startswith("Round "):
        try:
            return (0, int(label.split()[-1]))
        except ValueError:
            pass
    return (1, label)


def round_labels_present(spans_records: dict) -> list[str]:
    """All distinct round labels across every span's records, in display order."""
    labels = {r["Round"] for records in spans_records.values() for r in records}
    return sorted(labels, key=_round_sort_key)


def build_round_comparison_table(
    spans_records: dict, round_label: str, columns: list[str]
) -> pd.DataFrame:
    """
    Rows = stats (from `columns`, numeric ones only), columns = span
    labels. Each cell is that span's aggregate performance across every
    series they played in `round_label` -- None (renders as "--") if that
    span never reached this round.
    """
    numeric_columns = [c for c in columns if c not in ("Home Court", "Seed (approx)")]
    data = {}
    for span_label, records in spans_records.items():
        round_records = [r for r in records if r["Round"] == round_label]
        agg = aggregate_series_group(round_records)
        col = {}
        for stat in numeric_columns:
            getter = SERIES_COLUMN_DEFS[stat][0]
            try:
                col[stat] = getter(agg) if agg else None
            except Exception:
                col[stat] = None
        data[span_label] = col
    return pd.DataFrame(data)


# label -> (getter(record) -> value, display format or None for a plain
# string column, lower_is_better). Mirrors table.STAT_DEFS's shape/pattern
# on purpose, so the picker UI works the same way as the main stat table.
SERIES_COLUMN_DEFS = {
    "GP":            (lambda r: r["stats"]["gp"],                      "{:.0f}", False),
    "W":             (lambda r: r["w"],                                "{:.0f}", False),
    "L":             (lambda r: r["l"],                                "{:.0f}", True),
    "PTS/G":         (lambda r: r["stats"]["per_game"].get("PTS"),      "{:.1f}", False),
    "TRB/G":         (lambda r: r["stats"]["per_game"].get("REB"),      "{:.1f}", False),
    "AST/G":         (lambda r: r["stats"]["per_game"].get("AST"),      "{:.1f}", False),
    "STL/G":         (lambda r: r["stats"]["per_game"].get("STL"),      "{:.1f}", False),
    "BLK/G":         (lambda r: r["stats"]["per_game"].get("BLK"),      "{:.1f}", False),
    "TOV/G":         (lambda r: r["stats"]["per_game"].get("TOV"),      "{:.1f}", True),
    "PF/G":          (lambda r: r["stats"]["per_game"].get("PF"),       "{:.1f}", True),
    "FG%":           (lambda r: r["stats"]["shooting"]["FG_PCT"],       "{:.3f}", False),
    "3P%":           (lambda r: r["stats"]["shooting"]["FG3_PCT"],      "{:.3f}", False),
    "FT%":           (lambda r: r["stats"]["shooting"]["FT_PCT"],       "{:.3f}", False),
    "eFG%":          (lambda r: r["stats"]["shooting"]["EFG_PCT"],      "{:.3f}", False),
    "TS%":           (lambda r: r["stats"]["shooting"]["TS_PCT"],       "{:.3f}", False),
    "Home Court":    (lambda r: r["home_court"],                        None,     False),
    "Seed (approx)": (lambda r: r["seed"],                             "{:.0f}", False),
}
DEFAULT_SERIES_COLUMNS = ["GP", "W", "L", "PTS/G", "TS%", "Home Court", "Seed (approx)"]


def compute_series_records(
    store: NBADataStore, player_games: pd.DataFrame, wl_from_player_games: bool = False
) -> list[dict]:
    """
    One dict per series the player's TEAM played that postseason -- round,
    opponent, and result come from TEAM-level data (identify_series_for_player),
    not the player's own games, specifically so a round the player missed
    to injury still shows up correctly instead of being skipped (which
    would mis-number every round after it) and still counts toward
    championship credit if the team won it while they were out (stats for
    that series will show 0 games played -- see "player_played").

    This is the full computed data; build_series_table() flattens a
    user-chosen subset of columns from it for display -- kept separate so
    depth_summary() always sees the complete picture regardless of what
    the UI has toggled on.

    wl_from_player_games: when True, each record's "w"/"l" fields (the
    numbers shown in the GP/W/L columns and summed across series by
    aggregate_series_group) come from player_games' OWN WL column instead
    of the team's full series log. Used for DuoSpans, where player_games
    is already restricted to games the duo actually shared at real
    minutes (see NBADataStore.games_together) -- so their personal record
    doesn't get credited or blamed for games one half of the duo didn't
    play in. "Result" and "Champion" always reflect the TEAM's actual
    series outcome regardless of this flag -- those are facts about the
    series itself, not about who was on the floor for it.
    """
    series_infos = identify_series_for_player(store, player_games)
    if not series_infos:
        return []

    records = []
    depth_cache: dict[int, pd.DataFrame] = {}
    champion_cache: dict[int, int | None] = {}

    for info in series_infos:
        season, team_id, round_num, opponent = info["season"], info["team_id"], info["round_num"], info["opponent"]
        team_g, player_g = info["team_games"], info["player_games"]

        if season not in depth_cache:
            depth_cache[season] = league_round_depth(store, season)
            champion_cache[season] = season_champion_team_id(store, season)
        league_depth = depth_cache[season]
        total_rounds = int(league_depth["rounds_played"].max()) if not league_depth.empty else round_num

        # Result comes from the TEAM's games -- correct regardless of
        # whether the player suited up for any of this series.
        team_wins = int((team_g["WL"] == "W").sum())
        team_losses = int((team_g["WL"] == "L").sum())
        is_champion = (round_num == total_rounds) and (team_wins > team_losses) and (champion_cache[season] == team_id)
        home_court = _home_court_from_matchup(team_g.sort_values("GAME_DATE")["MATCHUP"].iloc[0])

        team_abbr = None
        if not player_g.empty and "TEAM_ABBREVIATION" in player_g.columns:
            team_abbr = player_g["TEAM_ABBREVIATION"].iloc[0]
        elif "TEAM_ABBREVIATION" in team_g.columns:
            team_abbr = team_g["TEAM_ABBREVIATION"].iloc[0]
        seed_info = estimate_conference_seed(store, team_abbr, season) if team_abbr else None

        played = len(player_g)
        if wl_from_player_games:
            record_w = int((player_g["WL"] == "W").sum()) if played else 0
            record_l = int((player_g["WL"] == "L").sum()) if played else 0
        else:
            record_w, record_l = team_wins, team_losses

        records.append({
            "Season": f"{season}-{str(season + 1)[-2:]}",
            "SeasonSort": season,
            "Round": _label_round(round_num, total_rounds),
            "RoundNum": round_num,
            "Opponent": opponent,
            "Result": "Won" if team_wins > team_losses else "Lost",
            "Champion": is_champion,
            "w": record_w,
            "l": record_l,
            "stats": _series_stat_block(player_g) if played else _empty_series_stat_block(),
            "home_court": home_court,
            "seed": seed_info["estimated_seed"] if seed_info else None,
            "player_played": played > 0,
            "team_games_in_series": len(team_g),
        })

    return sorted(records, key=lambda r: (r["SeasonSort"], r["RoundNum"]))


def build_series_table(records: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    """Flattens series records into a display DataFrame: identity columns
    (Season/Round/Opponent/Result/Champion/player_played) always first,
    then whichever SERIES_COLUMN_DEFS columns are requested, in that order."""
    columns = columns if columns is not None else DEFAULT_SERIES_COLUMNS
    rows = []
    for r in records:
        row = {"Season": r["Season"], "Round": r["Round"], "Opponent": r["Opponent"],
               "Result": r["Result"], "Champion": r["Champion"], "player_played": r.get("player_played", True)}
        for col in columns:
            if col not in SERIES_COLUMN_DEFS:
                continue
            getter = SERIES_COLUMN_DEFS[col][0]
            try:
                row[col] = getter(r)
            except Exception:
                row[col] = None
        rows.append(row)
    return pd.DataFrame(rows)


def depth_summary(records: list[dict]) -> dict:
    """Span-level playoff depth metrics from compute_series_records() --
    operates on the full records, not the user-filtered display table, so
    it stays correct no matter which columns are toggled on/off.

    "series_played"/"championships"/etc. count every series the player's
    TEAM played while they were on the roster that postseason, including
    ones they personally missed to injury (see "series_missed_to_injury")
    -- a player hurt for the clinching series still gets championship
    credit for their team winning it, rather than being silently excluded
    for having no box score rows in that specific series.
    """
    if not records:
        return {
            "seasons_in_playoffs": 0, "series_played": 0, "series_w": 0, "series_l": 0,
            "finals_apps": 0, "championships": 0, "best_round_label": None, "best_round_num": 0,
            "series_missed_to_injury": 0,
        }
    best = max(records, key=lambda r: r["RoundNum"])
    seasons = {r["Season"] for r in records}
    return {
        "seasons_in_playoffs": len(seasons),
        "series_played": len(records),
        "series_w": sum(1 for r in records if r["Result"] == "Won"),
        "series_l": sum(1 for r in records if r["Result"] == "Lost"),
        "finals_apps": sum(1 for r in records if r["Round"] == "Finals"),
        "championships": sum(1 for r in records if r["Champion"]),
        "best_round_label": best["Round"],
        "best_round_num": best["RoundNum"],
        "series_missed_to_injury": sum(1 for r in records if not r.get("player_played", True)),
    }


def estimate_conference_seed(store: NBADataStore, team_abbr: str, season: int) -> dict | None:
    """
    APPROXIMATE seed: ranks teams by regular-season win% within a
    hardcoded conference table. This is NOT official seeding -- it doesn't
    apply real NBA tiebreakers (head-to-head, division record, etc.) and
    doesn't account for play-in games (2020-present). Treat the result as
    a rough "how good was this team relative to its conference" signal,
    not an authoritative seed number. Returns None if the team's
    conference/data can't be determined.
    """
    conf = "East" if team_abbr in EASTERN_TEAMS else "West" if team_abbr in WESTERN_TEAMS else None
    if conf is None:
        return None

    reg = store.team_games_for_season(season, "regular")
    if reg.empty or "TEAM_ABBREVIATION" not in reg.columns:
        return None

    conf_teams = EASTERN_TEAMS if conf == "East" else WESTERN_TEAMS
    conf_games = reg[reg["TEAM_ABBREVIATION"].isin(conf_teams)]
    if conf_games.empty:
        return None

    records = conf_games.groupby("TEAM_ABBREVIATION").apply(
        lambda g: pd.Series({"wins": (g["WL"] == "W").sum(), "games": len(g)}),
        include_groups=False,
    ).reset_index()
    records["win_pct"] = records["wins"] / records["games"]
    records = records.sort_values("win_pct", ascending=False).reset_index(drop=True)

    match = records.index[records["TEAM_ABBREVIATION"] == team_abbr]
    if len(match) == 0:
        return None
    seed = int(match[0]) + 1
    return {"conference": conf, "estimated_seed": seed,
            "win_pct": float(records.loc[match[0], "win_pct"])}


def render_series_table_html(series_df: pd.DataFrame, columns: list[str] | None = None, title: str = "") -> str:
    """One row per series -- different shape from table.py's stat-rows
    table, so it gets its own small renderer. columns is the toggleable
    subset (identity columns Season/Round/Opponent/Result are always shown
    first, regardless of what's passed)."""
    if series_df.empty:
        return f'<div style="color:#888;">{title}: no playoff series in this span.</div>'

    columns = [c for c in (columns if columns is not None else DEFAULT_SERIES_COLUMNS) if c in series_df.columns]
    all_cols = ["Season", "Round", "Opponent", "Result"] + columns

    header = "".join(
        f'<th style="padding:5px 12px;text-align:left;color:#eee;border-bottom:1px solid #333;">{c}</th>'
        for c in all_cols
    )
    rows_html = []
    for _, row in series_df.iterrows():
        played = row.get("player_played", True)
        cells = []
        for col in all_cols:
            val = row.get(col)
            if col == "Season":
                cells.append(f'<td style="padding:5px 12px;color:#ddd;">{val}</td>')
            elif col == "Round":
                badge = " \U0001F3C6" if row.get("Champion") else ""
                cells.append(f'<td style="padding:5px 12px;color:#ddd;">{val}{badge}</td>')
            elif col == "Opponent":
                cells.append(f'<td style="padding:5px 12px;color:#ddd;">{val}</td>')
            elif col == "Result":
                color = "#7CFC9A" if val == "Won" else "#ddd"
                dnp = ' <span style="color:#888;font-weight:400;">(DNP)</span>' if not played else ""
                cells.append(f'<td style="padding:5px 12px;color:{color};font-weight:600;">{val}{dnp}</td>')
            elif col == "Home Court":
                color = "#7CFC9A" if val == "Home" else "#ddd"
                text = val if pd.notna(val) else "\u2014"
                cells.append(f'<td style="padding:5px 12px;color:{color};">{text}</td>')
            else:
                fmt = SERIES_COLUMN_DEFS.get(col, (None, "{}", False))[1]
                if not played:
                    text = "\u2014"
                elif pd.isna(val):
                    text = "\u2014"
                elif fmt:
                    text = fmt.format(val)
                else:
                    text = str(val)
                cell_color = "#555" if not played else "#ddd"
                cells.append(f'<td style="padding:5px 12px;color:{cell_color};">{text}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div style="font-family:-apple-system,sans-serif;">
      {f'<h4 style="margin-bottom:6px;color:#eee;">{title}</h4>' if title else ''}
      <table style="border-collapse:collapse;background:#111;">
        <tr>{header}</tr>
        {''.join(rows_html)}
      </table>
    </div>
    """