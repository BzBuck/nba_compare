"""
League percentile ranks: where a player's per-game value for a stat falls
relative to every other qualifying player in the league that same season.

Computed PER SEASON, never pooled across a span -- league context shifts
year to year (pace, 3-point rate), so blending a 2005 value into a
multi-season pool wouldn't mean anything. A span covering several seasons
gets a games-weighted average of that season's percentile.

Efficiency: one groupby over that season's full player-game table covers
every player at once (~500 players from ~26k game rows) -- cheap, and
cached per (store, season, season_type, min_games) so comparing many
players/spans that share a season only pays the cost once.

Every percentile column means the same thing regardless of the underlying
stat's direction: higher percentile = better. Turnovers are inverted
internally (low TOV -> high percentile) so you don't have to remember
which stats run backwards.
"""
from __future__ import annotations
from functools import lru_cache
import pandas as pd
from .data import NBADataStore
from .models import PlayerSpan

# Stat -> whether a HIGHER raw value is better. Only TOV runs the other way.
PERCENTILE_STATS = {
    "PTS": True, "REB": True, "AST": True, "STL": True, "BLK": True, "TOV": False,
    "FG_PCT": True, "FG3_PCT": True, "FT_PCT": True, "EFG_PCT": True, "TS_PCT": True,
}

# Shooting rates that get a "relative to league" version (see
# league_average_rates/span_league_average) -- unlike percentiles, "value
# minus league average" is meaningful for a Duo too, since it's just
# subtraction rather than a rank against a single-player distribution.
RELATIVE_SHOOTING_STATS = ["FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "TS_PCT"]


@lru_cache(maxsize=64)
def season_league_table(
    store: NBADataStore, season: int, season_type: str = "regular", min_games: int | None = None
) -> pd.DataFrame:
    """
    One row per qualifying player that season: per-game values for every
    stat in PERCENTILE_STATS, plus a "{stat}_PCTILE" column for each
    (0-100, direction-corrected so higher always means better).

    min_games defaults to 10 for regular season (filters out call-ups/
    tiny samples) and 1 for playoffs (series are short; playoff appearance
    itself is already a relevance filter).
    """
    if min_games is None:
        min_games = 10 if season_type == "regular" else 1

    games = store.all_player_games_for_season(season, season_type)
    if games.empty:
        return pd.DataFrame()

    grouped = games.groupby("PLAYER_ID").agg(
        GP=("PTS", "size"),
        PTS=("PTS", "sum"), REB=("REB", "sum"), AST=("AST", "sum"),
        STL=("STL", "sum"), BLK=("BLK", "sum"), TOV=("TOV", "sum"),
        FGM=("FGM", "sum"), FGA=("FGA", "sum"),
        FG3M=("FG3M", "sum"), FG3A=("FG3A", "sum"),
        FTM=("FTM", "sum"), FTA=("FTA", "sum"),
    ).reset_index()

    qualified = grouped[grouped["GP"] >= min_games].copy()
    if qualified.empty:
        return pd.DataFrame()

    gp = qualified["GP"]
    per_game = pd.DataFrame({
        "PLAYER_ID": qualified["PLAYER_ID"],
        "GP": gp,
        "PTS": qualified["PTS"] / gp,
        "REB": qualified["REB"] / gp,
        "AST": qualified["AST"] / gp,
        "STL": qualified["STL"] / gp,
        "BLK": qualified["BLK"] / gp,
        "TOV": qualified["TOV"] / gp,
        "FG_PCT": qualified["FGM"] / qualified["FGA"].replace(0, pd.NA),
        "FG3_PCT": qualified["FG3M"] / qualified["FG3A"].replace(0, pd.NA),
        "FT_PCT": qualified["FTM"] / qualified["FTA"].replace(0, pd.NA),
        "EFG_PCT": (qualified["FGM"] + 0.5 * qualified["FG3M"]) / qualified["FGA"].replace(0, pd.NA),
        "TS_PCT": qualified["PTS"] / (2 * (qualified["FGA"] + 0.44 * qualified["FTA"])).replace(0, pd.NA),
    })

    for stat, higher_is_better in PERCENTILE_STATS.items():
        direction = 1 if higher_is_better else -1
        per_game[f"{stat}_PCTILE"] = (direction * per_game[stat]).rank(pct=True) * 100

    return per_game


def player_season_percentiles(
    store: NBADataStore, player_id: int, season: int, season_type: str = "regular"
) -> dict | None:
    """{stat: percentile} for one player in one season, or None if they
    didn't meet the minimum-games qualifier that season."""
    table = season_league_table(store, season, season_type)
    if table.empty:
        return None
    row = table[table["PLAYER_ID"] == player_id]
    if row.empty:
        return None
    return {
        "gp": int(row.iloc[0]["GP"]),
        **{stat: row.iloc[0][f"{stat}_PCTILE"] for stat in PERCENTILE_STATS},
    }


def span_percentiles(store: NBADataStore, span: PlayerSpan, season_type: str = "regular") -> dict:
    """
    Games-weighted average percentile across every season in the span.
    Seasons where the player didn't qualify are excluded entirely (not
    counted as 0) rather than dragging the average down artificially.
    """
    per_season = [
        player_season_percentiles(store, span.player_id, season, season_type)
        for season in span.seasons
    ]
    per_season = [p for p in per_season if p is not None]
    if not per_season:
        return {stat: None for stat in PERCENTILE_STATS}

    total_gp = sum(p["gp"] for p in per_season)
    return {
        stat: sum(p[stat] * p["gp"] for p in per_season) / total_gp if total_gp else None
        for stat in PERCENTILE_STATS
    }


@lru_cache(maxsize=64)
def league_average_rates(store: NBADataStore, season: int, season_type: str = "regular") -> dict:
    """
    League-wide shooting rates for one season, from EVERY game that
    season (not just qualifying players the way season_league_table()
    filters -- a league average should reflect the whole league's actual
    shooting, same as Basketball-Reference's league-average row). Totals
    summed first, rates computed once -- same approach used everywhere
    else in this project, not an average of individual players' rates
    (which would overweight low-volume shooters).
    """
    games = store.all_player_games_for_season(season, season_type)
    if games.empty:
        return {}
    fgm, fga = games["FGM"].sum(), games["FGA"].sum()
    fg3m, fg3a = games["FG3M"].sum(), games["FG3A"].sum()
    ftm, fta = games["FTM"].sum(), games["FTA"].sum()
    pts = games["PTS"].sum()
    return {
        "FG_PCT": fgm / fga if fga else None,
        "FG3_PCT": fg3m / fg3a if fg3a else None,
        "FT_PCT": ftm / fta if fta else None,
        "EFG_PCT": (fgm + 0.5 * fg3m) / fga if fga else None,
        "TS_PCT": pts / (2 * (fga + 0.44 * fta)) if (fga or fta) else None,
    }


def span_league_average(
    store: NBADataStore, games_by_season: dict[int, int], season_type: str = "regular"
) -> dict:
    """
    Games-weighted average league rate across a set of seasons, weighted
    by games_by_season -- how many games THIS player/duo actually has in
    each season (not the league's own game counts), same games-weighting
    convention as span_percentiles, so a span leaning heavily on one
    season isn't diluted by treating every season equally. A season
    missing a given rate entirely (e.g. FG3_PCT before the 3-point line
    existed) is skipped for that stat rather than pulling the average
    toward zero.
    """
    result = {}
    for stat in RELATIVE_SHOOTING_STATS:
        weighted_sum, total_gp = 0.0, 0
        for season, gp in games_by_season.items():
            if not gp:
                continue
            rate = league_average_rates(store, season, season_type).get(stat)
            if rate is None:
                continue
            weighted_sum += gp * rate
            total_gp += gp
        result[stat] = weighted_sum / total_gp if total_gp else None
    return result