"""
Box-score sums that respect unrecorded games.

The early game logs have holes: a 1961-62 game can carry a player's points
and makes but no field goal attempts, rebounds or assists, and minutes
weren't kept for most games before the mid-1960s (see
data._blank_unrecorded_minutes). A blank there means "not recorded", not
zero, so:

- every per-game average is over the games that HAVE that stat, not all of
  them -- Elgin Baylor's 1961-62 MIN/G is 43.7 over the 11 games with
  minutes, not 10.0 over all 48;
- every rate uses only games with all of its parts recorded -- FG% is FGM
  over FGA from the games with both, never makes from 48 games over
  attempts from 37.

`coverage` (games behind each number) is what lets the table mark a value
resting on only some of a span's games, the way Basketball-Reference
underlines incomplete totals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import COUNTING_STATS

# 1979-80, the first season with a 3-point line (SEASON is the start year).
# Before it, a blank 3PM/3PA is a true zero, not a missing value.
FIRST_THREE_POINT_SEASON = 1979

# Each rate's parts -- it's computed only over games with all of them.
# TSA (true shot attempts) isn't a rate but needs FGA and FTA together.
MATCHED_COLS = {
    "FG_PCT": ("FGM", "FGA"),
    "FG3_PCT": ("FG3M", "FG3A"),
    "FT_PCT": ("FTM", "FTA"),
    "EFG_PCT": ("FGM", "FG3M", "FGA"),
    "TS_PCT": ("PTS", "FGA", "FTA"),
    "TSA": ("FGA", "FTA"),
}
SHOOTING_RATES = [k for k in MATCHED_COLS if k != "TSA"]
_COLS = COUNTING_STATS + ["MIN_NUM"]
_IDX = {c: i for i, c in enumerate(_COLS)}


def fill_pre_three(games: pd.DataFrame) -> pd.DataFrame:
    """`games` with blank 3PM/3PA set to 0 in seasons before the 3-point line."""
    if "SEASON" not in games.columns:
        return games
    pre = games["SEASON"] < FIRST_THREE_POINT_SEASON
    if not pre.any():
        return games
    games = games.copy()
    for col in ("FG3M", "FG3A"):
        games.loc[pre, col] = games.loc[pre, col].fillna(0)
    return games


def box_sums(games: pd.DataFrame) -> dict:
    """
    {"sums": {stat: total over games recording it}, "counts": {stat: those
    games}, "matched": {rate: ({part: total}, games with every part)}} for
    COUNTING_STATS and MIN_NUM. Plain sums can be added across groups
    (merge_box_sums), so a combination of series is still exact.
    """
    vals = games[_COLS].to_numpy(dtype=float, copy=True)
    if "SEASON" in games.columns:
        pre = games["SEASON"].to_numpy() < FIRST_THREE_POINT_SEASON
        for col in ("FG3M", "FG3A"):
            column = vals[:, _IDX[col]]
            column[pre & np.isnan(column)] = 0.0
    present = ~np.isnan(vals)
    sums = np.where(present, vals, 0.0).sum(axis=0)
    counts = present.sum(axis=0)
    matched = {}
    for key, parts in MATCHED_COLS.items():
        idx = [_IDX[c] for c in parts]
        rows = present[:, idx].all(axis=1)
        matched[key] = ({c: float(vals[rows, _IDX[c]].sum()) for c in parts}, int(rows.sum()))
    return {
        "sums": {c: float(s) for c, s in zip(_COLS, sums)},
        "counts": {c: int(n) for c, n in zip(_COLS, counts)},
        "matched": matched,
    }


def merge_box_sums(parts: list[dict]) -> dict:
    """box_sums over several groups of games at once, from each one's own."""
    out = {"sums": dict.fromkeys(_COLS, 0.0), "counts": dict.fromkeys(_COLS, 0), "matched": {}}
    for part in parts:
        for c in _COLS:
            out["sums"][c] += part["sums"][c]
            out["counts"][c] += part["counts"][c]
    for key, cols in MATCHED_COLS.items():
        out["matched"][key] = (
            {c: sum(p["matched"][key][0][c] for p in parts) for c in cols},
            sum(p["matched"][key][1] for p in parts),
        )
    return out


def per_game(box: dict, stats=COUNTING_STATS) -> dict:
    """Each stat averaged over the games that recorded it; None if none did."""
    return {c: box["sums"][c] / box["counts"][c] if box["counts"][c] else None for c in stats}


def shooting(box: dict) -> dict:
    """FG%, 3P%, FT%, eFG%, TS% -- each over the games with all its parts."""
    def part(key):
        return box["matched"][key][0]
    fg, fg3, ft, efg, ts = (part(k) for k in SHOOTING_RATES)
    return {
        "FG_PCT": fg["FGM"] / fg["FGA"] if fg["FGA"] else None,
        "FG3_PCT": fg3["FG3M"] / fg3["FG3A"] if fg3["FG3A"] else None,
        "FT_PCT": ft["FTM"] / ft["FTA"] if ft["FTA"] else None,
        "EFG_PCT": (efg["FGM"] + 0.5 * efg["FG3M"]) / efg["FGA"] if efg["FGA"] else None,
        # true shooting: points per shooting possession
        "TS_PCT": ts["PTS"] / (2 * (ts["FGA"] + 0.44 * ts["FTA"])) if (ts["FGA"] or ts["FTA"]) else None,
    }


def shooting_frame(games: pd.DataFrame, by) -> pd.DataFrame:
    """shooting() for every group of `games` at once (grouped by `by`), as a
    frame with one column per rate -- for league-wide tables."""
    games = fill_pre_three(games)
    sums = {}
    for key in SHOOTING_RATES:
        cols = list(MATCHED_COLS[key])
        sums[key] = games[cols].where(games[cols].notna().all(axis=1)).groupby(by).sum()

    def ratio(num, den):
        return num / den.replace(0, np.nan)
    fg, fg3, ft, efg, ts = (sums[k] for k in SHOOTING_RATES)
    return pd.DataFrame({
        "FG_PCT": ratio(fg["FGM"], fg["FGA"]),
        "FG3_PCT": ratio(fg3["FG3M"], fg3["FG3A"]),
        "FT_PCT": ratio(ft["FTM"], ft["FTA"]),
        "EFG_PCT": ratio(efg["FGM"] + 0.5 * efg["FG3M"], efg["FGA"]),
        "TS_PCT": ratio(ts["PTS"], 2 * (ts["FGA"] + 0.44 * ts["FTA"])),
    })
