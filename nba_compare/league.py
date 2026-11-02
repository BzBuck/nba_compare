"""
League mode: the whole league in one table -- one row per player (or per
player-season), one column per stat -- the way a season stats page shows
it, as opposed to the comparison table's one column per chosen span.

Every row is built from exactly the same stat code as a span in the
comparison table, so every built-in stat, custom formula and preset means
the same thing in both views. What's different is the scale: a season is
~500 players, so instead of building each player's full stat block the way
compare.aggregate_span does, the work is split up.

1. league_rows() fetches every player's games for the chosen seasons in
   ONE team/opponent join (NBADataStore.league_games_with_team_context),
   applies the season qualifiers and game filter to the whole league at
   once, then drops rows under the minimum games / minutes before anything
   expensive is computed for them.

2. family_values() computes one stat FAMILY (table.STAT_FAMILY -- a part
   of the stat block: box score, usage, consistency, Win Shares, ...) for
   every row. Only the families the chosen stats need are ever computed,
   so a table of per-game averages never pays for playoff series logs.

3. Each family has a measured cost (FAMILY_COST_MS). plan() prices the
   families a table needs, and when the total would be slow, holds the
   most expensive ones back until they're asked for -- the table shows
   without them first, rather than a stat added in passing quietly making
   every rerun crawl.

4. build_league_table() assembles the rows' blocks into one numeric
   DataFrame, evaluating custom formulas against them.

One league at a time: every league-wide number here (percentiles, league
averages, standings) is keyed by season within one league's store.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .data import NBADataStore
from .compare import (
    _box_block, _compute_usage, _compute_per_100, _compute_team_context, _compute_team_box,
    _compute_consistency, _relative_shooting, _league_averages, _regular_season_seed,
)
from .filters import GameFilter, apply_filter, player_seasons_passing
from .formulas import referenced_labels, safe_eval
from .models import PlayerSpan, season_str
from .table import STAT_DEFS, STAT_FAMILY, value_sample
from .win_shares import compute_win_shares
from . import percentiles as _percentiles
from . import playoffs as _playoffs

# Milliseconds per row, plus per player-season in the row (a combined
# multi-season row pays the second part once per season it covers), for
# each family -- measured with calibrate() below on NBA 2025 (one season)
# against 2016-2025 combined; rerun it after changing a family's code.
# Rough by nature: the point is telling a 1-second table from a 30-second
# one, not predicting either to the decimal.
FAMILY_COST_MS: dict[str, tuple[float, float]] = {
    "box":               (0.40, 0.00),
    "usage":             (0.65, 0.00),
    "per_100":           (0.85, 0.00),
    "team":              (1.75, 0.02),
    "team_box":          (3.00, 0.05),
    "consistency":       (1.65, 0.01),
    "relative_shooting": (0.55, 0.01),
    "league":            (0.15, 0.00),
    "avg_seed":          (0.10, 0.28),
    "win_shares":        (0.45, 1.15),
    "percentiles":       (0.02, 0.45),
    "depth":             (0.40, 2.80),
}

# Shown in the stat picker and the slow-table warning.
FAMILY_NAMES = {
    "box": "box score",
    "usage": "usage",
    "per_100": "per 100",
    "team": "team ratings",
    "relative_shooting": "relative shooting",
    "league": "league averages",
    "avg_seed": "avg seed",
    "percentiles": "percentiles",
    "win_shares": "Win Shares",
    "consistency": "consistency",
    "team_box": "team/opp box score",
    "depth": "playoff series",
}

# A family costing at least this per row is marked slow in the stat picker:
# about a second for one NBA season, but 7+ for ten seasons split by year.
SLOW_FAMILY_MS = 1.5

# A table estimated to take longer than this holds back its most expensive
# families until they're asked for.
WARN_SECONDS = 5.0


def family_is_slow(family: str) -> bool:
    per_row, per_season = FAMILY_COST_MS[family]
    return per_row + per_season >= SLOW_FAMILY_MS


# What one row of the table is: a player over all the chosen seasons, one
# player-season, or (playoffs only) one playoff series a player played in.
ROW_UNITS = {"player": "Player", "season": "Player-season", "series": "Playoff series"}


@dataclass(frozen=True)
class LeagueQuery:
    """Everything that decides which rows a league table has. Hashable, so
    the app can cache rows and family values under it."""
    league: str
    seasons: tuple[int, ...]
    season_type: str                  # "regular" or "playoffs"
    unit: str = "player"              # a ROW_UNITS key
    game_filter: GameFilter = GameFilter()
    min_games: int = 0
    min_minutes: float = 0.0          # MIN/G over the row's games

    def __post_init__(self):
        if self.unit not in ROW_UNITS:
            raise ValueError(f"unit must be one of {list(ROW_UNITS)}, not {self.unit!r}")
        if self.unit == "series" and self.season_type != "playoffs":
            raise ValueError("series rows only exist for playoff games")

    @property
    def narrowed(self) -> bool:
        """Whether the game filter drops games."""
        return self.game_filter.narrows(self.season_type)

    @property
    def whole_seasons(self) -> bool:
        """Whether each row's games are its seasons in full -- not under a
        game filter, and not for a single series. Same rule as aggregate_span:
        percentiles rank whole seasons, so they're blank otherwise, and Win
        Shares and playoff series logs use only the row's own games."""
        return not self.narrowed and self.unit != "series"


@dataclass
class LeagueRows:
    query: LeagueQuery
    store: NBADataStore
    groups: dict[tuple, pd.DataFrame]   # row key -> that row's games
    info: pd.DataFrame                  # Player / Season / ... / Champion, indexed by row key
    games_total: int                    # games before the filter, for the caption
    games_kept: int

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def player_seasons(self) -> int:
        return sum(g["SEASON"].nunique() for g in self.groups.values())


_ROW_KEYS = {
    "player": ["PLAYER_ID"],
    "season": ["PLAYER_ID", "SEASON"],
    # SERIES_ID counts a team's series within its postseason (playoffs.season_rounds).
    "series": ["PLAYER_ID", "SEASON", "TEAM_ID", "SERIES_ID"],
}
INFO_COLUMNS = {
    "player": ["Player", "Team", "Champion"],
    "season": ["Player", "Season", "Team", "Champion"],
    "series": ["Player", "Season", "Round", "Opp", "Team", "Series", "Champion"],
}


def league_rows(store: NBADataStore, query: LeagueQuery) -> LeagueRows:
    """The rows of a league table and each row's games: season qualifiers
    (judged on the full regular season, as in qualifying_seasons), then the
    game filter, then the row minimums."""
    games = store.league_games_with_team_context(list(query.seasons), query.season_type)
    games_total = len(games)
    keys = _ROW_KEYS[query.unit]

    conditions = query.game_filter.season_conditions
    if conditions:
        regular = store._load("regular")
        keep = player_seasons_passing(regular[regular.SEASON.isin(query.seasons)], store, conditions)
        games = games[pd.MultiIndex.from_frame(games[["PLAYER_ID", "SEASON"]]).isin(keep)]
    games = apply_filter(games.reset_index(drop=True), store, query.game_filter, query.season_type)
    series = None
    if query.unit == "series":
        series = _series_table(store, query.seasons)
        # A game the team logs can't place in a series has no row to go in.
        games = games.merge(series["games"], on=["GAME_ID", "TEAM_ID"], how="inner")

    if not games.empty:
        grouped = games.groupby(keys)
        size = grouped.size()
        # Over the games with minutes recorded. A row with none (before
        # 1951-52) can't be checked against a minutes floor: it passes only
        # when there isn't one.
        minutes = grouped["MIN_NUM"].mean()
        enough_minutes = (minutes >= query.min_minutes) | (query.min_minutes <= 0)
        passing = size.index[(size >= query.min_games) & enough_minutes]
        games = games[games.set_index(keys).index.isin(passing)]

    groups = {(k if isinstance(k, tuple) else (k,)): g for k, g in games.groupby(keys, sort=False)}
    champions = _champion_seasons(store, query.seasons)
    info = pd.DataFrame(
        [_row_info(key, g, query, champions, series) for key, g in groups.items()],
        index=pd.Index(list(groups), tupleize_cols=False),
        columns=INFO_COLUMNS[query.unit],
    )
    return LeagueRows(query, store, groups, info, games_total, len(games))


def _champion_seasons(store: NBADataStore, seasons) -> dict[int, tuple[int, set[int]]]:
    """{season: (champion TEAM_ID, players who played a playoff game for
    it)} -- the same "won the title" the Championships stat counts
    (playoffs.compute_series_records): a player gets the ring for appearing
    in the champion's postseason, whatever the regular-season table shows."""
    playoffs = store._load("playoffs")
    out = {}
    for season in seasons:
        champion = _playoffs.season_champion_team_id(store, int(season))
        if champion is None:
            continue
        ring = playoffs[(playoffs.SEASON == season) & (playoffs.TEAM_ID == champion)]
        out[int(season)] = (champion, set(ring["PLAYER_ID"].tolist()))
    return out


def _series_table(store: NBADataStore, seasons) -> dict:
    """Every team's playoff series in these seasons: "games" maps (GAME_ID,
    TEAM_ID) to SERIES_ID, and "series" describes each (SEASON, TEAM_ID,
    SERIES_ID) -- round, opponent, and the team's result in it."""
    games, series = [], []
    for season in seasons:
        bracket = _playoffs.season_rounds(store, int(season))
        if bracket.empty:
            continue
        rounds = int(bracket["ROUND"].max())
        games.append(bracket[["GAME_ID", "TEAM_ID", "SERIES_ID"]])
        for (team_id, series_id), g in bracket.groupby(["TEAM_ID", "SERIES_ID"]):
            wins, losses = int((g["WL"] == "W").sum()), int((g["WL"] == "L").sum())
            series.append({
                "SEASON": int(season), "TEAM_ID": team_id, "SERIES_ID": series_id,
                "Round": _playoffs._label_round(int(g["ROUND"].iloc[0]), rounds, store.league),
                "Opp": g["OPPONENT"].iloc[0],
                "Series": f"{'W' if wins > losses else 'L'} {wins}-{losses}",
            })
    if not games:
        return {"games": pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "SERIES_ID"]), "series": {}}
    return {
        "games": pd.concat(games, ignore_index=True),
        "series": {(s["SEASON"], s["TEAM_ID"], s["SERIES_ID"]): s for s in series},
    }


def _row_info(key: tuple, g: pd.DataFrame, query: LeagueQuery, champions: dict, series: dict | None) -> list:
    """Player name (as of the row's latest game), the teams played for in
    order -- "BKN/PHX" for a mid-season trade -- and whether the row is a
    title: a player-season the player won it in, or a series in a title
    run. A player row only when the whole table is one season, so a gold
    row means the same thing on every row -- over a range, the
    Championships stat counts them."""
    by_date = g.sort_values("GAME_DATE")
    teams = "/".join(dict.fromkeys(by_date["TEAM_ABBREVIATION"].dropna()))
    name = by_date["PLAYER_NAME"].iloc[-1]
    seasons = g["SEASON"].unique()
    season = int(seasons[0])
    team_id, ring = champions.get(season, (None, set()))
    if query.unit == "series":
        about = series["series"][(season, key[2], key[3])]
        return [name, season_str(season, query.league), about["Round"], about["Opp"], teams,
                about["Series"], key[2] == team_id]
    champion = int(key[0]) in ring
    if query.unit == "season":
        return [name, season_str(season, query.league), teams, champion]
    return [name, teams, champion and len(query.seasons) == 1]


# ---------- families ----------
# Each takes (row key, the row's games, LeagueRows) and returns the stat
# block keys it fills -- the same keys and values aggregate_span gives a
# span over the same games.

def _percentiles_part(key, g, rows):
    if not rows.query.whole_seasons:
        return {}
    span = PlayerSpan(int(key[0]), "", sorted(int(s) for s in g["SEASON"].unique()))
    return {"percentiles": _percentiles.span_percentiles(rows.store, span, rows.query.season_type)}


def _depth_part(key, g, rows):
    if rows.query.season_type != "playoffs":
        return {}
    store = rows.store
    if not rows.query.whole_seasons:
        records = _playoffs.compute_series_records(store, g, wl_from_player_games=True)
        records = [r for r in records if r["player_played"]]
    else:
        seasons = sorted(int(s) for s in g["SEASON"].unique())
        records = _playoffs.compute_series_records(store, store.games(int(key[0]), seasons, "playoffs"))
    return {"series_records": records, "depth": _playoffs.depth_summary(records)}


def _per_100_part(key, g, rows):
    possessions = _compute_per_100(g)
    return {"per_100": possessions["per_100"] if possessions else None, "possessions": possessions}


FAMILIES: dict[str, Callable[[tuple, pd.DataFrame, LeagueRows], dict]] = {
    "box": lambda key, g, rows: _box_block(g),
    "usage": lambda key, g, rows: {"usage": _compute_usage(g)},
    "per_100": _per_100_part,
    "team": lambda key, g, rows: {"team": _compute_team_context(g, rows.store.game_minutes)},
    "team_box": lambda key, g, rows: {"team_box": _compute_team_box(g)},
    "consistency": lambda key, g, rows: {"consistency": _compute_consistency(g)},
    "relative_shooting": lambda key, g, rows: {"relative_shooting": _relative_shooting(
        rows.store, g, _box_block(g)["shooting"], rows.query.season_type)},
    "league": lambda key, g, rows: {"league": _league_averages(rows.store, g, rows.query.season_type)},
    "avg_seed": lambda key, g, rows: (
        {"avg_seed": _regular_season_seed(rows.store, g)} if rows.query.season_type == "regular" else {}),
    "win_shares": lambda key, g, rows: {"win_shares": compute_win_shares(
        g, rows.store, rows.query.season_type, full_team_seasons=rows.query.whole_seasons)},
    "percentiles": _percentiles_part,
    "depth": _depth_part,
}
assert set(FAMILIES) == set(FAMILY_COST_MS) == set(FAMILY_NAMES) == set(STAT_FAMILY.values())


def family_applies(family: str, query: LeagueQuery) -> bool:
    """False for a family that's blank by definition here -- playoff series
    in a regular-season table, seed in a playoff one, percentiles under a
    game filter or for a single series -- so it's neither computed nor priced."""
    if family == "depth":
        return query.season_type == "playoffs"
    if family == "avg_seed":
        return query.season_type == "regular"
    if family == "percentiles":
        return query.whole_seasons
    return True


def family_values(rows: LeagueRows, family: str, on_progress: Callable[[float], None] | None = None) -> dict:
    """{row key: the block keys `family` fills} for every row."""
    fn = FAMILIES[family]
    out = {}
    n = len(rows.groups)
    for i, (key, g) in enumerate(rows.groups.items()):
        out[key] = fn(key, g, rows)
        if on_progress is not None and i % 25 == 0:
            on_progress(i / n)
    return out


# ---------- which families, and what they cost ----------

def families_for(labels: list[str], formulas: dict[str, str]) -> set[str]:
    """The families the stats in `labels` read -- a custom formula's are its
    built-in inputs' (see formulas.referenced_labels). Box score always,
    since it's cheap and the table's GP/MIN come from it."""
    needed = {"box"}
    for label in labels:
        if label in STAT_FAMILY:
            needed.add(STAT_FAMILY[label])
        elif label in formulas:
            needed |= {STAT_FAMILY[ref] for ref in referenced_labels(formulas[label], STAT_DEFS)}
    return needed


def estimate_seconds(families, n_rows: int, n_player_seasons: int) -> float:
    total_ms = 0.0
    for family in families:
        per_row, per_season = FAMILY_COST_MS[family]
        total_ms += per_row * n_rows + per_season * n_player_seasons
    return total_ms / 1000


def plan(rows: LeagueRows, needed: set[str], done: set[str]) -> tuple[set[str], set[str], float]:
    """
    (families to compute now, families to hold back, seconds the held ones
    would take). `done` are families already computed for these rows, which
    cost nothing. If everything left fits in WARN_SECONDS it all runs;
    otherwise the most expensive slow families (family_is_slow -- the ones
    the stat picker marks) are held back, one at a time, until what's left
    does. A cheap family always runs: a big table may take a few seconds
    for its basic columns, but never shows up without them.
    """
    todo = {f for f in needed if f not in done and family_applies(f, rows.query)}
    n, ps = len(rows), rows.player_seasons
    cost = {f: estimate_seconds([f], n, ps) for f in todo}
    held: set[str] = set()
    for family in sorted((f for f in todo if family_is_slow(f)), key=cost.get, reverse=True):
        if sum(cost[f] for f in todo - held) <= WARN_SECONDS:
            break
        held.add(family)
    return todo - held, held, sum(cost[f] for f in held)


# ---------- the table ----------

def _meets_minimums(sample: tuple[int, float | None] | None, query: LeagueQuery) -> bool:
    """Whether a value resting on only some of its row's games (value_sample)
    still meets the table's minimums on those games. The row's own GP and
    MIN/G passing them says nothing about a value built from fewer: it's left
    blank instead, and so is any formula using it."""
    if sample is None:
        return True
    games, minutes = sample
    if games < query.min_games:
        return False
    return minutes is None or query.min_minutes <= 0 or minutes / games >= query.min_minutes


def build_league_table(
    rows: LeagueRows, parts: dict[str, dict], labels: list[str], formulas: dict[str, str],
) -> pd.DataFrame:
    """
    One row per league row, Player (/Season) / Team then one column per
    label. `parts` is {family: family_values(...)} for whichever families
    have been computed; a stat whose family is missing comes out blank,
    like a stat a span has no data for. Custom formulas (label -> expr)
    are evaluated against each row's built-in values, so they mean exactly
    what they do in the comparison table. A value built from only some of
    its row's games has to meet the minimums on those games
    (_meets_minimums), or it's blank.
    """
    builtin = [l for l in labels if l in STAT_DEFS]
    formula_labels = [l for l in labels if l in formulas and l not in STAT_DEFS]
    refs = {l: referenced_labels(formulas[l], STAT_DEFS) for l in formula_labels}
    needed_vars = set(builtin).union(*refs.values()) if refs else set(builtin)
    getters = {l: STAT_DEFS[l][0] for l in needed_vars}

    records = []
    for key in rows.groups:
        block: dict = {}
        for values in parts.values():
            block.update(values.get(key, {}))
        variables = {}
        for label, getter in getters.items():
            try:
                variables[label] = getter(block)
            except Exception:
                variables[label] = None
            if not _meets_minimums(value_sample(block, label), rows.query):
                variables[label] = None
        row = {l: variables[l] for l in builtin}
        for label in formula_labels:
            try:
                row[label] = safe_eval(formulas[label], {r: variables[r] for r in refs[label]})
            except Exception:
                row[label] = None
        records.append(row)
    stats = pd.DataFrame(records, index=rows.info.index, columns=[l for l in labels if l in builtin or l in refs])
    return pd.concat([rows.info, stats.apply(pd.to_numeric, errors="coerce")], axis=1)


# ---------- calibration ----------

def calibrate(store: NBADataStore, single: int, span: range) -> dict[str, tuple[float, float]]:
    """
    Measures each family on one season (rows = player-seasons) and on a
    combined multi-season range, and solves for FAMILY_COST_MS's
    (per row, per player-season) pair. Run as `python -m nba_compare.league`.
    """
    import time
    samples = []
    for seasons, season_type in (((single,), "regular"), (tuple(span), "regular"),
                                 ((single,), "playoffs"), (tuple(span), "playoffs")):
        rows = league_rows(store, LeagueQuery(store.league, seasons, season_type, "player"))
        timings = {}
        for family in FAMILIES:
            if not family_applies(family, rows.query):
                continue
            family_values(rows, family)  # warm the per-season caches
            t = time.perf_counter()
            family_values(rows, family)
            timings[family] = (time.perf_counter() - t) * 1000
        samples.append((season_type, len(rows), rows.player_seasons, timings))

    result = {}
    for family in FAMILIES:
        pts = [(n, ps, t[family]) for _, n, ps, t in samples if family in t]
        (n1, p1, t1), (n2, p2, t2) = pts[0], pts[1]
        # t = a*n + b*ps; the single-season sample has n == ps.
        b = max(0.0, (t2 - t1 * n2 / n1) / (p2 - n2)) if p2 != n2 else 0.0
        a = max(0.0, (t1 - b * p1) / n1)
        result[family] = (round(a, 2), round(b, 2))
    return result


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    store = NBADataStore.from_config("NBA")
    latest = int(store._load("regular")["SEASON"].max())
    for family, (a, b) in calibrate(store, latest, range(latest - 9, latest + 1)).items():
        print(f'    "{family}": ({a:.2f}, {b:.2f}),')
