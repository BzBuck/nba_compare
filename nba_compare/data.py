"""
Loads your parquet game logs and turns raw per-game rows into
season-level aggregates for a given player + season list.

Expects the NBA stats API game log schema you already have:
SEASON_ID, PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME,
GAME_ID, GAME_DATE, MATCHUP, WL, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS,
FANTASY_PTS, VIDEO_AVAILABLE
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Union

import pandas as pd

# A parquet source: a path/URL pandas can read, or a callable producing one
# on demand (see NBADataStore).
PathSource = Union[str, Callable[[], str]]

# Counting stats that are safe to SUM across games (rates get recomputed from these).
COUNTING_STATS = [
    "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
]

# Team box-score columns every possession-based number is built from
# (pace, ORtg/DRtg, USG%, and all the per-100 stats). The team parquet only
# carries these in full from 1985 onward -- see _reconstructed_team_lines.
# A rebuilt value in one of THESE is what sets TEAM_EST.
TEAM_POSSESSION_STATS = ["FGA", "FTA", "TOV", "OREB", "PTS"]

# Every team/opponent box-score column joined onto a player's games (as
# TEAM_<stat> / OPP_<stat>) -- the possession columns above plus the rest of
# the line (FGM, REB, AST, BLK, ...), which feed the Team/Opp rows and
# formulas like AST% or TRB%. Same gaps before 1985, same rebuild.
TEAM_BOX_STATS = COUNTING_STATS

# How far a game's summed player minutes may sit from the official team
# minutes before the player rows are treated as an incomplete roster and
# not used to rebuild that team line (see _fill_missing_team_stats).
ROSTER_MINUTES_TOLERANCE = 5.0

# Minimum minutes EACH player in a duo must log in a game for it to count
# as "played together" -- excludes token appearances (a garbage-time minute
# in a blowout, an early exit to injury) that technically share a GAME_ID/
# TEAM_ID but don't represent meaningful shared floor time.
DUO_MIN_MINUTES = 10.0


def _to_minutes(min_col: pd.Series) -> pd.Series:
    """Handle both plain numeric minutes and legacy 'MM:SS' string formats."""
    if pd.api.types.is_numeric_dtype(min_col):
        return min_col.astype(float)

    def parse(v):
        if pd.isna(v):
            return 0.0
        s = str(v)
        if ":" in s:
            m, sec = s.split(":")
            return int(m) + int(sec) / 60.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    return min_col.map(parse)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Add SEASON (start-year int) and MIN_NUM (float minutes) columns."""
    df = df.copy()
    df["SEASON"] = df["SEASON_ID"].astype(str).str[1:].astype(int)
    df["MIN_NUM"] = _to_minutes(df["MIN"])
    return df


def _fill_missing_team_stats(
    merged: pd.DataFrame, recon: pd.DataFrame, prefix: str, id_col: str
) -> pd.DataFrame:
    """
    Fill NaN <prefix>_<stat> columns (every TEAM_BOX_STATS stat) from `recon` (team lines
    rebuilt by summing player rows -- see
    NBADataStore._reconstructed_team_lines), matched on GAME_ID + id_col.
    Official values are never overwritten; this only fills gaps.

    A rebuilt line is only trusted where its summed player minutes agree
    with the OFFICIAL team minutes for that game (within
    ROSTER_MINUTES_TOLERANCE). That's the check that the player rows are a
    complete roster and not a partial scrape -- and it handles overtime
    games for free, since their minutes total is higher than the usual 240
    on both sides of the comparison. Games whose official minutes are
    missing or zero (some 1960s rows) get no fill at all.

    Accumulates a boolean TEAM_EST column marking rows where at least one
    POSSESSION value (TEAM_POSSESSION_STATS) came from a rebuilt line rather
    than the official team logs -- that's what the comparison table marks
    with an asterisk on the possession-based rows. Each stat also gets its
    own <prefix>_<stat>_EST column, so a Team/Opp row is only marked when
    its own column was rebuilt, and a filled AST can't mark USG% (or the
    reverse).
    """
    if merged.empty or id_col not in merged.columns:
        return merged

    # TEAM_ID arrives as int64 on the player's own side but float64 on the
    # opponent side (the left join introduces NaN for unmatched games), and
    # pandas refuses to merge those against each other -- so both sides of
    # the key go to float64. Team IDs are ~1.6e9, exactly representable.
    key = "_join_team_id"
    left = merged.assign(**{key: merged[id_col].astype("float64")})
    src_cols = TEAM_BOX_STATS + ["MIN_NUM"]
    right = recon.assign(**{key: recon["TEAM_ID"].astype("float64")})
    right = right[["GAME_ID", key] + src_cols].rename(
        columns={c: f"_recon_{c}" for c in src_cols}
    )
    out = left.merge(right, on=["GAME_ID", key], how="left")

    official_min = out[f"{prefix}_MIN"]
    roster_complete = (
        (out["_recon_MIN_NUM"] - official_min).abs() <= ROSTER_MINUTES_TOLERANCE
    ) & (official_min > 0)

    estimated = pd.Series(False, index=out.index)
    for stat in TEAM_BOX_STATS:
        target = f"{prefix}_{stat}"
        available = out[f"_recon_{stat}"].where(roster_complete)
        gap = out[target].isna() & available.notna()
        out.loc[gap, target] = available[gap]
        out[f"{target}_EST"] = gap
        if stat in TEAM_POSSESSION_STATS:
            estimated |= gap

    prior = out["TEAM_EST"] if "TEAM_EST" in out.columns else False
    out["TEAM_EST"] = estimated | prior
    return out.drop(columns=[key] + [f"_recon_{c}" for c in src_cols])


class NBADataStore:
    """
    Lazily loads the parquet game logs and serves filtered/prepped rows.
    Only loads a file the first time it's actually needed.

    Each source may be given either as a path/URL anything pandas can read,
    or as a zero-argument callable returning one. The callable form is what
    keeps from_config() lazy: resolving a Hugging Face file can mean
    downloading it, and nothing should pay that cost for a dataset the
    session never touches (most comparisons never open the playoff logs).
    """

    def __init__(
        self,
        regular_path: PathSource = "nba_gamelogs.parquet",
        playoff_path: PathSource = "nba_playoffs_gamelogs.parquet",
        team_regular_path: PathSource = "nba_team_gamelogs.parquet",
        team_playoff_path: PathSource = "nba_team_playoffs_gamelogs.parquet",
    ):
        self._paths: dict[str, PathSource] = {
            "regular": regular_path,
            "playoffs": playoff_path,
            "team_regular": team_regular_path,
            "team_playoffs": team_playoff_path,
        }
        self._cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_config(cls) -> "NBADataStore":
        """
        Convenience constructor using the source configured in config.py --
        the Hugging Face dataset by default. Constructing the store touches
        no data and hits no network; each file is fetched on first use.
        """
        from . import config
        return cls(
            regular_path=partial(config.resolve, "regular"),
            playoff_path=partial(config.resolve, "playoffs"),
            team_regular_path=partial(config.resolve, "team_regular"),
            team_playoff_path=partial(config.resolve, "team_playoffs"),
        )

    def _load(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            source = self._paths[key]
            if callable(source):
                source = source()
            self._cache[key] = _prep(pd.read_parquet(source))
        return self._cache[key]

    def _reconstructed_team_lines(self, season_type: str) -> pd.DataFrame:
        """
        Team box-score lines rebuilt by SUMMING the player game logs per
        GAME_ID + TEAM_ID -- a fallback for the seasons where the official
        team parquet simply doesn't have them.

        The team logs carry no FGA/FTA/TOV/OREB at all before 1985, but the
        player logs go further back for some of them (FGA into the 1960s,
        OREB from 1973-74, TOV from 1977-78), so summing players recovers a
        usable team line for roughly 1977-1984 -- and nothing before that,
        since with no turnovers recorded there is no possession estimate to
        make at any level.

        A column is only rebuilt when EVERY player row in that game/team
        has it: one null in the roster would make the sum silently
        undercount, so it's left missing instead. Checked against 2020,
        where these sums reproduce official team FGA/FTA/OREB/PTS exactly;
        TOV comes in ~0.6/game low because team turnovers aren't charged to
        any individual player, which makes possessions built this way about
        0.6% light. Coverage over 1977-1984 runs from ~35% of games to ~93%
        (best in 1983), and the games that survive are clustered by team,
        so a rebuilt league-wide average is a few points off a true one --
        why anything resting on these is asterisked in the table rather than
        shown as equivalent to the post-1985 numbers.
        """
        cache_key = f"recon_{season_type}"
        if cache_key not in self._cache:
            df = self._load(season_type)
            cols = TEAM_BOX_STATS + ["MIN_NUM"]
            group_keys = [df["GAME_ID"], df["TEAM_ID"]]
            sums = df[cols].groupby(group_keys).sum(min_count=1)
            null_counts = df[cols].isna().groupby(group_keys).sum()
            self._cache[cache_key] = sums.mask(null_counts > 0).reset_index()
        return self._cache[cache_key]

    def games(self, player_id: int, seasons: list[int], season_type: str) -> pd.DataFrame:
        """season_type: 'regular' or 'playoffs'"""
        df = self._load(season_type)
        return df[(df.PLAYER_ID == player_id) & (df.SEASON.isin(seasons))]

    def team_games_for_season(self, season: int, season_type: str = "regular") -> pd.DataFrame:
        """ALL teams' games for one season (not filtered to one team) --
        needed to work out league-wide playoff depth and rough standings."""
        key = "team_regular" if season_type == "regular" else "team_playoffs"
        df = self._load(key)
        return df[df.SEASON == season]

    def all_player_games_for_season(self, season: int, season_type: str = "regular") -> pd.DataFrame:
        """ALL players' games for one season (not filtered to one player) --
        needed to build league-wide percentile distributions."""
        df = self._load(season_type)
        return df[df.SEASON == season]

    def games_with_team_context(self, player_id: int, seasons: list[int], season_type: str) -> pd.DataFrame:
        """
        Same as games(), but left-joins each row with:
        - that game's full TEAM box line (TEAM_MIN plus TEAM_<stat> for every
          TEAM_BOX_STATS stat), matched on GAME_ID + TEAM_ID -- for usage%,
          team pace/scoring context, and the Team rows/formula variables.
        - that game's full OPPONENT box line (OPP_MIN, OPP_<stat>), matched
          on GAME_ID with a different TEAM_ID -- for team defensive rating
          (points allowed per 100 opponent possessions) and the Opp rows.

        Rows where no match is found get NaN in the relevant columns --
        callers should filter those out before computing with them.

        Because the team parquet has no possession columns at all before
        1985, any gap left in a TEAM_*/OPP_* box column is then
        filled where possible from team lines rebuilt out of the player
        logs (see _reconstructed_team_lines), and the boolean TEAM_EST
        column marks the rows where that happened for a possession column
        (per-column <prefix>_<stat>_EST flags cover the rest), so downstream
        stats can be flagged as estimates rather than passed off as
        officially sourced.
        """
        player_df = self.games(player_id, seasons, season_type)
        team_key = "team_regular" if season_type == "regular" else "team_playoffs"
        team_df = self._load(team_key)

        team_slim = team_df[["GAME_ID", "TEAM_ID", "MIN_NUM"] + TEAM_BOX_STATS].rename(
            columns={"MIN_NUM": "TEAM_MIN", **{c: f"TEAM_{c}" for c in TEAM_BOX_STATS}}
        )
        merged = player_df.merge(team_slim, on=["GAME_ID", "TEAM_ID"], how="left")

        # OPP_MIN isn't used in any stat directly -- it's the completeness
        # check that lets a rebuilt OPPONENT line be trusted, the same way
        # TEAM_MIN does for the player's own team.
        opp_slim = team_df[["GAME_ID", "TEAM_ID", "MIN_NUM"] + TEAM_BOX_STATS].rename(
            columns={"TEAM_ID": "OPP_TEAM_ID", "MIN_NUM": "OPP_MIN", **{c: f"OPP_{c}" for c in TEAM_BOX_STATS}}
        )
        # Join on GAME_ID only (opponent has a different TEAM_ID by definition),
        # then keep just the row where the matched team ISN'T the player's own
        # team -- assumes exactly 2 teams per GAME_ID in the team parquet.
        merged = merged.merge(opp_slim, on="GAME_ID", how="left")
        merged = merged[(merged["OPP_TEAM_ID"].isna()) | (merged["OPP_TEAM_ID"] != merged["TEAM_ID"])]
        merged = merged.drop_duplicates(subset=["GAME_ID"], keep="first").reset_index(drop=True)

        recon = self._reconstructed_team_lines(season_type)
        merged = _fill_missing_team_stats(merged, recon, "TEAM", "TEAM_ID")
        merged = _fill_missing_team_stats(merged, recon, "OPP", "OPP_TEAM_ID")
        if "TEAM_EST" not in merged.columns:
            merged["TEAM_EST"] = False
        return merged

    def seasons_together(
        self, player_a_id: int, player_b_id: int, season_type: str = "regular",
        min_minutes: float = DUO_MIN_MINUTES,
    ) -> list[int]:
        """
        Seasons where the two players actually shared a team roster for at
        least one QUALIFYING game that season_type (matched on GAME_ID +
        TEAM_ID, each logging >= min_minutes -- see DUO_MIN_MINUTES) --
        narrower than the mere intersection of each player's individual
        seasons_played (see seasons_played()), which would also include
        seasons they were both merely active in the league on different
        teams. Same "played together" definition as games_together(), so a
        season offered here always yields at least one qualifying game
        there. Cheaper than games_together() -- just an ID join, no
        team-context merge -- since callers only need the season list, not
        full combined stats.
        """
        df = self._load(season_type)
        a = df.loc[df.PLAYER_ID == player_a_id, ["GAME_ID", "TEAM_ID", "SEASON", "MIN_NUM"]]
        b = df.loc[df.PLAYER_ID == player_b_id, ["GAME_ID", "TEAM_ID", "MIN_NUM"]]
        shared = a.merge(b, on=["GAME_ID", "TEAM_ID"], how="inner", suffixes=("", "_b"))
        shared = shared[(shared["MIN_NUM"] >= min_minutes) & (shared["MIN_NUM_b"] >= min_minutes)]
        return sorted(shared["SEASON"].unique().tolist())

    def games_together(
        self, player_a_id: int, player_b_id: int, seasons: list[int], season_type: str,
        min_minutes: float = DUO_MIN_MINUTES,
    ) -> pd.DataFrame:
        """
        Combined per-game rows for games where both players were on the
        SAME team roster for the SAME game (i.e. were teammates that game)
        AND each logged at least min_minutes (see DUO_MIN_MINUTES) -- a
        game where one of them only got token garbage-time/injury-exit
        minutes doesn't count as "played together" and is dropped before
        the box scores are combined. Matched on GAME_ID + TEAM_ID. Built
        from games_with_team_context() for each player, so the result
        carries the same TEAM_*/OPP_* context columns a single player's
        games would, and can be fed into _stat_block() /
        playoffs.compute_series_records() unmodified.

        COUNTING_STATS + MIN_NUM + PLUS_MINUS are summed across the two
        players. Everything else (GAME_ID, SEASON, TEAM_ID, MATCHUP,
        GAME_DATE, WL, TEAM_MIN, TEAM_FGA, TEAM_PTS, OPP_*, ...) is a
        game/team-level fact already identical for both players in a
        shared game, so it's kept as-is from player A's row rather than
        summed (summing it would double-count team/opponent context).

        Also stashes each player's own MIN_NUM/FGA/FTA/TOV (over these
        same qualifying games) under A_*/B_* columns. USG% is nonlinear in
        minutes (minutes is the denominator), so unlike everything else
        here it can't be correctly recomputed from combined totals -- it
        has to be computed per player and then added (see compare.py's
        _compute_usage). Everything else in this frame ignores these
        extra columns.
        """
        a = self.games_with_team_context(player_a_id, seasons, season_type)
        b = self.games_with_team_context(player_b_id, seasons, season_type)
        if a.empty or b.empty:
            return a.iloc[0:0].copy()

        sum_cols = COUNTING_STATS + ["MIN_NUM", "PLUS_MINUS"]
        b_slim = b[["GAME_ID", "TEAM_ID"] + sum_cols]

        merged = a.merge(b_slim, on=["GAME_ID", "TEAM_ID"], how="inner", suffixes=("", "_b"))
        merged = merged[(merged["MIN_NUM"] >= min_minutes) & (merged["MIN_NUM_b"] >= min_minutes)]

        for c in ("MIN_NUM", "FGA", "FTA", "TOV"):
            merged[f"A_{c}"] = merged[c]
            merged[f"B_{c}"] = merged[f"{c}_b"]

        for c in sum_cols:
            merged[c] = merged[c] + merged[f"{c}_b"]
            merged = merged.drop(columns=[f"{c}_b"])
        return merged.reset_index(drop=True)

    def seasons_played(self, player_id: int, season_type: str = "regular") -> list[int]:
        df = self._load(season_type)
        return sorted(df.loc[df.PLAYER_ID == player_id, "SEASON"].unique().tolist())

    def find_player_id(self, name_substring: str, season_type: str = "regular") -> pd.DataFrame:
        """Fuzzy lookup helper: returns matching (PLAYER_ID, PLAYER_NAME) pairs."""
        df = self._load(season_type)
        matches = df[df.PLAYER_NAME.str.contains(name_substring, case=False, na=False)]
        return matches[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates().reset_index(drop=True)

    def all_players(self, season_type: str = "regular") -> pd.DataFrame:
        """Full unique (PLAYER_ID, PLAYER_NAME) directory, sorted by name -- for search UIs."""
        df = self._load(season_type)
        return (
            df[["PLAYER_ID", "PLAYER_NAME"]]
            .drop_duplicates()
            .sort_values("PLAYER_NAME")
            .reset_index(drop=True)
        )