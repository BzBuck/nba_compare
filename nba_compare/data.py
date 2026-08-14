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
import pandas as pd

# Counting stats that are safe to SUM across games (rates get recomputed from these).
COUNTING_STATS = [
    "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
]

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


class NBADataStore:
    """
    Lazily loads your parquet files and serves filtered/prepped game logs.
    Only loads a file the first time it's actually needed.
    """

    def __init__(
        self,
        regular_path: str = "nba_gamelogs.parquet",
        playoff_path: str = "nba_playoffs_gamelogs.parquet",
        team_regular_path: str = "nba_team_gamelogs.parquet",
        team_playoff_path: str = "nba_team_playoffs_gamelogs.parquet",
    ):
        self._paths = {
            "regular": regular_path,
            "playoffs": playoff_path,
            "team_regular": team_regular_path,
            "team_playoffs": team_playoff_path,
        }
        self._cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_config(cls) -> "NBADataStore":
        """Convenience constructor using the paths in config.py."""
        from . import config
        return cls(
            regular_path=str(config.REGULAR_PATH),
            playoff_path=str(config.PLAYOFF_PATH),
            team_regular_path=str(config.TEAM_REGULAR_PATH),
            team_playoff_path=str(config.TEAM_PLAYOFF_PATH),
        )

    def _load(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = _prep(pd.read_parquet(self._paths[key]))
        return self._cache[key]

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
        - that game's TEAM totals (TEAM_MIN, TEAM_FGA, TEAM_FTA, TEAM_TOV,
          TEAM_PTS, TEAM_OREB), matched on GAME_ID + TEAM_ID -- for usage%
          and team pace/scoring context.
        - that game's OPPONENT totals (OPP_PTS, OPP_FGA, OPP_FTA, OPP_TOV,
          OPP_OREB), matched on GAME_ID with a different TEAM_ID -- for
          team defensive rating (points allowed per 100 opponent possessions).

        Rows where no match is found get NaN in the relevant columns --
        callers should filter those out before computing with them.
        """
        player_df = self.games(player_id, seasons, season_type)
        team_key = "team_regular" if season_type == "regular" else "team_playoffs"
        team_df = self._load(team_key)

        team_slim = team_df[["GAME_ID", "TEAM_ID", "MIN_NUM", "FGA", "FTA", "TOV", "PTS", "OREB"]].rename(
            columns={
                "MIN_NUM": "TEAM_MIN", "FGA": "TEAM_FGA", "FTA": "TEAM_FTA", "TOV": "TEAM_TOV",
                "PTS": "TEAM_PTS", "OREB": "TEAM_OREB",
            }
        )
        merged = player_df.merge(team_slim, on=["GAME_ID", "TEAM_ID"], how="left")

        opp_slim = team_df[["GAME_ID", "TEAM_ID", "PTS", "FGA", "FTA", "TOV", "OREB"]].rename(
            columns={
                "TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "FGA": "OPP_FGA",
                "FTA": "OPP_FTA", "TOV": "OPP_TOV", "OREB": "OPP_OREB",
            }
        )
        # Join on GAME_ID only (opponent has a different TEAM_ID by definition),
        # then keep just the row where the matched team ISN'T the player's own
        # team -- assumes exactly 2 teams per GAME_ID in the team parquet.
        merged = merged.merge(opp_slim, on="GAME_ID", how="left")
        merged = merged[(merged["OPP_TEAM_ID"].isna()) | (merged["OPP_TEAM_ID"] != merged["TEAM_ID"])]
        merged = merged.drop_duplicates(subset=["GAME_ID"], keep="first").reset_index(drop=True)
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