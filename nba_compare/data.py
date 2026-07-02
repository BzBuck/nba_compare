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