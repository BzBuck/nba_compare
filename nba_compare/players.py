"""
Lightweight player search for the UI: loads the unique player directory
once and filters it as the user types.
"""
from __future__ import annotations
import pandas as pd
from .data import NBADataStore


def player_directory(store: NBADataStore, season_type: str = "regular") -> pd.DataFrame:
    return store.all_players(season_type)


def search_players(query: str, directory: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Up to `limit` players whose name contains `query`, most career minutes
    first (see NBADataStore.all_players). Sorted here as well as there, since
    a two-league directory is two sorted ones stacked."""
    if not query or len(query) < 2:
        return directory.head(0)
    matches = directory[directory.PLAYER_NAME.str.contains(query, case=False, na=False)]
    if "CAREER_MIN" in matches.columns:
        matches = matches.sort_values(["CAREER_MIN", "CAREER_GP"], ascending=False, kind="stable")
    return matches.head(limit)