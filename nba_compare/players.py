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
    if not query or len(query) < 2:
        return directory.head(0)
    matches = directory[directory.PLAYER_NAME.str.contains(query, case=False, na=False)]
    return matches.head(limit)