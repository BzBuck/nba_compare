"""
Accolades (All-Star, All-NBA, MVP shares, championships, Finals MVP) are
NOT in play-by-play/box-score game logs, so this is a pluggable slot.

Two ways to fill this in later:
1. Scrape Basketball Reference's per-player "Awards" tables (fits your
   existing cloudscraper pipeline pattern).
2. Hand-build a small CSV with columns like:
   PLAYER_ID, SEASON, ALL_STAR, ALL_NBA_TIER, ALL_DEFENSE_TIER,
   MVP_SHARE, DPOY_SHARE, CHAMPION, FINALS_MVP

Once you have either, load it into an AccoladeStore and pass it into
compare.aggregate_span (small change: add an `accolades` param that
looks up rows by player_id + season list, same pattern as NBADataStore.games).
"""
from __future__ import annotations
import pandas as pd


class AccoladeStore:
    def __init__(self, csv_path: str | None = None):
        self.df = pd.read_csv(csv_path) if csv_path else pd.DataFrame(
            columns=["PLAYER_ID", "SEASON", "ALL_STAR", "ALL_NBA_TIER",
                     "ALL_DEFENSE_TIER", "MVP_SHARE", "DPOY_SHARE",
                     "CHAMPION", "FINALS_MVP"]
        )

    def block(self, player_id: int, seasons: list[int]) -> dict:
        rows = self.df[(self.df.PLAYER_ID == player_id) & (self.df.SEASON.isin(seasons))]
        if rows.empty:
            return {"all_star": 0, "all_nba": {}, "all_defense": {}, "mvp_shares": 0.0,
                    "championships": 0, "finals_mvp": 0}
        return {
            "all_star": int(rows.ALL_STAR.sum()),
            "all_nba": rows.ALL_NBA_TIER.dropna().value_counts().to_dict(),
            "all_defense": rows.ALL_DEFENSE_TIER.dropna().value_counts().to_dict(),
            "mvp_shares": float(rows.MVP_SHARE.sum()),
            "championships": int(rows.CHAMPION.sum()),
            "finals_mvp": int(rows.FINALS_MVP.sum()),
        }