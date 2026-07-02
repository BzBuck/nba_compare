"""
Central place for pointing nba_compare at the NBA Encyclopedia project's
data folder, so the parquet files stay a single source of truth instead
of being copied/duplicated here.

Layout assumed:
  Code/
  ├── NBA Encyclopedia/data/...
  └── nba_compare/            <- project root
      └── nba_compare/        <- this file lives here (the package)
"""
from pathlib import Path

# this file -> package dir -> project root -> Code/ -> sibling "NBA Encyclopedia"/data
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "NBA Encyclopedia" / "data"

REGULAR_PATH = DATA_DIR / "nba_gamelogs.parquet"
PLAYOFF_PATH = DATA_DIR / "nba_playoffs_gamelogs.parquet"
TEAM_REGULAR_PATH = DATA_DIR / "nba_team_gamelogs.parquet"
TEAM_PLAYOFF_PATH = DATA_DIR / "nba_team_playoffs_gamelogs.parquet"