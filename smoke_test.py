"""
Run from the project root (the folder this file lives in):

    cd "/Users/blakebuckner/Documents/Code/nba_compare"
    python smoke_test.py

Confirms: parquet paths resolve via config.py, SEASON_ID parsing works on
your real data, and the comparison + viz pipeline runs end to end.
"""
from nba_compare import PlayerSpan, NBADataStore, compare_spans, viz

store = NBADataStore.from_config()

# 1. Confirm the parquet files are actually loading and columns match.
matches = store.find_player_id("stephen curry")
print("player lookup:")
print(matches)

if matches.empty:
    raise SystemExit("No matches found — check config.DATA_DIR and your parquet paths.")
if len(matches) > 1:
    print(f"\nWarning: {len(matches)} matches found, using the first. "
          f"Narrow your search string if this isn't who you want.")

player_id = matches.iloc[0].PLAYER_ID
player_name = matches.iloc[0].PLAYER_NAME

# 2. Confirm SEASON_ID parsing works on your real data.
seasons = store.seasons_played(player_id)
print(f"\nseasons found for {player_name}:", seasons)

# 3. Run an actual comparison: full career vs. a specific season.
spans = [
    PlayerSpan.career(player_id, player_name, seasons),
    PlayerSpan.single_season(player_id, player_name, seasons[-1]),
]
result = compare_spans(spans, store)

print("\n--- summary (RS vs Playoffs) ---")
print(result.summary().to_string(index=False))

print("\n--- per-game, regular season ---")
print(result.wide_table("regular", "per_game").round(2))

# 4. Confirm figures build and open one in the browser.
fig = viz.grouped_bar(result, stats=["PTS", "AST", "REB", "STL", "TOV"], season_type="regular")
fig.show()

print("\nAll checks passed.")