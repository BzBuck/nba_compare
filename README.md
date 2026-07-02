# nba_compare

Compares NBA players across arbitrary time spans — full careers, single
seasons, or custom ranges — including regular season and playoff stats,
usage, team context, and game-to-game consistency, side by side.

The core idea: the unit being compared is a **span** (one player + one set
of seasons), not a "player." That's what lets you compare two different
players, two different eras of the *same* player, or any mix, through the
exact same code path.

## Setup

```bash
cd "/Users/blakebuckner/Documents/Code/nba_compare"
pip install -r requirements.txt
```

Point `config.py` at your data if the folder layout below doesn't already
match (see "Folder layout").

## Interactive UI

```bash
streamlit run app.py
```

- **Search and add players** by typing part of a name; add as many
  player+span rows as you want, including multiple spans of the same
  player (e.g. "prime LeBron" vs. "current LeBron").
- **Season range slider** per span (single season, a few years, or full
  career), plus independent toggles for regular season / playoffs.
- **Customize stats shown**: pick exactly which stats appear from the full
  catalog below (box score, usage, team context, consistency, your own
  custom formulas), then **drag to reorder** them.
- **Custom stat formulas**: sidebar form to define your own stat as a
  formula over existing ones — e.g. `PTS / USG_VOL_G` for points per used
  possession. See "Custom formulas" below for the full variable list and
  what's actually allowed in an expression.
- **Save / Load setup**: sidebar section to save your current players,
  seasons, and stat selection as a code (or file) you can paste back in
  later, in a different session, after the app's code has changed. See
  "Save / Load" below for why this is safe against future edits.

Results render as a Stathead-style table — one row per stat, one column
per span, best value in each row highlighted — split into separate
Regular Season and Playoffs tables. An Awards & Honors table appears too,
if you point the sidebar at an accolades CSV (see `accolades.py`).

## Quick test (no UI)

```bash
python smoke_test.py
```

Confirms your parquet paths resolve, `SEASON_ID` parsing works on real
data, and the comparison + chart pipeline runs end to end.

## Folder layout

```
Code/
├── NBA Encyclopedia/
│   └── data/
│       ├── nba_gamelogs.parquet
│       ├── nba_playoffs_gamelogs.parquet
│       ├── nba_team_gamelogs.parquet
│       └── nba_team_playoffs_gamelogs.parquet
└── nba_compare/                    <- project root, cd here to work
    ├── nba_compare/                <- the importable package
    │   ├── __init__.py
    │   ├── models.py                PlayerSpan
    │   ├── data.py                  NBADataStore -- loads/joins parquet
    │   ├── compare.py               stat computation + N-way comparison
    │   ├── table.py                 Stathead-style table + stat catalog
    │   ├── formulas.py              safe evaluator for custom formulas
    │   ├── session_config.py        save/load format for app setups
    │   ├── players.py               player search helper for the UI
    │   ├── accolades.py             pluggable Awards & Honors source
    │   ├── viz.py                   Plotly charts (library-level, see below)
    │   └── config.py                DATA_DIR — edit this if your data moves
    ├── app.py                       Streamlit UI, run this
    ├── smoke_test.py                quick end-to-end check
    ├── requirements.txt
    └── README.md
```

The package and the project root share the name `nba_compare` on purpose —
that's what lets `from nba_compare import ...` resolve when you run scripts
from the project root, without needing anything in the shared `Code/` folder.

## Stat catalog

Everything below lives in `table.STAT_DEFS`, one dict entry per row, so
adding/renaming a stat is a one-line change in `table.py`. All are
toggleable/reorderable in the app; the ★ ones are on by default.

**Box score** — ★GP, ★W, ★L, ★MIN/G, ★PTS/G, ★TRB/G, ★AST/G, ★STL/G,
★BLK/G, ★TOV/G, ★PF/G, ★+/-

**Shooting** — ★FG%, ★3P%, ★FT%, ★eFG%, ★TS%, ★TSA/G (true shot attempts =
FGA + .44·FTA, i.e. usage without the turnovers)

**Usage** — ★USG%, ★USG Vol/G (raw plays used per game, not a %), MIN%
(share of the team's total floor time this player occupied)

**Team context** — Team PTS/G, Team Poss/G, Team Pace (real two-team pace
formula, not a single-team estimate — see below), Team ORtg, Team DRtg,
Net Rtg (ORtg − DRtg), Team W%

**Consistency** — MIN/PTS/TRB/AST/STL/BLK/TOV/3PM/FGM/FTM/TSA/Usage
Vol/TS% CV% (coefficient of variation — see below), plus
MIN/PTS/TRB/AST/STL/BLK/TS% Floor (P10)

**Other** — +/- Std Dev

### What CV% actually means

Each stat's CV% is independent — a player's scoring volatility says
nothing about their rebounding volatility, so these are never combined
into one number. CV% = (game-to-game standard deviation ÷ mean) × 100. It
converts "points of swing" into "percent of a player's own average that
they swing by," so a 30-PPG star and a 10-PPG role player become directly
comparable on the same scale — lower means more predictable output game
to game. It's computed per stat independently, and isn't meaningful for
stats that can be zero or swing negative (plus-minus uses a raw standard
deviation instead, for that reason).

### What "Floor (P10)" means

The 10th percentile of the game log, not the true minimum. A single fluky
game (early exit, garbage-time line) shouldn't define "what to expect on
a bad night" — the floor means "about 1 game in 10 is this bad or worse,"
which is a more realistic idea of a bad-night baseline.

### Team ORtg / DRtg / Pace — what's real here, and what isn't

**Team-level** ORtg/DRtg/Pace are computed properly. ORtg = 100 × team
points ÷ team possessions; DRtg = 100 × opponent points ÷ opponent
possessions (needs the actual opposing team's box score for that game,
joined by `GAME_ID`). Possessions use the standard single-team estimate
(FGA − OREB + TOV + .44·FTA) applied to each side separately. Pace uses
the standard NBA formula — both sides' possessions, normalized to a
48-minute game via the team's actual minutes played (accounting for
overtime): `48 × ((team_poss + opp_poss) / (2 × (team_MIN / 5)))`. This
needed team minutes, which wasn't wired into anything until it got added
alongside player MIN/G.

**Individual (player-level) ORtg/DRtg are NOT implemented.** The real
Dean Oliver formula chains together roughly 15 intermediate terms (a
"qualified assist" estimate, team rebound rates, opponent defensive
rebounding, etc.), and small implementation mistakes produce numbers that
look plausible but are wrong. Rather than ship something unvalidated,
this was intentionally left out — ask if you want it built as its own
task, validated against known Basketball-Reference values before trusting it.

## Custom formulas

Combine any existing stat with `+ - * / **` and parentheses — e.g.
`PTS / USG_VOL_G` for points per used possession, or
`TEAM_ORTG - TEAM_DRTG` (though that one's already built in as Net Rtg).

This is **not** Python's `eval()` — `formulas.py` walks the parsed
expression tree and only allows numbers, known variable names, and basic
arithmetic. Function calls, attribute access, imports, anything else —
none of it exists to execute, so there's no code-injection surface, even
though the input is user-typed.

Available variables (also shown in the app's "Available variable names"
sidebar expander): counting stats per game (`PTS`, `REB`, `OREB`, `DREB`,
`AST`, `STL`, `BLK`, `TOV`, `PF`, `FGM`, `FGA`, `FG3M`, `FG3A`, `FTM`,
`FTA`), shooting (`FG_PCT`, `FG3_PCT`, `FT_PCT`, `EFG_PCT`, `TS_PCT`,
`TSA_G`), minutes (`MIN_G`, `MIN_PCT`), plus-minus (`PLUS_MINUS`,
`PLUS_MINUS_STD`), record (`W`, `L`, `GP`, `WIN_PCT`), usage (`USG_PCT`,
`USG_VOL_G`), team context (`TEAM_PTS_G`, `TEAM_POSS_G`, `TEAM_PACE`,
`TEAM_ORTG`, `TEAM_DRTG`, `TEAM_NET_RTG`), consistency (`*_CV` and
`*_FLOOR` for every stat that has one, e.g. `PTS_CV`, `PTS_FLOOR`,
`MIN_CV`, `TS_PCT_CV`).

## Save / Load

Sidebar → "Save / Load setup" → **Generate save code** produces a compact
text blob (or downloadable file) capturing your players, seasons, stat
selection/order, and custom formulas. Paste it back in — in the same
session or a completely different one — and **Load setup** rebuilds
everything.

This is designed to survive future edits to the code, not just work today:

- It's plain JSON (base64-wrapped), not a pickled Python object — a save
  file is just data, not a snapshot of class internals that a refactor
  could invalidate.
- Every field is read with a default, never assumed present — an old save
  missing a field a newer version added just gets a sensible default.
- Stats and custom formulas are re-validated against the *current* code
  before being applied. If a stat gets renamed/removed, or a formula
  references a variable that no longer exists, it's silently dropped (and
  reported to you) instead of crashing the whole load.

## Using it as a library (outside the app)

```python
from nba_compare import PlayerSpan, NBADataStore, compare_spans, viz

store = NBADataStore.from_config()

spans = [
    PlayerSpan.range(201939, "Stephen Curry", 2015, 2016, label="Curry 2015-16 & 2016-17"),
    PlayerSpan.single_season(201939, "Stephen Curry", 2021, label="Curry 2021-22"),
    PlayerSpan.career(2544, "LeBron James", store.seasons_played(2544)),
]
result = compare_spans(spans, store)

result.summary()                          # quick GP/PPG/TS% snapshot, RS vs Playoffs
result.wide_table("regular", "per_game")  # full per-game stat table
result.long_table()                       # tidy format for custom plotting

viz.grouped_bar(result, stats=["PTS", "AST", "REB"], season_type="regular").show()
viz.radar(result, stats=["PTS", "AST", "REB", "STL", "BLK"]).show()
```

`viz.py`'s Plotly charts and `ComparisonResult`'s `summary()`/
`wide_table()`/`long_table()` aren't used by `app.py` (the Streamlit table
view replaced them there), but they're kept as a lighter-weight path for
notebook/script use — `smoke_test.py` exercises this exact path.

## Notes on the underlying stats

- Shooting percentages (FG%, 3P%, FT%, eFG%, TS%) are recomputed from
  summed makes/attempts across a span, not averaged from per-game
  percentages — a 1-for-1 game shouldn't weigh the same as a 10-for-15 game.
- Regular season and playoffs are always kept as separate stat blocks,
  never blended, since playoff sample sizes are much smaller and noisier.
- Usage% is aggregated the way Basketball-Reference does it for
  multi-game spans: sum the raw components across the whole span, then
  apply the formula once — not an average of per-game percentages, which
  would overweight low-minute games.
- Advanced metrics needing full season/league context (Win Shares, BPM,
  VORP) aren't computable from raw box scores alone and aren't included.

## Extending it

Ideas that fit cleanly into this structure:
- Add a `per_100_poss` stat_group alongside `per_game`/`per_36` in `compare.py`.
- Add game-log-level filters to `PlayerSpan` (home/away only, vs. a
  specific opponent, only games the team won).
- Build individual (player-level) ORtg/DRtg — see the caveat above; worth
  validating against known values before trusting it.
- Wire in a real accolades source (`accolades.py` has a stub +
  instructions for scraping Basketball-Reference's Awards tables).