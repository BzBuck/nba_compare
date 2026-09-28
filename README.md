# nba_compare

Compares NBA and WNBA players across arbitrary time spans — full careers,
single seasons, or custom ranges — including regular season and playoff
stats, usage, per-possession rates, team context, and game-to-game
consistency, side by side. Rows can be single players or teammate duos,
restricted to the games two sides played against each other, and drawn
from either league or both at once.

The core idea: the unit being compared is a **span** (one player + one set
of seasons), not a "player." That's what lets you compare two different
players, two different eras of the *same* player, or any mix, through the
exact same code path.

## Setup

```bash
cd nba_compare
pip install -r requirements.txt
```

That's the whole setup — there's no data to download by hand.

## Where the data comes from

Four parquet game-log files per league are pulled from the Hugging Face dataset
[BBuckz/basketball-encyclopedia](https://huggingface.co/datasets/BBuckz/basketball-encyclopedia),
from its `nba/` and `wnba/` folders (same schema, `wnba_` prefix):

| logical name    | file                                 |
|-----------------|--------------------------------------|
| `regular`       | `nba_gamelogs.parquet`               |
| `playoffs`      | `nba_playoffs_gamelogs.parquet`      |
| `team_regular`  | `nba_team_gamelogs.parquet`          |
| `team_playoffs` | `nba_team_playoffs_gamelogs.parquet` |

Each file is downloaded on **first use** and then served from the local
Hugging Face cache (`~/.cache/huggingface`, or wherever `HF_HOME` points),
so only the first run of a fresh machine waits on the network — and once a
file is cached, it loads even with no connection. The download is lazy per
file, not all four up front: a comparison that never opens the playoff tabs
never fetches the playoff logs, and a session that never picks WNBA (or
Both) never fetches any WNBA file. `NBADataStore.from_config()` itself does no
I/O at all.

To publish new data, push it to the dataset repo; clients pick it up on
their next cold start. Nothing in this project needs to change.

### NBA, WNBA, or both

The **League** picker at the top of the app searches NBA players, WNBA
players, or both at once — so an NBA player can sit in the same table as a
WNBA player. Each league gets its own `NBADataStore`
(`NBADataStore.from_config("WNBA")`), and `compare_spans(spans, {"NBA": ...,
"WNBA": ...})` reads every span from its own league's store. That matters
because everything league-wide is keyed by season: percentiles, relative
shooting, standings and playoff structure for a WNBA row are measured
against the WNBA only. Pace is per 48 minutes in the NBA and per 40 in the
WNBA; per-game numbers aren't adjusted, so compare /36 or /100 across
leagues.

Two WNBA data gaps (team minutes and team turnovers recorded as `0` in the
early seasons) are repaired on load — see "Per 100 possessions" below.

### Overriding the source

Both of these are optional environment variables, read by
[`config.py`](nba_compare/config.py):

```bash
# read the parquet files out of a local folder instead of the Hub --
# offline work, or testing a rebuild before it's pushed. The folder can hold
# nba/ and wnba/ subfolders, or one league's files loose.
export NBA_COMPARE_DATA_DIR="../NBA Encyclopedia/data"

# pull a branch, tag, or commit sha other than main -- pin a sha when a
# run has to be reproducible against one version of the data
export NBA_COMPARE_HF_REVISION=main
```

The override folder is expected to use the same filenames the Hub repo does.
`HF_TOKEN` and `HF_HOME` are read by `huggingface_hub` itself; a token is
only needed if the dataset is ever made private.

## Interactive UI

```bash
streamlit run app.py
```

- **League picker** (NBA / WNBA / Both) at the top — which league's players
  the search covers. **Both** lets an NBA and a WNBA player share a table;
  see "NBA, WNBA, or both" above.
- **Search and add players** by typing part of a name; add as many
  player+span rows as you want, including multiple spans of the same
  player (e.g. "prime LeBron" vs. "current LeBron"). **Drag to reorder**
  which column each one appears in across every table (box score, round
  comparisons, series listing) — requires giving spans with the same
  default name distinct labels first, since dragging can't tell apart two
  identically-named columns.
- **Duo mode**: each row can be a single **Player** or a **Duo** — two
  teammates compared as their *combined* numbers for the games they
  actually shared on the floor together, not their individual careers
  added up. A game only counts if both players logged real minutes in it
  (see "Duo mode" below for exactly what that means and how it interacts
  with usage%, playoff series records, and awards).
- **Head-to-head toggle**: with exactly two rows (players or duos), only
  count the games the two sides played *against* each other — both columns
  are built from the same games, and W/L becomes the head-to-head record.
  See "Head-to-head" below.
- **Season range slider** per span (single season, a few years, or full
  career), plus independent toggles for regular season / playoffs.
- **Stat presets**: one click swaps the table to a themed set of rows —
  Traditional, Totals, Per 36, Era-adjusted, Efficiency, Advanced, Team
  impact, Playoff résumé, Consistency — and you can save your own current
  selection as a preset. See "Stat presets" below.
- **Customize stats shown**: pick exactly which stats appear from the full
  catalog below (box score, per-100, usage, team and opponent box scores,
  consistency, percentiles, your own custom formulas), then **drag to
  reorder** them.
- **Custom stat formulas**: sidebar form to define your own stat as a
  formula over existing ones, using the exact stat labels shown in the
  table — e.g. `PTS/G / USG Vol/G` for points per used possession — with
  its own decimal places and a "lower is better" flag for highlighting.
  See "Custom formulas" below for the full variable list and what's
  actually allowed in an expression.
- **Save / Load setup**: sidebar section to save your current players
  and duos, seasons, league, head-to-head setting, stat selection, custom
  formulas and saved presets as a code (or file) you can paste back in
  later, in a different session, after the app's code has changed. See
  "Save / Load" below for why this is safe against future edits.
- **Playoff series breakdown**: expander below the tables with two views —
  **Round-by-round comparison** (rows=stats, columns=spans, best value
  highlighted — same visual style as the main box score table, one
  section per round, so you can directly compare e.g. every span's
  combined Finals performance) and **By player** (each span's series
  listed chronologically). Season/Round/Opponent/Result are always
  shown in the "By player" view; everything else (GP, W, L, and any box
  score stat, plus Home Court and the approximate Seed) is toggleable and
  drag-reorderable — the same picker feeds both views. See "Playoff depth
  & series" below.

Results render as a Stathead-style table — one row per stat, one column
per span, best value in each row highlighted green and (with 3+ columns
being compared) worst highlighted red — split into separate Regular
Season and Playoffs tables. With exactly 2 columns, worst is never
highlighted, since it'd just be "not green" shown louder; ties at either
extreme aren't highlighted either. Short explanatory notes appear under the
tables for whichever stat families are showing (/100, /75, CV%, %ile,
Team/Opp, head-to-head, mixed leagues). An Awards & Honors table appears
too, if you point the sidebar at an accolades CSV (see `accolades.py`).

## Quick test (no UI)

```bash
python smoke_test.py
```

Prints the data source it resolved, then confirms the files load,
`SEASON_ID` parsing works on real data, and the comparison + chart pipeline
runs end to end.

## Folder layout

The project is self-contained — the data lives on the Hub (see "Where the
data comes from"), not in a sibling folder.

```
nba_compare/                        <- project root, cd here to work
    ├── nba_compare/                <- the importable package
    │   ├── __init__.py
    │   ├── models.py                PlayerSpan, DuoSpan
    │   ├── data.py                  NBADataStore -- loads/joins parquet
    │   ├── compare.py               stat computation + N-way comparison
    │   ├── table.py                 Stathead-style table + stat catalog
    │   ├── presets.py               built-in stat presets + their formulas
    │   ├── playoffs.py              series/round/championship identification
    │   ├── percentiles.py           league percentile ranks per season
    │   ├── formulas.py              safe evaluator for custom formulas
    │   ├── session_config.py        save/load format for app setups
    │   ├── players.py               player search helper for the UI
    │   ├── accolades.py             pluggable Awards & Honors source
    │   ├── viz.py                   Plotly charts (library-level, see below)
    │   └── config.py                where the parquet files come from, per league
    ├── app.py                       Streamlit UI, run this
    ├── smoke_test.py                quick end-to-end check
    ├── requirements.txt
    └── README.md
```

The package and the project root share the name `nba_compare` on purpose —
that's what lets `from nba_compare import ...` resolve when you run scripts
from the project root, with nothing to install or put on the path.

## Stat catalog

Everything below lives in `table.STAT_DEFS`, one dict entry per row, so
adding/renaming a stat is a one-line change in `table.py`. All are
toggleable/reorderable in the app; the • ones are on by default. (A • means
something else entirely and never appears here — in the rendered table it
marks a value built from rebuilt team data; see "Per 100 possessions" below.)

**Box score** — •GP, •W, •L, •MIN/G, •PTS/G, •TRB/G, ORB/G, DRB/G, •AST/G, •STL/G,
•BLK/G, •TOV/G, •PF/G, •+/-

**Shooting** — FGM/G, FGA/G, 3PM/G, 3PA/G, FTM/G, FTA/G, •FG%, •3P%, •FT%, •eFG%, •TS%, •TSA/G (true shot attempts =
FGA + .44·FTA, i.e. usage without the turnovers), rFG%/r3P%/rFT%/reFG%/rTS%
(each rate minus the games-weighted league average for the same season(s)
— see "Relative shooting" below)

**Per 100 possessions** — PTS/100, TRB/100, ORB/100, DRB/100, AST/100,
STL/100, BLK/100, TOV/100, PF/100, FGM/100, FGA/100, 3PM/100, 3PA/100,
FTM/100, FTA/100, Poss/G (the player's own estimated possessions per game).
Off by default, like /36. See "Per 100 possessions" below for the method and
the era limits.

**Usage** — •USG%, •USG Vol/G (raw plays used per game, not a %), MIN%
(share of the team's total floor time this player occupied)

**Team context** — Team Poss/G, Team Pace (real two-team pace
formula, not a single-team estimate — see below), Team ORtg, Team DRtg,
Net Rtg (ORtg − DRtg), Team MOV (plain, un-pace-adjusted average scoring
margin), Team W%, Avg Seed (approx) (regular-season-only average of the
same approximate conference seed used in the playoff series breakdown —
see "Playoff depth & series" below for its caveats)

**Team & opponent box score** — Team MIN/G, then the full box line per game
for the player's team and its opponents over the span's games: Team/Opp
PTS/G, TRB/G, ORB/G, DRB/G, AST/G, STL/G, BLK/G, TOV/G, PF/G, FGM/G, FGA/G,
3PM/G, 3PA/G, FTM/G, FTA/G. Labeled like the player's own rows ("Team AST/G"
next to "AST/G") so formulas such as AST% read the way Basketball-Reference
writes them. Each is averaged over the games that have it recorded; a column
rebuilt from player logs gets its own `*`.

**Consistency** — MIN/PTS/TRB/AST/STL/BLK/TOV/3PM/FGM/FTM/TSA/Usage
Vol/TS% CV% (coefficient of variation — see below), plus
MIN/PTS/TRB/AST/STL/BLK/TS% Floor (P10)

**Other** — +/- Std Dev

**Playoff depth** — Championships, Finals Apps, Series W, Series L, Best
Round Reached, Playoff Seasons, Series Missed (Injury) (all span-level
aggregates — see "Playoff depth & series" below for how these are derived)

**League percentiles** — PTS/TRB/AST/STL/BLK/TOV/FG%/3P%/FT%/eFG%/TS% %ile
(see "League percentiles" below)

Presets add more rows on top of these — /36, /75, totals, AST%, TRB%, GmSc
and so on — as formulas rather than built-ins; see "Stat presets" below.

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

### What "relative shooting" (rFG%, r3P%, rFT%, reFG%, rTS%) means

Each shooting rate minus the games-weighted league average for that exact
same set of seasons — e.g. `rTS%` of `+.032` means 3.2 percentage points
better than the league average across the seasons this span covers.
League averages are computed from every game that season (totals summed,
rate computed once — same convention used everywhere else here), not
just qualifying players. Unlike percentiles (below), this is meaningful
for a **Duo** span too, since it's plain subtraction against a league
baseline rather than a rank against a distribution of individual players.

### Per 100 possessions — and what the `*` means

`/100` rows are Basketball-Reference's "Per 100 Poss": the stat per 100 team
possessions the player was on the floor for. A box score never records how
many possessions a player was out there for, so it's the standard estimate —
the team's possessions over these exact games, prorated by the share of the
team's floor time he occupied:

```
team_poss   = FGA − OREB + TOV + .44·FTA      (summed over the span)
player_poss = player_MIN × team_poss / (team_MIN / 5)
per_100     = 100 × stat / player_poss
```

`team_MIN / 5` converts the team's player-minutes (~240 in the NBA, ~200 in
the WNBA) into minutes of game
clock, so `team_poss / (team_MIN/5)` is possessions per minute of game clock
— **this team's own measured pace over these exact games**, not a league
constant and not an era assumption. The one assumption is that the team ran
at its full-game pace while this player was on the floor, which is the same
assumption Basketball-Reference makes for its per-100 table.

That assumption is the whole reason these rows say something `/36` can't:
per-36 is blind to pace, so it reads a 107-possession 1980 Lakers game and a
90-possession 1991 Bulls game as the same amount of playing time.

**Why there's no constant-pace fallback.** The tempting shortcut for the old
seasons is to plug in a fixed league or era pace. Don't: per-100 then becomes
per-36 multiplied by a constant, which adds exactly zero information while
*looking* like a possession adjustment. It also isn't standard —
Basketball-Reference declines to publish pace before 1973-74 for the same
reason. Where the possessions can't be estimated, these rows show `—`.

**Era coverage, and the `*`.** The team logs only carry FGA/FTA/TOV/OREB from
**1985** onward. Before that they're empty, which would blank out every
per-100 row (and Pace/ORtg/DRtg, as it always has). But the *player* logs
reach further back for some of those columns, so where the official team line
is missing, it's rebuilt by summing every player row in that game — see
`data._reconstructed_team_lines`. A column is only rebuilt when every player
row in the game has it, and only when the summed player minutes match the
official team minutes (which is how a partial roster gets rejected, and how
overtime games pass automatically).

| Seasons | Source | Marked |
|---|---|---|
| NBA 1985– | official team box scores | no |
| NBA 1977–1984 | rebuilt from summed player rows | `*` |
| NBA –1976 | not computable at all | shows `—` |
| WNBA 1997, 2000, 2003 | team turnovers rebuilt from summed player rows | `*` |
| WNBA, every other season | official team box scores | no |

The WNBA team logs record team turnovers as `0` for every game of 1997, 2000
and 2003; since no team plays a turnover-free game, those zeros are treated
as missing and rebuilt like the NBA's gaps. The WNBA team logs also carry
team minutes of `0` for 1997–2003 (scrambled through 2004); those are
replaced by the players' summed minutes where they add up to a full game —
see `data._repair_team_lines`. Without that, USG%, per-100 and pace would
divide by zero for the league's first seasons.

Anything resting on a rebuilt line gets a **`*`** next to it in the table, with
a footnote. Two reasons it's a caveat and not just a footnote of pedantry:
coverage inside 1977-1984 is partial (roughly 35% of games in the thinnest
seasons up to ~93% in 1983) and clusters by team, and team turnovers that
aren't charged to any individual player are missing from the sum, which makes
rebuilt possessions run about 0.6% light. Checked against 2020, where these
sums reproduce official team FGA/FTA/OREB/PTS exactly.

Nothing before 1977 is recoverable from any source here: turnovers weren't
recorded (nor offensive rebounds before 1973-74), and without them there is
no possession estimate to make. Wilt's 1971-72 shows `—` on every /100 row —
correctly.

The star is only ever applied to rows that actually come out of the team box
score (`/100`, `Poss/G`, `USG%`, `MIN%`, `USG Vol/G`, and the Team
Pace/ORtg/DRtg/Net Rtg/PTS/Poss rows). A player's own `PTS/G` is official in
every season and never gets marked. `Team MOV` isn't marked either — it needs
only team and opponent points, which the official logs have back to 1946, so
it's computed off its own looser subset and is never an estimate.

**Duo spans** behave for `/100` exactly as they already do for `/36`: summed
stats over summed possessions is a minutes-weighted blend of the two players'
rates, not their combined output. `Poss/G` for a duo is likewise both
players' possessions added, the same way `MIN/G` is.

### Team ORtg / DRtg / Pace — what's real here, and what isn't

**Team-level** ORtg/DRtg/Pace are computed properly. ORtg = 100 × team
points ÷ team possessions; DRtg = 100 × opponent points ÷ opponent
possessions (needs the actual opposing team's box score for that game,
joined by `GAME_ID`). Possessions use the standard single-team estimate
(FGA − OREB + TOV + .44·FTA) applied to each side separately. Pace uses
the standard formula — both sides' possessions, normalized to one
regulation game via the team's actual minutes played (accounting for
overtime): `G × ((team_poss + opp_poss) / (2 × (team_MIN / 5)))`, where
`G` is 48 in the NBA and 40 in the WNBA (`config.GAME_MINUTES`), so each
league's pace reads on its own familiar scale.

These used to come back blank for every season before 1985, since the team
logs have no possession columns that far back. They now fill in for roughly
1977-1984 from rebuilt team lines, marked with a `*` — see "Per 100
possessions" above.

**Individual (player-level) ORtg/DRtg are NOT implemented.** The real
Dean Oliver formula chains together roughly 15 intermediate terms (a
"qualified assist" estimate, team rebound rates, opponent defensive
rebounding, etc.), and small implementation mistakes produce numbers that
look plausible but are wrong. Rather than ship something unvalidated,
this was intentionally left out — ask if you want it built as its own
task, validated against known Basketball-Reference values before trusting it.

### Playoff depth & series

There's no bracket/schedule data source here — series, rounds, and
championships are all inferred from the game logs themselves:

- **Series** = a consecutive run of games against the same opponent for
  one team in one season (opponent parsed from the `MATCHUP` field, which
  the NBA stats API always writes as `"OWN @ OPP"` or `"OWN vs. OPP"`). A
  team never faces two different playoff opponents interleaved within one
  postseason, so this reliably separates series without needing an
  explicit round/series ID.
- **Round number** is the *league* round, not just the team's Nth series:
  walking the postseason's series in the order they start, each one is one
  round past the latest round either team has already played
  (`playoffs._league_rounds`). The two only differ when there are byes — a
  top seed skipping straight to the semifinals (NBA 1950s–60s and 1975–83,
  WNBA 2016–21) plays its first series in round 3, not round 1. Counting per
  team used to leave those seasons with no champion detected at all.
- **Round labels**: NBA seasons with 4 rounds use First Round / Conf Semis /
  Conf Finals / Finals; other NBA formats use Round N / Finals. WNBA rounds
  are named back from the Finals (Finals, Semifinals, then First/Second
  Round), since the WNBA has run 2 to 4 rounds and has had no conference
  rounds since 2016.
- **Championship detection does NOT hardcode "round 4 = Finals."** It
  looks at *every* team's playoff series that season (from the team-level
  parquet) to find the actual deepest round reached league-wide that
  year, then checks whether this team reached it and won. That keeps it
  correct even in historical formats with a different number of rounds —
  tested against a synthetic 2-round bracket where it correctly labeled
  rounds "Round 1"/"Finals" (not the standard 4-round names) and only
  flagged the true championship series, not an earlier round win.
- **Seed is a clearly-flagged APPROXIMATION**, not real seeding: it ranks
  teams by regular-season win% within a hardcoded conference table, one
  per league (`playoffs.CONFERENCES` — the leagues share abbreviations like
  CHA, MIA and SEA for different teams). WNBA seasons from 2016 on are
  ranked league-wide, since the WNBA stopped seeding by conference. It
  does **not** apply real tiebreakers (head-to-head, division record,
  etc.) and doesn't account for NBA play-in games (2020–present). Treat it
  as a rough "how good was this team" signal, not an authoritative seed
  number.
- **Home Court is exact, not an approximation** — whoever hosted Game 1
  of a series had home court advantage for the whole series, by
  definition. That's read straight from the same `MATCHUP` field already
  used for opponent detection, on the series' first game specifically.
  Verified against a synthetic bracket with a team having home court in
  one series and playing on the road in another — both cases came back
  correct.
- **Round-by-round comparison groups by round LABEL, not round number.**
  "Finals" always means the championship round, whether that season had
  4 rounds or 2 — so a span's Finals runs across different-format eras
  still bucket together correctly. If a span reached a round multiple
  times, its column shows the combined performance across every
  appearance (totals summed, then rates recomputed from those sums — not
  an average of each series' rate, which would overweight short series).
  Verified against a career-long span vs. a single-season subset of the
  same player: their one shared Round 1 appearance showed identical
  numbers, while Finals differed correctly since the career span included
  multiple Finals runs the single-season span didn't.
- **Round/championship detection uses TEAM-level games, not just the
  player's own, specifically so injuries don't cost a player their
  championship credit.** Round numbers and results come from
  `team_series_structure()` (the team's full game log, which has no gaps),
  and the player's own games are matched into that structure by
  `GAME_ID` — a round the player has zero games in still produces a
  correct record (marked `player_played: False`, shown as **(DNP)** in
  the app) instead of silently disappearing. That matters twice over: a
  missed round no longer mis-numbers every round after it, and a player
  hurt for the clinching series still gets championship credit if the
  team won it while they were on the roster. Verified against a
  synthetic case where a player only has game rows for Round 1 of a
  3-round bracket their team wins entirely — Round 2 and the Finals both
  showed up correctly labeled, and Champion was correctly `True` on the
  Finals row despite zero player games there. Before this fix, that
  player would have shown zero championships for a season their team
  actually won.
  **Known limitation**: this requires the player to have appeared in at
  least one game for that team that postseason — there's no roster data
  here, only game logs, so a player who missed an *entire* postseason for
  a team can't be picked up this way regardless of the fix.
- **Series Missed (Injury)** counts the DNP series above directly — how
  many of a span's series it got championship/record credit for despite
  having zero box score rows in that specific series.
- **Duo spans use each half's own W/L, not the team's.** `compute_series_records()`
  takes a `wl_from_player_games` flag for this — a Duo's GP/W/L columns
  come from the two players' own combined games (see "Duo mode" below),
  not the team's full series record, so games one half of the duo sat out
  don't get credited or blamed on the pair. **Result** and **Champion**
  always reflect the team's actual series outcome regardless of this flag
  — those are facts about the series, not about who played in it.

### League percentiles

Where a player's per-game value for a stat ranks among every qualifying
player in **their own league, that same season** — computed per season, never
pooled across a span, since league context (pace, 3-point rate) shifts
year to year. A span covering multiple seasons shows a games-weighted
average of that season's percentile; seasons where the player didn't meet
the minimum-games qualifier are excluded from the average entirely, not
counted as a 0.

Covers PTS, TRB, AST, STL, BLK, TOV, FG%, 3P%, FT%, eFG%, TS%. Turnovers
are inverted internally (low TOV → high percentile), so every percentile
column means the same thing: higher is always better, regardless of the
underlying stat's direction — verified against a synthetic 6-player pool
where a low-scoring, low-turnover player correctly landed with a low PTS
percentile but a *high* TOV percentile, independently of each other.

Qualifier: minimum 10 games for regular season, 1 game for playoffs
(series are short, and playoff appearance is already a relevance filter).
Adjustable via `min_games` in `percentiles.season_league_table()`.

**Not shown for a Duo span** — a percentile ranks one value against the
distribution of *individual* players in the league, so a duo's combined
per-game total would trivially read near the 100th percentile against
that same distribution. `aggregate_duo_span()` omits the `percentiles`
key entirely for duo blocks (rendering as "--" in the table) rather than
showing a misleading number. Use the relative shooting stats above
instead for a duo — they subtract a league average rather than ranking
against a player distribution, so they stay meaningful.

**Efficiency**: one grouped calculation per (season, season_type) covers
every player in the league at once (~500 players from one pass over that
season's rows) — it's not done per-player. Results are cached per
`(store, season, season_type, min_games)`, so comparing multiple players
or spans that share a season only pays that cost once. A full multi-player,
multi-season comparison including percentiles ran in well under half a
second in testing.

**Not currently covered by percentiles**: usage%, team-context stats
(ORtg/DRtg/Pace), and the consistency (CV%/Floor) stats. Adding usage%
percentiles specifically would need the team-join run for every league
player each season, not just the stats already summable straight from
player game logs — a real efficiency step up from what's here now. Ask if
you want that built.

## Duo mode

Each row in the app is either a **Player** span or a **Duo** span. A Duo
compares two teammates' *combined* numbers for the games they actually
shared on the floor together — not the two players' individual careers
added up.

- **What counts as "played together"**: same `GAME_ID` + same `TEAM_ID`
  (so they were both on the same roster for the same game), AND each
  player logged at least 10 minutes that game (`data.DUO_MIN_MINUTES`).
  A token garbage-time appearance or an early exit to injury doesn't
  count — it'd technically share a `GAME_ID`/`TEAM_ID` but isn't
  meaningful shared floor time.
- **Season picker** for a Duo shows only seasons where they were actually
  teammates for at least one qualifying game (`NBADataStore.seasons_together`)
  — narrower than intersecting each player's individual seasons played,
  which would also include seasons they were both merely active in the
  league on different teams.
- **What's summed vs. kept as-is**: counting stats (`PTS`, `REB`, `AST`,
  etc.), minutes, and plus-minus are summed across the two players'
  matching games. Game/team-level facts that are already identical for
  both players in a shared game (season, team, matchup, opponent context,
  team totals) are kept as-is rather than summed, which would double-count
  them.
- **USG% is the one stat that can't just sum the raw totals** — minutes
  is USG%'s denominator, so summing minutes first before applying the
  formula would bias the ratio. Each player's own USG% share is computed
  individually (from their own minutes/plays over the same shared games)
  and *those* are added, since usage shares are what actually combine
  additively.
- **Percentiles are omitted for Duo spans** (shown as "--") — a percentile
  ranks against the distribution of individual players, and a duo's
  combined per-game total would trivially read near the 100th percentile
  against that distribution. Relative shooting (rFG%, rTS%, etc.) stays
  meaningful for a Duo instead, since it's just subtraction against a
  league average.
- **Playoff series GP/W/L** reflect the duo's own combined record over
  games they shared, not the team's full series record — see "Duo spans
  use each half's own W/L" under "Playoff depth & series" above.
- **Both players must be from the same league** — a cross-league pair
  shows "Different leagues -- never teammates" in Both mode.
- **Awards & Honors** are season-level data, not game-level, so a Duo's
  row is simply both players' individual awards summed over the span's
  seasons — not scoped to games they shared the way every other stat here is.

## Head-to-head

Turn on **Head-to-head** with exactly two rows (each a player or a duo) and
both columns are rebuilt from only the games the two sides played against
each other.

- **What counts as a meeting**: every player in both rows logged at least
  10 minutes (`data.H2H_MIN_MINUTES`), with the two rows on opposite teams.
  A duo only counts in games where both of its players qualified — the same
  rule as Duo mode.
- **Same games on both sides**: both rows use only the seasons *both* of
  their ranges cover, so neither column can include a meeting the other
  leaves out. W/L over those games is the head-to-head record.
- **Labels**: an auto label is rebuilt for the shared seasons and gains
  "vs. <opponent>"; a label you typed stays as typed.
- **Percentile rows show —** here: they rank whole seasons against the
  league and don't describe these particular games.
- **Playoff series** list only the series the two sides met in, with W/L
  from those games — the team's other series that postseason would
  otherwise show up as "(DNP)".
- Needs two rows with no player in common, from the same league — the app
  explains and falls back to the normal comparison otherwise.

## Stat presets

The **Stat presets** pills above the table swap the rows to a themed set in
one click:

| Preset | What it shows |
|---|---|
| Traditional | the default box score + shooting + usage rows |
| Totals | season-span totals (PTS, TRB, AST, … = per game × GP) |
| Per 36 | every counting stat per 36 minutes |
| Era-adjusted | Poss/G, Team Pace, /75 rates, relative shooting, USG%, %iles |
| Efficiency | shooting splits, 2P%, e3P%, 3PAr, FTr, points-share by shot type, AST/TOV, TOV% |
| Advanced | Basketball-Reference style PTS%/ORB%/DRB%/TRB%/AST%/STL%/BLK%/TOV%, USG%, MIN%, GmSc, PANTS |
| Team impact | Team W%, MOV, ORtg/DRtg/Net, Pace, Team/Opp PTS/G, +/-, MIN%, Avg Seed |
| Playoff résumé | championships, Finals apps, series W/L, best round, missed series |
| Consistency | every Floor (P10) and CV% row, plus +/- Std Dev |

The rows a preset needs that aren't built-in stats (Totals, /36, /75, 2P%,
AST%, GmSc, …) are **custom formulas over the existing stats**, defined in
`presets.PRESET_FORMULAS` — kept as formulas rather than built-ins so they
don't crowd the variable list every custom formula is written against.
Picking a preset adds the formulas it needs (listed in the sidebar under
"Added by preset") and switching to another removes them again; formulas you
added yourself are never touched. The pill stays highlighted while the rows
match a preset exactly, and clears once you edit them by hand.

A few of the formulas worth knowing:

- **/75** = the /100 rate × 0.75 — the same pace adjustment, at roughly a
  starter's per-game possession count.
- **AST%, ORB%, DRB%, TRB%, STL%, BLK%, PTS%** = the player's share of what
  was available while on the floor, using the Team/Opp box score rows and
  the player's share of team floor time (`(Team MIN/G / 5) / MIN/G`).
- **e3P%** = 3P% × 1.5 (a made three on the same footing as eFG%).
- **PANTS** = PTS/G − TSA/G, points beyond one per true shot attempt.
- **GmSc** = Hollinger's Game Score from per-game averages.

**Your own presets**: under "Customize stats shown", name the current rows
and **Save preset** — it keeps the rows in order plus the formulas behind
any of them, so it rebuilds them even in a session that doesn't have them.
Saving under an existing name updates it, a built-in name is refused, and
saved presets travel in your save code.

## Custom formulas

Combine any existing stat, using the exact label shown in the comparison
table (e.g. `PTS/G`, `TS%`, `USG Vol/G`), with `+ - * / **` and
parentheses — e.g. `PTS/G / USG Vol/G` for points per used possession, or
`Team ORtg - Team DRtg` (though that one's already built in as Net Rtg).

This is **not** Python's `eval()` — `formulas.py` first swaps every known
stat label in the expression for a safe placeholder identifier (so a
label with characters that aren't valid in a bare Python identifier, like
`MIN/G`, `TS%`, `+/-`, or `MIN Floor (P10)`, is treated as one atomic
value rather than parsed as an operation), then walks the substituted
expression as an AST that only allows numbers, those placeholders, and
basic arithmetic. Function calls, attribute access, imports, anything
else — none of it exists to execute, so there's no code-injection
surface, even though the input is user-typed.

Available variables are every label in the stat catalog above, shown
exactly as displayed in the table — also listed in the app's "Available
variable names" sidebar expander. Any stat added to `table.STAT_DEFS`
becomes usable in a formula automatically, with no separate variable-name
mapping to maintain. Custom formulas can only reference built-in stats,
not each other, which sidesteps chaining/self-reference/evaluation-order
issues entirely.

Each formula carries its own display format (0–3 decimal places) and a
**Lower is better** flag, so a formula like a turnover rate highlights its
lowest value as the best.

## Save / Load

Sidebar → "Save / Load setup" → **Generate save code** produces a compact
text blob (or downloadable file) capturing your players and duos,
seasons, league (NBA / WNBA / Both), head-to-head setting, stat
selection/order, custom formulas (with their decimals and lower-is-better
flag), saved presets, and accolades path. Paste it back in — in the same
session or a completely different one — and **Load setup** rebuilds
everything, switching the league picker to the one the save was made in.

This is designed to survive future edits to the code, not just work today:

- It's plain JSON (base64-wrapped), not a pickled Python object — a save
  file is just data, not a snapshot of class internals that a refactor
  could invalidate.
- Every field is read with a default, never assumed present — an old save
  missing a field a newer version added just gets a sensible default.
- Stats and custom formulas are re-validated against the *current* code
  before being applied. If a stat gets renamed/removed, or a formula
  references a variable that no longer exists, it's silently dropped (and
  reported to you) instead of crashing the whole load. Saved presets get
  the same check, and one left with no valid rows is dropped.
- The format is versioned (`session_config.CONFIG_VERSION`, currently 3).
  Older saves load unchanged: a save from before league support loads as
  NBA.

## Using it as a library (outside the app)

```python
from nba_compare import PlayerSpan, DuoSpan, NBADataStore, compare_spans, viz

store = NBADataStore.from_config()          # NBA by default; from_config("WNBA") for the WNBA

spans = [
    PlayerSpan.range(201939, "Stephen Curry", 2015, 2016, label="Curry 2015-16 & 2016-17"),
    PlayerSpan.single_season(201939, "Stephen Curry", 2021, label="Curry 2021-22"),
    PlayerSpan.career(2544, "LeBron James", store.seasons_played(2544)),
    DuoSpan(201939, "Stephen Curry", 202691, "Klay Thompson",
            seasons=store.seasons_together(201939, 202691), label="Splash Bros"),
]
result = compare_spans(spans, store)  # DuoSpan rows go through aggregate_duo_span() automatically

result.summary()                          # quick GP/PPG/TS% snapshot, RS vs Playoffs
result.wide_table("regular", "per_game")  # full per-game stat table
result.long_table()                       # tidy format for custom plotting

viz.grouped_bar(result, stats=["PTS", "AST", "REB"], season_type="regular").show()
viz.radar(result, stats=["PTS", "AST", "REB", "STL", "BLK"]).show()
```

To mix leagues, give each span its `league` and pass one store per league —
each span is then read from its own league's store:

```python
stores = {"NBA": NBADataStore.from_config("NBA"), "WNBA": NBADataStore.from_config("WNBA")}
spans = [
    PlayerSpan(2544, "LeBron James", [2012], league="NBA"),
    PlayerSpan(1628932, "A'ja Wilson", [2024], league="WNBA"),
]
result = compare_spans(spans, stores)
```

Head-to-head is the same span with `vs_player_ids` (and optionally
`vs_name`) set — `PlayerSpan(201939, "Stephen Curry", range(2014, 2019),
vs_player_ids=(2544,), vs_name="LeBron James")` — and works for a `DuoSpan`
too.

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
