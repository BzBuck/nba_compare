"""
Optional game filters: narrow WHICH games a span's numbers are built from
-- close games only, games against good teams, 30+ minute nights, road
wins, ... -- before any stat is computed. Every table row, custom formula
and playoff series line then describes only the games that passed.

A GameFilter is any number of numeric conditions ("field op value", e.g.
`|Margin| <= 10`, `Opp Seed <= 8`, `MIN >= 30`) plus a few categorical
ones (home/away, win/loss, overtime, specific opponents). All of them must
hold for a game to count. Fields are listed in GAME_FIELDS; adding one is a
single entry there.

Two kinds of field, which matters for duos and head-to-head:
- "player" fields read the row's own box score (a duo's is the two
  players' combined line), except MIN, which a duo must meet EACH -- "both
  played 30+" is what a minutes cutoff means for a pair.
- "game" fields describe the game from the row's team's side (margin,
  opponent's record, ...). In head-to-head, those are read from the first
  row's point of view and the player fields must hold for both rows, so
  the two columns are still built from the exact same games -- see
  compare.compare_spans.

Opponent W% and seed are the opponent's full regular-season record that
season (playoff games included -- a playoff opponent is ranked by its
regular season), and the seed is the same approximation used everywhere
else here (see playoffs.estimate_conference_seed). Opponent conference
reads the same hardcoded conference table, so it reflects today's
alignment -- realignments (Milwaukee moving East in 1980, ...) aren't
modeled, and teams the table doesn't cover fail an East/West choice.

Playoff-only parts (rounds, series home court, closeout/elimination, and
the "playoff" fields) narrow playoff games and leave the regular season
alone -- see playoffs.playoff_game_context for how a game's place in its
series is worked out.

Season qualifiers (GameFilter.season_conditions, fields in SEASON_FIELDS)
are a separate, coarser level: they keep or drop WHOLE seasons -- "only
seasons he played 60% of the schedule", "only 30+ MIN/G seasons" -- judged
on the full regular season before any game filter, so "close games only"
can't knock a season out for being under the GP cutoff. A dropped season
loses its playoff games too. Only availability/role fields are offered:
a production cutoff (PTS/G >= 25) would guarantee the very row it's
compared on. See compare.qualifying_seasons.

There's no starter/bench split: the game logs carry no starting-lineup
column, and inferring it from minutes would be a minutes filter anyway.
"""
from __future__ import annotations
import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import pandas as pd

from .data import NBADataStore
from . import playoffs as _playoffs

OPS: dict[str, Callable] = {
    ">=": operator.ge, "<=": operator.le, ">": operator.gt,
    "<": operator.lt, "=": operator.eq, "!=": operator.ne,
}
OP_SYMBOLS = {">=": "≥", "<=": "≤", ">": ">", "<": "<", "=": "=", "!=": "≠"}

LOCATIONS = ("Any", "Home", "Away")
RESULTS = ("Any", "W", "L")
GAME_LENGTHS = ("Any", "Regulation", "Overtime")
OPP_CONFERENCES = ("Any", "East", "West", "Same", "Other")

# Playoff-only parts. They narrow playoff games and never touch the regular
# season. Rounds are counted back from the Finals (playoffs.
# playoff_game_context), so "Last 4" is the conference finals in a 4-round
# NBA year, the semifinals in the WNBA, and round 2 of a 3-round year alike.
PLAYOFF_ROUNDS = {0: "Finals", 1: "Last 4", 2: "Last 8", 3: "Last 16"}
SERIES_HOME_COURT = ("Any", "With", "Without")
SERIES_SITUATIONS = ("Any", "Closeout", "Elimination", "Either", "Winner-take-all")
OPPONENT_MODES = ("Only", "Exclude")

# A game's team minutes above regulation (5 x game length) by more than
# this count as overtime -- one OT period adds 25 player-minutes.
OT_MINUTES_TOLERANCE = 10.0


class _Ctx:
    """Per-call cache of the derived columns several fields share."""

    def __init__(self, games: pd.DataFrame, store: NBADataStore, season_type: str):
        self.g, self.store, self.season_type = games, store, season_type

    @cached_property
    def opponent(self) -> pd.Series:
        return self.g["MATCHUP"].map(_playoffs._opponent_from_matchup)

    @cached_property
    def _standings(self) -> pd.DataFrame:
        return pd.concat(
            [_playoffs.season_standings(self.store, int(s)) for s in self.g["SEASON"].unique()],
            ignore_index=True,
        ).drop_duplicates(["SEASON", "TEAM_ABBREVIATION"]).set_index(["SEASON", "TEAM_ABBREVIATION"])

    def _standings_for(self, abbrs: pd.Series) -> pd.DataFrame:
        """That season's standings row for each game's team in `abbrs`,
        aligned to self.g's index."""
        keys = pd.MultiIndex.from_arrays([self.g["SEASON"], abbrs])
        return self._standings.reindex(keys).set_axis(self.g.index)

    @cached_property
    def opp_standings(self) -> pd.DataFrame:
        return self._standings_for(self.opponent)

    @cached_property
    def team_standings(self) -> pd.DataFrame:
        """The row's OWN team, game by game -- so a player traded mid-season
        is judged by whichever team each game was for."""
        return self._standings_for(self.g["TEAM_ABBREVIATION"])

    @cached_property
    def rest_days(self) -> pd.Series:
        rest = self.store.team_rest_days(self.season_type)
        keyed = rest.set_index(["GAME_ID", "TEAM_ID"])["REST_DAYS"]
        keys = pd.MultiIndex.from_arrays([self.g["GAME_ID"], self.g["TEAM_ID"]])
        return keyed.reindex(keys).set_axis(self.g.index)

    @cached_property
    def po_context(self) -> pd.DataFrame:
        """Each playoff game's place in its series (playoffs.playoff_game_context),
        aligned to self.g's index."""
        table = pd.concat(
            [_playoffs.playoff_game_context(self.store, int(s)) for s in self.g["SEASON"].unique()],
            ignore_index=True,
        ).drop_duplicates(["GAME_ID", "TEAM_ID"]).set_index(["GAME_ID", "TEAM_ID"])
        keys = pd.MultiIndex.from_arrays([self.g["GAME_ID"], self.g["TEAM_ID"]])
        return table.reindex(keys).set_axis(self.g.index)

    def conference_of(self, abbrs: pd.Series) -> pd.Series:
        table = _playoffs.CONFERENCES[self.store.league]
        lookup = {t: c for c, teams in table.items() for t in teams}
        return abbrs.map(lookup)

    @cached_property
    def margin(self) -> pd.Series:
        if "TEAM_PTS" not in self.g.columns or "OPP_PTS" not in self.g.columns:
            return pd.Series(float("nan"), index=self.g.index)
        return self.g["TEAM_PTS"] - self.g["OPP_PTS"]


@dataclass(frozen=True)
class FieldDef:
    getter: Callable[[_Ctx], pd.Series]
    kind: str                       # "player" or "game" -- see module docstring
    help: str
    default_op: str = ">="
    default_value: float = 0.0
    step: float = 1.0
    fmt: str = "%.0f"
    duo_cols: tuple[str, ...] = ()  # per-player columns a duo must EACH satisfy


def _col(name: str) -> Callable[[_Ctx], pd.Series]:
    return lambda ctx: ctx.g[name] if name in ctx.g.columns else pd.Series(float("nan"), index=ctx.g.index)


GAME_FIELDS: dict[str, FieldDef] = {
    "MIN": FieldDef(_col("MIN_NUM"), "player", "Minutes played. A duo must each meet it.",
                    default_value=30, duo_cols=("A_MIN_NUM", "B_MIN_NUM")),
    "PTS": FieldDef(_col("PTS"), "player", "Points scored (a duo's combined).", default_value=20),
    "TRB": FieldDef(_col("REB"), "player", "Total rebounds.", default_value=10),
    "AST": FieldDef(_col("AST"), "player", "Assists.", default_value=10),
    "STL": FieldDef(_col("STL"), "player", "Steals.", default_value=2),
    "BLK": FieldDef(_col("BLK"), "player", "Blocks.", default_value=2),
    "TOV": FieldDef(_col("TOV"), "player", "Turnovers.", default_op="<=", default_value=3),
    "3PM": FieldDef(_col("FG3M"), "player", "Threes made.", default_value=3),
    "FGA": FieldDef(_col("FGA"), "player", "Field goal attempts.", default_value=15),
    "+/-": FieldDef(_col("PLUS_MINUS"), "player", "Plus-minus (recorded from NBA 1996-97, WNBA 2008).", default_op=">", default_value=0),
    "Margin": FieldDef(lambda ctx: ctx.margin, "game",
                       "Final score margin from this row's team's side: +12 = won by 12, -5 = lost by 5.",
                       default_op=">=", default_value=10),
    "|Margin|": FieldDef(lambda ctx: ctx.margin.abs(), "game",
                         "Final margin either way. ≤ 10 is a close game, ≥ 20 a blowout.",
                         default_op="<=", default_value=10),
    "Opp W%": FieldDef(lambda ctx: ctx.opp_standings["win_pct"], "game",
                       "Opponent's regular-season win% that season.",
                       default_value=0.5, step=0.05, fmt="%.3f"),
    "Opp Seed": FieldDef(lambda ctx: ctx.opp_standings["seed"], "game",
                         "Opponent's approximate conference seed that season (win% rank -- no tiebreakers).",
                         default_op="<=", default_value=8),
    "Team W%": FieldDef(lambda ctx: ctx.team_standings["win_pct"], "game",
                        "This row's own team's regular-season win% that season -- e.g. ≥ .600 for "
                        "only the years on a contender.",
                        default_value=0.5, step=0.05, fmt="%.3f"),
    "Team Seed": FieldDef(lambda ctx: ctx.team_standings["seed"], "game",
                          "This row's own team's approximate conference seed that season (win% rank -- "
                          "no tiebreakers).",
                          default_op="<=", default_value=8),
    "Rest days": FieldDef(lambda ctx: ctx.rest_days, "game",
                          "Full days off since this row's team's previous game -- 0 is a back-to-back. "
                          "A season opener has none, so it fails any rest condition.",
                          default_op="=", default_value=0),
    # "playoff" fields only narrow playoff games -- see filter_mask.
    "Series game #": FieldDef(lambda ctx: ctx.po_context["SERIES_GAME"], "playoff",
                              "Playoffs only: which game of the series -- = 1 for Game 1s, = 7 for Game 7s.",
                              default_op="=", default_value=1),
    "Series lead": FieldDef(lambda ctx: ctx.po_context["SERIES_LEAD"], "playoff",
                            "Playoffs only: this row's team's series wins minus losses before the game -- "
                            "< 0 is trailing, = 0 tied, > 0 leading.",
                            default_op="<", default_value=0),
    "Team PTS": FieldDef(_col("TEAM_PTS"), "game", "Points this row's team scored.", default_value=110),
    "Opp PTS": FieldDef(_col("OPP_PTS"), "game", "Points the opponent scored.", default_op="<=", default_value=100),
}


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: float

    def describe(self) -> str:
        fd = GAME_FIELDS.get(self.field) or SEASON_FIELDS.get(self.field)
        value = (fd.fmt % self.value) if fd else f"{self.value:g}"
        if fd and fd.fmt == "%.3f":
            value = value.removeprefix("0")  # .500, the way W% is written
        return f"{self.field} {OP_SYMBOLS.get(self.op, self.op)} {value}"


@dataclass(frozen=True)
class GameFilter:
    conditions: tuple[Condition, ...] = ()
    location: str = "Any"
    result: str = "Any"
    game_length: str = "Any"
    opponents: tuple[str, ...] = ()     # team abbreviations, as in MATCHUP
    opponents_mode: str = "Only"        # "Only" these opponents, or "Exclude" them
    opp_conference: str = "Any"         # "East"/"West", or "Same"/"Other" as this row's team
    season_conditions: tuple[Condition, ...] = ()  # whole-season qualifiers, SEASON_FIELDS only
    po_rounds: tuple[int, ...] = ()     # PLAYOFF_ROUNDS keys; empty = every round
    po_home_court: str = "Any"          # SERIES_HOME_COURT
    po_situation: str = "Any"           # SERIES_SITUATIONS

    def narrows(self, season_type: str) -> bool:
        """Whether any GAME-level part applies to this season type -- what
        narrows its games within a season, and so what turns off its %ile
        rows. Season qualifiers don't count (they keep whole seasons, which
        percentiles describe fine), and the playoff-only parts don't count
        for the regular season."""
        general = bool(
            self.opponents
            or self.location != "Any" or self.result != "Any" or self.game_length != "Any"
            or self.opp_conference != "Any"
            or any(GAME_FIELDS[c.field].kind != "playoff" for c in self.conditions if c.field in GAME_FIELDS)
        )
        if season_type == "regular":
            return general
        return general or bool(self.conditions or self.po_rounds) or (
            self.po_home_court != "Any" or self.po_situation != "Any"
        )

    @property
    def filters_games(self) -> bool:
        """Whether any game-level part is set, for either season type."""
        return self.narrows("playoffs")

    @property
    def is_active(self) -> bool:
        return self.filters_games or bool(self.season_conditions)

    def describe(self) -> list[str]:
        """One short phrase per active part, for the caption above the tables."""
        out = [f"Seasons with {c.describe()}" for c in self.season_conditions]
        out += [c.describe() for c in self.conditions]
        if self.location != "Any":
            out.append(f"{self.location} games")
        if self.result != "Any":
            out.append("Wins" if self.result == "W" else "Losses")
        if self.game_length != "Any":
            out.append(f"{self.game_length} games")
        if self.opp_conference in ("East", "West"):
            out.append(f"vs. {self.opp_conference}")
        elif self.opp_conference != "Any":
            out.append(f"vs. {self.opp_conference.lower()} conference")
        if self.opponents:
            out.append(f"{'Only vs.' if self.opponents_mode == 'Only' else 'Excluding'} {', '.join(self.opponents)}")
        if self.po_rounds:
            out.append("Playoffs: " + ", ".join(PLAYOFF_ROUNDS[r] for r in sorted(self.po_rounds)))
        if self.po_home_court != "Any":
            out.append(f"Playoffs: {self.po_home_court.lower()} home court")
        if self.po_situation != "Any":
            out.append(f"Playoffs: {self.po_situation.lower()} games")
        return out

    def to_dict(self) -> dict:
        return {
            "conditions": [{"field": c.field, "op": c.op, "value": c.value} for c in self.conditions],
            "season_conditions": [{"field": c.field, "op": c.op, "value": c.value} for c in self.season_conditions],
            "location": self.location,
            "result": self.result,
            "game_length": self.game_length,
            "opponents": list(self.opponents),
            "opponents_mode": self.opponents_mode,
            "opp_conference": self.opp_conference,
            "po_rounds": list(self.po_rounds),
            "po_home_court": self.po_home_court,
            "po_situation": self.po_situation,
        }

    @classmethod
    def from_dict(cls, raw) -> "GameFilter":
        """Lenient, like session_config: anything unrecognized -- a field
        renamed since the save, a bad op, a non-number -- is dropped, never
        raised. Returns an inactive filter for garbage input."""
        if not isinstance(raw, dict):
            return cls()
        def read_conditions(items, fields):
            out = []
            for c in items if isinstance(items, list) else []:
                if not isinstance(c, dict) or c.get("field") not in fields or c.get("op") not in OPS:
                    continue
                try:
                    out.append(Condition(c["field"], c["op"], float(c.get("value"))))
                except (TypeError, ValueError):
                    continue
            return tuple(out)

        conditions = read_conditions(raw.get("conditions"), GAME_FIELDS)
        season_conditions = read_conditions(raw.get("season_conditions"), SEASON_FIELDS)

        def pick(key, allowed):
            return raw.get(key) if raw.get(key) in allowed else allowed[0]

        opponents = raw.get("opponents") or []
        rounds = raw.get("po_rounds") if isinstance(raw.get("po_rounds"), list) else []
        rounds = [r for r in rounds if isinstance(r, int) and not isinstance(r, bool)]
        return cls(
            conditions=conditions,
            season_conditions=season_conditions,
            location=pick("location", LOCATIONS),
            result=pick("result", RESULTS),
            game_length=pick("game_length", GAME_LENGTHS),
            opponents=tuple(o for o in opponents if isinstance(o, str)) if isinstance(opponents, list) else (),
            opponents_mode=pick("opponents_mode", OPPONENT_MODES),
            opp_conference=pick("opp_conference", OPP_CONFERENCES),
            po_rounds=tuple(sorted({r for r in rounds if r in PLAYOFF_ROUNDS})),
            po_home_court=pick("po_home_court", SERIES_HOME_COURT),
            po_situation=pick("po_situation", SERIES_SITUATIONS),
        )


def _holds(values: pd.Series, op: str, value: float) -> pd.Series:
    # A game whose field is unknown (no plus-minus before 1996, no seed for
    # a team the conference table doesn't cover) fails the condition rather
    # than slipping through -- including for "!=", where NaN != x is True.
    return values.notna() & OPS[op](values, value)


def filter_mask(
    games: pd.DataFrame, store: NBADataStore, game_filter: GameFilter, season_type: str,
    players_only: bool = False,
) -> pd.Series:
    """
    Boolean Series over `games`: True where the game passes every part of
    the filter. players_only skips the game fields and categorical parts --
    what the second row of a head-to-head is checked against (see the
    module docstring). The playoff-only parts apply only when season_type
    is "playoffs".
    """
    mask = pd.Series(True, index=games.index)
    if games.empty:
        return mask
    ctx = _Ctx(games, store, season_type)

    for c in game_filter.conditions:
        fd = GAME_FIELDS.get(c.field)
        if fd is None or c.op not in OPS or (players_only and fd.kind != "player"):
            continue
        if fd.kind == "playoff" and season_type != "playoffs":
            continue
        if fd.duo_cols and all(col in games.columns for col in fd.duo_cols):
            for col in fd.duo_cols:
                mask &= _holds(games[col], c.op, c.value)
        else:
            mask &= _holds(fd.getter(ctx), c.op, c.value)

    if players_only:
        return mask

    if game_filter.location != "Any":
        mask &= games["MATCHUP"].map(_playoffs._home_court_from_matchup) == game_filter.location
    if game_filter.result != "Any":
        mask &= games["WL"] == game_filter.result
    if game_filter.game_length != "Any":
        team_min = games["TEAM_MIN"] if "TEAM_MIN" in games.columns else pd.Series(float("nan"), index=games.index)
        # Pre-1963-64 NBA team logs record some team minutes as 0 -- unknown, not short.
        known = team_min > 0
        overtime = team_min > 5 * store.game_minutes + OT_MINUTES_TOLERANCE
        is_regulation = game_filter.game_length == "Regulation"
        mask &= known & (~overtime if is_regulation else overtime)
    if game_filter.opp_conference != "Any":
        opp_conf = ctx.conference_of(ctx.opponent)
        if game_filter.opp_conference in ("East", "West"):
            mask &= opp_conf == game_filter.opp_conference
        else:
            own_conf = ctx.conference_of(games["TEAM_ABBREVIATION"])
            same = opp_conf == own_conf
            mask &= opp_conf.notna() & own_conf.notna() & (same if game_filter.opp_conference == "Same" else ~same)
    if game_filter.opponents:
        listed = ctx.opponent.isin(game_filter.opponents)
        mask &= listed if game_filter.opponents_mode == "Only" else ~listed

    if season_type == "playoffs":
        po = ctx.po_context if (
            game_filter.po_rounds or game_filter.po_home_court != "Any" or game_filter.po_situation != "Any"
        ) else None
        if game_filter.po_rounds:
            mask &= po["ROUNDS_FROM_FINALS"].isin(game_filter.po_rounds)
        if game_filter.po_home_court != "Any":
            # == True / == False, so a game with no series context fails both.
            mask &= po["HOME_COURT"] == (game_filter.po_home_court == "With")
        if game_filter.po_situation != "Any":
            closeout, elim = po["CLOSEOUT"] == True, po["ELIMINATION"] == True  # noqa: E712 -- NaN-safe
            mask &= {
                "Closeout": closeout,
                "Elimination": elim,
                "Either": closeout | elim,
                "Winner-take-all": closeout & elim,
            }[game_filter.po_situation]
    return mask


def apply_filter(
    games: pd.DataFrame, store: NBADataStore, game_filter: GameFilter | None, season_type: str,
    players_only: bool = False,
) -> pd.DataFrame:
    """`games` narrowed to the rows passing the filter's game-level parts
    (unchanged when it has none). Season qualifiers are applied separately,
    by compare.qualifying_seasons."""
    if game_filter is None or not game_filter.narrows(season_type) or games.empty:
        return games
    return games[filter_mask(games, store, game_filter, season_type, players_only)].reset_index(drop=True)


# ---------- season qualifiers ----------

def _schedule_length(store: NBADataStore, season: int) -> int | None:
    """The most regular-season games any team played that season -- the
    schedule a GP% is out of (82, 66 in 1998-99, 34-44 in the WNBA...)."""
    cache_key = f"schedule_len_{season}"
    if cache_key not in store._cache:
        team_games = store.team_games_for_season(season, "regular")
        store._cache[cache_key] = int(team_games.groupby("TEAM_ID").size().max()) if not team_games.empty else None
    return store._cache[cache_key]


def _season_gp_pct(table: pd.DataFrame, store: NBADataStore) -> pd.Series:
    sched = pd.Series({s: _schedule_length(store, int(s)) for s in table.index}, dtype=float)
    return 100 * table["GP"] / sched


# Season-level fields: getter(per-season table, store) -> Series indexed by
# SEASON. The table has GP and MIN/G, plus A_MIN/G and B_MIN/G for a duo.
SEASON_FIELDS: dict[str, FieldDef] = {
    "GP": FieldDef(lambda t, store: t["GP"], "season",
                   "Regular-season games played that season (a duo: games they shared). "
                   "Season lengths differ -- GP% compares across leagues and lockout years.",
                   default_value=50),
    "GP%": FieldDef(_season_gp_pct, "season",
                    "Games played as a % of that season's schedule (82 in most NBA seasons, 66 in "
                    "2011-12, 34-44 in the WNBA) -- 60 drops injury-wrecked seasons.",
                    default_value=60, step=5),
    "MIN/G": FieldDef(lambda t, store: t["MIN/G"], "season",
                      "Minutes per game that season -- 30+ is starter-level. A duo must each meet it.",
                      default_value=30, duo_cols=("A_MIN/G", "B_MIN/G")),
}


def seasons_passing(games: pd.DataFrame, store: NBADataStore, conditions: tuple[Condition, ...]) -> list[int]:
    """
    The seasons in `games` (one span's FULL regular-season games -- see
    compare.qualifying_seasons) that meet every season condition. A season
    with no regular-season games at all can't be judged and doesn't pass.
    """
    if games.empty:
        return []
    by_season = games.groupby("SEASON")
    table = pd.DataFrame({"GP": by_season.size(), "MIN/G": by_season["MIN_NUM"].mean()})
    if "A_MIN_NUM" in games.columns:
        table["A_MIN/G"] = by_season["A_MIN_NUM"].mean()
        table["B_MIN/G"] = by_season["B_MIN_NUM"].mean()
    mask = pd.Series(True, index=table.index)
    for c in conditions:
        fd = SEASON_FIELDS.get(c.field)
        if fd is None or c.op not in OPS:
            continue
        if fd.duo_cols and all(col in table.columns for col in fd.duo_cols):
            for col in fd.duo_cols:
                mask &= _holds(table[col], c.op, c.value)
        else:
            mask &= _holds(fd.getter(table, store), c.op, c.value)
    return sorted(int(s) for s in table.index[mask])
