"""
Stat presets: one click swaps the comparison table to a themed set of rows
(Totals, Per 36, Era-adjusted, Advanced, ...).

A preset is just {"name", "stats", "formulas"}: the row labels to show, in
order, plus the custom formulas that produce whichever of those rows aren't
built-in stats. Built-in presets and the ones a user saves share that one
shape, so applying either goes through apply_preset().

Per-36, per-75, totals, shooting splits and the Basketball-Reference style
percentages are deliberately formulas over the existing stats rather than
new built-ins: they'd otherwise pile dozens more names into the variable
list that every custom formula has to be written against. A preset adds its
formulas when it's picked and takes them back out when another one is
(see apply_preset), so they only take up room while they're in use.

User presets are kept in a JSON file on disk (user_presets_path) so they
survive a refresh or restart: read once when a session starts, rewritten
whenever one is saved or deleted. The file holds the same preset shape a
save code carries, and goes through the same lenient reading
(session_config.read_user_presets) plus restore_user_presets' check
against the current stats -- so a file written by an older version of the
app still loads, minus whatever no longer exists.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .formulas import validate_formula
from .session_config import read_user_presets
from .table import STAT_DEFS, DEFAULT_STAT_LABELS

PRESETS_FILE_ENV = "NBA_COMPARE_PRESETS_FILE"
DEFAULT_PRESETS_FILE = Path.home() / ".nba_compare" / "presets.json"
PRESETS_FILE_VERSION = 1

# The format a formula without one displays with -- what every custom
# formula used before formulas could carry their own.
DEFAULT_FORMULA_FMT = "{:.3f}"


def _f(expr: str, fmt: str = "{:.1f}", lower: bool = False) -> dict:
    return {"expr": expr, "fmt": fmt, "lower": lower}


# (row label, player /G label, /100 label, lower is better) for the counting
# stats that get Totals, Per 36 and Per 75 versions.
_COUNTING = [
    ("PTS", "PTS/G", "PTS/100", False),
    ("TRB", "TRB/G", "TRB/100", False),
    ("ORB", "ORB/G", "ORB/100", False),
    ("DRB", "DRB/G", "DRB/100", False),
    ("AST", "AST/G", "AST/100", False),
    ("STL", "STL/G", "STL/100", False),
    ("BLK", "BLK/G", "BLK/100", False),
    ("TOV", "TOV/G", "TOV/100", True),
    ("PF", "PF/G", "PF/100", True),
    ("FGM", "FGM/G", "FGM/100", False),
    ("FGA", "FGA/G", "FGA/100", False),
    ("3PM", "3PM/G", "3PM/100", False),
    ("3PA", "3PA/G", "3PA/100", False),
    ("FTM", "FTM/G", "FTM/100", False),
    ("FTA", "FTA/G", "FTA/100", False),
]

# Minutes the player's team spent on the floor per player slot, i.e. the
# denominator in every Basketball-Reference "share of the floor" percentage.
_FLOOR = "(Team MIN/G / 5) / MIN/G"

# label -> formula spec for every row a built-in preset shows that isn't in
# STAT_DEFS. One shared registry, so two presets showing "TOV%" are showing
# the same formula.
PRESET_FORMULAS: dict[str, dict] = {
    # Totals -- per game times games played.
    "MIN": _f("MIN/G * GP", "{:.0f}"),
    **{label: _f(f"{per_game} * GP", "{:.0f}", lower) for label, per_game, _, lower in _COUNTING},
    # Per 36 minutes.
    **{f"{label}/36": _f(f"{per_game} * 36 / MIN/G", "{:.1f}", lower)
       for label, per_game, _, lower in _COUNTING},
    # Per 75 possessions -- the per-100 rate scaled to a more typical
    # single-player possession count, the common era-adjusted baseline.
    **{f"{label}/75": _f(f"{per_100} * 0.75", "{:.1f}", lower)
       for label, _, per_100, lower in _COUNTING},
    # Shot diet and scoring efficiency.
    "2P%": _f("(FGM/G - 3PM/G) / (FGA/G - 3PA/G)", "{:.3f}"),
    "2PA/G": _f("FGA/G - 3PA/G"),
    "3PAr": _f("3PA/G / FGA/G", "{:.3f}"),
    "FTr": _f("FTA/G / FGA/G", "{:.3f}"),
    "2P PTS Share": _f("2 * (FGM/G - 3PM/G) / PTS/G", "{:.3f}"),
    "3P PTS Share": _f("3 * 3PM/G / PTS/G", "{:.3f}"),
    "FT PTS Share": _f("FTM/G / PTS/G", "{:.3f}"),
    "AST/TOV": _f("AST/G / TOV/G", "{:.2f}"),
    "TOV%": _f("100 * TOV/G / (FGA/G + 0.44 * FTA/G + TOV/G)", lower=True),
    # 3P% on the same footing as eFG%: a made three is worth 1.5 made twos.
    "e3P%": _f("3P% * 1.5", "{:.3f}"),
    # Points scored beyond one per true shot attempt.
    "PANTS": _f("PTS/G - TSA/G"),
    # Basketball-Reference advanced percentages -- the player's share of
    # what was available while they were on the floor.
    "AST%": _f("100 * AST/G / (MIN/G / (Team MIN/G / 5) * Team FGM/G - FGM/G)"),
    "ORB%": _f(f"100 * ORB/G * {_FLOOR} / (Team ORB/G + Opp DRB/G)"),
    "DRB%": _f(f"100 * DRB/G * {_FLOOR} / (Team DRB/G + Opp ORB/G)"),
    "TRB%": _f(f"100 * TRB/G * {_FLOOR} / (Team TRB/G + Opp TRB/G)"),
    "STL%": _f(f"100 * STL/G * {_FLOOR} / (Opp FGA/G - Opp ORB/G + Opp TOV/G + 0.44 * Opp FTA/G)"),
    "BLK%": _f(f"100 * BLK/G * {_FLOOR} / (Opp FGA/G - Opp 3PA/G)"),
    "PTS%": _f(f"100 * PTS/G * {_FLOOR} / Team PTS/G"),
    # Win Shares, offense plus defense (see win_shares.py).
    "WS": _f("OWS + DWS"),
    # John Hollinger's Game Score, from per-game averages.
    "GmSc": _f(
        "PTS/G + 0.4 * FGM/G - 0.7 * FGA/G - 0.4 * (FTA/G - FTM/G) + 0.7 * ORB/G"
        " + 0.3 * DRB/G + STL/G + 0.7 * AST/G + 0.7 * BLK/G - 0.4 * PF/G - TOV/G"
    ),
}

_SHOOTING = ["FG%", "3P%", "FT%", "TS%"]

# name -> row labels, in display order. Dict order is the order the
# presets are offered in.
_PRESET_STATS: dict[str, list[str]] = {
    "Traditional": list(DEFAULT_STAT_LABELS),
    "Totals": ["GP", "W", "L", "MIN"] + [c[0] for c in _COUNTING] + _SHOOTING,
    "Per 36": ["GP", "MIN/G"] + [f"{c[0]}/36" for c in _COUNTING] + _SHOOTING,
    "Era-adjusted": [
        "GP", "Poss/G", "Team Pace",
        "PTS/75", "TRB/75", "AST/75", "STL/75", "BLK/75", "TOV/75", "3PA/75", "FTA/75",
        "rTS%", "reFG%", "r3P%", "rFT%", "USG%",
        "PTS %ile", "TRB %ile", "AST %ile", "TS% %ile",
    ],
    "Efficiency": [
        "TS%", "eFG%", "FG%", "2P%", "3P%", "e3P%", "FT%", "rTS%", "reFG%",
        "TSA/G", "USG%", "2PA/G", "3PA/G", "FTA/G", "3PAr", "FTr",
        "2P PTS Share", "3P PTS Share", "FT PTS Share", "AST/TOV", "TOV%",
    ],
    "Advanced": [
        "GP", "MIN/G", "TS%", "eFG%", "3PAr", "FTr",
        "PTS%", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "MIN%",
        "GmSc", "PANTS", "OWS", "DWS", "WS", "WS/48",
    ],
    "Team impact": [
        "GP", "W", "L", "Team W%", "Team MOV", "Team ORtg", "Team DRtg", "Net Rtg",
        "Team Pace", "Team PTS/G", "Opp PTS/G", "+/-", "MIN%", "Avg Seed (approx)",
    ],
    "Playoff résumé": [
        "GP", "Championships", "Finals Apps", "Series W", "Series L",
        "Best Round Reached", "Playoff Seasons", "Series Missed (Injury)",
    ],
    "Consistency": (
        [label for label in STAT_DEFS if "Floor (P10)" in label]
        + [label for label in STAT_DEFS if label.endswith("CV%")]
        + ["+/- Std Dev"]
    ),
}


def _formula_entry(label: str) -> dict:
    return {"label": label, **PRESET_FORMULAS[label]}


BUILTIN_PRESETS: list[dict] = [
    {
        "name": name,
        "stats": stats,
        "formulas": [_formula_entry(s) for s in stats if s not in STAT_DEFS],
    }
    for name, stats in _PRESET_STATS.items()
]
BUILTIN_PRESET_NAMES = {p["name"] for p in BUILTIN_PRESETS}


def _check_builtins() -> None:
    """Fail at import, not in front of the user, if a preset names a row
    that doesn't exist or a formula that doesn't parse against STAT_DEFS."""
    sample = {v: 1.0 for v in STAT_DEFS}
    for label, spec in PRESET_FORMULAS.items():
        if label in STAT_DEFS:
            raise ValueError(f"preset formula {label!r} shadows a built-in stat")
        error = validate_formula(spec["expr"], sample)
        if error:
            raise ValueError(f"preset formula {label!r}: {error}")
    for name, stats in _PRESET_STATS.items():
        unknown = [s for s in stats if s not in STAT_DEFS and s not in PRESET_FORMULAS]
        if unknown:
            raise ValueError(f"preset {name!r} names unknown stats: {unknown}")


_check_builtins()


def apply_preset(preset: dict, custom_formulas: list[dict]) -> tuple[list[dict], list[str]]:
    """
    (new custom_formulas, new stat order) for switching to `preset`.

    Formulas a preset adds are marked "managed". Switching presets drops
    every managed formula the new preset doesn't use, so they don't pile
    up in the sidebar -- but formulas the user added by hand are never
    touched. If the user already has an unmanaged formula under a label
    the preset wants, theirs is kept as-is rather than overwritten; a
    managed one under that label is replaced, since it only came from an
    earlier preset.
    """
    wanted = {f["label"]: f for f in preset["formulas"] if f["label"] in preset["stats"]}
    kept = [f for f in custom_formulas if not f.get("managed")]
    kept_labels = {f["label"] for f in kept}
    added = [
        {**wanted[label], "managed": True}
        for label in preset["stats"] if label in wanted and label not in kept_labels
    ]
    new_formulas = kept + added
    available = set(STAT_DEFS) | {f["label"] for f in new_formulas}
    return new_formulas, [s for s in preset["stats"] if s in available]


def matching_preset(stat_order: list[str], presets: list[dict]) -> str | None:
    """Name of the preset whose rows are exactly what's showing now, if any
    -- so the picker can show which one is active until the user edits it."""
    for preset in presets:
        if list(preset["stats"]) == list(stat_order):
            return preset["name"]
    return None


def preset_from_current(name: str, stat_order: list[str], custom_formulas: list[dict]) -> dict:
    """A user preset capturing what's showing now, including the custom
    formulas behind any of those rows (managed or not), so loading it later
    rebuilds them even in a session that doesn't have them."""
    shown = set(stat_order)
    return {
        "name": name,
        "stats": list(stat_order),
        "formulas": [
            {k: v for k, v in f.items() if k != "managed"}
            for f in custom_formulas if f["label"] in shown
        ],
    }


def restore_user_presets(presets: list[dict]) -> tuple[list[dict], list[str]]:
    """
    (presets still usable, names dropped) for user presets read from a save
    code or the presets file. Each preset's formulas are re-validated
    against the current formula engine and its rows narrowed to ones that
    still exist; a preset left with no rows, or named like a built-in
    (which always wins), is dropped. A later preset with the same name
    replaces an earlier one.
    """
    sample_vars = {v: 1.0 for v in STAT_DEFS}
    kept: dict[str, dict] = {}
    dropped = []
    for p in presets:
        formulas = [f for f in p["formulas"] if not validate_formula(f["expr"], sample_vars)]
        valid = set(STAT_DEFS) | {f["label"] for f in formulas}
        stats = [x for x in p["stats"] if x in valid]
        if not stats or p["name"] in BUILTIN_PRESET_NAMES:
            dropped.append(p["name"])
            continue
        kept.pop(p["name"], None)  # re-insert, so a replacement takes the later spot
        kept[p["name"]] = {**p, "stats": stats, "formulas": formulas}
    return list(kept.values()), dropped


def user_presets_path() -> Path:
    """Where user presets are kept: $NBA_COMPARE_PRESETS_FILE, or
    ~/.nba_compare/presets.json."""
    override = os.environ.get(PRESETS_FILE_ENV)
    return Path(override).expanduser() if override else DEFAULT_PRESETS_FILE


def presets_to_json(presets: list[dict]) -> str:
    return json.dumps({"version": PRESETS_FILE_VERSION, "presets": presets}, indent=2)


def presets_from_json(text: str | bytes) -> tuple[list[dict], list[str]]:
    """(presets, names dropped) from a presets file's contents -- the
    {"version", "presets"} object presets_to_json writes, or a bare list.
    Raises ValueError, with a message safe to show, if it isn't JSON."""
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Not a presets file ({e}).") from e
    if isinstance(raw, dict):
        raw = raw.get("presets", [])
    return restore_user_presets(read_user_presets(raw))


def load_user_presets(path: Path | None = None) -> tuple[list[dict], list[str], str | None]:
    """(presets, names dropped, error message) from the presets file. A
    missing file is just no presets yet; an unreadable one is reported and
    left alone, so it isn't overwritten before someone can look at it."""
    path = path or user_presets_path()
    if not path.exists():
        return [], [], None
    try:
        presets, dropped = presets_from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [], [], f"Couldn't read saved presets from {path}: {e}"
    return presets, dropped, None


def save_user_presets(presets: list[dict], path: Path | None = None) -> str | None:
    """Writes the presets file, replacing it in one step (a crash mid-write
    can't leave half a file). Returns an error message, or None."""
    path = path or user_presets_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".presets-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(presets_to_json(presets))
        os.replace(tmp, path)
    except OSError as e:
        return f"Couldn't save presets to {path}: {e}"
    return None
