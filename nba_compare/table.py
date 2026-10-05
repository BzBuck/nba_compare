"""
Builds a Basketball-Reference/Stathead-style comparison table: one row per
stat, one column per compared span, with the best value in each row
highlighted -- rather than the grouped-bar-chart approach.

Kept separate from viz.py since this renders as HTML/dataframe, not Plotly.
"""
from __future__ import annotations
import html
import pandas as pd
from .compare import ComparisonResult
from .data import TEAM_BOX_STATS
from .models import DuoSpan

# label -> (getter(stat_block) -> value, display format, lower_is_better)
# A dict (not a list) so the UI can let the user pick + reorder a subset by
# label. Consistency (CV%) rows live in here too, not a separate table --
# they're just more rows to toggle, same as any box score stat.
STAT_DEFS = {
    "GP":         (lambda b: b["games"],                               "{:.0f}", False),
    "W":          (lambda b: b["wins"],                                "{:.0f}", False),
    "L":          (lambda b: b["losses"],                              "{:.0f}", True),
    "MIN/G":      (lambda b: b["minutes_per_game"],                     "{:.1f}", False),
    "PTS/G":      (lambda b: b["per_game"]["PTS"],                      "{:.1f}", False),
    "TRB/G":      (lambda b: b["per_game"]["REB"],                      "{:.1f}", False),
    "ORB/G":      (lambda b: b["per_game"]["OREB"],                     "{:.1f}", False),
    "DRB/G":      (lambda b: b["per_game"]["DREB"],                     "{:.1f}", False),
    "AST/G":      (lambda b: b["per_game"]["AST"],                      "{:.1f}", False),
    "STL/G":      (lambda b: b["per_game"]["STL"],                      "{:.1f}", False),
    "BLK/G":      (lambda b: b["per_game"]["BLK"],                      "{:.1f}", False),
    "TOV/G":      (lambda b: b["per_game"]["TOV"],                      "{:.1f}", True),
    "PF/G":       (lambda b: b["per_game"]["PF"],                       "{:.1f}", True),
    "+/-":        (lambda b: b["plus_minus_per_game"],                  "{:+.1f}", False),
    "FGM/G":      (lambda b: b["per_game"]["FGM"],                       "{:.1f}", False),
    "FGA/G":      (lambda b: b["per_game"]["FGA"],                       "{:.1f}", False),
    "3PM/G":      (lambda b: b["per_game"]["FG3M"],                      "{:.1f}", False),
    "3PA/G":      (lambda b: b["per_game"]["FG3A"],                      "{:.1f}", False),
    "FTM/G":      (lambda b: b["per_game"]["FTM"],                       "{:.1f}", False),
    "FTA/G":      (lambda b: b["per_game"]["FTA"],                       "{:.1f}", False),
    "PTS/100":    (lambda b: (b.get("per_100") or {}).get("PTS"),         "{:.1f}", False),
    "TRB/100":    (lambda b: (b.get("per_100") or {}).get("REB"),         "{:.1f}", False),
    "ORB/100":    (lambda b: (b.get("per_100") or {}).get("OREB"),        "{:.1f}", False),
    "DRB/100":    (lambda b: (b.get("per_100") or {}).get("DREB"),        "{:.1f}", False),
    "AST/100":    (lambda b: (b.get("per_100") or {}).get("AST"),         "{:.1f}", False),
    "STL/100":    (lambda b: (b.get("per_100") or {}).get("STL"),         "{:.1f}", False),
    "BLK/100":    (lambda b: (b.get("per_100") or {}).get("BLK"),         "{:.1f}", False),
    "TOV/100":    (lambda b: (b.get("per_100") or {}).get("TOV"),         "{:.1f}", True),
    "PF/100":     (lambda b: (b.get("per_100") or {}).get("PF"),          "{:.1f}", True),
    "FGM/100":    (lambda b: (b.get("per_100") or {}).get("FGM"),         "{:.1f}", False),
    "FGA/100":    (lambda b: (b.get("per_100") or {}).get("FGA"),         "{:.1f}", False),
    "3PM/100":    (lambda b: (b.get("per_100") or {}).get("FG3M"),        "{:.1f}", False),
    "3PA/100":    (lambda b: (b.get("per_100") or {}).get("FG3A"),        "{:.1f}", False),
    "FTM/100":    (lambda b: (b.get("per_100") or {}).get("FTM"),         "{:.1f}", False),
    "FTA/100":    (lambda b: (b.get("per_100") or {}).get("FTA"),         "{:.1f}", False),
    "Poss/G":     (lambda b: (b.get("possessions") or {}).get("player_poss_per_game"), "{:.1f}", False),
    "FG%":        (lambda b: b["shooting"]["FG_PCT"],                   "{:.3f}", False),
    "3P%":        (lambda b: b["shooting"]["FG3_PCT"],                  "{:.3f}", False),
    "FT%":        (lambda b: b["shooting"]["FT_PCT"],                   "{:.3f}", False),
    "eFG%":       (lambda b: b["shooting"]["EFG_PCT"],                  "{:.3f}", False),
    "TS%":        (lambda b: b["shooting"]["TS_PCT"],                   "{:.3f}", False),
    "rFG%":       (lambda b: (b.get("relative_shooting") or {}).get("FG_PCT"),  "{:+.3f}", False),
    "r3P%":       (lambda b: (b.get("relative_shooting") or {}).get("FG3_PCT"), "{:+.3f}", False),
    "rFT%":       (lambda b: (b.get("relative_shooting") or {}).get("FT_PCT"),  "{:+.3f}", False),
    "reFG%":      (lambda b: (b.get("relative_shooting") or {}).get("EFG_PCT"), "{:+.3f}", False),
    "rTS%":       (lambda b: (b.get("relative_shooting") or {}).get("TS_PCT"),  "{:+.3f}", False),
    "TSA/G":      (lambda b: b["tsa_per_game"],                         "{:.1f}", False),
    "MIN%":       (lambda b: (b["usage"] or {}).get("min_pct"),         "{:.1f}", False),
    "USG%":       (lambda b: (b["usage"] or {}).get("usg_pct"),         "{:.1f}", False),
    "USG Vol/G":  (lambda b: (b["usage"] or {}).get("usage_per_game"),  "{:.1f}", False),
    "Team Poss/G": (lambda b: (b["team"] or {}).get("team_poss_per_game"), "{:.1f}", False),
    "Team Pace":  (lambda b: (b["team"] or {}).get("team_pace"),        "{:.1f}", False),
    "Team ORtg":  (lambda b: (b["team"] or {}).get("team_ortg"),        "{:.1f}", False),
    "Team DRtg":  (lambda b: (b["team"] or {}).get("team_drtg"),        "{:.1f}", True),
    "Net Rtg":    (lambda b: (b["team"] or {}).get("team_net_rtg"),     "{:+.1f}", False),
    "Team MOV":   (lambda b: (b["team"] or {}).get("team_mov"),         "{:+.1f}", False),
    "Team W%":    (lambda b: (b["wins"] / (b["wins"] + b["losses"]))
                             if b["wins"] is not None and (b["wins"] + b["losses"]) > 0 else None, "{:.3f}", False),
    "Avg Seed (approx)": (lambda b: b.get("avg_seed"),                  "{:.1f}", True),
    "MIN Floor (P10)": (lambda b: b["consistency"]["MIN"]["floor"],     "{:.1f}", False),
    "PTS Floor (P10)": (lambda b: b["consistency"]["PTS"]["floor"],     "{:.1f}", False),
    "TRB Floor (P10)": (lambda b: b["consistency"]["REB"]["floor"],     "{:.1f}", False),
    "AST Floor (P10)": (lambda b: b["consistency"]["AST"]["floor"],     "{:.1f}", False),
    "STL Floor (P10)": (lambda b: b["consistency"]["STL"]["floor"],     "{:.1f}", False),
    "BLK Floor (P10)": (lambda b: b["consistency"]["BLK"]["floor"],     "{:.1f}", False),
    "TS% Floor (P10)": (lambda b: b["consistency"]["TS_PCT"]["floor"],  "{:.3f}", False),
    "MIN CV%":    (lambda b: b["consistency"]["MIN"]["cv_pct"],         "{:.1f}", True),
    "PTS CV%":    (lambda b: b["consistency"]["PTS"]["cv_pct"],         "{:.1f}", True),
    "TRB CV%":    (lambda b: b["consistency"]["REB"]["cv_pct"],         "{:.1f}", True),
    "AST CV%":    (lambda b: b["consistency"]["AST"]["cv_pct"],         "{:.1f}", True),
    "STL CV%":    (lambda b: b["consistency"]["STL"]["cv_pct"],         "{:.1f}", True),
    "BLK CV%":    (lambda b: b["consistency"]["BLK"]["cv_pct"],         "{:.1f}", True),
    "TOV CV%":    (lambda b: b["consistency"]["TOV"]["cv_pct"],         "{:.1f}", True),
    "3PM CV%":    (lambda b: b["consistency"]["FG3M"]["cv_pct"],        "{:.1f}", True),
    "FGM CV%":    (lambda b: b["consistency"]["FGM"]["cv_pct"],         "{:.1f}", True),
    "FTM CV%":    (lambda b: b["consistency"]["FTM"]["cv_pct"],         "{:.1f}", True),
    "TSA CV%":    (lambda b: b["consistency"]["TSA"]["cv_pct"],         "{:.1f}", True),
    "Usage Vol CV%": (lambda b: b["consistency"]["USG_EVENTS"]["cv_pct"], "{:.1f}", True),
    "TS% CV%":    (lambda b: b["consistency"]["TS_PCT"]["cv_pct"],      "{:.1f}", True),
    "+/- Std Dev": (lambda b: b["plus_minus_std"],                      "{:.1f}", True),
    "Championships": (lambda b: (b.get("depth") or {}).get("championships"),   "{:.0f}", False),
    "Finals Apps": (lambda b: (b.get("depth") or {}).get("finals_apps"),       "{:.0f}", False),
    "Series W":   (lambda b: (b.get("depth") or {}).get("series_w"),           "{:.0f}", False),
    "Series L":   (lambda b: (b.get("depth") or {}).get("series_l"),           "{:.0f}", True),
    "Best Round Reached": (lambda b: (b.get("depth") or {}).get("best_round_num"), "{:.0f}", False),
    "Playoff Seasons": (lambda b: (b.get("depth") or {}).get("seasons_in_playoffs"), "{:.0f}", False),
    "Series Missed (Injury)": (lambda b: (b.get("depth") or {}).get("series_missed_to_injury"), "{:.0f}", True),
    "PTS %ile":   (lambda b: (b.get("percentiles") or {}).get("PTS"),               "{:.0f}", False),
    "TRB %ile":   (lambda b: (b.get("percentiles") or {}).get("REB"),               "{:.0f}", False),
    "AST %ile":   (lambda b: (b.get("percentiles") or {}).get("AST"),               "{:.0f}", False),
    "STL %ile":   (lambda b: (b.get("percentiles") or {}).get("STL"),               "{:.0f}", False),
    "BLK %ile":   (lambda b: (b.get("percentiles") or {}).get("BLK"),               "{:.0f}", False),
    "TOV %ile":   (lambda b: (b.get("percentiles") or {}).get("TOV"),               "{:.0f}", False),
    "FG% %ile":   (lambda b: (b.get("percentiles") or {}).get("FG_PCT"),            "{:.0f}", False),
    "3P% %ile":   (lambda b: (b.get("percentiles") or {}).get("FG3_PCT"),           "{:.0f}", False),
    "FT% %ile":   (lambda b: (b.get("percentiles") or {}).get("FT_PCT"),            "{:.0f}", False),
    "eFG% %ile":  (lambda b: (b.get("percentiles") or {}).get("EFG_PCT"),           "{:.0f}", False),
    "TS% %ile":   (lambda b: (b.get("percentiles") or {}).get("TS_PCT"),            "{:.0f}", False),
}
# The team's and opponent's full box score per game (compare._compute_team_box),
# labeled the same way as the player's own rows -- "Team AST/G" next to
# "AST/G" -- so a formula like AST% reads the way it's written on
# Basketball-Reference. Generated rather than listed out: it's the same
# getter 30 times over. Lower-is-better follows the team's point of view:
# its own turnovers and fouls are bad, and every opponent number is bad
# except the opponent's turnovers and fouls.
_BOX_LABELS = {
    "PTS": "PTS", "REB": "TRB", "OREB": "ORB", "DREB": "DRB", "AST": "AST", "STL": "STL",
    "BLK": "BLK", "TOV": "TOV", "PF": "PF", "FGM": "FGM", "FGA": "FGA", "FG3M": "3PM",
    "FG3A": "3PA", "FTM": "FTM", "FTA": "FTA",
}
_TEAM_BOX_KEYS = {"Team MIN/G": "TEAM_MIN"}
for _prefix, _name in (("TEAM", "Team"), ("OPP", "Opp")):
    for _stat in TEAM_BOX_STATS:
        _TEAM_BOX_KEYS[f"{_name} {_BOX_LABELS[_stat]}/G"] = f"{_prefix}_{_stat}"
for _label, _key in _TEAM_BOX_KEYS.items():
    _bad_for_team = (_key.endswith(("_TOV", "_PF")) if _key.startswith("TEAM_")
                     else not _key.endswith(("_TOV", "_PF")))
    STAT_DEFS[_label] = (
        lambda b, key=_key: (b.get("team_box") or {}).get(key),
        "{:.1f}", _bad_for_team,
    )

# Shown by default; advanced/team/consistency rows are opt-in since they
# answer a different question than raw production and would clutter the
# default view.
DEFAULT_STAT_LABELS = [
    "GP", "W", "L", "MIN/G", "PTS/G", "TRB/G", "AST/G", "STL/G", "BLK/G", "TOV/G", "PF/G", "+/-",
    "FG%", "3P%", "FT%", "eFG%", "TS%", "USG%", "USG Vol/G",
]

ROW_FORMATS = {label: fmt for label, (_getter, fmt, _lower) in STAT_DEFS.items()}
LOWER_IS_BETTER = {label for label, (_getter, _fmt, lower) in STAT_DEFS.items() if lower}

# Cells can carry a numbered footnote (superscript 1, 2, ... under the
# table), one per KIND of caveat. Numbers are handed out per table in the
# order the caveats first appear, so a table only ever shows the notes it
# uses, numbered from 1.
#
# ESTIMATED: a value that rests on rebuilt team box-score lines rather than
# the official ones -- see ESTIMATED_SOURCES and data.py's
# _reconstructed_team_lines. In practice this means 1977-1984.
ESTIMATED = "estimated"
ESTIMATED_NOTE = (
    "Built from team box-score lines rebuilt by summing each game's individual "
    "player rows \u2014 the official team logs are missing most box-score columns before 1985. "
    "Covers only part of each 1977\u20131984 season, and misses team turnovers that "
    "aren't charged to a player, so treat these as close estimates rather than "
    "settled numbers. Earlier seasons show \u2014: with no turnovers recorded there is "
    "no possession estimate to make."
)

# Which rows CAN carry the mark, and which part of the stat block knows
# whether they do. Every one of these is computed out of the team box score,
# so it's only these that a rebuilt team line can affect -- a player's own
# PTS/G is official in every season and never gets marked.
#
# "Team MOV" is deliberately absent: it needs only TEAM_PTS/OPP_PTS, which
# the official logs have back to 1946, so it's computed off its own subset
# and is never an estimate (see compare._compute_team_context).
ESTIMATED_SOURCES = {
    **{label: "possessions" for label in STAT_DEFS if label.endswith("/100")},
    "Poss/G": "possessions",
    "MIN%": "usage",
    "USG%": "usage",
    "USG Vol/G": "usage",
    **{label: ("team_box", key) for label, key in _TEAM_BOX_KEYS.items()},
    "Team Poss/G": "team",
    "Team Pace": "team",
    "Team ORtg": "team",
    "Team DRtg": "team",
    "Net Rtg": "team",
}


# PARTIAL: a value averaged over only SOME of the span's games, because the
# stat wasn't recorded for the rest. Its footnote lists each marked span's
# game count.
PARTIAL = "partial"
PARTIAL_NOTE = (
    "Plus-minus wasn't recorded before 1996-97 in the NBA (2008 in the WNBA), so for a "
    "span reaching earlier it's averaged over only the games that have it:"
)
FOOTNOTES = {ESTIMATED: ESTIMATED_NOTE, PARTIAL: PARTIAL_NOTE}

# Small and faint, so a footnote number beside a stat reads as a marker and
# not an exponent (the browser default is ~83% size, full strength); line-
# height 0 keeps a marked cell from growing taller than its row.
FOOTNOTE_MARK_STYLE = "font-size:.6em;opacity:.55;margin-left:2px;line-height:0;"

# Rows that can carry the PARTIAL footnote -> the stat-block key holding (games
# with the stat, games in the span).
PARTIAL_SOURCES = {
    "+/-": "plus_minus_coverage",
    "+/- Std Dev": "plus_minus_coverage",
}


def _partial_coverage(block: dict | None, label: str) -> tuple[int, int] | None:
    """(games with the stat, games) when this row's value covers only some
    of the span's games, else None. A span with NONE of them recorded isn't
    partial -- its value is simply missing and shows \u2014."""
    key = PARTIAL_SOURCES.get(label)
    if block is None or key is None or not block.get(key):
        return None
    recorded, games = block[key]
    return (recorded, games) if 0 < recorded < games else None


def _cell_mark(block: dict | None, label: str) -> tuple[str, str | None] | None:
    """(footnote kind, detail for this span) for its value in this row, or None."""
    if _is_estimated(block, label):
        return (ESTIMATED, None)
    partial = _partial_coverage(block, label)
    if partial:
        return (PARTIAL, f"{partial[0]:,} of {partial[1]:,} games")
    return None


def _is_estimated(block: dict | None, label: str) -> bool:
    """Whether this span's value for this row should carry the ESTIMATED footnote.
    A (source, key) entry reads a per-column flag -- the team_box block
    tracks "estimated" separately for each of its columns."""
    source = ESTIMATED_SOURCES.get(label)
    if block is None or source is None:
        return False
    if isinstance(source, tuple):
        source, key = source
        return bool(((block.get(source) or {}).get("estimated") or {}).get(key))
    return bool((block.get(source) or {}).get("estimated"))


def build_stat_table(
    result: ComparisonResult,
    season_type: str = "regular",
    stat_labels: list[str] | None = None,
    stat_defs: dict | None = None,
) -> pd.DataFrame:
    """
    Raw numeric table: rows = stats (in stat_labels order), columns = span
    labels. None where a span has no games. stat_defs defaults to the
    built-in STAT_DEFS; pass a merged dict (built-ins + custom formulas) to
    include user-defined stats -- see formulas.py.
    """
    stat_defs = stat_defs if stat_defs is not None else STAT_DEFS
    stat_labels = stat_labels if stat_labels is not None else DEFAULT_STAT_LABELS
    data = {}
    for agg in result.aggregates:
        block = agg[season_type]
        row = {}
        for label in stat_labels:
            if label not in stat_defs:
                continue
            getter = stat_defs[label][0]
            try:
                row[label] = getter(block) if block else None
            except Exception:
                row[label] = None
        data[agg["label"]] = row
    return pd.DataFrame(data).reindex([l for l in stat_labels if l in stat_defs])


def build_stat_flags(
    result: ComparisonResult,
    season_type: str = "regular",
    stat_labels: list[str] | None = None,
    stat_defs: dict | None = None,
) -> pd.DataFrame:
    """
    Same shape and order as build_stat_table, holding (footnote kind, detail)
    where that span's value for that row should be footnoted and None where
    it shouldn't: ESTIMATED for a value built from rebuilt team box-score
    lines, PARTIAL for one averaged over only part of the span's games
    (plus-minus before it was recorded). Pass to
    render_stat_table_html(flags=...).

    Built as a separate frame rather than as part of the table because
    build_stat_table has to stay purely numeric -- that's what the best/worst
    highlighting compares on. Custom formulas never flag, even when they
    reference a marked stat; a formula's provenance isn't tracked, and
    guessing at it would be worse than leaving it unmarked.
    """
    stat_defs = stat_defs if stat_defs is not None else STAT_DEFS
    stat_labels = stat_labels if stat_labels is not None else DEFAULT_STAT_LABELS
    data = {}
    for agg in result.aggregates:
        block = agg[season_type]
        data[agg["label"]] = {
            label: _cell_mark(block, label)
            for label in stat_labels if label in stat_defs
        }
    return pd.DataFrame(data).reindex([l for l in stat_labels if l in stat_defs])


def formats_and_lower_is_better(stat_defs: dict) -> tuple[dict, set]:
    """Derive the {label: format} and {lower_is_better labels} needed by
    render_stat_table_html from any stat_defs dict, including one merged
    with custom formulas (see formulas.py)."""
    formats = {label: fmt for label, (_getter, fmt, _lower) in stat_defs.items()}
    lower_is_better = {label for label, (_getter, _fmt, lower) in stat_defs.items() if lower}
    return formats, lower_is_better


def render_stat_table_html(
    df: pd.DataFrame,
    title: str = "",
    formats: dict | None = None,
    lower_is_better: set | None = None,
    flags: pd.DataFrame | None = None,
) -> str:
    """
    Formats + highlights the best (green) and worst (red) value per row,
    Stathead-style. Worst is only highlighted with 3+ columns being
    compared -- with exactly 2, every cell would be either best or worst,
    which is just noise (it's the same information as "not green," shown
    louder). Ties at either extreme aren't highlighted, since there's
    nothing distinct to flag.
    Returns a standalone HTML string -- pass to st.markdown(html, unsafe_allow_html=True).

    formats/lower_is_better default to the main STAT_DEFS rules; pass the
    output of formats_and_lower_is_better(your_stat_defs) for a table built
    from a different/merged stat_defs (e.g. including custom formulas).

    flags (from build_stat_flags, same shape as df) puts a superscript
    footnote number on each flagged cell and the numbered notes under the
    table -- numbered in order of first appearance, and only the notes
    actually used, so tables of complete, official numbers stay clean.
    PARTIAL_NOTE is followed by each footnoted column's detail (its game
    count). A plain True flag means ESTIMATED.
    """
    formats = formats if formats is not None else ROW_FORMATS
    lower_is_better = lower_is_better if lower_is_better is not None else LOWER_IS_BETTER

    def mark_for(row_label, col):
        if flags is None or row_label not in flags.index or col not in flags.columns:
            return None
        flag = flags.at[row_label, col]
        if flag is True:
            return (ESTIMATED, None)
        return flag if isinstance(flag, tuple) else None

    numbers: dict[str, int] = {}             # footnote kind -> its number in this table
    details: dict[str, dict[str, str]] = {}  # footnote kind -> {column: detail}

    def fmt_cell(row_label, value):
        if value is None or pd.isna(value):
            return "\u2014"
        return formats.get(row_label, "{}").format(value)

    rows_html = []
    for row_label in df.index:
        values = df.loc[row_label]
        numeric = values.dropna()
        best = None
        worst = None
        if len(numeric) > 1:
            best = numeric.min() if row_label in lower_is_better else numeric.max()
        if len(numeric) >= 3:
            worst = numeric.max() if row_label in lower_is_better else numeric.min()
            if worst == best:
                worst = None  # all tied at the extreme -- nothing distinct to flag red

        cells = []
        for col in df.columns:
            v = values[col]
            text = fmt_cell(row_label, v)
            mark = mark_for(row_label, col) if v is not None and not pd.isna(v) else None
            if mark:
                kind, detail = mark
                number = numbers.setdefault(kind, len(numbers) + 1)
                if detail:
                    details.setdefault(kind, {})[col] = detail
                text += f'<sup style="{FOOTNOTE_MARK_STYLE}">{number}</sup>'
            is_best = best is not None and v == best
            is_worst = (not is_best) and worst is not None and v == worst
            if is_best:
                style = "font-weight:600;background:#1f3d2b;color:#7CFC9A;"
            elif is_worst:
                style = "font-weight:600;background:#3d1f1f;color:#FC7C7C;"
            else:
                style = "color:#ddd;"
            cells.append(f'<td style="padding:5px 14px;{style}">{text}</td>')

        rows_html.append(
            f'<tr><td style="padding:5px 14px;color:#888;font-weight:600;">{row_label}</td>'
            f'{"".join(cells)}</tr>'
        )

    header_cells = "".join(
        f'<th style="padding:5px 14px;text-align:left;color:#eee;border-bottom:1px solid #333;">{c}</th>'
        for c in df.columns
    )
    notes = []
    for kind, number in sorted(numbers.items(), key=lambda kv: kv[1]):
        note = FOOTNOTES[kind]
        per_col = details.get(kind)
        if per_col:
            note += " " + "; ".join(f"{html.escape(str(c))}: {d}" for c, d in per_col.items()) + "."
        notes.append(f"<sup>{number}</sup> {note}")
    note_html = "".join(
        f'<div style="margin-top:8px;color:#888;font-size:.8em;max-width:820px;line-height:1.45;">'
        f'{note}</div>'
        for note in notes
    )
    return f"""
    <div style="font-family:-apple-system,sans-serif;">
      {f'<h4 style="margin-bottom:6px;color:#eee;">{title}</h4>' if title else ''}
      <table style="border-collapse:collapse;background:#111;">
        <tr><th style="padding:5px 14px;border-bottom:1px solid #333;"></th>{header_cells}</tr>
        {''.join(rows_html)}
      </table>
      {note_html}
    </div>
    """


def _sum_accolade_blocks(a: dict, b: dict) -> dict:
    """Combine two players' accolades.block() dicts into one, for a duo --
    scalar counts add, tier-count dicts (all_nba/all_defense) merge by key."""
    all_nba = dict(a["all_nba"])
    for k, v in b["all_nba"].items():
        all_nba[k] = all_nba.get(k, 0) + v
    all_defense = dict(a["all_defense"])
    for k, v in b["all_defense"].items():
        all_defense[k] = all_defense.get(k, 0) + v
    return {
        "all_star": a["all_star"] + b["all_star"],
        "all_nba": all_nba,
        "all_defense": all_defense,
        "mvp_shares": a["mvp_shares"] + b["mvp_shares"],
        "championships": a["championships"] + b["championships"],
        "finals_mvp": a["finals_mvp"] + b["finals_mvp"],
    }


def build_awards_table(spans, accolade_store) -> pd.DataFrame | None:
    """
    Not used by the app yet -- kept for the planned awards table (see
    accolades.py). Returns None if the accolade store has no data loaded
    (default state -- see accolades.py for how to wire in a real source). Awards are
    season-level, not game-level, data -- so a DuoSpan's row is simply both
    players' individual awards summed over the span's seasons, not scoped
    to games they shared as teammates the way the box-score stats are.
    """
    if accolade_store is None or accolade_store.df.empty:
        return None
    data = {}
    for span in spans:
        if isinstance(span, DuoSpan):
            block = _sum_accolade_blocks(
                accolade_store.block(span.player_a_id, span.seasons),
                accolade_store.block(span.player_b_id, span.seasons),
            )
        else:
            block = accolade_store.block(span.player_id, span.seasons)
        data[span.label] = {
            "All-Star": block["all_star"],
            "All-NBA": sum(block["all_nba"].values()) if block["all_nba"] else 0,
            "All-Defense": sum(block["all_defense"].values()) if block["all_defense"] else 0,
            "MVP shares": round(block["mvp_shares"], 2),
            "Championships": block["championships"],
            "Finals MVP": block["finals_mvp"],
        }
    return pd.DataFrame(data)