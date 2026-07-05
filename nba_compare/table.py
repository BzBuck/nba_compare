"""
Builds a Basketball-Reference/Stathead-style comparison table: one row per
stat, one column per compared span, with the best value in each row
highlighted -- rather than the grouped-bar-chart approach.

Kept separate from viz.py since this renders as HTML/dataframe, not Plotly.
"""
from __future__ import annotations
import pandas as pd
from .compare import ComparisonResult

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
    "AST/G":      (lambda b: b["per_game"]["AST"],                      "{:.1f}", False),
    "STL/G":      (lambda b: b["per_game"]["STL"],                      "{:.1f}", False),
    "BLK/G":      (lambda b: b["per_game"]["BLK"],                      "{:.1f}", False),
    "TOV/G":      (lambda b: b["per_game"]["TOV"],                      "{:.1f}", True),
    "PF/G":       (lambda b: b["per_game"]["PF"],                       "{:.1f}", True),
    "+/-":        (lambda b: b["plus_minus_per_game"],                  "{:+.1f}", False),
    "FG%":        (lambda b: b["shooting"]["FG_PCT"],                   "{:.3f}", False),
    "3P%":        (lambda b: b["shooting"]["FG3_PCT"],                  "{:.3f}", False),
    "FT%":        (lambda b: b["shooting"]["FT_PCT"],                   "{:.3f}", False),
    "eFG%":       (lambda b: b["shooting"]["EFG_PCT"],                  "{:.3f}", False),
    "TS%":        (lambda b: b["shooting"]["TS_PCT"],                   "{:.3f}", False),
    "TSA/G":      (lambda b: b["tsa_per_game"],                         "{:.1f}", False),
    "MIN%":       (lambda b: (b["usage"] or {}).get("min_pct"),         "{:.1f}", False),
    "USG%":       (lambda b: (b["usage"] or {}).get("usg_pct"),         "{:.1f}", False),
    "USG Vol/G":  (lambda b: (b["usage"] or {}).get("usage_per_game"),  "{:.1f}", False),
    "Team PTS/G": (lambda b: (b["team"] or {}).get("team_pts_per_game"),   "{:.1f}", False),
    "Team Poss/G": (lambda b: (b["team"] or {}).get("team_poss_per_game"), "{:.1f}", False),
    "Team Pace":  (lambda b: (b["team"] or {}).get("team_pace"),        "{:.1f}", False),
    "Team ORtg":  (lambda b: (b["team"] or {}).get("team_ortg"),        "{:.1f}", False),
    "Team DRtg":  (lambda b: (b["team"] or {}).get("team_drtg"),        "{:.1f}", True),
    "Net Rtg":    (lambda b: (b["team"] or {}).get("team_net_rtg"),     "{:+.1f}", False),
    "Team W%":    (lambda b: (b["wins"] / (b["wins"] + b["losses"]))
                             if b["wins"] is not None and (b["wins"] + b["losses"]) > 0 else None, "{:.3f}", False),
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
# Shown by default; advanced/team/consistency rows are opt-in since they
# answer a different question than raw production and would clutter the
# default view.
DEFAULT_STAT_LABELS = [
    "GP", "W", "L", "MIN/G", "PTS/G", "TRB/G", "AST/G", "STL/G", "BLK/G", "TOV/G", "PF/G", "+/-",
    "FG%", "3P%", "FT%", "eFG%", "TS%", "USG%", "USG Vol/G",
]

ROW_FORMATS = {label: fmt for label, (_getter, fmt, _lower) in STAT_DEFS.items()}
LOWER_IS_BETTER = {label for label, (_getter, _fmt, lower) in STAT_DEFS.items() if lower}


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
    """
    formats = formats if formats is not None else ROW_FORMATS
    lower_is_better = lower_is_better if lower_is_better is not None else LOWER_IS_BETTER

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
    return f"""
    <div style="font-family:-apple-system,sans-serif;">
      {f'<h4 style="margin-bottom:6px;color:#eee;">{title}</h4>' if title else ''}
      <table style="border-collapse:collapse;background:#111;">
        <tr><th style="padding:5px 14px;border-bottom:1px solid #333;"></th>{header_cells}</tr>
        {''.join(rows_html)}
      </table>
    </div>
    """


def build_awards_table(spans, accolade_store) -> pd.DataFrame | None:
    """
    Returns None if the accolade store has no data loaded (default state --
    see accolades.py for how to wire in a real source).
    """
    if accolade_store is None or accolade_store.df.empty:
        return None
    data = {}
    for span in spans:
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