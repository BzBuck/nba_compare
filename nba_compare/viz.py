"""
Plotly visualizations built on top of a ComparisonResult.
Three views: grouped bar (raw stat comparison), radar (normalized shape
comparison), and a combined RS/Playoff table figure.
"""
from __future__ import annotations
import plotly.graph_objects as go
from .compare import ComparisonResult


def grouped_bar(result: ComparisonResult, stats: list[str], season_type: str = "regular",
                 stat_group: str = "per_game", title: str | None = None) -> go.Figure:
    """One bar cluster per stat, one bar per span. Good for direct scale comparisons."""
    table = result.wide_table(season_type=season_type, stat_group=stat_group)
    fig = go.Figure()
    for span_label in table.columns:
        fig.add_trace(go.Bar(
            name=span_label,
            x=stats,
            y=[table.loc[s, span_label] if s in table.index else None for s in stats],
        ))
    fig.update_layout(
        barmode="group",
        title=title or f"{stat_group} comparison ({season_type})",
        template="plotly_dark",
    )
    return fig


def radar(result: ComparisonResult, stats: list[str], season_type: str = "regular",
           stat_group: str = "per_game", title: str | None = None) -> go.Figure:
    """
    Normalizes each stat 0-1 across the spans being compared (max = 1), so you
    can see relative 'shape' even when stats are on very different scales
    (e.g. AST vs PTS). Not meant for reading exact values -- use grouped_bar for that.
    """
    table = result.wide_table(season_type=season_type, stat_group=stat_group)
    table = table.reindex(stats)
    maxes = table.max(axis=1).replace(0, 1)
    normalized = table.div(maxes, axis=0)

    fig = go.Figure()
    for span_label in normalized.columns:
        values = normalized[span_label].fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=stats + [stats[0]],
            fill="toself",
            name=span_label,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title or f"Shape comparison ({season_type}, normalized)",
        template="plotly_dark",
    )
    return fig


def summary_table(result: ComparisonResult) -> go.Figure:
    df = result.summary()
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color="#333", font=dict(color="white")),
        cells=dict(values=[df[c] for c in df.columns], fill_color="#111", font=dict(color="white")),
    )])
    fig.update_layout(template="plotly_dark", title="Span summary (RS vs Playoffs)")
    return fig