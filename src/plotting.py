"""
Plotly figure builders.

Every function returns a ``plotly.graph_objects.Figure`` — no Streamlit
calls happen here, keeping visualisation logic testable and reusable.

Design language: Nature-journal quality. Thin axes, no gridlines,
precise tick labels, muted fills with crisp outlines, smart annotation
placement. Theme-adaptive (light/dark).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COL_SMILES, COL_CANONICAL, COL_PIC50
from .theme import CHART_COLORS, get_plotly_template, get_plotly_layout_defaults

# ---------------------------------------------------------------------------
# Shared visual constants — one source of truth
# ---------------------------------------------------------------------------

_FONT = "Inter, system-ui, -apple-system, sans-serif"
_MONO = "'SF Mono', 'Fira Code', 'Consolas', monospace"
_AXIS_COLOR = "rgba(128,128,128,0.35)"
_TICK_FONT_SIZE = 10
_LABEL_FONT_SIZE = 11
_ANNOT_FONT_SIZE = 10
_BAR_GAP = 0.04
_HIST_OPACITY = 0.6
_HIST_LINE_WIDTH = 0.8
_HIST_LINE_COLOR = "rgba(128,128,128,0.25)"

# Standardised chart heights
_H_SCATTER = 600
_H_HISTOGRAM = 340
_H_INSPECTOR = 360
_H_SIMILARITY = 320
_H_BAR_PER_ROW = 26
_H_BAR_MIN = 240
_H_GRID_PER_ROW = 240


def _base_layout(template: str) -> dict:
    """
    Shared layout overrides for all figures.

    Nature-style: no gridlines, thin axis lines, transparent background,
    tight margins, small precise tick labels.
    """
    return dict(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, size=_TICK_FONT_SIZE),
        margin=dict(l=48, r=20, t=32, b=40),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=_AXIS_COLOR,
            ticks="outside",
            ticklen=3,
            tickwidth=1,
            tickcolor=_AXIS_COLOR,
            tickfont=dict(size=_TICK_FONT_SIZE),
            title_font=dict(size=_LABEL_FONT_SIZE),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=_AXIS_COLOR,
            ticks="outside",
            ticklen=3,
            tickwidth=1,
            tickcolor=_AXIS_COLOR,
            tickfont=dict(size=_TICK_FONT_SIZE),
            title_font=dict(size=_LABEL_FONT_SIZE),
            zeroline=False,
        ),
        showlegend=False,
    )


# ---------------------------------------------------------------------------
# Chemical-space scatter
# ---------------------------------------------------------------------------

def scatter_chemical_space(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    hover_extra_cols: list[str] | None = None,
    reference_df: pd.DataFrame | None = None,
) -> go.Figure:
    """2-D scatter of dimensionality-reduced chemical space."""
    plot_df = df.copy()
    template = get_plotly_template()

    # Boolean colour detection
    is_bool_color = False
    if color_col and color_col in plot_df.columns:
        if plot_df[color_col].dtype == bool or set(plot_df[color_col].dropna().unique()).issubset({True, False, 0, 1}):
            is_bool_color = True
            plot_df[color_col] = plot_df[color_col].astype(str)

    # Hover data
    hover_cols: dict = {
        COL_SMILES: True,
        x_col: ":.3f",
        y_col: ":.3f",
    }
    if COL_CANONICAL in plot_df.columns:
        hover_cols[COL_CANONICAL] = True
    if color_col and color_col in plot_df.columns and not is_bool_color:
        hover_cols[color_col] = ":.3f"
    if hover_extra_cols:
        for c in hover_extra_cols:
            if c in plot_df.columns and c not in hover_cols and c != color_col:
                if plot_df[c].dtype == bool:
                    plot_df[c] = plot_df[c].astype(str)
                    hover_cols[c] = True
                elif pd.api.types.is_numeric_dtype(plot_df[c]):
                    hover_cols[c] = ":.3f"
                else:
                    hover_cols[c] = True

    color_kwarg = color_col if color_col and color_col in plot_df.columns else None

    if is_bool_color:
        fig = px.scatter(
            plot_df, x=x_col, y=y_col, color=color_kwarg,
            color_discrete_map={"True": CHART_COLORS["success"], "False": CHART_COLORS["danger"]},
            hover_data=hover_cols,
            labels={x_col: x_label or x_col, y_col: y_label or y_col},
            template=template, opacity=0.72,
        )
    else:
        fig = px.scatter(
            plot_df, x=x_col, y=y_col, color=color_kwarg,
            color_continuous_scale="Viridis",
            hover_data=hover_cols,
            labels={x_col: x_label or x_col, y_col: y_label or y_col},
            template=template, opacity=0.72,
        )

    fig.update_traces(marker=dict(
        size=5,
        line=dict(width=0.4, color="rgba(128,128,128,0.3)"),
    ))

    # Reference overlay
    if reference_df is not None and x_col in reference_df.columns and y_col in reference_df.columns:
        ref_hover = (
            reference_df[COL_SMILES].astype(str) if COL_SMILES in reference_df.columns
            else pd.Series("", index=reference_df.index)
        )
        fig.add_trace(go.Scatter(
            x=reference_df[x_col], y=reference_df[y_col],
            mode="markers", name="Reference",
            marker=dict(
                symbol="circle-open", color="rgba(128,128,128,0.45)",
                size=4, line=dict(width=0.7, color="rgba(128,128,128,0.45)"),
            ),
            text=ref_hover,
            hovertemplate="%{text}<br>%{x:.3f}, %{y:.3f}<extra>Reference</extra>",
            opacity=0.4,
        ))
        fig.data[0].name = "Current"
        fig.data[0].showlegend = True
        fig.update_layout(showlegend=True)

    layout = _base_layout(template)
    layout.update(height=_H_SCATTER)
    # Embedding axes: hide ticks — they're arbitrary coordinates
    layout["xaxis"].update(showticklabels=True, title_text=x_label or x_col)
    layout["yaxis"].update(showticklabels=True, title_text=y_label or y_col)
    fig.update_layout(**layout)

    if color_kwarg and not is_bool_color:
        fig.update_layout(coloraxis_colorbar=dict(
            title=dict(text=color_col, font=dict(size=_LABEL_FONT_SIZE)),
            thickness=12, len=0.55,
            tickfont=dict(size=_TICK_FONT_SIZE),
            outlinewidth=0,
        ))
    return fig


# ---------------------------------------------------------------------------
# Distribution histograms
# ---------------------------------------------------------------------------

def histogram(
    series: pd.Series,
    title: str = "",
    nbins: int = 50,
    color: str | None = None,
) -> go.Figure:
    """Simple histogram for a single numeric series."""
    template = get_plotly_template()
    fill = color or CHART_COLORS["primary"]

    fig = go.Figure(go.Histogram(
        x=series.dropna(),
        nbinsx=nbins,
        marker_color=fill,
        marker_line_color=_HIST_LINE_COLOR,
        marker_line_width=_HIST_LINE_WIDTH,
        opacity=_HIST_OPACITY,
    ))

    layout = _base_layout(template)
    layout.update(
        height=_H_HISTOGRAM,
        bargap=_BAR_GAP,
    )
    layout["xaxis"]["title_text"] = series.name or ""
    layout["yaxis"]["title_text"] = "Count"
    fig.update_layout(**layout)
    return fig


def descriptor_inspector(
    series: pd.Series,
    descriptor_name: str,
    nbins: int = 50,
) -> go.Figure:
    """
    Single-descriptor histogram with KDE overlay and annotated statistics.

    Smart placement: when mean and median are close, annotations are
    staggered vertically to prevent overlap.
    """
    vals = series.dropna()
    if vals.empty:
        fig = go.Figure()
        fig.update_layout(height=_H_INSPECTOR)
        return fig

    template = get_plotly_template()

    mean_val = float(vals.mean())
    median_val = float(vals.median())
    std_val = float(vals.std())
    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    n = len(vals)

    # Histogram (density-normalised)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=nbins,
        histnorm="probability density",
        marker_color=CHART_COLORS["primary"],
        marker_line_color=_HIST_LINE_COLOR,
        marker_line_width=_HIST_LINE_WIDTH,
        opacity=0.45,
        name="Distribution",
        hovertemplate=f"{descriptor_name}: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
    ))

    # KDE overlay
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(vals, bw_method="scott")
        x_grid = np.linspace(float(vals.min()), float(vals.max()), 200)
        kde_vals = kde(x_grid)
        fig.add_trace(go.Scatter(
            x=x_grid, y=kde_vals,
            mode="lines",
            line=dict(color=CHART_COLORS["primary"], width=1.8),
            name="KDE", hoverinfo="skip",
        ))
    except Exception:
        pass

    # Smart mean/median annotation placement
    data_range = float(vals.max()) - float(vals.min())
    closeness = abs(mean_val - median_val) / data_range if data_range > 0 else 0

    # If they're very close (<5% of range), stagger annotations
    if closeness < 0.05:
        mean_ay = -30
        median_ay = -55
    else:
        mean_ay = -30
        median_ay = -30

    # Mean annotation
    fig.add_vline(
        x=mean_val,
        line=dict(color=CHART_COLORS["danger"], width=1.2, dash="dash"),
    )
    fig.add_annotation(
        x=mean_val, y=1, yref="paper",
        text=f"<b>μ</b> = {mean_val:.2f}",
        showarrow=True, arrowhead=0, arrowwidth=1,
        arrowcolor=CHART_COLORS["danger"],
        ax=0, ay=mean_ay,
        font=dict(size=_ANNOT_FONT_SIZE, color=CHART_COLORS["danger"]),
        bgcolor="rgba(0,0,0,0)",
        borderpad=2,
    )

    # Median annotation
    fig.add_vline(
        x=median_val,
        line=dict(color=CHART_COLORS["warning"], width=1.2, dash="dot"),
    )
    fig.add_annotation(
        x=median_val, y=1, yref="paper",
        text=f"<b>med</b> = {median_val:.2f}",
        showarrow=True, arrowhead=0, arrowwidth=1,
        arrowcolor=CHART_COLORS["warning"],
        ax=0, ay=median_ay,
        font=dict(size=_ANNOT_FONT_SIZE, color=CHART_COLORS["warning"]),
        bgcolor="rgba(0,0,0,0)",
        borderpad=2,
    )

    # Statistics panel — compact, monospaced
    stat_text = (
        f"<b>n</b> = {n:,}  "
        f"<b>μ</b> = {mean_val:.3f}  "
        f"<b>σ</b> = {std_val:.3f}<br>"
        f"<b>med</b> = {median_val:.3f}  "
        f"<b>IQR</b> = [{q1:.2f}, {q3:.2f}]"
    )
    fig.add_annotation(
        text=stat_text,
        xref="paper", yref="paper",
        x=0.98, y=0.92,
        showarrow=False, align="right",
        font=dict(size=_ANNOT_FONT_SIZE, family=_MONO),
        bgcolor="rgba(128,128,128,0.06)",
        bordercolor="rgba(128,128,128,0.12)",
        borderwidth=1, borderpad=5,
    )

    layout = _base_layout(template)
    layout.update(
        height=_H_INSPECTOR,
        bargap=_BAR_GAP,
    )
    layout["xaxis"]["title_text"] = descriptor_name
    layout["yaxis"]["title_text"] = "Density"
    fig.update_layout(**layout)
    return fig


def distribution_grid(
    df: pd.DataFrame,
    columns: list[str],
    ncols: int = 3,
) -> go.Figure:
    """Compact grid of histograms — for overview, not primary display."""
    columns = [c for c in columns if c in df.columns]
    nrows = -(-len(columns) // ncols)

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=columns,
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )

    for i, col in enumerate(columns):
        r = i // ncols + 1
        c = i % ncols + 1
        vals = df[col].dropna()
        fig.add_trace(
            go.Histogram(
                x=vals, nbinsx=35,
                marker_color=CHART_COLORS["primary"],
                marker_line_color=_HIST_LINE_COLOR,
                marker_line_width=0.5,
                opacity=_HIST_OPACITY,
                name=col,
            ),
            row=r, col=c,
        )

    template = get_plotly_template()
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, size=9),
        height=_H_GRID_PER_ROW * nrows,
        showlegend=False,
        margin=dict(l=36, r=16, t=28, b=24),
        bargap=_BAR_GAP,
    )
    # Clean axes for all subplots
    for ax_key in [k for k in fig.layout.to_plotly_json() if k.startswith(("xaxis", "yaxis"))]:
        fig.layout[ax_key].update(
            showgrid=False,
            showline=True,
            linewidth=0.8,
            linecolor=_AXIS_COLOR,
            zeroline=False,
            ticks="outside",
            ticklen=2,
            tickwidth=0.8,
            tickcolor=_AXIS_COLOR,
            tickfont=dict(size=8),
        )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=_ANNOT_FONT_SIZE, family=_FONT))
    return fig


# ---------------------------------------------------------------------------
# Similarity histogram
# ---------------------------------------------------------------------------

def similarity_histogram(
    nn_similarities: np.ndarray,
    title: str = "",
) -> go.Figure:
    """Histogram of per-molecule nearest-neighbour similarities."""
    template = get_plotly_template()

    fig = go.Figure(go.Histogram(
        x=nn_similarities,
        nbinsx=50,
        marker_color=CHART_COLORS["secondary"],
        marker_line_color=_HIST_LINE_COLOR,
        marker_line_width=_HIST_LINE_WIDTH,
        opacity=_HIST_OPACITY,
    ))

    layout = _base_layout(template)
    layout.update(
        height=_H_SIMILARITY,
        bargap=_BAR_GAP,
    )
    layout["xaxis"]["title_text"] = "Tanimoto Similarity"
    layout["yaxis"]["title_text"] = "Count"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Scaffold frequency bar chart
# ---------------------------------------------------------------------------

def scaffold_bar_chart(
    freq_df: pd.DataFrame,
    title: str = "",
    max_bars: int = 15,
) -> go.Figure:
    """Horizontal bar chart of scaffold frequencies."""
    template = get_plotly_template()

    plot_df = freq_df.head(max_bars).copy()
    plot_df["label"] = plot_df["Scaffold"].apply(
        lambda s: s if len(str(s)) <= 35 else str(s)[:32] + "\u2026"
    )

    fig = go.Figure(go.Bar(
        y=plot_df["label"],
        x=plot_df["Count"],
        orientation="h",
        marker_color=CHART_COLORS["primary"],
        marker_line_color=_HIST_LINE_COLOR,
        marker_line_width=_HIST_LINE_WIDTH,
        opacity=0.75,
        text=plot_df["Fraction"].apply(lambda f: f"{f:.1%}"),
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate=(
            "Scaffold: %{customdata[0]}<br>"
            "Count: %{x}<br>"
            "Fraction: %{customdata[1]:.2%}<extra></extra>"
        ),
        customdata=plot_df[["Scaffold", "Fraction"]].values,
    ))

    layout = _base_layout(template)
    n_bars = len(plot_df)
    layout.update(
        height=max(_H_BAR_MIN, _H_BAR_PER_ROW * n_bars + 60),
        margin=dict(l=180, r=40, t=28, b=32),
    )
    layout["xaxis"]["title_text"] = "Count"
    # Y-axis: no tick lines for bar labels, just text
    layout["yaxis"].update(
        showline=False, ticks="",
        tickfont=dict(size=9, family=_MONO),
        title_text="",
    )
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")
    return fig
