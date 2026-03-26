"""
Plotly figure builders.

Every function returns a ``plotly.graph_objects.Figure`` — no Streamlit
calls happen here, keeping visualisation logic testable and reusable.

Charts use transparent backgrounds and theme-adaptive styling so they
inherit the host page's light or dark mode automatically.
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
    """
    2-D scatter plot of dimensionality-reduced chemical space.

    Parameters
    ----------
    hover_extra_cols : additional columns to include in hover tooltip.
    reference_df : if provided, overlay as muted hollow markers (campaign comparison).
    """
    plot_df = df.copy()
    template = get_plotly_template()
    layout_defaults = get_plotly_layout_defaults()

    # --- Determine if color column is boolean (categorical) ---
    is_bool_color = False
    if color_col and color_col in plot_df.columns:
        if plot_df[color_col].dtype == bool or set(plot_df[color_col].dropna().unique()).issubset({True, False, 0, 1}):
            is_bool_color = True
            plot_df[color_col] = plot_df[color_col].astype(str)

    # --- Build hover data ---
    hover_cols: dict = {
        COL_SMILES: True,
        x_col: ":.4f",
        y_col: ":.4f",
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

    # --- Main scatter ---
    color_kwarg = color_col if color_col and color_col in plot_df.columns else None

    if is_bool_color:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_kwarg,
            color_discrete_map={
                "True": CHART_COLORS["success"],
                "False": CHART_COLORS["danger"],
            },
            hover_data=hover_cols,
            labels={x_col: x_label or x_col, y_col: y_label or y_col},
            template=template,
            opacity=0.78,
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_kwarg,
            color_continuous_scale="Viridis",
            hover_data=hover_cols,
            labels={x_col: x_label or x_col, y_col: y_label or y_col},
            template=template,
            opacity=0.78,
        )

    fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="rgba(128,128,128,0.4)")))

    # --- Reference overlay (campaign comparison) ---
    if reference_df is not None and x_col in reference_df.columns and y_col in reference_df.columns:
        ref_hover = (
            reference_df[COL_SMILES].astype(str) if COL_SMILES in reference_df.columns
            else pd.Series("", index=reference_df.index)
        )
        fig.add_trace(
            go.Scatter(
                x=reference_df[x_col],
                y=reference_df[y_col],
                mode="markers",
                name="Reference",
                marker=dict(
                    symbol="circle-open",
                    color="rgba(128,128,128,0.5)",
                    size=5,
                    line=dict(width=0.8, color="rgba(128,128,128,0.5)"),
                ),
                text=ref_hover,
                hovertemplate="%{text}<br>%{x:.4f}, %{y:.4f}<extra>Reference</extra>",
                opacity=0.45,
            )
        )
        fig.data[0].name = "Current"
        fig.data[0].showlegend = True

    fig.update_layout(
        **layout_defaults,
        height=640,
        title=dict(text=title, font=dict(size=13), x=0, y=0.98) if title else {},
    )
    if color_kwarg and not is_bool_color:
        fig.update_layout(coloraxis_colorbar=dict(title=color_col, thickness=14, len=0.6))
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
    layout_defaults = get_plotly_layout_defaults()

    fig = px.histogram(
        x=series.dropna(),
        nbins=nbins,
        title=title,
        template=template,
        color_discrete_sequence=[color or CHART_COLORS["primary"]],
    )
    fig.update_layout(
        **layout_defaults,
        xaxis_title=series.name or "",
        yaxis_title="Count",
        height=320,
        bargap=0.05,
    )
    return fig


def descriptor_inspector(
    series: pd.Series,
    descriptor_name: str,
    nbins: int = 50,
) -> go.Figure:
    """
    Single-descriptor histogram with KDE overlay and annotated statistics.

    Shows histogram + smoothed density curve + vertical lines for mean and
    median, with an annotation block displaying n, μ, σ, median, IQR.
    """
    vals = series.dropna()
    if vals.empty:
        fig = go.Figure()
        fig.update_layout(title=descriptor_name, height=360)
        return fig

    template = get_plotly_template()
    layout_defaults = get_plotly_layout_defaults()

    mean_val = float(vals.mean())
    median_val = float(vals.median())
    std_val = float(vals.std())
    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    n = len(vals)

    # Histogram (normalised to density so KDE overlay scales correctly)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=nbins,
            histnorm="probability density",
            marker_color=CHART_COLORS["primary"],
            opacity=0.55,
            name="Distribution",
            hovertemplate=f"{descriptor_name}: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
        )
    )

    # KDE overlay — Gaussian kernel density estimate
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(vals, bw_method="scott")
        x_grid = np.linspace(float(vals.min()), float(vals.max()), 200)
        kde_vals = kde(x_grid)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=kde_vals,
                mode="lines",
                line=dict(color=CHART_COLORS["primary"], width=2),
                name="KDE",
                hoverinfo="skip",
            )
        )
    except Exception:
        pass  # degenerate data — skip KDE

    # Mean line
    fig.add_vline(
        x=mean_val,
        line=dict(color=CHART_COLORS["danger"], width=1.5, dash="dash"),
        annotation_text="μ",
        annotation_position="top",
        annotation_font=dict(size=11, color=CHART_COLORS["danger"]),
    )

    # Median line
    fig.add_vline(
        x=median_val,
        line=dict(color=CHART_COLORS["warning"], width=1.5, dash="dot"),
        annotation_text="med",
        annotation_position="top",
        annotation_font=dict(size=11, color=CHART_COLORS["warning"]),
    )

    # Statistics annotation block
    stat_text = (
        f"n = {n:,}<br>"
        f"μ = {mean_val:.3f}<br>"
        f"σ = {std_val:.3f}<br>"
        f"median = {median_val:.3f}<br>"
        f"IQR = [{q1:.3f}, {q3:.3f}]"
    )
    fig.add_annotation(
        text=stat_text,
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        showarrow=False,
        align="right",
        font=dict(size=11, family="monospace"),
        bgcolor="rgba(128,128,128,0.08)",
        bordercolor="rgba(128,128,128,0.15)",
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        template=template,
        **layout_defaults,
        xaxis_title=descriptor_name,
        yaxis_title="Density",
        height=380,
        bargap=0.03,
        showlegend=False,
    )
    return fig


def distribution_grid(
    df: pd.DataFrame,
    columns: list[str],
    ncols: int = 2,
) -> go.Figure:
    """Compact grid of histograms — for overview, not primary display."""
    columns = [c for c in columns if c in df.columns]
    nrows = -(-len(columns) // ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=columns,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    for i, col in enumerate(columns):
        r = i // ncols + 1
        c = i % ncols + 1
        vals = df[col].dropna()
        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=40,
                marker_color=CHART_COLORS["primary"],
                opacity=0.75,
                name=col,
            ),
            row=r, col=c,
        )

    layout_defaults = get_plotly_layout_defaults()
    fig.update_layout(
        template=get_plotly_template(),
        **layout_defaults,
        height=280 * nrows,
        showlegend=False,
    )
    for ax_key in [k for k in fig.layout.to_plotly_json() if k.startswith(("xaxis", "yaxis"))]:
        fig.layout[ax_key]["gridcolor"] = "rgba(128,128,128,0.12)"
    return fig


# ---------------------------------------------------------------------------
# Similarity histogram
# ---------------------------------------------------------------------------

def similarity_histogram(
    nn_similarities: np.ndarray,
    title: str = "Nearest-Neighbour Tanimoto Similarity",
) -> go.Figure:
    """Histogram of per-molecule nearest-neighbour similarities."""
    template = get_plotly_template()
    layout_defaults = get_plotly_layout_defaults()

    fig = px.histogram(
        x=nn_similarities,
        nbins=50,
        title=title,
        template=template,
        color_discrete_sequence=[CHART_COLORS["secondary"]],
    )
    fig.update_layout(
        **layout_defaults,
        xaxis_title="Tanimoto Similarity",
        yaxis_title="Count",
        height=340,
        bargap=0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# Scaffold frequency bar chart
# ---------------------------------------------------------------------------

def scaffold_bar_chart(
    freq_df: "pd.DataFrame",
    title: str = "Top Scaffolds by Frequency",
    max_bars: int = 15,
) -> go.Figure:
    """
    Horizontal bar chart of scaffold frequencies.

    Expects a DataFrame with columns: Scaffold, Count, Fraction.
    """
    template = get_plotly_template()
    layout_defaults = get_plotly_layout_defaults()

    plot_df = freq_df.head(max_bars).copy()
    # Truncate long SMILES for axis labels
    plot_df["label"] = plot_df["Scaffold"].apply(
        lambda s: s if len(str(s)) <= 40 else str(s)[:37] + "..."
    )

    fig = go.Figure(
        go.Bar(
            y=plot_df["label"],
            x=plot_df["Count"],
            orientation="h",
            marker_color=CHART_COLORS["primary"],
            opacity=0.8,
            text=plot_df["Fraction"].apply(lambda f: f"{f:.1%}"),
            textposition="auto",
            hovertemplate=(
                "Scaffold: %{customdata[0]}<br>"
                "Count: %{x}<br>"
                "Fraction: %{customdata[1]:.2%}<extra></extra>"
            ),
            customdata=plot_df[["Scaffold", "Fraction"]].values,
        )
    )

    fig.update_layout(
        template=template,
        **layout_defaults,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Count",
        yaxis_title="",
        height=max(280, 28 * len(plot_df) + 80),
        title=dict(text=title, font=dict(size=13), x=0, y=0.98) if title else {},
    )
    return fig
