"""
Reusable Streamlit UI components — compact app header, metrics strips,
section labels, sidebar controls, shortlist rendering, and download helpers.

Designed for a premium scientific-instrument feel: restrained typography,
muted labels, strong numeric hierarchy, no emoji, no dashboard clutter.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .config import (
    COL_PIC50,
    COL_VALID,
    COL_CANONICAL,
    NUMERIC_DESCRIPTOR_COLS,
    FLAG_COLS,
    COLOR_BY_OPTIONS,
    SECTION_SUBTITLES,
    APP_TITLE,
    APP_SUBTITLE,
)
from .filters import FilterSpec


# ---------------------------------------------------------------------------
# App header — compact, software-like
# ---------------------------------------------------------------------------

def render_app_header(
    filename: str | None = None,
    has_campaign: bool = False,
) -> None:
    """Compact top bar: product name + context + file badge."""
    parts = [
        f'<span class="app-header-title">{APP_TITLE}</span>',
        '<span class="app-header-sep">/</span>',
        f'<span class="app-header-context">{APP_SUBTITLE}</span>',
    ]
    badge = ""
    if filename:
        label = filename
        if has_campaign:
            label += " + reference"
        badge = f'<span class="app-header-badge">{label}</span>'

    st.markdown(
        f'<div class="app-header">{"".join(parts)}{badge}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section labels — replace st.subheader + emoji
# ---------------------------------------------------------------------------

def render_section_label(
    title: str,
    subtitle: str | None = None,
    subtitle_key: str | None = None,
) -> None:
    """
    Section heading with optional muted subtitle.
    Adds vertical spacing above to separate sections without dividers.
    """
    text = subtitle or (SECTION_SUBTITLES.get(subtitle_key, "") if subtitle_key else "")
    sub_html = f'<div class="section-subtitle">{text}</div>' if text else ""
    st.markdown(
        f'<div class="section-spacer"></div>'
        f'<div class="section-title">{title}</div>'
        f'{sub_html}',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metrics strip — instrument-panel readout
# ---------------------------------------------------------------------------

def render_metrics_strip(items: list[tuple[str, str]]) -> None:
    """
    Horizontal strip of value/label pairs.

    Each item is (label, value). Renders with large numeric emphasis
    and tiny uppercase labels — no card borders.
    """
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Metric cards (uses metrics strip internally)
# ---------------------------------------------------------------------------

def render_metric_cards(
    df: pd.DataFrame,
    has_pic50: bool,
    total_rows: int,
    n_valid: int,
    n_invalid: int,
    n_unique: int,
    similarity_metrics: dict | None = None,
) -> None:
    """Two-row metrics display: counts then averages."""

    # Row 1 — dataset counts
    render_metrics_strip([
        ("Total molecules", f"{total_rows:,}"),
        ("Valid", f"{n_valid:,}"),
        ("Invalid", f"{n_invalid:,}"),
        ("Unique", f"{n_unique:,}"),
    ])

    # Row 2 — descriptor averages + similarity
    valid_df = df[df[COL_VALID]] if COL_VALID in df.columns else df
    row2: list[tuple[str, str]] = []

    if has_pic50 and COL_PIC50 in valid_df.columns:
        avg = valid_df[COL_PIC50].mean()
        row2.append(("Avg pIC50", f"{avg:.3f}" if pd.notna(avg) else "\u2014"))

    for col, label in [("MolWt", "Avg MW"), ("LogP", "Avg LogP"), ("TPSA", "Avg TPSA")]:
        if col in valid_df.columns:
            avg = valid_df[col].mean()
            row2.append((label, f"{avg:.2f}" if pd.notna(avg) else "\u2014"))

    if similarity_metrics:
        row2.append(("Mean NN Sim", f"{similarity_metrics['mean_nn_similarity']:.3f}"))
        row2.append(("Diversity", f"{similarity_metrics['diversity_score']:.3f}"))

    if row2:
        st.markdown('<div style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
        render_metrics_strip(row2)


# ---------------------------------------------------------------------------
# Campaign upload (sidebar)
# ---------------------------------------------------------------------------

def render_campaign_upload():
    """
    Sidebar upload section with optional reference CSV.
    Returns (uploaded_current, uploaded_reference).
    """
    st.sidebar.markdown(
        '<div class="sidebar-group-label">Data Input</div>',
        unsafe_allow_html=True,
    )

    uploaded_current = st.sidebar.file_uploader(
        "Current round CSV",
        type=["csv"],
        help="Must contain a SMILES column. pIC50 is optional.",
        key="current_csv",
    )

    st.sidebar.markdown('<div style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-group-label">Campaign Comparison</div>',
        unsafe_allow_html=True,
    )
    uploaded_reference = st.sidebar.file_uploader(
        "Reference set CSV",
        type=["csv"],
        help="Upload a previous round or reference library.",
        key="reference_csv",
    )

    return uploaded_current, uploaded_reference


# ---------------------------------------------------------------------------
# Campaign metrics
# ---------------------------------------------------------------------------

def render_campaign_metrics(
    overlap_info: dict,
    comparison_df: pd.DataFrame | None = None,
) -> None:
    """Compact campaign comparison strip + optional descriptor table."""
    render_metrics_strip([
        ("Novel compounds", f"{overlap_info['novel_count']:,}"),
        ("Overlap", f"{overlap_info['overlap_count']:,}"),
        ("Reference unique", f"{overlap_info['ref_unique_count']:,}"),
    ])

    if comparison_df is not None and not comparison_df.empty:
        with st.expander("Descriptor comparison (current vs. reference)"):
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def render_sidebar_filters(
    df: pd.DataFrame,
    has_pic50: bool,
) -> FilterSpec:
    """Build sidebar filter widgets and return a populated FilterSpec."""
    st.sidebar.markdown(
        '<div class="sidebar-group-label" style="margin-top:1.6rem">Filters</div>',
        unsafe_allow_html=True,
    )

    valid_df = df[df[COL_VALID]] if COL_VALID in df.columns else df

    # Toggles
    valid_only = st.sidebar.checkbox("Valid molecules only", value=True)
    unique_only = st.sidebar.checkbox("Unique molecules only", value=False)
    lipinski_only = st.sidebar.checkbox(
        "Med-chem heuristic screen",
        value=False,
        help=(
            "MW \u2264 500, LogP \u2264 5, HBD \u2264 5, HBA \u2264 10, RotBond \u2264 10. "
            "Rule-based filter only \u2014 not a developability or stability predictor."
        ),
    )

    st.sidebar.markdown('<div style="margin-top:0.4rem"></div>', unsafe_allow_html=True)

    # Range sliders
    def _range_slider(label: str, col: str, default_min: float, default_max: float, step: float = 1.0):
        if col in valid_df.columns and not valid_df[col].dropna().empty:
            lo = float(valid_df[col].min())
            hi = float(valid_df[col].max())
        else:
            lo, hi = default_min, default_max
        return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), step=step)

    mw_range = _range_slider("Mol. Weight", "MolWt", 0.0, 1500.0, 5.0)
    logp_range = _range_slider("LogP", "LogP", -5.0, 10.0, 0.1)
    tpsa_range = _range_slider("TPSA", "TPSA", 0.0, 300.0, 1.0)

    def _max_slider(label: str, col: str, default_max: int):
        if col in valid_df.columns and not valid_df[col].dropna().empty:
            hi = int(valid_df[col].max())
        else:
            hi = default_max
        return st.sidebar.slider(label, min_value=0, max_value=max(hi, 1), value=max(hi, 1))

    hbd_max = _max_slider("Max HBD", "HBD", 20)
    hba_max = _max_slider("Max HBA", "HBA", 20)
    rotbond_max = _max_slider("Max Rot. Bonds", "RotatableBonds", 20)

    pic50_range = None
    if has_pic50 and COL_PIC50 in valid_df.columns and not valid_df[COL_PIC50].dropna().empty:
        lo = float(valid_df[COL_PIC50].min())
        hi = float(valid_df[COL_PIC50].max())
        pic50_range = st.sidebar.slider(
            "pIC50 range", min_value=lo, max_value=hi, value=(lo, hi), step=0.1,
        )

    return FilterSpec(
        mw_range=mw_range,
        logp_range=logp_range,
        tpsa_range=tpsa_range,
        hbd_max=hbd_max,
        hba_max=hba_max,
        rotbond_max=rotbond_max,
        pic50_range=pic50_range,
        valid_only=valid_only,
        unique_only=unique_only,
        lipinski_only=lipinski_only,
    )


# ---------------------------------------------------------------------------
# Colour-by dropdown
# ---------------------------------------------------------------------------

def render_color_by_dropdown(
    available_cols: list[str],
    has_pic50: bool,
) -> str | None:
    """Selectbox for choosing the color-by column in chemical space plots."""
    options = ["None"]
    for c in COLOR_BY_OPTIONS:
        if c in available_cols:
            if c == COL_PIC50 and not has_pic50:
                continue
            options.append(c)

    default_idx = options.index(COL_PIC50) if has_pic50 and COL_PIC50 in options else 0
    selected = st.selectbox("Colour by", options, index=default_idx, label_visibility="collapsed")
    return None if selected == "None" else selected


# ---------------------------------------------------------------------------
# Shortlist section
# ---------------------------------------------------------------------------

def render_shortlist_section(
    shortlist_df: pd.DataFrame,
    diverse_df: pd.DataFrame,
    has_pic50: bool,
) -> None:
    """Priority shortlist tables with download buttons."""

    # Scoring explanation — muted, not bold
    if has_pic50:
        st.markdown(
            '<div class="muted-text">'
            "Scoring: pIC50 (30%) + QED drug-likeness (30%) + structural diversity (20%) "
            "+ med-chem rule compliance (20%). Each factor is rank-normalised to [0, 1]."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="muted-text">'
            "Scoring: QED drug-likeness (40%) + structural diversity (30%) "
            "+ med-chem rule compliance (30%). Each factor is rank-normalised to [0, 1]. "
            "pIC50 not available \u2014 weight redistributed."
            "</div>",
            unsafe_allow_html=True,
        )

    tab_short, tab_diverse = st.tabs(["Top Ranked", "Diverse Representatives"])

    with tab_short:
        if shortlist_df.empty:
            st.info("No scored molecules available.")
        else:
            st.dataframe(shortlist_df, use_container_width=True, hide_index=True)
            csv = shortlist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download shortlist",
                data=csv,
                file_name="peptone_shortlist.csv",
                mime="text/csv",
            )

    with tab_diverse:
        if diverse_df.empty:
            st.info("Not enough molecules for diversity selection.")
        else:
            st.markdown(
                '<div class="muted-text">'
                "Selected via greedy max-min Tanimoto distance to maximise "
                "structural coverage across the chemical space."
                "</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(diverse_df, use_container_width=True, hide_index=True)
            csv = diverse_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download diverse representatives",
                data=csv,
                file_name="peptone_diverse_reps.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Download button
# ---------------------------------------------------------------------------

def render_download_button(df: pd.DataFrame) -> None:
    """Offer a CSV download of the processed (and optionally filtered) data."""
    export_cols = [c for c in df.columns if c not in ("mol",)]
    csv_bytes = df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download processed data",
        data=csv_bytes,
        file_name="peptone_v2_processed.csv",
        mime="text/csv",
    )
