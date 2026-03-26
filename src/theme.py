"""
Theme-aware styling for the Peptone Evaluator.

Provides global CSS injection and Plotly chart helpers that respect the
user's Streamlit theme (light or dark).  Never forces a colour mode —
uses Streamlit's native CSS variables so every surface adapts automatically.
"""

from __future__ import annotations

import streamlit as st


# ───────────────────────────────────────────────────────────────────────
# Accent palette (works well on both light and dark backgrounds)
# ───────────────────────────────────────────────────────────────────────

CHART_COLORS: dict[str, str] = {
    "primary": "#4361EE",
    "secondary": "#7209B7",
    "success": "#06D6A0",
    "warning": "#FFD166",
    "danger": "#EF476F",
    "neutral": "#8D99AE",
}

_FONT_STACK = "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"


# ───────────────────────────────────────────────────────────────────────
# Plotly helpers
# ───────────────────────────────────────────────────────────────────────

def get_plotly_template() -> str:
    """Return the Plotly template that matches the active Streamlit theme."""
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    return "plotly_dark" if base == "dark" else "plotly_white"


def get_plotly_layout_defaults() -> dict:
    """
    Shared Plotly layout overrides — transparent backgrounds, clean grid,
    and a system font stack so charts feel native to the app.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT_STACK, size=12),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.12)",
            zerolinecolor="rgba(128,128,128,0.18)",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.12)",
            zerolinecolor="rgba(128,128,128,0.18)",
        ),
        margin=dict(l=40, r=24, t=36, b=36),
    )


# ───────────────────────────────────────────────────────────────────────
# Global CSS injection
# ───────────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
/* ── Base overrides ───────────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent;
    backdrop-filter: none;
}

/* Reduce top padding so content starts higher */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ── Sidebar control rail ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    min-width: 280px !important;
    max-width: 320px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.82rem;
    line-height: 1.4;
}

/* Sidebar group labels */
.sidebar-group-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.45;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}

/* ── Section headers ──────────────────────────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    margin-bottom: 0.15rem;
    color: var(--text-color);
}
.section-subtitle {
    font-size: 0.78rem;
    opacity: 0.45;
    margin-top: 0px;
    margin-bottom: 1rem;
    line-height: 1.45;
    color: var(--text-color);
}
.section-spacer {
    margin-top: 2.8rem;
}

/* ── Metrics strip ────────────────────────────────────────────────── */
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.01em;
    color: var(--text-color);
}
.metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.4;
    margin-top: 0.2rem;
    color: var(--text-color);
}

/* Suppress default st.metric chrome that we're replacing */
[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    opacity: 0.4 !important;
}

/* ── App header ───────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.1rem;
}
.app-header-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-color);
    opacity: 0.55;
}
.app-header-sep {
    font-size: 0.72rem;
    opacity: 0.2;
    color: var(--text-color);
}
.app-header-context {
    font-size: 0.72rem;
    font-weight: 400;
    opacity: 0.35;
    color: var(--text-color);
}
.app-header-badge {
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    background: var(--secondary-background-color);
    color: var(--text-color);
    opacity: 0.7;
    margin-left: auto;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.2rem;
    border-bottom: 1px solid rgba(128,128,128,0.15);
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 0.5rem 1rem;
}
.stTabs [data-baseweb="tab-highlight"] {
    height: 2px;
}

/* ── Expanders ────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.88rem;
    font-weight: 500;
}
details[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.12) !important;
    border-radius: 4px !important;
}

/* ── Tables ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] th {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ── Download buttons ─────────────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.35rem 0.9rem;
    border-radius: 4px;
}

/* ── File uploader ────────────────────────────────────────────────── */
[data-testid="stFileUploader"] label {
    font-size: 0.8rem !important;
}

/* ── Plotly chart containers ──────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    padding: 0 !important;
}

/* ── Muted helper text ────────────────────────────────────────────── */
.muted-text {
    font-size: 0.78rem;
    opacity: 0.45;
    line-height: 1.5;
    color: var(--text-color);
}

/* ── Divider replacement ──────────────────────────────────────────── */
.subtle-divider {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.1);
    margin: 0.5rem 0;
}

/* ── Contextual help tooltips ────────────────────────────────────── */
.hint-wrap {
    position: relative;
    display: inline-block;
    cursor: help;
    vertical-align: middle;
}
.hint-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    font-size: 9px;
    font-weight: 600;
    font-style: normal;
    border-radius: 50%;
    border: 1px solid rgba(128,128,128,0.25);
    color: var(--text-color);
    opacity: 0.3;
    margin-left: 4px;
    transition: opacity 0.15s ease;
    line-height: 1;
    user-select: none;
}
.hint-wrap:hover .hint-icon {
    opacity: 0.7;
}
.hint-tip {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    width: max-content;
    max-width: 280px;
    padding: 8px 11px;
    font-size: 0.74rem;
    font-weight: 400;
    line-height: 1.45;
    letter-spacing: normal;
    text-transform: none;
    color: var(--text-color);
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 5px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    z-index: 9999;
    pointer-events: none;
    transition: opacity 0.15s ease, visibility 0.15s ease;
    white-space: normal;
}
.hint-wrap:hover .hint-tip {
    visibility: visible;
    opacity: 1;
}
/* Arrow */
.hint-tip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: var(--secondary-background-color);
}
</style>
"""


def inject_global_css() -> None:
    """Inject the full CSS design system into the Streamlit page. Call once."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
