"""
Theme-aware styling for the Peptone Evaluator.

Provides global CSS injection and Plotly chart helpers driven by a
dark_mode flag stored in st.session_state["dark_mode"] (default True).
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

def get_dark_mode() -> bool:
    """Return True when the app is in dark mode (reads session state)."""
    try:
        return bool(st.session_state.get("dark_mode", True))
    except Exception:
        return True


def get_plotly_template() -> str:
    """Return the Plotly template that matches the current dark/light mode."""
    return "plotly_dark" if get_dark_mode() else "plotly_white"


def get_plotly_layout_defaults() -> dict:
    """
    Shared Plotly layout overrides — transparent backgrounds, no gridlines,
    thin axis lines, and a system font stack.

    Note: the main chart functions in plotting.py use ``_base_layout()``
    directly for tighter control. This function remains for backward
    compatibility and edge cases.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT_STACK, size=10),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="rgba(128,128,128,0.35)",
            zeroline=False,
            ticks="outside",
            ticklen=3,
            tickwidth=1,
            tickcolor="rgba(128,128,128,0.35)",
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="rgba(128,128,128,0.35)",
            zeroline=False,
            ticks="outside",
            ticklen=3,
            tickwidth=1,
            tickcolor="rgba(128,128,128,0.35)",
        ),
        margin=dict(l=48, r=20, t=32, b=40),
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
    gap: 0.15rem;
    border-bottom: 1px solid rgba(128,128,128,0.12);
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.45rem 0.9rem;
}
.stTabs [data-baseweb="tab-highlight"] {
    height: 2px;
}

/* ── Expanders ────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.8rem;
    font-weight: 500;
}
details[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.1) !important;
    border-radius: 3px !important;
}

/* ── Tables ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] th {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
    font-size: 0.78rem !important;
}

/* ── Download buttons ─────────────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    font-size: 0.76rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 3px;
}

/* ── File uploader ────────────────────────────────────────────────── */
[data-testid="stFileUploader"] label {
    font-size: 0.78rem !important;
}

/* ── Plotly chart containers ──────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    padding: 0 !important;
}

/* ── st.info / st.warning — subtle ───────────────────────────────── */
[data-testid="stAlert"] {
    font-size: 0.8rem;
    padding: 0.6rem 0.9rem;
    border-radius: 3px;
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


_DARK_CSS = """
<style>
/* ── Dark mode backgrounds ────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
}
[data-testid="stHeader"] {
    background-color: #0e1117 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background-color: #1a1b27 !important;
}
</style>
"""

_LIGHT_CSS = """
<style>
/* ── Light mode — enhanced contrast ──────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #f5f6fa !important;
}
[data-testid="stHeader"] {
    background-color: #f5f6fa !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background-color: #e4e5ee !important;
}

/* Darker text for better readability */
.section-title { color: #111111 !important; }
.section-subtitle { color: #444444 !important; opacity: 1 !important; }
.metric-value { color: #111111 !important; }
.metric-label { color: #444444 !important; opacity: 1 !important; }
.app-header-title { color: #111111 !important; opacity: 0.8 !important; }
.app-header-context { color: #444444 !important; opacity: 1 !important; }
.app-header-sep { color: #444444 !important; opacity: 0.5 !important; }
.app-header-badge {
    background: #d8d9e8 !important;
    color: #111111 !important;
}
.muted-text { color: #444444 !important; opacity: 1 !important; }
.sidebar-group-label { color: #333333 !important; opacity: 0.7 !important; }

/* Slightly darker Streamlit default elements */
[data-testid="stMarkdownContainer"] p { color: #1a1a2e; }
label { color: #1a1a2e !important; }
</style>
"""


def inject_global_css(dark_mode: bool = True) -> None:
    """Inject the full CSS design system + theme overrides. Call once per render."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(_DARK_CSS if dark_mode else _LIGHT_CSS, unsafe_allow_html=True)
