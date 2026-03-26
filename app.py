"""
Peptone Generative Output Evaluator v2 — main Streamlit entrypoint.

Run with:
    streamlit run peptone_evaluator_v2/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the package is importable when Streamlit runs this file directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    COL_SMILES,
    COL_PIC50,
    COL_VALID,
    COL_CANONICAL,
    NUMERIC_DESCRIPTOR_COLS,
    FLAG_COLS,
    HOVER_EXTRA_COLS,
    LARGE_DATASET_WARN,
)
from src.theme import inject_global_css
from src.data_loader import load_csv, normalize_columns
from src.chemistry import parse_smiles_column, get_unique_mask
from src.descriptors import compute_descriptors, apply_rule_flags, compute_descriptor_summary
from src.fingerprints import compute_morgan_fingerprints
from src.dimensionality_reduction import reduce
from src.similarity import (
    compute_similarity_metrics,
    find_duplicates,
    top_similar_pairs,
    top_isolated_molecules,
    per_molecule_nn_table,
)
from src.prioritization import (
    compute_priority_scores,
    get_shortlist,
    select_diverse_representatives,
)
from src.campaign import compute_campaign_overlap, compare_descriptor_stats
from src.filters import apply_filters
from src.plotting import (
    scatter_chemical_space,
    descriptor_inspector,
    distribution_grid,
    similarity_histogram,
    scaffold_bar_chart,
)
from src.scaffolds import (
    compute_scaffolds,
    scaffold_frequency,
    scaffold_diversity_stats,
)
from src.substructure import (
    parse_substructure_query,
    substructure_search,
    highlight_substructure_svg,
    COMMON_SUBSTRUCTURES,
)
from src.export import sdf_bytes
from src.ui import (
    render_app_header,
    render_section_label,
    render_metrics_strip,
    render_metric_cards,
    render_campaign_upload,
    render_campaign_metrics,
    render_sidebar_filters,
    render_color_by_dropdown,
    render_shortlist_section,
    render_download_button,
)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + THEME
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Peptone Evaluator",
    page_icon="P",
    layout="wide",
)
inject_global_css()


# ═══════════════════════════════════════════════════════════════════════════
# CACHED ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Parsing molecules, computing descriptors, scaffolds & fingerprints\u2026")
def enrich_molecules(smiles_series: pd.Series, other_cols: pd.DataFrame):
    """Parse SMILES, compute descriptors, flags, scaffolds, and Morgan fingerprints."""
    df = other_cols.copy()
    df[COL_SMILES] = smiles_series
    df = parse_smiles_column(df)
    df = compute_descriptors(df)
    df = apply_rule_flags(df)
    df = compute_scaffolds(df)          # Murcko decomposition
    fp_matrix = compute_morgan_fingerprints(df)
    df["is_unique"] = get_unique_mask(df)
    df = df.drop(columns=["mol"])
    return df, fp_matrix


@st.cache_data(show_spinner="Running dimensionality reduction\u2026")
def cached_reduce(method: str, fp_matrix: np.ndarray, **kwargs):
    return reduce(method, fp_matrix, **kwargs)


@st.cache_data(show_spinner="Computing similarity metrics\u2026")
def cached_similarity(fp_matrix: np.ndarray):
    return compute_similarity_metrics(fp_matrix)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
uploaded_current, uploaded_reference = render_campaign_upload()


# ═══════════════════════════════════════════════════════════════════════════
# APP HEADER
# ═══════════════════════════════════════════════════════════════════════════
current_filename = uploaded_current.name if uploaded_current else None
render_app_header(
    filename=current_filename,
    has_campaign=uploaded_reference is not None,
)

if uploaded_current is None:
    st.markdown("---")
    st.markdown(
        "### Upload a CSV to begin\n\n"
        "Use the **sidebar** (click **›** if collapsed) to upload a CSV containing "
        "a `SMILES` column. A `pIC50` column is optional."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# LOAD & VALIDATE
# ═══════════════════════════════════════════════════════════════════════════
try:
    raw_df = load_csv(uploaded_current)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

try:
    norm_df, has_pic50 = normalize_columns(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if len(norm_df) > LARGE_DATASET_WARN:
    st.warning(
        f"Large dataset ({len(norm_df):,} rows). "
        "t-SNE and UMAP may take a while."
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENRICH
# ═══════════════════════════════════════════════════════════════════════════
other_cols = norm_df.drop(columns=[COL_SMILES])
df, fp_matrix = enrich_molecules(norm_df[COL_SMILES], other_cols)

total_rows = len(df)
n_valid = int(df[COL_VALID].sum())
n_invalid = total_rows - n_valid
n_unique = int(df["is_unique"].sum())


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE SET (optional)
# ═══════════════════════════════════════════════════════════════════════════
ref_df = None
ref_fp_matrix = None
has_campaign = False

if uploaded_reference is not None:
    try:
        ref_raw = load_csv(uploaded_reference)
        ref_norm, _ = normalize_columns(ref_raw)
        ref_other = ref_norm.drop(columns=[COL_SMILES])
        ref_df, ref_fp_matrix = enrich_molecules(ref_norm[COL_SMILES], ref_other)
        has_campaign = True
    except Exception as exc:
        st.sidebar.warning(f"Reference CSV issue: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY (compute on full set before filters)
# ═══════════════════════════════════════════════════════════════════════════
sim_metrics = None
if fp_matrix is not None and fp_matrix.shape[0] >= 2:
    sim_metrics = cached_similarity(fp_matrix)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ═══════════════════════════════════════════════════════════════════════════
filter_spec = render_sidebar_filters(df, has_pic50)
filtered_df = apply_filters(df, filter_spec)


# ═══════════════════════════════════════════════════════════════════════════
# ── PRIMARY SURFACE ──
# ═══════════════════════════════════════════════════════════════════════════

# Campaign comparison strip (if reference uploaded)
if has_campaign and ref_df is not None:
    render_section_label("Campaign Comparison", subtitle_key="campaign")
    overlap_info = compute_campaign_overlap(df, ref_df)
    comparison_stats = compare_descriptor_stats(df, ref_df)
    render_campaign_metrics(overlap_info, comparison_stats)

# Dataset overview metrics
render_section_label("Overview", subtitle_key="overview")
render_metric_cards(
    filtered_df,
    has_pic50=has_pic50,
    total_rows=total_rows,
    n_valid=n_valid,
    n_invalid=n_invalid,
    n_unique=n_unique,
    similarity_metrics=sim_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# ── CHEMICAL SPACE (hero section) ──
# ═══════════════════════════════════════════════════════════════════════════
render_section_label("Chemical Space", subtitle_key="chemical_space")

if fp_matrix is None or fp_matrix.shape[0] < 2:
    st.warning("Need at least 2 valid molecules for dimensionality reduction.")
else:
    # Colour-by selector (compact, inline)
    available_cols = [c for c in df.columns if c in filtered_df.columns]
    cb_col1, cb_col2 = st.columns([1, 4])
    with cb_col1:
        st.markdown(
            '<div class="metric-label" style="margin-top:0.5rem">Colour by</div>',
            unsafe_allow_html=True,
        )
    with cb_col2:
        color_col = render_color_by_dropdown(available_cols, has_pic50)

    # Combined FP matrix for campaign overlay
    if has_campaign and ref_fp_matrix is not None and ref_fp_matrix.shape[0] >= 2:
        combined_fp = np.vstack([fp_matrix, ref_fp_matrix])
        n_current = fp_matrix.shape[0]
    else:
        combined_fp = fp_matrix
        n_current = fp_matrix.shape[0]

    # Valid DataFrames for plotting
    valid_df = df[df[COL_VALID]].reset_index(drop=True)
    ref_valid_df = None
    if has_campaign and ref_df is not None:
        ref_valid_df = ref_df[ref_df[COL_VALID]].reset_index(drop=True)

    hover_extras = [c for c in HOVER_EXTRA_COLS if c in valid_df.columns]

    # Tabs
    tab_pca, tab_tsne, tab_umap = st.tabs(["PCA", "t-SNE", "UMAP"])

    def _render_scatter(tab, method, coords_full, xl, yl):
        """Render scatter in a tab with optional campaign overlay."""
        with tab:
            current_coords = coords_full[:n_current]
            plot_df = valid_df.copy()
            plot_df[xl] = current_coords[:, 0]
            plot_df[yl] = current_coords[:, 1]

            ref_plot_df = None
            if ref_valid_df is not None and coords_full.shape[0] > n_current:
                ref_coords = coords_full[n_current:]
                ref_plot_df = ref_valid_df.copy()
                ref_plot_df[xl] = ref_coords[:, 0]
                ref_plot_df[yl] = ref_coords[:, 1]

            st.plotly_chart(
                scatter_chemical_space(
                    plot_df, xl, yl,
                    color_col=color_col,
                    x_label=xl, y_label=yl,
                    hover_extra_cols=hover_extras,
                    reference_df=ref_plot_df,
                ),
                use_container_width=True,
            )

    # PCA
    coords_full, xl, yl = cached_reduce("PCA", combined_fp)
    _render_scatter(tab_pca, "PCA", coords_full, xl, yl)

    # t-SNE
    with tab_tsne:
        perp = st.slider(
            "Perplexity", min_value=5, max_value=100, value=30, step=5,
            help="Controls the balance between local and global structure.",
        )
    coords_full, xl, yl = cached_reduce("t-SNE", combined_fp, perplexity=perp)
    _render_scatter(tab_tsne, "t-SNE", coords_full, xl, yl)

    # UMAP
    with tab_umap:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            n_neighbors = st.slider(
                "n_neighbors", min_value=2, max_value=100, value=15, step=1,
                help="Larger values capture more global structure.",
            )
        with u_col2:
            min_dist = st.slider(
                "min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                help="Smaller values create tighter clusters.",
            )
    coords_full, xl, yl = cached_reduce(
        "UMAP", combined_fp, n_neighbors=n_neighbors, min_dist=min_dist,
    )
    _render_scatter(tab_umap, "UMAP", coords_full, xl, yl)


# ═══════════════════════════════════════════════════════════════════════════
# ── SECONDARY DIAGNOSTICS ──
# ═══════════════════════════════════════════════════════════════════════════

# ── Similarity & Diversity ────────────────────────────────────────────────
render_section_label("Similarity & Diversity", subtitle_key="similarity")

if sim_metrics is not None:
    valid_smiles_list = df.loc[df[COL_VALID], COL_CANONICAL].tolist()
    dup_info = find_duplicates(df.loc[df[COL_VALID], COL_CANONICAL])

    render_metrics_strip([
        ("Mean Pairwise Sim", f"{sim_metrics['mean_pairwise_similarity']:.4f}"),
        ("Mean NN Sim", f"{sim_metrics['mean_nn_similarity']:.4f}"),
        ("Diversity Score", f"{sim_metrics['diversity_score']:.4f}"),
        ("Duplicates", f"{dup_info['duplicate_count']:,} ({dup_info['duplicate_rate']:.1%})"),
    ])

    if sim_metrics["was_sampled"]:
        st.markdown(
            f'<div class="muted-text" style="margin-top:0.3rem">'
            f"Pairwise similarity computed on a random sample of "
            f"{sim_metrics['sample_size']:,} molecules.</div>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        similarity_histogram(sim_metrics["nn_similarities"]),
        use_container_width=True,
    )

    # Actionable tables
    sim_matrix = sim_metrics["sim_matrix"]
    sampled_idx = sim_metrics["sampled_indices"]
    sampled_smiles = [valid_smiles_list[i] for i in sampled_idx if i < len(valid_smiles_list)]

    with st.expander("Top 20 most similar pairs"):
        pairs_df = top_similar_pairs(sim_matrix, sampled_smiles)
        if not pairs_df.empty:
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough molecules to compute pairs.")

    with st.expander("Most isolated molecules (lowest NN similarity)"):
        isolated_df = top_isolated_molecules(sim_metrics["nn_similarities"], sampled_smiles)
        if not isolated_df.empty:
            st.dataframe(isolated_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data.")

    with st.expander("Per-molecule nearest-neighbour table"):
        nn_df = per_molecule_nn_table(sim_matrix, sampled_smiles)
        if not nn_df.empty:
            st.dataframe(nn_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data.")

    st.markdown(
        '<div class="muted-text">'
        "<b>How to read these metrics:</b> "
        "Mean NN similarity \u2014 average of each molecule\u2019s highest Tanimoto similarity "
        "to any other molecule. High values suggest clusters of very similar compounds. "
        "Diversity score = 1 \u2212 mean pairwise similarity. "
        "Tanimoto computed from 1024-bit Morgan fingerprints (radius 2, ECFP4-equivalent)."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Need at least 2 valid molecules to compute similarity metrics.")


# ── Scaffold Analysis ─────────────────────────────────────────────────────
render_section_label(
    "Scaffold Analysis",
    subtitle="Murcko decomposition of valid molecules. Measures whether the generative "
             "model is exploring diverse core structures or producing trivial variations.",
)

if "murcko_scaffold" in filtered_df.columns:
    valid_for_scaff = filtered_df[filtered_df[COL_VALID]]
    if not valid_for_scaff.empty and valid_for_scaff["murcko_scaffold"].notna().any():
        scaff_stats = scaffold_diversity_stats(valid_for_scaff, "murcko_scaffold")

        render_metrics_strip([
            ("Unique Scaffolds", f"{scaff_stats['unique_scaffolds']:,}"),
            ("Scaffold Ratio", f"{scaff_stats['scaffold_ratio']:.3f}"),
            ("Singletons", f"{scaff_stats['singleton_scaffolds']:,} "
                           f"({scaff_stats['singleton_fraction']:.1%})"),
            ("Top Scaffold", f"{scaff_stats['top_scaffold_count']:,} "
                             f"({scaff_stats['top_scaffold_fraction']:.1%})"),
        ])

        freq_df = scaffold_frequency(valid_for_scaff, "murcko_scaffold", top_n=15)
        if not freq_df.empty:
            st.plotly_chart(
                scaffold_bar_chart(freq_df, title="Top Murcko Scaffolds"),
                use_container_width=True,
            )

        with st.expander("Scaffold frequency table"):
            full_freq = scaffold_frequency(valid_for_scaff, "murcko_scaffold", top_n=50)
            if not full_freq.empty:
                st.dataframe(full_freq, use_container_width=True, hide_index=True)

        # Generic scaffold view
        with st.expander("Generic framework analysis"):
            gen_stats = scaffold_diversity_stats(valid_for_scaff, "generic_scaffold")
            st.markdown(
                '<div class="muted-text">'
                "Generic frameworks collapse all atoms to carbon and all bonds to single, "
                "grouping structurally similar scaffolds more aggressively."
                "</div>",
                unsafe_allow_html=True,
            )
            render_metrics_strip([
                ("Generic Frameworks", f"{gen_stats['unique_scaffolds']:,}"),
                ("Framework Ratio", f"{gen_stats['scaffold_ratio']:.3f}"),
            ])
            gen_freq = scaffold_frequency(valid_for_scaff, "generic_scaffold", top_n=15)
            if not gen_freq.empty:
                st.dataframe(gen_freq, use_container_width=True, hide_index=True)
    else:
        st.info("No valid scaffolds to analyse.")
else:
    st.info("Scaffold data not available.")


# ── Substructure Search ───────────────────────────────────────────────────
render_section_label(
    "Substructure Search",
    subtitle="Query by SMARTS or SMILES to find molecules containing a specific "
             "pharmacophore or functional group.",
)

ss_col1, ss_col2 = st.columns([3, 1])
with ss_col1:
    ss_query = st.text_input(
        "SMARTS or SMILES query",
        placeholder="e.g. c1ccccc1 (benzene) or [NH2] (primary amine)",
        label_visibility="collapsed",
    )
with ss_col2:
    ss_preset = st.selectbox(
        "Preset",
        ["Custom"] + list(COMMON_SUBSTRUCTURES.keys()),
        label_visibility="collapsed",
    )

# Resolve query
active_query = ""
if ss_preset != "Custom":
    active_query = COMMON_SUBSTRUCTURES[ss_preset]
elif ss_query:
    active_query = ss_query

if active_query:
    query_mol = parse_substructure_query(active_query)
    if query_mol is None:
        st.warning(f"Could not parse query: `{active_query}`")
    else:
        valid_for_ss = filtered_df[filtered_df[COL_VALID]]
        ss_mask = substructure_search(valid_for_ss, query_mol)
        n_matches = int(ss_mask.sum())
        n_searched = len(valid_for_ss)

        render_metrics_strip([
            ("Matches", f"{n_matches:,}"),
            ("Searched", f"{n_searched:,}"),
            ("Hit Rate", f"{n_matches / n_searched:.1%}" if n_searched else "—"),
        ])

        if n_matches > 0:
            match_df = valid_for_ss[ss_mask]
            display_cols = [c for c in [COL_CANONICAL, "MolWt", "LogP", "QED",
                                        "passes_lipinski_like", "murcko_scaffold"]
                           if c in match_df.columns]
            if COL_PIC50 in match_df.columns and has_pic50:
                display_cols = [COL_PIC50] + display_cols

            st.dataframe(
                match_df[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

            # Show up to 4 highlighted structures
            with st.expander("Highlighted structures (first 4 matches)"):
                svg_cols = st.columns(min(4, n_matches))
                for i, (_, row) in enumerate(match_df.head(4).iterrows()):
                    smi = row.get(COL_CANONICAL) or row.get(COL_SMILES)
                    svg = highlight_substructure_svg(str(smi), query_mol)
                    if svg:
                        with svg_cols[i]:
                            st.markdown(svg, unsafe_allow_html=True)
                            st.caption(str(smi)[:40])

            # Download matches
            csv_matches = match_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download matches (CSV)",
                data=csv_matches,
                file_name="peptone_substructure_matches.csv",
                mime="text/csv",
            )
        else:
            st.info("No molecules match this substructure in the current filtered set.")
else:
    st.markdown(
        '<div class="muted-text">'
        "Enter a SMARTS pattern or SMILES string above, or select a preset "
        "from the dropdown."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Descriptor Distributions ──────────────────────────────────────────────
render_section_label("Descriptor Distributions", subtitle_key="distributions")

dist_cols = [c for c in NUMERIC_DESCRIPTOR_COLS if c in filtered_df.columns]
if has_pic50 and COL_PIC50 in filtered_df.columns:
    dist_cols = [COL_PIC50] + dist_cols

valid_for_dist = filtered_df[filtered_df[COL_VALID]]

if dist_cols and not valid_for_dist.empty:
    # Summary statistics table — the primary scientific output
    summary_df = compute_descriptor_summary(filtered_df, dist_cols)
    if not summary_df.empty:
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

    # Single-descriptor inspector — select one to examine in detail
    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    inspect_col = st.selectbox(
        "Inspect descriptor",
        dist_cols,
        index=0,
        label_visibility="collapsed",
        help="Select a descriptor for detailed histogram with KDE and statistics.",
    )
    if inspect_col and inspect_col in valid_for_dist.columns:
        fig_inspect = descriptor_inspector(
            valid_for_dist[inspect_col],
            descriptor_name=inspect_col,
        )
        st.plotly_chart(fig_inspect, use_container_width=True)

    # Full grid in expander — for overview comparison
    with st.expander("All distributions"):
        fig_dist = distribution_grid(valid_for_dist, dist_cols)
        st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("No descriptor data to plot.")


# ═══════════════════════════════════════════════════════════════════════════
# ── DECISION LAYER ──
# ═══════════════════════════════════════════════════════════════════════════
render_section_label("Priority Shortlist", subtitle_key="shortlist")

if fp_matrix is not None and n_valid >= 2:
    nn_sims = sim_metrics["nn_similarities"] if sim_metrics else None
    sampled_idx_pri = sim_metrics["sampled_indices"] if sim_metrics else None

    scored_df = compute_priority_scores(
        df, nn_similarities=nn_sims, valid_indices=sampled_idx_pri, has_pic50=has_pic50,
    )

    shortlist = get_shortlist(scored_df)
    valid_df_for_div = df[df[COL_VALID]].reset_index(drop=True)
    diverse_reps = select_diverse_representatives(fp_matrix, valid_df_for_div)

    render_shortlist_section(shortlist, diverse_reps, has_pic50)
else:
    st.info("Need at least 2 valid molecules for prioritisation.")


# ═══════════════════════════════════════════════════════════════════════════
# ── UTILITIES ──
# ═══════════════════════════════════════════════════════════════════════════
render_section_label(
    "Data & Export",
    subtitle="Download processed data in CSV or SDF format. SDF files can be "
             "opened directly in PyMOL, MOE, Maestro, and other chemistry tools.",
)

with st.expander("Data preview"):
    tab_raw, tab_proc = st.tabs(["Raw", "Processed"])
    with tab_raw:
        st.dataframe(raw_df.head(5), use_container_width=True)
    with tab_proc:
        display_cols = [c for c in filtered_df.columns if c not in ("mol",)]
        st.dataframe(filtered_df[display_cols].head(5), use_container_width=True)

with st.expander("Download processed data"):
    dl_col1, dl_col2 = st.columns(2)

    # CSV download
    with dl_col1:
        render_download_button(filtered_df)

    # SDF download
    with dl_col2:
        sdf_valid_only = st.checkbox("Valid molecules only (SDF)", value=True)
        sdf_data = sdf_bytes(filtered_df, valid_only=sdf_valid_only)
        st.download_button(
            label="Download SDF",
            data=sdf_data,
            file_name="peptone_v2_processed.sdf",
            mime="chemical/x-mdl-sdfile",
        )

    display_cols = [c for c in filtered_df.columns if c not in ("mol",)]
    st.dataframe(filtered_df[display_cols], use_container_width=True)
