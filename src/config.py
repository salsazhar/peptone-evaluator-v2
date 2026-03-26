"""
Central configuration — constants, column names, default parameters, and thresholds.
Every module imports from here so magic numbers live in one place.
"""

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------
APP_TITLE = "Peptone Evaluator"
APP_SUBTITLE = "Generative Output Evaluation"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
COL_SMILES = "SMILES"
COL_PIC50 = "pIC50"
COL_CANONICAL = "canonical_SMILES"
COL_VALID = "is_valid"
COL_UNIQUE = "is_unique"

# ---------------------------------------------------------------------------
# Morgan fingerprint defaults
# ---------------------------------------------------------------------------
FP_RADIUS: int = 2
FP_NBITS: int = 1024

# ---------------------------------------------------------------------------
# Dimensionality-reduction defaults
# ---------------------------------------------------------------------------
DR_RANDOM_STATE: int = 42
TSNE_DEFAULT_PERPLEXITY: int = 30
UMAP_DEFAULT_N_NEIGHBORS: int = 15
UMAP_DEFAULT_MIN_DIST: float = 0.1

# ---------------------------------------------------------------------------
# Similarity / diversity
# ---------------------------------------------------------------------------
SIMILARITY_MAX_MOLECULES: int = 5_000  # sample cap for pairwise Tanimoto

# ---------------------------------------------------------------------------
# Filter range defaults (used when data range is unknown)
# ---------------------------------------------------------------------------
DEFAULT_MW_RANGE = (0.0, 1500.0)
DEFAULT_LOGP_RANGE = (-5.0, 10.0)
DEFAULT_TPSA_RANGE = (0.0, 300.0)
DEFAULT_HBD_MAX: int = 20
DEFAULT_HBA_MAX: int = 20
DEFAULT_ROTBOND_MAX: int = 20

# ---------------------------------------------------------------------------
# Rule-flag thresholds
# ---------------------------------------------------------------------------
LIPINSKI_MW_MAX: float = 500.0
LIPINSKI_LOGP_MAX: float = 5.0
LIPINSKI_HBD_MAX: int = 5
LIPINSKI_HBA_MAX: int = 10
LIPINSKI_ROTBOND_MAX: int = 10

HIGH_FLEXIBILITY_ROTBOND: int = 10
HIGH_LIPOPHILICITY_LOGP: float = 5.0
EXTREME_SIZE_MW_LOW: float = 150.0
EXTREME_SIZE_MW_HIGH: float = 800.0

# ---------------------------------------------------------------------------
# Descriptor columns (order matters for display)
# ---------------------------------------------------------------------------
DESCRIPTOR_COLS: list[str] = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA",
    "RotatableBonds", "RingCount", "FractionCsp3",
    "HeavyAtomCount", "FormalCharge", "QED", "MolFormula",
]

NUMERIC_DESCRIPTOR_COLS: list[str] = [
    c for c in DESCRIPTOR_COLS if c != "MolFormula"
]

FLAG_COLS: list[str] = [
    "passes_lipinski_like",
    "high_flexibility_flag",
    "high_lipophilicity_flag",
    "extreme_size_flag",
]

# ---------------------------------------------------------------------------
# Prioritisation
# ---------------------------------------------------------------------------
PRIORITY_WEIGHTS: dict[str, float] = {
    "pic50": 0.3,
    "qed": 0.3,
    "diversity": 0.2,
    "lipinski": 0.2,
}
PRIORITY_WEIGHTS_NO_PIC50: dict[str, float] = {
    "qed": 0.4,
    "diversity": 0.3,
    "lipinski": 0.3,
}
SHORTLIST_DEFAULT: int = 20
DIVERSE_REPS_DEFAULT: int = 10

# ---------------------------------------------------------------------------
# Enhanced similarity
# ---------------------------------------------------------------------------
TOP_SIMILAR_PAIRS: int = 20
TOP_ISOLATED: int = 10

# ---------------------------------------------------------------------------
# Chemical-space colour options
# ---------------------------------------------------------------------------
COLOR_BY_OPTIONS: list[str] = [
    "pIC50", "MolWt", "LogP", "TPSA", "QED",
    "HBD", "HBA", "RotatableBonds", "FractionCsp3",
    "passes_lipinski_like",
]

# Hover columns for scatter plots
HOVER_EXTRA_COLS: list[str] = [
    "MolWt", "LogP", "TPSA", "QED",
    "passes_lipinski_like", "MolFormula",
]

# ---------------------------------------------------------------------------
# Section subtitles
# ---------------------------------------------------------------------------
SECTION_SUBTITLES: dict[str, str] = {
    "overview": "High-level counts and averages for the uploaded molecular set.",
    "distributions": "Histograms of computed molecular descriptors across valid molecules.",
    "similarity": (
        "Tanimoto similarity metrics computed from 1024-bit Morgan fingerprints (radius 2, ECFP4). "
        "Use these to assess chemical diversity and identify redundancy."
    ),
    "chemical_space": (
        "2-D embeddings of the fingerprint space. Colour and inspect molecules to identify "
        "clusters, outliers, and structural trends."
    ),
    "shortlist": (
        "Ranked by a transparent composite score. Use this to identify molecules worth "
        "inspecting first. This is a heuristic ranking — not a prediction of efficacy."
    ),
    "campaign": (
        "Comparison of the current design round against a reference set. "
        "Novel molecules are those not present in the reference."
    ),
}

# ---------------------------------------------------------------------------
# Large-dataset warning threshold
# ---------------------------------------------------------------------------
LARGE_DATASET_WARN: int = 10_000
