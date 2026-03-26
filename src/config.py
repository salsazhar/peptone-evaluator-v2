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

# ---------------------------------------------------------------------------
# Glossary — definitions for metrics, descriptors, and concepts
# ---------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    # Dataset metrics
    "Total molecules": "Number of rows in the uploaded CSV.",
    "Valid": "Molecules with SMILES that RDKit could parse into a valid chemical structure.",
    "Invalid": "Rows where SMILES parsing failed. These are excluded from all analyses.",
    "Unique": "Distinct canonical SMILES among valid molecules. Duplicates share the same canonical form.",

    # Descriptor averages
    "Avg pIC50": "Mean of -log₁₀(IC50) across valid molecules. Higher = more potent.",
    "Avg MW": "Mean molecular weight (Da) across valid molecules.",
    "Avg LogP": "Mean octanol-water partition coefficient. Measures lipophilicity.",
    "Avg TPSA": "Mean topological polar surface area (Å²). Relates to membrane permeability.",

    # Similarity metrics
    "Mean NN Sim": (
        "Average nearest-neighbour Tanimoto similarity. Each molecule's highest "
        "similarity to any other molecule, averaged across the set. "
        "High values indicate clusters of very similar compounds."
    ),
    "Diversity": (
        "1 − mean pairwise Tanimoto similarity. Ranges from 0 (identical set) to 1 "
        "(maximally diverse). Computed from 1024-bit Morgan fingerprints."
    ),
    "Mean Pairwise Sim": (
        "Average Tanimoto similarity across all molecule pairs. "
        "Lower values indicate greater structural diversity."
    ),
    "Diversity Score": (
        "1 − mean pairwise Tanimoto similarity. A simple, transparent diversity metric."
    ),
    "Duplicates": "Molecules sharing the same canonical SMILES. Counted after canonicalisation.",

    # Descriptors
    "MolWt": "Exact molecular weight in Daltons.",
    "LogP": "Wildman-Crippen octanol-water partition coefficient. Measures lipophilicity.",
    "TPSA": "Topological polar surface area (Å²). Correlates with oral absorption and BBB penetration.",
    "HBD": "Hydrogen bond donor count. Lipinski rule: ≤ 5.",
    "HBA": "Hydrogen bond acceptor count. Lipinski rule: ≤ 10.",
    "RotatableBonds": "Number of freely rotating bonds. Affects conformational flexibility.",
    "RingCount": "Total number of rings (aromatic + aliphatic).",
    "FractionCsp3": "Fraction of sp3-hybridised carbons. Higher values suggest more 3-D character.",
    "HeavyAtomCount": "Number of non-hydrogen atoms.",
    "FormalCharge": "Net formal charge of the molecule.",
    "QED": "Quantitative Estimate of Drug-likeness (0–1). Composite of MW, LogP, HBD, HBA, PSA, RotBonds, Arom rings, Alerts.",
    "MolFormula": "Molecular formula derived from the structure.",
    "pIC50": "Negative log₁₀ of IC50 (M). Higher values indicate greater potency.",

    # Rule flags
    "passes_lipinski_like": (
        "True if MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10, RotBonds ≤ 10. "
        "Heuristic screen only — not a developability predictor."
    ),
    "high_flexibility_flag": "Rotatable bonds > 10. Highly flexible molecules may have poor oral bioavailability.",
    "high_lipophilicity_flag": "LogP > 5. May indicate poor solubility or high metabolic clearance.",
    "extreme_size_flag": "MW < 150 or MW > 800. Outside typical small-molecule drug range.",
    "Med-chem heuristic screen": (
        "Rule-based Lipinski-like filter: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10, "
        "RotBonds ≤ 10. This is a simple heuristic, not a stability or developability predictor."
    ),

    # Scaffold analysis
    "Unique Scaffolds": "Number of distinct Murcko scaffolds. Higher = more structural diversity.",
    "Scaffold Ratio": "Unique scaffolds / total molecules. 1.0 = every molecule has a unique scaffold.",
    "Singletons": "Scaffolds appearing exactly once. High singleton fraction indicates novelty.",
    "Top Scaffold": "Most frequently occurring Murcko scaffold and its share of the dataset.",
    "Generic Frameworks": "Murcko frameworks with all atoms → C and all bonds → single. Groups similar scaffolds more aggressively.",
    "Framework Ratio": "Unique generic frameworks / total molecules.",

    # Similarity section
    "Matches": "Number of molecules containing the queried substructure.",
    "Searched": "Total valid molecules searched.",
    "Hit Rate": "Fraction of searched molecules containing the substructure.",

    # Campaign comparison
    "Novel compounds": "Molecules in the current set whose canonical SMILES do not appear in the reference set.",
    "Overlap": "Molecules present in both the current and reference sets.",
    "Reference unique": "Distinct canonical SMILES in the reference set.",

    # Chemical space
    "PCA": "Principal Component Analysis. Linear projection preserving maximum variance.",
    "t-SNE": "t-distributed Stochastic Neighbour Embedding. Non-linear; preserves local structure.",
    "UMAP": "Uniform Manifold Approximation and Projection. Non-linear; preserves both local and global structure.",
    "Perplexity": "t-SNE parameter balancing local vs. global structure. Higher = more global.",
    "n_neighbors": "UMAP parameter controlling local neighbourhood size. Larger = more global.",
    "min_dist": "UMAP parameter controlling minimum distance between points. Smaller = tighter clusters.",

    # Priority shortlist
    "priority_score": "Composite ranking score combining potency, drug-likeness, diversity, and rule compliance.",
}
