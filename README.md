# Peptone Generative Output Evaluator v2

A modular Streamlit dashboard for evaluating generative molecular outputs via chemical-space analysis, descriptor computation, and rule-based filtering.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload a CSV with at least a **SMILES** column. A **pIC50** column is optional — if present, scatter plots will be coloured by binding affinity.

## Features

- **Descriptor computation** — MolWt, LogP, TPSA, HBD, HBA, Rotatable Bonds, Ring Count, Fraction Csp3, Heavy Atom Count, Formal Charge, QED, Molecular Formula
- **Rule-based flags** — Lipinski-like filter, high flexibility, high lipophilicity, extreme size
- **Dimensionality reduction** — PCA, t-SNE, and UMAP on 1024-bit Morgan fingerprints (ECFP4)
- **Similarity & diversity** — pairwise Tanimoto, nearest-neighbour similarity, diversity score (with automatic sampling for large datasets)
- **Interactive filters** — sidebar sliders for all descriptors, toggle for valid/unique/drug-like molecules
- **Export** — download processed data as CSV

## Project Structure

```
peptone_evaluator_v2/
├── app.py                          # Streamlit entrypoint
├── requirements.txt
├── README.md
└── src/
    ├── config.py                   # Constants and thresholds
    ├── data_loader.py              # CSV loading and column normalisation
    ├── chemistry.py                # SMILES parsing and canonicalisation
    ├── descriptors.py              # Molecular descriptors and rule flags
    ├── fingerprints.py             # Morgan fingerprint generation
    ├── dimensionality_reduction.py # PCA / t-SNE / UMAP
    ├── similarity.py               # Tanimoto similarity and diversity
    ├── filters.py                  # DataFrame filtering logic
    ├── plotting.py                 # Plotly figure builders
    └── ui.py                       # Streamlit UI components
```

## Design Decisions

- **Single cached enrichment pass** — RDKit Mol objects are created, used for descriptors and fingerprints, then discarded before Streamlit's cache serialises the result.
- **Filters apply to display, not computation** — dimensionality reduction runs on all valid molecules; filters only control which points are shown, preventing coordinate shifts on filter change.
- **Similarity sampling** — pairwise Tanimoto is computed on up to 5,000 molecules; larger datasets are randomly sampled and flagged in the UI.
- **Optional pIC50** — the app works fully without a pIC50 column; colour-by-activity features are conditional.

## Future Extensions

- Substructure search and highlighting
- Scaffold analysis (Murcko decomposition)
- Multi-file comparison (reference vs. generated set)
- Custom descriptor upload or user-defined QSAR models
- Batch export of filtered molecule sets in SDF format
- Integration with external databases (ChEMBL, PubChem)
