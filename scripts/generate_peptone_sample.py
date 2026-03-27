#!/usr/bin/env python3
"""
Generate data/peptone_campaign_sample.csv

Represents a realistic Peptone generative campaign output:
  - ~200 kinase-inhibitor-like molecules (anilinopyrimidine + indazole series)
  - No stereo annotations (@/@@) as in raw generative model output
  - pIC50 from N(mu=7.0, sigma=0.9) clipped [5.0, 9.5]; ~15% NaN
  - A few edge-case molecules (high MW, high LogP) mixed in

Run from repo root:
    python peptone_evaluator_v2/scripts/generate_peptone_sample.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

try:
    from rdkit import Chem
except ImportError:
    sys.exit("RDKit required: pip install rdkit-pypi")


def is_valid(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


# ---------------------------------------------------------------------------
# Molecule pool — two scaffold families + edge cases
# All without @/@@ stereo (mimics raw generative model output)
# ---------------------------------------------------------------------------

# === Family 1: 2-anilinopyrimidine-4-amine series ===
# Core: aniline connected to pyrimidine C2, amine at C4
# Varied at: aniline para-R1 and C4-amine R2

_ANILINO_R1 = [
    "C", "F", "Cl", "Br", "OC", "CC", "C(F)(F)F", "CN", "S(=O)(=O)C",
    "C#N", "OCC", "C(C)C", "CCC", "NC(=O)C", "OC(F)(F)F",
]
_ANILINO_R2 = [
    "N1CCOCC1",         # morpholine
    "N1CCN(C)CC1",      # N-methylpiperazine
    "N1CCNCC1",         # piperazine
    "N1CCCC1",          # pyrrolidine
    "NC1CCNCC1",        # 4-aminopiperidine
    "N1CCC(F)CC1",      # 4-fluoropiperidine
    "N1CCOC1",          # oxetane-amine
    "NC",               # methylamine
    "NCC",              # ethylamine
    "N(C)CC",           # dimethylamino-ethyl
    "N1CCCCC1",         # piperidine
    "N1CC(O)CC1",       # 3-hydroxypyrrolidine
]

# Build anilinopyrimidine SMILES: c1cnc(Nc2ccc({R1})cc2)nc1{R2}
FAMILY1 = []
for r1 in _ANILINO_R1:
    for r2 in _ANILINO_R2:
        smi = f"c1cnc(Nc2ccc({r1})cc2)nc1{r2}"
        if is_valid(smi):
            FAMILY1.append(smi)
        # meta-substituted aniline variant
        smi2 = f"c1cnc(Nc2cccc({r1})c2)nc1{r2}"
        if is_valid(smi2):
            FAMILY1.append(smi2)

# === Family 2: Indazole–pyrimidine series ===
_INDAZOLE_R = [
    "N1CCOCC1", "N1CCN(C)CC1", "N1CCNCC1", "N1CCCC1",
    "NC", "NCC", "N(C)C", "N1CCCCC1", "N1CC(O)CC1",
]

FAMILY2 = []
for r in _INDAZOLE_R:
    for smi in [
        f"c1cnc(Nc2ccc3[nH]ncc3c2)nc1{r}",   # 5-indazolyl
        f"c1cnc(Nc2ccc3cc[nH]n3c2)nc1{r}",    # 6-indazolyl
        f"c1cnc(Nc2cccc3[nH]ncc23)nc1{r}",    # 4-indazolyl
        f"c1cnc(Nc2ccc3ncc(C)c3c2)nc1{r}",    # methyl-indazolyl
    ]:
        if is_valid(smi):
            FAMILY2.append(smi)

# === Family 3: Pyrrolopyrimidine / purine-like series ===
_PURINE_R1 = ["C", "F", "Cl", "OC", "CC", "C(F)(F)F"]
_PURINE_R2 = [
    "N1CCOCC1", "N1CCN(C)CC1", "N1CCNCC1", "N1CCCC1", "NC", "NCC",
]

FAMILY3 = []
for r1 in _PURINE_R1:
    for r2 in _PURINE_R2:
        for smi in [
            f"c1nc2nc(Nc3ccc({r1})cc3)nc2[nH]1",           # 6-anilinopurine
            f"c1nc2nc({r2})nc2[nH]1",                       # 6-amino purine
            f"c1cnc2[nH]cc(Nc3ccc({r1})cc3)c2n1",           # 7-deazapurine
            f"c1cnc(Nc2ccc({r1})cc2)nc1{r2}",               # back-compat with F1
        ]:
            if is_valid(smi):
                FAMILY3.append(smi)

# === Edge-case molecules ===
EDGE_CASES = [
    # High MW (macrocycle-like)
    "O=C(NCCCCCCCCCCNC(=O)c1ccc(Nc2nccc(N3CCOCC3)n2)cc1)c1ccc(Nc2nccc(N3CCOCC3)n2)cc1",
    # High LogP
    "FC(F)(F)c1ccc(Nc2nccc(N3CCOCC3)n2)c(C(F)(F)F)c1",
    "c1ccc(Nc2nccc(Cc3ccc(Nc4nccc(N5CCOCC5)n4)cc3)n2)cc1",
    # Sulfonamide-containing
    "Cc1ccc(S(=O)(=O)Nc2ccnc(Nc3ccc(F)cc3)n2)cc1",
    "Cc1ccc(S(=O)(=O)Nc2ccnc(N3CCOCC3)n2)cc1",
    # Urea-linked
    "O=C(Nc1ccc(Nc2nccc(N3CCOCC3)n2)cc1)Nc1ccccc1",
    "O=C(Nc1ccc(F)cc1)Nc1ccc(Nc2nccc(N3CCOCC3)n2)cc1",
    # Amide series
    "O=C(c1ccc(F)cc1)Nc1ccnc(Nc2ccc(C)cc2)n1",
    "O=C(c1cccc(Cl)c1)Nc1ccnc(N2CCOCC2)n1",
    "O=C(CCN1CCOCC1)Nc1ccnc(Nc2ccc(C)cc2)n1",
    # Heterocyclic variants
    "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "Fc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "Clc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "COc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    # Bicyclic amines
    "c1cnc(N2CC3CCCC3C2)nc1Nc1ccc(C)cc1",
    "c1cnc(N2CC3CCCCC3C2)nc1Nc1ccc(F)cc1",
    # Oxazole / thiazole linked
    "c1cnc(Nc2ccc(-c3noc(C)n3)cc2)nc1N1CCOCC1",
    "c1cnc(Nc2ccc(-c3nc(C)cs3)cc2)nc1N1CCOCC1",
    # Pyridyl variants
    "c1cnc(Nc2ccncc2)nc1N1CCOCC1",
    "c1cnc(Nc2ccccn2)nc1N1CCOCC1",
    "c1cnc(Nc2cccnc2)nc1N1CCOCC1",
]
EDGE_CASES = [s for s in EDGE_CASES if is_valid(s)]

# ---------------------------------------------------------------------------
# Assemble pool
# ---------------------------------------------------------------------------
pool = FAMILY1 + FAMILY2 + FAMILY3 + EDGE_CASES
# Deduplicate by canonical SMILES
canonical_map: dict[str, str] = {}
for smi in pool:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        can = Chem.MolToSmiles(mol)
        canonical_map[can] = smi  # keep original (no-stereo) SMILES

unique_smiles = list(canonical_map.values())
random.shuffle(unique_smiles)

# Target ~200 molecules
TARGET = 200
if len(unique_smiles) > TARGET:
    unique_smiles = unique_smiles[:TARGET]

# ---------------------------------------------------------------------------
# Assign pIC50 values  (~85% have values, ~15% are NaN)
# ---------------------------------------------------------------------------
n = len(unique_smiles)
n_with_pic50 = int(n * 0.85)

raw_pic50 = np.random.normal(loc=7.0, scale=0.9, size=n_with_pic50)
raw_pic50 = np.clip(raw_pic50, 5.0, 9.5)
raw_pic50 = np.round(raw_pic50, 3)

pic50_values: list = list(raw_pic50) + [float("nan")] * (n - n_with_pic50)
random.shuffle(pic50_values)

# ---------------------------------------------------------------------------
# Build DataFrame and save
# ---------------------------------------------------------------------------
df = pd.DataFrame({"smiles": unique_smiles, "pIC50": pic50_values})

out_path = Path(__file__).resolve().parents[1] / "data" / "peptone_campaign_sample.csv"
out_path.parent.mkdir(exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} molecules to {out_path}")
print(f"  With pIC50:    {df['pIC50'].notna().sum()}")
print(f"  Without pIC50: {df['pIC50'].isna().sum()}")
print(f"  pIC50 range:   {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
