#!/usr/bin/env python3
"""
Generate data/oppenheimer_round2_5k.csv

Simulates a second Oppenheimer design round (~5,000 molecules) with
realistic evolution from the first round (peptone_campaign_sample.csv):
  - ~30% overlap with round 1 (shared canonical SMILES)
  - Expanded R-groups on existing scaffold families
  - New scaffold families (benzimidazole, quinazoline, aminopyridine,
    imidazopyridine, triazine)
  - Improved pIC50 distribution: N(mu=7.3, sigma=0.8), ~12% NaN
  - Edge-case molecules for filter stress-testing

Run from repo root:
    python peptone_evaluator_v2/scripts/generate_oppenheimer_round2.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(99)
np.random.seed(99)

try:
    from rdkit import Chem
except ImportError:
    sys.exit("RDKit required: pip install rdkit-pypi")


def is_valid(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def build_and_validate(templates: list[str]) -> list[str]:
    """Return list of valid SMILES from template list."""
    return [s for s in templates if is_valid(s)]


def combinatorial(cores: list[str], r_groups: dict[str, list[str]]) -> list[str]:
    """
    Build SMILES by substituting {R1}, {R2}, etc. in core templates.
    Returns only valid molecules.
    """
    results = []
    keys = sorted(r_groups.keys())

    def _recurse(template: str, idx: int):
        if idx == len(keys):
            if is_valid(template):
                results.append(template)
            return
        key = keys[idx]
        for val in r_groups[key]:
            _recurse(template.replace(key, val, 1), idx + 1)

    for core in cores:
        _recurse(core, 0)
    return results


# ---------------------------------------------------------------------------
# Round 1 molecules (for overlap) — exact same pools as generate_peptone_sample.py
# ---------------------------------------------------------------------------

_R1_ANILINO_R1 = [
    "C", "F", "Cl", "Br", "OC", "CC", "C(F)(F)F", "CN", "S(=O)(=O)C",
    "C#N", "OCC", "C(C)C", "CCC", "NC(=O)C", "OC(F)(F)F",
]
_R1_ANILINO_R2 = [
    "N1CCOCC1", "N1CCN(C)CC1", "N1CCNCC1", "N1CCCC1",
    "NC1CCNCC1", "N1CCC(F)CC1", "N1CCOC1", "NC", "NCC",
    "N(C)CC", "N1CCCCC1", "N1CC(O)CC1",
]

ROUND1_F1 = []
for r1 in _R1_ANILINO_R1:
    for r2 in _R1_ANILINO_R2:
        smi = f"c1cnc(Nc2ccc({r1})cc2)nc1{r2}"
        if is_valid(smi):
            ROUND1_F1.append(smi)
        smi2 = f"c1cnc(Nc2cccc({r1})c2)nc1{r2}"
        if is_valid(smi2):
            ROUND1_F1.append(smi2)

_R1_INDAZOLE_R = [
    "N1CCOCC1", "N1CCN(C)CC1", "N1CCNCC1", "N1CCCC1",
    "NC", "NCC", "N(C)C", "N1CCCCC1", "N1CC(O)CC1",
]

ROUND1_F2 = []
for r in _R1_INDAZOLE_R:
    for smi in [
        f"c1cnc(Nc2ccc3[nH]ncc3c2)nc1{r}",
        f"c1cnc(Nc2ccc3cc[nH]n3c2)nc1{r}",
        f"c1cnc(Nc2cccc3[nH]ncc23)nc1{r}",
        f"c1cnc(Nc2ccc3ncc(C)c3c2)nc1{r}",
    ]:
        if is_valid(smi):
            ROUND1_F2.append(smi)

_R1_PURINE_R1 = ["C", "F", "Cl", "OC", "CC", "C(F)(F)F"]
_R1_PURINE_R2 = [
    "N1CCOCC1", "N1CCN(C)CC1", "N1CCNCC1", "N1CCCC1", "NC", "NCC",
]

ROUND1_F3 = []
for r1 in _R1_PURINE_R1:
    for r2 in _R1_PURINE_R2:
        for smi in [
            f"c1nc2nc(Nc3ccc({r1})cc3)nc2[nH]1",
            f"c1nc2nc({r2})nc2[nH]1",
            f"c1cnc2[nH]cc(Nc3ccc({r1})cc3)c2n1",
            f"c1cnc(Nc2ccc({r1})cc2)nc1{r2}",
        ]:
            if is_valid(smi):
                ROUND1_F3.append(smi)

ROUND1_EDGE = build_and_validate([
    "O=C(NCCCCCCCCCCNC(=O)c1ccc(Nc2nccc(N3CCOCC3)n2)cc1)c1ccc(Nc2nccc(N3CCOCC3)n2)cc1",
    "FC(F)(F)c1ccc(Nc2nccc(N3CCOCC3)n2)c(C(F)(F)F)c1",
    "c1ccc(Nc2nccc(Cc3ccc(Nc4nccc(N5CCOCC5)n4)cc3)n2)cc1",
    "Cc1ccc(S(=O)(=O)Nc2ccnc(Nc3ccc(F)cc3)n2)cc1",
    "Cc1ccc(S(=O)(=O)Nc2ccnc(N3CCOCC3)n2)cc1",
    "O=C(Nc1ccc(Nc2nccc(N3CCOCC3)n2)cc1)Nc1ccccc1",
    "O=C(Nc1ccc(F)cc1)Nc1ccc(Nc2nccc(N3CCOCC3)n2)cc1",
    "O=C(c1ccc(F)cc1)Nc1ccnc(Nc2ccc(C)cc2)n1",
    "O=C(c1cccc(Cl)c1)Nc1ccnc(N2CCOCC2)n1",
    "O=C(CCN1CCOCC1)Nc1ccnc(Nc2ccc(C)cc2)n1",
    "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "Fc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "Clc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "COc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "c1cnc(N2CC3CCCC3C2)nc1Nc1ccc(C)cc1",
    "c1cnc(N2CC3CCCCC3C2)nc1Nc1ccc(F)cc1",
    "c1cnc(Nc2ccc(-c3noc(C)n3)cc2)nc1N1CCOCC1",
    "c1cnc(Nc2ccc(-c3nc(C)cs3)cc2)nc1N1CCOCC1",
    "c1cnc(Nc2ccncc2)nc1N1CCOCC1",
    "c1cnc(Nc2ccccn2)nc1N1CCOCC1",
    "c1cnc(Nc2cccnc2)nc1N1CCOCC1",
])

# Canonicalize round 1 pool for overlap matching
_round1_all = ROUND1_F1 + ROUND1_F2 + ROUND1_F3 + ROUND1_EDGE
_round1_canonical: dict[str, str] = {}
for smi in _round1_all:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        can = Chem.MolToSmiles(mol)
        _round1_canonical[can] = smi
ROUND1_POOL = list(_round1_canonical.values())
random.shuffle(ROUND1_POOL)

print(f"Round 1 pool size (unique): {len(ROUND1_POOL)}")


# ===========================================================================
# ROUND 2: EXPANDED SCAFFOLD FAMILIES
# ===========================================================================

# Large R-group libraries for maximum combinatorial coverage

# Aniline substituents (para and meta positions)
ANILINE_R = [
    "C", "F", "Cl", "Br", "OC", "CC", "C(F)(F)F", "CN", "C#N",
    "OCC", "C(C)C", "CCC", "NC(=O)C", "OC(F)(F)F", "S(=O)(=O)C",
    "N(C)C", "C(=O)N", "C(=O)NC", "C1CC1", "C1CCC1", "C(C)(C)C",
    "c1ccoc1", "c1ccsc1", "SC", "OCCF", "C(=O)O", "OCCC",
    "NS(=O)(=O)C", "C(=O)NCC", "OC(C)C", "NCC",
]

# Amine caps (C4 position on pyrimidine or equivalent)
AMINE_R = [
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
    "N1CCC(O)CC1",      # 4-hydroxypiperidine
    "N1CCN(CC)CC1",     # N-ethylpiperazine
    "N1CCC(N)CC1",      # 4-aminopiperidine
    "N1CC(C)CC1",       # 3-methylpyrrolidine
    "N1CCCC(O)C1",      # 3-hydroxypiperidine
    "NCC(C)C",          # isobutylamine
    "NCCC",             # propylamine
    "NC1CC1",           # cyclopropylamine
    "NC1CCC1",          # cyclobutylamine
    "N1CC(F)(F)C1",     # difluoroazetidine
    "N1CCC(OC)CC1",     # 4-methoxypiperidine
    "N1CCN(C(=O)C)CC1", # N-acetylpiperazine
    "NCC1CC1",          # cyclopropylmethylamine
    "N(CC)CC",          # diethylamine
    "N1CCCC1C",         # 2-methylpyrrolidine
    "N1CCCCC1C",        # 2-methylpiperidine
]


# --- Evolved Family 1: anilinopyrimidine (greatly expanded) ---
EVOLVED_F1 = combinatorial(
    cores=[
        "c1cnc(Nc2ccc({R1})cc2)nc1{R2}",   # para-aniline
        "c1cnc(Nc2cccc({R1})c2)nc1{R2}",    # meta-aniline
        "c1cnc(Nc2cc({R1})ccc2)nc1{R2}",    # ortho-aniline
    ],
    r_groups={"{R1}": ANILINE_R, "{R2}": AMINE_R},
)

# --- Evolved Family 2: indazole-pyrimidine (expanded) ---
EVOLVED_F2 = combinatorial(
    cores=[
        "c1cnc(Nc2ccc3[nH]ncc3c2)nc1{R2}",    # 5-indazolyl
        "c1cnc(Nc2ccc3cc[nH]n3c2)nc1{R2}",     # 6-indazolyl
        "c1cnc(Nc2cccc3[nH]ncc23)nc1{R2}",     # 4-indazolyl
        "c1cnc(Nc2ccc3ncc(C)c3c2)nc1{R2}",     # methyl-indazolyl
        "c1cnc(Nc2ccc3[nH]nc(C)c3c2)nc1{R2}",  # 3-methyl-indazolyl
    ],
    r_groups={"{R2}": AMINE_R},
)

# --- Evolved Family 3: purine-like (expanded) ---
EVOLVED_F3 = combinatorial(
    cores=[
        "c1nc2nc(Nc3ccc({R1})cc3)nc2[nH]1",     # 6-anilinopurine
        "c1cnc2[nH]cc(Nc3ccc({R1})cc3)c2n1",     # 7-deazapurine
        "c1nc(Nc2ccc({R1})cc2)c2nc[nH]c2n1",     # purine variant
    ],
    r_groups={"{R1}": ANILINE_R},
)


# --- Family 4: Benzimidazole-pyrimidine (NEW) ---
FAMILY4 = combinatorial(
    cores=[
        "c1ccc2[nH]c(-c3ccnc({R2})n3)nc2c1",             # unsubst benzimidazole-C
        "c1ccc2[nH]c(Nc3ccnc({R2})n3)nc2c1",              # unsubst benzimidazole-N
        "c1cc(F)c2[nH]c(-c3ccnc({R2})n3)nc2c1",           # 5-F
        "c1cc(Cl)c2[nH]c(-c3ccnc({R2})n3)nc2c1",          # 5-Cl
        "c1cc(C)c2[nH]c(-c3ccnc({R2})n3)nc2c1",           # 5-Me
        "c1cc(OC)c2[nH]c(-c3ccnc({R2})n3)nc2c1",          # 5-OMe
        "c1cc(C(F)(F)F)c2[nH]c(-c3ccnc({R2})n3)nc2c1",    # 5-CF3
        "c1c(F)cc2[nH]c(-c3ccnc({R2})n3)nc2c1",           # 6-F
        "c1c(Cl)cc2[nH]c(-c3ccnc({R2})n3)nc2c1",          # 6-Cl
        "c1c(C)cc2[nH]c(-c3ccnc({R2})n3)nc2c1",           # 6-Me
    ],
    r_groups={"{R2}": AMINE_R},
)


# --- Family 5: Quinazoline series (NEW — erlotinib/gefitinib-like) ---
FAMILY5 = combinatorial(
    cores=[
        "c1ccc2nc({R2})nc(Nc3ccc({R1})cc3)c2c1",          # 4-anilinoquinazoline
        "c1ccc2nc({R2})nc(Nc3cccc({R1})c3)c2c1",          # meta-aniline
        "COc1cc2nc({R2})nc(Nc3ccc({R1})cc3)c2cc1OC",      # 6,7-dimethoxy
        "COc1cc2nc({R2})nc(Nc3cccc({R1})c3)c2cc1OC",      # 6,7-dimethoxy meta
    ],
    r_groups={
        "{R1}": ANILINE_R[:20],  # subset to avoid over-generation
        "{R2}": AMINE_R[:20],
    },
)


# --- Family 6: 2-Aminopyridine core (NEW — scaffold hop from pyrimidine) ---
FAMILY6 = combinatorial(
    cores=[
        "c1ccnc(Nc2ccc({R1})cc2)c1{R2}",    # 2-anilino-3-amino pyridine
        "c1cc(Nc2ccc({R1})cc2)ncc1{R2}",     # 4-anilino-3-amino pyridine
        "c1ccnc(Nc2cccc({R1})c2)c1{R2}",     # meta-aniline variant
    ],
    r_groups={
        "{R1}": ANILINE_R[:20],
        "{R2}": AMINE_R[:20],
    },
)


# --- Family 7: Imidazo[1,2-a]pyridine (NEW) ---
FAMILY7 = combinatorial(
    cores=[
        "c1cnc2cc(Nc3ccc({R1})cc3)cn2c1",      # 6-anilino
        "c1cnc2cc(Nc3cccc({R1})c3)cn2c1",       # 6-anilino meta
        "c1cnc2c(Nc3ccc({R1})cc3)ccn2c1",       # 5-anilino
    ],
    r_groups={"{R1}": ANILINE_R},
)


# --- Family 8: Triazine series (NEW) ---
FAMILY8 = combinatorial(
    cores=[
        "c1nc(Nc2ccc({R1})cc2)nc(n1){R2}",    # 1,3,5-triazine
        "c1nc(Nc2cccc({R1})c2)nc(n1){R2}",    # meta-aniline triazine
    ],
    r_groups={
        "{R1}": ANILINE_R[:20],
        "{R2}": AMINE_R[:20],
    },
)


# --- Edge cases (expanded) ---
ROUND2_EDGE = build_and_validate([
    # Macrocyclic linkers
    "O=C(NCCCCCNC(=O)c1ccc(Nc2nccc(N3CCOCC3)n2)cc1)c1ccc(Nc2nccc(N3CCN(C)CC3)n2)cc1",
    "O=C(NCCCCCCNC(=O)c1ccc2[nH]c(-c3ccnc(N4CCOCC4)n3)nc2c1)c1ccccc1",
    # Very high MW
    "O=C(Nc1ccc(Nc2nccc(N3CCN(CCOC)CC3)n2)cc1)c1ccc(Nc2nccc(N3CCN(CCOC)CC3)n2)cc1",
    # Multi-ring systems
    "c1cnc(Nc2ccc3c(c2)CCN(C(=O)c2ccc(F)cc2)C3)nc1N1CCOCC1",
    "c1cnc(Nc2ccc3c(c2)OCO3)nc1N1CCN(C)CC1",
    "c1cnc(Nc2ccc3c(c2)OCCO3)nc1N1CCOCC1",
    # Spiro compounds
    "c1cnc(N2CCC3(CCOC3)CC2)nc1Nc1ccc(F)cc1",
    "c1cnc(N2CCC3(CCOCC3)CC2)nc1Nc1ccc(C)cc1",
    # Bridged bicyclic
    "c1cnc(NC2CC3CCC2C3)nc1Nc1ccc(F)cc1",
    "c1cnc(NC2CC3CCC(C3)C2)nc1Nc1ccc(Cl)cc1",
    # Heavily fluorinated
    "FC(F)(F)c1cc(C(F)(F)F)cc(Nc2nccc(N3CCOCC3)n2)c1",
    "FC(F)(F)c1ccc(Nc2nccc(N3CCN(C)CC3)n2)c(F)c1",
    # Sulfonamide variations
    "O=S(=O)(c1ccc(F)cc1)Nc1ccnc(Nc2ccc(C)cc2)n1",
    "O=S(=O)(c1cccc(Cl)c1)Nc1ccnc(N2CCOCC2)n1",
    "CS(=O)(=O)Nc1ccnc(Nc2ccc(OC)cc2)n1",
    # Urea expansions
    "O=C(Nc1ccc(Nc2nccc(N3CCN(C)CC3)n2)cc1)NC1CCCCC1",
    "O=C(Nc1cccc(Nc2nccc(N3CCOCC3)n2)c1)Nc1ccc(F)cc1",
    # Amide expansions
    "O=C(c1ccc(OC)cc1)Nc1ccnc(Nc2ccc(F)cc2)n1",
    "O=C(c1ccncc1)Nc1ccnc(N2CCOCC2)n1",
    "O=C(c1cccs1)Nc1ccnc(Nc2ccc(C)cc2)n1",
    "O=C(c1ccco1)Nc1ccnc(N2CCN(C)CC2)n1",
    # Oxazole/thiazole
    "c1cnc(Nc2ccc(-c3nc(CC)co3)cc2)nc1N1CCOCC1",
    "c1cnc(Nc2ccc(-c3nc(CC)cs3)cc2)nc1N1CCN(C)CC1",
    "c1cnc(Nc2ccc(-c3noc(CC)n3)cc2)nc1N1CCNCC1",
    # Pyridyl variants
    "c1cnc(Nc2ccnc(C)c2)nc1N1CCOCC1",
    "c1cnc(Nc2ccnc(F)c2)nc1N1CCN(C)CC1",
    "c1cnc(Nc2cc(C)ccn2)nc1N1CCCC1",
    # Bicyclic amine caps
    "c1cnc(N2CC3CCCC3C2)nc1Nc1ccc(OC)cc1",
    "c1cnc(N2CC3CCCCC3C2)nc1Nc1ccc(Cl)cc1",
    "c1cnc(N2CC3CCC3C2)nc1Nc1ccc(C)cc1",
    # Aminopyridine core swaps
    "c1ccnc(Nc2ccc(C)cc2)c1N1CCOCC1",
    "c1ccnc(Nc2ccc(F)cc2)c1N1CCN(C)CC1",
    "c1ccnc(Nc2ccc(Cl)cc2)c1N1CCNCC1",
    # Imidazopyridine
    "c1cnc2ncc(Nc3ccc(C)cc3)n2c1",
    "c1cnc2ncc(Nc3ccc(F)cc3)n2c1",
    "c1cnc2ncc(Nc3ccc(Cl)cc3)n2c1",
    "c1cnc2ncc(Nc3ccc(OC)cc3)n2c1",
    # High TPSA
    "O=C(NO)c1ccc(Nc2nccc(N3CCN(C(=O)O)CC3)n2)cc1",
    "O=C(O)CCN1CCN(c2ccnc(Nc3ccc(S(N)(=O)=O)cc3)n2)CC1",
])


# ---------------------------------------------------------------------------
# Assemble, deduplicate, sample
# ---------------------------------------------------------------------------

all_new_pools = {
    "Evolved F1 (anilinopyrimidine)": EVOLVED_F1,
    "Evolved F2 (indazole)": EVOLVED_F2,
    "Evolved F3 (purine)": EVOLVED_F3,
    "Family 4 (benzimidazole)": FAMILY4,
    "Family 5 (quinazoline)": FAMILY5,
    "Family 6 (aminopyridine)": FAMILY6,
    "Family 7 (imidazopyridine)": FAMILY7,
    "Family 8 (triazine)": FAMILY8,
    "Edge cases": ROUND2_EDGE,
}

print("Building pools...")
total_raw = 0
for name, pool in all_new_pools.items():
    print(f"  {name}: {len(pool)}")
    total_raw += len(pool)
print(f"  Total raw: {total_raw}")

# Combine all new molecules
all_new = []
for pool in all_new_pools.values():
    all_new.extend(pool)

# Canonicalize and deduplicate — exclude round 1 overlap from "new" pool
new_canonical: dict[str, str] = {}
for smi in all_new:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        can = Chem.MolToSmiles(mol)
        if can not in _round1_canonical:
            new_canonical[can] = smi

new_unique = list(new_canonical.values())
random.shuffle(new_unique)
print(f"\nNew unique molecules (excl. round 1 overlap): {len(new_unique)}")

# Target composition
TARGET = 5000
OVERLAP_TARGET = 1500  # ~30% from round 1

# Sample overlap from round 1
overlap_count = min(OVERLAP_TARGET, len(ROUND1_POOL))
overlap_smiles = ROUND1_POOL[:overlap_count]

# Fill remaining from new molecules
remaining = TARGET - overlap_count
if len(new_unique) < remaining:
    print(f"NOTE: {len(new_unique)} new molecules available, need {remaining}")
    remaining = len(new_unique)
new_smiles = new_unique[:remaining]

all_smiles = overlap_smiles + new_smiles
random.shuffle(all_smiles)

print(f"\nFinal dataset:")
print(f"  Overlap (from round 1): {len(overlap_smiles)}")
print(f"  New molecules: {len(new_smiles)}")
print(f"  Total: {len(all_smiles)}")

# ---------------------------------------------------------------------------
# Assign pIC50 — improved distribution for round 2
# ---------------------------------------------------------------------------
n = len(all_smiles)
n_with_pic50 = int(n * 0.88)  # 88% have values (up from 85% in round 1)

raw_pic50 = np.random.normal(loc=7.3, scale=0.8, size=n_with_pic50)
raw_pic50 = np.clip(raw_pic50, 5.0, 9.5)
raw_pic50 = np.round(raw_pic50, 3)

pic50_values: list = list(raw_pic50) + [float("nan")] * (n - n_with_pic50)
random.shuffle(pic50_values)

# ---------------------------------------------------------------------------
# Build DataFrame and save
# ---------------------------------------------------------------------------
df = pd.DataFrame({"smiles": all_smiles, "pIC50": pic50_values})

out_path = Path(__file__).resolve().parents[1] / "data" / "oppenheimer_round2_5k.csv"
out_path.parent.mkdir(exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\nSaved {len(df)} molecules to {out_path}")
print(f"  With pIC50:    {df['pIC50'].notna().sum()}")
print(f"  Without pIC50: {df['pIC50'].isna().sum()}")
print(f"  pIC50 range:   {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
print(f"  pIC50 mean:    {df['pIC50'].mean():.2f}")
print(f"  pIC50 std:     {df['pIC50'].std():.2f}")
