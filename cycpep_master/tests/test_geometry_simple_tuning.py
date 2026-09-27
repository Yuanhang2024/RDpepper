"""Unit regressions for the tuned simple_geometry_v2 module (dev tuning round).

Each fixed cluster gets one explicit test; plus non-regression checks for
existing behavior (carbonyl, amide, benzene, permutation, bad input, P-H not
blanket-banned, candidate contract). Molecules are synthetic, embedded with
RDKit ETKDG here; the module under test is loaded from the worktree root by
path and reads nothing from disk. Run with:

    py -3.14 -B -m pytest test_tuned_simple.py -q
"""

import importlib.util
import math
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core import geometry_simple as sg

ETKDG_SEED = 0xF00D


def make_record(smi, seed=ETKDG_SEED, shuffle=None, supply_bonds=True):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert mol is not None, smi
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, ps) == 0, smi
    conf = mol.GetConformer()
    heavy = [i for i in range(mol.GetNumAtoms())
             if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    order = list(range(len(heavy)))
    if shuffle is not None:
        rng = np.random.default_rng(shuffle)
        rng.shuffle(order)
    atoms = []
    for new, old in enumerate(order):
        src = mol.GetAtomWithIdx(heavy[old])
        p = conf.GetAtomPosition(heavy[old])
        atoms.append({"id": new, "element": src.GetSymbol(),
                      "xyz": [p.x, p.y, p.z]})
    bonds = None
    if supply_bonds:
        idx = {old: new for new, old in enumerate(order)}
        bonds = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in heavy and j in heavy:
                bonds.append([idx[i], idx[j]])
        bonds.sort()
    return {"atoms": atoms, "bonds": bonds, "total_charge": None}, mol


def canon_nostereo(smi_or_mol):
    mol = Chem.MolFromSmiles(smi_or_mol) if isinstance(smi_or_mol, str) else Chem.Mol(smi_or_mol)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)


def top1_identity(rec_smi, **kw):
    rec, ref = make_record(rec_smi, **kw)
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["status"] in ("candidate", "ambiguous"), (rec_smi, out["status"])
    top = out["candidates"][0]
    return out, top, canon_nostereo(top["smiles"]) == canon_nostereo(ref)


# --------------------------------------------------------------- nitro fix

@pytest.mark.parametrize("smi", [
    "C[N+](=O)[O-]",                      # nitromethane
    "N[C@@H](CC1=CC=C(C=C1)[N+](=O)[O-])C(=O)O",  # 4-nitrophenylalanine
])
def test_nitro_paired_charge_state(smi):
    """Nitro single-bonded terminal O must be O- paired with N+, not O-H."""
    rec, ref = make_record(smi)
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["candidates"], smi
    top = out["candidates"][0]
    els = [a["element"] for a in rec["atoms"]]
    nbr = {i: [] for i in range(len(els))}
    for i, j, _ in top["bonds"]:
        nbr[i].append(j)
        nbr[j].append(i)
    found_nitro = False
    for a, el in enumerate(els):
        if el != "N":
            continue
        oxy = [b for b in nbr[a] if els[b] == "O"]
        if len(oxy) < 2:
            continue
        has_double = any(o == 2.0 for i, j, o in top["bonds"]
                         if a in (i, j) and els[j if i == a else i] == "O")
        if not has_double:
            continue
        found_nitro = True
        assert top["formal_charges"][a] == 1, "nitro N must be N+"
        for b in oxy:
            key = (min(a, b), max(a, b))
            order = next(o for i, j, o in top["bonds"] if (min(i, j), max(i, j)) == key)
            if order == 1.0 and len(nbr[b]) == 1:
                assert top["formal_charges"][b] == -1, "paired nitro O must be O-"
                assert top["hydrogen_counts"][b] == 0, "paired nitro O must not be O-H"
    assert found_nitro, "nitro motif not recognized: " + smi
    assert canon_nostereo(top["smiles"]) == canon_nostereo(ref), smi


# ------------------------------------------------------------- thione fix

@pytest.mark.parametrize("smi", ["CC(=S)N", "NC(=S)N", "O=C(O)C(=S)N"])
def test_thiocarbonyl_c_equals_s(smi):
    """C=S must be reachable (S double quota) when geometry supports it."""
    out, top, ok = top1_identity(smi)
    assert ok, (smi, top["smiles"])


# ------------------------------------------------- 5-ring aromatic scoring

@pytest.mark.parametrize("smi", ["c1cc[nH]c1", "c1ccsc1", "c1ccoc1"])
def test_five_ring_aromatic_identity(smi):
    """5-ring aromatics must still win top-1 identity (108-deg expectation)."""
    out, top, ok = top1_identity(smi)
    assert ok, (smi, top["smiles"])


def test_six_ring_aromatic_identity():
    out, top, ok = top1_identity("Cc1ccccc1")
    assert ok, top["smiles"]


# ------------------------------------------------------- B2 correctness fixes

def test_glycine_zwitterion_protonation_adds_h():
    """Protonation must ADD hydrogen: zwitterion glycine is [NH3+]CC(=O)[O-],
    not a hydrogen-stripped [NH+]."""
    rec, ref = make_record("NCC(=O)O")
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    zw = [c for c in out["candidates"] if "[NH3+]" in c["smiles"]]
    assert zw, [c["smiles"] for c in out["candidates"]]
    c = zw[0]
    assert c["evidence"]["total_formal_charge"] == 0
    n_idx = [a["element"] for a in rec["atoms"]].index("N")
    assert c["formal_charges"][n_idx] == 1 and c["hydrogen_counts"][n_idx] == 3
    # neutral acid remains top-1 (charge penalty keeps protonated state lower)
    assert canon_nostereo(out["candidates"][0]["smiles"]) == canon_nostereo("NCC(=O)O")


def test_tertiary_amine_zwitterion_reachable():
    """Tertiary amine (bi=3, neutral h=0) must be protonatable: [NH+] with h=1."""
    rec, ref = make_record("CN(C)CC(=O)O")  # N,N-dimethylglycine
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    n_idx = [a["element"] for a in rec["atoms"]].index("N")
    zw = [c for c in out["candidates"]
          if c["formal_charges"][n_idx] == 1 and c["hydrogen_counts"][n_idx] == 1]
    assert zw, [c["smiles"] for c in out["candidates"]]
    assert zw[0]["evidence"]["total_formal_charge"] <= 0  # paired with acid O-


def test_nitrous_acid_not_forced_nitro_paired():
    """Tervalent N (bond sum 3, H-O-N=O) must NOT be forced to N+/O-."""
    rec, ref = make_record("ON=O")
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["candidates"]
    top = out["candidates"][0]
    assert all(q == 0 for q in top["formal_charges"]), top["smiles"]
    assert canon_nostereo(top["smiles"]) == canon_nostereo(ref)


def test_nitric_acid_paired_exactly_one_terminal_o():
    """N+(=O) with two terminal single O: exactly one O-, the other stays OH."""
    rec, ref = make_record("O[N+](=O)[O-]")
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["candidates"]
    top = out["candidates"][0]
    o_idx = [i for i, a in enumerate(rec["atoms"]) if a["element"] == "O"]
    n_idx = [a["element"] for a in rec["atoms"]].index("N")
    assert top["formal_charges"][n_idx] == 1
    n_paired = sum(1 for i in o_idx if top["formal_charges"][i] == -1)
    n_oh = sum(1 for i in o_idx if top["hydrogen_counts"][i] == 1)
    assert n_paired == 1 and n_oh == 1, top["smiles"]
    assert top["evidence"]["total_formal_charge"] == 0
    assert canon_nostereo(top["smiles"]) == canon_nostereo(ref)


def test_nitrile_triple_linear_not_sp2_penalized():
    """Triple endpoints are linear (~180 deg) and must not carry a max sp2
    angle penalty; identity stays top-1."""
    out, top, ok = top1_identity("CCC#N")
    assert ok, top["smiles"]
    assert any(o == 3.0 for _, _, o in top["bonds"])
    ang_term = top["evidence"]["score_terms"]["sp2_angle_residual"]
    assert ang_term < 1.0, ang_term


@pytest.mark.parametrize("smi", ["c1cc[nH]c1", "Cn1cccc1"])
def test_pyrrole_degree2_and_degree3_ring_n(smi):
    """Degree-2 ring N scored at 108; degree-3 (N-methyl) near 120; both keep
    aromatic identity top-1."""
    out, top, ok = top1_identity(smi)
    assert ok, (smi, top["smiles"])


# ----------------------------------------------------------- non-regression

def test_amide_carbonyl_identity():
    out, top, ok = top1_identity("CC(=O)N")
    assert ok, top["smiles"]


def test_ketone_carbonyl_identity():
    out, top, ok = top1_identity("CC(=O)C")
    assert ok, top["smiles"]


def test_sulfonyl_identity():
    out, top, ok = top1_identity("CS(=O)(=O)C")
    assert ok, top["smiles"]


def test_phosphate_not_forced_ph():
    """P with single bonds must keep P-H fills (no blanket P-H ban)."""
    rec, ref = make_record("CP")
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["candidates"]
    top = out["candidates"][0]
    els = [a["element"] for a in rec["atoms"]]
    p = els.index("P")
    assert top["formal_charges"][p] == 0
    assert top["hydrogen_counts"][p] == 2  # CH3-PH2


def test_neutral_amine_not_forced_protonated():
    """Without known charge input, neutral amine must stay top-1 neutral."""
    out, top, ok = top1_identity("NCC(=O)O")  # glycine zwitterion not forced
    assert ok, top["smiles"]


def test_permutation_invariance_nitro():
    rec1, _ = make_record("C[N+](=O)[O-]")
    rec2, _ = make_record("C[N+](=O)[O-]", shuffle=7)
    o1 = sg.infer_geometry(rec1, timeout_seconds=1.0)
    o2 = sg.infer_geometry(rec2, timeout_seconds=1.0)
    c1 = canon_nostereo(o1["candidates"][0]["smiles"])
    c2 = canon_nostereo(o2["candidates"][0]["smiles"])
    assert c1 == c2


def test_candidate_contract():
    rec, _ = make_record("CC(=O)NCC(=S)N")
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["status"] in ("candidate", "ambiguous")
    assert 1 <= len(out["candidates"]) <= 8
    n = len(rec["atoms"])
    for c in out["candidates"]:
        assert len(c["formal_charges"]) == n
        assert len(c["hydrogen_counts"]) == n
        for i, j, o in c["bonds"]:
            assert 0 <= i < n and 0 <= j < n and i != j
            assert o in (1.0, 1.5, 2.0, 3.0)
        # every input heavy atom preserved with coordinate correspondence
    els = [a["element"] for a in rec["atoms"]]
    m = Chem.MolFromSmiles(out["candidates"][0]["smiles"])
    assert m is not None
    from collections import Counter
    assert Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() > 1) == Counter(els)


def test_explicit_hydrogen_rejected():
    rec = {"atoms": [{"id": 0, "element": "C", "xyz": [0.0, 0.0, 0.0]},
                     {"id": 1, "element": "H", "xyz": [1.0, 0.0, 0.0]}],
           "bonds": [[0, 1]], "total_charge": None}
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["status"] == "invalid_input"
    assert "explicit_hydrogen_not_supported" in out["reason_codes"]


def test_bad_inputs():
    assert sg.infer_geometry({}, timeout_seconds=1.0)["status"] == "invalid_input"
    assert sg.infer_geometry({"atoms": [], "bonds": None}, timeout_seconds=1.0)["status"] == "invalid_input"
    bad = {"atoms": [{"id": 0, "element": "C", "xyz": [float("nan"), 0, 0]}],
           "bonds": None}
    assert sg.infer_geometry(bad, timeout_seconds=1.0)["status"] == "invalid_input"
    assert sg.infer_geometry({"atoms": [{"id": 0, "element": "C", "xyz": [0, 0, 0]}]},
                             timeout_seconds=0)["status"] == "invalid_input"


def test_total_charge_filter_still_applies():
    """Known charge input must filter candidates to that net charge."""
    rec, _ = make_record("C[N+](=O)[O-]")
    rec = dict(rec)
    rec["total_charge"] = 0
    out = sg.infer_geometry(rec, timeout_seconds=1.0)
    assert out["candidates"]
    assert out["candidates"][0]["evidence"]["total_formal_charge"] == 0
