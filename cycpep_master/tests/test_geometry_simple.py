"""Unit tests for simple_geometry.infer_geometry (simple_geometry_v2).

All test molecules are synthetic, generated inside this file with RDKit
(ETKDG, fixed seeds). The inference module under test reads nothing from
disk and does not import this file. Run with:

    py -3.14 -B -m pytest test_simple_geometry.py -q
"""

import json
import os
import sys
import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core import geometry_simple as sg

ETKDG_SEED = 0xF00D


def make_record(smi, seed=ETKDG_SEED, shuffle=None, supply_bonds=True):
    """Heavy-atom record (contract format) from a SMILES via 3D embed.

    Optional shuffle deterministically permutes atom order (ids stay
    zero-based positional) to exercise permutation invariance.
    """
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
        remap = {old: new for new, old in enumerate(order)}
        bonds = sorted([sorted([remap[b.GetBeginAtomIdx()], remap[b.GetEndAtomIdx()]])
                        for b in mol.GetBonds()
                        if b.GetBeginAtomIdx() in remap and b.GetEndAtomIdx() in remap])
    return {"atoms": atoms, "bonds": bonds, "total_charge": None}


def top(out):
    assert out["candidates"], "no candidates: %s" % out["reason_codes"]
    return out["candidates"][0]


def bond_pairs_orders(cand):
    return {(b[0], b[1]): b[2] for b in cand["bonds"]}


# --------------------------------------------------------------------------
# carbonyl / amide / amine
# --------------------------------------------------------------------------

def test_glycine_carbonyl_and_charge_states():
    out = sg.infer_geometry(make_record("NCC(=O)O"))
    assert out["status"] == "candidate"
    t = top(out)
    assert t["smiles"] == "NCC(=O)O"
    orders = bond_pairs_orders(t)
    assert 2.0 in orders.values()  # C=O inferred from geometry
    assert len(t["formal_charges"]) == 5 and len(t["hydrogen_counts"]) == 5
    totals = {c["evidence"]["total_formal_charge"] for c in out["candidates"]}
    assert totals == {0, -1}  # neutral acid + carboxylate (zwitterion folded in)
    assert any("[NH3+]" in c["smiles"] for c in out["candidates"])


def test_amide_vs_amine():
    amide = sg.infer_geometry(make_record("CC(=O)N"))
    amine = sg.infer_geometry(make_record("CCN"))
    assert top(amide)["smiles"] == "CC(N)=O"
    assert 2.0 in bond_pairs_orders(top(amide)).values()
    assert top(amine)["smiles"] == "CCN"
    assert set(bond_pairs_orders(top(amine)).values()) == {1.0}
    # geometry evidence distinguishes them: planar carbonyl C in the amide
    carb_c = max(amide["evidence"]["atom_geometry"], key=lambda a: a["angle_sum_deg"])
    assert carb_c["angle_sum_deg"] > 345.0


# --------------------------------------------------------------------------
# aromatic vs saturated rings
# --------------------------------------------------------------------------

def test_benzene_vs_cyclohexane():
    benz = sg.infer_geometry(make_record("c1ccccc1"))
    cyclo = sg.infer_geometry(make_record("C1CCCCC1"))
    assert top(benz)["smiles"] == "c1ccccc1"
    assert set(bond_pairs_orders(top(benz)).values()) == {1.5}
    assert top(cyclo)["smiles"] == "C1CCCCC1"
    assert set(bond_pairs_orders(top(cyclo)).values()) == {1.0}
    b_ring = benz["evidence"]["ring_geometry"][0]
    c_ring = cyclo["evidence"]["ring_geometry"][0]
    assert b_ring["decision"] == "aromatic" and c_ring["decision"] == "saturated"
    assert b_ring["plane_rms_a"] < c_ring["plane_rms_a"]
    assert b_ring["mean_angle_deg"] > c_ring["mean_angle_deg"]
    assert b_ring["mean_relative_bond_length"] < c_ring["mean_relative_bond_length"]


def test_pyrrole_vs_pyrrolidine():
    arom = sg.infer_geometry(make_record("c1cc[nH]c1"))
    sat = sg.infer_geometry(make_record("C1CCNC1"))
    assert top(arom)["smiles"] == "c1cc[nH]c1"
    rec = make_record("c1cc[nH]c1")
    n_h = [h for a, h in zip(rec["atoms"], top(arom)["hydrogen_counts"])
           if a["element"] == "N"]
    assert n_h == [1]  # donor N carries the ring H
    assert sum(top(arom)["hydrogen_counts"]) == 5  # 4x CH + NH
    assert 1.5 in set(bond_pairs_orders(top(arom)).values())
    assert top(sat)["smiles"] == "C1CCNC1"
    assert set(bond_pairs_orders(top(sat)).values()) == {1.0}


def test_tautomer_ambiguity_is_honest():
    out = sg.infer_geometry(make_record("c1nnn[nH]1"))  # tetrazole
    assert out["status"] == "ambiguous"
    assert len(out["candidates"]) == 2
    smiles = [c["smiles"] for c in out["candidates"]]
    assert smiles[0] != smiles[1]
    assert out["candidates"][1]["score"] - out["candidates"][0]["score"] < 0.20
    assert all(c["evidence"]["total_formal_charge"] == 0 for c in out["candidates"])


# --------------------------------------------------------------------------
# P / S handling
# --------------------------------------------------------------------------

def test_supported_ps_motifs():
    sulfone = sg.infer_geometry(make_record("CS(=O)(=O)C"))
    assert "sulfonyl_pattern" in sulfone["reason_codes"]
    n_s_o_double = sum(1 for b in top(sulfone)["bonds"] if b[2] == 2.0)
    assert n_s_o_double == 2
    phosph = sg.infer_geometry(make_record("CP(=O)(O)O"))
    assert "phosphoryl_pattern" in phosph["reason_codes"]
    assert top(phosph)["smiles"] == "CP(=O)(O)O"


def test_ps_conservative_and_ph_allowed():
    # dimethylphosphine CP(C): P-H arises from plain valence filling
    ph2 = sg.infer_geometry(make_record("CP(C)"))
    assert ph2["status"] == "candidate"
    p_h = [h for a, h in zip(make_record("CP(C)")["atoms"],
                             top(ph2)["hydrogen_counts"]) if a["element"] == "P"]
    assert p_h == [1]
    # degree-2 S-O: sulfenate vs oxide not separable by evidence -> flagged
    sulf = sg.infer_geometry(make_record("CSOC"))
    assert sulf["status"] == "candidate"
    assert "s_oxide_ambiguous" in sulf["reason_codes"]
    assert set(bond_pairs_orders(top(sulf)).values()) == {1.0}


def test_unknown_charge_states_nitro():
    out = sg.infer_geometry(make_record("CC[N+](=O)[O-]"))
    assert out["status"] == "candidate"
    assert "nitro_pattern" in out["reason_codes"]
    charges = top(out)["formal_charges"]
    assert 1 in charges  # N+ needed for valence; not forced neutral
    assert any("[O-]" in c["smiles"] for c in out["candidates"])
    assert any(c["evidence"]["total_formal_charge"] == 0 for c in out["candidates"])


# --------------------------------------------------------------------------
# determinism / permutation
# --------------------------------------------------------------------------

def test_atom_order_permutation_invariance():
    for smi in ("NCC(=O)O", "c1ccccc1", "C[C@H](N)C(=O)O"):
        base = sg.infer_geometry(make_record(smi))
        for seed in (11, 22, 33):
            perm = sg.infer_geometry(make_record(smi, shuffle=seed))
            assert perm["status"] == base["status"]
            assert top(perm)["smiles"] == top(base)["smiles"]
            assert len(perm["candidates"]) == len(base["candidates"])
            for a, b in zip(base["candidates"], perm["candidates"]):
                assert a["smiles"] == b["smiles"]
                assert abs(a["score"] - b["score"]) < 1e-6


def test_deterministic_repeat_calls():
    rec = make_record("NCC(=O)O")
    a = sg.infer_geometry(rec)
    b = sg.infer_geometry(rec)
    a["evidence"].pop("elapsed_ms")
    b["evidence"].pop("elapsed_ms")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------
# adjacency preservation, stereo, contract shape
# --------------------------------------------------------------------------

def test_supplied_adjacency_preserved():
    rec = make_record("NCC(=O)O")
    out = sg.infer_geometry(rec)
    pairs_out = set(bond_pairs_orders(top(out)))
    pairs_in = {tuple(sorted(b)) for b in rec["bonds"]}
    assert pairs_out == pairs_in  # no extra or missing heavy-heavy bonds


def test_stereo_from_coordinates():
    out = sg.infer_geometry(make_record("C[C@H](N)C(=O)O"))
    assert "@" in top(out)["smiles"]
    assert top(out)["evidence"]["stereo"] == "from_3d"


def test_contract_shape():
    out = sg.infer_geometry(make_record("NCC(=O)O"))
    assert out["algorithm"] == "simple_geometry_v2"
    assert isinstance(out["search_complete"], bool) and out["search_complete"]
    assert 1 <= len(out["candidates"]) <= 8
    scores = [c["score"] for c in out["candidates"]]
    assert scores == sorted(scores)  # rank 1 first; lower is better
    for c in out["candidates"]:
        assert isinstance(c["smiles"], str) and c["smiles"]
        assert set(c) >= {"smiles", "score", "bonds", "formal_charges",
                          "hydrogen_counts", "evidence"}
        assert all(b[2] in (1, 2, 3, 1.5) for b in c["bonds"])
        assert "score_terms" in c["evidence"]
    ev = out["evidence"]
    assert ev["geometry_usage"]["ring_plane_fits_and_relative_lengths_decide_aromaticity"]
    assert isinstance(ev["atom_geometry"], list) and ev["n_atoms"] == 5
    json.dumps(out)  # JSON ready


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------

def test_invalid_inputs():
    assert sg.infer_geometry({"atoms": [], "bonds": None})["status"] == "invalid_input"
    r = sg.infer_geometry({"atoms": [{"id": 0, "element": "Xx", "xyz": [0, 0, 0]}],
                           "bonds": None})
    assert r["status"] == "invalid_input" and "unknown_element:Xx" in r["reason_codes"]
    r = sg.infer_geometry({"atoms": [{"id": 0, "element": "C", "xyz": [0, 0, float("nan")]}],
                           "bonds": None})
    assert r["status"] == "invalid_input" and "nonfinite_coordinates" in r["reason_codes"]
    r = sg.infer_geometry({"atoms": [{"id": 0, "element": "C", "xyz": [0.0, 0, 0]},
                                     {"id": 1, "element": "C", "xyz": [1.5, 0, 0]}],
                           "bonds": [[0, 7]]})
    assert r["status"] == "invalid_input"
    assert sg.infer_geometry(["not", "a", "dict"])["status"] == "invalid_input"


def test_unresolved_impossible_valence():
    rec = {"atoms": [
        {"id": 0, "element": "C", "xyz": [0.0, 0.0, 0.0]},
        {"id": 1, "element": "C", "xyz": [1.2, 0.0, 0.0]},
        {"id": 2, "element": "O", "xyz": [0.6, 1.2, 0.0]},
        {"id": 3, "element": "C", "xyz": [0.6, -1.2, 0.0]},
        {"id": 4, "element": "C", "xyz": [1.8, 1.2, 0.0]},
        {"id": 5, "element": "C", "xyz": [-0.6, 1.2, 0.0]},
    ], "bonds": [[0, 2], [1, 2], [2, 3], [2, 4], [2, 5]], "total_charge": None}
    out = sg.infer_geometry(rec)  # oxygen with five bonds: no state exists
    assert out["status"] == "unresolved"
    assert out["candidates"] == []
    assert "no_sanitized_candidate" in out["reason_codes"]


def test_timeout_graceful():
    out = sg.infer_geometry(make_record("NCC(=O)O"), timeout_seconds=1e-9)
    assert out["status"] == "timeout"
    assert out["candidates"] == []
    assert out["search_complete"] is False
    assert "timeout_exceeded" in out["reason_codes"]
    json.dumps(out, allow_nan=False)  # no NaN/Infinity anywhere


def test_outputs_nan_free():
    for smi in ("NCC(=O)O", "c1ccccc1", "CS(=O)(=O)C", "c1nnn[nH]1"):
        json.dumps(sg.infer_geometry(make_record(smi)), allow_nan=False)


def test_malformed_geometry_never_raises():
    bad_records = [
        {},
        {"atoms": None},
        {"atoms": "abc"},
        {"atoms": [None]},
        {"atoms": [{"id": 0, "element": "C"}]},                      # xyz missing
        {"atoms": [{"id": 0, "element": "C", "xyz": None}]},
        {"atoms": [{"id": 0, "element": "C", "xyz": [1, 2]}]},
        {"atoms": [{"id": 0, "element": "C", "xyz": [1, 2, 3, 4]}]},
        {"atoms": [{"id": 0, "element": "C", "xyz": [1, "a", 3]}]},
        {"atoms": [{"id": 0, "element": None, "xyz": [0, 0, 0]}]},
        {"atoms": [{"id": 1, "element": "C", "xyz": [0, 0, 0]}]},    # non-positional id
        {"atoms": [{"id": 0, "element": "C", "xyz": [0.0, 0.0, 0.0]}],
         "bonds": "not-a-list"},
        {"atoms": [{"id": 0, "element": "C", "xyz": [0.0, 0.0, 0.0]},
                   {"id": 1, "element": "C", "xyz": [1.5, 0.0, 0.0]}],
         "bonds": [[0], [0, "1"], [0, None], [0, -1]]},
        {"atoms": [{"id": 0, "element": "C", "xyz": [0.0, 0.0, 0.0]},
                   {"id": 1, "element": "C", "xyz": [1.5, 0.0, 0.0]}],
         "bonds": [[0, 1]], "total_charge": 1.5},
    ]
    for rec in bad_records:
        out = sg.infer_geometry(rec)
        assert out["status"] == "invalid_input", rec
        assert out["candidates"] == []
        json.dumps(out, allow_nan=False)
    # integral-float total charge is accepted (common runner encoding)
    rec = make_record("CCN")
    rec["total_charge"] = 0.0
    assert sg.infer_geometry(rec)["status"] == "candidate"


def test_speed_bound():
    for smi in ("NCC(=O)O", "c1ccccc1", "CS(=O)(=O)C", "N[C@@H](Cc1cnc[nH]1)C(=O)O"):
        t0 = time.perf_counter()
        sg.infer_geometry(make_record(smi))
        assert (time.perf_counter() - t0) < 0.25, smi
