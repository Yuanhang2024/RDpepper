"""Minimal unit tests for complex_geometry_v2 (synthetic dev molecules only).

Covers the task-card chemistry set (carbonyl repair, aromatic vs saturated,
noisy geometry, charge ambiguity, P/S explicit valence, amide vs amine,
permutation, invalid input, bonds=null) plus the runner-required checks:
bounded behaviour under a 1.0 s per-record budget, JSON-finite output, and
exact preservation of supplied adjacency.  No external benchmark data or
truth dictionaries are read.
"""

import json
import math
import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core import geometry_complex as cg



# ---------------------------------------------------------------- helpers
def record_from_smiles(smi, seed=0xF00D):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0, smi
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    atoms = [{"id": a.GetIdx(), "element": a.GetSymbol(),
              "xyz": [float(v) for v in conf.GetAtomPosition(a.GetIdx())]}
             for a in mol.GetAtoms()]
    bonds = sorted({(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                     max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                    for b in mol.GetBonds()})
    return {"atoms": atoms, "bonds": [[i, j] for i, j in bonds],
            "total_charge": None}


def benzene_record(noise=0.0, seed=7):
    rng = np.random.RandomState(seed)
    atoms = []
    for k in range(6):
        xyz = [1.39 * math.cos(math.radians(60 * k)),
               1.39 * math.sin(math.radians(60 * k)), 0.0]
        if noise:
            xyz = list(np.array(xyz) + rng.normal(0, noise, 3))
        atoms.append({"id": k, "element": "C", "xyz": xyz})
    return {"atoms": atoms, "bonds": [[k, (k + 1) % 6] for k in range(6)],
            "total_charge": None}


def acetate_symmetric():
    return {"atoms": [
        {"id": 0, "element": "C", "xyz": [0.0, 0.0, 0.0]},
        {"id": 1, "element": "C", "xyz": [-1.52, 0.0, 0.0]},
        {"id": 2, "element": "O", "xyz": [0.70, 1.05, 0.05]},
        {"id": 3, "element": "O", "xyz": [0.70, -1.05, -0.05]},
    ], "bonds": [[0, 1], [0, 2], [0, 3]], "total_charge": None}


def canon(smi):
    return Chem.CanonSmiles(smi)


def top(out):
    assert out["candidates"], out
    return out["candidates"][0]


def heavy_atom_count(smi):
    return Chem.MolFromSmiles(smi).GetNumAtoms()


def all_finite_json(obj):
    text = json.dumps(obj, allow_nan=False)  # raises on inf/nan
    def walk(node):
        if isinstance(node, float):
            assert math.isfinite(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(obj)
    return text


def adjacency_intact(out, record):
    supplied = {(min(i, j), max(i, j)) for i, j in record["bonds"]}
    for cand in out["candidates"]:
        got = {(min(b[0], b[1]), max(b[0], b[1])) for b in cand["bonds"]}
        assert got == supplied
        assert all(b[2] in (1, 2, 3, 1.5) for b in cand["bonds"])


# ---------------------------------------------------------------- carbonyl
def test_carbonyl_double_not_single_plus_h():
    rec = record_from_smiles("CC=O")
    out = cg.infer_geometry(rec, timeout_seconds=1.0)
    cand = top(out)
    assert out["status"] in ("candidate", "ambiguous")
    assert canon(cand["smiles"]) == canon("CC=O")
    o_idx = next(a["id"] for a in rec["atoms"] if a["element"] == "O")
    # repair target: terminal O at 1.23 A must be C=O with zero H
    assert cand["hydrogen_counts"][o_idx] == 0
    co = [b for b in cand["bonds"] if o_idx in (b[0], b[1])][0]
    assert co[2] == 2


def test_formamide_and_amide_vs_amine():
    out = cg.infer_geometry(record_from_smiles("CC(N)=O"), timeout_seconds=1.0)
    assert canon(top(out)["smiles"]) == canon("CC(N)=O")
    rec = record_from_smiles("CCN")
    out = cg.infer_geometry(rec, timeout_seconds=1.0)
    cand = top(out)
    assert canon(cand["smiles"]) == canon("CCN")
    n_idx = next(a["id"] for a in rec["atoms"] if a["element"] == "N")
    assert cand["hydrogen_counts"][n_idx] == 2


# ------------------------------------------------------- aromatic vs saturated
def test_benzene_aromatic_not_saturated():
    out = cg.infer_geometry(benzene_record(), timeout_seconds=1.0)
    cand = top(out)
    assert out["status"] in ("candidate", "ambiguous")
    assert canon(cand["smiles"]) == canon("c1ccccc1")
    assert all(h == 1 for h in cand["hydrogen_counts"])
    assert all(b[2] == 1.5 for b in cand["bonds"])


def test_cyclohexane_saturated_not_aromatic():
    out = cg.infer_geometry(record_from_smiles("C1CCCCC1"), timeout_seconds=1.0)
    cand = top(out)
    assert canon(cand["smiles"]) == canon("C1CCCCC1")
    assert all(h == 2 for h in cand["hydrogen_counts"])
    assert all(b[2] == 1 for b in cand["bonds"])


def test_noisy_benzene_still_aromatic():
    out = cg.infer_geometry(benzene_record(noise=0.04), timeout_seconds=1.0)
    cand = top(out)
    assert canon(cand["smiles"]) == canon("c1ccccc1")
    assert all(b[2] == 1.5 for b in cand["bonds"])


def test_toluene():
    out = cg.infer_geometry(record_from_smiles("Cc1ccccc1"), timeout_seconds=1.0)
    assert canon(top(out)["smiles"]) == canon("Cc1ccccc1")


# ------------------------------------------------------------- charge ambiguity
def test_acetate_microstate_ambiguity_explored():
    out = cg.infer_geometry(acetate_symmetric(), timeout_seconds=1.0)
    smiles = {canon(c["smiles"]) for c in out["candidates"]}
    assert canon("CC(=O)O") in smiles and canon("CC(=O)[O-]") in smiles
    ev = out["evidence"]["charge_search"]
    assert ev["net_charge_states_explored"] == [-1, 0, 1]


def test_known_total_charge_constrains():
    rec = record_from_smiles("CC(N)=O")
    rec["total_charge"] = 0
    out = cg.infer_geometry(rec, timeout_seconds=1.0)
    assert all(c["evidence"]["charge_total"] == 0
               for c in out["candidates"])


# --------------------------------------------------------------------- P / S
def test_phosphate_valence_five_p_zero_h():
    out = cg.infer_geometry(record_from_smiles("O=P(O)(O)O"), timeout_seconds=1.0)
    cand = top(out)
    rec = record_from_smiles("O=P(O)(O)O")
    p_idx = [a["id"] for a in rec["atoms"] if a["element"] == "P"][0]
    assert cand["hydrogen_counts"][p_idx] == 0
    order_sum = sum(b[2] for b in cand["bonds"] if p_idx in (b[0], b[1]))
    assert order_sum == 5
    assert p_idx in cand["evidence"]["hypervalent_atoms"]


def test_sulfone_valence_six_tetrahedral():
    out = cg.infer_geometry(record_from_smiles("CS(C)(=O)=O"), timeout_seconds=1.0)
    cand = top(out)
    assert canon(cand["smiles"]) == canon("CS(C)(=O)=O")
    rec = record_from_smiles("CS(C)(=O)=O")
    s_idx = [a["id"] for a in rec["atoms"] if a["element"] == "S"][0]
    order_sum = sum(b[2] for b in cand["bonds"] if s_idx in (b[0], b[1]))
    assert order_sum == 6
    assert cand["formal_charges"][s_idx] == 0
    assert cand["evidence"]["hybridization"][s_idx] == "hypervalent_tetrahedral"


def test_phosphine_hydrogen_not_banned():
    out = cg.infer_geometry(
        {"atoms": [{"id": 0, "element": "P", "xyz": [0.0, 0.0, 0.0]}],
         "bonds": [], "total_charge": None}, timeout_seconds=1.0)
    cand = top(out)
    assert cand["hydrogen_counts"][0] == 3  # PH3 retained: no blanket P-H ban
    assert out["status"] in ("candidate", "ambiguous")


def test_thiolate_style_s_minus_available():
    out = cg.infer_geometry(
        {"atoms": [{"id": 0, "element": "S", "xyz": [0.0, 0.0, 0.0]}],
         "bonds": [], "total_charge": None}, timeout_seconds=1.0)
    smiles = [canon(c["smiles"]) for c in out["candidates"]]
    assert canon("S") in smiles  # SH2 among plausible states


# ---------------------------------------------------------------- permutation
def _permute(record, perm):
    # perm[new_id] = old_id
    inv = {old: new for new, old in enumerate(perm)}
    atoms = []
    for new_id, old_id in enumerate(perm):
        src = record["atoms"][old_id]
        atoms.append({"id": new_id, "element": src["element"],
                      "xyz": list(src["xyz"])})
    bonds = [[inv[i], inv[j]] for i, j in record["bonds"]]
    return {"atoms": atoms, "bonds": bonds,
            "total_charge": record.get("total_charge")}


@pytest.mark.parametrize("perm", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_permutation_invariance_acetaldehyde(perm):
    rec = record_from_smiles("CC=O")
    out = cg.infer_geometry(rec, timeout_seconds=1.0)
    out_p = cg.infer_geometry(_permute(rec, perm), timeout_seconds=1.0)
    assert canon(top(out)["smiles"]) == canon(top(out_p)["smiles"]) \
        == canon("CC=O")


@pytest.mark.parametrize("perm", [[0, 1, 2, 3, 4, 5], [5, 3, 1, 4, 2, 0]])
def test_permutation_invariance_benzene(perm):
    rec = benzene_record()
    out = cg.infer_geometry(rec, timeout_seconds=1.0)
    out_p = cg.infer_geometry(_permute(rec, perm), timeout_seconds=1.0)
    assert canon(top(out)["smiles"]) == canon(top(out_p)["smiles"]) \
        == canon("c1ccccc1")


# ------------------------------------------------------------- connectivity
def test_bonds_null_connectivity_inferred():
    rec = record_from_smiles("CC=O")
    rec["bonds"] = None
    out = cg.infer_geometry(rec, timeout_seconds=1.0)
    assert "connectivity_inferred" in out["reason_codes"]
    assert canon(top(out)["smiles"]) == canon("CC=O")


# ---------------------------------------------------------- schema & runner
def _record_set():
    return [
        record_from_smiles("CC=O"),
        record_from_smiles("CC(N)=O"),
        record_from_smiles("C1CCCCC1"),
        record_from_smiles("O=P(O)(O)O"),
        record_from_smiles("CS(C)(=O)=O"),
        benzene_record(),
        acetate_symmetric(),
    ]


def test_schema_scores_sorted_and_atoms_preserved():
    for rec in _record_set():
        out = cg.infer_geometry(rec, timeout_seconds=1.0)
        assert out["algorithm"] == "complex_geometry_v2"
        assert out["status"] in ("candidate", "ambiguous")
        assert 1 <= len(out["candidates"]) <= 8
        scores = [c["score"] for c in out["candidates"]]
        assert scores == sorted(scores)
        n = len(rec["atoms"])
        for c in out["candidates"]:
            assert len(c["formal_charges"]) == n
            assert len(c["hydrogen_counts"]) == n
            assert heavy_atom_count(c["smiles"]) == n
            assert isinstance(c["evidence"], dict)
        adjacency_intact(out, rec)
        all_finite_json(out)


def test_supplied_adjacency_intact_all_candidates():
    for rec in _record_set():
        out = cg.infer_geometry(rec, timeout_seconds=1.0)
        adjacency_intact(out, rec)


def test_one_second_budget_respected():
    for rec in _record_set():
        t0 = time.perf_counter()
        out = cg.infer_geometry(rec, timeout_seconds=1.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"{elapsed:.3f}s over budget"
        if not out["search_complete"]:
            assert any(code in out["reason_codes"] for code in (
                "beam_truncated", "charge_enumeration_truncated", "mol_build_budget_exhausted",
                "candidate_output_truncated", "deadline_exceeded",
            ))
        assert "deadline_exceeded" not in out["reason_codes"]


def test_tiny_timeout_truncates_honestly():
    out = cg.infer_geometry(benzene_record(), timeout_seconds=1e-9)
    assert out["search_complete"] is False
    assert out["status"] in ("timeout", "candidate", "ambiguous", "unresolved")
    assert "deadline_exceeded" in out["reason_codes"]
    all_finite_json(out)


def _strip_timings(obj):
    if isinstance(obj, dict):
        return {k: _strip_timings(v) for k, v in obj.items()
                if k != "elapsed_ms"}
    if isinstance(obj, list):
        return [_strip_timings(v) for v in obj]
    return obj


def test_determinism_same_input_same_output():
    rec = benzene_record()
    a = cg.infer_geometry(rec, timeout_seconds=1.0)
    b = cg.infer_geometry(rec, timeout_seconds=1.0)
    assert all_finite_json(_strip_timings(a)) == all_finite_json(_strip_timings(b))


def test_input_record_not_mutated():
    rec = record_from_smiles("CC=O")
    snapshot = json.dumps(rec)
    cg.infer_geometry(rec, timeout_seconds=1.0)
    assert json.dumps(rec) == snapshot


# ------------------------------------------------------------- invalid input
@pytest.mark.parametrize("bad,why", [
    ({"atoms": [], "bonds": None, "total_charge": None}, "empty"),
    ({"atoms": [{"id": 0, "element": "C", "xyz": [0, 0, 0]},
                {"id": 2, "element": "O", "xyz": [1.2, 0, 0]}],
      "bonds": [[0, 2]], "total_charge": None}, "noncontiguous"),
    ({"atoms": [{"id": 0, "element": "Xx", "xyz": [0, 0, 0]}],
      "bonds": None, "total_charge": None}, "unknown element"),
    ({"atoms": [{"id": 0, "element": "H", "xyz": [0, 0, 0]}],
      "bonds": None, "total_charge": None}, "explicit H"),
    ({"atoms": [{"id": 0, "element": "C", "xyz": [0, float("nan"), 0]}],
      "bonds": None, "total_charge": None}, "nan coord"),
    ({"atoms": [{"id": 0, "element": "C", "xyz": [0, 0, 0]}],
      "bonds": [[0, 0]], "total_charge": None}, "self loop"),
    ({"atoms": [{"id": 0, "element": "C", "xyz": [0, 0, 0]}],
      "bonds": [[0, 1]], "total_charge": None}, "endpoint range"),
    ({"atoms": [{"id": 0, "element": "C", "xyz": [0, 0, 0]}],
      "bonds": None, "total_charge": 0.5}, "charge float"),
    ("not a dict", "not dict"),
])
def test_invalid_inputs(bad, why):
    out = cg.infer_geometry(bad, timeout_seconds=1.0)
    assert out["status"] == "invalid_input", why
    assert out["candidates"] == []
    assert out["search_complete"] is False
    assert out["reason_codes"]
    all_finite_json(out)


@pytest.mark.parametrize("bad_t", [0, -1.0, float("nan"), float("inf"), "1"])
def test_invalid_timeout(bad_t):
    out = cg.infer_geometry(benzene_record(), timeout_seconds=bad_t)
    assert out["status"] == "invalid_input"
    assert "invalid_timeout_seconds" in out["reason_codes"]
