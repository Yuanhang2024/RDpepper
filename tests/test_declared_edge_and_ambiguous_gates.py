"""Focused tests for the two portfolio gate repairs.

1. declared_source_edges: CONECT serial pairs and 1555-symmetry LINK site
   pairs resolve to heavy serial edges; hydrogen/symmetric/foreign-chain
   records are ignored.
2. declared_edge_coverage: graph-bearing members are measured; groups with
   no graph cannot verify and stay uncovered; the missing-declared-edge
   rank bit outranks family priority.
3. geometry_candidate: an ``ambiguous`` simple-geometry result with scored
   candidates is admitted as an explicit hypothesis (reason codes retained),
   while ``unresolved``/``timeout`` remain fail-closed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from cycpep_master.bond_order_inference import (
    declared_edge_coverage,
    declared_source_edges,
)
from cycpep_master.core import geometry_candidate

_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  C   ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  O   ALA A   1       2.100   1.100   0.000  1.00  0.00           O
ATOM      4  N   GLY A   2       2.600  -1.000   0.000  1.00  0.00           N
ATOM      5  C   GLY A   2       4.000  -1.000   0.000  1.00  0.00           C
ATOM      6  H   GLY A   2       5.000  -1.000   0.000  1.00  0.00           H
CONECT    1    2
CONECT    2    1    3    4
CONECT    4    2    5    6
LINK         C   ALA A   1                 N   GLY A   2     1555   1555  1.34
LINK         C   ALA A   1                 N   XXX B   9     1555   1555  1.34
LINK         C   ALA A   1                 N   GLY A   2     1556   1555  1.34
END
"""


def _atoms_and_pdb(tmp_path: Path):
    source_atoms = [
        {"serial": 1, "name": "N", "residue": "ALA", "chain": "A",
         "residue_number": 1, "insertion_code": "", "element": "N",
         "xyz": [0.0, 0.0, 0.0]},
        {"serial": 2, "name": "C", "residue": "ALA", "chain": "A",
         "residue_number": 1, "insertion_code": "", "element": "C",
         "xyz": [1.458, 0.0, 0.0]},
        {"serial": 3, "name": "O", "residue": "ALA", "chain": "A",
         "residue_number": 1, "insertion_code": "", "element": "O",
         "xyz": [2.1, 1.1, 0.0]},
        {"serial": 4, "name": "N", "residue": "GLY", "chain": "A",
         "residue_number": 2, "insertion_code": "", "element": "N",
         "xyz": [2.6, -1.0, 0.0]},
        {"serial": 5, "name": "C", "residue": "GLY", "chain": "A",
         "residue_number": 2, "insertion_code": "", "element": "C",
         "xyz": [4.0, -1.0, 0.0]},
        {"serial": 6, "name": "H", "residue": "GLY", "chain": "A",
         "residue_number": 2, "insertion_code": "", "element": "H",
         "xyz": [5.0, -1.0, 0.0]},
    ]
    path = tmp_path / "mini.pdb"
    path.write_text(_PDB)
    return path, source_atoms


def test_declared_edges_from_conect_and_1555_link(tmp_path):
    path, atoms = _atoms_and_pdb(tmp_path)
    edges = declared_source_edges(path, atoms)
    # CONECT heavy pairs deduplicated + the 1555 LINK (same pair as CONECT 2-4)
    assert set(edges) == {(1, 2), (2, 3), (2, 4), (4, 5)}
    # foreign-chain LINK and 1556-symmetry LINK and hydrogen edges excluded


def test_coverage_requires_graph_bearing_member(tmp_path):
    _, atoms = _atoms_and_pdb(tmp_path)
    declared = [(1, 2), (2, 3), (2, 4), (4, 5)]
    full = {
        "engine": "geometry_simple_local",
        "candidate_graph": {
            "atoms": [{"serial": s, "element": a["element"]} for s, a in
                      ((1, atoms[0]), (2, atoms[1]), (3, atoms[2]),
                       (4, atoms[3]), (5, atoms[4]))],
            "bonds": [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 2, "b": 4},
                      {"a": 4, "b": 5}],
        },
    }
    missing = {
        "engine": "rdkit_pdb_proximity",
        "candidate_graph": {
            "atoms": full["candidate_graph"]["atoms"],
            "bonds": [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 2, "b": 4}],
        },
    }
    graphless = {"engine": "path_g_template", "family": "template"}
    assert declared_edge_coverage([full], declared)["covered"] is True
    partial = declared_edge_coverage([missing], declared)
    assert partial["covered"] is False and partial["covered_edge_count"] == 3
    none_graph = declared_edge_coverage([graphless], declared)
    assert none_graph["covered"] is False and none_graph["best_covering_engine"] is None
    assert declared_edge_coverage([graphless], [])["covered"] is True


def test_rank_bit_beats_family_priority(tmp_path):
    from cycpep_master.bond_order_inference import infer_bond_order_candidates
    path, _ = _atoms_and_pdb(tmp_path)
    report = infer_bond_order_candidates(str(path), "A")
    groups = report["identity_groups"]
    assert groups, "portfolio must produce at least one group for the mini chain"
    assert report["selected_candidate"]["declared_edge_coverage"]["covered"] is True


def test_ambiguous_geometry_is_admitted_but_unresolved_is_not(tmp_path, monkeypatch):
    path = tmp_path / "two.pdb"
    path.write_text(
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  O   LIG A   1       1.400   0.000   0.000  1.00  0.00           O\n"
        "CONECT    1    2\nEND\n"
    )
    atoms = [
        {"serial": 1, "name": "C", "residue": "LIG", "chain": "A",
         "residue_number": 1, "insertion_code": "", "element": "C",
         "xyz": [0.0, 0.0, 0.0]},
        {"serial": 2, "name": "O", "residue": "LIG", "chain": "A",
         "residue_number": 1, "insertion_code": "", "element": "O",
         "xyz": [1.4, 0.0, 0.0]},
    ]
    # unresolved / timeout statuses stay fail-closed
    for status in ("unresolved", "timeout"):
        monkeypatch.setattr(
            geometry_candidate,
            "infer_monomer_geometry",
            lambda record, *, mode, timeout_seconds, _s=status: {
                "status": _s, "candidates": [], "reason_codes": [_s]},
        )
        with pytest.raises(geometry_candidate.GeometryCandidateError):
            geometry_candidate.build_geometry_simple_molecule(str(path), atoms)

    # ambiguous WITH a scored candidate is admitted as an explicit hypothesis
    molecule, meta = None, None

    def fake_infer(record, *, mode, timeout_seconds):
        return {
            "status": "ambiguous",
            "algorithm": "geometry_simple_b2",
            "reason_codes": ["ring_aromatic_assigned_from_plane_angle_length"],
            "candidates": [
                {
                    "smiles": "CO",
                    "score": -1.0,
                    "bonds": [[0, 1, 1]],
                    "formal_charges": [0, 0],
                    "hydrogen_counts": [3, 1],
                    "evidence": {"elapsed_ms": 1},
                }
            ],
            "evidence": {"elapsed_ms": 1},
        }

    monkeypatch.setattr(geometry_candidate, "infer_monomer_geometry", fake_infer)
    molecule, meta = geometry_candidate.build_geometry_simple_molecule(
        str(path), atoms)
    assert molecule.GetNumAtoms() == 2
    assert meta["status"] == "ambiguous"
    assert meta["reason_codes"] == ["ring_aromatic_assigned_from_plane_angle_length"]
    assert meta["algorithm_status_note"]
