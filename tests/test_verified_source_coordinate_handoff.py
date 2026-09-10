"""Verified source-coordinate handoff for graphless result-first candidates.

Contract under test (``cycpep_master.export.conformer``):

* A result-first candidate without a source-bound graph may keep its verified
  chemistry while receiving source heavy-atom coordinates, but only when the
  candidate graph is a FULL element/bond-order isomorphism onto the perceived
  source heavy topology, every alternative mapping is symmetry-equivalent
  under candidate-graph automorphisms preserving identity/attachment/stereo,
  and the mapped 3D coordinates realize the candidate stereochemistry.
* Any refusal (element mismatch, count mismatch, topology mismatch, stereo
  mismatch, ambiguous selection) must fall back to the ORIGINAL X1
  ``smiles_only_no_source_graph`` route, recording the concrete blocker.
* The legacy call signature (no source path) behaves exactly as before.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.export.conformer import (
    _result_first_candidate_to_mol2,
    _verified_source_handoff_molecule,
)

SFTI_INPUT = (
    Path(__file__).resolve().parents[2]
    / ".zcode_v710_application_demo"
    / "current_source_run_001"
    / "1sfi"
    / "peptide.pdb"
)

SFTI_CANDIDATE_SMILES = (
    "CC[C@H](C)[C@@H]1NC(=O)[C@@H]2CCCN2C(=O)[C@@H]2CCCN2C(=O)[C@H]([C@@H](C)CC)"
    "NC(=O)[C@H](CO)NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@@H]2CSSC"
    "[C@H](NC1=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(=O)O)C(=O)N"
    "CC(=O)N[C@@H](CCCNC(N)=[NH2+])C(=O)N2"
)

# Three-atom chain C-C-O at realistic bond lengths.  RDKit proximity
# perception yields two single bonds, so the ethanol candidate (all single
# bonds) can map, while element/topology/count variants cannot.
CHAIN_PDB = """\
HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG L   1       1.520   0.000   0.000  1.00  0.00           C
HETATM    3  O1  LIG L   1       3.040   0.000   0.000  1.00  0.00           O
CONECT    1    2
CONECT    2    1    3
CONECT    3    2
END
"""

PROPANE_PDB = """\
HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG L   1       1.520   0.000   0.000  1.00  0.00           C
HETATM    3  C3  LIG L   1       3.040   0.000   0.000  1.00  0.00           C
CONECT    1    2
CONECT    2    1    3
CONECT    3    2
END
"""

# Equilateral triangle: proximity perception yields a 3-edge ring, so the
# propane candidate (2 edges, 3 atoms) matches as a substructure on equal
# atom counts but the source carries one EXTRA edge.  The handoff must
# reject the induced-graph inequality instead of certifying it.
CYCLOPROPANE_PDB = """\
HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG L   1       1.510   0.000   0.000  1.00  0.00           C
HETATM    3  C3  LIG L   1       0.755   1.308   0.000  1.00  0.00           C
CONECT    1    2    3
CONECT    2    1    3
CONECT    3    1    2
END
"""


def _hetatm(serial: int, name: str, x: float, y: float, z: float, element: str) -> str:
    """Fixed-column HETATM record (chain L, residue LIG 1)."""
    return (
        f"HETATM{serial:5d} {name:<4s} LIG L   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _tetrahedral_edge() -> float:
    return 1.54 / math.sqrt(3.0)


def _neopentane_pdb() -> str:
    a = _tetrahedral_edge()
    directions = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    lines = [_hetatm(1, "C0", 0.0, 0.0, 0.0, "C")]
    for serial, (dx, dy, dz) in enumerate(directions, 2):
        lines.append(
            _hetatm(serial, f"C{serial - 1}", a * dx, a * dy, a * dz, "C")
        )
    for serial in range(2, 6):
        lines.append(f"CONECT    1{serial:5d}")
        lines.append(f"CONECT{serial:5d}    1")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _embedded_heavy_pdb(smiles: str, seed: int = 7) -> str:
    """Write a heavy-atom PDB (chain L) realizing ``smiles`` in 3D."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    index_of = {atom.GetIdx(): position for position, atom in enumerate(heavy, 1)}
    lines = []
    conformer = mol.GetConformer()
    for position, atom in enumerate(heavy, 1):
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(
            _hetatm(
                position,
                f"{atom.GetSymbol()}{position}",
                point.x,
                point.y,
                point.z,
                atom.GetSymbol(),
            )
        )
    for position, atom in enumerate(heavy, 1):
        partners = [
            index_of[neighbor.GetIdx()]
            for neighbor in atom.GetNeighbors()
            if neighbor.GetIdx() in index_of
        ]
        if partners:
            row = f"CONECT{position:5d}" + "".join(f"{p:5d}" for p in partners[:4])
            lines.append(row)
    lines.append("END")
    return "\n".join(lines) + "\n"


def _graphless_result(smiles: str, *, ambiguous: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_smiles=None,
        candidate_graph=None,
        smiles=smiles,
        ambiguous=ambiguous,
    )


def _header_fields(block_or_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block_or_text.splitlines():
        if not line.startswith("#"):
            break
        for token in line.lstrip("#").split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields.setdefault(key, value)
    return fields


def _source_positions(path: Path) -> list[tuple[float, float, float]]:
    positions = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            positions.append(
                (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return positions


def _max_delta_to_source(mol: Chem.Mol, path: Path) -> float:
    conformer = mol.GetConformer()
    got = [
        tuple(conformer.GetAtomPosition(atom.GetIdx()))
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    source = _source_positions(path)
    assert len(got) == len(source)
    remaining = list(source)
    worst = 0.0
    for g in got:
        distances = [math.dist(g, s) for s in remaining]
        nearest = distances.index(min(distances))
        worst = max(worst, distances[nearest])
        remaining.pop(nearest)
    return worst


def test_happy_path_transfers_source_coordinates(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)

    molecule, detail = _verified_source_handoff_molecule("CCO", path, "L")

    assert molecule is not None
    assert detail["isomorphism_count"] == 1
    assert detail["symmetry_equivalent_alternatives"] is False
    assert detail["stereo_realization_verified"] is True
    assert detail["source_heavy_atoms"] == 3
    assert _max_delta_to_source(molecule, path) <= 1e-3
    # Candidate formal state is untouched: neutral ethanol.
    assert Chem.GetFormalCharge(molecule) == 0


def test_element_mismatch_refused(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)

    molecule, detail = _verified_source_handoff_molecule("CCS", path, "L")

    assert molecule is None
    assert "no element/bond-order isomorphism" in detail


def test_heavy_atom_count_mismatch_refused(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)

    molecule, detail = _verified_source_handoff_molecule("CCCO", path, "L")

    assert molecule is None
    assert "heavy-atom count mismatch" in detail


def test_topology_mismatch_refused(tmp_path):
    # Same element multiset and equal atom/bond counts as the source chain,
    # but a different connectivity: no isomorphism exists.
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)

    molecule, detail = _verified_source_handoff_molecule("COC", path, "L")

    assert molecule is None
    assert "no element/bond-order isomorphism" in detail


def test_source_extra_edge_refused(tmp_path):
    # Equal atom counts and a valid substructure match are NOT sufficient:
    # the perceived source ring has one extra edge the candidate lacks, so
    # the source is not the candidate's induced graph and the handoff must
    # be refused rather than certified.
    path = tmp_path / "cyclopropane.pdb"
    path.write_text(CYCLOPROPANE_PDB)

    molecule, detail = _verified_source_handoff_molecule("CCC", path, "L")

    assert molecule is None
    assert "bond count mismatch" in detail


def test_symmetry_equivalent_alternatives_accepted(tmp_path):
    # Neopentane: the four methyl carbons form one symmetry class, so 24
    # isomorphisms exist and every one is identity/attachment/stereo
    # preserving.  The handoff must accept and pick deterministically.
    path = tmp_path / "neopentane.pdb"
    path.write_text(_neopentane_pdb())

    molecule, detail = _verified_source_handoff_molecule("C(C)(C)(C)C", path, "L")

    assert molecule is not None
    assert detail["isomorphism_count"] == 24
    assert detail["symmetry_equivalent_alternatives"] is True
    assert _max_delta_to_source(molecule, path) <= 1e-3


def test_stereo_realization_gate(tmp_path):
    source_block = _embedded_heavy_pdb("CC[C@H](C)O")
    path = tmp_path / "butanol.pdb"
    path.write_text(source_block)

    matching, _ = _verified_source_handoff_molecule("CC[C@H](C)O", path, "L")
    flipped, detail = _verified_source_handoff_molecule(
        "CC[C@@H](C)O", path, "L"
    )

    assert matching is not None
    assert _max_delta_to_source(matching, path) <= 1e-3
    assert flipped is None
    assert "do not realize the candidate stereochemistry" in detail


def test_x3_written_with_full_mapping_receipt(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)
    output = tmp_path / "out.mol2"

    produced, error = _result_first_candidate_to_mol2(
        _graphless_result("CCO"),
        str(output),
        source_pdb_path=path,
        source_chain_id="L",
    )

    assert error is None and produced is not None
    text = Path(produced).read_text()
    fields = _header_fields(text)
    assert fields["coordinate_tier"] == "X3"
    assert fields["coordinate_source"] == "result_first_verified_source_handoff"
    assert fields["source_mapped_atoms"] == "3"
    assert fields["generated_heavy_atoms"] == "0"
    assert fields["stereo_realization_verified"] == "true"
    assert "fallback_origin" not in fields

    from cycpep_master.export.conformer import _mol2_roundtrip_full_inchikey

    roundtrip_key, roundtrip_error = _mol2_roundtrip_full_inchikey(text)
    assert roundtrip_error is None
    assert roundtrip_key == Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))


def test_mismatch_falls_back_to_original_x1_with_blocker(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)

    block, error = _result_first_candidate_to_mol2(
        _graphless_result("CCS"),
        None,
        source_pdb_path=path,
        source_chain_id="L",
    )

    assert error is None and block is not None
    fields = _header_fields(block)
    assert fields["coordinate_tier"] == "X1"
    assert fields["fallback_origin"] == "smiles_only_no_source_graph"
    assert fields["source_handoff_blocked"] == "true"
    assert "no_element/bond-order_isomorphism" in fields[
        "source_handoff_block_reason"
    ]


def test_ambiguous_selection_never_forces_handoff(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text(CHAIN_PDB)

    block, error = _result_first_candidate_to_mol2(
        _graphless_result("CCO", ambiguous=True),
        None,
        source_pdb_path=path,
        source_chain_id="L",
    )

    assert error is None and block is not None
    fields = _header_fields(block)
    assert fields["coordinate_tier"] == "X1"
    assert fields["fallback_origin"] == "smiles_only_no_source_graph"
    assert "candidate_selection_is_ambiguous" in fields[
        "source_handoff_block_reason"
    ]


def test_legacy_call_signature_unchanged():
    block, error = _result_first_candidate_to_mol2(_graphless_result("CCO"))

    assert error is None and block is not None
    fields = _header_fields(block)
    assert fields["coordinate_tier"] == "X1"
    assert fields["fallback_origin"] == "smiles_only_no_source_graph"
    assert "source_handoff_blocked" not in fields


@pytest.mark.skipif(
    not SFTI_INPUT.exists(), reason="1sfi development input not present"
)
def test_real_sfti_case_maps_all_105_heavy_atoms(tmp_path):
    molecule, detail = _verified_source_handoff_molecule(
        SFTI_CANDIDATE_SMILES, SFTI_INPUT, "I"
    )

    assert molecule is not None
    assert detail["source_heavy_atoms"] == 105
    assert detail["candidate_heavy_atoms"] == 105
    # The two isomorphisms differ only by the Phe12 ring mirror automorphism.
    assert detail["isomorphism_count"] == 2
    assert detail["symmetry_equivalent_alternatives"] is True
    assert _max_delta_to_source(molecule, SFTI_INPUT) <= 1e-3
    assert (
        Chem.MolToInchiKey(Chem.RemoveHs(Chem.Mol(molecule)))
        == Chem.MolToInchiKey(Chem.MolFromSmiles(SFTI_CANDIDATE_SMILES))
    )

    block, error = _result_first_candidate_to_mol2(
        _graphless_result(SFTI_CANDIDATE_SMILES),
        None,
        source_pdb_path=SFTI_INPUT,
        source_chain_id="I",
    )
    assert error is None and block is not None
    fields = _header_fields(block)
    assert fields["coordinate_tier"] == "X3"
    assert fields["source_mapped_atoms"] == "105"
    assert fields["generated_heavy_atoms"] == "0"
