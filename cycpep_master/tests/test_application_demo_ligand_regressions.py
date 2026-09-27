"""Regressions for the application-demo ligand PDBQT defects.

Covers the two production defects fixed in
``cycpep_master.docking.mol2_pdbqt``:

1. ``_prepare_parent_molecule`` compared heavy-atom snapshots by
   absolute RDKit index, so a parent MOL2 interleaving explicit
   hydrogens among heavy atoms (the 1BCK layout) was rejected purely
   because ``RemoveHs``/``AddHs`` renumbered atoms.  The comparison is
   now keyed by heavy-atom rank and still rejects any real heavy-atom
   graph, charge, or coordinate change.
2. Meeko's default Gasteiger charges are non-finite for some parent
   chemistries (the 1BM2 five-coordinate P-H phosphorus), which the
   PDBQT writer refuses.  Preparation now falls back to OpenBabel's
   standard Gasteiger model read through Meeko's
   ``charge_model="read"``, with the actual method reported in the
   audit and a warning code, or fails closed with a concrete reason
   when no finite real-charge recovery exists.

"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

pytest.importorskip("meeko")

from cycpep_master.docking import mol2_pdbqt  # noqa: E402
from cycpep_master.docking.mol2_input import (  # noqa: E402
    load_validated_mol2,
    write_validation_receipt,
)
from cycpep_master.docking.mol2_pdbqt import (  # noqa: E402
    _prepare_parent_molecule,
    mol2_to_ligand_pdbqt,
)
from cycpep_master.export.conformer import mol_to_mol2  # noqa: E402


def _embed(smiles: str) -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert molecule is not None
    assert AllChem.EmbedMolecule(molecule, randomSeed=17) == 0
    return molecule


def _interleave_hydrogens(molecule: Chem.Mol) -> Chem.Mol:
    """Reorder so explicit hydrogens sit between heavy atoms (1BCK)."""
    order: list[int] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() > 1:
            order.append(atom.GetIdx())
            order.extend(
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() == 1
            )
    assert sorted(order) == list(range(molecule.GetNumAtoms()))
    return Chem.RenumberAtoms(molecule, order)


def _validated_mol2(
    tmp_path: Path,
    smiles: str,
    *,
    interleave: bool = False,
) -> Path:
    molecule = _embed(smiles)
    if interleave:
        molecule = _interleave_hydrogens(molecule)
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetProp("_TriposAtomName", f"{atom.GetSymbol()}{index}")
        atom.SetProp("_TriposResidueName", "LIG")
        atom.SetProp("_TriposChainId", "L")
        atom.SetIntProp("_TriposResidueNumber", 1)
        atom.SetProp("_TriposInsertionCode", "")
    output = tmp_path / "parent.mol2"
    produced, error = mol_to_mol2(molecule, output_path=str(output))
    assert error is None and Path(produced) == output
    write_validation_receipt(
        output,
        coordinate_mode="source_bound",
        rigor="L2:Q",
        quality="exact",
        source_heavy_atom_mapping_complete=True,
        atom_provenance_complete=True,
        source_input_sha256="a" * 64,
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
        max_source_coordinate_delta_angstrom=0.0,
        evidence_manifest_sha256="b" * 64,
    )
    return output


def _pdbqt_charges(path: Path) -> list[float]:
    return [
        float(line[70:76])
        for line in path.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]


def test_interleaved_explicit_hydrogens_pass_heavy_invariants(tmp_path):
    output = _validated_mol2(tmp_path, "CCO", interleave=True)
    validated = load_validated_mol2(output)
    heavy = [
        atom.GetIdx() for atom in validated.molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    assert heavy != list(range(len(heavy)))  # H really interleaved

    prepared, audit = _prepare_parent_molecule(validated)

    assert audit["heavy_atom_invariants_valid"] is True
    assert audit["full_inchikey_before"] == audit["full_inchikey_after"]
    assert prepared.GetNumHeavyAtoms() == len(heavy)


def test_interleaved_explicit_hydrogens_produce_pdbqt(tmp_path):
    output = _validated_mol2(tmp_path, "CCO", interleave=True)
    destination = tmp_path / "ligand.pdbqt"

    audit, error = mol2_to_ligand_pdbqt(
        output, destination, flexibility_mode="fast"
    )

    assert error is None
    assert destination.is_file()
    assert audit["heavy_atom_invariants_valid"] is True
    assert audit["full_inchikey_before"] == audit["full_inchikey_after"]
    assert (
        audit["maximum_parent_coordinate_component_delta_angstrom"]
        <= 0.0011
    )
    charges = _pdbqt_charges(destination)
    assert charges and all(math.isfinite(q) for q in charges)


@pytest.mark.parametrize(
    "damage",
    [
        "coordinate_shift",
        "element_swap",
        "bond_removal",
    ],
)
def test_true_heavy_change_is_rejected(tmp_path, monkeypatch, damage):
    output = _validated_mol2(tmp_path, "CCO")
    validated = load_validated_mol2(output)
    original_addhs = Chem.AddHs

    def damaging_addhs(molecule, **kwargs):
        completed = original_addhs(molecule, **kwargs)
        heavy_indices = [
            atom.GetIdx()
            for atom in completed.GetAtoms()
            if atom.GetAtomicNum() > 1
        ]
        if damage == "coordinate_shift":
            position = completed.GetConformer().GetAtomPosition(
                heavy_indices[1]
            )
            completed.GetConformer().SetAtomPosition(
                heavy_indices[1],
                Point3D(position.x + 0.5, position.y, position.z),
            )
        elif damage == "element_swap":
            completed.GetAtomWithIdx(
                heavy_indices[1]
            ).SetAtomicNum(7)
        else:
            bond = completed.GetBondBetweenAtoms(
                heavy_indices[0], heavy_indices[1]
            )
            editable = Chem.RWMol(completed)
            editable.RemoveBond(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            )
            completed = editable.GetMol()
        return completed

    monkeypatch.setattr(
        mol2_pdbqt.Chem, "AddHs", damaging_addhs
    )
    with pytest.raises(
        RuntimeError,
        match="rejected: hydrogen completion changed",
    ):
        _prepare_parent_molecule(validated)


def test_normal_charges_keep_meeko_gasteiger_model(tmp_path):
    output = _validated_mol2(tmp_path, "CCO")
    destination = tmp_path / "ligand.pdbqt"

    audit, error = mol2_to_ligand_pdbqt(
        output, destination, flexibility_mode="fast"
    )

    assert error is None
    assert audit["charge_model"] == "meeko_gasteiger"
    assert audit["charge_fallback_applied"] is False
    assert audit["charge_fallback"] is None
    assert mol2_pdbqt.PARTIAL_CHARGE_FALLBACK_WARNING_CODE not in (
        audit["flexibility_warning_codes"]
    )


def test_nonfinite_gasteiger_falls_back_to_openbabel(tmp_path):
    pytest.importorskip("openbabel")
    # Five-coordinate P-H phosphorus: the same pathology as the 1BM2
    # parent, where Meeko's Gasteiger iteration yields inf/nan.
    output = _validated_mol2(tmp_path, "O[PH](O)(O)O")
    destination = tmp_path / "ligand.pdbqt"

    audit, error = mol2_to_ligand_pdbqt(
        output, destination, flexibility_mode="fast"
    )

    assert error is None
    assert destination.is_file()
    assert audit["charge_model"] == "openbabel_gasteiger"
    assert audit["charge_fallback_applied"] is True
    fallback = audit["charge_fallback"]
    assert fallback["trigger"] == (
        "meeko_gasteiger_nonfinite_partial_charges"
    )
    assert fallback["nonfinite_setup_atom_count"] > 0
    assert "OBChargeModel 'gasteiger'" in fallback["fallback_method"]
    assert fallback["openbabel_release"]
    assert fallback["identity_guard"] == (
        "sdf_roundtrip_full_inchikey_equal"
    )
    assert fallback["graph_or_coordinates_modified"] is False
    assert math.isfinite(float(fallback["total_partial_charge"]))
    assert mol2_pdbqt.PARTIAL_CHARGE_FALLBACK_WARNING_CODE in (
        audit["flexibility_warning_codes"]
    )
    charges = _pdbqt_charges(destination)
    assert charges and all(math.isfinite(q) for q in charges)
    assert abs(sum(charges)) < 0.01


def test_charge_fallback_fails_closed_when_unavailable(
    tmp_path, monkeypatch
):
    output = _validated_mol2(tmp_path, "O[PH](O)(O)O")
    destination = tmp_path / "ligand.pdbqt"

    def unavailable(_molecule):
        return None, None, "OpenBabel unavailable: simulated outage"

    monkeypatch.setattr(
        mol2_pdbqt, "_openbabel_gasteiger_charges", unavailable
    )
    audit, error = mol2_to_ligand_pdbqt(
        output, destination, flexibility_mode="fast"
    )

    assert audit is None
    assert error is not None
    assert error.startswith("not_supported:")
    assert "non-finite" in error
    assert "OpenBabel unavailable: simulated outage" in error
    assert not destination.exists()
