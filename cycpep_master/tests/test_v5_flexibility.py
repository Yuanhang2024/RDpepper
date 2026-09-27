from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core.artifacts import (
    ArtifactStatus,
    ArtifactType,
    ChemicalLevel,
    ConformerEnsembleArtifact,
    ConformerMember,
    CoordinateLevel,
    CoordinateOrigin,
    ENSEMBLE_SCHEMA_VERSION,
    EvidenceBasis,
    EvidenceProfile,
    artifact_payload_sha256,
    make_artifact_id,
)
from cycpep_master.docking.flexibility import (
    load_validated_flexibility_ensemble,
)
from cycpep_master.docking.mol2_input import (
    load_validated_mol2,
    sha256_path,
    write_validation_receipt,
)
from cycpep_master.docking.mol2_pdbqt import (
    mol2_to_ligand_pdbqt,
)
from cycpep_master.export.conformer import mol_to_mol2


def _write_member(
    root: Path,
    name: str,
    *,
    smiles: str = "CCCCCC",
    seed: int = 17,
    reverse_atom_order: bool = False,
) -> tuple[Path, Path]:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
    if reverse_atom_order:
        molecule = Chem.RenumberAtoms(
            molecule, list(reversed(range(molecule.GetNumAtoms())))
        )
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetProp("_TriposAtomName", f"{atom.GetSymbol()}{index}")
        atom.SetProp("_TriposResidueName", "LIG")
        atom.SetProp("_TriposChainId", "L")
        atom.SetIntProp("_TriposResidueNumber", 1)
        atom.SetProp("_TriposInsertionCode", "")
    path = root / f"{name}.mol2"
    produced, error = mol_to_mol2(molecule, output_path=str(path))
    assert error is None and produced == str(path)
    receipt = write_validation_receipt(
        path,
        coordinate_mode="regenerated",
        rigor="C3:S",
        quality="specified",
        source_heavy_atom_mapping_complete=False,
        atom_provenance_complete=True,
        source_input_sha256="a" * 64,
        topology_class="linear",
        macrocycle_ring_size=None,
        max_source_coordinate_delta_angstrom=None,
        evidence_manifest_sha256="b" * 64,
    )
    return path, receipt


def _member_row(path: Path, receipt: Path, conformer_id: str) -> dict:
    return ConformerMember(
        conformer_id=conformer_id,
        coordinate_origin=CoordinateOrigin.GENERATED,
        strategy="test_fixture",
        mol2_path=str(path),
        mol2_sha256=sha256_path(path),
        receipt_path=str(receipt),
        receipt_sha256=sha256_path(receipt),
        energy=None,
        qa={"passed": True},
        status="validated",
    ).to_dict()


def _manifest(
    root: Path,
    *,
    smiles: str = "CCCCCC",
    second_smiles: str | None = None,
    reverse_second: bool = False,
) -> tuple[Path, Path, str]:
    parent, parent_receipt = _write_member(
        root, "conf_001", smiles=smiles, seed=17
    )
    second, second_receipt = _write_member(
        root,
        "conf_002",
        smiles=second_smiles or smiles,
        seed=29,
        reverse_atom_order=reverse_second,
    )
    members = [
        _member_row(parent, parent_receipt, "conf_001"),
        _member_row(second, second_receipt, "conf_002"),
    ]
    payload = {
        "parent_graph_artifact_id": "c" * 64,
        "parent_graph_sha256": "d" * 64,
        "requested_count": 2,
        "members": members,
        "attempts": [],
        "template_evidence": [],
        "torsion_prior_error": None,
        "random_seed": 42,
        "num_threads": 1,
        "rdkit_version": "test",
    }
    artifact_id = make_artifact_id(
        ArtifactType.CONFORMER_ENSEMBLE,
        ("c" * 64,),
        payload,
    )
    artifact = ConformerEnsembleArtifact(
        artifact_type=ArtifactType.CONFORMER_ENSEMBLE,
        artifact_id=artifact_id,
        parent_artifact_ids=("c" * 64,),
        status=ArtifactStatus.MATERIALIZED,
        payload_sha256=artifact_payload_sha256(payload),
        evidence=EvidenceProfile(
            chemical_level=ChemicalLevel.C3,
            chemical_basis=EvidenceBasis.SPECIFIED,
            coordinate_origin=CoordinateOrigin.GENERATED,
            coordinate_level=CoordinateLevel.X1,
        ),
        requested_count=2,
        produced_count=2,
        members=tuple(
            ConformerMember(
                conformer_id=row["conformer_id"],
                coordinate_origin=CoordinateOrigin(
                    row["coordinate_origin"]
                ),
                strategy=row["strategy"],
                mol2_path=row["mol2_path"],
                mol2_sha256=row["mol2_sha256"],
                receipt_path=row["receipt_path"],
                receipt_sha256=row["receipt_sha256"],
                energy=row["energy"],
                qa=row["qa"],
                status=row["status"],
            )
            for row in members
        ),
    )
    document = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "artifact": artifact.to_dict(),
        "payload": payload,
    }
    manifest = root / "ensemble_manifest.json"
    manifest.write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )
    return parent, manifest, sha256_path(manifest)


def _reseal(document: dict) -> None:
    document["payload"]["members"] = copy.deepcopy(
        document["artifact"]["members"]
    )
    payload = document["payload"]
    document["artifact"]["payload_sha256"] = artifact_payload_sha256(
        payload
    )
    document["artifact"]["artifact_id"] = make_artifact_id(
        ArtifactType.CONFORMER_ENSEMBLE,
        tuple(document["artifact"]["parent_artifact_ids"]),
        payload,
    )


def test_validated_ensemble_manifest_is_consumed_without_embedding(
    tmp_path,
):
    parent_path, manifest, digest = _manifest(tmp_path)
    parent = load_validated_mol2(parent_path)

    ensemble, audit = load_validated_flexibility_ensemble(
        parent,
        manifest,
        expected_manifest_sha256=digest,
    )

    assert ensemble is not None
    assert ensemble.GetNumConformers() == 2
    assert audit["status"] == "loaded"
    assert audit["embedding_performed"] is False
    assert audit["manifest_sha256"] == digest


def test_expected_manifest_hash_drift_is_rejected(tmp_path):
    parent_path, manifest, _digest = _manifest(tmp_path)
    parent = load_validated_mol2(parent_path)

    ensemble, audit = load_validated_flexibility_ensemble(
        parent,
        manifest,
        expected_manifest_sha256="0" * 64,
    )

    assert ensemble is None
    assert audit["status"] == "invalid"
    assert "SHA-256 drift" in audit["reason"]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("mol2_sha256", "MOL2 hash drift"),
        ("receipt_sha256", "receipt hash drift"),
    ],
)
def test_member_hash_drift_is_rejected(tmp_path, field, reason):
    parent_path, manifest, _digest = _manifest(tmp_path)
    parent = load_validated_mol2(parent_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["artifact"]["members"][1][field] = "0" * 64
    _reseal(document)
    manifest.write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )

    ensemble, audit = load_validated_flexibility_ensemble(
        parent, manifest
    )

    assert ensemble is None
    assert reason in audit["reason"]


def test_duplicate_conformer_ids_are_rejected(tmp_path):
    parent_path, manifest, _digest = _manifest(tmp_path)
    parent = load_validated_mol2(parent_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["artifact"]["members"][1]["conformer_id"] = "conf_001"
    _reseal(document)
    manifest.write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )

    ensemble, audit = load_validated_flexibility_ensemble(
        parent, manifest
    )

    assert ensemble is None
    assert "duplicated" in audit["reason"]


def test_atom_order_drift_is_rejected(tmp_path):
    parent_path, manifest, _digest = _manifest(
        tmp_path, smiles="CCCOCN", reverse_second=True
    )
    parent = load_validated_mol2(parent_path)

    ensemble, audit = load_validated_flexibility_ensemble(
        parent, manifest
    )

    assert ensemble is None
    assert "atom order or graph" in audit["reason"]


def test_identity_drift_is_rejected(tmp_path):
    parent_path, manifest, _digest = _manifest(
        tmp_path, second_smiles="CCCCC"
    )
    parent = load_validated_mol2(parent_path)

    ensemble, audit = load_validated_flexibility_ensemble(
        parent, manifest
    )

    assert ensemble is None
    assert "identity differs" in audit["reason"]


def test_manifest_must_bind_the_selected_parent(tmp_path):
    _parent_path, manifest, _digest = _manifest(tmp_path)
    alternate, _receipt = _write_member(
        tmp_path, "alternate", seed=41
    )
    parent = load_validated_mol2(alternate)

    ensemble, audit = load_validated_flexibility_ensemble(
        parent, manifest
    )

    assert ensemble is None
    assert "does not bind the parent" in audit["reason"]


def test_pdbqt_balanced_mode_uses_existing_manifest_only(
    tmp_path, monkeypatch
):
    pytest.importorskip("meeko")
    from cycpep_master.docking import mol2_pdbqt as module

    parent_path, manifest, digest = _manifest(tmp_path)
    sigma_calls = []

    class UnavailablePrior:
        runtime_sha256 = hashlib.sha256(b"runtime").hexdigest()
        manifest_sha256 = hashlib.sha256(b"manifest").hexdigest()

        def query(self, _keys, *, flexibility_mode="balanced"):
            from cycpep_master.docking.torsion_prior import (
                TorsionPriorMatch,
            )

            return TorsionPriorMatch(
                status="unavailable",
                lookup_level=None,
                key=None,
                statistics=None,
                rigidity_score=None,
                confidence="none",
                eligible_to_freeze=False,
                reason="test",
            )

    def zero_sigma(ensemble, bonds, **_kwargs):
        sigma_calls.append((ensemble.GetNumConformers(), list(bonds)))
        return {bond: 0.0 for bond in bonds}

    monkeypatch.setattr(module, "bond_dihedral_sigma", zero_sigma)
    audit, error = mol2_to_ligand_pdbqt(
        parent_path,
        tmp_path / "balanced.pdbqt",
        torsdof_limit=0,
        flexibility_mode="balanced",
        ensemble_manifest_path=manifest,
        ensemble_manifest_sha256=digest,
        load_prior=lambda _path: UnavailablePrior(),
        embed_multiple=lambda *_args, **_kwargs: pytest.fail(
            "PDBQT must not embed"
        ),
    )

    assert error is None
    assert audit["effective_flexibility_mode"] == "balanced"
    assert audit["ensemble_fallback_triggered"] is True
    assert audit["embedding_performed"] is False
    assert sigma_calls and sigma_calls[0][0] == 2


def test_no_etkdg_calls_exist_in_pdbqt_or_torsion_budget_sources():
    package_root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("docking/mol2_pdbqt.py"),
        Path("docking/torsion_budget.py"),
    ):
        text = (package_root / relative).read_text(encoding="utf-8")
        assert "ETKDG" not in text
        assert "EmbedMolecule" not in text
        assert "EmbedMultipleConfs" not in text
