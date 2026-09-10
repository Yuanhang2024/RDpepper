from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from cycpep_master import application


PUBLIC_OPERATIONS = {
    "reconstruct",
    "reconstruct_structure",
    "reconstruct_unified",
    "reconstruct_exact_v1",
    "reconstruct_multichain",
    "reconstruct_result_first",
    "convert",
    "audit",
    "compare",
    "export",
    "export_best_available",
    "batch_export",
    "conformers",
    "template_lookup",
    "template_conformers",
    "admet",
    "protonate",
    "protonate_mol2",
    "validate_mol2",
    "prepare_ligand_from_sequence",
    "prepare_ligand_pdbqt_from_mol2",
    "prepare_ligand_pdbqt",
    "prepare_ligand_pdbqt_best_available",
    "prepare_ligand_pdbqt_from_pdb",
    "prepare_receptor_pdbqt",
    "validate_pdbqt",
    "docking_center",
    "run_prepared_vina",
    "dock",
    "batch_dock",
    "monomer_list",
    "monomer_add",
    "monomer_resolve",
}


def _fake_validation_receipt(mol2_path, **_kwargs):
    receipt = Path(str(mol2_path) + ".validation.json")
    receipt.write_text("{}\n", encoding="utf-8")
    return receipt


def test_capabilities_declares_every_public_operation():
    result = application.capabilities()

    assert result["status"] == "success"
    assert result["data"]["version"] == "7.1.0"
    assert set(result["data"]["operations"]) == PUBLIC_OPERATIONS
    assert "prepare_ligand_pdbqt_ensemble" not in application.__all__
    assert "prepare_ligand_pdbqt_ensemble" in (
        result["data"]["retired_operations"]
    )


def _atom_line(serial, name, x, y, z):
    line = (
        f"ATOM  {serial:5d} {name:<4s} LIG L   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
    )
    return line.ljust(76) + " C"


def _ring_pdb(n=8, bond=1.5):
    lines = ["HEADER    RING"]
    radius = bond / (2 * math.sin(math.pi / n))
    for index in range(n):
        angle = 2 * math.pi * index / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        lines.append(_atom_line(index + 1, f"C{index + 1}", x, y, 0.0))
    for index in range(n):
        partner = (index + 1) % n
        lines.append(f"CONECT{index + 1:5d}{partner + 1:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _peptide_like_ring_pdb(n_residues=3, bond=1.5):
    atom_names = ("N", "CA", "C")
    atom_count = n_residues * len(atom_names)
    radius = bond / (2 * math.sin(math.pi / atom_count))
    lines = ["HEADER    PEPTIDE-LIKE RING"]
    serial = 1
    for residue_number in range(1, n_residues + 1):
        for atom_name in atom_names:
            angle = 2 * math.pi * (serial - 1) / atom_count
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            element = "N" if atom_name == "N" else "C"
            line = (
                f"ATOM  {serial:5d} {atom_name:<4s} ALA L{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00"
            )
            lines.append(line.ljust(76) + f"{element:>2s}")
            serial += 1
    for left in range(1, atom_count + 1):
        right = left + 1 if left < atom_count else 1
        lines.append(f"CONECT{left:5d}{right:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_prepare_receptor_unsupported_element_is_typed_not_supported(tmp_path):
    source = tmp_path / "selenium.pdb"
    output = tmp_path / "selenium.pdbqt"
    source.write_text(
        "HETATM    1 SE   LIG A   1       0.000   0.000   0.000  1.00  0.00          Se  \n"
        "END\n",
        encoding="ascii",
    )
    output.write_text("stale output\n", encoding="ascii")

    result = application.prepare_receptor_pdbqt(source, output)

    assert result["status"] == "not_supported"
    assert result["error"] == (
        "not_supported: receptor atom type is unsupported by AutoDock4: 'Se'"
    )
    assert not output.exists()


def test_capabilities_includes_reconstruct_result_first():
    result = application.capabilities()

    assert "reconstruct_result_first" in result["data"]["operations"]


def test_capabilities_includes_reconstruct_unified():
    result = application.capabilities()

    assert "reconstruct_unified" in result["data"]["operations"]


def test_capabilities_includes_reconstruct_structure():
    result = application.capabilities()

    assert "reconstruct_structure" in result["data"]["operations"]


def test_convert_layers_map_connection_on_implicit_backbone_ports():
    result = application.convert_representation(
        "map", "smiles", "AG{cyc:1:R2-2:R1}"
    )

    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "PARTIAL"
    assert result["data"]["chemical_rigor"] == "C1:H"
    assert "implicit backbone bond" in result["error"]


def test_identity_conversion_still_applies_semantic_map_audit():
    result = application.convert_representation(
        "map", "map", "AG{cyc:1:R2-2:R1}"
    )

    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "PARTIAL"
    assert "implicit backbone bond" in result["error"]


@pytest.mark.parametrize("smiles", [None, "", " ", "not_a_smiles", "C1"])
def test_protonation_rejects_unparseable_smiles(smiles):
    result = application.protonate_smiles(smiles)

    assert result["status"] == "invalid_input"
    assert result.get("error")


def test_protonation_returns_parseable_output_for_valid_smiles():
    result = application.protonate_smiles("NCC(=O)O")

    assert result["status"] == "success"
    assert result["data"]["output_smiles"] == "[NH3+]CC(=O)[O-]"


def test_reconstruct_structure_service_contract():
    result = application.reconstruct_structure(
        "AAAAA{cyc:N-C}", mode="best-effort"
    )

    assert result["operation"] == "reconstruct_structure"
    assert result["status"] == "success"
    assert result["data"]["status"] == "success"
    assert result["data"]["quality"] == "high"
    assert result["data"]["mode"] == "best_effort"
    assert result["data"]["smiles"] is not None
    assert result["data"]["structure_profile"]["backbone_layout"] == "cyclic"


def test_reconstruct_unified_service_contract():
    result = application.reconstruct_unified(
        "AAAAA{cyc:N-C}", mode="best-effort"
    )

    assert result["operation"] == "reconstruct_unified"
    assert result["status"] == "success"
    assert result["data"]["status"] == "success"
    assert result["data"]["quality"] == "high"
    assert result["data"]["mode"] == "best_effort"
    assert result["data"]["smiles"] is not None
    assert result["data"]["structure_profile"]["backbone_layout"] == "cyclic"


def test_unified_service_forwards_result_first_controls(monkeypatch):
    observed = {}

    def fake_reconstruct(source, **kwargs):
        observed.update(source=source, **kwargs)
        return _downstream_reconstruction_payload(quality="topology")

    import cycpep_master.reconstruction as reconstruction_module

    monkeypatch.setattr(reconstruction_module, "reconstruct_structure", fake_reconstruct)
    result = application.reconstruct_structure(
        "input.pdb",
        chain_id="A",
        mode="best-effort",
        minimum_macrocycle_ring_size=11,
        require_empty_persistent_overlay=True,
    )

    assert result["status"] == "success"
    assert observed == {
        "source": "input.pdb",
        "chain_id": "A",
        "mode": "best-effort",
        "minimum_macrocycle_ring_size": 11,
        "require_empty_persistent_overlay": True,
        "radius_multiplier": None,
        "distance_ceiling": None,
        "allow_linear_topology": False,
    }


def test_reconstruct_result_first_forwards_macrocycle_threshold(
    tmp_path, monkeypatch
):
    from cycpep_master.remediation_v5 import StrictReconstructionResult

    strict = StrictReconstructionResult(
        status="rejected",
        warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
        path_used="V6_EVIDENCE_DIMENSION_AUDIT",
    )
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_prepared_structure_fail_closed_v6",
        lambda *args, **kwargs: strict,
    )
    path = tmp_path / "ring.pdb"
    path.write_text(_peptide_like_ring_pdb(), encoding="ascii")

    strict_payload = application.reconstruct_result_first(
        path, chain_id="L", minimum_macrocycle_ring_size=9
    )
    lenient_payload = application.reconstruct_result_first(
        path, chain_id="L", minimum_macrocycle_ring_size=10
    )

    assert strict_payload["data"]["quality"] == "topology"
    assert lenient_payload["data"]["quality"] == "partial"


def test_docking_discovery_only_returns_legacy_pdb_inputs(tmp_path):
    for name in ("a.pdb", "b.ent", "c.cif", "d.pdb.gz", "ignored.sdf"):
        (tmp_path / name).write_text("", encoding="ascii")

    assert [
        path.replace("\\", "/").rsplit("/", 1)[-1]
        for path in application.discover_docking_coordinate_files(tmp_path)
    ] == ["a.pdb", "b.ent"]


def test_batch_export_service_preserves_structured_results(tmp_path, monkeypatch):
    from cycpep_master.export import conformer

    observed = {}

    def fake_batch(items, output_dir, *, format, force_field):
        observed.update(
            items=items,
            output_dir=output_dir,
            format=format,
            force_field=force_field,
        )
        return [
            ("first", str(tmp_path / "first.sdf"), None),
            ("second", None, "embedding failed"),
        ]

    monkeypatch.setattr(conformer, "batch_export", fake_batch)
    result = application.batch_export_structures(
        {"first": "C", "second": "CC"},
        tmp_path,
        output_format="sdf",
        force_field="uff",
    )

    assert result["status"] == "partial"
    assert result["data"]["success_count"] == 1
    assert observed == {
        "items": [("first", "C"), ("second", "CC")],
        "output_dir": str(tmp_path),
        "format": "sdf",
        "force_field": "uff",
    }


def test_prepared_vina_service_forwards_public_contract(tmp_path, monkeypatch):
    from cycpep_master.docking import vina

    observed = {}

    def fake_run(ligand, receptor, center, box_size, output, **kwargs):
        observed.update(
            ligand=ligand,
            receptor=receptor,
            center=center,
            box_size=box_size,
            output=output,
            **kwargs,
        )
        return -7.25, None

    monkeypatch.setattr(vina, "run_vina", fake_run)
    ligand = tmp_path / "ligand.pdbqt"
    receptor = tmp_path / "receptor.pdbqt"
    output = tmp_path / "out.pdbqt"
    result = application.run_prepared_vina(
        ligand,
        receptor,
        output,
        center=(1, 2, 3),
        box_size=(20, 21, 22),
        exhaustiveness=17,
        num_modes=4,
    )

    assert result["status"] == "success"
    assert result["data"]["affinity_kcal_mol"] == -7.25
    assert observed == {
        "ligand": str(ligand),
        "receptor": str(receptor),
        "center": (1.0, 2.0, 3.0),
        "box_size": (20.0, 21.0, 22.0),
        "output": str(output),
        "exhaustiveness": 17,
        "num_modes": 4,
    }


def test_path_identity_normalizes_absolute_relative_and_parent_paths(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "nested" / "input.pdb"
    source.parent.mkdir()

    expected = application._path_identity(source)

    assert expected == application._path_identity("nested/../nested/input.pdb")
    assert expected == application._path_identity(str(source.absolute()))


def test_coordinate_export_rejects_same_file_aliases_without_deleting_input(
    tmp_path,
):
    source = tmp_path / "input.pdb"
    source.write_text("coordinate input\n", encoding="ascii")
    aliases = [source, tmp_path / "nested" / ".." / "input.pdb"]

    symlink = tmp_path / "symlink-input.pdb"
    try:
        symlink.symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        symlink = None
        symlink_error = exc
    else:
        symlink_error = None
    if symlink is not None:
        aliases.append(symlink)

    hardlink = tmp_path / "hardlink-input.pdb"
    try:
        os.link(source, hardlink)
    except (OSError, NotImplementedError) as exc:
        hardlink = None
        hardlink_error = exc
    else:
        hardlink_error = None
    if hardlink is not None:
        aliases.append(hardlink)

    for output in aliases:
        result = application.export_structure(
            source,
            output,
            source_kind="coordinate",
            output_format="mol2",
        )
        assert result["status"] == "invalid_input"
        assert "must not alias" in result["error"]
        assert source.read_text(encoding="ascii") == "coordinate input\n"

    if symlink_error is not None:
        pytest.skip(f"symlinks unavailable: {symlink_error}")
    if hardlink_error is not None:
        pytest.skip(f"hard links unavailable: {hardlink_error}")


@pytest.mark.parametrize(
    "error",
    [
        "Vina produced no output file: out.pdbqt",
        "Vina output is empty: out.pdbqt",
        "Vina output is not minimally parseable: out.pdbqt",
        "Vina output model 1 changed ligand atom invariant at position 1",
        "Vina ligand/output torsion-tree validation failed: malformed tree",
        "Vina ligand input must contain exactly one atom model",
    ],
)
def test_prepared_vina_maps_output_validation_errors_to_rejected(
    tmp_path, monkeypatch, error
):
    from cycpep_master.docking import vina

    monkeypatch.setattr(vina, "run_vina", lambda *_a, **_k: (None, error))

    result = application.run_prepared_vina(
        tmp_path / "ligand.pdbqt",
        tmp_path / "receptor.pdbqt",
        tmp_path / "out.pdbqt",
        center=(0, 0, 0),
    )

    assert result["status"] == "rejected"
    assert result["error"] == error


@pytest.mark.parametrize(
    "error",
    [
        "Vina failed (exit 1): Vina output is empty",
        "Vina execution error: Vina output model 1 changed ligand atom invariant",
        "Vina failed (exit 1): timed out while starting",
    ],
)
def test_prepared_vina_keeps_process_failures_failed(tmp_path, monkeypatch, error):
    from cycpep_master.docking import vina

    monkeypatch.setattr(vina, "run_vina", lambda *_a, **_k: (None, error))

    result = application.run_prepared_vina(
        tmp_path / "ligand.pdbqt",
        tmp_path / "receptor.pdbqt",
        tmp_path / "out.pdbqt",
        center=(0, 0, 0),
    )

    assert result["status"] == "failed"
    assert result["error"] == error


@pytest.mark.parametrize(
    "error",
    [
        "Vina output path must not alias the ligand input: ligand.pdbqt",
        "Vina output path must not alias the receptor input: receptor.pdbqt",
        "Cannot compare Vina input/output paths: invalid path",
    ],
)
def test_prepared_vina_maps_path_alias_errors_to_invalid_input(
    tmp_path, monkeypatch, error
):
    from cycpep_master.docking import vina

    monkeypatch.setattr(vina, "run_vina", lambda *_a, **_k: (None, error))

    result = application.run_prepared_vina(
        tmp_path / "ligand.pdbqt",
        tmp_path / "receptor.pdbqt",
        tmp_path / "out.pdbqt",
        center=(0, 0, 0),
    )

    assert result["status"] == "invalid_input"
    assert result["error"] == error


def _downstream_reconstruction_payload(*, quality="medium", smiles="CC"):
    return {
        "status": "success",
        "quality": quality,
        "result_origin": "result_first",
        "source_kind": "coordinate",
        "mode": "auto",
        "smiles": smiles,
        "graph": None if smiles else {"atoms": [], "bonds": []},
        "ambiguous": quality == "medium",
        "warnings": ["FORMAL_CHARGE_UNVERIFIED: formal charges unverified"],
        "warning_codes": ["FORMAL_CHARGE_UNVERIFIED"],
        "alternatives": [{"canonical_smiles": "C=C"}],
        "structure_profile": {"backbone_layout": "cyclic"},
        "provenance": {"ladder": "candidate_ensemble"},
    }


def test_export_accepts_reconstruction_envelope_and_preserves_context(
    tmp_path, monkeypatch
):
    import cycpep_master.export as export_module

    observed = {}

    def fake_export(smiles, output_path, **kwargs):
        observed.update(smiles=smiles, output_path=output_path, kwargs=kwargs)
        Path(output_path).write_text("mock mol2\n", encoding="utf-8")
        return output_path, None

    monkeypatch.setattr(export_module, "smiles_to_mol2", fake_export)
    monkeypatch.setattr(
        application,
        "_write_mol2_validation_receipt",
        _fake_validation_receipt,
    )
    payload = _downstream_reconstruction_payload()
    envelope = {
        "operation": "reconstruct_structure",
        "status": "success",
        "data": payload,
    }
    output = tmp_path / "result.mol2"

    result = application.export_structure(envelope, output)

    assert result["status"] == "success"
    assert observed["smiles"] == "CC"
    assert result["data"]["reconstruction"]["quality"] == "medium"
    assert result["data"]["reconstruction"]["warning_codes"] == [
        "FORMAL_CHARGE_UNVERIFIED"
    ]
    assert result["data"]["reconstruction"]["alternatives"]


def test_coordinate_mol2_export_uses_source_bound_materializer(
    tmp_path, monkeypatch
):
    observed = {}

    def fake_export(source, output_path=None, **kwargs):
        observed.update(source=source, output_path=output_path, **kwargs)
        Path(output_path).write_text("mock mol2\n", encoding="utf-8")
        return output_path, None

    def unexpected_regeneration(*_args, **_kwargs):
        raise AssertionError("PDB-to-MOL2 must not regenerate coordinates")

    monkeypatch.setattr(
        "cycpep_master.export.conformer.pdb_to_mol2", fake_export
    )
    monkeypatch.setattr(
        "cycpep_master.export.smiles_to_mol2", unexpected_regeneration
    )
    monkeypatch.setattr(
        application,
        "_reconstruct_coordinate_for_downstream",
        unexpected_regeneration,
    )
    monkeypatch.setattr(
        application,
        "_write_mol2_validation_receipt",
        _fake_validation_receipt,
    )
    output = tmp_path / "coordinate.mol2"

    result = application.export_structure(
        "input.pdb",
        output,
        source_kind="pdb",
        output_format="mol2",
        chain_id="A",
        path="result_first",
    )

    assert result["status"] == "success"
    assert result["data"]["coordinate_mode"] == "source_bound"
    assert result["data"]["fidelity_status"] == "source_bound"
    assert observed["source"] == "input.pdb"
    assert observed["chain_id"] == "A"
    assert observed["path"] == "result_first"


def test_coordinate_mol2_export_does_not_regenerate_after_mapping_failure(
    tmp_path, monkeypatch
):
    def unexpected_regeneration(*_args, **_kwargs):
        raise AssertionError("PDB-to-MOL2 must not regenerate coordinates")

    monkeypatch.setattr(
        "cycpep_master.export.conformer.pdb_to_mol2",
        lambda *_args, **_kwargs: (
            None,
            "not supported: source-bound mapping unavailable",
        ),
    )
    monkeypatch.setattr(
        "cycpep_master.export.smiles_to_mol2", unexpected_regeneration
    )
    result = application.export_structure(
        "input.pdb",
        tmp_path / "graph-only.mol2",
        source_kind="coordinate",
    )

    # Scientific mapping inability degrades to a typed diagnostic artifact;
    # it neither regenerates coordinates nor rejects the request.
    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "metadata"
    assert "source-bound mapping unavailable" in artifact["warnings"][0]
    assert not (tmp_path / "graph-only.mol2").exists()


def test_failed_reconstruction_envelope_cannot_promote_nested_success(
    tmp_path, monkeypatch
):
    def unexpected_promotion(*_args, **_kwargs):
        raise AssertionError("failed envelope must not reach a SMILES exporter")

    monkeypatch.setattr(
        "cycpep_master.export.smiles_to_mol2", unexpected_promotion
    )
    envelope = {
        "operation": "reconstruct_structure",
        "status": "failed",
        "data": _downstream_reconstruction_payload(),
    }

    result = application.export_structure(
        envelope,
        tmp_path / "must_not_exist.mol2",
    )

    # A failed envelope degrades to a diagnostic artifact and never
    # promotes its nested SMILES payload into a chemistry-bearing format.
    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    artifact = result["data"]["artifacts"][0]
    assert "envelope status is 'failed'" in artifact["warnings"][0]
    assert not (tmp_path / "must_not_exist.mol2").exists()


def test_graph_only_reconstruction_degrades_to_diagnostic_graph_artifact(
    tmp_path,
):
    payload = _downstream_reconstruction_payload(
        quality="topology", smiles=None
    )
    payload["graph"] = {"atoms": [{"serial": 1}], "bonds": []}
    payload["warnings"] = "FORMAL_CHARGE_UNVERIFIED: formal charges unverified"
    payload["warning_codes"] = "FORMAL_CHARGE_UNVERIFIED"
    payload["alternatives"] = {"canonical_smiles": "C=C"}
    output = tmp_path / "topology.mol2"

    result = application.export_structure(payload, output)

    # Graph-only evidence degrades to a Q1 diagnostic graph artifact
    # instead of a chemistry-bearing MOL2 or a rejection.
    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "degraded_format"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "graph_json"
    assert artifact["role"] == "diagnostic"
    companion = Path(str(output) + ".graph.json")
    document = json.loads(companion.read_text(encoding="utf-8"))
    assert document["graph"]["atoms"] == [{"serial": 1}]
    assert document["reconstruction"]["quality"] == "topology"
    assert document["reconstruction"]["warning_codes"] == [
        "FORMAL_CHARGE_UNVERIFIED"
    ]
    assert not output.exists()


def test_low_quality_smiles_is_attempted_by_each_downstream_consumer(
    tmp_path, monkeypatch
):
    import cycpep_master.export as export_module
    from cycpep_master.docking import workflow

    observed = []

    def fake_export(smiles, output_path, **kwargs):
        observed.append((
            (
                "mol2_parent"
                if str(output_path).endswith(".parent.mol2")
                else "mol2"
            ),
            smiles,
        ))
        Path(output_path).write_text("mock mol2\n", encoding="utf-8")
        return output_path, None

    def fake_pdbqt(parent_path, output_path, **kwargs):
        observed.append(("pdbqt_parent", str(parent_path)))
        Path(output_path).write_text("mock pdbqt\n", encoding="utf-8")
        return {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": "success",
            "data": {
                "output_path": str(output_path),
                "audit": {"source": "validated_parent_mol2"},
                "artifacts": [],
            },
        }

    def fake_dock(*args, **kwargs):
        observed.append(("dock", kwargs.get("ligand_smiles")))
        return -7.5, None

    monkeypatch.setattr(export_module, "smiles_to_mol2", fake_export)
    monkeypatch.setattr(
        application,
        "_write_mol2_validation_receipt",
        _fake_validation_receipt,
    )
    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_mol2", fake_pdbqt
    )
    monkeypatch.setattr(workflow, "dock_peptide", fake_dock)

    for quality in ("topology", "partial", "raw"):
        payload = _downstream_reconstruction_payload(
            quality=quality,
            smiles="CC",
        )
        export_result = application.export_structure(
            payload, tmp_path / f"{quality}.mol2"
        )
        pdbqt_result = application.prepare_ligand_pdbqt(
            payload,
            tmp_path / f"{quality}.pdbqt",
        )
        ensemble_result = application.prepare_ligand_pdbqt_ensemble(
            payload,
            tmp_path / f"{quality}_ensemble",
        )
        dock_result = application.dock_structure(
            "peptide.pdb",
            "receptor.pdb",
            center=(1, 2, 3),
            ligand_smiles=payload,
        )
        for result in (export_result, pdbqt_result, dock_result):
            assert result["status"] == "success"
            assert result["data"]["reconstruction"]["quality"] == quality
            assert result["data"]["reconstruction"]["warning_codes"] == [
                "FORMAL_CHARGE_UNVERIFIED"
            ]
        assert ensemble_result["status"] == "not_supported"

    assert [kind for kind, _value in observed] == [
        item
        for _quality in ("topology", "partial", "raw")
            for item in (
                "mol2",
                "pdbqt_parent",
                "dock",
            )
    ]


def test_downstream_exception_preserves_reconstruction_context(
    tmp_path, monkeypatch
):
    import cycpep_master.export as export_module

    def exploding_export(*args, **kwargs):
        raise RuntimeError("directed exporter failure")

    monkeypatch.setattr(export_module, "smiles_to_mol2", exploding_export)
    payload = _downstream_reconstruction_payload()

    result = application.export_structure(payload, tmp_path / "failed.mol2")

    assert result["status"] == "failed"
    assert "directed exporter failure" in result["error"]
    assert result["data"]["reconstruction"]["quality"] == "medium"
    assert result["data"]["reconstruction"]["warning_codes"] == [
        "FORMAL_CHARGE_UNVERIFIED"
    ]


def test_pdbqt_accepts_unified_result_and_preserves_context(
    tmp_path, monkeypatch
):
    from cycpep_master.reconstruction import reconstruct_structure

    observed = {}

    def fake_prepare(parent_path, output_path, **kwargs):
        observed.update(
            parent_path=str(parent_path),
            output_path=str(output_path),
            kwargs=kwargs,
        )
        Path(output_path).write_text("mock pdbqt\n", encoding="utf-8")
        return {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": "success",
            "data": {
                "output_path": str(output_path),
                "audit": {"source": "validated_parent_mol2"},
                "artifacts": [],
            },
        }

    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_mol2", fake_prepare
    )
    reconstruction = reconstruct_structure("AAAAA{cyc:N-C}")

    result = application.prepare_ligand_pdbqt(
        reconstruction, tmp_path / "ligand.pdbqt", num_confs=1
    )

    assert result["status"] == "success"
    assert observed["parent_path"].endswith(".parent.mol2")
    assert result["data"]["reconstruction"]["quality"] == "high"
    assert result["data"]["audit"]["source"] == "validated_parent_mol2"


def test_pdbqt_from_coordinate_uses_validated_parent_mol2(
    tmp_path, monkeypatch
):
    observed = {}

    def fake_export(_source, parent_path, **kwargs):
        Path(parent_path).write_text("mock mol2\n", encoding="utf-8")
        receipt = _fake_validation_receipt(parent_path)
        return {
            "operation": "export_best_available",
            "status": "success",
            "data": {
                "requested_format_status": "fulfilled",
                "validation_receipt_path": str(receipt),
                "artifacts": [],
            },
        }

    def fake_prepare(parent_path, output_path, **kwargs):
        observed.update(
            parent_path=str(parent_path),
            output_path=str(output_path),
            **kwargs,
        )
        Path(output_path).write_text("mock pdbqt\n", encoding="utf-8")
        return {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": "success",
            "data": {
                "output_path": str(output_path),
                "parent_coordinate_mode": "source_bound",
                "artifacts": [],
            },
        }

    monkeypatch.setattr(application, "export_best_available", fake_export)
    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_mol2", fake_prepare
    )
    result = application.prepare_ligand_pdbqt_from_pdb(
        "input.pdb",
        tmp_path / "ligand.pdbqt",
        chain_id="A",
        reconstruction_mode="strict",
        minimum_macrocycle_ring_size=10,
        require_empty_persistent_overlay=True,
    )

    assert result["status"] == "success"
    assert result["data"]["coordinate_source"] == "validated_parent_mol2"
    assert observed["parent_path"].endswith(".parent.mol2")
    assert observed["receipt_path"].endswith(".validation.json")


def test_dock_structure_default_auto_reconstructs_coordinate(monkeypatch):
    from cycpep_master.docking import workflow

    observed = {}

    def fake_reconstruct(source, **kwargs):
        observed.update(source=source, **kwargs)
        payload = _downstream_reconstruction_payload(quality="topology")
        return {
            "operation": "reconstruct_structure",
            "status": "success",
            "data": payload,
        }

    def fake_dock(*args, **kwargs):
        observed.update(dock_kwargs=kwargs)
        return -6.0, None

    monkeypatch.setattr(application, "reconstruct_structure", fake_reconstruct)
    monkeypatch.setattr(workflow, "dock_peptide", fake_dock)
    result = application.dock_structure(
        "peptide.pdb",
        "receptor.pdb",
        center=(1, 2, 3),
        reconstruction_mode="strict",
        minimum_macrocycle_ring_size=9,
    )

    assert result["status"] == "success"
    assert observed["source"] == "peptide.pdb"
    assert observed["mode"] == "strict"
    assert observed["minimum_macrocycle_ring_size"] == 9
    assert observed["dock_kwargs"]["ligand_preparation_mode"] == "audited_smiles"
    assert observed["dock_kwargs"]["ligand_smiles"] == "CC"
    assert result["data"]["reconstruction"]["quality"] == "topology"


def test_pdbqt_ensemble_input_error_preserves_reconstruction_context(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import ligand_pdbqt

    def invalid_ensemble(*args, **kwargs):
        raise ValueError("directed ensemble input failure")

    monkeypatch.setattr(
        ligand_pdbqt,
        "smiles_to_ligand_pdbqt_multi",
        invalid_ensemble,
    )
    payload = _downstream_reconstruction_payload()

    result = application.prepare_ligand_pdbqt_ensemble(
        payload,
        tmp_path / "ensemble",
    )

    assert result["status"] == "invalid_input"
    assert "directed ensemble input failure" in result["error"]
    assert result["data"]["reconstruction"]["quality"] == "medium"
    assert result["data"]["reconstruction"]["warning_codes"] == [
        "FORMAL_CHARGE_UNVERIFIED"
    ]


def test_dock_accepts_reconstruction_result_as_audited_smiles(monkeypatch):
    from cycpep_master.docking import workflow

    observed = {}

    def fake_dock(peptide, receptor, center, box_size, output_dir, **kwargs):
        observed.update(
            peptide=peptide,
            receptor=receptor,
            center=center,
            box_size=box_size,
            output_dir=output_dir,
            **kwargs,
        )
        kwargs["torsion_audit_out"].update({"status": "success"})
        return -6.5, None

    monkeypatch.setattr(workflow, "dock_peptide", fake_dock)
    reconstruction = _downstream_reconstruction_payload()

    result = application.dock_structure(
        "peptide.pdb",
        "receptor.pdb",
        center=(1, 2, 3),
        ligand_smiles=reconstruction,
    )

    assert result["status"] == "success"
    assert observed["ligand_preparation_mode"] == "audited_smiles"
    assert observed["ligand_smiles"] == "CC"
    assert result["data"]["reconstruction"]["quality"] == "medium"
    assert result["data"]["reconstruction"]["warnings"]


def test_batch_export_rejects_normalized_name_collision(tmp_path):
    result = application.batch_export_structures(
        [("a/b", "CC"), ("a\\b", "CCC")],
        tmp_path,
        output_format="sdf",
    )

    assert result["status"] == "failed"
    assert result["data"]["success_count"] == 0
    assert all("collision" in row["error"] for row in result["data"]["results"])
    assert not (tmp_path / "a_b.sdf").exists()


def test_admet_rejects_output_cardinality_mismatch(monkeypatch):
    import cycpep_master.admet as admet_module

    monkeypatch.setattr(
        admet_module,
        "run_admet",
        lambda _values: [{"input_smiles": "CC", "QED": 0.5}],
    )

    result = application.predict_admet(["CC", "CCC"])

    assert result["status"] == "failed"
    assert "cardinality" in result["error"]


def test_admet_rejects_output_identity_mismatch(monkeypatch):
    import cycpep_master.admet as admet_module

    monkeypatch.setattr(
        admet_module,
        "run_admet",
        lambda _values: [{"input_smiles": "CCC", "QED": 0.5}],
    )

    result = application.predict_admet(["CC"])

    assert result["status"] == "failed"
    assert "identity" in result["error"]


def test_template_lookup_rejects_missing_indexed_file(monkeypatch):
    from cycpep_master.docking import template_library

    monkeypatch.setattr(
        template_library,
        "find_template",
        lambda *_a, **_k: {"pdb_path": "missing/template.pdb"},
    )

    result = application.find_conformer_template("AAAA")

    assert result["status"] == "failed"
    assert "missing or empty" in result["error"]


def test_coordinate_sdf_does_not_use_smiles_from_failed_row(
    tmp_path, monkeypatch
):
    stale = tmp_path / "failed.sdf"
    stale.write_text("old result", encoding="utf-8")
    monkeypatch.setattr(
        application,
        "reconstruct_coordinates",
        lambda *_a, **_k: {
            "status": "failed",
            "data": {"results": [{"status": "failed", "smiles": "CC"}]},
        },
    )

    result = application.export_structure(
        "input.pdb", stale, source_kind="coordinate", output_format="sdf"
    )

    # A failed reconstruction degrades to a typed artifact and never uses
    # the stale SMILES from the failed row.
    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    assert not stale.exists()


@pytest.mark.parametrize(
    ("error", "expected_status", "expect_degrade"),
    [
        ("invalid_input: torsion limit must be non-negative", "invalid_input", False),
        ("not_supported: unsupported ligand element(s): Se", "success", True),
        ("rejected: malformed PDBQT output", "success", True),
        ("unexpected ligand worker failure", "failed", False),
    ],
)
def test_export_worker_errors_degrade_or_stay_typed(
    tmp_path, monkeypatch, error, expected_status, expect_degrade
):
    from cycpep_master import export as export_module

    monkeypatch.setattr(
        export_module,
        "smiles_to_mol2",
        lambda *_args, **_kwargs: (None, error),
    )

    result = application.export_structure("CC", tmp_path / "result.mol2")

    assert result["status"] == expected_status
    assert not (tmp_path / "result.mol2").exists()
    if expect_degrade:
        # Scientific inability degrades to a typed diagnostic artifact.
        assert result["data"]["requested_format_status"] in {
            "metadata_only", "degraded_format",
        }
        assert result["data"]["artifacts"][0]["warnings"] == [error]
    else:
        assert result["error"] == error


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        ("invalid_input: torsion limit must be non-negative", "invalid_input"),
        ("not_supported: unsupported ligand element(s): Se", "not_supported"),
        ("rejected: malformed PDBQT output", "rejected"),
        ("unexpected ligand worker failure", "failed"),
    ],
)
def test_prepare_ligand_pdbqt_preserves_typed_worker_error_status(
    tmp_path, monkeypatch, error, expected_status
):
    def fake_standard(*_args, **_kwargs):
        return {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": expected_status,
            "data": {},
            "error": error,
        }

    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_mol2", fake_standard
    )

    result = application.prepare_ligand_pdbqt(
        "CC", tmp_path / "result.pdbqt", num_confs=1
    )

    assert result["status"] == expected_status
    assert result["error"] == error


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        ("invalid_input: invalid SMILES", "invalid_input"),
        ("not_supported: unsupported ligand element(s): Se", "not_supported"),
        ("rejected: malformed output", "rejected"),
        ("worker crashed", "failed"),
    ],
)
def test_batch_export_preserves_typed_worker_error_status(
    tmp_path, monkeypatch, error, expected_status
):
    from cycpep_master.export import conformer

    monkeypatch.setattr(
        conformer,
        "batch_export",
        lambda *_args, **_kwargs: [("item", None, error)],
    )

    result = application.batch_export_structures(
        {"item": "CC"}, tmp_path
    )

    assert result["status"] == expected_status
    assert result["data"]["results"][0]["status"] == expected_status


def test_ligand_pdbqt_success_requires_nonempty_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        application,
        "prepare_ligand_pdbqt_from_mol2",
        lambda *_args, **_kwargs: {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": "failed",
            "data": {},
            "error": "converter produced no nonempty output",
        },
    )

    result = application.prepare_ligand_pdbqt(
        "CC", tmp_path / "missing.pdbqt", num_confs=1
    )

    assert result["status"] == "failed"
    assert "nonempty" in result["error"]


def test_ligand_pdbqt_ensemble_success_requires_every_output(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import ligand_pdbqt

    existing = tmp_path / "one.pdbqt"
    existing.write_text("one\n", encoding="utf-8")
    missing = tmp_path / "two.pdbqt"
    monkeypatch.setattr(
        ligand_pdbqt,
        "smiles_to_ligand_pdbqt_multi",
        lambda *_a, **_k: ([str(existing), str(missing)], None),
    )

    result = application.prepare_ligand_pdbqt_ensemble(
        "CC", tmp_path, n_conformers=2
    )

    assert result["status"] == "failed"
    assert "missing or empty" in result["error"]


def test_docking_center_rejects_explicit_empty_residue_list():
    result = application.docking_center("receptor.pdb", residue_ids=[])

    assert result["status"] == "invalid_input"


def test_batch_dock_preserves_typed_row_statuses(monkeypatch):
    from cycpep_master.docking import workflow

    monkeypatch.setattr(
        workflow,
        "batch_dock_peptides",
        lambda *_a, **_k: [
            ("timeout", None, "Vina timed out"),
            ("unsupported", None, "not_supported: selenium"),
        ],
    )

    result = application.batch_dock_structures(
        ["a.pdb", "b.pdb"], "r.pdb", center=(0, 0, 0)
    )

    assert result["status"] == "failed"
    assert [row["status"] for row in result["data"]["results"]] == [
        "timeout", "not_supported"
    ]


def test_json_ready_normalizes_nonfinite_and_mixed_sets():
    value = application.json_ready({
        "nan": float("nan"),
        "inf": float("inf"),
        "mixed": {1, "a"},
    })

    assert value["nan"] is None
    assert value["inf"] is None
    assert set(value["mixed"]) == {1, "a"}


def test_invalid_export_source_kind_preserves_existing_output(tmp_path):
    output = tmp_path / "old.mol2"
    output.write_text("validated old output\n", encoding="utf-8")

    result = application.export_structure(
        "CC", output, source_kind="bogus", output_format="mol2"
    )

    assert result["status"] == "invalid_input"
    assert output.read_text(encoding="utf-8") == "validated old output\n"


def test_batch_export_malformed_collection_returns_invalid_input(tmp_path):
    result = application.batch_export_structures(None, tmp_path)

    assert result["status"] == "invalid_input"


@pytest.mark.parametrize(
    ("center", "box_size"),
    [((float("nan"), 0, 0), (10, 10, 10)), ((0, 0, 0), (0, 10, 10))],
)
def test_prepared_vina_rejects_nonfinite_or_nonpositive_box(center, box_size):
    result = application.run_prepared_vina(
        "l.pdbqt", "r.pdbqt", "o.pdbqt", center=center, box_size=box_size
    )

    assert result["status"] == "invalid_input"


def test_export_rejects_invalid_fallback_policy_as_invalid_input(tmp_path):
    result = application.export_best_available(
        "CC", tmp_path / "ligand.mol2", fallback_policy="invalid"
    )

    assert result["status"] == "invalid_input"
    assert result["data"]["requested_format_status"] == "unavailable"
    assert not (tmp_path / "ligand.mol2").exists()


def test_reconstruct_coordinates_rejects_nondefault_geometry():
    result = application.reconstruct_coordinates(
        ["input.pdb"], radius_multiplier=1.31
    )

    assert result["status"] == "invalid_input"
    assert "does not consume non-default geometry" in result["error"]


def test_reconstruct_coordinates_forwards_fallback_policy(monkeypatch):
    captured = {}

    def fake_run_batch(*_args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("cycpep_master.pipeline.run_batch", fake_run_batch)

    application.reconstruct_coordinates(
        ["input.pdb"],
        fallback_policy="max_coverage",
        allow_linear_topology=True,
    )

    assert captured["fallback_policy"] == "max_coverage"
    assert captured["allow_linear_topology"] is True


def test_reconstruct_coordinates_cardinality_mismatch_is_not_success(monkeypatch):
    import cycpep_master.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "run_batch",
        lambda *_args, **_kwargs: [{"status": "success", "smiles": "CC"}],
    )

    result = application.reconstruct_coordinates(["a.pdb", "b.pdb"])

    assert result["status"] == "partial"
    assert "cardinality" in result["error"]
    assert result["data"]["requested_count"] == 2
    assert result["data"]["returned_count"] == 1
    assert result["data"]["cardinality_mismatch"] is True


def test_reconstruct_coordinates_empty_underlying_rows_is_failed(monkeypatch):
    import cycpep_master.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "run_batch", lambda *_a, **_k: [])

    result = application.reconstruct_coordinates(["a.pdb"])

    assert result["status"] == "failed"
    assert "cardinality" in result["error"]
    assert result["data"]["requested_count"] == 1
    assert result["data"]["returned_count"] == 0
    assert result["data"]["cardinality_mismatch"] is True


def test_batch_export_cardinality_mismatch_is_not_success(tmp_path, monkeypatch):
    import cycpep_master.export.conformer as conformer_module

    monkeypatch.setattr(
        conformer_module,
        "batch_export",
        lambda *_args, **_kwargs: [("first", str(tmp_path / "first.mol2"), None)],
    )

    result = application.batch_export_structures(
        {"first": "C", "second": "CC"}, tmp_path
    )

    assert result["status"] == "partial"
    assert "cardinality" in result["error"]
    assert result["data"]["requested_count"] == 2
    assert result["data"]["returned_count"] == 1
    assert result["data"]["cardinality_mismatch"] is True


def test_batch_dock_cardinality_mismatch_is_not_success(monkeypatch):
    import cycpep_master.docking.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "batch_dock_peptides",
        lambda *_args, **_kwargs: [("a", -5.0, None)],
    )

    result = application.batch_dock_structures(
        ["a.pdb", "b.pdb"], "r.pdb", center=(0, 0, 0)
    )

    assert result["status"] == "partial"
    assert "cardinality" in result["error"]
    assert result["data"]["requested_count"] == 2
    assert result["data"]["returned_count"] == 1
    assert result["data"]["cardinality_mismatch"] is True


def test_generate_template_conformers_fewer_than_requested_is_not_success(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import template_library

    monkeypatch.setattr(
        template_library,
        "generate_conformers",
        lambda *_args, **_kwargs: (object(), [1]),
    )
    output = tmp_path / "short.sdf"

    result = application.generate_template_conformers(
        "CC", "AAAA", output, n_conformers=3
    )

    assert result["status"] == "failed"
    assert "cardinality" in result["error"]
    assert result["data"]["requested_count"] == 3
    assert result["data"]["conformer_count"] == 1
    assert result["data"]["cardinality_mismatch"] is True
    assert not output.exists()


def test_export_structure_exception_cleans_partial_output_only(
    tmp_path, monkeypatch
):
    import cycpep_master.export as export_module

    output = tmp_path / "partial.mol2"
    unrelated = tmp_path / "unrelated.mol2"
    unrelated.write_text("keep me\n", encoding="utf-8")

    def exploding_export(smiles, output_path, **kwargs):
        Path(output_path).write_text("partial output\n", encoding="utf-8")
        raise RuntimeError("directed exporter failure after partial write")

    monkeypatch.setattr(export_module, "smiles_to_mol2", exploding_export)

    result = application.export_structure("CC", output)

    assert result["status"] == "failed"
    assert "directed exporter failure" in result["error"]
    assert not output.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep me\n"


# ---------------------------------------------------------------------------
# Coordinate-provenance receipts at the public boundary (regression: the
# ledger must never upgrade a regenerated export to X3/source_bound, and
# candidate-level identities are compared at the connectivity block).
# ---------------------------------------------------------------------------


def _peptide_ring_pdb(residue_count=3, bond=1.5):
    atom_count = residue_count * 3
    radius = bond / (2 * math.sin(math.pi / atom_count))
    lines = ["HEADER    PEPTIDE RING"]
    serial = 0
    for residue in range(1, residue_count + 1):
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
            serial += 1
            angle = 2 * math.pi * (serial - 1) / atom_count
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            line = (
                f"ATOM  {serial:5d} {name:>4s} ALA L{residue:4d}    "
                f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00"
            ).ljust(76)
            lines.append(line + f"{element:>2s}")
    for left in range(1, atom_count + 1):
        right = left + 1 if left < atom_count else 1
        lines.append(f"CONECT{left:5d}{right:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _read_receipt(mol2_path):
    import json

    receipt_path = Path(str(mol2_path) + ".validation.json")
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def test_smiles_mol2_export_receipt_is_regenerated_x1(tmp_path):
    output = tmp_path / "smiles.mol2"

    result = application.export_structure(
        "NCC(=O)O", output, source_kind="smiles", output_format="mol2"
    )

    assert result["status"] == "success"
    data = result["data"]
    assert data["coordinate_mode"] == "regenerated"
    assert data["coordinate_level"] == "X1"
    assert data["fidelity_status"] == "regenerated"
    assert data["rigor"] == "L2:H"
    assert data["quality"] == "hypothesis"
    receipt = _read_receipt(output)
    assert receipt["coordinate_mode"] == "regenerated"
    assert receipt["coordinate_level"] == "X1"
    assert receipt["generated_heavy_atom_count"] == 5
    assert receipt["mapped_heavy_atom_indices"] == []
    assert all(
        origin == "generated"
        for origin in receipt["atom_coordinate_origins"].values()
    )


def test_best_available_coordinate_mol2_receipt_stays_source_bound_x3(
    tmp_path,
):
    source = tmp_path / "ring.pdb"
    source.write_text(_peptide_ring_pdb(), encoding="ascii")
    output = tmp_path / "ring.mol2"

    result = application.export_best_available(
        str(source), output, source_kind="pdb", output_format="mol2"
    )

    assert result["status"] == "success"
    data = result["data"]
    assert data["coordinate_mode"] == "source_bound"
    assert data["coordinate_level"] == "X3"
    assert data["fidelity_status"] == "source_bound"
    receipt = _read_receipt(output)
    assert receipt["coordinate_mode"] == "source_bound"
    assert receipt["coordinate_level"] == "X3"
    assert receipt["generated_heavy_atom_indices"] == []


def test_candidate_identity_receipt_compares_connectivity_block(tmp_path):
    # A stereo-silent candidate identity exported from 3D coordinates must
    # not fail a full-InChIKey gate it never claimed, and must not silently
    # pass one either: the receipt records the connectivity-block check.
    source = tmp_path / "ring.pdb"
    source.write_text(_peptide_ring_pdb(), encoding="ascii")
    output = tmp_path / "ring.mol2"

    result = application.export_best_available(
        str(source), output, source_kind="pdb", output_format="mol2"
    )

    assert result["status"] == "success"
    receipt = _read_receipt(output)
    assert receipt["identity_expectation"] in {
        None, "connectivity_block",
    }


def test_prepare_ligand_pdbqt_parent_receipt_is_regenerated_x1(tmp_path):
    output = tmp_path / "ligand.pdbqt"

    result = application.prepare_ligand_pdbqt("NCC(=O)O", output)

    assert result["status"] == "success"
    parent = tmp_path / "ligand.pdbqt.parent.mol2"
    receipt = _read_receipt(parent)
    assert receipt["coordinate_mode"] == "regenerated"
    assert receipt["coordinate_level"] == "X1"
    assert receipt["generated_heavy_atom_count"] == 5
    assert receipt["mapped_heavy_atom_indices"] == []


def test_strict_scientific_export_failure_degrades_not_fails(tmp_path):
    # One residue cannot qualify as a cyclic peptide: strict_v6 export is
    # scientifically impossible, which degrades to a typed artifact rather
    # than a product-level failure.
    source = tmp_path / "fragment.pdb"
    source.write_text(
        "ATOM      1  N   ALA L   1       1.000   2.000   3.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA L   1       2.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA L   1       3.000   2.000   3.000  1.00  0.00           C\n"
        "CONECT    1    2\nCONECT    2    3\nEND\n",
        encoding="ascii",
    )
    output = tmp_path / "fragment.mol2"

    result = application.export_structure(
        str(source),
        output,
        source_kind="pdb",
        output_format="mol2",
        fallback_policy="strict_v6",
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "metadata"
    assert artifact["role"] == "diagnostic"
    assert artifact["warnings"]
    assert not output.exists()


def test_pdbqt_format_unavailable_for_low_rigor_parent(tmp_path):
    # A parent MOL2 whose chemistry is not settled (topology tier) must not
    # be laundered into a validated PDBQT: the format is honestly
    # unavailable while the parent artifact is still delivered.
    mol2 = tmp_path / "parent.mol2"
    produced = application.export_structure(
        "NCC(=O)O", mol2, output_format="mol2"
    )
    assert produced["status"] == "success"
    receipt_path = Path(str(mol2) + ".validation.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["quality"] = "topology"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = application.prepare_ligand_pdbqt_from_mol2(
        mol2, tmp_path / "ligand.pdbqt"
    )

    assert result["status"] == "success"
    data = result["data"]
    assert data["requested_format"] == "pdbqt"
    assert data["requested_format_status"] == "unavailable"
    assert "PDBQT requires settled chemistry" in data["reason"]
    artifact = data["artifacts"][0]
    assert artifact["format"] == "mol2"
    assert not (tmp_path / "ligand.pdbqt").exists()
