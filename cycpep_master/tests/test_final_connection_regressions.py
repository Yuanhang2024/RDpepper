from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master import application, remediation_v6
from cycpep_master.core import cyclization
from cycpep_master.docking.mol2_input import (
    Mol2ValidationError,
    load_validated_mol2,
    write_validation_receipt,
)
from cycpep_master.export.conformer import mol_to_mol2


def _pdb_atom(
    serial: int,
    residue: int,
    name: str,
    element: str,
    xyz: tuple[float, float, float],
) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:>4s} ALA A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
        f"{element:>2s}  "
    )


def _mol2_parent(tmp_path: Path) -> Path:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=23) == 0
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetProp("_TriposAtomName", f"{atom.GetSymbol()}{index}")
        atom.SetProp("_TriposResidueName", "LIG")
        atom.SetProp("_TriposChainId", "L")
        atom.SetIntProp("_TriposResidueNumber", 1)
        atom.SetProp("_TriposInsertionCode", "")
    output = tmp_path / "parent.mol2"
    produced, error = mol_to_mol2(molecule, output_path=str(output))
    assert error is None and Path(produced) == output
    return output


def test_request_local_identity_memo_is_shared_across_v3_v5_v6(
    monkeypatch
):
    from cycpep_master import remediation_v3, remediation_v5, remediation_v6
    from cycpep_master.core import identity_memo

    original = identity_memo.Chem.MolFromSmiles
    calls = []

    def counted(smiles, *args, **kwargs):
        calls.append(smiles)
        return original(smiles, *args, **kwargs)

    monkeypatch.setattr(identity_memo.Chem, "MolFromSmiles", counted)
    with identity_memo.identity_memo_context():
        assert remediation_v3._identity("CC")
        assert remediation_v5._canonical_identity("CC")[1]
        assert remediation_v6._candidate_identity("CC")

    assert calls == ["CC"]


def test_mapping_divergent_bundle_has_no_automatic_smiles_handoff():
    from cycpep_master import result_first

    candidates = [
        {
            "full_inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
            "canonical_smiles": "CC",
            "route": "a",
            "mapping_candidate_fingerprints": [["first"]],
        },
        {
            "full_inchikey": "ZZZZZZZZZZZZZZ-YYYYYYYYYY-X",
            "canonical_smiles": "CCC",
            "route": "e",
            "mapping_candidate_fingerprints": [["second"]],
        },
    ]
    strict = SimpleNamespace(
        status="rejected",
        qualified_success=False,
        rejection_reason="fixture",
        warning_codes=[],
        path_used="fixture",
    )
    provenance = {
        "mapping_aware": {},
        "ladder_attempts": [],
        "prepared": {},
    }

    result = result_first._mapping_aware_result(
        candidates, strict, provenance, []
    )
    is_reconstruction, smiles, context, error = (
        application._reconstruction_handoff(
            result, allow_candidate_smiles=True
        )
    )

    assert result.status == "success"
    assert result.quality == "medium"
    assert result.ambiguous is True
    assert result.smiles is None
    assert result.candidate_smiles is None
    assert isinstance(result.result, dict)
    assert len(result.result["candidates"]) == 2
    assert len(result.alternatives) == 2
    assert is_reconstruction is True
    assert smiles is None
    assert "no SMILES payload" in error
    assert context["ambiguous"] is True


def test_mapping_divergent_export_retains_candidate_bundle(
    tmp_path
):
    from cycpep_master import result_first

    candidates = [
        {
            "full_inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
            "canonical_smiles": "CC",
            "route": "a",
            "mapping_candidate_fingerprints": [["first"]],
        },
        {
            "full_inchikey": "ZZZZZZZZZZZZZZ-YYYYYYYYYY-X",
            "canonical_smiles": "CCC",
            "route": "e",
            "mapping_candidate_fingerprints": [["second"]],
        },
    ]
    strict = SimpleNamespace(
        status="rejected",
        qualified_success=False,
        rejection_reason="fixture",
        warning_codes=[],
        path_used="fixture",
    )
    result = result_first._mapping_aware_result(
        candidates,
        strict,
        {"mapping_aware": {}, "ladder_attempts": [], "prepared": {}},
        [],
    )

    exported = application.export_best_available(
        result, tmp_path / "ambiguous.mol2"
    )

    assert exported["status"] == "success"
    assert exported["data"]["requested_format_status"] == "metadata_only"
    artifact = Path(exported["data"]["artifacts"][0]["path"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(payload["chemistry_candidates"]) == 2
    assert not (tmp_path / "ambiguous.mol2").exists()


def test_result_first_binding_rejects_source_drift():
    from cycpep_master.export.conformer import _result_first_binding_error
    from cycpep_master.paths._map_utils import registry_epoch

    result = SimpleNamespace(provenance={
        "request_binding": {
            "source_sha256": "a" * 64,
            "normalized_sha256": "b" * 64,
            "normalized_chain_id": "L",
            "minimum_macrocycle_ring_size": 8,
            "require_empty_persistent_overlay": False,
            "infer_bond_orders": True,
            "registry_epoch": registry_epoch(),
        }
    })

    error = _result_first_binding_error(
        result,
        coordinate_input_evidence={
            "source_sha256": "c" * 64,
            "normalized_sha256": "b" * 64,
        },
        chain_id="L",
        minimum_macrocycle_ring_size=8,
        require_empty_persistent_overlay=False,
    )

    assert error == "result-first source binding mismatch: source_sha256"


def test_mol2_heavy_atom_ledger_uses_actual_atom_positions(tmp_path):
    mol2 = tmp_path / "interleaved.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\ninterleaved\n4 0 0 0 0\nSMALL\n"
        "NO_CHARGES\n@<TRIPOS>ATOM\n"
        "1 H1 0 0 0 H 1 LIG 0\n"
        "2 C1 1 0 0 C.3 1 LIG 0\n"
        "3 H2 2 0 0 H 1 LIG 0\n"
        "4 O1 3 0 0 O.2 1 LIG 0\n",
        encoding="ascii",
    )

    assert application._mol2_heavy_atom_indices(mol2) == [1, 3]


def test_x2_receipt_roundtrip_validates_complete_atom_origins(tmp_path):
    parent = _mol2_parent(tmp_path)
    receipt = write_validation_receipt(
        parent,
        coordinate_mode="template_completed",
        coordinate_level="X2",
        rigor="L2:H",
        quality="candidate",
        source_heavy_atom_mapping_complete=False,
        atom_provenance_complete=True,
        mapped_heavy_atom_indices=[0],
        generated_heavy_atom_indices=[1],
        atom_coordinate_origins={"0": "source", "1": "generated"},
    )

    validated = load_validated_mol2(parent, receipt_path=receipt)
    assert validated.coordinate_mode == "template_completed"
    assert validated.coordinate_level == "X2"

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["atom_coordinate_origins"]["1"] = "source"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Mol2ValidationError, match="atom origins"):
        load_validated_mol2(parent, receipt_path=receipt)


def test_atom_parse_cache_returns_defensive_copies(tmp_path):
    source = tmp_path / "cache.pdb"
    source.write_text(
        "\n".join([
            _pdb_atom(1, 1, "N", "N", (0.0, 0.0, 0.0)),
            _pdb_atom(2, 1, "C", "C", (1.3, 0.0, 0.0)),
            "END",
            "",
        ]),
        encoding="ascii",
    )
    cyclization._clear_atom_parse_cache()

    first = cyclization.read_atoms(str(source), "A")
    first[1]["name"] = "MUTATED"
    second = cyclization.read_atoms(str(source), "A")

    assert second[1]["name"] == "N"
    assert first is not second
    assert first[1] is not second[1]


def test_spatial_crosslink_scan_matches_bruteforce_oracle(tmp_path):
    source = tmp_path / "geometry.pdb"
    source.write_text(
        "\n".join([
            _pdb_atom(1, 1, "SG", "S", (0.0, 0.0, 0.0)),
            _pdb_atom(2, 2, "SG", "S", (2.05, 0.0, 0.0)),
            _pdb_atom(3, 1, "C", "C", (10.0, 0.0, 0.0)),
            _pdb_atom(4, 2, "N", "N", (11.3, 0.0, 0.0)),
            _pdb_atom(5, 1, "N", "N", (20.0, 0.0, 0.0)),
            _pdb_atom(6, 2, "C", "C", (21.3, 0.0, 0.0)),
            _pdb_atom(7, 3, "SG", "S", (100.0, 0.0, 0.0)),
            "END",
            "",
        ]),
        encoding="ascii",
    )
    atoms = cyclization.read_atoms(str(source), "A")
    position_of = {
        residue: position
        for position, residue in enumerate(
            sorted({atom["resid"] for atom in atoms.values()}), start=1
        )
    }
    candidates = [
        (serial, atom)
        for serial, atom in atoms.items()
        if atom["elem"] in cyclization._LINKABLE_ELEMS
        and atom["name"] not in cyclization._BACKBONE_OTHER
        and atom["resn"] not in cyclization._CAPS
    ]
    multiplier, ceiling = cyclization.resolve_geometry_params(None, None)
    expected = []
    for left in range(len(candidates)):
        serial_left, atom_left = candidates[left]
        for right in range(left + 1, len(candidates)):
            serial_right, atom_right = candidates[right]
            if atom_left["resid"] == atom_right["resid"]:
                continue
            if not cyclization._is_covalent_bond(
                atom_left,
                atom_right,
                radius_multiplier=multiplier,
                distance_ceiling=ceiling,
            ):
                continue
            if cyclization._is_forward_backbone_pair(
                atom_left["name"],
                position_of[atom_left["resid"]],
                atom_right["name"],
                position_of[atom_right["resid"]],
            ):
                continue
            expected.append((min(serial_left, serial_right), max(serial_left, serial_right)))

    observed = cyclization.geometric_crosslink_atom_pairs(
        str(source), "A"
    )

    assert observed == expected
    assert (1, 2) in observed
    assert (3, 4) not in observed
    assert (5, 6) in observed


def test_v6_fresh_replay_explicitly_bypasses_combo_cache(monkeypatch):
    smiles = "CC"
    inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    provenance = {
        "schema_version": "1.0.0-path-a-e-generation-provenance.1",
        "generator": "cycpep_master.paths.path_a.generate_with_evidence",
        "requested_route": "a",
        "geometric_cyclization_argument": False,
        "geometry_stage_executed": False,
    }
    base_evidence = {
        "route": "a",
        "geometry_inference_enabled": False,
        "generation_provenance": provenance,
    }
    supplied = {
        **base_evidence,
        "output_smiles": smiles,
        "error": None,
    }
    observed_kwargs = []

    def replay(*_args, **kwargs):
        observed_kwargs.append(kwargs)
        return smiles, None, dict(base_evidence)

    monkeypatch.setattr(remediation_v6, "generate_with_evidence", replay)
    fresh, dimension = remediation_v6._fresh_path_evidence_replay(
        [{
            "route": "a",
            "status": "success",
            "output_smiles": smiles,
            "output_inchikey": inchikey,
        }],
        {"a": supplied},
        "input.pdb",
        "A",
    )

    assert observed_kwargs == [{
        "geometric_cyclization": False,
        "_use_cache": False,
    }]
    assert dimension["passed"] is True
    assert fresh["a"] == supplied


def test_unified_result_preserves_result_first_candidate_contract():
    from cycpep_master.reconstruction import _from_result_first

    candidate_graph = {
        "atoms": [{"serial": 1, "element": "C", "xyz": [0, 0, 0]}],
        "bonds": [],
    }
    rf = SimpleNamespace(
        status="success",
        quality="candidate",
        source="rdkit_bond_order_inference",
        result=None,
        smiles=None,
        graph=None,
        ambiguous=False,
        warnings=[],
        warning_codes=["INFERRED_CHEMISTRY_UNQUALIFIED"],
        alternatives=[],
        provenance={},
        strict_status="rejected",
        strict_result=SimpleNamespace(status="rejected"),
        candidate_smiles="CC",
        candidate_graph=candidate_graph,
        chemistry_candidates=[{"smiles": "CC"}],
        bond_order_inference={"status": "candidate"},
        candidate_rigor="L2:H",
        artifact_status="unqualified_candidate",
        qualification_status="unqualified",
        chemical_rigor="C2:H",
        coordinate_evidence="X3",
    )

    unified = _from_result_first(
        rf,
        source_kind="pdb",
        mode="auto",
        chain="A",
        detection={"source_kind": "pdb"},
    )
    is_reconstruction, smiles, context, error = (
        application._reconstruction_handoff(
            unified, allow_candidate_smiles=True
        )
    )

    assert is_reconstruction is True
    assert error is None
    assert smiles == "CC"
    assert unified.candidate_graph == candidate_graph
    assert unified.artifact_status == "unqualified_candidate"
    assert context["strict_status"] == "rejected"


def test_path_h_bounded_shortest_path_preserves_decision_boundary():
    from cycpep_master.paths.path_h import _shortest_path_length

    adjacency = {
        index: {index - 1, index + 1}
        for index in range(1, 8)
    }
    adjacency[0] = {1}
    adjacency[8] = {7}

    assert _shortest_path_length(adjacency, 0, 6, limit=6) == 6
    assert _shortest_path_length(adjacency, 0, 7, limit=6) is None
    assert _shortest_path_length(adjacency, 0, 8, limit=6) is None


def test_combo_cache_key_canonicalizes_override_mapping_order(tmp_path):
    from cycpep_master.paths.path_a import _combo_cache_key

    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    common = {
        "collect_evidence": False,
        "radius_multiplier": None,
        "distance_ceiling": None,
    }
    first = _combo_cache_key(
        source,
        "A",
        False,
        mapping_overrides={1: {"serial": 2}, "x": [3, 4]},
        **common,
    )
    reordered = _combo_cache_key(
        source,
        "A",
        False,
        mapping_overrides={"x": [3, 4], 1: {"serial": 2}},
        **common,
    )
    changed = _combo_cache_key(
        source,
        "A",
        False,
        mapping_overrides={"x": [3, 5], 1: {"serial": 2}},
        **common,
    )

    assert first == reordered
    assert first != changed


def test_idempotent_pdb_alias_does_not_churn_registry_epoch():
    from cycpep_master.paths import _map_utils as map_utils

    with map_utils.isolated_monomer_registry():
        start = map_utils.registry_epoch()
        map_utils.register_pdb_alias("ZQX", "A")
        changed = map_utils.registry_epoch()
        map_utils.register_pdb_alias("ZQX", "A")
        unchanged = map_utils.registry_epoch()

        assert changed == start + 1
        assert unchanged == changed
