"""Pre-target specifications for the strict v4 fail-closed APIs.

These tests are committed as executable specifications but are intentionally
not run while the confirmatory target-execution barrier remains closed.
"""

from __future__ import annotations

import copy
import json

import pytest

from cycpep_master.chemical_audit import AuditResult, audit_assembly_json
from cycpep_master import paths
import cycpep_master.remediation_v3 as remediation
from cycpep_master.remediation_v3 import (
    convert_assembly_payload_fail_closed,
    reconstruct_pdb_fail_closed,
)


PORT_VALENCE = {
    "R1": (3, 3),
    "R2": (4, 4),
    "R3": (2, 2),
}


def _port(atom_map: int, name: str, cap: str | None, stereo: str) -> dict:
    declared, maximum = PORT_VALENCE[name]
    return {
        "atom_map": atom_map,
        "cap": cap,
        "stereo": stereo,
        "declared_valence": declared,
        "maximum_valence": maximum,
    }


def _valid_assembly() -> dict:
    symbols = ("A", "C", "S", "T")
    monomers = []
    for position, symbol in enumerate(symbols, start=1):
        stereo = "achiral" if symbol == "G" else "L"
        ports = {
            "R1": _port(position * 10 + 1, "R1", None, stereo),
            "R2": _port(position * 10 + 2, "R2", None, stereo),
        }
        if symbol in {"C", "S", "T"}:
            ports["R3"] = _port(position * 10 + 3, "R3", "H", stereo)
        monomers.append(
            {
                "monomer_id": f"m{position}",
                "symbol": symbol,
                "chain_id": "A",
                "formal_charge": 0,
                "ports": ports,
            }
        )
    connections = []
    for position in range(1, len(symbols)):
        connections.append(
            {
                "connection_id": f"c{position}",
                "source": {
                    "monomer_id": f"m{position}",
                    "port": "R2",
                },
                "target": {
                    "monomer_id": f"m{position + 1}",
                    "port": "R1",
                },
                "bond_type": "amide",
                "bond_order": 1,
                "role": "backbone",
            }
        )
    connections.append(
        {
            "connection_id": "c4",
            "source": {"monomer_id": "m4", "port": "R2"},
            "target": {"monomer_id": "m1", "port": "R1"},
            "bond_type": "amide",
            "bond_order": 1,
            "role": "ring",
        }
    )
    return {
        "schema_version": "1.0.0",
        "assembly_mode": "strict_cyclic_peptide",
        "component_policy": "single_covalent_entity",
        "declared_formal_charge": 0,
        "monomers": monomers,
        "connections": connections,
        "geometric_bond_candidates": [],
    }


def _payload(document: dict) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _invalid_assembly(invalid_class: str) -> dict:
    value = copy.deepcopy(_valid_assembly())
    monomers = value["monomers"]
    connections = value["connections"]
    if invalid_class == "UNKNOWN_MONOMER":
        monomers[0]["symbol"] = "UX001"
    elif invalid_class == "MISSING_R1":
        monomers[0]["ports"].pop("R1")
    elif invalid_class == "MISSING_R2":
        monomers[1]["ports"].pop("R2")
    elif invalid_class == "MISSING_R3":
        monomers[1]["ports"].pop("R3")
    elif invalid_class == "PORT_REUSE":
        connections.append(
            {
                "connection_id": "cx",
                "source": {"monomer_id": "m1", "port": "R2"},
                "target": {"monomer_id": "m3", "port": "R1"},
                "bond_type": "amide",
                "bond_order": 1,
                "role": "backbone",
            }
        )
    elif invalid_class == "ENDPOINT_OUT_OF_RANGE":
        connections[0]["target"]["monomer_id"] = "m999"
    elif invalid_class == "SELF_CONNECTION":
        connections[0] = {
            "connection_id": "c1",
            "source": {"monomer_id": "m2", "port": "R3"},
            "target": {"monomer_id": "m2", "port": "R3"},
            "bond_type": "disulfide",
            "bond_order": 1,
            "role": "ring",
        }
        monomers[1]["ports"]["R3"]["cap"] = None
    elif invalid_class == "INCOMPATIBLE_BOND_TYPE":
        connections[0]["bond_type"] = "disulfide"
    elif invalid_class == "CAP_INTERNAL_USE":
        monomers[0]["ports"]["R2"]["cap"] = "OH"
    elif invalid_class == "MULTIPLE_AMBIGUOUS_GEOMETRIC_BONDS":
        value["geometric_bond_candidates"] = [
            {
                "candidate_id": "g1",
                "source": {"monomer_id": "m2", "port": "R3"},
                "target": {"monomer_id": "m3", "port": "R3"},
                "distance_angstrom": 1.85,
                "confidence": 0.5,
            },
            {
                "candidate_id": "g2",
                "source": {"monomer_id": "m2", "port": "R3"},
                "target": {"monomer_id": "m4", "port": "R3"},
                "distance_angstrom": 1.85,
                "confidence": 0.5,
            },
        ]
    elif invalid_class == "FORMAL_CHARGE_CONFLICT":
        value["declared_formal_charge"] = 1
    elif invalid_class == "ATOM_MAPPING_MISSING":
        monomers[1]["ports"]["R3"]["atom_map"] = None
    elif invalid_class == "MULTICHAIN_WRONG_LINK":
        monomers[2]["chain_id"] = "B"
        monomers[3]["chain_id"] = "B"
    elif invalid_class == "UNSUPPORTED_STEREO_DECLARATION":
        monomers[0]["ports"]["R1"]["stereo"] = "D"
    elif invalid_class == "DECLARED_VALENCE_UNSUPPORTED":
        monomers[0]["ports"]["R1"]["declared_valence"] = 4
        monomers[0]["ports"]["R1"]["maximum_valence"] = 4
    elif invalid_class == "MONOMER_FORMAL_CHARGE_UNSUPPORTED":
        monomers[0]["formal_charge"] = 1
        monomers[1]["formal_charge"] = -1
    elif invalid_class == "UNUSED_PORT_CAP_UNSUPPORTED":
        monomers[1]["ports"]["R3"]["cap"] = "OH"
    elif invalid_class == "CONNECTION_ROLE_CONFLICT":
        connections[-1]["role"] = "interchain"
    else:
        raise AssertionError(f"unknown test invalid class {invalid_class}")
    return value


@pytest.mark.parametrize(
    "invalid_class",
    [
        "UNKNOWN_MONOMER",
        "MISSING_R1",
        "MISSING_R2",
        "MISSING_R3",
        "PORT_REUSE",
        "ENDPOINT_OUT_OF_RANGE",
        "SELF_CONNECTION",
        "INCOMPATIBLE_BOND_TYPE",
        "CAP_INTERNAL_USE",
        "MULTIPLE_AMBIGUOUS_GEOMETRIC_BONDS",
        "FORMAL_CHARGE_CONFLICT",
        "ATOM_MAPPING_MISSING",
        "MULTICHAIN_WRONG_LINK",
        "UNSUPPORTED_STEREO_DECLARATION",
        "DECLARED_VALENCE_UNSUPPORTED",
        "MONOMER_FORMAL_CHARGE_UNSUPPORTED",
        "UNUSED_PORT_CAP_UNSUPPORTED",
        "CONNECTION_ROLE_CONFLICT",
    ],
)
def test_strict_assembly_rejects_each_frozen_structured_class(invalid_class):
    payload = _payload(_invalid_assembly(invalid_class))
    audit = audit_assembly_json(payload)
    assert not audit.accepted
    assert invalid_class in audit.warning_codes

    result = convert_assembly_payload_fail_closed(payload)
    assert result.status == "rejected"
    assert result.output_smiles is None
    assert result.output_inchikey is None
    assert invalid_class in result.warning_codes


def test_standard_amino_acid_head_to_tail_assembly_explicitly_succeeds():
    payload = _payload(_valid_assembly())
    audit = audit_assembly_json(payload)
    assert audit.accepted
    assert audit.warning_codes == []

    result = convert_assembly_payload_fail_closed(payload)
    assert result.status == "success"
    assert result.output_smiles
    assert result.output_inchikey
    assert result.warning_codes == []
    assert result.path_used == "STRICT_ASSEMBLY_JSON:EXPLICIT_RGROUP_GRAPH"
    assert result.route_results[0]["route"] == "explicit_rgroup_assembly"
    assert result.route_results[0]["status"] == "success"
    assert result.output_smiles == (
        "C[C@@H]1NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CO)NC(=O)"
        "[C@H](CS)NC1=O"
    )
    assert result.output_inchikey == "SNMDEHHSDMLMAO-BGKGJTHRSA-N"


def test_strict_assembly_rejects_declared_charge_not_realized_in_output():
    document = _valid_assembly()
    for monomer in document["monomers"]:
        monomer["formal_charge"] = 1
    document["declared_formal_charge"] = len(document["monomers"])

    audit = audit_assembly_json(_payload(document))
    assert not audit.accepted
    assert "MONOMER_FORMAL_CHARGE_UNSUPPORTED" in audit.warning_codes

    result = convert_assembly_payload_fail_closed(_payload(document))
    assert result.status == "rejected"
    assert result.output_smiles is None
    assert "MONOMER_FORMAL_CHARGE_UNSUPPORTED" in result.warning_codes


def _install_route_stubs(
    monkeypatch,
    outputs: dict[str, str | None],
    *,
    f_candidate: str | None = None,
    h_candidate: str | None = None,
) -> None:
    monkeypatch.setattr(remediation, "audit_pdb_file", lambda _: AuditResult())

    def route_stub(route: str):
        def generate(_path: str, _chain: str):
            output = outputs.get(route)
            return output, None if output else f"Path {route.upper()} unavailable"

        return generate

    for route in ("a", "b", "c", "e", "g"):
        monkeypatch.setattr(paths, f"generate_{route}", route_stub(route))

    def geometric_stub(
        route: str, _path: str, _chain: str, **_kwargs
    ):
        candidate = f_candidate if route == "f" else h_candidate
        strict_output = outputs.get(route)
        return (
            strict_output,
            None if strict_output else f"Path {route.upper()} uncorroborated",
            {
                "candidate_identity": remediation._identity(candidate),
                "corroborator_identity": remediation._identity(
                    outputs.get("g")
                ),
                "corroborator_route": "g",
            },
        )

    monkeypatch.setattr(
        remediation,
        "_strict_geometric_route_detailed",
        geometric_stub,
    )


def test_v4_request_executes_shared_g_producer_once(
    monkeypatch, tmp_path
):
    from cycpep_master.paths import path_f, path_g, path_h

    monkeypatch.setattr(remediation, "audit_pdb_file", lambda _: AuditResult())
    for route in ("a", "b", "c", "e"):
        monkeypatch.setattr(
            paths, f"generate_{route}", lambda *_a, **_k: ("CC", None)
        )
    monkeypatch.setattr(path_f, "generate_f", lambda *_a, **_k: ("CC", None))
    monkeypatch.setattr(path_h, "generate_h", lambda *_a, **_k: ("CC", None))
    calls = []

    def shared_g(*_args, **_kwargs):
        calls.append("g")
        return "CC", None, {
            "route": "g",
            "allow_geometric_inference": True,
            "helm": "PEPTIDE1{A}$$$$",
            "map_payload": "A",
            "output_smiles": "CC",
        }

    monkeypatch.setattr(path_g, "generate_g_with_artifacts", shared_g)

    result = reconstruct_pdb_fail_closed(tmp_path / "fixture.pdb", "A")

    assert result.status == "success"
    assert calls == ["g"]
    assert [row["route"] for row in result.route_results] == list("abcefgh")


@pytest.mark.parametrize("success_count", [0, 1])
def test_zero_or_one_audited_success_is_insufficient(
    monkeypatch,
    tmp_path,
    success_count,
):
    outputs = {"a": "CC"} if success_count else {}
    _install_route_stubs(monkeypatch, outputs)

    result = reconstruct_pdb_fail_closed(tmp_path / "opaque.pdb", "A")
    assert result.status == "rejected"
    assert result.output_smiles is None
    assert result.output_inchikey is None
    assert result.warning_codes == ["INSUFFICIENT_CROSS_VALIDATION"]
    assert len(result.route_results) == 7


@pytest.mark.parametrize("routes", [("a", "c"), ("a", "e"), ("b", "g")])
def test_two_matching_routes_from_one_family_are_insufficient(
    monkeypatch, tmp_path, routes
):
    _install_route_stubs(
        monkeypatch, {route: "CC" for route in routes}
    )

    result = reconstruct_pdb_fail_closed(tmp_path / "opaque.pdb", "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["INSUFFICIENT_CROSS_VALIDATION"]
    assert len(result.route_results) == 7


def test_two_matching_audited_routes_succeed_and_preserve_route_evidence(
    monkeypatch,
    tmp_path,
):
    _install_route_stubs(monkeypatch, {"a": "CC", "b": "CC"})

    result = reconstruct_pdb_fail_closed(tmp_path / "opaque.pdb", "A")
    assert result.status == "success"
    assert result.path_used == "V4_CONSENSUS:A,B"
    assert result.output_inchikey
    assert len(result.route_results) == 7
    for row in result.route_results:
        assert set(
            (
                "status",
                "runtime_sec",
                "output_smiles",
                "output_inchikey",
                "candidate_identity",
                "corroborator_identity",
                "error",
                "warnings",
                "warning_codes",
            )
        ).issubset(row)
        assert row["runtime_sec"] >= 0.0


def test_distinct_successful_full_inchikeys_are_never_tie_broken(
    monkeypatch,
    tmp_path,
):
    _install_route_stubs(monkeypatch, {"a": "CC", "b": "CCC"})

    result = reconstruct_pdb_fail_closed(tmp_path / "opaque.pdb", "A")
    assert result.status == "rejected"
    assert result.output_smiles is None
    assert result.output_inchikey is None
    assert result.warning_codes == ["MULTI_PATH_FULL_INCHIKEY_CONFLICT"]


@pytest.mark.parametrize(
    ("f_candidate", "h_candidate"),
    [("CCC", None), ("CCC", "CCC")],
)
def test_f_h_candidate_disagreement_with_selected_identity_is_veto(
    monkeypatch, tmp_path, f_candidate, h_candidate
):
    _install_route_stubs(
        monkeypatch,
        {"a": "CC", "b": "CC"},
        f_candidate=f_candidate,
        h_candidate=h_candidate,
    )

    result = reconstruct_pdb_fail_closed(tmp_path / "opaque.pdb", "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["MULTI_PATH_FULL_INCHIKEY_CONFLICT"]


def test_f_h_candidate_conflict_cannot_be_tie_broken_by_other_routes(
    monkeypatch,
    tmp_path,
):
    _install_route_stubs(
        monkeypatch,
        {"a": "CC", "b": "CC"},
        f_candidate="CCC",
        h_candidate="CCCC",
    )

    result = reconstruct_pdb_fail_closed(tmp_path / "opaque.pdb", "A")
    assert result.status == "rejected"
    assert result.output_smiles is None
    assert result.output_inchikey is None
    assert result.warning_codes == ["MULTI_PATH_FULL_INCHIKEY_CONFLICT"]
    by_route = {row["route"]: row for row in result.route_results}
    assert by_route["f"]["candidate_identity"]["output_inchikey"]
    assert by_route["h"]["candidate_identity"]["output_inchikey"]
    assert (
        by_route["f"]["candidate_identity"]["output_inchikey"]
        != by_route["h"]["candidate_identity"]["output_inchikey"]
    )
