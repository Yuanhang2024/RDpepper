"""Regression tests for the additive fail-closed remediation API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cycpep_master.chemical_audit import (
    audit_biln,
    audit_helm,
    audit_map,
    audit_pdb_file,
    audit_pdb_text,
    audit_smiles,
)
from cycpep_master.remediation_v3 import (
    convert_payload_fail_closed,
    generate_f_strict,
    generate_h_strict,
    reconstruct_pdb_fail_closed,
)


V2_INPUTS = Path(__file__).resolve().parent / "fixtures" / "v2_inputs"
PAYLOADS = V2_INPUTS / "gold" / "payloads"
COORDINATE_CASES = V2_INPUTS / "gold" / "coordinate_gold_entities_v2.jsonl"
COORDINATE_PAYLOADS = V2_INPUTS / "source_snapshots" / "coordinate_gold"


def _payload(case_id: str) -> tuple[str, str]:
    matches = list(PAYLOADS.glob(f"{case_id}.*"))
    assert len(matches) == 1
    path = matches[0]
    return path.suffix.lstrip("."), path.read_text(encoding="utf-8")


def test_unpaired_biln_ring_closure_is_rejected():
    result = audit_biln("C(11,3)-A-G-V")
    assert not result.accepted
    assert "UNPAIRED_RING_CLOSURE" in result.warning_codes
    converted = convert_payload_fail_closed("biln", "C(11,3)-A-G-V")
    assert converted.status == "rejected"


def test_incomplete_map_and_helm_connections_are_rejected():
    map_result = audit_map("ACGV{cyc:1:R1-}")
    helm_result = audit_helm(
        "PEPTIDE1{A.C.G.V}$PEPTIDE1,PEPTIDE1,1:R1-$$$V2.0"
    )
    assert not map_result.accepted
    assert not helm_result.accepted
    assert "MALFORMED_RING_CLOSURE" in map_result.warning_codes
    assert "MALFORMED_RING_CLOSURE" in helm_result.warning_codes


@pytest.mark.parametrize(
    "case_id",
    ["FCV2-INV-STE-001", "FCV2-INV-STE-002", "FCV2-INV-STE-003"],
)
def test_duplicate_atom_map_stereo_conflicts_are_rejected(case_id):
    _, payload = _payload(case_id)
    audit = audit_smiles(payload)
    assert not audit.accepted
    assert "DUPLICATE_ATOM_MAP" in audit.warning_codes
    assert "CONFLICTING_STEREO_DECLARATION" in audit.warning_codes
    assert "MULTI_COMPONENT_CYCLIC_PEPTIDE" in audit.warning_codes
    assert convert_payload_fail_closed("smiles", payload).status == "rejected"


@pytest.mark.parametrize(
    "case_id",
    ["FCV2-INV-PDB-001", "FCV2-INV-PDB-003"],
)
def test_link_conect_partner_conflicts_are_rejected(case_id):
    path = next(PAYLOADS.glob(f"{case_id}.pdb"))
    audit = audit_pdb_file(path)
    assert not audit.accepted
    assert "PDB_LINK_CONECT_CONFLICT" in audit.warning_codes
    result = reconstruct_pdb_fail_closed(path, "A")
    assert result.status == "rejected"
    assert "PDB_LINK_CONECT_CONFLICT" in result.warning_codes


def test_valid_payload_controls_remain_accepted():
    biln = convert_payload_fail_closed("biln", "C(1,3)-A-A-A-C(1,3)")
    smiles = convert_payload_fail_closed("smiles", "N[C@@H](C)C(=O)O")
    assert biln.status == "success"
    assert smiles.status == "success"
    assert biln.warning_codes == []
    assert smiles.warning_codes == []


def test_pdb_audit_rejects_input_without_valid_atoms():
    audit = audit_pdb_text("garbage\nCONECT nope\nATOM bad\n")

    assert not audit.accepted
    assert "MALFORMED_PDB_ATOM_RECORD" in audit.warning_codes
    assert "NO_VALID_PDB_ATOMS" in audit.warning_codes


def test_unknown_monomer_warning_cannot_be_success():
    result = convert_payload_fail_closed("map", "A{nnr:DefinitelyUnknown}G")
    assert result.status == "rejected"
    assert "CHEMICAL_WARNING_PRESENT" in result.warning_codes


def _coordinate_cases():
    if not COORDINATE_CASES.exists():
        return []
    return [
        json.loads(line)
        for line in COORDINATE_CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    "case",
    _coordinate_cases(),
    ids=lambda row: row["case_id"],
)
def test_remediation_consensus_rejects_fh_candidate_conflict(case):
    path = COORDINATE_PAYLOADS / Path(case["input_relpath"]).name
    result = reconstruct_pdb_fail_closed(path, case["chain_id"])
    assert result.status == "rejected"
    assert result.output_smiles is None
    assert result.output_inchikey is None
    assert result.warning_codes == ["MULTI_PATH_FULL_INCHIKEY_CONFLICT"]
    by_route = {row["route"]: row for row in result.route_results}
    assert by_route["f"]["status"] == "rejected"
    assert by_route["h"]["status"] == "rejected"
    assert by_route["f"]["candidate_identity"]["output_inchikey"]
    assert by_route["h"]["candidate_identity"]["output_inchikey"]
    assert (
        by_route["f"]["candidate_identity"]["output_inchikey"]
        != by_route["h"]["candidate_identity"]["output_inchikey"]
    )
    assert result.path_used == "V4_AUDITED_CONSENSUS"


def test_strict_fh_do_not_emit_uncorroborated_full_identity():
    cases = _coordinate_cases()
    if not cases:
        pytest.skip("v2 coordinate remediation fixture is unavailable")
    case = cases[0]
    path = COORDINATE_PAYLOADS / Path(case["input_relpath"]).name
    f_output, f_error = generate_f_strict(path, case["chain_id"])
    h_output, h_error = generate_h_strict(path, case["chain_id"])
    assert f_output is None and "disagrees" in f_error
    assert h_output is None and "disagrees" in h_error
