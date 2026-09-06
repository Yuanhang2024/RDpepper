"""Content-addressed integration regression against the RCSB 9R9M structure."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cycpep_master.core.structure_io import prepare_coordinate_input
from cycpep_master.paths.path_a import generate_with_evidence
from cycpep_master.remediation_v6 import reconstruct_structure_fail_closed_v6


_REPO = Path(__file__).resolve().parents[2]
_SHA256 = "dfaa04345333bdd5008d4e80a8ecbace920ff7c179b8f3c2967a3355db6732e1"
_RCSB_9R9M = (
    _REPO / "data" / "external" / "rcsb_official_mmcif_v1" / "9R9M"
    / f"9R9M.{_SHA256.upper()}.cif"
)


def _official_9r9m() -> Path:
    if not _RCSB_9R9M.is_file():
        pytest.skip("content-addressed RCSB 9R9M external fixture is unavailable")
    observed = hashlib.sha256(_RCSB_9R9M.read_bytes()).hexdigest()
    assert observed == _SHA256
    return _RCSB_9R9M


@pytest.mark.parametrize("geometric_cyclization", [False, True])
def test_real_orn_maps_uniquely_through_a_family(geometric_cyclization):
    source = _official_9r9m()
    with prepare_coordinate_input(source, "I") as prepared:
        smiles, error, evidence = generate_with_evidence(
            str(prepared.pdb_path),
            prepared.chain_id,
            geometric_cyclization=geometric_cyclization,
        )
    assert error is None
    assert smiles
    orn = [
        row for row in evidence["residue_evidence"]
        if row["pdb_resname"] == "ORN"
    ]
    assert len(orn) == 1
    assert orn[0]["unified_symbol"] == "Orn"
    assert orn[0]["unified_source"] == "CycPeptMPDB"
    assert orn[0]["observed_heavy_atom_count"] == 8
    assert orn[0]["mapped_heavy_atom_count"] == 8
    assert orn[0]["mapped_template_heavy_atom_count"] == 8
    assert orn[0]["mapping_candidate_count"] == 1
    assert orn[0]["mapping_complete"] is True
    assert orn[0]["mapping_injective"] is True
    assert orn[0]["mapping_unique"] is True
    assert orn[0]["external_attachment_mapping_unique"] is True
    assert orn[0]["r3_anchor_template_atom_index"] == 7


def test_real_orn_mmcif_preserves_f_h_identity_veto():
    source = _official_9r9m()
    result = reconstruct_structure_fail_closed_v6(source, "I")

    assert result.status == "rejected"
    assert result.qualified_success is False
    assert result.warning_codes == ["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"]
    dimension = result.output_evidence["evidence_dimensions"][
        "diagnostic_identity_consistency"
    ]
    assert dimension["passed"] is False
    assert dimension["selected_full_inchikey"] == (
        "XIAHWYCVVWWASA-KLWONIAYSA-O"
    )
    assert len(dimension["disagreeing_full_inchikeys"]) == 2
    agreeing_routes = {
        row["route"]
        for row in result.route_results
        if row.get("status") == "success"
        and row.get("output_inchikey") == dimension[
            "selected_full_inchikey"
        ]
    }
    assert agreeing_routes == {"a", "e", "g"}
    assert result.input_evidence["coordinate_input"]["source_sha256"] == _SHA256
    assert result.input_evidence["coordinate_input"][
        "materialized_explicit_connection_count"
    ] == 1
