"""Focused safety tests for result-first degraded coordinate output."""

from __future__ import annotations

from dataclasses import asdict

from cycpep_master import result_first
from cycpep_master.core.rigor import rigor_from_result
from cycpep_master.core.structure_io import PreparedCoordinateInput
from cycpep_master.remediation_v5 import StrictReconstructionResult


GLU_MISSING_OE2 = """\
HEADER    GLU SIDECHAIN TRUNCATION
ATOM      1  N   GLU L   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  GLU L   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   GLU L   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   GLU L   1       4.000   2.000   3.000  1.00  0.00           O
ATOM      5  CB  GLU L   1       2.000   3.000   3.000  1.00  0.00           C
ATOM      6  CG  GLU L   1       2.000   4.000   3.000  1.00  0.00           C
ATOM      7  CD  GLU L   1       2.000   5.000   3.000  1.00  0.00           C
ATOM      8  OE1 GLU L   1       2.000   6.000   3.000  1.00  0.00           O
CONECT    1    2
CONECT    2    3
CONECT    3    4
CONECT    2    5
CONECT    5    6
CONECT    6    7
CONECT    7    8
END
"""

GLU_MISSING_BACKBONE = GLU_MISSING_OE2.replace(
    "ATOM      4  O   GLU L   1       4.000   2.000   3.000  1.00  0.00           O\n",
    "",
).replace("CONECT    3    4\n", "")

UNKNOWN_RESIDUE = GLU_MISSING_OE2.replace("GLU", "UNK")


# This audit bit is present on normalized production inputs.  It deliberately
# is absent from the ordinary test adapter below, preserving legacy fixtures.
AUDIT = {"projection_applied": True}


def _strict(*, status="rejected", qualified_success=False, rows=None):
    return StrictReconstructionResult(
        status=status,
        support_status="qualified" if status == "success" else "supported",
        output_smiles=None,
        output_inchikey=None,
        rejection_reason="fixture rejection",
        warning_codes=[],
        path_used="V6_EVIDENCE_DIMENSION_AUDIT",
        route_results=list(rows or []),
        output_evidence={},
        repair_codes=[],
        qualified_success=qualified_success,
    )


def _prepared(tmp_path, text, *, audit=None):
    path = tmp_path / "input.pdb"
    path.write_text(text, encoding="ascii")
    return PreparedCoordinateInput(
        pdb_path=path,
        chain_id="L",
        source_format="pdb",
        audit=dict(AUDIT if audit is None else audit),
    )


def _run(monkeypatch, prepared, strict=None):
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_prepared_structure_fail_closed_v6",
        lambda *args, **kwargs: strict or _strict(),
    )
    return result_first.reconstruct_prepared_structure(prepared)


def test_missing_sidechain_backbone_complete_is_degraded_partial(
    tmp_path, monkeypatch
):
    result = _run(monkeypatch, _prepared(tmp_path, GLU_MISSING_OE2))

    assert result.status == "success"
    assert result.quality == result_first.QUALITY_PARTIAL
    assert result.source == result_first.SOURCE_DEGRADED_TEMPLATE
    assert "DEGRADED_MISSING_SIDECHAIN" in result.warning_codes
    assert "DEGRADED_TEMPLATE_PROJECTION" in result.warning_codes
    data_quality = result.provenance["degraded"]["data_quality"]
    assert data_quality["entry_class"] == "degraded_sidechain"
    assert data_quality["residues"][0]["missing_template_atom_names"] == ["OE2"]
    assert result.smiles is None
    assert rigor_from_result(result).label == "L1:R"


def test_backbone_missing_or_unknown_identity_never_uses_template_partial(
    tmp_path, monkeypatch
):
    for text in (GLU_MISSING_BACKBONE, UNKNOWN_RESIDUE):
        result = _run(monkeypatch, _prepared(tmp_path, text))
        assert result.status == "success"
        assert result.source != result_first.SOURCE_DEGRADED_TEMPLATE
        assert result.quality in {
            result_first.QUALITY_TOPOLOGY,
            result_first.QUALITY_PARTIAL,
            result_first.QUALITY_RAW,
        }
        assert "DEGRADED_TEMPLATE_PROJECTION" not in result.warning_codes
        assert rigor_from_result(result).label in {"L1:H", "L1:R", "L0:C"}


def test_missing_atom_has_no_placeholder_or_chemical_claim(tmp_path, monkeypatch):
    result = _run(monkeypatch, _prepared(tmp_path, GLU_MISSING_OE2))

    assert result.source == result_first.SOURCE_DEGRADED_TEMPLATE
    assert all(atom["name"] != "OE2" for atom in result.graph["atoms"])
    assert all(
        "xyz" in atom and atom["xyz"] is not None
        for atom in result.graph["atoms"]
    )
    assert all(bond["a"] != 999 and bond["b"] != 999 for bond in result.graph["bonds"])
    assert all(bond["order"] is None for bond in result.graph["bonds"])
    projection = result.provenance["degraded"]["projection"]
    assert projection["missing_atoms"][0]["atom_names"] == ["OE2"]
    assert "formal_charge" not in result.graph
    assert "stereochemistry" not in result.graph
    assert result.smiles is None


def test_degraded_input_blocks_candidate_and_registry_promotion(
    tmp_path, monkeypatch
):
    strict = _strict(
        rows=[
            {
                "route": "a",
                "status": "success",
                "output_smiles": "NCC(=O)O",
                "output_inchikey": "candidate-is-diagnostic-only",
            }
        ]
    )
    monkeypatch.setattr(
        result_first,
        "_attempt_degraded_template_projection",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        result_first,
        "_attempt_registry_template",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("registry promotion must be gated")
        ),
    )
    monkeypatch.setattr(
        result_first,
        "_assessment_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("candidate promotion must be gated")
        ),
    )
    result = _run(monkeypatch, _prepared(tmp_path, GLU_MISSING_OE2), strict)

    assert result.quality == result_first.QUALITY_PARTIAL
    assert result.quality not in {
        result_first.QUALITY_EXACT,
        result_first.QUALITY_HIGH,
        result_first.QUALITY_MEDIUM,
    }
    assert result.provenance["degraded"]["candidate_registry_policy"] in {
        "blocked_before_promotion",
        "blocked",
    }
    assert rigor_from_result(result).label == "L1:R"


def test_clean_input_keeps_previous_ladder_behavior(tmp_path, monkeypatch):
    # A caller-owned PreparedCoordinateInput without normalization provenance is
    # intentionally outside the new quality audit gate.
    result = _run(
        monkeypatch,
        _prepared(tmp_path, GLU_MISSING_OE2, audit={}),
    )

    assert result.source == result_first.SOURCE_PARTIAL
    assert result.quality == result_first.QUALITY_PARTIAL
    assert "DEGRADED_MISSING_SIDECHAIN" not in result.warning_codes
    assert "degraded" not in result.provenance


def test_strict_exact_still_short_circuits_degraded_audit(tmp_path, monkeypatch):
    strict = _strict(status="success", qualified_success=True)
    strict.output_smiles = "NCC(=O)O"
    result = _run(monkeypatch, _prepared(tmp_path, GLU_MISSING_OE2), strict)

    assert result.quality == result_first.QUALITY_EXACT
    assert result.source == result_first.SOURCE_EXACT
    assert "degraded" not in result.provenance
    assert asdict(result.strict_result) == asdict(strict)
