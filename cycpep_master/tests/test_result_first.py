"""Tests for the result-first coordinate graph recovery facade."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from rdkit import Chem

from cycpep_master import application, result_first
from cycpep_master.core.structure_io import PreparedCoordinateInput
from cycpep_master.remediation_v5 import StrictReconstructionResult
from cycpep_master.remediation_v6 import _candidate_assessment


def _atom_line(serial, name, x, y, z):
    line = (
        f"ATOM  {serial:5d} {name:<4s} LIG L   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
    )
    line = line.ljust(76)
    return line + " C"


def ring_pdb(n, bond=1.5, conect=True, reverse=False):
    """A real regular ``n``-member ring of carbons with CONECT edges."""
    lines = ["HEADER    RING"]
    radius = bond / (2 * math.sin(math.pi / n))
    for index in range(n):
        angle = 2 * math.pi * index / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        lines.append(_atom_line(index + 1, f"C{index + 1}", x, y, 0.0))
    if conect:
        for index in range(n):
            partner = (index + 1) % n
            left = index + 1
            right = partner + 1
            if reverse and index % 2 == 0:
                left, right = right, left
            lines.append(f"CONECT{left:5d}{right:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def peptide_ring_pdb(residue_count=3, bond=1.5):
    """Minimal N-CA-C macrocycle with peptide identity evidence."""
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


SIMPLE_PDB = """HEADER    SIMPLE
ATOM      1  N   ALA L   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA L   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA L   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA L   1       4.000   2.000   3.000  1.00  0.00           O
CONECT    1    2
CONECT    2    3
END
"""
NO_CONECT_PDB = SIMPLE_PDB.replace(
    "CONECT    1    2\nCONECT    2    3\n", ""
)
NO_BONDS_PDB = """HEADER    NOBONDS
ATOM      1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  LIG L   1      10.000   0.000   0.000  1.00  0.00           C
ATOM      3  C3  LIG L   1      20.000   0.000   0.000  1.00  0.00           C
ATOM      4  C4  LIG L   1      30.000   0.000   0.000  1.00  0.00           C
END
"""
EMPTY_PDB = "HEADER    EMPTY\nEND\n"
FRAGMENTED_PDB = """HEADER    FRAG
ATOM      1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  LIG L   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C3  LIG L   1       3.000   0.000   0.000  1.00  0.00           C
ATOM      4  C4  LIG L   1       5.000   0.000   0.000  1.00  0.00           C
ATOM      5  C5  LIG L   1       6.500   0.000   0.000  1.00  0.00           C
ATOM      6  C6  LIG L   1       8.000   0.000   0.000  1.00  0.00           C
CONECT    1    2
CONECT    2    3
CONECT    4    5
CONECT    5    6
END
"""
DUP_SERIAL_PDB = SIMPLE_PDB.replace(
    "ATOM      2  CA  ALA", "ATOM      1  CA  ALA"
)
SELF_CONECT_PDB = SIMPLE_PDB.replace(
    "CONECT    2    3", "CONECT    2    2"
)
DANGLING_CONECT_PDB = SIMPLE_PDB.replace(
    "CONECT    2    3", "CONECT    2    9"
)
MALFORMED_ATOM_PDB = SIMPLE_PDB.replace(
    "ATOM      2  CA  ALA L   1       2.000   2.000   3.000",
    "ATOM      2  CA  ALA L   1       BADBAD  BAD    3.000",
)
MULTI_MODEL_PDB = """HEADER    MULTI
MODEL        1
ATOM      1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  LIG L   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C3  LIG L   1       3.000   0.000   0.000  1.00  0.00           C
ATOM      4  C4  LIG L   1       4.500   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  LIG L   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C3  LIG L   1       3.000   0.000   0.000  1.00  0.00           C
ATOM      4  C4  LIG L   1       4.500   0.000   0.000  1.00  0.00           C
CONECT    1    2
CONECT    1    4
CONECT    2    4
ENDMDL
END
"""


def _strict(
    status="rejected",
    *,
    smiles=None,
    rows=None,
    warning_codes=None,
    repair_codes=None,
    qualified_success=False,
    assessment=None,
    output_evidence=None,
):
    inchikey = None
    if smiles:
        molecule = Chem.MolFromSmiles(smiles)
        inchikey = Chem.MolToInchiKey(molecule) if molecule else None
    evidence = dict(output_evidence or {})
    if assessment is not None:
        evidence["candidate_assessment"] = assessment
    return StrictReconstructionResult(
        status=status,
        support_status="qualified" if status == "success" else "supported",
        output_smiles=smiles,
        output_inchikey=inchikey,
        rejection_reason=None if status == "success" else "fixture rejection",
        warning_codes=list(warning_codes or []),
        path_used="V6_EVIDENCE_DIMENSION_AUDIT",
        route_results=list(rows or []),
        output_evidence=evidence,
        repair_codes=list(repair_codes or []),
        qualified_success=bool(qualified_success),
    )


def _candidate_row(route, smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "route": route,
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": Chem.MolToInchiKey(molecule),
    }


def _candidate_row_with_wrong_declared_key(route, smiles):
    return {
        "route": route,
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": "X" * 27,
    }


def _assessment(
    rows,
    *,
    status="rejected",
    qualified_success=False,
    support_status="supported",
    output_smiles=None,
    repair_codes=None,
):
    output_inchikey = None
    if output_smiles:
        molecule = Chem.MolFromSmiles(output_smiles)
        assert molecule is not None
        output_inchikey = Chem.MolToInchiKey(molecule)
    return _candidate_assessment(
        rows,
        {},
        status=status,
        qualified_success=qualified_success,
        support_status=support_status,
        repair_codes=list(repair_codes or []),
        output_smiles=output_smiles,
        output_inchikey=output_inchikey,
    )


def _install_strict(monkeypatch, strict):
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_prepared_structure_fail_closed_v6",
        lambda *args, **kwargs: strict,
    )


def _prepared(tmp_path, text=SIMPLE_PDB, name="input.pdb", audit=None):
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return PreparedCoordinateInput(
        pdb_path=path,
        chain_id="L",
        source_format="pdb",
        audit=dict(audit or {}),
    )


def _kill_rdkit(monkeypatch):
    monkeypatch.setattr(result_first, "_proximity_mol", lambda path: None)
    monkeypatch.setattr(result_first, "_connectivity_mol", lambda path: None)
    monkeypatch.setattr(
        result_first, "_explicit_only_mol", lambda path: (None, {})
    )


def test_registry_template_stage_precedes_rdkit_fallback(tmp_path, monkeypatch):
    strict = _strict()
    _install_strict(monkeypatch, strict)
    prepared = _prepared(
        tmp_path,
        audit={"normalized_heavy_atom_count": 4},
    )
    monkeypatch.setattr(
        result_first,
        "_proximity_mol",
        lambda path: (_ for _ in ()).throw(
            AssertionError("proximity must follow registry stage")
        ),
    )

    result = result_first.reconstruct_prepared_structure(
        prepared,
        registry_assembler=lambda _prepared: {
            "smiles": "NCC=O",
            "route": "authoritative_template",
        },
    )

    assert result.status == "success"
    assert result.quality == "high"
    assert result.source == result_first.SOURCE_REGISTRY
    assert result.smiles == "NCC=O"
    assert result.provenance["registry_template"]["status"] == "qualified"
    assert result.provenance["registry_template"]["route"] == (
        "authoritative_template"
    )
    assert result.provenance["ladder_attempts"][0]["stage"] == (
        result_first.SOURCE_REGISTRY
    )


def test_registry_stage_respects_empty_overlay_policy(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    prepared = _prepared(
        tmp_path,
        audit={"normalized_heavy_atom_count": 2},
    )
    monkeypatch.setattr(
        result_first,
        "_overlay_policy",
        lambda required: {
            "require_empty_persistent_overlay": required,
            "status": "nonempty",
            "allowed": False,
            "audit": {"status": "nonempty"},
        },
    )
    result = result_first.reconstruct_prepared_structure(
        prepared,
        require_empty_persistent_overlay=True,
        registry_assembler=lambda _prepared: pytest.fail(
            "blocked registry adapter must not run"
        ),
    )

    assert result.status == "success"
    assert result.quality == "partial"
    attempt = result.provenance["ladder_attempts"][0]
    assert attempt["stage"] == result_first.SOURCE_REGISTRY
    assert result_first._WARNING_REGISTRY_OVERLAY_BLOCKED in attempt[
        "reason_codes"
    ]


def test_malformed_conect_retains_best_effort_raw_graph(tmp_path, monkeypatch):
    path = tmp_path / "malformed_conect.pdb"
    path.write_text(
        SIMPLE_PDB.replace(
            "CONECT    2    3",
            "CONECT    2    3  BAD!",
        ),
        encoding="ascii",
    )
    result = result_first.reconstruct_structure(path)

    assert result.status == "success"
    assert result.quality == "raw"
    assert result.source == result_first.SOURCE_RAW
    assert result.graph["atoms"]
    assert result.graph["bonds"]
    assert any(code in result.warning_codes for code in (
        "MALFORMED_CONECT_RAW_PRESERVED",
        "NORMALIZATION_ERROR_RAW_PRESERVED",
    ))
    assert result.provenance["normalization_error_recovered"] is True
    assert result.provenance["raw"]["malformed_conect_records"]


def _topology_reason_codes(result, stage="rdkit_proximity"):
    for attempt in result.provenance["ladder_attempts"]:
        if attempt.get("stage") == stage:
            return list(attempt.get("reason_codes", []))
    return []


# --------------------------------------------------------------------------
# Ladder stage 1: exact V6 short-circuit
# --------------------------------------------------------------------------


def test_exact_short_circuit_skips_all_fallbacks(tmp_path, monkeypatch):
    strict = _strict(
        status="success", smiles="NCC(=O)O", qualified_success=True
    )
    _install_strict(monkeypatch, strict)
    for name in (
        "_proximity_mol",
        "_connectivity_mol",
        "_explicit_only_mol",
        "_parse_raw_pdb_graph",
    ):
        monkeypatch.setattr(
            result_first,
            name,
            lambda path: (_ for _ in ()).throw(
                AssertionError("fallback invoked")
            ),
        )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "exact"
    assert result.source == "v6_strict_exact"
    assert result.result == "NCC(=O)O"
    assert result.smiles == "NCC(=O)O"
    assert result.ambiguous is False
    assert result.graph is None
    assert result.alternatives == []
    assert result.strict_status == "success"
    assert result.strict_result is strict
    assert result.provenance["ladder"] == "exact"
    assert len(result.provenance["ladder_attempts"]) == 1


def test_exact_surfaces_connection_reduction_without_mutating_strict(
    tmp_path, monkeypatch
):
    strict = _strict(
        status="success", smiles="NCC(=O)O", qualified_success=True
    )
    snapshot = asdict(strict)
    _install_strict(monkeypatch, strict)
    audit = {
        "source_pdb_conect_pair_count": 3,
        "first_model_pdb_conect_pair_count": 3,
        "selected_chain_pdb_conect_pair_count": 2,
        "materialized_explicit_connection_count": 2,
    }

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, audit=audit)
    )

    assert result.quality == "exact"
    assert result.warning_codes == [
        "SOURCE_CONECT_REDUCED_DURING_NORMALIZATION"
    ]
    assert asdict(strict) == snapshot


def test_strict_object_and_fields_unchanged(tmp_path, monkeypatch):
    strict = _strict(
        status="success",
        smiles="NCC(=O)O",
        qualified_success=True,
        warning_codes=["SOME_STRICT_CODE"],
    )
    snapshot = asdict(strict)
    _install_strict(monkeypatch, strict)

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.strict_result is strict
    assert strict.status == "success"
    assert strict.output_smiles == "NCC(=O)O"
    assert strict.qualified_success is True
    assert asdict(strict) == snapshot
    assert result.warning_codes == ["SOME_STRICT_CODE"]
    assert result.strict_status == "success"


def test_block_warning_prevents_exact_short_circuit(tmp_path, monkeypatch):
    strict = _strict(
        status="success",
        smiles="NCC(=O)O",
        qualified_success=True,
        warning_codes=["PDB_LINK_CONECT_CONFLICT"],
    )
    _install_strict(monkeypatch, strict)
    called = []

    def fail(name):
        def inner(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"{name} invoked")

        return inner

    for name in ("_proximity_mol", "_connectivity_mol", "_explicit_only_mol"):
        monkeypatch.setattr(result_first, name, fail(name))

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "raw"
    assert result.smiles is None
    assert result.source == "raw_pdb_graph"
    assert called == ["_proximity_mol", "_connectivity_mol", "_explicit_only_mol"]
    assert "STRICT_FALLBACK_BLOCKED" in result.warning_codes
    assert result.provenance["fallback_block"]["reason_codes"] == [
        "PDB_LINK_CONECT_CONFLICT"
    ]


def test_unqualified_strict_success_is_not_exact(tmp_path, monkeypatch):
    rows = [_candidate_row("b", "NCC(=O)O")]
    strict = _strict(
        status="success",
        smiles="NCC(=O)O",
        qualified_success=False,
        repair_codes=["TEST_REPAIR"],
        rows=rows,
        assessment=_assessment(
            rows,
            status="success",
            support_status="qualified",
            output_smiles="NCC(=O)O",
            repair_codes=["TEST_REPAIR"],
        ),
    )
    _install_strict(monkeypatch, strict)

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "high"
    assert result.source == "v6_candidate_unique"
    assert result.strict_result is strict
    assert "STRICT_SUCCESS_UNQUALIFIED" in result.warning_codes
    assert "STRICT_SUCCESS_WITHOUT_SMILES" not in result.warning_codes


# --------------------------------------------------------------------------
# Ladder stages 2-3: candidates from candidate_assessment only
# --------------------------------------------------------------------------


def test_unique_candidate_is_high(tmp_path, monkeypatch):
    rows = [_candidate_row("b", "NCC(=O)O")]
    _install_strict(
        monkeypatch,
        _strict(
            rows=rows,
            assessment=_assessment(rows),
            warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
        ),
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "high"
    assert result.source == "v6_candidate_unique"
    assert result.result == "NCC(=O)O"
    assert result.ambiguous is False
    assert result.alternatives == []
    assert "V6_STRICT_NOT_SUCCESS" in result.warning_codes
    assert "V6_INSUFFICIENT_EVIDENCE_DIMENSIONS" in result.warning_codes
    assert result.provenance["candidate_assessment"]["status"] == "valid"
    assert result.strict_status == "rejected"


def test_ensemble_medium_primary_and_alternatives(tmp_path, monkeypatch):
    rows = [
        _candidate_row("a", "CC(=O)O"),
        _candidate_row("c", "NCC(=O)O"),
        _candidate_row("e", "NCC(=O)O"),
    ]
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=_assessment(rows))
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "medium"
    assert result.source == "v6_candidate_ensemble"
    assert result.smiles is None
    assert result.ambiguous is True
    assert "MULTIPLE_CANDIDATE_IDENTITIES" in result.warning_codes
    # No identity is auto-selected under ambiguity: the primary payload is
    # the ordered candidate bundle itself (support count dominates, so
    # NCC(=O)O with routes c+e precedes CC(=O)O with route a).
    assert isinstance(result.result, dict)
    assert [row["canonical_smiles"] for row in result.result["candidates"]] == [
        "NCC(=O)O",
        "CC(=O)O",
    ]
    assert len(result.alternatives) == 2
    assert {row["canonical_smiles"] for row in result.alternatives} == {
        "CC(=O)O", "NCC(=O)O"
    }
    alternative = next(
        row for row in result.alternatives
        if row["canonical_smiles"] == "CC(=O)O"
    )
    assert alternative["routes"] == ["a"]
    assert alternative["supporting_route_count"] == 1
    assert (
        result.provenance["candidates"]["distinct_identity_count"] == 2
    )
    assert result.provenance["candidates"]["primary"] is None
    assert result.provenance["candidates"]["selection_policy"] == (
        "no automatic chemical identity selection under ambiguity"
    )


def test_ensemble_support_count_breaks_route_tie(tmp_path, monkeypatch):
    rows = [
        _candidate_row("a", "CC(=O)O"),
        _candidate_row("b", "CC(=O)O"),
        _candidate_row("a", "NCC(=O)O"),
    ]
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=_assessment(rows))
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality == "medium"
    assert result.smiles is None
    # support count dominates: CC(=O)O (routes a+b) precedes NCC(=O)O
    assert [row["canonical_smiles"] for row in result.result["candidates"]] == [
        "CC(=O)O",
        "NCC(=O)O",
    ]
    assert result.result["candidates"][1]["supporting_route_count"] == 1
    assert result.result["candidates"][0]["supporting_route_count"] == 2
    assert result.provenance["candidates"]["primary"] is None


def test_ensemble_route_priority_breaks_count_tie(tmp_path, monkeypatch):
    rows = [
        _candidate_row("a", "NCC(=O)O"),
        _candidate_row("c", "CC(=O)O"),
    ]
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=_assessment(rows))
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality == "medium"
    assert result.smiles is None
    # both count 1; route a beats c in bundle order
    assert [row["canonical_smiles"] for row in result.result["candidates"]] == [
        "NCC(=O)O",
        "CC(=O)O",
    ]
    assert result.result["candidates"][0]["primary_route"] == "a"
    assert result.provenance["candidates"]["primary"] is None


def test_ensemble_inchikey_tiebreak_is_stable(tmp_path, monkeypatch):
    rows = [
        _candidate_row("a", "NCC(=O)O"),
        _candidate_row("a", "CC(=O)O"),
    ]
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=_assessment(rows))
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))
    groups = result.provenance["candidates"]["identity_groups"]
    keys = sorted(group["full_inchikey"] for group in groups)

    assert result.quality == "medium"
    # bundle order preserves the group ordering (inchikey tiebreak), but
    # no primary identity is asserted
    assert result.provenance["candidates"]["primary"] is None
    assert (
        result.result["candidates"][0]["full_inchikey"] == keys[0]
    )
    assert len(result.alternatives) == 2
    assert (
        result.result["candidates"][0]["canonical_smiles"]
        == groups[0]["canonical_smiles"]
    )


def test_duplicate_route_rows_do_not_inflate_support_count(
    tmp_path, monkeypatch
):
    rows = [
        _candidate_row("a", "NCC(=O)O"),
        _candidate_row("a", "NCC(=O)O"),
        _candidate_row("c", "NCC(=O)O"),
    ]
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=_assessment(rows))
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality == "high"
    assert result.result == "NCC(=O)O"
    primary = result.provenance["candidates"]["primary"]
    assert primary["supporting_route_count"] == 2  # distinct routes a,c only
    assert primary["routes"] == ["a", "c"]
    assert primary["primary_route"] == "a"
    assert primary["admitted_route_row_count"] == 3  # rows, not votes


def test_declared_recomputed_mismatch_rows_never_elevate(
    tmp_path, monkeypatch
):
    rows = [_candidate_row_with_wrong_declared_key("a", "NCC(=O)O")]
    assessment = _assessment(rows)
    assert assessment["candidates"] == []
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=assessment)
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality not in {"high", "medium"}
    assert result.quality == "partial"
    assert result.provenance["candidate_assessment"]["status"] == "valid"
    assert "CANDIDATE_ASSESSMENT_INVALID" not in result.warning_codes


def test_semantically_invalid_assessment_is_skipped(tmp_path, monkeypatch):
    rows = [_candidate_row("b", "NCC(=O)O")]
    assessment = _assessment(rows)
    assessment["candidates"][0]["canonical_smiles"] = "CC(=O)O"
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=assessment)
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality not in {"high", "medium"}
    assert result.quality == "partial"
    assert "CANDIDATE_ASSESSMENT_INVALID" in result.warning_codes
    assert result.provenance["candidate_assessment"]["status"] == "invalid"


def test_assessment_result_context_must_match_outer_strict(
    tmp_path, monkeypatch
):
    rows = [_candidate_row("b", "NCC(=O)O")]
    assessment = _candidate_assessment(
        rows,
        {},
        status="success",
        qualified_success=True,
        support_status="qualified",
        output_smiles="NCC(=O)O",
        output_inchikey=Chem.MolToInchiKey(Chem.MolFromSmiles("NCC(=O)O")),
    )
    _install_strict(
        monkeypatch,
        _strict(rows=rows, assessment=assessment),
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality not in {"high", "medium"}
    assert "CANDIDATE_ASSESSMENT_INVALID" in result.warning_codes
    assert "result_context" in result.provenance["candidate_assessment"][
        "error"
    ]


def test_assessment_route_audits_must_match_outer_route_rows(
    tmp_path, monkeypatch
):
    assessment_rows = [_candidate_row("b", "NCC(=O)O")]
    strict_rows = [_candidate_row("b", "CC(=O)O")]
    _install_strict(
        monkeypatch,
        _strict(
            rows=strict_rows,
            assessment=_assessment(assessment_rows),
        ),
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality not in {"high", "medium"}
    assert "CANDIDATE_ASSESSMENT_INVALID" in result.warning_codes
    assert "route audits" in result.provenance["candidate_assessment"][
        "error"
    ]


def test_route_rows_without_assessment_not_accepted(tmp_path, monkeypatch):
    rows = [_candidate_row("b", "NCC(=O)O")]
    _install_strict(monkeypatch, _strict(rows=rows))

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality not in {"high", "medium"}
    assert result.quality == "partial"
    assert "CANDIDATE_ASSESSMENT_MISSING" in result.warning_codes
    assert result.provenance["candidate_assessment"]["status"] == "missing"


def test_f_h_rows_are_diagnostic_only_and_never_elevate(
    tmp_path, monkeypatch
):
    rows = [
        _candidate_row("f", "CC(=O)O"),
        _candidate_row("h", "NCC(=O)O"),
    ]
    _install_strict(monkeypatch, _strict(rows=rows))

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality not in {"high", "medium"}
    assert result.quality == "partial"
    assert "F_H_DIAGNOSTIC_ONLY" in result.warning_codes
    diagnostics = result.provenance["f_h_diagnostic_rows"]
    assert [row["route"] for row in diagnostics] == ["f", "h"]
    assert all(row["diagnostic_only"] for row in diagnostics)


def test_f_h_conflicts_do_not_demote_unique_chemical_candidate(
    tmp_path, monkeypatch
):
    rows = [
        _candidate_row("a", "NCC(=O)O"),
        _candidate_row("f", "CC(=O)O"),
        _candidate_row("h", "CCC(=O)O"),
    ]
    _install_strict(
        monkeypatch, _strict(rows=rows, assessment=_assessment(rows))
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality == "high"
    assert result.result == "NCC(=O)O"
    assert "F_H_DIAGNOSTIC_ONLY" in result.warning_codes


# --------------------------------------------------------------------------
# Ladder stages 4-7: topology qualification, explicit-only partial, raw
# --------------------------------------------------------------------------


def test_non_ring_simple_pdb_is_partial_not_topology(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "partial"
    assert result.source == "rdkit_explicit_only_partial"
    assert result.smiles is None
    assert result.ambiguous is False
    assert "PARTIAL_EXPLICIT_ONLY" in result.warning_codes
    attempts = result.provenance["ladder_attempts"]
    assert attempts[0]["stage"] == "rdkit_proximity"
    assert attempts[0]["ok"] is False
    assert "DETECTED_RING_SIZE_BELOW_THRESHOLD" in attempts[0][
        "reason_codes"
    ]
    assert attempts[1]["stage"] == "rdkit_determine_connectivity"
    assert attempts[1]["ok"] is False
    assert attempts[2]["stage"] == "rdkit_explicit_only_partial"
    assert attempts[2]["ok"] is True
    assert result.graph is not None
    assert len(result.graph["atoms"]) == 4
    assert all(bond["order"] is None for bond in result.graph["bonds"])
    payload = json.dumps(application.json_ready(result), sort_keys=True)
    assert "U2" not in payload and "U3" not in payload


def test_zero_bond_pdb_falls_to_raw_with_no_bonds(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, NO_BONDS_PDB)
    )

    assert result.status == "success"
    assert result.quality == "raw"
    assert result.source == "raw_pdb_graph"
    assert result.smiles is None
    assert result.ambiguous is False
    assert result.graph["bonds"] == []
    assert "NO_BONDS_AVAILABLE" in result.warning_codes
    assert "NO_BONDS_AVAILABLE" in _topology_reason_codes(
        result, "rdkit_proximity"
    )
    partial_attempt = [
        attempt
        for attempt in result.provenance["ladder_attempts"]
        if attempt["stage"] == "rdkit_explicit_only_partial"
    ][0]
    assert partial_attempt["ok"] is False
    assert "NO_BONDS_AVAILABLE" in partial_attempt["reason_codes"]


def test_fragmented_pdb_not_topology(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, FRAGMENTED_PDB)
    )

    assert result.quality not in {"topology"}
    assert result.quality == "partial"
    assert "MULTIPLE_HEAVY_ATOM_COMPONENTS" in _topology_reason_codes(
        result, "rdkit_proximity"
    )


def test_heavy_count_mismatch_not_topology(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, audit={"normalized_heavy_atom_count": 99})
    )

    assert result.quality == "partial"
    assert "HEAVY_ATOM_COUNT_MISMATCH" in _topology_reason_codes(
        result, "rdkit_proximity"
    )


def test_duplicate_serial_not_topology(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, DUP_SERIAL_PDB)
    )

    assert result.status == "success"
    assert result.quality in {"raw", "partial"}
    assert result.artifact_status in {"raw_coordinates", "partial_graph"}
    assert "DUPLICATE_ATOM_SERIAL" in _topology_reason_codes(
        result, "rdkit_proximity"
    )
    # readable atoms survive with an explicit damage ledger
    raw = result.provenance.get("raw") or {}
    assert raw.get("duplicate_atom_serials") == [1]
    assert len(result.graph["atoms"]) == 3


def test_missing_explicit_edge_not_topology(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, MULTI_MODEL_PDB)
    )

    assert result.status == "success"
    assert result.quality not in {"exact", "high", "medium", "topology"}
    assert result.artifact_status in {"partial_graph", "raw_coordinates"}
    assert "EXPLICIT_CONECT_NOT_PRESERVED" in _topology_reason_codes(
        result, "rdkit_proximity"
    )
    # second-model serials survive as duplicate-conflict evidence when the
    # artifact is the raw graph rather than an explicit-only partial
    if result.quality == "raw":
        assert result.provenance["raw"]["duplicate_atom_serials"] == [
            1, 2, 3, 4,
        ]


def test_peptide_ring_qualifies_topology(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, peptide_ring_pdb())
    )

    assert result.status == "success"
    assert result.quality == "topology"
    assert result.source == "rdkit_proximity"
    assert result.smiles is None
    assert result.ambiguous is False
    assert result.result == result.graph
    assert all(bond["order"] is None for bond in result.graph["bonds"])
    attempt = result.provenance["ladder_attempts"][0]
    assert attempt["ok"] is True
    assert attempt["largest_detected_ring_size"] == 9
    assert attempt["explicit_edges_preserved"] is True
    assert len(result.graph["bonds"]) >= 9


def test_topology_qualification_rejects_impossible_carbon_degree(tmp_path):
    molecule = Chem.RWMol()
    for _ in range(11):
        atom = Chem.Atom("C")
        atom.SetIntProp("_PDBSerial", molecule.GetNumAtoms() + 1)
        molecule.AddAtom(atom)
    for index in range(8):
        molecule.AddBond(index, (index + 1) % 8, Chem.BondType.SINGLE)
    for branch in (8, 9, 10):
        molecule.AddBond(0, branch, Chem.BondType.SINGLE)
    pdb_path = tmp_path / "empty.pdb"
    pdb_path.write_text("END\n", encoding="ascii")
    prepared = SimpleNamespace(audit={"normalized_heavy_atom_count": 11})

    reasons, details = result_first._topology_qualification(
        molecule.GetMol(), prepared, pdb_path, 8
    )

    assert "ABNORMAL_ATOM_DEGREE" in reasons
    assert details["abnormal_atom_degrees"][0]["degree"] == 5


def test_topology_qualification_rejects_atom_identity_drift(tmp_path):
    path = tmp_path / "identity-drift.pdb"
    path.write_text(peptide_ring_pdb(), encoding="ascii")
    molecule = Chem.MolFromPDBFile(
        str(path), proximityBonding=True, sanitize=False, removeHs=False
    )
    assert molecule is not None
    serials = [result_first._atom_serial(atom) for atom in molecule.GetAtoms()]
    mutated = Chem.RWMol(molecule)
    index = serials.index(2)
    mutated.GetAtomWithIdx(index).SetAtomicNum(7)
    prepared = SimpleNamespace(
        audit={"normalized_heavy_atom_count": molecule.GetNumHeavyAtoms()}
    )

    reasons, details = result_first._topology_qualification(
        mutated.GetMol(), prepared, path, 8
    )

    assert "ATOM_IDENTITY_NOT_PRESERVED" in reasons
    assert details["atom_identity_mismatches"][0]["serial"] == 2


def test_connectivity_topology_stage_with_real_ring(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    monkeypatch.setattr(result_first, "_proximity_mol", lambda path: None)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, peptide_ring_pdb())
    )

    assert result.quality == "topology"
    assert result.source == "rdkit_determine_connectivity"
    attempts = result.provenance["ladder_attempts"]
    assert attempts[0]["stage"] == "rdkit_proximity"
    assert attempts[0]["ok"] is False
    assert attempts[1]["stage"] == "rdkit_determine_connectivity"
    assert attempts[1]["ok"] is True
    assert "BOND_ORDERS_INFERRED" in result.warning_codes
    assert len(result.graph["bonds"]) >= 9


def _mutate_rdkit_edge(mol, left, right, *, replacement=None):
    """Mutate a real RDKit molecule by PDB serial for connectivity tests."""
    serials = [result_first._atom_serial(atom) for atom in mol.GetAtoms()]
    left_index = serials.index(left)
    right_index = serials.index(right)
    rw_mol = Chem.RWMol(mol)
    rw_mol.RemoveBond(left_index, right_index)
    if replacement is not None:
        replacement_left, replacement_right = replacement
        rw_mol.AddBond(
            serials.index(replacement_left),
            serials.index(replacement_right),
            Chem.BondType.SINGLE,
        )
    # DetermineConnectivity mutates its input in place; mirror that contract
    # after applying the directed test mutation.
    mol.__init__(rw_mol.GetMol())


@pytest.mark.parametrize("replacement", [None, (1, 3)])
def test_connectivity_rejects_deleted_or_replaced_explicit_edges(
    tmp_path, monkeypatch, replacement
):
    """A real RDKit post-call mutation must never be exposed as topology."""
    _install_strict(monkeypatch, _strict())
    monkeypatch.setattr(result_first, "_proximity_mol", lambda path: None)
    determine_connectivity = result_first.rdDetermineBonds.DetermineConnectivity

    def mutate_after_connectivity(mol):
        determine_connectivity(mol)
        _mutate_rdkit_edge(mol, 1, 2, replacement=replacement)

    monkeypatch.setattr(
        result_first.rdDetermineBonds,
        "DetermineConnectivity",
        mutate_after_connectivity,
    )

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, peptide_ring_pdb())
    )

    assert result.status == "success"
    assert result.quality == "partial"
    assert result.source == result_first.SOURCE_PARTIAL
    attempt = [
        row
        for row in result.provenance["ladder_attempts"]
        if row["stage"] == result_first.SOURCE_CONNECTIVITY
    ][0]
    assert attempt["ok"] is False
    audit = attempt["explicit_edge_audit"]
    assert audit["preserved"] is False
    assert [1, 2] in audit["missing_edges"]
    assert result_first._WARNING_EXPLICIT_EDGES_NOT_PRESERVED in attempt[
        "reason_codes"
    ]
    if replacement is not None:
        assert [1, 3] in audit["additional_edges"]


def test_connectivity_keeps_explicit_edges_while_allowing_new_edge(
    tmp_path, monkeypatch
):
    """Geometry may add edges, provided every explicit edge survives."""
    _install_strict(monkeypatch, _strict())
    monkeypatch.setattr(result_first, "_proximity_mol", lambda path: None)
    determine_connectivity = result_first.rdDetermineBonds.DetermineConnectivity

    def add_after_connectivity(mol):
        determine_connectivity(mol)
        _mutate_rdkit_edge(mol, 1, 2, replacement=(1, 3))
        # Restore the explicit input edge and retain the new geometry-like
        # edge, exercising the allowed-addition path of the audit.
        serials = [result_first._atom_serial(atom) for atom in mol.GetAtoms()]
        rw_mol = Chem.RWMol(mol)
        rw_mol.AddBond(
            serials.index(1), serials.index(2), Chem.BondType.SINGLE
        )
        mol.__init__(rw_mol.GetMol())

    monkeypatch.setattr(
        result_first.rdDetermineBonds,
        "DetermineConnectivity",
        add_after_connectivity,
    )

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, peptide_ring_pdb())
    )

    assert result.status == "success"
    assert result.quality == "topology"
    assert result.source == result_first.SOURCE_CONNECTIVITY
    attempt = [
        row
        for row in result.provenance["ladder_attempts"]
        if row["stage"] == result_first.SOURCE_CONNECTIVITY
    ][0]
    audit = attempt["explicit_edge_audit"]
    assert audit["preserved"] is True
    assert audit["additional_edges_allowed"] is True
    assert [1, 3] in audit["additional_edges"]


def test_connectivity_none_preserves_no_atoms_provenance(
    tmp_path, monkeypatch
):
    _install_strict(monkeypatch, _strict())
    monkeypatch.setattr(result_first, "_proximity_mol", lambda path: None)
    monkeypatch.setattr(result_first, "_connectivity_mol", lambda path: None)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, peptide_ring_pdb())
    )

    attempt = [
        row
        for row in result.provenance["ladder_attempts"]
        if row["stage"] == result_first.SOURCE_CONNECTIVITY
    ][0]
    assert attempt["ok"] is False
    assert "NO_ATOMS" in attempt["reason_codes"]
    assert result_first._WARNING_EXPLICIT_EDGES_NOT_PRESERVED in attempt[
        "reason_codes"
    ]
    assert attempt["explicit_edge_audit"]["preserved"] is False


def test_peptide_ring_through_reconstruct_structure_wiring(
    tmp_path, monkeypatch
):
    _install_strict(monkeypatch, _strict())
    path = tmp_path / "ring.pdb"
    path.write_text(peptide_ring_pdb(), encoding="ascii")

    result = result_first.reconstruct_structure(path, chain_id="L")

    assert result.status == "success"
    assert result.quality == "topology"
    assert result.provenance["prepared"]["normalized_heavy_atom_count"] == 9
    assert (
        result.provenance["prepared"][
            "materialized_explicit_connection_count"
        ]
        == 9
    )


def test_nonpeptide_macrocycle_is_not_topology_qualified(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, ring_pdb(8))
    )

    assert result.status == "success"
    assert result.quality in {"partial", "raw"}
    assert "NOT_PEPTIDE_ENTITY" in _topology_reason_codes(
        result, "rdkit_proximity"
    )


def test_ring_below_threshold_is_partial(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, ring_pdb(8)),
        minimum_macrocycle_ring_size=10,
    )

    assert result.quality == "partial"
    qualification = result.provenance["ladder_attempts"][0]["qualification"]
    assert qualification["largest_detected_ring_size"] == 8
    assert qualification["minimum_macrocycle_ring_size"] == 10


def test_partial_explicit_only_distinct_from_proximity(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.quality == "partial"
    # explicit-only partial exposes exactly the CONECT edges, never
    # distance-invented bonds (proximity would add an N-CA/CA-C/C-O set).
    assert result.graph["bonds"] == [
        {"a": 1, "b": 2, "order": None},
        {"a": 2, "b": 3, "order": None},
    ]
    assert result.provenance["partial"]["explicit_conect_pair_count"] == 2
    partial_attempt = [
        attempt
        for attempt in result.provenance["ladder_attempts"]
        if attempt["stage"] == "rdkit_explicit_only_partial"
    ][0]
    assert partial_attempt["explicit_only"] is True


def test_raw_graph_preserves_atoms_and_conect(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    _kill_rdkit(monkeypatch)

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "raw"
    assert result.source == "raw_pdb_graph"
    assert result.smiles is None
    assert result.graph == {
        "atoms": [
            {
                "serial": 1,
                "name": "N",
                "residue": "ALA",
                "residue_number": 1,
                "chain": "L",
                "element": "N",
                "xyz": [1.0, 2.0, 3.0],
            },
            {
                "serial": 2,
                "name": "CA",
                "residue": "ALA",
                "residue_number": 1,
                "chain": "L",
                "element": "C",
                "xyz": [2.0, 2.0, 3.0],
            },
            {
                "serial": 3,
                "name": "C",
                "residue": "ALA",
                "residue_number": 1,
                "chain": "L",
                "element": "C",
                "xyz": [3.0, 2.0, 3.0],
            },
            {
                "serial": 4,
                "name": "O",
                "residue": "ALA",
                "residue_number": 1,
                "chain": "L",
                "element": "O",
                "xyz": [4.0, 2.0, 3.0],
            },
        ],
        "bonds": [
            {"a": 1, "b": 2, "order": None},
            {"a": 2, "b": 3, "order": None},
        ],
    }
    assert "NO_BONDS_AVAILABLE" not in result.warning_codes


def test_raw_graph_without_bonds_warns(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    _kill_rdkit(monkeypatch)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, NO_CONECT_PDB)
    )

    assert result.status == "success"
    assert result.quality == "raw"
    assert result.graph["bonds"] == []
    assert "NO_BONDS_AVAILABLE" in result.warning_codes


@pytest.mark.parametrize(
    ("text", "damage_kind"),
    [
        (DUP_SERIAL_PDB, "duplicate_atom_serials"),
        (SELF_CONECT_PDB, "dropped_conect_token_count"),
        (DANGLING_CONECT_PDB, "dropped_conect_token_count"),
        (MALFORMED_ATOM_PDB, "skipped_atom_records"),
    ],
)
def test_raw_parser_preserves_damaged_inputs(
    tmp_path, monkeypatch, text, damage_kind
):
    _install_strict(monkeypatch, _strict())
    _kill_rdkit(monkeypatch)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, text)
    )

    assert result.status == "success"
    assert result.quality == "raw"
    assert result.artifact_status == "raw_coordinates"
    assert result.graph["atoms"]
    raw = result.provenance["raw"]
    damage = raw.get(damage_kind) or []
    assert damage, f"expected {damage_kind} ledger entry"
    assert "NO_READABLE_STRUCTURE" not in result.warning_codes
    raw_attempt = [
        attempt
        for attempt in result.provenance["ladder_attempts"]
        if attempt["stage"] == "raw_pdb_graph"
    ][0]
    assert raw_attempt["ok"] is True


def test_no_atoms_failed(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    _kill_rdkit(monkeypatch)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, EMPTY_PDB)
    )

    # Zero atoms is the only ladder-bottom case: a typed opaque artifact,
    # not a vanished input.
    assert result.status == "success"
    assert result.quality == "opaque"
    assert result.source == "opaque_input_artifact"
    assert result.artifact_status == "opaque_input"
    assert result.smiles is None
    assert result.graph is None
    assert isinstance(result.result, dict)
    assert result.result.get("failure_reason")
    assert "NO_READABLE_STRUCTURE" in result.warning_codes
    assert result.provenance["failure_reason"]


@pytest.mark.parametrize(
    "block_code",
    [
        "PDB_LINK_CONFLICT",
        "UNRESOLVED_PDB_LINK",
        "V6_INTERNAL_ERROR",
        "PDB_LINK_CONECT_CONFLICT",
        "V5_MALFORMED_CONECT_RECORD",
        "V5_DUPLICATE_ATOM_IDENTITY",
        "V5_STANDARD_RESIDUE_BOND_GEOMETRY_CONFLICT",
        "V5_CONFLICTING_CYCLIZATION_ENDPOINT",
        "V5_EXPLICIT_CONNECTION_VALENCE_CONFLICT",
        "V5_INPUT_AUDIT_FAILED",
    ],
)
def test_blocked_strict_only_tries_raw(tmp_path, monkeypatch, block_code):
    _install_strict(
        monkeypatch,
        _strict(warning_codes=[block_code]),
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    # Integrity conflicts block promotion only: the observation ladder
    # still runs and readable candidates are retained, unqualified.
    assert result.status == "success"
    assert result.quality not in {"exact", "high", "medium"}
    assert "STRICT_FALLBACK_BLOCKED" in result.warning_codes
    block = result.provenance["fallback_block"]
    assert block["reason_codes"] == [block_code]
    assert block["blocked"] is False
    assert block["promotion_blocked"] is True
    assert block["policy"] == "degrade_without_rejection"
    stages = [
        attempt["stage"] for attempt in result.provenance["ladder_attempts"]
    ]
    assert result_first.SOURCE_PROXIMITY in stages
    assert result_first.SOURCE_CONNECTIVITY in stages


@pytest.mark.parametrize(
    "fallback_code",
    [
        "V5_SEQRES_COORDINATE_MISMATCH",
        "V6_EXPLICIT_CLOSURE_BOND_ORDER_AMBIGUOUS",
        "V6_LOCAL_MONOMER_INFERENCE_NOT_UNIQUE",
        "V6_MULTIPOINT_SCAFFOLD_NOT_SUPPORTED",
        "V6_LIBRARY_VS_EMBEDDED_PROTONATION_CONFLICT",
    ],
)
def test_chemistry_uncertainty_codes_do_not_block_fallback(
    tmp_path, monkeypatch, fallback_code
):
    _install_strict(
        monkeypatch,
        _strict(warning_codes=[fallback_code]),
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert result.status == "success"
    assert result.quality == "partial"
    assert result.provenance.get("fallback_block") is None
    assert "STRICT_FALLBACK_BLOCKED" not in result.warning_codes
    assert fallback_code in result.warning_codes
    stages = [attempt["stage"] for attempt in result.provenance["ladder_attempts"]]
    assert stages == [
        result_first.SOURCE_PROXIMITY,
        result_first.SOURCE_CONNECTIVITY,
        result_first.SOURCE_PARTIAL,
    ]


def test_strict_call_exception_blocks_to_raw_only(tmp_path, monkeypatch):
    def exploding(*args, **kwargs):
        raise RuntimeError("v6 exploded")

    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_prepared_structure_fail_closed_v6",
        exploding,
    )

    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    # A strict implementation error is stage-local: it is recorded in
    # provenance and never blocks the observation ladder or promotion.
    assert result.status == "success"
    assert result.quality in {"raw", "partial"}
    assert result.provenance["strict_error"] == "RuntimeError: v6 exploded"
    assert "STRICT_FALLBACK_BLOCKED" not in result.warning_codes
    stages = [
        attempt["stage"] for attempt in result.provenance["ladder_attempts"]
    ]
    assert result_first.SOURCE_PROXIMITY in stages
    assert result_first.SOURCE_CONNECTIVITY in stages


def test_failed_preserves_prior_warning_codes(tmp_path, monkeypatch):
    _install_strict(
        monkeypatch,
        _strict(
            warning_codes=[
                "V6_PERSISTENT_OVERLAY_AUDIT_FAILED",
                "SOME_STRICT_CODE",
            ]
        ),
    )
    _kill_rdkit(monkeypatch)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, EMPTY_PDB)
    )

    # Ladder-bottom opaque artifact still carries every prior warning code.
    assert result.status == "success"
    assert result.quality == "opaque"
    assert result.artifact_status == "opaque_input"
    for code in (
        "V6_STRICT_NOT_SUCCESS",
        "STRICT_FALLBACK_BLOCKED",
        "V6_PERSISTENT_OVERLAY_AUDIT_FAILED",
        "SOME_STRICT_CODE",
        "NO_READABLE_STRUCTURE",
    ):
        assert code in result.warning_codes


def test_rdkit_postprocessing_exception_falls_through(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    def exploding_graph(mol):
        raise RuntimeError("graph post-processing failed")

    monkeypatch.setattr(result_first, "_mol_to_graph", exploding_graph)

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, ring_pdb(8))
    )

    assert result.status == "success"
    assert result.quality == "raw"
    errors = [
        attempt.get("error", "")
        for attempt in result.provenance["ladder_attempts"]
    ]
    assert any("graph post-processing failed" in error for error in errors)
    assert "NO_READABLE_STRUCTURE" not in result.warning_codes


def test_normalization_conect_reduction_warning_and_audit(
    tmp_path, monkeypatch
):
    _install_strict(monkeypatch, _strict())
    audit = {
        "source_pdb_conect_pair_count": 3,
        "first_model_pdb_conect_pair_count": 3,
        "selected_chain_pdb_conect_pair_count": 2,
        "materialized_explicit_connection_count": 2,
        "structured_connection_count": 0,
        "excluded_nonselected_residue_structured_connection_count": 1,
    }

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, audit=audit)
    )

    assert result.quality == "partial"
    assert "SOURCE_CONECT_REDUCED_DURING_NORMALIZATION" in result.warning_codes
    prepared = result.provenance["prepared"]
    assert prepared["source_pdb_conect_pair_count"] == 3
    assert prepared["materialized_explicit_connection_count"] == 2
    reduction = prepared["connection_reduction"]
    assert reduction["first_model_to_selected_chain"]["reduced_by"] == 1
    assert reduction["source_to_materialized"]["reduced_by"] == 1
    assert prepared["connection_audit"][
        "excluded_nonselected_residue_structured_connection_count"
    ] == 1


def test_damaged_dangling_conect_source_is_at_most_raw_failed(
    tmp_path, monkeypatch
):
    _install_strict(monkeypatch, _strict())

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, DANGLING_CONECT_PDB)
    )

    # A dangling CONECT edge degrades the artifact; it never removes the
    # input from the product surface.
    assert result.status == "success"
    assert result.quality in {"raw", "partial"}
    assert result.artifact_status in {"raw_coordinates", "partial_graph"}
    assert "EXPLICIT_CONECT_NOT_PRESERVED" in _topology_reason_codes(
        result, "rdkit_proximity"
    )
    assert "NO_READABLE_STRUCTURE" not in result.warning_codes


def test_graph_endpoint_canonicalization(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    reversed_pdb = SIMPLE_PDB.replace(
        "CONECT    1    2\nCONECT    2    3",
        "CONECT    2    1\nCONECT    3    2",
    )

    result = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path, reversed_pdb)
    )

    assert result.quality == "partial"
    assert result.graph["bonds"] == [
        {"a": 1, "b": 2, "order": None},
        {"a": 2, "b": 3, "order": None},
    ]
    serials = [atom["serial"] for atom in result.graph["atoms"]]
    assert serials == sorted(serials)


def test_graph_json_repeat_stable(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())

    first = result_first.reconstruct_prepared_structure(_prepared(tmp_path))
    second = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    assert json.dumps(
        application.json_ready(first), sort_keys=True
    ) == json.dumps(application.json_ready(second), sort_keys=True)
    json.dumps(application.json_ready(first))


def test_reconstruct_structure_input_error_is_failed(tmp_path, monkeypatch):
    _install_strict(monkeypatch, _strict())
    path = tmp_path / "empty.pdb"
    path.write_text(EMPTY_PDB, encoding="ascii")

    result = result_first.reconstruct_structure(path, chain_id="L")

    # Input errors produce a typed opaque artifact, not a vanished input.
    assert result.status == "success"
    assert result.quality == "opaque"
    assert result.source == "opaque_input_artifact"
    assert result.artifact_status == "opaque_input"
    assert isinstance(result.result, dict)
    assert result.result.get("error_code") or result.provenance.get(
        "error_code"
    )
    assert result.result.get("failure_reason") or result.provenance.get(
        "failure_reason"
    )


def test_application_facade_envelope_and_json(tmp_path, monkeypatch):
    strict = _strict(
        status="success", smiles="NCC(=O)O", qualified_success=True
    )
    _install_strict(monkeypatch, strict)
    path = tmp_path / "input.pdb"
    path.write_text(SIMPLE_PDB, encoding="ascii")

    payload = application.reconstruct_result_first(
        path,
        chain_id="L",
        minimum_macrocycle_ring_size=8,
    )

    assert payload["operation"] == "reconstruct_result_first"
    assert payload["status"] == "success"
    assert payload["data"]["quality"] == "exact"
    assert payload["data"]["strict_status"] == "success"
    assert payload["data"]["strict_result"]["output_smiles"] == "NCC(=O)O"
    json.dumps(payload)  # fully JSON serializable

    is_reconstruction, smiles, context, handoff_error = (
        application._reconstruction_handoff(payload)
    )
    assert is_reconstruction is True
    assert handoff_error is None
    assert smiles == "NCC(=O)O"
    assert context["strict_status"] == "success"
    assert context["strict_result"]["output_smiles"] == "NCC(=O)O"
    json.dumps(context)


def test_object_handoff_json_readies_strict_dataclass(tmp_path, monkeypatch):
    strict = _strict(
        status="success", smiles="NCC(=O)O", qualified_success=True
    )
    _install_strict(monkeypatch, strict)
    result = result_first.reconstruct_prepared_structure(_prepared(tmp_path))

    is_reconstruction, smiles, context, handoff_error = (
        application._reconstruction_handoff(result)
    )
    assert is_reconstruction is True
    assert handoff_error is None
    assert smiles == "NCC(=O)O"
    assert context["strict_status"] == "success"
    assert context["strict_result"]["output_smiles"] == "NCC(=O)O"
    json.dumps(context)
