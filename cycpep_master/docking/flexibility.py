"""Validated ensemble loading and typed flexibility evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from rdkit import Chem

from ..core.artifacts import (
    ArtifactStatus,
    ArtifactType,
    BudgetState,
    ClaimBoundary,
    ENSEMBLE_SCHEMA_VERSION,
    FlexibilityAssessmentArtifact,
    FlexibilityLevel,
    artifact_payload_sha256,
    evidence_from_dict,
    inherit_evidence,
    make_artifact_id,
    validate_artifact_identity,
    validate_inherited_evidence,
)
from .mol2_input import ValidatedMol2, load_validated_mol2, sha256_path


def _normalized(molecule: Chem.Mol) -> Chem.Mol:
    value = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
    return Chem.AddHs(value, addCoords=True)


def _atom_signature(molecule: Chem.Mol) -> list[tuple[Any, ...]]:
    return [
        (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            tuple(sorted(
                (
                    neighbor.GetIdx(),
                    str(
                        molecule.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        ).GetBondType()
                    ),
                )
                for neighbor in atom.GetNeighbors()
            )),
        )
        for atom in molecule.GetAtoms()
    ]


def _manifest_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _invalid_ensemble(
    source: Path,
    reason: str,
    *,
    status: str = "invalid",
    **details: Any,
) -> tuple[None, dict[str, Any]]:
    return None, {
        "status": status,
        "reason": reason,
        "manifest_path": str(source),
        **details,
    }


def load_validated_flexibility_ensemble(
    parent: ValidatedMol2,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[Chem.Mol | None, dict[str, Any]]:
    source = Path(manifest_path).resolve()
    if not source.is_file():
        return _invalid_ensemble(
            source,
            "ensemble manifest is missing",
            status="unavailable",
        )
    observed_manifest_sha256 = sha256_path(source)
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        return _invalid_ensemble(
            source, "ensemble manifest SHA-256 drift"
        )
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return _invalid_ensemble(
            source, f"cannot read ensemble manifest: {exc}"
        )
    if not isinstance(manifest, Mapping):
        return _invalid_ensemble(
            source, "ensemble manifest must be a JSON object"
        )
    if manifest.get("schema_version") != ENSEMBLE_SCHEMA_VERSION:
        return _invalid_ensemble(
            source, "unsupported ensemble manifest schema"
        )
    artifact = manifest.get("artifact")
    payload = manifest.get("payload")
    if not isinstance(artifact, Mapping) or not isinstance(
        payload, Mapping
    ):
        return _invalid_ensemble(
            source, "ensemble manifest lacks artifact or payload"
        )
    if artifact.get("artifact_type") != (
        ArtifactType.CONFORMER_ENSEMBLE.value
    ):
        return _invalid_ensemble(
            source, "manifest artifact is not a conformer ensemble"
        )
    if artifact.get("status") not in {
        ArtifactStatus.MATERIALIZED.value,
        ArtifactStatus.PARTIAL.value,
    }:
        return _invalid_ensemble(
            source, "ensemble artifact is not materialized"
        )
    parent_ids = artifact.get("parent_artifact_ids")
    if (
        not isinstance(parent_ids, list)
        or not parent_ids
        or any(not isinstance(value, str) for value in parent_ids)
        or len(set(parent_ids)) != len(parent_ids)
    ):
        return _invalid_ensemble(
            source, "ensemble parent artifact IDs are invalid"
        )
    try:
        observed_payload_hash = artifact_payload_sha256(payload)
        expected_artifact_id = make_artifact_id(
            ArtifactType.CONFORMER_ENSEMBLE,
            tuple(parent_ids),
            payload,
        )
    except Exception as exc:
        return _invalid_ensemble(
            source, f"ensemble payload is not canonical JSON: {exc}"
        )
    if artifact.get("payload_sha256") != observed_payload_hash:
        return _invalid_ensemble(
            source, "ensemble payload SHA-256 differs from artifact"
        )
    if artifact.get("artifact_id") != expected_artifact_id:
        return _invalid_ensemble(
            source, "ensemble artifact ID differs from its payload"
        )
    members = artifact.get("members")
    if not isinstance(members, list) or not members:
        return _invalid_ensemble(
            source,
            "ensemble manifest has no validated members",
            status="unavailable",
        )
    if payload.get("members") != members:
        return _invalid_ensemble(
            source, "artifact and payload member lists differ"
        )
    requested_count = artifact.get("requested_count")
    produced_count = artifact.get("produced_count")
    if (
        not isinstance(requested_count, int)
        or isinstance(requested_count, bool)
        or not isinstance(produced_count, int)
        or isinstance(produced_count, bool)
        or requested_count < produced_count
        or produced_count != len(members)
    ):
        return _invalid_ensemble(
            source, "ensemble member counts are inconsistent"
        )
    if (
        artifact.get("status") == ArtifactStatus.MATERIALIZED.value
        and produced_count != requested_count
    ) or (
        artifact.get("status") == ArtifactStatus.PARTIAL.value
        and not (0 < produced_count < requested_count)
    ):
        return _invalid_ensemble(
            source, "ensemble status and member counts are inconsistent"
        )
    normalized_parent = _normalized(parent.molecule)
    signature = _atom_signature(normalized_parent)
    ensemble = Chem.Mol(normalized_parent)
    ensemble.RemoveAllConformers()
    loaded = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    parent_bound = False
    for row in members:
        if not isinstance(row, Mapping):
            return _invalid_ensemble(
                source, "ensemble member must be an object"
            )
        conformer_id = row.get("conformer_id")
        if (
            not isinstance(conformer_id, str)
            or not conformer_id
            or conformer_id in seen_ids
        ):
            return _invalid_ensemble(
                source, "ensemble conformer IDs are missing or duplicated"
            )
        seen_ids.add(conformer_id)
        if row.get("status") != "validated":
            return _invalid_ensemble(
                source,
                f"ensemble member is not validated: {conformer_id}",
            )
        qa = row.get("qa")
        if not isinstance(qa, Mapping) or qa.get("passed") is not True:
            return _invalid_ensemble(
                source,
                f"ensemble member lacks passing QA: {conformer_id}",
            )
        if not row.get("mol2_path") or not row.get("receipt_path"):
            return _invalid_ensemble(
                source,
                f"ensemble member paths are missing: {conformer_id}",
            )
        path = _manifest_path(source.parent, row["mol2_path"])
        receipt = _manifest_path(
            source.parent, row["receipt_path"]
        )
        if path in seen_paths:
            return _invalid_ensemble(
                source, "ensemble MOL2 member paths are duplicated"
            )
        seen_paths.add(path)
        try:
            member = load_validated_mol2(
                path, receipt_path=receipt
            )
            molecule = _normalized(member.molecule)
        except Exception as exc:
            return _invalid_ensemble(
                source,
                f"ensemble member validation failed: {exc}",
                conformer_id=conformer_id,
            )
        if row.get("mol2_sha256") != member.sha256:
            return _invalid_ensemble(
                source,
                f"ensemble member MOL2 hash drift: {conformer_id}",
            )
        if row.get("receipt_sha256") != member.receipt_sha256:
            return _invalid_ensemble(
                source,
                f"ensemble member receipt hash drift: {conformer_id}",
            )
        if member.full_inchikey != parent.full_inchikey:
            return _invalid_ensemble(
                source, "ensemble member identity differs from parent"
            )
        if _atom_signature(molecule) != signature:
            return _invalid_ensemble(
                source, "ensemble member atom order or graph differs"
            )
        conformer = molecule.GetConformer()
        if any(
            not math.isfinite(value)
            for atom_index in range(molecule.GetNumAtoms())
            for value in tuple(conformer.GetAtomPosition(atom_index))
        ):
            return _invalid_ensemble(
                source, "ensemble member has nonfinite coordinates"
            )
        ensemble.AddConformer(
            Chem.Conformer(conformer), assignId=True
        )
        loaded.append({
            "conformer_id": conformer_id,
            "mol2_sha256": member.sha256,
            "receipt_sha256": member.receipt_sha256,
        })
        if (
            member.sha256 == parent.sha256
            and member.receipt_sha256 == parent.receipt_sha256
        ):
            parent_bound = True
    if not parent_bound:
        return _invalid_ensemble(
            source, "ensemble manifest does not bind the parent MOL2"
        )
    if ensemble.GetNumConformers() < 2:
        return _invalid_ensemble(
            source,
            "at least two validated conformers are required",
            status="unavailable",
            loaded_members=loaded,
        )
    ensemble.SetProp(
        "CYCPEP_TORSION_ENSEMBLE_SOURCE",
        "validated_mol2_manifest",
    )
    ensemble.SetProp(
        "CYCPEP_TORSION_EVIDENCE_LEVEL",
        "validated_sibling_coordinates",
    )
    return ensemble, {
        "status": "loaded",
        "manifest_path": str(source),
        "manifest_sha256": observed_manifest_sha256,
        "conformer_count": ensemble.GetNumConformers(),
        "loaded_members": loaded,
        "ensemble_artifact_id": artifact.get("artifact_id"),
        "requested_count": requested_count,
        "produced_count": produced_count,
        "embedding_performed": False,
    }


def flexibility_artifact_from_audit(
    parent_mol2_artifact: Mapping[str, Any],
    ensemble_artifact: Mapping[str, Any] | None,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    parent_profile = evidence_from_dict(
        parent_mol2_artifact["evidence"]
    )
    lookup_calibrated = any(
        row.get("eligible_to_freeze") is True
        and row.get("calibration_false_rigid_ci_high") is not None
        for row in audit.get("lookup_matches", [])
    )
    requested_mode = str(
        audit.get("requested_flexibility_mode") or "fast"
    )
    effective_mode = str(
        audit.get("effective_flexibility_mode") or "fast"
    )
    ensemble_audit = audit.get("ensemble")
    ensemble_loaded = (
        isinstance(ensemble_audit, Mapping)
        and ensemble_audit.get("status") == "loaded"
    )
    ensemble_used = bool(
        ensemble_loaded and audit.get("ensemble_fallback_triggered")
    )
    if audit.get("torsdof_limit") is None:
        level = FlexibilityLevel.F0
    elif (
        requested_mode == "thorough"
        and effective_mode == "thorough"
        and ensemble_used
    ):
        level = FlexibilityLevel.F3
    elif lookup_calibrated or ensemble_used:
        level = FlexibilityLevel.F2
    else:
        level = FlexibilityLevel.F1
    not_assessable = bool(
        audit.get("torsdof_limit") is not None
        and not audit.get("budget_satisfied")
        and (
            audit.get("flexibility_error")
            or (
                requested_mode != effective_mode
                and int(
                    audit.get("lookup_unresolved_bond_count") or 0
                )
                > 0
            )
        )
    )
    if audit.get("torsdof_limit") is None:
        budget = BudgetState.NOT_REQUESTED
    elif audit.get("budget_satisfied"):
        budget = BudgetState.SATISFIED
    elif not_assessable:
        budget = BudgetState.NOT_ASSESSABLE
    else:
        budget = BudgetState.UNSATISFIED
    evidence = inherit_evidence(
        parent_profile,
        flexibility_level=level,
        budget_state=budget,
    )
    parent_ids = [str(parent_mol2_artifact["artifact_id"])]
    if ensemble_artifact is not None:
        parent_ids.append(str(ensemble_artifact["artifact_id"]))
    ensemble_binding = (
        {
            "status": ensemble_audit.get("status"),
            "manifest_sha256": ensemble_audit.get(
                "manifest_sha256"
            ),
            "ensemble_artifact_id": ensemble_audit.get(
                "ensemble_artifact_id"
            ),
            "conformer_count": ensemble_audit.get(
                "conformer_count"
            ),
        }
        if isinstance(ensemble_audit, Mapping)
        else None
    )
    payload = {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "initial_torsdof": audit.get("initial_torsdof"),
        "final_torsdof": audit.get("final_torsdof"),
        "lookup_matches": list(audit.get("lookup_matches", [])),
        "frozen_bonds": list(audit.get("frozen_bonds", [])),
        "ensemble_binding": ensemble_binding,
        "torsion_prior_runtime_sha256": audit.get(
            "torsion_prior_runtime_sha256"
        ),
        "torsion_prior_manifest_sha256": audit.get(
            "torsion_prior_manifest_sha256"
        ),
        "budget_satisfied": audit.get("budget_satisfied"),
    }
    artifact_id = make_artifact_id(
        ArtifactType.FLEXIBILITY_ASSESSMENT,
        tuple(parent_ids),
        payload,
    )
    artifact = FlexibilityAssessmentArtifact(
        artifact_type=ArtifactType.FLEXIBILITY_ASSESSMENT,
        artifact_id=artifact_id,
        parent_artifact_ids=tuple(parent_ids),
        status=(
            ArtifactStatus.PARTIAL
            if budget == BudgetState.NOT_ASSESSABLE
            else ArtifactStatus.MATERIALIZED
        ),
        payload_sha256=artifact_payload_sha256(payload),
        evidence=evidence,
        warnings=tuple(
            ["TORSION_BUDGET_UNSATISFIED"]
            if budget == BudgetState.UNSATISFIED
            else ["TORSION_BUDGET_NOT_ASSESSABLE"]
            if budget == BudgetState.NOT_ASSESSABLE
            else []
        ),
        provenance={
            "source": (
                "validated_mol2_and_frozen_prior"
                if audit.get("torsion_prior_runtime_sha256")
                else "validated_mol2_without_available_prior"
            ),
            "audit": dict(audit),
        },
        claim_boundary=ClaimBoundary(
            allowed=(
                ("mechanistic torsion budget",)
                + (
                    ("calibrated false-rigid evidence",)
                    if lookup_calibrated
                    else ()
                )
                + (
                    ("validated ensemble torsion evidence",)
                    if ensemble_used
                    else ()
                )
            ),
            forbidden=(
                "biological flexibility",
                "docking benefit",
            ),
        ),
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        initial_torsdof=audit.get("initial_torsdof"),
        final_torsdof=audit.get("final_torsdof"),
        bond_evidence=tuple(
            dict(value) for value in audit.get("lookup_matches", [])
        ),
        budget_satisfied=audit.get("budget_satisfied"),
    )
    validate_inherited_evidence(
        parent_profile,
        evidence,
        allow_flexibility=True,
    )
    validate_artifact_identity(artifact, payload=payload)
    return artifact.to_dict()


__all__ = [
    "flexibility_artifact_from_audit",
    "load_validated_flexibility_ensemble",
]
