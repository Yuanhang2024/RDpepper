"""Collect audited entity-local monomers from sealed V6 result JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

from .derived_monomers import publish_overlay_bundle


_TERMINAL_STATUSES = {
    "success", "rejected", "failed", "timeout", "not_supported",
    "invalid_reference",
}
_SEALED_RUN_STATUSES = {
    "SEALED_RAW_RESULTS_PENDING_INDEPENDENT_SCORING",
    "SEALED_RAW_RESULTS",
}
_TARGET_TOOL = "cycpep_master_v6"
_EMPTY_OVERLAY_STATE = {"aliases": [], "entries": []}
_EMPTY_OVERLAY_STATE_SHA256 = hashlib.sha256(
    json.dumps(
        _EMPTY_OVERLAY_STATE, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at {path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        rows.append(row)
    return rows


def _scores(record: dict[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance")
    scores = provenance.get("scores") if isinstance(provenance, dict) else None
    if not isinstance(scores, dict):
        raise ValueError(f"record {record.get('case_id')} lacks provenance.scores")
    return scores


def _bootstrap(record: dict[str, Any]) -> dict[str, Any] | None:
    evidence = _scores(record).get("input_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(
            f"record {record.get('case_id')} lacks scores.input_evidence"
        )
    bootstrap = evidence.get("local_monomer_bootstrap")
    if bootstrap is None:
        return None
    if not isinstance(bootstrap, dict):
        raise ValueError(
            f"record {record.get('case_id')} has malformed local bootstrap"
        )
    if bootstrap.get("persistent_writes") != 0:
        raise ValueError(
            f"record {record.get('case_id')} did not run with an isolated overlay"
        )
    registry = bootstrap.get("registry_isolation")
    if bootstrap.get("status") == "ready":
        if not isinstance(registry, dict):
            raise ValueError(
                f"record {record.get('case_id')} lacks production registry isolation evidence"
            )
        if (
            registry.get("entity_local_overlay") is not True
            or registry.get("persistent_overlay_required_empty") is not True
            or registry.get("registry_restoration_verified") is not True
            or registry.get("persistent_overlay_at_start", {}).get("status") != "empty"
        ):
            raise ValueError(
                f"record {record.get('case_id')} failed production registry isolation audit"
            )
    elif registry is not None:
        raise ValueError(
            f"record {record.get('case_id')} activated a registry for non-ready inference"
        )
    return bootstrap


def _validate_record(record: dict[str, Any]) -> None:
    case_id = str(record.get("case_id", ""))
    entity_id = str(record.get("chemical_entity_id", ""))
    if not case_id or not entity_id:
        raise ValueError("V6 record lacks case_id or chemical_entity_id")
    if record.get("tool") != _TARGET_TOOL:
        raise ValueError(f"record {case_id} is not a {_TARGET_TOOL} result")
    if record.get("status") not in _TERMINAL_STATUSES:
        raise ValueError(f"record {case_id} has a nonterminal status")
    scores = _scores(record)
    evidence = scores.get("input_evidence")
    persistent = evidence.get("persistent_overlay_audit") if isinstance(
        evidence, dict
    ) else None
    if not isinstance(persistent, dict) or persistent.get("status") != "empty":
        raise ValueError(f"record {case_id} failed production persistent-overlay audit")
    if (
        persistent.get("disk_memory_count_consistent") is not True
        or persistent.get("disk_memory_semantic_consistent") is not True
        or persistent.get("state") != _EMPTY_OVERLAY_STATE
        or persistent.get("state_sha256") != _EMPTY_OVERLAY_STATE_SHA256
    ):
        raise ValueError(f"record {case_id} has an invalid empty-overlay state binding")
    for field in (
        "disk_derived_row_count", "disk_pdb_alias_count",
        "disk_manifest_entry_count",
        "memory_derived_row_count", "memory_pdb_alias_count",
    ):
        if persistent.get(field) != 0:
            raise ValueError(
                f"record {case_id} has nonzero persistent-overlay count {field}"
            )
    policy = evidence.get("entity_local_isolation")
    if (
        not isinstance(policy, dict)
        or policy.get("require_empty_persistent_overlay") is not True
        or policy.get("persistent_overlay_state_sha256")
        != persistent["state_sha256"]
    ):
        raise ValueError(f"record {case_id} lacks production isolation policy binding")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "fresh_process_per_job"
    ) is not True:
        raise ValueError(f"record {case_id} lacks supervised fresh-process evidence")
    bootstrap = _bootstrap(record)
    if bootstrap is not None and bootstrap.get("status") == "ready":
        if policy.get("activation_mode") != "entity_local_registry":
            raise ValueError(f"record {case_id} lacks entity-local activation evidence")


def _candidate(inference: dict[str, Any], entity_id: str) -> dict[str, Any]:
    smiles = str(inference.get("candidate_smiles") or "").strip()
    pdb_resname = str(inference.get("pdb_resname") or "").strip().upper()
    evidence = inference.get("evidence")
    if not smiles or not pdb_resname or not isinstance(evidence, dict):
        raise ValueError("qualified inference lacks monomer identity or evidence")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not Chem.MolToInchiKey(molecule):
        raise ValueError("qualified inference has no computable full InChIKey")
    input_hash = str(evidence.get("input_sha256") or "")
    if len(input_hash) != 64:
        raise ValueError("qualified inference lacks a SHA-256 input binding")
    return {
        "pdb_resname": pdb_resname,
        "smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "r3_mapped_smiles": inference.get("candidate_r3_mapped_smiles"),
        "r3_port": inference.get("r3_port"),
        "input_sha256": input_hash,
        "source_entity_id": entity_id,
    }


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    molecule = Chem.MolFromSmiles(str(candidate["smiles"]))
    if molecule is None:
        raise ValueError("candidate became unparseable during collection")
    port = candidate.get("r3_port")
    port_identity = json.dumps(
        {
            "mapped_smiles": candidate.get("r3_mapped_smiles"),
            "cap": port.get("cap") if isinstance(port, dict) else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        str(candidate["pdb_resname"]),
        Chem.MolToInchiKey(molecule),
        port_identity,
    )


def _quarantine(
    *,
    entity_id: str,
    pdb_resname: str = "",
    status: str,
    reason_codes: Iterable[str],
    input_sha256: str = "",
    candidate_graph_count: int = 0,
    details: dict[str, Any] | None = None,
) -> dict[str, str]:
    return {
        "record_id": entity_id,
        "pdb_resname": pdb_resname,
        "status": status,
        "reason_codes": ";".join(sorted(set(map(str, reason_codes)))),
        "input_sha256": input_sha256,
        "candidate_graph_count": str(candidate_graph_count),
        "details_json": json.dumps(
            details or {}, sort_keys=True, separators=(",", ":")
        ),
    }


def collect_v6_overlay_inputs(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Return deduplicated candidates and quarantine rows by chemical entity."""
    target_records = [row for row in records if row.get("tool") == _TARGET_TOOL]
    if not target_records:
        raise ValueError("result set contains no cycpep_master_v6 records")
    seen_cases = set()
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for record in target_records:
        _validate_record(record)
        case_id = str(record["case_id"])
        if case_id in seen_cases:
            raise ValueError(f"duplicate V6 case_id: {case_id}")
        seen_cases.add(case_id)
        by_entity.setdefault(str(record["chemical_entity_id"]), []).append(record)

    candidates: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, str]] = []
    accepted_entities = 0
    inferred_entities = 0
    for entity_id in sorted(by_entity):
        entity_records = sorted(by_entity[entity_id], key=lambda row: row["case_id"])
        entity_candidates: list[dict[str, Any]] = []
        entity_quarantine: list[dict[str, str]] = []
        record_identity_sets: list[set[tuple[str, str, str]]] = []
        entity_has_bootstrap = False
        entity_unqualified = False
        for record in entity_records:
            bootstrap = _bootstrap(record)
            if bootstrap is None:
                continue
            entity_has_bootstrap = True
            for row in bootstrap.get("quarantine_rows", []):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"record {record['case_id']} has malformed quarantine row"
                    )
                normalized = {key: str(value) for key, value in row.items()}
                normalized["record_id"] = entity_id
                entity_quarantine.append(normalized)
            accepted_record = (
                record.get("status") == "success"
                and record.get("chemical_audit_status") == "passed"
                and bootstrap.get("status") == "ready"
            )
            if not accepted_record:
                entity_unqualified = True
                continue
            inferences = bootstrap.get("inference_results")
            if not isinstance(inferences, list) or not inferences:
                raise ValueError(
                    f"successful bootstrap record {record['case_id']} lacks inferences"
                )
            for inference in inferences:
                if not isinstance(inference, dict) or inference.get("status") != "unique":
                    raise ValueError(
                        f"successful bootstrap record {record['case_id']} has "
                        "a non-unique inference"
                    )
                entity_candidates.append(_candidate(inference, entity_id))

        if not entity_has_bootstrap:
            continue
        inferred_entities += 1
        if any(
            record.get("status") != "success"
            or record.get("chemical_audit_status") != "passed"
            for record in entity_records
        ):
            entity_unqualified = True
        for record in entity_records:
            bootstrap = _bootstrap(record)
            if not isinstance(bootstrap, dict) or bootstrap.get("status") != "ready":
                continue
            record_candidates = [
                _candidate(inference, entity_id)
                for inference in bootstrap.get("inference_results", [])
                if isinstance(inference, dict) and inference.get("status") == "unique"
            ]
            record_identity_sets.append({
                _candidate_identity(candidate) for candidate in record_candidates
            })
        identities = {_candidate_identity(row) for row in entity_candidates}
        if entity_unqualified or not identities:
            quarantine_rows.extend(entity_quarantine)
            quarantine_rows.append(_quarantine(
                entity_id=entity_id,
                status="not_supported",
                reason_codes=["ENTITY_PAIRED_RECORD_NOT_UNIFORMLY_AUDITED"],
                candidate_graph_count=len(identities),
                details={"case_ids": [row["case_id"] for row in entity_records]},
            ))
            continue
        grouped_resnames: dict[str, set[tuple[str, str, str]]] = {}
        for identity in identities:
            grouped_resnames.setdefault(identity[0], set()).add(identity)
        conflicts = sorted(
            name for name, values in grouped_resnames.items() if len(values) != 1
        )
        if conflicts:
            quarantine_rows.extend(entity_quarantine)
            quarantine_rows.append(_quarantine(
                entity_id=entity_id,
                status="rejected",
                reason_codes=["PAIRED_RECORD_MONOMER_CONFLICT"],
                candidate_graph_count=len(identities),
                details={"conflicting_pdb_resnames": conflicts},
            ))
            continue
        if not record_identity_sets or any(
            identity_set != record_identity_sets[0]
            for identity_set in record_identity_sets[1:]
        ):
            quarantine_rows.extend(entity_quarantine)
            quarantine_rows.append(_quarantine(
                entity_id=entity_id,
                status="rejected",
                reason_codes=["PAIRED_RECORD_MONOMER_SET_CONFLICT"],
                candidate_graph_count=len(identities),
                details={
                    "candidate_sets": [
                        sorted(identity_set) for identity_set in record_identity_sets
                    ]
                },
            ))
            continue
        deduplicated = {}
        for row in entity_candidates:
            identity = _candidate_identity(row)
            key = (*identity, str(row["input_sha256"]))
            deduplicated[key] = row
        candidates.extend(deduplicated[key] for key in sorted(deduplicated))
        quarantine_rows.extend(entity_quarantine)
        accepted_entities += 1

    # A PDB residue code is a process-wide alias.  Publishing two chemical
    # meanings for the same code would make reconstruction order-dependent, so
    # quarantine every affected entity rather than selecting one graph.
    identities_by_resname: dict[str, set[tuple[str, str, str]]] = {}
    for candidate in candidates:
        identity = _candidate_identity(candidate)
        identities_by_resname.setdefault(identity[0], set()).add(identity)
    cross_entity_conflicts = sorted(
        name for name, identities in identities_by_resname.items()
        if len(identities) > 1
    )
    if cross_entity_conflicts:
        affected_entities = sorted({
            str(candidate["source_entity_id"])
            for candidate in candidates
            if str(candidate["pdb_resname"]) in cross_entity_conflicts
        })
        candidates = [
            candidate for candidate in candidates
            if str(candidate["source_entity_id"]) not in affected_entities
        ]
        for entity_id in affected_entities:
            quarantine_rows.append(_quarantine(
                entity_id=entity_id,
                status="rejected",
                reason_codes=["CROSS_ENTITY_PDB_RESNAME_GRAPH_CONFLICT"],
                candidate_graph_count=sum(
                    len(identities_by_resname[name])
                    for name in cross_entity_conflicts
                ),
                details={
                    "conflicting_pdb_resnames": cross_entity_conflicts,
                    "affected_entity_ids": affected_entities,
                },
            ))
        accepted_entities -= len(affected_entities)

    summary = {
        "v6_record_count": len(target_records),
        "unique_case_count": len(seen_cases),
        "unique_entity_count": len(by_entity),
        "entities_with_local_inference": inferred_entities,
        "accepted_entity_count": accepted_entities,
        "candidate_observation_count": len(candidates),
        "quarantine_row_count": len(quarantine_rows),
        "cross_entity_pdb_resname_conflict_count": len(cross_entity_conflicts),
    }
    return candidates, quarantine_rows, summary


def publish_from_sealed_results(
    results_path: str | Path,
    run_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    existing_bundle: str | Path | None = None,
    command: Iterable[str] | None = None,
) -> Path:
    results = Path(results_path).resolve()
    run_manifest_file = Path(run_manifest_path).resolve()
    manifest = json.loads(run_manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") not in _SEALED_RUN_STATUSES:
        raise ValueError("run manifest does not declare sealed raw results")
    results_hash = _sha256(results)
    if manifest.get("raw_results_sha256") != results_hash:
        raise ValueError("raw results hash differs from the run manifest")
    records = _jsonl(results)
    if manifest.get("result_count") != len(records):
        raise ValueError("raw result cardinality differs from the run manifest")
    candidates, quarantine_rows, summary = collect_v6_overlay_inputs(records)
    provenance = {
        "collector": "cycpep_master.core.overlay_collector",
        "collector_schema_version": "1.0.0",
        "command": list(command or []),
        "raw_results_path": str(results),
        "raw_results_sha256": results_hash,
        "run_manifest_path": str(run_manifest_file),
        "run_manifest_sha256": _sha256(run_manifest_file),
        **summary,
    }
    return publish_overlay_bundle(
        output_dir,
        candidates,
        quarantine_rows,
        existing_bundle=existing_bundle,
        publication_provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish an immutable monomer overlay from sealed V6 JSONL"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-bundle", type=Path)
    args = parser.parse_args(argv)
    invocation = ["cycpep-overlay", *(argv if argv is not None else sys.argv[1:])]
    output = publish_from_sealed_results(
        args.results,
        args.run_manifest,
        args.output,
        existing_bundle=args.existing_bundle,
        command=invocation,
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
