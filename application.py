"""Shared application services for the CLI and GUI.

This module is deliberately thin: it validates user-facing arguments, calls
the existing scientific components, and normalizes their heterogeneous return
types into JSON-serializable operation results.  It contains no reconstruction,
comparison, conformer, or docking algorithms of its own.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


SUPPORTED_COORDINATE_SUFFIXES = (
    ".pdb",
    ".ent",
    ".pdb.gz",
    ".ent.gz",
    ".cif",
    ".mmcif",
    ".cif.gz",
    ".mmcif.gz",
)
SUPPORTED_DOCKING_COORDINATE_SUFFIXES = (".pdb", ".ent")
RECONSTRUCTION_PATHS = ("v6", "a", "b", "c", "e", "f", "g", "h")
REPRESENTATION_KINDS = (
    "map",
    "helm",
    "biln",
    "smiles",
    "exact_v1",
    "edge_v1",
    "legacy_v5",
)
AUDIT_KINDS = ("map", "helm", "biln", "smiles", "assembly_json", "pdb")
TEMPLATE_STRATEGIES = ("full", "off", "random_same_length", "nearest_morgan")


def json_ready(value: Any) -> Any:
    """Convert public result objects into deterministic JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
            if key != "mol_3d"
        }
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_ready(item) for item in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _monomer_resolution_hints(
    payload: Any,
    *,
    kind: str | None = None,
) -> tuple[str, ...]:
    from .core.monomer_resolution import monomer_symbol_hints

    return monomer_symbol_hints(payload, kind=kind)


def _needs_monomer_resolution_scope(
    context: Mapping[str, Any] | None,
) -> bool:
    from .core.monomer_resolution import (
        needs_monomer_resolution_scope,
    )

    return needs_monomer_resolution_scope(context)


def _runtime_monomer_context(
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return (
        context
        if context is not None
        else {"include_persistent_user": True}
    )


def _is_monomer_resolution_error(value: Any) -> bool:
    text = str(value or "").upper()
    return any(
        marker in text
        for marker in (
            "UNKNOWN_MONOMER",
            "UNKNOWN MONOMER",
            "MONOMER_ID_UNRESOLVED",
            "NO UNIFIED SYMBOL",
            "NO UNIFIED MONOMER",
            "NOT IN LIBRARY",
        )
    )


def _attach_monomer_resolution_layer(
    result: dict[str, Any],
    ledger: Mapping[str, Any],
    *,
    requested_artifact_status: str,
) -> dict[str, Any]:
    data = result.setdefault("data", {})
    data.setdefault("monomer_resolution", json_ready(ledger))
    if str(ledger.get("status")) != "partial":
        return result
    rows = data.get("results")
    if isinstance(rows, list):
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("status") in {"success", "partial"}
            ):
                continue
            row["strict_status"] = row.get("status")
            row["status"] = "success"
            row["support_status"] = "lower_rigor"
            row["qualified_success"] = False
            row["artifact_status"] = "PARTIAL"
            row["chemical_rigor"] = str(
                ledger.get("chemical_rigor") or "C1:H"
            )
            row["requested_artifact_status"] = (
                requested_artifact_status
            )
    if result.get("status") not in {"success", "partial"}:
        result["strict_status"] = result.get("status")
        result["status"] = "success"
    data.setdefault("artifact_status", "PARTIAL")
    data.setdefault(
        "chemical_rigor",
        str(ledger.get("chemical_rigor") or "C1:H"),
    )
    data.setdefault(
        "requested_artifact_status", requested_artifact_status
    )
    return result


def _is_scalar_collection_input(value: Any) -> bool:
    return isinstance(value, (str, bytes, Path))


def _operation_result(
    operation: str,
    status: str,
    *,
    data: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = {
        "operation": operation,
        "status": status,
        "data": json_ready(dict(data or {})),
    }
    if error:
        result["error"] = str(error)
    return result


def _exception_result(
    operation: str,
    exc: Exception,
    *,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _operation_result(
        operation,
        "failed",
        data=data,
        error=f"{type(exc).__name__}: {exc}",
    )


def _materialize_worker_rows(value: Any) -> tuple[list[Any], str | None]:
    """Materialize a worker collection without letting malformed rows escape.

    Batch workers are expected to return an iterable of row objects.  Treating
    a mapping or scalar as that iterable would turn its keys into apparently
    valid rows, so preserve it as one raw row and report a shape error instead.
    The caller can then return a structured partial/failed result with the
    original payload available for audit.
    """
    if value is None:
        return [], "worker returned no row collection"
    if isinstance(value, (str, bytes, Mapping)):
        return [value], "worker returned a non-collection row payload"
    try:
        return list(value), None
    except Exception as exc:
        return [value], f"worker row collection could not be materialized: {exc}"


def _path_identity(value: Any) -> str | None:
    """Return a canonical, case-insensitive identity for a coordinate path.

    The original spelling is intentionally not retained here: callers use
    this value only for identity comparisons and keep their input spelling for
    user-facing reports.  ``strict=False`` also lets batch-contract checks
    compare paths before a worker has created an output file.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        canonical = Path(text).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        # Keep identity checks useful for malformed or cyclic paths without
        # allowing a path-normalization failure to escape the application API.
        canonical = Path(os.path.abspath(os.path.normpath(text)))
    return str(canonical).replace("\\", "/").casefold()


def _paths_alias(first: Any, second: Any) -> bool:
    """Return whether two path spellings refer to the same filesystem file."""
    first_identity = _path_identity(first)
    second_identity = _path_identity(second)
    if first_identity is None or second_identity is None:
        return False
    if first_identity == second_identity:
        return True
    try:
        return os.path.samefile(os.fspath(first), os.fspath(second))
    except (FileNotFoundError, OSError, TypeError, ValueError):
        # ``samefile`` cannot compare a not-yet-created destination.  The
        # canonical-path comparison above still covers direct and symlink
        # aliases in that case.
        return False


def _path_basename_identity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1].casefold() if text else None


def _file_stem_identity(value: Any) -> str | None:
    basename = _path_basename_identity(value)
    if basename is None:
        return None
    return basename.rsplit(".", 1)[0] if "." in basename else basename


def _reconstruction_row_identity(
    row: Mapping[str, Any], expected_ids: Sequence[str]
) -> str | None:
    """Resolve a pipeline row to an expected source path.

    Current pipeline rows carry ``source_path``.  ``file`` is retained as a
    compatibility fallback for older workers, but only when its basename maps
    to exactly one requested path; ambiguous basenames remain invalid.
    """
    if row.get("source_path") is not None:
        return _path_identity(row.get("source_path"))
    file_name = _path_basename_identity(row.get("file"))
    if file_name is None:
        return None
    candidates = [
        expected
        for expected in expected_ids
        if _path_basename_identity(expected) == file_name
    ]
    return candidates[0] if len(candidates) == 1 else file_name


def _reconstruction_row_report_identity(
    row: Mapping[str, Any], _expected_ids: Sequence[str]
) -> str | None:
    value = row.get("source_path")
    if value is None:
        value = row.get("file")
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _export_row_identity(
    row: Mapping[str, Any], _expected_ids: Sequence[str]
) -> str | None:
    value = row.get("name")
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _docking_row_identity(
    row: Mapping[str, Any], _expected_ids: Sequence[str]
) -> str | None:
    return _file_stem_identity(row.get("name"))


def _validate_batch_rows(
    rows: Sequence[Any],
    expected_ids: Sequence[str],
    *,
    identity_getter: Any,
    operation_label: str,
    collection_error: str | None = None,
    report_identity_getter: Any | None = None,
    report_expected_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Check batch row cardinality and identity without trusting worker order."""
    expected = list(expected_ids)
    actual: list[str | None] = []
    reported_expected = list(report_expected_ids or expected)
    reported_actual: list[str | None] = []
    malformed_indices: list[int] = []
    row_errors: list[str] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            malformed_indices.append(index)
            actual.append(None)
            reported_actual.append(None)
            row_errors.append(
                f"row {index} is not an object ({type(row).__name__})"
            )
            continue
        if row.get("malformed_worker_row") is True:
            malformed_indices.append(index)
            actual.append(None)
            reported_actual.append(None)
            row_errors.append(f"row {index} has invalid worker row shape")
            continue
        try:
            identity = identity_getter(row, expected)
        except Exception as exc:
            identity = None
            row_errors.append(f"row {index} identity extraction failed: {exc}")
        actual.append(identity)
        if report_identity_getter is None:
            reported_actual.append(identity)
        else:
            try:
                reported_actual.append(report_identity_getter(row, reported_expected))
            except Exception:
                reported_actual.append(None)
        if identity is None:
            row_errors.append(f"row {index} has no usable result identity")

    def repeated(values: Sequence[str | None]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in values:
            if value is not None and value in seen and value not in duplicates:
                duplicates.append(value)
            if value is not None:
                seen.add(value)
        return duplicates

    expected_duplicates = repeated(expected)
    actual_duplicates = repeated(actual)
    expected_set = set(expected)
    actual_set = {value for value in actual if value is not None}
    unknown = [
        value for value in dict.fromkeys(value for value in actual if value is not None)
        if value not in expected_set
    ]
    missing = [value for value in dict.fromkeys(expected) if value not in actual_set]

    def reported(
        identities: Sequence[str],
        normalized_values: Sequence[str | None],
        reported_values: Sequence[str | None],
    ) -> list[str]:
        values = []
        for identity in identities:
            index = next(
                i for i, candidate in enumerate(normalized_values)
                if candidate == identity
            )
            values.append(str(reported_values[index] or identity))
        return values

    reported_expected_duplicates = reported(
        expected_duplicates, expected, reported_expected
    )
    reported_actual_duplicates = reported(
        actual_duplicates, actual, reported_actual
    )
    reported_unknown = reported(unknown, actual, reported_actual)
    reported_missing = reported(missing, expected, reported_expected)
    order_mismatch = len(actual) != len(expected) or any(
        index >= len(expected) or value != expected[index]
        for index, value in enumerate(actual)
    )
    cardinality_mismatch = len(rows) != len(expected)

    problems: list[str] = []
    if collection_error:
        problems.append(collection_error)
    if cardinality_mismatch:
        problems.append(
            f"row cardinality mismatch: requested {len(expected)}, "
            f"returned {len(rows)}"
        )
    if malformed_indices:
        problems.append(
            "malformed worker rows at indices "
            + ", ".join(str(index) for index in malformed_indices)
        )
    if actual_duplicates:
        problems.append(
            "duplicate result identities: " + ", ".join(reported_actual_duplicates)
        )
    if expected_duplicates:
        problems.append(
            "duplicate requested identities: "
            + ", ".join(reported_expected_duplicates)
        )
    if unknown:
        problems.append("unknown result identities: " + ", ".join(reported_unknown))
    if missing:
        problems.append("missing result identities: " + ", ".join(reported_missing))
    if order_mismatch and not cardinality_mismatch:
        problems.append("result identity order differs from request order")
    problems.extend(row_errors)

    return {
        "contract_valid": not problems,
        "cardinality_mismatch": cardinality_mismatch,
        "identity_mismatch": bool(
            actual_duplicates
            or expected_duplicates
            or unknown
            or missing
            or order_mismatch
        ),
        "order_mismatch": order_mismatch,
        "duplicate_result_ids": reported_actual_duplicates,
        "duplicate_requested_ids": reported_expected_duplicates,
        "unknown_result_ids": reported_unknown,
        "missing_result_ids": reported_missing,
        "malformed_row_indices": malformed_indices,
        "expected_ids": reported_expected,
        "actual_ids": reported_actual,
        "error": f"{operation_label} result contract violation: "
        + "; ".join(problems)
        if problems
        else None,
    }


def _malformed_batch_row(operation_label: str, index: int, row: Any) -> dict[str, Any]:
    return {
        "row_index": index,
        "malformed_worker_row": True,
        "status": "failed",
        "error": (
            f"{operation_label} worker row {index} is malformed: "
            f"expected an object, got {type(row).__name__}"
        ),
        "raw_row": json_ready(row),
    }


def _cleanup_export_outputs(destination: Path, written: Any) -> None:
    """Remove only outputs produced/replaced by the current export call."""
    candidates = [destination]
    if written:
        try:
            candidates.append(Path(written))
        except (TypeError, ValueError):
            pass
    for candidate in candidates:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def _external_error_status(error: Any) -> str:
    """Preserve typed component failures across application facades.

    Component APIs historically return ``(value, error)`` pairs, so the
    application layer has to recover the status from the error text.  Only
    explicit, well-known markers are promoted; an unrecognised message stays
    ``failed`` rather than being guessed as a more specific state.
    """
    if isinstance(error, Mapping):
        declared = str(error.get("status") or "").strip().lower()
        if declared in {"invalid_input", "not_supported", "rejected", "timeout"}:
            return declared
        error = error.get("error") or error.get("message") or error
    lowered = str(error).strip().lower()
    for status in ("invalid_input", "not_supported", "rejected", "timeout"):
        if lowered == status or lowered.startswith(status + ":"):
            return status
    # Preserve failures reported by the process boundary even if stderr
    # happens to mention an output-validation term.
    if lowered.startswith("vina failed (exit") or lowered.startswith(
        "vina execution error:"
    ):
        return "failed"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if (
        "not_supported:" in lowered
        or lowered.startswith("not supported")
        or lowered.startswith("unsupported")
        or "not installed" in lowered
        or "not found" in lowered
    ):
        return "not_supported"
    if any(
        marker in lowered
        for marker in (
            "invariant mismatch",
            "malformed pdbqt",
            "rejected emitted pdbqt",
            "output validation failed",
            "vina produced no output file",
            "vina output is empty",
            "vina output is not minimally parseable",
            "vina output model ",
            "vina ligand/output invariant parse failed",
            "vina ligand/output torsion-tree validation failed",
            "vina ligand input must contain exactly one atom model",
            "failed to parse affinity from vina output",
            "rejected:",
        )
    ):
        return "rejected"
    if (
        "vina output path must not alias" in lowered
        or "cannot compare vina input/output paths" in lowered
        or lowered.startswith("invalid ")
        or lowered.startswith("invalid:")
        or lowered.startswith("invalid_")
        or lowered.startswith("input is invalid")
        or "must be " in lowered
    ):
        return "invalid_input"
    return "failed"


# Explicit scientific-inability markers from the coordinate exporters: a
# V5/V6 input-audit code prefix, the result-first terminal marker, the
# source-identity constraint codes, and the deterministic mapping-failure
# openings.  Anything else (disk, permission, process) stays infrastructure.
_SCIENTIFIC_INABILITY_PATTERNS = (
    re.compile(r"^V[56]_[A-Z0-9_]+:"),
    re.compile(r"^RESULT_FIRST_FAILED:"),
    re.compile(r"^SOURCE_IDENTITY_[A-Z_]+"),
    re.compile(r"^coordinate input preparation failed:"),
    re.compile(r"^incomplete PDB (atom|coordinate) mapping:"),
    re.compile(r"^3D embedding (failed|unavailable)"),
)


def _is_scientific_export_inability(error: Any) -> bool:
    """True when an export failed for chemistry/qualification reasons only.

    These attempts must degrade to the richest diagnostic artifact instead
    of returning a product-level failure; infrastructure errors do not.
    """
    if _external_error_status(error) in {"not_supported", "rejected"}:
        return True
    text = str(error).strip()
    return any(pattern.match(text) for pattern in _SCIENTIFIC_INABILITY_PATTERNS)


def _reconstruction_handoff(
    value: Any,
    *,
    allow_candidate_smiles: bool = False,
) -> tuple[bool, str | None, dict[str, Any] | None, str | None]:
    """Extract a downstream SMILES and audit context from a unified result.

    The helper accepts the public dataclass, its JSON-ready payload, or the
    application-service envelope.  Legacy strings and paths are deliberately
    left untouched.  Quality labels are audit metadata, not a global
    downstream gate: any successful result with an explicit SMILES payload is
    handed to the requested exporter.  Graph-only results remain unsupported
    by SMILES-only exporters without being promoted to invented chemistry.
    """
    candidate: Mapping[str, Any] | None = None
    envelope_status: str | None = None
    if isinstance(value, Mapping):
        candidate = value
        nested = value.get("data")
        if (
            value.get("operation")
            in {
                "reconstruct_structure",
                "reconstruct_unified",
                "reconstruct_result_first",
            }
            and isinstance(nested, Mapping)
        ):
            if value.get("status") is not None:
                envelope_status = str(value.get("status"))
            candidate = nested
    elif all(
        hasattr(value, field)
        for field in ("status", "quality", "smiles", "warning_codes")
    ):
        fields = (
            "status",
            "quality",
            "result_origin",
            "source",
            "source_kind",
            "mode",
            "rigor",
            "chemical_rigor",
            "smiles",
            "graph",
            "candidate_smiles",
            "candidate_graph",
            "chemistry_candidates",
            "bond_order_inference",
            "candidate_rigor",
            "ambiguous",
            "selection_tied",
            "warnings",
            "warning_codes",
            "alternatives",
            "structure_profile",
            "provenance",
            "strict_status",
            "strict_result",
            "integrity_findings",
            "artifact_status",
            "qualification_status",
            "coordinate_evidence",
        )
        candidate = {
            field: getattr(value, field)
            for field in fields
            if hasattr(value, field)
        }

    if not isinstance(candidate, Mapping):
        return False, None, None, None
    if not (
        "status" in candidate
        and "quality" in candidate
        and ("smiles" in candidate or "graph" in candidate)
    ):
        return False, None, None, None

    def list_field(name: str) -> list[Any]:
        raw = candidate.get(name)
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return [raw]

    context = {
        "status": candidate.get("status"),
        "quality": candidate.get("quality"),
        "result_origin": (
            candidate.get("result_origin") or candidate.get("source")
        ),
        "source_kind": candidate.get("source_kind"),
        "mode": candidate.get("mode"),
        "rigor": candidate.get("rigor"),
        "chemical_rigor": candidate.get("chemical_rigor"),
        "smiles": candidate.get("smiles"),
        "graph": candidate.get("graph"),
        "ambiguous": bool(candidate.get("ambiguous", False)),
        "selection_tied": bool(candidate.get("selection_tied", False)),
        "warnings": list_field("warnings"),
        "warning_codes": list_field("warning_codes"),
        "alternatives": list_field("alternatives"),
        "structure_profile": candidate.get("structure_profile"),
        "candidate_smiles": candidate.get("candidate_smiles"),
        "candidate_graph": candidate.get("candidate_graph"),
        "chemistry_candidates": list_field("chemistry_candidates"),
        "bond_order_inference": candidate.get("bond_order_inference") or {},
        "candidate_rigor": candidate.get("candidate_rigor"),
        "provenance": candidate.get("provenance") or {},
        # Preserve the strict V6 decision alongside the result-first output.
        # Downstream exporters need this audit context even when a lower
        # quality result is handed off successfully.
        "strict_status": candidate.get("strict_status"),
        "strict_result": candidate.get("strict_result"),
        # Cross-scope receipt evidence: diagnostic/integrity state and the
        # ladder's own qualification axes flow through the handoff so
        # receipts never have to reconstruct (or invent) them.
        "integrity_findings": list_field("integrity_findings"),
        "artifact_status": candidate.get("artifact_status"),
        "qualification_status": candidate.get("qualification_status"),
        "coordinate_evidence": candidate.get("coordinate_evidence"),
    }
    provenance = context["provenance"]
    request_binding = (
        provenance.get("request_binding")
        if isinstance(provenance, Mapping)
        else None
    )
    if isinstance(request_binding, Mapping):
        context["request_binding"] = dict(request_binding)
    if envelope_status is not None:
        context["envelope_status"] = envelope_status
    # Canonical chemical C-axis for the handoff channel: prefer an
    # explicitly installed canonical label; otherwise derive
    # conservatively with the diagnostic downgrade (a handed-off
    # 'high' is bound/incomplete evidence, never C2:R).
    context["chemical_rigor"] = _chemical_rigor_label(
        context, diagnostic=True
    )
    context = json_ready(context)
    if envelope_status is not None and envelope_status != "success":
        return (
            True,
            None,
            context,
            f"reconstruction envelope status is {envelope_status!r}",
        )
    status = str(candidate.get("status") or "failed")
    if status != "success":
        return (
            True,
            None,
            context,
            f"reconstruction result status is {status!r}",
        )
    quality = str(candidate.get("quality") or "unknown")
    smiles = candidate.get("smiles")
    if (
        allow_candidate_smiles
        and (not isinstance(smiles, str) or not smiles.strip())
    ):
        candidate_smiles = candidate.get("candidate_smiles")
        if isinstance(candidate_smiles, str) and candidate_smiles.strip():
            context["handoff_smiles_role"] = "candidate"
            context["handoff_warning_code"] = (
                "CANDIDATE_SMILES_HANDOFF"
            )
            return True, candidate_smiles.strip(), context, None
    if not isinstance(smiles, str) or not smiles.strip():
        return (
            True,
            None,
            context,
            f"reconstruction quality {quality!r} has no SMILES payload; "
            "this downstream path cannot serialize a graph-only result",
        )
    return True, smiles.strip(), context, None


_QUALITY_RIGOR = {
    "exact": "L2:Q",
    "high": "L2:R",
    "medium": "L2:R",
    "candidate": "L2:H",
    "hypothesis": "L1:H",
    "topology": "L1:H",
    "partial": "L1:R",
    "raw": "L0:C",
}


def _artifact_rigor(context: Mapping[str, Any] | None) -> str:
    value = context or {}
    explicit = value.get("candidate_rigor") or value.get("rigor")
    if isinstance(explicit, str) and explicit:
        return explicit
    quality = str(value.get("quality") or "").lower()
    return _QUALITY_RIGOR.get(quality, "L0:NONE")


def _compact_reconstruction_context(
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = context or {}
    return {
        "status": value.get("status"),
        "quality": value.get("quality"),
        "rigor": value.get("rigor"),
        "candidate_rigor": value.get("candidate_rigor"),
        "result_origin": value.get("result_origin"),
        "strict_status": value.get("strict_status"),
        "ambiguous": bool(value.get("ambiguous")),
        "warning_codes": list(value.get("warning_codes") or []),
        "handoff_smiles_role": value.get("handoff_smiles_role"),
        "handoff_warning_code": value.get("handoff_warning_code"),
        "artifact_status": value.get("artifact_status"),
        "qualification_status": value.get("qualification_status"),
        "chemical_rigor": value.get("chemical_rigor"),
        "coordinate_evidence": value.get("coordinate_evidence"),
        "integrity_findings": list(
            value.get("integrity_findings") or []
        ),
        "chemistry_candidate_count": len(
            value.get("chemistry_candidates") or []
        ),
    }


def _write_companion_json(
    destination: Path,
    suffix: str,
    payload: Mapping[str, Any],
) -> Path:
    output = Path(str(destination) + suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=str(output.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                json_ready(payload),
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return output


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_fingerprint(path: str | Path) -> tuple[Any, ...]:
    source = Path(path).resolve()
    stat = source.stat()
    return (
        str(source),
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@lru_cache(maxsize=128)
def _sha256_file_cached_by_fingerprint(
    fingerprint: tuple[Any, ...],
) -> str:
    return _sha256_file(str(fingerprint[0]))


def _cached_sha256_file(path: str | Path) -> str:
    return _sha256_file_cached_by_fingerprint(
        _file_fingerprint(path)
    )


def clear_application_resource_cache() -> None:
    """Clear process-local hashes used only by capabilities metadata."""
    _sha256_file_cached_by_fingerprint.cache_clear()


def clear_caches() -> None:
    """Clear safe process-local resource caches."""
    from .paths._map_utils import _REGISTRY_LOCK

    with _REGISTRY_LOCK:
        clear_application_resource_cache()
        from .exact_v1 import reset_exact_v1_registry_cache
        from .core.derived_monomers import _unified_identities
        from .paths.residue_template_factory import _template_from_values
        from .docking.torsion_prior import clear_torsion_prior_cache

        reset_exact_v1_registry_cache()
        _unified_identities.cache_clear()
        _template_from_values.cache_clear()
        clear_torsion_prior_cache()


def _evidence_digest(value: Any) -> str:
    payload = json.dumps(
        json_ready(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _full_inchikey_from_smiles(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(value)
        return Chem.MolToInchiKey(molecule) if molecule is not None else None
    except Exception:
        return None


_IDENTITY_QUALITIES = {
    "exact", "high", "medium", "candidate", "hypothesis",
}


def _expected_parent_inchikey(context: Any) -> tuple[str | None, str | None]:
    """Identity expectation at the rigor level the result can support.

    ``exact`` asserts a complete identity, so the full InChIKey must match.
    Candidate-level identities may carry stereo-silent SMILES while the 3D
    MOL2 read-back perceives stereocenters from geometry, so only the
    connectivity block is compared.  Topology/partial/raw outputs make no
    whole-molecule identity claim and set no gate.
    """
    if not isinstance(context, Mapping):
        return None, None
    quality = context.get("quality")
    if quality == "exact":
        return _full_inchikey_from_smiles(
            context.get("smiles") or context.get("candidate_smiles")
        ), None
    if quality in _IDENTITY_QUALITIES:
        full = _full_inchikey_from_smiles(
            context.get("smiles") or context.get("candidate_smiles")
        )
        return None, full
    return None, None


def _mol2_heavy_atom_indices_from_text(text: str) -> list[int]:
    """Return one-based non-hydrogen MOL2 atom ids from an ATOM block.

    Ids are the actual ``@<TRIPOS>ATOM`` id-column values parsed from the
    artifact, so hydrogen interleaving and writer ordering are honored;
    synthetic ranges are never used as provenance indices.
    """
    heavy: list[int] = []
    in_atoms = False
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        element = parts[5].split(".")[0].rstrip("0123456789")
        if element and element.upper() != "H":
            heavy.append(int(parts[0]))
    return heavy


def _mol2_heavy_atom_indices(path: str | Path) -> list[int]:
    """Read one-based non-hydrogen atom ids from a MOL2 ATOM block."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _mol2_heavy_atom_indices_from_text(text)


_MOL2_TIER_HEADER_CONVENTIONS = {
    "mol2_one_based",
    "legacy_zero_based",
}

#: Maximum number of leading ``#`` comment lines scanned for tier-header
#: fields; guards pathological files from being read as one giant header.
_MOL2_TIER_HEADER_MAX_LINES = 16


def _mol2_coordinate_ledger(path: str | Path) -> dict[str, Any]:
    """Parse MOL2 heavy atoms, tier header, and warning tokens once.

    The canonical ledger space is one-based MOL2 atom ids parsed from the
    artifact.  Headers carry ``atom_index_convention``; artifacts written
    before that marker are read in the legacy zero-based space and
    normalized up.  Absence of the header means the tier is unknown from
    the file itself; callers derive it from the export context instead of
    assuming X3.  Atom rows remain recoverable from malformed UTF-8, while
    a malformed or self-inconsistent header is not trusted.
    """
    result: dict[str, Any] = {
        "coordinate_level": None,
        "atom_index_convention": None,
        "mapped_heavy_atom_indices": [],
        "generated_heavy_atom_indices": [],
        "atom_coordinate_origins": {},
        "heavy_atom_indices": [],
        "warnings": [],
        "tier_evidence": {},
    }
    try:
        payload = Path(path).read_bytes()
    except OSError:
        return result

    heavy = _mol2_heavy_atom_indices_from_text(
        payload.decode("utf-8", errors="replace")
    )
    result["heavy_atom_indices"] = heavy

    try:
        text_lines = payload.decode("utf-8").splitlines()
    except UnicodeError:
        return result
    # Scan ALL leading comment lines (tier note, stereo-divergence note,
    # any future provenance note) so a MOL2 can carry coordinate tier,
    # embed/force-field evidence, and stereo-divergence warnings together
    # regardless of note ordering.
    fields: dict[str, str] = {}
    for line in text_lines[:_MOL2_TIER_HEADER_MAX_LINES]:
        if not line.startswith("#"):
            break
        for token in line.lstrip("#").split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields.setdefault(key, value)
    level = fields.get("coordinate_tier", "")
    if level not in {"X1", "X2", "X3"}:
        return result
    convention = fields.get("atom_index_convention", "legacy_zero_based")
    if convention not in _MOL2_TIER_HEADER_CONVENTIONS:
        return result
    heavy_space = (
        heavy if convention == "mol2_one_based"
        else [atom_id - 1 for atom_id in heavy]
    )
    valid_space = set(heavy_space)
    generated = sorted({
        int(value) for value in fields.get(
            "generated_heavy_atom_indices", ""
        ).split(",") if value.strip().isdigit()
    })
    if not set(generated) <= valid_space:
        return result
    mapped_field = fields.get("mapped_heavy_atom_indices")
    if mapped_field is not None:
        declared_mapped = [
            int(value) for value in mapped_field.split(",")
            if value.strip().isdigit()
        ]
        if (
            set(declared_mapped) & set(generated)
            or not set(declared_mapped) <= valid_space
        ):
            return result
    generated_set = set(generated)
    mapped = [index for index in heavy_space if index not in generated_set]
    tier_evidence: dict[str, Any] = {
        "header_atom_index_convention": convention,
        "fallback_origin": fields.get("fallback_origin") or None,
        "coordinate_source": fields.get("coordinate_source") or None,
    }
    for key in (
        "graph_divergence_detail",
        "optimization",
        "optimization_status",
        "embed_strategy",
        "force_field",
        "expected_full_inchikey",
        "observed_full_inchikey",
        "mapping_policy",
        "source_handoff_block_reason",
        "source_chain_id",
    ):
        if fields.get(key):
            tier_evidence[key] = fields[key]
    for key in (
        "etkdg_attempts",
        "isomorphism_count",
        "source_heavy_atoms",
        "candidate_heavy_atoms",
        "source_bonds",
        "candidate_bonds",
    ):
        raw = fields.get(key)
        if raw is not None and raw.isdigit():
            tier_evidence[key] = int(raw)
    for key in (
        "mmff_available",
        "graph_smiles_divergence",
        "stereo_realization_verified",
        "symmetry_equivalent_alternatives",
        "induced_edge_equality_verified",
        "source_handoff_blocked",
    ):
        if key in fields:
            tier_evidence[key] = fields[key] == "true"
    warnings: list[str] = []
    fallback_origin = tier_evidence.get("fallback_origin")
    if fallback_origin:
        warnings.append(f"COORDINATE_FALLBACK_ORIGIN_{fallback_origin.upper()}")
    if tier_evidence.get("graph_smiles_divergence"):
        warnings.append("GRAPH_SMILES_DIVERGENCE")
    if fields.get("stereo_roundtrip_divergence") == "true":
        warnings.append("STEREO_ROUNDTRIP_DIVERGENCE")
    if tier_evidence.get("etkdg_attempts") == 2:
        warnings.append("ETKDG_RANDOM_COORDS_RECOVERY")
    if str(tier_evidence.get("optimization_status") or "") in {
        "failed", "not_available",
    }:
        warnings.append(
            f"OPTIMIZATION_{tier_evidence['optimization_status'].upper()}"
        )
    if tier_evidence.get("mmff_available") is False:
        warnings.append("MMFF_UNAVAILABLE")
    if convention == "legacy_zero_based":
        # Normalize legacy zero-based header space up to the canonical
        # one-based ledger space.
        generated = [index + 1 for index in generated]
        mapped = [index + 1 for index in mapped]
    generated_set = set(generated)
    result.update({
        "coordinate_level": level,
        "atom_index_convention": "mol2_one_based",
        "mapped_heavy_atom_indices": mapped,
        "generated_heavy_atom_indices": generated,
        "atom_coordinate_origins": {
            str(index): (
                "generated" if index in generated_set else "source"
            )
            for index in mapped + generated
        },
        "warnings": warnings,
        "tier_evidence": tier_evidence,
    })
    return result


def _complete_ledger_from_mode(
    coordinate_mode: str,
    ledger: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Derive an honest coordinate ledger when the MOL2 carries no tier header.

    A header-less source-bound export is X3 by construction; a header-less
    regenerated export is fully generated, so every heavy atom is recorded
    as generated rather than left to an unsafe default.  Derived ledgers
    carry an explicit warning that the tier came from the export mode, not
    from an artifact header (criterion: lower-status evidence must be
    labeled, never silently assumed).
    """
    mode_to_level = {
        "source_bound": "X3",
        "template_completed": "X2",
        "regenerated": "X1",
    }
    level = ledger.get("coordinate_level")
    if level:
        return str(level), ledger
    level = mode_to_level.get(coordinate_mode)
    if level is None:
        return "unknown", ledger
    ledger = dict(ledger)
    ledger["coordinate_level"] = level
    ledger.setdefault("warnings", [])
    ledger.setdefault("tier_evidence", {})
    if (
        "COORDINATE_TIER_DERIVED_FROM_MODE"
        not in ledger["warnings"]
    ):
        ledger["warnings"] = list(ledger["warnings"]) + [
            "COORDINATE_TIER_DERIVED_FROM_MODE"
        ]
    if level == "X1" and not ledger.get("generated_heavy_atom_indices"):
        generated = list(ledger.get("heavy_atom_indices") or [])
        ledger["generated_heavy_atom_indices"] = generated
        ledger["atom_coordinate_origins"] = {
            str(index): "generated" for index in generated
        }
    return level, ledger


#: Index space expected by the frozen receipt writer
#: (``docking.mol2_input.build_validation_receipt`` validates provenance
#: indices against zero-based RDKit readback ``atom.GetIdx()``).  The
#: artifact ledger is canonical one-based MOL2 atom ids, so
#: :func:`_receipt_provenance_indices` converts at a single documented
#: point.  When the receipt writer adopts one-based ids
#: (``atom.GetIdx() + 1``), set this to ``"mol2_one_based"`` and the
#: conversion becomes an identity.
RECEIPT_ATOM_INDEX_CONVENTION = "rdkit_zero_based"


def _receipt_provenance_indices(
    ledger: Mapping[str, Any],
) -> tuple[list[int], list[int], dict[str, str]]:
    """Translate the one-based artifact ledger into receipt-writer space."""
    offset = 0 if RECEIPT_ATOM_INDEX_CONVENTION == "mol2_one_based" else -1
    mapped = [
        int(index) + offset
        for index in ledger.get("mapped_heavy_atom_indices") or []
    ]
    generated = [
        int(index) + offset
        for index in ledger.get("generated_heavy_atom_indices") or []
    ]
    origins: dict[str, str] = {}
    for key, value in (ledger.get("atom_coordinate_origins") or {}).items():
        try:
            index = int(key) + offset
        except (TypeError, ValueError):
            continue
        origins[str(index)] = str(value)
    return mapped, generated, origins


#: Canonical chemical C-axis, aligned with the staged ladder contract
#: (``result_first`` QUALITY_* tiers).  ``C3:Q`` is reserved for the
#: strict qualified exact product; a diagnostic channel may never emit it.
#: Legacy L labels are aliases of these canonical labels only.
_QUALITY_CHEMICAL_RIGOR = {
    "exact": "C3:Q",
    "high": "C2:R",
    "medium": "C2:H",
    "candidate": "C2:H",
    "hypothesis": "C1:H",
    "topology": "C1:H",
    "partial": "C1:R",
    "raw": "C0:C",
    "opaque": "C0:NONE",
}

_LEGACY_TO_CHEMICAL_RIGOR = {
    "L2:Q": "C3:Q",
    "L2:R": "C2:R",
    "L2:H": "C2:H",
    "L1:H": "C1:H",
    "L1:R": "C1:R",
    "L0:C": "C0:C",
    "L0:NONE": "C0:NONE",
}

#: Inverse alias map: wherever a canonical label and a legacy rigor label
#: are emitted together, the legacy alias is derived from the canonical
#: label so it can never out-claim it (e.g. canonical ``C2:H`` never
#: travels with an ``L2:R`` alias).
_CHEMICAL_RIGOR_TO_LEGACY = {
    "C3:Q": "L2:Q",
    "C2:R": "L2:R",
    "C2:H": "L2:H",
    "C1:H": "L1:H",
    "C1:R": "L1:R",
    "C0:C": "L0:C",
    "C0:NONE": "L0:NONE",
}

_CHEMICAL_RIGOR_LABEL_RE = re.compile(r"^C[0-3]:[A-Z]+$")


def _diagnostic_chemical_rigor_cap(label: str) -> str:
    """Cap a derived label at the diagnostic ceiling ``C2:H``.

    A diagnostic, degraded, or handed-off artifact can carry bound but
    incomplete identity evidence at most: ``C3:Q`` and ``C2:R`` downgrade
    to ``C2:H``; C1/C0 labels are never raised.
    """
    if label.startswith("C") and len(label) > 1 and label[1] >= "2":
        return "C2:H"
    return label


def _chemical_rigor_label(
    context: Mapping[str, Any] | None,
    legacy_rigor: str | None = None,
    *,
    diagnostic: bool = False,
) -> str:
    """Canonical chemical C-axis label for an export context.

    Prefers an explicitly installed canonical ``chemical_rigor`` (the
    ladder owns that label).  Only when absent is a label derived — from
    quality first, then the legacy L label — with the ladder's diagnostic
    downgrade semantics: diagnostic channels, failed/partial results, and
    non-success envelopes are capped at ``C2:H`` so recovered-chemistry
    (``C2:R``) and qualified-exact (``C3:Q``) claims survive only on the
    independently audited product itself.
    """
    value = context or {}
    explicit = value.get("chemical_rigor")
    if (
        isinstance(explicit, str)
        and _CHEMICAL_RIGOR_LABEL_RE.match(explicit)
    ):
        return explicit
    status = str(value.get("status") or "")
    envelope = value.get("envelope_status")
    degraded = (
        diagnostic
        or status in {"failed", "partial"}
        or (
            envelope is not None
            and str(envelope) != "success"
        )
    )
    quality = str(value.get("quality") or "").lower()
    label = _QUALITY_CHEMICAL_RIGOR.get(quality)
    if label is None:
        label = _LEGACY_TO_CHEMICAL_RIGOR.get(
            str(legacy_rigor or value.get("rigor") or ""), "C0:NONE"
        )
    if degraded:
        label = _diagnostic_chemical_rigor_cap(label)
    return label


def _legacy_rigor_alias(chemical_rigor: str, fallback: str) -> str:
    """Derive the legacy L-label alias of a canonical C-label."""
    return _CHEMICAL_RIGOR_TO_LEGACY.get(
        str(chemical_rigor), str(fallback)
    )


_COORDINATE_MODE_FROM_LEVEL = {
    "X3": "source_bound",
    "X2": "template_completed",
    "X1": "regenerated",
}

_FIDELITY_FROM_MODE = {
    "source_bound": "source_bound",
    "template_completed": "mixed",
    "regenerated": "regenerated",
}


def _coordinate_aliases(
    coordinate_level: str,
    coordinate_mode: str | None = None,
) -> tuple[str, str]:
    """Derive legacy coordinate aliases from the canonical X-axis tier."""
    mode = coordinate_mode or _COORDINATE_MODE_FROM_LEVEL.get(
        coordinate_level, "regenerated"
    )
    return mode, _FIDELITY_FROM_MODE.get(mode, "regenerated")


def _observed_mol2_identity(path: str | Path) -> dict[str, Any]:
    """Read a written MOL2 back and report its observed full InChIKey."""
    try:
        from .export.conformer import _mol2_roundtrip_full_inchikey

        content = Path(path).read_text(
            encoding="utf-8", errors="replace"
        )
        inchikey, error = _mol2_roundtrip_full_inchikey(content)
        return {"full_inchikey": inchikey, "error": error}
    except Exception as exc:
        return {"full_inchikey": None, "error": f"{type(exc).__name__}: {exc}"}


def _candidate_ambiguity_summary(
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Candidate-identity ambiguity facts for receipts and artifacts."""
    value = context or {}
    candidates = value.get("chemistry_candidates")
    alternatives = value.get("alternatives")
    candidate_count = None
    if isinstance(candidates, (list, tuple)):
        candidate_count = len(candidates)
    elif isinstance(alternatives, (list, tuple)):
        candidate_count = len(alternatives)
    return {
        "ambiguous": bool(value.get("ambiguous")),
        "chemistry_candidate_count": candidate_count,
        "candidate_rigor": value.get("candidate_rigor"),
        "handoff_smiles_role": value.get("handoff_smiles_role"),
        "selection_tied": bool(value.get("selection_tied")),
    }


def _strict_evidence_summary(
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Strict-ladder evidence facts for receipts and artifacts."""
    value = context or {}
    integrity_findings = value.get("integrity_findings")
    return {
        "strict_status": value.get("strict_status"),
        "qualification_status": value.get("qualification_status"),
        "integrity_finding_count": (
            len(integrity_findings)
            if isinstance(integrity_findings, (list, tuple))
            else None
        ),
        "warning_codes": list(value.get("warning_codes") or []),
        "result_origin": value.get("result_origin"),
    }


def _write_mol2_validation_receipt(
    mol2_path: str | Path,
    *,
    coordinate_mode: str,
    rigor: str,
    quality: str | None,
    source_heavy_atom_mapping_complete: bool,
    atom_provenance_complete: bool,
    source_input_sha256: str | None,
    evidence: Any,
    topology_class: str | None = None,
    macrocycle_ring_size: int | None = None,
    max_source_coordinate_delta_angstrom: float | None = None,
    expected_full_inchikey: str | None = None,
    expected_connectivity_inchikey: str | None = None,
    coordinate_level: str | None = None,
    mapped_heavy_atom_indices: list[int] | None = None,
    generated_heavy_atom_indices: list[int] | None = None,
    atom_coordinate_origins: Mapping[str, str] | None = None,
) -> Path:
    from .docking.mol2_input import (
        default_receipt_path,
        write_validation_receipt,
    )

    receipt = default_receipt_path(mol2_path)
    try:
        receipt.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"cannot clear stale MOL2 validation receipt: {exc}"
        ) from exc
    return write_validation_receipt(
        mol2_path,
        coordinate_mode=coordinate_mode,
        rigor=rigor,
        quality=quality,
        source_heavy_atom_mapping_complete=(
            source_heavy_atom_mapping_complete
        ),
        atom_provenance_complete=atom_provenance_complete,
        source_input_sha256=source_input_sha256,
        topology_class=topology_class,
        macrocycle_ring_size=macrocycle_ring_size,
        max_source_coordinate_delta_angstrom=(
            max_source_coordinate_delta_angstrom
        ),
        evidence_manifest_sha256=_evidence_digest(evidence),
        expected_full_inchikey=expected_full_inchikey,
        expected_connectivity_inchikey=expected_connectivity_inchikey,
        coordinate_level=coordinate_level,
        mapped_heavy_atom_indices=mapped_heavy_atom_indices,
        generated_heavy_atom_indices=generated_heavy_atom_indices,
        atom_coordinate_origins=atom_coordinate_origins,
    )


def _clear_mol2_validation_receipt(mol2_path: str | Path) -> None:
    from .docking.mol2_input import default_receipt_path

    receipt = default_receipt_path(mol2_path)
    try:
        receipt.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"cannot clear stale MOL2 validation receipt: {exc}"
        ) from exc


def _fallback_artifact_result(
    *,
    operation: str,
    destination: Path,
    requested_format: str,
    context: Mapping[str, Any] | None,
    reason: str | None,
) -> dict[str, Any]:
    value = context or {}
    graph = value.get("candidate_graph") or value.get("graph")
    candidates = list(
        value.get("chemistry_candidates")
        or value.get("alternatives")
        or []
    )
    # Fallback artifacts are diagnostic channels: canonical C-axis with the
    # diagnostic downgrade, legacy alias derived from the canonical label so
    # it can never out-claim it.  The X-axis value is passed through from
    # the ladder's own coordinate_evidence when it carries one; a fallback
    # artifact makes no independent coordinate claim.
    chemical_rigor = _chemical_rigor_label(value, diagnostic=True)
    rigor = _legacy_rigor_alias(chemical_rigor, _artifact_rigor(value))
    coordinate_evidence = value.get("coordinate_evidence")
    coordinate_level = (
        str(coordinate_evidence)
        if coordinate_evidence in {"X0", "X1", "X2", "X3"}
        else None
    )
    summary = _compact_reconstruction_context(value)
    if isinstance(graph, Mapping) and graph.get("atoms"):
        output = _write_companion_json(
            destination,
            ".graph.json",
            {
                "schema_version": "1.0.0-cycpep-artifact.1",
                "requested_format": requested_format,
                "requested_format_status": "degraded_format",
                "artifact": {
                    "format": "graph_json",
                    "rigor": rigor,
                    "chemical_rigor": chemical_rigor,
                    "coordinate_mode": "raw",
                    "coordinate_level": coordinate_level,
                    "role": "diagnostic",
                    "usable_for": ["inspection"],
                    "warnings": [reason] if reason else [],
                },
                "graph": graph,
                "chemistry_candidates": candidates,
                "reconstruction": summary,
            },
        )
        artifact = {
            "format": "graph_json",
            "path": str(output),
            "rigor": rigor,
            "chemical_rigor": chemical_rigor,
            "coordinate_mode": "raw",
            "coordinate_level": coordinate_level,
            "role": "diagnostic",
            "usable_for": ["inspection"],
            "warnings": [reason] if reason else [],
        }
        requested_status = "degraded_format"
    else:
        chemical_rigor = _chemical_rigor_label(
            {"quality": "opaque"}, diagnostic=True
        )
        rigor = _legacy_rigor_alias(chemical_rigor, "L0:NONE")
        coordinate_level = None
        output = _write_companion_json(
            destination,
            ".metadata.json",
            {
                "schema_version": "1.0.0-cycpep-artifact.1",
                "requested_format": requested_format,
                "requested_format_status": "metadata_only",
                "artifact": {
                    "format": "metadata",
                    "rigor": rigor,
                    "chemical_rigor": chemical_rigor,
                    "coordinate_mode": "none",
                    "coordinate_level": None,
                    "role": "diagnostic",
                    "usable_for": ["inspection"],
                    "warnings": [reason] if reason else [],
                },
                "chemistry_candidates": candidates,
                "reconstruction": summary,
            },
        )
        artifact = {
            "format": "metadata",
            "path": str(output),
            "rigor": rigor,
            "chemical_rigor": chemical_rigor,
            "coordinate_mode": "none",
            "coordinate_level": None,
            "role": "diagnostic",
            "usable_for": ["inspection"],
            "warnings": [reason] if reason else [],
        }
        requested_status = "metadata_only"
    return _operation_result(
        operation,
        "success",
        data={
            "requested_format": requested_format,
            "requested_format_status": requested_status,
            "chemical_rigor": chemical_rigor,
            "coordinate_level": coordinate_level,
            "artifacts": [artifact],
            "reconstruction": summary,
        },
    )


def _reconstruct_coordinate_for_downstream(
    source: str | Path,
    *,
    chain_id: str | list[str] | tuple[str, ...] | None,
    mode: str,
    minimum_macrocycle_ring_size: int,
    require_empty_persistent_overlay: bool,
    legacy_path: str | None = None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Resolve a coordinate input through the unified result-first facade.

    The legacy batch route is retained only as a compatibility fallback after
    the unified route cannot provide a serializable result.  This keeps old
    callers working while making the default downstream path result-first.
    """
    unified = reconstruct_structure(
        source,
        chain_id=chain_id,
        mode=mode,
        minimum_macrocycle_ring_size=int(minimum_macrocycle_ring_size),
        require_empty_persistent_overlay=bool(require_empty_persistent_overlay),
    )
    is_reconstruction, smiles, context, handoff_error = _reconstruction_handoff(
        unified
    )
    if is_reconstruction and handoff_error is None:
        return smiles, context, None

    # Preserve the historical explicit-path compatibility surface.  It is
    # never the first attempt and therefore cannot bypass the unified result.
    if legacy_path:
        try:
            legacy = reconstruct_coordinates(
                [source],
                path=legacy_path,
                chain_id=str(chain_id or "L"),
                require_empty_persistent_overlay=bool(
                    require_empty_persistent_overlay
                ),
            )
            rows = legacy.get("data", {}).get("results", [])
            row = rows[0] if rows else None
            if isinstance(row, Mapping) and row.get("status") == "success":
                legacy_payload = dict(row)
                legacy_payload.setdefault("status", "success")
                legacy_payload.setdefault("quality", "high")
                legacy_payload.setdefault("result_origin", "legacy_reconstruction")
                legacy_payload.setdefault("warnings", [])
                legacy_payload.setdefault("warning_codes", [])
                legacy_payload.setdefault("alternatives", [])
                legacy_payload.setdefault("provenance", {})
                return (
                    str(row["smiles"]) if row.get("smiles") else None,
                    json_ready(legacy_payload),
                    None if row.get("smiles") else (
                        "legacy reconstruction produced no serializable SMILES"
                    ),
                )
        except Exception:
            # Keep the unified error and provenance as the public explanation.
            pass

    return (
        None,
        context or {"status": "failed", "source": str(source)},
        handoff_error or "coordinate reconstruction produced no serializable SMILES",
    )


def discover_coordinate_files(directory: str | Path) -> list[str]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"coordinate directory does not exist: {root}")
    return [
        str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.name.lower().endswith(SUPPORTED_COORDINATE_SUFFIXES)
    ]


def discover_docking_coordinate_files(directory: str | Path) -> list[str]:
    """Return legacy PDB inputs accepted by the current docking stack."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"docking coordinate directory does not exist: {root}")
    return [
        str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file()
        and path.name.lower().endswith(SUPPORTED_DOCKING_COORDINATE_SUFFIXES)
    ]


def reconstruct_coordinates(
    inputs: Sequence[str | Path],
    *,
    path: str = "v6",
    chain_id: str = "L",
    target_chain_id: str = "R",
    run_admet: bool = False,
    compute_flexibility: bool = False,
    run_docking: bool = False,
    export_dir: str | Path | None = None,
    export_format: str = "mol2",
    csv_output: str | Path | None = None,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    allow_linear_topology: bool = False,
    fallback_policy: str = "strict_v6",
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the existing coordinate pipeline and preserve its audit payload.

    ``allow_linear_topology`` is forwarded to the V6 batch path. The legacy
    batch surface does not consume non-default geometry tolerances; callers
    must use ``reconstruct_structure`` / ``reconstruct_unified`` when those
    parameters differ from the package defaults.
    """
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import monomer_resolution_context

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                inputs, kind="coordinate"
            ),
        ) as resolution_ledger:
            result = reconstruct_coordinates(
                inputs,
                path=path,
                chain_id=chain_id,
                target_chain_id=target_chain_id,
                run_admet=run_admet,
                compute_flexibility=compute_flexibility,
                run_docking=run_docking,
                export_dir=export_dir,
                export_format=export_format,
                csv_output=csv_output,
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                allow_linear_topology=allow_linear_topology,
                fallback_policy=fallback_policy,
            )
        return _attach_monomer_resolution_layer(
            result,
            resolution_ledger,
            requested_artifact_status="RECONSTRUCTION_GRAPH_PARTIAL",
        )
    operation = "reconstruct"
    normalized_path = str(path).strip().lower()
    try:
        from .core.geometry_params import is_nondefault_geometry

        nondefault_geometry = is_nondefault_geometry(
            radius_multiplier, distance_ceiling
        )
    except ValueError as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if nondefault_geometry:
        return _operation_result(
            operation,
            "invalid_input",
            error=(
                "reconstruct_coordinates does not consume non-default "
                "geometry tolerances; use reconstruct_structure instead"
            ),
        )
    if normalized_path not in RECONSTRUCTION_PATHS:
        return _operation_result(
            operation,
            "invalid_input",
            error=f"unknown reconstruction path: {path!r}",
        )
    if export_format not in {"mol2", "sdf"}:
        return _operation_result(
            operation,
            "invalid_input",
            error="export_format must be 'mol2' or 'sdf'",
        )
    if _is_scalar_collection_input(inputs):
        return _operation_result(
            operation,
            "invalid_input",
            error="inputs must be a collection of coordinate paths",
        )
    try:
        source_paths = [str(Path(value)) for value in inputs]
    except (TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if not source_paths:
        return _operation_result(operation, "invalid_input", error="no inputs supplied")

    try:
        from .pipeline import run_batch

        entries = run_batch(
            source_paths,
            path=normalized_path,
            run_admet_flag=bool(run_admet),
            export_dir=str(export_dir) if export_dir else None,
            export_format=export_format,
            csv_output=str(csv_output) if csv_output else None,
            chain_id=str(chain_id),
            target_chain_id=str(target_chain_id),
            compute_rmsd=bool(compute_flexibility),
            run_docking=bool(run_docking),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
            allow_linear_topology=bool(allow_linear_topology),
            fallback_policy=fallback_policy,
        )
    except Exception as exc:
        return _exception_result(operation, exc)

    raw_entries, collection_error = _materialize_worker_rows(entries)
    expected_ids = [_path_identity(path) for path in source_paths]
    contract = _validate_batch_rows(
        raw_entries,
        expected_ids,
        identity_getter=_reconstruction_row_identity,
        operation_label="reconstruction",
        collection_error=collection_error,
        report_identity_getter=_reconstruction_row_report_identity,
        report_expected_ids=[str(path) for path in source_paths],
    )
    clean_entries = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping) or collection_error:
            clean_entries.append(_malformed_batch_row(operation, index, entry))
            continue
        clean = dict(entry)
        if "_row" in clean:
            clean["csv_row"] = clean.pop("_row")
        clean_entries.append(json_ready(clean))
    statuses = [str(entry.get("status", "failed")) for entry in clean_entries]
    requested_count = len(source_paths)
    returned_count = len(clean_entries)
    contract_error = contract["error"]
    if contract_error:
        status = (
            "partial"
            if any(status == "success" for status in statuses)
            else "failed"
        )
    elif statuses and all(status == "success" for status in statuses):
        status = "success"
    elif any(status == "success" for status in statuses):
        status = "partial"
    elif len(set(statuses)) == 1:
        status = statuses[0]
    else:
        status = "failed"
    return _operation_result(
        operation,
        status,
        data={
            "results": clean_entries,
            "count": len(clean_entries),
            "requested_count": requested_count,
            "returned_count": returned_count,
            "raw_rows": json_ready(raw_entries),
            **contract,
            "path": normalized_path,
            "chain_id": str(chain_id),
            "target_chain_id": str(target_chain_id),
            "csv_output": str(csv_output) if csv_output else None,
            "export_dir": str(export_dir) if export_dir else None,
        },
        error=contract_error,
    )


def reconstruct_multichain(
    coordinate_path: str | Path,
    *,
    chain_ids: Sequence[str] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose the existing diagnostic multi-chain assembly route."""
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                coordinate_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            result = reconstruct_multichain(
                coordinate_path,
                chain_ids=chain_ids,
            )
        return _attach_monomer_resolution_layer(
            result,
            resolution_ledger,
            requested_artifact_status="MULTICHAIN_GRAPH_PARTIAL",
        )
    operation = "reconstruct_multichain"
    if chain_ids is not None and _is_scalar_collection_input(chain_ids):
        return _operation_result(
            operation,
            "invalid_input",
            error="chain_ids must be a collection of chain identifiers",
        )
    try:
        normalized_chain_ids = list(chain_ids) if chain_ids is not None else None
    except TypeError as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if normalized_chain_ids == []:
        return _operation_result(
            operation,
            "invalid_input",
            error="chain_ids must not be an empty collection",
        )
    try:
        from .chemical_audit import audit_output_smiles
        from .paths import generate_multichain

        smiles, error = generate_multichain(
            str(coordinate_path), normalized_chain_ids
        )
        if error or not smiles:
            if _is_monomer_resolution_error(error):
                return _operation_result(
                    operation,
                    "success",
                    data={
                        "smiles": None,
                        "qualified_success": False,
                        "support_status": "symbolic_candidate",
                        "artifact_status": "PARTIAL",
                        "chemical_rigor": "C1:H",
                        "requested_artifact_status": (
                            "NOT_MATERIALIZABLE"
                        ),
                        "alternatives": [{
                            "kind": "coordinate",
                            "value": str(coordinate_path),
                            "claim_boundary": (
                                "source coordinates and symbolic residues"
                            ),
                        }],
                    },
                    error=error or "monomer resolution incomplete",
                )
            return _operation_result(
                operation,
                "failed",
                data={"qualified_success": False},
                error=error or "no molecular graph produced",
            )
        audit = audit_output_smiles(smiles)
        status = "success" if audit.accepted else "rejected"
        return _operation_result(
            operation,
            status,
            data={
                "smiles": smiles,
                "qualified_success": False,
                "support_status": "diagnostic",
                "audit": audit,
            },
            error=audit.reason,
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def reconstruct_result_first(
    input_path: str | Path,
    *,
    chain_id: str = "L",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    infer_bond_orders: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the result-first recovery ladder without changing strict V6."""
    operation = "reconstruct_result_first"
    resolution_stack = None
    try:
        from contextlib import ExitStack
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        from .result_first import reconstruct_structure

        resolution_stack = ExitStack()
        resolution_ledger = resolution_stack.enter_context(
            monomer_resolution_context(
                _runtime_monomer_context(monomer_context),
                required_symbols=_monomer_resolution_hints(
                    input_path, kind="coordinate"
                ),
            )
        )
        result = reconstruct_structure(
            input_path,
            chain_id=str(chain_id),
            minimum_macrocycle_ring_size=int(minimum_macrocycle_ring_size),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            infer_bond_orders=bool(infer_bond_orders),
        )
        payload = json_ready(result)
        from .core.rigor import rigor_from_result

        payload["rigor"] = rigor_from_result(payload).label
        payload["monomer_resolution"] = json_ready(
            resolution_ledger
        )
        status = str(payload.get("status") or "failed")
        error = None
        if status == "failed":
            provenance = payload.get("provenance") or {}
            error = str(
                provenance.get("failure_reason")
                or provenance.get("error")
                or "no readable structure"
            )
        return _operation_result(operation, status, data=payload, error=error)
    except Exception as exc:
        return _exception_result(operation, exc)
    finally:
        if resolution_stack is not None:
            resolution_stack.close()


def _run_unified_reconstruction(
    operation: str,
    source: str | Path,
    chain_id: str | list[str] | tuple[str, ...] | None = None,
    mode: str = "auto",
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolution_stack = None
    try:
        from contextlib import ExitStack
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        from .reconstruction import reconstruct_structure

        resolution_stack = ExitStack()
        resolution_ledger = resolution_stack.enter_context(
            monomer_resolution_context(
                _runtime_monomer_context(monomer_context),
                required_symbols=_monomer_resolution_hints(source),
            )
        )
        result = reconstruct_structure(
            source,
            chain_id=chain_id,
            mode=mode,
            minimum_macrocycle_ring_size=int(minimum_macrocycle_ring_size),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            allow_linear_topology=allow_linear_topology,
            **({
                "allow_chem_comp_evidence": allow_chem_comp_evidence,
                "chem_comp_evidence": chem_comp_evidence,
            } if allow_chem_comp_evidence or chem_comp_evidence is not None else {}),
        )
        payload = json_ready(result)
        if isinstance(payload, dict):
            payload.setdefault(
                "monomer_resolution",
                json_ready(resolution_ledger),
            )
        status = str(payload.get("status") or "failed")
        error = None
        if status == "failed":
            provenance = payload.get("provenance") or {}
            error = str(
                provenance.get("failure_reason")
                or provenance.get("error")
                or "no readable structure"
            )
        return _operation_result(operation, status, data=payload, error=error)
    except Exception as exc:
        return _exception_result(operation, exc)
    finally:
        if resolution_stack is not None:
            resolution_stack.close()


def reconstruct_structure(
    source: str | Path,
    chain_id: str | list[str] | tuple[str, ...] | None = None,
    mode: str = "auto",
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the unified structure-reconstruction orchestration layer.

    Accepts PDB/mmCIF coordinate paths or Sequence/HELM/MAP/BILN text with an
    optional explicit ``coordinate:/pdb:/mmcif:/sequence:/helm:/map:/biln:``
    prefix.  ``chain_id`` is an optional single chain or chain list; ``mode``
    is ``strict|auto|best_effort`` (the CLI spells ``best-effort``).

    ``allow_linear_topology`` (default ``False``) is forwarded to the strict
    V6 coordinate audit; a closure-less selected chain is then accepted as
    ``topology_class="linear"`` rather than rejected for lacking cyclization
    evidence.  The result-first ladder and legacy pipeline do not consume it.
    """
    return _run_unified_reconstruction(
        "reconstruct_structure",
        source,
        chain_id=chain_id,
        mode=mode,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
        allow_linear_topology=allow_linear_topology,
        monomer_context=monomer_context,
        allow_chem_comp_evidence=allow_chem_comp_evidence,
        chem_comp_evidence=chem_comp_evidence,
    )


def reconstruct_unified(
    source: str | Path,
    chain_id: str | list[str] | tuple[str, ...] | None = None,
    mode: str = "auto",
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility alias for ``reconstruct_structure``.

    Keeps the historical ``reconstruct_unified`` operation identifier so
    existing consumers keep receiving the same operation name.
    """
    return _run_unified_reconstruction(
        "reconstruct_unified",
        source,
        chain_id=chain_id,
        mode=mode,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
        allow_linear_topology=allow_linear_topology,
        monomer_context=monomer_context,
        allow_chem_comp_evidence=allow_chem_comp_evidence,
        chem_comp_evidence=chem_comp_evidence,
    )


def reconstruct_exact_v1(
    source: str | Path,
    *,
    chain_id: str = "L",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = True,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit exact_v1 only from qualified, unrepaired strict V6 evidence."""
    operation = "reconstruct_exact_v1"
    resolution_stack = None
    try:
        from contextlib import ExitStack
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        from .exact_v1 import EXACT, exact_v1_from_v6_result
        from .remediation_v6 import reconstruct_structure_fail_closed_v6

        resolution_stack = ExitStack()
        resolution_ledger = resolution_stack.enter_context(
            monomer_resolution_context(
                _runtime_monomer_context(monomer_context),
                required_symbols=_monomer_resolution_hints(
                    source, kind="coordinate"
                ),
            )
        )
        strict = reconstruct_structure_fail_closed_v6(
            source,
            chain_id=str(chain_id),
            minimum_macrocycle_ring_size=int(
                minimum_macrocycle_ring_size
            ),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
            allow_linear_topology=bool(allow_linear_topology),
            allow_chem_comp_evidence=bool(
                allow_chem_comp_evidence
            ),
            chem_comp_evidence=chem_comp_evidence,
        )
        exact = exact_v1_from_v6_result(strict)
        data = {
            "exact_v1": exact,
            "exactness_status": exact["exactness_status"],
            "reason_codes": list(exact["reason_codes"]),
            "graph_sha256": exact.get("graph_sha256"),
            "monomer_resolution": json_ready(resolution_ledger),
            "v6": {
                "status": strict.status,
                "support_status": strict.support_status,
                "qualified_success": bool(
                    strict.qualified_success
                ),
                "repair_codes": list(strict.repair_codes),
                "warning_codes": list(strict.warning_codes),
                "output_inchikey": strict.output_inchikey,
                "path_used": strict.path_used,
                "rejection_reason": strict.rejection_reason,
            },
        }
        if exact["exactness_status"] != EXACT:
            data.update({
                "artifact_status": "PARTIAL",
                "chemical_rigor": "C0:NONE",
                "requested_artifact_status": "EXACT_V1_ABSTAINED",
                "alternatives": [{
                    "kind": "v6_evidence",
                    "status": strict.status,
                    "support_status": strict.support_status,
                    "warning_codes": list(strict.warning_codes),
                }],
            })
            return _operation_result(
                operation,
                "success",
                data=data,
                error=(
                    strict.rejection_reason
                    or "strict V6 evidence cannot establish an exact "
                    "monomer-port graph"
                ),
            )
        return _operation_result(operation, "success", data=data)
    except (TypeError, ValueError) as exc:
        return _operation_result(
            operation, "rejected", error=str(exc)
        )
    except Exception as exc:
        return _exception_result(operation, exc)
    finally:
        if resolution_stack is not None:
            resolution_stack.close()
def convert_representation(
    source_kind: str,
    target_kind: str,
    payload: str | Mapping[str, Any],
    *,
    edge_max_rings: int = 3,
    edge_max_position: int = 32,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    operation = "convert"
    resolution_stack = None
    source = source_kind.strip().lower()
    target = target_kind.strip().lower()
    for name, value in (
        ("edge_max_rings", edge_max_rings),
        ("edge_max_position", edge_max_position),
    ):
        if (
            type(value) is not int
            or value < 1
        ):
            return _operation_result(
                operation,
                "invalid_input",
                error=f"{name} must be a positive integer",
            )
    if source not in REPRESENTATION_KINDS or target not in REPRESENTATION_KINDS:
        return _operation_result(
            operation,
            "invalid_input",
            error=(
                "source and target must be MAP, HELM, BILN, SMILES, "
                "exact_v1, edge_v1, or legacy_v5"
            ),
        )
    if source == "smiles" and target != "smiles":
        from .core.cyclic_peptide_graph import CyclicPeptideGraph

        abstained = CyclicPeptideGraph.abstain(
            ["GENERAL_SMILES_DECOMPOSITION_UNSUPPORTED"],
            source_kind="smiles",
        ).canonicalize()
        return _operation_result(
            operation,
            "success",
            data={
                "source_kind": source,
                "target_kind": target,
                "exact_v1": abstained,
                "exactness_status": abstained[
                    "exactness_status"
                ],
                "reason_codes": abstained["reason_codes"],
                "artifact_status": "PARTIAL",
                "chemical_rigor": "C1:H",
                "requested_artifact_status": "NOT_MATERIALIZABLE",
                "alternatives": [{
                    "kind": "SMILES",
                    "value": payload,
                    "claim_boundary": "atom graph only",
                }],
            },
            error="general SMILES-to-monomer decomposition is not supported",
        )
    try:
        from contextlib import ExitStack
        from .chemical_audit import audit_payload
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        from .exact_v1 import (
            ABSTAIN,
            biln_to_exact_v1,
            edge_v1_to_exact_v1,
            exact_v1_to_biln,
            exact_v1_to_edge_v1,
            exact_v1_to_helm,
            exact_v1_to_legacy_v5,
            exact_v1_to_map,
            exact_v1_to_smiles,
            helm_to_exact_v1,
            legacy_v5_to_exact_v1,
            map_to_exact_v1,
            parse_exact_v1_document,
        )
        from .representations import (
            biln_to_helm,
            biln_to_map,
            biln_to_smiles,
            helm_to_biln,
            helm_to_map,
            helm_to_smiles,
            map_to_biln,
            map_to_helm,
            map_to_smiles,
        )
        resolution_stack = ExitStack()
        resolution_ledger = resolution_stack.enter_context(
            monomer_resolution_context(
                _runtime_monomer_context(monomer_context),
                required_symbols=_monomer_resolution_hints(
                    payload, kind=source
                ),
            )
        )

        textual_kinds = {"map", "helm", "biln", "smiles"}
        audit = (
            audit_payload(source, payload)
            if source in textual_kinds
            else None
        )
        exact_document = None
        if source == "map":
            exact_document = map_to_exact_v1(payload)
        elif source == "helm":
            exact_document = helm_to_exact_v1(payload)
        elif source == "biln":
            exact_document = biln_to_exact_v1(payload)
        elif source == "edge_v1":
            exact_document = edge_v1_to_exact_v1(
                payload,
                max_rings=int(edge_max_rings),
                max_position=int(edge_max_position),
            )
        elif source == "legacy_v5":
            exact_document = legacy_v5_to_exact_v1(payload)
        elif source == "exact_v1":
            exact_document = parse_exact_v1_document(payload)

        if audit is not None and not audit.accepted:
            normalized_exact = bool(
                exact_document is not None
                and exact_document.get("exactness_status") != ABSTAIN
                and exact_document.get("normalization_codes")
            )
            abstained_exact = bool(
                exact_document is not None
                and exact_document.get("exactness_status") == ABSTAIN
            )
            if not normalized_exact and not abstained_exact:
                return _operation_result(
                    operation,
                    "rejected",
                    data={
                        "audit": audit,
                        "exact_v1": exact_document,
                    },
                    error=audit.reason,
                )

        if exact_document is not None and (
            exact_document.get("exactness_status") == ABSTAIN
        ):
            return _operation_result(
                operation,
                "success",
                data={
                    "source_kind": source,
                    "target_kind": target,
                    "exact_v1": exact_document,
                    "exactness_status": ABSTAIN,
                    "reason_codes": exact_document.get(
                        "reason_codes", []
                    ),
                    "artifact_status": "PARTIAL",
                    "chemical_rigor": "C1:H",
                    "requested_artifact_status": "NOT_MATERIALIZABLE",
                    "monomer_resolution": json_ready(
                        resolution_ledger
                    ),
                    "alternatives": [{
                        "kind": source,
                        "value": payload,
                        "claim_boundary": (
                            "symbolic monomer/connection topology only"
                        ),
                    }],
                },
                error=(
                    audit.reason
                    if audit is not None and not audit.accepted
                    else "exact_v1 abstained; conversion would require guessing"
                ),
            )

        if source == target:
            value: Any = (
                exact_document if source == "exact_v1" else payload
            )
        elif target == "exact_v1":
            if exact_document is None:
                raise KeyError((source, target))
            value = exact_document
        elif target == "edge_v1":
            if exact_document is None:
                raise KeyError((source, target))
            projection = exact_v1_to_edge_v1(
                exact_document,
                max_rings=int(edge_max_rings),
                max_position=int(edge_max_position),
                preserve_source_order=(source == "legacy_v5"),
            )
            if projection["status"] != "PROJECTED":
                return _operation_result(
                    operation,
                    "success",
                    data={
                        "source_kind": source,
                        "target_kind": target,
                        "exact_v1_graph_sha256": exact_document.get(
                            "graph_sha256"
                        ),
                        "projection": projection,
                        "artifact_status": "PARTIAL",
                        "chemical_rigor": resolution_ledger.get(
                            "chemical_rigor", "C3:S"
                        ),
                        "requested_artifact_status": (
                            "MODEL_PROJECTION_UNAVAILABLE"
                        ),
                        "alternatives": [{
                            "kind": "exact_v1",
                            "value": exact_document,
                        }],
                    },
                    error="exact_v1 is outside the edge_v1 model envelope",
                )
            value = projection["value"]
        elif target == "legacy_v5":
            if exact_document is None:
                raise KeyError((source, target))
            projection = exact_v1_to_legacy_v5(
                exact_document,
                max_rings=int(edge_max_rings),
                max_position=int(edge_max_position),
                preserve_source_order=(source == "edge_v1"),
            )
            if projection["status"] != "PROJECTED":
                return _operation_result(
                    operation,
                    "success",
                    data={
                        "source_kind": source,
                        "target_kind": target,
                        "exact_v1_graph_sha256": exact_document.get(
                            "graph_sha256"
                        ),
                        "projection": projection,
                        "artifact_status": "PARTIAL",
                        "chemical_rigor": resolution_ledger.get(
                            "chemical_rigor", "C3:S"
                        ),
                        "requested_artifact_status": (
                            "LEGACY_PROJECTION_UNAVAILABLE"
                        ),
                        "alternatives": [{
                            "kind": "exact_v1",
                            "value": exact_document,
                        }],
                    },
                    error="exact_v1 cannot be represented by legacy_v5",
                )
            value = projection["value"]
        elif exact_document is not None:
            serializers = {
                "map": exact_v1_to_map,
                "helm": exact_v1_to_helm,
                "biln": exact_v1_to_biln,
                "smiles": exact_v1_to_smiles,
            }
            value = serializers[target](exact_document)
        else:
            converters = {
                ("map", "helm"): map_to_helm,
                ("map", "biln"): map_to_biln,
                ("map", "smiles"): map_to_smiles,
                ("helm", "map"): helm_to_map,
                ("helm", "biln"): helm_to_biln,
                ("helm", "smiles"): helm_to_smiles,
                ("biln", "map"): biln_to_map,
                ("biln", "helm"): biln_to_helm,
                ("biln", "smiles"): biln_to_smiles,
            }
            value = converters[(source, target)](payload)

        output_audit = (
            audit_payload(target, value)
            if target in textual_kinds
            else None
        )
        qualified = bool(
            output_audit is None or output_audit.accepted
        )
        return _operation_result(
            operation,
            "success",
            data={
                "source_kind": source,
                "target_kind": target,
                "value": value,
                "input_audit": audit,
                "output_audit": output_audit,
                "exactness_status": (
                    exact_document.get("exactness_status")
                    if exact_document is not None
                    else None
                ),
                "exact_v1_graph_sha256": (
                    exact_document.get("graph_sha256")
                    if exact_document is not None
                    else None
                ),
                "artifact_status": (
                    "MATERIALIZED" if qualified else "PARTIAL"
                ),
                "qualified": qualified,
                "chemical_rigor": (
                    resolution_ledger.get(
                        "chemical_rigor", "C3:S"
                    )
                    if qualified
                    else "C2:R"
                ),
                "monomer_resolution": json_ready(
                    resolution_ledger
                ),
            },
            error=(
                output_audit.reason
                if output_audit is not None
                else None
            ),
        )
    except KeyError:
        return _operation_result(
            operation,
            "not_supported",
            error=f"conversion from {source} to {target} is not supported",
        )
    except (TypeError, ValueError) as exc:
        return _operation_result(operation, "rejected", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)
    finally:
        if resolution_stack is not None:
            resolution_stack.close()


def audit_chemistry(
    kind: str,
    *,
    payload: str | None = None,
    input_path: str | Path | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                payload if payload is not None else input_path,
                kind=str(kind).strip().lower(),
            ),
        ) as resolution_ledger:
            result = audit_chemistry(
                kind,
                payload=payload,
                input_path=input_path,
            )
        result.setdefault("data", {}).setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        return result
    operation = "audit"
    normalized = kind.strip().lower().replace("-", "_")
    try:
        from .chemical_audit import audit_payload, audit_pdb_file

        if payload is not None and input_path is not None:
            return _operation_result(
                operation,
                "invalid_input",
                error="provide either payload or input_path, not both",
            )
        if input_path is not None:
            if normalized != "pdb":
                text = Path(input_path).read_text(encoding="utf-8", errors="replace")
                audit = audit_payload(normalized, text)
            else:
                audit = audit_pdb_file(input_path)
        elif payload is not None:
            audit = audit_payload(normalized, payload)
        else:
            return _operation_result(
                operation, "invalid_input", error="payload or input_path is required"
            )
        issue_codes = {
            str(issue.code) for issue in getattr(audit, "issues", ())
        }
        if not audit.accepted and "UNKNOWN_MONOMER" in issue_codes:
            return _operation_result(
                operation,
                "success",
                data={
                    "kind": normalized,
                    "audit": audit,
                    "artifact_status": "PARTIAL",
                    "chemical_rigor": "C1:H",
                    "requested_artifact_status": (
                        "AUDIT_MONOMER_UNRESOLVED"
                    ),
                    "alternatives": [{
                        "kind": normalized,
                        "value": (
                            payload
                            if payload is not None
                            else str(input_path)
                        ),
                        "claim_boundary": (
                            "syntax and symbolic monomer inspection only"
                        ),
                    }],
                },
                error=audit.reason,
            )
        return _operation_result(
            operation,
            "success" if audit.accepted else "rejected",
            data={"kind": normalized, "audit": audit},
            error=audit.reason,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)


def compare_chemistry(
    left: str,
    right: str,
    *,
    mode: str = "strict",
) -> dict[str, Any]:
    operation = "compare"
    normalized = mode.strip().lower().replace("_", "-")
    try:
        if normalized == "strict":
            from .compare import compare_strict

            result = compare_strict(left, right)
            status = result.status
        elif normalized == "permissive":
            from .compare import compare

            matched, detail = compare(left, right)
            result = {"matched": matched, "detail": detail}
            status = "match" if matched else "mismatch"
        elif normalized in {"specified-stereo", "specified"}:
            from .compare.authoritative_stereo import compare_specified_stereo

            result = compare_specified_stereo(left, right)
            status = result.status
        else:
            return _operation_result(
                operation,
                "invalid_input",
                error="mode must be strict, permissive, or specified-stereo",
            )
        return _operation_result(
            operation,
            status,
            data={"mode": normalized, "comparison": result},
            error=getattr(result, "reason", None),
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def export_structure(
    source: Any,
    output_path: str | Path,
    *,
    source_kind: str = "smiles",
    output_format: str | None = None,
    chain_id: str = "L",
    path: str | None = "v6",
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    force_field: str = "mmff",
    num_confs: int = 10,
    random_seed: int = 42,
    monomer_context: Mapping[str, Any] | None = None,
    fallback_policy: str = "strict_v6",
) -> dict[str, Any]:
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                source, kind=source_kind
            ),
        ) as resolution_ledger:
            result = export_structure(
                source,
                output_path,
                source_kind=source_kind,
                output_format=output_format,
                chain_id=chain_id,
                path=path,
                reconstruction_mode=reconstruction_mode,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                force_field=force_field,
                num_confs=num_confs,
                random_seed=random_seed,
                fallback_policy=fallback_policy,
            )
        result.setdefault("data", {}).setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        return result
    operation = "export"
    try:
        from .max_coverage import coerce_policy
        fallback_policy = coerce_policy(fallback_policy)
    except Exception as exc:
        return _operation_result(
            operation,
            "invalid_input",
            data={
                "requested_format_status": "unavailable",
                "artifact_status": "opaque_input",
                "qualification_status": "not_assessable",
                "artifacts": [{
                    "format": "request",
                    "role": "diagnostic",
                    "status": "opaque_input",
                    "payload": {"fallback_policy": str(fallback_policy)},
                }],
            },
            error=f"invalid fallback policy: {exc}",
        )
    original_source = source
    original_source_kind = str(source_kind).strip().lower()
    destination = Path(output_path)
    normalized_source = source_kind.strip().lower()
    fmt = (output_format or destination.suffix.lstrip(".")).strip().lower()
    if fmt not in {"mol2", "sdf"}:
        return _operation_result(
            operation, "invalid_input", error="output format must be mol2 or sdf"
        )
    (
        is_reconstruction,
        reconstruction_smiles,
        reconstruction_context,
        handoff_error,
    ) = _reconstruction_handoff(source)
    resolved_source = source
    if is_reconstruction:
        if handoff_error:
            # Graph-only evidence (topology/partial/raw/bundle) cannot
            # honestly populate a bond-order-bearing format: degrade to the
            # richest diagnostic artifact instead of rejecting the request.
            return _fallback_artifact_result(
                operation=operation,
                destination=destination,
                requested_format=fmt,
                context=reconstruction_context,
                reason=(
                    f"{handoff_error}; {fmt.upper()} export requires a "
                    "serializable SMILES payload"
                ),
            )
        resolved_source = reconstruction_smiles
        normalized_source = "smiles"
    if normalized_source not in {"smiles", "coordinate", "pdb", "mmcif"}:
        return _operation_result(
            operation, "invalid_input", error="source_kind must be smiles or coordinate"
        )
    if normalized_source in {"coordinate", "pdb", "mmcif"} and _paths_alias(
        resolved_source, destination
    ):
        return _operation_result(
            operation,
            "invalid_input",
            error=(
                "coordinate input and output path must not alias: "
                f"source={resolved_source}, output={output_path}"
            ),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.unlink(missing_ok=True)
        if fmt == "mol2":
            _clear_mol2_validation_receipt(destination)
    except OSError as exc:
        return _operation_result(
            operation, "failed", error=f"cannot clear prior output: {exc}"
        )
    written = None
    coordinate_reconstruction_context = reconstruction_context
    try:
        from .export import smiles_to_mol2, smiles_to_sdf

        coordinate_mode = "regenerated"
        if normalized_source in {"coordinate", "pdb", "mmcif"}:
            if fmt == "mol2":
                source_bound_path = str(path or "v6").lower()
                if source_bound_path not in {
                    "a",
                    "e",
                    "v6",
                    "result_first",
                }:
                    return _operation_result(
                        operation,
                        "not_supported",
                        error=(
                            "coordinate-preserving MOL2 export has no "
                            f"source-bound materializer for path {path!r}"
                        ),
                    )
                from .export.conformer import pdb_to_mol2

                written, error = pdb_to_mol2(
                    str(resolved_source),
                    output_path=str(destination),
                    chain_id=str(chain_id),
                    path=source_bound_path,
                    fallback_policy=fallback_policy,
                    _minimum_macrocycle_ring_size=(
                        minimum_macrocycle_ring_size
                    ),
                    _require_empty_persistent_overlay=(
                        require_empty_persistent_overlay
                    ),
                )
                coordinate_mode = "source_bound"
            else:
                effective_legacy_path = (
                    path if path in RECONSTRUCTION_PATHS else None
                )
                (
                    smiles,
                    coordinate_reconstruction_context,
                    reconstruction_error,
                ) = _reconstruct_coordinate_for_downstream(
                    resolved_source,
                    chain_id=chain_id,
                    mode=reconstruction_mode,
                    minimum_macrocycle_ring_size=(
                        minimum_macrocycle_ring_size
                    ),
                    require_empty_persistent_overlay=(
                        require_empty_persistent_overlay
                    ),
                    legacy_path=effective_legacy_path,
                )
                if reconstruction_error or not smiles:
                    return _fallback_artifact_result(
                        operation=operation,
                        destination=destination,
                        requested_format=fmt,
                        context=coordinate_reconstruction_context,
                        reason=(
                            f"{reconstruction_error or 'coordinate reconstruction produced no SMILES'}; "
                            f"{fmt.upper()} export requires a serializable "
                            "SMILES payload"
                        ),
                    )
                coordinate_mode = "unified_reconstruction"
                written, error = smiles_to_sdf(
                    smiles,
                    output_path=str(destination),
                    force_field=force_field,
                    num_confs=num_confs,
                    random_seed=random_seed,
                )
        elif normalized_source == "smiles":
            exporter = smiles_to_mol2 if fmt == "mol2" else smiles_to_sdf
            written, error = exporter(
                resolved_source,
                output_path=str(destination),
                force_field=force_field,
                num_confs=num_confs,
                random_seed=random_seed,
            )
        else:
            return _operation_result(
                operation,
                "invalid_input",
                error="source_kind must be smiles or coordinate",
            )
        if error:
            error_status = _external_error_status(error)
            if _is_scientific_export_inability(error):
                # Scientific inability (no qualifying reconstruction) degrades
                # to the richest diagnostic artifact; only infrastructure
                # failures keep the failed envelope.
                _cleanup_export_outputs(destination, written)
                return _fallback_artifact_result(
                    operation=operation,
                    destination=destination,
                    requested_format=fmt,
                    context=coordinate_reconstruction_context,
                    reason=str(error),
                )
            _cleanup_export_outputs(destination, written)
            return _operation_result(
                operation,
                error_status,
                data=(
                    {"reconstruction": coordinate_reconstruction_context}
                    if coordinate_reconstruction_context is not None
                    else None
                ),
                error=error,
            )
        final_path = Path(written or destination)
        if not final_path.is_file() or final_path.stat().st_size == 0:
            _cleanup_export_outputs(destination, written)
            return _operation_result(
                operation, "failed", error="exporter produced no nonempty output"
            )
        coordinate_ledger = (
            _mol2_coordinate_ledger(final_path)
            if fmt == "mol2"
            else {
                "coordinate_level": None,
                "atom_index_convention": None,
                "mapped_heavy_atom_indices": [],
                "generated_heavy_atom_indices": [],
                "atom_coordinate_origins": {},
                "heavy_atom_indices": [],
                "warnings": [],
                "tier_evidence": {},
            }
        )
        coordinate_level, coordinate_ledger = _complete_ledger_from_mode(
            coordinate_mode, coordinate_ledger
        )
        if coordinate_level in {"X3", "X2", "X1"}:
            coordinate_mode = {
                "X3": "source_bound",
                "X2": "template_completed",
                "X1": "regenerated",
            }.get(coordinate_level, coordinate_mode)
        coordinate_mode, fidelity_status = _coordinate_aliases(
            coordinate_level, coordinate_mode
        )
        artifact_warnings = list(coordinate_ledger.get("warnings") or [])
        if coordinate_reconstruction_context is not None:
            for warning_code in (
                coordinate_reconstruction_context.get("handoff_warning_code"),
                coordinate_reconstruction_context.get("artifact_status"),
            ):
                if warning_code and warning_code not in artifact_warnings:
                    artifact_warnings.append(str(warning_code))
        data = {
            "requested_format": fmt,
            "requested_format_status": "fulfilled",
            "output_path": str(written or destination),
            "format": fmt,
            # Canonical axes: the chemical C-axis and the coordinate X-axis
            # are independent.  Legacy fields below (coordinate_mode,
            # fidelity_status, rigor, quality) remain as compatibility
            # aliases derived from the canonical pair.
            "coordinate_level": coordinate_level,
            "coordinate_mode": coordinate_mode,
            "fidelity_status": fidelity_status,
            "atom_index_convention": "mol2_one_based",
            "mapped_heavy_atom_indices": coordinate_ledger[
                "mapped_heavy_atom_indices"
            ],
            "generated_heavy_atom_indices": coordinate_ledger[
                "generated_heavy_atom_indices"
            ],
            "artifact_warnings": artifact_warnings,
        }
        if coordinate_reconstruction_context is not None:
            data["reconstruction"] = coordinate_reconstruction_context
        if fmt == "mol2":
            if coordinate_reconstruction_context is not None:
                receipt_context = coordinate_reconstruction_context
                rigor = _artifact_rigor(receipt_context)
                quality = receipt_context.get("quality")
            elif coordinate_mode == "source_bound":
                receipt_context = {
                    "path": path,
                    "coordinate_mode": coordinate_mode,
                    "quality": "exact" if path == "v6" else "high",
                }
                rigor = "L2:Q" if path == "v6" else "L2:R"
                quality = receipt_context["quality"]
            else:
                receipt_context = {
                    "source_kind": "smiles",
                    "coordinate_mode": coordinate_mode,
                    "quality": "hypothesis",
                }
                rigor = "L2:H"
                quality = "hypothesis"
            source_hash = None
            if isinstance(original_source, (str, Path)):
                if original_source_kind in {
                    "coordinate",
                    "pdb",
                    "mmcif",
                }:
                    candidate = Path(original_source)
                    if candidate.is_file():
                        source_hash = _sha256_file(candidate)
                else:
                    source_hash = hashlib.sha256(
                        str(original_source).encode("utf-8")
                    ).hexdigest()
            if is_reconstruction:
                expected_full_ik, expected_conn_ik = (
                    _expected_parent_inchikey(receipt_context)
                )
            else:
                expected_full_ik = _full_inchikey_from_smiles(
                    str(original_source)
                    if original_source_kind == "smiles"
                    else None
                )
                expected_conn_ik = None
            chemical_rigor = _chemical_rigor_label(
                receipt_context, legacy_rigor=rigor
            )
            # Legacy alias derived from the canonical label so it can
            # never out-claim it.
            rigor = _legacy_rigor_alias(chemical_rigor, rigor)
            receipt_mapped, receipt_generated, receipt_origins = (
                _receipt_provenance_indices(coordinate_ledger)
            )
            observed_identity = _observed_mol2_identity(final_path)
            # Verified source binding: propagate only what the handoff
            # context actually carries; never invent one.
            source_binding = None
            if isinstance(receipt_context, Mapping):
                binding = receipt_context.get("request_binding")
                if not isinstance(binding, Mapping):
                    provenance = receipt_context.get("provenance")
                    binding = (
                        provenance.get("request_binding")
                        if isinstance(provenance, Mapping)
                        else None
                    )
                if isinstance(binding, Mapping):
                    source_binding = dict(binding)
            receipt = _write_mol2_validation_receipt(
                final_path,
                coordinate_mode=coordinate_mode,
                rigor=rigor,
                quality=quality,
                source_heavy_atom_mapping_complete=(
                    coordinate_mode == "source_bound"
                ),
                atom_provenance_complete=True,
                source_input_sha256=source_hash,
                evidence={
                    "operation": operation,
                    "context": receipt_context,
                    "output_path": str(final_path),
                    "chemical_rigor": chemical_rigor,
                    "coordinate_level": coordinate_level,
                    "atom_index_convention": {
                        "artifact_header": "mol2_one_based",
                        "receipt_provenance": (
                            RECEIPT_ATOM_INDEX_CONVENTION
                        ),
                    },
                    "candidate_ambiguity": _candidate_ambiguity_summary(
                        receipt_context
                    ),
                    "strict_evidence": _strict_evidence_summary(
                        receipt_context
                    ),
                    "source_binding": source_binding,
                    "identity": {
                        "expected_full_inchikey": expected_full_ik,
                        "expected_connectivity_inchikey": expected_conn_ik,
                        "observed_full_inchikey": observed_identity.get(
                            "full_inchikey"
                        ),
                        "observed_identity_error": observed_identity.get(
                            "error"
                        ),
                    },
                    "fallback_warnings": artifact_warnings,
                    "tier_evidence": coordinate_ledger.get(
                        "tier_evidence"
                    )
                    or {},
                },
                max_source_coordinate_delta_angstrom=(
                    0.0 if coordinate_mode == "source_bound" else None
                ),
                expected_full_inchikey=expected_full_ik,
                expected_connectivity_inchikey=expected_conn_ik,
                coordinate_level=coordinate_level,
                mapped_heavy_atom_indices=receipt_mapped,
                generated_heavy_atom_indices=receipt_generated,
                atom_coordinate_origins=receipt_origins,
            )
            data["validation_receipt_path"] = str(receipt)
            data["validation_receipt_sha256"] = _sha256_file(receipt)
            # Legacy compatibility aliases of the canonical C/X axes.
            data["rigor"] = rigor
            data["quality"] = quality
            data["chemical_rigor"] = chemical_rigor
        data["artifacts"] = [{
            "format": fmt,
            "path": str(written or destination),
            "rigor": data.get("rigor", "L2:H"),
            "chemical_rigor": data.get(
                "chemical_rigor",
                _chemical_rigor_label(
                    coordinate_reconstruction_context,
                    legacy_rigor=data.get("rigor", "L2:H"),
                ),
            ),
            "coordinate_mode": coordinate_mode,
            "coordinate_level": coordinate_level,
            "role": "primary",
            "usable_for": ["inspection", "comparison"],
            "warnings": artifact_warnings,
        }]
        return _operation_result(
            operation,
            "success",
            data=data,
        )
    except Exception as exc:
        _cleanup_export_outputs(destination, written)
        return _exception_result(
            operation,
            exc,
            data=(
                {"reconstruction": coordinate_reconstruction_context}
                if coordinate_reconstruction_context is not None
                else None
            ),
        )


def _reconstruct_and_materialize_result_first_mol2(
    source: str | Path,
    destination: Path,
    *,
    chain_id: str,
    minimum_macrocycle_ring_size: int,
    require_empty_persistent_overlay: bool,
    fallback_policy: str,
    monomer_context: Mapping[str, Any] | None = None,
):
    """Reconstruct, bind, and materialize one result-first MOL2 (D1 fix).

    The monomer-resolution scope is established BEFORE reconstruction and
    binding and held through ``pdb_to_mol2``.  ``bind_reconstruction_result``
    stamps ``registry_epoch`` at bind time, while the fail-closed binding
    check inside ``pdb_to_mol2`` re-reads it under whatever scope is active
    at export; entering a scope between those two points bumps the epoch
    (scope activation is a registry-state mutation) and would fabricate a
    cross-registry mismatch.  Holding one scope across reconstruct -> bind
    -> export keeps the epochs identical without weakening the check: any
    real registry mutation between binding and export still fails closed.
    """
    from .core.monomer_resolution import (
        monomer_resolution_context,
        monomer_symbol_hints,
        needs_monomer_resolution_scope,
    )

    if not needs_monomer_resolution_scope(monomer_context):
        return _reconstruct_and_materialize_result_first_mol2_in_scope(
            source,
            destination,
            chain_id=chain_id,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=(
                require_empty_persistent_overlay
            ),
            fallback_policy=fallback_policy,
        )
    with monomer_resolution_context(
        (
            monomer_context
            if monomer_context is not None
            else {"include_persistent_user": True}
        ),
        required_symbols=monomer_symbol_hints(source, kind="coordinate"),
    ):
        return _reconstruct_and_materialize_result_first_mol2_in_scope(
            source,
            destination,
            chain_id=chain_id,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=(
                require_empty_persistent_overlay
            ),
            fallback_policy=fallback_policy,
        )


def _reconstruct_and_materialize_result_first_mol2_in_scope(
    source: str | Path,
    destination: Path,
    *,
    chain_id: str,
    minimum_macrocycle_ring_size: int,
    require_empty_persistent_overlay: bool,
    fallback_policy: str,
):
    """Scope-held body: requires an active monomer-resolution scope when
    one is needed; never enters a new scope after binding."""
    from .core.structure_io import prepare_coordinate_input
    from .export.conformer import pdb_to_mol2
    from .result_first import (
        bind_reconstruction_result,
        reconstruct_prepared_structure,
    )

    with prepare_coordinate_input(source, chain_id) as prepared:
        reconstruction = reconstruct_prepared_structure(
            prepared,
            minimum_macrocycle_ring_size=int(
                minimum_macrocycle_ring_size
            ),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
            infer_bond_orders=True,
        )
        bind_reconstruction_result(
            reconstruction,
            prepared,
            requested_chain_id=str(chain_id),
            minimum_macrocycle_ring_size=int(
                minimum_macrocycle_ring_size
            ),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
            infer_bond_orders=True,
        )
        written, error = pdb_to_mol2(
            str(prepared.pdb_path),
            output_path=str(destination),
            chain_id=str(prepared.chain_id),
            path="result_first",
            fallback_policy=fallback_policy,
            _prepared_coordinate_input=True,
            _coordinate_input_evidence=prepared.audit,
            _embedded_chem_comp_templates=prepared.audit.get(
                "embedded_chem_comp_templates"
            ),
            _strict_result=getattr(
                reconstruction, "strict_result", None
            ),
            _result_first_result=reconstruction,
            _minimum_macrocycle_ring_size=(
                minimum_macrocycle_ring_size
            ),
            _require_empty_persistent_overlay=(
                require_empty_persistent_overlay
            ),
        )
    return reconstruction, written, error


def export_best_available(
    source: Any,
    output_path: str | Path,
    *,
    source_kind: str = "smiles",
    output_format: str | None = None,
    chain_id: str = "L",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    force_field: str = "mmff",
    num_confs: int = 10,
    random_seed: int = 42,
    monomer_context: Mapping[str, Any] | None = None,
    fallback_policy: str = "max_coverage",
) -> dict[str, Any]:
    """Export the requested format or the best lower-rigor artifact.

    A successful operation means that at least one artifact was returned.
    ``requested_format_status`` states whether the requested chemical format
    was fulfilled, replaced by a graph artifact, or reduced to metadata.
    """
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                source, kind=source_kind
            ),
        ) as resolution_ledger:
            result = export_best_available(
                source,
                output_path,
                source_kind=source_kind,
                output_format=output_format,
                chain_id=chain_id,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                force_field=force_field,
                num_confs=num_confs,
                random_seed=random_seed,
                fallback_policy=fallback_policy,
            )
        result.setdefault("data", {}).setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        return result
    operation = "export_best_available"
    try:
        from .max_coverage import coerce_policy
        fallback_policy = coerce_policy(fallback_policy)
    except Exception as exc:
        return _operation_result(
            operation,
            "invalid_input",
            data={
                "requested_format_status": "unavailable",
                "artifact_status": "opaque_input",
                "qualification_status": "not_assessable",
                "artifacts": [{
                    "format": "request",
                    "role": "diagnostic",
                    "status": "opaque_input",
                    "payload": {"fallback_policy": str(fallback_policy)},
                }],
            },
            error=f"invalid fallback policy: {exc}",
        )
    destination = Path(output_path)
    normalized_source = str(source_kind).strip().lower()
    fmt = (output_format or destination.suffix.lstrip(".")).strip().lower()
    if fmt not in {"mol2", "sdf"}:
        return _operation_result(
            operation,
            "invalid_input",
            error="output format must be mol2 or sdf",
        )
    if fmt == "mol2":
        try:
            _clear_mol2_validation_receipt(destination)
        except Exception as exc:
            return _exception_result(operation, exc)
    reconstruction: Any | None = None
    context: dict[str, Any] | None = None
    source_bound_error: str | None = None
    if normalized_source in {"coordinate", "pdb", "mmcif"}:
        if _paths_alias(source, destination):
            return _operation_result(
                operation,
                "invalid_input",
                error=(
                    "coordinate input and output path must not alias: "
                    f"source={source}, output={output_path}"
                ),
            )
        written = None
        try:
            if fmt == "mol2":
                (
                    reconstruction,
                    written,
                    source_bound_error,
                ) = _reconstruct_and_materialize_result_first_mol2(
                    source,
                    destination,
                    chain_id=str(chain_id),
                    minimum_macrocycle_ring_size=int(
                        minimum_macrocycle_ring_size
                    ),
                    require_empty_persistent_overlay=bool(
                        require_empty_persistent_overlay
                    ),
                    fallback_policy=fallback_policy,
                    monomer_context=monomer_context,
                )
            else:
                from .result_first import (
                    reconstruct_structure as reconstruct_best,
                )

                reconstruction = reconstruct_best(
                    source,
                    chain_id=str(chain_id),
                    minimum_macrocycle_ring_size=int(
                        minimum_macrocycle_ring_size
                    ),
                    require_empty_persistent_overlay=bool(
                        require_empty_persistent_overlay
                    ),
                    infer_bond_orders=True,
                )
        except Exception as exc:
            return _fallback_artifact_result(
                operation=operation,
                destination=destination,
                requested_format=fmt,
                context={
                    "status": "failed",
                    "quality": None,
                    "warning_codes": ["RECONSTRUCTION_EXCEPTION"],
                },
                reason=f"{type(exc).__name__}: {exc}",
            )
        _, _, context, _ = _reconstruction_handoff(
            reconstruction, allow_candidate_smiles=True
        )
        if context is None:
            context = json_ready(reconstruction)
        try:
            from .core.rigor import rigor_from_result

            context["rigor"] = rigor_from_result(reconstruction).label
        except Exception:
            pass
        if fmt == "mol2":
            if (
                not source_bound_error
                and written
                and Path(written).is_file()
                and Path(written).stat().st_size > 0
            ):
                rigor = _artifact_rigor(context)
                coordinate_ledger = _mol2_coordinate_ledger(written)
                # Header-less result-first MOL2 on a coordinate source is
                # fully source-mapped by construction; headers override for
                # max-coverage X2/X1 outputs.
                coordinate_level, coordinate_ledger = (
                    _complete_ledger_from_mode("source_bound", coordinate_ledger)
                )
                coordinate_mode = {
                    "X3": "source_bound",
                    "X2": "template_completed",
                    "X1": "regenerated",
                }.get(coordinate_level, "regenerated")
                coordinate_mode, fidelity_status = _coordinate_aliases(
                    coordinate_level, coordinate_mode
                )
                expected_full_ik, expected_conn_ik = (
                    _expected_parent_inchikey(context)
                )
                chemical_rigor = _chemical_rigor_label(
                    context, legacy_rigor=rigor
                )
                rigor = _legacy_rigor_alias(chemical_rigor, rigor)
                receipt_mapped, receipt_generated, receipt_origins = (
                    _receipt_provenance_indices(coordinate_ledger)
                )
                artifact_warnings = list(
                    coordinate_ledger.get("warnings") or []
                )
                for warning_code in (
                    context.get("handoff_warning_code"),
                    context.get("artifact_status"),
                ):
                    if warning_code and warning_code not in artifact_warnings:
                        artifact_warnings.append(str(warning_code))
                source_binding = None
                reconstruction_provenance = getattr(
                    reconstruction, "provenance", None
                )
                if isinstance(reconstruction_provenance, dict):
                    binding = reconstruction_provenance.get(
                        "request_binding"
                    )
                    if isinstance(binding, dict):
                        source_binding = binding
                if source_binding is None:
                    context_binding = context.get("request_binding")
                    if isinstance(context_binding, dict):
                        source_binding = context_binding
                observed_identity = _observed_mol2_identity(written)
                receipt = _write_mol2_validation_receipt(
                    written,
                    coordinate_mode=coordinate_mode,
                    rigor=rigor,
                    quality=context.get("quality"),
                    source_heavy_atom_mapping_complete=(coordinate_level == "X3"),
                    atom_provenance_complete=True,
                    coordinate_level=coordinate_level,
                    mapped_heavy_atom_indices=receipt_mapped,
                    generated_heavy_atom_indices=receipt_generated,
                    atom_coordinate_origins=receipt_origins,
                    source_input_sha256=(
                        _sha256_file(source)
                        if Path(str(source)).is_file()
                        else None
                    ),
                    evidence={
                        "operation": operation,
                        "context": context,
                        "chain_id": str(chain_id),
                        "path": "result_first",
                        "chemical_rigor": chemical_rigor,
                        "coordinate_level": coordinate_level,
                        "atom_index_convention": {
                            "artifact_header": "mol2_one_based",
                            "receipt_provenance": (
                                RECEIPT_ATOM_INDEX_CONVENTION
                            ),
                        },
                        "source_binding": source_binding,
                        "candidate_ambiguity": _candidate_ambiguity_summary(
                            context
                        ),
                        "strict_evidence": _strict_evidence_summary(context),
                        "identity": {
                            "expected_full_inchikey": expected_full_ik,
                            "expected_connectivity_inchikey": (
                                expected_conn_ik
                            ),
                            "observed_full_inchikey": observed_identity.get(
                                "full_inchikey"
                            ),
                            "observed_identity_error": (
                                observed_identity.get("error")
                            ),
                        },
                        "fallback_warnings": artifact_warnings,
                        "tier_evidence": coordinate_ledger.get(
                            "tier_evidence"
                        )
                        or {},
                    },
                    topology_class=context.get("topology_class"),
                    macrocycle_ring_size=context.get(
                        "macrocycle_ring_size"
                    ),
                    max_source_coordinate_delta_angstrom=(
                        0.0 if coordinate_level == "X3" else None
                    ),
                    expected_full_inchikey=expected_full_ik,
                    expected_connectivity_inchikey=expected_conn_ik,
                )
                artifact = {
                    "format": "mol2",
                    "path": str(written),
                    "rigor": rigor,
                    "chemical_rigor": chemical_rigor,
                    "coordinate_mode": coordinate_mode,
                    "coordinate_level": coordinate_level,
                    "atom_index_convention": "mol2_one_based",
                    "generated_heavy_atom_indices": coordinate_ledger[
                        "generated_heavy_atom_indices"
                    ],
                    "role": "primary",
                    "usable_for": ["inspection", "comparison"],
                    "warnings": artifact_warnings,
                }
                return _operation_result(
                    operation,
                    "success",
                    data={
                        "requested_format": fmt,
                        "requested_format_status": "fulfilled",
                        # Canonical C/X axes; fidelity_status stays as a
                        # legacy alias of the coordinate mode.
                        "chemical_rigor": chemical_rigor,
                        "coordinate_level": coordinate_level,
                        "fidelity_status": fidelity_status,
                        "output_path": str(written),
                        "format": fmt,
                        "coordinate_mode": coordinate_mode,
                        "atom_index_convention": "mol2_one_based",
                        "mapped_heavy_atom_indices": coordinate_ledger[
                            "mapped_heavy_atom_indices"
                        ],
                        "generated_heavy_atom_indices": coordinate_ledger[
                            "generated_heavy_atom_indices"
                        ],
                        "artifact_warnings": artifact_warnings,
                        "validation_receipt_path": str(receipt),
                        "validation_receipt_sha256": _sha256_file(
                            receipt
                        ),
                        "artifacts": [artifact],
                        "reconstruction": _compact_reconstruction_context(
                            context
                        ),
                    },
                )
            return _fallback_artifact_result(
                operation=operation,
                destination=destination,
                requested_format=fmt,
                context=context,
                reason=(
                    source_bound_error
                    or "source-bound PDB-to-MOL2 export produced no output"
                ),
            )
        source = reconstruction
        normalized_source = "reconstruction"

    if normalized_source == "smiles" and isinstance(source, str):
        result = export_structure(
            source,
            destination,
            source_kind="smiles",
            output_format=fmt,
            force_field=force_field,
            num_confs=num_confs,
            random_seed=random_seed,
        )
        if (
            result["status"] == "success"
            and (result.get("data") or {}).get(
                "requested_format_status"
            ) == "fulfilled"
            and (result.get("data") or {}).get("output_path")
        ):
            # Provided-SMILES regeneration carries hypothesis-level
            # chemistry only; the legacy alias derives from the canonical
            # label so it can never out-claim it.
            chemical_rigor = _chemical_rigor_label(
                {"quality": "hypothesis"}
            )
            artifact = {
                "format": fmt,
                "path": result["data"]["output_path"],
                "rigor": _legacy_rigor_alias(chemical_rigor, "L2:H"),
                "chemical_rigor": chemical_rigor,
                "coordinate_mode": "regenerated",
                "coordinate_level": "X1",
                "role": "primary",
                "usable_for": ["inspection", "comparison"],
                "warnings": [],
            }
            result["operation"] = operation
            result["data"].update(
                {
                    "requested_format": fmt,
                    "requested_format_status": "fulfilled",
                    "fidelity_status": "regenerated",
                    "artifacts": [artifact],
                }
            )
            return result
        if result.get("status") == "success" and (
            (result.get("data") or {}).get("artifacts")
        ):
            result["operation"] = operation
            return result
        return _fallback_artifact_result(
            operation=operation,
            destination=destination,
            requested_format=fmt,
            context=None,
            reason=result.get("error"),
        )

    is_reconstruction, smiles, context, handoff_error = (
        _reconstruction_handoff(
            source, allow_candidate_smiles=True
        )
    )
    if is_reconstruction and smiles:
        result = export_structure(
            smiles,
            destination,
            source_kind="smiles",
            output_format=fmt,
            force_field=force_field,
            num_confs=num_confs,
            random_seed=random_seed,
        )
        if (
            result["status"] == "success"
            and (result.get("data") or {}).get(
                "requested_format_status"
            ) == "fulfilled"
            and (result.get("data") or {}).get("output_path")
        ):
            rigor = _artifact_rigor(context)
            chemical_rigor = _chemical_rigor_label(
                context, legacy_rigor=rigor
            )
            rigor = _legacy_rigor_alias(chemical_rigor, rigor)
            role = (
                "candidate"
                if (context or {}).get("handoff_smiles_role") == "candidate"
                else "primary"
            )
            warnings = [
                value
                for value in (
                    source_bound_error,
                    (context or {}).get("handoff_warning_code"),
                )
                if value
            ]
            artifact = {
                "format": fmt,
                "path": result["data"]["output_path"],
                "rigor": rigor,
                "chemical_rigor": chemical_rigor,
                "coordinate_mode": "regenerated",
                "coordinate_level": "X1",
                "role": role,
                "usable_for": ["inspection", "comparison"],
                "warnings": warnings,
            }
            result["operation"] = operation
            result["data"].update(
                {
                    "requested_format": fmt,
                    "requested_format_status": "fulfilled",
                    "fidelity_status": "regenerated",
                    "coordinate_mode": "regenerated",
                    "artifacts": [artifact],
                    "reconstruction": _compact_reconstruction_context(
                        context
                    ),
                }
            )
            return result
        if result.get("status") == "success" and (
            (result.get("data") or {}).get("artifacts")
        ):
            result["operation"] = operation
            result["data"].setdefault(
                "reconstruction", _compact_reconstruction_context(context)
            )
            return result
        handoff_error = result.get("error")
    return _fallback_artifact_result(
        operation=operation,
        destination=destination,
        requested_format=fmt,
        context=context,
        reason=handoff_error or source_bound_error,
    )


def batch_export_structures(
    smiles_items: Mapping[str, str] | Sequence[tuple[str, str]],
    output_dir: str | Path,
    *,
    output_format: str = "mol2",
    force_field: str = "mmff",
) -> dict[str, Any]:
    """Export a named SMILES collection through the existing batch exporter."""
    operation = "batch_export"
    fmt = str(output_format).strip().lower()
    if fmt not in {"mol2", "sdf"}:
        return _operation_result(
            operation, "invalid_input", error="output format must be mol2 or sdf"
        )
    if _is_scalar_collection_input(smiles_items):
        return _operation_result(
            operation,
            "invalid_input",
            error="smiles_items must be a mapping or collection of pairs",
        )
    try:
        items = list(
            smiles_items.items() if isinstance(smiles_items, Mapping) else smiles_items
        )
        normalized = [
            (str(name).strip(), str(smiles).strip()) for name, smiles in items
        ]
    except (TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if not normalized or any(not name or not smiles for name, smiles in normalized):
        return _operation_result(
            operation,
            "invalid_input",
            error="at least one non-empty name/SMILES pair is required",
        )
    try:
        from .export.conformer import batch_export

        worker_result = batch_export(
            normalized,
            str(output_dir),
            format=fmt,
            force_field=force_field,
        )
        raw_rows, collection_error = _materialize_worker_rows(worker_result)
        rows = []
        for index, raw_row in enumerate(raw_rows):
            if (
                collection_error
                or not isinstance(raw_row, (list, tuple))
                or len(raw_row) != 3
            ):
                rows.append(_malformed_batch_row(operation, index, raw_row))
                continue
            name, path, error = raw_row
            rows.append(
                {
                    "name": name,
                    "output_path": path,
                    "status": (
                        "success"
                        if path and not error
                        else _external_error_status(error)
                    ),
                    "error": error,
                }
            )
        expected_ids = [name for name, _smiles in normalized]
        contract = _validate_batch_rows(
            rows,
            expected_ids,
            identity_getter=_export_row_identity,
            operation_label="batch export",
            collection_error=collection_error,
        )
        success_count = sum(row["status"] == "success" for row in rows)
        requested_count = len(normalized)
        returned_count = len(rows)
        contract_error = contract["error"]
        if contract_error:
            status = "partial" if success_count else "failed"
        else:
            if success_count == len(rows):
                status = "success"
            elif success_count:
                status = "partial"
            else:
                failure_statuses = {
                    str(row.get("status") or "failed")
                    for row in rows
                }
                status = (
                    next(iter(failure_statuses))
                    if len(failure_statuses) == 1
                    else "failed"
                )
        return _operation_result(
            operation,
            status,
            data={
                "results": rows,
                "count": len(rows),
                "success_count": success_count,
                "requested_count": requested_count,
                "returned_count": returned_count,
                "raw_rows": json_ready(raw_rows),
                **contract,
                "output_dir": str(output_dir),
                "format": fmt,
            },
            error=contract_error,
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def find_conformer_template(
    generated_map: str,
    *,
    template_strategy: str = "full",
    random_seed: int = 42,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Query the packaged conformer library with its existing audit contract."""
    operation = "template_lookup"
    strategy = str(template_strategy).strip().lower()
    if strategy not in TEMPLATE_STRATEGIES:
        return _operation_result(
            operation,
            "invalid_input",
            error=f"template_strategy must be one of {list(TEMPLATE_STRATEGIES)}",
        )
    try:
        from .docking.template_library import find_template
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        audit: dict[str, Any] = {}
        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                generated_map, kind="map"
            ),
        ) as resolution_ledger:
            entry = find_template(
                generated_map,
                meta_out=audit,
                template_strategy=strategy,
                random_seed=int(random_seed),
            )
        if entry is None:
            audit_status = str(audit.get("status", "no_compatible_template"))
            status = (
                "rejected"
                if audit_status == "not_assessable"
                else "not_supported"
            )
            return _operation_result(
                operation,
                "success",
                data={
                    "template": None,
                    "audit": audit,
                    "artifact_status": "PARTIAL",
                    "requested_artifact_status": (
                        "TEMPLATE_UNAVAILABLE"
                    ),
                    "monomer_resolution": json_ready(
                        resolution_ledger
                    ),
                },
                error=str(audit.get("reason") or audit_status),
            )
        template = dict(entry)
        relative = str(template.get("pdb_path", "")).replace("\\", "/")
        if relative:
            resolved = (
                Path(__file__).resolve().parent / "data" / "templates" / relative
            )
            template["resolved_pdb_path"] = str(resolved)
            if not resolved.is_file() or resolved.stat().st_size == 0:
                return _operation_result(
                    operation,
                    "failed",
                    data={"template": template, "audit": audit},
                    error="template index references a missing or empty PDB file",
                )
        return _operation_result(
            operation,
            "success",
            data={
                "template": template,
                "audit": audit,
                "monomer_resolution": json_ready(
                    resolution_ledger
                ),
            },
        )
    except (TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)


def generate_template_conformers(
    smiles: str,
    generated_map: str,
    output_path: str | Path,
    *,
    n_conformers: int = 5,
    template_strategy: str = "full",
    random_seed: int = 42,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an audited template/ETKDG ensemble and serialize it as SDF."""
    operation = "template_conformers"
    destination = Path(output_path)
    strategy = str(template_strategy).strip().lower()
    if destination.suffix.lower() != ".sdf":
        return _operation_result(
            operation, "invalid_input", error="template ensembles require an .sdf output"
        )
    if strategy not in TEMPLATE_STRATEGIES:
        return _operation_result(
            operation,
            "invalid_input",
            error=f"template_strategy must be one of {list(TEMPLATE_STRATEGIES)}",
        )
    try:
        n_conformers = int(n_conformers)
    except (TypeError, ValueError, OverflowError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if n_conformers < 1:
        return _operation_result(
            operation, "invalid_input", error="n_conformers must be at least one"
        )
    try:
        destination.unlink(missing_ok=True)
    except OSError as exc:
        return _operation_result(
            operation, "failed", error=f"cannot clear prior SDF output: {exc}"
        )
    try:
        from rdkit import Chem
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        from .docking.template_library import generate_conformers

        audit: dict[str, Any] = {}
        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                generated_map, kind="map"
            ),
        ) as resolution_ledger:
            molecule, conformer_ids = generate_conformers(
                smiles,
                generated_map,
                n_conformers=int(n_conformers),
                meta_out=audit,
                template_strategy=strategy,
                random_seed=int(random_seed),
            )
        if molecule is None or not conformer_ids:
            failure = str(audit.get("failure_class", ""))
            status = "not_supported" if failure == "resource_limit" else "failed"
            detail = conformer_ids[0] if conformer_ids else audit.get("reason")
            return _operation_result(
                operation,
                status,
                data={
                    "audit": audit,
                    "conformer_count": 0,
                    "artifact_status": "PARTIAL",
                    "monomer_resolution": json_ready(
                        resolution_ledger
                    ),
                },
                error=str(detail or "no conformer generated"),
            )
        requested_count = int(n_conformers)
        generated_count = len(conformer_ids)
        if generated_count != requested_count:
            # A short conformer set is a cardinality mismatch, never a
            # top-level success; the partially generated SDF is not written.
            return _operation_result(
                operation,
                "failed",
                data={
                    "audit": audit,
                    "conformer_count": generated_count,
                    "requested_count": requested_count,
                    "cardinality_mismatch": True,
                    "conformer_ids": [int(value) for value in conformer_ids],
                    "monomer_resolution": json_ready(
                        resolution_ledger
                    ),
                },
                error=(
                    "template conformer cardinality mismatch: "
                    f"requested {requested_count}, generated {generated_count}"
                ),
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(destination))
        try:
            for ordinal, conformer_id in enumerate(conformer_ids, start=1):
                molecule.SetProp("CYCPEP_CONFORMER_INDEX", str(ordinal))
                writer.write(molecule, confId=int(conformer_id))
        finally:
            writer.close()
        if not destination.is_file() or destination.stat().st_size == 0:
            raise OSError("template conformer writer produced no nonempty output")
        record_count = destination.read_text(
            encoding="utf-8", errors="replace"
        ).count("$$$$")
        if record_count != len(conformer_ids):
            raise OSError(
                "template conformer SDF record count mismatch: "
                f"expected {len(conformer_ids)}, observed {record_count}"
            )
        return _operation_result(
            operation,
            "success",
            data={
                "output_path": str(destination),
                "conformer_count": len(conformer_ids),
                "conformer_ids": [int(value) for value in conformer_ids],
                "audit": audit,
                "monomer_resolution": json_ready(
                    resolution_ledger
                ),
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        return _operation_result(operation, "invalid_input", error=str(exc))
    except Exception as exc:
        destination.unlink(missing_ok=True)
        return _exception_result(operation, exc)


def conformer_statistics(
    smiles: str,
    *,
    num_confs: int = 50,
    random_seed: int = 42,
    force_field: str = "mmff",
    optimize: bool = True,
    energy_window: float | None = None,
    max_heavy_atoms: int | None = None,
) -> dict[str, Any]:
    operation = "conformers"
    try:
        from .export.conformer import (
            STABILITY_MAX_HEAVY_ATOMS,
            compute_conformer_ensemble_stats,
        )

        limit = STABILITY_MAX_HEAVY_ATOMS if max_heavy_atoms is None else max_heavy_atoms
        stats, error = compute_conformer_ensemble_stats(
            smiles,
            num_confs=int(num_confs),
            random_seed=int(random_seed),
            force_field=force_field,
            optimize=bool(optimize),
            energy_window=energy_window,
            max_heavy_atoms=limit,
        )
        if error or stats is None:
            status = "not_supported" if "too large" in str(error) else "failed"
            return _operation_result(operation, status, error=error)
        return _operation_result(operation, "success", data={"statistics": stats})
    except Exception as exc:
        return _exception_result(operation, exc)


def predict_admet(smiles_values: Sequence[str]) -> dict[str, Any]:
    operation = "admet"
    if _is_scalar_collection_input(smiles_values):
        return _operation_result(
            operation,
            "invalid_input",
            error="smiles_values must be a collection of SMILES strings",
        )
    values = [str(value).strip() for value in smiles_values if str(value).strip()]
    if not values:
        return _operation_result(operation, "invalid_input", error="no SMILES supplied")
    try:
        from .admet import run_admet

        predictions = run_admet(values)
        if (
            len(predictions) != len(values)
            or any(
                row.get("input_smiles") != value
                for row, value in zip(predictions, values)
            )
        ):
            return _operation_result(
                operation,
                "failed",
                data={"predictions": predictions},
                error="ADMET output cardinality or input identity mismatch",
            )
        errors = [row.get("error") for row in predictions if row.get("error")]
        if errors:
            status = (
                "not_supported"
                if all("not installed" in str(error).lower() for error in errors)
                else "failed"
            )
            return _operation_result(
                operation,
                status,
                data={"predictions": predictions},
                error=" | ".join(str(error) for error in errors),
            )
        return _operation_result(
            operation, "success", data={"predictions": predictions}
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def protonate_smiles(smiles: str) -> dict[str, Any]:
    operation = "protonate"
    if not isinstance(smiles, str) or not smiles.strip():
        return _operation_result(
            operation, "invalid_input", error="SMILES must be a nonempty string"
        )
    try:
        from rdkit import Chem

        from .docking.protonation import protonate_molecule_ph74

        input_molecule = Chem.MolFromSmiles(smiles)
        if input_molecule is None or input_molecule.GetNumAtoms() == 0:
            return _operation_result(
                operation, "invalid_input", error="input SMILES is not parseable"
            )
        output_molecule, microstate = protonate_molecule_ph74(input_molecule)
        output_smiles = Chem.MolToSmiles(Chem.RemoveHs(output_molecule))
        if output_molecule is None or output_molecule.GetNumAtoms() == 0:
            return _operation_result(
                operation,
                "failed",
                data={"input_smiles": smiles, "output_smiles": output_smiles},
                error="protonation produced an invalid SMILES",
            )
        return _operation_result(
            operation,
            "success",
            data={"input_smiles": smiles, "output_smiles": output_smiles,
                  "microstate": microstate},
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def protonate_mol2(
    mol2_path: str | Path,
    output_path: str | Path,
    *,
    policy: str = "physiological",
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write an explicit pH 7.4 microstate without replacing the source MOL2."""
    operation = "protonate_mol2"
    try:
        from rdkit import Chem
        from .docking.mol2_input import (
            default_receipt_path, load_validated_mol2, write_validation_receipt,
        )
        from .docking.protonation import protonate_molecule_ph74
        from .export.conformer import mol_to_mol2

        if policy != "physiological":
            return _operation_result(operation, "invalid_input", error="policy must be physiological (generic pH 7.4 rules)")
        source, destination = Path(mol2_path).resolve(), Path(output_path).resolve()
        output_receipt = default_receipt_path(destination)
        report_path = Path(str(destination) + ".microstate.json")
        if destination.suffix.lower() != ".mol2":
            return _operation_result(operation, "invalid_input", error="output must be a new .mol2 path")
        if any(path.exists() for path in (destination, output_receipt, report_path)) or _paths_alias(source, destination):
            return _operation_result(operation, "invalid_input", error="microstate output, receipt and report must not already exist or alias the input")
        parent = load_validated_mol2(source, receipt_path=receipt_path)
        molecule, microstate = protonate_molecule_ph74(parent.molecule)
        input_heavy = [a.GetIdx() for a in parent.molecule.GetAtoms() if a.GetAtomicNum() > 1]
        output_heavy = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1]
        if len(input_heavy) != len(output_heavy):
            raise ValueError("protonation changed the heavy-atom count")
        input_conf = parent.molecule.GetConformer()
        output_conf = molecule.GetConformer()
        if any(
            parent.molecule.GetAtomWithIdx(i).GetAtomicNum() != molecule.GetAtomWithIdx(j).GetAtomicNum()
            or max(abs(input_conf.GetAtomPosition(i)[k] - output_conf.GetAtomPosition(j)[k]) for k in range(3)) > 1e-6
            for i, j in zip(input_heavy, output_heavy)
        ):
            raise ValueError("protonation changed heavy-atom order or source coordinates")
        translated = dict(zip(input_heavy, output_heavy))
        old_receipt = parent.receipt
        mapped = [translated[i] for i in old_receipt["mapped_heavy_atom_indices"]]
        generated = [translated[i] for i in old_receipt["generated_heavy_atom_indices"]]
        origins = {str(i): "source" for i in mapped}
        origins.update({str(i): "generated" for i in generated})
        from rdkit.Chem import inchi
        expected = inchi.MolToInchiKey(molecule)
        if not expected:
            raise ValueError("protonated molecule has no full InChIKey")
        microstate = dict(microstate)
        microstate.update({
            "parent_mol2_path": str(source), "parent_mol2_sha256": parent.sha256,
            "parent_receipt_sha256": parent.receipt_sha256,
            "input_full_inchikey": parent.full_inchikey,
            "output_full_inchikey": expected,
            "input_smiles": Chem.MolToSmiles(Chem.RemoveHs(parent.molecule)),
            "output_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule)),
            "heavy_coordinates_preserved": True,
            "coordinate_level": parent.coordinate_level,
            "claim_boundary": "Generic pH 7.4 rule-based microstate; not experimental protonation, a pKa prediction, or a chemistry-accuracy upgrade.",
        })
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".protonate-", dir=str(destination.parent)) as temporary:
            temp_mol2 = Path(temporary) / destination.name
            content, error = mol_to_mol2(molecule)
            if error or not content:
                raise ValueError(error or "protonation produced no MOL2")
            content = (
                "# RDPEPPER_MICROSTATE_POLICY=physiological_ph7.4\n"
                "# RDPEPPER_WARNING=RULE_BASED_MICROSTATE_NOT_EXPERIMENTAL\n"
                f"# RDPEPPER_PARENT_MOL2_SHA256={parent.sha256}\n" + content
            )
            temp_mol2.write_text(content, encoding="utf-8", newline="\n")
            temp_receipt = write_validation_receipt(
                temp_mol2, coordinate_mode=parent.coordinate_mode,
                coordinate_level=parent.coordinate_level,
                rigor="L1:H", quality="hypothesis",
                source_heavy_atom_mapping_complete=parent.source_heavy_atom_mapping_complete,
                atom_provenance_complete=True,
                source_input_sha256=old_receipt.get("source_input_sha256"),
                topology_class=old_receipt.get("topology_class"),
                macrocycle_ring_size=old_receipt.get("macrocycle_ring_size"),
                max_source_coordinate_delta_angstrom=old_receipt.get("max_source_coordinate_delta_angstrom"),
                evidence_manifest_sha256=_evidence_digest(microstate),
                expected_full_inchikey=expected,
                mapped_heavy_atom_indices=mapped,
                generated_heavy_atom_indices=generated,
                atom_coordinate_origins=origins,
            )
            verified = load_validated_mol2(temp_mol2, receipt_path=temp_receipt)
            if Chem.MolToSmiles(Chem.RemoveHs(verified.molecule)) != microstate["output_smiles"]:
                raise ValueError("MOL2 roundtrip changed the assigned microstate")
            report_text = json.dumps(json_ready(microstate), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            temp_report = Path(temporary) / report_path.name
            temp_report.write_text(report_text, encoding="utf-8")
            for temporary_path, final_path in ((temp_mol2, destination), (temp_report, report_path), (temp_receipt, output_receipt)):
                with final_path.open("xb") as handle:
                    handle.write(temporary_path.read_bytes())
        return _operation_result(operation, "success", data={
            "output_path": str(destination), "validation_receipt_path": str(output_receipt),
            "microstate_report_path": str(report_path), "microstate": microstate,
            "coordinate_level": parent.coordinate_level,
            "formal_charge": verified.formal_charge,
            "full_inchikey": verified.full_inchikey,
            "warnings": ["RULE_BASED_MICROSTATE_NOT_EXPERIMENTAL"],
            "artifacts": [{"format": "mol2", "path": str(destination), "role": "primary",
                           "rigor": "L1:H", "coordinate_level": parent.coordinate_level,
                           "warnings": ["RULE_BASED_MICROSTATE_NOT_EXPERIMENTAL"]}],
        })
    except Exception as exc:
        return _exception_result(operation, exc)


def validate_mol2(
    mol2_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a MOL2 file and its mandatory sidecar receipt."""
    operation = "validate_mol2"
    try:
        from .docking.mol2_input import (
            Mol2ValidationError,
            load_validated_mol2,
            validation_error_status,
        )

        parent = load_validated_mol2(
            mol2_path, receipt_path=receipt_path
        )
        return _operation_result(
            operation,
            "success",
            data={
                "mol2_path": str(parent.path),
                "mol2_sha256": parent.sha256,
                "receipt_path": str(parent.receipt_path),
                "receipt_sha256": parent.receipt_sha256,
                "full_inchikey": parent.full_inchikey,
                "coordinate_mode": parent.coordinate_mode,
                "rigor": parent.rigor,
                "quality": parent.quality,
                "atom_count": parent.atom_count,
                "heavy_atom_count": parent.heavy_atom_count,
                "formal_charge": parent.formal_charge,
                "source_heavy_atom_mapping_complete": (
                    parent.source_heavy_atom_mapping_complete
                ),
                "atom_provenance_complete": (
                    parent.atom_provenance_complete
                ),
            },
        )
    except Mol2ValidationError as exc:
        return _operation_result(
            operation,
            validation_error_status(exc),
            error=f"validated MOL2 required: {exc}",
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def _verify_sdf_roundtrip(molecule, sdf_path: Path) -> float:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(
        str(sdf_path), removeHs=False, sanitize=True
    )
    records = [record for record in supplier]
    if len(records) != 1 or records[0] is None:
        raise ValueError("SDF roundtrip did not re-read exactly one molecule")
    roundtrip = records[0]
    if roundtrip.GetNumAtoms() != molecule.GetNumAtoms():
        raise ValueError("SDF roundtrip atom count differs from the source")
    if molecule.GetNumConformers() != 1 or roundtrip.GetNumConformers() != 1:
        raise ValueError("SDF roundtrip requires exactly one conformer")
    source_conformer = molecule.GetConformer()
    target_conformer = roundtrip.GetConformer()
    max_delta = 0.0
    for index in range(molecule.GetNumAtoms()):
        left = source_conformer.GetAtomPosition(index)
        right = target_conformer.GetAtomPosition(index)
        max_delta = max(
            max_delta,
            max(
                abs(left.x - right.x),
                abs(left.y - right.y),
                abs(left.z - right.z),
            ),
        )
    if max_delta > 0.001:
        raise ValueError(
            f"SDF roundtrip coordinates differ by {max_delta:.6f} A"
        )
    try:
        identity_verified = (
            Chem.MolToSmiles(Chem.RemoveHs(molecule))
            == Chem.MolToSmiles(Chem.RemoveHs(roundtrip))
        )
    except Exception as exc:
        raise ValueError(
            f"SDF roundtrip identity check failed: {exc}"
        ) from exc
    if not identity_verified:
        raise ValueError("SDF roundtrip identity differs from the source")
    return max_delta


def _export_read_mol2_sdf(
    molecule,
    export_sdf: str | Path,
    mol2_path: str | Path,
) -> dict[str, Any]:
    """Write the read molecule to a new SDF file and verify the roundtrip."""
    from rdkit import Chem

    output = Path(export_sdf).expanduser()
    if _paths_alias(output, mol2_path):
        raise ValueError(f"SDF export path aliases the MOL2 input: {output}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to overwrite existing SDF export: {output}")
    if molecule.GetNumConformers() != 1:
        raise ValueError("SDF export requires exactly one conformer")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp.sdf",
        dir=str(output.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        writer = Chem.SDWriter(str(temporary_path))
        try:
            writer.write(molecule)
        finally:
            writer.close()
        max_delta = _verify_sdf_roundtrip(molecule, temporary_path)
        # Exclusive create: a raced file at the destination must fail the
        # export instead of being overwritten.
        with output.open("xb") as handle:
            handle.write(temporary_path.read_bytes())
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "artifact_kind": "sdf_file",
        "path": str(output),
        "sha256": _cached_sha256_file(output),
        "roundtrip_max_coordinate_delta": max_delta,
        "roundtrip_identity_verified": True,
    }


def read_mol2(
    mol2_path: str | Path,
    *,
    compatibility: str = "rdkit_native",
    receipt_path: str | Path | None = None,
    export_sdf: str | Path | None = None,
) -> dict[str, Any]:
    """Read a MOL2 file under a compatibility mode and report the parse.

    ``rdkit_native`` (default) reproduces the stock RDKit MOL2 read;
    ``rdkit_charge_aware`` additionally restores formal charges declared in
    ``@<TRIPOS>UNITY_ATOM_ATTR``.  The data payload is the JSON reader
    report; no molecule object is placed in the envelope.
    """
    operation = "read_mol2"
    try:
        from .core.mol2_compat import load_mol2

        molecule, report = load_mol2(
            mol2_path,
            compatibility=compatibility,
            receipt_path=receipt_path,
        )
        data = dict(report)
        if export_sdf is not None:
            data["export_sdf"] = _export_read_mol2_sdf(
                molecule, export_sdf, mol2_path
            )
        return _operation_result(operation, "success", data=data)
    except (ValueError, OSError) as exc:
        return _operation_result(
            operation,
            "invalid_input",
            error=f"MOL2 read failed: {exc}",
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def _typed_evidence_from_mol2_audit(
    audit: Mapping[str, Any],
):
    from .core.artifacts import (
        ChemicalLevel,
        CoordinateLevel,
        CoordinateOrigin,
        EvidenceBasis,
        EvidenceProfile,
        FlexibilityLevel,
        FormatLevel,
    )

    rigor = str(audit.get("inherited_rigor") or "L0:NONE")
    level_text, _, basis_text = rigor.partition(":")
    try:
        level_number = int(level_text[1:])
    except (TypeError, ValueError):
        level_number = 0
    chemical_level = {
        0: ChemicalLevel.C0,
        1: ChemicalLevel.C1,
        2: ChemicalLevel.C2,
        3: ChemicalLevel.C3,
    }.get(max(0, min(3, level_number)), ChemicalLevel.C0)
    chemical_basis = {
        "S": EvidenceBasis.SPECIFIED,
        "Q": EvidenceBasis.QUALIFIED,
        "R": EvidenceBasis.REPAIRED,
        "H": EvidenceBasis.HYPOTHESIS,
        "C": EvidenceBasis.COORDINATE_ONLY,
        "NONE": EvidenceBasis.NONE,
    }.get(basis_text.upper(), EvidenceBasis.NONE)
    coordinate_mode = str(
        audit.get("parent_coordinate_mode") or "regenerated"
    )
    explicit_coordinate_level = str(
        audit.get("parent_coordinate_level") or ""
    )
    coordinate_origin, inferred_coordinate_level = {
        "source_bound": (
            CoordinateOrigin.SOURCE_BOUND,
            CoordinateLevel.X3,
        ),
        "template_completed": (
            CoordinateOrigin.TEMPLATE_BORROWED,
            CoordinateLevel.X2,
        ),
        "external_predicted": (
            CoordinateOrigin.EXTERNAL_PREDICTED,
            CoordinateLevel.X2,
        ),
        "experimental": (
            CoordinateOrigin.EXPERIMENTAL,
            CoordinateLevel.X3,
        ),
    }.get(
        coordinate_mode,
        (CoordinateOrigin.GENERATED, CoordinateLevel.X1),
    )
    coordinate_level = {
        "X1": CoordinateLevel.X1,
        "X2": CoordinateLevel.X2,
        "X3": CoordinateLevel.X3,
    }.get(explicit_coordinate_level, inferred_coordinate_level)
    return EvidenceProfile(
        chemical_level=chemical_level,
        chemical_basis=chemical_basis,
        coordinate_origin=coordinate_origin,
        coordinate_level=coordinate_level,
        format_level=FormatLevel.Q2,
        flexibility_level=FlexibilityLevel.F0,
    )


def _typed_mol2_parent_chain(
    mol2_path: str | Path,
    receipt_path: str | Path | None,
    audit: Mapping[str, Any],
    *,
    validated_parent=None,
) -> dict[str, Any]:
    from rdkit import Chem

    from .core.artifacts import (
        ArtifactStatus,
        ArtifactType,
        ChemicalGraphArtifact,
        ClaimBoundary,
        EvidenceProfile,
        InputArtifact,
        ValidatedMol2Artifact,
        artifact_payload_sha256,
        make_artifact_id,
        validate_artifact_identity,
        validate_inherited_evidence,
    )
    from .docking.mol2_input import load_validated_mol2

    parent = (
        validated_parent
        if validated_parent is not None
        else load_validated_mol2(
            mol2_path, receipt_path=receipt_path
        )
    )
    if (
        parent.sha256 != audit.get("parent_mol2_sha256")
        or parent.receipt_sha256
        != audit.get("parent_receipt_sha256")
    ):
        raise ValueError("typed MOL2 parent differs from PDBQT audit")
    input_payload = {
        "input_kind": "validated_mol2",
        "source_sha256": parent.sha256,
        "receipt_sha256": parent.receipt_sha256,
    }
    input_id = make_artifact_id(ArtifactType.INPUT, (), input_payload)
    input_artifact = InputArtifact(
        artifact_type=ArtifactType.INPUT,
        artifact_id=input_id,
        parent_artifact_ids=(),
        status=ArtifactStatus.MATERIALIZED,
        payload_sha256=artifact_payload_sha256(input_payload),
        evidence=EvidenceProfile(),
        provenance={
            "adapter": "validated_mol2_input",
            "receipt_sha256": parent.receipt_sha256,
        },
        claim_boundary=ClaimBoundary(
            allowed=("input provenance",),
            forbidden=("chemical identity", "3D accuracy"),
        ),
        input_kind="validated_mol2",
        source_sha256=parent.sha256,
        source_path=str(parent.path),
    )
    molecule = Chem.Mol(parent.molecule)
    smiles = Chem.MolToSmiles(
        Chem.RemoveHs(molecule),
        canonical=True,
        isomericSmiles=True,
    )
    mol2_evidence = _typed_evidence_from_mol2_audit(audit)
    graph_evidence = EvidenceProfile(
        chemical_level=mol2_evidence.chemical_level,
        chemical_basis=mol2_evidence.chemical_basis,
    )
    graph_payload = {
        "smiles": smiles,
        "full_inchikey": parent.full_inchikey,
        "formal_charge": parent.formal_charge,
        "topology_class": parent.receipt.get("topology_class"),
        "microstate_policy": "preserve_validated_parent_formal_state",
        "materializable": True,
        "source_mol2_sha256": parent.sha256,
    }
    graph_id = make_artifact_id(
        ArtifactType.CHEMICAL_GRAPH, (input_id,), graph_payload
    )
    graph_artifact = ChemicalGraphArtifact(
        artifact_type=ArtifactType.CHEMICAL_GRAPH,
        artifact_id=graph_id,
        parent_artifact_ids=(input_id,),
        status=ArtifactStatus.MATERIALIZED,
        payload_sha256=artifact_payload_sha256(graph_payload),
        evidence=graph_evidence,
        provenance={
            "adapter": "validated_mol2_graph",
            "monomer_decomposition": "not_asserted",
        },
        claim_boundary=ClaimBoundary(
            allowed=("validated atom graph identity",),
            forbidden=(
                "exact monomer-port decomposition",
                "experimental 3D accuracy",
            ),
        ),
        exact_v1={},
        smiles=smiles,
        full_inchikey=parent.full_inchikey,
        formal_charge=parent.formal_charge,
        topology_class=parent.receipt.get("topology_class"),
        microstate_policy="preserve_validated_parent_formal_state",
        materializable=True,
    )
    mol2_payload = {
        "conformer_id": "parent",
        "sha256": parent.sha256,
        "receipt_sha256": parent.receipt_sha256,
        "full_inchikey": parent.full_inchikey,
    }
    mol2_id = make_artifact_id(
        ArtifactType.VALIDATED_MOL2, (graph_id,), mol2_payload
    )
    mol2_artifact = ValidatedMol2Artifact(
        artifact_type=ArtifactType.VALIDATED_MOL2,
        artifact_id=mol2_id,
        parent_artifact_ids=(graph_id,),
        status=ArtifactStatus.MATERIALIZED,
        payload_sha256=artifact_payload_sha256(mol2_payload),
        evidence=mol2_evidence,
        provenance={
            "adapter": "validated_mol2_input",
            "receipt": dict(parent.receipt),
        },
        claim_boundary=ClaimBoundary(
            allowed=(
                "validated MOL2 identity",
                "validated coordinate provenance",
            ),
            forbidden=("docking benefit",),
        ),
        conformer_id="parent",
        path=str(parent.path),
        sha256=parent.sha256,
        receipt_path=str(parent.receipt_path),
        receipt_sha256=parent.receipt_sha256,
        full_inchikey=parent.full_inchikey,
    )
    validate_artifact_identity(
        input_artifact, payload=input_payload
    )
    validate_artifact_identity(
        graph_artifact, payload=graph_payload
    )
    validate_inherited_evidence(
        graph_evidence,
        mol2_evidence,
        allow_coordinate=True,
        allow_format=True,
    )
    validate_artifact_identity(
        mol2_artifact, payload=mol2_payload
    )
    return {
        "input_artifact": input_artifact.to_dict(),
        "chemical_graph": graph_artifact.to_dict(),
        "parent_mol2_artifact": mol2_artifact.to_dict(),
    }


def _validate_supplied_mol2_artifact(
    artifact: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    from .core.artifacts import (
        ArtifactStatus,
        ArtifactType,
        validate_artifact_record,
    )

    value = dict(artifact)
    if value.get("artifact_type") != ArtifactType.VALIDATED_MOL2.value:
        raise ValueError("parent artifact is not ValidatedMol2Artifact")
    if value.get("status") != ArtifactStatus.MATERIALIZED.value:
        raise ValueError("parent MOL2 artifact is not materialized")
    payload = {
        "conformer_id": value.get("conformer_id"),
        "path": value.get("path"),
        "sha256": value.get("sha256"),
        "receipt_path": value.get("receipt_path"),
        "receipt_sha256": value.get("receipt_sha256"),
        "full_inchikey": value.get("full_inchikey"),
    }
    validate_artifact_record(
        value,
        expected_type=ArtifactType.VALIDATED_MOL2,
        payload=payload,
    )
    if value.get("sha256") != audit.get("parent_mol2_sha256"):
        raise ValueError("parent MOL2 artifact hash differs from audit")
    if (
        value.get("receipt_sha256")
        != audit.get("parent_receipt_sha256")
    ):
        raise ValueError("parent MOL2 receipt hash differs from audit")
    if value.get("full_inchikey") != audit.get("parent_full_inchikey"):
        raise ValueError("parent MOL2 identity differs from audit")
    return value


def _pdbqt_qualification_state(
    role: str,
    audit: Mapping[str, Any],
) -> str:
    if role == "baseline" and audit.get("output_role") == "budgeted":
        return "PDBQT_BASELINE"
    if audit.get("torsdof_limit") is not None:
        return (
            "PDBQT_BUDGETED"
            if audit.get("budget_satisfied")
            else "PDBQT_BASELINE"
        )
    return "PDBQT_QUALIFIED"


def prepare_ligand_pdbqt_from_mol2(
    mol2_path: str | Path,
    output_path: str | Path,
    *,
    torsdof_limit: int | None = None,
    flexibility_mode: str = "balanced",
    torsion_prior_path: str | Path | None = None,
    ensemble_manifest_path: str | Path | None = None,
    ensemble_manifest_sha256: str | None = None,
    ensemble_size: int = 4,
    random_seed: int = 42,
    num_threads: int = 1,
    strict_budget: bool = False,
    receipt_path: str | Path | None = None,
    parent_mol2_artifact: Mapping[str, Any] | None = None,
    ensemble_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one ligand PDBQT strictly downstream of validated MOL2."""
    operation = "prepare_ligand_pdbqt_from_mol2"
    try:
        from .docking.mol2_input import (
            Mol2ValidationError,
            load_validated_mol2,
        )
        from .docking.mol2_pdbqt import mol2_to_ligand_pdbqt

        try:
            validated_parent = load_validated_mol2(
                mol2_path, receipt_path=receipt_path
            )
        except Mol2ValidationError as exc:
            return _operation_result(
                operation,
                "success",
                data={
                    "requested_format": "pdbqt",
                    "requested_format_status": "unavailable",
                    "qualification_status": "unqualified",
                    "reason": f"validated MOL2 required: {exc}",
                    "artifacts": ([{
                        "format": "mol2",
                        "path": str(mol2_path),
                        "role": "alternative",
                        "usable_for": ["inspection", "comparison"],
                        "warnings": [str(exc)],
                    }] if Path(mol2_path).is_file() else []),
                },
            )
        audit, error = mol2_to_ligand_pdbqt(
            validated_parent,
            output_path,
            torsdof_limit=torsdof_limit,
            flexibility_mode=flexibility_mode,
            torsion_prior_path=torsion_prior_path,
            ensemble_manifest_path=ensemble_manifest_path,
            ensemble_manifest_sha256=ensemble_manifest_sha256,
            ensemble_size=ensemble_size,
            random_seed=random_seed,
            num_threads=num_threads,
            strict_budget=strict_budget,
            receipt_path=receipt_path,
        )
        data: dict[str, Any] = {"audit": audit or {}}
        artifacts = []
        warning_codes = []
        if audit:
            from .core.artifacts import (
                ArtifactStatus,
                ArtifactType,
                ClaimBoundary,
                FormatLevel,
                PdbqtArtifact,
                artifact_payload_sha256,
                evidence_from_dict,
                inherit_evidence,
                make_artifact_id,
                validate_artifact_identity,
                validate_inherited_evidence,
            )
            from .docking.flexibility import (
                flexibility_artifact_from_audit,
            )
            from .docking.mol2_input import sha256_path

            audit_warnings = list(
                audit.get("flexibility_warning_codes") or []
            )
            if not audit.get("budget_satisfied"):
                downgraded_unresolved = bool(
                    audit.get("requested_flexibility_mode")
                    != audit.get("effective_flexibility_mode")
                    and int(
                        audit.get("lookup_unresolved_bond_count") or 0
                    )
                    > 0
                )
                audit_warnings.append(
                    "TORSION_BUDGET_NOT_ASSESSABLE"
                    if downgraded_unresolved
                    or audit.get("flexibility_error")
                    else "TORSION_BUDGET_UNSATISFIED"
                )
            if audit.get("flexibility_error"):
                audit_warnings.append(
                    "FLEXIBILITY_EVALUATION_FAILED_BASELINE_RETAINED"
                )
            warning_codes.extend(audit_warnings)
            baseline_path = audit.get("baseline_pdbqt_path")
            if baseline_path and Path(str(baseline_path)).is_file():
                artifacts.append({
                    "format": "pdbqt",
                    "path": str(baseline_path),
                    "role": "baseline",
                    "rigor": audit.get("inherited_rigor"),
                    "quality": audit.get("inherited_quality"),
                    "coordinate_mode": audit.get(
                        "parent_coordinate_mode"
                    ),
                    "usable_for": ["inspection", "docking_candidate"],
                    "warnings": list(audit_warnings),
                })
            prepared_path = audit.get("output_path")
            if prepared_path and Path(str(prepared_path)).is_file():
                artifact_warnings = list(audit_warnings)
                artifacts.append({
                    "format": "pdbqt",
                    "path": str(prepared_path),
                    "role": audit.get("output_role") or "primary",
                    "rigor": audit.get("inherited_rigor"),
                    "quality": audit.get("inherited_quality"),
                    "coordinate_mode": audit.get(
                        "parent_coordinate_mode"
                    ),
                    "usable_for": ["inspection", "docking_candidate"],
                    "warnings": artifact_warnings,
                })
                data["output_path"] = str(prepared_path)

            if parent_mol2_artifact is None:
                parent_chain = _typed_mol2_parent_chain(
                    mol2_path,
                    receipt_path,
                    audit,
                    validated_parent=validated_parent,
                )
                typed_parent = parent_chain[
                    "parent_mol2_artifact"
                ]
                data.update(parent_chain)
            else:
                typed_parent = _validate_supplied_mol2_artifact(
                    parent_mol2_artifact, audit
                )
                data["parent_mol2_artifact"] = typed_parent
            typed_ensemble = (
                dict(ensemble_artifact)
                if ensemble_artifact is not None
                else None
            )
            if typed_ensemble is None:
                ensemble_id = (
                    (audit.get("ensemble") or {}).get(
                        "ensemble_artifact_id"
                    )
                    if isinstance(audit.get("ensemble"), Mapping)
                    else None
                )
                if ensemble_id:
                    typed_ensemble = {"artifact_id": ensemble_id}
            flexibility_artifact = flexibility_artifact_from_audit(
                typed_parent, typed_ensemble, audit
            )
            data["flexibility_artifact"] = flexibility_artifact
            data["pdbqt_artifacts"] = []
            data["primary_pdbqt_artifact"] = None
            flexibility_profile = evidence_from_dict(
                flexibility_artifact["evidence"]
            )
            mol2_profile = evidence_from_dict(
                typed_parent["evidence"]
            )
            primary_path = (
                Path(str(prepared_path)).resolve()
                if prepared_path
                else None
            )
            for legacy_artifact in artifacts:
                artifact_path = Path(
                    str(legacy_artifact["path"])
                ).resolve()
                role = str(legacy_artifact["role"])
                qualification_state = _pdbqt_qualification_state(
                    role, audit
                )
                consumes_assessment = bool(
                    audit.get("torsdof_limit") is not None
                    and (
                        primary_path is not None
                        and artifact_path == primary_path
                    )
                )
                evidence_parent = (
                    flexibility_profile
                    if consumes_assessment
                    else mol2_profile
                )
                pdbqt_evidence = inherit_evidence(
                    evidence_parent,
                    format_level=FormatLevel.Q2,
                )
                pdbqt_payload = {
                    "sha256": sha256_path(artifact_path),
                    "role": role,
                    "qualification_state": qualification_state,
                    "parent_mol2_sha256": audit.get(
                        "parent_mol2_sha256"
                    ),
                    "tree_valid": bool(
                        audit.get("pdbqt_tree_valid")
                    ),
                    "atom_invariants_valid": bool(
                        audit.get("atom_invariants_valid")
                    ),
                    "budget_satisfied": audit.get(
                        "budget_satisfied"
                    ),
                }
                parent_ids = (
                    (
                        str(typed_parent["artifact_id"]),
                        str(flexibility_artifact["artifact_id"]),
                    )
                    if consumes_assessment
                    else (str(typed_parent["artifact_id"]),)
                )
                pdbqt_id = make_artifact_id(
                    ArtifactType.PDBQT,
                    parent_ids,
                    pdbqt_payload,
                )
                typed_pdbqt_object = PdbqtArtifact(
                    artifact_type=ArtifactType.PDBQT,
                    artifact_id=pdbqt_id,
                    parent_artifact_ids=parent_ids,
                    status=ArtifactStatus.MATERIALIZED,
                    payload_sha256=artifact_payload_sha256(
                        pdbqt_payload
                    ),
                    evidence=pdbqt_evidence,
                    warnings=tuple(sorted(set(audit_warnings))),
                    provenance={
                        "operation": operation,
                        "audit": dict(audit),
                        "assessment_artifact_id": (
                            flexibility_artifact["artifact_id"]
                        ),
                        "consumes_assessment": consumes_assessment,
                    },
                    claim_boundary=ClaimBoundary(
                        allowed=(
                            "PDBQT format qualification",
                            "mechanistic torsion budget",
                        ),
                        forbidden=(
                            "docking pose benefit",
                            "binding affinity benefit",
                        ),
                    ),
                    path=str(artifact_path),
                    sha256=str(pdbqt_payload["sha256"]),
                    role=role,
                    parent_mol2_sha256=str(
                        pdbqt_payload["parent_mol2_sha256"]
                    ),
                    tree_valid=bool(
                        pdbqt_payload["tree_valid"]
                    ),
                    atom_invariants_valid=bool(
                        pdbqt_payload["atom_invariants_valid"]
                    ),
                    qualification_state=qualification_state,
                    budget_satisfied=audit.get(
                        "budget_satisfied"
                    ),
                )
                validate_inherited_evidence(
                    evidence_parent,
                    pdbqt_evidence,
                    allow_format=True,
                )
                validate_artifact_identity(
                    typed_pdbqt_object, payload=pdbqt_payload
                )
                typed_pdbqt = typed_pdbqt_object.to_dict()
                data["pdbqt_artifacts"].append(typed_pdbqt)
                if (
                    primary_path is not None
                    and artifact_path == primary_path
                ):
                    data["primary_pdbqt_artifact"] = typed_pdbqt
            primary_typed = data.get("primary_pdbqt_artifact")
            data.update({
                "requested_format": "pdbqt",
                "requested_format_status": (
                    "fulfilled" if prepared_path else "degraded_format"
                ),
                "qualification_status": (
                    "qualified"
                    if prepared_path
                    and (
                        audit.get("torsdof_limit") is None
                        or audit.get("budget_satisfied")
                    )
                    else "unqualified"
                ),
                "budget_satisfied": audit.get("budget_satisfied"),
                "pdbqt_state": (
                    primary_typed.get("qualification_state")
                    if isinstance(primary_typed, Mapping)
                    else "PDBQT_BASELINE"
                ),
                "parent_mol2_sha256": audit.get("parent_mol2_sha256"),
                "parent_full_inchikey": audit.get(
                    "parent_full_inchikey"
                ),
                "inherited_rigor": audit.get("inherited_rigor"),
                "inherited_quality": audit.get("inherited_quality"),
                "parent_coordinate_mode": audit.get(
                    "parent_coordinate_mode"
                ),
                "parent_coordinate_level": audit.get(
                    "parent_coordinate_level"
                ),
                "parent_mapped_heavy_atom_indices": list(
                    audit.get("parent_mapped_heavy_atom_indices") or []
                ),
                "parent_generated_heavy_atom_indices": list(
                    audit.get("parent_generated_heavy_atom_indices") or []
                ),
                "parent_atom_coordinate_origins": dict(
                    audit.get("parent_atom_coordinate_origins") or {}
                ),
                "artifacts": artifacts,
                "warning_codes": sorted(set(warning_codes)),
            })
        if error:
            error_status = _external_error_status(error)
            if error_status in {"not_supported", "rejected"}:
                # The validated PDBQT gate remains fail-closed.  At the
                # product boundary, retain an already-written baseline PDBQT
                # or the parent MOL2 as an explicitly unqualified alternative.
                if not artifacts and Path(mol2_path).is_file():
                    artifacts.append({
                        "format": "mol2",
                        "path": str(mol2_path),
                        "role": "alternative",
                        "usable_for": ["inspection", "comparison"],
                        "warnings": [str(error)],
                    })
                has_pdbqt = any(
                    artifact.get("format") == "pdbqt"
                    for artifact in artifacts
                )
                data.update({
                    "requested_format": "pdbqt",
                    "requested_format_status": (
                        "degraded_format" if has_pdbqt else "unavailable"
                    ),
                    "qualification_status": "unqualified",
                    "reason": str(error),
                    "artifacts": artifacts,
                })
                return _operation_result(
                    operation, "success", data=data, error=None
                )
            return _operation_result(
                operation, error_status, data=data, error=error
            )
        if not artifacts:
            return _operation_result(
                operation,
                "failed",
                data=data,
                error="MOL2-to-PDBQT preparation produced no artifact",
            )
        return _operation_result(operation, "success", data=data)
    except Exception as exc:
        return _exception_result(operation, exc)


def prepare_ligand_from_sequence(
    sequence: str,
    output_dir: str | Path,
    *,
    cyclization: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    stereochemistry: Mapping[int | str, str] | None = None,
    terminal_modifications: Mapping[str, str] | None = None,
    protonation: str = "registry_default",
    conformer_count: int = 4,
    generate_pdbqt: bool = True,
    torsdof_limit: int | None = None,
    flexibility_mode: str = "balanced",
    torsion_prior_path: str | Path | None = None,
    template_strategy: str = "full",
    random_seed: int = 42,
    num_threads: int = 1,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare sequence-derived artifacts through one V5 DAG."""
    operation = "prepare_ligand_from_sequence"
    resolution_stack = None
    try:
        if type(conformer_count) is not int or conformer_count < 1:
            raise ValueError("conformer_count must be a positive integer")
        if type(random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        if type(num_threads) is not int or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        if flexibility_mode not in {"fast", "balanced", "thorough"}:
            raise ValueError(
                "flexibility_mode must be fast, balanced, or thorough"
            )
        if template_strategy not in TEMPLATE_STRATEGIES:
            raise ValueError(
                "template_strategy must be one of "
                + ", ".join(TEMPLATE_STRATEGIES)
            )
        from .export.conformer_ensemble import (
            materialize_mol2_ensemble,
        )
        from contextlib import ExitStack
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        from .sequence import (
            _sequence_tokens,
            build_molecule_from_sequence,
        )

        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        resolution_stack = ExitStack()
        resolution_ledger = resolution_stack.enter_context(
            monomer_resolution_context(
                _runtime_monomer_context(monomer_context),
                required_symbols=_sequence_tokens(sequence),
            )
        )
        chemical = build_molecule_from_sequence(
            sequence,
            cyclization=cyclization,
            stereochemistry=stereochemistry,
            terminal_modifications=terminal_modifications,
            protonation=protonation,
            _resolution_ledger=resolution_ledger,
        )
        chemical_graph = chemical["chemical_graph"]
        warnings = list(chemical.get("warnings") or [])
        data: dict[str, Any] = {
            "input_artifact": chemical["input_artifact"],
            "chemical_graph": chemical_graph,
            "mol2_ensemble": None,
            "mol2_artifacts": [],
            "flexibility_artifacts": [],
            "pdbqt_artifacts": [],
            "warnings": warnings,
            "alternatives": list(chemical.get("alternatives") or []),
            "provenance": dict(chemical.get("provenance") or {}),
            "monomer_resolution": json_ready(resolution_ledger),
        }
        if not chemical_graph.get("materializable"):
            data["requested_artifact_status"] = (
                "GRAPH_PARTIAL"
                if chemical_graph.get("status") == "PARTIAL"
                else "NOT_MATERIALIZABLE"
            )
            return _operation_result(operation, "success", data=data)

        try:
            ensemble_result = materialize_mol2_ensemble(
                chemical_graph,
                root / "mol2",
                ensemble_size=int(conformer_count),
                template_strategy=template_strategy,
                torsion_prior_path=torsion_prior_path,
                random_seed=int(random_seed),
                num_threads=int(num_threads),
            )
        except Exception as exc:
            warnings.append(
                "MOL2_ENSEMBLE_MATERIALIZATION_FAILED:"
                f"{type(exc).__name__}:{exc}"
            )
            data["warnings"] = list(dict.fromkeys(warnings))
            data["requested_artifact_status"] = "MOL2_UNAVAILABLE"
            return _operation_result(operation, "success", data=data)
        ensemble = ensemble_result["ensemble"]
        mol2_artifacts = list(
            ensemble_result["validated_mol2_artifacts"]
        )
        data["mol2_ensemble"] = ensemble
        data["mol2_artifacts"] = mol2_artifacts
        data["ensemble_manifest_path"] = ensemble_result[
            "manifest_path"
        ]
        warnings.extend(ensemble.get("warnings") or [])
        if not generate_pdbqt or not mol2_artifacts:
            data["requested_artifact_status"] = (
                "MOL2_COMPLETE"
                if len(mol2_artifacts) == conformer_count
                else "MOL2_PARTIAL"
                if mol2_artifacts
                else "MOL2_UNAVAILABLE"
            )
            data["warnings"] = list(dict.fromkeys(warnings))
            return _operation_result(operation, "success", data=data)

        pdbqt_root = root / "pdbqt"
        pdbqt_root.mkdir(parents=True, exist_ok=True)
        for mol2_artifact in mol2_artifacts:
            output = pdbqt_root / (
                f"{mol2_artifact['conformer_id']}.pdbqt"
            )
            try:
                prepared = prepare_ligand_pdbqt_from_mol2(
                    mol2_artifact["path"],
                    output,
                    receipt_path=mol2_artifact["receipt_path"],
                    torsdof_limit=torsdof_limit,
                    flexibility_mode=flexibility_mode,
                    torsion_prior_path=torsion_prior_path,
                    ensemble_manifest_path=ensemble_result[
                        "manifest_path"
                    ],
                    ensemble_manifest_sha256=ensemble_result[
                        "manifest_sha256"
                    ],
                    strict_budget=False,
                    parent_mol2_artifact=mol2_artifact,
                    ensemble_artifact=ensemble,
                )
            except Exception as exc:
                warnings.append(
                    f"PDBQT_FAILED:{mol2_artifact['conformer_id']}:"
                    f"{type(exc).__name__}:{exc}"
                )
                continue
            prepared_data = prepared.get("data") or {}
            flexibility = prepared_data.get(
                "flexibility_artifact"
            )
            if isinstance(flexibility, Mapping):
                data["flexibility_artifacts"].append(flexibility)
            warnings.extend(prepared_data.get("warning_codes") or [])
            prepared_path = prepared_data.get("output_path")
            primary_artifact = prepared_data.get(
                "primary_pdbqt_artifact"
            )
            if (
                prepared.get("status") != "success"
                or not prepared_path
                or not Path(str(prepared_path)).is_file()
                or not isinstance(primary_artifact, Mapping)
            ):
                warnings.append(
                    f"PDBQT_FAILED:{mol2_artifact['conformer_id']}:"
                    f"{prepared.get('error') or prepared.get('status')}"
                )
                continue
            data["pdbqt_artifacts"].append(
                dict(primary_artifact)
            )
        data["warnings"] = list(dict.fromkeys(warnings))
        data["requested_artifact_status"] = (
            "PDBQT_COMPLETE"
            if (
                ensemble.get("status") == "MATERIALIZED"
                and len(mol2_artifacts) == int(conformer_count)
                and len(data["pdbqt_artifacts"])
                == int(conformer_count)
            )
            else "PDBQT_PARTIAL"
            if data["pdbqt_artifacts"]
            else "PDBQT_UNAVAILABLE"
        )
        return _operation_result(operation, "success", data=data)
    except (TypeError, ValueError) as exc:
        return _operation_result(
            operation, "invalid_input", error=str(exc)
        )
    except Exception as exc:
        return _exception_result(operation, exc)
    finally:
        if resolution_stack is not None:
            resolution_stack.close()


def prepare_ligand_pdbqt(
    smiles: Any,
    output_path: str | Path,
    *,
    generated_map: str | None = None,
    num_confs: int = 10,
    random_seed: int = 42,
    rigid_macrocycles: bool = True,
    protonate: bool = True,
    torsdof_limit: int | None = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility facade that materializes validated MOL2 before Meeko."""
    operation = "prepare_ligand_pdbqt"
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        return _operation_result(
            operation,
            "invalid_input",
            error="random_seed must be an integer",
        )
    if (
        not isinstance(num_confs, int)
        or isinstance(num_confs, bool)
        or num_confs < 1
    ):
        return _operation_result(
            operation,
            "invalid_input",
            error="num_confs must be an integer >= 1",
        )
    if (
        not isinstance(torsion_num_threads, int)
        or isinstance(torsion_num_threads, bool)
        or torsion_num_threads < 1
    ):
        return _operation_result(
            operation,
            "invalid_input",
            error="torsion_num_threads must be an integer >= 1",
        )
    if (
        torsdof_limit is not None
        and (
            not isinstance(torsdof_limit, int)
            or isinstance(torsdof_limit, bool)
            or torsdof_limit < 0
        )
    ):
        return _operation_result(
            operation,
            "invalid_input",
            error="torsdof_limit must be non-negative",
        )
    if not isinstance(smiles, Mapping):
        candidate_path = Path(smiles) if isinstance(smiles, (str, Path)) else None
        if candidate_path is not None and candidate_path.name.lower().endswith(
            SUPPORTED_COORDINATE_SUFFIXES
        ):
            return prepare_ligand_pdbqt_from_pdb(
                candidate_path,
                output_path,
                chain_id="L",
                reconstruction_mode=reconstruction_mode,
                minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=require_empty_persistent_overlay,
                torsdof_limit=torsdof_limit,
                random_seed=random_seed,
                num_threads=torsion_num_threads,
                monomer_context=monomer_context,
            )
    (
        is_reconstruction,
        reconstruction_smiles,
        reconstruction_context,
        handoff_error,
    ) = _reconstruction_handoff(smiles)
    if is_reconstruction:
        if handoff_error:
            return _operation_result(
                operation,
                "success",
                data={
                    "reconstruction": reconstruction_context,
                    "audit": {},
                    "artifact_status": "PARTIAL",
                    "requested_artifact_status": "PDBQT_UNAVAILABLE",
                },
                error=(
                    f"{handoff_error}; PDBQT preparation requires a "
                    "serializable SMILES payload"
                ),
            )
        smiles = reconstruction_smiles
    try:
        from rdkit import Chem
        from .docking.ligand_pdbqt import _unsupported_ligand_error

        input_molecule = Chem.MolFromSmiles(str(smiles))
        if input_molecule is None:
            return _operation_result(
                operation,
                "invalid_input",
                error="input SMILES is not parseable",
            )
        unsupported = _unsupported_ligand_error(input_molecule)
        if unsupported:
            return _operation_result(
                operation,
                "success",
                data={
                    "artifact_status": "PARTIAL",
                    "requested_artifact_status": "PDBQT_UNAVAILABLE",
                    "input_smiles": str(smiles),
                },
                error=unsupported,
            )
        if not rigid_macrocycles:
            return _operation_result(
                operation,
                "invalid_input",
                error=(
                    "validated-MOL2 preparation requires "
                    "rigid_macrocycles=True"
                ),
            )
        from .docking.protonation import protonate_ph74
        from .export.conformer import mol_to_mol2, smiles_to_mol2

        parent_smiles = (
            protonate_ph74(str(smiles)) if protonate else str(smiles)
        )
        destination = Path(output_path)
        parent_mol2 = Path(str(destination) + ".parent.mol2")
        parent_coordinate_mode = "regenerated"
        template_metadata = None
        written = None
        export_warning = None
        if generated_map is not None:
            try:
                from .core.monomer_resolution import (
                    monomer_resolution_context,
                )
                from .docking.template_library import generate_conformers

                template_metadata = {}
                with monomer_resolution_context(
                    _runtime_monomer_context(monomer_context),
                    required_symbols=_monomer_resolution_hints(
                        generated_map, kind="map"
                    ),
                ) as resolution_ledger:
                    template_molecule, conformer_ids = generate_conformers(
                        parent_smiles,
                        generated_map,
                        n_conformers=1,
                        meta_out=template_metadata,
                        random_seed=int(random_seed),
                    )
                template_metadata["monomer_resolution"] = json_ready(
                    resolution_ledger
                )
                if template_molecule is not None and conformer_ids:
                    written, export_warning = mol_to_mol2(
                        template_molecule,
                        output_path=str(parent_mol2),
                    )
                    parent_coordinate_mode = (
                        "template_completed"
                        if int(
                            template_metadata.get(
                                "guided_conformer_count"
                            )
                            or 0
                        )
                        > 0
                        else "regenerated"
                    )
            except Exception as exc:
                template_metadata = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        if not written:
            written, export_warning = smiles_to_mol2(
                parent_smiles,
                output_path=str(parent_mol2),
                force_field="mmff",
                num_confs=int(num_confs),
                random_seed=int(random_seed),
            )
            parent_coordinate_mode = "regenerated"
        if not written or not parent_mol2.is_file():
            return _operation_result(
                operation,
                "success",
                data={
                    "artifact_status": "PARTIAL",
                    "chemical_rigor": "C2:H",
                    "inherited_rigor": (
                        (reconstruction_context or {}).get("rigor")
                    ),
                    "requested_artifact_status": "PDBQT_UNAVAILABLE",
                    "input_smiles": str(smiles),
                    "alternatives": [{
                        "kind": "smiles",
                        "value": str(smiles),
                        "claim_boundary": "chemical graph only",
                    }],
                    "template_metadata": template_metadata,
                },
                error=(
                    "validated parent MOL2 generation failed: "
                    f"{export_warning or 'no output'}"
                ),
            )
        context = reconstruction_context or {
            "quality": "hypothesis",
            "rigor": "L2:H",
            "source": "provided_smiles",
        }
        rigor = _artifact_rigor(context)
        parent_chemical_rigor = _chemical_rigor_label(
            context, legacy_rigor=rigor
        )
        rigor = _legacy_rigor_alias(parent_chemical_rigor, rigor)
        # The parent MOL2 is template-guided or ETKDG-generated from SMILES:
        # there is no experimental source coordinate, so the receipt reports
        # regenerated/X1 with every heavy atom generated.  Template guidance
        # stays in evidence metadata and must not claim the X2
        # source-completion semantics.
        parent_ledger = _mol2_coordinate_ledger(parent_mol2)
        if parent_ledger.get("coordinate_level") == "X1":
            generated_parent_indices = parent_ledger[
                "generated_heavy_atom_indices"
            ]
        else:
            # Header-less parent (template route): every heavy atom is
            # generated; ids are parsed one-based from the artifact.
            generated_parent_indices = parent_ledger["heavy_atom_indices"]
        (
            _receipt_mapped_indices,
            receipt_generated_parent_indices,
            receipt_parent_origins,
        ) = _receipt_provenance_indices({
            "mapped_heavy_atom_indices": [],
            "generated_heavy_atom_indices": generated_parent_indices,
            "atom_coordinate_origins": {
                str(index): "generated"
                for index in generated_parent_indices
            },
        })
        receipt = _write_mol2_validation_receipt(
            parent_mol2,
            coordinate_mode="regenerated",
            coordinate_level="X1",
            mapped_heavy_atom_indices=_receipt_mapped_indices,
            generated_heavy_atom_indices=receipt_generated_parent_indices,
            atom_coordinate_origins=receipt_parent_origins,
            rigor=rigor,
            quality=context.get("quality") or "hypothesis",
            source_heavy_atom_mapping_complete=False,
            atom_provenance_complete=True,
            source_input_sha256=hashlib.sha256(
                str(smiles).encode("utf-8")
            ).hexdigest(),
            evidence={
                "operation": operation,
                "reconstruction": context,
                "protonate": bool(protonate),
                "generated_map_role": (
                    "mol2_coordinate_guidance"
                    if generated_map is not None
                    else None
                ),
                "template_metadata": template_metadata,
                "chemical_rigor": parent_chemical_rigor,
                "coordinate_level": "X1",
                "atom_index_convention": {
                    "artifact_header": "mol2_one_based",
                    "receipt_provenance": (
                        RECEIPT_ATOM_INDEX_CONVENTION
                    ),
                },
                "candidate_ambiguity": _candidate_ambiguity_summary(
                    context
                ),
                "fallback_warnings": list(
                    parent_ledger.get("warnings") or []
                ),
                "tier_evidence": parent_ledger.get("tier_evidence")
                or {},
                "identity": {
                    "expected_full_inchikey": _full_inchikey_from_smiles(
                        parent_smiles
                    ),
                    "observed_full_inchikey": _observed_mol2_identity(
                        parent_mol2
                    ).get("full_inchikey"),
                },
                "export_warning": export_warning,
            },
            expected_full_inchikey=_full_inchikey_from_smiles(
                parent_smiles
            ),
        )
        prepared = prepare_ligand_pdbqt_from_mol2(
            parent_mol2,
            destination,
            receipt_path=receipt,
            torsdof_limit=torsdof_limit,
            flexibility_mode="balanced",
            ensemble_size=4,
            random_seed=int(random_seed),
            num_threads=int(torsion_num_threads),
        )
        prepared["operation"] = operation
        data = prepared.setdefault("data", {})
        data["parent_mol2_path"] = str(parent_mol2)
        data["parent_receipt_path"] = str(receipt)
        data["coordinate_materializer_contract"] = (
            "legacy_smiles_compatibility"
        )
        data["legacy_v4_handoff"] = {
            "generated_map_role": (
                "mol2_coordinate_guidance"
                if generated_map is not None
                else None
            ),
            "legacy_torsion_ensemble_size": int(
                torsion_ensemble_size
            ),
            "v4_balanced_ensemble_size": 4,
            "parent_export_warning": export_warning,
        }
        if reconstruction_context is not None:
            data["reconstruction"] = reconstruction_context
        warning_codes = data.setdefault("warning_codes", [])
        warning_codes.append(
            "LEGACY_SMILES_MOL2_PATH_WITHOUT_V5_ENSEMBLE_QA"
        )
        if generated_map is not None and parent_coordinate_mode != (
            "template_completed"
        ):
            warning_codes.append("V4_MOL2_TEMPLATE_GUIDANCE_FELL_BACK")
        if int(torsion_ensemble_size) != 4:
            warning_codes.append(
                "V4_LEGACY_ENSEMBLE_SIZE_REPLACED_BY_BALANCED_FOUR"
            )
        return prepared
    except Exception as exc:
        return _exception_result(
            operation,
            exc,
            data=(
                {"reconstruction": reconstruction_context}
                if reconstruction_context is not None
                else None
            ),
        )


def prepare_ligand_pdbqt_from_pdb(
    coordinate_path: str | Path,
    output_path: str | Path,
    *,
    chain_id: str = "L",
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    torsdof_limit: int | None = None,
    flexibility_mode: str = "balanced",
    torsion_prior_path: str | Path | None = None,
    ensemble_size: int = 4,
    random_seed: int = 42,
    num_threads: int = 1,
    strict_budget: bool = False,
    fallback_policy: str = "max_coverage",
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare PDBQT only after validated parent MOL2 export."""
    operation = "prepare_ligand_pdbqt_from_pdb"
    if _paths_alias(coordinate_path, Path(output_path)):
        return _operation_result(
            operation,
            "invalid_input",
            error=(
                "coordinate input and output path must not alias: "
                f"source={coordinate_path}, output={output_path}"
            ),
        )
    try:
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        parent_mol2 = Path(str(destination) + ".parent.mol2")
        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                coordinate_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            parent_export = export_best_available(
                coordinate_path,
                parent_mol2,
                source_kind="coordinate",
                output_format="mol2",
                chain_id=str(chain_id),
                minimum_macrocycle_ring_size=int(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=bool(
                    require_empty_persistent_overlay
                ),
                fallback_policy=fallback_policy,
            )
        export_data = parent_export.get("data") or {}
        receipt_path = export_data.get("validation_receipt_path")
        if (
            parent_export.get("status") != "success"
            or export_data.get("requested_format_status") != "fulfilled"
            or not parent_mol2.is_file()
            or not receipt_path
        ):
            return _operation_result(
                operation,
                "success",
                data={
                    "requested_format": "pdbqt",
                    "requested_format_status": (
                        "metadata_only"
                        if all(
                            artifact.get("format") == "metadata"
                            for artifact in (
                                export_data.get("artifacts") or []
                            )
                        )
                        else "degraded_format"
                    ),
                    "qualification_status": "not_assessable",
                    "parent_mol2_export": export_data,
                    "artifacts": export_data.get("artifacts") or [],
                    "artifact_status": "PARTIAL",
                    "requested_artifact_status": "PDBQT_UNAVAILABLE",
                    "monomer_resolution": json_ready(
                        resolution_ledger
                    ),
                },
                error=(
                    parent_export.get("error")
                    or "coordinate input did not produce validated MOL2"
                ),
            )
        prepared = prepare_ligand_pdbqt_from_mol2(
            parent_mol2,
            destination,
            receipt_path=receipt_path,
            torsdof_limit=torsdof_limit,
            flexibility_mode=flexibility_mode,
            torsion_prior_path=torsion_prior_path,
            ensemble_size=ensemble_size,
            random_seed=random_seed,
            num_threads=num_threads,
            strict_budget=strict_budget,
        )
        prepared["operation"] = operation
        data = prepared.setdefault("data", {})
        data.setdefault("input_path", str(coordinate_path))
        data.setdefault("parent_mol2_path", str(parent_mol2))
        data.setdefault("parent_receipt_path", str(receipt_path))
        data.setdefault("parent_mol2_export", export_data)
        data.setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        data.setdefault("coordinate_source", "validated_parent_mol2")
        data.setdefault(
            "original_coordinates_preserved",
            data.get("parent_coordinate_mode") == "source_bound",
        )
        data.setdefault(
            "legacy_reconstruction_mode_ignored_by_v4",
            reconstruction_mode,
        )
        return prepared
    except Exception as exc:
        return _exception_result(operation, exc)


def prepare_ligand_pdbqt_best_available(
    coordinate_path: str | Path,
    output_path: str | Path,
    *,
    chain_id: str = "L",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    generated_map: str | None = None,
    num_confs: int = 10,
    random_seed: int = 42,
    rigid_macrocycles: bool = True,
    protonate: bool = True,
    torsdof_limit: int | None = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
    fallback_policy: str = "max_coverage",
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare through validated MOL2 or return its lower artifact."""
    operation = "prepare_ligand_pdbqt_best_available"
    destination = Path(output_path)
    if _paths_alias(coordinate_path, destination):
        return _operation_result(
            operation,
            "invalid_input",
            error=(
                "coordinate input and output path must not alias: "
                f"source={coordinate_path}, output={output_path}"
            ),
        )
    if not rigid_macrocycles:
        return _operation_result(
            operation,
            "invalid_input",
            error="validated-MOL2 PDBQT requires rigid_macrocycles=True",
        )
    prepared = prepare_ligand_pdbqt_from_pdb(
        coordinate_path,
        destination,
        chain_id=chain_id,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        torsdof_limit=torsdof_limit,
        flexibility_mode="balanced",
        ensemble_size=4,
        random_seed=random_seed,
        num_threads=torsion_num_threads,
        fallback_policy=fallback_policy,
        monomer_context=monomer_context,
    )
    prepared["operation"] = operation
    data = prepared.setdefault("data", {})
    data["legacy_v4_handoff"] = {
        "generated_map_ignored": generated_map is not None,
        "num_confs_ignored_for_pdb_parent": int(num_confs),
        "protonate_ignored_parent_state_preserved": bool(protonate),
        "legacy_torsion_ensemble_size": int(torsion_ensemble_size),
        "v4_balanced_ensemble_size": 4,
    }
    artifacts = data.get("artifacts") or []
    if prepared.get("status") == "success":
        data.setdefault("requested_format", "pdbqt")
        if not data.get("requested_format_status"):
            if destination.is_file():
                data["requested_format_status"] = "fulfilled"
            elif artifacts:
                data["requested_format_status"] = (
                    "metadata_only"
                    if all(
                        artifact.get("format") == "metadata"
                        for artifact in artifacts
                    )
                    else "degraded_format"
                )
            else:
                data["requested_format_status"] = "unavailable"
        return prepared
    if artifacts:
        requested_status = (
            "metadata_only"
            if all(
                artifact.get("format") == "metadata"
                for artifact in artifacts
            )
            else "degraded_format"
        )
        return _operation_result(
            operation,
            "success",
            data={
                **data,
                "requested_format": "pdbqt",
                "requested_format_status": requested_status,
                "artifacts": artifacts,
            },
        )
    return prepared


def prepare_ligand_pdbqt_ensemble(
    smiles: Any,
    output_dir: str | Path,
    *,
    n_conformers: int = 3,
    generated_map: str | None = None,
    rigid_macrocycles: bool = True,
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
) -> dict[str, Any]:
    """Compatibility rejection for the retired direct PDBQT ensemble path."""
    operation = "prepare_ligand_pdbqt_ensemble"
    if not isinstance(smiles, Mapping):
        candidate_path = Path(smiles) if isinstance(smiles, (str, Path)) else None
        if candidate_path is not None and candidate_path.name.lower().endswith(
            SUPPORTED_COORDINATE_SUFFIXES
        ):
            reconstruction = reconstruct_structure(
                candidate_path,
                chain_id="L",
                mode=reconstruction_mode,
                minimum_macrocycle_ring_size=int(minimum_macrocycle_ring_size),
                require_empty_persistent_overlay=bool(
                    require_empty_persistent_overlay
                ),
            )
            reconstructed = prepare_ligand_pdbqt_ensemble(
                reconstruction,
                output_dir,
                n_conformers=n_conformers,
                generated_map=generated_map,
                rigid_macrocycles=rigid_macrocycles,
            )
            reconstructed["operation"] = operation
            reconstructed.setdefault("data", {}).setdefault(
                "coordinate_source", "unified_reconstruction"
            )
            return reconstructed
    try:
        n_conformers = int(n_conformers)
    except (TypeError, ValueError, OverflowError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if n_conformers < 1:
        return _operation_result(
            operation, "invalid_input", error="n_conformers must be at least one"
        )
    (
        is_reconstruction,
        reconstruction_smiles,
        reconstruction_context,
        handoff_error,
    ) = _reconstruction_handoff(smiles)
    if is_reconstruction:
        if handoff_error:
            return _operation_result(
                operation,
                "not_supported",
                data={"reconstruction": reconstruction_context},
                error=(
                    f"{handoff_error}; PDBQT ensemble preparation requires "
                    "a serializable SMILES payload"
                ),
            )
        smiles = reconstruction_smiles
    try:
        from .docking.ligand_pdbqt import smiles_to_ligand_pdbqt_multi

        ensemble_audit: dict[str, Any] = {}
        paths, error = smiles_to_ligand_pdbqt_multi(
            smiles,
            str(output_dir),
            n_confs=int(n_conformers),
            generated_map=generated_map,
            rigid_macrocycles=bool(rigid_macrocycles),
            audit_out=ensemble_audit,
        )
        if not error and len(paths) != int(n_conformers):
            error = (
                f"partial PDBQT ensemble: generated {len(paths)} "
                f"of {int(n_conformers)}"
            )
        if not error:
            missing = [
                str(path) for path in paths
                if not Path(path).is_file() or Path(path).stat().st_size == 0
            ]
            if missing:
                error = (
                    "PDBQT ensemble contains missing or empty outputs: "
                    + ", ".join(missing)
                )
        if error:
            status = _external_error_status(error)
            data = {
                "output_paths": paths,
                "count": len(paths),
                "audit": ensemble_audit,
            }
            if reconstruction_context is not None:
                data["reconstruction"] = reconstruction_context
            return _operation_result(
                operation,
                status,
                data=data,
                error=error,
            )
        data = {
            "output_dir": str(output_dir),
            "output_paths": paths,
            "count": len(paths),
            "generated_map_supplied": generated_map is not None,
            "audit": ensemble_audit,
        }
        if reconstruction_context is not None:
            data["reconstruction"] = reconstruction_context
        return _operation_result(
            operation,
            "success",
            data=data,
        )
    except (TypeError, ValueError) as exc:
        return _operation_result(
            operation,
            "invalid_input",
            data=(
                {"reconstruction": reconstruction_context}
                if reconstruction_context is not None
                else None
            ),
            error=str(exc),
        )
    except Exception as exc:
        return _exception_result(
            operation,
            exc,
            data=(
                {"reconstruction": reconstruction_context}
                if reconstruction_context is not None
                else None
            ),
        )


def prepare_receptor_pdbqt(
    coordinate_path: str | Path,
    output_path: str | Path,
    *,
    ph: float | None = None,
) -> dict[str, Any]:
    operation = "prepare_receptor_pdbqt"
    try:
        from .docking.receptor_pdbqt import pdb_to_receptor_pdbqt

        if ph is not None and (isinstance(ph, bool) or not math.isfinite(float(ph)) or not 0 < float(ph) <= 14):
            return _operation_result(operation, "invalid_input", error="pH must be finite in (0, 14]")
        error = pdb_to_receptor_pdbqt(
            str(coordinate_path), str(output_path),
            **({"ph": float(ph)} if ph is not None else {}),
        )
        if error:
            return _operation_result(
                operation, _external_error_status(error), error=error
            )
        destination = Path(output_path)
        if not destination.is_file() or destination.stat().st_size == 0:
            return _operation_result(
                operation,
                "failed",
                error="receptor PDBQT converter produced no nonempty output",
            )
        preparation_remarks = [
            line for line in destination.read_text(encoding="utf-8").splitlines()
            if line.startswith("REMARK")
        ]
        return _operation_result(
            operation, "success", data={
                "output_path": str(output_path),
                "preparation_remarks": preparation_remarks,
                "warnings": [line for line in preparation_remarks if "WARNING" in line],
            }
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def validate_pdbqt(
    *,
    payload: str | None = None,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    operation = "validate_pdbqt"
    if payload is not None and input_path is not None:
        return _operation_result(
            operation,
            "invalid_input",
            error="provide either payload or input_path, not both",
        )
    try:
        from .docking.pdbqt_validation import validate_pdbqt_torsion_tree

        text = payload
        if input_path is not None:
            text = Path(input_path).read_text(encoding="utf-8", errors="replace")
        if text is None:
            return _operation_result(
                operation, "invalid_input", error="payload or input_path is required"
            )
        audit = validate_pdbqt_torsion_tree(text)
        return _operation_result(operation, "success", data={"audit": audit})
    except (OSError, TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    except RuntimeError as exc:
        return _operation_result(operation, "rejected", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)


def docking_center(
    receptor_path: str | Path,
    *,
    residue_ids: Sequence[int] | None = None,
    chain_id: str | None = None,
) -> dict[str, Any]:
    operation = "docking_center"
    try:
        from .docking.box import get_binding_site_center, get_protein_center

        if residue_ids is not None:
            selected_residues = list(residue_ids)
            if not selected_residues:
                return _operation_result(
                    operation,
                    "invalid_input",
                    error="residue_ids must not be empty when supplied",
                )
            center = get_binding_site_center(
                str(receptor_path), selected_residues, chain_id=chain_id
            )
            mode = "binding_site"
        else:
            center = get_protein_center(
                str(receptor_path), chain_id=chain_id
            )
            mode = "protein"
        return _operation_result(
            operation,
            "success",
            data={"center": center, "mode": mode},
        )
    except (OSError, TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)


def run_prepared_vina(
    ligand_pdbqt: str | Path,
    receptor_pdbqt: str | Path,
    output_pdbqt: str | Path,
    *,
    center: Sequence[float],
    box_size: Sequence[float] = (25.0, 25.0, 25.0),
    exhaustiveness: int = 32,
    num_modes: int = 9,
    seed: int | None = None,
    cpu: int | None = None,
    max_evals: int | None = None,
    timeout_seconds: float = 600,
    mode: str = "docking",
) -> dict[str, Any]:
    """Run Vina directly on caller-prepared PDBQT files.

    ``mode`` selects the Vina operation: ``docking`` (global search, the
    default and only behavior before this parameter existed), ``score_only``
    (score the input pose as-is; Vina writes no output file, ``output_pdbqt``
    is ignored and reported as None), or ``local_only`` (local refinement of
    the input pose; a fresh verified output PDBQT is required).
    """
    operation = "run_prepared_vina"
    if not isinstance(mode, str) or mode not in ("docking", "score_only", "local_only"):
        return _operation_result(
            operation,
            "invalid_input",
            error="mode must be one of docking, score_only, local_only",
        )
    try:
        center = tuple(float(value) for value in center)
        box_size = tuple(float(value) for value in box_size)
        if (
            isinstance(exhaustiveness, bool)
            or isinstance(num_modes, bool)
            or int(exhaustiveness) != exhaustiveness
            or int(num_modes) != num_modes
        ):
            raise ValueError("exhaustiveness and num_modes must be integers")
        exhaustiveness = int(exhaustiveness)
        num_modes = int(num_modes)
        execution_controls = {}
        for name, value in (("seed", seed), ("cpu", cpu), ("max_evals", max_evals)):
            if value is not None:
                if isinstance(value, bool) or int(value) != value or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
                execution_controls[name] = int(value)
        if isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be positive and finite")
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if timeout_seconds != 600:
            execution_controls["timeout_seconds"] = timeout_seconds
    except (TypeError, ValueError, OverflowError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if len(center) != 3 or len(box_size) != 3:
        return _operation_result(
            operation,
            "invalid_input",
            error="center and box_size each require three values",
        )
    if not all(math.isfinite(value) for value in center):
        return _operation_result(
            operation, "invalid_input", error="center requires three finite values"
        )
    if not all(math.isfinite(value) and value > 0 for value in box_size):
        return _operation_result(
            operation,
            "invalid_input",
            error="box_size requires three positive finite values",
        )
    if exhaustiveness < 1 or num_modes < 1:
        return _operation_result(
            operation,
            "invalid_input",
            error="exhaustiveness and num_modes must be positive",
        )
    try:
        from .docking.vina import run_vina

        affinity, error = run_vina(
            str(ligand_pdbqt),
            str(receptor_pdbqt),
            center,
            box_size,
            str(output_pdbqt),
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            **execution_controls,
            **({} if mode == "docking" else {"mode": mode}),
        )
        if error:
            return _operation_result(
                operation, _external_error_status(error), error=error
            )
        return _operation_result(
            operation,
            "success",
            data={
                "affinity_kcal_mol": affinity,
                "ligand_pdbqt": str(ligand_pdbqt),
                "receptor_pdbqt": str(receptor_pdbqt),
                "output_pdbqt": None if mode == "score_only" else str(output_pdbqt),
                "output_pdbqt_written": mode != "score_only",
                "center": center,
                "box_size": box_size,
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
                "seed": seed,
                "seed_mode": "automatic" if seed in (None, 0) else "fixed",
                "cpu": cpu,
                "cpu_mode": "automatic" if cpu in (None, 0) else "fixed",
                "max_evals": max_evals,
                "timeout_seconds": timeout_seconds,
                "mode": mode,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)


def dock_structure(
    peptide_path: str | Path,
    receptor_path: str | Path,
    *,
    center: Sequence[float] | None = None,
    binding_site_residues: Sequence[int] | None = None,
    receptor_chain_id: str | None = None,
    box_size: Sequence[float] = (25.0, 25.0, 25.0),
    output_dir: str | Path | None = None,
    ligand_preparation_mode: str = "auto",
    ligand_smiles: Any | None = None,
    peptide_chain_id: str = "L",
    generated_map: str | None = None,
    torsdof_limit: int | None = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import monomer_resolution_context

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                peptide_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            result = dock_structure(
                peptide_path,
                receptor_path,
                center=center,
                binding_site_residues=binding_site_residues,
                receptor_chain_id=receptor_chain_id,
                box_size=box_size,
                output_dir=output_dir,
                ligand_preparation_mode=ligand_preparation_mode,
                ligand_smiles=ligand_smiles,
                peptide_chain_id=peptide_chain_id,
                generated_map=generated_map,
                torsdof_limit=torsdof_limit,
                torsion_ensemble_size=torsion_ensemble_size,
                torsion_num_threads=torsion_num_threads,
                reconstruction_mode=reconstruction_mode,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
            )
        result.setdefault("data", {}).setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        return result
    operation = "dock"
    try:
        normalized_box_size = tuple(float(value) for value in box_size)
    except (TypeError, ValueError, OverflowError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if len(normalized_box_size) != 3:
        return _operation_result(
            operation, "invalid_input", error="box_size requires three values"
        )
    reconstruction_context = None
    if ligand_smiles is None and ligand_preparation_mode == "auto":
        (
            reconstructed_smiles,
            reconstruction_context,
            reconstruction_error,
        ) = _reconstruct_coordinate_for_downstream(
            peptide_path,
            chain_id=peptide_chain_id,
            mode=reconstruction_mode,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            legacy_path=None,
        )
        if reconstruction_error or not reconstructed_smiles:
            return _operation_result(
                operation,
                "success",
                data={
                    "reconstruction": reconstruction_context,
                    "artifact_status": "PARTIAL",
                    "chemical_rigor": (
                        (reconstruction_context or {}).get("rigor")
                        or "C1:H"
                    ),
                    "requested_artifact_status": "DOCKING_UNAVAILABLE",
                },
                error=(
                    reconstruction_error
                    or "coordinate reconstruction produced no serializable SMILES"
                ),
            )
        ligand_smiles = reconstructed_smiles
        ligand_preparation_mode = "audited_smiles"
    if ligand_smiles is not None:
        (
            is_reconstruction,
            reconstruction_smiles,
            handoff_context,
            handoff_error,
        ) = _reconstruction_handoff(ligand_smiles)
        if handoff_context is not None:
            reconstruction_context = handoff_context
        if is_reconstruction:
            if handoff_error:
                return _operation_result(
                    operation,
                    "success",
                    data={
                        "reconstruction": reconstruction_context,
                        "artifact_status": "PARTIAL",
                        "chemical_rigor": (
                            (reconstruction_context or {}).get("rigor")
                            or "C1:H"
                        ),
                        "requested_artifact_status": (
                            "DOCKING_UNAVAILABLE"
                        ),
                    },
                    error=(
                        f"{handoff_error}; docking through PDBQT requires a "
                        "serializable SMILES payload"
                    ),
                )
            ligand_smiles = reconstruction_smiles
            ligand_preparation_mode = "audited_smiles"
    try:
        if center is None:
            center_result = docking_center(
                receptor_path,
                residue_ids=binding_site_residues,
                chain_id=receptor_chain_id,
            )
            if center_result["status"] != "success":
                data = {"center_result": center_result}
                if reconstruction_context is not None:
                    data["reconstruction"] = reconstruction_context
                return _operation_result(
                    operation,
                    center_result["status"],
                    data=data,
                    error=center_result.get("error"),
                )
            center = center_result["data"]["center"]
        if len(tuple(center)) != 3:
            return _operation_result(
                operation, "invalid_input", error="center requires three values"
            )

        from .docking.workflow import dock_peptide

        audit: dict[str, Any] = {}
        affinity, error = dock_peptide(
            str(peptide_path),
            str(receptor_path),
            tuple(float(value) for value in center),
            normalized_box_size,
            str(output_dir) if output_dir else None,
            cleanup=output_dir is None,
            ligand_preparation_mode=ligand_preparation_mode,
            ligand_smiles=ligand_smiles,
            peptide_chain_id=peptide_chain_id,
            receptor_chain_id=receptor_chain_id,
            generated_map=generated_map,
            torsdof_limit=torsdof_limit,
            torsion_ensemble_size=int(torsion_ensemble_size),
            torsion_num_threads=int(torsion_num_threads),
            torsion_audit_out=audit,
            reconstruction_mode=reconstruction_mode,
            minimum_macrocycle_ring_size=int(minimum_macrocycle_ring_size),
            require_empty_persistent_overlay=bool(
                require_empty_persistent_overlay
            ),
        )
        if error:
            status = _external_error_status(error)
            data = {"center": center, "box_size": box_size, "audit": audit}
            if reconstruction_context is not None:
                data["reconstruction"] = reconstruction_context
            return _operation_result(operation, status, data=data, error=error)
        data = {
            "affinity_kcal_mol": affinity,
            "center": center,
            "box_size": box_size,
            "output_dir": str(output_dir) if output_dir else None,
            "audit": audit,
        }
        if reconstruction_context is not None:
            data["reconstruction"] = reconstruction_context
        return _operation_result(
            operation,
            "success",
            data=data,
        )
    except Exception as exc:
        return _exception_result(
            operation,
            exc,
            data=(
                {"reconstruction": reconstruction_context}
                if reconstruction_context is not None
                else None
            ),
        )


def batch_dock_structures(
    peptide_paths: Sequence[str | Path],
    receptor_path: str | Path,
    *,
    center: Sequence[float],
    box_size: Sequence[float] = (25.0, 25.0, 25.0),
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dock multiple peptide PDB files using the existing batch contract."""
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import monomer_resolution_context

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=_monomer_resolution_hints(
                peptide_paths, kind="coordinate"
            ),
        ) as resolution_ledger:
            result = batch_dock_structures(
                peptide_paths,
                receptor_path,
                center=center,
                box_size=box_size,
            )
        result.setdefault("data", {}).setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        return result
    operation = "batch_dock"
    if _is_scalar_collection_input(peptide_paths):
        return _operation_result(
            operation,
            "invalid_input",
            error="peptide_paths must be a collection of coordinate paths",
        )
    try:
        inputs = [str(path) for path in peptide_paths]
    except TypeError as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if not inputs:
        return _operation_result(operation, "invalid_input", error="no peptides supplied")
    try:
        normalized_center = tuple(float(value) for value in center)
        normalized_box_size = tuple(float(value) for value in box_size)
    except (TypeError, ValueError, OverflowError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    if len(normalized_center) != 3 or len(normalized_box_size) != 3:
        return _operation_result(
            operation,
            "invalid_input",
            error="center and box_size each require three values",
        )
    try:
        from .docking.workflow import batch_dock_peptides

        worker_result = batch_dock_peptides(
            inputs,
            str(receptor_path),
            normalized_center,
            normalized_box_size,
        )
        raw_rows, collection_error = _materialize_worker_rows(worker_result)
        rows = []
        for index, raw_row in enumerate(raw_rows):
            if (
                collection_error
                or not isinstance(raw_row, (list, tuple))
                or len(raw_row) != 3
            ):
                rows.append(_malformed_batch_row(operation, index, raw_row))
                continue
            name, affinity, error = raw_row
            rows.append(
                {
                    "name": name,
                    "affinity_kcal_mol": affinity,
                    "status": (
                        "success" if error is None else _external_error_status(error)
                    ),
                    "error": error,
                }
            )
        expected_ids = [_file_stem_identity(path) for path in inputs]
        contract = _validate_batch_rows(
            rows,
            expected_ids,
            identity_getter=_docking_row_identity,
            operation_label="batch docking",
            collection_error=collection_error,
        )
        success_count = sum(row["status"] == "success" for row in rows)
        errors = " | ".join(
            str(row["error"]) for row in rows if row.get("error")
        )
        failure_statuses = {row["status"] for row in rows if row["status"] != "success"}
        requested_count = len(inputs)
        returned_count = len(rows)
        contract_error = contract["error"]
        if contract_error:
            status = "partial" if success_count else "failed"
        else:
            status = (
                "success"
                if success_count == len(rows)
                else "partial" if success_count
                else next(iter(failure_statuses)) if len(failure_statuses) == 1
                else "failed"
            )
        return _operation_result(
            operation,
            status,
            data={
                "results": rows,
                "count": len(rows),
                "success_count": success_count,
                "requested_count": requested_count,
                "returned_count": returned_count,
                "raw_rows": json_ready(raw_rows),
                **contract,
                "receptor_path": str(receptor_path),
                "center": normalized_center,
                "box_size": normalized_box_size,
            },
            error=" | ".join(
                part for part in (contract_error, errors or None) if part
            )
            or None,
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def add_monomer(
    symbol: str,
    smiles: str,
    *,
    r1: str | None = None,
    r2: str | None = None,
    r3: str | None = None,
    overwrite: bool = False,
    persist: bool = True,
) -> dict[str, Any]:
    operation = "monomer_add"
    try:
        from .core.monomer_admin import add_monomer as register

        record = register(
            symbol,
            smiles,
            r1=r1,
            r2=r2,
            r3=r3,
            overwrite=bool(overwrite),
            persist=bool(persist),
        )
        return _operation_result(operation, "success", data={"monomer": record})
    except (KeyError, TypeError, ValueError) as exc:
        return _operation_result(operation, "invalid_input", error=str(exc))
    except Exception as exc:
        return _exception_result(operation, exc)


def resolve_monomers(
    symbols: Sequence[str],
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve/extend monomers without persisting entity-local chemistry."""
    operation = "monomer_resolve"
    try:
        requested = [str(symbol).strip() for symbol in symbols]
        if not requested or any(not symbol for symbol in requested):
            return _operation_result(
                operation,
                "invalid_input",
                error="symbols must contain nonempty monomer identifiers",
            )
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=requested,
        ) as ledger:
            pass
        data = json_ready(ledger)
        return _operation_result(operation, "success", data=data)
    except Exception as exc:
        return _exception_result(operation, exc)


def list_monomers(
    query: str | None = None,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import monomer_resolution_context

        with monomer_resolution_context(
            _runtime_monomer_context(monomer_context),
            required_symbols=(
                (str(query),) if query else ()
            ),
        ) as resolution_ledger:
            result = list_monomers(query)
        result.setdefault("data", {}).setdefault(
            "monomer_resolution", json_ready(resolution_ledger)
        )
        return result
    operation = "monomer_list"
    try:
        from .paths import _map_utils

        needle = (query or "").strip().lower()
        rows = []
        for symbol in sorted(_map_utils.monomers2smi_dict, key=str.lower):
            if needle and needle not in symbol.lower():
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "smiles": _map_utils.monomers2smi_dict[symbol],
                    "rgroups": _map_utils.monomers2r_groups_dict.get(symbol, {}),
                }
            )
        return _operation_result(
            operation, "success", data={"monomers": rows, "count": len(rows)}
        )
    except Exception as exc:
        return _exception_result(operation, exc)


def capabilities() -> dict[str, Any]:
    operation = "capabilities"
    from . import __version__ as package_version
    optional_modules = {
        "gui": "PyQt5",
        "admet": "admet_ai",
        "docking_preparation": "meeko",
        "mmcif": "gemmi",
        "bond_order_inference": "openbabel",
    }
    availability = {
        name: importlib.util.find_spec(module) is not None
        for name, module in optional_modules.items()
    }
    try:
        from .docking.vina import find_vina

        vina_path = find_vina()
    except Exception:
        vina_path = None
    availability["vina"] = bool(vina_path)
    prior_path = (
        Path(__file__).resolve().parent
        / "data"
        / "torsion_priors"
        / "torsion_priors_runtime.json"
    )
    prior_metadata = None
    prior_error = None
    if prior_path.is_file():
        try:
            from .docking.torsion_prior import load_torsion_prior

            prior = load_torsion_prior(prior_path)
            prior_metadata = {
                "runtime_sha256": prior.runtime_sha256,
                "manifest_sha256": prior.manifest_sha256,
                "level_entry_counts": {
                    level: len(entries)
                    for level, entries in prior.levels.items()
                },
            }
        except Exception as exc:
            prior_error = f"{type(exc).__name__}: {exc}"
    availability["torsion_prior"] = prior_metadata is not None
    package_root = Path(__file__).resolve().parent
    v5_resource_paths = {
        "artifact_schema": (
            package_root / "schemas" / "v5_artifact.schema.json"
        ),
        "applicability_manifest": (
            package_root / "data" / "applicability_manifest.json"
        ),
        "evidence_dossier": (
            package_root / "data" / "v5_evidence_dossier.json"
        ),
        "artifact_contract": (
            package_root / "V5_ARTIFACT_CONTRACT.md"
        ),
        "reproducibility": (
            package_root / "V5_REPRODUCIBILITY.md"
        ),
    }
    v5_resources = {
        name: {
            "path": str(path),
            "sha256": (
                _cached_sha256_file(path) if path.is_file() else None
            ),
        }
        for name, path in v5_resource_paths.items()
    }
    availability["v5_resources"] = all(
        value["sha256"] is not None
        for value in v5_resources.values()
    )
    return _operation_result(
        operation,
        "success",
        data={
            "version": package_version,
            "operations": [
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
                "read_mol2",
                "prepare_ligand_from_sequence",
                "prepare_ligand_pdbqt_from_mol2",
                "prepare_ligand_pdbqt",
                "prepare_ligand_pdbqt_from_pdb",
                "prepare_ligand_pdbqt_best_available",
                "prepare_receptor_pdbqt",
                "validate_pdbqt",
                "docking_center",
                "run_prepared_vina",
                "dock",
                "batch_dock",
                "monomer_list",
                "monomer_add",
                "monomer_resolve",
            ],
            "maintenance_entry_points": [
                "cycpep-overlay",
                "cycpep-v5-reproduce",
                "python -m cycpep_master.docking.build_template_library",
                "python -m cycpep_master.docking.build_torsion_priors",
            ],
            "reconstruction_paths": RECONSTRUCTION_PATHS,
            "coordinate_suffixes": SUPPORTED_COORDINATE_SUFFIXES,
            "representation_kinds": REPRESENTATION_KINDS,
            "exact_representation": {
                "schema": "cycpep_exact_v1",
                "schema_path": str(
                    Path(__file__).resolve().parent
                    / "schemas"
                    / "exact_v1.schema.json"
                ),
                "general_smiles_decomposition": False,
                "edge_v1_default_max_rings": 3,
                "edge_v1_default_max_position": 32,
            },
            "monomer_resolution": {
                "entity_local_extensions": True,
                "custom_definitions": True,
                "standalone_ccd": True,
                "source_bound_component_templates": True,
                "on_demand_network_fetch_requires_opt_in": True,
                "unresolved_output": "C1:H_PARTIAL",
                "persistent_writes": False,
                "formal_parallelism": "one_entity_per_process",
            },
            "audit_kinds": AUDIT_KINDS,
            "template_strategies": TEMPLATE_STRATEGIES,
            "pdbqt_flexibility_modes": [
                "fast",
                "balanced",
                "thorough",
            ],
            "pdbqt_parent_contract": "validated_mol2_required",
            "retired_operations": {
                "prepare_ligand_pdbqt_ensemble": (
                    "Use a ConformerEnsembleArtifact and prepare each "
                    "validated MOL2 through prepare_ligand_pdbqt_from_mol2"
                )
            },
            "v5_artifact_chain": [
                "InputArtifact",
                "ChemicalGraphArtifact",
                "ConformerEnsembleArtifact",
                "ValidatedMol2Artifact",
                "FlexibilityAssessmentArtifact",
                "PdbqtArtifact",
            ],
            "v5_artifact_schema_path": str(
                Path(__file__).resolve().parent
                / "schemas"
                / "v5_artifact.schema.json"
            ),
            "applicability_manifest_path": str(
                Path(__file__).resolve().parent
                / "data"
                / "applicability_manifest.json"
            ),
            "v5_resources": v5_resources,
            "torsion_prior_path": (
                str(prior_path) if prior_metadata is not None else None
            ),
            "torsion_prior": prior_metadata,
            "torsion_prior_error": prior_error,
            "availability": availability,
            "vina_path": vina_path,
        },
    )


__all__ = [
    "AUDIT_KINDS",
    "RECONSTRUCTION_PATHS",
    "REPRESENTATION_KINDS",
    "SUPPORTED_COORDINATE_SUFFIXES",
    "SUPPORTED_DOCKING_COORDINATE_SUFFIXES",
    "TEMPLATE_STRATEGIES",
    "add_monomer",
    "audit_chemistry",
    "batch_dock_structures",
    "batch_export_structures",
    "capabilities",
    "clear_caches",
    "compare_chemistry",
    "conformer_statistics",
    "convert_representation",
    "discover_coordinate_files",
    "discover_docking_coordinate_files",
    "dock_structure",
    "docking_center",
    "export_best_available",
    "export_structure",
    "find_conformer_template",
    "generate_template_conformers",
    "json_ready",
    "list_monomers",
    "predict_admet",
    "prepare_ligand_pdbqt_from_mol2",
    "prepare_ligand_from_sequence",
    "prepare_ligand_pdbqt",
    "prepare_ligand_pdbqt_best_available",
    "prepare_ligand_pdbqt_from_pdb",
    "prepare_receptor_pdbqt",
    "protonate_smiles",
    "protonate_mol2",
    "read_mol2",
    "resolve_monomers",
    "reconstruct_coordinates",
    "reconstruct_exact_v1",
    "reconstruct_structure",
    "reconstruct_unified",
    "reconstruct_result_first",
    "reconstruct_multichain",
    "run_prepared_vina",
    "validate_mol2",
    "validate_pdbqt",
]
