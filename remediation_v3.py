"""Versioned fail-closed remediation APIs.

The historical v2 code remains byte-identical for frozen-result
reproducibility.  New callers should use this module for strict input audits,
strict structured assembly, strict F/H corroboration, and consensus
reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings

from rdkit import Chem

from . import __version__
from .core.identity_memo import identity_memo_context, molecular_identity
from .chemical_audit import (
    AuditResult,
    audit_assembly_json,
    audit_output_smiles,
    audit_payload,
    audit_pdb_file,
)


@dataclass
class RemediationResult:
    status: str
    output_smiles: str | None = None
    output_inchikey: str | None = None
    rejection_reason: str | None = None
    warning_codes: list[str] = field(default_factory=list)
    path_used: str | None = None
    route_results: list[dict] = field(default_factory=list)


def _inchikey(smiles: str | None) -> str | None:
    identity = molecular_identity(smiles)
    return identity.full_inchikey if identity is not None else None


def _canonical(smiles: str | None) -> str | None:
    identity = molecular_identity(smiles)
    return identity.canonical_smiles if identity is not None else None


def _reject_from_audit(audit: AuditResult) -> RemediationResult:
    warning_codes = list(audit.warning_codes)
    if (
        "UNKNOWN_MONOMER" in warning_codes
        and "CHEMICAL_WARNING_PRESENT" not in warning_codes
    ):
        warning_codes.insert(0, "CHEMICAL_WARNING_PRESENT")
    return RemediationResult(
        status="rejected",
        rejection_reason=audit.reason,
        warning_codes=warning_codes,
        path_used="INPUT_AUDIT",
    )


def convert_payload_fail_closed(kind: str, payload: str) -> RemediationResult:
    """Audit notation semantics before conversion and audit the output again."""

    input_audit = audit_payload(kind, payload)
    if not input_audit.accepted:
        return _reject_from_audit(input_audit)

    from .paths._map_utils import (
        get_smi_from_biln,
        get_smi_from_map,
        helm_to_map,
    )

    normalized = kind.strip().lower()
    output: str | None = None
    captured_codes: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if normalized == "biln":
                output = get_smi_from_biln(payload)
            elif normalized == "map":
                output = get_smi_from_map(payload)
            elif normalized == "helm":
                mapped = helm_to_map(payload)
                output = (
                    get_smi_from_map(mapped)
                    if mapped and not mapped.startswith("ERROR")
                    else None
                )
            elif normalized in {"smiles", "smi"}:
                output = _canonical(payload)
            captured_codes = sorted(
                {
                    f"{item.category.__name__}:{str(item.message)[:160]}"
                    for item in caught
                }
            )
    except Exception as exc:
        return RemediationResult(
            status="rejected",
            rejection_reason=f"{type(exc).__name__}: {exc}",
            warning_codes=["CONVERSION_EXCEPTION"],
            path_used="INPUT_AUDIT",
        )

    if captured_codes:
        return RemediationResult(
            status="rejected",
            rejection_reason="conversion emitted one or more chemical warnings",
            warning_codes=["CHEMICAL_WARNING_PRESENT", *captured_codes],
            path_used="INPUT_AUDIT",
        )
    output_audit = audit_output_smiles(output)
    if not output_audit.accepted:
        return RemediationResult(
            status="rejected",
            rejection_reason=output_audit.reason,
            warning_codes=output_audit.warning_codes,
            path_used="OUTPUT_AUDIT",
        )
    canonical = _canonical(output)
    return RemediationResult(
        status="success",
        output_smiles=canonical,
        output_inchikey=_inchikey(canonical),
        path_used="AUDITED_CONVERSION",
    )


def _assembly_map_payload(document: dict) -> str:
    """Translate an already-audited strict assembly graph without inference."""

    monomer_rows = document["monomers"]
    monomers = {row["monomer_id"]: row for row in monomer_rows}
    successor: dict[str, str] = {}
    sidechain_links: list[tuple[str, str]] = []
    for connection in document["connections"]:
        source = connection["source"]
        target = connection["target"]
        if connection["bond_type"] == "amide":
            if source["port"] == "R2":
                successor[source["monomer_id"]] = target["monomer_id"]
            else:
                successor[target["monomer_id"]] = source["monomer_id"]
        elif connection["bond_type"] == "disulfide":
            sidechain_links.append(
                (source["monomer_id"], target["monomer_id"])
            )

    start = monomer_rows[0]["monomer_id"]
    ordered_ids: list[str] = []
    current = start
    for _ in range(len(monomer_rows)):
        ordered_ids.append(current)
        current = successor[current]
    positions = {
        monomer_id: index
        for index, monomer_id in enumerate(ordered_ids, start=1)
    }
    sequence = "".join(str(monomers[item]["symbol"]) for item in ordered_ids)
    annotations = [f"{{cyc:1:R1-{len(ordered_ids)}:R2}}"]
    for first, second in sidechain_links:
        annotations.append(
            f"{{cyc:{positions[first]}:R3-{positions[second]}:R3}}"
        )
    return sequence + "".join(annotations)


def convert_assembly_payload_fail_closed(payload: str) -> RemediationResult:
    """Strictly audit and assemble a versioned ``assembly_json`` payload.

    Every connection is explicit.  Missing ports, geometric alternatives,
    charge changes, atom-map repair, cap removal, and cross-chain inference
    are rejected before any molecular assembly function is called.
    """

    started = time.perf_counter()
    audit = audit_assembly_json(payload)
    if not audit.accepted:
        return _reject_from_audit(audit)
    try:
        document = json.loads(payload)
        mapped = _assembly_map_payload(document)
    except (KeyError, TypeError, ValueError) as exc:
        return RemediationResult(
            status="rejected",
            rejection_reason=(
                "audited assembly graph could not be serialized without "
                f"inference: {type(exc).__name__}: {exc}"
            ),
            warning_codes=["ASSEMBLY_SERIALIZATION_FAILED"],
            path_used="STRICT_ASSEMBLY_JSON",
        )

    result = convert_payload_fail_closed("map", mapped)
    runtime_sec = time.perf_counter() - started
    if result.status != "success":
        return RemediationResult(
            status="rejected",
            rejection_reason=result.rejection_reason,
            warning_codes=result.warning_codes,
            path_used="STRICT_ASSEMBLY_JSON",
            route_results=[
                {
                    "route": "explicit_rgroup_assembly",
                    "status": result.status,
                    "runtime_sec": runtime_sec,
                    "output_smiles": result.output_smiles,
                    "output_inchikey": result.output_inchikey,
                    "candidate_identity": None,
                    "corroborator_identity": None,
                    "error": result.rejection_reason,
                    "warnings": list(result.warning_codes),
                    "warning_codes": list(result.warning_codes),
                }
            ],
        )
    assembled = Chem.MolFromSmiles(result.output_smiles or "")
    observed_charge = (
        sum(atom.GetFormalCharge() for atom in assembled.GetAtoms())
        if assembled is not None
        else None
    )
    declared_charge = document["declared_formal_charge"]
    if observed_charge != declared_charge:
        reason = (
            "assembled formal charge "
            f"{observed_charge!r} does not match declared formal charge "
            f"{declared_charge!r}"
        )
        return RemediationResult(
            status="rejected",
            rejection_reason=reason,
            warning_codes=["FORMAL_CHARGE_CONFLICT"],
            path_used="STRICT_ASSEMBLY_JSON",
            route_results=[
                {
                    "route": "explicit_rgroup_assembly",
                    "status": "rejected",
                    "runtime_sec": runtime_sec,
                    "output_smiles": None,
                    "output_inchikey": None,
                    "candidate_identity": None,
                    "corroborator_identity": None,
                    "error": reason,
                    "warnings": ["FORMAL_CHARGE_CONFLICT"],
                    "warning_codes": ["FORMAL_CHARGE_CONFLICT"],
                }
            ],
        )
    return RemediationResult(
        status="success",
        output_smiles=result.output_smiles,
        output_inchikey=result.output_inchikey,
        path_used="STRICT_ASSEMBLY_JSON:EXPLICIT_RGROUP_GRAPH",
        route_results=[
            {
                "route": "explicit_rgroup_assembly",
                "status": "success",
                "runtime_sec": runtime_sec,
                "output_smiles": result.output_smiles,
                "output_inchikey": result.output_inchikey,
                "candidate_identity": {
                    "output_smiles": result.output_smiles,
                    "output_inchikey": result.output_inchikey,
                },
                "corroborator_identity": None,
                "error": None,
                "warnings": [],
                "warning_codes": [],
            }
        ],
    )


def _identity(smiles: str | None) -> dict[str, str | None] | None:
    if not smiles:
        return None
    canonical = _canonical(smiles)
    return {
        "output_smiles": canonical or smiles,
        "output_inchikey": _inchikey(canonical or smiles),
    }


def _strict_geometric_route_detailed(
    route: str,
    pdb_path: str | Path,
    chain_id: str,
    *,
    corroborator_provider=None,
    pdb_audit: AuditResult | None = None,
) -> tuple[str | None, str | None, dict]:
    detail = {
        "candidate_identity": None,
        "corroborator_identity": None,
        "corroborator_route": "g",
        "warning_codes": [],
    }
    pdb_audit = pdb_audit or audit_pdb_file(pdb_path)
    if not pdb_audit.accepted:
        detail["warning_codes"] = list(pdb_audit.warning_codes)
        return None, pdb_audit.reason, detail

    from .paths.path_f import generate_f
    from .paths.path_g import generate_g
    from .paths.path_h import generate_h

    generator = generate_f if route == "f" else generate_h
    candidate, error = generator(str(pdb_path), chain_id)
    detail["candidate_identity"] = _identity(candidate)
    if not candidate:
        return (
            None,
            error or f"Path {route.upper()} produced no molecule",
            detail,
        )
    candidate_audit = audit_output_smiles(candidate)
    if not candidate_audit.accepted:
        detail["warning_codes"] = list(candidate_audit.warning_codes)
        return None, candidate_audit.reason, detail

    if corroborator_provider is None:
        reference, reference_error = generate_g(str(pdb_path), chain_id)
    else:
        corroborator = corroborator_provider()
        reference = corroborator["output"]
        reference_error = corroborator["error"]
        detail["warning_codes"] = sorted(set(
            detail.get("warning_codes", [])
            + list(corroborator.get("warning_codes", []))
        ))
    detail["corroborator_identity"] = _identity(reference)
    if not reference:
        return (
            None,
            "strict geometric identity requires independent library-driven "
            f"corroboration; Path G unavailable: {reference_error}",
            detail,
        )
    reference_audit = audit_output_smiles(reference)
    if not reference_audit.accepted:
        detail["warning_codes"] = list(reference_audit.warning_codes)
        return (
            None,
            "Path G corroborator failed output audit: "
            f"{reference_audit.reason}",
            detail,
        )
    candidate_key = _inchikey(candidate)
    reference_key = _inchikey(reference)
    if not candidate_key or candidate_key != reference_key:
        return (
            None,
            f"Path {route.upper()} approximate identity {candidate_key} "
            f"disagrees with library-driven identity {reference_key}",
            detail,
        )
    return _canonical(candidate), None, detail


def _strict_geometric_route(
    route: str,
    pdb_path: str | Path,
    chain_id: str,
) -> tuple[str | None, str | None]:
    output, error, _ = _strict_geometric_route_detailed(
        route, pdb_path, chain_id
    )
    return output, error


def generate_f_strict(
    pdb_path: str | Path,
    chain_id: str = "L",
) -> tuple[str | None, str | None]:
    """Path F accepted only when its full identity is independently corroborated."""

    return _strict_geometric_route("f", pdb_path, chain_id)


def generate_h_strict(
    pdb_path: str | Path,
    chain_id: str = "L",
) -> tuple[str | None, str | None]:
    """Path H accepted only when its full identity is independently corroborated."""

    return _strict_geometric_route("h", pdb_path, chain_id)


def reconstruct_pdb_fail_closed(
    pdb_path: str | Path,
    chain_id: str = "L",
    *,
    _execution_artifacts: dict | None = None,
    _pdb_audit: AuditResult | None = None,
) -> RemediationResult:
    """Accept two independent chemical families with no F/H identity veto."""
    with identity_memo_context():
        return _reconstruct_pdb_fail_closed_impl(
            pdb_path,
            chain_id,
            _execution_artifacts=_execution_artifacts,
            _pdb_audit=_pdb_audit,
        )


def _reconstruct_pdb_fail_closed_impl(
    pdb_path: str | Path,
    chain_id: str,
    *,
    _execution_artifacts: dict | None,
    _pdb_audit: AuditResult | None,
) -> RemediationResult:
    pdb_audit = _pdb_audit or audit_pdb_file(pdb_path)
    if not pdb_audit.accepted:
        return _reject_from_audit(pdb_audit)

    from .paths import generate_a, generate_b, generate_c, generate_e, generate_g

    generators = {
        "a": generate_a,
        "b": generate_b,
        "c": generate_c,
        "e": generate_e,
        "g": generate_g,
    }
    execution_artifacts = (
        _execution_artifacts if _execution_artifacts is not None else {}
    )
    shared_g: dict | None = None

    def provide_g() -> dict:
        nonlocal shared_g
        if shared_g is not None:
            if shared_g.get("exception") is not None:
                raise shared_g["exception"]
            return shared_g
        from .paths.path_g import generate_g_with_artifacts

        started = time.perf_counter()
        caught_codes: list[str] = []
        exception = None
        output = None
        error = None
        artifact = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output, error, artifact = generate_g_with_artifacts(
                    str(pdb_path), chain_id
                )
                caught_codes = sorted({
                    f"{item.category.__name__}:{str(item.message)[:160]}"
                    for item in caught
                })
        except Exception as exc:
            exception = exc
            error = f"{type(exc).__name__}: {exc}"
        shared_g = {
            "output": output,
            "error": error,
            "warning_codes": caught_codes,
            "runtime_sec": time.perf_counter() - started,
            "exception": exception,
            "artifact": artifact,
        }
        if artifact is not None:
            execution_artifacts[("g", True)] = artifact
        if exception is not None:
            raise exception
        return shared_g

    route_rows: list[dict] = []
    accepted: list[tuple[str, str, str]] = []
    for route in ("a", "b", "c", "e", "f", "g", "h"):
        generator = generators.get(route)
        output: str | None = None
        error: str | None = None
        caught_codes: list[str] = []
        detail: dict = {
            "candidate_identity": None,
            "corroborator_identity": None,
            "warning_codes": [],
        }
        raised = False
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if route in {"f", "h"}:
                    output, error, detail = _strict_geometric_route_detailed(
                        route,
                        pdb_path,
                        chain_id,
                        corroborator_provider=provide_g,
                        pdb_audit=pdb_audit,
                    )
                elif route == "g":
                    corroborator = provide_g()
                    output = corroborator["output"]
                    error = corroborator["error"]
                elif route == "b" and _execution_artifacts is not None:
                    from .paths.path_b import generate_with_artifacts

                    output, error, artifact = generate_with_artifacts(
                        str(pdb_path), chain_id
                    )
                    execution_artifacts[("b", True)] = artifact
                elif generator is not None:
                    output, error = generator(str(pdb_path), chain_id)
                caught_codes = sorted(
                    {
                        f"{item.category.__name__}:{str(item.message)[:160]}"
                        for item in caught
                    }
                )
                if route == "g" and shared_g is not None:
                    caught_codes = list(shared_g["warning_codes"])
        except Exception as exc:
            raised = True
            error = f"{type(exc).__name__}: {exc}"
        runtime_sec = time.perf_counter() - started
        if route == "g" and shared_g is not None:
            runtime_sec = float(shared_g["runtime_sec"])
        output_audit = audit_output_smiles(output)
        computed_identity = _identity(output)
        if detail.get("candidate_identity") is None:
            detail["candidate_identity"] = computed_identity
        route_warning_codes = sorted(
            set(caught_codes)
            .union(output_audit.warning_codes)
            .union(detail.get("warning_codes", []))
        )
        route_ok = bool(
            output
            and output_audit.accepted
            and not error
            and not route_warning_codes
            and computed_identity
            and computed_identity.get("output_inchikey")
        )
        key = (
            str(computed_identity["output_inchikey"])
            if route_ok and computed_identity
            else None
        )
        canonical = (
            str(computed_identity["output_smiles"])
            if route_ok and computed_identity
            else None
        )
        visible_identity = detail.get("candidate_identity")
        visible_smiles = (
            visible_identity.get("output_smiles")
            if isinstance(visible_identity, dict)
            else None
        )
        visible_key = (
            visible_identity.get("output_inchikey")
            if isinstance(visible_identity, dict)
            else None
        )
        if error is None and not route_ok:
            if caught_codes:
                error = "route emitted one or more chemical warnings"
            else:
                error = output_audit.reason or "route produced no audited identity"
        route_rows.append(
            {
                "route": route,
                "status": (
                    "success"
                    if route_ok
                    else ("failed" if raised else "rejected")
                ),
                "runtime_sec": runtime_sec,
                "output_smiles": visible_smiles,
                "output_inchikey": visible_key,
                "candidate_identity": detail.get("candidate_identity"),
                "corroborator_identity": detail.get(
                    "corroborator_identity"
                ),
                "error": error,
                "warnings": route_warning_codes,
                "warning_codes": route_warning_codes,
            }
        )
        if route_ok and key and canonical:
            accepted.append((route, canonical, key))

    route_families = {
        "a": "residue_template",
        "c": "residue_template",
        "e": "residue_template",
        "b": "monomer_library",
        "g": "monomer_library",
    }
    qualifying = [row for row in accepted if row[0] in route_families]
    successful_families = {
        route_families[route] for route, _smiles, _key in qualifying
    }
    if len(successful_families) < 2:
        return RemediationResult(
            status="rejected",
            rejection_reason=(
                f"{len(successful_families)} independent qualifying family/families; "
                "at least residue-template and monomer-library evidence are required"
            ),
            warning_codes=["INSUFFICIENT_CROSS_VALIDATION"],
            path_used="V4_AUDITED_CONSENSUS",
            route_results=route_rows,
        )

    unique_keys = sorted({row[2] for row in qualifying})
    f_h_candidate_keys = {
        str(identity["output_inchikey"])
        for row in route_rows
        if row["route"] in {"f", "h"}
        for identity in [row.get("candidate_identity")]
        if isinstance(identity, dict) and identity.get("output_inchikey")
    }
    selected_candidate_key = unique_keys[0] if len(unique_keys) == 1 else None
    f_h_conflict = bool(
        selected_candidate_key is not None
        and any(key != selected_candidate_key for key in f_h_candidate_keys)
    )
    if len(unique_keys) != 1 or len(f_h_candidate_keys) > 1 or f_h_conflict:
        conflict_detail = (
            f"; F/H candidate keys={sorted(f_h_candidate_keys)}"
            if f_h_candidate_keys
            else ""
        )
        return RemediationResult(
            status="rejected",
            rejection_reason=(
                f"{len(unique_keys)} distinct audited successful full "
                f"InChIKeys{conflict_detail}"
            ),
            warning_codes=["MULTI_PATH_FULL_INCHIKEY_CONFLICT"],
            path_used="V4_AUDITED_CONSENSUS",
            route_results=route_rows,
        )
    selected_route, selected_smiles, selected_key = qualifying[0]
    successful_routes = ",".join(row[0].upper() for row in accepted)
    return RemediationResult(
        status="success",
        output_smiles=selected_smiles,
        output_inchikey=selected_key,
        path_used=f"V4_CONSENSUS:{successful_routes}",
        route_results=route_rows,
    )


def remediation_version() -> str:
    return f"{__version__}+remediation.4"
