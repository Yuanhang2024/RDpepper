"""Unified, entity-local monomer resolution and runtime extension.

Every public workflow that consumes the monomer library can enter this
context.  Built-in and persistent user monomers retain their existing
precedence.  Explicit custom definitions and standalone CCD components are
compiled into the existing 235-column derived-row contract and activated only
for the current operation.  Unresolved definitions are returned in the ledger
instead of forcing the caller to reject the whole operation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import gzip
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen


EPHEMERAL_MONOMER_ID_START = 2_000_000_000
CCD_DOWNLOAD_URL = (
    "https://files.rcsb.org/ligands/download/{component_id}.cif"
)
_SAFE_COMPONENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,15}$")
_ACTIVE_LEDGER: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "cycpep_master_monomer_resolution_ledger",
    default=None,
)
_NNR_TOKEN = re.compile(r"\{nnr:([^{}]+)\}", re.IGNORECASE)
_BRACKET_TOKEN = re.compile(r"\[([^\[\]]+)\]")
_HELM_BLOCK = re.compile(
    r"PEPTIDE[1-9][0-9]*\{([^{}]*)\}", re.IGNORECASE
)
_BILN_TOKEN = re.compile(r"(?:^|-)([A-Z]|\[[^\[\]]+\])")


def active_monomer_resolution() -> Mapping[str, Any] | None:
    """Return the current operation-local resolution ledger, if any."""
    return _ACTIVE_LEDGER.get()


def needs_monomer_resolution_scope(
    context: Mapping[str, Any] | None,
) -> bool:
    """Return whether a public registry consumer must open a scope."""
    return context is not None or active_monomer_resolution() is None


def monomer_symbol_hints(
    payload: Any,
    *,
    kind: str | None = None,
) -> tuple[str, ...]:
    """Return syntax-level monomer identifiers without resolving chemistry.

    The helper intentionally performs no graph inference.  It exists so a
    request-scoped CCD provider can be asked for an unknown component before
    the strict notation parser consults the active registry.
    """
    if isinstance(payload, Mapping):
        values = []
        for row in payload.get("monomers") or ():
            if isinstance(row, Mapping):
                symbol = str(
                    row.get("monomer_symbol")
                    or row.get("symbol")
                    or ""
                ).strip()
                if symbol:
                    values.append(symbol)
        for row in payload.get("caps") or ():
            if isinstance(row, Mapping):
                symbol = str(
                    row.get("monomer_symbol")
                    or row.get("symbol")
                    or ""
                ).strip()
                if symbol:
                    values.append(symbol)
        return tuple(dict.fromkeys(values))
    if isinstance(payload, (list, tuple, set)):
        values = []
        for item in payload:
            values.extend(monomer_symbol_hints(item, kind=kind))
        return tuple(dict.fromkeys(values))
    text = str(payload or "").strip()
    if not text:
        return ()
    normalized = str(kind or "").strip().lower().replace("-", "_")
    path = Path(text)
    try:
        path_is_file = path.is_file()
    except OSError:
        path_is_file = False
    if (
        normalized in {"coordinate", "pdb", "mmcif"}
        or path_is_file
    ):
        return _coordinate_symbol_hints(path)

    values = [
        match.group(1).strip() for match in _NNR_TOKEN.finditer(text)
        if match.group(1).strip()
    ]
    values.extend(
        match.group(1).strip() for match in _BRACKET_TOKEN.finditer(text)
        if match.group(1).strip()
    )
    if normalized == "helm" or (not normalized and "PEPTIDE" in text.upper()):
        for block in _HELM_BLOCK.findall(text):
            for token in block.split("."):
                symbol = token.strip().strip("[]").strip()
                if symbol:
                    values.append(symbol)
    elif normalized == "biln":
        for match in _BILN_TOKEN.finditer(text):
            symbol = match.group(1).strip().strip("[]").strip()
            if symbol:
                values.append(symbol)
    elif normalized == "sequence":
        scrubbed = _NNR_TOKEN.sub("", _BRACKET_TOKEN.sub("", text))
        values.extend(
            character for character in scrubbed
            if character.isalpha() and character.isupper()
        )
    return tuple(dict.fromkeys(values))


def _coordinate_symbol_hints(path: Path) -> tuple[str, ...]:
    """Read component identifiers only; never infer bonds or identities."""
    if not path.is_file():
        return ()
    lower = path.name.lower()
    if lower.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        try:
            from .native_mmcif_graph import inspect_mmcif

            return tuple(dict.fromkeys(
                component
                for record in inspect_mmcif(path)
                if record.peptide_bearing
                for component in record.component_ids
                if component
            ))
        except Exception:
            return ()
    opener = gzip.open if lower.endswith(".gz") else open
    residues: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        with opener(path, "rt", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")):
                    value = line[17:20].strip()
                    if value:
                        key = (
                            line[21:22],
                            line[22:26],
                            line[26:27],
                            value,
                        )
                        row = residues.setdefault(
                            key, {"atoms": set(), "polymer": False}
                        )
                        row["atoms"].add(line[12:16].strip().upper())
                        row["polymer"] = (
                            row["polymer"] or line.startswith("ATOM  ")
                        )
    except OSError:
        return ()
    values = [
        key[3]
        for key, row in residues.items()
        if row["polymer"]
        or {"N", "CA", "C"}.issubset(row["atoms"])
    ]
    return tuple(dict.fromkeys(values))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_component_id(value: Any) -> str:
    component_id = str(value or "").strip().upper()
    if not _SAFE_COMPONENT_ID.fullmatch(component_id):
        raise ValueError(f"invalid CCD component ID: {value!r}")
    return component_id


def _read_component_file(path: str | Path) -> dict[str, dict]:
    from .mmcif_chem_comp import extract_embedded_chem_comp_templates

    return extract_embedded_chem_comp_templates(Path(path))


def _download_component(
    component_id: str,
    *,
    timeout_seconds: float,
) -> bytes:
    request = Request(
        CCD_DOWNLOAD_URL.format(component_id=component_id),
        headers={"User-Agent": "CycPep-Master/monomer-resolution"},
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read()


def _component_from_bytes(
    payload: bytes,
    component_id: str,
) -> dict:
    with tempfile.NamedTemporaryFile(
        suffix=f"_{component_id}.cif", delete=False
    ) as handle:
        path = Path(handle.name)
        handle.write(payload)
    try:
        templates = _read_component_file(path)
    finally:
        path.unlink(missing_ok=True)
    if component_id not in templates:
        raise ValueError(
            f"CCD payload does not contain component {component_id}"
        )
    return templates[component_id]


def _load_requested_components(
    component_ids: Mapping[str, str],
    *,
    ccd_files: Sequence[str | Path],
    ccd_directory: str | Path | None,
    allow_network: bool,
    network_timeout_seconds: float,
    cache_directory: str | Path | None,
) -> tuple[dict[str, dict], list[dict]]:
    templates: dict[str, dict] = {}
    errors: list[dict] = []
    for raw_path in ccd_files:
        try:
            templates.update(_read_component_file(raw_path))
        except Exception as exc:
            errors.append({
                "source": str(raw_path),
                "code": "CCD_FILE_UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
            })

    directory = Path(ccd_directory).resolve() if ccd_directory else None
    cache = Path(cache_directory).resolve() if cache_directory else None
    for raw_component in sorted(set(component_ids.values())):
        component_id = _safe_component_id(raw_component)
        if component_id in templates:
            continue
        candidates = []
        for root in (directory, cache):
            if root is None:
                continue
            candidates.extend((
                root / f"{component_id}.cif",
                root / component_id[0].lower() / f"{component_id}.cif",
            ))
        loaded = False
        for path in candidates:
            if not path.is_file():
                continue
            try:
                templates.update(_read_component_file(path))
                loaded = component_id in templates
            except Exception as exc:
                errors.append({
                    "component_id": component_id,
                    "source": str(path),
                    "code": "CCD_COMPONENT_PARSE_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if loaded:
                break
        if loaded or not allow_network:
            continue
        try:
            payload = _download_component(
                component_id,
                timeout_seconds=network_timeout_seconds,
            )
            templates[component_id] = _component_from_bytes(
                payload, component_id
            )
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                destination = cache / f"{component_id}.cif"
                with tempfile.NamedTemporaryFile(
                    dir=cache,
                    prefix=f".{component_id}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                temporary.replace(destination)
        except Exception as exc:
            errors.append({
                "component_id": component_id,
                "source": "rcsb_ccd",
                "code": "CCD_COMPONENT_FETCH_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return templates, errors


def _component_candidates(
    context: Mapping[str, Any],
    required_symbols: Sequence[str],
) -> tuple[list[dict], list[dict]]:
    from .mmcif_chem_comp import (
        materialize_standalone_peptide_component,
    )

    raw_ids = context.get("component_ids") or {}
    if isinstance(raw_ids, Mapping):
        component_ids = {
            str(symbol): _safe_component_id(component)
            for symbol, component in raw_ids.items()
        }
    else:
        component_ids = {
            _safe_component_id(component): _safe_component_id(component)
            for component in raw_ids
        }
    provided_templates = {}
    for raw_key, raw_component in (
        context.get("component_templates") or {}
    ).items():
        if not isinstance(raw_component, Mapping):
            errors = [{
                "component_id": str(raw_key),
                "code": "CCD_COMPONENT_TEMPLATE_INVALID",
            }]
            return [], errors
        component = dict(raw_component)
        component_id = _safe_component_id(
            component.get("component_id") or raw_key
        )
        provided_templates[component_id] = component
        if component_id not in set(component_ids.values()):
            component_ids.setdefault(str(raw_key), component_id)
    auto_resolve = context.get("auto_resolve_required_symbols")
    if auto_resolve is None:
        auto_resolve = bool(
            context.get("ccd_files")
            or context.get("ccd_directory")
            or context.get("allow_network")
        )
    if auto_resolve:
        from ..exact_v1 import MonomerRegistry

        registry = MonomerRegistry()
        for symbol in required_symbols:
            raw_symbol = str(symbol).strip()
            if not _SAFE_COMPONENT_ID.fullmatch(raw_symbol):
                continue
            try:
                from ..paths.residue_template_factory import resolve_symbol

                registry.resolve(resolve_symbol(raw_symbol))
                continue
            except Exception:
                pass
            component_ids.setdefault(
                raw_symbol, _safe_component_id(raw_symbol)
            )
    templates, errors = _load_requested_components(
        component_ids,
        ccd_files=tuple(context.get("ccd_files") or ()),
        ccd_directory=context.get("ccd_directory"),
        allow_network=bool(context.get("allow_network", False)),
        network_timeout_seconds=float(
            context.get("network_timeout_seconds", 30.0)
        ),
        cache_directory=context.get("cache_directory"),
    )
    for component_id, component in provided_templates.items():
        existing = templates.get(component_id)
        if (
            existing is not None
            and existing.get("component_snapshot_sha256")
            != component.get("component_snapshot_sha256")
        ):
            errors.append({
                "component_id": component_id,
                "code": "CCD_COMPONENT_SOURCE_CONFLICT",
                "selected_source": "request_component_template",
            })
        templates[component_id] = component
    r3_ports = context.get("r3_ports") or {}
    candidates = []
    for symbol, component_id in sorted(component_ids.items()):
        component = templates.get(component_id)
        if component is None:
            errors.append({
                "symbol": symbol,
                "component_id": component_id,
                "code": "CCD_COMPONENT_UNAVAILABLE",
            })
            continue
        port = (
            r3_ports.get(symbol)
            or r3_ports.get(component_id)
            or {}
        )
        try:
            compiled = materialize_standalone_peptide_component(
                component,
                r3_atom_name=port.get("atom_name"),
                r3_cap=port.get("cap"),
            )
            try:
                from rdkit import Chem
                from ..paths.residue_template_factory import (
                    get_residue_template_for_symbol,
                    resolve_symbol,
                )

                existing_symbol = resolve_symbol(symbol)
                existing = get_residue_template_for_symbol(
                    existing_symbol
                )
                existing_molecule = Chem.MolFromSmiles(
                    existing.free_smiles
                )
                existing_key = (
                    Chem.MolToInchiKey(existing_molecule)
                    if existing_molecule is not None
                    else None
                )
            except Exception:
                existing_symbol = None
                existing_key = None
            if existing_symbol is not None:
                if existing_key != compiled.get("full_inchikey"):
                    errors.append({
                        "symbol": symbol,
                        "component_id": component_id,
                        "code": "CCD_UNIFIED_IDENTITY_CONFLICT",
                        "existing_symbol": existing_symbol,
                        "existing_full_inchikey": existing_key,
                        "ccd_full_inchikey": compiled.get(
                            "full_inchikey"
                        ),
                    })
                    continue
                candidates.append({
                    "pdb_resname": symbol,
                    "force_alias_target": existing_symbol,
                    "graph_sha256": hashlib.sha256(
                        str(compiled["free_smiles"]).encode("utf-8")
                    ).hexdigest(),
                    "full_inchikey": compiled.get("full_inchikey"),
                    "resolution_mode": "standalone_ccd_exact_alias",
                    "source_entity_id": f"CCD:{component_id}",
                })
                continue
            candidates.append({
                "pdb_resname": symbol,
                "smiles": compiled["free_smiles"],
                "source_cxsmiles": compiled.get(
                    "ported_cxsmiles"
                ),
                "source_r_groups": compiled.get("r_groups"),
                "r3_port": (
                    {
                        "atom_name": compiled.get("r3_atom_name"),
                        "cap": compiled.get("r3_cap"),
                        "evidence_source": (
                            "standalone_ccd_component"
                        ),
                    }
                    if compiled.get("r3_atom_name")
                    else None
                ),
                "component_snapshot_sha256": compiled.get(
                    "component_snapshot_sha256"
                ),
                "component_source_input_sha256": compiled.get(
                    "source_input_sha256"
                ),
                "resolution_mode": "standalone_ccd_component",
                "source": "local_structure_derived",
                "source_entity_id": f"CCD:{component_id}",
            })
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "component_id": component_id,
                "code": "CCD_COMPONENT_NOT_MATERIALIZABLE",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return candidates, errors


def _explicit_candidates(
    definitions: Iterable[Mapping[str, Any]],
) -> tuple[list[dict], list[dict]]:
    candidates = []
    errors = []
    for index, raw in enumerate(definitions, start=1):
        definition = dict(raw)
        symbol = str(
            definition.get("symbol")
            or definition.get("monomer_id")
            or ""
        ).strip()
        smiles = str(definition.get("smiles") or "").strip()
        if not symbol or not smiles:
            errors.append({
                "index": index,
                "symbol": symbol or None,
                "code": "CUSTOM_MONOMER_DEFINITION_INCOMPLETE",
            })
            continue
        r3 = definition.get("r3_port") or {}
        candidates.append({
            "pdb_resname": symbol,
            "smiles": smiles,
            "r3_mapped_smiles": definition.get("r3_mapped_smiles"),
            "r3_port": dict(r3) if r3 else None,
            "source_cxsmiles": definition.get("source_cxsmiles"),
            "source_r_groups": definition.get("source_r_groups"),
            "source": "local_structure_derived",
            "resolution_mode": str(
                definition.get("resolution_mode")
                or "explicit_custom_definition"
            ),
            "source_entity_id": str(
                definition.get("source_entity_id")
                or f"CUSTOM:{symbol}"
            ),
        })
    return candidates, errors


def _compile_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    from .derived_monomers import build_derived_batch

    if not candidates:
        return [], [], [], []
    forced_aliases = [
        {
            "pdb_resname": str(candidate["pdb_resname"]).upper(),
            "target_symbol": str(candidate["force_alias_target"]),
            "graph_sha256": str(candidate.get("graph_sha256", "")),
            "full_inchikey": str(candidate.get("full_inchikey", "")),
            "input_sha256": "",
            "source_entity_id": str(
                candidate.get("source_entity_id", "")
            ),
            "resolution_mode": str(
                candidate.get("resolution_mode")
                or "source_exact_alias"
            ),
        }
        for candidate in candidates
        if candidate.get("force_alias_target")
    ]
    build_candidates = [
        candidate
        for candidate in candidates
        if not candidate.get("force_alias_target")
    ]
    if not build_candidates:
        return [], [], forced_aliases, []
    try:
        rows, manifests, aliases = build_derived_batch(
            build_candidates,
            starting_id=EPHEMERAL_MONOMER_ID_START,
        )
        return rows, manifests, [*forced_aliases, *aliases], []
    except Exception as batch_error:
        rows = []
        manifests = []
        aliases = []
        errors = [{
            "code": "MONOMER_EXTENSION_BATCH_FAILED",
            "error": f"{type(batch_error).__name__}: {batch_error}",
        }]
        for index, candidate in enumerate(build_candidates):
            try:
                built, entries, resolved_aliases = build_derived_batch(
                    [candidate],
                    starting_id=EPHEMERAL_MONOMER_ID_START + index,
                )
                rows.extend(built)
                manifests.extend(entries)
                aliases.extend(resolved_aliases)
            except Exception as exc:
                errors.append({
                    "symbol": candidate.get("pdb_resname"),
                    "code": "MONOMER_EXTENSION_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return rows, manifests, [*forced_aliases, *aliases], errors


def _aliases_for_new_rows(
    manifests: Sequence[Mapping[str, Any]],
) -> list[dict]:
    return [
        {
            "pdb_resname": str(entry["pdb_resname"]).upper(),
            "target_symbol": str(entry["symbol"]),
            "graph_sha256": str(entry["graph_sha256"]),
            "full_inchikey": str(entry["full_inchikey"]),
            "input_sha256": str(entry.get("input_sha256", "")),
            "source_entity_id": str(
                entry.get("source_entity_id", "")
            ),
            "resolution_mode": str(
                entry.get("resolution_mode")
                or "entity_local_extension"
            ),
        }
        for entry in manifests
    ]


def _resolved_ledger(
    required_symbols: Sequence[str],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    from ..exact_v1 import MonomerRegistry, reset_exact_v1_registry_cache

    reset_exact_v1_registry_cache()
    registry = MonomerRegistry()
    resolved = {}
    unresolved = []
    for symbol in required_symbols:
        try:
            try:
                definition = registry.resolve(symbol)
            except Exception:
                from ..paths.residue_template_factory import resolve_symbol

                definition = registry.resolve(resolve_symbol(symbol))
            resolved[str(symbol)] = {
                "stable_id": definition.stable_id,
                "resolved_symbol": definition.symbol,
                "stereo": definition.stereo,
                "ports": dict(definition.ports),
                "source": definition.source,
            }
        except Exception as exc:
            unresolved.append({
                "symbol": str(symbol),
                "code": "MONOMER_UNRESOLVED",
                "error": f"{type(exc).__name__}: {exc}",
            })
    conflicts = [
        row for row in base.get("errors", [])
        if (
            str(row.get("code", "")).endswith("IDENTITY_CONFLICT")
            or str(row.get("code", "")).endswith("SOURCE_CONFLICT")
        )
    ]
    resolution_errors = [
        row for row in base.get("errors", [])
        if str(row.get("code", "")) != "MONOMER_EXTENSION_BATCH_FAILED"
    ]
    if unresolved:
        chemical_rigor = "C1:H"
    elif conflicts or resolution_errors:
        chemical_rigor = "C2:H"
    elif base.get("component_definition_count"):
        chemical_rigor = "C3:Q"
    elif base.get("custom_definition_count"):
        chemical_rigor = "C3:S"
    elif base.get("derived_rows"):
        chemical_rigor = "C2:R"
    else:
        chemical_rigor = "C3:S"
    return {
        **dict(base),
        "status": (
            "partial"
            if unresolved or conflicts or resolution_errors
            else "resolved"
        ),
        "resolved": resolved,
        "unresolved": unresolved,
        "conflicts": conflicts,
        "resolution_errors": resolution_errors,
        "chemical_rigor": chemical_rigor,
        "registry_scope": "entity_local",
        "persistent_writes": 0,
    }


@contextmanager
def _monomer_resolution_context_impl(
    context: Mapping[str, Any] | None = None,
    *,
    required_symbols: Sequence[str] = (),
):
    """Activate custom/CCD monomers for one operation and return a ledger.

    The context never persists derived rows.  If compilation or activation
    fails, the caller still receives a partial ledger and can emit a lower-
    rigor artifact instead of rejecting the whole operation.
    """
    from ..paths._map_utils import (
        _REGISTRY_LOCK,
        isolated_monomer_registry,
    )

    specification = dict(context or {})
    requested_symbols = tuple(dict.fromkeys(
        [
            *(str(value) for value in required_symbols),
            *(
                str(value)
                for value in (
                    specification.get("required_symbols") or ()
                )
            ),
        ]
    ))
    explicit, explicit_errors = _explicit_candidates(
        specification.get("definitions") or ()
    )
    components, component_errors = _component_candidates(
        specification, requested_symbols
    )
    candidates = [*explicit, *components]
    rows, manifests, aliases, compile_errors = _compile_candidates(
        candidates
    )
    aliases = [
        *aliases,
        *_aliases_for_new_rows(manifests),
        *list(specification.get("pdb_aliases") or ()),
    ]
    rows = [
        *rows,
        *list(specification.get("derived_rows") or ()),
    ]
    base = {
        "schema_version": "1.0.0-monomer-resolution-context.1",
        "custom_definition_count": len(explicit),
        "component_definition_count": len(components),
        "derived_rows": [
            str(row.get("symbol", "")) for row in rows
        ],
        "component_manifests": [
            _json_ready(entry) for entry in manifests
        ],
        "pdb_aliases": [_json_ready(alias) for alias in aliases],
        "errors": [
            *explicit_errors,
            *component_errors,
            *compile_errors,
        ],
    }
    if context is None and not rows and not aliases:
        with _REGISTRY_LOCK:
            ledger = {
                **base,
                "status": "resolved",
                "resolved": {},
                "unresolved": [],
                "conflicts": [],
                "resolution_errors": [],
                "chemical_rigor": "C3:S",
                "registry_scope": "active_default",
                "persistent_writes": 0,
            }
            token = _ACTIVE_LEDGER.set(ledger)
            try:
                yield ledger
            finally:
                _ACTIVE_LEDGER.reset(token)
        return
    manager = isolated_monomer_registry(
        derived_rows=rows,
        pdb_aliases=aliases,
        include_persistent_user=bool(
            specification.get("include_persistent_user", False)
        ),
    )
    try:
        registry_audit = manager.__enter__()
    except Exception as exc:
        ledger = _resolved_ledger(requested_symbols, {
            **base,
            "errors": [
                *base["errors"],
                {
                    "code": "MONOMER_CONTEXT_ACTIVATION_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            ],
        })
        token = _ACTIVE_LEDGER.set(ledger)
        try:
            yield ledger
        finally:
            _ACTIVE_LEDGER.reset(token)
        return
    ledger = _resolved_ledger(requested_symbols, {
        **base,
        "registry_isolation": registry_audit,
    })
    token = _ACTIVE_LEDGER.set(ledger)
    try:
        yield ledger
    except BaseException:
        manager.__exit__(*sys.exc_info())
        raise
    else:
        manager.__exit__(None, None, None)
    finally:
        _ACTIVE_LEDGER.reset(token)


@contextmanager
def monomer_resolution_context(
    context: Mapping[str, Any] | None = None,
    *,
    required_symbols: Sequence[str] = (),
):
    """Serialize one complete request-scoped registry projection.

    Compilation is covered by the same re-entrant lock as activation and use.
    This prevents a second thread from compiling against another entity's
    temporary rows.  Process isolation remains the recommended formal batch
    execution model.
    """
    from ..paths._map_utils import _REGISTRY_LOCK

    with _REGISTRY_LOCK:
        with _monomer_resolution_context_impl(
            context,
            required_symbols=required_symbols,
        ) as ledger:
            yield ledger


__all__ = [
    "CCD_DOWNLOAD_URL",
    "EPHEMERAL_MONOMER_ID_START",
    "active_monomer_resolution",
    "monomer_symbol_hints",
    "monomer_resolution_context",
    "needs_monomer_resolution_scope",
]
