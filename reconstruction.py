"""Unified structure-reconstruction orchestration layer.

``reconstruct_structure`` dispatches PDB/mmCIF coordinates, plain linear
sequences, and HELM/MAP/BILN notation text through the existing CycPep Master
capabilities.  This module contains no reconstruction algorithm of its own: it
reuses the strict V6 pipeline, the result-first recovery ladder, the strict
notation parsers + deterministic monomer-registry assembly, and the existing
multi-chain assembly route.

Dispatch contract
-----------------
- ``strict``: coordinates only call strict V6; representations only call the
  existing strict parsers and deterministic assembly (no candidate promotion,
  no RDKit fallback).
- ``auto``: deterministic dispatch.  Coordinates run the result-first ladder
  first; if that yields only a low-grade topology/partial/raw result and no
  ``fallback_block`` was recorded, the existing registry (HELM/MAP) assembly
  route may replace it only after an independent heavy-atom/topology match.
- ``best_effort``: engaged only when the ``auto`` branch fails or when chain /
  text classification is ambiguous; it tries other compatible existing
  branches but never bypasses a result-first ``fallback_block`` and never
  re-interprets clearly coordinate input as text.

Source detection never claims a plain linear one-letter sequence is MAP, and
uses the native mmCIF adapter for chain selection and preserves native
``label_asym_id``/``auth_asym_id`` and ``struct_conn`` evidence when the
legacy PDB projection cannot assemble a result.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rdkit import Chem
from rdkit.Chem import inchi as rdkit_inchi

from .core.structure_io import CoordinateInputError, coordinate_format
from .core.monomer_resolution import needs_monomer_resolution_scope

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

QUALITY_EXACT = "exact"
QUALITY_HIGH = "high"
QUALITY_MEDIUM = "medium"
QUALITY_TOPOLOGY = "topology"
QUALITY_PARTIAL = "partial"
QUALITY_RAW = "raw"

ORIGIN_STRICT_V6 = "strict_v6"
ORIGIN_RESULT_FIRST = "result_first"
ORIGIN_REPRESENTATION_ASSEMBLY = "representation_assembly"
ORIGIN_COORDINATE_ASSEMBLY = "coordinate_assembly"
ORIGIN_MULTICHAIN_ASSEMBLY = "multichain_assembly"
ORIGIN_NATIVE_MMCIF_GRAPH = "native_mmcif_graph"
ORIGIN_BEST_EFFORT = "best_effort_fallback"
ORIGIN_FAILED = "failed"

MODES = ("strict", "auto", "best_effort")
SOURCE_KINDS = (
    "coordinate", "pdb", "mmcif", "sequence", "helm", "map", "biln",
    "unknown", "invalid",
)
_COORDINATE_KINDS = frozenset({"coordinate", "pdb", "mmcif"})
_NOTATION_ORDER = ("map", "helm", "biln", "sequence")

_SUPPORTED_COORDINATE_SUFFIXES = (
    ".pdb", ".ent", ".pdb.gz", ".ent.gz",
    ".cif", ".mmcif", ".cif.gz", ".mmcif.gz",
)

# IUPAC one-letter amino-acid codes (canonical + ambiguous/rare).
_ONE_LETTER_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYBJOUXZ")

_MIN_MACROCYCLE_RING_ATOMS = 8

_EXPLICIT_PREFIX_RE = re.compile(
    r"^(coordinate|pdb|mmcif|sequence|helm|map|biln):(.*)$",
    re.IGNORECASE | re.DOTALL,
)

_WARNING_INVALID_MODE = "UNIFIED_INVALID_MODE"
_WARNING_SOURCE_UNDETERMINED = "UNIFIED_SOURCE_UNDETERMINED"
_WARNING_AMBIGUOUS_SOURCE = "UNIFIED_AMBIGUOUS_SOURCE"
_WARNING_CHAIN_AMBIGUOUS = "UNIFIED_CHAIN_AMBIGUOUS"
_WARNING_REQUIRES_EXPLICIT_CHAINS = "UNIFIED_REQUIRES_EXPLICIT_CHAINS"
_WARNING_NATIVE_MMCIF_READ_FAILED = "UNIFIED_NATIVE_MMCIF_READ_FAILED"
_WARNING_NATIVE_MMCIF_GRAPH_FALLBACK = (
    "UNIFIED_NATIVE_MMCIF_GRAPH_FALLBACK"
)
_WARNING_NATIVE_MMCIF_DISCONNECTED = "UNIFIED_NATIVE_MMCIF_DISCONNECTED"
_WARNING_COMPRESSED_CHAIN_AUTOSELECT = "UNIFIED_COMPRESSED_CHAIN_AUTOSELECT_UNAVAILABLE"
_WARNING_STRICT_MULTICHAIN_UNSUPPORTED = "UNIFIED_STRICT_MULTICHAIN_UNSUPPORTED"
_WARNING_NO_PEPTIDE_CHAIN = "UNIFIED_NO_PEPTIDE_CHAIN"
_WARNING_EMPTY_CHAIN_LIST = "UNIFIED_EMPTY_CHAIN_LIST"
_WARNING_EMPTY_CHAIN_IDENTIFIER = "UNIFIED_EMPTY_CHAIN_IDENTIFIER"
_WARNING_CHAIN_ENUMERATION_FAILED = "UNIFIED_CHAIN_ENUMERATION_FAILED"
_WARNING_REQUESTED_CHAIN_NOT_AVAILABLE = (
    "UNIFIED_REQUESTED_CHAIN_NOT_AVAILABLE"
)
_WARNING_DUPLICATE_CHAIN_SELECTION = "UNIFIED_DUPLICATE_CHAIN_SELECTION"
_WARNING_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED = (
    "UNIFIED_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED"
)
_WARNING_INCHI_UNAVAILABLE = "UNIFIED_INCHI_UNAVAILABLE"
_WARNING_BEST_EFFORT_FALLBACK = "UNIFIED_BEST_EFFORT_FALLBACK"
_WARNING_ASSEMBLY_FAILED = "UNIFIED_REPRESENTATION_ASSEMBLY_FAILED"
_WARNING_INTERNAL_ERROR = "UNIFIED_INTERNAL_ERROR"
_WARNING_PREFIX_KIND_MISMATCH = "UNIFIED_PREFIX_KIND_MISMATCH"
_WARNING_COORDINATE_INPUT_ERROR = "UNIFIED_COORDINATE_INPUT_ERROR"
_WARNING_STRICT_NOT_QUALIFIED = "UNIFIED_STRICT_NOT_QUALIFIED"
_WARNING_REGISTRY_ASSEMBLY_NOT_QUALIFIED = (
    "UNIFIED_REGISTRY_ASSEMBLY_NOT_QUALIFIED"
)

_WARNING_TEXT = {
    _WARNING_INVALID_MODE: "invalid mode; expected strict, auto, or best_effort",
    _WARNING_SOURCE_UNDETERMINED: (
        "input could not be deterministically classified as a coordinate path "
        "or a supported notation"
    ),
    _WARNING_AMBIGUOUS_SOURCE: (
        "input text was ambiguous; a compatible notation was assembled in "
        "best_effort mode"
    ),
    _WARNING_CHAIN_AMBIGUOUS: (
        "multiple peptide chains present without a single connected component; "
        "a deterministic primary chain was used in best_effort mode"
    ),
    _WARNING_REQUIRES_EXPLICIT_CHAINS: (
        "multiple peptide chains are present but are not joined by explicit "
        "inter-chain SSBOND/LINK/CONECT records into one connected component; "
        "pass an explicit chain_id or chain list"
    ),
    _WARNING_NATIVE_MMCIF_READ_FAILED: (
        "native mmCIF parsing failed before chain selection or graph recovery"
    ),
    _WARNING_NATIVE_MMCIF_GRAPH_FALLBACK: (
        "the native mmCIF graph was returned after legacy assembly did not "
        "produce a usable structure"
    ),
    _WARNING_NATIVE_MMCIF_DISCONNECTED: (
        "selected peptide chains contain multiple explicit connectivity "
        "components; no inter-component bond was inferred"
    ),
    _WARNING_COMPRESSED_CHAIN_AUTOSELECT: (
        "chain auto-selection and multi-chain assembly for compressed "
        "coordinates are not supported; pass an explicit chain_id"
    ),
    _WARNING_STRICT_MULTICHAIN_UNSUPPORTED: (
        "strict mode only supports single-chain strict V6; pass one chain_id"
    ),
    _WARNING_NO_PEPTIDE_CHAIN: (
        "no peptide-bearing chain was found in the coordinate input"
    ),
    _WARNING_EMPTY_CHAIN_LIST: "the chain list is empty",
    _WARNING_EMPTY_CHAIN_IDENTIFIER: (
        "the explicit chain list contains an empty chain identifier"
    ),
    _WARNING_CHAIN_ENUMERATION_FAILED: (
        "peptide chain enumeration failed for the coordinate input"
    ),
    _WARNING_REQUESTED_CHAIN_NOT_AVAILABLE: (
        "one or more explicitly requested peptide chains are unavailable"
    ),
    _WARNING_DUPLICATE_CHAIN_SELECTION: (
        "the explicit chain list contains duplicate chain identifiers"
    ),
    _WARNING_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED: (
        "coordinate assembly result was not qualified by strict V6"
    ),
    _WARNING_INCHI_UNAVAILABLE: (
        "InChI/InChIKey could not be generated from the output SMILES; no "
        "fabricated identifier is reported"
    ),
    _WARNING_BEST_EFFORT_FALLBACK: (
        "a best_effort-compatible existing branch produced this result"
    ),
    _WARNING_ASSEMBLY_FAILED: (
        "representation could not be assembled by the strict parser and "
        "deterministic assembly route"
    ),
    _WARNING_INTERNAL_ERROR: (
        "unexpected orchestration failure; no readable structure produced"
    ),
    _WARNING_PREFIX_KIND_MISMATCH: (
        "explicit source prefix does not match the payload"
    ),
    _WARNING_COORDINATE_INPUT_ERROR: "coordinate input could not be processed",
    _WARNING_STRICT_NOT_QUALIFIED: (
        "strict V6 produced a structure without qualified success; "
        "repaired/unqualified output is not an exact reconstruction"
    ),
    _WARNING_REGISTRY_ASSEMBLY_NOT_QUALIFIED: (
        "registry assembly was not independently qualified against the "
        "coordinate fallback graph and was not selected"
    ),
}

_SS_SMARTS = Chem.MolFromSmarts("[#16X2]-[#16X2]")
_THIOETHER_SMARTS = Chem.MolFromSmarts("[#6X4]-[#16X2]-[#6X4]")
_ESTER_SMARTS = Chem.MolFromSmarts("[#6X3](=[OX1])-[#8X2]-[#6X3]")


@dataclass
class UnifiedReconstructionResult:
    """Typed result of the unified structure-reconstruction orchestration.

    ``status`` is ``success`` whenever a readable structure exists, otherwise
    ``failed``.  ``strict_result`` carries the unmodified strict V6 object when
    a coordinate branch produced one; orchestration never mutates it.
    """

    status: str
    quality: str | None
    result_origin: str | None
    source_kind: str | None
    mode: str
    result: Any
    smiles: str | None
    inchi: str | None
    inchikey: str | None
    graph: dict[str, Any] | None
    ambiguous: bool
    warnings: list[str]
    warning_codes: list[str]
    alternatives: list[dict[str, Any]]
    structure_profile: dict[str, Any] | None
    provenance: dict[str, Any]
    strict_result: Any = None
    strict_status: str | None = None
    candidate_smiles: str | None = None
    candidate_graph: dict[str, Any] | None = None
    chemistry_candidates: list[dict[str, Any]] = field(default_factory=list)
    bond_order_inference: dict[str, Any] = field(default_factory=dict)
    candidate_rigor: str | None = None
    artifact_status: str = "opaque_input"
    qualification_status: str = "not_assessable"
    chemical_rigor: str = "C0:NONE"
    coordinate_evidence: str = "X0"


def _warning_lines(codes: list[str]) -> list[str]:
    return [
        f"{code}: {_WARNING_TEXT.get(code, code)}" for code in codes
    ]


def _normalize_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        return ""
    return mode.strip().lower().replace("-", "_")


def _chain_id_summary(chain_id: Any) -> Any:
    if isinstance(chain_id, (list, tuple)):
        return [str(value) for value in chain_id]
    if chain_id is None:
        return None
    return str(chain_id)


def _compute_inchi(smiles: str | None) -> tuple[str | None, str | None, str | None]:
    """InChI/InChIKey from a cleanable SMILES; never fabricates on failure."""
    if not smiles:
        return None, None, None
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None, None, "output SMILES is not parseable by RDKit"
        value = rdkit_inchi.MolToInchi(molecule)
        key = rdkit_inchi.MolToInchiKey(molecule)
        if not value or not key:
            return None, None, "InChI generation returned an empty value"
        return str(value), str(key), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _build(
    *,
    status: str,
    quality: str | None,
    result_origin: str | None,
    source_kind: str | None,
    mode: str,
    result: Any,
    smiles: str | None,
    graph: dict[str, Any] | None,
    ambiguous: bool,
    warning_codes: list[str],
    alternatives: list[dict[str, Any]],
    structure_profile: dict[str, Any] | None,
    provenance: dict[str, Any],
    strict_result: Any = None,
    strict_status: str | None = None,
    candidate_smiles: str | None = None,
    candidate_graph: dict[str, Any] | None = None,
    chemistry_candidates: list[dict[str, Any]] | None = None,
    bond_order_inference: dict[str, Any] | None = None,
    candidate_rigor: str | None = None,
    artifact_status: str = "opaque_input",
    qualification_status: str = "not_assessable",
    chemical_rigor: str = "C0:NONE",
    coordinate_evidence: str = "X0",
    inchi_error: str | None = None,
) -> UnifiedReconstructionResult:
    codes = list(dict.fromkeys(warning_codes))
    inchi, inchikey, computed_error = _compute_inchi(smiles)
    if smiles and (inchi_error or computed_error):
        codes.append(_WARNING_INCHI_UNAVAILABLE)
    provenance = dict(provenance)
    provenance["inchi_error"] = inchi_error or computed_error
    return UnifiedReconstructionResult(
        status=status,
        quality=quality,
        result_origin=result_origin,
        source_kind=source_kind,
        mode=mode,
        result=result,
        smiles=smiles,
        inchi=inchi,
        inchikey=inchikey,
        graph=graph,
        ambiguous=ambiguous,
        warnings=_warning_lines(codes),
        warning_codes=codes,
        alternatives=list(alternatives),
        structure_profile=structure_profile,
        provenance=provenance,
        strict_result=strict_result,
        strict_status=(
            strict_status
            if strict_status is not None
            else getattr(strict_result, "status", None)
        ),
        candidate_smiles=candidate_smiles,
        candidate_graph=candidate_graph,
        chemistry_candidates=list(chemistry_candidates or []),
        bond_order_inference=dict(bond_order_inference or {}),
        candidate_rigor=candidate_rigor,
        artifact_status=artifact_status,
        qualification_status=qualification_status,
        chemical_rigor=chemical_rigor,
        coordinate_evidence=coordinate_evidence,
    )


def _failed_provenance(
    *,
    source_kind: str | None,
    mode: str,
    chain_id: Any,
    detection: dict[str, Any] | None,
    branch: str,
    failure_reason: str,
    underlying: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "detection": detection or {"source_kind": source_kind or "invalid"},
        "dispatch": {
            "mode": mode,
            "branch": branch,
            "chain_id": _chain_id_summary(chain_id),
        },
        "underlying": underlying or {},
        "failure_reason": failure_reason,
        "error": error,
    }


def _typed_failed(
    *,
    source: Any,
    chain_id: Any,
    mode: str,
    source_kind: str | None,
    detection: dict[str, Any] | None,
    branch: str,
    failure_reason: str,
    warning_codes: list[str],
    alternatives: list[dict[str, Any]] | None = None,
    underlying: dict[str, Any] | None = None,
    error: str | None = None,
    strict_result: Any = None,
) -> UnifiedReconstructionResult:
    provenance = _failed_provenance(
        source_kind=source_kind,
        mode=mode,
        chain_id=chain_id,
        detection=detection,
        branch=branch,
        failure_reason=failure_reason,
        underlying=underlying,
        error=error,
    )
    provenance["source"] = _source_preview(source)
    return _build(
        status=STATUS_FAILED,
        quality=None,
        result_origin=ORIGIN_FAILED,
        source_kind=source_kind,
        mode=mode,
        result=None,
        smiles=None,
        graph=None,
        ambiguous=False,
        warning_codes=warning_codes,
        alternatives=alternatives or [],
        structure_profile=None,
        provenance=provenance,
        strict_result=strict_result,
    )


def _source_preview(source: Any) -> str:
    text = str(source)
    if len(text) > 80:
        return text[:77] + "..."
    return text


# --------------------------------------------------------------------------
# Source classification
# --------------------------------------------------------------------------


def detect_source_kind(source: Any) -> str | None:
    """Return the deterministically detected source kind for ``source``.

    One of ``coordinate|pdb|mmcif|sequence|helm|map|biln``, or ``None`` when
    the input cannot be classified (invalid or undetermined text).
    """
    kind, _payload, error = _classify_source(source)
    if error:
        return None
    return kind


def _classify_source(source: Any) -> tuple[str, str, str | None]:
    """Return ``(kind, payload, error)``; error is None for usable inputs."""
    if isinstance(source, Path):
        return _classify_coordinate_path(str(source))
    if not isinstance(source, str):
        return "invalid", "", "input must be a path or a text string"
    text = source.strip()
    if not text:
        return "invalid", "", "input is empty"
    match = _EXPLICIT_PREFIX_RE.match(text)
    if match:
        prefix = match.group(1).lower()
        payload = match.group(2).strip()
        if not payload:
            return "invalid", "", f"{prefix}: prefix requires a payload"
        if prefix in _COORDINATE_KINDS:
            kind, path_text, error = _classify_coordinate_path(
                payload,
                require_format=None if prefix == "coordinate" else prefix,
            )
            if kind is None:
                return "invalid", payload, "coordinate file was not found"
            return (
                prefix if prefix in {"pdb", "mmcif"} else kind,
                path_text,
                error,
            )
        if prefix == "sequence" and not _is_symbolic_sequence(payload):
            return (
                "sequence",
                payload,
                "sequence: prefix requires a valid linear sequence token stream",
            )
        return prefix, payload, None
    classified = _classify_coordinate_path(text)
    if classified[0] == "coordinate":
        return classified
    if _is_helm(text):
        return "helm", text, None
    if _has_map_markers(text):
        return "map", text, None
    if _has_biln_markers(text):
        return "biln", text, None
    if _is_symbolic_sequence(text):
        return "sequence", text, None
    return "unknown", text, None


def _classify_coordinate_path(
    text: str, *, require_format: str | None = None
) -> tuple[str, str, str | None]:
    path = Path(text)
    suffix_supported = text.lower().endswith(_SUPPORTED_COORDINATE_SUFFIXES)
    if not (path.is_file() or suffix_supported):
        return None, text, None
    try:
        fmt = coordinate_format(path)
    except CoordinateInputError as exc:
        return "coordinate", text, f"{exc.code}: {exc}"
    if require_format and fmt != require_format:
        return (
            "coordinate",
            text,
            f"{_WARNING_PREFIX_KIND_MISMATCH}: prefix {require_format} "
            f"requires a {require_format} file, got {fmt}",
        )
    return (require_format or "coordinate"), text, None


def _is_pure_sequence(text: str) -> bool:
    return bool(text) and all(char in _ONE_LETTER_CODES for char in text)


def _is_symbolic_sequence(text: str) -> bool:
    if _is_pure_sequence(text):
        return True
    try:
        from .sequence import _sequence_tokens

        _sequence_tokens(text)
        return True
    except Exception:
        return False


def _is_helm(text: str) -> bool:
    return "PEPTIDE" in text.upper() and text.count("$") >= 4


def _has_map_markers(text: str) -> bool:
    return any(
        marker in text
        for marker in ("{nnr:", "{cyc:", "{br}", "{nt:", "{ct:")
    )


def _has_biln_markers(text: str) -> bool:
    return "-" in text or re.search(r"\(\d+\s*,\s*\d+\)", text) is not None


# --------------------------------------------------------------------------
# Structure profiles
# --------------------------------------------------------------------------


def _display_label(
    *,
    layout: str,
    residue_count: int | None,
    macrocycle_count: int,
    chain_count: int,
    branch_point_count: int,
) -> str:
    residue_part = f"{residue_count}-residue " if residue_count else ""
    chain_part = f", {chain_count} chain(s)" if chain_count > 1 else ""
    return (
        f"{layout} {residue_part}peptide, {macrocycle_count} macrocycle(s)"
        f"{chain_part}, {branch_point_count} branch point(s)"
    )


def _base_profile(
    *,
    chain_count: int = 1,
    is_multichain: bool = False,
    layout: str = "unknown",
    macrocycle_count: int = 0,
    macrocycle_count_metric: str = "unknown",
    crosslink_types: list[str] | None = None,
    branch_point_count: int = 0,
    evidence: str = "none",
    residue_count: int | None = None,
) -> dict[str, Any]:
    macrocycle_count = int(max(0, macrocycle_count))
    return {
        "chain_count": int(chain_count),
        "backbone_layout": layout,
        "macrocycle_count": macrocycle_count,
        "macrocycle_count_metric": macrocycle_count_metric,
        "crosslink_types": list(crosslink_types or []),
        "branch_point_count": int(branch_point_count),
        "is_multichain": bool(is_multichain),
        "display_label": _display_label(
            layout=layout,
            residue_count=residue_count,
            macrocycle_count=macrocycle_count,
            chain_count=int(chain_count),
            branch_point_count=int(branch_point_count),
        ),
        "profile_evidence": evidence,
    }


def _residue_graph(
    chains: list[list[str]],
    edges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> dict[str, Any]:
    return {
        "chains": [
            {"index": index + 1, "length": len(chain), "symbols": list(chain)}
            for index, chain in enumerate(chains)
        ],
        "edges": [
            {
                "from": list(left),
                "to": list(right),
                "rgroup_pair": _port_label(left[2], right[2]),
            }
            for left, right in edges
        ],
        "residue_count": sum(len(chain) for chain in chains),
        "graph_evidence": "strict_notation_parse",
    }


def _port_label(left_rgroup: int, right_rgroup: int) -> str:
    return f"R{min(left_rgroup, right_rgroup)}-R{max(left_rgroup, right_rgroup)}"


def _parse_map_graph(
    kind: str, text: str
) -> tuple[
    list[list[str]],
    list[tuple[tuple[int, int, int], tuple[int, int, int]]],
]:
    """Strict-parse one notation into (chains, edges) via existing converters."""
    from .paths._map_utils import _parse_map_strict, biln_to_helm, helm_to_map

    if kind in ("map", "sequence"):
        mapped = text
    elif kind == "helm":
        mapped = helm_to_map(text)
    elif kind == "biln":
        mapped = helm_to_map(biln_to_helm(text))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported representation kind: {kind}")
    return _parse_map_strict(mapped)


def _profile_from_residue_graph(
    chains: list[list[str]],
    edges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    *,
    is_multichain: bool = False,
) -> dict[str, Any]:
    residue_count = sum(len(chain) for chain in chains)
    chain_count = len(chains)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    # Materialize every residue before adding edges. A singleton chain (or an
    # otherwise isolated residue) is still a vertex in the residue graph and
    # must contribute to both V and the connected-component count. Omitting it
    # makes a cyclic component plus an isolated chain appear acyclic.
    for chain_index, chain in enumerate(chains):
        for position in range(1, len(chain) + 1):
            adjacency[(chain_index, position)]

    def add_edge(left: tuple[int, int], right: tuple[int, int]) -> None:
        adjacency[left].add(right)
        adjacency[right].add(left)

    for chain_index, chain in enumerate(chains):
        for position in range(1, len(chain)):
            add_edge((chain_index, position), (chain_index, position + 1))
    for left, right in edges:
        add_edge((left[0], left[1]), (right[0], right[1]))

    head_to_tail = any(
        left[0] == right[0]
        and {
            (left[1], left[2]),
            (right[1], right[2]),
        }
        == {
            (1, 1),
            (len(chains[left[0]]), 2),
        }
        for left, right in edges
    )
    backbone_edges = residue_count - chain_count
    edge_count = backbone_edges + len(edges)

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            parent[node], node = root, parent[node]
        return root

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for left, partners in adjacency.items():
        for partner in partners:
            union(left, partner)
    components = len({find(node) for node in adjacency}) if adjacency else 0
    cycle_rank = max(0, edge_count - residue_count + components)

    degree: Counter[tuple[int, int]] = Counter()
    for node, partners in adjacency.items():
        degree[node] = len(partners)
    branch_point_count = sum(1 for value in degree.values() if value > 2)
    crosslink_types = sorted(
        {_port_label(left[2], right[2]) for left, right in edges}
    )
    return _base_profile(
        chain_count=chain_count,
        is_multichain=is_multichain or chain_count > 1,
        layout="cyclic" if head_to_tail else "linear",
        macrocycle_count=cycle_rank,
        macrocycle_count_metric="cycle_rank",
        crosslink_types=crosslink_types,
        branch_point_count=branch_point_count,
        evidence="residue_connection_graph",
        residue_count=residue_count,
    )


def _profile_from_strict_evidence(
    strict: Any,
    *,
    chain_count: int = 1,
) -> dict[str, Any]:
    evidence = getattr(strict, "input_evidence", None) or {}
    topology = str(evidence.get("topology") or "")
    bonds = list(evidence.get("cyclization_bonds") or [])
    residue_count = evidence.get("residue_count")
    layout = "cyclic" if topology and topology != "linear" else "linear"
    macrocycle_map = {
        "linear": 0, "monocyclic": 1, "bicyclic": 2, "tricyclic": 3,
    }
    if topology in macrocycle_map:
        macrocycle_count = macrocycle_map[topology]
        macrocycle_count_metric = "strict_topology_macrocycle_count"
    else:
        macrocycle_count = len(bonds)
        macrocycle_count_metric = "cyclization_bond_count"
    crosslink_types = sorted(
        {
            str(bond.get("bond_type"))
            for bond in bonds
            if bond.get("bond_type") and str(bond.get("bond_type")) != "peptide"
        }
    )
    position_count: Counter[int] = Counter()
    for bond in bonds:
        position_count[int(bond["position_1"])] += 1
        position_count[int(bond["position_2"])] += 1
    branch_point_count = sum(
        1 for value in position_count.values() if value >= 2
    )
    return _base_profile(
        chain_count=chain_count,
        layout=layout,
        macrocycle_count=macrocycle_count,
        macrocycle_count_metric=macrocycle_count_metric,
        crosslink_types=crosslink_types,
        branch_point_count=branch_point_count,
        evidence="strict_v6_input_evidence",
        residue_count=int(residue_count) if residue_count else None,
    )


def _profile_from_smiles(
    smiles: str,
    *,
    chain_count: int = 1,
    is_multichain: bool = False,
) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return _base_profile(
            chain_count=chain_count,
            is_multichain=is_multichain,
            layout="unknown",
            evidence="smiles_unavailable",
        )
    ring_info = molecule.GetRingInfo()
    macrocycle_count = sum(
        1
        for ring in ring_info.AtomRings()
        if len(ring) >= _MIN_MACROCYCLE_RING_ATOMS
    )
    crosslink_types: list[str] = []
    if molecule.HasSubstructMatch(_SS_SMARTS):
        crosslink_types.append("disulfide")
    if molecule.HasSubstructMatch(_THIOETHER_SMARTS):
        crosslink_types.append("thioether")
    if molecule.HasSubstructMatch(_ESTER_SMARTS):
        crosslink_types.append("ester")
    branch_point_count = sum(
        1
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetDegree() >= 4
    )
    return _base_profile(
        chain_count=chain_count,
        is_multichain=is_multichain,
        layout="cyclic" if macrocycle_count > 0 else "linear",
        macrocycle_count=macrocycle_count,
        macrocycle_count_metric="large_ring_count",
        crosslink_types=crosslink_types,
        branch_point_count=branch_point_count,
        evidence="smiles_ring_and_smarts_analysis",
    )


def _profile_from_graph(
    graph: dict[str, Any],
    *,
    chain_count: int = 1,
    is_multichain: bool = False,
) -> dict[str, Any]:
    atoms = list(graph.get("atoms") or [])
    bonds = list(graph.get("bonds") or [])
    if not atoms:
        return _base_profile(
            chain_count=chain_count,
            is_multichain=is_multichain,
            layout="unknown",
            evidence="empty_graph",
        )
    residues = {
        (atom.get("chain"), atom.get("residue_number"))
        for atom in atoms
        if atom.get("chain") or atom.get("residue_number")
    }
    serial_degree: Counter[int] = Counter()
    for bond in bonds:
        serial_degree[int(bond["a"])] += 1
        serial_degree[int(bond["b"])] += 1
    parent = {int(atom["serial"]): int(atom["serial"]) for atom in atoms}

    def find(serial: int) -> int:
        root = serial
        while parent[root] != root:
            root = parent[root]
        while parent[serial] != serial:
            parent[serial], serial = root, parent[serial]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for bond in bonds:
        union(int(bond["a"]), int(bond["b"]))
    topology = Chem.RWMol()
    serial_to_index: dict[int, int] = {}
    for atom in atoms:
        serial = int(atom["serial"])
        serial_to_index[serial] = topology.AddAtom(Chem.Atom(6))
    added_edges: set[tuple[int, int]] = set()
    for bond in bonds:
        left, right = sorted((int(bond["a"]), int(bond["b"])))
        edge = (left, right)
        if left == right or edge in added_edges:
            continue
        if left not in serial_to_index or right not in serial_to_index:
            continue
        topology.AddBond(
            serial_to_index[left],
            serial_to_index[right],
            Chem.BondType.SINGLE,
        )
        added_edges.add(edge)
    ring_sizes = [len(ring) for ring in Chem.GetSymmSSSR(topology.GetMol())]
    macrocycle_count = sum(
        size >= _MIN_MACROCYCLE_RING_ATOMS for size in ring_sizes
    )
    branch_point_count = sum(
        1 for value in serial_degree.values() if value >= 4
    )
    return _base_profile(
        chain_count=chain_count,
        is_multichain=is_multichain,
        layout="cyclic" if macrocycle_count > 0 else "linear",
        macrocycle_count=macrocycle_count,
        macrocycle_count_metric="large_ring_sssr_count",
        crosslink_types=[],
        branch_point_count=branch_point_count,
        evidence="atom_graph_large_ring_analysis",
        residue_count=len(residues) if residues else None,
    )


def _profile_for_coordinate(
    smiles: str | None,
    graph: dict[str, Any] | None,
    strict: Any,
    *,
    chain_count: int = 1,
    is_multichain: bool = False,
) -> dict[str, Any] | None:
    if smiles:
        return _profile_from_smiles(
            smiles, chain_count=chain_count, is_multichain=is_multichain
        )
    if graph:
        return _profile_from_graph(
            graph, chain_count=chain_count, is_multichain=is_multichain
        )
    if strict is not None and getattr(strict, "input_evidence", None):
        return _profile_from_strict_evidence(
            strict, chain_count=chain_count
        )
    return None


# --------------------------------------------------------------------------
# Coordinate branches
# --------------------------------------------------------------------------


def _run_result_first(
    path_text: str,
    chain: str,
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
) -> Any:
    from .result_first import reconstruct_structure as result_first_reconstruct

    def registry_assembler(prepared: Any) -> dict[str, Any]:
        return _try_registry_assembly_prepared(prepared)

    return result_first_reconstruct(
        path_text,
        chain_id=chain,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
        registry_assembler=registry_assembler,
    )


def _try_registry_assembly(
    path_text: str, chain: str
) -> tuple[str | None, str | None]:
    """Existing HELM/MAP monomer-registry assembly route (path B).

    Uses the shared ``prepare_coordinate_input`` normalization so PDB, mmCIF,
    and compressed inputs all run the same existing assembly entry point on
    the projected PDB (identical to the pipeline's A-H usage).
    """
    try:
        from .paths import generate_b
        from .core.structure_io import prepare_coordinate_input

        with prepare_coordinate_input(path_text, chain) as prepared:
            payload = _try_registry_assembly_prepared(prepared)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    smiles = payload.get("smiles")
    error = payload.get("error")
    if not smiles:
        return None, error or "registry assembly produced no SMILES"
    return smiles, None


def _try_registry_assembly_prepared(prepared: Any) -> dict[str, Any]:
    """Run Path B on an already normalized input without a second prepare."""
    try:
        from .chemical_audit import audit_output_smiles
        from .paths import generate_b

        smiles, error = generate_b(
            str(prepared.pdb_path), chain_id=str(prepared.chain_id)
        )
        if not smiles:
            return {
                "smiles": None,
                "error": error or "registry assembly produced no SMILES",
                "route": "path_b_registry",
            }
        audit = audit_output_smiles(smiles)
        if not audit.accepted:
            return {
                "smiles": None,
                "error": audit.reason or "registry assembly SMILES rejected by output audit",
                "route": "path_b_registry",
                "output_audit": {"accepted": False, "reason": audit.reason},
            }
        return {
            "smiles": smiles,
            "route": "path_b_registry",
            "output_audit": {"accepted": True},
        }
    except Exception as exc:
        return {
            "smiles": None,
            "error": f"{type(exc).__name__}: {exc}",
            "route": "path_b_registry",
        }


def _element_atomic_number(element: Any) -> int:
    try:
        symbol = str(element or "").strip()
        if not symbol:
            return 0
        return int(Chem.GetPeriodicTable().GetAtomicNumber(symbol.title()))
    except Exception:
        return 0


def _topology_signature(
    elements: dict[int, str],
    edges: list[tuple[int, int]],
) -> dict[str, Any] | None:
    periodic_table = Chem.GetPeriodicTable()
    normalized_elements: dict[int, str] = {}
    for index, element in elements.items():
        atomic_number = _element_atomic_number(element)
        if atomic_number <= 0:
            return None
        normalized_elements[index] = periodic_table.GetElementSymbol(
            atomic_number
        )
    elements = normalized_elements
    heavy = {
        index for index, element in elements.items()
        if _element_atomic_number(element) > 1
    }
    if not heavy:
        return None
    heavy_edges: set[tuple[int, int]] = set()
    for left, right in edges:
        if left == right or left not in elements or right not in elements:
            return None
        if left in heavy and right in heavy:
            heavy_edges.add(tuple(sorted((left, right))))
    molecule = Chem.RWMol()
    indices = {
        index: molecule.AddAtom(Chem.Atom(elements[index]))
        for index in sorted(heavy)
    }
    try:
        for left, right in sorted(heavy_edges):
            molecule.AddBond(indices[left], indices[right], Chem.BondType.SINGLE)
        topology = molecule.GetMol()
        for atom in topology.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
        for bond in topology.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        canonical = Chem.MolToSmiles(
            topology, canonical=True, isomericSmiles=False
        )
    except Exception:
        return None
    return {
        "canonical_topology": canonical,
        "heavy_atom_count": len(heavy),
        "heavy_element_counts": dict(sorted(
            Counter(elements[index] for index in heavy).items()
        )),
        "heavy_edge_count": len(heavy_edges),
    }


def _graph_topology_signature(graph: dict[str, Any]) -> dict[str, Any] | None:
    elements: dict[int, str] = {}
    for row in list(graph.get("atoms") or []):
        if not isinstance(row, dict):
            return None
        try:
            serial = int(row["serial"])
        except (KeyError, TypeError, ValueError):
            return None
        if serial in elements:
            return None
        element = str(row.get("element") or "").strip().upper()
        if _element_atomic_number(element) <= 0:
            return None
        elements[serial] = element
    edges: list[tuple[int, int]] = []
    for row in list(graph.get("bonds") or []):
        if not isinstance(row, dict):
            return None
        try:
            edges.append((int(row["a"]), int(row["b"])))
        except (KeyError, TypeError, ValueError):
            return None
    return _topology_signature(elements, edges)


def _smiles_topology_signature(smiles: str) -> dict[str, Any] | None:
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None:
        return None
    elements = {
        atom.GetIdx(): atom.GetSymbol().strip().upper()
        for atom in molecule.GetAtoms()
    }
    edges = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
    ]
    return _topology_signature(elements, edges)


def _qualify_registry_assembly(
    smiles: str,
    result_first: Any,
) -> dict[str, Any]:
    """Qualify registry SMILES against the selected coordinate graph."""
    graph = getattr(result_first, "graph", None)
    if not isinstance(graph, dict) or not graph.get("atoms"):
        return {
            "qualified": False,
            "status": "not_qualified",
            "reason": "LOW_GRADE_GRAPH_UNAVAILABLE",
        }
    graph_signature = _graph_topology_signature(graph)
    if graph_signature is None:
        return {
            "qualified": False,
            "status": "not_qualified",
            "reason": "LOW_GRADE_GRAPH_INVALID",
        }
    smiles_signature = _smiles_topology_signature(smiles)
    if smiles_signature is None:
        return {
            "qualified": False,
            "status": "not_qualified",
            "reason": "REGISTRY_SMILES_NOT_PARSEABLE",
            "coordinate_signature": graph_signature,
        }
    mismatches = [
        key
        for key in (
            "canonical_topology",
            "heavy_atom_count",
            "heavy_element_counts",
            "heavy_edge_count",
        )
        if graph_signature[key] != smiles_signature[key]
    ]
    return {
        "qualified": not mismatches,
        "status": "qualified" if not mismatches else "not_qualified",
        "reason": None if not mismatches else "GRAPH_SIGNATURE_MISMATCH",
        "mismatches": mismatches,
        "coordinate_signature": graph_signature,
        "registry_signature": smiles_signature,
    }


def _coordinate_assembly_result(
    *,
    source_kind: str,
    mode: str,
    chain: str,
    path_text: str,
    smiles: str,
    route: str,
    strict: Any,
    detection: dict[str, Any],
    alternatives: list[dict[str, Any]],
    extra_codes: list[str],
    result_first: Any = None,
    registry_audit: dict[str, Any] | None = None,
    quality: str = QUALITY_HIGH,
    ambiguous: bool = False,
) -> UnifiedReconstructionResult:
    profile = _profile_from_smiles(smiles, chain_count=1)
    underlying = {
        "route": route,
        "coordinate_input": path_text,
    }
    if result_first is not None:
        underlying["result_first"] = {
            "status": result_first.status,
            "quality": result_first.quality,
            "source": result_first.source,
            "ambiguous": bool(result_first.ambiguous),
            "warnings": list(result_first.warnings),
            "warning_codes": list(result_first.warning_codes),
            "alternatives": list(result_first.alternatives),
            "graph": result_first.graph,
            "provenance": result_first.provenance,
        }
    if registry_audit is not None:
        underlying["registry_assembly"] = registry_audit
    provenance = {
        "detection": detection,
        "dispatch": {
            "mode": mode,
            "branch": route,
            "chain_id": chain,
        },
        "underlying": underlying,
    }
    strict_codes = list(getattr(strict, "warning_codes", None) or [])
    return _build(
        status=STATUS_SUCCESS,
        quality=quality,
        result_origin=ORIGIN_COORDINATE_ASSEMBLY,
        source_kind=source_kind,
        mode=mode,
        result={"route": route, "smiles": smiles, "coordinate_input": path_text},
        smiles=smiles,
        graph=None,
        ambiguous=ambiguous,
        warning_codes=[
            _WARNING_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED,
            *strict_codes,
            *extra_codes,
        ],
        alternatives=alternatives,
        structure_profile=profile,
        provenance=provenance,
        strict_result=strict,
    )


def _from_result_first(
    rf: Any,
    *,
    source_kind: str,
    mode: str,
    chain: str,
    detection: dict[str, Any],
    registry_audit: dict[str, Any] | None = None,
) -> UnifiedReconstructionResult:
    profile = _profile_for_coordinate(
        rf.smiles or rf.candidate_smiles,
        rf.graph or rf.candidate_graph,
        rf.strict_result,
        chain_count=1,
    )
    provenance = {
        "detection": detection,
        "dispatch": {
            "mode": mode,
            "branch": "result_first",
            "chain_id": chain,
        },
        "underlying": rf.provenance,
        "result_first_quality": rf.quality,
    }
    warning_codes = list(rf.warning_codes)
    if registry_audit is not None:
        provenance["registry_assembly"] = registry_audit
        if not registry_audit.get("qualified"):
            warning_codes.append(_WARNING_REGISTRY_ASSEMBLY_NOT_QUALIFIED)
    return _build(
        status=STATUS_SUCCESS,
        quality=rf.quality,
        result_origin=(
            ORIGIN_STRICT_V6
            if rf.quality == QUALITY_EXACT
            else ORIGIN_RESULT_FIRST
        ),
        source_kind=source_kind,
        mode=mode,
        result=rf,
        smiles=rf.smiles,
        graph=rf.graph,
        ambiguous=rf.ambiguous,
        warning_codes=warning_codes,
        alternatives=list(rf.alternatives),
        structure_profile=profile,
        provenance=provenance,
        strict_result=rf.strict_result,
        strict_status=rf.strict_status,
        candidate_smiles=rf.candidate_smiles,
        candidate_graph=rf.candidate_graph,
        chemistry_candidates=list(rf.chemistry_candidates),
        bond_order_inference=dict(rf.bond_order_inference),
        candidate_rigor=rf.candidate_rigor,
        artifact_status=rf.artifact_status,
        qualification_status=rf.qualification_status,
        chemical_rigor=rf.chemical_rigor,
        coordinate_evidence=rf.coordinate_evidence,
    )


def _strict_coordinate(
    path_text: str,
    chain: str,
    source_kind: str,
    mode: str,
    detection: dict[str, Any],
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
) -> UnifiedReconstructionResult:
    from . import remediation_v6

    try:
        strict = remediation_v6.reconstruct_structure_fail_closed_v6(
            path_text,
            chain_id=chain,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            allow_linear_topology=allow_linear_topology,
            **(
                {
                    "allow_chem_comp_evidence": allow_chem_comp_evidence,
                    "chem_comp_evidence": chem_comp_evidence,
                }
                if allow_chem_comp_evidence or chem_comp_evidence is not None
                else {}
            ),
        )
    except Exception as exc:
        return _typed_failed(
            source=path_text,
            chain_id=chain,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="strict_v6",
            failure_reason=f"strict V6 raised: {type(exc).__name__}: {exc}",
            warning_codes=[_WARNING_INTERNAL_ERROR],
            error=f"{type(exc).__name__}: {exc}",
        )
    if (
        strict.status != STATUS_SUCCESS
        or not strict.output_smiles
        or not bool(strict.qualified_success)
    ):
        qualified_output = (
            strict.status == STATUS_SUCCESS and bool(strict.output_smiles)
        )
        if qualified_output:
            failure_reason = (
                "strict V6 produced a structure without qualified success "
                "(repaired/unqualified output cannot be exact)"
            )
            warning_codes = list(
                dict.fromkeys(
                    [_WARNING_STRICT_NOT_QUALIFIED, *strict.warning_codes]
                )
            )
        else:
            failure_reason = (
                strict.rejection_reason
                or f"strict V6 did not produce a structure (status={strict.status})"
            )
            warning_codes = list(strict.warning_codes)
        return _typed_failed(
            source=path_text,
            chain_id=chain,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="strict_v6",
            failure_reason=failure_reason,
            warning_codes=warning_codes,
            underlying={
                "strict_status": strict.status,
                "support_status": strict.support_status,
                "path_used": strict.path_used,
                "qualified_success": bool(strict.qualified_success),
                "repair_codes": list(strict.repair_codes),
            },
            strict_result=strict,
        )
    profile = _profile_for_coordinate(
        strict.output_smiles, None, strict, chain_count=1
    )
    provenance = {
        "detection": detection,
        "dispatch": {"mode": mode, "branch": "strict_v6", "chain_id": chain},
        "underlying": {
            "strict_status": strict.status,
            "path_used": strict.path_used,
            "qualified_success": bool(strict.qualified_success),
        },
    }
    if str(strict.input_evidence.get("topology_class")) == "linear":
        # Linear topology is a top-level provenance declaration.  It is
        # deliberately kept out of the frozen path-a/e generation-provenance
        # dicts, which remediation_v6 replays by exact equality.
        provenance["topology_class"] = "linear"
    return _build(
        status=STATUS_SUCCESS,
        quality=QUALITY_EXACT,
        result_origin=ORIGIN_STRICT_V6,
        source_kind=source_kind,
        mode=mode,
        result=strict,
        smiles=strict.output_smiles,
        graph=None,
        ambiguous=False,
        warning_codes=list(strict.warning_codes),
        alternatives=[],
        structure_profile=profile,
        provenance=provenance,
        strict_result=strict,
    )


def _auto_single_chain_coordinate(
    path_text: str,
    chain: str,
    source_kind: str,
    mode: str,
    detection: dict[str, Any],
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
) -> UnifiedReconstructionResult:
    rf = _run_result_first(
        path_text,
        chain,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    if rf.status != STATUS_SUCCESS:
        registry_audit = None
        if mode == "best_effort" and not rf.provenance.get("fallback_block"):
            smiles, error = _try_registry_assembly(path_text, chain)
            if smiles:
                registry_audit = _qualify_registry_assembly(smiles, rf)
                if registry_audit["qualified"]:
                    return _coordinate_assembly_result(
                        source_kind=source_kind,
                        mode=mode,
                        chain=chain,
                        path_text=path_text,
                        smiles=smiles,
                        route="registry_assembly_best_effort",
                        strict=rf.strict_result,
                        detection=detection,
                        alternatives=[
                            {
                                "route": "result_first",
                                "status": "failed",
                                "error": rf.provenance.get("failure_reason"),
                            }
                        ],
                        extra_codes=[_WARNING_BEST_EFFORT_FALLBACK],
                        result_first=rf,
                        registry_audit=registry_audit,
                        quality=QUALITY_MEDIUM,
                        ambiguous=False,
                    )
            elif error:
                registry_audit = {
                    "qualified": False,
                    "status": "not_available",
                    "reason": "REGISTRY_ASSEMBLY_FAILED",
                    "error": error,
                }
        warning_codes = list(rf.warning_codes)
        underlying = dict(rf.provenance or {})
        if registry_audit is not None:
            underlying["registry_assembly"] = registry_audit
            warning_codes.append(_WARNING_REGISTRY_ASSEMBLY_NOT_QUALIFIED)
        return _typed_failed(
            source=path_text,
            chain_id=chain,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="result_first",
            failure_reason=(
                rf.provenance.get("failure_reason")
                or rf.provenance.get("error")
                or "no readable structure at any result-first ladder stage"
            ),
            warning_codes=warning_codes,
            underlying=underlying,
            error=rf.provenance.get("error"),
            strict_result=rf.strict_result,
        )
    if (
        rf.quality in (QUALITY_EXACT, QUALITY_HIGH, QUALITY_MEDIUM)
        or rf.provenance.get("fallback_block")
        or rf.provenance.get("registry_template") is not None
        or rf.provenance.get("normalization_error_recovered")
    ):
        return _from_result_first(
            rf,
            source_kind=source_kind,
            mode=mode,
            chain=chain,
            detection=detection,
            registry_audit=rf.provenance.get("registry_template"),
        )
    # Only topology/partial/raw without fallback_block may be replaced by the
    # registry route after an independent graph-consistency qualification.
    smiles, error = _try_registry_assembly(path_text, chain)
    if smiles:
        registry_audit = _qualify_registry_assembly(smiles, rf)
        if registry_audit["qualified"]:
            return _coordinate_assembly_result(
                source_kind=source_kind,
                mode=mode,
                chain=chain,
                path_text=path_text,
                smiles=smiles,
                route="registry_assembly",
                strict=rf.strict_result,
                detection=detection,
                alternatives=[
                    {
                        "route": "result_first",
                        "status": "success",
                        "quality": rf.quality,
                    }
                ],
                extra_codes=[],
                result_first=rf,
                registry_audit=registry_audit,
                quality=QUALITY_MEDIUM,
                ambiguous=False,
            )
    else:
        registry_audit = {
            "qualified": False,
            "status": "not_available",
            "reason": "REGISTRY_ASSEMBLY_FAILED",
            "error": error,
        }
    return _from_result_first(
        rf,
        source_kind=source_kind,
        mode=mode,
        chain=chain,
        detection=detection,
        registry_audit=registry_audit,
    )


def _with_chain_ambiguity(
    result: UnifiedReconstructionResult,
    *,
    alternatives: list[dict[str, Any]],
    note: str,
) -> UnifiedReconstructionResult:
    codes = list(
        dict.fromkeys([*result.warning_codes, _WARNING_CHAIN_AMBIGUOUS])
    )
    provenance = dict(result.provenance)
    normalized_alternatives = [
        _alternative_summary(item)
        for item in list(result.alternatives) + list(alternatives)
    ]
    provenance["chain_ambiguity"] = {
        "resolved": True,
        "note": note,
        "alternatives": normalized_alternatives,
    }
    return UnifiedReconstructionResult(
        status=result.status,
        quality=result.quality,
        result_origin=result.result_origin,
        source_kind=result.source_kind,
        mode=result.mode,
        result=result.result,
        smiles=result.smiles,
        inchi=result.inchi,
        inchikey=result.inchikey,
        graph=result.graph,
        ambiguous=True,
        warnings=_warning_lines(codes),
        warning_codes=codes,
        alternatives=normalized_alternatives,
        structure_profile=result.structure_profile,
        provenance=provenance,
        strict_result=result.strict_result,
        strict_status=result.strict_status,
        candidate_smiles=result.candidate_smiles,
        candidate_graph=result.candidate_graph,
        chemistry_candidates=list(result.chemistry_candidates),
        bond_order_inference=dict(result.bond_order_inference),
        candidate_rigor=result.candidate_rigor,
        artifact_status=result.artifact_status,
        qualification_status=result.qualification_status,
        chemical_rigor=result.chemical_rigor,
        coordinate_evidence=result.coordinate_evidence,
    )


def _graph_summary(graph: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(graph, dict):
        return None
    atoms = list(graph.get("atoms") or [])
    bonds = list(graph.get("bonds") or [])
    heavy = sum(
        1
        for atom in atoms
        if isinstance(atom, dict)
        and str(atom.get("element") or "").strip().upper() not in {"", "H"}
    )
    return {
        "atom_count": len(atoms),
        "heavy_atom_count": heavy,
        "bond_count": len(bonds),
        "has_coordinates": all(
            isinstance(atom, dict)
            and isinstance(atom.get("xyz"), (list, tuple))
            and len(atom.get("xyz")) == 3
            for atom in atoms
        )
        if atoms
        else False,
    }


def _alternative_summary(
    candidate: Any,
    *,
    chain_id: str | None = None,
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Normalize chain, assembly, and result-first alternatives alike."""
    if isinstance(candidate, UnifiedReconstructionResult):
        summary: dict[str, Any] = {
            "status": status or candidate.status,
            "quality": candidate.quality,
            "result_origin": candidate.result_origin,
            "source_kind": candidate.source_kind,
            "mode": candidate.mode,
            "smiles": candidate.smiles,
            "graph_summary": _graph_summary(candidate.graph),
            "ambiguous": bool(candidate.ambiguous),
            "warnings": list(candidate.warnings),
            "warning_codes": list(candidate.warning_codes),
            "structure_profile": candidate.structure_profile,
            "provenance": dict(candidate.provenance),
        }
        if chain_id is not None:
            summary["chain_id"] = chain_id
        if reason:
            summary["reason"] = reason
        return summary
    raw = dict(candidate) if isinstance(candidate, dict) else {}
    summary = {
        "status": status or raw.get("status", "not_selected"),
        "quality": raw.get("quality"),
        "result_origin": raw.get("result_origin") or raw.get("source"),
        "source_kind": raw.get("source_kind"),
        "mode": raw.get("mode"),
        "smiles": raw.get("smiles") or raw.get("canonical_smiles"),
        "graph_summary": raw.get("graph_summary")
        or _graph_summary(raw.get("graph")),
        "ambiguous": bool(raw.get("ambiguous", False)),
        "warnings": list(raw.get("warnings") or []),
        "warning_codes": list(raw.get("warning_codes") or []),
        "structure_profile": raw.get("structure_profile"),
        "provenance": dict(raw.get("provenance") or {}),
    }
    for key, value in raw.items():
        if key not in summary and key != "graph":
            summary[key] = value
    if chain_id is not None:
        summary["chain_id"] = chain_id
    if reason:
        summary["reason"] = reason
    return summary


_QUALITY_RANK = {
    QUALITY_EXACT: 6,
    QUALITY_HIGH: 5,
    QUALITY_MEDIUM: 4,
    QUALITY_TOPOLOGY: 3,
    QUALITY_PARTIAL: 2,
    QUALITY_RAW: 1,
}


def _multichain_result_first_fallback(
    path_text: str,
    chain_ids: list[str],
    source_kind: str,
    mode: str,
    detection: dict[str, Any],
    *,
    minimum_macrocycle_ring_size: int,
    require_empty_persistent_overlay: bool,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    assembly_reason: str,
) -> UnifiedReconstructionResult | None:
    """Select a usable chain result after a multi-chain assembler fails."""
    candidates: list[tuple[int, str, UnifiedReconstructionResult]] = []
    failed: list[dict[str, Any]] = []
    for index, chain in enumerate(chain_ids):
        rf = _run_result_first(
            path_text,
            chain,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
        )
        if rf.status != STATUS_SUCCESS:
            failed.append(
                _alternative_summary(
                    {
                        "chain_id": chain,
                        "status": rf.status,
                        "quality": rf.quality,
                        "source": rf.source,
                        "smiles": rf.smiles,
                        "graph": rf.graph,
                        "warnings": rf.warnings,
                        "warning_codes": rf.warning_codes,
                        "provenance": rf.provenance,
                    },
                    chain_id=chain,
                    status=rf.status,
                    reason=rf.provenance.get("failure_reason"),
                )
            )
            continue
        candidate = _from_result_first(
            rf,
            source_kind=source_kind,
            mode=mode,
            chain=chain,
            detection=detection,
        )
        candidates.append(
            (_QUALITY_RANK.get(candidate.quality or "", 0), chain, candidate)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], chain_ids.index(item[1])))
    _rank, primary_chain, primary = candidates[0]
    alternatives = [
        _alternative_summary(candidate, chain_id=chain)
        for _item_rank, chain, candidate in candidates[1:]
    ]
    alternatives.extend(failed)
    alternatives.append(
        _alternative_summary(
            {
                "status": STATUS_FAILED,
                "quality": None,
                "source": ORIGIN_MULTICHAIN_ASSEMBLY,
                "reason": assembly_reason,
            },
            status=STATUS_FAILED,
            reason=assembly_reason,
        )
    )
    return _with_chain_ambiguity(
        primary,
        alternatives=alternatives,
        note=(
            "multi-chain assembly failed; deterministic primary chain "
            f"{primary_chain!r} selected by quality and input order"
        ),
    )


def _explicit_chain_components(
    path_text: str, chain_ids: list[str]
) -> tuple[list[list[str]], dict[str, Any]]:
    """Group chains using only explicit PDB connectivity records.

    Automatic chain selection must not infer a covalent inter-chain bond from
    coordinate proximity.  SSBOND and LINK carry chain identifiers directly;
    CONECT carries atom serials, so its endpoints are resolved against the
    first-model atom records for the already selected peptide chains.  Ambiguous
    or non-peptide endpoints are ignored rather than guessed.  The returned
    evidence is intentionally JSON-shaped so the dispatch decision remains
    auditable in ``provenance``.
    """
    from collections import defaultdict as _defaultdict

    parent = {chain: chain for chain in chain_ids}
    evidence: dict[str, Any] = {
        "ssbond": [],
        "link": [],
        "conect": [],
        "errors": [],
        "inference": "explicit_records_only",
    }

    def find(chain: str) -> str:
        root = chain
        while parent[root] != root:
            root = parent[root]
        while parent[chain] != chain:
            parent[chain], chain = root, parent[chain]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    def use_record(kind: str, record: dict[str, Any], left: str, right: str) -> None:
        if left in parent and right in parent and left != right:
            union(left, right)
            item = dict(record)
            item.update({"chain1": left, "chain2": right})
            evidence[kind].append(item)

    try:
        from .core.cyclization import read_atoms, read_conect, read_link, read_ssbond

        for bond in read_ssbond(path_text):
            use_record(
                "ssbond",
                {key: value for key, value in bond.items()},
                str(bond.get("chain1", "")),
                str(bond.get("chain2", "")),
            )
        for link in read_link(path_text):
            use_record(
                "link",
                {key: value for key, value in link.items()},
                str(link.get("chain1", "")),
                str(link.get("chain2", "")),
            )

        serial_to_chains: dict[int, set[str]] = _defaultdict(set)
        for chain in chain_ids:
            for serial in read_atoms(path_text, chain):
                serial_to_chains[int(serial)].add(chain)
        for left_serial, right_serial in read_conect(path_text):
            left_chains = serial_to_chains.get(int(left_serial), set())
            right_chains = serial_to_chains.get(int(right_serial), set())
            # A serial reused across chains or an endpoint outside the selected
            # peptide set is not safe for automatic grouping.
            if len(left_chains) != 1 or len(right_chains) != 1:
                continue
            left = next(iter(left_chains))
            right = next(iter(right_chains))
            if left == right or left not in parent or right not in parent:
                continue
            use_record(
                "conect",
                {"serial1": int(left_serial), "serial2": int(right_serial)},
                left,
                right,
            )
    except (OSError, TypeError, ValueError) as exc:
        # A failed optional evidence parser must not turn into a coordinate
        # proximity guess.  Preserve the failure for the caller's audit trail.
        evidence["errors"].append(f"{type(exc).__name__}: {exc}")

    grouped: dict[str, list[str]] = {}
    for chain in chain_ids:
        grouped.setdefault(find(chain), []).append(chain)
    return list(grouped.values()), evidence


def _ssbond_chain_components(
    path_text: str, chain_ids: list[str]
) -> list[list[str]]:
    """Backward-compatible view of explicit chain components."""
    groups, _evidence = _explicit_chain_components(path_text, chain_ids)
    return groups


def _profile_from_multichain(
    path_text: str, chain_ids: list[str]
) -> dict[str, Any] | None:
    try:
        from .paths._map_utils import helm_to_map
        from .paths.path_b import build_helm_multichain

        helm = build_helm_multichain(path_text, list(chain_ids))
        if not helm:
            return None
        mapped = helm_to_map(helm)
        if not mapped:
            return None
        chains, edges = _parse_map_graph("map", mapped)
        return _profile_from_residue_graph(
            chains, edges, is_multichain=True
        )
    except Exception:
        return None


def _multichain_assembly(
    path_text: str,
    chain_ids: list[str],
    source_kind: str,
    mode: str,
    detection: dict[str, Any],
) -> UnifiedReconstructionResult:
    from .paths import generate_multichain

    try:
        smiles, error = generate_multichain(path_text, list(chain_ids))
    except Exception as exc:
        smiles, error = None, f"{type(exc).__name__}: {exc}"
    if not smiles:
        return _typed_failed(
            source=path_text,
            chain_id=list(chain_ids),
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="multichain_assembly",
            failure_reason=error or "multi-chain assembly produced no SMILES",
            warning_codes=[_WARNING_ASSEMBLY_FAILED],
            error=error,
        )
    from .chemical_audit import audit_output_smiles

    audit = audit_output_smiles(smiles)
    if not audit.accepted:
        return _typed_failed(
            source=path_text,
            chain_id=list(chain_ids),
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="multichain_assembly",
            failure_reason=(
                audit.reason or "multi-chain SMILES rejected by output audit"
            ),
            warning_codes=[_WARNING_ASSEMBLY_FAILED],
            error=audit.reason,
        )
    profile = _profile_from_multichain(path_text, chain_ids)
    provenance = {
        "detection": detection,
        "dispatch": {
            "mode": mode,
            "branch": "multichain_assembly",
            "chain_id": list(chain_ids),
        },
        "underlying": {"route": "generate_multichain"},
    }
    return _build(
        status=STATUS_SUCCESS,
        quality=QUALITY_HIGH,
        result_origin=ORIGIN_MULTICHAIN_ASSEMBLY,
        source_kind=source_kind,
        mode=mode,
        result={
            "route": "multichain_assembly",
            "smiles": smiles,
            "chain_ids": list(chain_ids),
        },
        smiles=smiles,
        graph=None,
        ambiguous=False,
        warning_codes=[_WARNING_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED],
        alternatives=[],
        structure_profile=profile,
        provenance=provenance,
        strict_result=None,
    )


def _best_effort_single_chain_fallback(
    path_text: str,
    chain_ids: list[str],
    source_kind: str,
    mode: str,
    detection: dict[str, Any],
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
) -> UnifiedReconstructionResult | None:
    """Deterministic per-chain result-first fallback."""
    return _multichain_result_first_fallback(
        path_text,
        chain_ids,
        source_kind,
        mode,
        detection,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
        assembly_reason="multi-chain assembly produced no usable SMILES",
    )


def _reconstruct_coordinate(
    path_text: str,
    source_kind: str,
    classification_error: str | None,
    chain_id: Any,
    mode: str,
    detection: dict[str, Any],
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
) -> UnifiedReconstructionResult:
    if classification_error:
        warning_code = (
            _WARNING_PREFIX_KIND_MISMATCH
            if classification_error.startswith(_WARNING_PREFIX_KIND_MISMATCH)
            else _WARNING_COORDINATE_INPUT_ERROR
        )
        return _typed_failed(
            source=path_text,
            chain_id=chain_id,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="coordinate_input",
            failure_reason=classification_error,
            warning_codes=[warning_code],
            error=classification_error,
        )
    try:
        fmt = coordinate_format(Path(path_text))
    except CoordinateInputError as exc:
        return _typed_failed(
            source=path_text,
            chain_id=chain_id,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="coordinate_input",
            failure_reason=f"{exc.code}: {exc}",
            warning_codes=[_WARNING_COORDINATE_INPUT_ERROR],
            error=f"{exc.code}: {exc}",
        )
    detection = dict(detection)
    detection["coordinate_format"] = fmt
    if not Path(path_text).is_file():
        return _typed_failed(
            source=path_text,
            chain_id=chain_id,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="coordinate_input",
            failure_reason=f"coordinate input does not exist: {path_text}",
            warning_codes=[_WARNING_COORDINATE_INPUT_ERROR],
            error=f"COORDINATE_INPUT_NOT_FOUND: {path_text}",
        )
    chains, single, resolution_error = _resolve_coordinate_chains(
        path_text, fmt, chain_id
    )
    if resolution_error:
        return _typed_failed(
            source=path_text,
            chain_id=chain_id,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="chain_resolution",
            failure_reason=resolution_error["message"],
            warning_codes=[resolution_error["code"]],
            underlying={"chain_resolution": resolution_error},
        )
    native_graph = None
    native_graph_error = None
    assembly_chains = list(chains)
    if fmt == "mmcif":
        assembly_chains = _native_mmcif_dispatch_ids(path_text, chains)
        try:
            from .core.native_mmcif_graph import read_native_mmcif

            native_graph = read_native_mmcif(
                path_text,
                chain_ids=chains,
                model_num=1,
                peptide_only=True,
            )
            detection = dict(detection)
            detection["native_mmcif"] = {
                "selected_chain_ids": list(native_graph.selected_chain_ids),
                "chain_dispatch_ids": list(assembly_chains),
                "warnings": list(native_graph.warnings),
            }
        except Exception as exc:
            native_graph_error = f"{type(exc).__name__}: {exc}"
            detection = dict(detection)
            detection["native_mmcif"] = {
                "selected_chain_ids": list(chains),
                "chain_dispatch_ids": list(assembly_chains),
                "read_error": native_graph_error,
            }
    if single:
        assembly_chain = assembly_chains[0]
        if mode == "strict":
            return _strict_coordinate(
                path_text,
                assembly_chain,
                source_kind,
                mode,
                detection,
                minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=require_empty_persistent_overlay,
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
        result = _auto_single_chain_coordinate(
            path_text,
            assembly_chain,
            source_kind,
            mode,
            detection,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
        )
        if result.status == STATUS_SUCCESS or native_graph is None:
            return result
        native = _native_mmcif_result(
            graph=native_graph,
            groups=[list(chains)],
            source_kind=source_kind,
            mode=mode,
            chain_id=chain_id,
            detection=detection,
            assembly_reason=result.provenance.get("failure_reason")
            or native_graph_error,
            ambiguous=False,
            alternatives=[_alternative_summary(result, chain_id=chains[0])],
        )
        return native or result
    # Multiple peptide chains.
    if mode == "strict":
        return _typed_failed(
            source=path_text,
            chain_id=chain_id,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="chain_resolution",
            failure_reason=(
                "strict mode supports only single-chain strict V6; multiple "
                f"peptide chains found: {chains}"
            ),
            warning_codes=[_WARNING_STRICT_MULTICHAIN_UNSUPPORTED],
            underlying={"peptide_chains": chains},
        )
    if isinstance(chain_id, (list, tuple)):
        # An explicit list is an atomic caller selection.  Never silently
        # discard requested chains in best_effort mode; the existing
        # multi-chain assembler must either preserve the full list or fail.
        assembled = _multichain_assembly(
            path_text, assembly_chains, source_kind, mode, detection
        )
        if assembled.status == STATUS_SUCCESS:
            return assembled
        if native_graph is not None:
            native = _native_mmcif_result(
                graph=native_graph,
                groups=[list(chains)],
                source_kind=source_kind,
                mode=mode,
                chain_id=chain_id,
                detection=detection,
                assembly_reason=assembled.provenance.get("failure_reason")
                or native_graph_error,
                ambiguous=False,
                alternatives=[_alternative_summary(assembled)],
            )
            if native is not None:
                return native
        fallback = _multichain_result_first_fallback(
            path_text,
            assembly_chains,
            source_kind,
            mode,
            detection,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            assembly_reason=assembled.provenance.get("failure_reason")
            or "explicit multi-chain assembly failed",
        )
        if fallback is not None:
            return fallback
        return assembled
    if fmt == "mmcif" and native_graph is not None:
        groups, explicit_evidence = _native_mmcif_components(
            native_graph, chains
        )
    else:
        groups, explicit_evidence = _explicit_chain_components(path_text, chains)
    detection = dict(detection)
    detection["explicit_connection_evidence"] = explicit_evidence
    if len(groups) == 1:
        assembled = _multichain_assembly(
            path_text, assembly_chains, source_kind, mode, detection
        )
        if assembled.status == STATUS_SUCCESS:
            return assembled
        if native_graph is not None:
            native = _native_mmcif_result(
                graph=native_graph,
                groups=groups,
                source_kind=source_kind,
                mode=mode,
                chain_id=chain_id,
                detection=detection,
                assembly_reason=assembled.provenance.get("failure_reason")
                or native_graph_error,
                ambiguous=False,
                alternatives=[_alternative_summary(assembled)],
            )
            if native is not None:
                return native
        fallback = _multichain_result_first_fallback(
            path_text,
            assembly_chains,
            source_kind,
            mode,
            detection,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            assembly_reason=assembled.provenance.get("failure_reason")
            or "multi-chain assembly failed",
        )
        if fallback is not None:
            return fallback
        return assembled
    if mode == "auto":
        if native_graph is not None:
            native = _native_mmcif_result(
                graph=native_graph,
                groups=groups,
                source_kind=source_kind,
                mode=mode,
                chain_id=chain_id,
                detection=detection,
                assembly_reason=(
                    "multiple peptide chains are not joined by explicit "
                    "inter-chain struct_conn records"
                ) or native_graph_error,
                ambiguous=True,
                alternatives=[],
            )
            if native is not None:
                native.provenance.setdefault("chain_resolution", {})
                native.provenance["chain_resolution"].update({
                    "peptide_chains": chains,
                    "ssbond_components": groups,
                    "explicit_connection_evidence": explicit_evidence,
                })
                return native
        fallback = _multichain_result_first_fallback(
            path_text,
            assembly_chains,
            source_kind,
            mode,
            detection,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            assembly_reason=(
                "multiple peptide chains are not joined by explicit "
                "inter-chain SSBOND/LINK/CONECT records"
            ),
        )
        if fallback is not None:
            fallback.provenance.setdefault("chain_resolution", {})
            fallback.provenance["chain_resolution"].update({
                "peptide_chains": chains,
                "ssbond_components": groups,
                "explicit_connection_evidence": explicit_evidence,
            })
            return fallback
        return _typed_failed(
            source=path_text,
            chain_id=chain_id,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="chain_resolution",
            failure_reason=(
                "multiple peptide chains are not joined by explicit "
                "inter-chain SSBOND/LINK/CONECT records and all fallback "
                "chains failed: "
                f"{chains}"
            ),
            warning_codes=[_WARNING_REQUIRES_EXPLICIT_CHAINS],
            underlying={
                "peptide_chains": chains,
                "ssbond_components": groups,
                "explicit_connection_evidence": explicit_evidence,
            },
        )
    # best_effort: deterministic primary candidate + alternatives/ambiguous.
    primary_group = max(
        groups, key=lambda group: (len(group), -chains.index(group[0]))
    )
    remaining = [chain for chain in chains if chain not in primary_group]
    result = _multichain_assembly(
        path_text,
        [assembly_chains[chains.index(label)] for label in primary_group],
        source_kind,
        mode,
        detection,
    )
    if result.status == STATUS_SUCCESS:
        return _with_chain_ambiguity(
            result,
            alternatives=[
                {"chain_id": other, "status": "not_selected"}
                for other in remaining
            ],
            note=(
                "multiple disconnected chain groups; deterministic primary "
                f"group {primary_group!r} selected"
            ),
        )
    if native_graph is not None:
        native = _native_mmcif_result(
            graph=native_graph,
            groups=groups,
            source_kind=source_kind,
            mode=mode,
            chain_id=chain_id,
            detection=detection,
            assembly_reason=(
                "multi-chain primary assembly failed before fallback"
            ) or native_graph_error,
            ambiguous=True,
            alternatives=[],
        )
        if native is not None:
            return native
    fallback = _best_effort_single_chain_fallback(
        path_text,
        assembly_chains,
        source_kind,
        mode,
        detection,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    if fallback is not None:
        return fallback
    return _typed_failed(
        source=path_text,
        chain_id=chain_id,
        mode=mode,
        source_kind=source_kind,
        detection=detection,
        branch="chain_resolution",
        failure_reason=(
            "multi-chain and single-chain branches failed for all peptide "
            "chains in best_effort mode"
        ),
        warning_codes=[
            _WARNING_REQUIRES_EXPLICIT_CHAINS,
            _WARNING_BEST_EFFORT_FALLBACK,
        ],
        underlying={
            "peptide_chains": chains,
            "ssbond_components": groups,
            "explicit_connection_evidence": explicit_evidence,
        },
    )


def _resolve_coordinate_chains(
    path_text: str, fmt: str, chain_id: Any
) -> tuple[list[str], bool, dict[str, Any] | None]:
    """Resolve chain_id into concrete PDB chain ids for a coordinate input."""
    compressed = path_text.lower().endswith(".gz")
    if isinstance(chain_id, (list, tuple)):
        raw_ids = [str(value) for value in chain_id]
        if not raw_ids:
            return [], False, {
                "code": _WARNING_EMPTY_CHAIN_LIST,
                "message": "the chain list is empty",
            }
        empty_positions = [
            index for index, value in enumerate(raw_ids) if not value.strip()
        ]
        if empty_positions:
            return [], False, {
                "code": _WARNING_EMPTY_CHAIN_IDENTIFIER,
                "message": (
                    "the explicit chain list contains empty identifiers at "
                    f"positions: {empty_positions}"
                ),
                "empty_chain_id_positions": empty_positions,
            }
        ids = [value.strip() for value in raw_ids]
        if fmt == "mmcif":
            return _native_mmcif_select_labels(path_text, chain_id)
        if compressed:
            return [], False, {
                "code": _WARNING_COMPRESSED_CHAIN_AUTOSELECT,
                "message": (
                    "multi-chain assembly for compressed coordinates is not "
                    "supported; pass an explicit chain_id"
                ),
            }
        duplicate_ids = sorted(
            {value for value in ids if ids.count(value) > 1}
        )
        if duplicate_ids:
            return [], False, {
                "code": _WARNING_DUPLICATE_CHAIN_SELECTION,
                "message": (
                    "the explicit chain list contains duplicate identifiers: "
                    f"{duplicate_ids}"
                ),
                "duplicate_chain_ids": duplicate_ids,
            }
        try:
            available = _auto_peptide_chain_ids(path_text)
        except Exception as exc:
            return [], False, {
                "code": _WARNING_CHAIN_ENUMERATION_FAILED,
                "message": (
                    "peptide chain enumeration failed while validating the "
                    f"explicit chain list: {type(exc).__name__}: {exc}"
                ),
            }
        missing = [value for value in ids if value not in available]
        if missing:
            return [], False, {
                "code": _WARNING_REQUESTED_CHAIN_NOT_AVAILABLE,
                "message": (
                    "explicitly requested peptide chain(s) are unavailable: "
                    f"{missing}; available peptide chains: {available}"
                ),
                "requested_chain_ids": ids,
                "missing_chain_ids": missing,
                "available_peptide_chain_ids": available,
            }
        return ids, len(ids) == 1, None
    if fmt == "mmcif":
        return _native_mmcif_select_labels(path_text, chain_id)
    if chain_id is not None:
        return [str(chain_id)], True, None
    if compressed:
        return [], False, {
            "code": _WARNING_COMPRESSED_CHAIN_AUTOSELECT,
            "message": (
                "chain auto-selection for compressed coordinates is not "
                "supported; pass an explicit chain_id"
            ),
            }
    try:
        chains = _auto_peptide_chain_ids(path_text)
    except Exception as exc:
        return [], False, {
            "code": _WARNING_CHAIN_ENUMERATION_FAILED,
            "message": (
                f"peptide chain enumeration failed: {type(exc).__name__}: {exc}"
            ),
        }
    if not chains:
        return [], False, {
            "code": _WARNING_NO_PEPTIDE_CHAIN,
            "message": "no peptide-bearing chain was found in the coordinate input",
        }
    if len(chains) == 1:
        return chains, True, None
    return chains, False, None


def _auto_peptide_chain_ids(path_text: str) -> list[str]:
    """Conservatively identify peptide-capable PDB chains for auto selection.

    Path B intentionally treats every non-solvent residue as potentially
    peptide-like because callers historically supplied a chain explicitly.
    The unified API cannot use that permissive rule for automatic selection:
    a ligand-only HETATM chain must not become a raw peptide result.  A chain
    therefore needs either recognizable N-CA-C backbone atoms in one residue,
    or at least two residues registered as peptide monomers.
    """
    from .paths import _map_utils as map_utils
    from .paths.path_b import _AA_3TO1
    from .core.pdb_utils import first_model_records

    skip = {
        "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "FE",
        "MN", "CU", "SO4", "PO4", "GOL", "EDO", "DMS", "NAG", "MAN",
    }
    residues: dict[
        str, dict[tuple[str, str, str], dict[str, Any]]
    ] = defaultdict(dict)
    chain_order: list[str] = []
    with Path(path_text).open(encoding="utf-8", errors="replace") as handle:
        for line in first_model_records(handle):
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            chain = line[21:22]
            residue_name = line[17:20].strip().upper()
            if not residue_name or residue_name in skip:
                continue
            if chain not in residues:
                chain_order.append(chain)
            key = (line[22:26], line[26:27], residue_name)
            row = residues[chain].setdefault(
                key, {"name": residue_name, "atoms": set()}
            )
            row["atoms"].add(line[12:16].strip().upper())

    selected: list[str] = []
    registry = map_utils.monomers2smi_dict
    for chain in chain_order:
        rows = list(residues[chain].values())
        registered = [
            row["name"] in _AA_3TO1
            or row["name"] in registry
            or map_utils.resolve_pdb_alias(row["name"]) in registry
            for row in rows
        ]
        backbone = [
            {"N", "CA", "C"}.issubset(row["atoms"]) for row in rows
        ]
        registered_count = sum(registered)
        backbone_evidence = (
            sum(backbone) >= 2
            or any(
                has_backbone and is_registered
                for has_backbone, is_registered in zip(backbone, registered)
            )
        )
        if backbone_evidence or registered_count >= 2:
            selected.append(chain)
    return selected


def _native_mmcif_records(path_text: str) -> tuple[Any, ...]:
    """Return native peptide-chain records for one mmCIF input.

    The import is deliberately lazy: PDB and notation-only callers must not
    require the native adapter at dispatch time.  The adapter itself is the
    sole source used for mmCIF chain enumeration; no PDB projection or
    coordinate-proximity inference is involved here.
    """

    from .core.native_mmcif_graph import inspect_mmcif, list_peptide_chains

    # Keep the all-chain inspection and peptide-only projection tied to the
    # same adapter semantics.  This also leaves non-peptide records available
    # to the adapter for provenance without allowing them into auto selection.
    inspected = tuple(inspect_mmcif(path_text))
    peptide_labels = {
        record.label_asym_id
        for record in list_peptide_chains(path_text)
    }
    return tuple(
        record for record in inspected if record.label_asym_id in peptide_labels
    )


def _native_mmcif_select_labels(
    path_text: str, chain_id: Any
) -> tuple[list[str], bool, dict[str, Any] | None]:
    """Resolve an mmCIF chain request to native label asym IDs.

    ``auth_asym_id`` is accepted as a user-facing alias, but the returned
    identifiers are always ``label_asym_id`` so that native graph selection is
    unambiguous.  Explicit lists retain caller order; automatic selection
    follows the adapter's deterministic record order.
    """

    try:
        records = _native_mmcif_records(path_text)
    except Exception as exc:
        return [], False, {
            "code": _WARNING_NATIVE_MMCIF_READ_FAILED,
            "message": (
                "native mmCIF chain enumeration failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }

    if isinstance(chain_id, (list, tuple)):
        raw_ids = [str(value) for value in chain_id]
        if not raw_ids:
            return [], False, {
                "code": _WARNING_EMPTY_CHAIN_LIST,
                "message": "the chain list is empty",
            }
        empty_positions = [
            index for index, value in enumerate(raw_ids) if not value.strip()
        ]
        if empty_positions:
            return [], False, {
                "code": _WARNING_EMPTY_CHAIN_IDENTIFIER,
                "message": (
                    "the explicit chain list contains empty identifiers at "
                    f"positions: {empty_positions}"
                ),
                "empty_chain_id_positions": empty_positions,
            }
        requested = [value.strip() for value in raw_ids]
        duplicate_ids = sorted({value for value in requested if requested.count(value) > 1})
        if duplicate_ids:
            return [], False, {
                "code": _WARNING_DUPLICATE_CHAIN_SELECTION,
                "message": (
                    "the explicit chain list contains duplicate identifiers: "
                    f"{duplicate_ids}"
                ),
                "duplicate_chain_ids": duplicate_ids,
            }
    elif chain_id is not None:
        requested = [str(chain_id).strip()]
        if not requested[0]:
            return [], False, {
                "code": _WARNING_EMPTY_CHAIN_IDENTIFIER,
                "message": "the explicit chain identifier is empty",
            }
    else:
        selected = [record.label_asym_id for record in records]
        if not selected:
            return [], False, {
                "code": _WARNING_NO_PEPTIDE_CHAIN,
                "message": "no peptide-bearing chain was found in the mmCIF input",
            }
        return selected, len(selected) == 1, None

    by_label = {record.label_asym_id: record for record in records}
    by_auth: dict[str, list[Any]] = {}
    for record in records:
        if record.auth_asym_id:
            by_auth.setdefault(record.auth_asym_id, []).append(record)
    selected: list[str] = []
    ambiguous: list[str] = []
    missing: list[str] = []
    for value in requested:
        record = by_label.get(value)
        if record is None:
            matches = by_auth.get(value, [])
            if len(matches) == 1:
                record = matches[0]
            elif len(matches) > 1:
                ambiguous.append(value)
                continue
        if record is None:
            missing.append(value)
        else:
            selected.append(record.label_asym_id)
    if ambiguous:
        return [], False, {
            "code": _WARNING_REQUESTED_CHAIN_NOT_AVAILABLE,
            "message": (
                "mmCIF author chain identifier is ambiguous: "
                f"{sorted(ambiguous)}"
            ),
            "ambiguous_chain_ids": sorted(ambiguous),
        }
    if missing:
        available = [record.label_asym_id for record in records]
        return [], False, {
            "code": _WARNING_REQUESTED_CHAIN_NOT_AVAILABLE,
            "message": (
                "explicitly requested mmCIF peptide chain(s) are unavailable: "
                f"{missing}; available native peptide chains: {available}"
            ),
            "requested_chain_ids": requested,
            "missing_chain_ids": missing,
            "available_peptide_chain_ids": available,
        }
    duplicate_labels = sorted({
        value for value in selected if selected.count(value) > 1
    })
    if duplicate_labels:
        return [], False, {
            "code": _WARNING_DUPLICATE_CHAIN_SELECTION,
            "message": (
                "the explicit mmCIF chain selection resolves to duplicate "
                f"native labels: {duplicate_labels}"
            ),
            "duplicate_chain_ids": duplicate_labels,
        }
    return selected, len(selected) == 1, None


def _native_mmcif_dispatch_ids(
    path_text: str, label_ids: list[str]
) -> list[str]:
    """Map native labels to legacy projected author chain IDs when safe."""

    try:
        records = _native_mmcif_records(path_text)
    except Exception:
        return list(label_ids)
    by_label = {record.label_asym_id: record for record in records}
    candidate = [
        by_label.get(label).auth_asym_id or label
        if by_label.get(label) is not None else label
        for label in label_ids
    ]
    # A projected PDB chain cannot represent two native label chains that share
    # one author asym ID.  Leave labels intact so the native fallback remains
    # available instead of silently merging them.
    if len(set(candidate)) != len(candidate):
        return list(label_ids)
    return candidate


def _native_mmcif_components(
    graph: Any, label_ids: list[str]
) -> tuple[list[list[str]], dict[str, Any]]:
    """Group native mmCIF chains using resolved explicit ``struct_conn`` rows."""

    parent = {label: label for label in label_ids}
    evidence: dict[str, Any] = {
        "struct_conn": [],
        "chem_comp_bond": [],
        "inference": "native_struct_conn_only",
    }

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            parent[value], value = root, parent[value]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for bond in graph.bonds:
        if not bond.resolved or not bond.within_selection:
            continue
        left = bond.endpoint_1.label_asym_id
        right = bond.endpoint_2.label_asym_id
        if left not in parent or right not in parent or left == right:
            continue
        union(left, right)
        evidence.setdefault(bond.source, []).append(bond.to_dict())

    grouped: dict[str, list[str]] = {}
    for label in label_ids:
        grouped.setdefault(find(label), []).append(label)
    return list(grouped.values()), evidence


def _native_mmcif_common_graph(graph: Any) -> dict[str, Any]:
    """Project native adapter records to the unified graph contract."""

    atoms: list[dict[str, Any]] = []
    atom_serials: dict[str, int] = {}
    for serial, atom in enumerate(graph.atoms, start=1):
        atom_serials[atom.atom_id] = serial
        coordinates = atom.coordinates
        atoms.append({
            "serial": serial,
            "atom_id": atom.atom_id,
            "source_atom_id": atom.source_atom_id,
            "element": atom.element,
            "element_raw": atom.element_raw,
            "chain": atom.chain_id,
            "residue_number": atom.label_seq_id or atom.auth_seq_id,
            "residue_name": atom.label_comp_id or atom.auth_comp_id,
            "atom_name": atom.label_atom_id or atom.auth_atom_id,
            "x": coordinates[0] if coordinates is not None else None,
            "y": coordinates[1] if coordinates is not None else None,
            "z": coordinates[2] if coordinates is not None else None,
            "formal_charge": atom.formal_charge,
            "occupancy": atom.occupancy,
            "b_iso": atom.b_iso,
            "model_num": atom.model_num,
            "group_pdb": atom.group_pdb,
            "label_asym_id": atom.label_asym_id,
            "auth_asym_id": atom.auth_asym_id,
            "label_seq_id": atom.label_seq_id,
            "auth_seq_id": atom.auth_seq_id,
            "label_comp_id": atom.label_comp_id,
            "auth_comp_id": atom.auth_comp_id,
        })

    bonds: list[dict[str, Any]] = []
    source_bonds: list[dict[str, Any]] = []
    for bond in graph.bonds:
        row = bond.to_dict()
        row["a"] = atom_serials.get(bond.atom_id_1)
        row["b"] = atom_serials.get(bond.atom_id_2)
        source_bonds.append(row)
        if (
            bond.resolved and bond.within_selection
            and row["a"] is not None and row["b"] is not None
            and row["a"] != row["b"]
        ):
            bonds.append({
                "a": row["a"],
                "b": row["b"],
                "order": bond.order,
                "source": bond.source,
                "connection_id": bond.connection_id,
                "connection_type": bond.connection_type,
            })
    return {
        "atoms": atoms,
        "bonds": bonds,
        "source_bonds": source_bonds,
        "chains": [chain.to_dict() for chain in graph.chains],
        "entities": [entity.to_dict() for entity in graph.entities],
        "selected_chain_ids": list(graph.selected_chain_ids),
        "model_num": graph.model_num,
        "warnings": list(graph.warnings),
        "provenance": dict(graph.provenance),
        "graph_evidence": "native_mmcif_atom_site_struct_conn",
    }


def _native_mmcif_result(
    *,
    graph: Any,
    groups: list[list[str]],
    source_kind: str,
    mode: str,
    chain_id: Any,
    detection: dict[str, Any],
    assembly_reason: str | None,
    ambiguous: bool,
    alternatives: list[dict[str, Any]] | None = None,
) -> UnifiedReconstructionResult | None:
    """Return a usable native graph after legacy assembly has failed."""

    common_graph = _native_mmcif_common_graph(graph)
    atoms = common_graph["atoms"]
    if not atoms:
        return None
    resolved_bonds = common_graph["bonds"]
    complete_atoms = all(
        atom.get("element") and all(
            atom.get(axis) is not None for axis in ("x", "y", "z")
        )
        for atom in atoms
    )
    if not resolved_bonds:
        quality = QUALITY_RAW
    elif complete_atoms and len(resolved_bonds) == len(common_graph["source_bonds"]):
        quality = QUALITY_TOPOLOGY
    else:
        quality = QUALITY_PARTIAL
    profile = _profile_from_graph(
        common_graph,
        chain_count=len(graph.selected_chain_ids),
        is_multichain=len(graph.selected_chain_ids) > 1,
    )
    warning_codes = [
        _WARNING_NATIVE_MMCIF_GRAPH_FALLBACK,
        _WARNING_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED,
    ]
    if len(groups) > 1:
        warning_codes.append(_WARNING_NATIVE_MMCIF_DISCONNECTED)
    underlying: dict[str, Any] = {
        "route": "native_mmcif_graph",
        "native_graph": graph.to_dict(),
        "explicit_connection_components": groups,
        "explicit_connection_evidence": _native_mmcif_components(
            graph, list(graph.selected_chain_ids)
        )[1],
    }
    if assembly_reason:
        underlying["legacy_assembly_failure"] = assembly_reason
    provenance = {
        "detection": detection,
        "dispatch": {
            "mode": mode,
            "branch": "native_mmcif_graph",
            "chain_id": _chain_id_summary(chain_id),
        },
        "underlying": underlying,
        "native_mmcif": {
            "selected_chain_ids": list(graph.selected_chain_ids),
            "groups": groups,
            "warnings": list(graph.warnings),
        },
    }
    return _build(
        status=STATUS_SUCCESS,
        quality=quality,
        result_origin=ORIGIN_NATIVE_MMCIF_GRAPH,
        source_kind=source_kind,
        mode=mode,
        result={"route": "native_mmcif_graph", "graph": common_graph},
        smiles=None,
        graph=common_graph,
        ambiguous=ambiguous,
        warning_codes=warning_codes,
        alternatives=alternatives or [],
        structure_profile=profile,
        provenance=provenance,
        strict_result=None,
    )


# --------------------------------------------------------------------------
# Representation branches
# --------------------------------------------------------------------------


def _assemble_representation(kind: str, text: str) -> tuple[str | None, str | None]:
    """Strict parser + deterministic monomer-registry assembly only."""
    from .chemical_audit import audit_output_smiles

    try:
        if kind == "sequence":
            from .paths import _map_utils
            from .sequence import _sequence_tokens

            symbols = _sequence_tokens(text)
            map_text = "".join(
                _map_utils._symbol_to_map.get(
                    symbol, _map_utils._auto_map_denotion(symbol)
                )
                for symbol in symbols
            )
            smiles = _map_utils.get_smi_from_map(map_text)
            if not smiles:
                return None, (
                    "linear sequence assembly produced no SMILES (unknown "
                    "monomer or unsupported residue)"
                )
        elif kind == "map":
            from .representations import map_to_smiles

            smiles = map_to_smiles(text)
        elif kind == "helm":
            from .representations import helm_to_smiles

            smiles = helm_to_smiles(text)
        elif kind == "biln":
            from .representations import biln_to_smiles

            smiles = biln_to_smiles(text)
        else:  # pragma: no cover - guarded by callers
            return None, f"unsupported representation kind: {kind}"
    except (TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    audit = audit_output_smiles(smiles)
    if not audit.accepted:
        return None, audit.reason or "assembled SMILES rejected by output audit"
    return smiles, None


def _representation_result(
    *,
    kind: str,
    text: str,
    mode: str,
    detection: dict[str, Any],
) -> UnifiedReconstructionResult:
    smiles, error = _assemble_representation(kind, text)
    if not smiles:
        from .core.monomer_resolution import monomer_symbol_hints
        from .exact_v1 import MonomerRegistry

        unresolved_symbols = []
        registry = MonomerRegistry()
        for symbol in monomer_symbol_hints(text, kind=kind):
            try:
                registry.resolve(symbol)
            except Exception:
                unresolved_symbols.append(symbol)
        if not unresolved_symbols:
            return _typed_failed(
                source=text,
                chain_id=None,
                mode=mode,
                source_kind=kind,
                detection=detection,
                branch=f"{kind}_assembly",
                failure_reason=f"{kind} assembly failed: {error}",
                warning_codes=[_WARNING_ASSEMBLY_FAILED],
                error=error,
            )
        exact = None
        reason_codes = []
        if kind in {"map", "helm", "biln"}:
            try:
                from .exact_v1 import notation_to_exact_v1

                exact = notation_to_exact_v1(kind, text)
                reason_codes = list(exact.get("reason_codes") or [])
            except Exception as exc:
                reason_codes = [
                    f"SYMBOLIC_PARSE_FAILED:{type(exc).__name__}:{exc}"
                ]
        else:
            reason_codes = [
                "SEQUENCE_MONOMER_RESOLUTION_INCOMPLETE"
            ]
        symbolic_graph = {
            "schema": "cycpep_symbolic_graph_v1",
            "source_kind": kind,
            "source_value": text,
            "exact_v1": exact,
            "materializable": False,
            "chemical_rigor": "C1:H",
            "unresolved_monomers": list(dict.fromkeys(
                unresolved_symbols
            )),
            "reason_codes": reason_codes,
        }
        provenance = _failed_provenance(
            source_kind=kind,
            mode=mode,
            chain_id=None,
            detection=detection,
            branch=f"{kind}_assembly",
            failure_reason=f"{kind} assembly failed: {error}",
            underlying={
                "artifact_status": "PARTIAL",
                "requested_artifact_status": "NOT_MATERIALIZABLE",
                "strict_assembly_error": error,
            },
            error=error,
        )
        provenance["source"] = _source_preview(text)
        return _build(
            status=STATUS_SUCCESS,
            quality=QUALITY_PARTIAL,
            result_origin=ORIGIN_REPRESENTATION_ASSEMBLY,
            source_kind=kind,
            mode=mode,
            result={
                "route": f"{kind}_symbolic_candidate",
                "artifact_status": "PARTIAL",
                "chemical_rigor": "C1:H",
                "requested_artifact_status": "NOT_MATERIALIZABLE",
                "reason_codes": reason_codes,
                "unresolved_monomers": list(dict.fromkeys(
                    unresolved_symbols
                )),
                "exact_v1": exact,
            },
            smiles=None,
            graph=symbolic_graph,
            ambiguous=False,
            warning_codes=[_WARNING_ASSEMBLY_FAILED],
            alternatives=[{
                "kind": kind,
                "value": text,
                "claim_boundary": (
                    "symbolic monomer/topology candidate only"
                ),
            }],
            structure_profile=None,
            provenance=provenance,
            strict_result=None,
        )
    try:
        chains, edges = _parse_map_graph(kind, text)
    except Exception:
        chains, edges = [], []
    profile = (
        _profile_from_residue_graph(chains, edges)
        if chains
        else _profile_from_smiles(smiles)
    )
    graph = _residue_graph(chains, edges) if chains else None
    provenance = {
        "detection": detection,
        "dispatch": {"mode": mode, "branch": f"{kind}_assembly", "chain_id": None},
        "underlying": {"route": f"{kind}_assembly", "source_kind": kind},
    }
    return _build(
        status=STATUS_SUCCESS,
        quality=QUALITY_HIGH,
        result_origin=ORIGIN_REPRESENTATION_ASSEMBLY,
        source_kind=kind,
        mode=mode,
        result={"route": f"{kind}_assembly", "source_kind": kind, "text": text},
        smiles=smiles,
        graph=graph,
        ambiguous=False,
        warning_codes=[],
        alternatives=[],
        structure_profile=profile,
        provenance=provenance,
        strict_result=None,
    )


def _best_effort_notation_attempts(
    text: str,
    mode: str,
    detection: dict[str, Any],
    excluded: set[str],
) -> UnifiedReconstructionResult:
    attempts: list[dict[str, Any]] = []
    for kind in _NOTATION_ORDER:
        if kind in excluded:
            continue
        candidate = _representation_result(
            kind=kind, text=text, mode=mode, detection=detection
        )
        attempts.append(
            {
                "source_kind": kind,
                "status": candidate.status,
                "error": candidate.provenance.get("failure_reason"),
            }
        )
        if candidate.status == STATUS_SUCCESS:
            codes = list(
                dict.fromkeys(
                    [
                        _WARNING_AMBIGUOUS_SOURCE,
                        _WARNING_BEST_EFFORT_FALLBACK,
                        *candidate.warning_codes,
                    ]
                )
            )
            provenance = dict(candidate.provenance)
            provenance["notation_attempts"] = attempts
            return UnifiedReconstructionResult(
                status=candidate.status,
                quality=candidate.quality,
                result_origin=candidate.result_origin,
                source_kind=candidate.source_kind,
                mode=candidate.mode,
                result=candidate.result,
                smiles=candidate.smiles,
                inchi=candidate.inchi,
                inchikey=candidate.inchikey,
                graph=candidate.graph,
                ambiguous=True,
                warnings=_warning_lines(codes),
                warning_codes=codes,
                alternatives=attempts,
                structure_profile=candidate.structure_profile,
                provenance=provenance,
                strict_result=candidate.strict_result,
                strict_status=candidate.strict_status,
                candidate_smiles=candidate.candidate_smiles,
                candidate_graph=candidate.candidate_graph,
                chemistry_candidates=list(candidate.chemistry_candidates),
                bond_order_inference=dict(candidate.bond_order_inference),
                candidate_rigor=candidate.candidate_rigor,
                artifact_status=candidate.artifact_status,
                qualification_status=candidate.qualification_status,
                chemical_rigor=candidate.chemical_rigor,
                coordinate_evidence=candidate.coordinate_evidence,
            )
    return _typed_failed(
        source=text,
        chain_id=None,
        mode=mode,
        source_kind=(
            str(detection.get("source_kind"))
            if detection.get("source_kind") in SOURCE_KINDS
            else "unknown"
        ),
        detection=detection,
        branch="best_effort_notation",
        failure_reason="no compatible notation could be assembled",
        warning_codes=[
            _WARNING_SOURCE_UNDETERMINED,
            _WARNING_BEST_EFFORT_FALLBACK,
        ],
        alternatives=attempts,
        error="; ".join(
            f"{attempt['source_kind']}: {attempt['error']}"
            for attempt in attempts
        ),
    )


def _reconstruct_representation(
    kind: str,
    text: str,
    mode: str,
    detection: dict[str, Any],
) -> UnifiedReconstructionResult:
    result = _representation_result(
        kind=kind, text=text, mode=mode, detection=detection
    )
    if (
        mode != "best_effort"
        or result.status == STATUS_SUCCESS
        or detection.get("explicit_prefix") is not None
    ):
        return result
    return _best_effort_notation_attempts(
        text, mode, detection, excluded={kind}
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def reconstruct_structure(
    source: Any,
    chain_id: Any = None,
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
) -> UnifiedReconstructionResult:
    """Unified structure reconstruction across coordinates and notations.

    ``chain_id``: explicit string selects one chain; a list/tuple selects the
    PDB multi-chain route.  ``None`` auto-selects the unique peptide chain
    (or requires explicit chains when multiple disconnected groups exist).
    ``mode`` is one of ``strict|auto|best_effort`` (the CLI spells
    ``best-effort``).

    ``minimum_macrocycle_ring_size`` and ``require_empty_persistent_overlay``
    are forwarded to strict V6 and the result-first coordinate ladder. They
    are accepted for all source kinds so callers can use one stable API;
    notation-only routes do not consume these coordinate-specific settings.

    ``radius_multiplier`` and ``distance_ceiling`` tune geometric covalent-
    radius cyclization detection; both ``None`` (default) select the standard
    behaviour. They are accepted for all source kinds; notation-only routes
    do not consume these coordinate-specific settings.

    ``allow_linear_topology`` (default ``False``) is forwarded to the strict V6
    coordinate audit: when set, a selected chain with neither explicit nor
    geometrically-inferred cyclization evidence is accepted as
    ``topology_class="linear"`` instead of being rejected for lacking closure
    evidence. It is honored on the ``strict`` coordinate path this round; the
    result-first ladder and legacy pipeline do not consume it yet.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            (
                monomer_context
                if monomer_context is not None
                else {"include_persistent_user": True}
            ),
            required_symbols=monomer_symbol_hints(source),
        ) as resolution_ledger:
            resolved = reconstruct_structure(
                source,
                chain_id=chain_id,
                mode=mode,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
        if monomer_context is not None:
            resolved.provenance.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    normalized_mode = _normalize_mode(mode)
    if normalized_mode not in MODES:
        return _typed_failed(
            source=source,
            chain_id=chain_id,
            mode=str(mode),
            source_kind="invalid",
            detection={"source_kind": "invalid"},
            branch="mode_validation",
            failure_reason=(
                f"invalid mode: {mode!r}; expected strict, auto, or best_effort"
            ),
            warning_codes=[_WARNING_INVALID_MODE],
        )
    try:
        kind, payload, classification_error = _classify_source(source)
    except Exception as exc:
        return _typed_failed(
            source=source,
            chain_id=chain_id,
            mode=normalized_mode,
            source_kind="invalid",
            detection={"source_kind": "invalid"},
            branch="source_classification",
            failure_reason=(
                f"source classification failed: {type(exc).__name__}: {exc}"
            ),
            warning_codes=[_WARNING_INTERNAL_ERROR],
            error=f"{type(exc).__name__}: {exc}",
        )
    explicit_prefix = None
    if isinstance(source, str):
        prefix_match = _EXPLICIT_PREFIX_RE.match(source.strip())
        if prefix_match:
            explicit_prefix = prefix_match.group(1).lower()
    detection = {
        "source_kind": kind,
        "payload_preview": _source_preview(payload),
        "classification_error": classification_error,
        "explicit_prefix": explicit_prefix,
    }
    try:
        if kind in _COORDINATE_KINDS:
            return _reconstruct_coordinate(
                payload,
                kind,
                classification_error,
                chain_id,
                normalized_mode,
                detection,
                minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=require_empty_persistent_overlay,
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
        if kind in ("sequence", "helm", "map", "biln"):
            if classification_error:
                return _typed_failed(
                    source=source,
                    chain_id=chain_id,
                    mode=normalized_mode,
                    source_kind=kind,
                    detection=detection,
                    branch="source_classification",
                    failure_reason=classification_error,
                    warning_codes=[_WARNING_PREFIX_KIND_MISMATCH],
                    error=classification_error,
                )
            return _reconstruct_representation(
                kind, payload, normalized_mode, detection
            )
        if kind == "invalid":
            return _typed_failed(
                source=source,
                chain_id=chain_id,
                mode=normalized_mode,
                source_kind="invalid",
                detection=detection,
                branch="source_classification",
                failure_reason=classification_error or "input is invalid",
                warning_codes=[_WARNING_SOURCE_UNDETERMINED],
                error=classification_error,
            )
        # unknown / undetermined text
        if normalized_mode == "best_effort":
            return _best_effort_notation_attempts(
                payload, normalized_mode, detection, excluded=set()
            )
        return _typed_failed(
            source=source,
            chain_id=chain_id,
            mode=normalized_mode,
            source_kind="unknown",
            detection=detection,
            branch="source_classification",
            failure_reason=(
                "input could not be deterministically classified as a "
                "coordinate path or a supported notation"
            ),
            warning_codes=[_WARNING_SOURCE_UNDETERMINED],
        )
    except Exception as exc:
        return _typed_failed(
            source=source,
            chain_id=chain_id,
            mode=normalized_mode,
            source_kind=kind,
            detection=detection,
            branch="dispatch",
            failure_reason=(
                f"unexpected orchestration failure: {type(exc).__name__}: {exc}"
            ),
            warning_codes=[_WARNING_INTERNAL_ERROR],
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "MODES",
    "SOURCE_KINDS",
    "STATUS_FAILED",
    "STATUS_SUCCESS",
    "UnifiedReconstructionResult",
    "detect_source_kind",
    "reconstruct_structure",
]
