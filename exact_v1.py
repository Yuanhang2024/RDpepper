"""Exact, lossless cyclic-peptide monomer-port graph representation.

``exact_v1`` is the chemistry-facing intermediate representation.  It is not
a model token sequence.  MAP, HELM, BILN, qualified V6 reconstruction, legacy
V5, and edge_v1 may enter through audited adapters; edge_v1 remains a bounded,
lossy model projection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

from .core.cyclic_peptide_graph import (
    ABSTAIN,
    EXACT,
    EXACT_V1_SCHEMA,
    BondEndpoint,
    CyclicPeptideGraph,
    CyclicPeptideGraphError,
    MonomerNode,
    PortBond,
    TerminalCap,
    canonical_exact_v1_bytes,
    canonical_json_bytes,
)
from .core.monomer_resolution import needs_monomer_resolution_scope
from .paths import _map_utils


DEFAULT_EDGE_MAX_POSITION = 32
DEFAULT_EDGE_MAX_RINGS = 3
EDGE_BOND_TYPES = frozenset({
    "HT", "SS", "SC", "EST", "HSC", "THIO", "ALK",
})
CAP_SYMBOLS = frozenset({"ac", "nme", "nh2"})

_STANDARD_ONE_TO_EDGE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
_EDGE_STANDARD_TO_ONE = {
    value: key for key, value in _STANDARD_ONE_TO_EDGE.items()
}
_THREE_TITLE_TO_ONE = {
    value.title(): key for key, value in _STANDARD_ONE_TO_EDGE.items()
}
_LEGACY_TYPE_RE = re.compile(
    r"^<cyc([1-9][0-9]*)_(HT|SS|SC|EST|HSC|THIO|ALK)>$"
)
_LEGACY_POS_RE = re.compile(
    r"^<cyc([1-9][0-9]*)_([1-9][0-9]*)_([1-9][0-9]*)>$"
)
_EDGE_BOND_RE = re.compile(
    r"^<BOND_(HT|SS|SC|EST|HSC|THIO|ALK)>$"
)
_EDGE_SRC_RE = re.compile(r"^<SRC_POS_([1-9][0-9]*)>$")
_EDGE_DST_RE = re.compile(r"^<DST_POS_([1-9][0-9]*)>$")


class ExactV1Error(ValueError):
    """An exact_v1 operation cannot be completed without guessing."""


class ExactV1Abstained(ExactV1Error):
    """The source was intentionally represented as ABSTAIN."""


@dataclass(frozen=True)
class MonomerDefinition:
    stable_id: str
    symbol: str
    stereo: str
    ports: Mapping[str, str]
    source: str


@dataclass(frozen=True)
class EdgeOperation:
    ring: int
    bond_type: str
    src: int | None
    dst: int | None


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _row_digest(row: Mapping[str, Any]) -> str:
    selected = {
        key: str(row.get(key, "")).strip()
        for key in (
            "symbol",
            "source",
            "version",
            "smiles_canonical",
            "smiles_original",
            "CXSMILES",
            "R1",
            "R2",
            "R3",
        )
    }
    return hashlib.sha256(canonical_json_bytes(selected)).hexdigest()


@lru_cache(maxsize=1)
def _full_unified_rows() -> dict[str, tuple[dict[str, str], ...]]:
    path = _package_root() / "unified_monomer_library.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in csv.DictReader(handle):
            symbol = str(row.get("symbol", "")).strip()
            if symbol:
                grouped.setdefault(symbol, []).append(dict(row))
    return {
        symbol: tuple(rows) for symbol, rows in grouped.items()
    }


def _port_atoms(
    symbol: str,
    row: Mapping[str, Any],
) -> dict[str, str]:
    annotated = _map_utils.monomers2smi_dict.get(symbol)
    if not annotated:
        return {}
    cxsmiles = str(row.get("CXSMILES", ""))
    explicit_ports = {
        port for port in ("R1", "R2", "R3")
        if f"_{port}" in cxsmiles
    }
    normalized = re.sub(r":_R([123])", r":\1", str(annotated))
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        return {}
    result = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        map_number = atom.GetAtomMapNum()
        if map_number not in {1, 2, 3} or atom.GetDegree() != 1:
            continue
        port = f"R{map_number}"
        if port in explicit_ports:
            result[port] = atom.GetNeighbors()[0].GetSymbol()
    return result


def _ported_smiles_identity(symbol: str) -> str | None:
    annotated = _map_utils.monomers2smi_dict.get(symbol)
    if not annotated:
        return None
    normalized = re.sub(r":_R([123])", r":\1", str(annotated))
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        return None
    return Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )


def _cx_ported_identity(row: Mapping[str, Any]) -> str | None:
    cxsmiles = str(row.get("CXSMILES", "")).strip()
    if not cxsmiles:
        return None
    try:
        annotated = _map_utils.get_smi_from_cxsmiles(cxsmiles)
    except Exception:
        return None
    normalized = re.sub(r":_R([123])", r":\1", str(annotated))
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        return None
    return Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )


def _stereo_identity(row: Mapping[str, Any], symbol: str) -> str:
    smiles = next(
        (
            str(row.get(name, "")).strip()
            for name in (
                "smiles_canonical",
                "smiles_original",
                "replaced_SMILES",
            )
            if str(row.get(name, "")).strip()
        ),
        "",
    )
    if not smiles:
        runtime = str(
            _map_utils.monomers2smi_dict.get(symbol, "")
        )
        if not runtime:
            raise ExactV1Error("MONOMER_STEREO_UNRESOLVED")
        runtime = re.sub(r":_R([123])", r":\1", runtime)
        molecule = Chem.MolFromSmiles(runtime)
    else:
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ExactV1Error("MONOMER_STEREO_UNRESOLVED")
    centers = Chem.FindMolChiralCenters(
        molecule,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    if any(label == "?" for _index, label in centers):
        raise ExactV1Error("MONOMER_STEREO_UNRESOLVED")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    nonstereo = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=False
    )
    if canonical == nonstereo:
        return "ACHIRAL"
    return "ISOMERIC_SMILES_SHA256:" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


class MonomerRegistry:
    """Read-only exact identity view over the active monomer resources."""

    def __init__(self):
        full = _full_unified_rows()
        self._full = full
        self._active = {
            str(symbol): dict(row)
            for symbol, row in _map_utils._unified_by_symbol.items()
        }
        self._active.update({
            str(symbol): dict(row)
            for symbol, row in _map_utils._active_user_rows.items()
        })
        symbols = set(_map_utils.monomers2smi_dict)
        self._resolvable_symbols = symbols.intersection(
            set(full) | set(self._active)
        )
        self._definitions: dict[str, MonomerDefinition] = {}
        self._exact_aliases: dict[str, str] = {}

        for row_symbol, rows in full.items():
            for row in rows:
                alias = str(
                    row.get("best_cycpep_match_symbol", "")
                ).strip()
                try:
                    exact_alias = float(
                        row.get("tanimoto_similarity", "nan")
                    ) == 1.0
                except (TypeError, ValueError):
                    exact_alias = False
                if (
                    exact_alias
                    and alias
                    and row_symbol in self._resolvable_symbols
                ):
                    alias_identity = _ported_smiles_identity(alias)
                    row_identity = _ported_smiles_identity(row_symbol)
                    if (
                        alias_identity is not None
                        and alias_identity == row_identity
                    ):
                        self._exact_aliases[alias] = row_symbol
        for alias, target in _map_utils._MONOMER_ALIASES.items():
            if (
                target in self._resolvable_symbols
                and _ported_smiles_identity(alias)
                == _ported_smiles_identity(target)
            ):
                self._exact_aliases[alias] = target

    def _definition(self, symbol: str) -> MonomerDefinition:
        cached = self._definitions.get(symbol)
        if cached is not None:
            return cached
        if symbol not in self._resolvable_symbols:
            raise ExactV1Error(f"MONOMER_ID_UNRESOLVED:{symbol}")
        active_identity = _ported_smiles_identity(symbol)
        candidates = [
            row for row in self._full.get(symbol, ())
            if (
                active_identity is not None
                and _cx_ported_identity(row) == active_identity
            )
        ]
        candidates.sort(key=lambda row: (
            int(str(row.get("monomer_id", "")).strip())
            if str(row.get("monomer_id", "")).strip().isdigit()
            else 2**63,
            _row_digest(row),
        ))
        base = dict(candidates[0]) if candidates else {}
        base.update(self._active.get(symbol, {}))
        base.setdefault("symbol", symbol)
        base.setdefault("source", "active_library")
        ports = _port_atoms(symbol, base)
        if not ports:
            raise ExactV1Error(
                f"MONOMER_EXPLICIT_PORT_IDENTITY_UNRESOLVED:{symbol}"
            )
        numeric = str(base.get("monomer_id", "")).strip()
        identity_digest = _row_digest(base)[:16]
        stable_id = (
            f"unified:{int(numeric)}:{identity_digest}"
            if numeric.lstrip("+-").isdigit()
            else (
                f"library:{str(base.get('source') or 'active')}:"
                f"{symbol}:{identity_digest}"
            )
        )
        definition = MonomerDefinition(
            stable_id=stable_id,
            symbol=symbol,
            stereo=_stereo_identity(base, symbol),
            ports=dict(sorted(ports.items())),
            source=str(base.get("source") or "active_library"),
        )
        self._definitions[symbol] = definition
        return definition

    def resolve(self, symbol: str) -> MonomerDefinition:
        raw = str(symbol).strip()
        canonical = self._exact_aliases.get(raw)
        if canonical is None:
            canonical = (
                raw if raw in self._resolvable_symbols else None
            )
        if canonical is None:
            pdb_alias = _map_utils.resolve_pdb_alias(raw)
            canonical = (
                str(pdb_alias)
                if pdb_alias
                else None
            )
        if canonical is None:
            raise ExactV1Error(f"MONOMER_ID_UNRESOLVED:{raw}")
        return self._definition(canonical)

    def validate_identity(
        self,
        *,
        symbol: str,
        monomer_id: str,
        stereo: str,
    ) -> None:
        definition = self.resolve(symbol)
        if definition.stable_id != monomer_id:
            raise ExactV1Error("MONOMER_ID_REGISTRY_MISMATCH")
        if definition.stereo != stereo:
            raise ExactV1Error("MONOMER_STEREO_REGISTRY_MISMATCH")


_REGISTRY_CACHE: MonomerRegistry | None = None


def _registry() -> MonomerRegistry:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = MonomerRegistry()
    return _REGISTRY_CACHE


def reset_exact_v1_registry_cache() -> None:
    """Clear the read-only resolver cache after an explicit registry change."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None


def _abstain(
    reasons: Iterable[str],
    *,
    source_kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return CyclicPeptideGraph.abstain(
        reasons,
        source_kind=source_kind,
        metadata=metadata,
    ).canonicalize()


def _chain_id(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"CHAIN_{index + 1}"


def _classify_bond(
    left_port: str,
    left_atom: str | None,
    right_port: str,
    right_atom: str | None,
    *,
    same_chain: bool,
    left_position: int,
    right_position: int,
    chain_length: int,
) -> str:
    ports = {left_port, right_port}
    atoms = {str(left_atom or ""), str(right_atom or "")}
    if (
        same_chain
        and ports == {"R1", "R2"}
        and {left_position, right_position} == {1, chain_length}
    ):
        return "HT"
    if ports == {"R1", "R3"}:
        return "HSC"
    if ports == {"R2", "R3"}:
        return "SIDECHAIN_TO_TAIL"
    if atoms == {"S"}:
        return "SS"
    if atoms == {"C", "N"}:
        return "ISOPEPTIDE"
    if atoms == {"C", "O"}:
        return "ESTER"
    if atoms == {"C", "S"}:
        return "THIOETHER"
    if atoms == {"C"}:
        return "ALKYL"
    if left_port == "R3" and right_port == "R3":
        return "SIDECHAIN"
    return "CROSSLINK"


def _normalize_declared_bond_type(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return {
        "PEPTIDE": "HT",
        "HEAD_TO_TAIL": "HT",
        "HT": "HT",
        "DISULFIDE": "SS",
        "SS": "SS",
        "ISOPEPTIDE": "ISOPEPTIDE",
        "ESTER": "ESTER",
        "THIOETHER": "THIOETHER",
        "HSC": "HSC",
        "SIDECHAIN": "SIDECHAIN",
        "SC": "SIDECHAIN",
        "SIDECHAIN_TO_TAIL": "SIDECHAIN_TO_TAIL",
        "ALKYL": "ALKYL",
        "ALK": "ALKYL",
        "CROSSLINK": "CROSSLINK",
        "UNKNOWN": None,
        "": None,
    }.get(normalized)


def _build_from_components(
    chains: Sequence[Sequence[str]],
    edges: Sequence[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ],
    *,
    source_kind: str,
    edge_details: Sequence[Mapping[str, Any] | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = _registry()
    reasons: list[str] = []
    nodes: list[MonomerNode] = []
    caps: list[TerminalCap] = []
    parser_position_to_node: dict[tuple[int, int], MonomerNode] = {}
    definitions: dict[int, MonomerDefinition] = {}
    node_id = 0
    peptide_nodes_by_chain: dict[int, list[MonomerNode]] = {}
    pending_caps: list[tuple[int, str, MonomerDefinition]] = []

    for chain_index, raw_chain in enumerate(chains):
        chain = list(raw_chain)
        if not chain:
            reasons.append("EMPTY_CHAIN")
            continue
        leading_cap = bool(chain and chain[0] == "ac")
        trailing_cap = bool(chain and chain[-1] in {"nme", "nh2"})
        for position, symbol in enumerate(chain, 1):
            if symbol in CAP_SYMBOLS:
                if not (
                    (symbol == "ac" and position == 1)
                    or (
                        symbol in {"nme", "nh2"}
                        and position == len(chain)
                    )
                ):
                    reasons.append("CAP_NOT_AT_CHAIN_TERMINUS")
                try:
                    pending_caps.append((
                        chain_index,
                        "N" if symbol == "ac" else "C",
                        registry.resolve(symbol),
                    ))
                except ExactV1Error as exc:
                    reasons.append(str(exc))
                continue
            try:
                definition = registry.resolve(symbol)
            except ExactV1Error as exc:
                reasons.append(str(exc))
                continue
            node_id += 1
            semantic_position = (
                position - int(leading_cap)
            )
            node = MonomerNode(
                node_id=node_id,
                monomer_id=definition.stable_id,
                monomer_symbol=definition.symbol,
                stereo=definition.stereo,
                chain_id=_chain_id(chain_index),
                original_position=semantic_position,
            )
            nodes.append(node)
            definitions[node.node_id] = definition
            parser_position_to_node[(chain_index, position)] = node
            peptide_nodes_by_chain.setdefault(chain_index, []).append(node)

        expected_count = (
            len(chain) - int(leading_cap) - int(trailing_cap)
        )
        if len(peptide_nodes_by_chain.get(chain_index, [])) != expected_count:
            reasons.append("CHAIN_MONOMER_RESOLUTION_INCOMPLETE")

    if reasons:
        return _abstain(reasons, source_kind=source_kind)

    bonds: list[PortBond] = []
    for chain_nodes in peptide_nodes_by_chain.values():
        for left, right in zip(chain_nodes, chain_nodes[1:]):
            left_definition = definitions[left.node_id]
            right_definition = definitions[right.node_id]
            if (
                "R2" not in left_definition.ports
                or "R1" not in right_definition.ports
            ):
                reasons.append("BACKBONE_PORT_UNAVAILABLE")
                continue
            bonds.append(PortBond(
                bond_type="PEPTIDE",
                src=BondEndpoint(
                    left.node_id,
                    "R2",
                    left_definition.ports.get("R2") or None,
                ),
                dst=BondEndpoint(
                    right.node_id,
                    "R1",
                    right_definition.ports.get("R1") or None,
                ),
            ))

    details = list(edge_details or [None] * len(edges))
    if len(details) != len(edges):
        return _abstain(
            ["EDGE_DETAIL_CARDINALITY_MISMATCH"],
            source_kind=source_kind,
        )
    for edge_index, (left_raw, right_raw) in enumerate(edges):
        left = parser_position_to_node.get((left_raw[0], left_raw[1]))
        right = parser_position_to_node.get((right_raw[0], right_raw[1]))
        if left is None or right is None:
            reasons.append("CONNECTION_ENDPOINT_NOT_A_PEPTIDE_MONOMER")
            continue
        left_port = f"R{int(left_raw[2])}"
        right_port = f"R{int(right_raw[2])}"
        left_definition = definitions[left.node_id]
        right_definition = definitions[right.node_id]
        if (
            left_port not in left_definition.ports
            or right_port not in right_definition.ports
        ):
            reasons.append("CONNECTION_PORT_UNAVAILABLE")
            continue
        left_atom = left_definition.ports.get(left_port) or None
        right_atom = right_definition.ports.get(right_port) or None
        same_chain = left.chain_id == right.chain_id
        chain_length = len(
            peptide_nodes_by_chain[left_raw[0]]
        ) if same_chain else -1
        inferred_type = _classify_bond(
            left_port,
            left_atom,
            right_port,
            right_atom,
            same_chain=same_chain,
            left_position=left.original_position,
            right_position=right.original_position,
            chain_length=chain_length,
        )
        detail = details[edge_index] or {}
        declared = _normalize_declared_bond_type(
            detail.get("bond_type")
        )
        if declared is not None and declared != inferred_type:
            compatible = {
                ("SIDECHAIN", "ISOPEPTIDE"),
                ("SIDECHAIN", "ESTER"),
                ("SIDECHAIN", "THIOETHER"),
                ("SIDECHAIN", "ALKYL"),
            }
            if (declared, inferred_type) not in compatible:
                reasons.append("BOND_TYPE_PORT_CHEMISTRY_CONFLICT")
                continue
        bond_type = inferred_type
        bonds.append(PortBond(
            bond_type=bond_type,
            src=BondEndpoint(left.node_id, left_port, left_atom),
            dst=BondEndpoint(right.node_id, right_port, right_atom),
        ))

    for chain_index, terminus, definition in pending_caps:
        chain_nodes = peptide_nodes_by_chain.get(chain_index, [])
        if not chain_nodes:
            reasons.append("CAP_TARGET_UNAVAILABLE")
            continue
        target = chain_nodes[0] if terminus == "N" else chain_nodes[-1]
        cap_port = "R2" if terminus == "N" else "R1"
        target_port = "R1" if terminus == "N" else "R2"
        target_definition = definitions[target.node_id]
        if (
            cap_port not in definition.ports
            or target_port not in target_definition.ports
        ):
            reasons.append("CAP_CONNECTION_PORT_UNAVAILABLE")
            continue
        caps.append(TerminalCap(
            cap_id=f"cap:{definition.stable_id}",
            monomer_id=definition.stable_id,
            monomer_symbol=definition.symbol,
            chain_id=_chain_id(chain_index),
            terminus=terminus,
            port=cap_port,
            atom=definition.ports.get(cap_port) or None,
            target_node_id=target.node_id,
            target_port=target_port,
            target_atom=(
                target_definition.ports.get(target_port) or None
            ),
        ))

    if reasons:
        return _abstain(reasons, source_kind=source_kind)
    return CyclicPeptideGraph(
        monomers=tuple(nodes),
        bonds=tuple(bonds),
        caps=tuple(caps),
        source_kind=source_kind,
        metadata=dict(metadata or {}),
    ).canonicalize()


def _parse_notation(kind: str, payload: str):
    normalized = kind.strip().lower()
    if normalized == "map":
        return _map_utils._parse_map_strict(payload)
    if normalized == "helm":
        return _map_utils._parse_helm_strict(payload)
    if normalized == "biln":
        return _map_utils._parse_biln_strict(payload)
    raise ExactV1Error(f"UNSUPPORTED_EXACT_V1_SOURCE:{kind}")


def _legacy_cap_offset_map(payload: str) -> str | None:
    """Normalize the historical MAP convention that counted a leading cap."""
    if "{nt:" not in payload:
        return None
    edge_pattern = re.compile(
        r"\{cyc:([1-9][0-9]*):R([1-3])-"
        r"([1-9][0-9]*):R([1-3])\}"
    )
    matches = list(edge_pattern.finditer(payload))
    if not matches:
        return None
    working = re.sub(r"\{(?:nt|ct):[^{}]+\}", "", payload)
    working = edge_pattern.sub("", working)
    working = working.replace("{cyc:N-C}", "")
    segments = working.split("{br}")
    try:
        residue_count = sum(
            len(_map_utils._map_tokenize_strict(segment))
            for segment in segments
        )
    except Exception:
        return None
    endpoints = [
        int(value)
        for match in matches
        for value in (match.group(1), match.group(3))
    ]
    if (
        not endpoints
        or min(endpoints) < 2
        or max(endpoints) > residue_count + 1
    ):
        return None

    def shift(match: re.Match) -> str:
        return (
            f"{{cyc:{int(match.group(1)) - 1}:R{match.group(2)}-"
            f"{int(match.group(3)) - 1}:R{match.group(4)}}}"
        )

    return edge_pattern.sub(shift, payload)


def notation_to_exact_v1(
    kind: str,
    payload: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_symbol_hints,
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(payload, kind=kind),
        ):
            return notation_to_exact_v1(kind, payload)
    """Convert one strict notation to canonical exact_v1 or ABSTAIN."""
    normalized = str(kind).strip().lower()
    try:
        chains, edges = _parse_notation(normalized, payload)
        return _build_from_components(
            chains, edges, source_kind=normalized
        )
    except Exception as exc:
        if (
            normalized == "map"
            and "endpoint is out of range" in str(exc)
        ):
            migrated = _legacy_cap_offset_map(payload)
            if migrated is not None:
                try:
                    chains, edges = _parse_notation(
                        normalized, migrated
                    )
                    return _build_from_components(
                        chains,
                        edges,
                        source_kind="map_legacy_cap_offset",
                        metadata={
                            "normalization_codes": [
                                "LEGACY_CAP_INCLUSIVE_POSITION_NORMALIZED"
                            ]
                        },
                    )
                except Exception:
                    pass
        return _abstain(
            [f"NOTATION_PARSE_FAILED:{type(exc).__name__}:{exc}"],
            source_kind=normalized,
        )


def map_to_exact_v1(
    payload: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return notation_to_exact_v1(
        "map", payload, monomer_context=monomer_context
    )


def helm_to_exact_v1(
    payload: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return notation_to_exact_v1(
        "helm", payload, monomer_context=monomer_context
    )


def biln_to_exact_v1(
    payload: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return notation_to_exact_v1(
        "biln", payload, monomer_context=monomer_context
    )


def _coerce_document(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
) -> dict[str, Any]:
    if isinstance(value, CyclicPeptideGraph):
        document = value.canonicalize()
    elif isinstance(value, Mapping):
        document = dict(value)
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ExactV1Error("exact_v1 input is not JSON") from exc
        if not isinstance(parsed, dict):
            raise ExactV1Error("exact_v1 JSON must be an object")
        document = parsed
    else:
        raise ExactV1Error("unsupported exact_v1 input type")
    _validate_document_shape(document)
    if document.get("schema") != EXACT_V1_SCHEMA:
        raise ExactV1Error("unsupported exact_v1 schema")
    if document.get("exactness_status") == ABSTAIN:
        raise ExactV1Abstained(
            ",".join(document.get("reason_codes") or ["ABSTAIN"])
        )
    graph = CyclicPeptideGraph.from_exact_document(document)
    registry = _registry()
    for row in document.get("monomers", []):
        registry.validate_identity(
            symbol=str(row["monomer_symbol"]),
            monomer_id=str(row["monomer_id"]),
            stereo=str(row["stereo"]),
        )
    for cap in document.get("caps", []):
        registry.validate_identity(
            symbol=str(cap["monomer_symbol"]),
            monomer_id=str(cap["monomer_id"]),
            stereo=registry.resolve(
                str(cap["monomer_symbol"])
            ).stereo,
        )
    _validate_registry_bound_graph(document, registry)
    canonical = graph.canonicalize()
    if (
        canonical_exact_v1_bytes(canonical)
        != canonical_exact_v1_bytes(document)
    ):
        raise ExactV1Error("exact_v1 document is not canonical")
    return document


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing:
        raise ExactV1Error(
            f"{label} lacks required fields: {missing}"
        )
    if extra:
        raise ExactV1Error(
            f"{label} has unsupported fields: {extra}"
        )


def _validate_document_shape(document: Mapping[str, Any]) -> None:
    top = {
        "schema",
        "monomers",
        "bonds",
        "caps",
        "chain_breaks",
        "canonical_order",
        "original_to_canonical",
        "canonical_to_original",
        "graph_sha256",
        "exactness_status",
        "reason_codes",
        "normalization_codes",
        "projection_trace",
        "source_kind",
    }
    _require_keys(
        document,
        required=top,
        allowed=top,
        label="exact_v1 document",
    )
    for name in (
        "monomers",
        "bonds",
        "caps",
        "chain_breaks",
        "canonical_order",
        "reason_codes",
        "normalization_codes",
    ):
        if not isinstance(document[name], list):
            raise ExactV1Error(f"exact_v1 {name} must be an array")
    if not isinstance(document["original_to_canonical"], Mapping):
        raise ExactV1Error(
            "exact_v1 original_to_canonical must be an object"
        )
    if not isinstance(document["canonical_to_original"], Mapping):
        raise ExactV1Error(
            "exact_v1 canonical_to_original must be an object"
        )
    trace = document["projection_trace"]
    if trace is not None:
        if not isinstance(trace, Mapping):
            raise ExactV1Error(
                "exact_v1 projection_trace must be an object or null"
            )
        trace_fields = {"source_kind", "operations"}
        _require_keys(
            trace,
            required=trace_fields,
            allowed=trace_fields,
            label="exact_v1 projection_trace",
        )
        if not isinstance(trace["operations"], list):
            raise ExactV1Error(
                "exact_v1 projection_trace operations must be an array"
            )
        operation_fields = {"bond_type", "src", "dst"}
        for index, operation in enumerate(trace["operations"]):
            if not isinstance(operation, Mapping):
                raise ExactV1Error(
                    "exact_v1 projection operation must be an object"
                )
            _require_keys(
                operation,
                required=operation_fields,
                allowed=operation_fields,
                label=f"exact_v1 projection operation {index}",
            )
    monomer_fields = {
        "node_id",
        "monomer_id",
        "monomer_symbol",
        "stereo",
        "modifications",
        "chain_id",
        "canonical_position",
        "original_chain_id",
        "original_position",
    }
    for index, row in enumerate(document["monomers"]):
        if not isinstance(row, Mapping):
            raise ExactV1Error(
                f"exact_v1 monomer {index} must be an object"
            )
        _require_keys(
            row,
            required=monomer_fields,
            allowed=monomer_fields,
            label=f"exact_v1 monomer {index}",
        )
    endpoint_fields = {"node_id", "port", "atom"}
    bond_fields = {"bond_type", "src", "dst", "bond_order"}
    for index, row in enumerate(document["bonds"]):
        if not isinstance(row, Mapping):
            raise ExactV1Error(
                f"exact_v1 bond {index} must be an object"
            )
        _require_keys(
            row,
            required=bond_fields,
            allowed=bond_fields,
            label=f"exact_v1 bond {index}",
        )
        for side in ("src", "dst"):
            endpoint = row[side]
            if not isinstance(endpoint, Mapping):
                raise ExactV1Error(
                    f"exact_v1 bond {index} {side} must be an object"
                )
            _require_keys(
                endpoint,
                required=endpoint_fields,
                allowed=endpoint_fields,
                label=f"exact_v1 bond {index} {side}",
            )
    cap_fields = {
        "cap_id",
        "monomer_id",
        "monomer_symbol",
        "chain_id",
        "terminus",
        "port",
        "atom",
        "target_node_id",
        "target_port",
        "target_atom",
    }
    for index, row in enumerate(document["caps"]):
        if not isinstance(row, Mapping):
            raise ExactV1Error(
                f"exact_v1 cap {index} must be an object"
            )
        _require_keys(
            row,
            required=cap_fields,
            allowed=cap_fields,
            label=f"exact_v1 cap {index}",
        )
    break_fields = {"left_chain_id", "right_chain_id"}
    for index, row in enumerate(document["chain_breaks"]):
        if not isinstance(row, Mapping):
            raise ExactV1Error(
                f"exact_v1 chain break {index} must be an object"
            )
        _require_keys(
            row,
            required=break_fields,
            allowed=break_fields,
            label=f"exact_v1 chain break {index}",
        )


def _validate_registry_bound_graph(
    document: Mapping[str, Any],
    registry: MonomerRegistry,
) -> None:
    rows = {
        int(row["node_id"]): row for row in document["monomers"]
    }
    definitions = {
        node_id: registry.resolve(str(row["monomer_symbol"]))
        for node_id, row in rows.items()
    }
    chain_lengths = Counter(
        str(row["chain_id"]) for row in document["monomers"]
    )
    for index, bond in enumerate(document["bonds"]):
        endpoints = []
        for side in ("src", "dst"):
            endpoint = bond[side]
            node_id = int(endpoint["node_id"])
            definition = definitions[node_id]
            port = str(endpoint["port"])
            expected_atom = definition.ports.get(port)
            if expected_atom is None:
                raise ExactV1Error(
                    f"bond {index} uses unavailable monomer port"
                )
            if endpoint.get("atom") != (expected_atom or None):
                raise ExactV1Error(
                    f"bond {index} endpoint atom differs from registry"
                )
            endpoints.append((
                rows[node_id],
                port,
                expected_atom or None,
            ))
        left, right = endpoints
        if bond["bond_type"] == "PEPTIDE":
            by_port = {left[1]: left[0], right[1]: right[0]}
            if set(by_port) != {"R1", "R2"}:
                raise ExactV1Error(
                    "PEPTIDE bond must connect R2 to R1"
                )
            if (
                by_port["R1"]["chain_id"]
                != by_port["R2"]["chain_id"]
                or int(by_port["R1"]["canonical_position"])
                != int(by_port["R2"]["canonical_position"]) + 1
            ):
                raise ExactV1Error(
                    "PEPTIDE bond does not follow chain order"
                )
            continue
        same_chain = left[0]["chain_id"] == right[0]["chain_id"]
        inferred = _classify_bond(
            left[1],
            left[2],
            right[1],
            right[2],
            same_chain=same_chain,
            left_position=int(left[0]["canonical_position"]),
            right_position=int(right[0]["canonical_position"]),
            chain_length=(
                chain_lengths[str(left[0]["chain_id"])]
                if same_chain
                else -1
            ),
        )
        if str(bond["bond_type"]) != inferred:
            raise ExactV1Error(
                f"bond {index} type differs from registry-bound ports"
            )

    for index, cap in enumerate(document["caps"]):
        definition = registry.resolve(str(cap["monomer_symbol"]))
        cap_port = str(cap["port"])
        expected_cap_atom = definition.ports.get(cap_port)
        if expected_cap_atom is None or cap.get("atom") != (
            expected_cap_atom or None
        ):
            raise ExactV1Error(
                f"cap {index} port atom differs from registry"
            )
        target_id = int(cap["target_node_id"])
        target_definition = definitions[target_id]
        target_port = str(cap["target_port"])
        expected_target_atom = target_definition.ports.get(
            target_port
        )
        if expected_target_atom is None or cap.get("target_atom") != (
            expected_target_atom or None
        ):
            raise ExactV1Error(
                f"cap {index} target atom differs from registry"
            )


def _ensure_notation_serializable(
    document: Mapping[str, Any],
) -> None:
    if any(row.get("modifications") for row in document["monomers"]):
        raise ExactV1Error(
            "exact_v1 modifications are not notation-serializable"
        )
    # Registry binding also proves that each semantic bond type is exactly
    # recoverable from its monomer IDs and ports.
    _validate_registry_bound_graph(document, _registry())


def exact_v1_to_json(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    indent: int | None = 2,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_json(value, indent=indent)
    document = _coerce_document(value)
    return json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    ) + ("\n" if indent is not None else "")


def validate_exact_v1(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate hashes, canonical order, and registry-bound identities."""
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return validate_exact_v1(value)
    return _coerce_document(value)


def parse_exact_v1_document(
    value: Mapping[str, Any] | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an EXACT or ABSTAIN document without promoting ABSTAIN."""
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_symbol_hints,
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return parse_exact_v1_document(value)
    if isinstance(value, Mapping):
        document = dict(value)
    elif isinstance(value, str):
        try:
            document = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ExactV1Error("exact_v1 input is not JSON") from exc
        if not isinstance(document, dict):
            raise ExactV1Error("exact_v1 JSON must be an object")
    else:
        raise ExactV1Error("unsupported exact_v1 input type")
    _validate_document_shape(document)
    if document.get("schema") != EXACT_V1_SCHEMA:
        raise ExactV1Error("unsupported exact_v1 schema")
    if document.get("exactness_status") == ABSTAIN:
        CyclicPeptideGraph.from_exact_document(document)
        return document
    return _coerce_document(document)


def _nodes_by_chain(document: Mapping[str, Any]):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in document["monomers"]:
        grouped.setdefault(str(row["chain_id"]), []).append(dict(row))
    return {
        chain_id: sorted(
            rows, key=lambda row: int(row["canonical_position"])
        )
        for chain_id, rows in sorted(grouped.items())
    }


def exact_v1_to_map(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_symbol_hints,
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_map(value)
    document = _coerce_document(value)
    _ensure_notation_serializable(document)
    chains = _nodes_by_chain(document)
    symbol_to_map = dict(_map_utils._symbol_to_map)
    fragments = []
    global_position: dict[int, int] = {}
    running = 0
    for chain_id, rows in chains.items():
        text = ""
        for row in rows:
            symbol = str(row["monomer_symbol"])
            text += symbol_to_map.get(
                symbol, _map_utils._auto_map_denotion(symbol)
            )
            running += 1
            global_position[int(row["node_id"])] = running
        fragments.append(text)
    tags = []
    closure_bonds = [
        row for row in document["bonds"]
        if row["bond_type"] != "PEPTIDE"
    ]
    for row in closure_bonds:
        src = row["src"]
        dst = row["dst"]
        src_node = int(src["node_id"])
        dst_node = int(dst["node_id"])
        if (
            row["bond_type"] == "HT"
            and len(chains) == 1
            and {
                (src_node, str(src["port"])),
                (dst_node, str(dst["port"])),
            }
            == {
                (1, "R1"),
                (len(document["monomers"]), "R2"),
            }
        ):
            tags.append("{cyc:N-C}")
            continue
        left = (
            global_position[src_node],
            str(src["port"]),
        )
        right = (
            global_position[dst_node],
            str(dst["port"]),
        )
        if left > right:
            left, right = right, left
        tags.append(
            f"{{cyc:{left[0]}:{left[1]}-{right[0]}:{right[1]}}}"
        )
    cap_tags = []
    ordered_caps = sorted(
        document["caps"],
        key=lambda cap: (
            0 if str(cap["terminus"]) == "N" else 1,
            str(cap["monomer_symbol"]),
        ),
    )
    for cap in ordered_caps:
        symbol = str(cap["monomer_symbol"])
        code = {
            ("N", "ac"): "{nt:ACE}",
            ("C", "nme"): "{ct:NME}",
            ("C", "nh2"): "{ct:NH2}",
        }.get((str(cap["terminus"]), symbol))
        if code is None:
            raise ExactV1Error("CAP_NOT_MAP_SERIALIZABLE")
        cap_tags.append(code)
    if cap_tags and len(chains) != 1:
        raise ExactV1Error("MULTICHAIN_CAPS_NOT_MAP_SERIALIZABLE")
    return (
        "{br}".join(fragments)
        + "".join(tags)
        + "".join(cap_tags)
    )


def exact_v1_to_helm(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_symbol_hints,
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_helm(value)
    document = _coerce_document(value)
    _ensure_notation_serializable(document)
    grouped = _nodes_by_chain(document)
    chain_ids = list(grouped)
    chain_index = {
        chain_id: index for index, chain_id in enumerate(chain_ids)
    }
    caps_by_chain: dict[str, dict[str, str]] = {}
    for cap in document["caps"]:
        caps_by_chain.setdefault(str(cap["chain_id"]), {})[
            str(cap["terminus"])
        ] = str(cap["monomer_symbol"])
    chains = []
    node_endpoint: dict[int, tuple[int, int]] = {}
    for chain_id in chain_ids:
        rows = grouped[chain_id]
        symbols = []
        leading = caps_by_chain.get(chain_id, {}).get("N")
        trailing = caps_by_chain.get(chain_id, {}).get("C")
        if leading:
            symbols.append(leading)
        for row in rows:
            symbols.append(str(row["monomer_symbol"]))
            node_endpoint[int(row["node_id"])] = (
                chain_index[chain_id],
                len(symbols),
            )
        if trailing:
            symbols.append(trailing)
        chains.append(symbols)
    edges = []
    for bond in document["bonds"]:
        if bond["bond_type"] == "PEPTIDE":
            continue
        left = bond["src"]
        right = bond["dst"]
        left_chain, left_position = node_endpoint[int(left["node_id"])]
        right_chain, right_position = node_endpoint[int(right["node_id"])]
        edges.append((
            (
                left_chain,
                left_position,
                int(str(left["port"]).removeprefix("R")),
            ),
            (
                right_chain,
                right_position,
                int(str(right["port"]).removeprefix("R")),
            ),
        ))
    return _map_utils._format_helm(chains, edges)


def exact_v1_to_biln(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_symbol_hints,
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_biln(value)
    return _map_utils.helm_to_biln(exact_v1_to_helm(value))


def exact_v1_to_smiles(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_symbol_hints,
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_smiles(value)
    map_text = exact_v1_to_map(value)
    smiles = _map_utils.get_smi_from_map(map_text)
    if not smiles:
        raise ExactV1Error("EXACT_V1_SMILES_ASSEMBLY_FAILED")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ExactV1Error("EXACT_V1_SMILES_NOT_PARSEABLE")
    return Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )


def exact_v1_equivalent(
    left: Mapping[str, Any] | CyclicPeptideGraph | str,
    right: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> bool:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=(
                *monomer_symbol_hints(left),
                *monomer_symbol_hints(right),
            ),
        ):
            return exact_v1_equivalent(left, right)
    try:
        return canonical_exact_v1_bytes(
            _coerce_document(left)
        ) == canonical_exact_v1_bytes(_coerce_document(right))
    except (ExactV1Error, CyclicPeptideGraphError):
        return False


def chemical_graph_equivalent(
    left: Mapping[str, Any] | CyclicPeptideGraph | str,
    right: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> bool:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=(
                *monomer_symbol_hints(left),
                *monomer_symbol_hints(right),
            ),
        ):
            return chemical_graph_equivalent(left, right)
    try:
        left_smiles = exact_v1_to_smiles(left)
        right_smiles = exact_v1_to_smiles(right)
    except (ExactV1Error, CyclicPeptideGraphError):
        return False
    left_mol = Chem.MolFromSmiles(left_smiles)
    right_mol = Chem.MolFromSmiles(right_smiles)
    if left_mol is None or right_mol is None:
        return False
    return Chem.MolToSmiles(
        left_mol, canonical=True, isomericSmiles=True
    ) == Chem.MolToSmiles(
        right_mol, canonical=True, isomericSmiles=True
    )


def _edge_residue_token(symbol: str) -> str:
    if symbol in _STANDARD_ONE_TO_EDGE:
        return f"<{_STANDARD_ONE_TO_EDGE[symbol]}>"
    if any(character in symbol for character in "<>"):
        raise ExactV1Error("MONOMER_NOT_EDGE_V1_TOKENIZABLE")
    return f"<{symbol}>"


def _edge_bond_type(row: Mapping[str, Any]) -> str | None:
    bond_type = str(row["bond_type"])
    ports = {str(row["src"]["port"]), str(row["dst"]["port"])}
    return {
        "HT": "HT",
        "SS": "SS",
        "ISOPEPTIDE": "SC",
        "ESTER": "EST",
        "THIOETHER": "THIO",
        "HSC": "HSC",
        "SIDECHAIN": "SC",
        "SIDECHAIN_TO_TAIL": "SC",
        "ALKYL": "ALK",
    }.get(bond_type) or (
        "HSC" if ports == {"R1", "R3"} else None
    )


def exact_v1_to_edge_v1(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    max_rings: int = DEFAULT_EDGE_MAX_RINGS,
    max_position: int = DEFAULT_EDGE_MAX_POSITION,
    preserve_source_order: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_edge_v1(
                value,
                max_rings=max_rings,
                max_position=max_position,
                preserve_source_order=preserve_source_order,
            )
    try:
        document = _coerce_document(value)
    except (ExactV1Error, CyclicPeptideGraphError) as exc:
        return {
            "status": "UNPROJECTABLE",
            "value": None,
            "reason_codes": [str(exc)],
        }
    chains = _nodes_by_chain(document)
    ordered_monomers = list(document["monomers"])
    projection_position = {
        int(row["node_id"]): int(row["node_id"])
        for row in ordered_monomers
    }
    if preserve_source_order:
        source_chains = {
            str(row["original_chain_id"])
            for row in ordered_monomers
        }
        if len(source_chains) == 1:
            ordered_monomers = sorted(
                ordered_monomers,
                key=lambda row: int(row["original_position"]),
            )
            projection_position = {
                int(row["node_id"]): position
                for position, row in enumerate(ordered_monomers, 1)
            }
    reasons = []
    if len(chains) != 1:
        reasons.append("EDGE_V1_MULTICHAIN_UNSUPPORTED")
    if document["caps"]:
        reasons.append("EDGE_V1_CAPS_UNSUPPORTED")
    if any(row.get("modifications") for row in document["monomers"]):
        reasons.append("EDGE_V1_MODIFICATIONS_UNSUPPORTED")
    residue_count = len(ordered_monomers)
    if residue_count > int(max_position):
        reasons.append("EDGE_V1_POSITION_CAPACITY_EXCEEDED")
    closures = [
        row for row in document["bonds"]
        if row["bond_type"] != "PEPTIDE"
    ]
    if len(closures) > int(max_rings):
        reasons.append("EDGE_V1_RING_CAPACITY_EXCEEDED")
    operations = []
    for row in closures:
        projected_type = _edge_bond_type(row)
        if projected_type not in EDGE_BOND_TYPES:
            reasons.append(
                f"EDGE_V1_BOND_TYPE_UNSUPPORTED:{row['bond_type']}"
            )
            continue
        if row["bond_type"] == "HT":
            src, dst = 1, residue_count
        else:
            src = projection_position[int(row["src"]["node_id"])]
            dst = projection_position[int(row["dst"]["node_id"])]
        if src > dst:
            src, dst = dst, src
        if not (1 <= src < dst <= int(max_position)):
            reasons.append("EDGE_V1_ENDPOINT_OUT_OF_RANGE")
        operations.append((projected_type, src, dst))
    trace = document.get("projection_trace")
    if preserve_source_order and isinstance(trace, Mapping):
        traced_operations = []
        for row in trace.get("operations", []):
            src, dst = int(row["src"]), int(row["dst"])
            if src > dst:
                src, dst = dst, src
            traced_operations.append((
                str(row["bond_type"]),
                src,
                dst,
            ))
        if sorted(traced_operations) != sorted(operations):
            reasons.append("EDGE_V1_PROJECTION_TRACE_MISMATCH")
        else:
            operations = traced_operations
    if reasons:
        return {
            "status": "UNPROJECTABLE",
            "value": None,
            "reason_codes": list(dict.fromkeys(reasons)),
        }
    tokens = []
    if not operations:
        tokens.extend(["<LINEAR>", "<NO_SRC>", "<NO_DST>"])
    else:
        for bond_type, src, dst in operations:
            tokens.extend([
                f"<BOND_{bond_type}>",
                f"<SRC_POS_{src}>",
                f"<DST_POS_{dst}>",
            ])
    while len(tokens) < int(max_rings) * 3:
        tokens.extend(["<NO_BOND>", "<NO_SRC>", "<NO_DST>"])
    tokens.extend(
        _edge_residue_token(str(row["monomer_symbol"]))
        for row in ordered_monomers
    )
    return {
        "status": "PROJECTED",
        "value": "".join(tokens),
        "reason_codes": [],
        "max_rings": int(max_rings),
        "max_position": int(max_position),
        "ordering": (
            "source" if preserve_source_order else "canonical"
        ),
        "source_graph_sha256": document["graph_sha256"],
    }


def model_projection_equivalent(
    left: Mapping[str, Any] | CyclicPeptideGraph | str,
    right: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    max_rings: int = DEFAULT_EDGE_MAX_RINGS,
    max_position: int = DEFAULT_EDGE_MAX_POSITION,
    monomer_context: Mapping[str, Any] | None = None,
) -> bool:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=(
                *monomer_symbol_hints(left),
                *monomer_symbol_hints(right),
            ),
        ):
            return model_projection_equivalent(
                left,
                right,
                max_rings=max_rings,
                max_position=max_position,
            )
    left_projection = exact_v1_to_edge_v1(
        left, max_rings=max_rings, max_position=max_position
    )
    right_projection = exact_v1_to_edge_v1(
        right, max_rings=max_rings, max_position=max_position
    )
    return (
        left_projection["status"] == "PROJECTED"
        and right_projection["status"] == "PROJECTED"
        and left_projection["value"] == right_projection["value"]
    )


def _scan_model_tokens(sequence: str) -> list[str]:
    tokens = []
    index = 0
    while index < len(sequence):
        if sequence[index] == "<":
            end = sequence.find(">", index)
            if end < 0:
                raise ExactV1Error("UNCLOSED_MODEL_TOKEN")
            tokens.append(sequence[index:end + 1])
            index = end + 1
            continue
        if sequence[index].isspace():
            index += 1
            continue
        three = sequence[index:index + 3]
        if three in _THREE_TITLE_TO_ONE:
            tokens.append(three)
            index += 3
            continue
        tokens.append(sequence[index])
        index += 1
    return tokens


def _model_residue_symbol(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        content = token[1:-1]
        return _EDGE_STANDARD_TO_ONE.get(content, content)
    if token in _THREE_TITLE_TO_ONE:
        return _THREE_TITLE_TO_ONE[token]
    if token in _STANDARD_ONE_TO_EDGE:
        return token
    return token


def _operations_to_exact(
    operations: Sequence[EdgeOperation],
    residues: Sequence[str],
    *,
    source_kind: str,
) -> dict[str, Any]:
    chain = [_model_residue_symbol(token) for token in residues]
    edges = []
    details = []
    projection_operations = []
    for operation in operations:
        if operation.bond_type in {"NONE", "LINEAR"}:
            continue
        if operation.src is None or operation.dst is None:
            return _abstain(
                ["MODEL_EDGE_ENDPOINT_MISSING"],
                source_kind=source_kind,
            )
        ports = {
            "HT": (1, 2),
            "SS": (3, 3),
            "SC": (3, 3),
            "EST": (3, 3),
            "HSC": (1, 3),
            "THIO": (3, 3),
            "ALK": (3, 3),
        }.get(operation.bond_type)
        if ports is None:
            return _abstain(
                ["MODEL_EDGE_BOND_TYPE_UNSUPPORTED"],
                source_kind=source_kind,
            )
        edges.append((
            (0, int(operation.src), ports[0]),
            (0, int(operation.dst), ports[1]),
        ))
        details.append({"bond_type": operation.bond_type})
        projection_operations.append({
            "bond_type": operation.bond_type,
            "src": int(operation.src),
            "dst": int(operation.dst),
        })
    return _build_from_components(
        [chain],
        edges,
        source_kind=source_kind,
        edge_details=details,
        metadata={
            "projection_trace": {
                "source_kind": source_kind,
                "operations": projection_operations,
            }
        },
    )


def edge_v1_to_exact_v1(
    sequence: str,
    *,
    max_rings: int = DEFAULT_EDGE_MAX_RINGS,
    max_position: int = DEFAULT_EDGE_MAX_POSITION,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import monomer_resolution_context

        with monomer_resolution_context(monomer_context):
            return edge_v1_to_exact_v1(
                sequence,
                max_rings=max_rings,
                max_position=max_position,
            )
    try:
        if (
            type(max_rings) is not int
            or max_rings < 1
            or type(max_position) is not int
            or max_position < 1
        ):
            raise ExactV1Error(
                "EDGE_V1_CAPACITY_MUST_BE_POSITIVE"
            )
        tokens = _scan_model_tokens(sequence)
        prefix = int(max_rings) * 3
        if len(tokens) <= prefix:
            raise ExactV1Error("EDGE_V1_TOKEN_COUNT_INVALID")
        operations = []
        terminal_seen = False
        for ring in range(1, int(max_rings) + 1):
            bond, src, dst = tokens[(ring - 1) * 3:ring * 3]
            if bond == "<NO_BOND>":
                if src != "<NO_SRC>" or dst != "<NO_DST>":
                    raise ExactV1Error("EDGE_V1_EMPTY_SLOT_HAS_ENDPOINT")
                terminal_seen = True
                operations.append(EdgeOperation(ring, "NONE", None, None))
                continue
            if bond == "<LINEAR>":
                if src != "<NO_SRC>" or dst != "<NO_DST>":
                    raise ExactV1Error("EDGE_V1_LINEAR_SLOT_HAS_ENDPOINT")
                if terminal_seen or ring != 1:
                    raise ExactV1Error(
                        "EDGE_V1_LINEAR_SLOT_ORDER_INVALID"
                    )
                terminal_seen = True
                operations.append(EdgeOperation(ring, "LINEAR", None, None))
                continue
            if terminal_seen:
                raise ExactV1Error("EDGE_V1_ACTIVE_SLOT_AFTER_TERMINAL")
            bond_match = _EDGE_BOND_RE.fullmatch(bond)
            src_match = _EDGE_SRC_RE.fullmatch(src)
            dst_match = _EDGE_DST_RE.fullmatch(dst)
            if not (bond_match and src_match and dst_match):
                raise ExactV1Error("EDGE_V1_TRIPLE_MALFORMED")
            operations.append(EdgeOperation(
                ring,
                bond_match.group(1),
                int(src_match.group(1)),
                int(dst_match.group(1)),
            ))
        residues = tokens[prefix:]
        if len(residues) > int(max_position):
            raise ExactV1Error(
                "EDGE_V1_POSITION_CAPACITY_EXCEEDED"
            )
        for operation in operations:
            if (
                operation.src is not None
                and operation.src > int(max_position)
            ) or (
                operation.dst is not None
                and operation.dst > int(max_position)
            ):
                raise ExactV1Error(
                    "EDGE_V1_ENDPOINT_OUT_OF_RANGE"
                )
        return _operations_to_exact(
            operations, residues, source_kind="edge_v1"
        )
    except Exception as exc:
        return _abstain(
            [f"EDGE_V1_PARSE_FAILED:{type(exc).__name__}:{exc}"],
            source_kind="edge_v1",
        )


def legacy_v5_to_exact_v1(
    sequence: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import monomer_resolution_context

        with monomer_resolution_context(monomer_context):
            return legacy_v5_to_exact_v1(sequence)
    try:
        types: dict[int, str] = {}
        endpoints: dict[int, tuple[int, int]] = {}
        residues = []
        linear = False
        for token in _scan_model_tokens(sequence):
            type_match = _LEGACY_TYPE_RE.fullmatch(token)
            if type_match:
                ring = int(type_match.group(1))
                if ring in types:
                    raise ExactV1Error(
                        "LEGACY_V5_DUPLICATE_RING_TYPE"
                    )
                types[ring] = type_match.group(2)
                continue
            endpoint_match = _LEGACY_POS_RE.fullmatch(token)
            if endpoint_match:
                ring = int(endpoint_match.group(1))
                if ring in endpoints:
                    raise ExactV1Error(
                        "LEGACY_V5_DUPLICATE_RING_ENDPOINT"
                    )
                endpoints[ring] = (
                    int(endpoint_match.group(2)),
                    int(endpoint_match.group(3)),
                )
                continue
            if token == "<linear>":
                linear = True
                continue
            residues.append(token)
        if linear and types:
            raise ExactV1Error(
                "LEGACY_V5_LINEAR_AND_CYCLIC_CONFLICT"
            )
        if set(endpoints) - set(types):
            raise ExactV1Error(
                "LEGACY_V5_ENDPOINT_WITHOUT_RING_TYPE"
            )
        operations = []
        if linear and not types:
            operations.append(EdgeOperation(1, "LINEAR", None, None))
        for ring in sorted(types):
            bond_type = types[ring]
            pair = endpoints.get(ring)
            if pair is None and bond_type == "HT" and residues:
                pair = (1, len(residues))
            operations.append(EdgeOperation(
                ring,
                bond_type,
                pair[0] if pair else None,
                pair[1] if pair else None,
            ))
        if not operations:
            operations.append(EdgeOperation(1, "LINEAR", None, None))
        return _operations_to_exact(
            operations, residues, source_kind="legacy_v5"
        )
    except Exception as exc:
        return _abstain(
            [f"LEGACY_V5_PARSE_FAILED:{type(exc).__name__}:{exc}"],
            source_kind="legacy_v5",
        )


def exact_v1_to_legacy_v5(
    value: Mapping[str, Any] | CyclicPeptideGraph | str,
    *,
    max_rings: int = DEFAULT_EDGE_MAX_RINGS,
    max_position: int = DEFAULT_EDGE_MAX_POSITION,
    preserve_source_order: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(value),
        ):
            return exact_v1_to_legacy_v5(
                value,
                max_rings=max_rings,
                max_position=max_position,
                preserve_source_order=preserve_source_order,
            )
    projection = exact_v1_to_edge_v1(
        value,
        max_rings=max_rings,
        max_position=max_position,
        preserve_source_order=preserve_source_order,
    )
    if projection["status"] != "PROJECTED":
        return {
            "status": "UNPROJECTABLE",
            "value": None,
            "reason_codes": projection["reason_codes"],
        }
    tokens = _scan_model_tokens(str(projection["value"]))
    prefix = int(max_rings) * 3
    type_tokens = []
    position_tokens = []
    active = False
    for ring in range(1, int(max_rings) + 1):
        bond, src, dst = tokens[(ring - 1) * 3:ring * 3]
        if bond == "<LINEAR>":
            type_tokens.append("<linear>")
            active = True
            break
        if bond == "<NO_BOND>":
            continue
        bond_type = _EDGE_BOND_RE.fullmatch(bond).group(1)
        src_pos = _EDGE_SRC_RE.fullmatch(src).group(1)
        dst_pos = _EDGE_DST_RE.fullmatch(dst).group(1)
        type_tokens.append(f"<cyc{ring}_{bond_type}>")
        position_tokens.append(
            f"<cyc{ring}_{src_pos}_{dst_pos}>"
        )
        active = True
    if not active:
        type_tokens.append("<linear>")
    residues = [
        _model_residue_symbol(token) for token in tokens[prefix:]
    ]
    return {
        "status": "PROJECTED",
        "value": "".join(type_tokens + residues + position_tokens),
        "reason_codes": [],
        "source_graph_sha256": projection["source_graph_sha256"],
    }


def exact_v1_from_v6_result(result: Any) -> dict[str, Any]:
    """Build exact_v1 only from a qualified, unrepaired V6 result."""

    def field(name: str, default=None):
        if isinstance(result, Mapping):
            return result.get(name, default)
        return getattr(result, name, default)

    if (
        field("status") != "success"
        or field("support_status") != "qualified"
        or field("qualified_success") is not True
        or list(field("repair_codes", []) or [])
        or list(field("warning_codes", []) or [])
    ):
        return _abstain(
            ["V6_RESULT_NOT_QUALIFIED_EXACT"],
            source_kind="v6",
        )
    output_evidence = dict(field("output_evidence", {}) or {})
    dimensions = dict(output_evidence.get("evidence_dimensions", {}) or {})
    required_dimensions = {
        "library_chemistry",
        "atom_mapping",
        "chemical_graph_audit",
        "stereochemistry",
        "mapping_evidence_binding",
    }
    if any(
        not isinstance(dimensions.get(name), Mapping)
        or dimensions[name].get("passed") is not True
        for name in required_dimensions
    ):
        return _abstain(
            ["V6_EXACT_IDENTITY_DIMENSIONS_NOT_QUALIFIED"],
            source_kind="v6",
        )
    output_inchikey = field("output_inchikey")
    route_results = list(field("route_results", []) or [])
    candidates = []
    for row in route_results:
        if (
            not isinstance(row, Mapping)
            or row.get("status") != "success"
            or row.get("output_inchikey") != output_inchikey
        ):
            continue
        evidence = row.get("evidence_dimensions_input")
        if not isinstance(evidence, Mapping):
            continue
        residue_rows = list(evidence.get("residue_evidence") or [])
        if not residue_rows:
            continue
        ordered = sorted(
            residue_rows,
            key=lambda value: int(value.get("residue_position", -1)),
        )
        positions = [int(value.get("residue_position", -1)) for value in ordered]
        if positions != list(range(1, len(ordered) + 1)):
            continue
        symbols = [str(value.get("unified_symbol") or "") for value in ordered]
        if any(not symbol for symbol in symbols):
            continue
        input_evidence = dict(field("input_evidence", {}) or {})
        edges = []
        details = []
        closure_parse_failed = False
        for closure in input_evidence.get("cyclization_bonds", []):
            try:
                left_port = str(closure["rgroup_1"]).upper()
                right_port = str(closure["rgroup_2"]).upper()
                edges.append((
                    (
                        0,
                        int(closure["position_1"]),
                        int(left_port.removeprefix("R")),
                    ),
                    (
                        0,
                        int(closure["position_2"]),
                        int(right_port.removeprefix("R")),
                    ),
                ))
                details.append({
                    "bond_type": closure.get("bond_type"),
                })
            except Exception:
                closure_parse_failed = True
                break
        if closure_parse_failed:
            return _abstain(
                ["V6_CYCLIZATION_EVIDENCE_MALFORMED"],
                source_kind="v6",
            )
        candidate = _build_from_components(
            [symbols],
            edges,
            source_kind="v6",
            edge_details=details,
        )
        if candidate.get("exactness_status") == EXACT:
            candidates.append(candidate)
    if not candidates:
        return _abstain(
            ["V6_MONOMER_PORT_EVIDENCE_UNAVAILABLE"],
            source_kind="v6",
        )
    reference = candidates[0]
    if any(not exact_v1_equivalent(reference, item) for item in candidates[1:]):
        return _abstain(
            ["V6_ROUTE_EXACT_V1_DISAGREEMENT"],
            source_kind="v6",
        )
    try:
        assembled = Chem.MolFromSmiles(exact_v1_to_smiles(reference))
        assembled_key = (
            Chem.MolToInchiKey(assembled) if assembled is not None else None
        )
    except Exception:
        assembled_key = None
    if not output_inchikey or assembled_key != output_inchikey:
        return _abstain(
            ["V6_EXACT_V1_FULL_IDENTITY_MISMATCH"],
            source_kind="v6",
        )
    return reference


__all__ = [
    "ABSTAIN",
    "DEFAULT_EDGE_MAX_POSITION",
    "DEFAULT_EDGE_MAX_RINGS",
    "EXACT",
    "EXACT_V1_SCHEMA",
    "ExactV1Abstained",
    "ExactV1Error",
    "MonomerRegistry",
    "biln_to_exact_v1",
    "chemical_graph_equivalent",
    "edge_v1_to_exact_v1",
    "exact_v1_equivalent",
    "exact_v1_from_v6_result",
    "exact_v1_to_biln",
    "exact_v1_to_edge_v1",
    "exact_v1_to_helm",
    "exact_v1_to_json",
    "exact_v1_to_legacy_v5",
    "exact_v1_to_map",
    "exact_v1_to_smiles",
    "helm_to_exact_v1",
    "legacy_v5_to_exact_v1",
    "map_to_exact_v1",
    "model_projection_equivalent",
    "notation_to_exact_v1",
    "parse_exact_v1_document",
    "reset_exact_v1_registry_cache",
    "validate_exact_v1",
]
