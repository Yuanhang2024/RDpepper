"""Lossless monomer-port graph primitives for cyclic peptides.

The graph is intentionally independent from model token representations.  It
stores stable monomer identities, directed peptide-chain order, explicit port
connections, terminal caps, and source-to-canonical traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Sequence


EXACT_V1_SCHEMA = "cycpep_exact_v1"
EXACT = "EXACT"
ABSTAIN = "ABSTAIN"

PORTS = frozenset({"R1", "R2", "R3"})
BOND_TYPES = frozenset({
    "PEPTIDE",
    "HT",
    "SS",
    "ISOPEPTIDE",
    "ESTER",
    "THIOETHER",
    "HSC",
    "SIDECHAIN",
    "SIDECHAIN_TO_TAIL",
    "ALKYL",
    "CROSSLINK",
})
MAX_CANONICALIZATION_STATES = 100_000


class CyclicPeptideGraphError(ValueError):
    """A monomer-port graph violates the exact_v1 contract."""


class CanonicalizationError(CyclicPeptideGraphError):
    """The graph cannot be canonically ordered without guessing."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class MonomerNode:
    node_id: int
    monomer_id: str
    monomer_symbol: str
    stereo: str
    modifications: tuple[str, ...] = ()
    chain_id: str = "A"
    original_position: int = 1


@dataclass(frozen=True)
class BondEndpoint:
    node_id: int
    port: str
    atom: str | None = None

    def key(self) -> tuple[int, str, str]:
        return self.node_id, self.port, self.atom or ""


@dataclass(frozen=True)
class PortBond:
    bond_type: str
    src: BondEndpoint
    dst: BondEndpoint
    bond_order: str = "SINGLE"


@dataclass(frozen=True)
class TerminalCap:
    cap_id: str
    monomer_id: str
    monomer_symbol: str
    chain_id: str
    terminus: str
    port: str
    atom: str | None = None
    target_node_id: int = 0
    target_port: str = ""
    target_atom: str | None = None


def _chain_name(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return f"CHAIN_{index + 1}"


def _endpoint_dict(endpoint: BondEndpoint, mapping: Mapping[int, int]) -> dict:
    return {
        "node_id": int(mapping[endpoint.node_id]),
        "port": endpoint.port,
        "atom": endpoint.atom,
    }


def _canonical_endpoint_pair(
    bond: PortBond, mapping: Mapping[int, int]
) -> tuple[dict, dict]:
    left = _endpoint_dict(bond.src, mapping)
    right = _endpoint_dict(bond.dst, mapping)
    left_key = (left["node_id"], left["port"], left["atom"] or "")
    right_key = (right["node_id"], right["port"], right["atom"] or "")
    return (left, right) if left_key <= right_key else (right, left)


def canonical_graph_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the graph bytes governed by ``graph_sha256``.

    Source-position trace fields are deliberately excluded.  They may differ
    between equivalent rotated inputs while the canonical graph remains
    byte-identical.
    """
    monomers = []
    for row in document.get("monomers", []):
        monomers.append({
            "node_id": int(row["node_id"]),
            "monomer_id": str(row["monomer_id"]),
            "monomer_symbol": str(row["monomer_symbol"]),
            "stereo": str(row["stereo"]),
            "modifications": list(row.get("modifications", [])),
            "chain_id": str(row["chain_id"]),
            "position": int(row["canonical_position"]),
        })
    bonds = []
    for row in document.get("bonds", []):
        bonds.append({
            "bond_type": str(row["bond_type"]),
            "src": {
                "node_id": int(row["src"]["node_id"]),
                "port": str(row["src"]["port"]),
                "atom": row["src"].get("atom"),
            },
            "dst": {
                "node_id": int(row["dst"]["node_id"]),
                "port": str(row["dst"]["port"]),
                "atom": row["dst"].get("atom"),
            },
            "bond_order": str(row.get("bond_order", "SINGLE")),
        })
    caps = []
    for row in document.get("caps", []):
        caps.append({
            "cap_id": str(row["cap_id"]),
            "monomer_id": str(row["monomer_id"]),
            "monomer_symbol": str(row["monomer_symbol"]),
            "chain_id": str(row["chain_id"]),
            "terminus": str(row["terminus"]),
            "port": str(row["port"]),
            "atom": row.get("atom"),
            "target_node_id": int(row["target_node_id"]),
            "target_port": str(row["target_port"]),
            "target_atom": row.get("target_atom"),
        })
    return {
        "schema": EXACT_V1_SCHEMA,
        "monomers": sorted(monomers, key=lambda row: row["node_id"]),
        "bonds": sorted(
            bonds,
            key=lambda row: (
                row["bond_type"],
                row["src"]["node_id"],
                row["src"]["port"],
                row["dst"]["node_id"],
                row["dst"]["port"],
            ),
        ),
        "caps": sorted(
            caps,
            key=lambda row: (
                row["chain_id"],
                row["terminus"],
                row["monomer_id"],
                row["target_node_id"],
            ),
        ),
        "chain_breaks": sorted(
            (
                {
                    "left_chain_id": str(row["left_chain_id"]),
                    "right_chain_id": str(row["right_chain_id"]),
                }
                for row in document.get("chain_breaks", [])
            ),
            key=lambda row: (
                row["left_chain_id"],
                row["right_chain_id"],
            ),
        ),
    }


def canonical_exact_v1_bytes(document: Mapping[str, Any]) -> bytes:
    """Return canonical graph bytes used for exact equivalence and hashing."""
    if document.get("exactness_status") != EXACT:
        raise CyclicPeptideGraphError(
            "canonical bytes require exactness_status=EXACT"
        )
    return canonical_json_bytes(canonical_graph_payload(document))


@dataclass(frozen=True)
class CyclicPeptideGraph:
    monomers: tuple[MonomerNode, ...] = ()
    bonds: tuple[PortBond, ...] = ()
    caps: tuple[TerminalCap, ...] = ()
    exactness_status: str = EXACT
    reason_codes: tuple[str, ...] = ()
    source_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def abstain(
        cls,
        reason_codes: Iterable[str],
        *,
        source_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CyclicPeptideGraph":
        reasons = tuple(dict.fromkeys(str(code) for code in reason_codes if code))
        return cls(
            exactness_status=ABSTAIN,
            reason_codes=reasons or ("UNSPECIFIED_ABSTENTION",),
            source_kind=source_kind,
            metadata=dict(metadata or {}),
        )

    def _chains(self) -> dict[str, tuple[MonomerNode, ...]]:
        grouped: dict[str, list[MonomerNode]] = {}
        for node in self.monomers:
            grouped.setdefault(node.chain_id, []).append(node)
        return {
            chain_id: tuple(sorted(
                nodes,
                key=lambda node: (node.original_position, node.node_id),
            ))
            for chain_id, nodes in grouped.items()
        }

    def validation_reasons(self) -> list[str]:
        if self.exactness_status == ABSTAIN:
            return list(self.reason_codes)
        reasons: list[str] = []
        if self.exactness_status != EXACT:
            reasons.append("INVALID_EXACTNESS_STATUS")
        node_ids = [node.node_id for node in self.monomers]
        if not node_ids:
            reasons.append("MONOMER_GRAPH_EMPTY")
        if any(type(value) is not int or value < 1 for value in node_ids):
            reasons.append("INVALID_NODE_ID")
        if len(node_ids) != len(set(node_ids)):
            reasons.append("DUPLICATE_NODE_ID")
        nodes = {node.node_id: node for node in self.monomers}
        for node in self.monomers:
            if not node.monomer_id:
                reasons.append("MONOMER_ID_MISSING")
            if not node.monomer_symbol:
                reasons.append("MONOMER_SYMBOL_MISSING")
            if not node.stereo or node.stereo == "UNKNOWN":
                reasons.append("MONOMER_STEREO_UNRESOLVED")
            if not node.chain_id:
                reasons.append("CHAIN_ID_MISSING")
            if node.original_position < 1:
                reasons.append("INVALID_ORIGINAL_POSITION")
        for chain_nodes in self._chains().values():
            positions = [node.original_position for node in chain_nodes]
            if positions != list(range(1, len(chain_nodes) + 1)):
                reasons.append("CHAIN_POSITIONS_NOT_CONTIGUOUS")

        used_ports: set[tuple[int, str]] = set()
        seen_bonds: set[tuple] = set()
        for bond in self.bonds:
            if bond.bond_type not in BOND_TYPES:
                reasons.append("UNSUPPORTED_BOND_TYPE")
            if bond.bond_order != "SINGLE":
                reasons.append("UNSUPPORTED_BOND_ORDER")
            if bond.src.node_id not in nodes or bond.dst.node_id not in nodes:
                reasons.append("BOND_ENDPOINT_UNKNOWN")
                continue
            if bond.src.node_id == bond.dst.node_id:
                reasons.append("SELF_BOND")
            for endpoint in (bond.src, bond.dst):
                if endpoint.port not in PORTS:
                    reasons.append("INVALID_PORT")
                key = (endpoint.node_id, endpoint.port)
                if key in used_ports:
                    reasons.append("PORT_REUSED")
                used_ports.add(key)
            endpoint_keys = sorted((bond.src.key(), bond.dst.key()))
            identity = (
                bond.bond_type,
                tuple(endpoint_keys),
                bond.bond_order,
            )
            if identity in seen_bonds:
                reasons.append("DUPLICATE_BOND")
            seen_bonds.add(identity)

        chains = self._chains()
        for chain_id, chain_nodes in chains.items():
            node_by_position = {
                node.original_position: node for node in chain_nodes
            }
            expected = {
                (
                    node_by_position[position].node_id,
                    node_by_position[position + 1].node_id,
                )
                for position in range(1, len(chain_nodes))
            }
            observed = set()
            for bond in self.bonds:
                if bond.bond_type != "PEPTIDE":
                    continue
                left = nodes.get(bond.src.node_id)
                right = nodes.get(bond.dst.node_id)
                if left is None or right is None:
                    continue
                if left.chain_id != chain_id or right.chain_id != chain_id:
                    continue
                ordered = tuple(sorted(
                    (left, right),
                    key=lambda node: node.original_position,
                ))
                observed.add((ordered[0].node_id, ordered[1].node_id))
            if observed != expected:
                reasons.append("BACKBONE_BOND_SET_MISMATCH")

        caps_seen: set[tuple[str, str]] = set()
        for cap in self.caps:
            if cap.chain_id not in chains:
                reasons.append("CAP_CHAIN_UNKNOWN")
            if cap.terminus not in {"N", "C"}:
                reasons.append("INVALID_CAP_TERMINUS")
            expected_port = "R2" if cap.terminus == "N" else "R1"
            if cap.port != expected_port:
                reasons.append("CAP_PORT_MISMATCH")
            if cap.target_node_id not in nodes:
                reasons.append("CAP_TARGET_UNKNOWN")
            else:
                target = nodes[cap.target_node_id]
                chain_nodes = chains.get(cap.chain_id, ())
                expected_target = (
                    chain_nodes[0].node_id
                    if cap.terminus == "N" and chain_nodes
                    else chain_nodes[-1].node_id
                    if chain_nodes
                    else None
                )
                expected_target_port = (
                    "R1" if cap.terminus == "N" else "R2"
                )
                if (
                    target.chain_id != cap.chain_id
                    or cap.target_node_id != expected_target
                    or cap.target_port != expected_target_port
                ):
                    reasons.append("CAP_TARGET_MISMATCH")
                target_key = (cap.target_node_id, cap.target_port)
                if target_key in used_ports:
                    reasons.append("PORT_REUSED")
                used_ports.add(target_key)
            key = (cap.chain_id, cap.terminus)
            if key in caps_seen:
                reasons.append("DUPLICATE_TERMINAL_CAP")
            caps_seen.add(key)
        return list(dict.fromkeys(reasons))

    def _chain_is_head_to_tail(
        self, chain_nodes: Sequence[MonomerNode]
    ) -> bool:
        if not chain_nodes:
            return False
        first = chain_nodes[0].node_id
        last = chain_nodes[-1].node_id
        expected = {
            (first, "R1"),
            (last, "R2"),
        }
        for bond in self.bonds:
            if bond.bond_type != "HT":
                continue
            endpoints = {
                (bond.src.node_id, bond.src.port),
                (bond.dst.node_id, bond.dst.port),
            }
            if endpoints == expected:
                return True
        return False

    def _state_count(self, chains: Mapping[str, Sequence[MonomerNode]]) -> int:
        count = math.factorial(len(chains))
        for nodes in chains.values():
            if self._chain_is_head_to_tail(nodes):
                count *= max(1, len(nodes))
        return count

    def canonicalize(self) -> dict[str, Any]:
        if self.exactness_status != EXACT:
            return {
                "schema": EXACT_V1_SCHEMA,
                "monomers": [],
                "bonds": [],
                "caps": [],
                "chain_breaks": [],
                "canonical_order": [],
                "original_to_canonical": {},
                "canonical_to_original": {},
                "graph_sha256": None,
                "exactness_status": ABSTAIN,
                "reason_codes": list(self.reason_codes),
                "normalization_codes": list(
                    self.metadata.get("normalization_codes", [])
                ),
                "projection_trace": self.metadata.get(
                    "projection_trace"
                ),
                "source_kind": self.source_kind,
            }
        validation = self.validation_reasons()
        if validation:
            return CyclicPeptideGraph.abstain(
                validation,
                source_kind=self.source_kind,
                metadata=self.metadata,
            ).canonicalize()

        chains = self._chains()
        if self._state_count(chains) > MAX_CANONICALIZATION_STATES:
            return CyclicPeptideGraph.abstain(
                ["CANONICALIZATION_STATE_LIMIT_EXCEEDED"],
                source_kind=self.source_kind,
                metadata=self.metadata,
            ).canonicalize()

        nodes_by_id = {node.node_id: node for node in self.monomers}
        best: tuple[bytes, bytes, dict[str, Any]] | None = None
        chain_ids = tuple(sorted(chains))
        for permutation in itertools.permutations(chain_ids):
            rotation_options = []
            for chain_id in permutation:
                chain_nodes = chains[chain_id]
                rotation_options.append(
                    tuple(range(len(chain_nodes)))
                    if self._chain_is_head_to_tail(chain_nodes)
                    else (0,)
                )
            for rotations in itertools.product(*rotation_options):
                ordered_nodes: list[MonomerNode] = []
                canonical_chain_by_original: dict[str, str] = {}
                canonical_position_by_node: dict[int, int] = {}
                for chain_index, (chain_id, rotation) in enumerate(
                    zip(permutation, rotations)
                ):
                    canonical_chain = _chain_name(chain_index)
                    canonical_chain_by_original[chain_id] = canonical_chain
                    original_nodes = list(chains[chain_id])
                    rotated = (
                        original_nodes[rotation:] + original_nodes[:rotation]
                    )
                    for position, node in enumerate(rotated, 1):
                        canonical_position_by_node[node.node_id] = position
                    ordered_nodes.extend(rotated)
                mapping = {
                    node.node_id: index
                    for index, node in enumerate(ordered_nodes, 1)
                }
                cyclic_chain_nodes = {
                    chain_id: {
                        node.node_id for node in chains[chain_id]
                    }
                    for chain_id in permutation
                    if self._chain_is_head_to_tail(chains[chain_id])
                }
                canonical_position = canonical_position_by_node
                monomer_rows = [
                    {
                        "node_id": mapping[node.node_id],
                        "monomer_id": node.monomer_id,
                        "monomer_symbol": node.monomer_symbol,
                        "stereo": node.stereo,
                        "modifications": list(node.modifications),
                        "chain_id": canonical_chain_by_original[node.chain_id],
                        "position": canonical_position_by_node[node.node_id],
                    }
                    for node in ordered_nodes
                ]
                bond_rows = []
                for bond in self.bonds:
                    bond_type = bond.bond_type
                    if bond_type in {"PEPTIDE", "HT"}:
                        left_node = nodes_by_id[bond.src.node_id]
                        right_node = nodes_by_id[bond.dst.node_id]
                        if (
                            left_node.chain_id == right_node.chain_id
                            and left_node.chain_id in cyclic_chain_nodes
                        ):
                            size = len(chains[left_node.chain_id])
                            port_positions = {
                                bond.src.port: canonical_position[
                                    bond.src.node_id
                                ],
                                bond.dst.port: canonical_position[
                                    bond.dst.node_id
                                ],
                            }
                            if set(port_positions) == {"R1", "R2"}:
                                bond_type = (
                                    "HT"
                                    if (
                                        port_positions["R2"] == size
                                        and port_positions["R1"] == 1
                                    )
                                    else "PEPTIDE"
                                )
                    src, dst = _canonical_endpoint_pair(bond, mapping)
                    bond_rows.append({
                        "bond_type": bond_type,
                        "src": src,
                        "dst": dst,
                        "bond_order": bond.bond_order,
                    })
                bond_rows.sort(key=lambda row: (
                    row["bond_type"],
                    row["src"]["node_id"],
                    row["src"]["port"],
                    row["dst"]["node_id"],
                    row["dst"]["port"],
                ))
                cap_rows = [
                    {
                        "cap_id": cap.cap_id,
                        "monomer_id": cap.monomer_id,
                        "monomer_symbol": cap.monomer_symbol,
                        "chain_id": canonical_chain_by_original[cap.chain_id],
                        "terminus": cap.terminus,
                        "port": cap.port,
                        "atom": cap.atom,
                        "target_node_id": mapping[cap.target_node_id],
                        "target_port": cap.target_port,
                        "target_atom": cap.target_atom,
                    }
                    for cap in self.caps
                ]
                cap_rows.sort(key=lambda row: (
                    row["chain_id"],
                    row["terminus"],
                    row["monomer_id"],
                    row["target_node_id"],
                ))
                canonical_chain_ids = [
                    _chain_name(index) for index in range(len(permutation))
                ]
                chain_breaks = [
                    {
                        "left_chain_id": canonical_chain_ids[index],
                        "right_chain_id": canonical_chain_ids[index + 1],
                    }
                    for index in range(len(canonical_chain_ids) - 1)
                ]
                payload = {
                    "schema": EXACT_V1_SCHEMA,
                    "monomers": monomer_rows,
                    "bonds": bond_rows,
                    "caps": cap_rows,
                    "chain_breaks": chain_breaks,
                }
                trace = {
                    "canonical_order": [
                        node.node_id for node in ordered_nodes
                    ],
                    "rotations": list(rotations),
                    "permutation": list(permutation),
                }
                candidate = (
                    canonical_json_bytes(payload),
                    canonical_json_bytes(trace),
                    {
                        "payload": payload,
                        "mapping": mapping,
                        "ordered_nodes": ordered_nodes,
                    },
                )
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:  # pragma: no cover - nonempty validated graph
            raise CanonicalizationError("no canonical graph state was produced")

        selected = best[2]
        payload = selected["payload"]
        mapping = selected["mapping"]
        ordered_nodes = selected["ordered_nodes"]
        source_by_canonical = {
            mapping[node.node_id]: node for node in ordered_nodes
        }
        document_monomers = []
        for row in payload["monomers"]:
            source = source_by_canonical[row["node_id"]]
            document_monomers.append({
                "node_id": row["node_id"],
                "monomer_id": row["monomer_id"],
                "monomer_symbol": row["monomer_symbol"],
                "stereo": row["stereo"],
                "modifications": list(row["modifications"]),
                "chain_id": row["chain_id"],
                "canonical_position": row["position"],
                "original_chain_id": source.chain_id,
                "original_position": source.original_position,
            })
        original_to_canonical = {
            f"{node.chain_id}:{node.original_position}": mapping[node.node_id]
            for node in sorted(
                self.monomers,
                key=lambda value: (
                    value.chain_id,
                    value.original_position,
                    value.node_id,
                ),
            )
        }
        canonical_to_original = {
            str(canonical): key
            for key, canonical in sorted(original_to_canonical.items())
        }
        graph_hash = hashlib.sha256(best[0]).hexdigest()
        document = {
            "schema": EXACT_V1_SCHEMA,
            "monomers": document_monomers,
            "bonds": payload["bonds"],
            "caps": payload["caps"],
            "chain_breaks": payload["chain_breaks"],
            "canonical_order": [
                node.node_id for node in ordered_nodes
            ],
            "original_to_canonical": original_to_canonical,
            "canonical_to_original": canonical_to_original,
            "graph_sha256": graph_hash,
            "exactness_status": EXACT,
            "reason_codes": [],
            "normalization_codes": list(
                self.metadata.get("normalization_codes", [])
            ),
            "projection_trace": self.metadata.get("projection_trace"),
            "source_kind": self.source_kind,
        }
        if hashlib.sha256(canonical_exact_v1_bytes(document)).hexdigest() != (
            graph_hash
        ):
            raise CanonicalizationError("canonical graph hash is unstable")
        return document

    @classmethod
    def from_exact_document(
        cls, document: Mapping[str, Any]
    ) -> "CyclicPeptideGraph":
        if document.get("schema") != EXACT_V1_SCHEMA:
            raise CyclicPeptideGraphError("unsupported exact_v1 schema")
        status = str(document.get("exactness_status") or "")
        if status == ABSTAIN:
            return cls.abstain(
                document.get("reason_codes") or ["UNSPECIFIED_ABSTENTION"],
                source_kind=document.get("source_kind"),
            )
        if status != EXACT:
            raise CyclicPeptideGraphError("invalid exactness_status")
        expected_hash = str(document.get("graph_sha256") or "")
        observed_hash = hashlib.sha256(
            canonical_exact_v1_bytes(document)
        ).hexdigest()
        if expected_hash != observed_hash:
            raise CyclicPeptideGraphError(
                "exact_v1 graph_sha256 differs from canonical bytes"
            )
        nodes = tuple(
            MonomerNode(
                node_id=int(row["node_id"]),
                monomer_id=str(row["monomer_id"]),
                monomer_symbol=str(row["monomer_symbol"]),
                stereo=str(row["stereo"]),
                modifications=tuple(row.get("modifications", [])),
                chain_id=str(row["chain_id"]),
                original_position=int(row["canonical_position"]),
            )
            for row in document.get("monomers", [])
        )
        bonds = tuple(
            PortBond(
                bond_type=str(row["bond_type"]),
                src=BondEndpoint(
                    node_id=int(row["src"]["node_id"]),
                    port=str(row["src"]["port"]),
                    atom=row["src"].get("atom"),
                ),
                dst=BondEndpoint(
                    node_id=int(row["dst"]["node_id"]),
                    port=str(row["dst"]["port"]),
                    atom=row["dst"].get("atom"),
                ),
                bond_order=str(row.get("bond_order", "SINGLE")),
            )
            for row in document.get("bonds", [])
        )
        caps = tuple(
            TerminalCap(
                cap_id=str(row["cap_id"]),
                monomer_id=str(row["monomer_id"]),
                monomer_symbol=str(row["monomer_symbol"]),
                chain_id=str(row["chain_id"]),
                terminus=str(row["terminus"]),
                port=str(row["port"]),
                atom=row.get("atom"),
                target_node_id=int(row["target_node_id"]),
                target_port=str(row["target_port"]),
                target_atom=row.get("target_atom"),
            )
            for row in document.get("caps", [])
        )
        graph = cls(
            monomers=nodes,
            bonds=bonds,
            caps=caps,
            source_kind=document.get("source_kind"),
        )
        reasons = graph.validation_reasons()
        if reasons:
            raise CyclicPeptideGraphError(
                "invalid exact_v1 graph: " + ", ".join(reasons)
            )
        return graph


__all__ = [
    "ABSTAIN",
    "BOND_TYPES",
    "EXACT",
    "EXACT_V1_SCHEMA",
    "BondEndpoint",
    "CanonicalizationError",
    "CyclicPeptideGraph",
    "CyclicPeptideGraphError",
    "MonomerNode",
    "PortBond",
    "TerminalCap",
    "canonical_exact_v1_bytes",
    "canonical_graph_payload",
    "canonical_json_bytes",
]
