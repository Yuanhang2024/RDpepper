"""Fail-closed semantic audits for cyclic-peptide inputs and outputs.

This module is intentionally additive.  The frozen v2 benchmark hashes the
historical parser and reconstruction modules, so remediation APIs live in new
files rather than mutating the code that produced the frozen raw results.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import math
import re
import warnings
from typing import Any, Iterable, Mapping

from rdkit import Chem
from .core.monomer_resolution import needs_monomer_resolution_scope


@dataclass(frozen=True)
class AuditIssue:
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class AuditResult:
    issues: tuple[AuditIssue, ...] = ()

    @property
    def accepted(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def warning_codes(self) -> list[str]:
        return sorted({issue.code for issue in self.issues})

    @property
    def reason(self) -> str | None:
        if self.accepted:
            return None
        return " | ".join(issue.message for issue in self.issues)


def _result(issues: Iterable[AuditIssue]) -> AuditResult:
    return AuditResult(tuple(issues))


def _duplicate_endpoints(
    endpoints: Iterable[tuple[str, str]],
    *,
    notation: str,
) -> list[AuditIssue]:
    seen: set[tuple[str, str]] = set()
    issues: list[AuditIssue] = []
    for endpoint in endpoints:
        if endpoint in seen:
            issues.append(
                AuditIssue(
                    "DUPLICATE_ATTACHMENT_ENDPOINT",
                    f"{notation} attachment endpoint "
                    f"{endpoint[0]}:{endpoint[1]} is used more than once",
                )
            )
        seen.add(endpoint)
    return issues


def audit_biln(text: str) -> AuditResult:
    """Reject malformed or unpaired BILN cross-link annotations."""

    issues: list[AuditIssue] = []
    if not text.strip():
        return _result([AuditIssue("EMPTY_INPUT", "BILN input is empty")])
    if text.count("(") != text.count(")"):
        issues.append(
            AuditIssue(
                "UNBALANCED_BILN_ANNOTATION",
                "BILN contains unbalanced cross-link parentheses",
            )
        )
    annotations = re.findall(r"\(([^()]*)\)", text)
    bonds: dict[str, list[str]] = defaultdict(list)
    for annotation in annotations:
        match = re.fullmatch(r"\s*(\d+)\s*,\s*([123])\s*", annotation)
        if not match:
            issues.append(
                AuditIssue(
                    "MALFORMED_BILN_CROSSLINK",
                    f"invalid BILN cross-link annotation ({annotation})",
                )
            )
            continue
        bonds[match.group(1)].append(match.group(2))
    for bond_id, endpoints in sorted(bonds.items()):
        if len(endpoints) != 2:
            issues.append(
                AuditIssue(
                    "UNPAIRED_RING_CLOSURE",
                    f"BILN bond {bond_id} has {len(endpoints)} endpoint(s), expected 2",
                )
            )
    try:
        from .paths._map_utils import _parse_biln_strict, _symbol_to_map

        chains, _edges = _parse_biln_strict(text)
    except (TypeError, ValueError) as exc:
        issues.append(
            AuditIssue(
                _biln_parser_error_code(str(exc)),
                str(exc),
            )
        )
        return _result(issues)
    for chain in chains:
        for symbol in chain:
            if symbol not in _symbol_to_map:
                issues.append(
                    AuditIssue(
                        "UNKNOWN_MONOMER",
                        f"BILN references unregistered monomer {symbol!r}",
                    )
                )
    return _result(issues)


def _biln_parser_error_code(message: str) -> str:
    if "exactly twice" in message:
        return "UNPAIRED_RING_CLOSURE"
    if "out of range" in message or "unknown chain" in message:
        return "BILN_ENDPOINT_OUT_OF_RANGE"
    if "occupied" in message or "unavailable" in message:
        return "INVALID_BILN_PORT"
    if "endpoint is reused" in message:
        return "DUPLICATE_ATTACHMENT_ENDPOINT"
    return "MALFORMED_BILN_SEQUENCE"


_MAP_CYC_TAG = re.compile(r"\{cyc:([^{}]*)\}")
_MAP_DETAILED = re.compile(
    r"\s*(N|\d+):(R[123])-(C|\d+):(R[123])\s*"
)
_MAP_SIMPLE = re.compile(r"\s*(N|\d+)-(C|\d+)\s*")


def audit_map(text: str) -> AuditResult:
    """Reject incomplete MAP cyclization tags and reused attachment ports."""

    issues: list[AuditIssue] = []
    if not text.strip():
        return _result([AuditIssue("EMPTY_INPUT", "MAP input is empty")])
    if text.count("{") != text.count("}"):
        issues.append(
            AuditIssue("UNBALANCED_MAP_BRACES", "MAP contains unbalanced braces")
        )
    tags = list(_MAP_CYC_TAG.finditer(text))
    if "{cyc:" in text and not tags:
        issues.append(
            AuditIssue(
                "MALFORMED_RING_CLOSURE",
                "MAP contains an incomplete cyclization tag",
            )
        )
    endpoints: list[tuple[str, str]] = []
    for tag in tags:
        payload = tag.group(1)
        detailed = _MAP_DETAILED.fullmatch(payload)
        simple = _MAP_SIMPLE.fullmatch(payload)
        if detailed:
            endpoints.extend(
                [
                    (detailed.group(1), detailed.group(2)),
                    (detailed.group(3), detailed.group(4)),
                ]
            )
        elif not simple:
            issues.append(
                AuditIssue(
                    "MALFORMED_RING_CLOSURE",
                    f"invalid MAP cyclization tag {tag.group(0)}",
                )
            )
    issues.extend(_duplicate_endpoints(endpoints, notation="MAP"))
    try:
        from .paths._map_utils import _parse_map_strict, map_to_helm_dict

        # Unregistered NNAA is a hard unknown-monomer error. Registered NNAA
        # tokens are parsed directly so their actual port availability is
        # included in the semantic audit.
        for token in sorted(
            {
                match.group(0)
                for match in re.finditer(r"\{nnr:[^{}]+\}", text)
                if match.group(0) not in map_to_helm_dict
            }
        ):
            issues.append(
                AuditIssue(
                    "UNKNOWN_MONOMER",
                    f"MAP references unregistered NNAA {token}",
                )
            )
        _parse_map_strict(text)
    except (TypeError, ValueError) as exc:
        issues.append(AuditIssue("INVALID_MAP_SEMANTICS", str(exc)))
    return _result(issues)


_HELM_CONNECTION = re.compile(
    r"\s*([A-Za-z0-9_]+),([A-Za-z0-9_]+),"
    r"(\d+):(R[123])-(\d+):(R[123])\s*"
)


def audit_helm(text: str) -> AuditResult:
    """Reject malformed HELM connection records before semantic conversion."""

    issues: list[AuditIssue] = []
    if not text.strip():
        return _result([AuditIssue("EMPTY_INPUT", "HELM input is empty")])
    parts = text.split("$")
    if len(parts) < 4:
        issues.append(
            AuditIssue(
                "MALFORMED_HELM_SECTIONS",
                "HELM must contain sequence, connection, group, and annotation sections",
            )
        )
        return _result(issues)
    if not re.search(r"[A-Za-z0-9_]+\{[^{}]+\}", parts[0]):
        issues.append(
            AuditIssue("MALFORMED_HELM_POLYMER", "HELM polymer block is missing")
        )
    endpoints: list[tuple[str, str]] = []
    if parts[1].strip():
        for entry in parts[1].split("|"):
            match = _HELM_CONNECTION.fullmatch(entry)
            if not match:
                issues.append(
                    AuditIssue(
                        "MALFORMED_RING_CLOSURE",
                        f"invalid HELM connection record {entry!r}",
                    )
                )
                continue
            endpoints.extend(
                [
                    (f"{match.group(1)}:{match.group(3)}", match.group(4)),
                    (f"{match.group(2)}:{match.group(5)}", match.group(6)),
                ]
            )
    issues.extend(_duplicate_endpoints(endpoints, notation="HELM"))
    try:
        from .paths._map_utils import _parse_helm_strict, _symbol_to_map

        chains, _edges = _parse_helm_strict(text)
    except (TypeError, ValueError) as exc:
        issues.append(
            AuditIssue(
                _helm_parser_error_code(str(exc)),
                str(exc),
            )
        )
        return _result(issues)
    for chain in chains:
        for symbol in chain:
            if symbol not in _symbol_to_map:
                issues.append(
                    AuditIssue(
                        "UNKNOWN_MONOMER",
                        f"HELM references unregistered monomer {symbol!r}",
                    )
                )
    return _result(issues)


def _helm_parser_error_code(message: str) -> str:
    if "all five sections" in message:
        return "MALFORMED_HELM_SECTIONS"
    if "unknown polymer" in message:
        return "HELM_UNKNOWN_POLYMER"
    if "out of range" in message or "occupied" in message or "unavailable" in message:
        return "HELM_ENDPOINT_OUT_OF_RANGE"
    if "invalid HELM connection" in message:
        return "MALFORMED_RING_CLOSURE"
    if "endpoint is reused" in message:
        return "DUPLICATE_ATTACHMENT_ENDPOINT"
    return "MALFORMED_HELM_POLYMER"


def audit_smiles(
    text: str,
    *,
    require_single_component: bool = True,
) -> AuditResult:
    """Reject unparsable, multi-component, or ambiguously mapped SMILES."""

    issues: list[AuditIssue] = []
    if not text.strip():
        return _result([AuditIssue("EMPTY_INPUT", "SMILES input is empty")])
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return _result(
            [AuditIssue("SMILES_PARSE_FAILED", "RDKit could not parse the SMILES")]
        )
    by_map: dict[int, list[Chem.ChiralType]] = defaultdict(list)
    for atom in mol.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map > 0:
            by_map[atom_map].append(atom.GetChiralTag())
    for atom_map, tags in sorted(by_map.items()):
        if len(tags) > 1:
            issues.append(
                AuditIssue(
                    "DUPLICATE_ATOM_MAP",
                    f"atom-map identifier {atom_map} occurs {len(tags)} times",
                )
            )
            defined = {
                tag
                for tag in tags
                if tag != Chem.ChiralType.CHI_UNSPECIFIED
            }
            if len(defined) > 1:
                issues.append(
                    AuditIssue(
                        "CONFLICTING_STEREO_DECLARATION",
                        f"atom-map identifier {atom_map} has incompatible "
                        "tetrahedral descriptors",
                    )
                )
    components = len(Chem.GetMolFrags(mol))
    if require_single_component and components != 1:
        issues.append(
            AuditIssue(
                "MULTI_COMPONENT_CYCLIC_PEPTIDE",
                f"cyclic-peptide identity has {components} connected components",
            )
        )
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        issues.append(
            AuditIssue(
                "DUMMY_ATOM_PRESENT",
                "molecular output contains one or more dummy atoms",
            )
        )
    return _result(issues)


_STRICT_ASSEMBLY_PORTS: dict[str, frozenset[str]] = {
    "A": frozenset(("R1", "R2")),
    "C": frozenset(("R1", "R2", "R3")),
    "F": frozenset(("R1", "R2")),
    "G": frozenset(("R1", "R2")),
    "L": frozenset(("R1", "R2")),
    "S": frozenset(("R1", "R2", "R3")),
    "T": frozenset(("R1", "R2", "R3")),
    "V": frozenset(("R1", "R2")),
}
_STRICT_ASSEMBLY_PORT_VALENCES: dict[str, int] = {
    "R1": 3,
    "R2": 4,
    "R3": 2,
}
_STRICT_ASSEMBLY_BACKBONE_CAPS: dict[str, str] = {
    "R1": "H",
    "R2": "OH",
}
_MISSING_PORT_CODES = {
    "R1": "MISSING_R1",
    "R2": "MISSING_R2",
    "R3": "MISSING_R3",
}
_ASSEMBLY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "assembly_mode",
        "component_policy",
        "declared_formal_charge",
        "monomers",
        "connections",
        "geometric_bond_candidates",
    }
)
_ASSEMBLY_MONOMER_KEYS = frozenset(
    {
        "monomer_id",
        "symbol",
        "chain_id",
        "formal_charge",
        "ports",
    }
)
_ASSEMBLY_PORT_KEYS = frozenset(
    {
        "atom_map",
        "cap",
        "stereo",
        "declared_valence",
        "maximum_valence",
    }
)
_ASSEMBLY_CONNECTION_KEYS = frozenset(
    {
        "connection_id",
        "source",
        "target",
        "bond_type",
        "bond_order",
        "role",
    }
)
_ASSEMBLY_ENDPOINT_KEYS = frozenset({"monomer_id", "port"})
_ASSEMBLY_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "source",
        "target",
        "distance_angstrom",
        "confidence",
    }
)


class _DuplicateAssemblyKey(ValueError):
    """Raised when JSON contains a key that a normal decoder would overwrite."""


def _strict_json_object(payload: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise _DuplicateAssemblyKey(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    document = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(document, dict):
        raise TypeError("assembly_json top level must be an object")
    return document


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _assembly_endpoint(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict) or set(value) != _ASSEMBLY_ENDPOINT_KEYS:
        return None
    monomer_id = value.get("monomer_id")
    port = value.get("port")
    if not isinstance(monomer_id, str) or not monomer_id:
        return None
    if not isinstance(port, str) or port not in {"R1", "R2", "R3"}:
        return None
    return monomer_id, port


def _assembly_template_profile(symbol: str) -> dict[str, Any] | None:
    """Return the construction defaults for a registered monomer.

    ``assembly_json`` currently delegates chemistry to the MAP assembler.  A
    declaration is therefore meaningful only when it agrees with the same
    registered template.  This helper intentionally reports only properties
    that are already observable from that template; it does not infer new
    stereochemistry or remap atom indices.
    """

    try:
        from .paths._map_utils import get_smi_from_map

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smiles = get_smi_from_map(symbol)
        molecule = Chem.MolFromSmiles(smiles or "")
    except Exception:
        return None
    if molecule is None:
        return None
    return {
        "formal_charge": sum(
            atom.GetFormalCharge() for atom in molecule.GetAtoms()
        ),
        "stereo": (
            "L"
            if any(
                atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                for atom in molecule.GetAtoms()
            )
            else "achiral"
        ),
    }


def _assembly_default_cap(symbol: str, port_name: str) -> str | None:
    """Return the cap the current assembler applies to an unused port."""

    if port_name in _STRICT_ASSEMBLY_BACKBONE_CAPS:
        return _STRICT_ASSEMBLY_BACKBONE_CAPS[port_name]
    try:
        from .paths._map_utils import monomers2r_groups_dict

        value = monomers2r_groups_dict.get(symbol, {}).get(port_name)
        if value is not None and str(value).strip() not in {"", "-"}:
            return str(value).strip().upper()
    except Exception:
        pass
    # The strict schema includes R3 for legacy side-chain declarations such
    # as S/T even when the live template has no separate R3 entry.  H is the
    # historical default used by those declarations.
    return "H" if port_name == "R3" else None


def audit_assembly_json(
    payload: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> AuditResult:
    """Audit the frozen ``strict_fail_closed_v1`` assembly contract.

    This validator deliberately accepts only the versioned, explicit schema
    used by the confirmatory invalid-input protocol.  It never infers a
    missing attachment point, resolves geometry, adds a cap, changes charge,
    or joins chains.  Such behavior would turn an invalid declaration into a
    silent repair.  Port ``atom_map`` values are metadata-only in this
    version: they are required to be positive, complete, and unique, but are
    not projected into the MAP/SMILES assembly because that projection is not
    implemented.
    """

    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = payload
        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(parsed),
        ):
            return audit_assembly_json(payload)
    if not isinstance(payload, str) or not payload.strip():
        return _result(
            [AuditIssue("EMPTY_INPUT", "assembly_json input is empty")]
        )
    try:
        document = _strict_json_object(payload)
    except (json.JSONDecodeError, TypeError, _DuplicateAssemblyKey) as exc:
        return _result(
            [
                AuditIssue(
                    "INVALID_ASSEMBLY_JSON",
                    f"assembly_json cannot be decoded without loss: {exc}",
                )
            ]
        )

    issues: list[AuditIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(AuditIssue(code, message))

    unexpected_top_level = sorted(set(document) - _ASSEMBLY_TOP_LEVEL_KEYS)
    missing_top_level = sorted(_ASSEMBLY_TOP_LEVEL_KEYS - set(document))
    if unexpected_top_level:
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            f"assembly_json has unsupported top-level fields {unexpected_top_level}",
        )
    if missing_top_level:
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            f"assembly_json is missing required fields {missing_top_level}",
        )
    if document.get("schema_version") != "1.0.0":
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "assembly_json schema_version must be exactly '1.0.0'",
        )
    if document.get("assembly_mode") != "strict_cyclic_peptide":
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "assembly_mode must be exactly 'strict_cyclic_peptide'",
        )
    if document.get("component_policy") != "single_covalent_entity":
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "component_policy must be exactly 'single_covalent_entity'",
        )

    monomers_raw = document.get("monomers")
    connections_raw = document.get("connections")
    candidates_raw = document.get("geometric_bond_candidates")
    if not isinstance(monomers_raw, list) or not monomers_raw:
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "monomers must be a nonempty array",
        )
        monomers_raw = []
    if not isinstance(connections_raw, list) or not connections_raw:
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "connections must be a nonempty array",
        )
        connections_raw = []
    if not isinstance(candidates_raw, list):
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "geometric_bond_candidates must be an array",
        )
        candidates_raw = []

    monomers: dict[str, dict[str, Any]] = {}
    atom_map_owners: dict[int, list[tuple[str, str]]] = defaultdict(list)
    formal_charge_sum = 0
    for index, monomer in enumerate(monomers_raw):
        if not isinstance(monomer, dict):
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer at index {index} is not an object",
            )
            continue
        extra = sorted(set(monomer) - _ASSEMBLY_MONOMER_KEYS)
        missing = sorted(_ASSEMBLY_MONOMER_KEYS - set(monomer))
        if extra or missing:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer at index {index} has extra={extra}, missing={missing}",
            )
        monomer_id = monomer.get("monomer_id")
        if not isinstance(monomer_id, str) or not monomer_id:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer at index {index} has no nonempty monomer_id",
            )
            continue
        if monomer_id in monomers:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer_id {monomer_id!r} occurs more than once",
            )
            continue
        monomers[monomer_id] = monomer

        symbol = monomer.get("symbol")
        required_ports = _STRICT_ASSEMBLY_PORTS.get(str(symbol))
        if required_ports is None:
            add(
                "UNKNOWN_MONOMER",
                f"monomer {monomer_id!r} has unsupported symbol {symbol!r}",
            )
            required_ports = frozenset()
        template_profile = _assembly_template_profile(str(symbol))

        chain_id = monomer.get("chain_id")
        if not isinstance(chain_id, str) or not chain_id:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer {monomer_id!r} has no nonempty chain_id",
            )

        formal_charge = monomer.get("formal_charge")
        if not isinstance(formal_charge, int) or isinstance(formal_charge, bool):
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer {monomer_id!r} formal_charge must be an integer",
            )
        else:
            formal_charge_sum += formal_charge
            if (
                template_profile is not None
                and formal_charge != template_profile["formal_charge"]
            ):
                add(
                    "MONOMER_FORMAL_CHARGE_UNSUPPORTED",
                    f"monomer {monomer_id!r} declares formal_charge="
                    f"{formal_charge}, but the registered {symbol!r} template "
                    f"constructs with formal_charge="
                    f"{template_profile['formal_charge']}",
                )

        ports = monomer.get("ports")
        if not isinstance(ports, dict):
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer {monomer_id!r} ports must be an object",
            )
            ports = {}
        for required_port in sorted(required_ports):
            if required_port not in ports:
                add(
                    _MISSING_PORT_CODES[required_port],
                    f"monomer {monomer_id!r} is missing required {required_port}",
                )
        unexpected_ports = sorted(set(ports) - required_ports)
        if unexpected_ports:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"monomer {monomer_id!r} declares unsupported ports "
                f"{unexpected_ports}",
            )
        for port_name, port in ports.items():
            if not isinstance(port, dict):
                add(
                    "INVALID_ASSEMBLY_SCHEMA",
                    f"{monomer_id}:{port_name} declaration is not an object",
                )
                continue
            extra = sorted(set(port) - _ASSEMBLY_PORT_KEYS)
            missing = sorted(_ASSEMBLY_PORT_KEYS - set(port))
            if extra or missing:
                add(
                    "INVALID_ASSEMBLY_SCHEMA",
                    f"{monomer_id}:{port_name} has extra={extra}, missing={missing}",
                )
            atom_map = port.get("atom_map")
            if not _positive_int(atom_map):
                add(
                    "ATOM_MAPPING_MISSING",
                    f"{monomer_id}:{port_name} has no positive integer atom_map",
                )
            else:
                atom_map_owners[int(atom_map)].append((monomer_id, port_name))
            declared_valence = port.get("declared_valence")
            maximum_valence = port.get("maximum_valence")
            if (
                not _positive_int(declared_valence)
                or not _positive_int(maximum_valence)
                or int(declared_valence) > int(maximum_valence)
            ):
                add(
                    "INVALID_ASSEMBLY_SCHEMA",
                    f"{monomer_id}:{port_name} has an invalid valence declaration",
                )
            expected_valence = _STRICT_ASSEMBLY_PORT_VALENCES.get(port_name)
            if (
                expected_valence is not None
                and _positive_int(declared_valence)
                and _positive_int(maximum_valence)
                and (
                    int(declared_valence) != expected_valence
                    or int(maximum_valence) != expected_valence
                )
            ):
                add(
                    "DECLARED_VALENCE_UNSUPPORTED",
                    f"{monomer_id}:{port_name} declares valence "
                    f"{declared_valence}/{maximum_valence}, but the current "
                    f"assembler supports only {expected_valence}/{expected_valence}",
                )
            cap = port.get("cap")
            if cap is not None and not isinstance(cap, str):
                add(
                    "INVALID_ASSEMBLY_SCHEMA",
                    f"{monomer_id}:{port_name} cap must be null or a string",
                )
            stereo = port.get("stereo")
            if stereo not in {"L", "D", "achiral", "unspecified"}:
                add(
                    "INVALID_ASSEMBLY_SCHEMA",
                    f"{monomer_id}:{port_name} has unsupported stereo {stereo!r}",
                )
            elif (
                template_profile is not None
                and stereo != template_profile["stereo"]
            ):
                add(
                    "UNSUPPORTED_STEREO_DECLARATION",
                    f"{monomer_id}:{port_name} declares stereo={stereo!r}, "
                    f"but the registered {symbol!r} template supports only "
                    f"stereo={template_profile['stereo']!r}",
                )

    for atom_map, owners in sorted(atom_map_owners.items()):
        if len(owners) > 1:
            add(
                "ATOM_MAPPING_CONFLICT",
                f"atom_map {atom_map} is assigned to multiple ports {owners}",
            )

    declared_formal_charge = document.get("declared_formal_charge")
    if (
        not isinstance(declared_formal_charge, int)
        or isinstance(declared_formal_charge, bool)
    ):
        add(
            "INVALID_ASSEMBLY_SCHEMA",
            "declared_formal_charge must be an integer",
        )
    elif declared_formal_charge != formal_charge_sum:
        add(
            "FORMAL_CHARGE_CONFLICT",
            "declared assembly formal charge "
            f"{declared_formal_charge} differs from monomer sum "
            f"{formal_charge_sum}",
        )

    endpoint_use: dict[tuple[str, str], list[str]] = defaultdict(list)
    oriented_backbone: list[tuple[str, str]] = []
    amide_edges: list[tuple[str, str, str, str]] = []
    connection_ids: set[str] = set()
    for index, connection in enumerate(connections_raw):
        if not isinstance(connection, dict):
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"connection at index {index} is not an object",
            )
            continue
        extra = sorted(set(connection) - _ASSEMBLY_CONNECTION_KEYS)
        missing = sorted(_ASSEMBLY_CONNECTION_KEYS - set(connection))
        if extra or missing:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"connection at index {index} has extra={extra}, missing={missing}",
            )
        connection_id = connection.get("connection_id")
        if not isinstance(connection_id, str) or not connection_id:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"connection at index {index} has no nonempty connection_id",
            )
            connection_id = f"index-{index}"
        elif connection_id in connection_ids:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"connection_id {connection_id!r} occurs more than once",
            )
        connection_ids.add(connection_id)

        source = _assembly_endpoint(connection.get("source"))
        target = _assembly_endpoint(connection.get("target"))
        if source is None or target is None:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"connection {connection_id!r} has a malformed endpoint",
            )
            continue
        if source[0] not in monomers or target[0] not in monomers:
            add(
                "ENDPOINT_OUT_OF_RANGE",
                f"connection {connection_id!r} references an unknown monomer",
            )
            continue
        if source[0] == target[0]:
            add(
                "SELF_CONNECTION",
                f"connection {connection_id!r} joins monomer {source[0]!r} "
                "to itself",
            )

        endpoint_is_declared = True
        for endpoint in (source, target):
            ports = monomers[endpoint[0]].get("ports")
            if not isinstance(ports, dict) or endpoint[1] not in ports:
                endpoint_is_declared = False
                required = _STRICT_ASSEMBLY_PORTS.get(
                    str(monomers[endpoint[0]].get("symbol")),
                    frozenset(),
                )
                if endpoint[1] not in required:
                    add(
                        "ENDPOINT_OUT_OF_RANGE",
                        f"connection {connection_id!r} references unavailable "
                        f"endpoint {endpoint[0]}:{endpoint[1]}",
                    )
        if endpoint_is_declared:
            endpoint_use[source].append(connection_id)
            if target != source:
                endpoint_use[target].append(connection_id)

        bond_type = connection.get("bond_type")
        bond_order = connection.get("bond_order")
        single_bond = (
            isinstance(bond_order, int)
            and not isinstance(bond_order, bool)
            and bond_order == 1
        )
        source_symbol = str(monomers[source[0]].get("symbol"))
        target_symbol = str(monomers[target[0]].get("symbol"))
        amide = (
            bond_type == "amide"
            and single_bond
            and (source[1], target[1]) in {("R2", "R1"), ("R1", "R2")}
        )
        disulfide = (
            bond_type == "disulfide"
            and single_bond
            and source[1] == target[1] == "R3"
            and source_symbol == target_symbol == "C"
        )
        if not (amide or disulfide):
            add(
                "INCOMPATIBLE_BOND_TYPE",
                f"connection {connection_id!r} has bond_type={bond_type!r}, "
                f"bond_order={bond_order!r} for {source[1]}-{target[1]}",
            )
        elif amide:
            if source[1] == "R2":
                oriented = (source[0], target[0])
            else:
                oriented = (target[0], source[0])
            oriented_backbone.append(oriented)
            amide_edges.append(
                (
                    connection_id,
                    oriented[0],
                    oriented[1],
                    str(connection.get("role")),
                )
            )

        source_chain = monomers[source[0]].get("chain_id")
        target_chain = monomers[target[0]].get("chain_id")
        if source_chain != target_chain and connection.get("role") != "interchain":
            add(
                "MULTICHAIN_WRONG_LINK",
                f"connection {connection_id!r} crosses chain "
                f"{source_chain!r}->{target_chain!r} without interchain role",
            )
        if connection.get("role") not in {"backbone", "ring", "interchain"}:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"connection {connection_id!r} has unsupported role "
                f"{connection.get('role')!r}",
            )
        elif source_chain == target_chain and connection.get("role") == "interchain":
            add(
                "CONNECTION_ROLE_CONFLICT",
                f"connection {connection_id!r} is marked interchain but both "
                f"endpoints belong to chain {source_chain!r}",
            )
        elif amide and connection.get("role") not in {"backbone", "ring"}:
            add(
                "CONNECTION_ROLE_CONFLICT",
                f"amide connection {connection_id!r} must be marked backbone "
                "or ring",
            )
        elif disulfide and connection.get("role") != "ring":
            add(
                "CONNECTION_ROLE_CONFLICT",
                f"disulfide connection {connection_id!r} must be marked ring",
            )

    chain_ids = {
        row.get("chain_id")
        for row in monomers.values()
        if isinstance(row.get("chain_id"), str)
    }
    if len(chain_ids) > 1:
        add(
            "MULTICHAIN_WRONG_LINK",
            "strict single-entity assembly declares more than one peptide chain",
        )

    for endpoint, uses in sorted(endpoint_use.items()):
        if len(uses) > 1:
            add(
                "PORT_REUSE",
                f"attachment endpoint {endpoint[0]}:{endpoint[1]} is reused "
                f"by connections {uses}",
            )
        port = monomers[endpoint[0]].get("ports", {}).get(endpoint[1])
        if isinstance(port, dict) and port.get("cap") not in (None, ""):
            add(
                "CAP_INTERNAL_USE",
                f"connected endpoint {endpoint[0]}:{endpoint[1]} still has "
                f"cap {port.get('cap')!r}",
            )

    # Free ports are capped by the MAP assembler using the registered
    # monomer defaults.  A different declaration would be silently ignored
    # by ``_assembly_map_payload`` and therefore must be rejected explicitly.
    for monomer_id, monomer in monomers.items():
        ports = monomer.get("ports")
        if not isinstance(ports, dict):
            continue
        symbol = str(monomer.get("symbol"))
        for port_name, port in ports.items():
            if not isinstance(port, dict):
                continue
            endpoint = (monomer_id, port_name)
            if endpoint in endpoint_use:
                continue
            expected_cap = _assembly_default_cap(symbol, port_name)
            if expected_cap is None:
                continue
            cap = port.get("cap")
            if not isinstance(cap, str) or cap.strip().upper() != expected_cap:
                add(
                    "UNUSED_PORT_CAP_UNSUPPORTED",
                    f"unused endpoint {monomer_id}:{port_name} declares cap "
                    f"{cap!r}, but the current assembler applies "
                    f"{expected_cap!r}",
                )

    successor: dict[str, str] = {}
    predecessor: dict[str, str] = {}
    for source_id, target_id in oriented_backbone:
        if source_id in successor and successor[source_id] != target_id:
            add(
                "INVALID_ASSEMBLY_TOPOLOGY",
                f"monomer {source_id!r} has multiple backbone successors",
            )
        if target_id in predecessor and predecessor[target_id] != source_id:
            add(
                "INVALID_ASSEMBLY_TOPOLOGY",
                f"monomer {target_id!r} has multiple backbone predecessors",
            )
        successor[source_id] = target_id
        predecessor[target_id] = source_id
    if monomers and (
        len(oriented_backbone) != len(monomers)
        or set(successor) != set(monomers)
        or set(predecessor) != set(monomers)
    ):
        add(
            "INVALID_ASSEMBLY_TOPOLOGY",
            "R1/R2 backbone declarations do not form a complete head-to-tail cycle",
        )
    elif monomers:
        start = next(iter(monomers))
        visited: set[str] = set()
        current = start
        for _ in range(len(monomers)):
            if current in visited or current not in successor:
                break
            visited.add(current)
            current = successor[current]
        if current != start or visited != set(monomers):
            add(
                "INVALID_ASSEMBLY_TOPOLOGY",
                "R1/R2 backbone contains disconnected or non-cyclic components",
            )

    # The assembler treats every amide edge as part of the same directed
    # backbone cycle and has no independent branch/role implementation.  A
    # strict cyclic declaration therefore has exactly one amide ``ring`` edge
    # closing the directed path; all other amide edges are ``backbone``.
    if len(amide_edges) == len(monomers) and monomers:
        ring_edges = [edge for edge in amide_edges if edge[3] == "ring"]
        if len(ring_edges) != 1:
            add(
                "CONNECTION_ROLE_CONFLICT",
                "strict cyclic assembly requires exactly one amide ring "
                f"closure, found {len(ring_edges)}",
            )
        else:
            _, ring_source, ring_target, _ = ring_edges[0]
            remaining = [
                (source_id, target_id)
                for connection_id, source_id, target_id, role in amide_edges
                if connection_id != ring_edges[0][0]
            ]
            remaining_successor = {
                source_id: target_id for source_id, target_id in remaining
            }
            remaining_predecessor = {
                target_id: source_id for source_id, target_id in remaining
            }
            starts = set(monomers) - set(remaining_predecessor)
            ends = set(monomers) - set(remaining_successor)
            valid_path = len(starts) == 1 and len(ends) == 1
            if valid_path:
                path_start = next(iter(starts))
                path_end = next(iter(ends))
                current = path_start
                visited = set()
                for _ in range(len(monomers)):
                    if current in visited:
                        break
                    visited.add(current)
                    if current not in remaining_successor:
                        break
                    current = remaining_successor[current]
                valid_path = (
                    visited == set(monomers)
                    and current == path_end
                    and ring_source == path_end
                    and ring_target == path_start
                )
            if not valid_path:
                add(
                    "CONNECTION_ROLE_CONFLICT",
                    "amide ring role does not close the backbone path; "
                    "branch or internal-ring semantics are not implemented",
                )

    candidates_by_endpoint: dict[
        tuple[str, str], list[tuple[tuple[str, str], Any, Any]]
    ] = defaultdict(list)
    for index, candidate in enumerate(candidates_raw):
        if not isinstance(candidate, dict):
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"geometric candidate at index {index} is not an object",
            )
            continue
        extra = sorted(set(candidate) - _ASSEMBLY_CANDIDATE_KEYS)
        missing = sorted(_ASSEMBLY_CANDIDATE_KEYS - set(candidate))
        if extra or missing:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"geometric candidate at index {index} has "
                f"extra={extra}, missing={missing}",
            )
        source = _assembly_endpoint(candidate.get("source"))
        target = _assembly_endpoint(candidate.get("target"))
        if source is None or target is None:
            add(
                "INVALID_ASSEMBLY_SCHEMA",
                f"geometric candidate at index {index} has a malformed endpoint",
            )
            continue
        score = (
            candidate.get("distance_angstrom"),
            candidate.get("confidence"),
        )
        candidates_by_endpoint[source].append((target, *score))
        candidates_by_endpoint[target].append((source, *score))

    ambiguous = False
    for endpoint, candidates in candidates_by_endpoint.items():
        partners = {row[0] for row in candidates}
        scores = {(row[1], row[2]) for row in candidates}
        if len(partners) > 1 and len(scores) == 1:
            ambiguous = True
            add(
                "MULTIPLE_AMBIGUOUS_GEOMETRIC_BONDS",
                f"endpoint {endpoint[0]}:{endpoint[1]} has "
                f"{len(partners)} equally scored geometric partners",
            )
    if candidates_raw and not ambiguous:
        add(
            "UNRESOLVED_GEOMETRIC_BOND",
            "strict assembly_json cannot infer a bond from geometric candidates",
        )

    return _result(issues)


def _pdb_atom_key(line: str, *, second: bool = False) -> tuple[str, str, str, str]:
    if second:
        return (
            line[42:46].strip(),
            line[51:52].strip(),
            line[52:56].strip(),
            line[56:57].strip(),
        )
    return (
        line[12:16].strip(),
        line[21:22].strip(),
        line[22:26].strip(),
        line[26:27].strip(),
    )


def audit_pdb_text(text: str) -> AuditResult:
    """Detect mutually exclusive inter-residue LINK and CONECT evidence.

    CONECT is allowed to be a subset or superset of LINK.  Rejection occurs
    only when the same LINK endpoint has a different cross-residue CONECT
    partner, which avoids treating ordinary intra-residue CONECT bonds as a
    conflict.
    """

    issues: list[AuditIssue] = []
    from .core.pdb_utils import first_model_records

    lines = list(first_model_records(text.splitlines()))
    serial_by_key: dict[tuple[str, str, str, str], int] = {}
    residue_by_serial: dict[int, tuple[str, str, str]] = {}
    atom_record_count = 0
    malformed_atom_record_count = 0
    duplicate_serials = set()
    duplicate_identities = set()
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_record_count += 1
        try:
            serial = int(line[6:11])
            int(line[22:26])
            coordinates = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("non-finite atom coordinate")
            if not line[12:16].strip() or not line[17:20].strip():
                raise ValueError("missing atom or residue name")
        except (IndexError, ValueError):
            malformed_atom_record_count += 1
            continue
        key = _pdb_atom_key(line)
        if serial in residue_by_serial:
            duplicate_serials.add(serial)
        if key in serial_by_key:
            duplicate_identities.add(key)
        serial_by_key[key] = serial
        residue_by_serial[serial] = (key[1], key[2], key[3])

    if malformed_atom_record_count:
        issues.append(
            AuditIssue(
                "MALFORMED_PDB_ATOM_RECORD",
                f"PDB contains {malformed_atom_record_count} malformed "
                "ATOM/HETATM record(s)",
            )
        )
    if duplicate_serials:
        issues.append(AuditIssue(
            "DUPLICATE_PDB_ATOM_SERIAL",
            f"PDB repeats atom serials {sorted(duplicate_serials)}",
        ))
    if duplicate_identities:
        issues.append(AuditIssue(
            "DUPLICATE_PDB_ATOM_IDENTITY",
            "PDB repeats one or more chain/residue/atom identities",
        ))
    if not residue_by_serial:
        issues.append(
            AuditIssue(
                "NO_VALID_PDB_ATOMS",
                "PDB contains no valid ATOM/HETATM coordinate records"
                if atom_record_count
                else "PDB contains no ATOM/HETATM coordinate records",
            )
        )

    link_edges: set[tuple[int, int]] = set()
    link_partners: dict[int, set[int]] = defaultdict(set)
    for line in lines:
        if not line.startswith("LINK"):
            continue
        first = serial_by_key.get(_pdb_atom_key(line))
        second = serial_by_key.get(_pdb_atom_key(line, second=True))
        if first is None or second is None:
            issues.append(
                AuditIssue(
                    "UNRESOLVED_PDB_LINK",
                    "LINK record cannot be resolved to two selected-chain atoms",
                )
            )
            continue
        edge = tuple(sorted((first, second)))
        link_edges.add(edge)
        link_partners[first].add(second)
        link_partners[second].add(first)

    for serial, partners in sorted(link_partners.items()):
        if len(partners) > 1:
            issues.append(
                AuditIssue(
                    "PDB_LINK_CONFLICT",
                    f"PDB atom serial {serial} has multiple LINK partners "
                    f"{sorted(partners)}",
                )
            )

    cross_conect: dict[int, set[int]] = defaultdict(set)
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        try:
            serials = [int(token) for token in line.split()[1:]]
        except ValueError:
            issues.append(AuditIssue(
                "MALFORMED_PDB_CONECT_RECORD",
                "CONECT record contains a non-integer atom serial",
            ))
            continue
        if len(serials) < 2:
            issues.append(AuditIssue(
                "MALFORMED_PDB_CONECT_RECORD",
                "CONECT record contains fewer than two atom serials",
            ))
            continue
        source = serials[0]
        for partner in serials[1:]:
            if source == partner:
                issues.append(AuditIssue(
                    "PDB_SELF_CONNECTION",
                    f"CONECT contains self-connection for serial {source}",
                ))
                continue
            missing = [
                serial for serial in (source, partner)
                if serial not in residue_by_serial
            ]
            if missing:
                issues.append(AuditIssue(
                    "UNRESOLVED_PDB_CONECT",
                    f"CONECT references absent atom serials {sorted(set(missing))}",
                ))
                continue
            if residue_by_serial[source] != residue_by_serial[partner]:
                cross_conect[source].add(partner)
                cross_conect[partner].add(source)

    reported: set[tuple[int, int, int]] = set()
    for first, second in sorted(link_edges):
        for endpoint, expected in ((first, second), (second, first)):
            alternatives = cross_conect.get(endpoint, set()) - {expected}
            for alternative in sorted(alternatives):
                marker = (endpoint, expected, alternative)
                if marker in reported:
                    continue
                reported.add(marker)
                issues.append(
                    AuditIssue(
                        "PDB_LINK_CONECT_CONFLICT",
                        f"PDB atom serial {endpoint} links to {expected} in LINK "
                        f"but to {alternative} in cross-residue CONECT",
                    )
                )
    return _result(issues)


def audit_pdb_file(path: str | Path) -> AuditResult:
    path = Path(path)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return audit_pdb_text(handle.read())
    return audit_pdb_text(path.read_text(encoding="utf-8", errors="replace"))


def audit_payload(
    kind: str,
    payload: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> AuditResult:
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        normalized_kind = kind.strip().lower()
        hint_payload: Any = payload
        if normalized_kind in {"assembly_json", "assembly-json"}:
            try:
                hint_payload = json.loads(payload)
            except Exception:
                pass
        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                hint_payload, kind=normalized_kind
            ),
        ):
            return audit_payload(kind, payload)
    normalized = kind.strip().lower()
    if normalized in {"assembly_json", "assembly-json"}:
        return audit_assembly_json(payload)
    if normalized == "biln":
        return audit_biln(payload)
    if normalized == "map":
        return audit_map(payload)
    if normalized == "helm":
        return audit_helm(payload)
    if normalized in {"smiles", "smi"}:
        return audit_smiles(payload)
    if normalized == "pdb":
        return audit_pdb_text(payload)
    return _result(
        [
            AuditIssue(
                "NOT_SUPPORTED",
                f"no audited cyclic-peptide validator for input format {kind!r}",
            )
        ]
    )


def audit_output_smiles(text: str | None) -> AuditResult:
    if text is None:
        return _result(
            [AuditIssue("NO_MOLECULAR_OUTPUT", "no molecular output was produced")]
        )
    return audit_smiles(text, require_single_component=True)
