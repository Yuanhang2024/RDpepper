"""Sequence input adapter for the unified V5 artifact pipeline."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .core.applicability import evaluate_sequence_applicability
from .core.monomer_resolution import needs_monomer_resolution_scope
from .core.artifacts import (
    ArtifactStatus,
    ArtifactType,
    ChemicalGraphArtifact,
    ChemicalLevel,
    ClaimBoundary,
    CoordinateLevel,
    CoordinateOrigin,
    EvidenceBasis,
    EvidenceProfile,
    FlexibilityLevel,
    FormatLevel,
    InputArtifact,
    artifact_payload_sha256,
    make_artifact_id,
    validate_artifact_identity,
)
from .exact_v1 import (
    ABSTAIN,
    EXACT,
    ExactV1Error,
    MonomerRegistry,
    exact_v1_to_smiles,
    map_to_exact_v1,
)
from .paths import _map_utils


SEQUENCE_SCHEMA_VERSION = "1.0.0-cycpep-v5-sequence.1"
_PLAIN_AA = frozenset("ARNDCEQGHILKMFPSTWYV")


def _sequence_tokens(sequence: str) -> list[str]:
    text = str(sequence or "").strip()
    if not text:
        raise ValueError("sequence must be nonempty")
    tokens = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace() or character == "-":
            index += 1
            continue
        if text.startswith("{nnr:", index):
            end = text.find("}", index)
            if end < 0:
                raise ValueError("unclosed {nnr:...} sequence token")
            symbol = text[index + 5:end].strip()
            if not symbol:
                raise ValueError("empty {nnr:...} sequence token")
            tokens.append(symbol)
            index = end + 1
            continue
        if character == "[":
            end = text.find("]", index)
            if end < 0:
                raise ValueError("unclosed [monomer] sequence token")
            symbol = text[index + 1:end].strip()
            if not symbol:
                raise ValueError("empty [monomer] sequence token")
            tokens.append(symbol)
            index = end + 1
            continue
        if character in _PLAIN_AA:
            tokens.append(character)
            index += 1
            continue
        raise ValueError(
            f"unsupported sequence character at position {index + 1}: "
            f"{character!r}"
        )
    if not tokens:
        raise ValueError("sequence contains no monomers")
    return tokens


def _apply_stereochemistry(
    symbols: list[str],
    stereochemistry: Mapping[int | str, str] | None,
) -> list[str]:
    if not stereochemistry:
        return symbols
    registry = MonomerRegistry()
    result = list(symbols)
    for raw_position, value in stereochemistry.items():
        position = int(raw_position)
        if position < 1 or position > len(result):
            raise ValueError(
                f"stereochemistry position {position} is out of range"
            )
        directive = str(value).strip()
        if directive.upper() == "L":
            continue
        if directive.upper() == "D":
            candidate = "d" + result[position - 1]
        else:
            candidate = directive
        definition = registry.resolve(candidate)
        result[position - 1] = definition.symbol
    return result


def _canonicalize_symbols(symbols: Sequence[str]) -> list[str]:
    """Resolve aliases through the active operation-local registry."""
    registry = MonomerRegistry()
    resolved = []
    for symbol in symbols:
        try:
            resolved.append(registry.resolve(symbol).symbol)
        except Exception:
            resolved.append(symbol)
    return resolved


def _terminal_tags(
    modifications: Mapping[str, str] | None,
) -> list[str]:
    if not modifications:
        return []
    tags = []
    normalized = {
        str(key).strip().upper(): str(value).strip().upper()
        for key, value in modifications.items()
    }
    if "N" in normalized:
        if normalized["N"] != "ACE":
            raise ValueError("supported N-terminal modification is ACE")
        tags.append("{nt:ACE}")
    if "C" in normalized:
        if normalized["C"] not in {"NME", "NH2"}:
            raise ValueError(
                "supported C-terminal modifications are NME and NH2"
            )
        tags.append(f"{{ct:{normalized['C']}}}")
    return tags


def _endpoint(value: Mapping[str, Any]) -> tuple[int, str]:
    position = int(value["position"])
    port = str(value["port"]).strip().upper()
    if port not in {"R1", "R2", "R3"}:
        raise ValueError(f"unsupported cyclization port: {port}")
    return position, port


def _cyclization_tags(
    cyclization: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    residue_count: int,
) -> tuple[list[str], str, bool, list[str]]:
    if isinstance(cyclization, str):
        mode = cyclization.strip().lower().replace("_", "-")
        if mode in {"head-to-tail", "headtotail", "ht", "n-c"}:
            return ["{cyc:N-C}"], "head_to_tail", False, ["HT"]
        if mode in {"none", "linear"}:
            return [], "linear", False, []
        if mode in {"infer", "auto"}:
            return ["{cyc:N-C}"], "head_to_tail", True, ["HT"]
        raise ValueError(
            "cyclization string must be head-to-tail, linear, or infer"
        )
    rows = (
        [cyclization]
        if isinstance(cyclization, Mapping)
        else list(cyclization)
    )
    if not rows:
        return [], "linear", False, []
    tags = []
    types = []
    expected_bond_types = []
    type_aliases = {
        "HT": "HT",
        "HEAD_TO_TAIL": "HT",
        "SS": "SS",
        "DISULFIDE": "SS",
        "ISOPEPTIDE": "ISOPEPTIDE",
        "ESTER": "ESTER",
        "THIOETHER": "THIOETHER",
        "SIDECHAIN_TO_TAIL": "SIDECHAIN_TO_TAIL",
    }
    topology_names = {
        "HT": "head_to_tail",
        "SS": "disulfide",
        "ISOPEPTIDE": "isopeptide",
        "ESTER": "ester",
        "THIOETHER": "thioether",
        "SIDECHAIN_TO_TAIL": "sidechain_to_tail",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("cyclization rows must be objects")
        left = _endpoint(row["src"])
        right = _endpoint(row["dst"])
        for position, _port in (left, right):
            if position < 1 or position > residue_count:
                raise ValueError(
                    f"cyclization position {position} is out of range"
                )
        tags.append(
            f"{{cyc:{left[0]}:{left[1]}-"
            f"{right[0]}:{right[1]}}}"
        )
        raw_type = str(row.get("bond_type") or "").strip().upper()
        normalized_type = type_aliases.get(raw_type)
        if normalized_type is None:
            raise ValueError(
                "explicit cyclization bond_type must be one of "
                + ", ".join(sorted(type_aliases))
            )
        types.append(topology_names[normalized_type])
        expected_bond_types.append(normalized_type)
    topology = types[0] if len(set(types)) == 1 else "mixed"
    return tags, topology, False, expected_bond_types


def _map_sequence(symbols: Sequence[str]) -> str:
    return "".join(
        _map_utils._symbol_to_map.get(
            symbol, _map_utils._auto_map_denotion(symbol)
        )
        for symbol in symbols
    )


def _input_artifact(
    payload: Mapping[str, Any],
) -> InputArtifact:
    payload_hash = artifact_payload_sha256(payload)
    artifact_id = make_artifact_id(
        ArtifactType.INPUT, (), payload
    )
    return InputArtifact(
        artifact_type=ArtifactType.INPUT,
        artifact_id=artifact_id,
        parent_artifact_ids=(),
        status=ArtifactStatus.MATERIALIZED,
        payload_sha256=payload_hash,
        evidence=EvidenceProfile(),
        claim_boundary=ClaimBoundary(
            allowed=("input provenance",),
            forbidden=("chemical identity", "3D accuracy"),
        ),
        provenance={
            "adapter": "build_molecule_from_sequence",
            "schema_version": SEQUENCE_SCHEMA_VERSION,
        },
        input_kind="sequence",
        source_sha256=hashlib.sha256(
            str(payload["sequence"]).encode("utf-8")
        ).hexdigest(),
        source_value=str(payload["sequence"]),
    )


def build_molecule_from_sequence(
    sequence: str,
    *,
    cyclization: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    stereochemistry: Mapping[int | str, str] | None = None,
    terminal_modifications: Mapping[str, str] | None = None,
    protonation: str = "registry_default",
    monomer_context: Mapping[str, Any] | None = None,
    _resolution_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a typed ChemicalGraphArtifact without generating coordinates."""
    raw_symbols = _sequence_tokens(sequence)
    if (
        _resolution_ledger is None
        and needs_monomer_resolution_scope(monomer_context)
    ):
        from .core.monomer_resolution import (
            monomer_resolution_context,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=raw_symbols,
        ) as ledger:
            return build_molecule_from_sequence(
                sequence,
                cyclization=cyclization,
                stereochemistry=stereochemistry,
                terminal_modifications=terminal_modifications,
                protonation=protonation,
                _resolution_ledger=(
                    ledger if monomer_context is not None else {}
                ),
            )
    if _resolution_ledger is None:
        from .core.monomer_resolution import active_monomer_resolution

        _resolution_ledger = active_monomer_resolution()
    resolution_snapshot = copy.deepcopy(
        dict(_resolution_ledger or {})
    )
    symbols = _canonicalize_symbols(_apply_stereochemistry(
        raw_symbols, stereochemistry
    ))
    (
        tags,
        topology_class,
        inferred,
        expected_bond_types,
    ) = _cyclization_tags(
        cyclization, residue_count=len(symbols)
    )
    terminal_tags = _terminal_tags(terminal_modifications)
    map_text = _map_sequence(symbols) + "".join(tags + terminal_tags)
    request_payload = {
        "sequence": sequence,
        "symbols": symbols,
        "cyclization": cyclization,
        "stereochemistry": {
            str(key): str(value)
            for key, value in (stereochemistry or {}).items()
        },
        "terminal_modifications": {
            str(key): str(value)
            for key, value in (
                terminal_modifications or {}
            ).items()
        },
        "protonation": str(protonation),
        "map": map_text,
        "monomer_resolution": resolution_snapshot,
    }
    input_artifact = _input_artifact(request_payload)
    exact = map_to_exact_v1(map_text)
    observed_explicit_types = sorted(
        str(row.get("bond_type"))
        for row in exact.get("bonds", [])
        if row.get("bond_type") != "PEPTIDE"
    )
    if (
        exact.get("exactness_status") == EXACT
        and sorted(expected_bond_types) != observed_explicit_types
    ):
        exact = dict(exact)
        exact["exactness_status"] = ABSTAIN
        exact["reason_codes"] = list(dict.fromkeys([
            *list(exact.get("reason_codes") or []),
            "EXPLICIT_BOND_TYPE_MISMATCH",
        ]))
    warnings = []
    if inferred:
        warnings.append("CYCLIZATION_HYPOTHESIS_HEAD_TO_TAIL")
    if (
        resolution_snapshot
        and resolution_snapshot.get("status") == "partial"
    ):
        warnings.append("MONOMER_RESOLUTION_PARTIAL")
    contains_nnaa = any(len(symbol) > 1 for symbol in symbols)
    applicability = evaluate_sequence_applicability(
        residue_count=len(symbols),
        topology_class=topology_class,
        contains_nnaa=contains_nnaa,
    )
    warnings.extend(applicability["warnings"])

    smiles = None
    inchikey = None
    parent_inchikey = None
    formal_charge = None
    error = None
    microstate_inferred = False
    microstate_report = None
    if exact["exactness_status"] == EXACT:
        try:
            smiles = exact_v1_to_smiles(exact)
            parent_molecule = Chem.MolFromSmiles(smiles)
            if parent_molecule is None:
                raise ValueError(
                    "assembled sequence parent SMILES is invalid"
                )
            parent_inchikey = Chem.MolToInchiKey(parent_molecule)
            policy = str(protonation).strip().lower()
            if policy in {"registry_default", "preserve"}:
                pass
            elif policy in {"physiological", "ph7.4", "ph74"}:
                from .docking.protonation import protonate_molecule_ph74

                assigned, microstate_report = protonate_molecule_ph74(parent_molecule)
                smiles = Chem.MolToSmiles(assigned)
                microstate_inferred = True
                warnings.append("PHYSIOLOGICAL_MICROSTATE_HEURISTIC")
                warnings.extend(microstate_report.get("warnings", []))
            else:
                raise ValueError(
                    "protonation must be registry_default or physiological"
                )
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("assembled sequence SMILES is invalid")
            smiles = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )
            inchikey = Chem.MolToInchiKey(molecule)
            formal_charge = int(Chem.GetFormalCharge(molecule))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = ", ".join(exact["reason_codes"])

    materializable = bool(smiles and inchikey and error is None)
    if materializable:
        resolution_rigor = str(
            resolution_snapshot.get("chemical_rigor") or ""
        )
        chemical_level = (
            ChemicalLevel.C2
            if inferred
            or microstate_inferred
            or resolution_rigor == "C2:H"
            else ChemicalLevel.C3
        )
        if (
            inferred
            or microstate_inferred
            or resolution_rigor == "C2:H"
        ):
            basis = EvidenceBasis.HYPOTHESIS
        elif (
            resolution_snapshot
            and resolution_snapshot.get("chemical_rigor") == "C3:Q"
        ):
            basis = EvidenceBasis.QUALIFIED
        else:
            basis = EvidenceBasis.SPECIFIED
    else:
        chemical_level = ChemicalLevel.C1
        basis = EvidenceBasis.HYPOTHESIS
    evidence = EvidenceProfile(
        chemical_level=chemical_level,
        chemical_basis=basis,
    )
    graph_payload = {
        "exact_v1": exact,
        "smiles": smiles,
        "full_inchikey": inchikey,
        "formal_charge": formal_charge,
        "topology_class": topology_class,
        "microstate_policy": protonation,
        "parent_full_inchikey": parent_inchikey,
        "exact_v1_identity_match": (
            parent_inchikey == inchikey
            if parent_inchikey and inchikey
            else None
        ),
        "materializable": materializable,
        "symbolic_graph": {
            "symbols": list(symbols),
            "cyclization": cyclization,
            "terminal_modifications": dict(
                terminal_modifications or {}
            ),
            "topology_class": topology_class,
        },
        "monomer_resolution": resolution_snapshot,
    }
    payload_hash = artifact_payload_sha256(graph_payload)
    artifact_id = make_artifact_id(
        ArtifactType.CHEMICAL_GRAPH,
        (input_artifact.artifact_id,),
        graph_payload,
    )
    chemical_artifact = ChemicalGraphArtifact(
        artifact_type=ArtifactType.CHEMICAL_GRAPH,
        artifact_id=artifact_id,
        parent_artifact_ids=(input_artifact.artifact_id,),
        status=(
            ArtifactStatus.MATERIALIZED
            if materializable
            else ArtifactStatus.PARTIAL
        ),
        payload_sha256=payload_hash,
        evidence=evidence,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance={
            "adapter": "sequence_to_exact_v1",
            "map": map_text,
            "applicability": applicability,
            "error": error,
            "symbolic_graph": graph_payload["symbolic_graph"],
            "monomer_resolution": resolution_snapshot,
            **({"microstate": microstate_report} if microstate_report is not None else {}),
        },
        claim_boundary=ClaimBoundary(
            allowed=(
                "specified chemical graph",
                "format materialization",
            ) if materializable else (
                "input inspection",
                "symbolic monomer topology",
                "lower-rigor candidate routing",
            ),
            forbidden=(
                "experimental 3D accuracy",
                "biological flexibility",
                "docking benefit",
            ),
        ),
        exact_v1=exact,
        smiles=smiles,
        full_inchikey=inchikey,
        formal_charge=formal_charge,
        topology_class=topology_class,
        microstate_policy=str(protonation),
        parent_full_inchikey=parent_inchikey,
        exact_v1_identity_match=(
            parent_inchikey == inchikey
            if parent_inchikey and inchikey
            else None
        ),
        materializable=materializable,
        monomer_resolution=resolution_snapshot,
    )
    validate_artifact_identity(
        input_artifact, payload=request_payload
    )
    validate_artifact_identity(
        chemical_artifact, payload=graph_payload
    )
    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "input_artifact": input_artifact.to_dict(),
        "chemical_graph": chemical_artifact.to_dict(),
        "warnings": list(chemical_artifact.warnings),
        "alternatives": (
            []
            if materializable
            else [{
                "artifact_type": "SymbolicChemicalGraphCandidate",
                "status": "PARTIAL",
                "chemical_rigor": "C1:H",
                "symbols": list(symbols),
                "topology_class": topology_class,
                "cyclization": cyclization,
                "reason_codes": list(
                    exact.get("reason_codes") or []
                ),
            }]
        ),
        "provenance": {
            "map": map_text,
            "applicability": applicability,
            "monomer_resolution": resolution_snapshot,
        },
    }


__all__ = [
    "SEQUENCE_SCHEMA_VERSION",
    "build_molecule_from_sequence",
]
