"""Typed, content-addressed artifacts for the V5 preparation DAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
from typing import Any, Mapping, Sequence

from .cyclic_peptide_graph import canonical_json_bytes


ARTIFACT_SCHEMA_VERSION = "1.0.0-cycpep-v5-artifact.1"
ENSEMBLE_SCHEMA_VERSION = "1.0.0-cycpep-v5-ensemble.1"


class ArtifactType(str, Enum):
    INPUT = "InputArtifact"
    CHEMICAL_GRAPH = "ChemicalGraphArtifact"
    CONFORMER_ENSEMBLE = "ConformerEnsembleArtifact"
    VALIDATED_MOL2 = "ValidatedMol2Artifact"
    FLEXIBILITY_ASSESSMENT = "FlexibilityAssessmentArtifact"
    PDBQT = "PdbqtArtifact"


class ArtifactStatus(str, Enum):
    MATERIALIZED = "MATERIALIZED"
    PARTIAL = "PARTIAL"
    NOT_MATERIALIZABLE = "NOT_MATERIALIZABLE"
    FAILED = "FAILED"


class ChemicalLevel(str, Enum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


class EvidenceBasis(str, Enum):
    SPECIFIED = "S"
    QUALIFIED = "Q"
    REPAIRED = "R"
    HYPOTHESIS = "H"
    COORDINATE_ONLY = "C"
    NONE = "NONE"


class CoordinateOrigin(str, Enum):
    NONE = "none"
    GENERATED = "generated"
    TORSION_GUIDED = "torsion_guided"
    TEMPLATE_BORROWED = "template_borrowed"
    SOURCE_BOUND = "source_bound"
    EXTERNAL_PREDICTED = "external_predicted"
    EXPERIMENTAL = "experimental"


class CoordinateLevel(str, Enum):
    X0 = "X0"
    X1 = "X1"
    X2 = "X2"
    X3 = "X3"


class FormatLevel(str, Enum):
    Q0 = "Q0"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"


class FlexibilityLevel(str, Enum):
    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"


class BudgetState(str, Enum):
    NOT_REQUESTED = "not_requested"
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    NOT_ASSESSABLE = "not_assessable"


_CHEMICAL_ORDER = {
    ChemicalLevel.C0: 0,
    ChemicalLevel.C1: 1,
    ChemicalLevel.C2: 2,
    ChemicalLevel.C3: 3,
}
_COORDINATE_ORDER = {
    CoordinateLevel.X0: 0,
    CoordinateLevel.X1: 1,
    CoordinateLevel.X2: 2,
    CoordinateLevel.X3: 3,
}
_FORMAT_ORDER = {
    FormatLevel.Q0: 0,
    FormatLevel.Q1: 1,
    FormatLevel.Q2: 2,
    FormatLevel.Q3: 3,
}
_FLEX_ORDER = {
    FlexibilityLevel.F0: 0,
    FlexibilityLevel.F1: 1,
    FlexibilityLevel.F2: 2,
    FlexibilityLevel.F3: 3,
}


def structured_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


_LOCATOR_KEYS = frozenset({
    "path",
    "source_path",
    "manifest_path",
    "mol2_path",
    "receipt_path",
    "output_path",
    "template_pdb_path",
    "pdb_path",
})


def _identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(item)
            for key, item in value.items()
            if str(key) not in _LOCATOR_KEYS
            and not str(key).endswith("_path")
        }
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    return value


def make_artifact_id(
    artifact_type: ArtifactType | str,
    parent_artifact_ids: Sequence[str],
    payload: Mapping[str, Any],
) -> str:
    kind = (
        artifact_type.value
        if isinstance(artifact_type, ArtifactType)
        else str(artifact_type)
    )
    return structured_sha256({
        "artifact_type": kind,
        "parent_artifact_ids": list(parent_artifact_ids),
        "payload": _identity_value(payload),
    })


@dataclass(frozen=True)
class EvidenceProfile:
    chemical_level: ChemicalLevel = ChemicalLevel.C0
    chemical_basis: EvidenceBasis = EvidenceBasis.NONE
    coordinate_origin: CoordinateOrigin = CoordinateOrigin.NONE
    coordinate_level: CoordinateLevel = CoordinateLevel.X0
    format_level: FormatLevel = FormatLevel.Q0
    flexibility_level: FlexibilityLevel = FlexibilityLevel.F0
    budget_state: BudgetState = BudgetState.NOT_REQUESTED

    @property
    def chemical_rigor(self) -> str:
        return f"{self.chemical_level.value}:{self.chemical_basis.value}"

    def to_dict(self) -> dict[str, str]:
        return {
            "chemical_level": self.chemical_level.value,
            "chemical_basis": self.chemical_basis.value,
            "chemical_rigor": self.chemical_rigor,
            "coordinate_origin": self.coordinate_origin.value,
            "coordinate_level": self.coordinate_level.value,
            "format_level": self.format_level.value,
            "flexibility_level": self.flexibility_level.value,
            "budget_state": self.budget_state.value,
        }


@dataclass(frozen=True)
class ClaimBoundary:
    allowed: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "allowed": sorted(set(self.allowed)),
            "forbidden": sorted(set(self.forbidden)),
        }


@dataclass(frozen=True)
class ArtifactBase:
    artifact_type: ArtifactType
    artifact_id: str
    parent_artifact_ids: tuple[str, ...]
    status: ArtifactStatus
    payload_sha256: str
    evidence: EvidenceProfile
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    claim_boundary: ClaimBoundary = field(default_factory=ClaimBoundary)
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type.value,
            "artifact_id": self.artifact_id,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "status": self.status.value,
            "payload_sha256": self.payload_sha256,
            "evidence": self.evidence.to_dict(),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "claim_boundary": self.claim_boundary.to_dict(),
        }


@dataclass(frozen=True)
class InputArtifact(ArtifactBase):
    input_kind: str = ""
    source_sha256: str = ""
    source_path: str | None = None
    source_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "input_kind": self.input_kind,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "source_value": self.source_value,
        }


@dataclass(frozen=True)
class ChemicalGraphArtifact(ArtifactBase):
    exact_v1: Mapping[str, Any] = field(default_factory=dict)
    smiles: str | None = None
    full_inchikey: str | None = None
    formal_charge: int | None = None
    topology_class: str | None = None
    microstate_policy: str = "registry_default"
    parent_full_inchikey: str | None = None
    exact_v1_identity_match: bool | None = None
    materializable: bool = False
    monomer_resolution: Mapping[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "exact_v1": dict(self.exact_v1),
            "smiles": self.smiles,
            "full_inchikey": self.full_inchikey,
            "formal_charge": self.formal_charge,
            "topology_class": self.topology_class,
            "microstate_policy": self.microstate_policy,
            "parent_full_inchikey": self.parent_full_inchikey,
            "exact_v1_identity_match": self.exact_v1_identity_match,
            "materializable": self.materializable,
            "monomer_resolution": dict(self.monomer_resolution),
        }


@dataclass(frozen=True)
class ConformerMember:
    conformer_id: str
    coordinate_origin: CoordinateOrigin
    strategy: str
    mol2_path: str | None
    mol2_sha256: str | None
    receipt_path: str | None
    receipt_sha256: str | None
    energy: float | None
    qa: Mapping[str, Any]
    status: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["coordinate_origin"] = self.coordinate_origin.value
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class ConformerEnsembleArtifact(ArtifactBase):
    requested_count: int = 0
    produced_count: int = 0
    members: tuple[ConformerMember, ...] = ()
    manifest_path: str | None = None
    manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "requested_count": self.requested_count,
            "produced_count": self.produced_count,
            "members": [member.to_dict() for member in self.members],
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class ValidatedMol2Artifact(ArtifactBase):
    conformer_id: str = ""
    path: str = ""
    sha256: str = ""
    receipt_path: str = ""
    receipt_sha256: str = ""
    full_inchikey: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "conformer_id": self.conformer_id,
            "path": self.path,
            "sha256": self.sha256,
            "receipt_path": self.receipt_path,
            "receipt_sha256": self.receipt_sha256,
            "full_inchikey": self.full_inchikey,
        }


@dataclass(frozen=True)
class FlexibilityAssessmentArtifact(ArtifactBase):
    requested_mode: str = "fast"
    effective_mode: str = "fast"
    initial_torsdof: int | None = None
    final_torsdof: int | None = None
    bond_evidence: tuple[Mapping[str, Any], ...] = ()
    budget_satisfied: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "initial_torsdof": self.initial_torsdof,
            "final_torsdof": self.final_torsdof,
            "bond_evidence": [
                dict(value) for value in self.bond_evidence
            ],
            "budget_satisfied": self.budget_satisfied,
        }


@dataclass(frozen=True)
class PdbqtArtifact(ArtifactBase):
    path: str = ""
    sha256: str = ""
    role: str = "baseline"
    parent_mol2_sha256: str = ""
    tree_valid: bool = False
    atom_invariants_valid: bool = False
    qualification_state: str = "PDBQT_QUALIFIED"
    budget_satisfied: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "path": self.path,
            "sha256": self.sha256,
            "role": self.role,
            "parent_mol2_sha256": self.parent_mol2_sha256,
            "tree_valid": self.tree_valid,
            "atom_invariants_valid": self.atom_invariants_valid,
            "qualification_state": self.qualification_state,
            "budget_satisfied": self.budget_satisfied,
        }


def artifact_payload_sha256(payload: Mapping[str, Any]) -> str:
    return structured_sha256(payload)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_evidence_profile(evidence: EvidenceProfile) -> None:
    if (
        evidence.coordinate_level == CoordinateLevel.X0
    ) != (evidence.coordinate_origin == CoordinateOrigin.NONE):
        raise ValueError(
            "coordinate origin and coordinate evidence level disagree"
        )
    if (
        evidence.flexibility_level == FlexibilityLevel.F0
        and evidence.budget_state
        not in {
            BudgetState.NOT_REQUESTED,
            BudgetState.NOT_ASSESSABLE,
        }
    ):
        raise ValueError(
            "a torsion budget cannot be satisfied without flexibility "
            "evidence"
        )


def validate_artifact_identity(
    artifact: ArtifactBase,
    *,
    payload: Mapping[str, Any] | None = None,
) -> None:
    if not _is_sha256(artifact.artifact_id):
        raise ValueError("artifact_id must be a SHA-256")
    if not _is_sha256(artifact.payload_sha256):
        raise ValueError("payload_sha256 must be a SHA-256")
    if len(set(artifact.parent_artifact_ids)) != len(
        artifact.parent_artifact_ids
    ):
        raise ValueError("parent_artifact_ids must be unique")
    if any(
        not _is_sha256(value)
        for value in artifact.parent_artifact_ids
    ):
        raise ValueError("parent_artifact_ids must be SHA-256 values")
    validate_evidence_profile(artifact.evidence)
    if payload is not None:
        if artifact.payload_sha256 != artifact_payload_sha256(payload):
            raise ValueError("artifact payload SHA-256 mismatch")
        expected_id = make_artifact_id(
            artifact.artifact_type,
            artifact.parent_artifact_ids,
            payload,
        )
        if artifact.artifact_id != expected_id:
            raise ValueError("artifact ID is not bound to its payload")


def validate_artifact_record(
    record: Mapping[str, Any],
    *,
    expected_type: ArtifactType | None = None,
    payload: Mapping[str, Any] | None = None,
) -> None:
    if record.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported typed artifact schema")
    try:
        artifact_type = ArtifactType(str(record["artifact_type"]))
        ArtifactStatus(str(record["status"]))
        evidence = evidence_from_dict(record["evidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid typed artifact record: {exc}") from exc
    if expected_type is not None and artifact_type != expected_type:
        raise ValueError(
            f"expected {expected_type.value}, got {artifact_type.value}"
        )
    artifact_id = record.get("artifact_id")
    payload_sha = record.get("payload_sha256")
    parents = record.get("parent_artifact_ids")
    if not _is_sha256(artifact_id):
        raise ValueError("artifact_id must be a SHA-256")
    if not _is_sha256(payload_sha):
        raise ValueError("payload_sha256 must be a SHA-256")
    if (
        not isinstance(parents, list)
        or len(set(parents)) != len(parents)
        or any(not _is_sha256(value) for value in parents)
    ):
        raise ValueError("parent_artifact_ids must be unique SHA-256 values")
    validate_evidence_profile(evidence)
    if payload is not None:
        if payload_sha != artifact_payload_sha256(payload):
            raise ValueError("artifact payload SHA-256 mismatch")
        if artifact_id != make_artifact_id(
            artifact_type, tuple(parents), payload
        ):
            raise ValueError("artifact ID is not bound to its payload")


def validate_inherited_evidence(
    parent: EvidenceProfile,
    child: EvidenceProfile,
    *,
    allow_coordinate: bool = False,
    allow_format: bool = False,
    allow_flexibility: bool = False,
) -> None:
    """Reject downstream promotion outside the stage-owned dimension."""
    if child.chemical_level != parent.chemical_level:
        raise ValueError("downstream artifact changed chemical level")
    if child.chemical_basis != parent.chemical_basis:
        raise ValueError("downstream artifact changed chemical basis")
    if not allow_coordinate and (
        child.coordinate_level != parent.coordinate_level
        or child.coordinate_origin != parent.coordinate_origin
    ):
        raise ValueError("stage changed coordinate evidence")
    if allow_coordinate and (
        _COORDINATE_ORDER[child.coordinate_level]
        < _COORDINATE_ORDER[parent.coordinate_level]
    ):
        raise ValueError("stage downgraded coordinate evidence")
    if not allow_format and child.format_level != parent.format_level:
        raise ValueError("stage changed format qualification")
    if allow_format and (
        _FORMAT_ORDER[child.format_level]
        < _FORMAT_ORDER[parent.format_level]
    ):
        raise ValueError("stage downgraded format qualification")
    if not allow_flexibility and (
        child.flexibility_level != parent.flexibility_level
        or child.budget_state != parent.budget_state
    ):
        raise ValueError("stage changed flexibility evidence")
    if allow_flexibility and (
        _FLEX_ORDER[child.flexibility_level]
        < _FLEX_ORDER[parent.flexibility_level]
    ):
        raise ValueError("stage downgraded flexibility evidence")
    validate_evidence_profile(child)


def inherit_evidence(
    parent: EvidenceProfile,
    *,
    coordinate_origin: CoordinateOrigin | None = None,
    coordinate_level: CoordinateLevel | None = None,
    format_level: FormatLevel | None = None,
    flexibility_level: FlexibilityLevel | None = None,
    budget_state: BudgetState | None = None,
) -> EvidenceProfile:
    return EvidenceProfile(
        chemical_level=parent.chemical_level,
        chemical_basis=parent.chemical_basis,
        coordinate_origin=coordinate_origin or parent.coordinate_origin,
        coordinate_level=coordinate_level or parent.coordinate_level,
        format_level=format_level or parent.format_level,
        flexibility_level=(
            flexibility_level or parent.flexibility_level
        ),
        budget_state=budget_state or parent.budget_state,
    )


def evidence_from_dict(value: Mapping[str, Any]) -> EvidenceProfile:
    return EvidenceProfile(
        chemical_level=ChemicalLevel(value["chemical_level"]),
        chemical_basis=EvidenceBasis(value["chemical_basis"]),
        coordinate_origin=CoordinateOrigin(value["coordinate_origin"]),
        coordinate_level=CoordinateLevel(value["coordinate_level"]),
        format_level=FormatLevel(value["format_level"]),
        flexibility_level=FlexibilityLevel(
            value["flexibility_level"]
        ),
        budget_state=BudgetState(value["budget_state"]),
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactBase",
    "ArtifactStatus",
    "ArtifactType",
    "BudgetState",
    "ChemicalGraphArtifact",
    "ChemicalLevel",
    "ClaimBoundary",
    "ConformerEnsembleArtifact",
    "ConformerMember",
    "CoordinateLevel",
    "CoordinateOrigin",
    "ENSEMBLE_SCHEMA_VERSION",
    "EvidenceBasis",
    "EvidenceProfile",
    "FlexibilityAssessmentArtifact",
    "FlexibilityLevel",
    "FormatLevel",
    "InputArtifact",
    "PdbqtArtifact",
    "ValidatedMol2Artifact",
    "artifact_payload_sha256",
    "evidence_from_dict",
    "inherit_evidence",
    "make_artifact_id",
    "structured_sha256",
    "validate_artifact_identity",
    "validate_artifact_record",
    "validate_evidence_profile",
    "validate_inherited_evidence",
]
