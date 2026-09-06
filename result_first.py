"""Result-first coordinate chemistry graph recovery facade.

This module is fully decoupled from the strict V6 fail-closed pipeline: the
V6 entry points, pipeline ``path="v6"`` default, and CLI/GUI byte and field
semantics are untouched.  V6 is treated as the *top* of a deterministic
degradation ladder instead of a terminal verdict.  The product contract is
degrade-without-rejection: every input yields a typed, status-closed
artifact; insufficient evidence lowers the artifact tier and is recorded,
never silently dropped and never promoted.

Recovery ladder (first hit wins, no fallback is invoked afterwards):

1. V6 strict exact success -> quality ``exact`` (short-circuits; no fallback
   calls; the unmodified strict result is stored in ``strict_result``)
2. one distinct candidate identity in the validated
   ``strict.output_evidence["candidate_assessment"]`` -> quality ``high``
3. multiple candidate identities -> ordered candidate bundle with NO
   automatically selected identity (``smiles=None``, ``ambiguous=True``)
   -> quality ``medium``
4. RDKit ``MolFromPDBFile(proximityBonding=True, sanitize=False)`` mol that
   passes the independent topology qualification -> quality ``topology``
5. RDKit ``proximityBonding=False`` + ``rdDetermineBonds.DetermineConnectivity``
   mol that passes the topology qualification -> quality ``topology``
6. explicit-only unsanitized RDKit mol built from normalized-PDB atom records
   and CONECT edges -> quality ``partial``
7. advisory-parsed atom-coordinate + CONECT graph with an explicit damage
   ledger (malformed records, dangling/self edges, duplicate serials are
   recorded, not fatal) -> quality ``raw``
8. no readable atom at any stage -> typed ``opaque`` input artifact (byte
   snapshot, error code, format probe); the input never disappears

Quality definitions (no scientific-correctness claim is ever made):

- ``exact``: byte/field copy of the strict V6 success output.
- ``high``: a single candidate identity from the validated candidate
  assessment; chemistry not independently verified.
- ``medium``: multiple candidate identities; the bundle is ordered by
  distinct supporting route count (descending), best route priority
  (a>b>c>e>g), then full InChIKey / canonical SMILES, but no identity is
  automatically selected; all candidates are retained as alternatives.
- ``topology``: RDKit-read molecule passing the topology qualification
  (unique atom serials, heavy-atom count vs normalized audit, single
  heavy-atom component, bonds present, deterministic detected ring size
  >= ``minimum_macrocycle_ring_size``, all normalized-PDB explicit CONECT
  edges preserved, no self bonds).  Graph bond orders are unknown.
- ``partial``: readable explicit-only chemical graph (atoms + CONECT) built
  as an unsanitized RDKit mol; bond orders unknown; never SMILES.
- ``raw``: advisory-parsed atom + CONECT graph with a damage ledger; zero
  bonds allowed with ``NO_BONDS_AVAILABLE``; never SMILES.
- ``opaque``: no readable atom; the payload retains the input path, byte
  count, format probe, and error code.

Contract notes:

- ``status`` is always ``success`` for scientific outcomes; infrastructure
  failures still map to a typed envelope.  A top-level product rejection no
  longer exists.
- ``quality`` is one of
  ``exact|high|medium|candidate|hypothesis|topology|partial|raw|opaque``.
  ``warning_codes`` and ``warnings`` are preserved.
- Facade ``success`` does NOT imply the paper U2/U3 criteria.
- F/H route rows are diagnostic only and never elevate candidates.  When
  every readable F/H row agrees on one chemical identity, that identity is
  exposed at hypothesis quality capped at ``C1:H``; divergent F/H identities
  are never collapsed to an arbitrary primary — every identity is retained
  in an ambiguous, unqualified candidate set (``smiles=None``,
  ``ambiguous=True``, warning ``F_H_IDENTITY_DIVERGENT``).  F/H chemistry is
  never qualified regardless of agreement.  The F/H promotion stage is
  evaluated on the non-degraded path only; degraded fall-through never
  promotes F/H rows (they stay observable inside the stored, unmodified
  ``strict_result``).
- The ladder is monotone and molecule-first: molecule-capable stages run
  before graph-only template projection and the topology/partial/raw
  observation ladder, so a stronger molecule-level recovery is never
  shadowed by a weaker graph artifact.  On non-degraded inputs the
  molecule-capable stages are strict/mapping-aware/assessment candidates,
  registry assembly, bond-order inference, F/H diagnostic promotion, and
  proximity-molecule perception.  On degraded inputs the order is bond-order
  inference (executed at most once per request), proximity-molecule
  perception, template projection, then the observation ladder.  A C1:H
  proximity molecule that supersedes a graph-only C1:R projection is
  intended maximum-acceptance behavior, not a shadowing defect.
- The proximity-molecule fallback never silently selects the largest
  fragment: a multi-component read declines molecule promotion, records a
  complete component ledger (``provenance["proximity_molecule"]``), and
  carries ``PROXIMITY_COMPONENTS_DECLINED`` so the observation ladder can
  preserve every component honestly.
- Recovery quality and chemical rigor are orthogonal.  ``quality`` is never
  rewritten by rigor logic, but a ``C2:R`` recovered-chemistry label is
  never granted to a result that carries a diagnostic/integrity condition
  (integrity findings, F/H diagnostic-only or divergence labels, engine
  disagreement, mapping divergence, registry/template assistance, degraded
  input classes): such artifacts are labeled ``C2:H`` instead.
  ``chemical_rigor`` is the canonical label; the per-candidate
  ``candidate_rigor`` alias is letter-clamped to it at result construction,
  so an alias such as ``L2:R`` can never out-claim a canonical ``C2:H``.
  Downstream consumers must treat ``chemical_rigor`` as canonical
  (integration contract with the MOL2/application layer).
- Candidate elevation consumes ONLY ``strict.output_evidence[
  "candidate_assessment"]`` after
  ``candidate_assessment.validate_candidate_assessment_semantics``; route
  rows are never re-read to recompute candidates, so duplicate route rows
  cannot inflate ``supporting_route_count``.
- Known integrity-conflict codes block PROMOTION only (``degrade_without_
  rejection``): the observation ladder still runs, readable candidates are
  retained unqualified, and the block reasons live in
  ``provenance.fallback_block`` and the nested ``strict_evidence``.  A
  strict implementation exception is stage-local: it is recorded in
  ``provenance.strict_error`` and never blocks anything.
- Every RDKit-readable normalized case retains at least an
  unqualified/topology/partial/raw artifact; readable candidates are never
  dropped without provenance (``provenance.readable_candidates``).
- Topology/partial/raw results always carry a structured graph with unknown
  (null) bond orders and ``smiles=None``.  Inferred chemistry is exposed
  separately as an audited candidate.  Only exact/high populate the
  qualified downstream ``smiles`` handoff.
- RDKit fallbacks always use the entity already selected by
  ``prepare_coordinate_input``; chain selection is never repeated, and atom
  serial/name/residue/coordinates are preserved.
- Low-level exporters remain independent of this recovery ladder.  The shared
  ``application`` services accept this result (or its JSON payload), use the
  primary SMILES when present, and propagate ``quality``/warnings/alternatives.
  Graph-only topology/partial/raw results return typed ``not_supported`` for
  bond-order-dependent MOL2/SDF/PDBQT/docking instead of fabricating chemistry.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from . import remediation_v6
from .candidate_assessment import validate_candidate_assessment_semantics
from .core.structure_io import CoordinateInputError, prepare_coordinate_input
from .core.monomer_resolution import needs_monomer_resolution_scope
from .remediation_v5 import StrictReconstructionResult

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

QUALITY_EXACT = "exact"
QUALITY_HIGH = "high"
QUALITY_MEDIUM = "medium"
QUALITY_CANDIDATE = "candidate"
QUALITY_HYPOTHESIS = "hypothesis"
QUALITY_TOPOLOGY = "topology"
QUALITY_PARTIAL = "partial"
QUALITY_RAW = "raw"
QUALITY_OPAQUE = "opaque"
QUALITIES = (
    QUALITY_EXACT,
    QUALITY_HIGH,
    QUALITY_MEDIUM,
    QUALITY_CANDIDATE,
    QUALITY_HYPOTHESIS,
    QUALITY_TOPOLOGY,
    QUALITY_PARTIAL,
    QUALITY_RAW,
    QUALITY_OPAQUE,
)

SOURCE_EXACT = "v6_strict_exact"
SOURCE_UNIQUE_CANDIDATE = "v6_candidate_unique"
SOURCE_ENSEMBLE_CANDIDATE = "v6_candidate_ensemble"
SOURCE_MAPPING_AWARE = "coordinate_mapping_aware_recovery"
SOURCE_REGISTRY = "registry_template_assisted"
SOURCE_DEGRADED_TEMPLATE = "degraded_template_projection"
SOURCE_BOND_ORDER_CONSENSUS = "bond_order_inference_consensus"
SOURCE_BOND_ORDER_CANDIDATE = "bond_order_inference_candidate"
SOURCE_BOND_ORDER_HYPOTHESIS = "bond_order_inference_hypothesis"
SOURCE_PROXIMITY = "rdkit_proximity"
SOURCE_CONNECTIVITY = "rdkit_determine_connectivity"
SOURCE_PARTIAL = "rdkit_explicit_only_partial"
SOURCE_RAW = "raw_pdb_graph"
SOURCE_OPAQUE = "opaque_input_artifact"
SOURCE_FAILED = "failed"
SOURCES = (
    SOURCE_EXACT,
    SOURCE_UNIQUE_CANDIDATE,
    SOURCE_ENSEMBLE_CANDIDATE,
    SOURCE_MAPPING_AWARE,
    SOURCE_REGISTRY,
    SOURCE_DEGRADED_TEMPLATE,
    SOURCE_PROXIMITY,
    SOURCE_CONNECTIVITY,
    SOURCE_PARTIAL,
    SOURCE_RAW,
    SOURCE_OPAQUE,
    SOURCE_FAILED,
)

# Routes that may produce candidates.  F/H are diagnostic only and never
# elevate a candidate (mirrors the strict V6 chemical-route policy).
_CHEMICAL_ROUTES = ("a", "b", "c", "e", "g")
_DIAGNOSTIC_ROUTES = ("f", "h")
_ROUTE_PRIORITY = {route: index for index, route in enumerate(_CHEMICAL_ROUTES)}

_WARNING_V6_NOT_SUCCESS = "V6_STRICT_NOT_SUCCESS"
_WARNING_STRICT_SUCCESS_WITHOUT_SMILES = "STRICT_SUCCESS_WITHOUT_SMILES"
_WARNING_STRICT_SUCCESS_UNQUALIFIED = "STRICT_SUCCESS_UNQUALIFIED"
_WARNING_MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATE_IDENTITIES"
_WARNING_REGISTRY_TEMPLATE_ASSISTED = "REGISTRY_TEMPLATE_ASSISTED"
_WARNING_REGISTRY_NOT_QUALIFIED = "REGISTRY_TEMPLATE_NOT_QUALIFIED"
_WARNING_REGISTRY_OVERLAY_BLOCKED = "REGISTRY_TEMPLATE_OVERLAY_BLOCKED"
_WARNING_MALFORMED_CONECT_RAW = "MALFORMED_CONECT_RAW_PRESERVED"
_WARNING_NORMALIZATION_RAW = "NORMALIZATION_ERROR_RAW_PRESERVED"
_WARNING_DAMAGED_RAW = "DAMAGED_SOURCE_RAW_PRESERVED"
_WARNING_F_H_DIAGNOSTIC_ONLY = "F_H_DIAGNOSTIC_ONLY"
_WARNING_BOND_ORDERS_INFERRED = "BOND_ORDERS_INFERRED"
_WARNING_FORMAL_CHARGE_UNVERIFIED = "FORMAL_CHARGE_UNVERIFIED"
_WARNING_STEREOCHEMISTRY_UNVERIFIED = "STEREOCHEMISTRY_UNVERIFIED"
_WARNING_EXPLICIT_EDGES_NOT_PRESERVED = "EXPLICIT_CONECT_NOT_PRESERVED"
_WARNING_NO_BONDS_AVAILABLE = "NO_BONDS_AVAILABLE"
_WARNING_PARTIAL_EXPLICIT_ONLY = "PARTIAL_EXPLICIT_ONLY"
_WARNING_NO_READABLE_STRUCTURE = "NO_READABLE_STRUCTURE"
_WARNING_CANDIDATE_ASSESSMENT_MISSING = "CANDIDATE_ASSESSMENT_MISSING"
_WARNING_CANDIDATE_ASSESSMENT_INVALID = "CANDIDATE_ASSESSMENT_INVALID"
_WARNING_SOURCE_CONECT_REDUCED = "SOURCE_CONECT_REDUCED_DURING_NORMALIZATION"
_WARNING_STRICT_FALLBACK_BLOCKED = "STRICT_FALLBACK_BLOCKED"
_WARNING_DEGRADED_MISSING_SIDECHAIN = "DEGRADED_MISSING_SIDECHAIN"
_WARNING_DEGRADED_MISSING_BACKBONE = "DEGRADED_MISSING_BACKBONE"
_WARNING_DEGRADED_TRUNCATED_RESIDUE = "DEGRADED_TRUNCATED_RESIDUE"
_WARNING_DEGRADED_ABNORMAL_ATOM_NAME = "DEGRADED_ABNORMAL_ATOM_NAME"
_WARNING_DEGRADED_ELEMENT_MISMATCH = "DEGRADED_ELEMENT_MISMATCH"
_WARNING_DEGRADED_UNKNOWN_RESIDUE = "DEGRADED_UNKNOWN_RESIDUE"
_WARNING_DEGRADED_SEQUENCE_GAP = "DEGRADED_SEQUENCE_GAP"
_WARNING_DEGRADED_EXTERNAL_ENDPOINT_UNRESOLVED = "DEGRADED_EXTERNAL_ENDPOINT_UNRESOLVED"
_WARNING_DEGRADED_TEMPLATE_PROJECTION = "DEGRADED_TEMPLATE_PROJECTION"
_WARNING_DEGRADED_CANDIDATE_PROMOTION_BLOCKED = "DEGRADED_CANDIDATE_PROMOTION_BLOCKED"
_WARNING_BOND_ORDER_CANDIDATE = "BOND_ORDER_CANDIDATE_INFERRED"
_WARNING_INFERRED_CHEMISTRY_UNQUALIFIED = "INFERRED_CHEMISTRY_UNQUALIFIED"
_WARNING_INFERENCE_ENGINES_DISAGREE = "INFERENCE_ENGINES_DISAGREE"
_WARNING_OPENBABEL_UNAVAILABLE = "OPENBABEL_INFERENCE_UNAVAILABLE"
_WARNING_MAPPING_AMBIGUOUS_IDENTITY_INVARIANT = "COORDINATE_MAPPING_AMBIGUOUS_IDENTITY_INVARIANT"
_WARNING_MAPPING_IDENTITY_DIVERGENT = "COORDINATE_MAPPING_IDENTITY_DIVERGENT"
_WARNING_F_H_IDENTITY_DIVERGENT = "F_H_IDENTITY_DIVERGENT"
_WARNING_PROXIMITY_MOLECULE_FALLBACK = "PROXIMITY_MOLECULE_FALLBACK"
_WARNING_PROXIMITY_COMPONENTS_DECLINED = "PROXIMITY_COMPONENTS_DECLINED"

# Warning codes that mark a result's chemical identity as diagnostic-only,
# divergent, or integrity-conflicted.  High/medium recovery quality keeps its
# recovery tier, but the chemical_rigor provenance letter can never be "R"
# while any of these conditions is present: the artifact is labeled C2:H so a
# recovered-chemistry claim is never inherited from quality alone.  Mirrors
# ``core.rigor._DIAGNOSTIC_INTEGRITY_WARNING_CODES`` (kept decoupled there).
_RIGOR_DOWNGRADE_WARNING_CODES = frozenset({
    _WARNING_F_H_DIAGNOSTIC_ONLY,
    _WARNING_F_H_IDENTITY_DIVERGENT,
    _WARNING_REGISTRY_TEMPLATE_ASSISTED,
    _WARNING_INFERENCE_ENGINES_DISAGREE,
    _WARNING_MAPPING_IDENTITY_DIVERGENT,
    _WARNING_STRICT_FALLBACK_BLOCKED,
    _WARNING_DEGRADED_MISSING_SIDECHAIN,
    _WARNING_DEGRADED_MISSING_BACKBONE,
    _WARNING_DEGRADED_TRUNCATED_RESIDUE,
    _WARNING_DEGRADED_ABNORMAL_ATOM_NAME,
    _WARNING_DEGRADED_ELEMENT_MISMATCH,
    _WARNING_DEGRADED_UNKNOWN_RESIDUE,
    _WARNING_DEGRADED_SEQUENCE_GAP,
    _WARNING_DEGRADED_EXTERNAL_ENDPOINT_UNRESOLVED,
    _WARNING_DEGRADED_TEMPLATE_PROJECTION,
    _WARNING_DEGRADED_CANDIDATE_PROMOTION_BLOCKED,
})

# Topology qualification reason codes (recorded in ladder-attempt provenance).
_REASON_NO_ATOMS = "NO_ATOMS"
_REASON_DUPLICATE_ATOM_SERIAL = "DUPLICATE_ATOM_SERIAL"
_REASON_ATOM_IDENTITY_NOT_PRESERVED = "ATOM_IDENTITY_NOT_PRESERVED"
_REASON_HEAVY_ATOM_COUNT_MISMATCH = "HEAVY_ATOM_COUNT_MISMATCH"
_REASON_MULTIPLE_HEAVY_ATOM_COMPONENTS = "MULTIPLE_HEAVY_ATOM_COMPONENTS"
_REASON_NOT_PEPTIDE_ENTITY = "NOT_PEPTIDE_ENTITY"
_REASON_DETECTED_RING_SIZE_BELOW_THRESHOLD = (
    "DETECTED_RING_SIZE_BELOW_THRESHOLD"
)
_REASON_SELF_BOND = "SELF_BOND"
_REASON_ABNORMAL_ATOM_DEGREE = "ABNORMAL_ATOM_DEGREE"

# Strict V6 warning codes (or call exceptions) that mark the strict result as
# clearly untrustworthy: such results are never elevated to
# high/medium/topology/partial and only the raw stage is attempted.
_FALLBACK_BLOCK_WARNING_CODES = frozenset({
    "PDB_LINK_CONFLICT",
    "UNRESOLVED_PDB_LINK",
    "PDB_LINK_CONECT_CONFLICT",
    "V5_PLAIN_PDB_REQUIRED",
    "V5_MULTIMODEL_INPUT_REJECTED",
    "V5_MALFORMED_PDB_NUMBER",
    "V5_DUPLICATE_ATOM_SERIAL",
    "V5_MALFORMED_ATOM_RECORD",
    "V5_MISSING_ELEMENT_FIELD",
    "V5_MULTISEGMENT_CHAIN_REJECTED",
    "V5_ALTLOC_INPUT_REJECTED",
    "V5_DUPLICATE_ATOM_IDENTITY",
    "V5_AMBIGUOUS_RESIDUE_NUMBER",
    "V5_NO_SELECTED_CHAIN_ATOMS",
    "V5_MALFORMED_SSBOND_RECORD",
    "V5_MALFORMED_LINK_RECORD",
    "V5_MALFORMED_CONECT_RECORD",
    "V5_INTERCHAIN_CONNECTION_UNSUPPORTED",
    "V5_TRUNCATED_CONNECTION_ENDPOINT",
    "V5_EXPLICIT_CONNECTION_VALENCE_CONFLICT",
    "V5_STANDARD_RESIDUE_BOND_GEOMETRY_CONFLICT",
    "V5_CONFLICTING_CYCLIZATION_ENDPOINT",
    "V5_INPUT_CONTEXT_FAILED",
    "V5_INPUT_AUDIT_FAILED",
    "V6_INPUT_AUDIT_FAILED",
    "V6_LIBRARY_VS_EMBEDDED_COMPONENT_CONFLICT",
    "V6_LOCAL_MONOMER_EVIDENCE_CONFLICT",
    "V6_LOCAL_MONOMER_INPUT_REJECTED",
    "V6_LOCAL_OVERLAY_ACTIVATION_REJECTED",
    "V6_PERSISTENT_OVERLAY_AUDIT_FAILED",
    "V6_PERSISTENT_OVERLAY_NOT_EMPTY",
    "V6_SELECTED_IDENTITY_DRIFT",
    "V6_INTERNAL_ERROR",
    "V6_INVALID_MACROCYCLE_THRESHOLD",
})

# Explicit-connection audit fields produced by ``prepare_coordinate_input``
# that must be surfaced in ``provenance["prepared"]``.
_CONNECTION_AUDIT_FIELDS = (
    "source_pdb_conect_pair_count",
    "first_model_pdb_conect_pair_count",
    "selected_chain_pdb_conect_pair_count",
    "materialized_explicit_connection_count",
    "structured_connection_count",
    "first_model_structured_connection_count",
    "ignored_nonselected_model_structured_connection_count",
    "ignored_later_model_structured_connection_count",
    "model_invariant_unscoped_connection_count",
    "structured_connection_altloc_mismatch_count",
    "excluded_nonselected_residue_structured_connection_count",
    "cross_object_pdb_conect_pair_count",
    "unresolved_object_pdb_conect_pair_count",
    "ambiguous_object_structured_connection_count",
    "cross_object_structured_connection_count",
)

_WARNING_TEXT = {
    _WARNING_V6_NOT_SUCCESS: (
        "strict V6 did not qualify; a lower ladder stage produced the result"
    ),
    _WARNING_STRICT_SUCCESS_WITHOUT_SMILES: (
        "strict V6 reported success without an output_smiles"
    ),
    _WARNING_STRICT_SUCCESS_UNQUALIFIED: (
        "strict V6 returned a structure that was not a qualified success"
    ),
    _WARNING_MULTIPLE_CANDIDATES: (
        "multiple distinct candidate identities; a candidate bundle is "
        "returned without an automatically selected identity"
    ),
    _WARNING_REGISTRY_TEMPLATE_ASSISTED: (
        "a registry/template assembly was selected after an independent "
        "coordinate audit; coordinate identity remains lower-confidence"
    ),
    _WARNING_REGISTRY_NOT_QUALIFIED: (
        "registry/template assembly was unavailable or failed the coordinate "
        "heavy-atom audit"
    ),
    _WARNING_REGISTRY_OVERLAY_BLOCKED: (
        "registry/template assembly was skipped because the requested overlay "
        "policy was not satisfied"
    ),
    _WARNING_MALFORMED_CONECT_RAW: (
        "malformed CONECT records were retained only as a best-effort raw "
        "atom/edge graph"
    ),
    _WARNING_NORMALIZATION_RAW: (
        "coordinate normalization did not close; readable source atoms/edges "
        "were retained as an unqualified raw artifact"
    ),
    _WARNING_DAMAGED_RAW: (
        "malformed atom records or duplicate serials were skipped while the "
        "raw graph was preserved; see provenance.raw damage ledger"
    ),
    _WARNING_F_H_DIAGNOSTIC_ONLY: (
        "F/H rows are diagnostic only and never elevate candidates"
    ),
    _WARNING_BOND_ORDERS_INFERRED: (
        "bonds inferred from coordinates, not authoritative"
    ),
    _WARNING_FORMAL_CHARGE_UNVERIFIED: "formal charges unverified",
    _WARNING_STEREOCHEMISTRY_UNVERIFIED: "stereochemistry unverified",
    _WARNING_EXPLICIT_EDGES_NOT_PRESERVED: (
        "not all explicit CONECT edges of the normalized PDB were preserved "
        "by this stage"
    ),
    _WARNING_NO_BONDS_AVAILABLE: (
        "no explicit bonds available; raw atom graph only"
    ),
    _WARNING_PARTIAL_EXPLICIT_ONLY: (
        "partial graph derived from explicit CONECT edges only; bond orders "
        "unknown"
    ),
    _WARNING_NO_READABLE_STRUCTURE: (
        "no readable non-empty structure at any ladder stage"
    ),
    _WARNING_CANDIDATE_ASSESSMENT_MISSING: (
        "strict output_evidence contains no candidate_assessment; candidate "
        "stage skipped"
    ),
    _WARNING_CANDIDATE_ASSESSMENT_INVALID: (
        "candidate_assessment failed semantic validation; candidate stage "
        "skipped"
    ),
    _WARNING_SOURCE_CONECT_REDUCED: (
        "explicit source CONECT edges were reduced during entity "
        "selection/normalization; counts in "
        "provenance['prepared']['connection_reduction']"
    ),
    _WARNING_STRICT_FALLBACK_BLOCKED: (
        "strict qualification/promotion was blocked, but lower observable "
        "artifacts continued through the degradation ladder"
    ),
    _WARNING_DEGRADED_MISSING_SIDECHAIN: "backbone is complete but template side-chain atoms are missing",
    _WARNING_DEGRADED_MISSING_BACKBONE: "one or more required backbone atoms are missing",
    _WARNING_DEGRADED_TRUNCATED_RESIDUE: "residue coordinate record is truncated",
    _WARNING_DEGRADED_ABNORMAL_ATOM_NAME: "an observed atom name could not be safely mapped",
    _WARNING_DEGRADED_ELEMENT_MISMATCH: "an observed atom element conflicts with its template",
    _WARNING_DEGRADED_UNKNOWN_RESIDUE: "residue identity has no unique trusted template",
    _WARNING_DEGRADED_SEQUENCE_GAP: "a residue-number gap cannot be safely bridged",
    _WARNING_DEGRADED_EXTERNAL_ENDPOINT_UNRESOLVED: "an explicit connection endpoint is unresolved",
    _WARNING_DEGRADED_TEMPLATE_PROJECTION: "primary result is an observed partial template projection",
    _WARNING_DEGRADED_CANDIDATE_PROMOTION_BLOCKED: "candidate and registry promotion were blocked by missing coordinate evidence",
    _WARNING_BOND_ORDER_CANDIDATE: (
        "one or more source-composition-bound bond-order candidates were "
        "inferred"
    ),
    _WARNING_INFERRED_CHEMISTRY_UNQUALIFIED: (
        "the selected inferred candidate is serializable but is not a "
        "strict-qualified chemical result"
    ),
    _WARNING_INFERENCE_ENGINES_DISAGREE: (
        "bond-order inference engines produced distinct connectivity "
        "identities; alternatives were retained"
    ),
    _WARNING_OPENBABEL_UNAVAILABLE: (
        "the optional Open Babel bond-order inference engine was unavailable"
    ),
    _WARNING_MAPPING_AMBIGUOUS_IDENTITY_INVARIANT: "multiple coordinate mappings produced one complete chemical identity",
    _WARNING_MAPPING_IDENTITY_DIVERGENT: "multiple coordinate mappings produced distinct complete chemical identities",
    _WARNING_F_H_IDENTITY_DIVERGENT: (
        "F/H diagnostic rows produced distinct chemical identities; every "
        "identity is retained as an unqualified ambiguous candidate set "
        "with no automatically selected primary"
    ),
    _WARNING_PROXIMITY_MOLECULE_FALLBACK: (
        "molecule-level SMILES recovered by proximity perception with "
        "valence-inferred bond orders; never a strict-qualified result"
    ),
    _WARNING_PROXIMITY_COMPONENTS_DECLINED: (
        "proximity perception produced multiple disconnected components; "
        "molecule promotion was declined instead of silently selecting the "
        "largest fragment, and the complete component ledger is recorded in "
        "provenance['proximity_molecule']"
    ),
}

_TWO_LETTER_ELEMENTS = frozenset({
    "AC", "AG", "AL", "AM", "AR", "AS", "AT", "AU", "BA", "BE", "BH", "BI",
    "BK", "BR", "CA", "CD", "CE", "CF", "CL", "CM", "CN", "CO", "CR", "CS",
    "CU", "DB", "DS", "DY", "ER", "ES", "EU", "FE", "FL", "FM", "FR", "GA",
    "GD", "GE", "HE", "HF", "HG", "HO", "HS", "IN", "IR", "LA", "LI", "LR",
    "LU", "LV", "MC", "MD", "MG", "MN", "MO", "MT", "NA", "NB", "ND", "NE",
    "NH", "NI", "NO", "NP", "OG", "OS", "PA", "PB", "PD", "PM", "PO", "PR",
    "PT", "PU", "RA", "RB", "RE", "RF", "RG", "RH", "RN", "RU", "SB", "SC",
    "SE", "SG", "SI", "SM", "SN", "SR", "TA", "TB", "TC", "TE", "TH", "TI",
    "TL", "TM", "TS", "U", "V", "W", "XE", "YB", "ZN", "ZR",
})

# Common protein atom names whose first two letters collide with element
# symbols (CA = C-alpha, CD/CD1, ND/NE/NH, OG, SD, ...).  Only used when a
# PDB record lacks the element column; normalized inputs always carry it.
_PROTEIN_ATOM_NAME_ELEMENT_COLLISIONS = frozenset({
    "CA", "CD", "CE", "CG", "ND", "NE", "NH", "OD", "OE", "OG", "OH", "SD",
    "SG",
})


@dataclass
class ReconstructionResult:
    """Typed result of the result-first recovery ladder.

    ``strict_result`` is the unmodified strict V6 object (``json_ready``
    serializes it via ``asdict``); the facade never mutates it.
    """

    status: str
    quality: str | None
    source: str | None
    result: Any
    smiles: str | None
    graph: dict[str, Any] | None
    ambiguous: bool
    warnings: list[str]
    warning_codes: list[str]
    alternatives: list[dict[str, Any]]
    provenance: dict[str, Any]
    strict_status: str | None
    strict_result: Any = None
    candidate_smiles: str | None = None
    candidate_graph: dict[str, Any] | None = None
    chemistry_candidates: list[dict[str, Any]] = field(default_factory=list)
    bond_order_inference: dict[str, Any] = field(default_factory=dict)
    candidate_rigor: str | None = None
    artifact_status: str = "opaque_input"
    qualification_status: str = "not_assessable"
    chemical_rigor: str = "C0:NONE"
    coordinate_evidence: str = "X0"
    integrity_findings: list[str] = field(default_factory=list)
    strict_evidence: dict[str, Any] = field(default_factory=dict)
    normalized_source_sha256: str | None = None
    graph_sha256: str | None = None


def _warnings_from_codes(codes: list[str]) -> list[str]:
    return [
        f"{code}: {_WARNING_TEXT.get(code, code)}" for code in codes
    ]


def _artifact_contract(
    *,
    quality: str | None,
    graph: dict[str, Any] | None,
    provenance: dict[str, Any],
    strict: StrictReconstructionResult | None,
    warning_codes: list[str] | None = None,
) -> dict[str, Any]:
    mapping = {
        QUALITY_EXACT: ("qualified_molecule", "qualified", "C3:Q", "X3"),
        QUALITY_HIGH: ("unqualified_candidate", "unqualified", "C2:R", "X3"),
        QUALITY_MEDIUM: ("unqualified_candidate", "unqualified", "C2:H", "X3"),
        QUALITY_CANDIDATE: ("unqualified_candidate", "unqualified", "C2:H", "X3"),
        QUALITY_HYPOTHESIS: ("unqualified_candidate", "unqualified", "C1:H", "X3"),
        QUALITY_TOPOLOGY: ("topology", "unqualified", "C1:H", "X3"),
        QUALITY_PARTIAL: ("partial_graph", "unqualified", "C1:R", "X3"),
        QUALITY_RAW: ("raw_coordinates", "not_assessable", "C0:C", "X3"),
        QUALITY_OPAQUE: ("opaque_input", "not_assessable", "C0:NONE", "X0"),
    }
    artifact_status, qualification_status, chemical_rigor, coordinate_evidence = (
        mapping.get(quality, mapping[QUALITY_OPAQUE])
    )
    graph_sha256 = None
    if isinstance(graph, dict):
        graph_sha256 = hashlib.sha256(
            _json_roundtrip(graph).encode("ascii")
        ).hexdigest()
    prepared = provenance.get("prepared") or {}
    normalized_source_sha256 = next(
        (
            str(prepared[key])
            for key in (
                "normalized_pdb_sha256",
                "normalized_sha256",
                "source_sha256",
                "input_sha256",
            )
            if prepared.get(key)
        ),
        None,
    )
    integrity_findings = list(dict.fromkeys(
        list(provenance.get("integrity_findings") or [])
        + list((provenance.get("fallback_block") or {}).get("reason_codes") or [])
    ))
    # Recovery quality and chemical rigor are orthogonal: quality is never
    # rewritten here, but a C2:R "recovered chemistry" letter is never
    # inherited from high quality while a diagnostic/integrity condition is
    # present.  Such artifacts are labeled C2:H (identity evidence is
    # heuristic/unresolved); no stronger tier is ever granted.
    if chemical_rigor == "C2:R" and (
        integrity_findings
        or any(
            code in _RIGOR_DOWNGRADE_WARNING_CODES
            for code in (warning_codes or [])
        )
    ):
        chemical_rigor = "C2:H"
    strict_evidence = {
        "status": strict.status if strict is not None else None,
        "qualified_success": bool(strict.qualified_success) if strict is not None else None,
        "rejection_reason": strict.rejection_reason if strict is not None else None,
        "warning_codes": list(strict.warning_codes) if strict is not None else [],
        "path_used": strict.path_used if strict is not None else None,
        "error": provenance.get("strict_error"),
    }
    return {
        "artifact_status": artifact_status,
        "qualification_status": qualification_status,
        "chemical_rigor": chemical_rigor,
        "coordinate_evidence": coordinate_evidence,
        "integrity_findings": integrity_findings,
        "strict_evidence": strict_evidence,
        "normalized_source_sha256": normalized_source_sha256,
        "graph_sha256": graph_sha256,
    }


# Conservative strength ordering of rigor provenance letters.  ``Q`` is the
# strict-qualified contract, ``R`` a recovered-chemistry claim, ``H``
# heuristic identity evidence, ``C`` raw coordinates, ``NONE`` no claim.
_RIGOR_LETTER_STRENGTH = {"Q": 4, "R": 3, "H": 2, "C": 1, "NONE": 0}


def _clamp_candidate_rigor(
    candidate_rigor: str | None,
    chemical_rigor: str,
) -> str | None:
    """Never let the per-candidate rigor alias out-claim the canonical label.

    ``candidate_rigor`` is a pass-through alias (for example the bond-order
    inference report's ``L2:R``), while ``chemical_rigor`` is the canonical
    C-axis contract computed by ``_artifact_contract``.  When a
    diagnostic/integrity condition downgrades the canonical letter, the
    alias is conservatively clamped to the same letter (its own prefix
    dimension is preserved) so no consumer can read a recovered-chemistry
    claim the canonical contract does not support.  Unparseable or empty
    aliases pass through unchanged.
    """
    if not isinstance(candidate_rigor, str) or ":" not in candidate_rigor:
        return candidate_rigor
    if not isinstance(chemical_rigor, str) or ":" not in chemical_rigor:
        return candidate_rigor
    alias_prefix, alias_letter = candidate_rigor.rsplit(":", 1)
    canonical_letter = chemical_rigor.rsplit(":", 1)[1].strip().upper()
    alias_strength = _RIGOR_LETTER_STRENGTH.get(alias_letter.strip().upper())
    canonical_strength = _RIGOR_LETTER_STRENGTH.get(canonical_letter)
    if alias_strength is None or canonical_strength is None:
        return candidate_rigor
    if alias_strength > canonical_strength:
        return f"{alias_prefix}:{canonical_letter}"
    return candidate_rigor


def _build_result(
    *,
    status: str,
    quality: str | None,
    source: str | None,
    result: Any,
    smiles: str | None,
    graph: dict[str, Any] | None,
    ambiguous: bool,
    warning_codes: list[str],
    alternatives: list[dict[str, Any]],
    provenance: dict[str, Any],
    strict: StrictReconstructionResult | None,
    candidate_smiles: str | None = None,
    candidate_graph: dict[str, Any] | None = None,
    chemistry_candidates: list[dict[str, Any]] | None = None,
    bond_order_inference: dict[str, Any] | None = None,
    candidate_rigor: str | None = None,
) -> ReconstructionResult:
    codes = list(dict.fromkeys(warning_codes))
    all_alternatives = list(alternatives or []) + list(
        provenance.get("readable_candidates") or []
    )
    normalized_alternatives = _normalize_alternatives(
        all_alternatives,
        parent_status=status,
        parent_quality=quality,
        parent_source=source,
        parent_warning_codes=codes,
    )
    contract = _artifact_contract(
        quality=quality,
        graph=graph or candidate_graph,
        provenance=provenance,
        strict=strict,
        warning_codes=codes,
    )
    return ReconstructionResult(
        status=status,
        quality=quality,
        source=source,
        result=result,
        smiles=smiles,
        graph=graph,
        ambiguous=ambiguous,
        warnings=_warnings_from_codes(codes),
        warning_codes=codes,
        alternatives=normalized_alternatives,
        provenance=provenance,
        strict_status=strict.status if strict is not None else None,
        strict_result=strict,
        candidate_smiles=candidate_smiles,
        candidate_graph=candidate_graph,
        chemistry_candidates=list(chemistry_candidates or []),
        bond_order_inference=dict(bond_order_inference or {}),
        candidate_rigor=_clamp_candidate_rigor(
            candidate_rigor, contract["chemical_rigor"]
        ),
        **contract,
    )


def _normalize_alternatives(
    alternatives: list[dict[str, Any]] | None,
    *,
    parent_status: str,
    parent_quality: str | None,
    parent_source: str | None,
    parent_warning_codes: list[str],
) -> list[dict[str, Any]]:
    """Give every alternative the same small, JSON-ready result summary.

    Older callers supplied identity-only candidate rows or chain-only rows.
    Keeping those domain fields while adding the common result fields lets
    clients render alternatives without knowing which ladder branch produced
    them.  The input rows are copied and never mutated.
    """
    normalized: list[dict[str, Any]] = []
    for raw in alternatives or []:
        item = dict(raw) if isinstance(raw, dict) else {"value": raw}
        item.setdefault("status", "not_selected")
        item.setdefault("quality", parent_quality)
        item.setdefault("source", parent_source or "alternative")
        item.setdefault("result_origin", item.get("source"))
        smiles = item.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            candidate_smiles = item.get("canonical_smiles")
            if isinstance(candidate_smiles, str) and candidate_smiles:
                smiles = candidate_smiles
        item.setdefault("smiles", smiles)
        item.setdefault("graph", None)
        item.setdefault("ambiguous", bool(item.get("alternatives")))
        item.setdefault("warnings", [])
        item.setdefault("warning_codes", list(parent_warning_codes))
        item.setdefault("structure_profile", None)
        item.setdefault("provenance", {})
        normalized.append(item)
    return normalized


def _connection_reduction(audit: dict[str, Any]) -> dict[str, Any]:
    """Quantify explicit-edge loss across entity selection/normalization."""
    source = audit.get("source_pdb_conect_pair_count")
    first_model = audit.get("first_model_pdb_conect_pair_count")
    selected_chain = audit.get("selected_chain_pdb_conect_pair_count")
    materialized = audit.get("materialized_explicit_connection_count")
    reductions: dict[str, Any] = {}
    steps = (
        ("source_to_first_model", source, first_model),
        ("first_model_to_selected_chain", first_model, selected_chain),
        (
            "selected_chain_to_materialized",
            selected_chain,
            materialized,
        ),
        ("source_to_materialized", source, materialized),
    )
    for name, before, after in steps:
        if (
            isinstance(before, int)
            and isinstance(after, int)
            and before > after
        ):
            reductions[name] = {
                "before": before,
                "after": after,
                "reduced_by": before - after,
            }
    return reductions


def _prepared_summary(prepared: Any) -> dict[str, Any]:
    audit = getattr(prepared, "audit", None) or {}
    summary: dict[str, Any] = {
        "source_format": str(getattr(prepared, "source_format", "")),
        "chain_id": str(getattr(prepared, "chain_id", "")),
        "normalized_chain_id": audit.get("normalized_chain_id"),
        "normalized_heavy_atom_count": audit.get(
            "normalized_heavy_atom_count"
        ),
        "source_sha256": audit.get("source_sha256"),
        "normalized_sha256": audit.get("normalized_sha256"),
    }
    for field in _CONNECTION_AUDIT_FIELDS:
        if field in audit:
            summary[field] = audit[field]
    summary["connection_audit"] = {
        key: value
        for key, value in audit.items()
        if isinstance(key, str)
        and (
            "conect" in key.lower() or "connection" in key.lower()
        )
    }
    summary["connection_reduction"] = _connection_reduction(audit)
    return summary


def _initial_provenance(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    strict_error: str | None,
) -> dict[str, Any]:
    return {
        "ladder": None,
        "prepared": _prepared_summary(prepared),
        "strict_status": strict.status if strict is not None else None,
        "strict_qualified_success": (
            bool(strict.qualified_success) if strict is not None else None
        ),
        "strict_path_used": strict.path_used if strict is not None else None,
        "strict_rejection_reason": (
            strict.rejection_reason if strict is not None else None
        ),
        "strict_warning_codes": (
            list(strict.warning_codes) if strict is not None else []
        ),
        "strict_error": strict_error,
        "ladder_attempts": [],
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def bind_reconstruction_result(
    result: ReconstructionResult,
    prepared: Any,
    *,
    requested_chain_id: str,
    minimum_macrocycle_ring_size: int,
    require_empty_persistent_overlay: bool,
    infer_bond_orders: bool,
) -> ReconstructionResult:
    from .paths._map_utils import registry_epoch

    result.provenance["request_binding"] = {
        "schema_version": "1.0.0-result-first-binding.1",
        "source_sha256": prepared.audit.get("source_sha256"),
        "normalized_sha256": prepared.audit.get("normalized_sha256"),
        "requested_chain_id": str(requested_chain_id),
        "normalized_chain_id": str(prepared.chain_id),
        "minimum_macrocycle_ring_size": int(
            minimum_macrocycle_ring_size
        ),
        "require_empty_persistent_overlay": bool(
            require_empty_persistent_overlay
        ),
        "infer_bond_orders": bool(infer_bond_orders),
        "registry_epoch": int(registry_epoch()),
    }
    return result


def reconstruct_structure(
    input_path: str | Path,
    chain_id: str = "L",
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    registry_assembler: Callable[[Any], Any] | None = None,
    infer_bond_orders: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> ReconstructionResult:
    """Run the result-first ladder on a coordinate file.

    The input is normalized through the shared ``prepare_coordinate_input``
    layer (same entity/chain selection and audit as strict V6), then delegated
    to :func:`reconstruct_prepared_structure`.  Input errors and unexpected
    failures map to a typed ``failed`` result with provenance instead of
    raising.
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
            required_symbols=monomer_symbol_hints(
                input_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            resolved = reconstruct_structure(
                input_path,
                chain_id=chain_id,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                registry_assembler=registry_assembler,
                infer_bond_orders=infer_bond_orders,
            )
        if monomer_context is not None:
            resolved.provenance.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    try:
        with prepare_coordinate_input(input_path, chain_id) as prepared:
            result = reconstruct_prepared_structure(
                prepared,
                minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                registry_assembler=registry_assembler,
                infer_bond_orders=bool(infer_bond_orders),
            )
            return bind_reconstruction_result(
                result,
                prepared,
                requested_chain_id=str(chain_id),
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                infer_bond_orders=bool(infer_bond_orders),
            )
    except CoordinateInputError as exc:
        raw_fallback = _raw_source_fallback(
            input_path,
            chain_id,
            coordinate_error=exc,
        )
        if raw_fallback is not None:
            return raw_fallback
        return _failed_result(
            {
                "ladder": "opaque_input",
                "prepared": {
                    "input_path": str(Path(input_path)),
                    "chain_id": str(chain_id),
                },
                "strict_status": None,
                "strict_error": None,
                "error_code": getattr(exc, "code", "COORDINATE_INPUT_ERROR"),
                "not_supported": bool(getattr(exc, "not_supported", False)),
                "error": f"{type(exc).__name__}: {exc}",
                "ladder_attempts": [],
                "failure_reason": "coordinate input could not be prepared",
                "integrity_findings": [
                    str(getattr(exc, "code", "COORDINATE_INPUT_ERROR"))
                ],
            },
            None,
            base_codes=[_WARNING_NO_READABLE_STRUCTURE],
        )
    except Exception as exc:
        return _failed_result(
            {
                "ladder": "opaque_input",
                "prepared": {
                    "input_path": str(Path(input_path)),
                    "chain_id": str(chain_id),
                },
                "strict_status": None,
                "strict_error": None,
                "error_code": "UNEXPECTED_INPUT_PREPARATION_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "ladder_attempts": [],
                "failure_reason": "coordinate input could not be prepared",
                "integrity_findings": ["UNEXPECTED_INPUT_PREPARATION_ERROR"],
            },
            None,
            base_codes=[_WARNING_NO_READABLE_STRUCTURE],
        )


def _raw_source_fallback(
    input_path: str | Path,
    chain_id: str | None,
    *,
    coordinate_error: CoordinateInputError,
) -> ReconstructionResult | None:
    """Retain readable atoms when normalization failed only on CONECT syntax.

    The normal path intentionally uses strict normalization.  A malformed
    CONECT line should not erase otherwise readable coordinate evidence from
    the result-first product API, however.  This fallback accepts valid atom
    records and valid fixed-width CONECT pairs, records every dropped token,
    and never promotes the result above ``raw``.
    """
    path = Path(input_path)
    if path.name.lower().endswith(".gz") or path.suffix.lower() not in {
        ".pdb",
        ".ent",
    }:
        return None
    try:
        graph, audit = _parse_raw_source_graph(path, chain_id)
        payload = _json_roundtrip(graph)
    except Exception:
        return None
    if not graph.get("atoms"):
        return None
    error_code = str(getattr(coordinate_error, "code", ""))
    warning_codes = [
        _WARNING_MALFORMED_CONECT_RAW
        if "CONECT" in error_code
        else _WARNING_NORMALIZATION_RAW
    ]
    if not graph.get("bonds"):
        warning_codes.append(_WARNING_NO_BONDS_AVAILABLE)
    provenance = {
        "ladder": SOURCE_RAW,
        "prepared": {
            "input_path": str(path),
            "chain_id": chain_id,
            "normalization_error": {
                "code": getattr(coordinate_error, "code", None),
                "message": str(coordinate_error),
            },
        },
        "strict_status": None,
        "strict_error": None,
        "ladder_attempts": [
            {
                "stage": SOURCE_RAW,
                "quality": QUALITY_RAW,
                "ok": True,
                "atom_count": len(graph["atoms"]),
                "bond_count": len(graph["bonds"]),
                "malformed_conect_record_count": len(
                    audit["malformed_records"]
                ),
                "dropped_conect_token_count": audit["dropped_token_count"],
                "skipped_atom_record_count": len(
                    audit.get("skipped_atom_records", [])
                ),
                "duplicate_atom_serial_count": len(
                    audit.get("duplicate_atom_serials", [])
                ),
                "graph_json_sha256": hashlib.sha256(
                    payload.encode("ascii")
                ).hexdigest(),
            }
        ],
        "raw": {
            "atom_count": len(graph["atoms"]),
            "bond_count": len(graph["bonds"]),
            "malformed_conect_records": audit["malformed_records"],
            "dropped_conect_token_count": audit["dropped_token_count"],
            "skipped_atom_records": audit.get("skipped_atom_records", []),
            "duplicate_atom_serials": audit.get(
                "duplicate_atom_serials", []
            ),
            "graph_json_sha256": hashlib.sha256(
                payload.encode("ascii")
            ).hexdigest(),
        },
        "normalization_error_recovered": True,
        "integrity_findings": [error_code or "COORDINATE_INPUT_ERROR"],
    }
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_RAW,
        source=SOURCE_RAW,
        result=graph,
        smiles=None,
        graph=graph,
        ambiguous=False,
        warning_codes=warning_codes,
        alternatives=[],
        provenance=provenance,
        strict=None,
    )


def reconstruct_prepared_structure(
    prepared: Any,
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    radius_multiplier: float | None = None,
    distance_ceiling: float | None = None,
    registry_assembler: Callable[[Any], Any] | None = None,
    infer_bond_orders: bool = False,
    monomer_context: Mapping[str, Any] | None = None,
) -> ReconstructionResult:
    """Run the result-first ladder on one prepared coordinate input.

    ``prepared`` is the same ``PreparedCoordinateInput`` object used by the
    strict V6 API and must be consumed while its context manager is open (its
    ``pdb_path`` is a temporary normalized PDB).  The strict V6 pipeline is
    invoked first and its result is stored unmodified in ``strict_result``.
    No public-API path raises: unexpected ladder failures degrade to a typed
    ``failed`` result.
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
            required_symbols=monomer_symbol_hints(
                prepared.pdb_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            resolved = reconstruct_prepared_structure(
                prepared,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                registry_assembler=registry_assembler,
                infer_bond_orders=infer_bond_orders,
            )
        if monomer_context is not None:
            resolved.provenance.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    strict: StrictReconstructionResult | None = None
    strict_error: str | None = None
    try:
        strict = remediation_v6.reconstruct_prepared_structure_fail_closed_v6(
            prepared,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
        )
    except Exception as exc:  # strict V6 is fail-closed; never fatal here
        strict_error = f"{type(exc).__name__}: {exc}"

    provenance = _initial_provenance(prepared, strict, strict_error)
    connection_codes = _connection_warning_codes(provenance)

    # ``radius_multiplier`` / ``distance_ceiling`` terminate here: the strict
    # V6 call above is an immutable contract and the ladder stages below never
    # invoke geometric covalent-radius cyclization.
    try:
        return _run_ladder(
            prepared,
            strict,
            strict_error,
            provenance,
            connection_codes,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
            registry_assembler=registry_assembler,
            infer_bond_orders=bool(infer_bond_orders),
        )
    except Exception as exc:  # requirement: never raise from the public API
        provenance["ladder"] = "failed"
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        provenance["failure_reason"] = (
            "unexpected ladder failure; no readable structure produced"
        )
        return _failed_result(
            provenance, strict, base_codes=connection_codes
        )


def _connection_warning_codes(provenance: dict[str, Any]) -> list[str]:
    reduction = provenance["prepared"].get("connection_reduction") or {}
    if reduction:
        return [_WARNING_SOURCE_CONECT_REDUCED]
    return []


def _run_ladder(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    strict_error: str | None,
    provenance: dict[str, Any],
    connection_codes: list[str],
    *,
    minimum_macrocycle_ring_size: int,
    require_empty_persistent_overlay: bool,
    registry_assembler: Callable[[Any], Any] | None,
    infer_bond_orders: bool,
) -> ReconstructionResult:
    codes = list(connection_codes)
    if strict is None or strict.status != STATUS_SUCCESS:
        codes.insert(0, _WARNING_V6_NOT_SUCCESS)
    elif strict.qualified_success is not True:
        codes.append(_WARNING_STRICT_SUCCESS_UNQUALIFIED)
    if (
        strict is not None
        and strict.status == STATUS_SUCCESS
        and not strict.output_smiles
    ):
        codes.append(_WARNING_STRICT_SUCCESS_WITHOUT_SMILES)
    integrity_findings = _strict_fallback_blocked(strict, strict_error)
    promotion_blocked = bool(integrity_findings)
    if promotion_blocked:
        codes.append(_WARNING_STRICT_FALLBACK_BLOCKED)
        if strict is not None:
            codes.extend(strict.warning_codes)
        provenance["integrity_findings"] = list(integrity_findings)
        provenance["fallback_block"] = {
            "blocked": False,
            "promotion_blocked": True,
            "reason_codes": list(integrity_findings),
            "policy": "degrade_without_rejection",
            "note": (
                "strict qualification and automatic chemistry promotion are "
                "blocked, but observable candidates/topology/partial/raw evidence "
                "continue through the degradation ladder"
            ),
        }

    if (
        strict is not None
        and strict.status == STATUS_SUCCESS
        and strict.qualified_success is True
        and strict.output_smiles
        and not promotion_blocked
    ):
        return _exact_result(
            strict, provenance, additional_codes=connection_codes
        )

    if strict is not None:
        # Lower ladder stages retain why strict V6 did not qualify.  The
        # exact branch above keeps its historical warning construction.
        codes.extend(strict.warning_codes)

    degraded = _audit_degraded_input(prepared)
    if degraded["status"] != "clean":
        _install_degraded_gate(provenance, degraded)
        degraded_codes = list(degraded["warning_codes"])
        codes.extend(degraded_codes)
        codes.append(_WARNING_DEGRADED_CANDIDATE_PROMOTION_BLOCKED)
        # Monotone molecule-first order for degraded inputs: the
        # molecule-capable stages (bond-order inference, executed at most
        # once per request, then proximity-molecule perception) run before
        # any graph-only template projection or observation stage, so a
        # stronger molecule-level recovery is never shadowed by an early
        # partial graph artifact.  Candidate/registry promotion stays
        # blocked; the block label is carried by every degraded artifact.
        provenance["degraded"]["promotion_policy"] = "observation_ladder_only"
        if infer_bond_orders:
            inferred_result = _attempt_bond_order_inference(
                prepared,
                strict,
                provenance,
                base_codes=codes,
            )
            if inferred_result is not None:
                provenance["degraded"]["candidate_registry_policy"] = (
                    "inferred_candidates_exposed_unqualified"
                )
                return inferred_result

        proximity_mol = _attempt_proximity_molecule(
            prepared, strict, provenance, base_codes=codes,
        )
        if proximity_mol is not None:
            return proximity_mol

        partial_result = _attempt_degraded_template_projection(
            prepared, strict, provenance, base_codes=codes, audit=degraded
        )
        if partial_result is not None:
            return partial_result

        # Observation-only ladder as final fallback
        result = _attempt_proximity_topology(
            prepared, strict, provenance, base_codes=codes,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        )
        if result is None:
            result = _attempt_connectivity_topology(
                prepared, strict, provenance, base_codes=codes,
                minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            )
        if result is None:
            result = _attempt_partial(
                prepared, strict, provenance, base_codes=codes
            )
        if result is None:
            result = _attempt_raw(prepared, strict, provenance, base_codes=codes)
        if result is not None:
            return result
        return _failed_result(provenance, strict, base_codes=codes)

    try:
        mapping_candidates, mapping_reason = _mapping_aware_candidates(
            prepared, strict, provenance, codes
        )
    except Exception as exc:
        mapping_candidates = []
        mapping_reason = "mapping_aware_stage_exception"
        provenance.setdefault("mapping_aware", {})["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        provenance["ladder_attempts"].append({
            "stage": SOURCE_MAPPING_AWARE,
            "quality": QUALITY_MEDIUM,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
    if mapping_candidates:
        if not promotion_blocked:
            return _mapping_aware_result(
                mapping_candidates, strict, provenance, codes
            )
        # Maximum acceptance: when blocked, still promote but let the
        # assessment path run first — it may produce a higher-quality result.
        # Record mapping-aware as fallback and continue.
        for candidate in mapping_candidates:
            row = dict(candidate)
            row.update({
                "status": "diagnostic_promoted",
                "quality": QUALITY_MEDIUM,
                "source": SOURCE_MAPPING_AWARE,
            })
            provenance.setdefault("readable_candidates", []).append(row)
    if mapping_reason not in {None, "no_ambiguous_mapping_evidence"}:
        provenance.setdefault("mapping_aware", {})["skipped_reason"] = mapping_reason

    rows = list(strict.route_results) if strict is not None else []
    diagnostic_rows = _diagnostic_rows(rows)
    if diagnostic_rows:
        codes.append(_WARNING_F_H_DIAGNOSTIC_ONLY)
        provenance["f_h_diagnostic_rows"] = diagnostic_rows

    candidates = _assessment_candidates(strict, provenance, codes)
    if candidates and promotion_blocked:
        # Maximum acceptance: promote with diagnostic label instead of
        # blocking.  Quality stays at hypothesis level and the original
        # warning codes (including F_H_DIAGNOSTIC_ONLY) are preserved.
        for candidate in candidates:
            row = dict(candidate)
            row.update({
                "status": "diagnostic_promoted",
                "quality": QUALITY_MEDIUM,
                "source": SOURCE_ENSEMBLE_CANDIDATE,
            })
            provenance.setdefault("readable_candidates", []).append(row)
        # Do NOT clear candidates — let the normal selection path run
        # (preserves high/medium quality from assessment).  This branch is
        # strict promotion blocking on an otherwise non-degraded input
        # (integrity conflict codes), NOT input degradation: the policy is
        # recorded under the existing fallback_block ledger so the
        # "degraded" provenance key stays reserved for the input-degradation
        # gate and the two conditions can never be confused downstream.
        provenance.setdefault("fallback_block", {})["promotion_policy"] = (
            "diagnostic_promoted"
        )
    if candidates:
        primary, alternatives = _select_primary(candidates)
        provenance["candidates"] = {
            "distinct_identity_count": len(candidates),
            "primary": primary,
            "assessment_source": (
                "strict.output_evidence.candidate_assessment"
            ),
            "identity_groups": [
                {
                    "full_inchikey": group["full_inchikey"],
                    "canonical_smiles": group["canonical_smiles"],
                    "routes": group["routes"],
                    "primary_route": group["primary_route"],
                    "supporting_route_count": group[
                        "supporting_route_count"
                    ],
                    "admitted_route_row_count": group[
                        "admitted_route_row_count"
                    ],
                }
                for group in candidates
            ],
        }
        if len(candidates) == 1:
            provenance["ladder"] = "candidate_unique"
            provenance["ladder_attempts"].append(
                {
                    "stage": SOURCE_UNIQUE_CANDIDATE,
                    "quality": QUALITY_HIGH,
                    "ok": True,
                    "supporting_route_count": primary[
                        "supporting_route_count"
                    ],
                }
            )
            return _candidate_result(
                quality=QUALITY_HIGH,
                source=SOURCE_UNIQUE_CANDIDATE,
                primary=primary,
                alternatives=[],
                warning_codes=codes,
                provenance=provenance,
                strict=strict,
            )
        codes.append(_WARNING_MULTIPLE_CANDIDATES)
        provenance["ladder"] = "candidate_ensemble"
        provenance["ladder_attempts"].append(
            {
                "stage": SOURCE_ENSEMBLE_CANDIDATE,
                "quality": QUALITY_MEDIUM,
                "ok": True,
                "distinct_identity_count": len(candidates),
                "primary_route": primary["primary_route"],
                "primary_supporting_route_count": primary[
                    "supporting_route_count"
                ],
            }
        )
        candidate_rows = [
            {
                **dict(candidate),
                "status": "not_selected",
                "quality": QUALITY_MEDIUM,
                "source": SOURCE_ENSEMBLE_CANDIDATE,
            }
            for candidate in candidates
        ]
        provenance["candidates"]["primary"] = None
        provenance["candidates"]["selection_policy"] = (
            "no automatic chemical identity selection under ambiguity"
        )
        return _build_result(
            status=STATUS_SUCCESS,
            quality=QUALITY_MEDIUM,
            source=SOURCE_ENSEMBLE_CANDIDATE,
            result={"candidates": candidate_rows},
            smiles=None,
            graph=None,
            ambiguous=True,
            warning_codes=codes,
            alternatives=candidate_rows,
            provenance=provenance,
            strict=strict,
            chemistry_candidates=candidate_rows,
            candidate_rigor="C2:H",
        )

    registry_result = _attempt_registry_template(
        prepared,
        strict,
        provenance,
        base_codes=codes,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        registry_assembler=registry_assembler,
    )
    if registry_result is not None:
        # Maximum acceptance: always return; warning codes carry labels.
        return registry_result

    if infer_bond_orders:
        inferred_result = _attempt_bond_order_inference(
            prepared,
            strict,
            provenance,
            base_codes=codes,
        )
        if inferred_result is not None:
            # Maximum acceptance: always return; warning codes carry labels.
            return inferred_result

    # F/H diagnostic promotion: diagnostic rows never elevate a candidate and
    # never reach the qualified tier.  When every readable F/H row agrees on
    # one chemical identity, that single identity may be exposed at hypothesis
    # quality capped at C1:H; divergent identities are all retained as an
    # ambiguous, unqualified candidate set with no selected primary.
    if diagnostic_rows:
        f_h_result = _f_h_diagnostic_promotion(
            diagnostic_rows, strict, provenance, codes
        )
        if f_h_result is not None:
            return f_h_result

    # Proximity molecule fallback: RDKit distance perception with bond
    # orders as the last molecule-level resort before topology/raw.
    proximity_mol_result = _attempt_proximity_molecule(
        prepared, strict, provenance, base_codes=codes,
    )
    if proximity_mol_result is not None:
        return proximity_mol_result

    # RDKit / raw ladder.  Each stage returns None (and records itself in
    # provenance) when it cannot produce a readable non-empty structure.
    result = _attempt_proximity_topology(
        prepared,
        strict,
        provenance,
        base_codes=codes,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
    )
    if result is None:
        result = _attempt_connectivity_topology(
            prepared,
            strict,
            provenance,
            base_codes=codes,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        )
    if result is None:
        result = _attempt_partial(
            prepared, strict, provenance, base_codes=codes
        )
    if result is None:
        result = _attempt_raw(prepared, strict, provenance, base_codes=codes)
    if result is not None:
        return result

    return _failed_result(provenance, strict, base_codes=codes)


def _opaque_input_payload(provenance: dict[str, Any]) -> dict[str, Any]:
    prepared = provenance.get("prepared") or {}
    input_path = prepared.get("input_path")
    payload: dict[str, Any] = {
        "artifact_type": "opaque_input",
        "input_path": input_path,
        "format": prepared.get("format") or prepared.get("source_format"),
        "error_code": provenance.get("error_code"),
        "error": provenance.get("error"),
        "failure_reason": provenance.get("failure_reason"),
    }
    if input_path:
        path = Path(str(input_path))
        try:
            data = path.read_bytes()
            payload.update({
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "input_byte_count": len(data),
                "input_bytes_readable": True,
            })
        except (OSError, ValueError):
            payload["input_bytes_readable"] = False
    return payload


def _failed_result(
    provenance: dict[str, Any],
    strict: StrictReconstructionResult | None,
    *,
    base_codes: list[str],
) -> ReconstructionResult:
    """Lowest degradation tier; every scientific input remains status-closed."""
    provenance["ladder"] = "opaque_input"
    provenance.setdefault(
        "failure_reason", "no readable non-empty structure at any ladder stage"
    )
    codes = list(base_codes)
    if strict is not None:
        codes.extend(strict.warning_codes)
    codes.append(_WARNING_NO_READABLE_STRUCTURE)
    payload = _opaque_input_payload(provenance)
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_OPAQUE,
        source=SOURCE_OPAQUE,
        result=payload,
        smiles=None,
        graph=None,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
    )


# --------------------------------------------------------------------------
# Ladder stages 1-3: strict V6 and candidates
# --------------------------------------------------------------------------


def _exact_result(
    strict: StrictReconstructionResult,
    provenance: dict[str, Any],
    *,
    additional_codes: list[str] | None = None,
) -> ReconstructionResult:
    """Ladder stage 1: strict V6 success, short-circuit, no fallbacks."""
    codes = list(strict.warning_codes) + list(additional_codes or [])
    provenance["ladder"] = "exact"
    provenance["ladder_attempts"].append(
        {
            "stage": SOURCE_EXACT,
            "quality": QUALITY_EXACT,
            "ok": True,
            "strict_path_used": strict.path_used,
            "strict_qualified_success": bool(strict.qualified_success),
        }
    )
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_EXACT,
        source=SOURCE_EXACT,
        result=strict.output_smiles,
        smiles=strict.output_smiles,
        graph=None,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
    )


def _strict_fallback_blocked(
    strict: StrictReconstructionResult | None,
    strict_error: str | None,
) -> list[str]:
    """Return block reason codes for clearly untrustworthy strict results."""
    reasons: list[str] = []
    # An implementation/runtime exception is not chemical evidence and cannot
    # justify erasing observable RDKit/raw candidates. It remains in
    # provenance["strict_error"] while the degradation ladder continues.
    if strict is not None:
        for code in strict.warning_codes:
            if code in _FALLBACK_BLOCK_WARNING_CODES:
                reasons.append(code)
    return reasons


def _mapping_aware_candidates(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    codes: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Reassemble every bounded A/E mapping combination before promotion."""
    if strict is None:
        return [], "strict_result_missing"
    route_inputs = []
    for row in strict.route_results:
        if row.get("route") not in {"a", "e"} or row.get("status") != "success":
            continue
        evidence = row.get("evidence_dimensions_input")
        if not isinstance(evidence, dict):
            continue
        residues = evidence.get("residue_evidence")
        if not isinstance(residues, list):
            continue
        ambiguous = any(
            int(item.get("mapping_candidate_count", 1)) > 1
            or item.get("external_attachment_mapping_unique") is False
            for item in residues if isinstance(item, dict)
        )
        if ambiguous:
            route_inputs.append((str(row["route"]), evidence))
    if not route_inputs:
        return [], "no_ambiguous_mapping_evidence"

    # Prefer A, then E, and never combine evidence from different routes.
    route, source = sorted(route_inputs, key=lambda item: item[0])[0]
    choices = []
    for residue in sorted(
        source.get("residue_evidence", []),
        key=lambda item: int(item.get("residue_position", -1)),
    ):
        candidates = residue.get("mapping_candidates")
        if not isinstance(candidates, list) or not candidates:
            return [], "mapping_candidate_ledger_missing"
        choices.append((int(residue["residue_position"]), candidates))
    combination_count = 1
    for _position, candidates in choices:
        combination_count *= len(candidates)
    cap = 20000
    provenance["mapping_aware"] = {
        "route": route,
        "candidate_counts_by_position": {
            str(position): len(candidates) for position, candidates in choices
        },
        "combination_count": combination_count,
        "combination_cap": cap,
        "truncated": combination_count > cap,
    }
    if combination_count > cap:
        return [], "mapping_combination_limit_exceeded"

    from .paths.path_a import generate_with_evidence

    identities: dict[str, dict[str, Any]] = {}
    for selected in product(*(candidates for _position, candidates in choices)):
        overrides = {
            position: selected[index]["serial_to_template_atom_index"]
            for index, (position, _candidates) in enumerate(choices)
        }
        smiles, error, evidence = generate_with_evidence(
            str(prepared.pdb_path),
            str(prepared.chain_id),
            geometric_cyclization=route == "e",
            mapping_overrides=overrides,
        )
        if error or not isinstance(smiles, str) or not smiles:
            continue
        closures = evidence.get("closure_evidence", [])
        if not all(
            isinstance(closure, dict)
            and closure.get("endpoints_resolved_uniquely") is True
            and closure.get("materialized") is True
            and closure.get("identity_changes_when_removed") is True
            for closure in closures
        ):
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        key = Chem.MolToInchiKey(molecule)
        if not key:
            continue
        fingerprint = tuple(
            selected[index].get("candidate_fingerprint")
            for index in range(len(selected))
        )
        identities.setdefault(key, {
            "full_inchikey": key,
            "canonical_smiles": canonical,
            "route": route,
            "mapping_candidate_fingerprints": [],
            "closure_evidence": closures,
        })["mapping_candidate_fingerprints"].append(list(fingerprint))

    for candidate in identities.values():
        # Preserve a stable, auditable representation without using set
        # iteration order.
        seen = []
        for fingerprint in candidate["mapping_candidate_fingerprints"]:
            if fingerprint not in seen:
                seen.append(fingerprint)
        candidate["mapping_candidate_fingerprints"] = seen
    return list(identities.values()), None


def _mapping_aware_result(
    candidates: list[dict[str, Any]],
    strict: StrictReconstructionResult,
    provenance: dict[str, Any],
    codes: list[str],
) -> ReconstructionResult:
    candidates.sort(key=lambda item: (item["full_inchikey"], item["canonical_smiles"]))
    primary = candidates[0]
    alternatives = [
        {
            "full_inchikey": item["full_inchikey"],
            "canonical_smiles": item["canonical_smiles"],
            "route": item["route"],
            "mapping_candidate_fingerprints": item["mapping_candidate_fingerprints"],
        }
        for item in candidates[1:]
    ]
    invariant = len(candidates) == 1
    warning = (
        _WARNING_MAPPING_AMBIGUOUS_IDENTITY_INVARIANT
        if invariant else _WARNING_MAPPING_IDENTITY_DIVERGENT
    )
    provenance["ladder"] = SOURCE_MAPPING_AWARE
    provenance["mapping_aware"].update({
        "status": "identity_invariant" if invariant else "identity_divergent",
        "distinct_identity_count": len(candidates),
        "primary": primary,
        "alternatives": alternatives,
    })
    provenance["ladder_attempts"].append({
        "stage": SOURCE_MAPPING_AWARE,
        "quality": QUALITY_HIGH if invariant else QUALITY_MEDIUM,
        "ok": True,
        "distinct_identity_count": len(candidates),
    })
    if invariant:
        return _candidate_result(
            quality=QUALITY_HIGH,
            source=SOURCE_MAPPING_AWARE,
            primary={"output_smiles": primary["canonical_smiles"], **primary},
            alternatives=[],
            warning_codes=list(codes) + [warning],
            provenance=provenance,
            strict=strict,
        )
    candidate_rows = [
        {
            **dict(item),
            "status": "not_selected",
            "quality": QUALITY_MEDIUM,
            "source": SOURCE_MAPPING_AWARE,
        }
        for item in candidates
    ]
    provenance["mapping_aware"]["primary"] = None
    provenance["mapping_aware"]["selection_policy"] = (
        "no automatic chemical identity selection under mapping divergence"
    )
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_MEDIUM,
        source=SOURCE_MAPPING_AWARE,
        result={"candidates": candidate_rows},
        smiles=None,
        graph=None,
        ambiguous=True,
        warning_codes=list(codes) + [warning],
        alternatives=candidate_rows,
        provenance=provenance,
        strict=strict,
        chemistry_candidates=candidate_rows,
        candidate_rigor="C2:H",
    )


def _assessment_candidates(
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    codes: list[str],
) -> list[dict[str, Any]]:
    """Consume only ``output_evidence["candidate_assessment"]``.

    Route rows are never re-read to recompute candidates: ``routes`` and
    ``supporting_route_count`` come directly from the assessment, so
    duplicate route rows cannot inflate support counts.  A missing or
    semantically invalid assessment skips the candidate stage with an
    explicit warning and provenance entry.
    """
    if strict is None:
        return []
    evidence = strict.output_evidence
    if not isinstance(evidence, dict):
        evidence = {}
    assessment = evidence.get("candidate_assessment")
    if not isinstance(assessment, dict):
        codes.append(_WARNING_CANDIDATE_ASSESSMENT_MISSING)
        provenance["candidate_assessment"] = {
            "status": "missing",
            "reason": (
                "strict.output_evidence.candidate_assessment is absent"
            ),
        }
        return []
    try:
        validate_candidate_assessment_semantics(assessment)
        _validate_assessment_binding(assessment, strict)
    except Exception as exc:
        codes.append(_WARNING_CANDIDATE_ASSESSMENT_INVALID)
        provenance["candidate_assessment"] = {
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return []
    provenance["candidate_assessment"] = {
        "status": "valid",
        "schema_version": assessment.get("schema_version"),
    }
    raw_candidates = assessment.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return []

    groups: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        full_inchikey = candidate.get("full_inchikey")
        canonical_smiles = candidate.get("canonical_smiles")
        routes = candidate.get("routes")
        if (
            not isinstance(full_inchikey, str)
            or not full_inchikey
            or not isinstance(canonical_smiles, str)
            or not canonical_smiles
            or not isinstance(routes, list)
            or not routes
        ):
            continue
        count = candidate.get("supporting_route_count")
        if not isinstance(count, int):
            count = len(routes)
        best_route = min(
            routes,
            key=lambda route: _ROUTE_PRIORITY.get(
                route, len(_ROUTE_PRIORITY)
            ),
        )
        groups.append(
            {
                "full_inchikey": full_inchikey,
                "canonical_smiles": canonical_smiles,
                "routes": list(routes),
                "primary_route": best_route,
                "supporting_route_count": count,
                "admitted_route_row_count": candidate.get(
                    "admitted_route_row_count"
                ),
            }
        )
    groups.sort(
        key=lambda group: (
            -group["supporting_route_count"],
            _ROUTE_PRIORITY.get(group["primary_route"], len(_ROUTE_PRIORITY)),
            group["full_inchikey"],
            group["canonical_smiles"],
        )
    )
    return groups


def _route_identity_input_sha256(row: dict[str, Any]) -> str:
    payload = {
        "route": row.get("route"),
        "status": row.get("status"),
        "output_smiles": row.get("output_smiles"),
        "output_inchikey": row.get("output_inchikey"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_assessment_binding(
    assessment: dict[str, Any],
    strict: StrictReconstructionResult,
) -> None:
    """Bind a valid candidate assessment to its enclosing strict result."""
    context = assessment.get("result_context")
    expected_context = {
        "status": strict.status,
        "support_status": strict.support_status,
        "qualified_success": bool(strict.qualified_success),
        "repair_codes": sorted(set(strict.repair_codes)),
    }
    if context != expected_context:
        raise ValueError(
            "candidate assessment result_context does not match strict result"
        )

    audit_hashes = sorted(
        audit["identity_input_sha256"]
        for audit in assessment.get("route_identity_audits", [])
    )
    route_hashes = sorted(
        _route_identity_input_sha256(row) for row in strict.route_results
    )
    if audit_hashes != route_hashes:
        raise ValueError(
            "candidate assessment route audits do not match strict "
            "route_results"
        )


def _select_primary(
    groups: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary_group = groups[0]
    primary = {
        "full_inchikey": primary_group["full_inchikey"],
        "canonical_smiles": primary_group["canonical_smiles"],
        "routes": list(primary_group["routes"]),
        "primary_route": primary_group["primary_route"],
        "supporting_route_count": primary_group["supporting_route_count"],
        "admitted_route_row_count": primary_group["admitted_route_row_count"],
        "output_smiles": primary_group["canonical_smiles"],
    }
    alternatives = [
        {
            "full_inchikey": group["full_inchikey"],
            "canonical_smiles": group["canonical_smiles"],
            "routes": list(group["routes"]),
            "primary_route": group["primary_route"],
            "supporting_route_count": group["supporting_route_count"],
            "admitted_route_row_count": group["admitted_route_row_count"],
        }
        for group in groups[1:]
    ]
    return primary, alternatives


def _candidate_result(
    *,
    quality: str,
    source: str,
    primary: dict[str, Any],
    alternatives: list[dict[str, Any]],
    warning_codes: list[str],
    provenance: dict[str, Any],
    strict: StrictReconstructionResult | None,
) -> ReconstructionResult:
    return _build_result(
        status=STATUS_SUCCESS,
        quality=quality,
        source=source,
        result=primary["output_smiles"],
        smiles=primary["output_smiles"],
        graph=None,
        ambiguous=bool(alternatives),
        warning_codes=warning_codes,
        alternatives=alternatives,
        provenance=provenance,
        strict=strict,
    )


def _attempt_bond_order_inference(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
) -> ReconstructionResult | None:
    """Expose inferred chemistry without changing strict qualification.

    The inference portfolio binds every admitted identity to the observed
    heavy-atom composition.  Multi-family or template-supported candidates
    may populate the historical downstream SMILES handoff; single-engine and
    geometry-only candidates remain explicit opt-in materialization inputs.
    """
    from .bond_order_inference import infer_bond_order_candidates

    codes = list(base_codes)
    strict_candidates = _assessment_candidates(strict, provenance, codes)
    try:
        report = infer_bond_order_candidates(
            str(prepared.pdb_path),
            str(prepared.chain_id),
            strict_candidates=strict_candidates,
        )
    except Exception as exc:
        provenance["bond_order_inference"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        provenance["ladder_attempts"].append(
            {
                "stage": "bond_order_inference",
                "quality": QUALITY_CANDIDATE,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None

    selected = report.get("selected_candidate")
    public_report = dict(report)
    if isinstance(selected, dict):
        public_selected = {
            key: value
            for key, value in selected.items()
            if key != "candidate_graph"
        }
        public_report["selected_candidate"] = public_selected
    provenance["bond_order_inference"] = public_report
    if not isinstance(selected, dict):
        provenance["ladder_attempts"].append(
            {
                "stage": "bond_order_inference",
                "quality": QUALITY_CANDIDATE,
                "ok": False,
                "candidate_count": int(report.get("candidate_count") or 0),
            }
        )
        return None

    quality = str(selected.get("quality") or QUALITY_CANDIDATE)
    if quality not in {
        QUALITY_HIGH,
        QUALITY_CANDIDATE,
        QUALITY_HYPOTHESIS,
    }:
        quality = QUALITY_CANDIDATE
    source = {
        QUALITY_HIGH: SOURCE_BOND_ORDER_CONSENSUS,
        QUALITY_CANDIDATE: SOURCE_BOND_ORDER_CANDIDATE,
        QUALITY_HYPOTHESIS: SOURCE_BOND_ORDER_HYPOTHESIS,
    }[quality]
    candidate_smiles = str(selected["canonical_smiles"])
    candidate_graph = selected.get("candidate_graph")
    top_level_smiles = (
        candidate_smiles if quality == QUALITY_HIGH else None
    )
    identity_groups = list(report.get("identity_groups") or [])
    selected_id = selected.get("candidate_id")
    alternatives = [
        group
        for group in identity_groups
        if group.get("candidate_id") != selected_id
    ]

    codes.extend(
        [
            _WARNING_BOND_ORDER_CANDIDATE,
            _WARNING_INFERRED_CHEMISTRY_UNQUALIFIED,
        ]
    )
    if len(identity_groups) > 1:
        codes.append(_WARNING_INFERENCE_ENGINES_DISAGREE)
    if report.get("openbabel_available") is not True:
        codes.append(_WARNING_OPENBABEL_UNAVAILABLE)
    provenance["ladder"] = source
    provenance["ladder_attempts"].append(
        {
            "stage": "bond_order_inference",
            "quality": quality,
            "ok": True,
            "candidate_count": len(identity_groups),
            "selected_candidate_id": selected_id,
            "selected_evidence_class": selected.get("evidence_class"),
            "selected_rigor": selected.get("rigor"),
            "selected_engine": selected.get("selected_engine"),
            "selected_has_coordinate_graph": isinstance(
                candidate_graph, dict
            ),
        }
    )
    return _build_result(
        status=STATUS_SUCCESS,
        quality=quality,
        source=source,
        result=(
            top_level_smiles
            if top_level_smiles is not None
            else candidate_graph or public_report["selected_candidate"]
        ),
        smiles=top_level_smiles,
        graph=candidate_graph,
        ambiguous=bool(alternatives) or bool(report.get("selection_tied")),
        warning_codes=codes,
        alternatives=alternatives,
        provenance=provenance,
        strict=strict,
        candidate_smiles=candidate_smiles,
        candidate_graph=candidate_graph,
        chemistry_candidates=identity_groups,
        bond_order_inference=public_report,
        candidate_rigor=str(selected.get("rigor") or ""),
    )


def _diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        route = str(row.get("route", ""))
        if route not in _DIAGNOSTIC_ROUTES:
            continue
        diagnostics.append(
            {
                "route": route,
                "status": str(row.get("status", "")),
                "output_smiles": row.get("output_smiles"),
                "output_inchikey": row.get("output_inchikey"),
                "diagnostic_only": True,
            }
        )
    diagnostics.sort(
        key=lambda row: (row["route"], str(row["output_inchikey"] or ""))
    )
    return diagnostics


def _canonical_identity(smiles: str) -> tuple[str | None, str | None]:
    """Return ``(inchikey, canonical_smiles)`` for a parseable SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    try:
        inchikey = Chem.MolToInchiKey(molecule) or None
    except Exception:
        inchikey = None
    return inchikey, Chem.MolToSmiles(molecule)


def _f_h_diagnostic_promotion(
    diagnostic_rows: list[dict[str, Any]],
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    codes: list[str],
) -> ReconstructionResult | None:
    """Expose F/H diagnostic identities without a qualified-tier claim.

    F/H route rows are diagnostic only: they never elevate a candidate and
    never produce a qualified artifact.  Identity agreement across every
    readable F/H row permits exactly one capped C1:H hypothesis identity.
    Divergent identities are never collapsed to an arbitrary primary: every
    identity is retained in an ordered, ambiguous, unqualified candidate set
    (``smiles=None``, ``ambiguous=True``).
    """
    identities: dict[str, dict[str, Any]] = {}
    for row in diagnostic_rows:
        smiles_value = row.get("output_smiles")
        if not isinstance(smiles_value, str) or not smiles_value:
            continue
        inchikey, canonical = _canonical_identity(smiles_value)
        if canonical is None:
            # Unreadable diagnostic SMILES: retained above in
            # provenance["f_h_diagnostic_rows"], never promoted.
            continue
        key = inchikey or canonical
        entry = identities.setdefault(key, {
            "full_inchikey": inchikey,
            "canonical_smiles": canonical,
            "routes": [],
        })
        route = str(row.get("route", ""))
        if route and route not in entry["routes"]:
            entry["routes"].append(route)
    if not identities:
        return None
    ordered = sorted(
        identities.values(),
        key=lambda entry: (
            entry["full_inchikey"] or "",
            entry["canonical_smiles"],
        ),
    )
    agreed = len(ordered) == 1
    provenance["f_h_diagnostic_identity"] = {
        "status": "identity_agreement" if agreed else "identity_divergent",
        "distinct_identity_count": len(ordered),
        "identities": [
            {
                "full_inchikey": entry["full_inchikey"],
                "canonical_smiles": entry["canonical_smiles"],
                "routes": sorted(entry["routes"]),
            }
            for entry in ordered
        ],
        "rigor_cap": "C1:H",
        "promotion_policy": (
            "single agreed identity exposed at hypothesis quality (C1:H cap)"
            if agreed
            else (
                "all identities retained; no automatic selection under "
                "identity divergence"
            )
        ),
    }
    provenance["ladder"] = "f_h_diagnostic_promotion"
    if agreed:
        identity = ordered[0]
        provenance["ladder_attempts"].append({
            "stage": "f_h_diagnostic_promotion",
            "quality": QUALITY_HYPOTHESIS,
            "ok": True,
            "distinct_identity_count": 1,
            "routes": sorted(identity["routes"]),
        })
        return _build_result(
            status=STATUS_SUCCESS,
            quality=QUALITY_HYPOTHESIS,
            source="f_h_diagnostic_promotion",
            result=identity["canonical_smiles"],
            smiles=identity["canonical_smiles"],
            graph=None,
            ambiguous=False,
            warning_codes=list(codes),
            alternatives=[],
            provenance=provenance,
            strict=strict,
            candidate_smiles=identity["canonical_smiles"],
            candidate_graph=None,
            candidate_rigor="C1:H",
        )
    codes.append(_WARNING_F_H_IDENTITY_DIVERGENT)
    candidate_rows = [
        {
            "full_inchikey": entry["full_inchikey"],
            "canonical_smiles": entry["canonical_smiles"],
            "routes": sorted(entry["routes"]),
            "status": "not_selected",
            "quality": QUALITY_HYPOTHESIS,
            "source": "f_h_diagnostic_promotion",
        }
        for entry in ordered
    ]
    provenance["ladder_attempts"].append({
        "stage": "f_h_diagnostic_promotion",
        "quality": QUALITY_HYPOTHESIS,
        "ok": True,
        "distinct_identity_count": len(ordered),
        "ambiguous": True,
    })
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_HYPOTHESIS,
        source="f_h_diagnostic_promotion",
        result={"candidates": candidate_rows},
        smiles=None,
        graph=None,
        ambiguous=True,
        warning_codes=list(codes),
        alternatives=candidate_rows,
        provenance=provenance,
        strict=strict,
        chemistry_candidates=candidate_rows,
        candidate_rigor="C1:H",
    )


# --------------------------------------------------------------------------
# Degraded input audit and protected observation projection
# --------------------------------------------------------------------------


def _audit_degraded_input(prepared: Any) -> dict[str, Any]:
    """Audit coordinate completeness without weakening strict V6 semantics."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "clean",
        "entry_class": "clean",
        "primary_policy": "existing_ladder",
        "residues": [],
        "warning_codes": [],
    }
    try:
        atoms = _parse_pdb_atom_records(Path(prepared.pdb_path))
    except Exception as exc:
        result.update({
            "status": "unrecoverable",
            "entry_class": "unrecoverable",
            "warning_codes": [_WARNING_DEGRADED_TRUNCATED_RESIDUE],
            "error": f"{type(exc).__name__}: {exc}",
        })
        return result
    by_residue: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for atom in atoms:
        key = (
            str(atom["chain"]), int(atom["residue_number"]),
            "", str(atom["residue"]).strip().upper(),
        )
        by_residue.setdefault(key, []).append(atom)
    ordered = sorted(by_residue.items(), key=lambda item: item[0])
    result["residue_counts"] = {"total": len(ordered)}
    classes: list[str] = []
    previous_number: int | None = None
    for position, (key, residue_atoms) in enumerate(ordered, start=1):
        chain, residue_number, icode, residue_name = key
        row: dict[str, Any] = {
            "position": position,
            "chain": chain,
            "residue_number": residue_number,
            "icode": icode,
            "observed_resname": residue_name,
            "observed_atom_names": sorted(
                str(atom["name"]).strip().upper() for atom in residue_atoms
            ),
            "mapped_atom_names": [],
            "missing_template_atom_names": [],
            "observed_unresolved_atom_names": [],
            "element_mismatches": [],
            "backbone": {
                "required": ["N", "CA", "C", "O"],
                "present": [], "missing": [], "passed": False,
            },
        }
        if previous_number is not None and residue_number > previous_number + 1:
            row["sequence_gap"] = True
        previous_number = residue_number
        try:
            from .paths.residue_template_factory import get_residue_template
            from .core.pdb_parser import standard_pdb_atom_name_map
            template = get_residue_template(residue_name)
            name_map = standard_pdb_atom_name_map(residue_name, template.smiles)
            if not name_map:
                raise ValueError("no trusted PDB atom-name map")
            template_atoms = {
                name.upper(): template.mol.GetAtomWithIdx(index).GetSymbol().upper()
                for name, index in name_map.items()
            }
            row["resolved_symbol"] = template.symbol
            row["template_graph_sha256"] = template.free_graph_sha256
            row["mapping_basis"] = "exact_pdb_atom_names"
            seen_names: dict[str, list[dict[str, Any]]] = {}
            for atom in residue_atoms:
                seen_names.setdefault(
                    str(atom["name"]).strip().upper(), []
                ).append(atom)
            mapped: dict[str, dict[str, Any]] = {}
            for name, observed in sorted(seen_names.items()):
                if name not in template_atoms or len(observed) != 1:
                    row["observed_unresolved_atom_names"].append(name)
                    continue
                atom = observed[0]
                expected_element = template_atoms[name]
                actual_element = str(atom["element"]).strip().upper()
                if actual_element != expected_element:
                    row["element_mismatches"].append({
                        "name": name, "expected": expected_element,
                        "observed": actual_element,
                        "serial": int(atom["serial"]),
                    })
                    row["observed_unresolved_atom_names"].append(name)
                    continue
                mapped[name] = atom
            row["mapped_atom_names"] = sorted(mapped)
            missing = sorted(set(template_atoms) - set(mapped))
            row["missing_template_atom_names"] = missing
            backbone_present = sorted(
                name for name in ("N", "CA", "C", "O") if name in mapped
            )
            backbone_missing = sorted(
                set(("N", "CA", "C", "O")) - set(backbone_present)
            )
            row["backbone"].update({
                "present": backbone_present,
                "missing": backbone_missing,
                "passed": not backbone_missing,
            })
            row["mapped_serials"] = {
                name: int(atom["serial"]) for name, atom in sorted(mapped.items())
            }
            if backbone_missing:
                row["output_status"] = "template_partial_unavailable"
                classes.append("degraded_backbone")
            elif missing and all(
                name not in {"N", "CA", "C", "O"} for name in missing
            ):
                row["output_status"] = "sidechain_partial_recoverable"
                classes.append("degraded_sidechain")
            elif row["observed_unresolved_atom_names"] or row["element_mismatches"]:
                row["output_status"] = "template_partial_unavailable"
                classes.append("degraded_identity")
            else:
                row["output_status"] = "clean"
                classes.append("clean")
        except Exception as exc:
            row.update({
                "output_status": "template_partial_unavailable",
                "identity_error": f"{type(exc).__name__}: {exc}",
            })
            classes.append("degraded_identity")
        result["residues"].append(row)
    if not ordered:
        return result
    # PreparedCoordinateInput fixtures and caller-owned adapters may not carry
    # the immutable normalization audit.  They retain the historical ladder;
    # only normalized production inputs enter this quality-sensitive branch.
    if not bool((getattr(prepared, "audit", None) or {}).get("projection_applied")):
        result["status"] = "clean"
        result["entry_class"] = "clean"
        result["warning_codes"] = []
        return result
    if any(row.get("sequence_gap") for row in result["residues"]):
        result["warning_codes"].append(_WARNING_DEGRADED_SEQUENCE_GAP)
    if "degraded_identity" in classes:
        entry_class = "degraded_identity"
        result["warning_codes"].append(_WARNING_DEGRADED_UNKNOWN_RESIDUE)
    elif "degraded_backbone" in classes:
        entry_class = "degraded_backbone"
        result["warning_codes"].append(_WARNING_DEGRADED_MISSING_BACKBONE)
    elif "degraded_sidechain" in classes:
        entry_class = "degraded_sidechain"
        result["warning_codes"].append(_WARNING_DEGRADED_MISSING_SIDECHAIN)
    else:
        entry_class = "clean"
    if entry_class != "clean":
        result["status"] = "degraded"
        result["entry_class"] = entry_class
        result["primary_policy"] = (
            "partial_projection_only" if entry_class == "degraded_sidechain"
            else "observation_ladder_only"
        )
        result["warning_codes"].append(
            _WARNING_DEGRADED_TEMPLATE_PROJECTION
            if entry_class == "degraded_sidechain"
            else _WARNING_DEGRADED_TRUNCATED_RESIDUE
        )
    result["residue_counts"] = {
        "total": len(ordered),
        "clean": sum(row.get("output_status") == "clean" for row in result["residues"]),
        "degraded": sum(row.get("output_status") != "clean" for row in result["residues"]),
    }
    return result


def _install_degraded_gate(provenance: dict[str, Any], audit: dict[str, Any]) -> None:
    provenance["degraded"] = {
        "status": "degraded",
        "data_quality": audit,
        "candidate_registry_policy": "blocked_before_promotion",
    }


def _attempt_degraded_template_projection(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
    audit: dict[str, Any],
) -> ReconstructionResult | None:
    if audit.get("entry_class") != "degraded_sidechain":
        return None
    try:
        graph, details = _build_degraded_template_graph(
            Path(prepared.pdb_path), audit
        )
        if not graph.get("atoms"):
            return None
        payload = _json_roundtrip(graph)
    except Exception as exc:
        provenance.setdefault("degraded", {})["projection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return None
    provenance["ladder"] = SOURCE_DEGRADED_TEMPLATE
    provenance["degraded"].update({
        "entry_class": "degraded_sidechain",
        "projection": details,
        "graph_json_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    })
    provenance["ladder_attempts"].append({
        "stage": SOURCE_DEGRADED_TEMPLATE, "quality": QUALITY_PARTIAL,
        "ok": True, "atom_count": len(graph["atoms"]),
        "bond_count": len(graph["bonds"]),
    })
    codes = list(base_codes) + [_WARNING_DEGRADED_TEMPLATE_PROJECTION]
    return _build_result(
        status=STATUS_SUCCESS, quality=QUALITY_PARTIAL,
        source=SOURCE_DEGRADED_TEMPLATE, result=graph, smiles=None,
        graph=graph, ambiguous=False, warning_codes=codes, alternatives=[],
        provenance=provenance, strict=strict,
    )


def _build_degraded_template_graph(
    pdb_path: Path, audit: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    atoms = _parse_pdb_atom_records(pdb_path)
    rows = [row for row in audit.get("residues", [])]
    by_key = {
        (str(row["chain"]), int(row["residue_number"]), str(row["observed_resname"]).upper()): row
        for row in rows
    }
    graph_atoms: list[dict[str, Any]] = []
    serial_to_name: dict[int, str] = {}
    for atom in atoms:
        key = (str(atom["chain"]), int(atom["residue_number"]), str(atom["residue"]).upper())
        row = by_key.get(key)
        mapped = set(row.get("mapped_atom_names", [])) if row else set()
        name = str(atom["name"]).strip().upper()
        status = "mapped_observed" if name in mapped else "observed_unresolved"
        graph_atoms.append({**atom, "mapping_status": status})
        if status == "mapped_observed":
            serial_to_name[int(atom["serial"])] = name
    bonds: set[tuple[int, int]] = set()
    # Preserve explicit source edges only when both endpoint identities are
    # observed.  Template edges between two exact-mapped observed atoms are
    # also permitted, but remain order-unknown and never bridge a missing atom.
    for left, right in _parse_conect_pairs(pdb_path):
        if left in serial_to_name and right in serial_to_name:
            bonds.add(tuple(sorted((left, right))))
    for row in rows:
        mapped_serials = {
            str(name).upper(): int(serial)
            for name, serial in (row.get("mapped_serials") or {}).items()
        }
        if not mapped_serials:
            continue
        try:
            from .paths.residue_template_factory import get_residue_template
            from .core.pdb_parser import standard_pdb_atom_name_map
            template = get_residue_template(row["observed_resname"])
            name_map = standard_pdb_atom_name_map(
                row["observed_resname"], template.smiles
            )
            inverse = {
                int(index): str(name).upper() for name, index in name_map.items()
            }
            for bond in template.mol.GetBonds():
                left_name = inverse.get(bond.GetBeginAtomIdx())
                right_name = inverse.get(bond.GetEndAtomIdx())
                if left_name in mapped_serials and right_name in mapped_serials:
                    bonds.add(tuple(sorted((
                        mapped_serials[left_name], mapped_serials[right_name]
                    ))))
        except Exception:
            # The audit already enforces the strict identity gate.  A template
            # edge lookup failure must reduce the graph, never invent one.
            continue
    details = {
        "atom_count": len(graph_atoms),
        "bond_count": len(bonds),
        "missing_atoms": [
            {
                "chain": row["chain"], "residue_number": row["residue_number"],
                "residue": row["observed_resname"],
                "atom_names": list(row.get("missing_template_atom_names", [])),
            }
            for row in rows if row.get("missing_template_atom_names")
        ],
        "unresolved_atoms": [
            {
                "serial": int(atom["serial"]), "name": atom["name"],
                "residue": atom["residue"],
            }
            for atom in graph_atoms if atom["mapping_status"] == "observed_unresolved"
        ],
        "bond_source": "observed_explicit_endpoints_only",
    }
    graph_atoms.sort(key=lambda atom: int(atom["serial"]))
    graph = {
        "atoms": graph_atoms,
        "bonds": [
            {"a": left, "b": right, "order": None}
            for left, right in sorted(bonds)
        ],
    }
    return graph, details




def _default_registry_assembly(prepared: Any) -> dict[str, Any]:
    """Run the existing registry assembler against the selected normalized PDB.

    This is deliberately a small adapter around Path B.  It does not mutate
    the Unified library or invent an overlay.  A caller can replace it with an
    authoritative embedded-template adapter through ``registry_assembler``.
    """
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
                "error": audit.reason or "registry SMILES failed output audit",
                "route": "path_b_registry",
                "output_audit": {
                    "accepted": False,
                    "reason": audit.reason,
                },
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


def _registry_payload(value: Any) -> tuple[str | None, str | None, dict[str, Any]]:
    """Accept the tuple/string/dict forms used by existing registry adapters."""
    if isinstance(value, str):
        return value, None, {"route": "custom_registry"}
    if isinstance(value, tuple):
        smiles = value[0] if len(value) > 0 else None
        error = value[1] if len(value) > 1 else None
        return (
            str(smiles) if isinstance(smiles, str) and smiles else None,
            str(error) if error else None,
            {"route": "custom_registry"},
        )
    if isinstance(value, dict):
        smiles = value.get("smiles")
        error = value.get("error")
        metadata = {
            str(key): item
            for key, item in value.items()
            if key not in {"smiles", "error"}
        }
        return (
            str(smiles) if isinstance(smiles, str) and smiles else None,
            str(error) if error else None,
            metadata,
        )
    return None, "registry adapter returned an unsupported payload", {}


def _overlay_policy(require_empty_persistent_overlay: bool) -> dict[str, Any]:
    """Return the current overlay state and the policy applied to this call."""
    try:
        from .paths._map_utils import persistent_overlay_audit

        audit = persistent_overlay_audit()
    except Exception as exc:
        audit = {
            "status": "audit_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    status = str(audit.get("status", "unknown"))
    allowed = not require_empty_persistent_overlay or status == "empty"
    return {
        "require_empty_persistent_overlay": bool(
            require_empty_persistent_overlay
        ),
        "status": status,
        "allowed": allowed,
        "audit": audit,
    }


def _attempt_registry_template(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
    require_empty_persistent_overlay: bool,
    registry_assembler: Callable[[Any], Any] | None,
) -> ReconstructionResult | None:
    """Try Path B/authoritative-template recovery before geometry inference."""
    # A hand-constructed PreparedCoordinateInput may omit the normalization
    # audit.  The default registry adapter must not manufacture a result for
    # such an object; an explicit custom adapter may still opt in for tests or
    # a caller-owned authoritative template.
    if (
        registry_assembler is None
        and (getattr(prepared, "audit", None) or {}).get(
            "normalized_heavy_atom_count"
        )
        is None
    ):
        return None
    stage: dict[str, Any] = {
        "stage": SOURCE_REGISTRY,
        "quality": QUALITY_HIGH,
        "ok": False,
    }
    policy = _overlay_policy(require_empty_persistent_overlay)
    stage["overlay_policy"] = policy
    if not policy["allowed"]:
        stage["reason_codes"] = [_WARNING_REGISTRY_OVERLAY_BLOCKED]
        provenance["ladder_attempts"].append(stage)
        provenance["registry_template"] = {
            "status": "blocked",
            "overlay_policy": policy,
        }
        return None
    adapter = registry_assembler or _default_registry_assembly
    try:
        smiles, error, metadata = _registry_payload(adapter(prepared))
    except Exception as exc:
        smiles, error, metadata = (
            None,
            f"{type(exc).__name__}: {exc}",
            {},
        )
    audit: dict[str, Any] = {
        "status": "not_available",
        "route": metadata.get("route", "registry_template"),
        "overlay_policy": policy,
        **metadata,
    }
    if not smiles:
        stage["reason_codes"] = [_WARNING_REGISTRY_NOT_QUALIFIED]
        stage["error"] = error or "registry/template produced no SMILES"
        provenance["ladder_attempts"].append(stage)
        audit.update({"status": "not_available", "error": stage["error"]})
        provenance["registry_template"] = audit
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("registry/template SMILES is not parseable")
        observed_heavy = molecule.GetNumHeavyAtoms()
        expected_heavy = (getattr(prepared, "audit", None) or {}).get(
            "normalized_heavy_atom_count"
        )
        if expected_heavy is None:
            raise ValueError(
                "registry/template coordinate audit lacks normalized heavy-atom "
                "count"
            )
        if observed_heavy != int(expected_heavy):
            raise ValueError(
                "registry/template heavy-atom count mismatch: "
                f"expected {int(expected_heavy)}, observed {observed_heavy}"
            )
        source_elements = Counter(
            row["element"] for row in _parse_pdb_atom_records(Path(prepared.pdb_path))
            if str(row["element"]).upper() != "H"
        )
        candidate_elements = Counter(
            atom.GetSymbol().upper() for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() != 1
        )
        if source_elements != candidate_elements:
            raise ValueError(
                "registry/template heavy-element composition mismatch"
            )
        source_ring = (getattr(prepared, "audit", None) or {}).get(
            "macrocycle_ring_size"
        )
        candidate_ring = _largest_detected_ring_size(molecule)
        if source_ring is not None and int(source_ring) >= 8 and candidate_ring < 8:
            raise ValueError(
                "registry/template macrocycle topology mismatch: "
                f"source ring {int(source_ring)}, candidate ring {candidate_ring}"
            )
    except Exception as exc:
        stage["reason_codes"] = [_WARNING_REGISTRY_NOT_QUALIFIED]
        stage["error"] = f"{type(exc).__name__}: {exc}"
        provenance["ladder_attempts"].append(stage)
        audit.update({"status": "not_qualified", "error": stage["error"]})
        provenance["registry_template"] = audit
        if isinstance(smiles, str) and smiles:
            provenance.setdefault("readable_candidates", []).append({
                "status": "not_selected",
                "quality": QUALITY_MEDIUM,
                "source": SOURCE_REGISTRY,
                "smiles": smiles,
                "canonical_smiles": smiles,
                "qualification_failures": [_WARNING_REGISTRY_NOT_QUALIFIED],
                "error": stage["error"],
            })
        return None
    audit.update(
        {
            "status": "qualified",
            "heavy_atom_count": molecule.GetNumHeavyAtoms(),
            "formal_charge": Chem.GetFormalCharge(molecule),
            "template_source": (
                "embedded_mmcif_chem_comp"
                if (getattr(prepared, "audit", None) or {}).get(
                    "embedded_chem_comp_template_count", 0
                )
                else "unified_monomer_registry"
            ),
        }
    )
    stage.update(
        {
            "ok": True,
            "smiles": smiles,
            "heavy_atom_count": molecule.GetNumHeavyAtoms(),
            "audit": audit,
        }
    )
    provenance["ladder_attempts"].append(stage)
    provenance["registry_template"] = audit
    provenance["ladder"] = SOURCE_REGISTRY
    codes = list(base_codes) + [_WARNING_REGISTRY_TEMPLATE_ASSISTED]
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_HIGH,
        source=SOURCE_REGISTRY,
        result=smiles,
        smiles=smiles,
        graph=None,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
    )


# --------------------------------------------------------------------------
# Ladder stages 5-8: RDKit topology, explicit-only partial, raw graph
# --------------------------------------------------------------------------


def _retain_readable_candidate(
    provenance: dict[str, Any],
    *,
    source: str,
    mol: Any,
    qualification_failures: list[str],
    qualification: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "status": "not_selected",
        "quality": QUALITY_TOPOLOGY,
        "source": source,
        "result_origin": source,
        "candidate_available": bool(mol is not None and mol.GetNumAtoms() > 0),
        "qualification_failures": list(qualification_failures),
        "qualification": dict(qualification),
        "error": error,
        "smiles": None,
    }
    if candidate["candidate_available"]:
        try:
            graph = _mol_to_graph(mol)
            candidate["graph"] = graph
            candidate["graph_sha256"] = hashlib.sha256(
                _json_roundtrip(graph).encode("ascii")
            ).hexdigest()
            candidate["atom_count"] = int(mol.GetNumAtoms())
            candidate["bond_count"] = int(mol.GetNumBonds())
        except Exception as exc:
            candidate["graph_error"] = f"{type(exc).__name__}: {exc}"
    provenance.setdefault("readable_candidates", []).append(candidate)
    return candidate


def _attempt_proximity_topology(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
    minimum_macrocycle_ring_size: int,
) -> ReconstructionResult | None:
    """Ladder stage 4: proximity-bonded unsanitized RDKit topology."""
    pdb_path = Path(prepared.pdb_path)
    stage = {
        "stage": SOURCE_PROXIMITY,
        "quality": QUALITY_TOPOLOGY,
        "ok": False,
    }
    try:
        mol = _proximity_mol(pdb_path)
        reason_codes, qualification = _topology_qualification(
            mol,
            prepared,
            pdb_path,
            minimum_macrocycle_ring_size,
        )
        retained = _retain_readable_candidate(
            provenance,
            source=SOURCE_PROXIMITY,
            mol=mol,
            qualification_failures=reason_codes,
            qualification=qualification,
        )
        if reason_codes:
            stage["reason_codes"] = reason_codes
            stage["qualification"] = qualification
            stage["candidate_graph_sha256"] = retained.get("graph_sha256")
            provenance["ladder_attempts"].append(stage)
            return None
        metrics = _metrics_from_topology_qualification(qualification)
        graph = retained.get("graph") or _mol_to_graph(mol)
    except Exception as exc:
        stage["error"] = f"{type(exc).__name__}: {exc}"
        provenance["ladder_attempts"].append(stage)
        return None
    stage.update({"ok": True, **metrics, "qualification": {"passed": True}})
    provenance["ladder_attempts"].append(stage)
    provenance["ladder"] = "rdkit_proximity"
    return _topology_result(
        source=SOURCE_PROXIMITY,
        metrics=metrics,
        graph=graph,
        strict=strict,
        provenance=provenance,
        base_codes=base_codes,
    )


def _attempt_connectivity_topology(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
    minimum_macrocycle_ring_size: int,
) -> ReconstructionResult | None:
    """Ladder stage 5: DetermineConnectivity topology (no proximity)."""
    pdb_path = Path(prepared.pdb_path)
    stage = {
        "stage": SOURCE_CONNECTIVITY,
        "quality": QUALITY_TOPOLOGY,
        "ok": False,
    }
    try:
        # CONECT edges are hard input constraints.  Capture them before
        # DetermineConnectivity runs: the RDKit call is allowed to add
        # geometry-supported edges, but it must never delete or replace an
        # explicit input edge and then silently expose the resulting graph.
        explicit_edges = _parse_conect_pairs(pdb_path)
        mol = _connectivity_mol(pdb_path)
        explicit_edge_audit = _audit_explicit_edges(mol, explicit_edges)
        stage["explicit_edge_audit"] = explicit_edge_audit
        # Keep the normal no-molecule qualification path intact so failures
        # such as ``NO_ATOMS`` remain visible alongside the edge audit.
        if (
            mol is not None
            and mol.GetNumAtoms() > 0
            and not explicit_edge_audit["preserved"]
        ):
            stage["reason_codes"] = [
                _WARNING_EXPLICIT_EDGES_NOT_PRESERVED
            ]
            stage["qualification"] = {
                "passed": False,
                "explicit_edge_audit": explicit_edge_audit,
            }
            retained = _retain_readable_candidate(
                provenance,
                source=SOURCE_CONNECTIVITY,
                mol=mol,
                qualification_failures=stage["reason_codes"],
                qualification=stage["qualification"],
            )
            stage["candidate_graph_sha256"] = retained.get("graph_sha256")
            provenance["ladder_attempts"].append(stage)
            return None
        reason_codes, qualification = _topology_qualification(
            mol,
            prepared,
            pdb_path,
            minimum_macrocycle_ring_size,
        )
        if not explicit_edge_audit["preserved"]:
            if _WARNING_EXPLICIT_EDGES_NOT_PRESERVED not in reason_codes:
                reason_codes.append(_WARNING_EXPLICIT_EDGES_NOT_PRESERVED)
            qualification["explicit_edge_audit"] = explicit_edge_audit
        retained = _retain_readable_candidate(
            provenance,
            source=SOURCE_CONNECTIVITY,
            mol=mol,
            qualification_failures=reason_codes,
            qualification=qualification,
        )
        if reason_codes:
            stage["reason_codes"] = reason_codes
            stage["qualification"] = qualification
            stage["candidate_graph_sha256"] = retained.get("graph_sha256")
            provenance["ladder_attempts"].append(stage)
            return None
        metrics = _metrics_from_topology_qualification(qualification)
        graph = retained.get("graph") or _mol_to_graph(mol)
    except Exception as exc:
        stage["error"] = f"{type(exc).__name__}: {exc}"
        provenance["ladder_attempts"].append(stage)
        return None
    stage.update({"ok": True, **metrics, "qualification": {"passed": True}})
    provenance["ladder_attempts"].append(stage)
    provenance["ladder"] = "rdkit_determine_connectivity"
    return _topology_result(
        source=SOURCE_CONNECTIVITY,
        metrics=metrics,
        graph=graph,
        strict=strict,
        provenance=provenance,
        base_codes=base_codes,
    )


def _attempt_partial(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
) -> ReconstructionResult | None:
    """Ladder stage 6: explicit-only unsanitized RDKit mol from CONECT.

    Built from normalized-PDB atom records and explicit CONECT edges with
    UNSPECIFIED bond types.  This is a genuinely different reader from the
    proximity/connectivity stages: it never invents bonds by distance.  It
    requires at least one explicit bond; zero-bond inputs fall through to the
    raw stage (``NO_BONDS_AVAILABLE``).
    """
    pdb_path = Path(prepared.pdb_path)
    stage = {
        "stage": SOURCE_PARTIAL,
        "quality": QUALITY_PARTIAL,
        "ok": False,
    }
    try:
        mol, partial_details = _explicit_only_mol(pdb_path)
        if mol is None or mol.GetNumAtoms() == 0:
            stage["error"] = "no explicit bonds; no chemical graph to expose"
            stage["reason_codes"] = [_WARNING_NO_BONDS_AVAILABLE]
            provenance["ladder_attempts"].append(stage)
            return None
        if mol.GetNumBonds() == 0:
            stage["error"] = "no explicit bonds; no chemical graph to expose"
            stage["reason_codes"] = [_WARNING_NO_BONDS_AVAILABLE]
            provenance["ladder_attempts"].append(stage)
            return None
        conect_pairs = _parse_conect_pairs(pdb_path)
        metrics = _mol_metrics(mol, conect_pairs)
        graph = _mol_to_graph(mol)
    except Exception as exc:
        stage["error"] = f"{type(exc).__name__}: {exc}"
        provenance["ladder_attempts"].append(stage)
        return None
    stage.update({"ok": True, **metrics, "explicit_only": True})
    provenance["ladder_attempts"].append(stage)
    provenance["ladder"] = SOURCE_PARTIAL
    codes = list(base_codes) + [_WARNING_PARTIAL_EXPLICIT_ONLY]
    provenance["partial"] = {
        "atom_count": mol.GetNumAtoms(),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "bond_count": mol.GetNumBonds(),
        "explicit_conect_pair_count": partial_details[
            "explicit_conect_pair_count"
        ],
        "graph_bond_orders": "unknown",
        "smiles_derived": False,
    }
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_PARTIAL,
        source=SOURCE_PARTIAL,
        result=graph,
        smiles=None,
        graph=graph,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
    )


def _attempt_proximity_molecule(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
) -> ReconstructionResult | None:
    """Last molecule-level resort: RDKit proximity perception with bond orders.

    Uses MolFromPDBFile(proximityBonding=True, sanitize=True) which infers
    connectivity from interatomic distances and bond orders from valence.
    The result is labeled hypothesis quality with C1:H rigor.  This stage
    only fires when every validated path (strict, mapping-aware,
    assessment, registry, bond-order inference, F/H diagnostic) produced
    nothing, and sits above the topology/partial/raw observation-only
    ladder so that a molecule-level SMILES is always attempted before
    degrading to a graph artifact.

    Multi-component reads are never collapsed to the largest fragment:
    promotion is declined, every component is recorded in a complete
    deterministic ledger (``provenance["proximity_molecule"]``), and the
    warning ``PROXIMITY_COMPONENTS_DECLINED`` is carried by every weaker
    artifact so all components remain honestly observable downstream.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    pdb_path = getattr(prepared, "pdb_path", None)
    if pdb_path is None:
        provenance.setdefault("ladder_attempts", []).append({
            "stage": "proximity_molecule",
            "quality": QUALITY_HYPOTHESIS,
            "ok": False,
            "reason": "no prepared pdb path",
        })
        return None
    try:
        molecule = Chem.MolFromPDBFile(
            str(pdb_path),
            sanitize=True,
            proximityBonding=True,
            removeHs=True,
        )
    except Exception as exc:
        provenance.setdefault("ladder_attempts", []).append({
            "stage": "proximity_molecule",
            "quality": QUALITY_HYPOTHESIS,
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        })
        return None
    if molecule is None or molecule.GetNumAtoms() == 0:
        provenance.setdefault("ladder_attempts", []).append({
            "stage": "proximity_molecule",
            "quality": QUALITY_HYPOTHESIS,
            "ok": False,
            "reason": "no molecule from proximity perception",
        })
        return None
    smiles = Chem.MolToSmiles(molecule)
    if not smiles:
        provenance.setdefault("ladder_attempts", []).append({
            "stage": "proximity_molecule",
            "quality": QUALITY_HYPOTHESIS,
            "ok": False,
            "reason": "empty SMILES",
        })
        return None
    if len(Chem.GetMolFrags(molecule)) > 1:
        # No silent largest-fragment selection.  Every component is recorded
        # in a complete ledger and molecule promotion is declined; the
        # observation ladder below preserves all atoms/components honestly.
        components = _proximity_component_ledger(molecule)
        provenance["proximity_molecule"] = {
            "status": "declined_multiple_components",
            "component_count": len(components),
            "components": components,
            "policy": (
                "no silent largest-fragment selection; molecule promotion "
                "declined, the observation ladder preserves every component"
            ),
        }
        provenance.setdefault("ladder_attempts", []).append({
            "stage": "proximity_molecule",
            "quality": QUALITY_HYPOTHESIS,
            "ok": False,
            "reason_codes": [_WARNING_PROXIMITY_COMPONENTS_DECLINED],
            "component_count": len(components),
        })
        # Surface the decline on every weaker artifact produced afterwards.
        if _WARNING_PROXIMITY_COMPONENTS_DECLINED not in base_codes:
            base_codes.append(_WARNING_PROXIMITY_COMPONENTS_DECLINED)
        return None
    codes = list(base_codes)
    codes.append(_WARNING_PROXIMITY_MOLECULE_FALLBACK)
    provenance["ladder"] = "proximity_molecule"
    provenance["ladder_attempts"].append({
        "stage": "proximity_molecule",
        "quality": QUALITY_HYPOTHESIS,
        "ok": True,
        "atom_count": molecule.GetNumAtoms(),
        "component_count": 1,
    })
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_HYPOTHESIS,
        source="proximity_molecule",
        result=smiles,
        smiles=smiles,
        graph=None,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
        candidate_smiles=smiles,
        candidate_graph=None,
        candidate_rigor="C1:H",
    )


def _proximity_component_ledger(molecule: Any) -> list[dict[str, Any]]:
    """Complete, deterministic component ledger for a fragmented molecule."""
    from rdkit import Chem as _Chem

    components: list[dict[str, Any]] = []
    for fragment in _Chem.GetMolFrags(molecule, asMols=True):
        components.append(
            {
                "atom_count": int(fragment.GetNumAtoms()),
                "heavy_atom_count": int(fragment.GetNumHeavyAtoms()),
                "canonical_smiles": _Chem.MolToSmiles(fragment),
                "atom_serials": sorted(
                    _atom_serial(atom) for atom in fragment.GetAtoms()
                ),
            }
        )
    components.sort(
        key=lambda item: (
            -item["atom_count"],
            item["canonical_smiles"],
            item["atom_serials"],
        )
    )
    return components


def _attempt_raw(
    prepared: Any,
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    *,
    base_codes: list[str],
) -> ReconstructionResult | None:
    """Ladder stage 7: atom coordinates + CONECT graph with a damage ledger.

    Damage is advisory, never terminal: malformed CONECT records, dangling
    or self edges, malformed atom lines, and duplicate serials are dropped
    and recorded instead of failing the stage.  A graph with at least one
    atom succeeds even with zero bonds (warning ``NO_BONDS_AVAILABLE``).
    Only a total absence of atoms or a serialization failure makes this
    stage fail.
    """
    pdb_path = Path(prepared.pdb_path)
    stage = {
        "stage": SOURCE_RAW,
        "quality": QUALITY_RAW,
        "ok": False,
    }
    try:
        graph, damage_audit = _parse_raw_source_graph(
            pdb_path, getattr(prepared, "chain_id", None)
        )
        payload = _json_roundtrip(graph)
    except Exception as exc:
        stage["error"] = f"{type(exc).__name__}: {exc}"
        provenance["ladder_attempts"].append(stage)
        return None
    skipped_atom_records = damage_audit.get("skipped_atom_records", [])
    duplicate_atom_serials = damage_audit.get("duplicate_atom_serials", [])
    dropped_token_count = damage_audit.get("dropped_token_count", 0)
    malformed_record_count = len(damage_audit.get("malformed_records", []))
    stage.update(
        {
            "ok": True,
            "atom_count": len(graph["atoms"]),
            "bond_count": len(graph["bonds"]),
            "explicit_conect_pair_count": len(graph["bonds"]),
            "malformed_conect_record_count": malformed_record_count,
            "dropped_conect_token_count": dropped_token_count,
            "skipped_atom_record_count": len(skipped_atom_records),
            "duplicate_atom_serial_count": len(duplicate_atom_serials),
            "graph_json_sha256": hashlib.sha256(
                payload.encode("ascii")
            ).hexdigest(),
        }
    )
    provenance["ladder_attempts"].append(stage)
    provenance["ladder"] = "raw_pdb_graph"
    codes = list(base_codes)
    if not graph["bonds"]:
        codes.append(_WARNING_NO_BONDS_AVAILABLE)
    if malformed_record_count or dropped_token_count:
        codes.append(_WARNING_MALFORMED_CONECT_RAW)
    if skipped_atom_records or duplicate_atom_serials:
        codes.append(_WARNING_DAMAGED_RAW)
    provenance["raw"] = {
        "atom_count": len(graph["atoms"]),
        "bond_count": len(graph["bonds"]),
        "malformed_conect_records": damage_audit.get(
            "malformed_records", []
        ),
        "dropped_conect_token_count": dropped_token_count,
        "skipped_atom_records": skipped_atom_records,
        "duplicate_atom_serials": duplicate_atom_serials,
        "graph_json_sha256": stage["graph_json_sha256"],
    }
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_RAW,
        source=SOURCE_RAW,
        result=graph,
        smiles=None,
        graph=graph,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
    )


def _topology_result(
    *,
    source: str,
    metrics: dict[str, Any],
    graph: dict[str, Any],
    strict: StrictReconstructionResult | None,
    provenance: dict[str, Any],
    base_codes: list[str],
) -> ReconstructionResult:
    codes = list(base_codes) + [
        _WARNING_BOND_ORDERS_INFERRED,
        _WARNING_FORMAL_CHARGE_UNVERIFIED,
        _WARNING_STEREOCHEMISTRY_UNVERIFIED,
    ]
    provenance[source] = {
        "atom_count": metrics["atom_count"],
        "heavy_atom_count": metrics["heavy_atom_count"],
        "heavy_atom_component_count": metrics["heavy_atom_component_count"],
        "bond_count": metrics["bond_count"],
        "largest_detected_ring_size": metrics["largest_detected_ring_size"],
        "explicit_conect_pair_count": metrics["explicit_conect_pair_count"],
        "explicit_edges_preserved": metrics["explicit_edges_preserved"],
        "missing_explicit_edges": metrics["missing_explicit_edges"],
        "graph_bond_orders": "unknown",
    }
    return _build_result(
        status=STATUS_SUCCESS,
        quality=QUALITY_TOPOLOGY,
        source=source,
        result=graph,
        smiles=None,
        graph=graph,
        ambiguous=False,
        warning_codes=codes,
        alternatives=[],
        provenance=provenance,
        strict=strict,
    )


def _topology_qualification(
    mol: Any,
    prepared: Any,
    pdb_path: Path,
    minimum_macrocycle_ring_size: int,
) -> tuple[list[str], dict[str, Any]]:
    """Independent topology eligibility check (requirement 4).

    Returns ``(reason_codes, details)``; an empty ``reason_codes`` list means
    the stage qualifies.  Reason codes are deterministic.
    """
    reason_codes: list[str] = []
    details: dict[str, Any] = {}
    if mol is None or mol.GetNumAtoms() == 0:
        reason_codes.append(_REASON_NO_ATOMS)
        details["atom_count"] = 0 if mol is None else mol.GetNumAtoms()
        return reason_codes, details

    serials = [_atom_serial(atom) for atom in mol.GetAtoms()]
    details["atom_count"] = mol.GetNumAtoms()
    serial_counts = Counter(serials)
    duplicate_serials = sorted(
        serial for serial, count in serial_counts.items() if count > 1
    )
    details["duplicate_serials"] = duplicate_serials
    if duplicate_serials:
        reason_codes.append(_REASON_DUPLICATE_ATOM_SERIAL)

    atom_identity_mismatches = _atom_identity_mismatches(mol, pdb_path)
    details["atom_identity_mismatches"] = atom_identity_mismatches
    if atom_identity_mismatches:
        reason_codes.append(_REASON_ATOM_IDENTITY_NOT_PRESERVED)

    heavy_atom_count = mol.GetNumHeavyAtoms()
    details["heavy_atom_count"] = heavy_atom_count
    audit = getattr(prepared, "audit", None) or {}
    expected_heavy = audit.get("normalized_heavy_atom_count")
    if expected_heavy is not None:
        details["normalized_heavy_atom_count"] = int(expected_heavy)
        if heavy_atom_count != int(expected_heavy):
            reason_codes.append(_REASON_HEAVY_ATOM_COUNT_MISMATCH)

    component_count = _heavy_atom_components(mol)
    details["heavy_atom_component_count"] = component_count
    if component_count != 1:
        reason_codes.append(_REASON_MULTIPLE_HEAVY_ATOM_COMPONENTS)

    from .core.pdb_utils import read_first_model_lines
    from .paths import _map_utils as map_utils
    from .paths.path_b import _AA_3TO1

    residues: dict[tuple[str, str, str, str], set[str]] = {}
    for line in read_first_model_lines(str(pdb_path)):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        key = (
            line[21:22],
            line[22:26],
            line[26:27],
            line[17:20].strip().upper(),
        )
        residues.setdefault(key, set()).add(line[12:16].strip().upper())
    peptide_like = 0
    peptide_chains: set[str] = set()
    registry = map_utils.monomers2smi_dict
    for (chain, _number, _icode, residue_name), atom_names in residues.items():
        registered = (
            residue_name in _AA_3TO1
            or residue_name in registry
            or map_utils.resolve_pdb_alias(residue_name) in registry
        )
        has_backbone = {"N", "CA", "C"}.issubset(atom_names)
        if registered or has_backbone:
            peptide_like += 1
            peptide_chains.add(chain)
    details["peptide_like_residue_count"] = peptide_like
    details["peptide_like_chain_count"] = len(peptide_chains)
    if peptide_like < 2 or len(peptide_chains) != 1:
        reason_codes.append(_REASON_NOT_PEPTIDE_ENTITY)

    bond_count = mol.GetNumBonds()
    details["bond_count"] = bond_count
    if bond_count == 0:
        reason_codes.append(_WARNING_NO_BONDS_AVAILABLE)

    ring_size = _largest_detected_ring_size(mol)
    details["largest_detected_ring_size"] = ring_size
    details["minimum_macrocycle_ring_size"] = minimum_macrocycle_ring_size
    if ring_size < minimum_macrocycle_ring_size:
        reason_codes.append(_REASON_DETECTED_RING_SIZE_BELOW_THRESHOLD)

    conect_pairs = _parse_conect_pairs(pdb_path)
    serial_index = {serial: index for index, serial in enumerate(serials)}
    missing_edges: list[list[int]] = []
    for left, right in conect_pairs:
        left_index = serial_index.get(left)
        right_index = serial_index.get(right)
        if (
            left_index is None
            or right_index is None
            or mol.GetBondBetweenAtoms(left_index, right_index) is None
        ):
            missing_edges.append([left, right])
    details["explicit_conect_pair_count"] = len(conect_pairs)
    details["missing_explicit_edges"] = missing_edges
    if missing_edges:
        reason_codes.append(_WARNING_EXPLICIT_EDGES_NOT_PRESERVED)

    self_bonds: list[list[int]] = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() == bond.GetEndAtomIdx():
            self_bonds.append(
                [serials[bond.GetBeginAtomIdx()]] * 2
            )
    details["self_bonds"] = self_bonds
    if self_bonds:
        reason_codes.append(_REASON_SELF_BOND)

    maximum_degree = {
        1: 1,
        6: 4,
        7: 4,
        8: 2,
        9: 1,
        15: 6,
        16: 6,
        17: 1,
        35: 1,
        53: 1,
    }
    abnormal_degrees = []
    for atom in mol.GetAtoms():
        limit = maximum_degree.get(atom.GetAtomicNum())
        degree = atom.GetDegree()
        if limit is not None and degree > limit:
            abnormal_degrees.append({
                "serial": serials[atom.GetIdx()],
                "element": atom.GetSymbol(),
                "degree": degree,
                "maximum_degree": limit,
            })
    details["abnormal_atom_degrees"] = abnormal_degrees
    if abnormal_degrees:
        reason_codes.append(_REASON_ABNORMAL_ATOM_DEGREE)

    return reason_codes, details


def _atom_identity_mismatches(
    mol: Any,
    pdb_path: Path,
) -> list[dict[str, Any]]:
    """Compare an RDKit PDB read against the normalized atom records.

    Topology fallback is allowed to infer edges, but it must not silently
    change the atom identity carried by the normalized input.  The comparison
    is keyed by PDB serial and covers the element plus the residue/name
    identity used by the downstream graph serializer.
    """
    if mol is None:
        return []
    try:
        source_rows = _parse_pdb_atom_records(pdb_path)
    except (OSError, UnicodeError, ValueError):
        # The primary qualification path owns malformed/duplicate/empty
        # input diagnostics.  Do not replace those established reason codes
        # with an auxiliary identity-audit exception.
        return []
    expected = {int(row["serial"]): row for row in source_rows}
    observed: dict[int, dict[str, Any]] = {}
    duplicate_serials: set[int] = set()
    for atom in mol.GetAtoms():
        serial = _atom_serial(atom)
        info = atom.GetPDBResidueInfo()
        row = {
            "serial": serial,
            "name": info.GetName().strip() if info else "",
            "residue": info.GetResidueName().strip() if info else "",
            "residue_number": int(info.GetResidueNumber()) if info else 0,
            "chain": info.GetChainId().strip() if info else "",
            "element": atom.GetSymbol().strip().upper(),
        }
        if serial in observed:
            duplicate_serials.add(serial)
        observed[serial] = row

    mismatches: list[dict[str, Any]] = []
    for serial in sorted(set(expected) | set(observed)):
        expected_row = expected.get(serial)
        observed_row = observed.get(serial)
        if expected_row is None or observed_row is None:
            mismatches.append({
                "serial": serial,
                "expected": expected_row,
                "observed": observed_row,
                "reason": "serial_missing_or_extra",
            })
            continue
        expected_identity = {
            "name": str(expected_row["name"]).strip().upper(),
            "residue": str(expected_row["residue"]).strip().upper(),
            "residue_number": int(expected_row["residue_number"]),
            "chain": str(expected_row["chain"]).strip(),
            "element": str(expected_row["element"]).strip().upper(),
        }
        observed_identity = {
            "name": str(observed_row["name"]).strip().upper(),
            "residue": str(observed_row["residue"]).strip().upper(),
            "residue_number": int(observed_row["residue_number"]),
            "chain": str(observed_row["chain"]).strip(),
            "element": str(observed_row["element"]).strip().upper(),
        }
        if expected_identity != observed_identity:
            mismatches.append({
                "serial": serial,
                "expected": expected_identity,
                "observed": observed_identity,
                "reason": "identity_changed",
            })
    for serial in sorted(duplicate_serials):
        mismatches.append({
            "serial": serial,
            "expected": expected.get(serial),
            "observed": observed.get(serial),
            "reason": "duplicate_observed_serial",
        })
    return mismatches


# --------------------------------------------------------------------------
# RDKit helpers (module-level so tests can substitute them)
# --------------------------------------------------------------------------


def _proximity_mol(pdb_path: Path) -> Any:
    return Chem.MolFromPDBFile(
        str(pdb_path), proximityBonding=True, sanitize=False, removeHs=False
    )


def _connectivity_mol(pdb_path: Path) -> Any:
    mol = Chem.MolFromPDBFile(
        str(pdb_path), proximityBonding=False, sanitize=False, removeHs=False
    )
    if mol is None:
        return None
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def _explicit_only_mol(
    pdb_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Build an explicit-only unsanitized RWMol+Conformer from PDB records.

    Atoms come from ATOM/HETATM records (duplicate serials rejected) and
    bonds exclusively from CONECT edges with ``UNSPECIFIED`` bond type.
    """
    atoms = _parse_pdb_atom_records(pdb_path)
    conect_pairs = _parse_conect_pairs(pdb_path)
    present = {atom["serial"] for atom in atoms}
    edges: list[tuple[int, int]] = []
    for left, right in conect_pairs:
        if left == right:
            raise ValueError(f"self CONECT edge {left}-{left}")
        if left not in present or right not in present:
            raise ValueError(
                f"dangling CONECT endpoint in edge {left}-{right}"
            )
        edges.append((left, right))
    if not edges:
        return None, {
            "atom_count": len(atoms),
            "explicit_conect_pair_count": 0,
        }

    mol = Chem.RWMol()
    for atom in atoms:
        rd_atom = Chem.Atom(atom["element"])
        info = Chem.AtomPDBResidueInfo()
        info.SetSerialNumber(int(atom["serial"]))
        info.SetName(atom["name"])
        info.SetResidueName(atom["residue"])
        info.SetResidueNumber(int(atom["residue_number"]))
        info.SetChainId(atom["chain"])
        rd_atom.SetPDBResidueInfo(info)
        mol.AddAtom(rd_atom)
    serial_index = {atom["serial"]: index for index, atom in enumerate(atoms)}
    for left, right in edges:
        mol.AddBond(
            serial_index[left],
            serial_index[right],
            Chem.BondType.UNSPECIFIED,
        )
    molecule = mol.GetMol()
    conformer = Chem.Conformer(len(atoms))
    for index, atom in enumerate(atoms):
        conformer.SetAtomPosition(
            index, (atom["xyz"][0], atom["xyz"][1], atom["xyz"][2])
        )
    molecule.AddConformer(conformer, assignId=True)
    return molecule, {
        "atom_count": len(atoms),
        "explicit_conect_pair_count": len(edges),
    }


def _atom_serial(atom: Any) -> int:
    info = atom.GetPDBResidueInfo()
    if info is not None:
        try:
            return int(info.GetSerialNumber())
        except (TypeError, ValueError):
            pass
    for prop in ("pdbSerial", "serialNumber"):
        if atom.HasProp(prop):
            try:
                return int(atom.GetProp(prop))
            except (TypeError, ValueError):
                pass
    return atom.GetIdx() + 1


def _mol_to_graph(mol: Any) -> dict[str, Any]:
    """Structured graph with stable ordering and canonical undirected edges.

    Atoms are sorted by serial; every undirected bond is canonicalized to
    ``(min(serial), max(serial))``; bond orders are always null (unknown) for
    ladder stages below exact/high/medium.  Repeated serialization is
    deterministic.
    """
    conformer = mol.GetConformer()
    atom_rows: list[dict[str, Any]] = []
    serial_by_index: list[int] = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        serial = _atom_serial(atom)
        info = atom.GetPDBResidueInfo()
        position = conformer.GetAtomPosition(index)
        atom_rows.append(
            {
                "serial": serial,
                "name": info.GetName().strip() if info else "",
                "residue": info.GetResidueName().strip() if info else "",
                "residue_number": int(info.GetResidueNumber()) if info else 0,
                "chain": info.GetChainId().strip() if info else "",
                "element": atom.GetSymbol(),
                "xyz": [
                    float(position.x),
                    float(position.y),
                    float(position.z),
                ],
            }
        )
        serial_by_index.append(serial)
    atoms = sorted(atom_rows, key=lambda row: row["serial"])
    edge_pairs: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        left = serial_by_index[bond.GetBeginAtomIdx()]
        right = serial_by_index[bond.GetEndAtomIdx()]
        edge_pairs.add((min(left, right), max(left, right)))
    bonds = [
        {"a": left, "b": right, "order": None}
        for left, right in sorted(edge_pairs)
    ]
    return {"atoms": atoms, "bonds": bonds}


def _metrics_from_topology_qualification(
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Project metrics already computed by a successful qualification."""
    missing = list(qualification["missing_explicit_edges"])
    return {
        "atom_count": qualification["atom_count"],
        "heavy_atom_count": qualification["heavy_atom_count"],
        "heavy_atom_component_count": qualification[
            "heavy_atom_component_count"
        ],
        "bond_count": qualification["bond_count"],
        "largest_detected_ring_size": qualification[
            "largest_detected_ring_size"
        ],
        "explicit_conect_pair_count": qualification[
            "explicit_conect_pair_count"
        ],
        "explicit_edges_preserved": not missing,
        "missing_explicit_edges": missing,
        "sanitize": "none",
    }


def _mol_metrics(mol: Any, conect_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    serials = [_atom_serial(atom) for atom in mol.GetAtoms()]
    serial_index = {serial: index for index, serial in enumerate(serials)}
    missing_explicit: list[list[int]] = []
    for left, right in conect_pairs:
        left_index = serial_index.get(left)
        right_index = serial_index.get(right)
        if (
            left_index is None
            or right_index is None
            or mol.GetBondBetweenAtoms(left_index, right_index) is None
        ):
            missing_explicit.append([left, right])
    return {
        "atom_count": mol.GetNumAtoms(),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "heavy_atom_component_count": _heavy_atom_components(mol),
        "bond_count": mol.GetNumBonds(),
        "largest_detected_ring_size": _largest_detected_ring_size(mol),
        "explicit_conect_pair_count": len(conect_pairs),
        "explicit_edges_preserved": not missing_explicit,
        "missing_explicit_edges": missing_explicit,
        "sanitize": "none",
    }


def _mol_edge_pairs(mol: Any) -> set[tuple[int, int]]:
    """Return the molecule's undirected edges in PDB-serial space."""
    if mol is None:
        return set()
    serials = [_atom_serial(atom) for atom in mol.GetAtoms()]
    pairs: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        left = serials[bond.GetBeginAtomIdx()]
        right = serials[bond.GetEndAtomIdx()]
        pairs.add(tuple(sorted((left, right))))
    return pairs


def _audit_explicit_edges(
    mol: Any,
    explicit_edges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Check hard CONECT edges after an RDKit connectivity operation.

    Additional geometry-inferred edges are intentionally permitted.  The
    audit only rejects a result when one of the input explicit edges is
    absent from the returned graph; this covers both deletion and endpoint
    replacement without imposing a geometry-derived edge set.
    """
    required = {tuple(sorted(edge)) for edge in explicit_edges}
    observed = _mol_edge_pairs(mol)
    missing = sorted(required - observed)
    additional = sorted(observed - required)
    return {
        "required_edges": [list(edge) for edge in sorted(required)],
        "observed_edges": [list(edge) for edge in sorted(observed)],
        "missing_edges": [list(edge) for edge in missing],
        "additional_edges": [list(edge) for edge in additional],
        "additional_edges_allowed": True,
        "preserved": not missing,
    }


def _heavy_atom_components(mol: Any) -> int:
    heavy = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    parent = {index: index for index in heavy}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in parent and right in parent:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left
    return len({find(index) for index in heavy}) if heavy else 0


def _largest_detected_ring_size(mol: Any) -> int:
    """Deterministic detected ring size.

    For every bond, compute the length of the shortest cycle through that
    bond (BFS), then return the maximum over all bonds.  This is a
    deterministic detection metric, not a "largest simple ring" claim.
    """
    adjacency: dict[int, list[int]] = {
        atom.GetIdx(): [] for atom in mol.GetAtoms()
    }
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        adjacency[left].append(right)
        adjacency[right].append(left)
        edges.append((left, right))
    largest = 0
    for left, right in edges:
        size = _shortest_cycle_through_edge(left, right, adjacency)
        if size and size > largest:
            largest = size
    return largest


def _shortest_cycle_through_edge(
    left: int,
    right: int,
    adjacency: dict[int, list[int]],
) -> int:
    queue: deque[tuple[int, int]] = deque([(left, 0)])
    visited = {left}
    while queue:
        node, depth = queue.popleft()
        for neighbor in adjacency[node]:
            if node == left and neighbor == right:
                continue
            if neighbor == right:
                return depth + 2
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return 0


# --------------------------------------------------------------------------
# PDB-native parsing helpers (raw and explicit-only partial stages)
# --------------------------------------------------------------------------


def _read_pdb_lines(pdb_path: Path) -> list[str]:
    try:
        return (
            Path(pdb_path)
            .read_text(encoding="ascii", errors="strict")
            .splitlines()
        )
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"PDB cannot be read: {type(exc).__name__}: {exc}"
        ) from exc


def _parse_int_field(raw: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_float_field(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _element_from_name(name: str) -> str:
    letters = "".join(character for character in name if character.isalpha())
    if not letters:
        return ""
    symbol = letters[:2].upper()
    if (
        len(letters) == 2
        and symbol in _TWO_LETTER_ELEMENTS
        and symbol not in _PROTEIN_ATOM_NAME_ELEMENT_COLLISIONS
    ):
        return symbol
    return letters[0].upper()


def _parse_conect_pairs(pdb_path: Path) -> list[tuple[int, int]]:
    """Strictly parse CONECT records; malformed records raise."""
    pairs: set[tuple[int, int]] = set()
    for line in _read_pdb_lines(pdb_path):
        if not line.startswith("CONECT"):
            continue
        serials: list[int] = []
        for index in range(6, len(line.rstrip("\r\n")), 5):
            token = line[index : index + 5].strip()
            if not token:
                continue
            parsed = _parse_int_field(token)
            if parsed is None:
                raise ValueError(
                    f"malformed CONECT record: {line.rstrip()!r}"
                )
            serials.append(parsed)
        if not serials:
            raise ValueError(
                f"malformed CONECT record without serials: {line.rstrip()!r}"
            )
        head, *partners = serials
        for partner in partners:
            pairs.add(tuple(sorted((head, partner))))
    return sorted(pairs)


def _parse_pdb_atom_records(pdb_path: Path) -> list[dict[str, Any]]:
    """Strictly parse ATOM/HETATM records; nothing is silently dropped."""
    atoms: list[dict[str, Any]] = []
    seen_serials: set[int] = set()
    for line in _read_pdb_lines(pdb_path):
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        serial = _parse_int_field(line[6:11])
        name = line[12:16].strip()
        residue = line[17:20].strip()
        chain = line[21:22].strip()
        residue_number = _parse_int_field(line[22:26]) or 0
        x = _parse_float_field(line[30:38])
        y = _parse_float_field(line[38:46])
        z = _parse_float_field(line[46:54])
        element = line[76:78].strip() or _element_from_name(name)
        if serial is None or x is None or y is None or z is None:
            raise ValueError(
                f"malformed ATOM/HETATM record: {line.rstrip()!r}"
            )
        if not element:
            raise ValueError(
                f"unparseable atom element in record: {line.rstrip()!r}"
            )
        if serial in seen_serials:
            raise ValueError(f"duplicate atom serial {serial}")
        seen_serials.add(serial)
        atoms.append(
            {
                "serial": serial,
                "name": name,
                "residue": residue,
                "residue_number": residue_number,
                "chain": chain,
                "element": element,
                "xyz": [x, y, z],
            }
        )
    if not atoms:
        raise ValueError("no ATOM/HETATM records found")
    return atoms


def _parse_raw_pdb_graph(pdb_path: Path) -> dict[str, Any]:
    atoms = _parse_pdb_atom_records(pdb_path)
    present = {atom["serial"] for atom in atoms}
    bonds = []
    for left, right in _parse_conect_pairs(pdb_path):
        if left == right:
            raise ValueError(f"self CONECT edge {left}-{left}")
        if left not in present or right not in present:
            raise ValueError(
                f"dangling CONECT endpoint in edge {left}-{right}"
            )
        bonds.append({"a": left, "b": right, "order": None})
    bonds.sort(key=lambda bond: (bond["a"], bond["b"]))
    return {"atoms": atoms, "bonds": bonds}


def _parse_raw_source_graph(
    pdb_path: Path,
    chain_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse a readable PDB while treating malformed records as advisory."""
    atom_damage: dict[str, Any] = {}
    atoms = _parse_pdb_atom_records_for_chain(
        pdb_path, chain_id, damage=atom_damage
    )
    present = {atom["serial"] for atom in atoms}
    bonds: set[tuple[int, int]] = set()
    malformed_records: list[str] = []
    dropped_token_count = 0
    for line in _read_pdb_lines(pdb_path):
        if not line.startswith("CONECT"):
            continue
        serials: list[int] = []
        malformed = False
        for index in range(6, len(line), 5):
            token = line[index : index + 5].strip()
            if not token:
                continue
            parsed = _parse_int_field(token)
            if parsed is None:
                malformed = True
                dropped_token_count += 1
                continue
            serials.append(parsed)
        if malformed or len(serials) < 2:
            malformed_records.append(line.rstrip())
        if not serials:
            continue
        head, *partners = serials
        if head not in present:
            if partners:
                dropped_token_count += len(partners)
            continue
        for partner in partners:
            if partner not in present or partner == head:
                dropped_token_count += 1
                continue
            bonds.add(tuple(sorted((head, partner))))
    graph = {
        "atoms": sorted(atoms, key=lambda row: row["serial"]),
        "bonds": [
            {"a": left, "b": right, "order": None}
            for left, right in sorted(bonds)
        ],
    }
    return graph, {
        "malformed_records": malformed_records,
        "dropped_token_count": dropped_token_count,
        "skipped_atom_records": atom_damage.get("skipped_atom_records", []),
        "duplicate_atom_serials": atom_damage.get(
            "duplicate_atom_serials", []
        ),
    }


def _parse_pdb_atom_records_for_chain(
    pdb_path: Path,
    chain_id: str | None,
    damage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Strictly parse atoms from one chain without depending on CONECT.

    With a ``damage`` ledger the parser degrades instead of raising: a
    malformed record or a duplicate serial skips the line and is recorded,
    so readable coordinate evidence survives as a raw artifact with an
    explicit conflict ledger rather than collapsing to opaque input.
    """
    atoms: list[dict[str, Any]] = []
    seen_serials: set[int] = set()
    selected_chain = None if chain_id is None else str(chain_id)
    for line in _read_pdb_lines(pdb_path):
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        serial = _parse_int_field(line[6:11])
        name = line[12:16].strip()
        residue = line[17:20].strip()
        chain = line[21:22].strip()
        if selected_chain is not None and chain != selected_chain:
            continue
        residue_number = _parse_int_field(line[22:26]) or 0
        x = _parse_float_field(line[30:38])
        y = _parse_float_field(line[38:46])
        z = _parse_float_field(line[46:54])
        element = line[76:78].strip() or _element_from_name(name)
        if serial is None or x is None or y is None or z is None:
            if damage is None:
                raise ValueError(f"malformed ATOM/HETATM record: {line!r}")
            damage.setdefault("skipped_atom_records", []).append(
                line.rstrip()
            )
            continue
        if not element:
            if damage is None:
                raise ValueError(
                    f"unparseable atom element in record: {line!r}"
                )
            damage.setdefault("skipped_atom_records", []).append(
                line.rstrip()
            )
            continue
        if serial in seen_serials:
            if damage is None:
                raise ValueError(f"duplicate atom serial {serial}")
            damage.setdefault("duplicate_atom_serials", []).append(serial)
            continue
        seen_serials.add(serial)
        atoms.append(
            {
                "serial": serial,
                "name": name,
                "residue": residue,
                "residue_number": residue_number,
                "chain": chain,
                "element": element,
                "xyz": [x, y, z],
            }
        )
    if not atoms:
        raise ValueError("no selected ATOM/HETATM records found")
    return atoms


def _json_roundtrip(graph: dict[str, Any]) -> str:
    payload = json.dumps(
        graph, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    restored = json.loads(payload)
    if restored != graph:
        raise ValueError("graph JSON roundtrip mismatch")
    return payload


__all__ = [
    "QUALITIES",
    "QUALITY_EXACT",
    "QUALITY_HIGH",
    "QUALITY_MEDIUM",
    "QUALITY_TOPOLOGY",
    "QUALITY_PARTIAL",
    "QUALITY_RAW",
    "SOURCES",
    "SOURCE_EXACT",
    "SOURCE_UNIQUE_CANDIDATE",
    "SOURCE_ENSEMBLE_CANDIDATE",
    "SOURCE_REGISTRY",
    "SOURCE_DEGRADED_TEMPLATE",
    "SOURCE_PROXIMITY",
    "SOURCE_CONNECTIVITY",
    "SOURCE_PARTIAL",
    "SOURCE_RAW",
    "SOURCE_FAILED",
    "STATUS_FAILED",
    "STATUS_SUCCESS",
    "ReconstructionResult",
    "reconstruct_prepared_structure",
    "reconstruct_structure",
]
