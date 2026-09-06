# CycPep Master V5 artifact contract

V5 uses one typed, content-bound object graph:

```text
InputArtifact
  -> ChemicalGraphArtifact
  -> ConformerEnsembleArtifact
       -> ValidatedMol2Artifact[]

ValidatedMol2Artifact + ConformerEnsembleArtifact + torsion-prior hashes
  -> FlexibilityAssessmentArtifact

ValidatedMol2Artifact [+ FlexibilityAssessmentArtifact]
  -> PdbqtArtifact
```

Sequence, MAP, HELM, BILN, SMILES, PDB and mmCIF are input routes, not
independent docking pipelines. The new sequence facade is
`application.prepare_ligand_from_sequence`. Historical PDB and SMILES
facades remain compatible, but all ligand PDBQT serialization converges on
`prepare_ligand_pdbqt_from_mol2`.

## Stage ownership

- Chemical graph stage owns monomer resolution, ports, bonds,
  stereochemistry, terminal state and selected microstate identity.
  Every public consumer uses the same request-scoped monomer-resolution
  context. Request-local definitions and CCD/PRD components are projected
  into the existing registry only for the operation and are never persisted.
- Within the V5 typed-artifact path, MOL2 materialization owns every generated
  coordinate operation:
  template constraints, ETKDG, torsion-prior guidance, closure checks,
  minimization, clash/strain screening and RMSD diversity.
- Template resources return coordinate evidence only.
- The torsion-prior resource supplies local angular/flexibility evidence
  only.
- The PDBQT stage consumes validated MOL2 and optional existing ensemble
  coordinates. It never calls ETKDG or changes the parent heavy-atom graph,
  formal charge or coordinates.

Legacy compatibility exporters remain available and may use their historical
RDKit embedding implementations. They are outside the V5 typed-artifact claim
and must not be used to infer that every repository-level embedding call has a
single implementation owner.

Graph-only or contradictory inputs remain `NOT_MATERIALIZABLE`; no MOL2 or
PDBQT is fabricated. An individual conformer or PDBQT failure does not
invalidate successful sibling MOL2 artifacts. A failed torsion budget retains
the Meeko baseline PDBQT.

An unavailable monomer is not a terminal operation rejection. The chemical
stage returns a symbolic candidate plus `monomer_resolution` evidence and a
lower chemical level. Exact/strict V5/V6 evidence may still record ABSTAIN,
`rejected` or `not_supported` internally; those nested states delimit claims
and are not promoted. Malformed syntax, invalid arguments and irreversible
write errors remain ordinary input/operational failures.

Monomer evidence mapping:

- explicit request definition: `C3:S`;
- source-bound CCD/PRD graph with charge, stereo and required ports:
  `C3:Q`;
- source conflict or auditable inference: `C2:H`;
- unresolved identity with symbolic monomer/topology artifact: `C1:H`.

## Evidence axes

Each artifact carries five independent fields:

- `chemical_rigor`: `C0`–`C3` plus basis
  `S/Q/R/H/C/NONE`.
- `coordinate_origin` and `coordinate_level`: `X0`–`X3`.
- `format_level`: `Q0`–`Q3`.
- `flexibility_level`: `F0`–`F3`.
- `budget_state`: `not_requested`, `satisfied`, `unsatisfied` or
  `not_assessable`.

Downstream stages preserve chemical evidence exactly. Only the conformer
materializer may add coordinate evidence, only validators/writers may add
format qualification, and only flexibility assessment may add `F` evidence
or a budget state.

`F0` means no torsion budget was requested. `F1` is a mechanistic
Meeko/prior-only assessment. `F2` requires calibrated lookup evidence or
validated sibling-conformer evidence. `F3` is reserved for thorough mode with
a complete validated ensemble.

## PDBQT typed states

- `PDBQT_QUALIFIED`: valid PDBQT; no torsion budget was requested.
- `PDBQT_BASELINE`: valid Meeko baseline retained without a satisfied
  requested budget, or the companion baseline of a budgeted output.
- `PDBQT_BUDGETED`: requested TORSDOF limit is satisfied.

File validity and torsion-budget success are deliberately separate.

## Identity and integrity

Every artifact has:

- a deterministic `artifact_id` bound to artifact type, semantic payload and
  parent IDs;
- a `payload_sha256`;
- unique parent artifact IDs;
- warnings, provenance and an explicit claim boundary.

Filesystem locator fields are excluded from logical artifact IDs. Therefore
the same input, resources, seed and thread count produce the same logical IDs
in different output directories. MOL2 and receipt file hashes remain
mandatory and are verified when an ensemble is loaded. An optional expected
ensemble-manifest SHA-256 rejects manifest replacement.

The machine-readable schema is `schemas/v5_artifact.schema.json`.

## Claim boundary

V5 artifacts support claims about chemical identity evidence, coordinate
provenance, MOL2/PDBQT format qualification and mechanistic torsion-budget
assessment. They do not, without separate held-out experiments, establish
experimental structure accuracy, biological flexibility, docking-pose
improvement, binding affinity or selectivity.
