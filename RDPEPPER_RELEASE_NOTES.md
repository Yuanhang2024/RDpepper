# RDpepper 7.1.0

Version 7.1.0 completes the maximum-acceptance repair line while preserving the
strict A–H scientific contract as the highest `C3:Q` evidence tier.

## Scientific and product contract

- Recovery quality, chemical rigor (`C3:Q` through `C0:NONE`), and coordinate
  evidence (`X3` through `X0`) are independent fields. Diagnostic or integrity
  warnings can lower chemical rigor without erasing a readable artifact.
- F/H diagnostic identities are never promoted to qualified chemistry. Agreement
  can produce only a `C1:H` hypothesis; disagreement retains every identity as an
  ambiguous candidate set with no automatically selected primary.
- Result-first and degraded paths are molecule-first without duplicate bond-order
  inference. Multi-component proximity input is preserved as a complete ledger
  rather than silently reduced to its largest fragment.
- Parseable, unambiguous SMILES can be materialized as X1 MOL2. Request binding,
  graph/SMILES divergence, embedding and force-field outcomes, actual one-based
  MOL2 atom identifiers, component integrity, and mandatory readback are recorded
  in artifact evidence and validation receipts.

## Evidence boundary

- The integrated repair line passed byte-compilation plus the focused result-ladder
  (56/56) and MOL2/binding (44/44) micro-smoke suites. The full pytest gate was
  not run for 7.1.0, so this release must not inherit the 7.0.0 full-suite claim.
- Formal BIRD and HighDB evaluation has not been rerun for 7.1.0. Candidate031
  remains development evidence; the successor candidate032 harness is frozen
  separately and no full-cohort run is part of this release step.
- This is a local desktop Python distribution release. Public GitHub/PyPI upload
  is not authorized.


# RDpepper 7.0.0

Version 7.0.0 establishes a degradation-only, status-closed product contract
and consolidates the current optimized reconstruction and materialization
implementation as the next local release baseline.

## Scientific and product contract

- Every normally accessible request returns the richest typed artifact it can
  support. Strict V4/V5/V6 `rejected`, `not_supported`, and `failed` states
  remain nested strict evidence and no longer erase the product artifact.
- Chemical qualification, coordinate evidence, requested-format fulfillment,
  and flexibility evidence are orthogonal. Lower-rigor topology, partial,
  raw-coordinate, or opaque artifacts never masquerade as qualified molecules.
- Mapping-derived identity disagreement returns an ambiguous candidate bundle
  with no automatically selected SMILES. F/H candidates remain diagnostics and
  retain their strict identity-veto role.
- PDBQT preparation consumes a receipt-validated MOL2 parent and preserves the
  parent C/X/Q evidence. An unavailable PDBQT request degrades to an honest
  alternative artifact rather than laundering a topology or raw parent.

## Engineering and performance

- One request now reuses the shared Path G producer, B/G forward artifacts,
  molecular identity calculations, immutable first-model PDB text, strict PDB
  audit, bound result-first reconstruction, and one ValidatedMol2 load.
- Coordinate normalization, result-first reconstruction, and source-bound MOL2
  materialization share one prepared-input scope.
- Successful topology fallback reuses qualification metrics rather than
  repeating component, ring, and explicit-edge analysis. MOL2 coordinate-ledger
  parsing and MMFF property construction avoid redundant reads and setup.
- Dead private docking/parser branches, duplicate canonical-JSON helpers, and
  no-op evidence parameters were removed without shortening the scientific
  route portfolio or fresh A/E replay.

## Evidence boundary

- This release does not claim new benchmark accuracy, coverage, or runtime
  statistics. Formal BIRD/HighDB/MOL2 benchmark execution has not been rerun
  against the 7.0.0 source baseline.
- Existing frozen results and historical QA receipts retain their original
  source hashes and software versions. Production drift requires a successor
  scientific freeze before new results may support manuscript claims.
- Public upload remains unauthorized while redistribution terms for the derived
  NNAA tranche are unresolved. The 7.0.0 desktop bundle is a local release
  baseline, not a GitHub or PyPI publication authorization.


# RDpepper 6.1.0

Version 6.1.0 adds the optional max-coverage fallback layer. Full design and
implementation record: `V7_MAX_COVERAGE_FALLBACK_DESIGN.md`.

## Scientific and engineering changes

- New module `max_coverage` provides pure fallback primitives: input
  diagnosis (class A), notation salvage with placeholders (class B),
  coordinate tiering X3/X2/X1 (class D), and salvage payloads (class E);
  covered by `tests/test_max_coverage.py`.
- `export.conformer.pdb_to_mol2` accepts `fallback_policy` (`strict_v6`
  default, `max_coverage` opt-in): under max_coverage, records with
  incomplete source-coordinate mapping are emitted anyway - mapped atoms
  keep source coordinates, small gaps are filled by local bond-geometry
  placement (ETKDG embedding only as fallback) - and the MOL2 header
  records the coordinate tier and mapped/generated atom counts. Identity
  gates (full-InChIKey agreement, roundtrip verification) are enforced
  unchanged; default behavior is byte-identical to 6.0.0.
- `application.export_structure` and the CLI `export --fallback-policy`
  option expose the policy publicly.
- Evaluated on the fixed 1,571-record MOL2 cohort (candidate028 ablation):
  338/343 not_supported records convert to X2 (coverage 77.8% -> 99.4%).

## Compatibility

- Default behavior unchanged (53/53 export + fallback tests; full suite
  green). Fallback outputs are additive and always carry tier receipts.


# RDpepper 6.0.0

Version 6.0.0 makes torsion-prior guidance a real constraint inside the MOL2
conformer materialization layer.  Full design and implementation record:
`V6_TORSION_PRIOR_MATERIALIZATION_PLAN.md`.

## Scientific and engineering changes

- Template-constrained embedding attempts now also apply calibrated torsion
  priors to rotatable bonds the template does not cover; the standalone
  `torsion_prior_guided` strategy applies priors to every matched bond (the
  legacy 8-bond cap is removed).
- Prior table means are enforced during relaxation: RDKit native torsion
  constraints (`MMFFAddTorsionConstraint` / `UFFAddTorsionConstraint`) are used
  when available with a deterministic project-and-relax fallback; every
  constrained bond reports target/final/delta and bonds beyond tolerance are
  recorded as `prior_unsatisfied` instead of being silently accepted.
- Materialization and PDBQT preparation now query the runtime prior table with
  one vocabulary and one key derivation: the shared
  `classify_macrocycle_topology` core (also used by the table builder) derives
  `topology_class` and `macrocycle_ring_size` from the exact_v1 graph, and both
  values are written into the validated-MOL2 receipt.
- Prior-guided quartet selection mirrors the observation-side conventions and
  never selects hydrogens.
- PDBQT-layer behavior, the typed artifact contract, `exact_v1`, and the
  monomer library are unchanged; ensembles remain seed-deterministic.

# RDpepper 5.1.0

RDpepper is the public successor name for CycPep Master.  Version 5.1.0 adds a
new distribution, import facade, command-line identity, and GUI branding while
preserving the existing `cycpep_master` implementation namespace.

## Public naming

- PyPI distribution: `rdpepper`
- Python facade: `import rdpepper`
- CLI: `rdpepper`
- GUI: `rdpepper-gui`
- Intended source repository: `ChimeraModel/RDpepper`

The legacy `cycpep`, `cycpep-gui`, `cycpep-overlay`, and
`cycpep-v5-reproduce` entry points remain installed.

## Scientific and engineering changes

- Request-scoped monomer resolution can extend the active monomer view from
  caller-provided definitions or independently supplied CCD/PRD components
  without mutating the packaged library.
- Source-bound CCD double-bond stereochemistry is preserved conservatively
  when the component definition uniquely supports an E/Z assignment.
- Sequence inputs that cannot be materialized remain typed symbolic or
  lower-rigor artifacts instead of being silently promoted to qualified
  molecules.
- The typed artifact chain, validated-MOL2 ownership of conformer generation,
  and MOL2-only PDBQT entry point remain unchanged.

## Compatibility

Existing source layouts and `cycpep_master.*` imports remain valid.  Machine
schemas, frozen manifests, and artifact identifiers retain their historical
names.  See `MIGRATION_RDPEPPER.md` for the supported migration surface.

## Evidence boundary

The public rename is not a scientific experiment.  Existing HighDB, BIRD,
MOL2, PDBQT, and exact-v1 results retain the software version and source
capsule recorded by their original receipts.  New dynamic sequence evidence
supports operational output, molecular-graph coverage, conditional identity
agreement, and downstream format completion; it does not establish
experimental conformer accuracy, biological flexibility, docking benefit,
affinity, or selectivity.

The final release bundle must include a successor RDpepper 5.1.0 packaging QA
receipt.  Historical V5 receipts are included only as explicitly labeled
historical evidence and are not rewritten as 5.1.0 results.

## Third-party NNAA data

The 9,998-row compiled NNAA tranche is attributed to the diverse 10,000 amino
acid library reported by Amarasinghe et al., *J. Chem. Inf. Model.* 2022,
DOI 10.1021/acs.jcim.2c00193. The local article identifies a TXT Associated
Content file but does not state an open redistribution license. GitHub/PyPI
publication of those derived rows therefore remains on hold until the
publisher/data terms or explicit permission are confirmed.
