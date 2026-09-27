# RDpepper V6.0.0 — Prior-Respecting Conformer Materialization

- Status: **IMPLEMENTED 2026-08-26.** Production code landed; see
  "Implementation record" below. Benchmark successor freeze and manuscript
  rebuild are tracked as separate phases per AGENTS.md.
- Scope owner: primary agent (this record defines the change; implementation is a
  separate, explicitly scheduled phase).
- Version semantics: **major bump (6.0.0)** because validated MOL2 conformers are
  content-addressed (`core/artifacts.make_artifact_id`); every sequence-arm MOL2
  byte stream and every downstream artifact ID changes, so V5.1.0 evidence chains
  cannot absorb this incrementally.

## Implementation record (2026-08-26)

- V6-1/V6-2/V6-3: `export/conformer_ensemble.py` — `_prior_constraints`
  replaces `_apply_torsion_priors` (all Lipinski-rotatable bonds, no 8-bond
  cap, applicability = matched ∧ confidence ∈ {high, medium} ∧
  `circular_std_deg` ≤ `PRIOR_APPLY_MAX_STD_DEG` = 30°); `_constrained_relaxation`
  projects table means then relaxes under native RDKit torsion constraints
  (`MMFFAddTorsionConstraint` / `UFFAddTorsionConstraint`, force constant 100,
  window ±10°, `Initialize()` before minimize) with a deterministic
  project-and-relax fallback (≤3 iterations); template attempts apply priors to
  bonds without template coordinates; per-bond target/final/delta audit with
  `prior_unsatisfied` marking (no hard rejects).
- V6-4: `docking/torsion_prior.py::classify_macrocycle_topology` extracted as the
  single classification core; `docking/torsion_observations.py::_topology` now
  delegates to it (pure extraction, build-side behavior unchanged);
  `export/conformer_ensemble.py::_derived_topology` derives the same vocabulary
  plus `macrocycle_ring_size` from the exact_v1 document, and both the runtime
  prior query keys and the MOL2 validation receipt now carry the derived values
  (the PDBQT layer inherits the unified keys through the receipt).
- V6-5: `docking/torsion_prior.py::prior_guidance_quartet` — heavy-atom quartet
  selection mirroring the observation-side conventions (phi/psi/omega anchor on
  neighbouring-residue N/CA/C with cyclic backbone wrap-around, chi1 anchors on
  the backbone N, closure-like bonds take the index-first heavy neighbour).
- V6-6: `qa.prior_guidance` audit extended with `constraint`
  {mechanism, iterations, tolerance_deg, bonds[], prior_unsatisfied};
  attempt statuses, strategy names, and `coordinate_mode` vocabulary are
  unchanged.
- Version: pyproject.toml / `__init__.py` / `build_rdpepper_release.py` bumped
  to 6.0.0 (QA receipt filename constant updated; receipt generation itself
  remains a release-phase step).
- Tests: `tests/test_v6_prior_materialization.py` (9 tests) plus fixture update
  in `tests/test_v5_conformer_materializer.py`; seed determinism verified via
  identical ensemble manifest SHA-256 across runs.  Full suite green after
  fixing three stale assertions (two version literals, one pre-existing
  fixture/routing mismatch): confirmation pass 1324 passed / 24 skipped /
  0 failed (GUI file covered by the initial full run).

## Post-freeze note (2026-08-27, review-driven)

The Candidate026 review round found one audit-labeling defect:
`_constrained_relaxation` recorded `mechanism="rdkit_torsion_constraint"`
even when the constraint list was empty.  Fixed to record
`mechanism="no_constraints"` while preserving the single minimization pass
(behavior for empty constraint lists is otherwise unchanged; verified by a
dedicated probe and the V6 test file).  This is an audit-label-only change
with zero effect on any Candidate026 scientific number; the
Candidate026 source capsule predates the fix and stands as frozen history,
and the fix ships with the 6.0.0 release build.  The same review established
the actual trigger statistics for the reported cohort (template attempts 0,
prior-guided constraints applied in 1 of 34 cases / 1 bond total,
0 prior-unsatisfied bonds), which the Candidate027 manuscript wording must
disclose.

## Motivation

V5.1.0 materializes MOL2 conformers from a sequence-only graph via three
independent strategies (`export/conformer_ensemble.py:633-651`):

1. `template_1..N` — ETKDGv3 embedding constrained by a template coord map,
   MMFF relaxation with template atoms as fixed points.
2. `torsion_prior_guided` — one unconstrained ETKDG embedding, then a
   post-hoc lookup pass, then an **unconstrained** MMFF relaxation.
3. `diverse_etkdg_1..M` — seeded ETKDG variants, unconstrained relaxation.

The prior-guided pass does not respect the prior table it reads: dihedral means
are written with `SetDihedralDeg` and then free to drift during minimization,
the template branch never consults the prior table at all, and the table keys
used here diverge from the PDBQT layer's keys for the same bond. The change
recorded here makes prior guidance a real, auditable constraint inside the
materialization layer.

## Current-behavior gaps (V5.1.0 anchors)

- G1. Priors are applied only *after* embedding and only in the standalone
  `torsion_prior_guided` attempt (`export/conformer_ensemble.py:263`
  `_apply_torsion_priors`, invoked at `:680-689`); template attempts never mix
  in prior knowledge for bonds the template does not cover.
- G2. Applied means are unconstrained during optimization
  (`_optimize` at `export/conformer_ensemble.py:327` supports fixed points
  only), so the final dihedral is not guaranteed to match the table mean; the
  table value only biases which basin MMFF settles into.
- G3. At most 8 bonds are treated (`if len(applied) >= 8: break`), which
  truncates macrocycle-rich graphs.
- G4. `macrocycle_ring_size=None` is passed here
  (`export/conformer_ensemble.py:684`) while the PDBQT layer queries the same
  runtime table with the parent receipt's real `topology_class` and
  `macrocycle_ring_size` (`docking/mol2_pdbqt.py:519-541`); lookup coverage and
  keys differ between the two layers.
- G5. Quartet neighbor selection sorts by atom index only; the PDBQT layer's
  equivalent sort deprioritizes hydrogens (`docking/torsion_budget.py:140-161`),
  so the same bond can be written onto a hydrogen quartet here.

## Planned changes (V6.0.0)

- V6-1 (addresses G1). Hybrid template+prior attempts: when a template coord
  map is active, bonds not covered by template constraints also receive prior
  lookup and constrained guidance in the same attempt.
- V6-2 (addresses G2). Prior-respecting relaxation: applied dihedrals are held
  by torsion constraints (or an equivalent staged-freeze scheme) during
  minimization; each audit records requested mean, post-minimization dihedral,
  and their delta. A bond whose delta exceeds tolerance is reported as
  `prior_unsatisfied`, never silently accepted.
- V6-3 (addresses G3). Remove the 8-bond cap; apply every matched bond. Keep a
  summary cap only for audit readability, not for behavior.
- V6-4 (addresses G4). Materialization queries the runtime prior with the same
  key builder and the same inputs (`topology_class`, real
  `macrocycle_ring_size` from the chemical graph / receipt) as the PDBQT layer,
  so one table serves both layers identically.
- V6-5 (addresses G5). Quartet selection deprioritizes hydrogens, matching
  `docking/torsion_budget.py`.
- V6-6. Provenance: prior `runtime_sha256` / `manifest_sha256` and per-bond
  lookup levels stay in the attempt audit; the coordinate-origin taxonomy
  gains an explicit decision (template-covered vs prior-guided vs regenerated
  bonds are distinguishable in the ensemble member audit).

Non-goals: no conformer generation moves into the PDBQT layer; no
`exact_v1` / representation changes; no monomer-library changes; no docking
or accuracy claims are added by the code change itself.

## Acceptance criteria

- Full production test suite passes with 0 failures (V5.1.0 baseline: 1274
  passed, 24 skipped).
- Seed-deterministic ensembles; QA gates unchanged (InChIKey match, closure,
  strain <= 250 kcal/mol, RMSD >= 0.15 A dedup, clash checks).
- With the packaged prior unavailable, behavior degrades to the documented
  V5 diverse/template paths and the audit says so explicitly.
- A prior-respect metric is reported for every constrained bond (see V6-2);
  silent drift is impossible by construction.

## Evidence and freeze impact (blocking, per AGENTS.md)

Production drift and publication evidence are separate phases. Implementing
V6 invalidates the sequence-arm conformer outputs and their hashes, therefore:

- BIRD 132-record sequence arm: MOL2/PDBQT downstream must be rerun under a
  successor freeze; `analysis_005` / `final_002` successors replace the frozen
  candidate024 artifacts for any *new* manuscript numbers.
- candidate025 remains the frozen 5.1.0 evidence generation; a successor
  candidate (not in-place edits) carries V6 numbers.
- The still-pending six supplementary-data placements
  (`supplementary_data/inherited_v501/*`, `supplementary_data/dynamic_sequence/*`)
  merge into the successor freeze rather than being finished against 5.1.0,
  unless a 5.1.0-only submission package is decided first.
- Unaffected at graph level: exact_v1 round-trip counts, BIRD coordinate-arm
  U2/U3 results, HighDB operational coverage, monomer-library statistics.
  Manuscript version-consistency statements are still required wherever
  historical V5 cohorts and V6 sequence-arm numbers coexist.
