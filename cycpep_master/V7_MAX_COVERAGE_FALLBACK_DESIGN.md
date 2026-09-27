# Max-Coverage Fallback Design: eliminating hard rejections (V7 proposal)

Status: **IMPLEMENTED IN RDpepper 6.1.0** (module `cycpep_master/max_coverage.py`;
`pdb_to_mol2`/`export_structure`/CLI `--fallback-policy` wiring; cohort ablation
`paper_pipeline_vnext/candidate028_fallback_mol2/`, 338/343 -> X2, coverage
77.8% -> 99.4%). Default behavior is unchanged: every fallback below is gated
behind an explicit `fallback_policy="max_coverage"` and the package default
remains `"strict_v6"`, so the V6.0.0 frozen behavior and all manuscript
numbers are untouched until a successor freeze opts in.

## Goal

Two invariants replace every hard rejection:

1. **Maximal coverage**: an input structure always yields the richest artifact
   it can support (record -> graph -> MOL2 -> PDBQT), instead of stopping.
2. **Per-structure rigor output**: every produced artifact carries an explicit
   evidence tier on the existing axes (C/X/Q/F); degradation is expressed as a
   lower tier with an itemized audit, never as silence.

## Rejection inventory -> fallback mapping

| Class (current behavior) | Volume | Fallback in `max_coverage` mode | Rigor output |
|---|---|---|---|
| A. Entry `invalid_input` on data content (unparseable SMILES/notation payload) | op-level | `diagnose_input` returns error kind + position + salvageable fragment instead of bare rejection | operation result `status="degraded_input"` with diagnosis |
| B. Notation/monomer hard validation (`_map_utils` ValueError family; one bad token rejects the whole input) | 40/132 sequence partials end here | `salvage_notation_prefix` replaces the failing token with a placeholder monomer and continues parsing; salvaged prefix retained | C1:H partial graph + `placeholders` + error positions |
| C. `strict` mode reject | 781/2504 HighDB | **unchanged by design** - strict is an explicit user contract; result-first remains the max-coverage route | (contract) |
| D. MOL2 `not_supported` (source-coordinate mapping incomplete) | 343/1571 | `apply_coordinate_tier` emits the MOL2 anyway: mapped atoms keep source coordinates, unmapped atoms keep embedded coordinates | X3 (all mapped, 0 A) / X2 (mixed; generated atom list in receipt) / X1 (no mapping) |
| D'. PDBQT without parent MOL2 | 25/225 | follows from D: a parent MOL2 now always exists (possibly X1/X2); PDBQT inherits the parent X tier | inherits X tier; mode audit unchanged |
| E. Internal fail-closed states (`unrecoverable`, `template_partial_unavailable`, `audit_failed`) | small | attach `salvage_payload` (deepest parsed prefix / partial graph) to the state | state + payload, tier unchanged |
| F. Library/manifest integrity; malformed parameters | - | **unchanged** (installation corruption and caller bugs must fail loudly) | - |

## X-axis tier semantics (new, explicit)

- **X3** every heavy atom carries a source-mapped coordinate; displacement 0 A.
- **X2** >= 1 heavy atom uses an embedded (generated) coordinate; receipt lists
  `generated_atom_indices`, counts, and per-atom origin.
- **X1** no source mapping available; all coordinates are embedded; the graph
  (C-tier) is still source-bound.
- **X0** reserved: no trustworthy coordinate basis at all (not produced by the
  MOL2 fallback; X1 is the floor, so every structure has an output).

## Integration points (where the policy plugs in)

- `application.py` operations accept `fallback_policy` (default `strict_v6`);
  in `max_coverage` they route data-content failures through `diagnose_input`.
- Sequence adapter: notation parse errors route through
  `salvage_notation_prefix` before falling to the C1:H shell.
- MOL2 materializer: after embedding, `apply_coordinate_tier(mol, mapping)`
  decides X3/X2/X1 and writes `coordinate_origin` into the receipt; the MOL2
  writer runs regardless of mapping completeness.
- PDBQT stage: consumes any parent MOL2 and copies the parent X tier into its
  audit.

## What deliberately does NOT change

- C-axis gating: a molecule is never *invented* to raise coverage; placeholders
  stay C1:H.
- Truth isolation, provenance receipts, and the torsion-prior mechanism.
- The frozen V6.0.0 numbers: switching an experiment to `max_coverage` is a
  successor-freeze decision, reported as new cohorts (e.g., coverage-ablation
  arm), never silently merged with existing ones.
