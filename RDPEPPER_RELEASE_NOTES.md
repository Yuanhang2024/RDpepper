# RDpepper Release Notes

## 7.1.0

Version 7.1.0 completes the maximum-acceptance product line while preserving
the strict evidence contract as the highest tier.

### Scientific and product contract

- Chemical rigor (`C3:Q` ... `C0:NONE`) and coordinate evidence
  (`X3` ... `X0`) are independent fields. Integrity warnings can lower
  chemical rigor without erasing a readable artifact; readable but
  under-evidenced input is downgraded and labeled rather than rejected as a
  product failure.
- An ambiguous candidate-assessment ensemble keeps the candidate set without
  an automatically selected primary identity. F/H diagnostic
  identities are never promoted to qualified chemistry; agreement yields at
  most a `C1:H` hypothesis and the family quorum / F-H disagreement veto is
  retained.
- Bond-order inference may select one candidate for MOL2 materialization
  while ambiguity and alternatives remain in reconstruction evidence; compact
  operation summaries need not contain the whole nested evidence bundle. A parseable, unambiguous SMILES can be materialized as X1 MOL2
  with mandatory readback; request binding, graph/SMILES divergence,
  embedding and force-field outcomes, actual MOL2 atom identifiers, component
  integrity, and fallback/ETKDG/force-field warnings all enter the auditable
  receipt.
- `C` tiers are evidence statements, not probabilities; `X` tiers record
  coordinate origin, not accuracy. Multi-component proximity input is
  preserved as a complete ledger rather than silently reduced to its largest
  fragment.
- The recommended best-available export route (used for the current paper
  runs) is
  `export_best_available(source, dest, source_kind='pdb', output_format='mol2', chain_id='L', fallback_policy='max_coverage')`.
  Strict V6 reconstruction (`--path v6`) remains available for qualified-evidence
  audits; a fallback-policy value alone does not establish strict qualification.

### QA and evaluation summary

- Release QA for 7.1.0: byte-compilation of the integrated source, a
  56-assertion result-ladder micro-suite (56/56), a 44-assertion
  MOL2/binding micro-suite (44/44), and a clean-environment wheel install
  with `pip check`. The full pytest gate was **not** run for this release,
  and no full-suite pass is claimed.
- Subsequent evaluation runs completed on the fixed 7.1.0 source: the BIRD
  corpus (806 inputs), HighDB (2,504 inputs), a fixed 480-member comparison
  set, CREMP (480), and same-graph writer/reader consistency checks. BIRD and HighDB
  informed prior development; CREMP is computed-coordinate stress evidence,
  and the writer/reader experiment uses known graphs. These results do not
  constitute unseen experimental holdout validation. Detailed protocols and
  results belong to the separate manuscript evidence package.

### Distribution, provenance, and license

- This local candidate is a documentation- and privacy-cleaned distribution of
  the frozen internal 7.1.0 source. Production Python files and runtime data
  are unchanged. The original source identity is recorded in
  `ORIGINAL_SOURCE.json` (source tree SHA-256
  `44ee88ab3a9ceda4ad0b8c384ab76d418e7c10fa59e1bceb61027aea308bba9a`); the
  public tree is not a byte-identical copy of the complete internal source
  tree.
- Distributed through the public GitHub repository and its Releases page.
  No PyPI publication is claimed for this release.
- Licensing: source code under MIT; the NNAA collection under CC BY-NC 4.0;
  the CycPeptMPDB subset under CC BY 4.0; the HELM-GPT monomer subset under
  the HELM-GPT project MIT license; structural templates/priors under
  CC BY 4.0 (`THIRD_PARTY_DATA.md`, `NOTICE`).

## 7.0.0 (historical)

Established the degradation-only, status-closed product contract: every
normally accessible request returns the richest typed artifact it can support,
with strict `rejected` / `not_supported` / `failed` states retained as nested
evidence rather than erasing the product artifact. Chemical qualification,
coordinate evidence, requested-format fulfillment, and flexibility evidence
became orthogonal axes; mapping-derived identity disagreement returns an
ambiguous candidate bundle with no automatically selected SMILES; PDBQT
preparation consumes a receipt-validated MOL2 parent and degrades honestly
when unavailable. Engineering work reused one shared Path-G producer,
identity calculations, and prepared-input scopes across a request.

## 6.1.0 (historical)

Added the optional max-coverage fallback layer (`max_coverage` module;
`fallback_policy` parameter on `export.conformer.pdb_to_mol2`,
`application.export_structure`, and the CLI export command): records with
incomplete source-coordinate mapping are emitted at an explicit X2/X1 tier —
mapped atoms keep source coordinates, small gaps are completed from local
bond geometry — while identity gates (full-InChIKey agreement, round-trip
verification) are enforced unchanged. This was an optional recovery mode; strict defaults remained available.

## 6.0.0 (historical)

Made calibrated torsion priors real constraints inside the MOL2 conformer
materialization layer: template-constrained embedding applies priors to
rotatable bonds the template does not cover; prior means are enforced via
RDKit native torsion constraints (MMFF/UFF) with a deterministic
project-and-relax fallback; every constrained bond reports
target/final/delta, and out-of-tolerance bonds are recorded as
`prior_unsatisfied`. Materialization and PDBQT preparation share one prior
vocabulary and key derivation, with `topology_class` and
`macrocycle_ring_size` written into the validated-MOL2 receipt. Ensembles
retain recorded random seeds.

## 5.1.0 (historical)

Public renaming from CycPep Master to RDpepper: distribution name `rdpepper`,
`import rdpepper` facade, `rdpepper` CLI, and `rdpepper-gui`, while the
`cycpep_master` implementation namespace, `cycpep` / `cycpep-gui` commands,
historical schemas, artifact identifiers, and frozen manifests remain valid.
Added request-scoped monomer resolution (caller-supplied definitions or
CCD/PRD components extending the active view without mutating the packaged
library), conservative preservation of source-bound CCD double-bond
stereochemistry when the component uniquely supports E/Z, and typed
symbolic/lower-rigor artifacts for unmaterializable sequence inputs. See
`MIGRATION_RDPEPPER.md`.

Releases before 5.1.0 were internal (CycPep Master) versions and are not part
of this public distribution.
