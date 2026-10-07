# exact_v1 migration matrix

## Source To exact_v1

| Source | Route | Exact gate | Information retained |
|---|---|---|---|
| MAP | strict MAP parser | all monomers and used ports registry-bound | chains, caps, ports, crosslinks, stereo-by-monomer |
| HELM | direct strict HELM parser | PEPTIDE polymers and R1-R3 links only | multi-chain polymer graph and canonical connection semantics |
| BILN | direct strict BILN parser | bracket and bond-ID grammar valid | multi-chain graph; source bond IDs are normalized |
| PDB/mmCIF | qualified V6 postprocessor | unrepaired qualified result plus passed identity dimensions and full identity replay | V6 monomer, closure, stereo, and source evidence |
| legacy_v5 | package-local compatibility parser | every operation resolves to explicit ports and stable monomers | supported legacy topology; no invented fourth ring |
| edge_v1 | package-local bounded parser | operation triples and monomer tokens valid | model projection semantics only |
| general SMILES | unsupported | unique audited monomer decomposition unavailable | ABSTAIN |

## exact_v1 To Target

| Target | Behavior | Fail-closed boundary |
|---|---|---|
| MAP | canonical chain/port serialization | unsupported cap or chain combination rejects |
| HELM | exact_v1 to MAP to strict canonical HELM | unsupported HELM semantics reject |
| BILN | exact_v1 to canonical HELM/BILN | unsupported bond annotation rejects |
| SMILES | active audited monomer assembly | invalid assembly or identity parse rejects |
| edge_v1 | bounded model projection | caps, multi-chain, >3 closures, >32 positions, unsupported bond type reject |
| legacy_v5 | projection through the edge envelope | same capacity limits as edge_v1 |

## Compatibility Modes

- Canonical exact_v1 and canonical edge_v1 are the default.
- `preserve_source_order=True` reproduces frozen legacy_v5-to-edge_v1 bytes
  when a head-to-tail ring has an arbitrary source starting residue.
- model projection equality is never reported as exact equality.
- The existing `chimera_encoder_decoder` package remains optional and is used
  only by compatibility tests, not by the installed runtime.

## Packaged Real-Structure Audit

The generated report is:

- `benchmarks/exact_v1_compatibility_report.json`
- `benchmarks/exact_v1_compatibility_report.md`

Input: packaged 794-entry template index, SHA-256
`179aae8a4280270c44f200242f82a26f0a12d6edbe7b70fcd096017d98ba4eb4`.

Results:

- exact_v1: 793/794;
- canonical MAP/HELM/BILN roundtrip: 793/793;
- NNAA exact: 44/45;
- multiring exact: 9/9;
- edge_v1 projected: 326/793;
- historical cap-inclusive positions normalized: 467;
- source-SMILES connectivity identity: 784/793;
- full source microstate identity: 171/793.

All nine connectivity mismatches are Scaffold mixed-topology records. They
remain visible in the row-level report. The low full-microstate match rate is
primarily a formal-charge/protonation difference between source template
SMILES and the active monomer registry; connectivity and complete microstate
are reported separately.

The sole ABSTAIN is synthetic entry `15_none_syn_111`; one monomer has an
unresolved stereocenter in the active registry. It remains unrepresented
rather than being silently treated as achiral.

This audit measures compatibility of the packaged index, not population
accuracy or experimental correctness.
