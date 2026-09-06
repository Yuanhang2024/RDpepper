# CycPep Master exact_v1

## Purpose

`exact_v1` is CycPep Master's lossless chemistry-facing intermediate
representation for cyclic peptides. It records monomer identities, directed
peptide-chain order, explicit R1/R2/R3 connections, terminal caps, stereo
identity, and source-to-canonical mapping.

It is not a model token sequence. `edge_v1` and `legacy_v5` are bounded model
projections and cannot establish exact chemical equivalence.

The sibling `chimera_encoder_decoder.graph_exact_v1` research atom-graph
sidecar is a different contract. It is not imported, modified, or used as the
runtime implementation of this package-local monomer-port IR.

## Stable Monomer Identity

The current numeric `monomer_id` is snapshot-scoped. exact_v1 therefore binds
it to the identity-bearing monomer row:

```text
unified:<numeric-id>:<identity-content-sha256-prefix>
```

Library-only caps use a content-addressed library ID. A monomer definition is
accepted only when its active CXSMILES contains an explicit anchor for every
used port. Automatically inferred R3 anchors do not qualify.

Aliases are case-sensitive and accepted only when their ported molecular
graphs are identical. Nearest-neighbor `best_cycpep_match_symbol` metadata is
not treated as an alias.

## Canonicalization

- Linear chains preserve N-to-C direction and are never reversed.
- A head-to-tail chain may be cyclically rotated. The canonicalizer moves the
  normative `HT` break so adjacent amide edges remain `PEPTIDE` and the wrap
  edge remains `HT`.
- Multi-chain inputs are ordered by a port-aware canonical graph comparison.
- Caps, crosslinks, monomer stereo, modifications, and chain breaks
  participate in the canonical graph bytes.
- Canonicalization is fail-closed when the state space exceeds the fixed
  safety limit.

`graph_sha256` is calculated from canonical chemistry bytes. Source trace
fields (`original_*`, `canonical_order`, and bidirectional position maps) are
excluded from that hash because equivalent rotated inputs legitimately have
different source positions.

`projection_trace` is excluded as well. It stores legacy model slot order only
for explicit compatibility projection and is checked against the exact graph
before use; it cannot change exact equivalence.

Use `canonical_exact_v1_bytes()` when byte equality of the canonical chemistry
record is required.

## Exactness And ABSTAIN

Every document has:

```text
exactness_status = EXACT | ABSTAIN
reason_codes = [...]
normalization_codes = [...]
```

Unknown monomers, unresolved stereo, missing ports, reused ports,
contradictory bond semantics, malformed connections, and hash drift produce
`ABSTAIN` or rejection. No all-single-bond or sequence-only guess is emitted.

The historical cap-inclusive MAP convention is migrated only when all of the
following are true:

1. ordinary strict MAP parsing fails for an out-of-range endpoint;
2. an N-terminal cap is present;
3. every explicit endpoint lies in `2..N+1`;
4. subtracting one from every endpoint yields a strict valid graph.

The resulting exact document records
`LEGACY_CAP_INCLUSIVE_POSITION_NORMALIZED`.

## Public API

```python
from cycpep_master.exact_v1 import (
    map_to_exact_v1,
    helm_to_exact_v1,
    biln_to_exact_v1,
    exact_v1_to_map,
    exact_v1_to_helm,
    exact_v1_to_biln,
    exact_v1_to_smiles,
    exact_v1_to_edge_v1,
    exact_v1_equivalent,
    chemical_graph_equivalent,
    model_projection_equivalent,
)
```

The three equivalence APIs are deliberately separate:

- `exact_v1_equivalent`: canonical monomer-port graphs are identical.
- `chemical_graph_equivalent`: assembled atom graphs, stereo, isotope state,
  and formal charge are identical.
- `model_projection_equivalent`: bounded `edge_v1` projections match.

The third relation does not prove either of the first two.

## V6 Integration

`application.reconstruct_exact_v1()` is additive. It does not alter
`StrictReconstructionResult` or historical reconstruction output.

Emission requires:

- `status=success`;
- `support_status=qualified`;
- `qualified_success=true`;
- no repair or warning codes;
- passed library chemistry, atom mapping, chemical graph, stereochemistry,
  and mapping-binding dimensions;
- route-level monomer evidence agreement;
- exact_v1-to-SMILES full InChIKey equality with V6.

Otherwise the operation returns a typed ABSTAIN result.

## edge_v1 Boundary

The default model envelope is:

- maximum positions: 32;
- maximum closure bonds: 3;
- supported tokens: HT, SS, SC, EST, HSC, THIO, and ALK;
- no caps or multiple chains.

Overflow and unsupported semantics return `UNPROJECTABLE`. Connections are
never truncated.

Canonical projection is the default. `preserve_source_order=True` exists only
for byte-compatible legacy migration, including multi-ring slot order.

## Frozen History

The implementation does not modify remediation_v5, remediation_v6 result
shapes, historical V5 benchmarks, frozen raw outputs, or the unified monomer
library.

Any experimental or publication claim using exact_v1 requires a successor
source freeze.
