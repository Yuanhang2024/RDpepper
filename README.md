# RDpepper 7.1.0

RDpepper is an evidence-graded cheminformatics engine for cyclic peptides. It
reconstructs an auditable chemical graph from protein-bound PDB/mmCIF
coordinate records (receptor + peptide ligand) or from sequence/HELM/MAP/BILN
notation, and provides HELM/MAP/BILN/SMILES conversion, non-natural amino acid
and multi-chain handling, validated MOL2 materialization, PDBQT preparation,
ADMET prediction, and optional AutoDock Vina integration.

RDpepper is built on RDKit and adds residue-aware reconstruction and explicit evidence reporting:
**converting a protein-bound cyclic-peptide PDB structure into a cyclic-peptide
chemical graph with explicit evidence grades.** It parses PDB chains and
SSBOND/LINK/CONECT records, assembles residues from templates and the monomer
library, detects cyclization, and hands the result to RDKit for molecule
construction and perception.

## About this distribution

This is the public GitHub distribution of RDpepper 7.1.0. Bundled
third-party data keep their source licenses: the NNAA collection is
CC BY-NC 4.0, the CycPeptMPDB subset is CC BY 4.0, the HELM-GPT monomer
subset follows the HELM-GPT project MIT license, and structural
templates/priors are CC BY 4.0. See `NOTICE` and `THIRD_PARTY_DATA.md`.

The candidate is a documentation- and privacy-cleaned copy of the frozen
source. Production Python code and runtime data are unchanged. The original
frozen-source identity is recorded in `ORIGINAL_SOURCE.json` (source tree
SHA-256
`44ee88ab3a9ceda4ad0b8c384ab76d418e7c10fa59e1bceb61027aea308bba9a`);
the public tree is not a byte-identical copy of the complete internal source
tree. Version history: `RDPEPPER_RELEASE_NOTES.md`.

The public distribution name, repository name, and new code entry points are
`RDpepper` / `rdpepper`. For compatibility, the implementation package
`cycpep_master`, the `cycpep` / `cycpep-gui` commands, and historical schema,
artifact, and provenance identifiers remain valid and are not rewritten by the
renaming (see `MIGRATION_RDPEPPER.md`).

## Requirements

- **Python** >= 3.10 (declared package requirement). Release QA for 7.1.0 was
  performed on CPython 3.14 under Windows; other versions and platforms have
  not been systematically tested for this release.
- **Core dependencies** (installed automatically): RDKit 2026.3.3, Gemmi
  >= 0.7.0, pandas >= 2.0, numpy >= 1.24.

## Installation

Install from this repository or from the wheel published on the
Releases page. This release makes no claim about PyPI availability.

```bash
# Option 1: wheel downloaded from the GitHub Releases page
python -m pip install ./rdpepper-7.1.0-py3-none-any.whl

# Option 2: from a checkout of this repository
python -m pip install .
```

Optional features install as extras on the same command:

```bash
python -m pip install "./rdpepper-7.1.0-py3-none-any.whl[inference,docking]"
python -m pip install ".[gui,admet]"
```

| Extra | Provides |
|---|---|
| `inference` | Multi-engine bond-order inference (Open Babel) and BIRD comparison |
| `docking` | PDBQT preparation (Meeko) and Vina integration helpers |
| `gui` | PyQt5 desktop workspace (`rdpepper-gui`) |
| `admet` | ADMET prediction (admet-ai); the step is skipped when absent |
| `dev` | pytest and schema validation for the test suite |

The AutoDock Vina executable is not bundled; provide it via the `VINA_BIN`
environment variable or system `PATH` when using the docking integration.

## Quick start (command line)

```bash
# Reconstruct one PDB/mmCIF file (peptide chain L by default)
rdpepper --pdb example.pdb --chain L

# Strict, evidence-qualified fail-closed V6 route
rdpepper --pdb example.pdb --path v6

# Explicit diagnostic candidate route
rdpepper --pdb example.pdb --path g

# Batch a directory and write a CSV summary
rdpepper --dir ./input --csv results.csv

# Export 3D MOL2 while reconstructing
rdpepper --dir ./input --export-dir ./output --export-format mol2

# Notation conversion
rdpepper convert --from map --to smiles '{nnr:7T2}AC'

# Sequence -> validated MOL2 ensemble -> PDBQT readiness budget
rdpepper prepare-sequence ACDEFG ./prepared \
  --cyclization head-to-tail --conformers 4 --flexibility-mode balanced

# Desktop workspace (requires the [gui] extra)
rdpepper-gui
```

Run `rdpepper <command> --help` for the full service surface (`reconstruct`,
`convert`, `audit`, `compare`, `export`, `pdbqt`, `dock`, `monomer`, and
more). All service commands return unified JSON results. A source checkout can
also run `python run.py ...` and `python -m rdpepper ...`; these and the
legacy `cycpep` / `cycpep-gui` commands use the same implementation.

## Python API

```python
import rdpepper

print(rdpepper.__version__)          # "7.1.0"

# New code uses the facade; deep implementation modules stay cycpep_master.*
from rdpepper import application, reconstruct_structure

# Unified reconstruction: PDB/mmCIF paths or sequence/HELM/MAP/BILN text
result = reconstruct_structure("input/example.pdb", chain_id="L", mode="auto")
```

### Best-available MOL2 export (route used for the current paper runs)

```python
from rdpepper import application

result = application.export_best_available(
    "input/example.pdb",           # source structure
    "output/example.mol2",         # destination
    source_kind="pdb",
    output_format="mol2",
    chain_id="L",
    fallback_policy="max_coverage",
)
data = result["data"]
data["requested_format_status"]    # fulfilled | degraded_format | metadata_only
data["artifacts"]                  # inspect artifact-specific evidence and warnings
```

Under `fallback_policy="max_coverage"`, MOL2 export is not hard-rejected when
the source-coordinate mapping is incomplete: mapped atoms keep their source
coordinates, small gaps are completed from local bond geometry, and the
artifact records an explicit coordinate tier (see below). Identity checks remain in force. The fallback-policy parameter controls
materialization recovery; it is not itself a guarantee of strict-qualified
chemistry in the result-first API. Use the explicit V6 reconstruction route
(`--path v6`) when requesting strict evidence qualification.

Top-level `status=success` means an inspectable artifact was returned, not
that qualified chemistry was established. Always also read
`requested_format_status`, `artifact_status`, `chemical_rigor`, and
`qualified_success` before using a result.

### Request-scoped monomer resolution

Monomer-resolution-aware entry points accept a `monomer_context` mapping
(CLI: `--monomer-context-json`) that can extend the active monomer view for
one operation only — inline definitions, CCD/PRD component snapshots, or a
local CCD directory:

```json
{
  "definitions": [{"symbol": "MyAA", "smiles": "N[C@@H](CCl)C(=O)O"}],
  "ccd_directory": "./ccd",
  "allow_network": false
}
```

Supplied definitions and CCD/PRD components are recorded in the resolution
ledger. A component record alone does not establish whole-molecule
qualification. Unresolved or conflicting monomers retain lower-tier evidence
and warnings.
`allow_network` defaults to `false`; enabling it fetches only the specific
unresolved component IDs from RCSB CCD. Extensions never mutate the packaged
or user monomer library.

## Evidence model

The artifact contract records separate evidence axes where applicable:

- **`C0–C3`** — chemical-graph evidence, with suffixes `S/Q/R/H/C/NONE`
  (specified, qualified, repaired, hypothesis, coordinate-only, none).
- **`X0–X3`** — coordinate evidence: X3 source-bound, X2 mixed/completed,
  X1 regenerated, X0 unavailable; detailed coordinate modes are recorded separately.
- **`Q0–Q3`** — format qualification (validated MOL2; tree/atom-invariant
  checked PDBQT).
- **`F0–F3`** — flexibility evidence, with budget status recorded separately.

Ambiguity semantics:

- An ambiguous candidate-assessment ensemble retains its candidate set without
  an automatically selected primary identity.
- Bond-order inference may select a *materialization* candidate while retaining
  ambiguity and alternatives in reconstruction evidence. Compact operation
  summaries need not embed the entire nested evidence bundle.
- `C` tiers are statements about how the chemical graph was established
  (evidence), **not** a probability that it is correct.
- `X` tiers record where the coordinates came from (origin), **not** their
  accuracy.

Known scope boundaries, stated once: lower-tier outputs (`topology`,
`partial`, `raw`) are structured edge lists with `bond_order=null` — they are
never disguised as all-single-bond SMILES; stereochemistry in
inference-materialized candidates is not verified against the full
InChIKey-qualified V6 contract; the torsion-prior and template libraries are
flexibility/conformation evidence only and do not establish docking, pose, or
affinity claims.

## Reconstruction routes

V6 aggregates multi-source evidence and emits a structure only when
qualification conditions hold. A–H are diagnostic candidate routes (A/C/E,
B/G, and F/H share major ancestors; they are not seven independent votes).

| Route | Method | Best for |
|---|---|---|
| **V6** (`--path v6`) | Multi-source evidence agreement + qualification audit | Auditable, evidence-qualified recovery |
| **A** | Per-residue atom-template assembly + CONECT crosslinks | Standard-amino-acid cyclic peptides |
| **B** | HELM -> MAP -> SMILES via the monomer library | NNAA-containing peptides |
| **C** | Path A + HETATM cap merging (ACE/NME) | Explicit HETATM caps |
| **E** | Path A + geometric covalent-radius cyclization | Structures without CONECT (e.g., relaxed models) |
| **F** | Geometric bond graph + heuristic bond orders + formula check | Untemplated special residues |
| **G** | Special-residue library, symbol-level assembly | Stapled / lanthionine / depsipeptide chemistry |
| **H** | CONECT-driven connectivity + heuristic bond orders | Complex peptides with full linkage records |

Supported cyclization: head-to-tail, disulfide, side-chain-to-backbone
isopeptide, ester/lactone (depsipeptide), thioether/lanthionine, hydrocarbon
staples, and inter-chain crosslinks (e.g., insulin-style multi-chain peptides
via `generate_multichain`). Detection prefers SSBOND/LINK/CONECT records and
falls back to element + covalent-radius geometry.

The 20 standard amino acids plus ACE/NME caps are built in; everything else
resolves through the monomer library.

## Monomer and template libraries

`unified_monomer_library.csv` — 13,152 monomers: CycPeptMPDB (384, with
source annotations and descriptors), a compiled 9,998-row NNAA collection derived from
the diverse 10,000 amino-acid library of Amarasinghe et al.,
*J. Chem. Inf. Model.* 2022, 62, 2999–3007
(https://doi.org/10.1021/acs.jcim.2c00193), and HELM-GPT (2,770).

The packaged conformer template index contains 794 entries (CPBind 331,
CPSea 228, Scaffold 9, synthetic 226); production exposes a source-restricted
read-only view of 340 entries (CPBind + Scaffold) so template evidence scope
never silently expands.

## Tests and benchmark scope

This candidate includes the existing software tests but not the publication
benchmark harness, raw benchmark runs, external gold datasets, or article PDFs.
The selected CI smoke tests use synthetic inputs. Optional external test
fixtures can be supplied through `RDPEPPER_TEST_4INS_PDB` and
`RDPEPPER_TEST_SPECIAL_RESIDUES_DIR`; absent fixture files remain skipped.
No full-suite or cross-platform pass is inferred from the release smoke tests.

## License and third-party data

- **Source code**: MIT (see `LICENSE`).
- **Compiled NNAA library data** (`source=NNAA` rows): Creative Commons
  Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
- **Other third-party resources** remain under their own terms with notice:
  RDKit (BSD-3), Open Babel (GPL-2.0, optional inference engine), Meeko,
  AutoDock Vina (Apache-2.0), CPBind/CPSea (CC BY 4.0,
  Zenodo 10.5281/zenodo.17324994), AfCycDesign scaffolds (CC BY 4.0,
  Zenodo 10.5281/zenodo.15164650), CycPeptMPDB, and MAP↔SMILES conversion
  framework code with retained inline MIT notices and MAP_HELM_SMILES attribution
  (Copyright (c) 2021-2024 Charles Xu and others).

The package as a whole is not MIT. Full per-resource terms and attributions:
`THIRD_PARTY_DATA.md` and `NOTICE`.

## Citation

If RDpepper is useful in your research, please cite **RDpepper 7.1.0** as
identified by this release and its `ORIGINAL_SOURCE.json` source-tree hash. A
formal publication reference will accompany the manuscript; none is claimed
here in advance.
