# RDpepper User Manual — release 7.3.0

RDpepper reconstructs an auditable chemical graph from cyclic-peptide
coordinates (PDB/mmCIF) or sequence notation (HELM/MAP/BILN), materializes
validated MOL2 artifacts with per-atom provenance, and prepares receipt-backed
PDBQT inputs for docking. Every artifact carries explicit evidence grades:
recovery quality, chemical-graph rigor (`C0–C3`), and coordinate evidence
(`X0–X3`) are recorded independently, and degraded outputs are labeled rather
than silently rejected.

This manual describes the **7.3.0 wheel** (`rdpepper-7.3.0-py3-none-any.whl`).
All CLI examples below use options that exist in the released parser; the
command surface was captured with `rdpepper <command> --help` from the wheel
itself, and the three worked examples in [`case_studies/`](../case_studies/)
were executed end to end against this wheel before this document was written.

## 1. Installation

```bash
pip install rdpepper==7.3.0                       # pinned core install
pip install "rdpepper[inference,docking]==7.3.0"  # + Open Babel engine, Meeko PDBQT
pip install "rdpepper[gui,admet]==7.3.0"          # + PyQt5 GUI, ADMET prediction
```

Version pinning matters: this manual documents the 7.3.0 CLI and behavior,
so the examples below assume exactly that release.

**Core dependencies** (installed with the package, from the wheel metadata):
`rdkit==2026.3.3`, `gemmi>=0.7.0`, `pandas>=2.0`, `numpy>=1.24`.

**Optional extras**: `inference` (openbabel-wheel), `docking` (meeko,
scipy), `gui` (PyQt5), `admet` (admet-ai), `dev` (pytest, jsonschema),
`prior-build` (pyarrow, duckdb). Optional dependencies add capabilities
without affecting core results: without the Open Babel engine, bond-order
inference runs with the available engines (RDKit plus the template/geometry
candidates) and flags the absent engine with
`OPENBABEL_INFERENCE_UNAVAILABLE`; without admet-ai the ADMET step is
skipped; without Meeko, PDBQT preparation is unavailable. The AutoDock Vina
executable is never bundled — provide it via `VINA_BIN` or `PATH`.

**Platforms.** `Requires-Python >=3.10`; Python classifiers declare
3.10–3.14, and the package readme lists Linux/macOS/Windows as supported
systems. Release-level verification for this line was performed
on CPython 3.14.5 (Windows x64) with RDKit 2026.03.3 and Gemmi 0.7.5; the
commands and case studies in this manual were executed in exactly that
environment. No full pytest-suite pass is claimed for 7.3.0 in this manual,
and other version/platform combinations are supported by declaration, not by
a systematic test matrix.

The `rdpepper` command and `import rdpepper` are the stable public surface.
The implementation namespace `cycpep_master`, the `cycpep`/`cycpep-gui`
commands, and `python -m rdpepper` remain valid compatibility entry points.

## 2. Quick start (CLI)

### Reconstruct a peptide chain from a PDB file

```bash
# Unified dispatch (single source, JSON result)
rdpepper reconstruct example.pdb --chain L

# Strict, evidence-qualified fail-closed V6 route (legacy batch pipeline)
rdpepper reconstruct example.pdb --path v6 --chain L

# Legacy one-file / directory form
rdpepper --pdb example.pdb --chain L
rdpepper --dir ./inputs --csv results.csv
```

Reconstruction output is a JSON envelope (service commands) or a status line
(legacy form). Read `status`, `qualified_success`, and `chemical_rigor` — a
`success` envelope means "an inspectable artifact was returned", not
"qualified chemistry was established".

### Export a MOL2 with a validation receipt

```bash
rdpepper export example.pdb output/example.mol2 \
    --source-kind pdb --format mol2 --chain L \
    --seed 42 --num-confs 10 --fallback-policy max_coverage
```

This writes `output/example.mol2` **and** the companion receipt
`output/example.mol2.validation.json`. Notes, verified against the released
parser:

- `export` takes **two positional arguments** (`source output`); format
  selection is `--format {mol2,sdf}`, source kind is
  `--source-kind {smiles,coordinate,pdb,mmcif}`.
- Without `--strict-format` this command uses the best-available route
  (`export_best_available`); adding `--strict-format` switches to
  `export_structure`, which exits 1 when the requested format cannot be
  produced instead of degrading to a graph or metadata artifact.
- `--fallback-policy {strict_v6,max_coverage}` controls coordinate-gap
  handling for source-bound MOL2: `strict_v6` rejects incomplete
  source-coordinate mapping; `max_coverage` emits the MOL2 with mapped atoms
  at source coordinates and gaps completed from local bond geometry, carrying
  an `X2`/`X1` coordinate-tier receipt. Identity gates are unchanged by the
  policy. The default is `strict_v6` with `--strict-format`, `max_coverage`
  otherwise.
- `--monomer-context-json` supplies request-scoped custom monomer
  definitions, CCD files/directory, or component IDs (Section 6).

### Read a MOL2 back with receipt verification

```bash
rdpepper read-mol2 output/example.mol2 \
    --compatibility rdkit_charge_aware \
    --receipt output/example.mol2.validation.json \
    --export-sdf output/example.sdf
```

`read-mol2` reads a MOL2 into the RDKit graph layer. The compatibility
mode is an explicit per-call choice — `--compatibility rdkit_native`
(default) uses RDKit's own perception; `rdkit_charge_aware` restores the
formal charges declared in the file's
`@<TRIPOS>UNITY_ATOM_ATTR` record **before** RDKit sanitization, and the
reader never switches modes automatically. This is a
file-fidelity step, not chemistry inference: it does not decide whether the
declared charges are chemically correct, and it is not protonation. With
`--receipt`, the reader binds the file to its export-time receipt by SHA-256
and cross-checks the recorded identity (`receipt_verification.status`).
The reader's report states its own boundary: it describes one reader's parse
of the file, not an independent chemical truth.

### Protonate a validated MOL2 (pH 7.4 dominant microstate)

```bash
rdpepper protonate-mol2 output/example.mol2 output/example_ph74.mol2 \
    --receipt output/example.mol2.validation.json
```

Applies the deterministic pH 7.4 dominant-microstate policy while preserving
heavy-atom coordinates; a rule-based bookkeeping policy, not site-specific
pKa prediction.

### Prepare docking inputs (requires `[docking]` extra)

```bash
# Ligand PDBQT from a validated parent MOL2
rdpepper pdbqt ligand-mol2 output/example.mol2 output/example.pdbqt \
    --receipt output/example.mol2.validation.json

# Ligand PDBQT directly from peptide PDB coordinates
rdpepper pdbqt ligand-pdb example.pdb output/example.pdbqt --chain L

# Receptor PDBQT; --ph is a generic residue-state policy, not pKa prediction
rdpepper pdbqt receptor receptor.pdb output/receptor.pdbqt

# Box center from binding-site residues, then Vina (VINA_BIN or PATH)
rdpepper dock-center receptor.pdb --residues 42 57 88 --chain A
rdpepper vina ligand.pdbqt receptor.pdbqt docked.pdbqt \
    --center 12.5 15.6 24.4 --box-size 25 25 25 --seed 17
```

PDBQT consumes only receipt-validated MOL2 parents and inherits the parent's
evidence tiers; docking outputs are flexibility/pose evidence only and never
establish affinity. The `pdbqt` command is a group with the subcommands
`ligand` (from SMILES), `ligand-mol2` (from a validated parent MOL2),
`ligand-pdb` (from peptide PDB coordinates), `receptor`, and `validate`.

### Convert notation, build from sequence, inspect monomers

```bash
rdpepper convert --from map --to smiles '{nnr:7T2}AC'
rdpepper prepare-sequence ACDEFG ./prepared \
    --cyclization head-to-tail --conformers 4 --flexibility-mode balanced
rdpepper monomer list --query meA
rdpepper capabilities            # operations + dependency availability report
```

`convert` accepts `--from/--to` in `{map,helm,biln,smiles,exact_v1,edge_v1,legacy_v5}`.
Run `rdpepper <command> --help` for any command not shown here; the full
service list is: `capabilities, reconstruct, reconstruct-unified,
reconstruct-exact, reconstruct-result-first, convert, audit, compare, export,
batch-export, conformers, template, admet, protonate, protonate-mol2,
read-mol2, prepare-sequence, pdbqt, dock-center, vina, dock, batch-dock,
monomer`.

## 3. Exit codes and the JSON envelope

- Service commands print one JSON object (`--compact` for one line,
  `--json-out FILE` to also save it) and exit `0` only for
  `status ∈ {success, partial, match, mismatch}`; `invalid_input`, `failed`,
  `not_supported`, and write errors exit `1`.
- `export`/`pdbqt ligand-pdb` with `--strict-format` exit `1` whenever
  `requested_format_status ≠ "fulfilled"` — i.e. when a lower-rigor graph or
  metadata artifact was returned instead of the requested file format.
- The legacy `--pdb/--dir` form exits `1` on error outcomes and prints
  `STATUS:`/`RIGOR:`/`SMILES:` lines for scripts.
- `status="success"` means an artifact was returned. Before using a result,
  also read `requested_format_status` (`fulfilled` | `degraded_format` |
  `metadata_only` | `unavailable`), `artifact_status`,
  `qualification_status`, `chemical_rigor`, and `qualified_success`.

## 4. Evidence model (C0–C3 / X0–X3 and neighbors)

Recovery quality, chemical-graph evidence, coordinate evidence, format
qualification (`Q`), and flexibility evidence (`F`) are **independent axes**
in the artifact contract. The two you will see most often:

**Chemical-graph rigor `C0–C3`** — how the chemical graph was *established*:

| Label | Meaning (as implemented in the 7.3.0 ladder) |
|---|---|
| `C3:Q` | Qualified exact product: strict multi-source agreement plus qualification audit. Reserved for the strict qualified route; a diagnostic channel can never emit it. |
| `C2:R` | Recovered chemistry (e.g. high-quality result-first recovery). Automatically downgraded to `C2:H` while any diagnostic/integrity condition is present. |
| `C2:H` | Bound but heuristic/unresolved identity evidence (medium/candidate quality). Diagnostic outputs are capped at this level. |
| `C1:H` | Hypothesis-grade identity (single diagnostic family, unresolved, or agreed F/H identity). |
| `C1:R` | Partial graph; unresolved portions are explicit, never guessed. |
| `C0:C` | Raw coordinates only (connectivity recorded, no bond orders). |
| `C0:NONE` | Opaque input; no trustworthy chemistry basis. |

**Coordinate evidence `X0–X3`** — where the coordinates came from:

| Label | Meaning |
|---|---|
| `X3` | Every heavy atom's coordinates are mapped from the input coordinates; the receipt records the maximum source displacement (0.0 at the MOL2 writer's four-decimal coordinate precision). |
| `X2` | Mixed: ≥1 heavy atom uses an embedded/generated coordinate; the receipt lists `generated_heavy_atom_indices` and counts. |
| `X1` | No source mapping; all coordinates are embedded/regenerated. The chemical evidence (C-axis) is graded independently of coordinate origin. |
| `X0` | No trustworthy coordinate basis (e.g. opaque input). |

Legacy `L` labels (e.g. `L2:Q`, `L1:H`) are aliases of the canonical `C`
labels, emitted only where the canonical label allows them; an alias can
never out-claim its canonical label. Ambiguous identity is preserved: an
ambiguous candidate set retains all candidates with **no automatically
selected primary**, and lower-tier outputs are structured edge lists with
`bond_order=null` — never disguised as all-single-bond SMILES. `C` tiers are
evidence statements, not probabilities; `X` tiers are origin statements, not
accuracy statements.

## 5. Readers and the MOL2 compatibility layer

Many MOL2 files in the wild carry integer formal charges in a separate
`@<TRIPOS>UNITY_ATOM_ATTR` record while the `@<TRIPOS>ATOM` charge column
holds partial charges; RDKit's native MOL2 parser does not restore those
formal charges. The `rdkit_charge_aware` mode of `read-mol2` /
`rdpepper.read_mol2`:

1. validates the MOL2 `ATOM` table structure;
2. parses the declared UNITY formal charges from
   `@<TRIPOS>UNITY_ATOM_ATTR` (the partial-charge column is never applied as
   formal charge);
3. applies the declared charges **before** RDKit sanitization (sanitization
   is mandatory; the reader does not fall back to `rdkit_native` on its
   own — the mode is always the caller's explicit choice);
4. with `--receipt`, verifies the file against its export-time validation
   receipt (SHA-256 binding, identity cross-check).

The report is a reader's parse of the file — receipt verification is an
internal consistency check against the export receipt, not independent
chemical truth. On the frozen evaluation cohorts used in the manuscript, this
reader recovered 45/45 native-failed BIRD artifacts and 1,456/1,456
HighDB artifacts (1,501/1,501 combined); those are recovery statistics for
previously exported RDpepper-style artifacts, measured under the conditions
reported in the manuscript, and are not a guarantee for arbitrary third-party
MOL2 files.

## 6. Monomer resolution and optional CCD lookup

The 20 standard amino acids plus ACE/NME caps are built in; everything else
resolves through the packaged unified monomer library (13,152 entries). All
monomer-consuming entry points accept a **request-scoped** context
(`--monomer-context-json` / `monomer_context=`) with inline definitions,
CCD/PRD component snapshots, or a local CCD directory:

```bash
rdpepper prepare-sequence '[MyAA]AC' ./prepared-custom \
    --cyclization head-to-tail \
    --monomer-context-json '{"definitions":[{"symbol":"MyAA","smiles":"N[C@@H](CCl)C(=O)O"}]}'

rdpepper convert --from map --to smiles '{nnr:7T2}AC' \
    --monomer-context-json '{"ccd_directory":"./ccd","auto_resolve_required_symbols":true}'
```

Extensions live for the current operation only: they never mutate
`unified_monomer_library.csv`, the derived overlay, or a user library, and
every supplied component is recorded in the resolution ledger. Unresolved
symbols stay lower-rigor (`unresolved_output: C1:H_PARTIAL`) with the reason
in the ledger; unknown residues are **never guessed** into higher tiers —
local geometric inference may recover their graph at hypothesis/candidate
grade with explicit warning codes.

A standalone opt-in CCD resolver for unknown PDB residue codes is controlled
by environment variables and is **disabled by default**:

```bash
RDPEPPER_AUTO_CCD=1                  # enable (default: disabled)
RDPEPPER_CCD_CACHE_DIR=/path/to/ccd  # default: <LOCALAPPDATA>/rdpepper/ccd
RDPEPPER_CCD_ALLOW_NETWORK=0         # offline cache-only (default; =1 permits
                                     # fetching missing components from RCSB)
```

A disabled feature, an offline cache miss, or an unparsable component CIF
falls back silently to the existing inference ladder; no negative lookup is
persisted.

## 7. Caching (scope and guarantees)

RDpepper keeps a **performance-only** identity cache for the monomer-library
fold, keyed by the library file's SHA-256 fingerprint (so any library change
invalidates it automatically):

- default location: `%LOCALAPPDATA%/rdpepper/` (Windows) or the
  `TEMP`/`TMP`/working-directory fallback used by the same code path
  elsewhere;
- `RDPEPPER_IDENTITY_CACHE=<path>` relocates it (the file's directory is
  used);
- `RDPEPPER_DISABLE_IDENTITY_CACHE=1` disables the entire cache stack and
  forces recompute — results are identical, only runtime changes.

The cache never influences scientific output: a fingerprint mismatch or
unreadable location falls back to recomputing the original fold. A first run
in a fresh environment performs the one-time library fold; subsequent
processes reuse it. The companion case-study runner pins this cache inside
its own directory so runs stay self-contained. No controlled end-to-end
speedup ratio is claimed here; absolute runtimes for the reference machine
are recorded per command in each case-study run's `results.json`
(latest copy: `case_studies/latest_results.json`).

## 8. Python API

```python
import rdpepper

print(rdpepper.__version__)              # "7.3.0"

# Unified reconstruction: coordinates or sequence/HELM/MAP/BILN text
result = rdpepper.reconstruct_structure("input/example.pdb", chain_id="L", mode="auto")

# Best-available MOL2 export (same service as `rdpepper export`)
result = rdpepper.application.export_best_available(
    "input/example.pdb", "output/example.mol2",
    source_kind="pdb", output_format="mol2", chain_id="L",
    fallback_policy="max_coverage",
)
data = result["data"]
data["requested_format_status"]   # fulfilled | degraded_format | metadata_only
data["chemical_rigor"]            # C-axis label
data["coordinate_level"]          # X3 | X2 | X1 | X0

# Charge-aware MOL2 readback with receipt verification
report = rdpepper.read_mol2(
    "output/example.mol2",
    compatibility="rdkit_charge_aware",
    receipt_path="output/example.mol2.validation.json",
)
report["data"]["receipt_verification"]["status"]   # "verified"
```

The `rdpepper` package is a facade; deep modules live under `cycpep_master.*`
and `import cycpep_master` remains supported (`rdpepper.reconstruct_structure
is cycpep_master.reconstruct_structure`).

## 9. Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| `MONOMER_UNRESOLVED` / `unresolved` ledger entries | The registry reported no mapping for that residue code in this run; the graph may come from local inference at `C1:H`/`C2:H` | Provide `--monomer-context-json` definitions or a CCD directory, or use the labeled tier as recorded |
| `requested_format_status="degraded_format"` | MOL2/SDF could not be produced; a graph/metadata artifact was returned | Inspect `data.artifacts[]`; add `--strict-format` if you want a hard failure instead |
| Exit 1 from a service command | `status` not in {success, partial, match, mismatch}, or `--strict-format` unfulfilled | Read the JSON envelope's `error` / warning codes |
| `OPENBABEL_INFERENCE_UNAVAILABLE` | The Open Babel engine is absent; inference proceeded with the available engines and flagged it | `pip install "rdpepper[inference]==7.3.0"` to add the second engine |
| Vina step reports unavailable | Vina executable not found | Set `VINA_BIN` or add it to `PATH` (it is never bundled) |
| `MOL2 read failed` | File is not a MOL2 or is corrupted | Check the file starts with `#` or `@<TRIPOS>MOLECULE`; native parse failure of a UNITY-style file is exactly what `--compatibility rdkit_charge_aware` addresses |
| Ambiguous result, no single SMILES | Identity candidates disagree by design | Inspect `chemistry_candidates`/`alternatives` in the evidence; do not assume a primary was chosen |

## 10. Worked examples

The [`case_studies/`](../case_studies/) directory contains three executable
examples on real wwPDB BIRD/PRDCC inputs (a depsipeptide macrocycle and two
peptides without a declared inter-residue cyclic linkage, one of them
phosphothreonine-containing), each reconstructed, exported, and read back
with receipt verification, plus `run_all.py`, which returns a nonzero exit
code on any failure, and a machine-readable results file per run. The
recorded results there (`C1:H`/`C2:H`, `X3`) are the observed evidence
grades; the runs' non-standard residue codes remained unresolved in their
resolution ledgers, which the receipts record verbatim.

## 11. License, data terms, and citation

- **Code**: MIT (`LICENSE`). Bundled third-party data retain their own
  licenses, including the compiled NNAA tranche (CC BY-NC 4.0), the
  CycPeptMPDB subset and structural templates/priors (CC BY 4.0), and the
  HELM-GPT monomer subset (that project's MIT license) — see `NOTICE` and
  `THIRD_PARTY_DATA.md`. Installing the Python package does not relicense
  third-party data.
- **Repository**: the maintained source repository is
  [github.com/Yuanhang2024/RDpepper](https://github.com/Yuanhang2024/RDpepper).
  (The PyPI project metadata lists a different URL whose repository is not
  reachable; published release metadata cannot be edited, so this manual
  states the working location explicitly.)
- **Software archive**: RDpepper V7.3.0,
  [DOI 10.5281/zenodo.23201365](https://doi.org/10.5281/zenodo.23201365)
  (concept DOI 10.5281/zenodo.23201364), archived 2026-10-07. The archived
  **wheel** in that record is the verified release artifact (byte-identical
  to the PyPI 7.3.0 wheel); the record's source **zip** is a
  `7.3.0.dev0` snapshot and is not the exact evaluated source, so cite the
  wheel or the PyPI release when byte identity matters.
- **Citation**: Li, Jinghang; Yang, Yongliang. *RDpepper: evidence-graded
  cyclic-peptide reconstruction and validated MOL2-to-PDBQT preparation.*
  Software. A manuscript describing the software is in revision; no journal
  acceptance or publication status is claimed here. Correspondence:
  everbright99@163.com (Prof. Yongliang Yang, Quantova Therapeutics, Ltd.,
  Shanghai).
