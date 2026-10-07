# RDpepper Case Studies (release 7.3.0)

Three executable examples that reconstruct a chemical graph from a real wwPDB
BIRD/PRDCC coordinate file, export a MOL2 artifact with a validation receipt,
and read that MOL2 back through the charge-aware compatibility reader with
receipt verification. Everything on this page was checked against the actual
7.3.0 wheel (`rdpepper-7.3.0-py3-none-any.whl`); the observed results below
are copied from the recorded reference run.

## Inputs

All three inputs are BIRD PRDCC → PDB conversions and use **chain L**. The
topology column states what the file's own CONECT records declare, together
with what the reference-run graph contained.

| File | Chain | Residues (file order) | Linkage declared in the file |
|---|---|---|---|
| `PRD_000227.pdb` | L | MHW, THR, DBB, PRO, MEA, MHV, 004 | CONECT 15↔51 declares a Thr side-chain **OG1 ester to the C-terminal carboxyl of residue 004**, closing the macrocycle; the remaining CONECT records are the backbone links. |
| `PRD_000331.pdb` | L | PHQ, ASP, GLU, VAL, ASA | Backbone links only. No inter-residue cyclic linkage is declared (no closing CONECT, SSBOND, or LINK record), and the reference-run graph contained no cyclic link. |
| `PRD_000807.pdb` | L | HCI, PRO, LEU, HIS, SER, TPO, ALA, NH2 | Backbone links only, ending in a C-terminal amide (NH2). No inter-residue cyclic linkage is declared, and the reference-run graph contained no cyclic link. Contains a phosphothreonine (TPO with P, O1P, O2P, O3P atoms). |

## Observed results (reference run, wheel 7.3.0)

Environment: CPython 3.14.5, Windows, RDKit 2026.03.3, Gemmi 0.7.5. The
runner records exact versions, paths, commands, timings, and hashes in the
machine-readable results file of each run.

| Case | Export | Format | Chemical rigor | Coordinates | Readback |
|---|---|---|---|---|---|
| PRD_000227 | `success` | fulfilled | `C1:H` | `X3` source-bound, 60/60 heavy atoms mapped | sanitized, InChIKey `FEPMHVLSLDOMQC-IYPFLVAKSA-N`, receipt verified |
| PRD_000331 | `success` | fulfilled | `C2:H` | `X3` source-bound, 42/42 heavy atoms mapped | sanitized, InChIKey `RKEUSPKRYPGTDQ-AYMMHLMVSA-N`, receipt verified |
| PRD_000807 | `success` | fulfilled | `C2:H` | `X3` source-bound, 58/58 heavy atoms mapped | sanitized, InChIKey `ATYOYHNMUAPYOW-ZEJNGAGRSA-N`, receipt verified |

Each case's graph was produced by the result-first bond-order inference
ladder (recorded as `result_origin: bond_order_inference_hypothesis` for
PRD_000227 and `bond_order_inference_candidate` for PRD_000331/807), so the
chemistry is labeled `C1:H`/`C2:H` — hypothesis/candidate grade, not a
qualified library-templated graph. The per-case resolution ledgers state the
outcome for every residue code, e.g. THR→T, PRO→P, DBB→dAbu, MEA→meA
resolved for PRD_000227, while MHW, MHV, and 004 remained unresolved in that
run (ledger: "No Unified symbol mapping for PDB residue …"); PHQ/ASA and
HCI/TPO remained unresolved in their runs the same way. If a CCD component
snapshot or a `--monomer-context-json` definition is supplied for those
codes, the evidence labels can rise — the receipts are the record.

## Running one case by hand

```bash
# 1. Reconstruct chain L and export MOL2 (best-available route; writes
#    PRD_000227.mol2.validation.json next to the output)
rdpepper export data/PRD_000227.pdb outputs/PRD_000227.mol2 \
    --source-kind pdb --format mol2 --chain L \
    --seed 42 --num-confs 10 --fallback-policy max_coverage

# 2. Read the MOL2 back with receipt verification
rdpepper read-mol2 outputs/PRD_000227.mol2 \
    --compatibility rdkit_charge_aware \
    --receipt outputs/PRD_000227.mol2.validation.json
```

Both commands print a JSON envelope. Check `status`, and for exports also
`requested_format_status` (`fulfilled` means the MOL2 was produced). Run
`rdpepper <command> --help` for the authoritative options of any subcommand
(for example `pdbqt receptor --ph` for a generic residue-state pH policy on
receptor preparation).

## Running all three cases

```bash
cd case_studies
python run_all.py                 # fresh run directory under outputs/
python run_all.py --output-dir mydir --expected-version 7.3.0
```

The runner returns a nonzero exit code on failure:

- **exit 2** before any case runs when an input file is missing, the wheel
  target has no rdpepper package, or the installed version does not match
  the pinned expected version (7.3.0 by default; the mismatch is recorded as
  an informative failure rather than a release-verification claim);
- **exit 1** when any case fails a check — a nonzero CLI exit, a missing
  MOL2/receipt/JSON artifact, a MOL2 missing its required Tripos records,
  a receipt whose SHA-256 does not match the hash recorded in the export
  result, a format request that was not fulfilled, or a readback without a
  verified receipt and a sanitized graph;
- **exit 0** only when all three cases pass end to end.

Each run writes its own directory `outputs/run_YYYYMMDD_HHMMSS/` with the
MOL2s, receipts, per-case JSON, `run.log`, and `results.json`; a pointer
copy of the latest results is kept at `latest_results.json`. Nothing is ever
deleted or overwritten — an existing run directory is refused — and earlier
runs stay intact (preserved runs live under `history/`). Set
`RDPEPPER_WHEEL_TARGET` to a `pip install --target` directory to pin the
code under test; the runner records `rdpepper.__version__` and
`rdpepper.__file__` so the install actually used is visible. The runner
points `RDPEPPER_IDENTITY_CACHE` at `case_studies/.cache/` so no cache files
are written outside this directory. The recorded examples used the `inference` extra. Open Babel supplies an
additional bond-order-inference engine; Meeko is not used by these examples.

## Scope: what these cases do not do

- These PRD files contain only the peptide chain; there is no receptor
  structure here, so no docking is run. For the full known-site redocking
  study (eight protein–cyclic-peptide complexes, including the eIF4E
  complex, three seeds per complex, 24 total), see the manuscript Supporting
  Information; the receptor structures and per-case study data are not
  redistributed in this repository.
- Runtime varies with hardware and cache state (the first run in a fresh
  environment also builds the monomer-library fold cache); see the `timings`
  block of each run's `results.json` for the recorded per-command seconds
  rather than trusting a single number.

## Data provenance

Input PDB files are wwPDB BIRD PRDCC reference conversions (files carry the
header `BIRD PRDCC -> PDB conversion of PRDCC_000227/331/807`). Reference
chemistry in PRDCC is not consulted by the pipeline; reconstruction here
uses only the coordinates, CONECT records, and the packaged monomer library.
The copies in `data/` are byte-identical to the repository's frozen case
inputs (each file's SHA-256 is recorded in the run results).

## License note

The package code is MIT. Bundled third-party data retain their own
licenses, including the compiled NNAA tranche (CC BY-NC 4.0), the
CycPeptMPDB subset and structural templates/priors (CC BY 4.0), and the
HELM-GPT monomer subset (that project's MIT license) — see `NOTICE` and
`THIRD_PARTY_DATA.md` in the repository root.
