# RDpepper Case Studies

Three automated case studies (run by `run_all.py`) plus one manual
docking demonstration (Case Study 4).

## Quick Start

```bash
pip install rdpepper==7.3.0
cd case_studies
python run_all.py
```

The runner creates a timestamped directory under `outputs/run_YYYYMMDD_HHMMSS/`
and writes a `summary.json` with per-case status. Exit code 0 = all passed.

## Case Study 1: NNAA cyclic peptide (PRD_000227)

A cyclic peptide containing non-natural amino acids (codes `004`, `MHW`, `MHV`).
Tests library-external residue resolution and head-to-tail cyclization.

- Input: `data/PRD_000227.pdb` (chain L)
- Expected: X3 source-bound coordinates, 60 heavy atoms, formal charge 0,
  chemical rigor C1:H (hypothesis tier — 3 unresolved NNAA monomers:
  004, MHW, MHV; resolved via bond-order inference)
- Reference output: `reference_run/PRD_000227/`

Manual command (equivalent to what `run_all.py` does):
```bash
rdpepper export data/PRD_000227.pdb outputs/PRD_000227.mol2 \
    --source-kind pdb --format mol2 --chain L
```

## Case Study 2: Thioether cyclized peptide (PRD_000331)

A cystine-knot-like peptide with thioether cyclization.

- Input: `data/PRD_000331.pdb` (chain L)
- Expected: X3 source-bound, 42 heavy atoms, rigor C2:H
- Reference output: `reference_run/PRD_000331/`

## Case Study 3: TPO-containing peptide (PRD_000807)

A cyclic peptide with phosphothreonine (TPO).

- Input: `data/PRD_000807.pdb` (chain L)
- Expected: X3 source-bound, 58 heavy atoms, rigor C2:H
- Reference output: `reference_run/PRD_000807/`

## Case Study 4: Docking demonstration (7EZW, eIF4E) — manual

> This case is **not** run by `run_all.py` (requires Open Babel and Vina).
> Follow the steps below manually.

A full docking-chain: reconstruct a disulfide-cyclic peptide ligand,
prepare physiological microstate, generate PDBQT, dock with Vina.

**Input files** (in `data/`):
- `7ezw_ligand.pdb` — the peptide ligand (chain B)
- `7ezw_receptor.pdb` — the eIF4E receptor (chain A)

**Step 1: Prepare receptor PDBQT**
```bash
rdpepper pdbqt receptor data/7ezw_receptor.pdb outputs/7ezw_receptor.pdbqt \
    --chain A
```

**Step 2: Prepare ligand PDBQT**
```bash
rdpepper pdbqt ligand-pdb data/7ezw_ligand.pdb outputs/7ezw_ligand.pdbqt \
    --chain B --ph 7.4
```

**Step 3: Dock with Vina** (requires `vina` installed)
```bash
vina --receptor outputs/7ezw_receptor.pdbqt \
     --ligand outputs/7ezw_ligand.pdbqt \
     --center_x 12.56 --center_y 15.61 --center_z 24.43 \
     --size_x 32.5 --size_y 27.1 --size_z 32.9 \
     --exhaustiveness 16 --seed 17 \
     --out outputs/7ezw_docked.pdbqt
```

**Expected**: Top-1 RMSD to the crystal ligand is approximately 1.3–1.5 Å
for seed 17 (see SI Table S18 of the manuscript for the full per-seed
results). Reference ligand artifacts (parent MOL2, physiological
microstate MOL2, PDBQT) are in `reference_run/7ezw/`.

## Python API Example

See `api_example.py` for a complete script demonstrating:
1. PDB → MOL2 reconstruction with validation receipt
2. Charge-aware MOL2 readback
3. SMILES → MOL2 direct input

```bash
python api_example.py
```

## Comparing Your Results

After running `run_all.py`, compare MOL2 files with the reference:
```bash
# find the latest run directory
diff outputs/run_*/PRD_000227.mol2 reference_run/PRD_000227/PRD_000227.mol2
```

Note: coordinate-level output may differ slightly across RDKit versions;
the InChIKey identity check (in the validation receipt) is the definitive
comparison.

## Data Provenance

Input PDB files are from the wwPDB BIRD (Biologically Interesting
molecule Reference Dictionary). The 7EZW receptor/ligand pair is from
PDB entry 7EZW (eIF4E–cyclic peptide complex). Reference outputs were
generated with RDpepper 7.3.0 on Windows (tree SHA-256 08cc3951…).

## Full Reference Results

See `reference_run/summary.json` for the complete machine-readable
record of the automated reference run (3 cases, all passed).
