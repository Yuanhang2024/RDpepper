# RDpepper Case Studies

Four end-to-end examples covering reconstruction, MOL2 export, charge-aware
readback, SMILES input, and docking preparation.

## Quick Start

```bash
# Install RDpepper
pip install rdpepper

# Run all cases
cd case_studies
python run_all.py

# Or run individual examples
python api_example.py
```

## Case Study 1: NNAA cyclic peptide (PRD_000227)

A cyclic peptide containing non-natural amino acids (codes `004`, `MHW`, `MHV`).
Tests library-external residue resolution and head-to-tail cyclization.

- Input: `data/PRD_000227.pdb` (chain L)
- Expected: X3 source-bound coordinates, 60 heavy atoms, formal charge 0
- Reference output: `reference_run/PRD_000227/`

```bash
cycpep export data/PRD_000227.pdb --chain L --format mol2 \
    --output outputs/PRD_000227.mol2 --best-available
```

## Case Study 2: Thioether cyclized peptide (PRD_000331)

A cystine-knot-like peptide with thioether cyclization.

- Input: `data/PRD_000331.pdb` (chain L)
- Expected: X3 source-bound, 42 heavy atoms
- Reference output: `reference_run/PRD_000331/`

## Case Study 3: TPO-containing peptide (PRD_000807)

A cyclic peptide with phosphothreonine (TPO), testing phosphorylated
residue handling.

- Input: `data/PRD_000807.pdb` (chain L)
- Expected: X3 source-bound, 58 heavy atoms
- Reference output: `reference_run/PRD_000807/`

## Case Study 4: Docking preparation (7EZW, eIF4E)

A full docking-chain demonstration: reconstruct a disulfide-cyclic peptide
ligand, prepare physiological microstate, generate PDBQT, dock with Vina.

- Input: `data/7ezw_ligand.pdb` (chain B) + `data/7ezw_receptor.pdb`
- Expected: Top-1 RMSD ≈ 1.4 Å to crystal pose (seed 17)
- Reference output: `reference_run/7ezw/`

```bash
# Prepare ligand
cycpep prepare-ligand data/7ezw_ligand.pdb --chain B \
    --ph 7.4 --output outputs/7ezw_ligand.pdbqt

# Dock (requires vina)
vina --receptor data/7ezw_receptor.pdbqt \
     --ligand outputs/7ezw_ligand.pdbqt \
     --center_x 12.56 --center_y 15.61 --center_z 24.43 \
     --size_x 32.5 --size_y 27.1 --size_z 32.9 \
     --exhaustiveness 16 --seed 17 \
     --out outputs/7ezw_docked.pdbqt
```

## Python API Example

See `api_example.py` for a complete end-to-end script demonstrating:
1. PDB → MOL2 reconstruction with validation receipt
2. Charge-aware MOL2 readback
3. SMILES → MOL2 direct input

## Expected Outputs

Each case has reference outputs in `reference_run/`:
- `*.mol2` — the expected MOL2 file
- `*.mol2.validation.json` — the expected validation receipt

Compare your outputs with:
```bash
diff outputs/PRD_000227.mol2 reference_run/PRD_000227/PRD_000227.mol2
```

## Data Provenance

Input PDB files are from the wwPDB BIRD (Biologically Interesting molecule
Reference Dictionary). Reference chemistry is from the PRDCC frozen truth.
No truth files are used by the production pipeline; they are for post-hoc
evaluation only.
