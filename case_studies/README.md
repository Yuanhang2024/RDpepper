# RDpepper Case Studies

Three end-to-end examples showing reconstruction → MOL2 → readback → docking
preparation, using real wwPDB BIRD entries with known reference chemistry.

## Case Study 1: Standard cyclic peptide (BIRD PRD_000227, 004-containing)

A cyclic peptide containing a non-natural amino acid (code `004`,
(2S)-amino(phenyl)ethanoic acid). RDpepper resolves the NNAA through the
monomer library, detects head-to-tail cyclization, and exports X3
source-bound coordinates.

```bash
# 1. Reconstruct and export MOL2
cycpep export case_studies/data/PRD_000227.pdb --chain L \
    --format mol2 --output outputs/PRD_000227.mol2 --best-available

# 2. Read back with charge awareness
cycpep read outputs/PRD_000227.mol2 --compatibility rdkit_charge_aware

# 3. Prepare for docking
cycpep prepare-ligand case_studies/data/PRD_000227.pdb --chain L \
    --ph 7.4 --output outputs/PRD_000227.pdbqt
```

Expected: `status=success`, coordinate_level=X3, L1/L2/L3 identity check.

## Case Study 2: Disulfide-cyclic peptide (BIRD PRD_000331, thioether)

A cystine-knot peptide with 3 disulfide bonds. Tests multi-bond cyclization
topology detection and S-S bond-order assignment.

```bash
cycpep export case_studies/data/PRD_000331.pdb --chain A \
    --format mol2 --output outputs/PRD_000331.mol2 --best-available
```

Expected: `status=success`, topology_class=SS, 3 SSBOND records detected.

## Case Study 3: Known-site redocking (PDB 7EZW, eIF4E / disulfide peptide)

A full docking demonstration: reconstruct → MOL2 → physiological microstate →
PDBQT → Vina docking (3 seeds). Compare the top-ranked pose to the crystal
ligand position (Top-1 RMSD ~1.3 Å).

```bash
# Full chain
cycpep prepare-ligand case_studies/data/7ezw_ligand.pdb --chain B \
    --ph 7.4 --output outputs/7ezw_ligand.pdbqt

# Dock with Vina (requires vina installed)
vina --receptor case_studies/data/7ezw_receptor.pdbqt \
     --ligand outputs/7ezw_ligand.pdbqt \
     --center_x 12.56 --center_y 15.61 --center_z 24.43 \
     --size_x 32.5 --size_y 27.1 --size_z 32.9 \
     --exhaustiveness 16 --seed 17 \
     --out outputs/7ezw_docked.pdbqt
```

Expected: Top-1 RMSD ≈ 1.4 Å (seed 17), <2 Å in 3/3 seeds.

## Running all case studies

```bash
cd case_studies
bash run_all.sh   # or: py -3.14 run_all.py
```

## Data provenance

All input PDB files are from the wwPDB BIRD (Biologically Interesting
molecule Reference Dictionary) reference set. Reference chemistry is from
the PRDCC (PRD Canonical Chemistry) frozen truth. No truth files are used
by the production pipeline; they are for post-hoc evaluation only.
