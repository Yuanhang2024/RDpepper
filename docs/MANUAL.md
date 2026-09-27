# RDpepper User Manual

RDpepper reconstructs cyclic-peptide chemical graphs from PDB coordinate files
and exports validated MOL2 artifacts with per-atom provenance.

## Installation

```bash
pip install rdpepper
```

Requirements: Python ≥ 3.10, RDKit, gemmi. Optional: PyQt5 (GUI), Open Babel
(bond-order inference), admet-ai (ADMET).

## Quick Start (CLI)

### Reconstruct a cyclic peptide from a PDB file

```bash
cycpep reconstruct path/to/structure.pdb --chain L --format mol2 --output result.mol2
```

This produces:
- `result.mol2` — the MOL2 with SYBYL atom types, integer bond orders, and
  UNITY integer formal charges
- `result.mol2.validation.json` — a validation receipt with the InChIKey
  identity check, coordinate provenance, and charge assignment

### Export the best available artifact (with fallback)

```bash
cycpep export path/to/structure.pdb --chain L --format mol2 --output result.mol2 --best-available
```

When strict reconstruction fails, this falls back to the best-supported
artifact with explicit rigor labels and coordinate-tier provenance.

### Read an existing MOL2 file (with charge-aware compatibility)

```bash
cycpep read path/to/file.mol2 --compatibility rdkit_charge_aware
```

This reads a MOL2 through a staged path: native RDKit parsing first; on
failure, the compatibility loader restores UNITY formal charges from the
`@<TRIPOS>UNITY_ATOM_ATTR` record, preserving coordinates and identity.

### Prepare a ligand for docking (PDBQT)

```bash
cycpep prepare-ligand path/to/structure.pdb --chain L --ph 7.4 --output ligand.pdbqt
```

This chains: reconstruction → MOL2 → physiological microstate (pH 7.4) →
PDBQT with torsion-tree preparation.

## Quick Start (GUI)

```bash
cycpep-gui
```

The GUI provides: file open, chain selection, reconstruction, MOL2 export,
docking preparation, and result visualization.

## Quick Start (Python API)

```python
from cycpep_master import application

# Reconstruct and export MOL2
result = application.export_best_available(
    "path/to/structure.pdb",
    "output.mol2",
    source_kind="coordinate",
    output_format="mol2",
    chain_id="L",
)
print(result["status"])       # "success"
print(result["data"]["smiles"])  # canonical SMILES

# Read an existing MOL2 with charge awareness
read_result = application.read_mol2(
    "path/to/existing.mol2",
    compatibility="rdkit_charge_aware",
)
print(read_result["data"]["total_formal_charge"])

# Prepare a ligand for docking
dock_result = application.prepare_ligand_pdbqt(
    "path/to/structure.pdb",
    "ligand.pdbqt",
    chain_id="L",
)
```

## Coordinate Provenance Tiers

Every exported heavy atom carries one of two provenance labels:

- **X3 (source_bound)** — the atom's coordinates are mapped directly from the
  source PDB file; `max_source_coordinate_delta_angstrom = 0.0` in the
  validation receipt.
- **X1 (regenerated)** — the atom's coordinates were regenerated from the
  recovered chemical graph (ETKDG + MMFF optimization).

The validation receipt's `coordinate_level` field records the tier.

## Chemical Rigor Labels

The pipeline assigns per-artifact rigor labels:

- **C (confirmed)** — strict reconstruction succeeded; all evidence gates passed.
- **X (cross-validated)** — multiple families agree but at least one gate
  was relaxed.
- **Q (qualified)** — a single-family result with explicit uncertainty.
- **F (fallback)** — a parseable SMILES was materialized as a best-effort
  artifact when strict reconstruction failed.

## Charge-Aware MOL2 Compatibility Reader

Many existing MOL2 files (written by UCSF ChimeraX, Open Babel, or other
tools) fail direct RDKit readback because of UNITY formal-charge records
that RDKit's native parser does not restore. RDpepper's compatibility reader:

1. Validates the MOL2 ATOM table structure
2. Parses the `@<TRIPOS>UNITY_ATOM_ATTR` record for integer formal charges
3. Applies the charges to the RDKit molecule after sanitization
4. Verifies the result against the companion validation receipt (if present)

On the wwPDB BIRD reference set, this recovered 45/45 native-failed
artifacts; on a 2,504-entry HighDB set, 1,456/1,456 (100%).

## Optional: Auto-CCD Resolution

For library-external residue codes (e.g., non-natural amino acids not in
the monomer library), RDpepper can optionally resolve against the PDB
Chemical Component Dictionary:

```bash
export RDPEPPER_AUTO_CCD=1
export RDPEPPER_CCD_CACHE_DIR=/path/to/ccd/cache
export RDPEPPER_CCD_ALLOW_NETWORK=0  # offline; cache-only
cycpep reconstruct path/to/structure.pdb --chain L
```

## Performance Caching

RDpepper uses content-addressed durable caches under
`%LOCALAPPDATA%/rdpepper/` (Windows) or `~/.local/share/rdpepper/` (Linux/macOS)
to accelerate repeated processing. These caches are:

- **Content-addressed**: keyed by SHA-256 over the source file bytes
- **Self-invalidating**: any library edit changes the key and forces recompute
- **Kill switch**: `RDPEPPER_DISABLE_IDENTITY_CACHE=1` disables the entire stack

First run after install pays a one-time ~15 s cold-seeding pass; subsequent
runs are ~7× faster (median 4.9 s vs 34.0 s per BIRD entry).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `not_supported` status | Residue not in monomer library and de-novo inference failed | Check the residue is a valid amino acid; consider `RDPEPPER_AUTO_CCD=1` |
| `rejected` status | Reconstruction succeeded but strict gate failed | Inspect the validation receipt for the specific gate |
| MOL2 fails RDKit readback | UNITY charges not restored by native parser | Use `cycpep read --compatibility rdkit_charge_aware` |
| Slow first run | Durable caches seeding | Wait ~15 s; subsequent runs are fast |
| `MOL2 read failed` | File corrupted or wrong format | Check the file starts with `#` or `@<TRIPOS>MOLECULE` |

## License

MIT License. See [LICENSE](../LICENSE).

## Citation

If you use RDpepper in your research, please cite:

> Yang, Y. RDpepper: Evidence-Graded Cyclic-Peptide Reconstruction and
> Validated MOL2 Preparation. *J. Chem. Inf. Model.* (submitted).
> https://github.com/Yuanhang2024/RDpepper
