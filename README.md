# RDpepper 7.3.0

RDpepper prepares cyclic-peptide chemical graphs from PDB/mmCIF coordinates and exports MOL2 files with coordinate provenance, evidence labels, and validation receipts. Python, command-line, and optional desktop interfaces share the same application layer.

The workflow combines residue and monomer definitions, cross-link information, chemical reconstruction, and graded recovery. It records the distinction between a qualified structure, a recoverable candidate, and a topology or metadata result. An explicit charge-aware MOL2 reader restores declared formal charges before RDKit sanitization.

## Install

```bash
python -m pip install rdpepper==7.3.0

# Add Open Babel inference and Meeko docking preparation
python -m pip install "rdpepper[inference,docking]==7.3.0"

# Optional desktop interface
python -m pip install "rdpepper[gui]==7.3.0"
```

The core dependencies are RDKit 2026.3.3, Gemmi, NumPy, and pandas. AutoDock Vina is an external executable; provide it through `VINA_BIN` or `PATH` to run docking. See the [user manual](docs/MANUAL.md) for optional extras and configuration.

The declared Python requirement is 3.10 or later. The documented examples were checked on Windows with CPython 3.14.5, RDKit 2026.03.3, and Gemmi 0.7.5. The example records identify the installation and dependency versions used.

## Prepare and read a structure

```bash
# Reconstruct the selected peptide chain
rdpepper reconstruct peptide.pdb --chain L

# Export a MOL2 and its validation receipt
rdpepper export peptide.pdb output/peptide.mol2 \
    --source-kind pdb --format mol2 --chain L \
    --fallback-policy max_coverage

# Read back using declared UNITY formal charges, with receipt verification
rdpepper read-mol2 output/peptide.mol2 \
    --compatibility rdkit_charge_aware \
    --receipt output/peptide.mol2.validation.json \
    --export-sdf output/peptide.sdf
```

Inspect the returned format status, chemical evidence, coordinate origin, and warnings before selecting an output for downstream work. A returned artifact may carry a lower evidence level even when file export succeeds. Receipt verification checks the exported file against its recorded identity; reference-chemistry evaluation is a separate operation.

The reader modes are explicit. `rdkit_native` uses RDKit's native MOL2 parser. `rdkit_charge_aware` restores formal charges from `@<TRIPOS>UNITY_ATOM_ATTR` before sanitization. The MOL2 numeric charge column and UNITY declarations are separate fields; this operation does not calculate partial charges or predict protonation.

## Run the case studies

Three small, included coordinate examples demonstrate export, inspection of the evidence record, and receipt-verified readback:

```bash
cd case_studies
python run_all.py
```

Each run writes a new results directory and returns a nonzero exit code if a required input, command, output, or verification fails. The [case-study guide](case_studies/README.md) describes the inputs, recorded outcomes, and result files. The [reference run summary](case_studies/reference_run/summary.json) records verification against the published 7.3.0 wheel.

## Interfaces and evidence

- **Python:** `import rdpepper`; the implementation namespace `cycpep_master` remains supported.
- **CLI:** `rdpepper`; `cycpep` is a compatibility alias.
- **GUI:** `rdpepper-gui` after installing the `gui` extra.
- **Representations:** HELM, BILN, MAP, SMILES, and explicit monomer–port representation interfaces.
- **Downstream preparation:** validated MOL2 to Meeko PDBQT, with optional rule-based microstate preparation and Vina integration.

Chemical evidence (`C`), coordinate origin (`X`), and format qualification are recorded separately. X3 denotes source-mapped coordinates; regenerated coordinates remain explicitly identified. Full definitions and operation-specific behavior are in the [manual](docs/MANUAL.md) and [artifact contract](V5_ARTIFACT_CONTRACT.md).

Monomer definitions can be supplied for an individual request. Optional CCD resolution and library caches are documented in the manual. Runtime caches are keyed to their source libraries and can be disabled with `RDPEPPER_DISABLE_IDENTITY_CACHE=1`.

## Source layout and reproducibility

The implementation is stored at the repository root and packaged as `cycpep_master`. The `rdpepper/` directory supplies the public facade. A source checkout can be installed with:

```bash
python -m pip install .
```

[Evaluated source provenance](provenance/EVALUATED_SOURCE.json) identifies the 7.3.0 implementation and its relationship to the published wheel. Documentation and packaging metadata can be updated independently of the evaluated implementation. Earlier commits and release tags remain in the repository history.

The maintained repository is [Yuanhang2024/RDpepper](https://github.com/Yuanhang2024/RDpepper). The [PyPI 7.3.0 release](https://pypi.org/project/rdpepper/7.3.0/) supplies the wheel and source distribution. The release wheel is also archived at [Zenodo, version DOI 10.5281/zenodo.23201365](https://doi.org/10.5281/zenodo.23201365).

## License and citation

The software code is licensed under [MIT](LICENSE). Bundled third-party resources retain their source-specific terms, including the NNAA collection's non-commercial condition. See [NOTICE](NOTICE), [third-party data attribution](THIRD_PARTY_DATA.md), and [license texts](licenses/).

For the archived software:

> Li, Jinghang; Yang, Yongliang. *RDpepper V7.3.0*. Zenodo, 2026. https://doi.org/10.5281/zenodo.23201365.

Correspondence: Prof. Yongliang Yang, everbright99@163.com.
