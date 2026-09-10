# RDpepper repository guide

This guide maps the files and folders of the public RDpepper 7.2.0
distribution to what they contain and how they fit together. Usage,
the evidence model, and reconstruction routes are covered in the
[README](../README.md); this document is about the tree itself.

## Package shape and naming

- The runtime implementation lives **at the repository root** as a flat
  Python package. The build maps this root onto the `cycpep_master`
  package namespace (`package-dir = {cycpep_master = ".", rdpepper = "rdpepper"}`),
  so `import cycpep_master.core.mol2_format` resolves to `core/mol2_format.py`.
- [`rdpepper/`](../rdpepper) is the **public facade**: it re-exports the
  stable API (`rdpepper.reconstruct_structure`, `rdpepper.read_mol2`,
  `rdpepper.load_mol2`, `rdpepper.application`, ...) and provides the
  `rdpepper` / `rdpepper-gui` entry points. New code should import
  `rdpepper`; the `cycpep_master` import path, the legacy `cycpep` /
  `cycpep-gui` commands, and historical schema/artifact identifiers remain
  valid and are not rewritten.
- [`application.py`](../application.py) is the **shared service layer**
  behind both the CLI and the GUI. It is deliberately thin: it validates
  user-facing arguments, calls the scientific components, and normalizes
  their results into JSON-serializable operation results. It contains no
  reconstruction, comparison, conformer, or docking algorithms of its own.

### Entry points

| Command | Module | Notes |
|---|---|---|
| `rdpepper` | `rdpepper.cli` -> `cli/main.py` | Full CLI service surface; JSON results |
| `rdpepper-gui` | `rdpepper.gui` -> `gui/__main__.py` | PyQt5 desktop workspace |
| `rdpepper-overlay` / `cycpep-overlay` | `core/overlay_collector.py` | Overlay collection utility |
| `rdpepper-reproduce` / `cycpep-v5-reproduce` | `v5_reproducibility.py` | Deterministic V5 reproduction receipts |
| `cycpep`, `cycpep-gui` | `cli/main.py`, `gui/__main__.py` | Legacy aliases of the same services |

From a source checkout, `python run.py ...` and `python -m rdpepper ...`
reach the same implementation as the installed `rdpepper` command.

## Directory map

| Directory | Contents |
|---|---|
| [`core/`](../core) | Coordinate parsing (`pdb_parser.py`, `native_mmcif_graph.py`, `structure_io.py`), the cyclic-peptide chemistry graph (`cyclic_peptide_graph.py`, `cyclization.py`, `molecule.py`), monomer resolution and administration (`monomer_resolution.py`, `monomer_admin.py`, `derived_monomers.py`), special residues, MOL2 writing and compatibility reading (`mol2_format.py`, `mol2_compat.py`), evidence rigor (`rigor.py`), and CXSMILES generation |
| [`paths/`](../paths) | Diagnostic reconstruction routes: `path_a.py` (template assembly; route E is Path A with the geometric covalent-radius cyclization fallback, exposed via `paths/__init__.py`), `path_b.py` (HELM -> MAP -> SMILES via the monomer library), `path_c.py` (A + HETATM caps), `path_f.py`/`path_g.py`/`path_h.py` (special-residue and CONECT-driven routes), plus `residue_template_factory.py` and MAP utilities. Routes share major ancestors (A/C/E, B/G, F/H); they are candidates for V6 evidence adjudication, not seven independent votes |
| [`export/`](../export) | Conformer generation and validated MOL2/SDF ensemble export (`conformer.py`, `conformer_ensemble.py`), including source-coordinate preservation and coordinate-tier materialization |
| [`docking/`](../docking) | Ligand/receptor PDBQT preparation (`ligand_pdbqt.py`, `receptor_pdbqt.py`, `mol2_pdbqt.py`), deterministic pH 7.4 protonation policy (`protonation.py`), Meeko interface, box and flexibility budgeting, template library tooling, and the AutoDock Vina wrapper (`vina.py`, `vina_wrapper.py`, `workflow.py`) |
| [`gui/`](../gui) | PyQt5 desktop workspace: `main_window.py` (tabs for reconstruction, notation, 3D/properties, PDBQT & docking, monomer library, capabilities), `params_panel.py`, `workers.py`, `render.py` |
| [`cli/`](../cli) | Command-line interface over the shared application services (`main.py`) |
| [`admet/`](../admet) | Optional ADMET prediction wrapper around `admet-ai` (`predictor.py`); the step is skipped when the extra is not installed |
| [`compare/`](../compare) | SMILES comparison and authoritative-stereo comparison (`smiles_compare.py`, `authoritative_stereo.py`) |
| [`libraries/`](../libraries) | Per-source monomer sub-libraries (`core.csv`, `caps.csv`, `curated_cycpep.csv`, `nnaa_diverse.csv`, `helm_gpt.csv`, `special.csv`) with `manifest.json` |
| [`data/`](../data) | Runtime data: conformer template index and per-size/cyclization-type template PDBs (`templates/`), torsion priors (`torsion_priors/`), `applicability_manifest.json`, `v5_evidence_dossier.json` |
| [`schemas/`](../schemas) | JSON schemas for artifact contracts (`candidate_assessment.schema.json`, `exact_v1.schema.json`, `v5_artifact.schema.json`) |
| [`tests/`](../tests) | pytest suite with bundled synthetic fixtures (`fixtures/`) and golden files (`golden/`); see [tests/README.md](../tests/README.md) |
| [`licenses/`](../licenses) | Retained third-party license texts (CC-BY-4.0, CC-BY-NC-4.0, ChEMBL, HELM-GPT MIT) |

## Top-level modules and runtime data

| File | Role |
|---|---|
| `application.py` | Shared CLI/GUI service layer (see above) |
| `reconstruction.py` | `reconstruct_structure`: unified dispatch of PDB/mmCIF paths and sequence/HELM/MAP/BILN text |
| `representations.py` | Deterministic HELM/MAP/BILN -> SMILES conversion API |
| `sequence.py` | Sequence input adapter for the V5 artifact pipeline (used by `prepare-sequence`) |
| `pipeline.py` | Pipeline orchestrator: PDB -> SMILES -> compare -> ADMET -> export -> CSV |
| `result_first.py` | Result-first recovery facade: degrade-without-rejection artifact ladder on top of the same components |
| `max_coverage.py` | Max-coverage fallback primitives used by `fallback_policy="max_coverage"` |
| `bond_order_inference.py` | Auditable multi-engine bond-order candidate inference (materialization separated from qualification) |
| `exact_v1.py` | Exact monomer-port graph representation shared by audited adapters |
| `candidate_assessment.py` | Independent semantic validation of V6 candidate-assessment records |
| `chemical_audit.py`, `remediation_v3.py`, `remediation_v5.py`, `remediation_v6.py` | Fail-closed audit and versioned reconstruction adjudicators (v3 historical, v5 family-aware, v6 evidence-dimension acceptance) |
| `v5_reproducibility.py` | Deterministic V5 reproduction receipts |
| `build_monomer_library.py`, `build_sublibraries.py`, `validate_monomer_library.py` | Library build, sub-library derivation, and end-to-end monomer validation utilities |
| `unified_monomer_library.csv` | Packaged monomer library (13,152 entries: CycPeptMPDB 384, NNAA 9,998, HELM-GPT 2,770) |
| `special_residue_library.csv`, `derived_monomer_library.csv`, `derived_monomer_manifest.json`, `derived_monomer_quarantine.csv` | Special-residue and derived-monomer runtime data |
| `run.py` | Compatibility runner for installed or source-checkout use |
| `pyproject.toml`, `requirements.txt`, `environment.yml` | Packaging and dependency declarations (see below) |
| `ORIGINAL_SOURCE.json` | Provenance record of the 7.1.0 public tree and its frozen parent source |

## Optional components

Optional capability is opt-in through package extras; nothing optional is
silently required for core reconstruction:

| Extra | Provides | Where |
|---|---|---|
| `inference` | Open Babel multi-engine bond-order inference | `bond_order_inference.py` |
| `docking` | Meeko PDBQT preparation and Vina integration helpers | `docking/` |
| `gui` | PyQt5 desktop workspace (`rdpepper-gui`) | `gui/` |
| `admet` | `admet-ai` ADMET prediction (step skips when absent) | `admet/` |
| `dev` | pytest and jsonschema for the test suite | `tests/` |
| `prior-build` | pyarrow/duckdb for rebuilding torsion priors | `docking/build_torsion_priors.py` |

The **AutoDock Vina executable is not bundled** with this distribution.
The docking integration discovers it through the `VINA_BIN` environment
variable or `PATH`. Docking outputs remain flexibility/pose evidence only.

The 7.2.0 **MOL2 compatibility reader** (`read_mol2` / `load_mol2`,
`rdpepper read-mol2`) defaults to RDKit's native MOL2 perception; the
opt-in `rdkit_charge_aware` mode restores the formal charges declared in
the file's UNITY atom attributes before RDKit sanitization. It does not
rewrite the MOL2, does not infer chemical correctness, and is not a
protonation step; `--export-sdf` writes an optional new SDF artifact. In
the GUI the same function sits in the "PDBQT & docking" tab, under the
"Protonation & MOL2" sub-tab, in the "Read MOL2" group.

## Running the tests

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -p no:cacheprovider --import-mode=importlib
```

Most tests run on bundled synthetic fixtures. Tests that need external
structure sets (for example the gold CPBind set or insulin/special-residue
PDBs that are not part of this repository) skip automatically when the
files are absent. This public tree ships no CI workflow; no full-suite or
cross-platform pass is implied by the release smoke tests.

## What this public tree intentionally does not contain

- The publication benchmark harness, raw benchmark runs, external gold
  datasets, and article/manuscript files.
- The AutoDock Vina executables (external dependency; see `VINA_BIN`).
- Historical internal QA receipts, source manifests, and planning
  documents; the exclusions of the 7.1.0 public tree are recorded in
  `ORIGINAL_SOURCE.json`.

## License map

- **Source code**: MIT (`LICENSE`).
- **Bundled data**: CC BY-NC 4.0 (NNAA collection), CC BY 4.0
  (CycPeptMPDB subset, CPBind/CPSea and AfCycDesign-derived
  templates/priors), HELM-GPT MIT (monomer subset). Texts are preserved
  in `licenses/`, `NOTICE`, and `THIRD_PARTY_DATA.md`.

The package as a whole is **not** MIT; per-resource terms apply. See the
README's license section for the full list.
