# Tests

pytest suite for the RDpepper distribution. The runtime implementation is
the flat repository root packaged as `cycpep_master`; these tests import
`cycpep_master.*` and run from the repository root.

## Running

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -p no:cacheprovider --import-mode=importlib
```

`.[dev]` installs pytest and jsonschema on top of the core dependencies
declared in `pyproject.toml` (rdkit, gemmi, pandas, numpy); tests that use
optional features need the matching extras (`.[gui]`, `.[docking]`,
`.[inference]`, `.[admet]`).

Most tests run on bundled synthetic inputs (`fixtures/v2_inputs`,
`fixtures/mol2_compat`, `golden/vina_wrapper`). This public tree ships no
CI workflow.

## Structure-dependent tests (external data, not bundled)

Some tests need real protein-bound cyclic-peptide structures and skip when
they are absent:

- `test_cyclization.py`, `test_geometry_params.py`, and
  `test_linear_topology_contract.py` use the session `pdb_files` fixture
  from `conftest.py`, which discovers gold-standard CPBind PDBs under
  `TestFiles_Example/CPBind_Examples` resolved one directory above the
  repository root (~850 MB structure set; not part of this repository).
  `test_cyclization.py` asserts A==B==C==E cross-path agreement on those
  structures.
- `test_branched.py` (insulin 4INS multichain) and
  `test_special_residues.py` (stapled p53, daptomycin, nisin) reference
  external fixture locations and skip when the files are absent.

If the external data is absent these tests `pytest.skip` rather than fail.
A skip means the check did not run, not that it passed; no full-suite or
cross-platform pass is implied. No environment-variable fixture override
is wired in this release.
