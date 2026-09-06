# Tests

cycpep_master ships two kinds of tests:

## Library-only unit tests (run anywhere, no external data)

These exercise the monomer library, MAP/HELM/BILN conversion, sub-library
derivation, and the conformer guard — they need only the repo itself:

```bash
pip install -e . pytest
python -m pytest cycpep_master/tests/ \
  -p no:cacheprovider --import-mode=importlib \
  --ignore=cycpep_master/tests/test_cyclization.py \
  --ignore=cycpep_master/tests/test_branched.py \
  --ignore=cycpep_master/tests/test_conformer_guard.py
```

These are the tests run in CI (`.github/workflows/tests.yml`).

## Structure-dependent tests (need gold-standard PDBs)

`test_cyclization.py`, `test_branched.py`, and `test_conformer_guard.py`
assert A==B==C==E cross-path agreement on a set of real protein-bound cyclic-
peptide PDBs. They require the `TestFiles_Example/` directory (~850 MB,
CPBind + Scaffold + PepPCBench structures), which is **not** bundled in this
repo. To run them:

1. Obtain the structure set (CPBind / Scaffold from their respective sources).
2. Place PDBs under `../TestFiles_Example/CPBind_Examples/` relative to this
   package (see `conftest.py:_PDB_DIR`).
3. Run the full suite:

```bash
python -m pytest cycpep_master/tests/ -p no:cacheprovider --import-mode=importlib
```

If the data is absent these tests `pytest.skip` rather than fail.
