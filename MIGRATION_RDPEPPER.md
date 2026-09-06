# Migrating from CycPep Master to RDpepper

RDpepper 5.1.0 changed the public distribution and product name without
renaming the implementation package. This keeps existing Python imports,
serialized provenance, command lines, and historical artifact identifiers
valid. The current release is 7.1.0.

## Installation

Install the accompanying local wheel or source checkout. This is a held
publication-preparation candidate, not an uploaded GitHub/PyPI release:

```bash
python -m pip install ./rdpepper-7.1.0-py3-none-any.whl
python -m pip install .
```

An installed RDpepper distribution provides both top-level imports:

```python
import rdpepper
import cycpep_master

assert rdpepper.__version__ == cycpep_master.__version__
```

The distribution identity is `rdpepper`. A dependency declaration on the old
distribution name `cycpep-master` is not automatically satisfied by a
different distribution name; downstream package metadata should depend on
`rdpepper>=7.1` (or the applicable bound for your compatibility surface).

## Python API

New code should use the public facade:

```python
from rdpepper import application, reconstruct_structure
```

Existing imports continue to work unchanged:

```python
from cycpep_master import application, reconstruct_structure
```

RDpepper exposes the stable top-level API (`reconstruct_structure`,
`application`, `prepare_ligand_from_sequence`, `resolve_monomers`,
`map_to_exact_v1`, `monomer_resolution_context`, and the core result types)
through `rdpepper`. Deep implementation modules intentionally remain under
`cycpep_master.*`; RDpepper does not alias the entire module tree, because
loading one implementation under two module names could split class identity,
registries, and caches. Migrate imports module by module rather than relying
on a full-tree alias.

## Command line and GUI

Use the new commands in documentation and automation:

```bash
rdpepper --version
rdpepper reconstruct example.pdb
rdpepper-gui
```

The compatibility commands remain installed and execute the same
implementation:

```bash
cycpep --version
cycpep reconstruct example.pdb
cycpep-gui
```

A source checkout additionally supports `python run.py ...` and
`python -m rdpepper ...` with the same behavior. Both command names are present in this frozen version.

## Stable historical identifiers

The following are not renamed and must not be rewritten when migrating:

- the implementation package and source-layout convention `cycpep_master`;
- `cycpep_exact_v1` and existing schema-version strings;
- artifact IDs, hashes, frozen manifests, benchmark receipts, and provenance
  created by earlier releases;
- historical evidence that explicitly records earlier software versions.

Branding does not alter chemical results or elevate their evidence level.
