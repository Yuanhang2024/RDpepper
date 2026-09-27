# Migrating from CycPep Master to RDpepper

RDpepper 5.1.0 changes the public distribution and product name without
renaming the implementation package.  This keeps existing Python imports,
serialized provenance, command lines, and historical artifact identifiers
valid.

## Installation

New environments should install the RDpepper distribution:

```bash
python -m pip install rdpepper
```

An installed RDpepper wheel provides both top-level imports:

```python
import rdpepper
import cycpep_master

assert rdpepper.__version__ == cycpep_master.__version__
```

The PyPI distribution identity is `rdpepper`.  A dependency declaration on the
old distribution name `cycpep-master` is not automatically satisfied by a
different distribution name; downstream package metadata should therefore be
updated to depend on `rdpepper>=5.1,<6`.

## Python API

New code should use the public facade:

```python
from rdpepper import application, reconstruct_structure
```

Existing imports continue to work:

```python
from cycpep_master import application, reconstruct_structure
```

RDpepper 5.1 exposes the stable top-level API and
`rdpepper.application`.  Deep implementation modules intentionally remain
under `cycpep_master.*`.  RDpepper does not alias the entire module tree,
because loading one implementation under two module names could split class
identity, registries, and caches.

## Command line and GUI

Use the new commands in documentation and automation:

```bash
rdpepper --version
rdpepper reconstruct peptide.cif
rdpepper-gui
```

The compatibility commands remain available and execute the same
implementation:

```bash
cycpep --version
cycpep reconstruct peptide.cif
cycpep-gui
```

No removal date is assigned to the compatibility commands in the 5.1 series.

## Stable historical identifiers

The following are not renamed:

- the source implementation package and working-directory convention
  `cycpep_master`;
- `cycpep_exact_v1` and existing schema-version strings;
- artifact IDs, hashes, frozen manifests, benchmark receipts, and provenance
  created by earlier releases;
- historical evidence that explicitly records software version 5.0.1 or
  earlier.

Branding does not alter chemical results or elevate their evidence level.
