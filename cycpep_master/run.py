"""Compatibility entry point for an installed or source-checkout RDpepper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SOURCE_ROOT = Path(__file__).resolve().parent
if importlib.util.find_spec("cycpep_master") is None:
    spec = importlib.util.spec_from_file_location(
        "cycpep_master",
        _SOURCE_ROOT / "__init__.py",
        submodule_search_locations=[str(_SOURCE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot initialize the cycpep_master compatibility package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cycpep_master"] = module
    spec.loader.exec_module(module)

from cycpep_master.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
