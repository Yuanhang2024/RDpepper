"""RDpepper command-line entry point."""

from __future__ import annotations

from collections.abc import Sequence

from cycpep_master.cli.main import main as _compatibility_main


def main(argv: Sequence[str] | None = None) -> int:
    return _compatibility_main(
        argv,
        program_name="rdpepper",
        product_name="RDpepper",
        distribution_name="rdpepper",
    )


if __name__ == "__main__":
    raise SystemExit(main())
