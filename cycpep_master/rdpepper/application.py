"""Compatibility-preserving access to :mod:`cycpep_master.application`."""

from __future__ import annotations

from cycpep_master import application as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
