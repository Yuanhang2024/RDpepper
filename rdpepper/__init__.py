"""Public RDpepper facade over the compatibility ``cycpep_master`` package.

The implementation remains in :mod:`cycpep_master` so existing imports,
serialized provenance, and historical workflows continue to work.  New code
should prefer ``import rdpepper``.
"""

from __future__ import annotations

from cycpep_master import __version__

__all__ = [
    "__version__",
    "CyclicPeptideGraph",
    "UnifiedReconstructionResult",
    "application",
    "exact_v1_equivalent",
    "load_mol2",
    "map_to_exact_v1",
    "monomer_resolution_context",
    "prepare_ligand_from_sequence",
    "read_mol2",
    "reconstruct_structure",
    "resolve_monomers",
]


def __getattr__(name: str):
    if name == "application":
        from cycpep_master import application

        return application
    if name in {
        "CyclicPeptideGraph",
        "UnifiedReconstructionResult",
        "exact_v1_equivalent",
        "load_mol2",
        "map_to_exact_v1",
        "monomer_resolution_context",
        "prepare_ligand_from_sequence",
        "read_mol2",
        "reconstruct_structure",
        "resolve_monomers",
    }:
        import cycpep_master

        return getattr(cycpep_master, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
