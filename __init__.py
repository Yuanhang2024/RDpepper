__version__ = "7.2.0"

__all__ = [
    "__version__",
    "CyclicPeptideGraph",
    "UnifiedReconstructionResult",
    "exact_v1_equivalent",
    "map_to_exact_v1",
    "monomer_resolution_context",
    "prepare_ligand_from_sequence",
    "load_mol2",
    "read_mol2",
    "resolve_monomers",
    "reconstruct_structure",
]


def __getattr__(name):
    if name == "reconstruct_structure":
        from .reconstruction import reconstruct_structure

        return reconstruct_structure
    if name == "UnifiedReconstructionResult":
        from .reconstruction import UnifiedReconstructionResult

        return UnifiedReconstructionResult
    if name == "CyclicPeptideGraph":
        from .core.cyclic_peptide_graph import CyclicPeptideGraph

        return CyclicPeptideGraph
    if name in {"exact_v1_equivalent", "map_to_exact_v1"}:
        from . import exact_v1

        return getattr(exact_v1, name)
    if name == "prepare_ligand_from_sequence":
        from .application import prepare_ligand_from_sequence

        return prepare_ligand_from_sequence
    if name == "resolve_monomers":
        from .application import resolve_monomers

        return resolve_monomers
    if name == "load_mol2":
        from .core.mol2_compat import load_mol2

        return load_mol2
    if name == "read_mol2":
        from .application import read_mol2

        return read_mol2
    if name == "monomer_resolution_context":
        from .core.monomer_resolution import monomer_resolution_context

        return monomer_resolution_context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
