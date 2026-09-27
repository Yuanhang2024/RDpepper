"""AutoDock Vina Python wrapper for peptide-protein docking.

Requirements:
  - AutoDock Vina binary on PATH, or via the VINA_BIN env var, or bundled in
    cycpep_master/vina/ (e.g. vina_1.2.7_win.exe on Windows; `vina` on Linux).
    Download from https://github.com/ccxvii/vina/releases if absent.
  - MGLTools or OpenBabel for PDBQT conversion (optional, can use meeko)
  - Meeko (pip install meeko) for modern PDBQT generation

Usage:
    from docking.vina_wrapper import dock_peptide

    affinity, output = dock_peptide(
        peptide_pdb="peptide.pdb",
        receptor_pdb="protein.pdb",
        center=(10.5, 20.3, 15.2),
        box_size=(25, 25, 25),
    )
"""

import subprocess
from typing import Optional, Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

EXHAUSTIVENESS = 32  # Search thoroughness (default 8, increase for accuracy)
NUM_MODES = 9  # Number of binding modes to generate


def _find_vina() -> Optional[str]:
    """Locate the AutoDock Vina executable in a cross-platform way.

    Search order:
      1. ``VINA_BIN`` environment variable (explicit user override).
      2. ``vina`` on PATH (``shutil.which``) — works on Linux/macOS after
         ``pip install vina`` or a system install, and on Windows if added to
         PATH.
      3. A bundled binary under the package ``vina/`` directory, selected by
         platform (``vina_1.2.7_win.exe`` on Windows; ``vina`` on Linux).

    Returns the executable path or None.
    """
    from .vina import find_vina

    return find_vina()


# ══════════════════════════════════════════════════════════════════════════════
# PDBQT Conversion (using Meeko - pure Python, no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def pdb_to_pdbqt_meeko(pdb_path: str, pdbqt_path: str, is_receptor: bool = False) -> Optional[str]:
    """Prepare PDBQT via validated ligand MOL2 or rigid receptor typing."""
    if is_receptor:
        return _pdb_to_pdbqt_simple_receptor(pdb_path, pdbqt_path)
    from .ligand_pdbqt import pdb_to_ligand_pdbqt

    return pdb_to_ligand_pdbqt(pdb_path, pdbqt_path)


def _autodock_atom_type(atom) -> str:
    """Map an RDKit atom to an AutoDock (Vina) atom type.

    Rigid-receptor typing: carbons are aromatic (A) or aliphatic (C); N/O are
    split into H-bond acceptors (NA/OA) vs plain (N/O) by whether they carry a
    lone pair available for acceptance (heuristic: N/O with any implicit/explicit
    H are donors typed N/O, others acceptors NA/OA); S->SA, H(polar)->HD. This is
    the standard AutoDock4 receptor typing Vina 1.1.2 expects.
    """
    from .receptor_pdbqt import autodock_atom_type

    return autodock_atom_type(atom)


def _pdb_to_pdbqt_simple_receptor(pdb_path: str, pdbqt_path: str) -> Optional[str]:
    """Convert a receptor PDB to a rigid PDBQT (no torsion tree).

    Receptors dock rigidly, so the PDBQT is just ATOM records carrying an
    AutoDock atom type + a Gasteiger partial charge. RDKit parses the structure
    (sanitize-tolerant, since receptor PDBs from AlphaFold/CPBind can have minor
    valence quirks), computes Gasteiger charges, and we emit AutoDock4-style
    PDBQT lines. Nonpolar hydrogens are merged (dropped); polar H stay as HD.
    Returns None on success, an error string on failure.
    """
    from .receptor_pdbqt import pdb_to_receptor_pdbqt

    return pdb_to_receptor_pdbqt(
        pdb_path, pdbqt_path, atom_type=_autodock_atom_type
    )


def protonate_ph74(smiles: str) -> str:
    """Set the dominant pH-7.4 protonation state on a peptide SMILES.

    The token→MAP→SMILES chain (and PDB→SMILES paths) emit NEUTRAL forms (Lys
    -NH2, Asp/Glu -COOH, Arg neutral guanidine). Docking electrostatics (meeko/
    Gasteiger, which honors formal charge) then under-represents charged residues,
    and logP/TPSA-based permeability proxies over-estimate membrane passage.
    Peptide ionizable groups have pKa far from 7.4, so the dominant microstate is
    deterministic — no full pKa predictor needed. Complete rule set (net at 7.4):
      ANIONS (deprotonate):
        carboxylic acid  -C(=O)[OH] -> -COO(-)    (Asp/Glu sidechain, C-term)
        sulfonic acid    -S(=O)2[OH] -> -SO3(-)    (pKa <0; NNAA)
        phosphate/phosphonate P(=O)[OH] -> P-O(-)  (both OH -> net ~-2; pSer/pThr/pTyr, NNAA)
      CATIONS (protonate):
        aliphatic 1° amine -[NH2] on sp3 C -> -NH3(+)   (Lys sidechain, N-term)
        aliphatic 2° amine -[NH]<          -> -NH2(+)<  (NNAA N-alkyl)
        aliphatic 3° amine  N<             -> -N(+)<     (NNAA)
        guanidine (Arg)                    -> guanidinium (+1)
        amidine                            -> amidinium  (+1; NNAA)
      NEUTRAL (documented simplification — pKa near/above 7.4, dominant state ~neutral):
        His imidazole (pKa ~6), Cys thiol (~8.3), Tyr phenol (~10)
    EXCLUDED (never charged): backbone/side amide N (!$(NC=O)), aromatic N of
    indole/imidazole (!$([nX3])), enamine/imine N (!$(N=*)) — verified: a
    Trp/His/Pro + R/K/D/E peptide yields net 0 with 0 aromatic-N charged.
    Verified: meeko carries these formal charges into the PDBQT (RRRR peptide
    net +9.0, ILKEL net -2.0), vs neutral form which loses them (net ~0).
    Returns the protonated SMILES, or the input unchanged on any failure.
    """
    from .protonation import protonate_ph74 as apply_protonation

    return apply_protonation(smiles)


def _bond_dihedral_sigma(
    mol_3d,
    bond_atom_pairs,
    n_confs=8,
    random_seed=42,
    num_threads=1,
    metadata_out=None,
):
    """Per-bond rotational flexibility from a quick conformer ensemble.

    For each rotatable bond (given as a 0-based atom-index pair), embeds an
    ETKDG ensemble of `mol_3d` and measures the circular standard deviation of
    that bond's dihedral across conformers. Low sigma = the bond barely rotates
    across sampled shapes (effectively rigid); high sigma = a genuinely flexible
    torsion. Returns {(<i>,<j>): sigma_degrees}. Bonds whose sigma can't be
    computed (no neighbor to define a dihedral) map to +inf so they are the LAST
    to be frozen (never freeze a bond we couldn't measure).

    This is the ranking used by `_apply_adaptive_torsion_budget` to decide which
    bonds to rigidify when a peptide exceeds a configured torsion ceiling.
    """
    from .torsion_budget import bond_dihedral_sigma

    return bond_dihedral_sigma(
        mol_3d,
        bond_atom_pairs,
        n_confs=n_confs,
        random_seed=random_seed,
        num_threads=num_threads,
        metadata_out=metadata_out,
    )


def _actual_rotatable_bonds(setup):
    """Return unique RDKit atom-index bonds represented by Meeko tree edges."""
    from .torsion_budget import actual_rotatable_bonds

    return actual_rotatable_bonds(setup)


def _setup_atom_snapshot(setup):
    """Capture atom fields that torsion-tree rebuilding must not change."""
    from .torsion_budget import setup_atom_snapshot

    return setup_atom_snapshot(setup)


def _validate_setup_atom_invariants(before, setup, tolerance=1e-12):
    """Fail closed if Meeko flexibility rebuilding changes atom properties."""
    from .torsion_budget import validate_setup_atom_invariants

    return validate_setup_atom_invariants(
        before, setup, tolerance=tolerance, snapshot=_setup_atom_snapshot
    )


def _torsdof_from_setup(setup):
    from .torsion_budget import torsdof_from_setup

    return torsdof_from_setup(setup)


def _apply_adaptive_torsion_budget(
    setup,
    mol_3d,
    preparator,
    limit=57,
    n_confs=8,
    random_seed=42,
    num_threads=1,
):
    """Freeze low-dispersion Meeko bonds at setup level, then rebuild its tree.

    Meeko tree serials are deliberately not used as RDKit indices. The selected
    bonds come from ``rigid_body_connectivity``, whose endpoints retain the input
    RDKit atom indices. After each selection, Meeko recalculates the flexibility
    model so the final PDBQT is serialized as a valid tree rather than edited as
    text.
    """
    from .torsion_budget import apply_adaptive_torsion_budget

    return apply_adaptive_torsion_budget(
        setup,
        mol_3d,
        preparator,
        limit=limit,
        n_confs=n_confs,
        random_seed=random_seed,
        num_threads=num_threads,
        list_bonds=_actual_rotatable_bonds,
        count_torsdof=_torsdof_from_setup,
        calculate_sigma=_bond_dihedral_sigma,
    )


def _validate_pdbqt_torsion_tree(pdbqt_string):
    """Fail closed on malformed ROOT/BRANCH structure and TORSDOF drift."""
    from .pdbqt_validation import validate_pdbqt_torsion_tree

    return validate_pdbqt_torsion_tree(pdbqt_string)


def _atomic_write_text(path, text):
    from .pdbqt_validation import atomic_write_text

    return atomic_write_text(path, text)


def _freeze_lowest_sigma_bonds(*_args, **_kwargs):
    """Disabled unsafe legacy string editor retained for explicit failure."""
    from .torsion_budget import freeze_lowest_sigma_bonds

    return freeze_lowest_sigma_bonds(*_args, **_kwargs)


def smiles_to_ligand_pdbqt(
    smiles: str,
    pdbqt_path: str,
    num_confs: int = 10,
    random_seed: int = 42,
    rigid_macrocycles: bool = True,
    generated_map: Optional[str] = None,
    n_template_confs: int = 1,
    conf_out: Optional[dict] = None,
    protonate: bool = True,
    torsdof_limit: Optional[int] = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
) -> Optional[str]:
    """SMILES → 3D → meeko ligand PDBQT.

    The canonical SMILES→PDBQT path for docking. If `generated_map` is given,
    first tries Scaffold-template guided conformer generation via
    `docking.template_library.generate_conformers` (top template(s) if a bucket
    exists, ETKDG max-min otherwise). If no map is given or template generation
    fails, falls back to ``export.conformer._embed_3d`` (multi-seed ETKDGv3 +
    useRandomCoords).

    For large cyclic peptides the default flexible PDBQT (full ROOT/BRANCH
    torsion tree) is REJECTED by Uni-Dock ("Could not parse PDBQT of ligand").
    ``rigid_macrocycles=True`` (meeko's native option) freezes macrocycle ring
    bonds, collapsing the torsion tree enough for Uni-Dock to accept it.

    `protonate=True` (default) applies pH-7.4 protonation (`protonate_ph74`) to
    the SMILES up front so charged residues carry real formal charges into the
    docking electrostatics and permeability descriptors. Verified not to harm
    template matching (fingerprint is robust to the few charge/H changes; some
    peptides even hit a template they missed neutral). Set False for the raw
    neutral form.

    If `conf_out` (a dict) is given, on success it is filled with `mol_3d` (the
    RDKit Mol with the docked conformer) and `source` ("template" = a library
    template matched, "etkdg" = Scene-B unguided pool, "embed3d" = fallback). The
    RFT reward uses this to harvest de-novo conformers (source != "template") of
    well-binding peptides back into the template library. It also records
    `template_meta` (the `generate_conformers` audit, or None when no map was
    given) and `template_error` (why a template-guided conformer was not used,
    or None) so template success and ETKDG fallback are never conflated.

    `torsdof_limit` (default None = off): if set, and the rigid-macrocycle Meeko
    setup still has more than `torsdof_limit` rotatable bonds, rank its actual
    RDKit-indexed tree edges by dihedral dispersion across multiple compatible
    template conformers. If fewer than two matched templates are usable, an
    ETKDG ensemble is used with an explicit lower evidence label. The lowest-
    dispersion bonds are rigidified and Meeko rebuilds the complete tree. Set
    to 57 for AutoDock-GPU. The emitted tree is parsed and audited before atomic
    export.

    Returns None on success, error string on failure.
    """
    from .ligand_pdbqt import smiles_to_ligand_pdbqt as prepare_ligand
    from .pdbqt_validation import validate_pdbqt_torsion_tree

    return prepare_ligand(
        smiles,
        pdbqt_path,
        num_confs=num_confs,
        random_seed=random_seed,
        rigid_macrocycles=rigid_macrocycles,
        generated_map=generated_map,
        n_template_confs=n_template_confs,
        conf_out=conf_out,
        protonate=protonate,
        torsdof_limit=torsdof_limit,
        torsion_ensemble_size=torsion_ensemble_size,
        torsion_num_threads=torsion_num_threads,
        protonate_smiles=protonate_ph74,
        snapshot_setup=_setup_atom_snapshot,
        count_torsdof=_torsdof_from_setup,
        apply_budget=_apply_adaptive_torsion_budget,
        validate_invariants=_validate_setup_atom_invariants,
        validate_tree=validate_pdbqt_torsion_tree,
        write_text=_atomic_write_text,
    )


def smiles_to_ligand_pdbqt_multi(
    smiles: str,
    out_dir: str,
    n_confs: int = 3,
    generated_map: Optional[str] = None,
    rigid_macrocycles: bool = True,
) -> Tuple[List[str], Optional[str]]:
    """Compatibility facade for the retired direct multi-PDBQT path.

    The delegated implementation returns ``([], not_supported_error)``.
    Materialize a validated MOL2 ensemble and prepare each member through the
    single MOL2-to-PDBQT entrypoint instead.
    """
    from .ligand_pdbqt import smiles_to_ligand_pdbqt_multi as prepare_many

    return prepare_many(
        smiles,
        out_dir,
        n_confs=n_confs,
        generated_map=generated_map,
        rigid_macrocycles=rigid_macrocycles,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Vina Docking
# ══════════════════════════════════════════════════════════════════════════════

def run_vina(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    output_pdbqt: str,
    exhaustiveness: int = EXHAUSTIVENESS,
    num_modes: int = NUM_MODES,
) -> Tuple[Optional[float], Optional[str]]:
    """Run AutoDock Vina docking.

    Returns:
        (best_affinity_kcal_mol, error_message)
    """
    from .vina import run_vina as execute_vina

    return execute_vina(
        ligand_pdbqt,
        receptor_pdbqt,
        center,
        box_size,
        output_pdbqt,
        exhaustiveness,
        num_modes,
        find_executable=_find_vina,
        run_process=subprocess.run,
        timeout_error=subprocess.TimeoutExpired,
        parse_affinity=_parse_vina_affinity,
    )


def _parse_vina_affinity(vina_stdout: str) -> Optional[float]:
    """Parse best binding affinity from Vina stdout."""
    # Vina output format:
    #   mode |   affinity | dist from best mode
    #        | (kcal/mol) | rmsd l.b.| rmsd u.b.
    #   -----+------------+----------+----------
    #      1       -8.2       0.000       0.000

    from .vina import parse_vina_affinity

    return parse_vina_affinity(vina_stdout)


# ══════════════════════════════════════════════════════════════════════════════
# High-level API
# ══════════════════════════════════════════════════════════════════════════════

def dock_peptide(
    peptide_pdb: str,
    receptor_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    output_dir: Optional[str] = None,
    cleanup: bool = True,
    *,
    ligand_preparation_mode: str = "original_pdb",
    ligand_smiles: Optional[str] = None,
    peptide_chain_id: str = "L",
    receptor_chain_id: Optional[str] = None,
    generated_map: Optional[str] = None,
    torsdof_limit: Optional[int] = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
    torsion_audit_out: Optional[dict] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Dock a peptide (ligand) to a protein receptor using AutoDock Vina.

    Args:
        peptide_pdb: Path to peptide PDB file (ligand)
        receptor_pdb: Path to receptor PDB file (protein)
        center: (x, y, z) coordinates of docking box center in Angstroms
        box_size: (x, y, z) dimensions of docking box in Angstroms
        output_dir: Directory to save output files (temp dir if None)
        cleanup: Delete a temporary output directory after docking
        ligand_preparation_mode: ``original_pdb`` preserves the supplied ligand
            coordinates. ``audited_smiles`` explicitly generates new coordinates
            from ``ligand_smiles`` or a qualified V6 reconstruction of
            ``peptide_pdb``.
        ligand_smiles: Optional explicit chemical graph for ``audited_smiles``.
            If omitted, V6 must reconstruct a qualified graph from ``peptide_pdb``.
        peptide_chain_id: Peptide chain used by V6 reconstruction.
        generated_map: Optional MAP used for template-guided conformer generation.
        torsdof_limit: Optional adaptive torsion ceiling for ``audited_smiles``.
        torsion_ensemble_size: Template/fallback ensemble size used to rank
            torsions.
        torsion_num_threads: Threads used for the torsion-ranking ensemble.
        torsion_audit_out: Serializable preparation and torsion audit details.

    Returns:
        (best_affinity_kcal_mol, error_message)
        Affinity is negative (more negative = stronger binding).
        If error, affinity is None and error message is returned.

    Example:
        >>> affinity, err = dock_peptide(
        ...     "peptide.pdb",
        ...     "protein.pdb",
        ...     center=(10.5, 20.3, 15.2),
        ...     box_size=(30, 30, 30),
        ... )
        >>> if err:
        ...     print(f"Docking failed: {err}")
        ... else:
        ...     print(f"Binding affinity: {affinity:.2f} kcal/mol")
    """
    from .workflow import _dock_peptide_impl

    return _dock_peptide_impl(
        peptide_pdb,
        receptor_pdb,
        center,
        box_size,
        output_dir,
        cleanup,
        ligand_preparation_mode=ligand_preparation_mode,
        ligand_smiles=ligand_smiles,
        peptide_chain_id=peptide_chain_id,
        receptor_chain_id=receptor_chain_id,
        generated_map=generated_map,
        torsdof_limit=torsdof_limit,
        torsion_ensemble_size=torsion_ensemble_size,
        torsion_num_threads=torsion_num_threads,
        torsion_audit_out=torsion_audit_out,
        convert_pdb=pdb_to_pdbqt_meeko,
        prepare_smiles=smiles_to_ligand_pdbqt,
        execute_vina=run_vina,
    )


def batch_dock_peptides(
    peptide_pdb_list: List[str],
    receptor_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
) -> List[Tuple[str, Optional[float], Optional[str]]]:
    """Dock multiple peptides to the same receptor.

    Returns:
        List of (peptide_name, affinity_kcal_mol, error_message)
    """
    from .workflow import batch_dock_peptides as dock_many

    return dock_many(
        peptide_pdb_list,
        receptor_pdb,
        center,
        box_size,
        dock=dock_peptide,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Auto box center detection (convenience)
# ══════════════════════════════════════════════════════════════════════════════

def get_protein_center(pdb_path: str) -> Tuple[float, float, float]:
    """Get geometric center of a protein PDB (useful for initial box placement)."""
    from .box import get_protein_center as calculate_center

    return calculate_center(pdb_path)


def get_binding_site_center(
    pdb_path: str,
    residue_ids: List[int],
    *,
    chain_id: Optional[str] = None,
    include_hydrogens: bool = False,
) -> Tuple[float, float, float]:
    """Get center of specific residues (known binding site).

    Args:
        pdb_path: Protein PDB file
        residue_ids: List of residue numbers (PDB numbering) defining the binding site
        chain_id: Optional PDB chain identifier. Without it, matching residue
            numbers from every chain are included.
        include_hydrogens: Include H/D atoms in the geometric center.

    Returns:
        (x, y, z) center of the binding site
    """
    from .box import get_binding_site_center as calculate_center

    return calculate_center(
        pdb_path,
        residue_ids,
        chain_id=chain_id,
        include_hydrogens=include_hydrogens,
    )


if __name__ == "__main__":
    # Example usage
    print("=== Vina Wrapper Test ===")

    # Test 1: Check Vina executable
    vina_exe = _find_vina()
    if vina_exe:
        print(f"✓ Vina found: {vina_exe}")
    else:
        print("✗ Vina not found. Set VINA_BIN, install vina, or bundle it.")
        print("  Download from: https://github.com/ccsb-scripps/AutoDock-Vina/releases")
        exit(1)

    # Test 2: Check Meeko
    try:
        import meeko
        print(f"✓ Meeko installed: {meeko.__version__}")
    except ImportError:
        print("✗ Meeko not installed")
        print("  Install with: pip install meeko")
        exit(1)

    print("\nWrapper ready. Example usage:")
    print("""
    from docking.vina_wrapper import dock_peptide

    affinity, err = dock_peptide(
        peptide_pdb="path/to/peptide.pdb",
        receptor_pdb="path/to/protein.pdb",
        center=(10.5, 20.3, 15.2),  # box center coordinates
        box_size=(25, 25, 25),       # box dimensions
    )

    if err:
        print(f"Error: {err}")
    else:
        print(f"Affinity: {affinity:.2f} kcal/mol")
    """)
