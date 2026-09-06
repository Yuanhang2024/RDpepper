"""Ligand PDBQT compatibility facades and shared serialization helpers."""

from pathlib import Path
from typing import List, Optional, Tuple

from .pdbqt_validation import (
    _AUTODOCK4_TYPES,
    atomic_write_text,
    validate_pdbqt_torsion_tree,
)
from .protonation import protonate_ph74
from .receptor_pdbqt import autodock_atom_type
from .torsion_budget import (
    apply_adaptive_torsion_budget,
    setup_atom_snapshot,
    torsdof_from_setup,
    validate_setup_atom_invariants,
)


def _writer_text(result):
    """Normalize Meeko writer return contracts across supported versions."""
    if isinstance(result, str):
        if not result.strip():
            raise RuntimeError("PDBQT writer returned empty text")
        return result
    if isinstance(result, tuple) and len(result) == 3:
        pdbqt_string, is_ok, error_message = result
        if not is_ok:
            raise RuntimeError(f"PDBQT write failed: {error_message}")
        if not isinstance(pdbqt_string, str) or not pdbqt_string.strip():
            raise RuntimeError("PDBQT writer returned empty text")
        return pdbqt_string
    raise RuntimeError("unsupported Meeko PDBQT writer return contract")


def _unsupported_ligand_elements(molecule) -> list[str]:
    """Return element symbols that cannot be represented by AutoDock4."""
    unsupported = {
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
        if autodock_atom_type(atom) not in _AUTODOCK4_TYPES
        and not (
            atom.GetAtomicNum() == 1 and autodock_atom_type(atom) == ""
        )
    }
    return sorted(unsupported)


def _unsupported_ligand_error(molecule) -> str | None:
    elements = _unsupported_ligand_elements(molecule)
    if not elements:
        return None
    return (
        "not_supported: unsupported ligand element(s): "
        + ", ".join(elements)
    )


def _pdbqt_connectivity_from_molecule(molecule, pdbqt_string):
    """Map RDKit bonds to emitted PDBQT serials via Meeko's INDEX MAP."""
    pairs = []
    for line in pdbqt_string.splitlines():
        if line.startswith("REMARK INDEX MAP "):
            values = line.removeprefix("REMARK INDEX MAP ").split()
            if len(values) % 2:
                raise RuntimeError("malformed PDBQT INDEX MAP")
            try:
                pairs.extend(
                    (int(values[index]), int(values[index + 1]))
                    for index in range(0, len(values), 2)
                )
            except ValueError:
                raise RuntimeError(
                    "malformed PDBQT INDEX MAP"
                ) from None
    if not pairs:
        raise RuntimeError("PDBQT is missing Meeko INDEX MAP")
    serial_by_rdkit_index = {}
    for one_based_index, serial in pairs:
        rdkit_index = one_based_index - 1
        if (
            serial in serial_by_rdkit_index.values()
            or rdkit_index in serial_by_rdkit_index
        ):
            raise RuntimeError("PDBQT INDEX MAP is not one-to-one")
        if rdkit_index < 0 or rdkit_index >= molecule.GetNumAtoms():
            raise RuntimeError(
                "PDBQT INDEX MAP references an unknown source atom"
            )
        serial_by_rdkit_index[rdkit_index] = serial
    connectivity = []
    for bond in molecule.GetBonds():
        left = serial_by_rdkit_index.get(bond.GetBeginAtomIdx())
        right = serial_by_rdkit_index.get(bond.GetEndAtomIdx())
        if left is not None and right is not None:
            connectivity.append((left, right))
    return connectivity


def _preferred_ligand_chain(pdb_path: str | Path) -> str:
    chains = []
    for line in Path(pdb_path).read_text(
        encoding="ascii", errors="replace"
    ).splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21:22].strip()
        if chain and chain not in chains:
            chains.append(chain)
    if "L" in chains:
        return "L"
    return chains[0] if chains else "L"


def pdb_to_ligand_pdbqt(pdb_path: str, pdbqt_path: str) -> Optional[str]:
    """Compatibility facade through source-bound validated MOL2."""
    try:
        from ..application import prepare_ligand_pdbqt_from_pdb

        result = prepare_ligand_pdbqt_from_pdb(
            pdb_path,
            pdbqt_path,
            chain_id=_preferred_ligand_chain(pdb_path),
        )
    except Exception as exc:
        return (
            "failed: validated-MOL2 ligand preparation failed: "
            f"{type(exc).__name__}: {exc}"
        )
    destination = Path(pdbqt_path)
    if (
        result.get("status") == "success"
        and destination.is_file()
        and destination.stat().st_size > 0
    ):
        return None
    status = str(result.get("status") or "failed")
    error = str(
        result.get("error")
        or "validated-MOL2 ligand preparation produced no PDBQT"
    )
    return error if error.startswith(f"{status}:") else f"{status}: {error}"


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
    *,
    protonate_smiles=protonate_ph74,
    snapshot_setup=setup_atom_snapshot,
    count_torsdof=torsdof_from_setup,
    apply_budget=apply_adaptive_torsion_budget,
    validate_invariants=validate_setup_atom_invariants,
    validate_tree=validate_pdbqt_torsion_tree,
    write_text=atomic_write_text,
) -> Optional[str]:
    """Compatibility facade through a generated validated parent MOL2."""
    from ..application import prepare_ligand_pdbqt as prepare_compat

    result = prepare_compat(
        smiles,
        pdbqt_path,
        generated_map=generated_map,
        num_confs=num_confs,
        random_seed=random_seed,
        rigid_macrocycles=rigid_macrocycles,
        protonate=protonate,
        torsdof_limit=torsdof_limit,
        torsion_ensemble_size=torsion_ensemble_size,
        torsion_num_threads=torsion_num_threads,
    )
    audit = (result.get("data") or {}).get("audit") or {}
    if conf_out is not None:
        conf_out.clear()
        conf_out.update(audit)
    destination = Path(pdbqt_path)
    if (
        result.get("status") == "success"
        and destination.is_file()
        and destination.stat().st_size > 0
    ):
        return None
    return str(
        result.get("error")
        or (result.get("data") or {}).get(
            "requested_artifact_status"
        )
        or result.get("status")
        or "failed"
    )


def smiles_to_ligand_pdbqt_multi(
    smiles: str,
    out_dir: str,
    n_confs: int = 3,
    generated_map: Optional[str] = None,
    rigid_macrocycles: bool = True,
    random_seed: int = 42,
    protonate: bool = True,
    audit_out: Optional[dict] = None,
) -> Tuple[List[str], Optional[str]]:
    """Reject direct multi-output PDBQT; conformers are evidence only."""
    if audit_out is not None:
        audit_out.clear()
        audit_out.update({
            "status": "not_supported",
            "reason": "V5_REQUIRES_VALIDATED_MOL2_ENSEMBLE_PIPELINE",
        })
    return [], (
        "not_supported: V5 does not support direct multi-conformer PDBQT "
        "output; materialize a validated MOL2 ensemble first"
    )
