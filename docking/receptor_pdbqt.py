"""Rigid receptor preparation and AutoDock atom typing."""

import math
import os
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from .pdbqt_validation import _AUTODOCK4_TYPES, atomic_write_text


def _clear_prior_pdbqt_output(
    input_path: str,
    output_path: Path,
    *,
    error_prefix: str,
) -> Optional[str]:
    """Clear an old PDBQT only when it cannot alias the input structure.

    PDB and PDBQT paths are user-controlled and may be the same path, a
    symlink, or hard links to one inode.  Checking those relationships before
    unlinking prevents a failed conversion from deleting its source.
    """
    source = Path(input_path)
    try:
        if source.resolve(strict=False) == output_path.resolve(strict=False):
            return f"{error_prefix}: input and output paths refer to the same file"
        if source.exists() and output_path.exists() and os.path.samefile(
            source, output_path
        ):
            return f"{error_prefix}: input and output are hard-link aliases"
    except (OSError, RuntimeError) as exc:
        return f"{error_prefix}: cannot verify input/output paths: {exc}"
    try:
        output_path.unlink(missing_ok=True)
    except OSError as exc:
        return f"{error_prefix}: {exc}"
    return None


def _is_amide_like_nitrogen(atom) -> bool:
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() not in {"C", "S", "P"}:
            continue
        for other in neighbor.GetNeighbors():
            if other.GetIdx() == atom.GetIdx():
                continue
            bond = atom.GetOwningMol().GetBondBetweenAtoms(
                neighbor.GetIdx(), other.GetIdx()
            )
            if (
                bond is not None
                and bond.GetBondType() == Chem.BondType.DOUBLE
                and other.GetSymbol() in {"O", "S", "N"}
            ):
                return True
    return False


def autodock_atom_type(atom) -> str:
    """Map an RDKit atom to the receptor AutoDock4 atom type."""
    symbol = atom.GetSymbol()
    has_hydrogen = atom.GetTotalNumHs() > 0 or any(
        neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
    )
    if symbol == "C":
        return "A" if atom.GetIsAromatic() else "C"
    if symbol == "N":
        return (
            "N"
            if has_hydrogen
            or atom.GetFormalCharge() > 0
            or _is_amide_like_nitrogen(atom)
            else "NA"
        )
    if symbol == "O":
        return "OA"
    if symbol == "S":
        return "S" if has_hydrogen else "SA"
    if symbol == "H":
        neighbors = atom.GetNeighbors()
        if neighbors and neighbors[0].GetSymbol() in ("N", "O", "S"):
            return "HS" if neighbors[0].GetSymbol() == "S" else "HD"
        return ""
    return symbol


def pdb_to_receptor_pdbqt(
    pdb_path: str,
    pdbqt_path: str,
    *,
    atom_type=autodock_atom_type,
) -> Optional[str]:
    """Convert a receptor PDB to a rigid PDBQT without a torsion tree."""
    output_path = Path(pdbqt_path)
    clear_error = _clear_prior_pdbqt_output(
        pdb_path,
        output_path,
        error_prefix="cannot clear prior receptor PDBQT output",
    )
    if clear_error:
        return clear_error
    try:
        molecule = Chem.MolFromPDBFile(
            pdb_path, removeHs=False, sanitize=False
        )
        if molecule is None:
            return f"RDKit failed to parse receptor {pdb_path}"
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            pass
        # Classify unsupported AutoDock elements before charge assignment.
        # RDKit may emit a non-finite Gasteiger charge for elements such as Se;
        # that is a capability boundary, not an undifferentiated conversion
        # failure.
        unsupported_types = sorted(
            {
                atom_type(atom)
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() != 1
                and atom_type(atom) not in _AUTODOCK4_TYPES
            }
        )
        if unsupported_types:
            return (
                "not_supported: receptor atom type is unsupported by "
                "AutoDock4: "
                + ", ".join(
                    repr(atom_type_name) for atom_type_name in unsupported_types
                )
            )
        try:
            AllChem.ComputeGasteigerCharges(molecule)
        except Exception as exc:
            return (
                "not_supported: receptor Gasteiger charge assignment failed: "
                f"{type(exc).__name__}: {exc}"
            )
        conformer = (
            molecule.GetConformer() if molecule.GetNumConformers() else None
        )
        if conformer is None:
            return "Receptor PDB has no 3D coordinates"

        charges: dict[int, float] = {}
        for atom in molecule.GetAtoms():
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
            except Exception as exc:
                return f"receptor Gasteiger charge is unavailable: {exc}"
            if not math.isfinite(charge):
                return "receptor Gasteiger charge is non-finite"
            charges[atom.GetIdx()] = charge
        for atom in molecule.GetAtoms():
            if atom_type(atom) != "" or atom.GetAtomicNum() != 1:
                continue
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                return "omitted receptor hydrogen has no unique heavy-atom parent"
            parent = neighbors[0].GetIdx()
            charges[parent] += charges[atom.GetIdx()]

        lines = ["REMARK  rigid receptor PDBQT (cycpep_master)"]
        serial = 0
        for atom in molecule.GetAtoms():
            autodock_type = atom_type(atom)
            if autodock_type == "":
                continue
            if autodock_type not in _AUTODOCK4_TYPES:
                return (
                    "not_supported: receptor atom type is unsupported by "
                    f"AutoDock4: {autodock_type!r}"
                )
            serial += 1
            position = conformer.GetAtomPosition(atom.GetIdx())
            if not all(math.isfinite(value) for value in (
                position.x, position.y, position.z
            )):
                return "receptor PDB contains non-finite coordinates"
            info = atom.GetPDBResidueInfo()
            name = (info.GetName() if info else atom.GetSymbol()).strip()[:4]
            residue_name = (
                info.GetResidueName() if info else "UNK"
            ).strip()[:3]
            chain = (info.GetChainId() if info else "A") or "A"
            residue_number = info.GetResidueNumber() if info else 1
            charge = charges[atom.GetIdx()]
            lines.append(
                "ATOM  %5d %-4s %-3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f    %6.3f %-2s"
                % (
                    serial % 100000,
                    name.ljust(4)[:4],
                    residue_name,
                    chain[:1],
                    residue_number % 10000,
                    position.x,
                    position.y,
                    position.z,
                    1.0,
                    0.0,
                    charge,
                    autodock_type,
                )
            )
        lines.append("TER")
        if serial == 0:
            return "no atoms written to receptor PDBQT"
        atomic_write_text(output_path, "\n".join(lines) + "\n")
        return None
    except Exception as exc:
        return f"receptor PDBQT conversion failed: {exc}"
