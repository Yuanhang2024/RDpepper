"""Rigid receptor preparation and AutoDock atom typing.

``pdb_to_receptor_pdbqt`` keeps its historical RDKit-default behavior
unless the caller passes ``ph``.  With ``ph`` set (supported range
(0, 10.5]), an explicit residue-state protonation policy is applied with
provenance REMARKs; it can never silently degrade to the unprotonated
default: an unavailable policy backend or an unsupported pH fails the
conversion.

On the default (``ph=None``) path, sanitization is never swallowed: an
undeclared spurious proximity edge at a valence-invalid
standard-amino-acid atom is repaired (recorded in REMARKs) when exactly
one such candidate exists, declared covalent records
(LINK/SSBOND/CONECT), peptide and disulfide links are preserved, and
anything unrepairable or ambiguous fails closed naming the atom.  The
optional ``ph`` policy keeps its historical internal behavior unchanged.
Genuine monoatomic Na/K/Mg/Ca/Zn/Ni/Cl ions carry formal ionic charges
(explicit input declarations take precedence over the model
oxidation-state assumptions recorded in REMARKs) so their Gasteiger
charges stay finite; coordination is represented as non-bonded for the
rigid docking representation without asserting the physical interaction
is absent.  Variable-valent metals and other PEOE-parameterless contexts
remain typed ``not_supported`` instead of being guessed.
"""

import math
import os
import re
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
            # Vina accepts HD for polar hydrogens but does not recognize HS.
            return "HD"
        return ""
    return symbol


def _label_added_hydrogens(molecule) -> None:
    """Avoid AddHs(addResidueInfo=True), which can hang with existing hydrogens."""
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetPDBResidueInfo() is not None:
            continue
        parents = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() != 1
        ]
        if len(parents) != 1:
            continue
        parent_info = parents[0].GetPDBResidueInfo()
        if parent_info is None:
            continue
        atom.SetPDBResidueInfo(
            Chem.AtomPDBResidueInfo(
                "H",
                parent_info.GetSerialNumber(),
                parent_info.GetAltLoc(),
                parent_info.GetResidueName(),
                parent_info.GetResidueNumber(),
                parent_info.GetChainId(),
                parent_info.GetInsertionCode(),
                parent_info.GetOccupancy(),
                parent_info.GetTempFactor(),
                parent_info.GetIsHeteroAtom(),
                parent_info.GetSecondaryStructure(),
                parent_info.GetSegmentNumber(),
            )
        )


# --- Sanitize repair and monoatomic ion charge policy ---------------------
#
# Non-finite receptor Gasteiger charges have two causes addressed here:
#
# 1. A geometrically distorted input can gain a spurious undeclared
#    inter-residue proximity edge (for example the C(i)--CD(i+1) contact of
#    a distorted proline), over-valencing a backbone carbon; the failed
#    sanitization then turns every charge into NaN.  Only an undeclared,
#    non-canonical, standard-amino-acid inter-residue edge at the
#    valence-invalid atom is repairable, and only when exactly one such
#    candidate exists; LINK/SSBOND/CONECT-declared bonds, peptide C--N
#    links, and CYS SG--SG disulfides are always preserved, and everything
#    else fails closed naming the atom.
# 2. PEOE/Gasteiger has no parameters for coordinated metals, so their NaN
#    spreads over the 12-iteration bond radius.  A genuine monoatomic ion
#    residue of an element with a single common biological oxidation state
#    (Na, K, Mg, Ca, Zn, Ni) is detached from its coordination-only bonds
#    (a rigid-docking representation choice recorded in REMARKs; no claim
#    the physical coordination is absent) and carries its formal ionic
#    charge: the input's explicit declaration when present, otherwise the
#    model value.  The assumed oxidation states (notably Ni) are model
#    assumptions, not experimental determination.  Variable-valent metals
#    and other parameterless contexts stay unsupported, never guessed.

_STANDARD_AMINO_ACIDS = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR "
    "TRP TYR VAL".split()
)
_MONOATOMIC_ION_FORMAL_CHARGES = {
    "Na": 1,
    "K": 1,
    "Mg": 2,
    "Ca": 2,
    "Zn": 2,
    "Ni": 2,
    "Cl": -1,
}
_MONOATOMIC_ION_RESIDUE_NAMES = {
    "Na": frozenset({"NA"}),
    "K": frozenset({"K"}),
    "Mg": frozenset({"MG"}),
    "Ca": frozenset({"CA"}),
    "Zn": frozenset({"ZN"}),
    "Ni": frozenset({"NI"}),
    "Cl": frozenset({"CL"}),
}
# Receptor-side ion atom types, verified against the bundled Vina 1.2.7
# binary: "Na" (sodium) and "K" (potassium) parse as distinct receptor atom
# types and differ from "NA" (the nitrogen acceptor).  They are
# intentionally NOT added to the shared ligand type whitelist in
# pdbqt_validation, which keeps ligand torsion-tree semantics unchanged.
_RECEPTOR_ION_ATOM_TYPES = frozenset({"Na", "K"})
# Elements whose oxidation state must not be assumed for a receptor.
_VARIABLE_VALENT_ELEMENTS = frozenset({"Fe", "Cu", "Mn", "Co"})
# Elements covered by the classic PEOE parameter table in ordinary organic
# bonding contexts; non-finite charges on these indicate an unusual context
# (for example neutral tetravalent phosphate P) that needs a chemistry
# reference rather than a guessed charge.
_ORGANIC_PEOE_ELEMENTS = frozenset(
    {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
)


def _atom_residue_key(atom):
    info = atom.GetPDBResidueInfo()
    if info is None:
        return None
    return (
        (info.GetChainId() or "A").strip(),
        info.GetResidueNumber(),
        info.GetInsertionCode().strip(),
        info.GetResidueName().strip(),
    )


def _atom_site(atom) -> str:
    info = atom.GetPDBResidueInfo()
    if info is None:
        return f"#{atom.GetIdx()}<{atom.GetSymbol()}>"
    return (
        f"{info.GetName().strip()} {info.GetResidueName().strip()} "
        f"{(info.GetChainId() or 'A').strip()}{info.GetResidueNumber()}"
        f"{info.GetInsertionCode().strip()}"
    )


def _link_site_key(chain, resseq, icode, resname, name):
    return (
        (chain or "A").strip(),
        int(resseq),
        icode.strip(),
        resname.strip(),
        name.strip(),
    )


def _read_declared_pdb_edges(pdb_path: str) -> dict:
    """Collect the covalent bonds the input PDB declares explicitly.

    Returns ``{"conect", "link", "ssbond", "ok"}``.  ``ok`` is False when
    the records cannot be read reliably; callers then treat every edge as
    declared (repair is disabled) instead of risking deletion of a real
    crosslink.
    """
    declared = {"conect": set(), "link": set(), "ssbond": set(), "ok": True}
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        declared["ok"] = False
        return declared
    for line in lines:
        record = line[:6].strip()
        try:
            if record == "CONECT":
                serials = [
                    int(line[start:end])
                    for start, end in (
                        (6, 11), (11, 16), (16, 21), (21, 26), (26, 31),
                    )
                    if line[start:end].strip()
                ]
                for partner in serials[1:]:
                    declared["conect"].add(frozenset((serials[0], partner)))
            elif record == "LINK" and len(line) >= 57:
                left = _link_site_key(
                    chain=line[21],
                    resseq=line[22:26],
                    icode=line[26],
                    resname=line[17:20],
                    name=line[12:16],
                )
                right = _link_site_key(
                    chain=line[51],
                    resseq=line[52:56],
                    icode=line[56],
                    resname=line[47:50],
                    name=line[42:46],
                )
                declared["link"].add(frozenset((left, right)))
            elif record == "SSBOND" and len(line) >= 35:
                left = (
                    (line[15] or "A").strip(),
                    int(line[17:21]),
                    line[21].strip(),
                )
                right = (
                    (line[29] or "A").strip(),
                    int(line[31:35]),
                    line[35:36].strip(),
                )
                declared["ssbond"].add(frozenset((left, right)))
        except (ValueError, IndexError):
            declared["ok"] = False
    return declared


def _bond_declared_in_input(atom, other, declared) -> bool:
    info = atom.GetPDBResidueInfo()
    other_info = other.GetPDBResidueInfo()
    if info is None or other_info is None:
        return True  # cannot verify: preserve conservatively
    serial_pair = frozenset((info.GetSerialNumber(), other_info.GetSerialNumber()))
    if serial_pair in declared["conect"]:
        return True
    link_pair = frozenset(
        (
            (
                (info.GetChainId() or "A").strip(),
                info.GetResidueNumber(),
                info.GetInsertionCode().strip(),
                info.GetResidueName().strip(),
                info.GetName().strip(),
            ),
            (
                (other_info.GetChainId() or "A").strip(),
                other_info.GetResidueNumber(),
                other_info.GetInsertionCode().strip(),
                other_info.GetResidueName().strip(),
                other_info.GetName().strip(),
            ),
        )
    )
    if link_pair in declared["link"]:
        return True
    if (
        info.GetName().strip() == "SG"
        and other_info.GetName().strip() == "SG"
        and info.GetResidueName().strip() == "CYS"
        and other_info.GetResidueName().strip() == "CYS"
    ):
        residues = frozenset(
            (
                (
                    (info.GetChainId() or "A").strip(),
                    info.GetResidueNumber(),
                    info.GetInsertionCode().strip(),
                ),
                (
                    (other_info.GetChainId() or "A").strip(),
                    other_info.GetResidueNumber(),
                    other_info.GetInsertionCode().strip(),
                ),
            )
        )
        if residues in declared["ssbond"]:
            return True
    return False


def _canonical_inter_residue_link(atom, other) -> bool:
    info = atom.GetPDBResidueInfo()
    other_info = other.GetPDBResidueInfo()
    if info is None or other_info is None:
        return True  # cannot verify: treat as canonical (preserve)
    names = {info.GetName().strip(), other_info.GetName().strip()}
    resnames = {info.GetResidueName().strip(), other_info.GetResidueName().strip()}
    if names == {"C", "N"} and resnames <= _STANDARD_AMINO_ACIDS:
        return True
    if names == {"SG", "SG"} and resnames == {"CYS"}:
        return True
    return False


def _is_spurious_proximity_edge(atom, other, declared) -> bool:
    """Whether an edge is an undeclared non-canonical standard-AA link."""
    for candidate in (atom, other):
        info = candidate.GetPDBResidueInfo()
        if (
            info is None
            or info.GetResidueName().strip() not in _STANDARD_AMINO_ACIDS
        ):
            return False
    if _atom_residue_key(atom) == _atom_residue_key(other):
        return False
    if _canonical_inter_residue_link(atom, other):
        return False
    if _bond_declared_in_input(atom, other, declared):
        return False
    return True


def _sanitize_exception_atom_index(exc):
    """Atom index of a MolSanitizeException, or None when unavailable.

    Some RDKit builds do not expose ``GetAtomIdx`` on the raised Python
    wrapper; the message's ``atom # <n>`` is the fallback.
    """
    getter = getattr(exc, "GetAtomIdx", None)
    if getter is not None:
        try:
            value = getter()
        except Exception:
            value = None
        if isinstance(value, int) and value >= 0:
            return value
    match = re.search(r"atom # (\d+)", str(exc))
    return int(match.group(1)) if match else None


def _sanitize_with_spurious_edge_repair(molecule, declared):
    """Sanitize, repairing only a single undeclared spurious proximity edge.

    Returns ``(molecule, error, repaired_edges)``.  A repair candidate is an
    edge of ``_is_spurious_proximity_edge`` at the atom named by the valence
    exception.  Exactly one candidate is required: multiple candidates fail
    closed as ambiguous instead of deleting edges to force a pass, and no
    candidate fails closed naming the atom.  Sanitization failures are never
    swallowed on this path.
    """
    repaired = []
    while True:
        try:
            Chem.SanitizeMol(molecule)
            return molecule, None, repaired
        except Chem.rdchem.MolSanitizeException as exc:
            index = _sanitize_exception_atom_index(exc)
            label = "unknown atom"
            candidates = []
            if index is not None and 0 <= index < molecule.GetNumAtoms():
                atom = molecule.GetAtomWithIdx(index)
                label = _atom_site(atom)
                if declared.get("ok"):
                    for bond in atom.GetBonds():
                        other = bond.GetOtherAtom(atom)
                        if _is_spurious_proximity_edge(atom, other, declared):
                            candidates.append((atom, other))
            if len(candidates) > 1:
                listed = "; ".join(
                    f"{_atom_site(a)} - {_atom_site(b)}" for a, b in candidates
                )
                return molecule, (
                    f"receptor PDB cannot be sanitized: {type(exc).__name__}: "
                    f"{exc} (atom {label}); {len(candidates)} undeclared "
                    f"spurious inter-residue edges meet this atom [{listed}]; "
                    "refusing to choose among multiple deletions"
                ), repaired
            if not candidates:
                return molecule, (
                    f"receptor PDB cannot be sanitized: {type(exc).__name__}: "
                    f"{exc} (atom {label}); no undeclared spurious "
                    "inter-residue edge is repairable at this atom; explicit "
                    "covalent declarations (LINK/SSBOND/CONECT), peptide and "
                    "disulfide links are preserved"
                ), repaired
            atom, other = candidates[0]
            editor = Chem.RWMol(molecule)
            editor.RemoveBond(atom.GetIdx(), other.GetIdx())
            repaired.append((_atom_site(atom), _atom_site(other)))
            molecule = editor.GetMol()
        except Exception as exc:
            return molecule, (
                f"receptor PDB cannot be sanitized: {type(exc).__name__}: {exc}"
            ), repaired


def _apply_monoatomic_ion_policy(molecule):
    """Assign formal ionic charges to genuine monoatomic receptor ions.

    Returns ``(molecule, error, remarks)``.  A genuine ion is a
    single-atom residue whose element matches and whose residue name AND
    atom name are both the accepted label for that element (for example
    residue/atom ``ZN`` of Zn), which rules out unrelated single-atom
    residues.  The formal charge is the input's explicit declaration when
    present, otherwise the model oxidation state; the assumed states are
    recorded in REMARKs as model assumptions, not experimental
    determination.  Coordination bonds are removed for the rigid docking
    representation only and each removed contact is recorded; no physical
    interaction is asserted absent.

    Ordering limitation: sanitization (with edge repair) runs before this
    policy, so a metal whose parsed valence itself fails sanitization would
    fail closed before detachment could help.  No such case occurs in the
    observed inputs; widening the policy is deliberately left undone.
    """
    residues: dict = {}
    for atom in molecule.GetAtoms():
        key = _atom_residue_key(atom)
        if key is not None:
            residues.setdefault(key, []).append(atom)
    editor = Chem.RWMol(molecule)
    records = []
    for key in sorted(residues, key=lambda item: (item[0], item[1], item[2], item[3])):
        atoms = residues[key]
        if len(atoms) != 1:
            continue
        atom = atoms[0]
        model_charge = _MONOATOMIC_ION_FORMAL_CHARGES.get(atom.GetSymbol())
        if model_charge is None:
            continue
        accepted = _MONOATOMIC_ION_RESIDUE_NAMES[atom.GetSymbol()]
        info = atom.GetPDBResidueInfo()
        if key[3] not in accepted or info.GetName().strip() not in accepted:
            continue
        explicit = atom.GetFormalCharge()
        charge = explicit if explicit else model_charge
        source = (
            "input explicit formal charge"
            if explicit
            else "model oxidation-state assumption, not experimentally determined"
        )
        contacts = [_atom_site(neighbor) for neighbor in atom.GetNeighbors()]
        for neighbor in list(atom.GetNeighbors()):
            editor.RemoveBond(atom.GetIdx(), neighbor.GetIdx())
        target = editor.GetAtomWithIdx(atom.GetIdx())
        target.SetFormalCharge(charge)
        target.UpdatePropertyCache(strict=False)
        records.append((_atom_site(atom), charge, source, contacts))
    if not records:
        return molecule, None, []
    molecule = editor.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        index = _sanitize_exception_atom_index(exc)
        site = "unknown atom"
        if index is not None and 0 <= index < molecule.GetNumAtoms():
            site = _atom_site(molecule.GetAtomWithIdx(index))
        return molecule, (
            "receptor PDB cannot be sanitized after monoatomic ion charge "
            f"assignment: {type(exc).__name__}: {exc} (atom {site}); an "
            "input-declared ion formal charge is not chemically "
            "representable"
        ), []
    remarks = [
        "REMARK  ion charges: monoatomic Na/K/Mg/Ca/Zn/Ni/Cl ions carry formal "
        "ionic charges (+1/+1/+2/+2/+2/+2/-1); Gasteiger/PEOE has no parameters "
        "for coordinated metals; oxidation states are explicit model "
        "assumptions (Ni oxidation is NOT experimentally determined here)",
        "REMARK  WARNING metal coordination: default-valence hydrogen completion "
        "after non-bonded representation can protonate coordinating groups; "
        "site-specific protonation and coordination chemistry require review"
    ]
    for site, charge, source, contacts in records:
        remark = f"REMARK  ion {site}: formal charge {charge:+d} ({source})"
        if contacts:
            listed = ", ".join(contacts[:4]) + ("..." if len(contacts) > 4 else "")
            remark += (
                f"; {len(contacts)} coordination contact(s) [{listed}] "
                "represented as non-bonded in this rigid docking "
                "representation; no assertion that the physical interaction "
                "is absent"
            )
        remarks.append(remark)
    return molecule, None, remarks


def _nonfinite_charge_error(atoms) -> str:
    """Typed error for non-finite Gasteiger charges, naming the causes."""
    groups: dict = {}
    for atom in atoms:
        info = atom.GetPDBResidueInfo()
        resname = info.GetResidueName().strip() if info else "UNK"
        groups.setdefault((atom.GetSymbol(), resname), []).append(atom)
    described = []
    for index, ((symbol, resname), members) in enumerate(
        sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        if index == 6:
            described.append(f"... and {len(groups) - 6} more element/residue groups")
            break
        described.append(
            f"{symbol} in {resname} at {_atom_site(members[0])} x{len(members)}"
        )
    reasons = []
    root_symbols = sorted(
        {symbol for symbol, _ in groups if symbol not in _ORGANIC_PEOE_ELEMENTS}
    )
    if root_symbols:
        variable = [
            symbol
            for symbol in root_symbols
            if symbol in _VARIABLE_VALENT_ELEMENTS
        ]
        fixed = [
            symbol
            for symbol in root_symbols
            if symbol not in _VARIABLE_VALENT_ELEMENTS
        ]
        if fixed:
            reasons.append(
                "no PEOE charge parameters for element " + "/".join(fixed)
            )
        if variable:
            reasons.append(
                "variable oxidation state for "
                + "/".join(variable)
                + "; assigning a formal charge requires a chemistry reference"
            )
    organic_symbols = sorted(
        {symbol for symbol, _ in groups if symbol in _ORGANIC_PEOE_ELEMENTS}
    )
    if organic_symbols:
        reasons.append(
            "bonding context without PEOE parameters for "
            + "/".join(organic_symbols)
            + " (for example neutral tetravalent phosphate P); the charge "
            "model requires a chemistry reference"
        )
    return (
        "not_supported: receptor Gasteiger charge is non-finite for "
        + "; ".join(described)
        + ("; " + "; ".join(reasons) if reasons else "")
        + "; refusing to assign a guessed charge"
    )


# --- Optional explicit-pH protonation policy ----------------------------
#
# ``ph=None`` (default) is behavior-compatible with the historical
# converter: RDKit template-valence hydrogens plus the "pH not assigned"
# warning.
#
# ``ph=<float>`` applies a residue-state template policy:
#   * ASP/GLU side chains and C/OXT terminal carboxylates are anionic
#     (deprotonated, formal charge -1 on one oxygen) when pH is above the
#     model pKa and neutral acids otherwise;
#   * LYS (ammonium) and ARG (guanidinium, net +1 placed only on the C=N
#     nitrogen) are cationic; the accepted range stops at the LYS constant
#     (10.5) because neutral Lys/Arg transitions are not assigned -- a
#     higher pH fails closed instead of pretending an arbitrary
#     ionization state was honored;
#   * HIS is never auto-assigned: explicit input hydrogens are the user
#     state, and their absence is reported as an undetermined microstate.
#
# The thresholds are generic textbook constants, not per-residue pKa
# predictions, and the output REMARKs say so.  OpenBabel's
# ``AddHydrogens(..., correctForPH=True)`` was evaluated for this path and
# not adopted: its PDB residue bookkeeping (renumbering/renaming on pocket
# receptors) conflicts with the exact heavy-atom identity requirement
# enforced here.

_PROTONATION_POLICY_NAME = "residue_template"
_ASP_MODEL_PKA = 3.9
_GLU_MODEL_PKA = 4.1
_LYS_MODEL_PKA = 10.5
_ARG_MODEL_PKA = 12.5
_CTERM_MODEL_PKA = 4.1  # C/OXT carboxylate shares the GLU constant
_HIS_RING_NITROGENS = ("ND1", "NE2")


def _load_protonation_policy_backend():
    """Resolve the pH policy engine, raising when it is unavailable.

    Routing ``ph`` requests through this loader guarantees fail-closed
    behavior: an unavailable engine must fail the conversion instead of
    silently emitting the unprotonated neutral default.
    """
    if not all(
        (
            hasattr(Chem, "RWMol"),
            hasattr(Chem, "AddHs"),
            hasattr(Chem.Atom, "SetFormalCharge"),
        )
    ):
        raise ImportError("RDKit molecule-editing APIs are unavailable")
    return _apply_residue_template_protonation


def _group_atoms_by_residue(molecule) -> dict:
    """Map (chain, resnum, icode, resname) -> {PDB atom name -> atom}."""
    groups: dict = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (
            (info.GetChainId() or "A").strip(),
            info.GetResidueNumber(),
            info.GetInsertionCode(),
            info.GetResidueName().strip(),
        )
        groups.setdefault(key, {})[info.GetName().strip()] = atom
    return groups


def _is_hetero_residue(atoms: dict) -> bool:
    return any(
        atom.GetPDBResidueInfo().GetIsHeteroAtom() for atom in atoms.values()
    )


def _pick_carboxylate_target(carbon, oxygens):
    """Choose the oxygen that carries the -1.

    The hydroxyl-style oxygen is preferred: the one with an explicit H if
    the input was a neutral acid, else the non-double-bonded C-O.
    """
    best = None
    best_score = -1
    for oxygen in oxygens:
        score = 0
        if any(
            neighbor.GetAtomicNum() == 1 for neighbor in oxygen.GetNeighbors()
        ):
            score += 2
        bond = oxygen.GetOwningMol().GetBondBetweenAtoms(
            carbon.GetIdx(), oxygen.GetIdx()
        )
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            score += 1
        if score > best_score:
            best, best_score = oxygen, score
    return best


def _apply_residue_template_protonation(molecule, ph: float):
    """Set the standard ionization states for ``ph``.

    Returns ``(molecule, remarks, sites)`` where ``sites`` records the
    enforced per-atom state so hydrogens that RDKit templates add later
    cannot silently undo the policy.
    """
    groups = _group_atoms_by_residue(molecule)
    sites = []
    strip_hydrogens = []
    stats = {
        "acid": 0,
        "acid_neutral": 0,
        "cterm": 0,
        "lys": 0,
        "arg": 0,
        "his": 0,
        "his_h": 0,
    }
    for key in sorted(groups):
        resname = key[3]
        atoms = groups[key]
        if _is_hetero_residue(atoms):
            continue
        if resname in ("ASP", "GLU"):
            carbon_name = "CG" if resname == "ASP" else "CD"
            o_names = ("OD1", "OD2") if resname == "ASP" else ("OE1", "OE2")
            model_pka = _ASP_MODEL_PKA if resname == "ASP" else _GLU_MODEL_PKA
            carbon = atoms.get(carbon_name)
            oxygens = [atoms.get(name) for name in o_names]
            complete = (
                carbon is not None
                and carbon.GetSymbol() == "C"
                and all(
                    oxygen is not None and oxygen.GetSymbol() == "O"
                    for oxygen in oxygens
                )
            )
            if complete and ph > model_pka:
                target = _pick_carboxylate_target(carbon, oxygens)
                for oxygen in oxygens:
                    strip_hydrogens.extend(
                        neighbor.GetIdx()
                        for neighbor in oxygen.GetNeighbors()
                        if neighbor.GetAtomicNum() == 1
                    )
                sites.append(
                    {
                        "kind": "carboxylate",
                        "residue": key,
                        "atom": target.GetPDBResidueInfo()
                        .GetName()
                        .strip(),
                        "oxygens": list(o_names),
                        "formal_charge": -1,
                    }
                )
                stats["acid"] += 1
            elif complete:
                # The parsed neutral-acid template state already matches.
                stats["acid_neutral"] += 1
        if "OXT" in atoms and ph > _CTERM_MODEL_PKA:
            carbon = atoms.get("C")
            oxygens = [atoms.get(name) for name in ("O", "OXT")]
            if (
                carbon is not None
                and carbon.GetSymbol() == "C"
                and all(
                    oxygen is not None and oxygen.GetSymbol() == "O"
                    for oxygen in oxygens
                )
            ):
                target = _pick_carboxylate_target(carbon, oxygens)
                for oxygen in oxygens:
                    strip_hydrogens.extend(
                        neighbor.GetIdx()
                        for neighbor in oxygen.GetNeighbors()
                        if neighbor.GetAtomicNum() == 1
                    )
                sites.append(
                    {
                        "kind": "carboxylate",
                        "residue": key,
                        "atom": target.GetPDBResidueInfo().GetName().strip(),
                        "oxygens": ["O", "OXT"],
                        "formal_charge": -1,
                    }
                )
                stats["cterm"] += 1
        if resname == "LYS" and ph <= _LYS_MODEL_PKA:
            nz = atoms.get("NZ")
            if nz is not None and nz.GetSymbol() == "N":
                sites.append(
                    {
                        "kind": "cation",
                        "residue": key,
                        "atom": "NZ",
                        "required_h": 3,
                        "formal_charge": 1,
                    }
                )
                stats["lys"] += 1
        elif resname == "ARG" and ph <= _ARG_MODEL_PKA:
            cz = atoms.get("CZ")
            nitrogens = {name: atoms.get(name) for name in ("NE", "NH1", "NH2")}
            if (
                cz is not None
                and cz.GetSymbol() == "C"
                and all(
                    nitrogen is not None and nitrogen.GetSymbol() == "N"
                    for nitrogen in nitrogens.values()
                )
            ):
                charged = None
                for name in ("NH2", "NH1", "NE"):
                    bond = molecule.GetBondBetweenAtoms(
                        cz.GetIdx(), nitrogens[name].GetIdx()
                    )
                    if (
                        bond is not None
                        and bond.GetBondType() == Chem.BondType.DOUBLE
                    ):
                        charged = name
                        break
                if charged is None:
                    # Without a perceived C=N bond the exact Kekule charge
                    # placement is unavailable; NH2 is the conventional
                    # guanidinium nitrogen and the enforced net charge is
                    # still exactly +1.
                    charged = "NH2"
                for name in ("NE", "NH1", "NH2"):
                    # NE also bonds CD, so it carries one H in either the
                    # neutral or the C=N/+1 form; the guanidinium +1 sits
                    # only on the C=N nitrogen, all others stay neutral.
                    sites.append(
                        {
                            "kind": "cation",
                            "residue": key,
                            "atom": name,
                            "required_h": 1 if name == "NE" else 2,
                            "formal_charge": 1 if name == charged else 0,
                        }
                    )
                stats["arg"] += 1
        elif resname == "HIS":
            stats["his"] += 1
            for name in _HIS_RING_NITROGENS:
                nitrogen = atoms.get(name)
                if nitrogen is None:
                    continue
                stats["his_h"] += sum(
                    1
                    for neighbor in nitrogen.GetNeighbors()
                    if neighbor.GetAtomicNum() == 1
                )

    editor = Chem.RWMol(molecule)
    for index in sorted(set(strip_hydrogens), reverse=True):
        editor.RemoveAtom(index)
    edited = editor.GetMol()
    edited_groups = _group_atoms_by_residue(edited)
    edited_atoms = []
    for site in sites:
        atom = edited_groups[site["residue"]][site["atom"]]
        atom.SetFormalCharge(site["formal_charge"])
        edited_atoms.append(atom)
    if edited_atoms:
        for atom in edited_atoms:
            atom.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(edited)
        except Exception:
            pass

    remarks = [
        "REMARK  protonation policy: "
        f"{_PROTONATION_POLICY_NAME} (cycpep_master) at pH {ph:g}",
        "REMARK  protonation model pKa: ASP 3.9, GLU 4.1, LYS 10.5, "
        "ARG 12.5, C/OXT 4.1 (generic textbook constants; per-residue pKa "
        "NOT computed)",
        "REMARK  protonation applied: "
        f"{stats['acid']} Asp/Glu carboxylate(s) anionic, "
        f"{stats['acid_neutral']} neutral acid, "
        f"{stats['cterm']} terminal carboxylate(s) anionic, "
        f"{stats['lys']} Lys ammonium, {stats['arg']} Arg guanidinium "
        "cationic (net +1 per Arg)",
    ]
    if stats["his_h"]:
        remarks.append(
            "REMARK  protonation HIS: explicit input hydrogens kept as the "
            f"user state ({stats['his']} His, {stats['his_h']} ring N-H); "
            "microstate not re-determined"
        )
    elif stats["his"]:
        remarks.append(
            "REMARK  protonation HIS: no explicit input hydrogens; RDKit "
            "template tautomer retained; protonation/tautomer microstate "
            "UNDETERMINED (uncertain; requires review)"
        )
    else:
        remarks.append("REMARK  protonation HIS: no histidine residues present")
    remarks.append(
        "REMARK  protonation engine: RDKit residue-state template edits "
        "(cycpep_master residue policy)"
    )
    return edited, remarks, sites


def _hydrogen_position(conformer, parent):
    """Position a new hydrogen opposite the mean substituent direction."""
    origin = conformer.GetAtomPosition(parent.GetIdx())
    base = (origin.x, origin.y, origin.z)
    total = [0.0, 0.0, 0.0]
    for neighbor in parent.GetNeighbors():
        position = conformer.GetAtomPosition(neighbor.GetIdx())
        vector = (
            position.x - base[0],
            position.y - base[1],
            position.z - base[2],
        )
        length = math.sqrt(
            sum(component * component for component in vector)
        )
        if length > 1e-6:
            for index in range(3):
                total[index] += vector[index] / length
    magnitude = math.sqrt(sum(component * component for component in total))
    if magnitude <= 1e-6:
        total, magnitude = [1.0, 0.0, 0.0], 1.0
    return tuple(base[i] - 1.01 * total[i] / magnitude for i in range(3))


def _enforce_protonation_state(molecule, sites):
    """Force the requested hydrogen counts/charges after template AddHs.

    Fail closed: a policy atom that vanished raises, so a pH request can
    never silently degrade to the neutral default.
    """
    groups = _group_atoms_by_residue(molecule)
    remove = []
    charge_edits = []
    for site in sites:
        atoms = groups.get(site["residue"], {})
        if site["kind"] == "carboxylate":
            target = atoms.get(site["atom"])
            if target is None or target.GetSymbol() != "O":
                raise RuntimeError(
                    f"carboxylate oxygen {site['atom']} of "
                    f"{site['residue']} vanished after AddHs"
                )
            charge_edits.append(
                (site["residue"], site["atom"], site["formal_charge"])
            )
            for name in site["oxygens"]:
                oxygen = atoms.get(name)
                if oxygen is None:
                    continue
                remove.extend(
                    neighbor.GetIdx()
                    for neighbor in oxygen.GetNeighbors()
                    if neighbor.GetAtomicNum() == 1
                )
        else:
            nitrogen = atoms.get(site["atom"])
            if nitrogen is None or nitrogen.GetSymbol() != "N":
                raise RuntimeError(
                    f"nitrogen {site['atom']} of {site['residue']} "
                    "vanished after AddHs"
                )
            charge_edits.append(
                (site["residue"], site["atom"], site["formal_charge"])
            )
            hydrogens = [
                neighbor
                for neighbor in nitrogen.GetNeighbors()
                if neighbor.GetAtomicNum() == 1
            ]
            remove.extend(
                hydrogen.GetIdx()
                for hydrogen in hydrogens[site["required_h"] :]
            )
    editor = Chem.RWMol(molecule)
    for index in sorted(set(remove), reverse=True):
        editor.RemoveAtom(index)
    provisional = editor.GetMol()
    groups = _group_atoms_by_residue(provisional)
    for residue_key, name, charge in charge_edits:
        groups[residue_key][name].SetFormalCharge(charge)

    editor = Chem.RWMol(provisional)
    conformer = editor.GetConformer()
    groups = _group_atoms_by_residue(editor)
    for site in sites:
        if site["kind"] != "cation":
            continue
        nitrogen = groups[site["residue"]][site["atom"]]
        while (
            sum(
                1
                for neighbor in nitrogen.GetNeighbors()
                if neighbor.GetAtomicNum() == 1
            )
            < site["required_h"]
        ):
            index = editor.AddAtom(Chem.Atom(1))
            editor.AddBond(nitrogen.GetIdx(), index, Chem.BondType.SINGLE)
            conformer.SetAtomPosition(
                index, _hydrogen_position(conformer, nitrogen)
            )
    return editor.GetMol()


def pdb_to_receptor_pdbqt(
    pdb_path: str,
    pdbqt_path: str,
    *,
    atom_type=autodock_atom_type,
    ph: Optional[float] = None,
) -> Optional[str]:
    """Convert a receptor PDB to a rigid PDBQT without a torsion tree.

    ``ph=None`` keeps the historical RDKit-default protonation (with its
    "pH not assigned" warning).  A numeric ``ph`` in the supported range
    (0, 10.5] applies the explicit residue-state protonation policy,
    records provenance REMARKs, and fails the conversion when that policy
    is unavailable or the pH is outside the range it can honor.
    """
    output_path = Path(pdbqt_path)
    clear_error = _clear_prior_pdbqt_output(
        pdb_path,
        output_path,
        error_prefix="cannot clear prior receptor PDBQT output",
    )
    if clear_error:
        return clear_error
    if ph is not None:
        try:
            ph = float(ph)
        except (TypeError, ValueError):
            return f"invalid pH {ph!r}: must be a finite number in (0, 14]"
        if not math.isfinite(ph) or not 0.0 < ph <= 14.0:
            return f"invalid pH {ph!r}: must be a finite number in (0, 14]"
        if ph > _LYS_MODEL_PKA:
            return (
                f"not_supported: pH {ph:g} is outside the residue-state "
                f"policy supported range (0, {_LYS_MODEL_PKA:g}]; neutral "
                "Lys/Arg states are not assigned, refusing to guess the "
                "ionization state"
            )
    try:
        molecule = Chem.MolFromPDBFile(
            pdb_path, removeHs=False, sanitize=False
        )
        if molecule is None:
            return f"RDKit failed to parse receptor {pdb_path}"
        declared_edges = _read_declared_pdb_edges(pdb_path)
        molecule, sanitize_error, repaired_edges = (
            _sanitize_with_spurious_edge_repair(molecule, declared_edges)
        )
        if sanitize_error:
            return sanitize_error
        if not molecule.GetNumConformers():
            return "Receptor PDB has no 3D coordinates"
        molecule, ion_error, ion_policy_remarks = (
            _apply_monoatomic_ion_policy(molecule)
        )
        if ion_error:
            return ion_error
        protonation_remarks: Optional[list] = None
        policy_sites: Optional[list] = None
        if ph is not None:
            try:
                backend = _load_protonation_policy_backend()
                molecule, protonation_remarks, policy_sites = backend(
                    molecule, ph
                )
            except Exception as exc:
                return (
                    "not_supported: receptor protonation policy at pH "
                    f"{ph:g} is unavailable: {type(exc).__name__}: {exc}; "
                    "refusing to emit the unprotonated neutral default"
                )
        hydrogen_remark = (
            "REMARK  WARNING hydrogens: completion unavailable; kept as parsed"
        )
        protonated = None
        try:
            protonated = Chem.AddHs(molecule, addCoords=True)
        except Exception as exc:
            hydrogen_remark = (
                "REMARK  WARNING hydrogens: RDKit AddHs failed "
                f"({type(exc).__name__}); hydrogens kept as parsed; "
                "nonpolar H merged into heavy-atom charges"
            )
        if protonated is not None:
            molecule = protonated
            hydrogen_remark = (
                "REMARK  hydrogens: polar H explicit (HD, RDKit AddHs); "
                "nonpolar H merged into heavy-atom charges"
            )
        if policy_sites is not None:
            try:
                molecule = _enforce_protonation_state(molecule, policy_sites)
            except Exception as exc:
                return (
                    "not_supported: receptor protonation policy could not "
                    f"be enforced: {type(exc).__name__}: {exc}"
                )
        _label_added_hydrogens(molecule)
        # Classify unsupported AutoDock elements before charge assignment.
        # RDKit may emit a non-finite Gasteiger charge for elements such as Se;
        # that is a capability boundary, not an undifferentiated conversion
        # failure.  Receptor-side verified ion types (Na/K) extend the shared
        # whitelist here only; ligand typing semantics are untouched.
        unsupported_types = sorted(
            {
                atom_type(atom)
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() != 1
                and atom_type(atom) not in _AUTODOCK4_TYPES
                and atom_type(atom) not in _RECEPTOR_ION_ATOM_TYPES
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
        conformer = molecule.GetConformer()

        charges: dict[int, float] = {}
        nonfinite_atoms = []
        for atom in molecule.GetAtoms():
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
            except Exception as exc:
                return f"receptor Gasteiger charge is unavailable: {exc}"
            if not math.isfinite(charge):
                nonfinite_atoms.append(atom)
                continue
            charges[atom.GetIdx()] = charge
        if nonfinite_atoms:
            return _nonfinite_charge_error(nonfinite_atoms)
        for atom in molecule.GetAtoms():
            if atom_type(atom) != "" or atom.GetAtomicNum() != 1:
                continue
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                return "omitted receptor hydrogen has no unique heavy-atom parent"
            parent = neighbors[0].GetIdx()
            charges[parent] += charges[atom.GetIdx()]

        lines = [
            "REMARK  rigid receptor PDBQT (cycpep_master)",
            hydrogen_remark,
        ]
        for left_site, right_site in repaired_edges:
            lines.append(
                f"REMARK  proximity repair: removed undeclared inter-residue "
                f"edge {left_site} - {right_site} (valence repair; not "
                "declared by LINK/SSBOND/CONECT and not a peptide or "
                "disulfide link)"
            )
        lines.extend(ion_policy_remarks)
        if protonation_remarks is None:
            lines.append(
                "REMARK  WARNING protonation: RDKit default valence; pH not assigned; ionization-sensitive residues require review"
            )
        else:
            lines.extend(protonation_remarks)
        serial = 0
        for atom in molecule.GetAtoms():
            autodock_type = atom_type(atom)
            if autodock_type == "":
                continue
            if autodock_type not in _AUTODOCK4_TYPES and (
                autodock_type not in _RECEPTOR_ION_ATOM_TYPES
            ):
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
            name = (
                (info.GetName() if info else "").strip()
                or atom.GetSymbol()
            )[:4]
            residue_name = (
                info.GetResidueName() if info else "UNK"
            ).strip()[:3]
            chain = (info.GetChainId() if info else "A") or "A"
            residue_number = info.GetResidueNumber() if info else 1
            charge = charges[atom.GetIdx()]
            lines.append(
                "ATOM  %5d %-4s %-3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f    %6.3f %-2s"
                % (
                    serial % 100000,
                    name.ljust(4)[:4],
                    residue_name,
                    chain[:1],
                    residue_number % 10000,
                    (info.GetInsertionCode() if info else " ")[:1] or " ",
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
