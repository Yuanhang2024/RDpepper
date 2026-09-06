"""Compare only stereochemical constraints explicitly specified by a reference InChI."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey


_DEFINED_BOND_STEREO = {
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
}
_MAX_GRAPH_ISOMORPHISMS = 10_000


@dataclass(frozen=True)
class SpecifiedStereoComparisonResult:
    """Result of an auxiliary, reference-constrained stereo comparison.

    This result is intentionally separate from strict full-InChIKey identity.
    An unspecified reference stereocentre is not treated as either matching or
    mismatching an observed assignment.
    """

    status: str
    nonstereo_inchikey_match: bool | None
    specified_stereo_match: bool | None
    reference_specified_atom_stereo_count: int | None
    reference_unspecified_atom_stereo_count: int | None
    reference_specified_bond_stereo_count: int | None
    matched_specified_atom_stereo_count: int | None
    matched_specified_bond_stereo_count: int | None
    matched_specified_stereo_fraction: float | None
    reference_full_inchikey: str | None
    observed_full_inchikey: str | None
    reason: str | None = None


def _parse_reference_inchi(value: str):
    if not isinstance(value, str) or not value.startswith("InChI="):
        return None, "reference is not an InChI string"
    try:
        molecule = Chem.MolFromInchi(value, sanitize=True, removeHs=True)
    except Exception as exc:
        return None, f"reference InChI parse raised {type(exc).__name__}: {exc}"
    if molecule is None:
        return None, "invalid reference InChI"
    return molecule, None


def _parse_observed(value: str):
    if not isinstance(value, str) or not value.strip():
        return None, "empty observed structure"
    try:
        if value.startswith("InChI="):
            molecule = Chem.MolFromInchi(value, sanitize=True, removeHs=True)
        else:
            molecule = Chem.MolFromSmiles(value)
    except Exception as exc:
        return None, f"observed structure parse raised {type(exc).__name__}: {exc}"
    if molecule is None:
        return None, "invalid observed structure"
    return molecule, None


def _identity(molecule: Chem.Mol) -> tuple[str | None, str | None]:
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(molecule)
        full = MolToInchiKey(molecule) or None
        nonstereo = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(nonstereo)
        nonstereo_key = MolToInchiKey(nonstereo) or None
    except Exception:
        return None, None
    return full, nonstereo_key


def _reference_stereo_counts(molecule: Chem.Mol) -> tuple[int, int, int]:
    atom_centres = Chem.FindMolChiralCenters(
        molecule,
        includeUnassigned=True,
        includeCIP=True,
        useLegacyImplementation=False,
    )
    specified_atoms = sum(label != "?" for _index, label in atom_centres)
    unspecified_atoms = sum(label == "?" for _index, label in atom_centres)
    specified_bonds = sum(
        bond.GetStereo() in _DEFINED_BOND_STEREO for bond in molecule.GetBonds()
    )
    return specified_atoms, unspecified_atoms, specified_bonds


def _best_constraint_match(
    reference: Chem.Mol,
    observed: Chem.Mol,
) -> tuple[int, int] | None:
    reference = Chem.Mol(reference)
    try:
        observed_inchi = MolToInchi(observed)
        observed = Chem.MolFromInchi(observed_inchi, sanitize=True, removeHs=True)
    except Exception:
        return None
    if observed is None:
        return None
    Chem.AssignStereochemistry(reference, cleanIt=True, force=True)
    Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
    reference_atoms = {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in reference.GetAtoms()
        if atom.HasProp("_CIPCode")
    }
    reference_bonds = {
        bond.GetIdx(): bond.GetStereo()
        for bond in reference.GetBonds()
        if bond.GetStereo() in _DEFINED_BOND_STEREO
    }
    mappings = observed.GetSubstructMatches(
        reference,
        useChirality=False,
        uniquify=False,
        maxMatches=_MAX_GRAPH_ISOMORPHISMS + 1,
    )
    if not mappings or len(mappings) > _MAX_GRAPH_ISOMORPHISMS:
        return None
    best = (0, 0)
    for mapping in mappings:
        atom_matches = sum(
            observed.GetAtomWithIdx(mapping[index]).HasProp("_CIPCode")
            and observed.GetAtomWithIdx(mapping[index]).GetProp("_CIPCode") == label
            for index, label in reference_atoms.items()
        )
        bond_matches = 0
        for bond_index, stereo in reference_bonds.items():
            reference_bond = reference.GetBondWithIdx(bond_index)
            observed_bond = observed.GetBondBetweenAtoms(
                mapping[reference_bond.GetBeginAtomIdx()],
                mapping[reference_bond.GetEndAtomIdx()],
            )
            bond_matches += bool(
                observed_bond is not None and observed_bond.GetStereo() == stereo
            )
        best = max(best, (atom_matches, bond_matches))
    return best


def compare_specified_stereo(
    reference_inchi: str,
    observed_structure: str,
) -> SpecifiedStereoComparisonResult:
    """Compare the observed molecule with constraints specified by an InChI.

    The comparison is eligible only when removing stereochemistry yields the
    same full InChIKey on both sides. RDKit chirality-aware graph matching then
    enforces reference R/S and E/Z assignments while leaving reference ``?``
    assignments unconstrained. This auxiliary result must not replace strict
    full-InChIKey scoring.
    """

    reference, reference_error = _parse_reference_inchi(reference_inchi)
    observed, observed_error = _parse_observed(observed_structure)
    if reference is None or observed is None:
        return SpecifiedStereoComparisonResult(
            status="invalid_input",
            nonstereo_inchikey_match=None,
            specified_stereo_match=None,
            reference_specified_atom_stereo_count=None,
            reference_unspecified_atom_stereo_count=None,
            reference_specified_bond_stereo_count=None,
            matched_specified_atom_stereo_count=None,
            matched_specified_bond_stereo_count=None,
            matched_specified_stereo_fraction=None,
            reference_full_inchikey=None,
            observed_full_inchikey=None,
            reason="; ".join(
                reason for reason in (reference_error, observed_error) if reason
            ),
        )

    reference_full, reference_nonstereo = _identity(reference)
    observed_full, observed_nonstereo = _identity(observed)
    specified_atoms, unspecified_atoms, specified_bonds = _reference_stereo_counts(
        reference
    )
    if not all(
        (reference_full, reference_nonstereo, observed_full, observed_nonstereo)
    ):
        return SpecifiedStereoComparisonResult(
            status="invalid_input",
            nonstereo_inchikey_match=None,
            specified_stereo_match=None,
            reference_specified_atom_stereo_count=specified_atoms,
            reference_unspecified_atom_stereo_count=unspecified_atoms,
            reference_specified_bond_stereo_count=specified_bonds,
            matched_specified_atom_stereo_count=None,
            matched_specified_bond_stereo_count=None,
            matched_specified_stereo_fraction=None,
            reference_full_inchikey=reference_full,
            observed_full_inchikey=observed_full,
            reason="InChIKey generation failed",
        )

    nonstereo_match = reference_nonstereo == observed_nonstereo
    if not nonstereo_match:
        return SpecifiedStereoComparisonResult(
            status="not_comparable",
            nonstereo_inchikey_match=False,
            specified_stereo_match=None,
            reference_specified_atom_stereo_count=specified_atoms,
            reference_unspecified_atom_stereo_count=unspecified_atoms,
            reference_specified_bond_stereo_count=specified_bonds,
            matched_specified_atom_stereo_count=None,
            matched_specified_bond_stereo_count=None,
            matched_specified_stereo_fraction=None,
            reference_full_inchikey=reference_full,
            observed_full_inchikey=observed_full,
            reason="nonstereo_inchikey_mismatch",
        )

    if specified_atoms + specified_bonds == 0:
        return SpecifiedStereoComparisonResult(
            status="not_comparable",
            nonstereo_inchikey_match=True,
            specified_stereo_match=None,
            reference_specified_atom_stereo_count=specified_atoms,
            reference_unspecified_atom_stereo_count=unspecified_atoms,
            reference_specified_bond_stereo_count=specified_bonds,
            matched_specified_atom_stereo_count=None,
            matched_specified_bond_stereo_count=None,
            matched_specified_stereo_fraction=None,
            reference_full_inchikey=reference_full,
            observed_full_inchikey=observed_full,
            reason="reference_has_no_specified_stereo_constraints",
        )

    matched = _best_constraint_match(reference, observed)
    if matched is None:
        return SpecifiedStereoComparisonResult(
            status="not_comparable",
            nonstereo_inchikey_match=True,
            specified_stereo_match=None,
            reference_specified_atom_stereo_count=specified_atoms,
            reference_unspecified_atom_stereo_count=unspecified_atoms,
            reference_specified_bond_stereo_count=specified_bonds,
            matched_specified_atom_stereo_count=None,
            matched_specified_bond_stereo_count=None,
            matched_specified_stereo_fraction=None,
            reference_full_inchikey=reference_full,
            observed_full_inchikey=observed_full,
            reason="graph_isomorphism_missing_or_limit_exceeded",
        )
    matched_atoms, matched_bonds = matched
    total = specified_atoms + specified_bonds
    matched_total = matched_atoms + matched_bonds
    stereo_match = matched_total == total
    return SpecifiedStereoComparisonResult(
        status="match" if stereo_match else "mismatch",
        nonstereo_inchikey_match=True,
        specified_stereo_match=stereo_match,
        reference_specified_atom_stereo_count=specified_atoms,
        reference_unspecified_atom_stereo_count=unspecified_atoms,
        reference_specified_bond_stereo_count=specified_bonds,
        matched_specified_atom_stereo_count=matched_atoms,
        matched_specified_bond_stereo_count=matched_bonds,
        matched_specified_stereo_fraction=matched_total / total,
        reference_full_inchikey=reference_full,
        observed_full_inchikey=observed_full,
        reason=None if stereo_match else "specified_stereo_constraint_mismatch",
    )
