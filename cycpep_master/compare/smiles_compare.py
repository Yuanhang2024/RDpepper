"""SMILES comparison utilities."""
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchiKey


@dataclass(frozen=True)
class StrictComparisonResult:
    status: str
    strict_graph_match: bool | None
    full_inchikey_match: bool | None
    connectivity_match: bool | None
    canonical_isomeric_a: str | None
    canonical_isomeric_b: str | None
    full_inchikey_a: str | None
    full_inchikey_b: str | None
    molecular_formula_a: str | None
    molecular_formula_b: str | None
    reason: str | None = None


def _strict_identity(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "empty SMILES"
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return None, f"SMILES parse raised {type(exc).__name__}: {exc}"
    if molecule is None:
        return None, "invalid SMILES"
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True,
            allHsExplicit=False,
        )
        inchikey = MolToInchiKey(molecule) or None
        formula = rdMolDescriptors.CalcMolFormula(molecule)
    except Exception as exc:
        return None, f"identity generation failed: {type(exc).__name__}: {exc}"
    if not inchikey or len(inchikey.split("-")) != 3:
        return None, "full InChIKey generation failed"
    return {
        "canonical": canonical,
        "inchikey": inchikey,
        "formula": formula,
    }, None


def compare_strict(a, b) -> StrictComparisonResult:
    """Compare exact molecular identities without permissive fallbacks."""
    left, left_error = _strict_identity(a)
    right, right_error = _strict_identity(b)
    if left is None or right is None:
        return StrictComparisonResult(
            status="invalid_input",
            strict_graph_match=None,
            full_inchikey_match=None,
            connectivity_match=None,
            canonical_isomeric_a=left["canonical"] if left else None,
            canonical_isomeric_b=right["canonical"] if right else None,
            full_inchikey_a=left["inchikey"] if left else None,
            full_inchikey_b=right["inchikey"] if right else None,
            molecular_formula_a=left["formula"] if left else None,
            molecular_formula_b=right["formula"] if right else None,
            reason="; ".join(
                reason for reason in (left_error, right_error) if reason
            ),
        )
    graph_match = left["canonical"] == right["canonical"]
    return StrictComparisonResult(
        status="match" if graph_match else "mismatch",
        strict_graph_match=graph_match,
        full_inchikey_match=left["inchikey"] == right["inchikey"],
        connectivity_match=(
            left["inchikey"].split("-", 1)[0]
            == right["inchikey"].split("-", 1)[0]
        ),
        canonical_isomeric_a=left["canonical"],
        canonical_isomeric_b=right["canonical"],
        full_inchikey_a=left["inchikey"],
        full_inchikey_b=right["inchikey"],
        molecular_formula_a=left["formula"],
        molecular_formula_b=right["formula"],
    )


def rdkit_canonical(smiles):
    """Normalize SMILES to RDKit canonical form (no stereo, no isotopes)."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        for a in mol.GetAtoms():
            a.SetIsotope(0)
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def compare(a, b):
    """Legacy permissive comparison with cascading fallback strategies.

    Scientific identity checks must use :func:`compare_strict`.
    """
    if not a or not b:
        return False, "None"

    ca = rdkit_canonical(a)
    cb = rdkit_canonical(b)
    if ca is None or cb is None:
        return False, "canonical fail"
    if ca == cb:
        return True, "match"

    ma = Chem.MolFromSmiles(ca)
    mb = Chem.MolFromSmiles(cb)
    if not ma or not mb:
        return False, "mol fail"

    # Do not let the permissive fingerprint fallback call molecules with
    # different composition equivalent.  This guard intentionally precedes
    # charge/stereo normalization below, because those normalizations are
    # useful for legacy route comparisons but must not erase atom loss/gain.
    formula_a = rdMolDescriptors.CalcMolFormula(ma)
    formula_b = rdMolDescriptors.CalcMolFormula(mb)
    heavy_a = ma.GetNumHeavyAtoms()
    heavy_b = mb.GetNumHeavyAtoms()
    if formula_a != formula_b or heavy_a != heavy_b:
        return False, (
            f"composition mismatch: formula {formula_a} vs {formula_b}; "
            f"heavy_atoms {heavy_a} vs {heavy_b}"
        )

    for mol in (ma, mb):
        Chem.SanitizeMol(mol)
        rdmolops.RemoveStereochemistry(mol)
        for a in mol.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsotope(0)
            a.SetAtomMapNum(0)

    na = Chem.MolToSmiles(ma, canonical=True, isomericSmiles=False)
    nb = Chem.MolToSmiles(mb, canonical=True, isomericSmiles=False)
    if na == nb:
        return True, "connect"

    if MolToInchiKey(ma) == MolToInchiKey(mb):
        return True, "InChIKey"

    fa = AllChem.GetMACCSKeysFingerprint(ma)
    fb = AllChem.GetMACCSKeysFingerprint(mb)
    t = AllChem.DataStructs.TanimotoSimilarity(fa, fb)
    if t > 0.99:
        return True, f"fp {t:.4f}"
    return False, f"fp {t:.4f}"
