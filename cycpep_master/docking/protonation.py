"""Deterministic dominant-state protonation rules used for docking.

Two layers share one rule engine:

* :func:`protonate_molecule_ph74` — strict molecule-level policy application.
  It returns a NEW molecule plus a JSON-ready microstate report and never
  consults deposited references, benchmark truth, or site-specific pKa models.
* :func:`protonate_ph74` — legacy SMILES-in/SMILES-out facade that delegates
  to the strict engine for parseable input and retains its historical
  invalid-input passthrough.

Documented policy defaults (dominant microstate at pH 7.4; these are rules,
not experimentally validated predictions):

* deprotonated: carboxylic acids, sulfonic acids, and phosphoric
  acids/phosphates (at most two acidic oxygens per phosphorus);
* protonated: primary/secondary/tertiary aliphatic amines (the protonated
  tertiary amine carries ONE hydrogen), guanidines and amidines (charge and
  the added hydrogen on the imino nitrogen);
* conservatively neutral (dominant-state default, explicitly reported in the
  microstate report): histidine/imidazole-type aromatic five-rings with two
  nitrogens, aliphatic thiols, phenols;
* never modified: amides, sulfonamides, aromatic amines, nitriles, atoms that
  already carry a formal charge (rules only target the neutral form), and
  non-target ions; quaternary ammonium keeps zero hydrogens; every component
  of a salt is retained.

The molecule-level helper preserves the heavy-atom relative order, the bond
topology, all fragments, heavy-atom 3D coordinates, atom metadata and stereo
tags; hydrogen coordinates may be rebuilt.  The input molecule is never
modified.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

POLICY_NAME = "physiological"
POLICY_PH = 7.4

CLAIM_BOUNDARY = {
    "allowed": [
        "deterministic dominant-microstate assignment at pH 7.4",
        "charge/hydrogen bookkeeping provenance",
    ],
    "forbidden": [
        "experimental protonation-state validation",
        "site-specific pKa prediction",
        "recovery of deposited-model protonation",
    ],
}

WARNING_HISTIDINE_NEUTRAL = "HISTIDINE_IMIDAZOLE_NEUTRAL_CONSERVATIVE"
WARNING_CHARGED_HISTIDINE_PRESERVED = "CHARGED_HISTIDINE_PRESERVED"
WARNING_THIOL_PHENOL_NEUTRAL = "THIOL_PHENOL_NEUTRAL_CONSERVATIVE"
WARNING_UNFOLDABLE_HYDROGEN = "UNFOLDABLE_HYDROGEN_PRESERVED"
WARNING_STEREO_CIP_CHANGED = "STEREO_CIP_LABEL_CHANGED"
WARNING_ISOTOPE_SITE_SKIPPED = "ISOTOPE_HYDROGEN_SITE_SKIPPED"

# Amine rules exclude any nitrogen bonded to a non-carbon heavy atom
# (amides, sulfonamides, hydrazines, hydroxylamines, ...) and imine-type
# nitrogen; the imino rules below cover C=N protonation separately.
_AMINE_EXCLUSIONS = "!$(N[!#6;!#1]);!$(N[C,S,P]=O);!$(N=*)"

_SMARTS_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
_SMARTS_SULFON = Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")
_SMARTS_AMINE_PRIMARY = Chem.MolFromSmarts(
    f"[NX3;H2;{_AMINE_EXCLUSIONS}][CX4]"
)
_SMARTS_AMINE_SECONDARY = Chem.MolFromSmarts(
    f"[NX3;H1;{_AMINE_EXCLUSIONS}]([CX4])[CX4]"
)
_SMARTS_AMINE_TERTIARY = Chem.MolFromSmarts(
    f"[NX3;H0;{_AMINE_EXCLUSIONS}]([CX4])([CX4])[CX4]"
)
_SMARTS_GUANIDINE = Chem.MolFromSmarts("[NX2]=[CX3]([NX3])[NX3]")
_SMARTS_AMIDINE = Chem.MolFromSmarts("[NX2]=[CX3][NX3;!$(NC=O)]")
_SMARTS_THIOL = Chem.MolFromSmarts("[#6][SX2H1]")
_SMARTS_PHENOL = Chem.MolFromSmarts("[OX2H1][c]")


class ProtonationPolicyError(ValueError):
    """Raised when the strict protonation policy cannot be applied."""


def _is_protium(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 1 and atom.GetIsotope() in (0, 1)


def _is_foldable_protium(atom: Chem.Atom) -> bool:
    return (
        _is_protium(atom)
        and atom.GetDegree() == 1
        and atom.GetNeighbors()[0].GetAtomicNum() != 1
    )


def _survivor_indices(molecule: Chem.Mol) -> list[int]:
    """Input atom indices that survive the protium fold, in input order.

    Survivors are every non-protium atom (heavy atoms and isotopic
    hydrogens) plus any protium RemoveHs cannot fold.  Folded/output atom
    indices equal positions in this list, so ``survivors[k]`` is the
    original input index of folded atom ``k``.
    """
    return [
        atom.GetIdx() for atom in molecule.GetAtoms()
        if not _is_foldable_protium(atom)
    ]


def _fold_explicit_hydrogens(
    molecule: Chem.Mol,
) -> tuple[Chem.RWMol, bool]:
    """Return a heavy-ordered copy with protium folded into H counts.

    Uses ``Chem.RemoveHs`` (stereo-tag safe) and then pins every heavy atom
    to exact ``NumExplicitHs``/``NoImplicit`` semantics, so SMARTS total-H
    queries and exact per-site hydrogen targets are unambiguous.  Isotopic
    hydrogens are preserved as explicit atoms.  The second value reports
    whether unfoldable protium was kept as an explicit atom.
    """
    try:
        folded = Chem.RemoveHs(Chem.Mol(molecule), sanitize=False)
    except Exception as exc:  # noqa: BLE001 - surfaced as policy failure
        raise ProtonationPolicyError(
            f"explicit hydrogens cannot be folded: {exc}"
        ) from exc
    editable = Chem.RWMol(folded)
    editable.UpdatePropertyCache(strict=False)
    kept_unfoldable = False
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 1:
            if _is_protium(atom):
                kept_unfoldable = True
            continue
        total_h = atom.GetTotalNumHs()
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_h)
    return editable, kept_unfoldable


def _sanitize_strict(mol: Chem.Mol) -> None:
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - surfaced as policy failure
        raise ProtonationPolicyError(f"molecule is not sanitizable: {exc}") from exc


def _phosphorus_sites(probe: Chem.Mol) -> list[dict[str, Any]]:
    """Deprotonate up to two acidic oxygens per phosphorus (legacy rule)."""
    sites: list[dict[str, Any]] = []
    for phosphorus in probe.GetAtoms():
        if phosphorus.GetAtomicNum() != 15:
            continue
        acidic: list[int] = []
        already_negative = 0
        for oxygen in phosphorus.GetNeighbors():
            bond = probe.GetBondBetweenAtoms(
                phosphorus.GetIdx(), oxygen.GetIdx()
            )
            if (
                oxygen.GetAtomicNum() == 8
                and bond.GetBondType() == Chem.BondType.SINGLE
                and oxygen.GetDegree() == 1
            ):
                if oxygen.GetFormalCharge() == -1 and oxygen.GetTotalNumHs() == 0:
                    already_negative += 1
                    acidic.append(oxygen.GetIdx())
                elif oxygen.GetTotalNumHs() >= 1 and oxygen.GetFormalCharge() == 0:
                    acidic.append(oxygen.GetIdx())
        target_negative = min(2, len(acidic))
        budget = target_negative - already_negative
        for index in sorted(acidic):
            if budget <= 0:
                break
            atom = probe.GetAtomWithIdx(index)
            if atom.GetFormalCharge() == 0:
                sites.append({
                    "index": index,
                    "rule": "phosphoric_acid_deprotonated",
                    "charge": -1,
                    "hydrogens": 0,
                })
                budget -= 1
    return sites


def _pattern_sites(
    probe: Chem.Mol,
    smarts: Chem.Mol,
    rule: str,
    charge: int,
    hydrogen_target_from_delta: int,
    *,
    target: int = 0,
) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for match in probe.GetSubstructMatches(smarts):
        index = int(match[target])
        atom = probe.GetAtomWithIdx(index)
        if atom.GetFormalCharge() != 0:
            continue
        sites.append({
            "index": index,
            "rule": rule,
            "charge": charge,
            "hydrogens": atom.GetTotalNumHs() + hydrogen_target_from_delta,
        })
    return sites


def _has_isotope_hydrogen_neighbor(atom: Chem.Atom) -> bool:
    return any(
        neighbour.GetAtomicNum() == 1 and neighbour.GetIsotope() >= 2
        for neighbour in atom.GetNeighbors()
    )


def _detect_sites(probe: Chem.Mol) -> tuple[list[dict[str, Any]], list[int]]:
    """Ordered, de-duplicated neutral-microstate sites on the folded probe.

    Sites whose atom carries an isotopic-hydrogen neighbour (D/T) are
    explicitly refused rather than double-counting hydrogen; their indices
    are returned separately for the microstate report.
    """
    sites: list[dict[str, Any]] = []
    seen: set[int] = set()
    skipped_isotope: list[int] = []

    def add(candidates: list[dict[str, Any]]) -> None:
        for site in candidates:
            if site["index"] in seen:
                continue
            seen.add(site["index"])
            if _has_isotope_hydrogen_neighbor(probe.GetAtomWithIdx(site["index"])):
                skipped_isotope.append(site["index"])
                continue
            sites.append(site)

    add(_pattern_sites(probe, _SMARTS_CARBOXYL, "carboxylic_acid_deprotonated", -1, -1, target=2))
    add(_pattern_sites(probe, _SMARTS_SULFON, "sulfonic_acid_deprotonated", -1, -1, target=3))
    add(_phosphorus_sites(probe))
    add(_pattern_sites(probe, _SMARTS_AMINE_PRIMARY, "primary_aliphatic_amine_protonated", 1, 1))
    add(_pattern_sites(probe, _SMARTS_AMINE_SECONDARY, "secondary_aliphatic_amine_protonated", 1, 1))
    add(_pattern_sites(probe, _SMARTS_AMINE_TERTIARY, "tertiary_aliphatic_amine_protonated", 1, 1))
    add(_pattern_sites(probe, _SMARTS_GUANIDINE, "guanidine_protonated", 1, 1))
    add(_pattern_sites(probe, _SMARTS_AMIDINE, "amidine_protonated", 1, 1))
    return sites, sorted(skipped_isotope)


def _imidazole_report(probe: Chem.Mol) -> tuple[list[int], list[int]]:
    """(neutral imidazole-like nitrogens, charged azolium nitrogen indices).

    Imidazole-like = aromatic five-membered ring carrying exactly two
    nitrogens (histidine imidazole, substituted imidazoles).  The dominant
    pH-7.4 microstate is neutral, so neutral rings are only reported; an
    already-charged ring nitrogen is PRESERVED and explicitly reported
    rather than silently neutralized or further protonated.
    """
    ring_info = probe.GetRingInfo()
    neutral: set[int] = set()
    charged: list[int] = []
    for ring in ring_info.AtomRings():
        if len(ring) != 5:
            continue
        atoms = [probe.GetAtomWithIdx(index) for index in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        nitrogens = [atom for atom in atoms if atom.GetAtomicNum() == 7]
        if len(nitrogens) != 2:
            continue
        for atom in nitrogens:
            if atom.GetFormalCharge() == 1:
                charged.append(atom.GetIdx())
            else:
                neutral.add(atom.GetIdx())
    return sorted(neutral - set(charged)), sorted(set(charged))


def _input_formal_charge(molecule: Chem.Mol) -> int:
    return sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())


def _heavy_signature(molecule: Chem.Mol) -> tuple:
    """Heavy-atom identity sequence (order-sensitive conservation check)."""
    return tuple(
        (atom.GetAtomicNum(), atom.GetIsotope())
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 1 or atom.GetIsotope() >= 2
    )


def _heavy_positions(molecule: Chem.Mol) -> list[tuple[float, float, float]] | None:
    conformer = molecule.GetConformer() if molecule.GetNumConformers() else None
    if conformer is None:
        return None
    positions = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() < 2:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        positions.append((point.x, point.y, point.z))
    return positions


def _stereo_signature(molecule: Chem.Mol, survivors: list[int]) -> list[tuple[int, str, str | None]]:
    """(survivor position, chiral tag, CIP code or None) per tagged survivor.

    Compared by survivor position (heavy relative order), so interleaved
    hydrogens cannot alias the mapping.  Raises on stereochemistry
    assignment failure instead of silently returning no evidence.
    """
    copy = Chem.Mol(molecule)
    Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    position_of_index = {
        input_index: position for position, input_index in enumerate(survivors)
    }
    signature = []
    for atom in copy.GetAtoms():
        position = position_of_index.get(atom.GetIdx())
        if position is None:
            continue
        tag = str(atom.GetChiralTag())
        if tag == str(Chem.ChiralType.CHI_UNSPECIFIED):
            continue
        cip = str(atom.GetProp("_CIPCode")) if atom.HasProp("_CIPCode") else None
        signature.append((position, tag, cip))
    return signature


def _heavy_bond_topology(molecule: Chem.Mol, survivors: list[int]) -> set[tuple[int, int, str]]:
    """Rank-indexed survivor bond multiset (bond type names as-is)."""
    position_of_index = {
        input_index: position for position, input_index in enumerate(survivors)
    }
    topology: set[tuple[int, int, str]] = set()
    for bond in molecule.GetBonds():
        left = position_of_index.get(bond.GetBeginAtomIdx())
        right = position_of_index.get(bond.GetEndAtomIdx())
        if left is None or right is None:
            continue
        topology.add((min(left, right), max(left, right), bond.GetBondType().name))
    return topology


def _tags_changed(
    before: list[tuple[int, str, str | None]],
    after: list[tuple[int, str, str | None]],
) -> tuple[bool, bool]:
    """(chiral tags changed, CIP labels changed) by survivor position.

    Chiral-tag changes are integrity failures; CIP labels may legitimately
    reorder when a formal-charge change reorders Cahn-Ingold priorities.
    """
    tags_before = [(index, tag) for index, tag, _cip in before]
    tags_after = [(index, tag) for index, tag, _cip in after]
    cip_before = [(index, cip) for index, _tag, cip in before if cip]
    cip_after = [(index, cip) for index, _tag, cip in after if cip]
    return tags_before != tags_after, cip_before != cip_after


def protonate_molecule_ph74(molecule: Chem.Mol) -> tuple[Chem.Mol, dict[str, Any]]:
    """Apply the documented pH-7.4 dominant-state policy to ``molecule``.

    Returns ``(new_molecule, microstate_report)``.  The new molecule preserves
    the heavy-atom relative order, bond topology, fragment count, heavy-atom
    coordinates, atom metadata and stereo tags of the input; hydrogens may be
    rebuilt (coordinates included).  The input molecule is not modified.

    Raises :class:`ProtonationPolicyError` (a ``ValueError`` subclass) for
    inputs that are not RDKit molecules, have no atoms, or cannot be
    sanitized/policy-applied; failures surface instead of returning fake
    success.
    """
    if not isinstance(molecule, Chem.Mol):
        raise ProtonationPolicyError(
            f"expected an RDKit Mol, got {type(molecule).__name__}"
        )
    if molecule.GetNumAtoms() == 0:
        raise ProtonationPolicyError("molecule has no atoms")

    probe, _ = _fold_explicit_hydrogens(molecule)
    # Sanitize the RWMol itself so its own RingInfo (used by the imidazole
    # survey below) is populated.
    _sanitize_strict(probe)
    survivors = _survivor_indices(molecule)
    if probe.GetNumAtoms() != len(survivors):
        raise ProtonationPolicyError(
            "explicit-hydrogen fold diverged from the RemoveHs survivor set"
        )

    sites, skipped_isotope = _detect_sites(probe)
    neutral_imidazole, charged_imidazole = _imidazole_report(probe)
    thiols = [int(match[1]) for match in probe.GetSubstructMatches(_SMARTS_THIOL)]
    phenols = [int(match[0]) for match in probe.GetSubstructMatches(_SMARTS_PHENOL)]

    warnings: list[str] = []
    if neutral_imidazole:
        warnings.append(WARNING_HISTIDINE_NEUTRAL)
    if charged_imidazole:
        warnings.append(WARNING_CHARGED_HISTIDINE_PRESERVED)
    if thiols or phenols:
        warnings.append(WARNING_THIOL_PHENOL_NEUTRAL)
    if skipped_isotope:
        warnings.append(WARNING_ISOTOPE_SITE_SKIPPED)

    has_foldable_protium = any(
        _is_foldable_protium(atom) for atom in molecule.GetAtoms()
    )

    input_charge = _input_formal_charge(molecule)
    input_stereo = _stereo_signature(molecule, survivors)
    changed_atoms: list[dict[str, Any]] = []

    # Single application path: the folded copy carries exact per-atom
    # NoImplicit + NumExplicitHs semantics, so every site edit is an exact
    # count/charge assignment (no implicit-valence recomputation is relied
    # upon).  Hydrogens are re-materialized only for inputs that carried
    # explicit protium; implicit-style inputs keep equivalent exact-count
    # atoms (identical canonical SMILES).
    output, kept_unfoldable = _fold_explicit_hydrogens(molecule)
    if kept_unfoldable:
        warnings.append(WARNING_UNFOLDABLE_HYDROGEN)
    for site in sites:
        atom = output.GetAtomWithIdx(site["index"])
        old_charge = atom.GetFormalCharge()
        old_hydrogens = atom.GetTotalNumHs()
        atom.SetFormalCharge(site["charge"])
        atom.SetNumExplicitHs(site["hydrogens"])
        changed_atoms.append({
            "atom_index": site["index"],
            "input_atom_index": survivors[site["index"]],
            "element": atom.GetSymbol(),
            "rule": site["rule"],
            "formal_charge": [old_charge, site["charge"]],
            "total_h": [old_hydrogens, site["hydrogens"]],
        })
    _sanitize_strict(output.GetMol())
    # Verify the exact hydrogen counts BEFORE materialization: after AddHs
    # the counts move into explicit atoms and GetTotalNumHs() reads zero.
    for entry, site in zip(changed_atoms, sites):
        atom = output.GetAtomWithIdx(site["index"])
        if atom.GetTotalNumHs() != site["hydrogens"]:
            raise ProtonationPolicyError(
                "hydrogen bookkeeping mismatch at atom "
                f"{site['index']}: expected {site['hydrogens']} total H, "
                f"got {atom.GetTotalNumHs()}"
            )
    if has_foldable_protium:
        output = Chem.AddHs(
            output.GetMol(), addCoords=output.GetNumConformers() > 0
        )
        _sanitize_strict(output)
        for site in sites:
            atom = output.GetAtomWithIdx(site["index"])
            attached = sum(
                1
                for neighbour in atom.GetNeighbors()
                if neighbour.GetAtomicNum() == 1 and neighbour.GetIsotope() < 2
            )
            if attached + atom.GetTotalNumHs() != site["hydrogens"]:
                raise ProtonationPolicyError(
                    "materialized hydrogen mismatch at atom "
                    f"{site['index']}: expected {site['hydrogens']} H, "
                    f"got {attached + atom.GetTotalNumHs()}"
                )
    else:
        output = output.GetMol()

    # Integrity checks: survivor order, bonds, fragments, coordinates, tags.
    if _heavy_signature(output) != _heavy_signature(molecule):
        raise ProtonationPolicyError("heavy-atom order changed")
    output_survivors = _survivor_indices(output)
    if len(output_survivors) != len(survivors):
        raise ProtonationPolicyError("survivor count changed")
    if _heavy_bond_topology(molecule, survivors) != _heavy_bond_topology(
        output, output_survivors
    ):
        raise ProtonationPolicyError("survivor bond topology changed")
    if len(Chem.GetMolFrags(output)) != len(Chem.GetMolFrags(molecule)):
        raise ProtonationPolicyError("fragment count changed")

    old_positions = _heavy_positions(molecule)
    new_positions = _heavy_positions(output)
    heavy_coordinates_unchanged = old_positions == new_positions
    if not heavy_coordinates_unchanged:
        raise ProtonationPolicyError("heavy-atom coordinates changed")

    output_stereo = _stereo_signature(output, output_survivors)
    tags_changed, cip_changed = _tags_changed(input_stereo, output_stereo)
    if tags_changed:
        raise ProtonationPolicyError("stereo tags changed")
    if cip_changed:
        # CIP labels may legitimately reorder when a formal-charge change
        # reorders Cahn-Ingold priorities; tags themselves survived.
        warnings.append(WARNING_STEREO_CIP_CHANGED)

    report = {
        "policy": POLICY_NAME,
        "ph": POLICY_PH,
        "rule_basis": (
            "documented dominant-microstate defaults; not experimentally "
            "validated and not site-specific pKa predictions"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "input_formal_charge": input_charge,
        "output_formal_charge": _input_formal_charge(output),
        "changed_atoms": changed_atoms,
        "indexing": (
            "atom_index: folded/output atom index (heavy relative order); "
            "input_atom_index: index in the original input molecule"
        ),
        "conservative_neutral_sites": {
            "histidine_imidazole_like": neutral_imidazole,
            "thiol": thiols,
            "phenol": phenols,
        },
        "preserved_charged_histidine_like": charged_imidazole,
        "skipped_isotope_hydrogen_sites": skipped_isotope,
        "explicit_hydrogens_rebuilt": bool(has_foldable_protium),
        "preservation": {
            "heavy_atom_relative_order": True,
            "heavy_bond_topology": True,
            "fragment_count": [
                len(Chem.GetMolFrags(molecule)),
                len(Chem.GetMolFrags(output)),
            ],
            "heavy_coordinates_unchanged": True,
            "stereo_tags_unchanged": True,
        },
        "warnings": warnings,
    }
    return output, report


def protonate_ph74(smiles: str) -> str:
    """Set the documented dominant pH 7.4 protonation state.

    Carboxylic, sulfonic, and phosphoric acids are deprotonated. Aliphatic
    amines, guanidines, and amidines are protonated. Histidine imidazole,
    cysteine thiol, and tyrosine phenol remain neutral. The input is returned
    unchanged when parsing fails or the strict policy application fails; this
    legacy facade preserves its historical contract, so callers that need
    failures to surface must use :func:`protonate_molecule_ph74` directly.
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return smiles
        output, _report = protonate_molecule_ph74(molecule)
        return Chem.MolToSmiles(output)
    except Exception:  # noqa: BLE001 - legacy passthrough contract
        return smiles
