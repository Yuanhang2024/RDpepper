"""Deterministic dominant-state protonation rules used for docking."""

from rdkit import Chem


def protonate_ph74(smiles: str) -> str:
    """Set the documented dominant pH 7.4 protonation state.

    Carboxylic, sulfonic, and phosphoric acids are deprotonated. Aliphatic
    amines, guanidines, and amidines are protonated. Histidine imidazole,
    cysteine thiol, and tyrosine phenol remain neutral. The input is returned
    unchanged if parsing or sanitization fails.
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return smiles
        editable = Chem.RWMol(molecule)

        def set_state(index, charge, hydrogens):
            atom = editable.GetAtomWithIdx(index)
            atom.SetFormalCharge(charge)
            atom.SetNumExplicitHs(hydrogens)

        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        ):
            set_state(match[2], -1, 0)
        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")
        ):
            set_state(match[3], -1, 0)
        for phosphorus in molecule.GetAtoms():
            if phosphorus.GetAtomicNum() != 15:
                continue
            acidic_oxygens = []
            for oxygen in phosphorus.GetNeighbors():
                bond = molecule.GetBondBetweenAtoms(
                    phosphorus.GetIdx(), oxygen.GetIdx()
                )
                if (
                    oxygen.GetAtomicNum() == 8
                    and bond.GetBondType() == Chem.BondType.SINGLE
                    and oxygen.GetDegree() == 1
                    and (
                        oxygen.GetTotalNumHs() > 0
                        or oxygen.GetFormalCharge() == -1
                    )
                ):
                    acidic_oxygens.append(oxygen)
            target_negative = min(2, len(acidic_oxygens))
            already_negative = sum(
                oxygen.GetFormalCharge() == -1
                for oxygen in acidic_oxygens
            )
            for oxygen in sorted(
                acidic_oxygens, key=lambda atom: atom.GetIdx()
            ):
                if already_negative >= target_negative:
                    break
                if oxygen.GetFormalCharge() == 0:
                    set_state(oxygen.GetIdx(), -1, 0)
                    already_negative += 1

        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(Na)][CX4]")
        ):
            set_state(match[0], 1, 3)
        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts(
                "[NX3;H1;!$(NC=O);!$(N=*);!$([nX3])]([CX4])[CX4]"
            )
        ):
            set_state(match[0], 1, 2)
        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts(
                "[NX3;H0;!$(NC=O);!$(N=*);!$([nX3])]([CX4])([CX4])[CX4]"
            )
        ):
            set_state(match[0], 1, 0)
        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts("[NX2]=[CX3]([NX3])[NX3]")
        ):
            atom = editable.GetAtomWithIdx(match[0])
            atom.SetFormalCharge(1)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        for match in molecule.GetSubstructMatches(
            Chem.MolFromSmarts("[NX2]=[CX3][NX3;!$(NC=O)]")
        ):
            atom = editable.GetAtomWithIdx(match[0])
            if atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)

        output = editable.GetMol()
        Chem.SanitizeMol(output)
        return Chem.MolToSmiles(output)
    except Exception:
        return smiles
