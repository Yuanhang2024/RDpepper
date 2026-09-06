"""Amino acid and capping group SMILES templates."""

AA_SMILES = {
    'ALA': 'N[C@@H](C)C(=O)', 'ARG': 'N[C@@H](CCCNC(=[NH2+])N)C(=O)',
    'ASN': 'N[C@@H](CC(N)=O)C(=O)', 'ASP': 'N[C@@H](CC(=O)O)C(=O)',
    'CYS': 'N[C@@H](CS)C(=O)', 'GLN': 'N[C@@H](CCC(N)=O)C(=O)',
    'GLU': 'N[C@@H](CCC(=O)O)C(=O)', 'GLY': 'NCC(=O)',
    'HIS': 'N[C@@H](Cc1c[nH]c[nH+]1)C(=O)', 'ILE': 'N[C@@H]([C@@H](C)CC)C(=O)',
    'LEU': 'N[C@@H](CC(C)C)C(=O)', 'LYS': 'N[C@@H](CCCC[NH3+])C(=O)',
    'MET': 'N[C@@H](CCSC)C(=O)', 'PHE': 'N[C@@H](Cc1ccccc1)C(=O)',
    'PRO': 'N1CCC[C@H]1C(=O)', 'SER': 'N[C@@H](CO)C(=O)',
    'THR': 'N[C@@H]([C@@H](C)O)C(=O)', 'TRP': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)',
    'TYR': 'N[C@@H](Cc1ccc(O)cc1)C(=O)', 'VAL': 'N[C@@H](C(C)C)C(=O)',
}

CAP_SMILES = {'ACE': 'CC(=O)', 'NME': 'NC'}

ALL_SMILES = {}
ALL_SMILES.update(AA_SMILES)
ALL_SMILES.update(CAP_SMILES)

AC_AA_SMILES = {aa: f'CC(=O){smi}' for aa, smi in AA_SMILES.items()}
AA_NHME_SMILES = {aa: smi[:-5] + 'C(=O)NC' for aa, smi in AA_SMILES.items()}
