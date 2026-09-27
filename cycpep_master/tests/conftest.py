"""Shared pytest fixtures and helpers for cycpep_master tests."""
import glob
import os

import pytest
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Repo layout: this file is cycpep_master/tests/conftest.py
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CYCPEP_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_CYCPEP_DIR)
_PDB_DIR = os.path.join(_REPO_ROOT, "TestFiles_Example", "CPBind_Examples")


def canonical(smiles):
    """Canonical isomeric SMILES, or None if unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def count_smarts(smiles, smarts):
    """Number of substructure matches of `smarts` in `smiles` (0 if no parse)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return 0
    patt = Chem.MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(patt))


def has_dummy(smiles):
    """True if the molecule still carries an uncapped dummy atom '*'."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return False
    return any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())


def gold_pdbs():
    """Sorted list of gold-standard CPBind PDB paths (may be empty)."""
    return sorted(glob.glob(os.path.join(_PDB_DIR, "*.pdb")))


@pytest.fixture(scope="session")
def pdb_files():
    files = gold_pdbs()
    if not files:
        pytest.skip(f"no gold-standard PDBs found under {_PDB_DIR}")
    return files
