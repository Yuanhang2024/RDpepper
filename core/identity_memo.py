"""Request-local molecular identity memoization for strict reconstruction."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from rdkit import Chem


@dataclass(frozen=True, slots=True)
class MolecularIdentity:
    input_smiles: str
    canonical_smiles: str
    full_inchikey: str
    inchi_connectivity_block: str
    inchi_second_block: str
    inchi_nonprotonation_key: str
    inchi_protonation_flag: str
    heavy_atom_composition: tuple[tuple[str, int], ...]
    formal_charge: int


_ACTIVE_IDENTITY_MEMO: ContextVar[
    dict[str, MolecularIdentity | None] | None
] = ContextVar("cycpep_identity_memo", default=None)


@contextmanager
def identity_memo_context() -> Iterator[dict[str, MolecularIdentity | None]]:
    active = _ACTIVE_IDENTITY_MEMO.get()
    if active is not None:
        yield active
        return
    memo: dict[str, MolecularIdentity | None] = {}
    token = _ACTIVE_IDENTITY_MEMO.set(memo)
    try:
        yield memo
    finally:
        _ACTIVE_IDENTITY_MEMO.reset(token)


def molecular_identity(smiles: str | None) -> MolecularIdentity | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    memo = _ACTIVE_IDENTITY_MEMO.get()
    if memo is not None and smiles in memo:
        return memo[smiles]
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            identity = None
        else:
            canonical = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )
            full_key = Chem.MolToInchiKey(molecule)
            blocks = full_key.split("-") if full_key else []
            if len(blocks) != 3:
                identity = None
            else:
                composition: dict[str, int] = {}
                for atom in molecule.GetAtoms():
                    if atom.GetAtomicNum() <= 1:
                        continue
                    symbol = atom.GetSymbol().upper()
                    composition[symbol] = composition.get(symbol, 0) + 1
                identity = MolecularIdentity(
                    input_smiles=smiles,
                    canonical_smiles=canonical,
                    full_inchikey=full_key,
                    inchi_connectivity_block=blocks[0],
                    inchi_second_block=blocks[1],
                    inchi_nonprotonation_key="-".join(blocks[:2]),
                    inchi_protonation_flag=blocks[2],
                    heavy_atom_composition=tuple(sorted(composition.items())),
                    formal_charge=int(Chem.GetFormalCharge(molecule)),
                )
    except Exception:
        identity = None
    if memo is not None:
        memo[smiles] = identity
        if identity is not None:
            memo.setdefault(identity.canonical_smiles, identity)
    return identity
