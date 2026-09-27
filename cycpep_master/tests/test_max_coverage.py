"""Tests for the max-coverage fallback primitives (V7 proposal)."""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.max_coverage import (
    MAX_COVERAGE,
    STRICT_V6,
    CoordinateTier,
    apply_coordinate_tier,
    coerce_policy,
    diagnose_input,
    salvage_notation_prefix,
    salvage_payload,
)


# ------------------------------------------------------------------ policy

def test_policy_default_is_strict_v6():
    assert coerce_policy(None) == STRICT_V6
    assert coerce_policy(MAX_COVERAGE) == MAX_COVERAGE
    with pytest.raises(ValueError):
        coerce_policy("yolo")


# ------------------------------------------------------------- class A

def _smiles_parser(text: str):
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError("SMILES not parseable")
    return mol


def test_diagnose_input_reports_prefix_and_position():
    d = diagnose_input("CCO(C", parser=_smiles_parser)
    assert d.status == "degraded_input"
    assert d.error_kind == "content_parse_error"
    assert d.salvageable_prefix == "CCO"
    assert d.failure_position == 3
    assert "parseable" in d.message


def test_diagnose_input_passes_clean_payload():
    d = diagnose_input("CCO", parser=_smiles_parser)
    assert d.error_kind == "none"
    assert d.salvageable_prefix == "CCO"


def test_diagnose_input_keeps_parameter_errors_hard():
    with pytest.raises(ValueError):
        diagnose_input("", parser=_smiles_parser)


# ------------------------------------------------------------- class B

def _biln_tokens(text: str):
    return [t for t in text.split("-") if t]


def _known_monomer(token: str):
    if token not in {"Ac", "Aib", "K", "NH2", "G", "P"}:
        raise ValueError(f"unknown monomer {token!r}")
    return token


def test_salvage_notation_prefix_replaces_bad_tokens():
    s = salvage_notation_prefix(
        "Ac-Aib-Cxxx-K-NH2", tokenizer=_biln_tokens, parse_token=_known_monomer)
    assert s.status == "notation_partial"
    assert s.tokens == ["Ac", "Aib", "\u0000UNK", "K", "NH2"]
    assert len(s.placeholders) == 1
    assert s.placeholders[0]["position"] == 2
    assert s.placeholders[0]["raw_token"] == "Cxxx"
    assert s.salvaged_monomer_count == 4


def test_salvage_notation_prefix_clean_returns_complete():
    s = salvage_notation_prefix(
        "Ac-P-G-K-NH2", tokenizer=_biln_tokens, parse_token=_known_monomer)
    assert s.status == "notation_complete"
    assert s.placeholders == []


# ------------------------------------------------------------- class D

def _ethanol_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def test_coordinate_tier_x3_full_mapping_zero_displacement():
    mol = _ethanol_conformer()
    conf = mol.GetConformer()
    coords = {i: conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())}
    source = {i: (p.x + 1.0, p.y, p.z) for i, p in coords.items()}
    tier = apply_coordinate_tier(conf, mol.GetNumAtoms(), source)
    assert isinstance(tier, CoordinateTier)
    assert tier.tier == "X3"
    assert tier.generated_atom_count == 0
    after = conf.GetAtomPosition(0)
    assert (after.x, after.y, after.z) == pytest.approx(source[0])


def test_coordinate_tier_x2_partial_mapping_lists_generated():
    mol = _ethanol_conformer()
    conf = mol.GetConformer()
    source = {0: (1.0, 2.0, 3.0)}
    tier = apply_coordinate_tier(conf, mol.GetNumAtoms(), source)
    assert tier.tier == "X2"
    assert tier.mapped_atom_count == 1
    assert tier.generated_atom_indices == [i for i in range(mol.GetNumAtoms()) if i != 0]
    assert conf.GetAtomPosition(0).x == pytest.approx(1.0)


def test_coordinate_tier_x1_no_mapping_keeps_embedding():
    mol = _ethanol_conformer()
    conf = mol.GetConformer()
    before = conf.GetAtomPosition(2)
    tier = apply_coordinate_tier(conf, mol.GetNumAtoms(), None)
    assert tier.tier == "X1"
    assert tier.generated_atom_count == mol.GetNumAtoms()
    after = conf.GetAtomPosition(2)
    assert (after.x, after.y, after.z) == (before.x, before.y, before.z)


def test_coordinate_tier_out_of_range_indices_ignored():
    mol = _ethanol_conformer()
    conf = mol.GetConformer()
    source = {0: (0.0, 0.0, 0.0), 99: (5.0, 5.0, 5.0)}
    tier = apply_coordinate_tier(conf, mol.GetNumAtoms(), source)
    assert tier.tier == "X2"
    assert 99 not in tier.generated_atom_indices


# ------------------------------------------------------------- class E

def test_salvage_payload_attaches_fragment():
    state = {"status": "unrecoverable", "error": "X"}
    out = salvage_payload(state, partial={"monomers": ["A", "B"]})
    assert out["salvage"] == {"monomers": ["A", "B"]}
    assert state == {"status": "unrecoverable", "error": "X"}  # original untouched


def test_salvage_payload_without_partial_is_identity():
    state = {"status": "audit_failed"}
    assert salvage_payload(state, partial=None) is state
