"""Public representation API preserves monomer and connection semantics."""
from __future__ import annotations

import pytest
from rdkit import Chem

from cycpep_master.compare import compare_strict
from cycpep_master.representations import (
    biln_to_helm,
    biln_to_map,
    biln_to_smiles,
    helm_to_biln,
    helm_to_map,
    helm_to_smiles,
    map_to_biln,
    map_to_helm,
    map_to_smiles,
)


CASES = [
    "AAAAA{cyc:N-C}",
    "CAAAC{cyc:1:R3-5:R3}",
    "CAAAC{cyc:N-C}{cyc:1:R3-5:R3}",
    "AC{br}CG{cyc:2:R3-3:R3}",
]


@pytest.mark.parametrize("source", CASES)
def test_map_helm_biln_roundtrip_preserves_canonical_map(source):
    helm = map_to_helm(source)
    biln = helm_to_biln(helm)
    assert helm_to_map(helm) == source
    assert biln_to_map(biln) == source
    assert map_to_biln(source) == biln
    assert biln_to_helm(biln) == helm


@pytest.mark.parametrize("source", CASES)
def test_all_serializations_assemble_same_strict_molecular_identity(source):
    helm = map_to_helm(source)
    biln = map_to_biln(source)
    map_smiles = map_to_smiles(source)
    helm_smiles = helm_to_smiles(helm)
    biln_smiles = biln_to_smiles(biln)
    assert Chem.MolFromSmiles(map_smiles) is not None
    assert compare_strict(map_smiles, helm_smiles).strict_graph_match is True
    assert compare_strict(map_smiles, biln_smiles).strict_graph_match is True


@pytest.mark.parametrize(
    ("source", "fragment_count"),
    [("A", 1), ("A{br}G", 2), ("A{br}GG", 2)],
)
def test_isolated_monomers_and_chain_segments_assemble(source, fragment_count):
    helm = map_to_helm(source)
    biln = map_to_biln(source)
    map_smiles = map_to_smiles(source)
    helm_smiles = helm_to_smiles(helm)
    biln_smiles = biln_to_smiles(biln)

    molecule = Chem.MolFromSmiles(map_smiles)
    assert molecule is not None
    assert len(Chem.GetMolFrags(molecule)) == fragment_count
    assert compare_strict(map_smiles, helm_smiles).strict_graph_match is True
    assert compare_strict(map_smiles, biln_smiles).strict_graph_match is True


@pytest.mark.parametrize(
    "function,value",
    [
        (helm_to_map, "PEPTIDE1{A.A}$$$$V2.0 trailing"),
        (map_to_helm, "AA{cyc:1:R4-2:R2}"),
        (biln_to_helm, "A(1,3)-A"),
        (map_to_smiles, "AA{cyc:1:R4-2:R2}"),
        (map_to_smiles, "AG{cyc:1:R2-2:R1}"),
    ],
)
def test_public_api_fails_closed_on_invalid_serializations(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_registered_multichar_monomer_roundtrips_across_notations():
    source = "{nnr:DAB}A"
    helm = map_to_helm(source)
    biln = helm_to_biln(helm)

    assert helm == "PEPTIDE1{[DAB].A}$$$$V2.0"
    assert biln == "[DAB]-A"
    assert helm_to_map(helm) == source
    assert biln_to_map(biln) == source
    assert biln_to_helm(biln) == helm


def test_terminal_caps_roundtrip_through_bracketed_biln_tokens():
    source = "{nt:ACE}A{ct:NME}"
    canonical_map = "A{nt:ACE}{ct:NME}"
    helm = map_to_helm(source)
    biln = helm_to_biln(helm)

    assert biln == "[ac]-A-[nme]"
    assert helm_to_map(helm) == canonical_map
    assert biln_to_map(biln) == canonical_map
    assert biln_to_helm(biln) == helm


@pytest.mark.parametrize("symbol", ["Phe(3-Cl)", "Et-Gly", "Ser(Ph(2-Cl))"])
def test_punctuated_monomer_names_roundtrip_through_biln(symbol):
    helm = f"PEPTIDE1{{[{symbol}].A}}$$$$V2.0"
    biln = helm_to_biln(helm)

    assert biln == f"[{symbol}]-A"
    assert biln_to_helm(biln) == helm


def test_map_edges_use_uncapped_residue_positions_and_shift_for_helm():
    source = "{nt:ACE}CA{cyc:1:R3-2:R2}"
    helm = map_to_helm(source)

    assert "2:R3-3:R2" in helm
    assert helm_to_map(helm) == "CA{cyc:1:R3-2:R2}{nt:ACE}"


def test_terminal_cap_cannot_leave_occupied_backbone_port_available():
    with pytest.raises(ValueError, match="R1 is occupied"):
        map_to_helm("{nt:ACE}AA{cyc:1:R1-2:R2}")


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{A.[ac].C}$$$$V2.0",
        "PEPTIDE1{A.[nme].C}$$$$V2.0",
        "PEPTIDE1{[ac].[ac].A}$$$$V2.0",
        "PEPTIDE1{A.[nme].[nh2]}$$$$V2.0",
    ],
)
def test_helm_rejects_misplaced_or_duplicate_terminal_caps(helm):
    with pytest.raises(ValueError, match="terminal cap|multiple C-terminal"):
        helm_to_map(helm)
