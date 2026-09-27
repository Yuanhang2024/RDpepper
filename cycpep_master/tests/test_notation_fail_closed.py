"""Strict notation validation and multichain round-trip regressions."""

import pytest

from cycpep_master.paths._map_utils import (
    biln_to_helm,
    helm_to_biln,
    helm_to_map,
    map_to_helm,
)


@pytest.mark.parametrize("value", [
    "PEPTIDE1{A.G.V}",
    "PEPTIDE1{A.G.V}|PEPTIDE1{C.C}$$$$V2.0",
    "PEPTIDE1{}$$$$V2.0",
    "PEPTIDE1{A.G.V}$PEPTIDE9,PEPTIDE1,1:R1-3:R2$$$V2.0",
    "PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R4-3:R2$$$V2.0",
    "PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R1-99:R2$$$V2.0",
])
def test_invalid_helm_is_rejected(value):
    with pytest.raises(ValueError):
        helm_to_map(value)
    with pytest.raises(ValueError):
        helm_to_biln(value)


@pytest.mark.parametrize("value", [
    "A--C",
    "A-C.",
    "A(1,3)-C",
    "A(1,3)-C(1,3)-G(1,3)",
    "A(1,4)-C(1,3)",
    "A(1,3)-C(1,3)",
    "C(1,3)-A(1,3)-C(2,3)-C(2,3)",
])
def test_invalid_biln_is_rejected(value):
    with pytest.raises(ValueError):
        biln_to_helm(value)


@pytest.mark.parametrize("value", [
    "AGV{cyc:1:R1-3:R2",
    "AGV{cyc:}",
    "AGV{cyc:1:R1-3:R2}{cyc:1:R1-3:R2}",
    "AGV{cyc:1:R1-99:R2}",
    "AGV{br}",
    "{br}AGV",
])
def test_invalid_map_is_rejected(value):
    with pytest.raises(ValueError):
        map_to_helm(value)


def test_map_helm_map_preserves_multichain_connections():
    source = "AC{br}CA{cyc:2:R3-3:R3}"
    assert helm_to_map(map_to_helm(source)) == source


def test_three_chain_map_helm_map_preserves_two_connections():
    source = "AC{br}CAC{br}CC{cyc:2:R3-3:R3}{cyc:5:R3-7:R3}"
    assert helm_to_map(map_to_helm(source)) == source


def test_glutamate_r3_is_a_valid_registered_port():
    """The unified monomer library defines Glu R3; do not overfit a bad gold label."""
    result = map_to_helm("ELGYSRI{cyc:1:R3-7:R2}")
    assert "1:R3-7:R2" in result
