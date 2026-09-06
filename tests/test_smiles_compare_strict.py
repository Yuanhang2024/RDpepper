from cycpep_master.compare import compare, compare_strict


def test_strict_compare_accepts_equivalent_serializations_and_atom_maps():
    result = compare_strict("C[C@H](N)C(=O)O", "N[C@@H]([CH3:7])C(O)=O")
    assert result.status == "match"
    assert result.strict_graph_match is True
    assert result.full_inchikey_match is True
    assert result.connectivity_match is True
    assert result.molecular_formula_a == result.molecular_formula_b


def test_strict_compare_rejects_enantiomers_but_reports_same_connectivity():
    result = compare_strict("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")
    assert result.status == "mismatch"
    assert result.strict_graph_match is False
    assert result.full_inchikey_match is False
    assert result.connectivity_match is True


def test_strict_compare_rejects_charge_and_isotope_changes():
    charged = compare_strict("CC(=O)O", "CC(=O)[O-]")
    assert charged.strict_graph_match is False
    assert charged.full_inchikey_match is False

    isotopic = compare_strict("CC", "[13CH3]C")
    assert isotopic.strict_graph_match is False
    assert isotopic.full_inchikey_match is False
    assert isotopic.connectivity_match is True


def test_strict_compare_returns_invalid_input_instead_of_similarity():
    result = compare_strict("not-a-smiles", "CC")
    assert result.status == "invalid_input"
    assert result.strict_graph_match is None
    assert result.reason == "invalid SMILES"


def test_legacy_compare_remains_explicitly_permissive():
    legacy_match, _detail = compare(
        "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"
    )
    assert legacy_match is True
    assert compare_strict(
        "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"
    ).strict_graph_match is False
