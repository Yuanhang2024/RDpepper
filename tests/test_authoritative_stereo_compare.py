from rdkit import Chem

from cycpep_master.compare import compare_specified_stereo


ALANINE_S = (
    "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1"
)
ALANINE_UNSPECIFIED = (
    "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2?"
)
THREONINE_PARTIAL = (
    "InChI=1S/C4H9NO3/c1-2(6)3(5)4(7)8/h2-3,6H,5H2,1H3,(H,7,8)"
    "/t2?,3-/m0/s1"
)


def test_matches_equivalent_specified_reference_stereo():
    result = compare_specified_stereo(ALANINE_S, "N[C@@H](C)C(=O)O")
    assert result.status == "match"
    assert result.nonstereo_inchikey_match is True
    assert result.specified_stereo_match is True
    assert result.reference_specified_atom_stereo_count == 1
    assert result.matched_specified_atom_stereo_count == 1
    assert result.matched_specified_stereo_fraction == 1.0


def test_rejects_opposite_and_missing_observed_assignment():
    opposite = compare_specified_stereo(ALANINE_S, "N[C@H](C)C(=O)O")
    missing = compare_specified_stereo(ALANINE_S, "NC(C)C(=O)O")
    assert opposite.status == "mismatch"
    assert opposite.specified_stereo_match is False
    assert opposite.matched_specified_atom_stereo_count == 0
    assert opposite.matched_specified_stereo_fraction == 0.0
    assert missing.status == "mismatch"
    assert missing.specified_stereo_match is False


def test_reference_question_mark_is_not_forced_or_counted_as_a_pass():
    left = compare_specified_stereo(ALANINE_UNSPECIFIED, "N[C@@H](C)C(=O)O")
    right = compare_specified_stereo(ALANINE_UNSPECIFIED, "N[C@H](C)C(=O)O")
    for result in (left, right):
        assert result.status == "not_comparable"
        assert result.nonstereo_inchikey_match is True
        assert result.specified_stereo_match is None
        assert result.reference_specified_atom_stereo_count == 0
        assert result.reference_unspecified_atom_stereo_count == 1


def test_partial_reference_ignores_question_mark_but_enforces_defined_centre():
    reference = Chem.MolFromInchi(THREONINE_PARTIAL)
    assert reference is not None
    base = Chem.MolToSmiles(reference, canonical=True, isomericSmiles=True)
    matches = compare_specified_stereo(THREONINE_PARTIAL, base)
    assert matches.status == "match"
    assert matches.reference_specified_atom_stereo_count == 1
    assert matches.reference_unspecified_atom_stereo_count == 1
    assert matches.matched_specified_atom_stereo_count == 1

    specified = next(
        atom for atom in reference.GetAtoms() if atom.HasProp("_CIPCode")
    )
    specified.InvertChirality()
    inverted = Chem.MolToSmiles(reference, canonical=True, isomericSmiles=True)
    mismatch = compare_specified_stereo(THREONINE_PARTIAL, inverted)
    assert mismatch.status == "mismatch"
    assert mismatch.matched_specified_atom_stereo_count == 0


def test_nonstereo_identity_mismatch_is_not_stereo_failure():
    result = compare_specified_stereo(ALANINE_S, "N[C@@H](CC)C(=O)O")
    assert result.status == "not_comparable"
    assert result.nonstereo_inchikey_match is False
    assert result.specified_stereo_match is None
    assert result.reason == "nonstereo_inchikey_mismatch"


def test_defined_double_bond_stereo_is_enforced():
    reference = Chem.MolToInchi(Chem.MolFromSmiles("C/C=C/C"))
    same = compare_specified_stereo(reference, "C/C=C/C")
    opposite = compare_specified_stereo(reference, "C/C=C\\C")
    assert same.status == "match"
    assert same.reference_specified_bond_stereo_count == 1
    assert same.matched_specified_bond_stereo_count == 1
    assert opposite.status == "mismatch"
    assert opposite.matched_specified_bond_stereo_count == 0


def test_invalid_input_is_explicit():
    result = compare_specified_stereo("not-inchi", "CC")
    assert result.status == "invalid_input"
    assert result.specified_stereo_match is None
    assert result.reason == "reference is not an InChI string"
