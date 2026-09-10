from __future__ import annotations

from types import SimpleNamespace

from rdkit import Chem

from cycpep_master import bond_order_inference, result_first
from cycpep_master.export.conformer import (
    _candidate_graph_to_mol,
    _mol2_roundtrip_full_inchikey,
    _result_first_candidate_to_mol2,
)
from cycpep_master.remediation_v5 import StrictReconstructionResult


SIMPLE_PDB = """\
HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  O1  LIG L   1       1.430   0.000   0.000  1.00  0.00           O
CONECT    1    2
CONECT    2    1
END
"""


def _graph():
    return {
        "schema_version": "chemical-graph-1",
        "atoms": [
            {
                "serial": 1,
                "name": "C1",
                "residue": "LIG",
                "chain": "L",
                "residue_number": 1,
                "insertion_code": "",
                "element": "C",
                "xyz": [0.0, 0.0, 0.0],
                "formal_charge": 0,
            },
            {
                "serial": 2,
                "name": "O1",
                "residue": "LIG",
                "chain": "L",
                "residue_number": 1,
                "insertion_code": "",
                "element": "O",
                "xyz": [1.43, 0.0, 0.0],
                "formal_charge": 0,
            },
        ],
        "bonds": [
            {
                "a": 1,
                "b": 2,
                "order": 1.0,
                "is_aromatic": False,
                "stereo": "STEREONONE",
            }
        ],
    }


def _report(*, quality="candidate", rigor="L2:H"):
    molecule = Chem.MolFromSmiles("CO")
    key = Chem.MolToInchiKey(molecule)
    group = {
        "candidate_id": "candidate-1",
        "canonical_smiles": "CO",
        "full_inchikey": key,
        "inchi_connectivity_block": key.split("-", 1)[0],
        "inchi_nonprotonation_key": "-".join(key.split("-")[:2]),
        "formal_charge": 0,
        "heavy_atom_composition": {"C": 1, "O": 1},
        "supporting_engines": ["openbabel_pdb"],
        "supporting_families": ["bond_order_perception"],
        "engine_count": 1,
        "independent_family_count": 1,
        "evidence_class": "single_engine_bond_order_perception",
        "quality": quality,
        "rigor": rigor,
        "qualified_chemistry": False,
        "full_inchikey_variants": [key],
        "canonical_smiles_variants": ["CO"],
        "selected_engine": "openbabel_pdb",
        "candidate_graph": _graph(),
        "mapping_audit": {"source_atom_mapping_complete": True},
    }
    public = {
        field: value
        for field, value in group.items()
        if field != "candidate_graph"
    }
    return {
        "schema_version": "bond-order-inference-1",
        "status": "parseable",
        "source_heavy_atom_composition": {"C": 1, "O": 1},
        "candidate_count": 1,
        "selected_candidate_id": "candidate-1",
        "selected_candidate": group,
        "selection_tied": False,
        "identity_groups": [public],
        "engine_attempts": [],
        "openbabel_available": True,
        "qualified_chemistry": False,
    }


def _prepared(tmp_path):
    path = tmp_path / "simple.pdb"
    path.write_text(SIMPLE_PDB, encoding="ascii")
    return SimpleNamespace(
        pdb_path=path,
        chain_id="L",
        source_format="pdb",
        audit={"normalized_heavy_atom_count": 2},
    )


def _install_strict(monkeypatch):
    strict = StrictReconstructionResult(
        status="rejected",
        support_status="supported",
        qualified_success=False,
        warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
        path_used="V6_EVIDENCE_DIMENSION_AUDIT",
        output_evidence={},
    )
    monkeypatch.setattr(
        result_first.remediation_v6,
        "reconstruct_prepared_structure_fail_closed_v6",
        lambda *args, **kwargs: strict,
    )


def test_bond_order_inference_is_explicit_opt_in(
    tmp_path, monkeypatch
):
    _install_strict(monkeypatch)
    monkeypatch.setattr(
        bond_order_inference,
        "infer_bond_order_candidates",
        lambda *args, **kwargs: _report(),
    )

    default = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path)
    )
    inferred = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path), infer_bond_orders=True
    )

    assert default.candidate_smiles is None
    assert inferred.quality == result_first.QUALITY_CANDIDATE
    assert inferred.smiles is None
    assert inferred.candidate_smiles == "CO"
    assert inferred.candidate_rigor == "L2:H"
    assert inferred.candidate_graph["bonds"][0]["order"] == 1.0
    assert "INFERRED_CHEMISTRY_UNQUALIFIED" in inferred.warning_codes


def test_multi_family_high_candidate_uses_existing_smiles_handoff(
    tmp_path, monkeypatch
):
    _install_strict(monkeypatch)
    monkeypatch.setattr(
        bond_order_inference,
        "infer_bond_order_candidates",
        lambda *args, **kwargs: _report(quality="high", rigor="L2:R"),
    )

    inferred = result_first.reconstruct_prepared_structure(
        _prepared(tmp_path), infer_bond_orders=True
    )

    assert inferred.quality == result_first.QUALITY_HIGH
    assert inferred.smiles == "CO"
    assert inferred.candidate_smiles == "CO"
    assert inferred.candidate_rigor == "L2:R"
    assert inferred.strict_result.qualified_success is False


def test_source_heavy_atom_mismatch_is_not_admitted(
    tmp_path, monkeypatch
):
    path = tmp_path / "simple.pdb"
    path.write_text(SIMPLE_PDB, encoding="ascii")
    monkeypatch.setattr(
        "cycpep_master.paths.path_g.generate_g",
        lambda *args, **kwargs: ("CCC", None),
    )
    monkeypatch.setattr(
        "cycpep_master.paths.path_h.generate_h",
        lambda *args, **kwargs: (None, "no candidate"),
    )
    monkeypatch.setattr(
        bond_order_inference,
        "_prepare_openbabel_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ImportError("Open Babel unavailable")
        ),
    )
    monkeypatch.setattr(
        bond_order_inference,
        "_prepare_rdkit_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("RDKit candidate unavailable")
        ),
    )
    from cycpep_master.core.geometry_candidate import GeometryCandidateError
    monkeypatch.setattr(
        "cycpep_master.core.geometry_candidate.build_geometry_simple_molecule",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            GeometryCandidateError("geometry candidate disabled in mismatch test")
        ),
    )

    report = bond_order_inference.infer_bond_order_candidates(
        path,
        "L",
        strict_candidates=[
            {
                "canonical_smiles": "CCC",
                "routes": ["a"],
                "supporting_route_count": 1,
            }
        ],
    )

    assert report["status"] == "not_supported"
    assert report["candidate_count"] == 0
    assert sum(
        attempt.get("reason") == "source_heavy_atom_composition_mismatch"
        for attempt in report["engine_attempts"]
    ) == 2


def test_candidate_graph_mol2_preserves_source_coordinates_and_connectivity():
    result = SimpleNamespace(
        candidate_smiles="CO",
        candidate_graph=_graph(),
        smiles=None,
    )

    block, error = _result_first_candidate_to_mol2(result)

    assert error is None
    assert block is not None
    roundtrip_key, roundtrip_error = _mol2_roundtrip_full_inchikey(block)
    expected_key = Chem.MolToInchiKey(Chem.MolFromSmiles("CO"))
    assert roundtrip_error is None
    assert roundtrip_key.split("-", 1)[0] == expected_key.split("-", 1)[0]
    atom_lines = block.split("@<TRIPOS>ATOM\n", 1)[1].split(
        "@<TRIPOS>UNITY_ATOM_ATTR", 1
    )[0].split("@<TRIPOS>BOND", 1)[0].strip().splitlines()
    heavy = [line.split() for line in atom_lines[:2]]
    assert [float(value) for value in heavy[0][2:5]] == [0.0, 0.0, 0.0]
    assert [float(value) for value in heavy[1][2:5]] == [1.43, 0.0, 0.0]


def test_aromatic_heteroatom_state_survives_candidate_graph_roundtrip():
    source = Chem.MolFromSmiles("c1cc[nH]c1")
    atoms = []
    for index, atom in enumerate(source.GetAtoms(), 1):
        atoms.append(
            {
                "serial": index,
                "name": f"{atom.GetSymbol()}{index}",
                "residue": "LIG",
                "chain": "L",
                "residue_number": 1,
                "insertion_code": "",
                "element": atom.GetSymbol(),
                "xyz": [float(index), 0.0, 0.0],
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "num_explicit_hs": atom.GetNumExplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "num_radical_electrons": atom.GetNumRadicalElectrons(),
                "is_aromatic": atom.GetIsAromatic(),
            }
        )
    bonds = [
        {
            "a": bond.GetBeginAtomIdx() + 1,
            "b": bond.GetEndAtomIdx() + 1,
            "order": bond.GetBondTypeAsDouble(),
            "is_aromatic": bond.GetIsAromatic(),
            "stereo": str(bond.GetStereo()),
        }
        for bond in source.GetBonds()
    ]

    molecule, error = _candidate_graph_to_mol(
        {"atoms": atoms, "bonds": bonds},
        Chem.MolToSmiles(source),
    )

    assert error is None
    assert molecule is not None
    assert Chem.MolToInchiKey(molecule).split("-", 1)[0] == (
        Chem.MolToInchiKey(source).split("-", 1)[0]
    )
