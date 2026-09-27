from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from rdkit import Chem

from cycpep_master.remediation_v6 import _candidate_assessment


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _row(route: str, smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "route": route,
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": Chem.MolToInchiKey(molecule),
    }


def _validator():
    schema = json.loads(
        (PACKAGE_ROOT / "schemas" / "candidate_assessment.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize(
    ("rows", "mode"),
    [
        ([], "no_candidate"),
        ([_row("a", "N[C@@H](C)C(=O)O")], "candidate_unique"),
        (
            [
                _row("a", "N[C@@H](C)C(=O)O"),
                _row("b", "N[C@H](C)C(=O)O"),
            ],
            "candidate_ensemble",
        ),
    ],
)
def test_candidate_assessment_examples_validate(rows, mode):
    assessment = _candidate_assessment(
        rows, {}, status="rejected", qualified_success=False
    )
    _validator().validate(assessment)
    assert assessment["mode"] == mode


def test_candidate_assessment_compatibility_aliases_are_equal():
    assessment = _candidate_assessment(
        [_row("a", "N[C@@H](C)C(=O)O")],
        {},
        status="rejected",
        qualified_success=False,
    )

    assert assessment["automatic_selection_permitted"] == assessment[
        "unattended_selection_qualified"
    ]
    assert assessment["invalid_candidate_routes"] == assessment[
        "inadmissible_candidate_routes"
    ]
