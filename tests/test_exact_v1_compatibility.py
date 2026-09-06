from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cycpep_master.exact_v1 import ABSTAIN, map_to_exact_v1


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    PACKAGE_ROOT
    / "benchmarks"
    / "exact_v1_compatibility_report.json"
)
RUNNER = (
    PACKAGE_ROOT
    / "benchmarks"
    / "exact_v1_compatibility.py"
)
INDEX = (
    PACKAGE_ROOT
    / "data"
    / "templates"
    / "templates_index.json"
)
EXACT_IMPLEMENTATION = PACKAGE_ROOT / "exact_v1.py"
GRAPH_IMPLEMENTATION = (
    PACKAGE_ROOT / "core" / "cyclic_peptide_graph.py"
)
EXACT_SCHEMA = PACKAGE_ROOT / "schemas" / "exact_v1.schema.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_v1_compatibility_report_is_hash_bound_and_closed():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    overall = report["overall"]

    assert report["runner_sha256"] == _sha256(RUNNER)
    assert report["input_index_sha256"] == _sha256(INDEX)
    assert report["exact_implementation_sha256"] == _sha256(
        EXACT_IMPLEMENTATION
    )
    assert report["graph_implementation_sha256"] == _sha256(
        GRAPH_IMPLEMENTATION
    )
    assert report["exact_schema_sha256"] == _sha256(EXACT_SCHEMA)
    assert overall["records"] == 794
    assert overall["exact_records"] == 793
    assert overall["all_notation_roundtrip_records"] == 793
    assert overall["nnaa_records"] == 45
    assert overall["nnaa_exact_records"] == 44
    assert overall["multiring_records"] == 9
    assert overall["multiring_exact_records"] == 9
    assert sum(
        values["records"]
        for values in report["by_source"].values()
    ) == 794
    keys = [row["template_key"] for row in report["rows"]]
    assert len(keys) == len(set(keys)) == 794
    abstained = [
        row for row in report["rows"] if row["exact"] is not True
    ]
    assert [
        (
            row["template_key"],
            row["reason_codes"],
        )
        for row in abstained
    ] == [(
        "15_none_syn_111",
        [
            "MONOMER_STEREO_UNRESOLVED",
            "CHAIN_MONOMER_RESOLUTION_INCOMPLETE",
        ],
    )]


def test_edge_projection_is_strictly_bounded_in_report():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    overall = report["overall"]

    assert overall["edge_v1_projected_records"] == 326
    assert overall["edge_v1_projected_records"] < (
        overall["exact_records"]
    )
    assert overall["normalization_counts"] == {
        "LEGACY_CAP_INCLUSIVE_POSITION_NORMALIZED": 467
    }


def test_the_sole_nonexact_template_is_a_literal_abstention():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    document = map_to_exact_v1(index["15_none_syn_111"]["map"])

    assert document["exactness_status"] == ABSTAIN
    assert document["reason_codes"] == [
        "MONOMER_STEREO_UNRESOLVED",
        "CHAIN_MONOMER_RESOLUTION_INCOMPLETE",
    ]
