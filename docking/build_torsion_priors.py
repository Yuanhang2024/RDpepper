"""Build entity-weighted V4 torsion priors from full structure libraries."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from .torsion_observations import (
    CHI_DEFINITIONS,
    parse_structure_file,
)
from .torsion_prior import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
)


BUILD_SCHEMA_VERSION = "1.0.0-cycpep-torsion-build.2"
RIGID_SIGMA_DEG = 20.0
FALSE_RIGID_DEVIATION_DEG = 30.0
FALSE_RIGID_TAIL_MASS = 0.05
HISTOGRAM_BINS = 24
SOURCE_CONFIGS = (
    {
        "name": "CPBind",
        "source_class": "predicted_complex",
        "chain_id": "L",
        "structure_subdir": "CPBind_pdb",
        "index_relative": "CPBind_properties/CPBind_index.txt",
    },
    {
        "name": "CPSea_PDB",
        "source_class": "pdb_derived_complex",
        "chain_id": "L",
        "structure_subdir": "CPSea_PDB_pdb",
        "index_relative": "CPSea_PDB_index.txt",
    },
    {
        "name": "AfCycDesign",
        "source_class": "predicted_design",
        "chain_id": "A",
        "structure_subdir": None,
        "index_relative": None,
    },
)


OBSERVATION_SCHEMA = {
    "entity_key": "string",
    "full_inchikey": "string",
    "source_name": "string",
    "source_class": "string",
    "structure_id": "string",
    "model_id": "int32",
    "chain_id": "string",
    "sequence": "string",
    "residue_count": "int32",
    "topology_class": "string",
    "macrocycle_ring_size": "int32",
    "file_path": "string",
    "file_sha256": "string",
    "graph_route": "string",
    "torsion_name": "string",
    "torsion_kind": "string",
    "angle_deg": "float64",
    "quartet_atom_indices": "list<int32>",
    "central_bond_atom_indices": "list<int32>",
    "exact_key": "string",
    "residue_class_ring_key": "string",
    "morgan_key": "string",
    "generic_key": "string",
}
STRUCTURE_SCHEMA = {
    "entity_key": "string",
    "full_inchikey": "string",
    "source_name": "string",
    "source_class": "string",
    "structure_id": "string",
    "model_id": "int32",
    "chain_id": "string",
    "sequence": "string",
    "residue_count": "int32",
    "topology_class": "string",
    "macrocycle_ring_size": "int32",
    "file_path": "string",
    "file_sha256": "string",
    "graph_route": "string",
    "atom_count": "int32",
    "heavy_atom_count": "int32",
    "formal_charge": "int32",
    "closure_bond_count": "int32",
    "status": "string",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrow_schema(specification: dict[str, str]):
    import pyarrow as pa

    types = {
        "string": pa.string(),
        "int32": pa.int32(),
        "float64": pa.float64(),
        "list<int32>": pa.list_(pa.int32()),
    }
    return pa.schema([
        pa.field(name, types[type_name])
        for name, type_name in specification.items()
    ])


def _write_parquet(path: Path, rows: list[dict[str, Any]], schema) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        use_dictionary=True,
    )
    os.replace(temporary, path)


def _structure_files(root: Path, config: dict[str, Any]) -> list[Path]:
    if config["name"] == "AfCycDesign":
        files = sorted(
            [
                *root.joinpath("paper_set").glob("*.pdb"),
                *root.joinpath("14-16_paper_set").glob("*.pdb"),
            ],
            key=lambda path: path.name.lower(),
        )
    else:
        structure_root = root / str(config["structure_subdir"])
        index = root / str(config["index_relative"])
        identifiers = [
            line.strip()
            for line in index.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                f"{config['name']} index contains duplicate identifiers"
            )
        files = [
            structure_root / f"{identifier}.pdb"
            for identifier in identifiers
        ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{config['name']} structure files are missing: "
            + ", ".join(missing[:5])
        )
    return files


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _parse_task(
    task: tuple[str, str, str, str, int | None, str | None],
) -> dict[str, Any]:
    (
        path,
        source_name,
        source_class,
        chain_id,
        expected_length,
        expected_topology,
    ) = task
    try:
        return parse_structure_file(
            path,
            source_name=source_name,
            source_class=source_class,
            chain_id=chain_id,
            expected_residue_count=expected_length,
            expected_topology_class=expected_topology,
        )
    except Exception as exc:
        source = Path(path)
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "file_path": str(source),
            "file_sha256": (
                _sha256(source) if source.is_file() else None
            ),
            "observations": [],
            "structures": [],
        }


def _circular_distance(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _wilson_upper(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    z = 1.959963984540054
    estimate = numerator / denominator
    scale = 1.0 + z * z / denominator
    center = (
        estimate + z * z / (2.0 * denominator)
    ) / scale
    half = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / denominator
            + z * z / (4.0 * denominator * denominator)
        )
        / scale
    )
    return min(1.0, center + half)


def _angle_bin(angle_deg: float) -> int:
    normalized = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return min(
        HISTOGRAM_BINS - 1,
        max(
            0,
            int(
                math.floor(
                    (normalized + 180.0)
                    / (360.0 / HISTOGRAM_BINS)
                )
            ),
        ),
    )


def _normalized_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("cosine_mean") is None or row.get("sine_mean") is None:
        angle = float(row["entity_angle_deg"])
        cosine_mean = math.cos(math.radians(angle))
        sine_mean = math.sin(math.radians(angle))
    else:
        cosine_mean = float(row["cosine_mean"])
        sine_mean = float(row["sine_mean"])
    histogram = row.get("histogram_counts")
    if histogram is None:
        histogram = [0.0] * HISTOGRAM_BINS
        histogram[_angle_bin(float(row["entity_angle_deg"]))] = float(
            row.get("n_observations") or 1
        )
    histogram = np.asarray(histogram, dtype=float)
    total = float(histogram.sum())
    if total <= 0.0:
        raise ValueError("torsion summary histogram is empty")
    return {
        **row,
        "entity_key": str(row["entity_key"]),
        "source_class": str(row["source_class"]),
        "structure_key": (
            str(row["structure_key"])
            if row.get("structure_key") is not None
            else None
        ),
        "cosine_mean": cosine_mean,
        "sine_mean": sine_mean,
        "histogram_probability": histogram / total,
        "n_observations": int(row.get("n_observations") or 0),
        "n_structures": int(row.get("n_structures") or 0),
    }


def _circular_mean_std(
    cosine_mean: float,
    sine_mean: float,
) -> tuple[float, float, float]:
    resultant = min(
        1.0, math.hypot(float(cosine_mean), float(sine_mean))
    )
    mean = math.degrees(
        math.atan2(float(sine_mean), float(cosine_mean))
    )
    circular_std = (
        math.degrees(math.sqrt(-2.0 * math.log(resultant)))
        if resultant > 1e-12
        else 180.0
    )
    return mean, circular_std, resultant


def _vector_distribution(
    cosine_mean: float,
    sine_mean: float,
    histogram_probability: np.ndarray,
) -> dict[str, Any]:
    mean, circular_std, resultant = _circular_mean_std(
        cosine_mean, sine_mean
    )
    histogram = np.asarray(histogram_probability, dtype=float)
    histogram_total = float(histogram.sum())
    if histogram_total <= 0.0:
        histogram = np.zeros(HISTOGRAM_BINS, dtype=float)
    else:
        histogram = histogram / histogram_total
    nonzero = histogram[histogram > 0]
    entropy = float(
        -(nonzero * np.log(nonzero)).sum()
        / math.log(HISTOGRAM_BINS)
    )
    width = 360.0 / HISTOGRAM_BINS
    modes = []
    for index, value in enumerate(histogram):
        if (
            value > 0
            and value >= histogram[index - 1]
            and value >= histogram[(index + 1) % HISTOGRAM_BINS]
        ):
            modes.append({
                "center_deg": -180.0 + (index + 0.5) * width,
                "entity_weight_fraction": float(value),
            })
    modes.sort(
        key=lambda item: (
            -item["entity_weight_fraction"],
            item["center_deg"],
        )
    )
    return {
        "circular_mean_deg": mean,
        "circular_std_deg": circular_std,
        "resultant_length": resultant,
        "normalized_entropy": entropy,
        "modes": modes[:4],
        "histogram_entity_weight_fraction": [
            float(value) for value in histogram
        ],
    }


def _aggregate_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_entity: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_entity.setdefault(row["entity_key"], {}).setdefault(
            row["source_class"], []
        ).append(row)
    entities = []
    for entity_key, source_groups in sorted(by_entity.items()):
        source_summaries = []
        for source_class, source_rows in sorted(source_groups.items()):
            source_summaries.append({
                "source_class": source_class,
                "cosine_mean": float(np.mean([
                    row["cosine_mean"] for row in source_rows
                ])),
                "sine_mean": float(np.mean([
                    row["sine_mean"] for row in source_rows
                ])),
                "histogram_probability": np.mean(
                    [
                        row["histogram_probability"]
                        for row in source_rows
                    ],
                    axis=0,
                ),
            })
        entities.append({
            "entity_key": entity_key,
            "cosine_mean": float(np.mean([
                row["cosine_mean"] for row in source_summaries
            ])),
            "sine_mean": float(np.mean([
                row["sine_mean"] for row in source_summaries
            ])),
            "histogram_probability": np.mean(
                [
                    row["histogram_probability"]
                    for row in source_summaries
                ],
                axis=0,
            ),
            "source_classes": sorted(source_groups),
        })
    return entities


def _is_false_rigid(
    held: dict[str, Any],
    predicted_angle: float,
) -> bool:
    _held_mean, held_std, _held_resultant = _circular_mean_std(
        held["cosine_mean"],
        held["sine_mean"],
    )
    width = 360.0 / HISTOGRAM_BINS
    tail_mass = sum(
        float(weight)
        for index, weight in enumerate(
            held["histogram_probability"]
        )
        if _circular_distance(
            -180.0 + (index + 0.5) * width,
            predicted_angle,
        )
        > FALSE_RIGID_DEVIATION_DEG
    )
    return bool(
        held_std > RIGID_SIGMA_DEG
        or tail_mass > FALSE_RIGID_TAIL_MASS
    )


def _structure_calibration(
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    if len(rows) <= 2:
        return 0, 0
    source_totals: dict[str, dict[str, float]] = {}
    for row in rows:
        total = source_totals.setdefault(
            row["source_class"],
            {"count": 0.0, "cosine": 0.0, "sine": 0.0},
        )
        total["count"] += 1.0
        total["cosine"] += row["cosine_mean"]
        total["sine"] += row["sine_mean"]
    false_rigid = 0
    evaluable = 0
    for held in rows:
        source_vectors = []
        for source_class, total in source_totals.items():
            count = total["count"] - (
                1.0 if source_class == held["source_class"] else 0.0
            )
            if count <= 0.0:
                continue
            source_vectors.append((
                (
                    total["cosine"]
                    - (
                        held["cosine_mean"]
                        if source_class == held["source_class"]
                        else 0.0
                    )
                )
                / count,
                (
                    total["sine"]
                    - (
                        held["sine_mean"]
                        if source_class == held["source_class"]
                        else 0.0
                    )
                )
                / count,
            ))
        if not source_vectors:
            continue
        cosine_mean = float(np.mean([
            value[0] for value in source_vectors
        ]))
        sine_mean = float(np.mean([
            value[1] for value in source_vectors
        ]))
        training_mean, training_std, _training_resultant = (
            _circular_mean_std(
            cosine_mean,
            sine_mean,
            )
        )
        if training_std > RIGID_SIGMA_DEG:
            continue
        evaluable += 1
        if _is_false_rigid(held, training_mean):
            false_rigid += 1
    return false_rigid, evaluable


def _entity_calibration(
    entities: list[dict[str, Any]],
) -> tuple[int, int]:
    count = len(entities)
    if count <= 2:
        return 0, 0
    cosine_sum = float(sum(
        row["cosine_mean"] for row in entities
    ))
    sine_sum = float(sum(
        row["sine_mean"] for row in entities
    ))
    false_rigid = 0
    evaluable = 0
    for held in entities:
        denominator = count - 1
        cosine_mean = (
            cosine_sum - held["cosine_mean"]
        ) / denominator
        sine_mean = (
            sine_sum - held["sine_mean"]
        ) / denominator
        training_mean, training_std, _training_resultant = (
            _circular_mean_std(
            cosine_mean,
            sine_mean,
            )
        )
        if training_std > RIGID_SIGMA_DEG:
            continue
        evaluable += 1
        if _is_false_rigid(held, training_mean):
            false_rigid += 1
    return false_rigid, evaluable


def _circular_statistics(
    rows: list[dict[str, Any]],
    *,
    calibration_unit: str = "entity",
) -> dict[str, Any]:
    if calibration_unit not in {"entity", "structure"}:
        raise ValueError("calibration_unit must be entity or structure")
    normalized = [_normalized_summary_row(row) for row in rows]
    entities = _aggregate_rows(normalized)
    if not entities:
        raise ValueError("torsion statistics require observations")
    consensus = _vector_distribution(
        float(np.mean([
            row["cosine_mean"] for row in entities
        ])),
        float(np.mean([
            row["sine_mean"] for row in entities
        ])),
        np.mean(
            [
                row["histogram_probability"] for row in entities
            ],
            axis=0,
        ),
    )
    source_distributions = {}
    source_entity_counts = {}
    for source_class in sorted({
        row["source_class"] for row in normalized
    }):
        source_entities = _aggregate_rows([
            row
            for row in normalized
            if row["source_class"] == source_class
        ])
        source_entity_counts[source_class] = len(source_entities)
        source_distributions[source_class] = {
            "n_entities": len(source_entities),
            **_vector_distribution(
                float(np.mean([
                    row["cosine_mean"] for row in source_entities
                ])),
                float(np.mean([
                    row["sine_mean"] for row in source_entities
                ])),
                np.mean(
                    [
                        row["histogram_probability"]
                        for row in source_entities
                    ],
                    axis=0,
                ),
            ),
        }
    source_means = {
        source: distribution["circular_mean_deg"]
        for source, distribution in source_distributions.items()
    }
    source_consistency = max(
        (
            _circular_distance(left, right)
            for left in source_means.values()
            for right in source_means.values()
        ),
        default=0.0,
    )
    entity_false, entity_evaluable = _entity_calibration(entities)
    structure_false, structure_evaluable = _structure_calibration(
        normalized
    )
    false_rigid, evaluable = (
        (structure_false, structure_evaluable)
        if calibration_unit == "structure"
        else (entity_false, entity_evaluable)
    )
    false_upper = _wilson_upper(false_rigid, evaluable)

    source_false = 0
    source_evaluable = 0
    for held_source in source_distributions:
        held_rows = [
            row
            for row in normalized
            if row["source_class"] == held_source
        ]
        held_entities = _aggregate_rows(held_rows)
        held_entity_keys = {
            row["entity_key"] for row in held_entities
        }
        training_entities = _aggregate_rows([
            row
            for row in normalized
            if row["entity_key"] not in held_entity_keys
        ])
        if len(training_entities) <= 2:
            continue
        training = _vector_distribution(
            float(np.mean([
                row["cosine_mean"] for row in training_entities
            ])),
            float(np.mean([
                row["sine_mean"] for row in training_entities
            ])),
            np.mean(
                [
                    row["histogram_probability"]
                    for row in training_entities
                ],
                axis=0,
            ),
        )
        if training["circular_std_deg"] > RIGID_SIGMA_DEG:
            continue
        for held in held_entities:
            source_evaluable += 1
            if _is_false_rigid(
                held, training["circular_mean_deg"]
            ):
                source_false += 1
    source_false_upper = _wilson_upper(
        source_false, source_evaluable
    )
    rigidity_score = max(
        0.0,
        min(
            1.0,
            consensus["resultant_length"]
            * (
                1.0
                - min(
                    consensus["circular_std_deg"] / 90.0,
                    1.0,
                )
            )
            * (1.0 - consensus["normalized_entropy"]),
        ),
    )
    population_count = (
        int(sum(row["n_structures"] for row in normalized))
        if calibration_unit == "structure"
        else len(entities)
    )
    if (
        population_count >= 20
        and len(source_distributions) >= 2
        and source_consistency <= 20.0
        and false_upper is not None
        and false_upper <= 0.10
    ):
        confidence = "high"
    elif (
        population_count >= 5
        and false_upper is not None
        and false_upper <= 0.25
    ):
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "n_entities": len(entities),
        "n_observations": int(sum(
            row["n_observations"] for row in normalized
        )),
        "n_structures": int(sum(
            row["n_structures"] for row in normalized
        )),
        "entity_weighting": (
            "equal entity; equal source within entity; equal structure "
            "or bond component within entity-source"
        ),
        "source_entity_counts": source_entity_counts,
        "source_distributions": source_distributions,
        **consensus,
        "source_means_deg": source_means,
        "maximum_source_mean_delta_deg": source_consistency,
        "leave_one_entity_out_false_rigid_count": entity_false,
        "leave_one_entity_out_evaluable_count": entity_evaluable,
        "leave_one_entity_out_false_rigid_ci_high": _wilson_upper(
            entity_false, entity_evaluable
        ),
        "leave_one_structure_out_false_rigid_count": structure_false,
        "leave_one_structure_out_evaluable_count": structure_evaluable,
        "leave_one_structure_out_false_rigid_ci_high": _wilson_upper(
            structure_false, structure_evaluable
        ),
        "leave_one_source_out_entity_disjoint": True,
        "leave_one_source_out_false_rigid_count": source_false,
        "leave_one_source_out_evaluable_count": source_evaluable,
        "leave_one_source_out_false_rigid_ci_high": (
            source_false_upper
        ),
        "calibration_unit": calibration_unit,
        "calibration_false_rigid_count": false_rigid,
        "calibration_evaluable_count": evaluable,
        "calibration_false_rigid_ci_high": false_upper,
        "rigidity_score": rigidity_score,
        "confidence": confidence,
    }


def _compile_level(connection, level: str, key_column: str) -> dict[str, Any]:
    bin_width = 360.0 / HISTOGRAM_BINS
    histogram_columns = ",\n".join(
        (
            "sum(CASE WHEN angle_bin = "
            f"{index} THEN 1 ELSE 0 END)::BIGINT"
        )
        for index in range(HISTOGRAM_BINS)
    )
    grouping = (
        ", structure_key"
        if level == "exact"
        else ""
    )
    selected_structure = (
        ", structure_key"
        if level == "exact"
        else ", NULL::VARCHAR AS structure_key"
    )
    if level == "exact":
        population_expression = (
            "count(DISTINCT file_sha256 || ':' || model_id)"
        )
        population_minimum = 3
    elif level == "generic":
        population_expression = "count(DISTINCT entity_key)"
        population_minimum = 10
    else:
        population_expression = "count(DISTINCT entity_key)"
        population_minimum = 5
    query = f"""
        WITH eligible_keys AS (
            SELECT {key_column} AS lookup_key
            FROM observations
            WHERE {key_column} IS NOT NULL
            GROUP BY {key_column}
            HAVING {population_expression} >= {population_minimum}
        ),
        normalized AS (
            SELECT
                observation.{key_column} AS lookup_key,
                observation.entity_key,
                observation.source_class,
                observation.file_sha256 || ':' || observation.model_id
                    AS structure_key,
                cos(radians(observation.angle_deg)) AS cosine,
                sin(radians(observation.angle_deg)) AS sine,
                least(
                    {HISTOGRAM_BINS - 1},
                    greatest(
                        0,
                        floor(
                            (
                                (
                                    (observation.angle_deg + 180.0) % 360.0
                                    + 360.0
                                ) % 360.0
                            ) / {bin_width}
                        )::INTEGER
                    )
                ) AS angle_bin
            FROM observations AS observation
            INNER JOIN eligible_keys AS eligible
                ON observation.{key_column} = eligible.lookup_key
        )
        SELECT
            lookup_key,
            entity_key,
            source_class
            {selected_structure},
            avg(cosine) AS cosine_mean,
            avg(sine) AS sine_mean,
            count(*)::BIGINT AS n_observations,
            count(DISTINCT structure_key)::BIGINT AS n_structures,
            [{histogram_columns}] AS histogram_counts
        FROM normalized
        GROUP BY lookup_key, entity_key, source_class {grouping}
        ORDER BY lookup_key, entity_key, source_class {grouping}
    """
    reader = connection.sql(query).to_arrow_reader(batch_size=50000)
    entries = {}
    active_key = None
    active_rows = []

    def commit(key, rows):
        if key is None or not rows:
            return
        statistics = _circular_statistics(
            rows,
            calibration_unit=(
                "structure" if level == "exact" else "entity"
            ),
        )
        if level == "exact":
            population_ready = statistics["n_structures"] >= 3
        elif level == "generic":
            population_ready = statistics["n_entities"] >= 10
        else:
            population_ready = statistics["n_entities"] >= 5
        retain = bool(
            population_ready
            and statistics["confidence"] in {"high", "medium"}
            and statistics["rigidity_score"] >= 0.65
            and statistics["calibration_false_rigid_ci_high"]
            is not None
            and statistics["calibration_false_rigid_ci_high"] <= 0.25
        )
        if retain:
            entries[str(key)] = statistics

    for batch in reader:
        data = batch.to_pylist()
        for row in data:
            key = row["lookup_key"]
            if active_key is not None and key != active_key:
                commit(active_key, active_rows)
                active_rows = []
            active_key = key
            active_rows.append(row)
    commit(active_key, active_rows)
    return entries


def _merge_and_compile(
    output_dir: Path,
    *,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "offline prior compilation requires duckdb; install "
            "cycpep-master[prior-build]"
        ) from exc
    observations_glob = (
        output_dir / "work" / "observations" / "*.parquet"
    ).as_posix()
    structures_glob = (
        output_dir / "work" / "structures" / "*.parquet"
    ).as_posix()
    observations_output = output_dir / "torsion_observations.parquet"
    entities_output = output_dir / "torsion_entities.parquet"
    connection = duckdb.connect(
        str(output_dir / "work" / "compile.duckdb")
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW observations AS "
        f"SELECT * FROM read_parquet('{observations_glob}', "
        "union_by_name=true)"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW structures AS "
        f"SELECT * FROM read_parquet('{structures_glob}', "
        "union_by_name=true)"
    )
    connection.execute(
        f"COPY (SELECT * FROM observations) TO "
        f"'{observations_output.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                entity_key,
                any_value(full_inchikey) AS full_inchikey,
                any_value(sequence) AS sequence,
                any_value(residue_count) AS residue_count,
                any_value(topology_class) AS topology_class,
                any_value(macrocycle_ring_size)
                    AS macrocycle_ring_size,
                count(DISTINCT file_sha256 || ':' || model_id)::BIGINT
                    AS structure_model_count,
                count(DISTINCT source_name)::INTEGER AS source_count,
                string_agg(DISTINCT source_name, ',')
                    AS source_names,
                1.0 / count(DISTINCT file_sha256 || ':' || model_id)
                    AS structure_model_weight
            FROM structures
            GROUP BY entity_key
        ) TO '{entities_output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    levels = {
        "exact": _compile_level(
            connection, "exact", "exact_key"
        ),
        "residue_class_ring": _compile_level(
            connection,
            "residue_class_ring",
            "residue_class_ring_key",
        ),
        "morgan": _compile_level(
            connection, "morgan", "morgan_key"
        ),
        "generic": _compile_level(
            connection, "generic", "generic_key"
        ),
    }
    runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "source_summary": source_summary,
        "thresholds": {
            "rigid_sigma_deg": RIGID_SIGMA_DEG,
            "false_rigid_deviation_deg": (
                FALSE_RIGID_DEVIATION_DEG
            ),
            "false_rigid_tail_mass": FALSE_RIGID_TAIL_MASS,
            "histogram_bins": HISTOGRAM_BINS,
        },
        "levels": levels,
    }
    runtime_path = output_dir / "torsion_priors_runtime.json"
    _atomic_json(runtime_path, runtime)
    definitions_path = output_dir / "torsion_definitions.json"
    _atomic_json(
        definitions_path,
        {
            "schema_version": "1.0.0-cycpep-torsion-definitions.1",
            "backbone": {
                "phi": ["C(-1)", "N", "CA", "C"],
                "psi": ["N", "CA", "C", "N(+1)"],
                "omega": ["CA", "C", "N(+1)", "CA(+1)"],
            },
            "sidechain_chi": CHI_DEFINITIONS,
            "closure_policy": (
                "deterministic first heavy neighbor at each closure endpoint"
            ),
        },
    )
    return {
        "runtime_path": runtime_path,
        "definitions_path": definitions_path,
        "observations_path": observations_output,
        "entities_path": entities_output,
        "level_entry_counts": {
            level: len(entries) for level, entries in levels.items()
        },
    }


def build(
    *,
    cpbind_root: Path,
    cpsea_root: Path,
    afcycdesign_root: Path,
    output_dir: Path,
    max_workers: int,
    shard_size: int,
    limit_per_source: int | None,
    resume: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"refusing to overwrite torsion build: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "work"
    checkpoint_path = work / "checkpoint.json"
    protocol = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "sources": {
            "CPBind": str(cpbind_root.resolve()),
            "CPSea_PDB": str(cpsea_root.resolve()),
            "AfCycDesign": str(afcycdesign_root.resolve()),
        },
        "source_classes": {
            "CPBind": "predicted_complex",
            "CPSea_PDB": "pdb_derived_complex",
            "AfCycDesign": "predicted_design",
            "unbound": "not_available",
        },
        "max_workers": int(max_workers),
        "shard_size": int(shard_size),
        "limit_per_source": limit_per_source,
        "entity_weighting": (
            "equal entity weight; equal source weight within entity; "
            "equal structure or bond-component weight within "
            "entity-source"
        ),
        "calibration": {
            "exact": "leave-one-structure-out",
            "other_levels": "leave-one-entity-out",
            "source": "leave-one-source-out with held entities removed",
            "false_rigid_tail_mass": FALSE_RIGID_TAIL_MASS,
        },
        "randomness": "none",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(
            protocol, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    protocol["protocol_sha256"] = protocol_hash
    protocol_path = output_dir / "build_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("torsion build protocol drift")
    else:
        _atomic_json(protocol_path, protocol)
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {
            "schema_version": BUILD_SCHEMA_VERSION,
            "protocol_sha256": protocol_hash,
            "completed_shards": {},
        }
    )
    if checkpoint["protocol_sha256"] != protocol_hash:
        raise ValueError("torsion build checkpoint protocol drift")

    roots = {
        "CPBind": cpbind_root,
        "CPSea_PDB": cpsea_root,
        "AfCycDesign": afcycdesign_root,
    }
    observation_schema = _arrow_schema(OBSERVATION_SCHEMA)
    structure_schema = _arrow_schema(STRUCTURE_SCHEMA)
    status_counts = Counter()
    source_summary = {}
    for config in SOURCE_CONFIGS:
        files = _structure_files(roots[config["name"]], config)
        metadata: dict[str, tuple[int | None, str | None]] = {}
        if config["name"] == "CPBind":
            basic_path = (
                roots["CPBind"]
                / "CPBind_properties"
                / "CPBind_Basic.tsv"
            )
            topology_map = {
                "HEADTAIL": "head_to_tail",
                "DISULFIDE": "disulfide",
                "ISOPEPTIDE": "isopeptide",
            }
            with basic_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    metadata[str(row["id"])] = (
                        int(row["length"]),
                        topology_map.get(str(row["cyclic_type"])),
                    )
        elif config["name"] == "CPSea_PDB":
            for path in files:
                fields = path.stem.rsplit("_", 3)
                metadata[path.stem] = (
                    int(fields[-2]) - int(fields[-3]) + 1,
                    None,
                )
        else:
            for path in files:
                if path.name.startswith("hallucinated_"):
                    expected_length = 14
                else:
                    expected_length = int(path.name.split("_", 1)[0])
                metadata[path.stem] = (
                    expected_length,
                    "head_to_tail",
                )
        if limit_per_source is not None:
            files = files[:limit_per_source]
        source_summary[config["name"]] = {
            "file_count": len(files),
            "source_class": config["source_class"],
            "chain_id": config["chain_id"],
        }
        for shard_index, shard in enumerate(
            _chunks(files, shard_size)
        ):
            shard_id = f"{config['name']}-{shard_index:06d}"
            if shard_id in checkpoint["completed_shards"]:
                receipt = checkpoint["completed_shards"][shard_id]
                for name in ("observations", "structures", "status"):
                    path = output_dir / receipt[name]["path"]
                    if (
                        not path.is_file()
                        or _sha256(path) != receipt[name]["sha256"]
                    ):
                        raise ValueError(
                            f"completed shard drift: {shard_id}/{name}"
                        )
                continue
            tasks = [
                (
                    str(path),
                    config["name"],
                    config["source_class"],
                    config["chain_id"],
                    *metadata.get(path.stem, (None, None)),
                )
                for path in shard
            ]
            with ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                results = list(
                    executor.map(_parse_task, tasks, chunksize=8)
                )
            observations = [
                row
                for result in results
                for row in result["observations"]
            ]
            structures = [
                row
                for result in results
                for row in result["structures"]
            ]
            statuses = [
                {
                    "file_path": result["file_path"],
                    "file_sha256": result["file_sha256"],
                    "status": result["status"],
                    "error_json": (
                        json.dumps(
                            result["error"],
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        if result["error"] is not None
                        else None
                    ),
                    "observation_count": len(result["observations"]),
                    "structure_model_count": len(result["structures"]),
                }
                for result in results
            ]
            status_schema = _arrow_schema({
                "file_path": "string",
                "file_sha256": "string",
                "status": "string",
                "error_json": "string",
                "observation_count": "int32",
                "structure_model_count": "int32",
            })
            outputs = {
                "observations": (
                    work / "observations" / f"{shard_id}.parquet"
                ),
                "structures": (
                    work / "structures" / f"{shard_id}.parquet"
                ),
                "status": work / "status" / f"{shard_id}.parquet",
            }
            _write_parquet(
                outputs["observations"],
                observations,
                observation_schema,
            )
            _write_parquet(
                outputs["structures"], structures, structure_schema
            )
            _write_parquet(
                outputs["status"], statuses, status_schema
            )
            checkpoint["completed_shards"][shard_id] = {
                name: {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                for name, path in outputs.items()
            }
            _atomic_json(checkpoint_path, checkpoint)
            status_counts.update(
                result["status"] for result in results
            )
            print(
                f"[torsion-prior] {shard_id}: files={len(shard)} "
                f"structures={len(structures)} "
                f"observations={len(observations)}",
                flush=True,
            )
    compiled = _merge_and_compile(
        output_dir, source_summary=source_summary
    )
    status_glob = (
        output_dir / "work" / "status" / "*.parquet"
    ).as_posix()
    import duckdb

    connection = duckdb.connect()
    status_rows = connection.execute(
        f"SELECT file_path, file_sha256, status "
        f"FROM read_parquet('{status_glob}') ORDER BY file_path"
    ).fetchall()
    source_digest = hashlib.sha256()
    final_status_counts = Counter()
    for file_path, file_hash, status in status_rows:
        source_digest.update(
            f"{file_path}\t{file_hash}\n".encode("utf-8")
        )
        final_status_counts[str(status)] += 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "build_protocol_sha256": _sha256(protocol_path),
        "source_content_manifest_sha256": source_digest.hexdigest(),
        "source_file_count": len(status_rows),
        "source_status_counts": dict(
            sorted(final_status_counts.items())
        ),
        "source_summary": source_summary,
        "runtime_sha256": _sha256(compiled["runtime_path"]),
        "torsion_observations_sha256": _sha256(
            compiled["observations_path"]
        ),
        "torsion_entities_sha256": _sha256(
            compiled["entities_path"]
        ),
        "torsion_definitions_sha256": _sha256(
            compiled["definitions_path"]
        ),
        "level_entry_counts": compiled["level_entry_counts"],
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "network_access": False,
        "completed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    manifest_path = output_dir / "torsion_prior_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        **manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpbind-root", type=Path, required=True)
    parser.add_argument("--cpsea-root", type=Path, required=True)
    parser.add_argument("--afcycdesign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least one")
    if args.shard_size < 1:
        parser.error("--shard-size must be at least one")
    result = build(
        cpbind_root=args.cpbind_root,
        cpsea_root=args.cpsea_root,
        afcycdesign_root=args.afcycdesign_root,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        shard_size=args.shard_size,
        limit_per_source=args.limit_per_source,
        resume=args.resume,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
