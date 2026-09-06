"""Deterministic V5 sequence-pipeline reproduction receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping

from . import __version__
from .application import prepare_ligand_from_sequence
from .core.artifacts import structured_sha256


RECEIPT_SCHEMA_VERSION = "1.0.0-cycpep-v5-reproduction.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _resource_hashes() -> dict[str, str | None]:
    root = Path(__file__).resolve().parent
    relative_paths = (
        "unified_monomer_library.csv",
        "data/applicability_manifest.json",
        "data/templates/templates_index.json",
        "data/torsion_priors/torsion_priors_runtime.json",
        "data/torsion_priors/torsion_prior_manifest_runtime.json",
        "schemas/exact_v1.schema.json",
        "schemas/v5_artifact.schema.json",
        "data/v5_evidence_dossier.json",
    )
    return {
        relative: (
            _sha256(root / relative)
            if (root / relative).is_file()
            else None
        )
        for relative in relative_paths
    }


def _artifact_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = [
        data.get("input_artifact"),
        data.get("chemical_graph"),
        data.get("mol2_ensemble"),
        *list(data.get("mol2_artifacts") or []),
        *list(data.get("flexibility_artifacts") or []),
        *list(data.get("pdbqt_artifacts") or []),
    ]
    return sorted(
        (
            {
                "artifact_type": value.get("artifact_type"),
                "artifact_id": value.get("artifact_id"),
                "status": value.get("status"),
                "parent_artifact_ids": list(
                    value.get("parent_artifact_ids") or []
                ),
                "content_sha256": value.get("sha256"),
                "receipt_sha256": value.get("receipt_sha256"),
                "manifest_sha256": value.get("manifest_sha256"),
            }
            for value in values
            if isinstance(value, Mapping)
        ),
        key=lambda row: (
            str(row["artifact_type"]),
            str(row["artifact_id"]),
        ),
    )


def reproduce_v5(
    output_dir: str | Path,
    *,
    sequence: str = "ACDEFG",
    cyclization: str = "head-to-tail",
    conformer_count: int = 4,
    generate_pdbqt: bool = True,
    random_seed: int = 42,
    num_threads: int = 1,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to use nonempty reproduction directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    result = prepare_ligand_from_sequence(
        sequence,
        destination,
        cyclization=cyclization,
        conformer_count=conformer_count,
        generate_pdbqt=generate_pdbqt,
        flexibility_mode="balanced",
        template_strategy="off",
        random_seed=random_seed,
        num_threads=num_threads,
    )
    data = result.get("data") or {}
    resources = _resource_hashes()
    artifact_rows = _artifact_rows(data)
    semantic_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "package_version": __version__,
        "request": {
            "sequence": sequence,
            "cyclization": cyclization,
            "conformer_count": conformer_count,
            "generate_pdbqt": generate_pdbqt,
            "random_seed": random_seed,
            "num_threads": num_threads,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rdkit": importlib.metadata.version("rdkit"),
            "meeko": (
                importlib.metadata.version("meeko")
                if generate_pdbqt
                else None
            ),
        },
        "resources": resources,
        "operation_status": result.get("status"),
        "requested_artifact_status": data.get(
            "requested_artifact_status"
        ),
        "warnings": list(data.get("warnings") or []),
        "artifacts": artifact_rows,
    }
    receipt = {
        **semantic_receipt,
        "reproduction_digest": structured_sha256(semantic_receipt),
    }
    receipt_path = destination / "v5_reproduction_receipt.json"
    _atomic_json(receipt_path, receipt)
    return {
        "result": result,
        "receipt": receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic CycPep Master V5 smoke workflow"
    )
    parser.add_argument("output_dir")
    parser.add_argument("--sequence", default="ACDEFG")
    parser.add_argument(
        "--cyclization",
        choices=["head-to-tail", "linear", "infer"],
        default="head-to-tail",
    )
    parser.add_argument("--conformers", type=int, default=4)
    parser.add_argument("--no-pdbqt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        outcome = reproduce_v5(
            args.output_dir,
            sequence=args.sequence,
            cyclization=args.cyclization,
            conformer_count=args.conformers,
            generate_pdbqt=not args.no_pdbqt,
            random_seed=args.seed,
            num_threads=args.threads,
        )
    except Exception as exc:
        print(f"V5 reproduction failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps({
        "status": outcome["result"].get("status"),
        "receipt_path": outcome["receipt_path"],
        "receipt_sha256": outcome["receipt_sha256"],
        "reproduction_digest": outcome["receipt"][
            "reproduction_digest"
        ],
    }, indent=2, sort_keys=True))
    return 0 if outcome["result"].get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "main",
    "reproduce_v5",
]
