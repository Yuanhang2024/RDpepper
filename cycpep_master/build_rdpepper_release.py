"""Build a non-overwriting RDpepper 7.1.0 local release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile


VERSION = "7.1.0"
DISTRIBUTION = "rdpepper"
QA_RECEIPT_NAME = "RDPEPPER_7_1_0_QA_RECEIPT.json"
SOURCE_BASELINE_NAME = "SOURCE_BASELINE.json"

_REQUIRED_WHEEL_MEMBERS = frozenset({
    "cycpep_master/core/artifacts.py",
    "cycpep_master/core/identity_memo.py",
    "cycpep_master/docking/mol2_input.py",
    "cycpep_master/docking/mol2_pdbqt.py",
    "rdpepper/__init__.py",
    "cycpep_master/schemas/candidate_assessment.schema.json",
    "cycpep_master/schemas/exact_v1.schema.json",
    "cycpep_master/schemas/v5_artifact.schema.json",
    "cycpep_master/data/torsion_priors/torsion_priors_runtime.json",
})
_FORBIDDEN_MEMBER_PARTS = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "benchmarks",
    "tests",
    "build",
    "dist",
})
_FORBIDDEN_MEMBER_NAMES = frozenset({
    "NUL",
    "user_monomer_library.csv",
    "torsion_prior_manifest.json",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            value for value in source.rglob("*") if value.is_file()
        ):
            relative = Path(source.name) / path.relative_to(source)
            info = zipfile.ZipInfo(relative.as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def _unsafe_archive_member(
    name: str,
    *,
    allow_sdist_metadata: bool = False,
) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in normalized:
        return "unsafe archive path"
    for index, part in enumerate(path.parts):
        if part in _FORBIDDEN_MEMBER_PARTS:
            return f"forbidden archive path component {part!r}"
        if part.endswith(".egg-info") and not (
            allow_sdist_metadata
            and index == 1
            and part == f"{DISTRIBUTION}.egg-info"
        ):
            return f"forbidden archive path component {part!r}"
        if any(0xE000 <= ord(character) <= 0xF8FF for character in part):
            return "private-use character in archive path"
    if path.name in _FORBIDDEN_MEMBER_NAMES:
        return f"forbidden archive member {path.name!r}"
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return f"compiled Python member {path.name!r}"
    if path.name.endswith((".bak", ".backup", ".orig", ".rej", "~")):
        return f"backup member {path.name!r}"
    return None


def _validate_member_names(
    names: set[str],
    *,
    label: str,
    allow_sdist_metadata: bool = False,
) -> None:
    problems = [
        f"{name}: {reason}"
        for name in sorted(names)
        if (
            reason := _unsafe_archive_member(
                name,
                allow_sdist_metadata=allow_sdist_metadata,
            )
        ) is not None
    ]
    if problems:
        raise ValueError(f"unsafe {label} members: " + "; ".join(problems))


def _validate_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = {name.rstrip("/") for name in archive.namelist() if name}
        _validate_member_names(names, label="wheel")
        missing = sorted(_REQUIRED_WHEEL_MEMBERS - names)
        if missing:
            raise ValueError(
                "wheel is missing required production members: "
                + ", ".join(missing)
            )
        metadata_names = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        headers = {
            key.strip(): value.strip()
            for line in metadata.splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
        }
        if headers.get("Name") != DISTRIBUTION:
            raise ValueError("wheel METADATA distribution name mismatch")
        if headers.get("Version") != VERSION:
            raise ValueError("wheel METADATA version mismatch")


def _validate_sdist(sdist: Path) -> None:
    prefix = f"{DISTRIBUTION}-{VERSION}/"
    with tarfile.open(sdist, "r:gz") as archive:
        names = {
            member.name.rstrip("/")
            for member in archive.getmembers()
            if member.name
        }
    _validate_member_names(
        names,
        label="sdist",
        allow_sdist_metadata=True,
    )
    sdist_members = {
        member.removeprefix("cycpep_master/")
        if member.startswith("cycpep_master/")
        else member
        for member in _REQUIRED_WHEEL_MEMBERS
    }
    required = {
        prefix + member for member in sdist_members
    } | {
        prefix + "pyproject.toml",
        prefix + "README.md",
        prefix + "LICENSE",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "sdist is missing required production members: "
            + ", ".join(missing)
        )


def _validate_source_archive(source_archive: Path) -> dict:
    prefix = f"{DISTRIBUTION}-{VERSION}-current-source/"
    with zipfile.ZipFile(source_archive) as archive:
        names = {
            name.rstrip("/") for name in archive.namelist() if name
        }
        problems = []
        for name in sorted(names):
            normalized = name.replace("\\", "/")
            path = PurePosixPath(normalized)
            if not normalized.startswith(prefix):
                problems.append(f"{name}: unexpected source root")
                continue
            if path.is_absolute() or ".." in path.parts or ":" in normalized:
                problems.append(f"{name}: unsafe archive path")
                continue
            for part in path.parts:
                if part in {
                    "__pycache__", ".pytest_cache", ".ruff_cache",
                    ".mypy_cache", "benchmarks", "build", "dist",
                } or part.endswith(".egg-info"):
                    problems.append(
                        f"{name}: forbidden source path component {part!r}"
                    )
                    break
                if any(
                    0xE000 <= ord(character) <= 0xF8FF
                    for character in part
                ):
                    problems.append(
                        f"{name}: private-use character in source path"
                    )
                    break
            if path.name in _FORBIDDEN_MEMBER_NAMES:
                problems.append(
                    f"{name}: forbidden source member {path.name!r}"
                )
            if path.suffix.lower() in {".pyc", ".pyo"}:
                problems.append(f"{name}: compiled Python source member")
        if problems:
            raise ValueError(
                "unsafe current-source members: " + "; ".join(problems)
            )
        required = {
            prefix + "build_rdpepper_release.py",
            prefix + "core/identity_memo.py",
            prefix + "docking/mol2_input.py",
            prefix + "tests/test_application.py",
            prefix + "source_manifest.json",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                "current-source archive is missing required members: "
                + ", ".join(missing)
            )
        manifest = json.loads(
            archive.read(prefix + "source_manifest.json").decode("utf-8")
        )
    if manifest.get("version") != VERSION:
        raise ValueError("current-source manifest version mismatch")
    return manifest


def build_release(
    wheel_path: str | Path,
    sdist_path: str | Path,
    source_archive_path: str | Path,
    output_dir: str | Path,
    *,
    zip_path: str | Path | None = None,
    delivery_path: str | Path | None = None,
) -> dict:
    root = Path(__file__).resolve().parent
    wheel = Path(wheel_path).resolve()
    sdist = Path(sdist_path).resolve()
    source_archive = Path(source_archive_path).resolve()
    destination = Path(output_dir).resolve()
    archive_path = (
        Path(zip_path).resolve()
        if zip_path is not None
        else destination.with_suffix(".zip")
    )
    delivery = (
        Path(delivery_path).resolve()
        if delivery_path is not None
        else archive_path.with_name(f"RDpepper-{VERSION}-DELIVERY.json")
    )
    expected_wheel = f"{DISTRIBUTION}-{VERSION}-py3-none-any.whl"
    expected_sdist = f"{DISTRIBUTION}-{VERSION}.tar.gz"
    expected_source_archive = (
        f"{DISTRIBUTION}-{VERSION}-current-source.zip"
    )
    if not wheel.is_file():
        raise FileNotFoundError(f"wheel is missing: {wheel}")
    if not sdist.is_file():
        raise FileNotFoundError(f"sdist is missing: {sdist}")
    if not source_archive.is_file():
        raise FileNotFoundError(
            f"current-source archive is missing: {source_archive}"
        )
    if wheel.name != expected_wheel:
        raise ValueError(f"unexpected wheel filename: {wheel.name}")
    if sdist.name != expected_sdist:
        raise ValueError(f"unexpected sdist filename: {sdist.name}")
    if source_archive.name != expected_source_archive:
        raise ValueError(
            f"unexpected current-source filename: {source_archive.name}"
        )
    for target, label in (
        (destination, "release directory"),
        (archive_path, "release archive"),
        (delivery, "delivery receipt"),
    ):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {label}: {target}")

    _validate_wheel(wheel)
    _validate_sdist(sdist)
    source_manifest = _validate_source_archive(source_archive)

    qa_path = root / QA_RECEIPT_NAME
    baseline_path = root / SOURCE_BASELINE_NAME
    qa_receipt = json.loads(qa_path.read_text(encoding="utf-8"))
    source_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if qa_receipt.get("version") != VERSION:
        raise ValueError("QA receipt version does not match release version")
    if qa_receipt.get("packaging_status") != "PASS":
        raise ValueError("QA receipt does not authorize local packaging")
    if source_baseline.get("version") != VERSION:
        raise ValueError("source baseline version does not match release version")
    if source_baseline.get("source_tree_sha256") != source_manifest.get(
        "source_tree_sha256"
    ):
        raise ValueError(
            "source baseline does not match current-source manifest"
        )
    expected_artifacts = qa_receipt.get("artifacts") or {}
    for kind, path in (
        ("wheel", wheel),
        ("sdist", sdist),
        ("current_source", source_archive),
    ):
        expected = expected_artifacts.get(kind) or {}
        if expected.get("sha256") != _sha256(path):
            raise ValueError(f"{kind} SHA-256 does not match QA receipt")
        if expected.get("size") != path.stat().st_size:
            raise ValueError(f"{kind} size does not match QA receipt")

    destination.mkdir(parents=True)
    sources = {
        wheel.name: wheel,
        sdist.name: sdist,
        source_archive.name: source_archive,
        "README.md": root / "README.md",
        "LICENSE": root / "LICENSE",
        "THIRD_PARTY_DATA.md": root / "THIRD_PARTY_DATA.md",
        "RDPEPPER_RELEASE_NOTES.md": root / "RDPEPPER_RELEASE_NOTES.md",
        "MIGRATION_RDPEPPER.md": root / "MIGRATION_RDPEPPER.md",
        "V7_MAX_COVERAGE_FALLBACK_DESIGN.md": (
            root / "V7_MAX_COVERAGE_FALLBACK_DESIGN.md"
        ),
        QA_RECEIPT_NAME: qa_path,
        SOURCE_BASELINE_NAME: baseline_path,
        "evidence/historical_releases/RDPEPPER_6_1_0_QA_RECEIPT.json": (
            root / "RDPEPPER_6_1_0_QA_RECEIPT.json"
        ),
        "evidence/historical_releases/RDPEPPER_5_1_0_QA_RECEIPT.json": (
            root / "RDPEPPER_5_1_0_QA_RECEIPT.json"
        ),
        "evidence/historical_v5/V5_QA_RECEIPT.json": (
            root / "V5_QA_RECEIPT.json"
        ),
        "evidence/historical_v5/v5_evidence_dossier.json": (
            root / "data" / "v5_evidence_dossier.json"
        ),
        "evidence/applicability_manifest.json": (
            root / "data" / "applicability_manifest.json"
        ),
        "schemas/candidate_assessment.schema.json": (
            root / "schemas" / "candidate_assessment.schema.json"
        ),
        "schemas/exact_v1.schema.json": (
            root / "schemas" / "exact_v1.schema.json"
        ),
        "schemas/v5_artifact.schema.json": (
            root / "schemas" / "v5_artifact.schema.json"
        ),
    }
    missing = [
        relative
        for relative, source in sources.items()
        if not source.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "release inputs are missing: " + ", ".join(missing)
        )
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    payload_files = [
        {
            "path": path.relative_to(destination).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            value for value in destination.rglob("*") if value.is_file()
        )
    ]
    manifest = {
        "schema_version": "1.1.0-rdpepper-release.1",
        "software": "RDpepper",
        "version": VERSION,
        "release_profile": "local_python_distribution_bundle",
        "packaging_status": qa_receipt["packaging_status"],
        "publication_status": qa_receipt["publication_status"],
        "upload_authorized": qa_receipt["upload_authorized"],
        "formal_benchmark_status": qa_receipt["formal_benchmark_status"],
        "source_tree_sha256": source_baseline["source_tree_sha256"],
        "manifest_scope": (
            "payload files present before release_manifest.json and "
            "SHA256SUMS.txt are generated"
        ),
        "files": payload_files,
    }
    manifest_path = destination / "release_manifest.json"
    _write_json(manifest_path, manifest)
    checksum_rows = [
        *payload_files,
        {
            "path": "release_manifest.json",
            "size": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
    ]
    checksum_path = destination / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{row['sha256']}  {row['path']}\n"
            for row in checksum_rows
        ),
        encoding="ascii",
        newline="\n",
    )
    _deterministic_zip(destination, archive_path)
    delivery_receipt = {
        "schema_version": "1.0.0-rdpepper-local-delivery.1",
        "software": "RDpepper",
        "version": VERSION,
        "release_directory": str(destination),
        "release_manifest_sha256": _sha256(manifest_path),
        "sha256sums_sha256": _sha256(checksum_path),
        "zip_path": str(archive_path),
        "zip_sha256": _sha256(archive_path),
        "zip_size": archive_path.stat().st_size,
        "source_tree_sha256": source_baseline["source_tree_sha256"],
        "upload_authorized": False,
    }
    _write_json(delivery, delivery_receipt)
    return {
        "release_dir": str(destination),
        "release_manifest": str(manifest_path),
        "release_manifest_sha256": _sha256(manifest_path),
        "zip_path": str(archive_path),
        "zip_sha256": _sha256(archive_path),
        "delivery_receipt": str(delivery),
        "delivery_receipt_sha256": _sha256(delivery),
        "file_count": len(checksum_rows) + 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--sdist", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip")
    parser.add_argument("--delivery")
    args = parser.parse_args(argv)
    try:
        result = build_release(
            args.wheel,
            args.sdist,
            args.source_archive,
            args.output_dir,
            zip_path=args.zip,
            delivery_path=args.delivery,
        )
    except Exception as exc:
        print(
            "RDpepper release build failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "DISTRIBUTION", "build_release", "main"]
