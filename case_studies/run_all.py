#!/usr/bin/env python3
"""Run the three RDpepper case studies end-to-end against an installed wheel.

Each invocation writes its own run directory (default
``outputs/run_YYYYMMDD_HHMMSS``) containing the MOL2 artifacts, validation
receipts, per-case JSON results, ``run.log``, and the machine-readable
``results.json``. A pointer copy of the latest results is kept at
``latest_results.json`` next to this script. Existing files are never
deleted or overwritten: if a target path in a fresh run directory already
exists, that case fails loudly instead of replacing evidence. Earlier runs
remain untouched (see ``history/`` for preserved runs).

Failure behavior: the runner returns a nonzero exit code on failure —
exit 2 for preflight problems (missing input file, unusable wheel target,
version mismatch), exit 1 when any case fails its checks. Every early
failure path still writes a machine-readable ``results.json`` describing the
failure. Exit 0 means all three cases passed end to end.

Which installation is used:
  * If RDPEPPER_WHEEL_TARGET is set, it must contain an installable rdpepper
    package; it is placed on PYTHONPATH for every subprocess, and the
    preflight records rdpepper.__version__ and rdpepper.__file__ as proof of
    the code under test.
  * If unset, the ambient interpreter environment is used (same preflight
    proof recorded).
  * The preflight asserts the pinned expected version (default 7.3.0;
    override with --expected-version or RDPEPPER_EXPECTED_VERSION). A
    different version is recorded as an informative failure: the run does
    not proceed under a release-verification claim it cannot support.

Optional dependencies (openbabel, meeko) are probed with find_spec and never
imported when absent; these cases do not require them — bond-order inference
proceeds with the available engines and flags a missing Open Babel engine.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT_ROOT = HERE / "outputs"
LATEST_RESULTS = HERE / "latest_results.json"
DEFAULT_WHEEL_TARGET = (HERE.parent / "validation" / "wheel_install").resolve()
DEFAULT_EXPECTED_VERSION = "7.3.0"
SUBPROCESS_TIMEOUT = 1800  # seconds, per CLI invocation

# Chain IDs read from the actual input files (all three use chain L).
CASES = [
    {
        "name": "PRD_000227",
        "pdb": DATA / "PRD_000227.pdb",
        "chain": "L",
        "description": (
            "7 residues (MHW, THR, DBB, PRO, MEA, MHV, 004). CONECT declares "
            "a Thr side-chain OG1 ester to the C-terminal carboxyl of residue "
            "004 (15-51), which closes the macrocycle."
        ),
    },
    {
        "name": "PRD_000331",
        "pdb": DATA / "PRD_000331.pdb",
        "chain": "L",
        "description": (
            "5 residues (PHQ, ASP, GLU, VAL, ASA). No inter-residue cyclic "
            "linkage is declared in the CONECT records (and no SSBOND/LINK "
            "record exists); the reference-run graph contained no cyclic "
            "link."
        ),
    },
    {
        "name": "PRD_000807",
        "pdb": DATA / "PRD_000807.pdb",
        "chain": "L",
        "description": (
            "8 residues (HCI, PRO, LEU, HIS, SER, TPO, ALA, NH2), "
            "C-terminal amide, containing a phosphothreonine (TPO with "
            "P/O1P/O2P/O3P atoms). No inter-residue cyclic linkage is "
            "declared in the CONECT records."
        ),
    },
]


class RunContext:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_path = run_dir / "run.log"
        self.results_path = run_dir / "results.json"
        self._log = open(self.log_path, "w", encoding="utf-8", newline="\n")
        self.results: dict = {}

    def log(self, line: str = "") -> None:
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        text = f"[{stamp}] {line}" if line else ""
        print(text, flush=True)
        self._log.write(text + "\n")
        self._log.flush()

    def finish(self, exit_code: int) -> int:
        self.results["exit_code"] = exit_code
        self.results["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.results_path.write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:  # pointer copy for a stable path; run-dir file is canonical
            LATEST_RESULTS.write_text(
                json.dumps(self.results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        self._log.close()
        return exit_code


def sha256_of(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def base_env(wheel_target: Path | None) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if wheel_target is not None:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(wheel_target) + os.pathsep + existing if existing
            else str(wheel_target)
        )
    # Keep every cache write inside this directory: the default cache root is
    # <LOCALAPPDATA>/rdpepper, which case runs must not touch.
    cache_pick = HERE / ".cache" / "identity.pickle"
    cache_pick.parent.mkdir(parents=True, exist_ok=True)
    env["RDPEPPER_IDENTITY_CACHE"] = str(cache_pick)
    return env


def run_cli(argv: list[str], env: dict, *, label: str, ctx: "RunContext") -> tuple[int, str]:
    ctx.log(f"$ {' '.join(argv)}")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(ctx.run_dir),
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        ctx.log(f"  TIMEOUT after {elapsed:.0f}s")
        ctx.results.setdefault("timeouts", []).append(
            {"label": label, "seconds": round(elapsed, 1)}
        )
        return 124, ""
    elapsed = time.monotonic() - start
    ctx.log(f"  exit={proc.returncode} elapsed={elapsed:.1f}s")
    if proc.stderr.strip():
        for line in proc.stderr.strip().splitlines()[-5:]:
            ctx.log(f"  stderr| {line}")
    ctx.results.setdefault("timings", {})[label] = round(elapsed, 1)
    return proc.returncode, proc.stdout


def dependency_probe_code() -> str:
    return (
        "import json, importlib.util as u\n"
        "def probe(name):\n"
        "    spec = u.find_spec(name)\n"
        "    if spec is None:\n"
        "        return {'available': False, 'version': None}\n"
        "    mod = __import__(name)\n"
        "    return {'available': True,\n"
        "            'version': getattr(mod, '__version__', None)}\n"
        "print(json.dumps({k: probe(k) for k in\n"
        "  ['rdkit', 'gemmi', 'pandas', 'numpy', 'openbabel', 'meeko']}))\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the three RDpepper case studies (nonzero exit on failure)."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Run directory for artifacts, log and results.json. Default: "
            "outputs/run_YYYYMMDD_HHMMSS (a fresh directory; existing files "
            "are never deleted or overwritten)."
        ),
    )
    parser.add_argument(
        "--expected-version",
        default=os.environ.get("RDPEPPER_EXPECTED_VERSION", DEFAULT_EXPECTED_VERSION),
        help=(
            "Pinned rdpepper version this run verifies (default: "
            f"{DEFAULT_EXPECTED_VERSION}). A different installed version is "
            "recorded as an informative preflight failure."
        ),
    )
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else (OUTPUT_ROOT / f"run_{stamp}")
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        print(
            f"FAIL: run directory is not empty (existing evidence is never "
            f"overwritten): {run_dir}",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(run_dir)
    ctx.results = {
        "runner": "case_studies/run_all.py",
        "schema": "rdpepper-case-studies-run/2",
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "run_dir": str(run_dir),
        "log_file": str(ctx.log_path),
        "expected_rdpepper_version": args.expected_version,
        "cases": [],
        "preflight": {},
    }

    # ---- resolve the installation under test --------------------------------
    wheel_target_arg = os.environ.get("RDPEPPER_WHEEL_TARGET")
    wheel_target: Path | None = None
    if wheel_target_arg:
        wheel_target = Path(wheel_target_arg).resolve()
        ctx.results["wheel_target"] = str(wheel_target)
        if not (wheel_target / "rdpepper" / "__init__.py").is_file():
            ctx.results["status"] = "failed_preflight"
            ctx.results["preflight"] = {
                "status": "failed",
                "reason": "wheel_target_missing_rdpepper",
                "detail": (
                    "RDPEPPER_WHEEL_TARGET has no rdpepper package: "
                    f"{wheel_target}"
                ),
            }
            ctx.log(
                f"FAIL: RDPEPPER_WHEEL_TARGET has no rdpepper package: {wheel_target}"
            )
            return ctx.finish(2)
    elif DEFAULT_WHEEL_TARGET.is_dir():
        wheel_target = DEFAULT_WHEEL_TARGET
        ctx.results["wheel_target"] = str(wheel_target)
    else:
        ctx.results["wheel_target"] = None

    wheel_source_file = os.environ.get("RDPEPPER_WHEEL_FILE")
    ctx.results["wheel_source_file"] = wheel_source_file
    if wheel_source_file and Path(wheel_source_file).is_file():
        ctx.results["wheel_source_sha256"] = sha256_of(Path(wheel_source_file))

    env = base_env(wheel_target)

    # ---- preflight: wheel identity, version pin, dependency probe -----------
    ctx.log("Preflight: import rdpepper and record its origin.")
    rc, out = run_cli(
        [sys.executable, "-c",
         "import json, rdpepper, cycpep_master; "
         "print(json.dumps({'rdpepper_version': rdpepper.__version__, "
         "'rdpepper_file': rdpepper.__file__, "
         "'cycpep_master_file': cycpep_master.__file__}))"],
        env,
        label="preflight_import",
        ctx=ctx,
    )
    if rc != 0:
        ctx.results["status"] = "failed_preflight"
        ctx.results["preflight"] = {
            "status": "failed",
            "reason": "rdpepper_import_failed",
            "detail": "cannot import rdpepper in the configured environment",
        }
        ctx.log("FAIL: cannot import rdpepper in the configured environment.")
        return ctx.finish(2)
    try:
        identity = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        ctx.results["status"] = "failed_preflight"
        ctx.results["preflight"] = {
            "status": "failed",
            "reason": "preflight_unparseable",
        }
        ctx.log("FAIL: preflight identity output was not valid JSON.")
        return ctx.finish(2)

    expected = str(args.expected_version)
    version_ok = identity.get("rdpepper_version") == expected
    ctx.results["preflight"] = {
        "status": "ok" if version_ok else "failed",
        "rdpepper_version": identity.get("rdpepper_version"),
        "expected_version": expected,
        "version_matches_expected": version_ok,
        "rdpepper_file": identity.get("rdpepper_file"),
        "cycpep_master_file": identity.get("cycpep_master_file"),
    }
    ctx.log(
        f"  rdpepper {identity.get('rdpepper_version')} from "
        f"{identity.get('rdpepper_file')} (expected {expected})"
    )
    if not version_ok:
        ctx.results["status"] = "failed_preflight"
        ctx.results["preflight"]["reason"] = "version_mismatch"
        ctx.results["preflight"]["detail"] = (
            f"found {identity.get('rdpepper_version')}, expected {expected}; "
            "this configuration is not the pinned release and is not claimed "
            "as release verification (set --expected-version or "
            "RDPEPPER_EXPECTED_VERSION to verify a different pin explicitly)"
        )
        ctx.log(
            f"FAIL: rdpepper version {identity.get('rdpepper_version')} != "
            f"pinned {expected}; recording an informative failure instead of "
            f"a release-verification claim."
        )
        return ctx.finish(2)

    rc, out = run_cli(
        [sys.executable, "-c", dependency_probe_code()],
        env,
        label="preflight_dependencies",
        ctx=ctx,
    )
    if rc == 0:
        try:
            deps = json.loads(out.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            deps = {}
    else:
        deps = {}
    ctx.results["preflight"]["dependencies"] = deps
    ctx.results["preflight"]["dependency_note"] = (
        "Required by these cases: rdkit, gemmi, pandas, numpy (via the "
        "rdpepper import). Optional: openbabel enables the second "
        "bond-order-inference engine (its absence is flagged by the tool and "
        "inference proceeds with the available engines); meeko is not used "
        "by these reconstruction/readback cases."
    )
    ctx.log(f"  dependencies: {json.dumps(deps, sort_keys=True)}")

    missing = [str(c["pdb"]) for c in CASES if not c["pdb"].is_file()]
    if missing:
        for path in missing:
            ctx.log(f"FAIL: required input missing: {path}")
        ctx.results["status"] = "failed_preflight"
        ctx.results["preflight"]["reason"] = "missing_inputs"
        ctx.results["missing_inputs"] = missing
        return ctx.finish(2)

    # ---- cases ---------------------------------------------------------------
    failures = 0
    for case in CASES:
        record: dict = {
            "name": case["name"],
            "chain": case["chain"],
            "description": case["description"],
            "input_pdb": str(case["pdb"]),
            "input_sha256": sha256_of(case["pdb"]),
            "steps": [],
            "status": "running",
        }
        ctx.log("=" * 72)
        ctx.log(f"CASE {case['name']}  chain={case['chain']}")
        mol2 = run_dir / f"{case['name']}.mol2"
        receipt = run_dir / f"{case['name']}.mol2.validation.json"
        export_json = run_dir / f"{case['name']}.export.json"
        read_json = run_dir / f"{case['name']}.read.json"

        targets_exist = [
            str(p) for p in (mol2, receipt, export_json, read_json) if p.exists()
        ]

        ok = True

        def step(name: str, passed: bool, detail: dict) -> bool:
            entry = {"step": name, "ok": bool(passed), **detail}
            record["steps"].append(entry)
            ctx.log(
                f"  [{'PASS' if passed else 'FAIL'}] {name}"
                + (f": {detail['note']}" if detail.get("note") else "")
            )
            return passed

        if targets_exist:
            ok = step(
                "fresh_target_paths",
                False,
                {
                    "note": "run directory must be fresh; refusing to "
                    "overwrite: " + ", ".join(targets_exist)
                },
            )
        else:
            # 1) reconstruction + MOL2 export via the release wheel
            argv = [
                sys.executable, "-m", "rdpepper",
                "export", str(case["pdb"]), str(mol2),
                "--source-kind", "pdb",
                "--format", "mol2",
                "--chain", case["chain"],
                "--seed", "42",
                "--num-confs", "10",
                "--fallback-policy", "max_coverage",
                "--json-out", str(export_json),
                "--compact",
            ]
            rc, _ = run_cli(argv, env, label=f"{case['name']}:export", ctx=ctx)
            ok = step(
                "export_command", rc == 0, {"exit_code": rc, "command": argv}
            ) and ok

            ok = step(
                "export_json_written",
                export_json.is_file(),
                {
                    "path": str(export_json),
                    "sha256": sha256_of(export_json) if export_json.is_file() else None,
                },
            ) and ok
            export_data: dict = {}
            if export_json.is_file():
                try:
                    payload = json.loads(export_json.read_text(encoding="utf-8"))
                    export_data = payload.get("data", {}) or {}
                    record["export"] = {
                        "operation": payload.get("operation"),
                        "status": payload.get("status"),
                        "requested_format_status": export_data.get(
                            "requested_format_status"
                        ),
                        "chemical_rigor": export_data.get("chemical_rigor"),
                        "coordinate_level": export_data.get("coordinate_level"),
                        "coordinate_mode": export_data.get("coordinate_mode"),
                        "mapped_heavy_atoms": len(
                            export_data.get("mapped_heavy_atom_indices") or []
                        ),
                        "generated_heavy_atoms": len(
                            export_data.get("generated_heavy_atom_indices") or []
                        ),
                        "receipt_sha256": export_data.get("validation_receipt_sha256"),
                        "warning_codes": export_data.get("reconstruction", {}).get(
                            "warning_codes"
                        ),
                        "result_origin": export_data.get("reconstruction", {}).get(
                            "result_origin"
                        ),
                        "unresolved_monomers": [
                            {
                                "symbol": u.get("symbol"),
                                "ledger_error": u.get("error"),
                            }
                            for u in (
                                export_data.get("monomer_resolution", {}).get(
                                    "unresolved"
                                )
                                or []
                            )
                        ],
                        "resolved_monomers": {
                            k: v.get("resolved_symbol")
                            for k, v in (
                                export_data.get("monomer_resolution", {}).get(
                                    "resolved"
                                )
                                or {}
                            ).items()
                        },
                    }
                except json.JSONDecodeError as exc:
                    ok = step("export_json_parse", False, {"note": str(exc)}) and ok

            mol2_bytes = mol2.stat().st_size if mol2.is_file() else 0
            mol2_head_records: list[str] = []
            mol2_has_atom_record = False
            if mol2.is_file():
                with open(mol2, "r", encoding="utf-8", errors="replace") as handle:
                    for offset, line in enumerate(handle):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("#"):
                            continue  # writer emits # metadata comment lines first
                        mol2_head_records.append(stripped)
                        if stripped == "@<TRIPOS>ATOM":
                            mol2_has_atom_record = True
                        if offset > 400:
                            break
            mol2_contract_ok = (
                mol2.is_file()
                and mol2_bytes > 0
                and mol2_head_records[:1] == ["@<TRIPOS>MOLECULE"]
                and mol2_has_atom_record
            )
            ok = step(
                "mol2_written",
                mol2_contract_ok,
                {
                    "path": str(mol2),
                    "bytes": mol2_bytes,
                    "first_record": mol2_head_records[0] if mol2_head_records else None,
                    "has_atom_record": mol2_has_atom_record,
                    "sha256": sha256_of(mol2) if mol2.is_file() else None,
                    "note": "after optional # comment lines, the MOL2 must "
                    "open with @<TRIPOS>MOLECULE and contain an "
                    "@<TRIPOS>ATOM record (the writer's actual contract)",
                },
            ) and ok

            receipt_data: dict = {}
            if receipt.is_file():
                try:
                    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    ok = step("receipt_json_parse", False, {"note": str(exc)}) and ok
            ok = step(
                "receipt_written",
                receipt.is_file() and bool(receipt_data),
                {
                    "path": str(receipt),
                    "sha256": sha256_of(receipt) if receipt.is_file() else None,
                    "receipt_keys": sorted(receipt_data)[:12] if receipt_data else [],
                },
            ) and ok

            actual_receipt_sha = sha256_of(receipt) if receipt.is_file() else None
            declared_sha = export_data.get("validation_receipt_sha256")
            ok = step(
                "receipt_sha_matches_export",
                bool(actual_receipt_sha) and actual_receipt_sha == declared_sha,
                {
                    "declared_in_export_json": declared_sha,
                    "actual_file_sha256": actual_receipt_sha,
                    "note": "the receipt referenced by the export result must "
                    "be the file on disk (contract check)",
                },
            ) and ok

            ok = step(
                "export_status",
                export_data.get("requested_format_status") == "fulfilled",
                {
                    "status": record.get("export", {}).get("status"),
                    "requested_format_status": export_data.get(
                        "requested_format_status"
                    ),
                    "note": "the MOL2 request must be fulfilled, not degraded",
                },
            ) and ok

            # 2) receipt-verified readback through the compatibility reader
            if mol2.is_file():
                argv = [
                    sys.executable, "-m", "rdpepper",
                    "read-mol2", str(mol2),
                    "--compatibility", "rdkit_charge_aware",
                    "--receipt", str(receipt),
                    "--json-out", str(read_json),
                    "--compact",
                ]
                rc, _ = run_cli(argv, env, label=f"{case['name']}:read", ctx=ctx)
                ok = step(
                    "read_command", rc == 0, {"exit_code": rc, "command": argv}
                ) and ok
                ok = step(
                    "read_json_written",
                    read_json.is_file(),
                    {
                        "path": str(read_json),
                        "sha256": sha256_of(read_json) if read_json.is_file() else None,
                        "note": "a zero exit code without the JSON report is "
                        "a failure",
                    },
                ) and ok
                read_data: dict = {}
                if read_json.is_file():
                    try:
                        payload = json.loads(read_json.read_text(encoding="utf-8"))
                        read_data = payload.get("data", {}) or {}
                        verification = read_data.get("receipt_verification", {}) or {}
                        record["readback"] = {
                            "status": payload.get("status"),
                            "reader_mode": read_data.get("reader_mode"),
                            "sanitized": read_data.get("sanitized"),
                            "atom_count": read_data.get("atom_count"),
                            "heavy_atom_count": read_data.get("heavy_atom_count"),
                            "total_formal_charge": read_data.get(
                                "total_formal_charge"
                            ),
                            "full_inchikey": read_data.get("full_inchikey"),
                            "receipt_verification": verification.get("status"),
                            "coordinates_verified": read_data.get(
                                "coordinates_verified"
                            ),
                        }
                        ok = step(
                            "readback_receipt_verified",
                            verification.get("status") == "verified",
                            {"receipt_verification": verification.get("status")},
                        ) and ok
                        ok = step(
                            "readback_sanitized",
                            read_data.get("sanitized") is True
                            and bool(read_data.get("full_inchikey")),
                            {
                                "full_inchikey": read_data.get("full_inchikey"),
                                "note": "sanitization must succeed and yield "
                                "an InChIKey",
                            },
                        ) and ok
                    except json.JSONDecodeError as exc:
                        ok = step("read_json_parse", False, {"note": str(exc)}) and ok
                else:
                    ok = step(
                        "readback_receipt_verified",
                        False,
                        {"note": "no read JSON to verify"},
                    ) and ok

        record["status"] = "passed" if ok else "failed"
        if not ok:
            failures += 1
        ctx.results["cases"].append(record)

    # ---- summary --------------------------------------------------------------
    ctx.log("=" * 72)
    total = len(ctx.results["cases"])
    passed = sum(1 for c in ctx.results["cases"] if c["status"] == "passed")
    ctx.results["summary"] = {"total": total, "passed": passed, "failed": failures}
    ctx.results["status"] = "passed" if failures == 0 else "failed"
    ctx.log(
        f"Cases: {passed}/{total} passed, {failures} failed. "
        f"Machine JSON: {ctx.results_path}"
    )
    if failures:
        ctx.log("RESULT: FAILED (see failed cases above).")
        return ctx.finish(1)
    ctx.log("RESULT: ALL CASES PASSED.")
    return ctx.finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
