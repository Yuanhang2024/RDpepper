"""AutoDock Vina discovery, invocation, and output parsing."""

import math
import os
import subprocess
from typing import Optional, Tuple


EXHAUSTIVENESS = 32
NUM_MODES = 9
VINA_MODES = ("docking", "score_only", "local_only")


def find_vina() -> Optional[str]:
    """Locate Vina by explicit environment, PATH, then bundled binary."""
    import shutil
    import sys

    explicit = os.environ.get("VINA_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    on_path = shutil.which("vina")
    if on_path:
        return on_path
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled_dir = os.path.join(package_root, "vina")
    candidate = os.path.join(
        bundled_dir,
        "vina_1.2.7_win.exe" if sys.platform.startswith("win") else "vina",
    )
    return candidate if os.path.exists(candidate) else None


def parse_vina_affinity(vina_stdout: str) -> Optional[float]:
    """Parse the best binding affinity from Vina stdout."""
    lines = vina_stdout.split("\n")
    for index, line in enumerate(lines):
        if "mode" in line and "affinity" in line:
            for candidate in lines[index + 1:]:
                if candidate.strip() and not candidate.startswith("-"):
                    parts = candidate.split()
                    if len(parts) >= 2:
                        try:
                            mode = int(parts[0])
                            affinity = float(parts[1])
                        except ValueError:
                            continue
                        if mode > 0 and math.isfinite(affinity):
                            return affinity
    return None


def parse_vina_score_only_energy(vina_stdout: str) -> Optional[float]:
    """Parse the reported binding energy from --score_only/--local_only stdout.

    Vina 1.2.7 prints "Estimated Free Energy of Binding : <value> (kcal/mol)"
    for both modes instead of the ranked mode table emitted by a docking run.
    """
    for line in vina_stdout.split("\n"):
        if "Estimated Free Energy of Binding" not in line:
            continue
        for token in line.split(":", 1)[-1].split():
            try:
                value = float(token)
            except ValueError:
                continue
            if math.isfinite(value):
                return value
    return None


def _minimally_parseable_output_pdbqt(path: str) -> bool:
    """True when the file looks like a freshly written Vina output PDBQT."""
    try:
        _pdbqt_model_signatures(path)
    except (OSError, ValueError):
        return False
    return True


def _paths_alias(first_path: str, second_path: str) -> bool:
    """Return whether two paths refer to the same filesystem object."""
    first = os.path.abspath(os.fspath(first_path))
    second = os.path.abspath(os.fspath(second_path))
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(
        os.path.realpath(second)
    ):
        return True
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, OSError):
        return False


def _clear_prior_output(path: str) -> Optional[str]:
    """Remove a prior Vina output, reporting failures without touching inputs."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"Cannot clear prior Vina output {path}: {exc}"
    return None


def _pdbqt_atom_signature(line: str) -> tuple[int, str, float]:
    """Parse the ligand invariants that Vina must preserve for every pose."""
    try:
        serial = int(line[6:11])
        coordinates = tuple(
            float(line[start:end])
            for start, end in ((30, 38), (38, 46), (46, 54))
        )
    except (IndexError, ValueError) as exc:
        raise ValueError("malformed PDBQT atom record") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("non-finite PDBQT atom coordinates")
    tokens = line.split()
    if len(tokens) < 12:
        raise ValueError("PDBQT atom lacks charge or AutoDock type")
    try:
        charge = float(tokens[-2])
    except ValueError as exc:
        raise ValueError("non-numeric PDBQT atom charge") from exc
    if not math.isfinite(charge) or not tokens[-1]:
        raise ValueError("invalid PDBQT atom charge or AutoDock type")
    return serial, tokens[-1], charge


def _pdbqt_model_signatures(path: str) -> list[tuple[tuple[int, str, float], ...]]:
    """Read one implicit ligand model or one or more explicit Vina models."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    if not any(line.strip() for line in lines):
        raise ValueError("empty PDBQT")

    models: list[tuple[tuple[int, str, float], ...]] = []
    implicit: list[tuple[int, str, float]] = []
    current: list[tuple[int, str, float]] | None = None
    saw_model = False
    for line in lines:
        tag = line[:6].strip()
        if tag == "MODEL":
            if current is not None or implicit:
                raise ValueError("nested or mixed PDBQT models")
            saw_model = True
            current = []
        elif tag == "ENDMDL":
            if current is None or not current:
                raise ValueError("empty or unmatched PDBQT model")
            models.append(tuple(current))
            current = None
        elif tag in {"ATOM", "HETATM"}:
            signature = _pdbqt_atom_signature(line)
            if saw_model:
                if current is None:
                    raise ValueError("PDBQT atom outside MODEL block")
                current.append(signature)
            else:
                implicit.append(signature)
    if current is not None:
        raise ValueError("unterminated PDBQT model")
    if saw_model:
        if not models:
            raise ValueError("PDBQT contains no complete model")
    elif implicit:
        models = [tuple(implicit)]
    else:
        raise ValueError("PDBQT contains no atoms")
    for model in models:
        serials = [row[0] for row in model]
        if len(serials) != len(set(serials)):
            raise ValueError("PDBQT model repeats atom serials")
    return models


def _pdbqt_model_payloads(path: str) -> list[str]:
    """Return implicit input or explicit Vina model payloads without wrappers."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    saw_model = any(line[:6].strip() == "MODEL" for line in lines)
    if not saw_model:
        return ["\n".join(lines) + "\n"]
    payloads = []
    current = None
    for line in lines:
        tag = line[:6].strip()
        if tag == "MODEL":
            if current is not None:
                raise ValueError("nested PDBQT MODEL block")
            current = []
        elif tag == "ENDMDL":
            if current is None:
                raise ValueError("unmatched PDBQT ENDMDL")
            payloads.append("\n".join(current) + "\n")
            current = None
        elif current is not None:
            current.append(line)
    if current is not None or not payloads:
        raise ValueError("incomplete PDBQT MODEL blocks")
    return payloads


def _verify_vina_output(
    output_pdbqt: str,
    ligand_pdbqt: str | None = None,
) -> Optional[str]:
    """Return an error string when Vina's output is absent or unparseable."""
    try:
        stat = os.stat(output_pdbqt)
    except OSError:
        return f"Vina produced no output file: {output_pdbqt}"
    if stat.st_size <= 0:
        return f"Vina output is empty: {output_pdbqt}"
    if not _minimally_parseable_output_pdbqt(output_pdbqt):
        return f"Vina output is not minimally parseable: {output_pdbqt}"
    if ligand_pdbqt is not None:
        try:
            ligand_models = _pdbqt_model_signatures(ligand_pdbqt)
            output_models = _pdbqt_model_signatures(output_pdbqt)
            ligand_payloads = _pdbqt_model_payloads(ligand_pdbqt)
            output_payloads = _pdbqt_model_payloads(output_pdbqt)
            if len(ligand_payloads) != 1:
                raise ValueError("ligand input contains multiple torsion trees")
        except (OSError, ValueError) as exc:
            return f"Vina ligand/output invariant parse failed: {exc}"
        if len(ligand_models) != 1:
            return "Vina ligand input must contain exactly one atom model"
        expected = ligand_models[0]
        for model_index, observed in enumerate(output_models, start=1):
            if len(observed) != len(expected):
                return (
                    f"Vina output model {model_index} atom-count mismatch: "
                    f"{len(observed)} != {len(expected)}"
                )
            for atom_index, (before, after) in enumerate(
                zip(expected, observed), start=1
            ):
                if before[:2] != after[:2] or not math.isclose(
                    before[2], after[2], rel_tol=0.0, abs_tol=5e-4
                ):
                    return (
                        f"Vina output model {model_index} changed ligand atom "
                        f"invariant at position {atom_index}: {after} != {before}"
                    )
        try:
            from .pdbqt_validation import validate_pdbqt_torsion_tree

            validate_pdbqt_torsion_tree(ligand_payloads[0])
            for payload in output_payloads:
                validate_pdbqt_torsion_tree(payload)
        except RuntimeError as exc:
            return f"Vina ligand/output torsion-tree validation failed: {exc}"
    return None


def run_vina(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    output_pdbqt: str,
    exhaustiveness: int = EXHAUSTIVENESS,
    num_modes: int = NUM_MODES,
    *,
    seed: Optional[int] = None,
    cpu: Optional[int] = None,
    max_evals: Optional[int] = None,
    timeout_seconds: float = 600,
    mode: str = "docking",
    find_executable=find_vina,
    run_process=subprocess.run,
    timeout_error=subprocess.TimeoutExpired,
    parse_affinity=parse_vina_affinity,
) -> Tuple[Optional[float], Optional[str]]:
    """Run Vina using injectable process seams for the compatibility facade.

    Success requires a freshly written, non-empty, minimally parseable output
    PDBQT; stale or unparseable output is rejected with an error. ``mode``
    selects the Vina operation: ``docking`` (global search, default),
    ``score_only`` (score the input pose; Vina writes no output file, so
    ``--out`` is not passed and the output path is never created, cleared,
    or removed), or ``local_only`` (local refinement of the input pose; a
    verified fresh output PDBQT is required, as in docking).
    """
    if not isinstance(mode, str) or mode not in VINA_MODES:
        return None, (
            "Invalid Vina mode: "
            f"{mode!r}; expected one of {', '.join(VINA_MODES)}"
        )
    controls = {}
    try:
        for name, value in (("seed", seed), ("cpu", cpu), ("max_evals", max_evals)):
            if value is None:
                continue
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            controls[name] = int(value)
        if any(value < 0 for value in controls.values()):
            raise ValueError("seed, cpu and max_evals must be non-negative")
        if isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be positive and finite")
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"Invalid Vina execution controls: {exc}"
    caller_cwd = os.getcwd()
    resolved_ligand = os.path.abspath(os.path.join(caller_cwd, ligand_pdbqt))
    resolved_receptor = os.path.abspath(os.path.join(caller_cwd, receptor_pdbqt))
    resolved_output = os.path.abspath(os.path.join(caller_cwd, output_pdbqt))
    try:
        if _paths_alias(resolved_output, resolved_ligand):
            return None, (
                "Vina output path must not alias the ligand input: "
                f"{resolved_output}"
            )
        if _paths_alias(resolved_output, resolved_receptor):
            return None, (
                "Vina output path must not alias the receptor input: "
                f"{resolved_output}"
            )
    except (TypeError, ValueError, OSError) as exc:
        return None, f"Cannot compare Vina input/output paths: {exc}"

    clear_error = None
    if mode != "score_only":
        clear_error = _clear_prior_output(resolved_output)
    if clear_error is not None:
        return None, clear_error

    executable = find_executable()
    if executable is None:
        return None, (
            "Vina executable not found. Set VINA_BIN, install vina "
            "(pip install vina / system package), or bundle it in "
            "cycpep_master/vina/."
        )

    try:
        center_values = tuple(float(value) for value in center)
        size_values = tuple(float(value) for value in box_size)
        exhaustiveness = int(exhaustiveness)
        num_modes = int(num_modes)
    except (TypeError, ValueError):
        return None, "Invalid Vina numeric parameters"
    if len(center_values) != 3 or not all(
        math.isfinite(value) for value in center_values
    ):
        return None, "Invalid Vina center: expected three finite values"
    if len(size_values) != 3 or not all(
        math.isfinite(value) and value > 0 for value in size_values
    ):
        return None, "Invalid Vina box size: expected three positive finite values"
    if exhaustiveness < 1 or num_modes < 1:
        return None, "Invalid Vina search parameters: values must be positive"
    resolved_executable = os.path.abspath(executable)
    if not os.path.isfile(resolved_ligand):
        return None, f"Vina ligand input does not exist: {resolved_ligand}"
    if not os.path.isfile(resolved_receptor):
        return None, f"Vina receptor input does not exist: {resolved_receptor}"

    cx, cy, cz = center_values
    sx, sy, sz = size_values
    command = [
        resolved_executable,
        "--receptor", resolved_receptor,
        "--ligand", resolved_ligand,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x", str(sx),
        "--size_y", str(sy),
        "--size_z", str(sz),
    ]
    if mode != "score_only":
        command.extend(["--out", resolved_output])
    command.extend([
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ])
    for name, value in controls.items():
        command.extend([f"--{name}", str(value)])
    if mode in ("score_only", "local_only"):
        command.append(f"--{mode}")
    vina_cwd = os.path.dirname(resolved_executable) or None
    success = False
    try:
        result = run_process(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=vina_cwd,
        )
        if result.returncode != 0:
            return None, f"Vina failed (exit {result.returncode}): {result.stderr}"
        if mode == "docking":
            affinity = parse_affinity(result.stdout)
        else:
            affinity = parse_vina_score_only_energy(result.stdout)
        if affinity is None or not math.isfinite(affinity):
            return None, "Failed to parse affinity from Vina output"
        if mode != "score_only":
            output_error = _verify_vina_output(resolved_output, resolved_ligand)
            if output_error is not None:
                return None, output_error
        success = True
        return affinity, None
    except timeout_error:
        if timeout_seconds == 600:
            return None, "Vina timed out (>10 min)"
        return None, f"Vina timed out (>{timeout_seconds:g} s)"
    except Exception as exc:
        return None, f"Vina execution error: {exc}"
    finally:
        if not success and mode != "score_only":
            try:
                os.remove(resolved_output)
            except (FileNotFoundError, OSError):
                pass
