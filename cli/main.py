"""Command-line interface for all RDpepper application services.

The historical ``cycpep --pdb/--dir`` interface remains stable.  New
subcommands expose the rest of the public functionality through the shared
``cycpep_master.application`` service layer and emit machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence

from .. import __version__
from .. import application as services
from ..pipeline import run_batch


_COMMANDS = {
    "capabilities",
    "reconstruct",
    "reconstruct-unified",
    "reconstruct-exact",
    "reconstruct-result-first",
    "convert",
    "audit",
    "compare",
    "export",
    "batch-export",
    "conformers",
    "template",
    "admet",
    "protonate",
    "prepare-sequence",
    "pdbqt",
    "dock-center",
    "vina",
    "dock",
    "batch-dock",
    "monomer",
}


def _find_coordinate_files(directory: str) -> List[str]:
    """Compatibility wrapper around shared coordinate discovery."""
    return services.discover_coordinate_files(directory)


def _derive_export_dir(args) -> Optional[str]:
    if args.export_dir:
        return args.export_dir
    if args.pdb:
        return os.path.dirname(os.path.abspath(args.pdb))
    if args.dir:
        absolute = os.path.abspath(args.dir.rstrip("/\\"))
        suffix = "_sdf" if getattr(args, "export_format", None) == "sdf" else "_mol2"
        return os.path.join(
            os.path.dirname(absolute), os.path.basename(absolute) + suffix
        )
    return None


def _result_outcome(entry: dict) -> str:
    if entry.get("status") == "success":
        return "qualified" if entry.get("qualified_success") else "unqualified"
    return "error"


def _print_result(entry: dict, verbose: bool = False):
    outcome = _result_outcome(entry)
    tag = {
        "qualified": "[OK]",
        "unqualified": "[UNQUALIFIED]",
        "error": "[ERROR]",
    }[outcome]
    print(f"  {entry['file']} {tag}")
    _print_rigor(entry)
    if verbose and entry.get("smiles"):
        print(f"    SMILES: {entry['smiles'][:80]}...")
    if verbose and entry.get("admet") and "error" not in entry["admet"]:
        values = entry["admet"]
        print(
            f"    logP={values.get('logP', '?')}  QED={values.get('QED', '?')}  "
            f"AMES={values.get('AMES', '?')}  hERG={values.get('hERG', '?')}"
        )
    if entry.get("export_path"):
        print(f"    -> {os.path.basename(entry['export_path'])}")


def _legacy_parser(
    *,
    program_name: str = "cycpep",
    product_name: str = "CycPep Master",
    distribution_name: str = "cycpep-master",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=program_name,
        description=(
            f"{product_name} - fail-closed cyclic-peptide chemical reconstruction "
            "from PDB or mmCIF."
        ),
        epilog=(
            "Complete service commands: capabilities, reconstruct, "
            "reconstruct-unified, reconstruct-exact, convert, "
            "audit, compare, export, batch-export, conformers, template, admet, "
            "protonate, pdbqt, dock-center, vina, dock, batch-dock, monomer. "
            f"Use '{program_name} <command> --help'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb", help="Single PDB/mmCIF coordinate file")
    group.add_argument("--dir", help="Directory of coordinate files")
    parser.add_argument(
        "-p",
        "--path",
        default="v6",
        choices=services.RECONSTRUCTION_PATHS,
        help=(
            "v6 = evidence-qualified fail-closed reconstruction (default); "
            "a-h = explicit diagnostic candidate path"
        ),
    )
    parser.add_argument("--chain", default="L", help="Peptide chain ID")
    parser.add_argument("--target-chain", default="R", help="Target chain ID")
    parser.add_argument("--admet", action="store_true", help="Run ADMET")
    parser.add_argument("--no-admet", action="store_true", help="Skip ADMET")
    parser.add_argument(
        "--flexibility-proxy",
        "--stability",
        dest="flexibility_proxy",
        action="store_true",
        help="Compute the conformer-dispersion flexibility proxy",
    )
    parser.add_argument("--docking", action="store_true", help="Run Vina docking")
    parser.add_argument(
        "--require-empty-persistent-overlay",
        action="store_true",
        help="Require an empty persistent monomer overlay for V6",
    )
    parser.add_argument("--export-dir", help="3D export directory")
    parser.add_argument(
        "--export-format", choices=["mol2", "sdf"], help="Enable 3D export"
    )
    parser.add_argument("--csv", help="Output CSV path")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--version",
        action="version",
        version=f"{distribution_name} {__version__}",
    )
    return parser


def _legacy_main(
    argv: Sequence[str],
    *,
    program_name: str = "cycpep",
    product_name: str = "CycPep Master",
    distribution_name: str = "cycpep-master",
) -> int:
    parser = _legacy_parser(
        program_name=program_name,
        product_name=product_name,
        distribution_name=distribution_name,
    )
    args = parser.parse_args(list(argv))
    if args.require_empty_persistent_overlay and args.path != "v6":
        parser.error("--require-empty-persistent-overlay requires --path v6")
    do_admet = args.admet and not args.no_admet
    export_dir = _derive_export_dir(args) if args.export_format else None
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    if args.pdb:
        pdb_paths = [args.pdb]
        csv_output = None if args.no_csv else (
            args.csv
            if args.csv
            else os.path.join(
                os.path.dirname(os.path.abspath(args.pdb)),
                os.path.splitext(os.path.basename(args.pdb))[0] + ".csv",
            )
        )
    else:
        try:
            pdb_paths = _find_coordinate_files(args.dir)
        except ValueError as exc:
            parser.error(str(exc))
        if not pdb_paths:
            print(f"No supported PDB/mmCIF files found in {args.dir}")
            return 1
        csv_output = args.csv
        print(f"Processing {len(pdb_paths)} files from {args.dir} (path={args.path})...")

    try:
        results = run_batch(
            pdb_paths,
            path=args.path,
            run_admet_flag=do_admet,
            export_dir=export_dir,
            export_format=args.export_format or "mol2",
            csv_output=csv_output,
            chain_id=args.chain,
            target_chain_id=args.target_chain,
            compute_rmsd=args.flexibility_proxy,
            run_docking=args.docking,
            require_empty_persistent_overlay=args.require_empty_persistent_overlay,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.pdb:
        result = results[0] if results else {}
        exit_code = 1 if _result_outcome(result) == "error" else 0
        print(
            f"STATUS: {result.get('status', 'unknown')} "
            f"support={result.get('support_status', 'unknown')} "
            f"qualified={result.get('qualified_success', False)}"
        )
        if _result_outcome(result) == "error":
            if result.get("warning_codes"):
                print(f"CODES: {','.join(result['warning_codes'])}")
            print(
                "ERROR: "
                + str(
                    result.get("error")
                    or result.get("rejection_reason")
                    or "unknown error"
                )
            )
        else:
            if result.get("repair_codes"):
                print(f"REPAIRS: {','.join(result['repair_codes'])}")
            _print_rigor(result)
            print(f"SMILES: {result.get('smiles', '')}")
            if result.get("map"):
                print(f"MAP: {result['map']}")
            if result.get("helm"):
                print(f"HELM: {result['helm']}")
            if args.flexibility_proxy:
                proxy = result.get("flexibility_proxy", {})
                print(
                    "FLEXIBILITY_PROXY: "
                    f"status={proxy.get('status', 'not_assessable')} "
                    f"score={proxy.get('flexibility_proxy', 'NA')} "
                    f"heavy_rmsd={proxy.get('rmsd_mean', 'NA')} "
                    f"backbone_rmsd={proxy.get('backbone_rmsd_mean', 'NA')} "
                    f"sidechain_rmsd={proxy.get('sidechain_rmsd_mean', 'NA')} "
                    f"energy_sd={proxy.get('energy_std', 'NA')}"
                )
                if proxy.get("error"):
                    print(f"FLEXIBILITY_PROXY_ERROR: {proxy['error']}")
            if args.verbose and result.get("admet"):
                print("ADMET:")
                for key, value in result["admet"].items():
                    print(f"  {key}: {value}")
            if result.get("export_path"):
                print(f"Exported to: {result['export_path']}")
            if csv_output:
                print(f"CSV: {csv_output}")
    else:
        outcomes = [_result_outcome(result) for result in results]
        exit_code = 1 if "error" in outcomes else 0
        for result in results:
            _print_result(result, verbose=args.verbose)
        print(
            f"\nDone: {outcomes.count('qualified')} qualified, "
            f"{outcomes.count('unqualified')} unqualified, "
            f"{outcomes.count('error')} errors, {len(results)} total"
        )
        if csv_output:
            print(f"CSV: {csv_output}")
        if export_dir:
            print(f"Export: {export_dir}")
    return exit_code


def _add_json_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json-out", help="Also write the result JSON to this file")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )
    return parsed


def _read_payload(value: str | None, input_file: str | None) -> str:
    if value is not None and input_file is not None:
        raise ValueError("payload and --input-file are mutually exclusive")
    if input_file:
        return Path(input_file).read_text(encoding="utf-8", errors="replace")
    if value == "-":
        return sys.stdin.read()
    if value is None:
        raise ValueError("payload or --input-file is required")
    return value


def _read_smiles_manifest(path: str | Path) -> list[tuple[str, str]]:
    """Read a JSON object or list of named SMILES records."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = []
        for index, row in enumerate(payload, start=1):
            if not isinstance(row, dict) or "name" not in row or "smiles" not in row:
                raise ValueError(
                    f"manifest row {index} must contain name and smiles"
                )
            items.append((row["name"], row["smiles"]))
    else:
        raise ValueError("SMILES manifest must be a JSON object or list")
    return [(str(name), str(smiles)) for name, smiles in items]


def _json_object_argument(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid JSON object: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _reconstruction_options(args) -> dict:
    """Return only explicitly requested reconstruction options."""
    options = {}
    minimum = getattr(args, "minimum_macrocycle_ring_size", None)
    if minimum is not None:
        options["minimum_macrocycle_ring_size"] = minimum
    if getattr(args, "require_empty_persistent_overlay", False):
        options["require_empty_persistent_overlay"] = True
    for name in ("radius_multiplier", "distance_ceiling"):
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    if getattr(args, "allow_linear_topology", False):
        options["allow_linear_topology"] = True
    if getattr(args, "infer_bond_orders", False):
        options["infer_bond_orders"] = True
    return options


def _rigor_from_envelope(result: dict) -> str | None:
    """Get the displayed rigor label when a result provides one."""
    for key in ("rigor", "chemical_rigor"):
        if result.get(key):
            return str(result[key])
    data = result.get("data")
    if isinstance(data, dict):
        for key in ("rigor", "chemical_rigor"):
            if data.get(key):
                return str(data[key])
        graph = data.get("chemical_graph")
        if isinstance(graph, dict):
            evidence = graph.get("evidence")
            if isinstance(evidence, dict) and evidence.get(
                "chemical_rigor"
            ):
                return str(evidence["chemical_rigor"])
    return None


def _print_rigor(result: dict) -> None:
    rigor = _rigor_from_envelope(result)
    if rigor:
        print(f"RIGOR: {rigor}")


def _emit_json(result: dict, args) -> int:
    output = getattr(args, "json_out", None)
    text = json.dumps(
        services.json_ready(result),
        ensure_ascii=False,
        sort_keys=True,
        indent=None if getattr(args, "compact", False) else 2,
    )
    if output:
        try:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            result = {
                "operation": result.get("operation", "cli"),
                "status": "failed",
                "data": {},
                "error": f"cannot write JSON output: {type(exc).__name__}: {exc}",
            }
            text = json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=None if getattr(args, "compact", False) else 2,
            )
    print(text)
    data = result.get("data") if isinstance(result, dict) else None
    requested_format_status = (
        data.get("requested_format_status")
        if isinstance(data, dict)
        else None
    )
    if (
        getattr(args, "strict_format", False)
        and requested_format_status is not None
        and requested_format_status != "fulfilled"
    ):
        return 1
    return 0 if result.get("status") in {
        "success",
        "partial",
        "match",
        "mismatch",
    } else 1


def _service_parser(
    *,
    program_name: str = "cycpep",
    product_name: str = "CycPep Master",
    distribution_name: str = "cycpep-master",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=program_name,
        description=f"Complete {product_name} CLI (JSON operation results)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{distribution_name} {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capabilities", help="List operations and dependencies")
    _add_json_options(cap)

    reconstruct = sub.add_parser("reconstruct", help="Recover graphs from coordinates")
    reconstruct.add_argument("inputs", nargs="*", help="Coordinate files")
    reconstruct.add_argument(
        "--mode",
        choices=["auto", "strict", "best-effort"],
        default=None,
        help=(
            "unified dispatch mode (default: auto); with explicit --path, "
            "only strict is accepted and the selected legacy path remains "
            "authoritative"
        ),
    )
    reconstruct.add_argument("--dir", help="Add all supported files in a directory")
    reconstruct.add_argument(
        "--path",
        choices=services.RECONSTRUCTION_PATHS,
        default=None,
        help=(
            "legacy pipeline path (v6 = fail-closed V6, a-h = diagnostic "
            "candidate path). Selects the legacy batch pipeline; omit to use "
            "the unified reconstruct_structure dispatch"
        ),
    )
    reconstruct.add_argument("--chain", default=None, help="Peptide chain ID")
    reconstruct.add_argument(
        "--target-chain", default=None, help="Target chain ID"
    )
    reconstruct.add_argument("--multichain", action="store_true")
    reconstruct.add_argument(
        "--chains",
        nargs="*",
        help=(
            "Explicit multi-chain IDs. The coordinate input may follow this list; "
            "a trailing supported coordinate filename is detected automatically."
        ),
    )
    reconstruct.add_argument("--admet", action="store_true")
    reconstruct.add_argument("--flexibility-proxy", action="store_true")
    reconstruct.add_argument("--docking", action="store_true")
    reconstruct.add_argument("--export-dir")
    reconstruct.add_argument("--export-format", choices=["mol2", "sdf"], default=None)
    reconstruct.add_argument("--csv")
    reconstruct.add_argument("--require-empty-persistent-overlay", action="store_true")
    reconstruct.add_argument("--minimum-macrocycle-ring-size", type=int)
    reconstruct.add_argument(
        "--radius-multiplier", type=float,
        help="Covalent-radius tolerance multiplier (default: 1.3; non-default is flagged)",
    )
    reconstruct.add_argument(
        "--distance-ceiling", type=float,
        help="Maximum inferred bond distance in Angstrom (default: 3.0; non-default is flagged)",
    )
    reconstruct.add_argument(
        "--allow-linear-topology", action="store_true",
        help="Accept a selected coordinate chain without cyclization evidence as linear",
    )
    reconstruct.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help="Entity-local custom definitions or CCD resolution settings",
    )
    _add_json_options(reconstruct)

    result_first = sub.add_parser(
        "reconstruct-result-first", help="Run the result-first coordinate recovery ladder"
    )
    result_first.add_argument("source", help="Coordinate file")
    result_first.add_argument("--chain", default="L", help="Peptide chain ID")
    result_first.add_argument("--minimum-macrocycle-ring-size", type=int)
    result_first.add_argument("--require-empty-persistent-overlay", action="store_true")
    result_first.add_argument("--radius-multiplier", type=float,
                              help="Covalent-radius tolerance multiplier (default: 1.3)")
    result_first.add_argument("--distance-ceiling", type=float,
                              help="Maximum inferred bond distance in Angstrom (default: 3.0)")
    result_first.add_argument(
        "--infer-bond-orders",
        action="store_true",
        help=(
            "Expose source-composition-bound bond-order candidates with "
            "audited rigor labels"
        ),
    )
    result_first.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help="Entity-local custom or CCD monomer extensions",
    )
    _add_json_options(result_first)

    unified = sub.add_parser(
        "reconstruct-unified",
        help=(
            "Unified reconstruction: PDB/mmCIF coordinates or Sequence/HELM/"
            "MAP/BILN text with optional source prefix"
        ),
    )
    unified.add_argument("source", help="Coordinate path or notation text")
    unified.add_argument(
        "--mode",
        choices=["auto", "strict", "best-effort"],
        default="auto",
        help="strict = V6-only coordinates / strict notation assembly; "
        "best-effort = compatible fallback branches on auto failure",
    )
    unified.add_argument("--minimum-macrocycle-ring-size", type=int)
    unified.add_argument("--require-empty-persistent-overlay", action="store_true")
    unified.add_argument(
        "--radius-multiplier", type=float,
        help="Covalent-radius tolerance multiplier (default: 1.3; non-default is flagged)",
    )
    unified.add_argument(
        "--distance-ceiling", type=float,
        help="Maximum inferred bond distance in Angstrom (default: 3.0; non-default is flagged)",
    )
    unified.add_argument(
        "--allow-linear-topology", action="store_true",
        help="Accept a selected coordinate chain without cyclization evidence as linear",
    )
    unified.add_argument(
        "--chain",
        nargs="*",
        help=(
            "Single chain ID, or a chain list for the PDB multi-chain route; "
            "omit to auto-select the unique peptide chain"
        ),
    )
    _add_json_options(unified)

    exact = sub.add_parser(
        "reconstruct-exact",
        help=(
            "Emit canonical exact_v1 from qualified, unrepaired V6 "
            "PDB/mmCIF evidence"
        ),
    )
    unified.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help="Entity-local custom or CCD monomer extensions",
    )
    exact.add_argument("source", help="PDB/mmCIF coordinate path")
    exact.add_argument("--chain", default="L")
    exact.add_argument(
        "--minimum-macrocycle-ring-size", type=int, default=8
    )
    exact.add_argument(
        "--allow-linear-topology", action="store_true"
    )
    exact.add_argument(
        "--allow-persistent-overlay",
        action="store_true",
        help=(
            "Permit the persistent derived monomer overlay; exact_v1 "
            "otherwise requires an empty persistent overlay"
        ),
    )
    exact.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help="Entity-local custom or CCD monomer extensions",
    )
    _add_json_options(exact)

    convert = sub.add_parser(
        "convert",
        help=(
            "Convert MAP/HELM/BILN/exact_v1 and bounded "
            "edge_v1/legacy_v5 projections"
        ),
    )
    convert.add_argument("--from", dest="source_kind", required=True, choices=services.REPRESENTATION_KINDS)
    convert.add_argument("--to", dest="target_kind", required=True, choices=services.REPRESENTATION_KINDS)
    convert.add_argument("payload", nargs="?")
    convert.add_argument("--input-file")
    convert.add_argument(
        "--edge-max-rings", type=_positive_int, default=3
    )
    convert.add_argument(
        "--edge-max-position", type=_positive_int, default=32
    )
    convert.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help=(
            "Entity-local custom/CCD monomer definitions; unresolved "
            "monomers remain lower-rigor candidates"
        ),
    )
    _add_json_options(convert)

    audit = sub.add_parser("audit", help="Run fail-closed chemistry audits")
    audit.add_argument("--kind", required=True, choices=services.AUDIT_KINDS)
    audit.add_argument("payload", nargs="?")
    audit.add_argument("--input-file")
    audit.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    _add_json_options(audit)

    compare = sub.add_parser("compare", help="Compare molecular identities")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument(
        "--mode",
        choices=["strict", "permissive", "specified-stereo"],
        default="strict",
    )
    _add_json_options(compare)

    export = sub.add_parser("export", help="Export MOL2 or SDF")
    export.add_argument("source")
    export.add_argument("output")
    export.add_argument(
        "--source-kind", choices=["smiles", "coordinate", "pdb", "mmcif"],
        default="smiles",
    )
    export.add_argument("--format", choices=["mol2", "sdf"])
    export.add_argument("--chain", default="L")
    export.add_argument("--path", choices=services.RECONSTRUCTION_PATHS, default="v6")
    export.add_argument(
        "--reconstruction-mode",
        choices=["auto", "strict", "best-effort"],
        default="auto",
    )
    export.add_argument("--minimum-macrocycle-ring-size", type=int, default=8)
    export.add_argument("--require-empty-persistent-overlay", action="store_true")
    export.add_argument("--force-field", choices=["mmff", "uff"], default="mmff")
    export.add_argument("--num-confs", type=int, default=10)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    export.add_argument(
        "--strict-format",
        action="store_true",
        help=(
            "Fail when MOL2/SDF cannot be produced instead of returning "
            "a lower-rigor graph or metadata artifact"
        ),
    )
    export.add_argument(
        "--fallback-policy",
        choices=["strict_v6", "max_coverage"],
        default=None,
        help=(
            "Coordinate-gap policy for source-bound MOL2 export. Defaults to "
            "strict_v6 with --strict-format and max_coverage otherwise; strict_v6 "
            "rejects records with incomplete source-coordinate mapping; "
            "max_coverage emits them with an X2/X1 coordinate-tier receipt "
            "(identity gates unchanged)"
        ),
    )
    _add_json_options(export)

    batch_export = sub.add_parser(
        "batch-export", help="Export a JSON manifest of named SMILES"
    )
    batch_export.add_argument("manifest")
    batch_export.add_argument("output_dir")
    batch_export.add_argument("--format", choices=["mol2", "sdf"], default="mol2")
    batch_export.add_argument("--force-field", choices=["mmff", "uff"], default="mmff")
    _add_json_options(batch_export)

    conformers = sub.add_parser("conformers", help="Compute conformer dispersion")
    conformers.add_argument("smiles")
    conformers.add_argument("--num-confs", type=int, default=50)
    conformers.add_argument("--seed", type=int, default=42)
    conformers.add_argument("--force-field", choices=["mmff", "uff"], default="mmff")
    conformers.add_argument("--no-optimize", action="store_true")
    conformers.add_argument("--energy-window", type=float)
    conformers.add_argument("--max-heavy-atoms", type=int)
    _add_json_options(conformers)

    template = sub.add_parser("template", help="Query or use the conformer library")
    template_sub = template.add_subparsers(dest="template_command", required=True)
    template_lookup = template_sub.add_parser("lookup", help="Find the closest template")
    template_lookup.add_argument("map")
    template_lookup.add_argument(
        "--strategy", choices=services.TEMPLATE_STRATEGIES, default="full"
    )
    template_lookup.add_argument("--seed", type=int, default=42)
    template_lookup.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    _add_json_options(template_lookup)
    template_generate = template_sub.add_parser(
        "generate", help="Write a template/ETKDG conformer ensemble as SDF"
    )
    template_generate.add_argument("smiles")
    template_generate.add_argument("map")
    template_generate.add_argument("output")
    template_generate.add_argument("--num-confs", type=int, default=5)
    template_generate.add_argument(
        "--strategy", choices=services.TEMPLATE_STRATEGIES, default="full"
    )
    template_generate.add_argument("--seed", type=int, default=42)
    template_generate.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    _add_json_options(template_generate)

    admet = sub.add_parser("admet", help="Predict ADMET properties")
    admet.add_argument("smiles", nargs="*")
    admet.add_argument("--input-file", help="One SMILES per line")
    _add_json_options(admet)

    protonate = sub.add_parser("protonate", help="Apply the pH 7.4 protonation rules")
    protonate.add_argument("smiles")
    _add_json_options(protonate)

    prepare_sequence = sub.add_parser(
        "prepare-sequence",
        help=(
            "Materialize a cyclic-peptide sequence as validated MOL2 and "
            "optional PDBQT artifacts"
        ),
    )
    prepare_sequence.add_argument("sequence")
    prepare_sequence.add_argument("output_dir")
    prepare_sequence.add_argument(
        "--cyclization",
        choices=["head-to-tail", "linear", "infer"],
        required=True,
    )
    prepare_sequence.add_argument(
        "--stereochemistry-json",
        type=_json_object_argument,
    )
    prepare_sequence.add_argument(
        "--terminal-modifications-json",
        type=_json_object_argument,
    )
    prepare_sequence.add_argument(
        "--protonation",
        choices=["registry_default", "physiological"],
        default="registry_default",
    )
    prepare_sequence.add_argument("--conformers", type=int, default=4)
    prepare_sequence.add_argument("--no-pdbqt", action="store_true")
    prepare_sequence.add_argument("--torsdof-limit", type=int)
    prepare_sequence.add_argument(
        "--flexibility-mode",
        choices=["fast", "balanced", "thorough"],
        default="balanced",
    )
    prepare_sequence.add_argument("--torsion-prior")
    prepare_sequence.add_argument(
        "--template-strategy",
        choices=services.TEMPLATE_STRATEGIES,
        default="full",
    )
    prepare_sequence.add_argument("--seed", type=int, default=42)
    prepare_sequence.add_argument("--threads", type=int, default=1)
    prepare_sequence.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help=(
            "Entity-local custom definitions, CCD files/directory, or "
            "explicit component IDs"
        ),
    )
    _add_json_options(prepare_sequence)

    pdbqt = sub.add_parser("pdbqt", help="Prepare or validate PDBQT")
    pdbqt_sub = pdbqt.add_subparsers(dest="pdbqt_command", required=True)
    ligand = pdbqt_sub.add_parser("ligand", help="Prepare ligand PDBQT from SMILES")
    ligand.add_argument("smiles")
    ligand.add_argument("output")
    ligand.add_argument("--map")
    ligand.add_argument("--num-confs", type=int, default=10)
    ligand.add_argument("--seed", type=int, default=42)
    ligand.add_argument("--flexible-macrocycles", action="store_true")
    ligand.add_argument("--no-protonate", action="store_true")
    ligand.add_argument("--torsdof-limit", type=int)
    ligand.add_argument("--torsion-ensemble-size", type=int, default=8)
    ligand.add_argument("--torsion-threads", type=int, default=1)
    ligand.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    _add_json_options(ligand)
    ligand_mol2 = pdbqt_sub.add_parser(
        "ligand-mol2",
        help="Prepare ligand PDBQT from a validated parent MOL2",
    )
    ligand_mol2.add_argument("input")
    ligand_mol2.add_argument("output")
    ligand_mol2.add_argument("--receipt")
    ligand_mol2.add_argument("--torsdof-limit", type=int)
    ligand_mol2.add_argument(
        "--flexibility-mode",
        choices=["fast", "balanced", "thorough"],
        default="balanced",
    )
    ligand_mol2.add_argument("--torsion-prior")
    ligand_mol2.add_argument("--ensemble-manifest")
    ligand_mol2.add_argument("--ensemble-manifest-sha256")
    ligand_mol2.add_argument("--ensemble-size", type=int, default=4)
    ligand_mol2.add_argument("--seed", type=int, default=42)
    ligand_mol2.add_argument("--threads", type=int, default=1)
    ligand_mol2.add_argument("--strict-budget", action="store_true")
    _add_json_options(ligand_mol2)
    ligand_pdb = pdbqt_sub.add_parser(
        "ligand-pdb", help="Prepare ligand PDBQT from existing PDB coordinates"
    )
    ligand_pdb.add_argument("input")
    ligand_pdb.add_argument("output")
    ligand_pdb.add_argument("--chain", default="L")
    ligand_pdb.add_argument(
        "--reconstruction-mode",
        choices=["auto", "strict", "best-effort"],
        default="auto",
    )
    ligand_pdb.add_argument("--minimum-macrocycle-ring-size", type=int, default=8)
    ligand_pdb.add_argument("--require-empty-persistent-overlay", action="store_true")
    ligand_pdb.add_argument(
        "--fallback-policy",
        choices=["strict_v6", "max_coverage"],
        default=None,
        help=(
            "Defaults to strict_v6 with --strict-format and max_coverage "
            "for best-available preparation"
        ),
    )
    ligand_pdb.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    ligand_pdb.add_argument(
        "--strict-format",
        action="store_true",
        help=(
            "Require PDBQT instead of returning a lower-rigor graph or "
            "metadata artifact"
        ),
    )
    _add_json_options(ligand_pdb)
    receptor = pdbqt_sub.add_parser("receptor", help="Prepare receptor PDBQT")
    receptor.add_argument("input")
    receptor.add_argument("output")
    _add_json_options(receptor)
    validate = pdbqt_sub.add_parser("validate", help="Audit ligand PDBQT torsion tree")
    validate.add_argument("input", nargs="?", help="PDBQT file, or '-' for stdin")
    validate.add_argument("--payload", help="PDBQT text supplied directly")
    _add_json_options(validate)

    center = sub.add_parser("dock-center", help="Calculate a docking box center")
    center.add_argument("receptor")
    center.add_argument("--residues", nargs="*", type=int)
    center.add_argument("--chain")
    _add_json_options(center)

    vina = sub.add_parser("vina", help="Run Vina on prepared PDBQT files")
    vina.add_argument("ligand")
    vina.add_argument("receptor")
    vina.add_argument("output")
    vina.add_argument("--center", nargs=3, type=float, required=True)
    vina.add_argument("--box-size", nargs=3, type=float, default=(25.0, 25.0, 25.0))
    vina.add_argument("--exhaustiveness", type=int, default=32)
    vina.add_argument("--num-modes", type=int, default=9)
    _add_json_options(vina)

    dock = sub.add_parser("dock", help="Run the Vina integration")
    dock.add_argument("peptide")
    dock.add_argument("receptor")
    dock.add_argument("--center", nargs=3, type=float)
    dock.add_argument("--binding-site-residues", nargs="*", type=int)
    dock.add_argument("--receptor-chain")
    dock.add_argument("--box-size", nargs=3, type=float, default=(25.0, 25.0, 25.0))
    dock.add_argument("--output-dir")
    dock.add_argument(
        "--ligand-preparation-mode",
        choices=["auto", "original_pdb", "audited_smiles"],
        default="auto",
    )
    dock.add_argument("--smiles")
    dock.add_argument("--peptide-chain", default="L")
    dock.add_argument("--map")
    dock.add_argument("--torsdof-limit", type=int)
    dock.add_argument("--torsion-ensemble-size", type=int, default=8)
    dock.add_argument("--torsion-threads", type=int, default=1)
    dock.add_argument(
        "--reconstruction-mode",
        choices=["auto", "strict", "best-effort"],
        default="auto",
    )
    dock.add_argument("--minimum-macrocycle-ring-size", type=int, default=8)
    dock.add_argument("--require-empty-persistent-overlay", action="store_true")
    dock.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help="Entity-local custom definitions or CCD resolution settings",
    )
    _add_json_options(dock)

    batch_dock = sub.add_parser(
        "batch-dock", help="Dock multiple peptide PDB files to one receptor"
    )
    batch_dock.add_argument("receptor")
    batch_dock.add_argument("peptides", nargs="*")
    batch_dock.add_argument("--dir", help="Add all PDB/ENT files in a directory")
    batch_dock.add_argument("--center", nargs=3, type=float, required=True)
    batch_dock.add_argument("--box-size", nargs=3, type=float, default=(25.0, 25.0, 25.0))
    batch_dock.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        help="Shared entity-local custom definitions or CCD settings",
    )
    _add_json_options(batch_dock)

    monomer = sub.add_parser("monomer", help="Inspect or add monomers")
    monomer_sub = monomer.add_subparsers(dest="monomer_command", required=True)
    monomer_list = monomer_sub.add_parser("list")
    monomer_list.add_argument("--query")
    monomer_list.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
    )
    _add_json_options(monomer_list)
    monomer_add = monomer_sub.add_parser("add")
    monomer_add.add_argument("symbol")
    monomer_add.add_argument("smiles")
    monomer_add.add_argument("--r1")
    monomer_add.add_argument("--r2")
    monomer_add.add_argument("--r3")
    monomer_add.add_argument("--overwrite", action="store_true")
    monomer_add.add_argument("--runtime-only", action="store_true")
    _add_json_options(monomer_add)
    monomer_resolve = monomer_sub.add_parser("resolve")
    monomer_resolve.add_argument("symbols", nargs="+")
    monomer_resolve.add_argument(
        "--monomer-context-json",
        type=_json_object_argument,
        required=True,
    )
    _add_json_options(monomer_resolve)
    return parser


def _reject_legacy_reconstruct_flags(parser: argparse.ArgumentParser, args) -> None:
    """Reject legacy batch options under the default unified dispatch.

    The unified ``reconstruct_structure`` service accepts a single source and
    does not run the legacy batch pipeline (directory discovery, ADMET,
    flexibility, docking, export, CSV, multi-chain flags).  Silently ignoring
    those flags would change the scientific path, so require an explicit
    ``--path`` instead.
    """
    legacy_flags = []
    if args.dir:
        legacy_flags.append("--dir")
    if args.multichain:
        legacy_flags.append("--multichain")
    if args.admet:
        legacy_flags.append("--admet")
    if args.flexibility_proxy:
        legacy_flags.append("--flexibility-proxy")
    if args.docking:
        legacy_flags.append("--docking")
    if args.export_dir:
        legacy_flags.append("--export-dir")
    if args.export_format is not None:
        legacy_flags.append("--export-format")
    if args.csv:
        legacy_flags.append("--csv")
    if args.target_chain is not None:
        legacy_flags.append("--target-chain")
    if legacy_flags:
        parser.error(
            "unified reconstruct does not support the legacy batch option(s) "
            + ", ".join(legacy_flags)
            + "; pass an explicit --path to use the legacy batch pipeline"
        )


def _retired_pdbqt_ensemble_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="cycpep pdbqt ensemble",
        description=(
            "Retired compatibility command. It always returns "
            "not_supported; use a validated MOL2 ensemble and run each "
            "member through 'cycpep pdbqt ligand-mol2'."
        ),
    )
    parser.add_argument("smiles")
    parser.add_argument("output_dir")
    parser.add_argument("--map")
    parser.add_argument("--num-confs", type=int, default=3)
    parser.add_argument("--flexible-macrocycles", action="store_true")
    _add_json_options(parser)
    args = parser.parse_args(list(argv))
    result = services.prepare_ligand_pdbqt_ensemble(
        args.smiles,
        args.output_dir,
        n_conformers=args.num_confs,
        generated_map=args.map,
        rigid_macrocycles=not args.flexible_macrocycles,
    )
    return _emit_json(result, args)


def _service_main(
    argv: Sequence[str],
    *,
    program_name: str = "cycpep",
    product_name: str = "CycPep Master",
    distribution_name: str = "cycpep-master",
) -> int:
    tokens = list(argv)
    if tokens[:2] == ["pdbqt", "ensemble"]:
        return _retired_pdbqt_ensemble_main(tokens[2:])
    parser = _service_parser(
        program_name=program_name,
        product_name=product_name,
        distribution_name=distribution_name,
    )
    args = parser.parse_args(tokens)
    try:
        if args.command == "capabilities":
            result = services.capabilities()
        elif args.command == "reconstruct-unified":
            chain_ids = list(args.chain or [])
            chain = chain_ids[0] if len(chain_ids) == 1 else (chain_ids or None)
            reconstruction_options = _reconstruction_options(args)
            result = services.reconstruct_unified(
                args.source,
                chain_id=chain,
                mode=args.mode.replace("-", "_"),
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
                **reconstruction_options,
            )
        elif args.command == "reconstruct-exact":
            result = services.reconstruct_exact_v1(
                args.source,
                chain_id=args.chain,
                minimum_macrocycle_ring_size=(
                    args.minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    not args.allow_persistent_overlay
                ),
                allow_linear_topology=args.allow_linear_topology,
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
            )
        elif args.command == "reconstruct-result-first":
            reconstruction_options = _reconstruction_options(args)
            result = services.reconstruct_result_first(
                args.source,
                chain_id=args.chain,
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
                **reconstruction_options,
            )
        elif args.command == "reconstruct":
            inputs = list(args.inputs)
            chain_ids = list(args.chains or [])
            mode = (args.mode or "auto").replace("-", "_")
            if args.path is None:
                _reject_legacy_reconstruct_flags(parser, args)
                if len(inputs) != 1:
                    parser.error(
                        "unified reconstruct supports exactly one SOURCE; "
                        "pass an explicit --path to use the legacy batch pipeline"
                    )
                if args.chain is not None and chain_ids:
                    parser.error("use either --chain or --chains, not both")
                reconstruction_options = _reconstruction_options(args)
                result = services.reconstruct_structure(
                    inputs[0],
                    chain_id=chain_ids or args.chain,
                    mode=mode,
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                    **reconstruction_options,
                )
            else:
                if args.mode is not None and args.mode != "strict":
                    parser.error(
                        f"--mode {args.mode} is a unified-mode option and "
                        "cannot be combined with an explicit --path; "
                        "drop --mode or omit --path"
                    )
                if args.multichain and not inputs and chain_ids:
                    possible_input = chain_ids[-1]
                    if any(
                        possible_input.lower().endswith(suffix)
                        for suffix in services.SUPPORTED_COORDINATE_SUFFIXES
                    ):
                        inputs.append(chain_ids.pop())
                if chain_ids and not args.multichain:
                    parser.error("--chains requires --multichain with explicit --path")
                if args.dir:
                    inputs.extend(services.discover_coordinate_files(args.dir))
                if args.multichain:
                    if len(inputs) != 1:
                        parser.error(
                            "--multichain requires exactly one coordinate input"
                        )
                    result = services.reconstruct_multichain(
                        inputs[0],
                        chain_ids=chain_ids or None,
                        **({
                            "monomer_context": args.monomer_context_json
                        } if args.monomer_context_json is not None else {}),
                    )
                else:
                    result = services.reconstruct_coordinates(
                        inputs,
                        path=args.path,
                        chain_id=args.chain or "L",
                        target_chain_id=args.target_chain or "R",
                        run_admet=args.admet,
                        compute_flexibility=args.flexibility_proxy,
                        run_docking=args.docking,
                        export_dir=args.export_dir,
                        export_format=args.export_format or "mol2",
                        csv_output=args.csv,
                        require_empty_persistent_overlay=args.require_empty_persistent_overlay,
                        radius_multiplier=args.radius_multiplier,
                        distance_ceiling=args.distance_ceiling,
                        allow_linear_topology=args.allow_linear_topology,
                        **({
                            "monomer_context": args.monomer_context_json
                        } if args.monomer_context_json is not None else {}),
                    )
        elif args.command == "convert":
            result = services.convert_representation(
                args.source_kind,
                args.target_kind,
                _read_payload(args.payload, args.input_file).strip(),
                edge_max_rings=args.edge_max_rings,
                edge_max_position=args.edge_max_position,
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
            )
        elif args.command == "audit":
            if args.payload is not None and args.input_file is not None:
                raise ValueError("payload and --input-file are mutually exclusive")
            if args.input_file:
                result = services.audit_chemistry(
                    args.kind,
                    input_path=args.input_file,
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                )
            else:
                result = services.audit_chemistry(
                    args.kind,
                    payload=_read_payload(args.payload, None),
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                )
        elif args.command == "compare":
            result = services.compare_chemistry(args.left, args.right, mode=args.mode)
        elif args.command == "export":
            export_service = (
                services.export_structure
                if args.strict_format
                else services.export_best_available
            )
            result = export_service(
                args.source,
                args.output,
                source_kind=args.source_kind,
                output_format=args.format,
                chain_id=args.chain,
                minimum_macrocycle_ring_size=args.minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=args.require_empty_persistent_overlay,
                force_field=args.force_field,
                num_confs=args.num_confs,
                random_seed=args.seed,
                fallback_policy=(
                    args.fallback_policy
                    or (
                        "strict_v6"
                        if args.strict_format
                        else "max_coverage"
                    )
                ),
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
                **(
                    {
                        "path": args.path,
                        "reconstruction_mode": (
                            args.reconstruction_mode.replace("-", "_")
                        ),
                    }
                    if args.strict_format
                    else {}
                ),
            )
        elif args.command == "batch-export":
            result = services.batch_export_structures(
                _read_smiles_manifest(args.manifest),
                args.output_dir,
                output_format=args.format,
                force_field=args.force_field,
            )
        elif args.command == "conformers":
            result = services.conformer_statistics(
                args.smiles,
                num_confs=args.num_confs,
                random_seed=args.seed,
                force_field=args.force_field,
                optimize=not args.no_optimize,
                energy_window=args.energy_window,
                max_heavy_atoms=args.max_heavy_atoms,
            )
        elif args.command == "template":
            if args.template_command == "lookup":
                result = services.find_conformer_template(
                    args.map,
                    template_strategy=args.strategy,
                    random_seed=args.seed,
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                )
            else:
                result = services.generate_template_conformers(
                    args.smiles,
                    args.map,
                    args.output,
                    n_conformers=args.num_confs,
                    template_strategy=args.strategy,
                    random_seed=args.seed,
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                )
        elif args.command == "admet":
            values = list(args.smiles)
            if args.input_file:
                values.extend(
                    line.strip()
                    for line in Path(args.input_file).read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if line.strip()
                )
            result = services.predict_admet(values)
        elif args.command == "protonate":
            result = services.protonate_smiles(args.smiles)
        elif args.command == "prepare-sequence":
            result = services.prepare_ligand_from_sequence(
                args.sequence,
                args.output_dir,
                cyclization=args.cyclization,
                stereochemistry=args.stereochemistry_json,
                terminal_modifications=(
                    args.terminal_modifications_json
                ),
                protonation=args.protonation,
                conformer_count=args.conformers,
                generate_pdbqt=not args.no_pdbqt,
                torsdof_limit=args.torsdof_limit,
                flexibility_mode=args.flexibility_mode,
                torsion_prior_path=args.torsion_prior,
                template_strategy=args.template_strategy,
                random_seed=args.seed,
                num_threads=args.threads,
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
            )
        elif args.command == "pdbqt":
            if args.pdbqt_command == "ligand":
                result = services.prepare_ligand_pdbqt(
                    args.smiles,
                    args.output,
                    generated_map=args.map,
                    num_confs=args.num_confs,
                    random_seed=args.seed,
                    rigid_macrocycles=not args.flexible_macrocycles,
                    protonate=not args.no_protonate,
                    torsdof_limit=args.torsdof_limit,
                    torsion_ensemble_size=args.torsion_ensemble_size,
                    torsion_num_threads=args.torsion_threads,
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                )
            elif args.pdbqt_command == "ligand-mol2":
                result = services.prepare_ligand_pdbqt_from_mol2(
                    args.input,
                    args.output,
                    receipt_path=args.receipt,
                    torsdof_limit=args.torsdof_limit,
                    flexibility_mode=args.flexibility_mode,
                    torsion_prior_path=args.torsion_prior,
                    ensemble_manifest_path=args.ensemble_manifest,
                    ensemble_manifest_sha256=(
                        args.ensemble_manifest_sha256
                    ),
                    ensemble_size=args.ensemble_size,
                    random_seed=args.seed,
                    num_threads=args.threads,
                    strict_budget=args.strict_budget,
                )
            elif args.pdbqt_command == "ligand-pdb":
                pdbqt_service = (
                    services.prepare_ligand_pdbqt_from_pdb
                    if args.strict_format
                    else services.prepare_ligand_pdbqt_best_available
                )
                result = pdbqt_service(
                    args.input,
                    args.output,
                    chain_id=args.chain,
                    minimum_macrocycle_ring_size=args.minimum_macrocycle_ring_size,
                    require_empty_persistent_overlay=args.require_empty_persistent_overlay,
                    fallback_policy=(
                        args.fallback_policy
                        or (
                            "strict_v6"
                            if args.strict_format
                            else "max_coverage"
                        )
                    ),
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                    **(
                        {
                            "reconstruction_mode": (
                                args.reconstruction_mode.replace("-", "_")
                            )
                        }
                        if args.strict_format
                        else {}
                    ),
                )
            elif args.pdbqt_command == "receptor":
                result = services.prepare_receptor_pdbqt(args.input, args.output)
            else:
                if args.payload is not None and args.input is not None:
                    raise ValueError("payload and input are mutually exclusive")
                if args.payload is not None:
                    result = services.validate_pdbqt(payload=args.payload)
                elif args.input == "-":
                    result = services.validate_pdbqt(payload=sys.stdin.read())
                elif args.input:
                    result = services.validate_pdbqt(input_path=args.input)
                else:
                    result = services.validate_pdbqt()
        elif args.command == "dock-center":
            result = services.docking_center(
                args.receptor, residue_ids=args.residues, chain_id=args.chain
            )
        elif args.command == "vina":
            result = services.run_prepared_vina(
                args.ligand,
                args.receptor,
                args.output,
                center=args.center,
                box_size=args.box_size,
                exhaustiveness=args.exhaustiveness,
                num_modes=args.num_modes,
            )
        elif args.command == "dock":
            result = services.dock_structure(
                args.peptide,
                args.receptor,
                center=args.center,
                binding_site_residues=args.binding_site_residues,
                receptor_chain_id=args.receptor_chain,
                box_size=args.box_size,
                output_dir=args.output_dir,
                ligand_preparation_mode=args.ligand_preparation_mode,
                ligand_smiles=args.smiles,
                peptide_chain_id=args.peptide_chain,
                generated_map=args.map,
                torsdof_limit=args.torsdof_limit,
                torsion_ensemble_size=args.torsion_ensemble_size,
                torsion_num_threads=args.torsion_threads,
                reconstruction_mode=args.reconstruction_mode.replace("-", "_"),
                minimum_macrocycle_ring_size=args.minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=args.require_empty_persistent_overlay,
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
            )
        elif args.command == "batch-dock":
            inputs = list(args.peptides)
            if args.dir:
                inputs.extend(services.discover_docking_coordinate_files(args.dir))
            result = services.batch_dock_structures(
                inputs,
                args.receptor,
                center=args.center,
                box_size=args.box_size,
                **({
                    "monomer_context": args.monomer_context_json
                } if args.monomer_context_json is not None else {}),
            )
        elif args.command == "monomer":
            if args.monomer_command == "list":
                result = services.list_monomers(
                    args.query,
                    **({
                        "monomer_context": args.monomer_context_json
                    } if args.monomer_context_json is not None else {}),
                )
            elif args.monomer_command == "resolve":
                result = services.resolve_monomers(
                    args.symbols,
                    monomer_context=args.monomer_context_json,
                )
            else:
                result = services.add_monomer(
                    args.symbol,
                    args.smiles,
                    r1=args.r1,
                    r2=args.r2,
                    r3=args.r3,
                    overwrite=args.overwrite,
                    persist=not args.runtime_only,
                )
        else:
            parser.error(f"unsupported command {args.command}")
    except (OSError, TypeError, ValueError) as exc:
        result = {
            "operation": args.command,
            "status": "invalid_input",
            "data": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return _emit_json(result, args)


def main(
    argv: Sequence[str] | None = None,
    *,
    program_name: str = "cycpep",
    product_name: str = "CycPep Master",
    distribution_name: str = "cycpep-master",
) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in _COMMANDS:
        return _service_main(
            tokens,
            program_name=program_name,
            product_name=product_name,
            distribution_name=distribution_name,
        )
    return _legacy_main(
        tokens,
        program_name=program_name,
        product_name=product_name,
        distribution_name=distribution_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
