"""Pipeline orchestrator: PDB -> SMILES -> Compare -> ADMET -> Export -> CSV."""
import contextlib
import csv as _csv
import os
from dataclasses import asdict

from .paths import PATH_MAP
from .paths.path_b import build_helm_from_pdb, _AA_3TO1
from .core.monomer_resolution import needs_monomer_resolution_scope
from .paths._map_utils import helm_to_map, map_to_helm, helm_to_biln
from .compare import compare
from .admet import run_admet
from .export import smiles_to_sdf
from .export.conformer import compute_conformer_ensemble_stats
from .core import parse_chain_sequence
from .core.cyclization import detect_cyclization
from .core.structure_io import (
    CoordinateInputError,
    coordinate_format,
    prepare_coordinate_input,
)
from .core.rigor import rigor_from_result

# ── Constants ─────────────────────────────────────────────────────────

ADMET_PROPERTIES = [
    'molecular_weight', 'logP', 'QED', 'Lipinski', 'AMES', 'hERG',
    'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca', 'Caco2_Wang', 'PPBR_AZ',
    'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'HIA_Hou', 'Bioavailability_Ma',
    'BBB_Martins', 'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith',
    'CYP2D6_Veith', 'CYP3A4_Veith', 'DILI', 'ClinTox', 'tpsa',
    'hydrogen_bond_acceptors', 'hydrogen_bond_donors',
]

# Base CSV columns are always present; conditional columns are appended by
# _build_csv_fields based on which computations are enabled.
_BASE_FIELDS = [
    'filename', 'input_format', 'helm', 'smiles', 'map', 'biln',
    'cyclization_type', 'status', 'reconstruction_status',
    'support_status', 'qualified_success',
    'repair_codes', 'warning_codes', 'rejection_reason',
    'rigor_level', 'rigor_provenance', 'rigor',
]

_FLEXIBILITY_FIELDS = [
    'flexibility_status', 'flexibility_proxy', 'rmsd',
    'backbone_rmsd_mean', 'sidechain_rmsd_mean', 'energy_std',
    'flexibility_num_confs', 'flexibility_num_kept', 'flexibility_pair_count',
    'flexibility_error',
]

_ADMET_STATUS_FIELDS = ['admet_status', 'admet_error']
_DOCKING_FIELDS = ['docking_status', 'docking_score', 'docking_error']
_EXPORT_FIELDS = [
    'export_status', 'export_format', 'export_requested_format_status',
    'export_coordinate_mode', 'export_coordinate_level',
    'export_validation_receipt_path', 'export_validation_receipt_sha256',
    'export_path', 'export_error',
]


def _coordinate_format_hint(source_file):
    try:
        return coordinate_format(source_file)
    except CoordinateInputError:
        return None


def _joined_codes(value):
    if not value:
        return ''
    if isinstance(value, str):
        return value
    return ';'.join(str(code) for code in value)


def _base_csv_row(entry):
    return {
        'filename': entry.get('file', ''),
        'input_format': entry.get('input_format') or '',
        'helm': entry.get('helm') or '',
        'smiles': entry.get('smiles') or '',
        'map': entry.get('map') or '',
        'biln': entry.get('biln') or '',
        'cyclization_type': entry.get('cyclization_type') or '',
        'status': entry.get('status') or 'failed',
        'reconstruction_status': (
            entry.get('reconstruction_status') or entry.get('status') or 'failed'
        ),
        'support_status': entry.get('support_status') or 'unknown',
        'qualified_success': bool(entry.get('qualified_success', False)),
        'repair_codes': _joined_codes(entry.get('repair_codes')),
        'warning_codes': _joined_codes(entry.get('warning_codes')),
        'rejection_reason': (
            entry.get('rejection_reason') or entry.get('error') or ''
        ),
        'rigor_level': entry.get('rigor_level') or '',
        'rigor_provenance': entry.get('rigor_provenance') or '',
        'rigor': entry.get('rigor') or '',
    }


def _apply_rigor(entry, *, strict=None, quality=None):
    """Attach additive rigor labels without changing reconstruction fields."""
    candidate = dict(entry)
    if quality is not None:
        candidate['quality'] = quality
    if strict is not None:
        candidate['strict_result'] = strict
    level = rigor_from_result(candidate)
    entry.update({
        'rigor_level': level.recovery_level,
        'rigor_provenance': level.provenance,
        'rigor': level.label,
    })
    return entry


def _comparison_field_names(primary_path, secondary_path):
    """Return stable, path-specific CSV fields for a route comparison."""
    prefix = f'compare_{primary_path}_vs_{secondary_path}'
    return prefix, f'{prefix}_detail'


def _build_csv_fields(run_admet_flag=False, compute_rmsd=False,
                      run_docking=False, run_export=False,
                      has_target=True, do_compare=True,
                      comparison_fields=None):
    """Assemble CSV field list from base + conditional columns.

    Conditional columns appear only when their option is enabled, so the CSV
    header adapts to what was actually computed.
    """
    fields = list(_BASE_FIELDS)
    if has_target:
        fields.append('target_sequence')
    if do_compare:
        fields += list(comparison_fields or (
            'comparison_status', 'comparison_detail'
        ))
    if compute_rmsd:
        fields += _FLEXIBILITY_FIELDS
    if run_docking:
        fields += _DOCKING_FIELDS
    if run_export:
        fields += _EXPORT_FIELDS
    if run_admet_flag:
        fields += _ADMET_STATUS_FIELDS
        fields += [f'admet_{p}' for p in ADMET_PROPERTIES]
    return fields


def _requested_stage_names(*, run_admet_flag, compute_rmsd,
                           run_docking, export_dir):
    stages = []
    if run_admet_flag:
        stages.append('admet')
    if compute_rmsd:
        stages.append('flexibility')
    if run_docking:
        stages.append('docking')
    if export_dir:
        stages.append('export')
    return stages


def _finalize_workflow_status(entry, requested_stages):
    reconstruction_status = str(
        entry.get('reconstruction_status') or entry.get('status') or 'failed'
    )
    entry['reconstruction_status'] = reconstruction_status
    incomplete = [
        stage for stage in requested_stages
        if entry.get(f'{stage}_status') != 'success'
    ]
    entry['workflow_warning_codes'] = [
        f'{stage.upper()}_STAGE_NOT_SUCCESS' for stage in incomplete
    ]
    if reconstruction_status != 'success':
        entry['status'] = reconstruction_status
    elif incomplete:
        entry['status'] = 'partial'
    else:
        entry['status'] = 'success'
    return entry


def _mark_requested_stages_not_assessable(entry, requested_stages, reason):
    for stage in requested_stages:
        entry[f'{stage}_status'] = 'not_assessable'
        entry[f'{stage}_error'] = reason
    return _finalize_workflow_status(entry, requested_stages)


# ── Pipeline ───────────────────────────────────────────────────────────

def run_batch(pdb_paths, path='v6', run_admet_flag=False,
               export_dir=None, export_format='mol2', csv_output=None,
               chain_id='L', target_chain_id='R',
               compute_rmsd=False, run_docking=False,
               require_empty_persistent_overlay=False,
               allow_linear_topology=False,
               fallback_policy='strict_v6',
               monomer_context=None):
    """Process coordinate files through audited V6 or an explicit raw path.

    PDB, PDB.GZ, mmCIF and mmCIF.GZ inputs share the same deterministic input
    preparation layer. ``path='v6'`` is the fail-closed default; A-H are
    diagnostic candidate generators and must be requested explicitly.
    Formal runs may require an empty persistent overlay explicitly; the
    default remains suitable for ordinary interactive use.
    """
    from .max_coverage import coerce_policy

    fallback_policy = coerce_policy(fallback_policy)
    pdb_paths = list(pdb_paths)
    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            (
                monomer_context
                if monomer_context is not None
                else {"include_persistent_user": True}
            ),
            required_symbols=monomer_symbol_hints(
                pdb_paths, kind="coordinate"
            ),
        ):
            return run_batch(
                pdb_paths,
                path=path,
                run_admet_flag=run_admet_flag,
                export_dir=export_dir,
                export_format=export_format,
                csv_output=csv_output,
                chain_id=chain_id,
                target_chain_id=target_chain_id,
                compute_rmsd=compute_rmsd,
                run_docking=run_docking,
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                allow_linear_topology=allow_linear_topology,
                fallback_policy=fallback_policy,
            )
    if path not in {'v6', *PATH_MAP.keys()}:
        raise ValueError(f"unknown reconstruction path: {path}")
    prepared_inputs = []
    with contextlib.ExitStack() as stack:
        for source in pdb_paths:
            try:
                prepared = stack.enter_context(
                    prepare_coordinate_input(source, chain_id)
                )
                strict = None
                if path == 'v6':
                    from .remediation_v6 import (
                        reconstruct_prepared_structure_fail_closed_v6,
                    )
                    strict = reconstruct_prepared_structure_fail_closed_v6(
                        prepared,
                        require_empty_persistent_overlay=(
                            require_empty_persistent_overlay
                        ),
                        allow_linear_topology=allow_linear_topology,
                    )
                target_prepared = None
                target_error = None
                if target_chain_id == chain_id:
                    target_prepared = prepared
                else:
                    try:
                        target_prepared = stack.enter_context(
                            prepare_coordinate_input(source, target_chain_id)
                        )
                    except CoordinateInputError as exc:
                        target_error = exc
                prepared_inputs.append((
                    str(source), prepared, strict, None,
                    target_prepared, target_error,
                ))
            except CoordinateInputError as exc:
                strict = None
                if path == 'v6':
                    from .remediation_v6 import coordinate_input_error_result_v6
                    strict = coordinate_input_error_result_v6(
                        source,
                        chain_id,
                        exc,
                        require_empty_persistent_overlay=(
                            require_empty_persistent_overlay
                        ),
                    )
                prepared_inputs.append((
                    str(source), None, strict, exc, None, None,
                ))
        return _run_batch_prepared(
            prepared_inputs,
            path=path,
            run_admet_flag=run_admet_flag,
            export_dir=export_dir,
            export_format=export_format,
            csv_output=csv_output,
            target_chain_id=target_chain_id,
            compute_rmsd=compute_rmsd,
            run_docking=run_docking,
            require_empty_persistent_overlay=(
                require_empty_persistent_overlay
            ),
            fallback_policy=fallback_policy,
        )


def _run_batch_prepared(prepared_inputs, path='v6', run_admet_flag=False,
              export_dir=None, export_format='mol2', csv_output=None,
              target_chain_id='R', compute_rmsd=False, run_docking=False,
              require_empty_persistent_overlay=False,
              fallback_policy='strict_v6'):
    """Execute already prepared coordinate inputs.

    Process multiple PDB files: SMILES + HELM/MAP/BILN + (optional)
    compare / ADMET / stability(RMSD) / docking / 3D export / CSV.

    Unified output contract — CSV columns are conditional on the options:
      Always:    filename, helm, smiles, map, biln, cyclization_type
      + target_sequence            (when target_chain resolves)
      + compare_<requested>_vs_<reference>[_detail]
                                    (when both path SMILES generate)
      + flexibility component fields (when compute_rmsd=True)
      + docking_score              (when run_docking=True)
      + admet_*                    (when run_admet_flag=True)

    Args:
        prepared_inputs: internal source/prepared/result/error tuples.
        path: fail-closed ``v6`` or one explicit A-H diagnostic path.
        run_admet_flag: run ADMET prediction if True.
        export_dir: if set, export 3D MOL2/SDF here (one file per PDB, named
            by stem). None = no export.
        export_format: 'mol2' or 'sdf'.
        csv_output: if set, write CSV here.
        chain_id: peptide chain ID (default 'L').
        target_chain_id: target protein chain ID (default 'R').
        compute_rmsd: if True, run the computational conformer-dispersion proxy.
        run_docking: if True, run Vina docking using the original PDB peptide
            chain as ligand against the target chain (Vina exe auto-detected).

    Returns list of result dicts.
    """
    gen = PATH_MAP.get(path)
    comparison_path = None
    other = None
    comparison_fields = None
    if path != 'v6':
        # Path C is the historical comparison reference for every other
        # diagnostic route; when C itself is requested, compare it with A.
        comparison_path = 'a' if path == 'c' else 'c'
        other = PATH_MAP[comparison_path]
        comparison_fields = _comparison_field_names(path, comparison_path)

    results = []
    used_export_paths = set()
    requested_stages = _requested_stage_names(
        run_admet_flag=run_admet_flag,
        compute_rmsd=compute_rmsd,
        run_docking=run_docking,
        export_dir=export_dir,
    )
    for (
        source_file,
        prepared,
        strict,
        preparation_error,
        target_prepared,
        target_error,
    ) in prepared_inputs:
        bn = os.path.basename(source_file)
        if prepared is None:
            if strict is not None:
                entry = {
                    'file': bn,
                    'source_path': source_file,
                    'pdb_path': None,
                    'input_format': _coordinate_format_hint(source_file),
                    **asdict(strict),
                }
                entry['smiles'] = entry['output_smiles']
                entry['error'] = entry['rejection_reason']
                _mark_requested_stages_not_assessable(
                    entry, requested_stages,
                    'coordinate input preparation did not produce a structure',
                )
                _apply_rigor(entry, strict=strict)
                entry['_row'] = _base_csv_row(entry)
                results.append(entry)
                continue
            entry = {
                'file': bn,
                'source_path': source_file,
                'pdb_path': None,
                'input_format': _coordinate_format_hint(source_file),
                'status': 'rejected',
                'support_status': 'unknown',
                'qualified_success': False,
                'repair_codes': [],
                'warning_codes': [],
                'error': (
                    'coordinate input preparation failed: '
                    f'{preparation_error}'
                ),
            }
            entry['rejection_reason'] = entry['error']
            _mark_requested_stages_not_assessable(
                entry, requested_stages,
                'coordinate input preparation did not produce a structure',
            )
            _apply_rigor(entry, strict=strict)
            entry['_row'] = _base_csv_row(entry)
            results.append(entry)
            continue
        pdb_file = str(prepared.pdb_path)
        effective_chain_id = prepared.chain_id
        input_format = prepared.source_format

        # ── Sequence + topology ──
        target_pdb_file = (
            str(target_prepared.pdb_path) if target_prepared is not None else None
        )
        effective_target_chain_id = (
            target_prepared.chain_id if target_prepared is not None else None
        )
        target_res = (
            parse_chain_sequence(target_pdb_file, effective_target_chain_id)
            if target_pdb_file and effective_target_chain_id
            else []
        )
        # Receptor (target chain) sequence as ONE-letter codes for ESM. The old
        # three-letter title-case form ('LeuSerIle...') was fed raw to the ESM
        # tokenizer, which split each residue into [first-letter, <unk>] -> half
        # the receptor tokens were noise. One-letter is what ESM expects; non-
        # standard residues fall back to 'X'.
        target_seq = ''.join(_AA_3TO1.get(r['name'], 'X') for r in target_res) if target_res else ''

        peptide_res = parse_chain_sequence(pdb_file, effective_chain_id)
        if not peptide_res:
            entry = {
                'file': bn, 'source_path': source_file,
                'pdb_path': source_file, 'input_format': input_format,
                'status': 'rejected', 'support_status': 'unknown',
                'qualified_success': False, 'repair_codes': [],
                'warning_codes': ['NO_SELECTED_CHAIN_RESIDUES'],
                'error': f'no {effective_chain_id}-chain residues',
            }
            entry['rejection_reason'] = entry['error']
            _mark_requested_stages_not_assessable(
                entry, requested_stages,
                'selected chain contains no reconstructable residues',
            )
            _apply_rigor(entry, strict=strict)
            entry['_row'] = _base_csv_row(entry)
            results.append(entry)
            continue

        cycl_info = detect_cyclization(pdb_file, effective_chain_id)
        cycl_type = cycl_info.topology

        # ── HELM / MAP / BILN (monomer-level, NNAA symbol preserved) ──
        helm = ''
        map_str = ''
        biln = ''
        try:
            helm = build_helm_from_pdb(pdb_file, effective_chain_id) or ''
            if helm:
                m = helm_to_map(helm)
                if m and not m.startswith('ERROR'):
                    map_str = m
                    biln = helm_to_biln(helm) or ''
        except Exception as exc:
            representation_error = f'{type(exc).__name__}: {exc}'
        else:
            representation_error = None

        # ── SMILES (requested path + independent comparison path) ──
        if path == 'v6':
            smi = strict.output_smiles
            err = strict.rejection_reason
            smi_alt = None
            alt_err = None
        else:
            smi, err = gen(pdb_file, effective_chain_id)
            smi_alt, alt_err = other(pdb_file, effective_chain_id)
        cmp_result, cmp_detail = '', ''
        if smi and smi_alt:
            ok, detail = compare(smi, smi_alt)
            cmp_result = 'PASS' if ok else 'DIFF'
            cmp_detail = detail

        # An explicit diagnostic route is the primary contract.  A reference
        # route may be useful for comparison or a caller-controlled fallback,
        # but it must never replace a failed requested route in the result.
        primary_smi = smi

        entry = {
            'file': bn, 'source_path': source_file, 'pdb_path': source_file,
            'input_format': input_format,
            'smiles': primary_smi,
            'error': err if not primary_smi else None,
            'status': strict.status if strict else ('success' if primary_smi else 'failed'),
            'support_status': strict.support_status if strict else 'unknown',
            'qualified_success': strict.qualified_success if strict else False,
            'repair_codes': list(strict.repair_codes) if strict else [],
        }
        if path != 'v6':
            entry.update({
                'primary_path': path,
                'primary_path_status': 'success' if smi else 'failed',
                'primary_path_error': err,
                'comparison_path': comparison_path,
                'comparison_path_status': (
                    'success' if smi_alt else 'failed'
                ),
                'comparison_path_error': alt_err,
            })
            if smi_alt:
                entry['comparison_path_smiles'] = smi_alt
            if not smi:
                entry['fallback_status'] = (
                    'available' if smi_alt else 'unavailable'
                )
                entry['fallback_source'] = comparison_path
                entry['fallback_smiles'] = smi_alt
                entry['fallback_warning'] = (
                    'PRIMARY_PATH_FAILED_FALLBACK_AVAILABLE'
                    if smi_alt
                    else 'PRIMARY_PATH_FAILED_NO_FALLBACK'
                )
        if representation_error:
            entry['representation_error'] = representation_error
        if strict:
            strict_fields = asdict(strict)
            entry.update(strict_fields)
        if target_error is not None:
            entry['target_input_error'] = {
                'code': target_error.code,
                'message': str(target_error),
            }

        # ── ADMET (conditional) ──
        if run_admet_flag:
            if not primary_smi:
                entry['admet_status'] = 'not_assessable'
                entry['admet_error'] = 'no reconstructed SMILES available'
            else:
                admet_result = run_admet([primary_smi])
                entry['admet'] = admet_result[0] if admet_result else {}
                admet_error = entry['admet'].get('error') if entry['admet'] else (
                    'ADMET returned no result row'
                )
                if admet_error:
                    entry['admet_status'] = (
                        'not_supported'
                        if 'not installed' in str(admet_error).lower()
                        else 'failed'
                    )
                    entry['admet_error'] = str(admet_error)
                else:
                    entry['admet_status'] = 'success'
                    entry['admet_error'] = None

        # ── Computational flexibility proxy (conditional) ──
        flexibility_result = None
        if compute_rmsd:
            if not primary_smi:
                flexibility_result = {
                    'status': 'not_assessable',
                    'error': 'no qualified SMILES available',
                }
            else:
                try:
                    stats, proxy_error = compute_conformer_ensemble_stats(
                        primary_smi, num_confs=10
                    )
                    flexibility_result = stats if stats is not None else {
                        'status': 'not_assessable',
                        'error': proxy_error or 'unspecified proxy failure',
                    }
                except Exception as exc:
                    flexibility_result = {
                        'status': 'failed',
                        'error': f'{type(exc).__name__}: {exc}',
                    }
            entry['flexibility_proxy'] = flexibility_result
            entry['flexibility_status'] = flexibility_result.get(
                'status', 'failed'
            )
            entry['flexibility_error'] = flexibility_result.get('error')

        # ── Docking (conditional; uses original PDB peptide chain as ligand) ──
        docking_val = ''
        if run_docking:
            if not target_res:
                entry['docking_status'] = 'not_assessable'
                entry['docking_error'] = (
                    str(target_error) if target_error is not None
                    else 'target chain contains no reconstructable residues'
                )
            else:
                try:
                    docking_val = _dock_prepared_structures(
                        pdb_file, target_pdb_file
                    )
                    entry['docking_status'] = 'success'
                    entry['docking_error'] = None
                except Exception as exc:
                    entry['docking_status'] = 'failed'
                    entry['docking_error'] = f'{type(exc).__name__}: {exc}'
            entry['docking_score'] = docking_val

        # ── 3D export (conditional) ──
        # MOL2 prefers the Path A atom mapping so source PDB coordinates are
        # retained. Other paths and failed Path A assemblies are explicitly
        # labelled as regenerated. SDF is always generated from SMILES.
        if export_dir:
            entry['export_format'] = export_format
            if not primary_smi:
                entry['export_status'] = 'not_assessable'
                entry['export_coordinate_mode'] = 'not_exported'
                entry['export_error'] = 'no reconstructed SMILES available'
            else:
                name = os.path.splitext(bn)[0]
                out_path = os.path.join(export_dir, name + '.' + export_format)
                path_key = os.path.normcase(os.path.abspath(out_path))
                if path_key in used_export_paths:
                    stem = f"{name}_{input_format}"
                    out_path = os.path.join(export_dir, stem + '.' + export_format)
                    path_key = os.path.normcase(os.path.abspath(out_path))
                if path_key in used_export_paths:
                    entry['export_error'] = 'duplicate normalized export path'
                    entry['export_coordinate_mode'] = 'not_exported'
                    out_path = None
                else:
                    used_export_paths.add(path_key)
                export_err = None
                if out_path is None:
                    export_err = entry['export_error']
                elif export_format == 'mol2':
                    from . import application as application_service

                    source_bound_path = path in {'a', 'e', 'v6'}
                    export_result = application_service.export_structure(
                        pdb_file if source_bound_path else primary_smi,
                        out_path,
                        source_kind=(
                            'coordinate' if source_bound_path else 'smiles'
                        ),
                        output_format='mol2',
                        chain_id=effective_chain_id,
                        path=path if source_bound_path else None,
                        require_empty_persistent_overlay=(
                            require_empty_persistent_overlay
                        ),
                        fallback_policy=fallback_policy,
                    )
                    export_data = export_result.get('data') or {}
                    requested_status = export_data.get(
                        'requested_format_status'
                    )
                    coordinate_mode = export_data.get(
                        'coordinate_mode'
                    ) or 'not_exported'
                    entry['export_requested_format_status'] = (
                        requested_status
                    )
                    entry['export_coordinate_level'] = export_data.get(
                        'coordinate_level'
                    )
                    entry['export_validation_receipt_path'] = (
                        export_data.get('validation_receipt_path')
                    )
                    entry['export_validation_receipt_sha256'] = (
                        export_data.get('validation_receipt_sha256')
                    )
                    if (
                        export_result.get('status') != 'success'
                        or requested_status != 'fulfilled'
                        or not export_data.get('output_path')
                    ):
                        export_err = (
                            export_result.get('error')
                            or f'MOL2 request {requested_status or "failed"}'
                        )
                    else:
                        out_path = str(export_data['output_path'])
                elif export_format == 'sdf':
                    coordinate_mode = 'regenerated'
                    _, export_err = smiles_to_sdf(
                        primary_smi, output_path=out_path
                    )
                    entry['export_requested_format_status'] = (
                        'fulfilled' if not export_err else 'unavailable'
                    )
                    entry['export_coordinate_level'] = 'X1'
                else:
                    coordinate_mode = 'not_exported'
                    export_err = f"unsupported export format: {export_format}"

                entry['export_coordinate_mode'] = coordinate_mode
                if export_err:
                    entry['export_status'] = 'failed'
                    entry['export_error'] = export_err
                else:
                    entry['export_status'] = 'success'
                    entry['export_error'] = None
                    entry['export_path'] = out_path

        _finalize_workflow_status(entry, requested_stages)
        _apply_rigor(
            entry,
            strict=strict,
            quality=("exact" if path == "v6" and strict and strict.status == "success"
                     and strict.output_smiles else ("high" if primary_smi else None)),
        )

        # ── Build CSV row (keys must match _build_csv_fields) ──
        entry.update({
            'helm': helm,
            'map': map_str,
            'biln': biln,
            'cyclization_type': cycl_type,
        })
        row = _base_csv_row(entry)
        if target_res:
            row['target_sequence'] = target_seq
        if smi and smi_alt and comparison_fields:
            row[comparison_fields[0]] = cmp_result
            row[comparison_fields[1]] = cmp_detail
        if compute_rmsd:
            proxy = flexibility_result or {}
            row.update({
                'flexibility_status': proxy.get('status', 'not_assessable'),
                'flexibility_proxy': proxy.get('flexibility_proxy', ''),
                'rmsd': proxy.get('rmsd_mean', ''),
                'backbone_rmsd_mean': proxy.get('backbone_rmsd_mean', ''),
                'sidechain_rmsd_mean': proxy.get('sidechain_rmsd_mean', ''),
                'energy_std': proxy.get('energy_std', ''),
                'flexibility_num_confs': proxy.get('num_confs', ''),
                'flexibility_num_kept': proxy.get('num_kept', ''),
                'flexibility_pair_count': proxy.get('pair_count', ''),
                'flexibility_error': proxy.get('error', ''),
            })
        if run_docking:
            row.update({
                'docking_status': entry.get('docking_status', ''),
                'docking_score': docking_val,
                'docking_error': entry.get('docking_error') or '',
            })
        if export_dir:
            row.update({
                field: entry.get(field) or '' for field in _EXPORT_FIELDS
            })
        if run_admet_flag:
            row['admet_status'] = entry.get('admet_status', '')
            row['admet_error'] = entry.get('admet_error') or ''
            for prop in ADMET_PROPERTIES:
                row[f'admet_{prop}'] = entry.get('admet', {}).get(prop, '')

        entry['_row'] = row
        results.append(entry)

    # ── Write CSV (header from _build_csv_fields, matching row keys) ──
    if csv_output and results:
        rows = [r['_row'] for r in results if '_row' in r]
        fields = _build_csv_fields(
            run_admet_flag=run_admet_flag, compute_rmsd=compute_rmsd,
            run_docking=run_docking, run_export=bool(export_dir),
            has_target=any(r.get('target_sequence') is not None for r in rows),
            do_compare=any(
                comparison_fields and comparison_fields[0] in r
                for r in rows
            ),
            comparison_fields=comparison_fields,
        )
        with open(csv_output, 'w', newline='', encoding='utf-8') as f:
            writer = _csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'CSV saved: {csv_output} ({len(rows)} records)')

    return results


def _dock_prepared_structures(ligand_pdb, receptor_pdb):
    """Dock two already isolated, normalized PDB coordinate projections."""
    from .docking.vina_wrapper import dock_peptide

    center, box_size = _box_from_pdb(receptor_pdb)
    affinity, error = dock_peptide(ligand_pdb, receptor_pdb, center, box_size)
    if error:
        raise RuntimeError(error)
    return affinity if affinity is not None else ''


def _box_from_pdb(pdb_path, padding=20.0):
    """Compute docking box center/size from receptor atom coordinates."""
    xs, ys, zs = [], [], []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')) or len(line) < 54:
                continue
            try:
                xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
            except ValueError:
                continue
    if not xs:
        raise ValueError("receptor PDB contains no finite atom coordinates")
    cx = (min(xs) + max(xs)) / 2; cy = (min(ys) + max(ys)) / 2; cz = (min(zs) + max(zs)) / 2
    sx = max(xs) - min(xs) + padding; sy = max(ys) - min(ys) + padding; sz = max(zs) - min(zs) + padding
    return (cx, cy, cz), (sx, sy, sz)
