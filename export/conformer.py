"""3D conformer generation, energy minimization, and MOL2 export.

Uses RDKit: SMILES -> 2D -> 3D embedding (ETKDGv3) -> MMFF94 optimization -> MOL2.
"""
import os
import math
import statistics
import warnings

from rdkit import Chem
from ..core.monomer_resolution import needs_monomer_resolution_scope
from rdkit.Chem import AllChem

from ..core.mol2_format import (
    mol2_unity_formal_charges as _mol2_unity_formal_charges,
)


ETKDG_ATTEMPT_TIMEOUT_SECONDS = 30

#: Above this heavy-atom count, full-molecule ETKDG is not a viable product
#: path (observed: a 550-heavy-atom 3-disulfide domain cannot materialize a
#: single conformer even at a 300 s timeout).  Large entities get ONE
#: bounded random-coordinate attempt; failure is a typed, instant
#: unavailability instead of a multi-minute stall.
ETKDG_LARGE_ENTITY_HEAVY_ATOMS = 300
ETKDG_LARGE_ENTITY_TIMEOUT_SECONDS = 60


def _embed_3d(mol, num_confs=10, random_seed=42, *, retain_all=False):
    """Generate 3D conformer(s) via ETKDGv3, return the lowest-energy one.

    num_confs raised from 1 (historical default) to 10: a single-seed embed
    frequently FAILS for large cyclic peptides (>100 heavy atoms), silently
    returning 0 conformers (observed: 135-ha linear peptide 0/1, but 10/10
    with multi-seed). 10 keeps embed under ~15s for 135-ha peptides while
    making near-100% of macrocycles embeddable. Falls back to useRandomCoords
    (random-coordinate init, no distance geometry) if ETKDG yields nothing.

    Entities beyond ``ETKDG_LARGE_ENTITY_HEAVY_ATOMS`` heavy atoms are
    outside the practical embedding envelope: they get ONE bounded
    random-coordinate attempt and a typed unavailability on failure
    instead of minutes of doomed distance-geometry work.
    """
    if not isinstance(num_confs, int) or isinstance(num_confs, bool) or num_confs < 1:
        return None, "num_confs must be an integer >= 1"
    mol = Chem.Mol(mol)
    mol = Chem.AddHs(mol)

    if mol.GetNumHeavyAtoms() > ETKDG_LARGE_ENTITY_HEAVY_ATOMS:
        attempt = AllChem.ETKDGv3()
        attempt.randomSeed = random_seed
        attempt.numThreads = 0
        attempt.useRandomCoords = True
        attempt.useExpTorsionAnglePrefs = True
        attempt.useBasicKnowledge = True
        attempt.useMacrocycleTorsions = True
        attempt.timeout = ETKDG_LARGE_ENTITY_TIMEOUT_SECONDS
        status = AllChem.EmbedMultipleConfs(
            mol, numConfs=1, params=attempt
        )
        if len(status) == 0 or mol.GetNumConformers() == 0:
            return None, (
                "3D embedding unavailable for large entity ("
                f"{mol.GetNumHeavyAtoms()} heavy atoms above the "
                f"{ETKDG_LARGE_ENTITY_HEAVY_ATOMS}-heavy-atom embedding "
                "envelope)"
            )
        mol.SetProp('CYCPEP_ETKDG_ATTEMPTS', '1')
        mol.SetProp(
            'CYCPEP_EMBED_STRATEGY', 'etkdgv3_random_coords_bounded'
        )
        return mol, None

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    params.timeout = ETKDG_ATTEMPT_TIMEOUT_SECONDS

    status = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    etkdg_attempts = 1
    embed_strategy = 'etkdgv3'
    if len(status) == 0:
        # Fallback: random-coordinate init (no distance geometry) — helps when
        # the macrocycle distance matrix is poorly conditioned.
        retry = AllChem.ETKDGv3()
        retry.randomSeed = random_seed
        retry.numThreads = 0
        retry.useRandomCoords = True
        retry.useExpTorsionAnglePrefs = True
        retry.useBasicKnowledge = True
        retry.useMacrocycleTorsions = True
        retry.timeout = ETKDG_ATTEMPT_TIMEOUT_SECONDS
        status = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            params=retry,
        )
        etkdg_attempts = 2
        embed_strategy = 'etkdgv3_random_coords'
    if len(status) == 0:
        return None, "3D embedding failed (no conformers generated)"

    # ETKDG with a timeout can report conformer ids that failed to
    # materialize on large macrocycles (observed: 550-ha 3-disulfide
    # peptide); every id must be checked against the molecule before use.
    present_ids = {conf.GetId() for conf in mol.GetConformers()}
    conf_ids = [cid for cid in status if cid in present_ids]
    if not conf_ids:
        return None, "3D embedding failed (no usable conformers generated)"

    energies = []
    try:
        mmff_properties = AllChem.MMFFGetMoleculeProperties(mol)
    except ValueError:
        mmff_properties = None
        properties_available = False
    else:
        properties_available = True
    if properties_available:
        for cid in conf_ids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, mmff_properties, confId=cid
                )
                if ff:
                    energies.append((ff.CalcEnergy(), cid))
            except ValueError:
                # A conformer id that RDKit cannot service (stale or partially
                # materialized) must not abort the whole export.
                continue

    if energies and not retain_all:
        energies.sort()
        best_cid = energies[0][1]
        for cid in conf_ids:
            if cid != best_cid:
                mol.RemoveConformer(cid)

    mol.SetProp('CYCPEP_ETKDG_ATTEMPTS', str(etkdg_attempts))
    mol.SetProp('CYCPEP_EMBED_STRATEGY', embed_strategy)
    return mol, None


def _optimize(mol, force_field='mmff', max_iters=2000):
    """Energy-minimize the molecule using MMFF94 or UFF."""
    if force_field not in {'mmff', 'uff'}:
        return Chem.Mol(mol), "force_field must be 'mmff' or 'uff'"
    mol = Chem.Mol(mol)
    props = (
        AllChem.MMFFGetMoleculeProperties(mol)
        if force_field == 'mmff'
        else None
    )
    actual_force_field = force_field
    mol.SetProp(
        'CYCPEP_MMFF_AVAILABLE',
        'true' if force_field == 'mmff' and props is not None else 'false',
    )

    if force_field == 'mmff' and props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            actual_force_field = 'uff'
    else:
        ff = AllChem.UFFGetMoleculeForceField(mol)
        actual_force_field = 'uff'

    mol.SetProp('CYCPEP_REQUESTED_FORCE_FIELD', force_field)
    mol.SetProp('CYCPEP_FORCE_FIELD', actual_force_field)

    if ff is None:
        mol.SetProp('CYCPEP_OPTIMIZATION_STATUS', 'not_available')
        return mol, "no force field available"

    ff.Initialize()
    status = ff.Minimize(maxIts=max_iters)
    if status != 0:
        mol.SetProp('CYCPEP_OPTIMIZATION_STATUS', 'failed')
        mol.SetProp(
            'CYCPEP_OPTIMIZATION_ERROR',
            f"optimization did not converge (status={status})",
        )
        return mol, mol.GetProp('CYCPEP_OPTIMIZATION_ERROR')

    mol.SetProp('CYCPEP_OPTIMIZATION_STATUS', 'converged')
    return mol, None


def _ensure_optimization_provenance(mol, requested_force_field, error):
    """Attach an auditable optimization outcome to a molecule."""
    if not mol.HasProp('CYCPEP_REQUESTED_FORCE_FIELD'):
        mol.SetProp('CYCPEP_REQUESTED_FORCE_FIELD', requested_force_field)
    if not mol.HasProp('CYCPEP_FORCE_FIELD'):
        mol.SetProp('CYCPEP_FORCE_FIELD', requested_force_field)
    if error:
        current_status = (
            mol.GetProp('CYCPEP_OPTIMIZATION_STATUS')
            if mol.HasProp('CYCPEP_OPTIMIZATION_STATUS')
            else None
        )
        if current_status != 'not_available':
            mol.SetProp('CYCPEP_OPTIMIZATION_STATUS', 'failed')
        mol.SetProp('CYCPEP_OPTIMIZATION_ERROR', str(error))
    elif not mol.HasProp('CYCPEP_OPTIMIZATION_STATUS'):
        mol.SetProp('CYCPEP_OPTIMIZATION_STATUS', 'converged')


def _clear_prior_output(output_path, format_name):
    """Remove a caller-provided artifact before validating new input.

    Direct export APIs return a tuple rather than raising for invalid input.
    Clearing first prevents callers from mistaking an older artifact for the
    result of the failed request.
    """
    if output_path is None:
        return None
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        return None, f"cannot clear prior {format_name} output: {exc}"
    return None


def _paths_alias(first_path, second_path):
    """Return whether two paths refer to the same filesystem object."""
    if first_path is None or second_path is None:
        return False
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


def _reject_output_alias(input_path, output_path, format_name):
    """Reject an output that would overwrite its source input."""
    if output_path is None:
        return None
    try:
        aliases = _paths_alias(input_path, output_path)
    except (TypeError, ValueError, OSError) as exc:
        return None, f"cannot compare {format_name} input/output paths: {exc}"
    if aliases:
        return None, (
            f"cannot write {format_name} output: input and output paths "
            "refer to the same file"
        )
    return None


def smiles_to_mol2(smiles, output_path=None, force_field='mmff',
                   num_confs=10, random_seed=42):
    """Convert a SMILES string to a MOL2 file with 3D coordinates.

    Returns (output_path, error).
    """
    clear_error = _clear_prior_output(output_path, 'MOL2')
    if clear_error:
        return clear_error
    if force_field not in {'mmff', 'uff'}:
        return None, "force_field must be 'mmff' or 'uff'"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"invalid SMILES: {smiles[:60]}"

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"sanitize failed: {e}"

    mol_3d, err = _embed_3d(mol, num_confs=num_confs, random_seed=random_seed)
    if err:
        return None, err

    mol_opt, err = _optimize(mol_3d, force_field=force_field)
    if err:
        warnings.warn(err)

    # V4 validated MOL2 is a preparation parent, so explicit hydrogens must
    # remain serialized. Removing them makes Tripos readers guess protonation
    # and can change the complete InChIKey (for example neutral alcohols).
    mol_out = Chem.Mol(mol_opt)
    _ensure_optimization_provenance(mol_out, force_field, err)

    for a in mol_out.GetAtoms():
        a.SetProp('_TriposAtomName', f'{a.GetSymbol()}{a.GetIdx() + 1}')

    try:
        Chem.SanitizeMol(mol_out)
    except Exception:
        pass

    # MolToMolBlock emits MDL/SDF syntax, not Tripos MOL2. Route all MOL2
    # output through the dedicated writer so extension and format agree.
    written, write_error = mol_to_mol2(mol_out, output_path=output_path)
    if write_error:
        return None, write_error
    try:
        if output_path:
            with open(written, encoding="utf-8") as handle:
                content = handle.read()
        else:
            content = written
    except Exception:
        return None, "MOL2 roundtrip read failed"
    heavy_ids = _mol2_one_based_heavy_atom_ids(content)
    note = _coordinate_tier_note(
        "X1",
        mapped_one_based=[],
        generated_one_based=heavy_ids,
        coordinate_source="smiles_etkdgv3",
        etkdg_attempts=mol_out.GetProp('CYCPEP_ETKDG_ATTEMPTS')
        if mol_out.HasProp('CYCPEP_ETKDG_ATTEMPTS') else None,
        embed_strategy=mol_out.GetProp('CYCPEP_EMBED_STRATEGY')
        if mol_out.HasProp('CYCPEP_EMBED_STRATEGY') else None,
        force_field=mol_out.GetProp('CYCPEP_FORCE_FIELD')
        if mol_out.HasProp('CYCPEP_FORCE_FIELD') else None,
        mmff_available=mol_out.GetProp('CYCPEP_MMFF_AVAILABLE')
        if mol_out.HasProp('CYCPEP_MMFF_AVAILABLE') else None,
        optimization_status=mol_out.GetProp('CYCPEP_OPTIMIZATION_STATUS')
        if mol_out.HasProp('CYCPEP_OPTIMIZATION_STATUS') else None,
    )
    written = _prepend_mol2_tier_note(written, note, output_path)
    # Mandatory readback: a validated regenerated MOL2 must round-trip to
    # the requested SMILES connectivity.  Connectivity divergence fails
    # closed; full-InChIKey-only divergence (e.g. stereo perceived from 3D
    # geometry) is retained but recorded explicitly in a second leading
    # comment line placed AFTER the tier note so both survive header
    # parsing.
    roundtrip_inchikey, roundtrip_error = _mol2_roundtrip_full_inchikey(
        note + content
    )
    expected = Chem.MolFromSmiles(smiles)
    expected_inchikey = Chem.MolToInchiKey(expected) if expected else None
    expected_connectivity = (
        expected_inchikey.split("-", 1)[0] if expected_inchikey else None
    )
    roundtrip_connectivity = (
        roundtrip_inchikey.split("-", 1)[0] if roundtrip_inchikey else None
    )
    if roundtrip_error or not expected_connectivity or (
        roundtrip_connectivity != expected_connectivity
    ):
        if output_path and written:
            try:
                os.remove(written)
            except OSError:
                pass
        return None, (
            roundtrip_error
            or f"MOL2 readback connectivity mismatch: "
            f"{roundtrip_connectivity} != {expected_connectivity}"
        )
    stereo_note = ""
    if (
        expected_inchikey
        and roundtrip_inchikey
        and expected_inchikey != roundtrip_inchikey
    ):
        stereo_note = (
            "# stereo_roundtrip_divergence=true "
            f"expected_full_inchikey={expected_inchikey} "
            f"observed_full_inchikey={roundtrip_inchikey}\n"
        )
    if stereo_note:
        # The tier note is already the first line; insert the stereo note
        # directly after it (comment lines do not affect MOL2 parsing, and
        # the ledger scans every leading comment line).
        if output_path:
            with open(written, encoding="utf-8") as handle:
                raw = handle.read()
            with open(written, "w", encoding="utf-8", newline="") as handle:
                handle.write(note + stereo_note + raw[len(note):])
        else:
            written = note + stereo_note + written[len(note):]
    # Keep the historical two-value return shape while surfacing a failed
    # minimization in the second value.  The serialized artifact still carries
    # the same provenance for callers that intentionally retain it.
    return written, err


def smiles_to_sdf(smiles, output_path, force_field='mmff',
                  num_confs=1, random_seed=42):
    """Convert SMILES to an SDF file with 3D coordinates."""
    clear_error = _clear_prior_output(output_path, 'SDF')
    if clear_error:
        return clear_error
    if force_field not in {'mmff', 'uff'}:
        return None, "force_field must be 'mmff' or 'uff'"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"invalid SMILES: {smiles[:60]}"

    Chem.SanitizeMol(mol)
    mol_3d, err = _embed_3d(mol, num_confs=num_confs, random_seed=random_seed)
    if err:
        return None, err

    mol_opt, opt_err = _optimize(mol_3d, force_field=force_field)
    if opt_err:
        warnings.warn(opt_err)
    mol_out = Chem.RemoveHs(mol_opt, sanitize=False)
    _ensure_optimization_provenance(mol_out, force_field, opt_err)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        with Chem.SDWriter(output_path) as w:
            if w.write(mol_out) is False:
                raise OSError("RDKit SDF writer rejected molecule")
    except Exception as exc:
        try:
            os.remove(output_path)
        except OSError:
            pass
        return None, f"SDF write failed: {exc}"

    # Preserve the legacy (path, error) tuple while making a failed
    # minimization visible to the caller instead of warning-only success.
    return output_path, opt_err


def batch_export(smiles_map, output_dir, format='mol2', force_field='mmff'):
    """Export multiple SMILES to 3D files."""
    if format not in {'mol2', 'sdf'}:
        raise ValueError("format must be 'mol2' or 'sdf'")
    if force_field not in {'mmff', 'uff'}:
        raise ValueError("force_field must be 'mmff' or 'uff'")
    items = list(smiles_map.items() if isinstance(smiles_map, dict) else smiles_map)
    results = []
    normalized_names = []
    normalization_errors = []
    for name, _ in items:
        safe_name = str(name).replace('/', '_').replace('\\', '_')
        error = None
        if not safe_name or safe_name in {'.', '..'}:
            error = "output name must not be empty or a dot path"
        elif ':' in safe_name:
            error = "output name must not contain ':'"
        elif any(ord(char) < 32 for char in safe_name):
            error = "output name contains a control character"
        normalized_names.append(safe_name)
        normalization_errors.append(error)
    normalized_keys = [
        value.casefold()
        for value, error in zip(normalized_names, normalization_errors)
        if error is None
    ]
    collisions = {
        value for value in normalized_keys if normalized_keys.count(value) > 1
    }

    for (name, smi), safe_name, name_error in zip(
        items, normalized_names, normalization_errors
    ):
        ext = '.mol2' if format == 'mol2' else '.sdf'
        if name_error is not None:
            results.append((name, None, f"invalid output name: {name_error}"))
            continue
        path = os.path.join(output_dir, safe_name + ext)

        if safe_name.casefold() in collisions:
            results.append((
                name,
                None,
                f"normalized output name collision: {safe_name!r}",
            ))
            continue

        if format == 'mol2':
            written, err = smiles_to_mol2(
                smi, output_path=path, force_field=force_field
            )
        else:
            written, err = smiles_to_sdf(
                smi, output_path=path, force_field=force_field
            )
        if err and written:
            try:
                os.remove(written)
            except OSError:
                pass

        results.append((name, path if not err else None, err))

    return results

# ── Ported from mirror: ensemble stats + mol2 export (Lite was missing these) ─
# Formal M5 development data showed the 50-conformer proxy entering its timeout
# regime at 94 heavy atoms. Fail closed just below that observed boundary.
STABILITY_MAX_HEAVY_ATOMS = 90
ENSEMBLE_ETKDG_TIMEOUT_SECONDS = 5
_PEPTIDE_BACKBONE_SMARTS = Chem.MolFromSmarts("[NX3,NX4][CX4][CX3](=[OX1])")


def _backbone_and_sidechain_indices(mol):
    backbone = sorted({
        atom_index
        for match in mol.GetSubstructMatches(_PEPTIDE_BACKBONE_SMARTS)
        for atom_index in match
    })
    sidechain = sorted(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in backbone
    )
    return backbone, sidechain


def _aligned_subset_rmsds(mol, conformer_ids, align_indices, measure_indices):
    """Measure a fixed atom subset after identity-mapped backbone alignment."""
    import math
    from rdkit.Chem import rdMolAlign

    if len(align_indices) < 3 or not measure_indices:
        return []
    atom_map = [(index, index) for index in align_indices]
    values = []
    for left_index, left_cid in enumerate(conformer_ids):
        for right_cid in conformer_ids[left_index + 1:]:
            probe = Chem.Mol(mol)
            try:
                rdMolAlign.AlignMol(
                    probe,
                    mol,
                    prbCid=left_cid,
                    refCid=right_cid,
                    atomMap=atom_map,
                )
                probe_conf = probe.GetConformer(left_cid)
                reference_conf = mol.GetConformer(right_cid)
                squared = 0.0
                for atom_index in measure_indices:
                    left = probe_conf.GetAtomPosition(atom_index)
                    right = reference_conf.GetAtomPosition(atom_index)
                    squared += (
                        (left.x - right.x) ** 2
                        + (left.y - right.y) ** 2
                        + (left.z - right.z) ** 2
                    )
                values.append(math.sqrt(squared / len(measure_indices)))
            except Exception:
                continue
    return values


def compute_conformer_ensemble_stats(
    smiles,
    num_confs=50,
    random_seed=42,
    force_field='mmff',
    optimize=True,
    energy_window=None,
    max_heavy_atoms=STABILITY_MAX_HEAVY_ATOMS,
):
    """Generate an ensemble and return computational dispersion statistics.

    A molecule that samples a wider modeled conformational space has higher
    pairwise RMSD and energy dispersion. This is a computational proxy, not an
    experimental stability measurement.

    Args:
        smiles: input SMILES string
        num_confs: number of conformers to embed (more = better statistics)
        random_seed: ETKDGv3 random seed
        force_field: 'mmff' or 'uff' for energy minimization
        optimize: minimize each conformer before measuring energy/RMSD
        energy_window: if set (kcal/mol), only keep confs within this window
            of the global minimum before computing RMSD stats
        max_heavy_atoms: refuse inputs larger than this (returns an error
            instead of hanging in ETKDGv3). Defaults to
            STABILITY_MAX_HEAVY_ATOMS; pass None to disable the guard.

    Returns:
        (stats_dict, error). On success error is None and stats_dict has
        requested_force_field, force_field (the actual field used), and
        optimization_status (converged, failed, or not_run), in addition to:
          num_confs        — conformers successfully embedded
          num_kept         — conformers after energy-window filtering
          energy_min       — lowest MMFF/UFF energy (kcal/mol)
          energy_max       — highest energy among kept confs
          energy_mean      — mean energy
          energy_std       — energy standard deviation (spread)
          energy_range     — max - min (kcal/mol)
          rmsd_mean        — mean symmetry-aware heavy-atom RMSD (Å)
          backbone_rmsd_*  — backbone RMSD after identity-mapped alignment
          sidechain_rmsd_* — side-chain RMSD after backbone alignment
          flexibility_proxy — frozen composite computational proxy
    """
    if force_field not in {'mmff', 'uff'}:
        return None, "force_field must be 'mmff' or 'uff'"
    if not isinstance(num_confs, int) or isinstance(num_confs, bool) or num_confs < 1:
        return None, "num_confs must be an integer >= 1"
    if energy_window is not None:
        try:
            window_value = float(energy_window)
        except (TypeError, ValueError, OverflowError):
            return None, (
                "energy_window must be a finite non-negative number, "
                f"got {energy_window!r}"
            )
        if not math.isfinite(window_value) or window_value < 0:
            return None, (
                "energy_window must be a finite non-negative number, "
                f"got {energy_window!r}"
            )
        energy_window = window_value
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"invalid SMILES: {smiles[:60]}"

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"sanitize failed: {e}"

    if max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return None, (
            f"molecule too large for conformer ensemble: "
            f"{mol.GetNumHeavyAtoms()} heavy atoms > {max_heavy_atoms} "
            f"(resource_limit; raise max_heavy_atoms only for an explicitly "
            f"budgeted development run)")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    params.timeout = ENSEMBLE_ETKDG_TIMEOUT_SECONDS

    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if len(cids) == 0:
        return None, "3D embedding failed (no conformers generated)"

    # ── Energy minimize + collect per-conformer energy ──
    props = AllChem.MMFFGetMoleculeProperties(mol) if force_field == 'mmff' else None
    energies = []  # (energy, cid)
    nonconverged_count = 0
    actual_force_fields = []
    for cid in cids:
        actual_force_field = force_field
        if force_field == 'mmff' and props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                actual_force_field = 'uff'
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            actual_force_field = 'uff'
        if ff is None:
            continue
        actual_force_fields.append(actual_force_field)
        if optimize:
            ff.Initialize()
            if ff.Minimize(maxIts=1000) != 0:
                nonconverged_count += 1
        energies.append((ff.CalcEnergy(), cid))

    if not energies:
        return None, "no force field available for energy evaluation"

    energies.sort()
    e_min = energies[0][0]

    # ── Optional energy-window filtering ──
    if energy_window is not None:
        kept = [(e, c) for (e, c) in energies if e - e_min <= energy_window]
    else:
        kept = energies
    kept_cids = [c for (_, c) in kept]
    kept_energies = [e for (e, _) in kept]
    if len(kept_cids) < 2:
        return None, (
            "not assessable: conformer dispersion requires at least two "
            f"force-field conformers after filtering; observed {len(kept_cids)}"
        )

    # ── Pairwise heavy-atom RMSD over kept conformers ──
    # Use a heavy-atom-only copy so H positions don't dominate RMSD.
    mol_noh = Chem.RemoveHs(mol)
    rmsds = []
    n = len(kept_cids)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # GetBestRMS aligns then computes symmetry-corrected RMSD
                probe = Chem.Mol(mol_noh)
                r = AllChem.GetBestRMS(
                    probe, mol_noh, prbId=kept_cids[i], refId=kept_cids[j]
                )
                rmsds.append(r)
            except Exception:
                continue

    backbone_indices, sidechain_indices = _backbone_and_sidechain_indices(mol_noh)
    if len(backbone_indices) < 3:
        return None, (
            "not assessable: peptide backbone could not be identified for "
            "backbone/side-chain decomposition"
        )
    backbone_rmsds = _aligned_subset_rmsds(
        mol_noh, kept_cids, backbone_indices, backbone_indices
    )
    sidechain_rmsds = _aligned_subset_rmsds(
        mol_noh, kept_cids, backbone_indices, sidechain_indices
    )
    if not backbone_rmsds:
        return None, "not assessable: no pairwise backbone RMSD could be computed"

    # ── Aggregate ──
    e_max = max(kept_energies)
    e_mean = statistics.mean(kept_energies)
    e_std = statistics.pstdev(kept_energies) if len(kept_energies) > 1 else 0.0

    if rmsds:
        rmsd_mean = statistics.mean(rmsds)
        rmsd_max = max(rmsds)
        rmsd_std = statistics.pstdev(rmsds) if len(rmsds) > 1 else 0.0
    else:
        return None, "not assessable: no pairwise heavy-atom RMSD could be computed"

    backbone_mean = statistics.mean(backbone_rmsds)
    backbone_max = max(backbone_rmsds)
    backbone_std = (
        statistics.pstdev(backbone_rmsds) if len(backbone_rmsds) > 1 else 0.0
    )
    sidechain_mean = statistics.mean(sidechain_rmsds) if sidechain_rmsds else None
    sidechain_max = max(sidechain_rmsds) if sidechain_rmsds else None
    sidechain_std = (
        statistics.pstdev(sidechain_rmsds) if len(sidechain_rmsds) > 1 else 0.0
    ) if sidechain_rmsds else None

    # Frozen computational proxy. It is not an experimental stability metric.
    flexibility_proxy = rmsd_mean + 0.1 * e_std

    force_field_values = sorted(set(actual_force_fields))
    actual_force_field = (
        force_field_values[0]
        if len(force_field_values) == 1
        else 'mixed'
    )
    optimization_status = (
        'not_run'
        if not optimize
        else ('failed' if nonconverged_count else 'converged')
    )
    stats = {
        "num_confs": len(cids),
        "num_kept": len(kept_cids),
        "pair_count": len(rmsds),
        "status": "success" if optimization_status != 'failed' else "failed",
        "optimization_status": optimization_status,
        "optimization_converged": (
            None if not optimize else nonconverged_count == 0
        ),
        "optimization_nonconverged_count": nonconverged_count,
        "optimization_requested": bool(optimize),
        "proxy_definition": "mean_pairwise_heavy_atom_rmsd + 0.1 * energy_sd",
        "energy_units": "kcal/mol",
        "rmsd_units": "angstrom",
        "random_seed": random_seed,
        "requested_force_field": force_field,
        "force_field": actual_force_field,
        "force_field_fallback": actual_force_field != force_field,
        "force_fields_used": force_field_values,
        "energy_min": round(e_min, 3),
        "energy_max": round(e_max, 3),
        "energy_mean": round(e_mean, 3),
        "energy_std": round(e_std, 3),
        "energy_range": round(e_max - e_min, 3),
        "rmsd_mean": round(rmsd_mean, 3),
        "rmsd_max": round(rmsd_max, 3),
        "rmsd_std": round(rmsd_std, 3),
        "backbone_atom_count": len(backbone_indices),
        "backbone_rmsd_mean": round(backbone_mean, 3),
        "backbone_rmsd_max": round(backbone_max, 3),
        "backbone_rmsd_std": round(backbone_std, 3),
        "sidechain_atom_count": len(sidechain_indices),
        "sidechain_assessable": bool(sidechain_rmsds),
        "sidechain_rmsd_mean": (
            round(sidechain_mean, 3) if sidechain_mean is not None else None
        ),
        "sidechain_rmsd_max": (
            round(sidechain_max, 3) if sidechain_max is not None else None
        ),
        "sidechain_rmsd_std": (
            round(sidechain_std, 3) if sidechain_std is not None else None
        ),
        "flexibility_proxy": round(flexibility_proxy, 3),
        # Backward-compatible alias; new scientific outputs use flexibility_proxy.
        "flexibility": round(flexibility_proxy, 3),
    }
    optimization_error = None
    if optimization_status == 'failed':
        optimization_error = (
            "optimization did not converge for "
            f"{nonconverged_count}/{len(cids)} conformers"
        )
    return stats, optimization_error


def _assign_sybyl_atom_type(atom):
    """Assign a SYBYL atom type based on RDKit atom properties."""
    elem = atom.GetSymbol()
    hyb = atom.GetHybridization()
    if atom.GetIsAromatic():
        return f'{elem}.ar'
    if elem == 'C':
        if hyb == Chem.HybridizationType.SP3:
            return 'C.3'
        elif hyb == Chem.HybridizationType.SP2:
            return 'C.2'
        return 'C.1'
    elif elem == 'N':
        if atom.GetFormalCharge() > 0 and atom.GetDegree() >= 4:
            return 'N.4'
        if any(
            neighbor.GetSymbol() == 'C'
            and any(
                other.GetSymbol() in {'O', 'S'}
                and atom.GetOwningMol().GetBondBetweenAtoms(
                    neighbor.GetIdx(), other.GetIdx()
                ).GetBondType() == Chem.BondType.DOUBLE
                for other in neighbor.GetNeighbors()
                if other.GetIdx() != atom.GetIdx()
            )
            for neighbor in atom.GetNeighbors()
        ):
            return 'N.am'
        if hyb == Chem.HybridizationType.SP3:
            return 'N.3'
        elif hyb == Chem.HybridizationType.SP2:
            return 'N.2'
        return 'N.1'
    elif elem == 'O':
        if atom.GetFormalCharge() < 0 and any(
            neighbor.GetSymbol() == 'C'
            and any(other.GetSymbol() == 'O' for other in neighbor.GetNeighbors())
            for neighbor in atom.GetNeighbors()
        ):
            return 'O.co2'
        if hyb == Chem.HybridizationType.SP3:
            return 'O.3'
        return 'O.2'
    elif elem == 'S':
        if hyb == Chem.HybridizationType.SP3:
            return 'S.3'
        return 'S.2'
    elif elem == 'P':
        return 'P.3'
    return elem


def _mol2_roundtrip_full_inchikey(content):
    """Read a MOL2 block with RDKit while honoring Tripos formal charges."""
    try:
        molecule = Chem.MolFromMol2Block(
            content,
            sanitize=False,
            removeHs=False,
        )
        if molecule is None:
            return None, "RDKit could not read the MOL2 block"
        for atom_index, charge in _mol2_unity_formal_charges(content).items():
            if atom_index < 0 or atom_index >= molecule.GetNumAtoms():
                return None, (
                    "MOL2 formal-charge atom index is out of range: "
                    f"{atom_index + 1}"
                )
            molecule.GetAtomWithIdx(atom_index).SetFormalCharge(charge)
        Chem.SanitizeMol(molecule)
        inchikey = Chem.MolToInchiKey(molecule)
        if not inchikey:
            return None, "RDKit did not produce a MOL2 roundtrip InChIKey"
        return inchikey, None
    except Exception as exc:
        return None, f"MOL2 roundtrip identity check failed: {exc}"


def _mol2_one_based_heavy_atom_ids(content):
    """Parse actual one-based heavy-atom ids from a MOL2 ATOM block.

    Tier headers and receipts must never carry synthetic ranges: every id
    emitted here is read back from the artifact's own ``@<TRIPOS>ATOM``
    id column, so hydrogen interleaving and writer reordering are honored.
    """
    heavy = []
    in_atoms = False
    for line in content.splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        element = parts[5].split(".")[0]
        if element and element.upper() != "H":
            heavy.append(int(parts[0]))
    return sorted(heavy)


def _coordinate_tier_note(tier, *, mapped_one_based, generated_one_based,
                           **tokens):
    """Build the leading MOL2 coordinate-tier comment line.

    ``mapped_one_based``/``generated_one_based`` must come from
    :func:`_mol2_one_based_heavy_atom_ids` (or be derived from ids parsed
    out of the artifact).  Extra keyword tokens become ``key=value`` fields;
    ``None`` values are skipped and booleans serialize as ``true``/``false``.
    """
    fields = [
        f"coordinate_tier={tier}",
        f"source_mapped_atoms={len(mapped_one_based)}",
        f"generated_heavy_atoms={len(generated_one_based)}",
        "generated_heavy_atom_indices="
        + ",".join(str(index) for index in generated_one_based),
        "mapped_heavy_atom_indices="
        + ",".join(str(index) for index in mapped_one_based),
        "atom_index_convention=mol2_one_based",
    ]
    for key, value in tokens.items():
        if value is None:
            continue
        if isinstance(value, bool):
            fields.append(f"{key}={'true' if value else 'false'}")
        else:
            fields.append(f"{key}={value}")
    return "# " + " ".join(fields) + "\n"


def _prepend_mol2_tier_note(produced, note, output_path):
    """Prepend a tier note to a written MOL2 path or in-memory block."""
    if output_path:
        with open(produced, encoding="utf-8") as handle:
            raw = handle.read()
        with open(produced, "w", encoding="utf-8", newline="") as handle:
            handle.write(note + raw)
        return produced
    return note + str(produced)


def mol_to_mol2(mol, output_path=None):
    """Write an RDKit Mol (with existing 3D conformer) to Tripos MOL2 format.

    Returns (output_path, error).
    """
    if output_path:
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            return None, f"cannot clear prior MOL2 output: {exc}"
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
        # Kekulization below deliberately clears aromatic flags so that MOL2
        # bond records contain concrete 1/2 orders.  Capture SYBYL atom types
        # first; otherwise aromatic atoms are mislabelled as C.2/N.2.
        sybyl_atom_types = [
            _assign_sybyl_atom_type(atom) for atom in mol.GetAtoms()
        ]
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as exc:
        return None, f"MOL2 graph normalization failed: {exc}"
    if mol.GetNumConformers() == 0:
        return None, "molecule has no 3D conformer"
    conf = mol.GetConformer()
    for atom_index in range(mol.GetNumAtoms()):
        position = conf.GetAtomPosition(atom_index)
        if not all(math.isfinite(value) for value in (
            position.x, position.y, position.z
        )):
            return None, f"non-finite MOL2 coordinate at atom {atom_index + 1}"

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    lines = []
    lines.append("@<TRIPOS>MOLECULE")
    lines.append("molecule")
    residue_keys = []
    atom_residues = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None:
            key = (
                info.GetChainId().strip(), info.GetResidueNumber(),
                info.GetInsertionCode().strip(), info.GetResidueName().strip(),
            )
            atom_name = info.GetName().strip() or atom.GetSymbol()
        else:
            key = (
                atom.GetProp('_TriposChainId') if atom.HasProp('_TriposChainId') else '',
                atom.GetIntProp('_TriposResidueNumber')
                if atom.HasProp('_TriposResidueNumber') else 1,
                atom.GetProp('_TriposInsertionCode')
                if atom.HasProp('_TriposInsertionCode') else '',
                atom.GetProp('_TriposResidueName')
                if atom.HasProp('_TriposResidueName') else 'RES',
            )
            atom_name = (
                atom.GetProp('_TriposAtomName')
                if atom.HasProp('_TriposAtomName') else atom.GetSymbol()
            )
        if key not in residue_keys:
            residue_keys.append(key)
        atom_residues.append((residue_keys.index(key) + 1, key, atom_name))

    lines.append(f"{num_atoms} {num_bonds} {len(residue_keys)} 0 0")
    lines.append("SMALL")
    lines.append("USER_CHARGES")
    lines.append("")

    provenance = []
    for prop in (
        'CYCPEP_REQUESTED_FORCE_FIELD',
        'CYCPEP_FORCE_FIELD',
        'CYCPEP_MMFF_AVAILABLE',
        'CYCPEP_OPTIMIZATION_STATUS',
        'CYCPEP_OPTIMIZATION_ERROR',
        'CYCPEP_ETKDG_ATTEMPTS',
        'CYCPEP_EMBED_STRATEGY',
    ):
        if mol.HasProp(prop):
            provenance.append(f"{prop}={mol.GetProp(prop)}")
    if provenance:
        lines.append("@<TRIPOS>COMMENT")
        lines.extend(provenance)

    lines.append("@<TRIPOS>ATOM")
    for i, atom in enumerate(mol.GetAtoms(), 1):
        pos = conf.GetAtomPosition(i - 1)
        sybyl = sybyl_atom_types[i - 1]
        charge = atom.GetFormalCharge()
        subst_id, residue_key, atom_name = atom_residues[i - 1]
        _, residue_number, insertion_code, residue_name = residue_key
        subst_name = f"{residue_name}{residue_number}{insertion_code}"[:8]
        lines.append(
            f"{i:>6} {atom_name:<8}"
            f" {pos.x:>10.4f} {pos.y:>10.4f} {pos.z:>10.4f}"
            f" {sybyl:<6} {subst_id:>4} {subst_name:<8} {charge:>7.4f}"
        )

    charged_atoms = [
        (atom.GetIdx() + 1, atom.GetFormalCharge())
        for atom in mol.GetAtoms()
        if atom.GetFormalCharge()
    ]
    if charged_atoms:
        # The MOL2 charge column contains partial charges. Formal charges use
        # Tripos UNITY attributes so independent readers can reconstruct the
        # same protonation state.
        lines.append("@<TRIPOS>UNITY_ATOM_ATTR")
        for atom_id, charge in charged_atoms:
            lines.append(f"{atom_id} 1")
            lines.append(f"charge {charge}")

    lines.append("@<TRIPOS>BOND")
    bond_type_map = {
        Chem.BondType.SINGLE: '1', Chem.BondType.DOUBLE: '2',
        Chem.BondType.TRIPLE: '3',
    }
    for i, bond in enumerate(mol.GetBonds(), 1):
        bond_type = bond.GetBondType()
        if bond_type not in bond_type_map:
            return None, (
                f"unsupported MOL2 bond type at bond {i}: {bond_type.name}"
            )
        order = bond_type_map[bond_type]
        lines.append(
            f"{i:>6} {bond.GetBeginAtomIdx() + 1:>5}"
            f" {bond.GetEndAtomIdx() + 1:>5} {order}"
        )

    lines.append("@<TRIPOS>SUBSTRUCTURE")
    for index, (chain, residue_number, insertion_code, residue_name) in enumerate(
        residue_keys, 1
    ):
        root_atom = next(
            atom_index + 1
            for atom_index, (subst_id, _, _) in enumerate(atom_residues)
            if subst_id == index
        )
        subst_name = f"{residue_name}{residue_number}{insertion_code}"[:8]
        chain_value = chain or '****'
        lines.append(
            f"{index:>6} {subst_name:<8} {root_atom:>5} RESIDUE 1 {chain_value:<4}"
        )

    content = '\n'.join(lines) + '\n'
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
        except Exception as exc:
            try:
                os.remove(output_path)
            except OSError:
                pass
            return None, f"MOL2 write failed: {exc}"
        return output_path, None
    return content, None


def _candidate_graph_to_mol(candidate_graph, candidate_smiles):
    """Materialize one source-bound inferred graph as an RDKit molecule.

    Layering contract: this helper returns a ``Chem.Mol`` or an error and
    never materializes a file.  When the graph diverges from the candidate
    SMILES it returns a ``GRAPH_SMILES_DIVERGENCE:``-prefixed error; the
    decision to rematerialize from SMILES at X1 belongs solely to the outer
    materialization layer (:func:`_result_first_candidate_to_mol2`).
    """
    if not isinstance(candidate_graph, dict):
        return None, "result-first candidate has no source-bound chemical graph"
    atoms = candidate_graph.get("atoms")
    bonds = candidate_graph.get("bonds")
    if not isinstance(atoms, list) or not atoms:
        return None, "result-first candidate graph has no atoms"
    if not isinstance(bonds, list) or not bonds:
        return None, "result-first candidate graph has no bonds"
    expected = Chem.MolFromSmiles(str(candidate_smiles or ""))
    if expected is None or expected.GetNumAtoms() == 0:
        return None, "result-first candidate SMILES is not parseable"

    editable = Chem.RWMol()
    serial_to_index = {}
    try:
        for index, row in enumerate(atoms):
            serial = int(row["serial"])
            if serial in serial_to_index:
                return None, f"duplicate candidate graph serial {serial}"
            atom = Chem.Atom(str(row["element"]))
            atom.SetFormalCharge(int(row.get("formal_charge") or 0))
            atom.SetIsotope(int(row.get("isotope") or 0))
            atom.SetNumExplicitHs(int(row.get("num_explicit_hs") or 0))
            atom.SetNoImplicit(bool(row.get("no_implicit", False)))
            atom.SetNumRadicalElectrons(
                int(row.get("num_radical_electrons") or 0)
            )
            info = Chem.AtomPDBResidueInfo()
            info.SetSerialNumber(serial)
            info.SetName(str(row.get("name") or atom.GetSymbol()))
            info.SetResidueName(str(row.get("residue") or "RES"))
            info.SetResidueNumber(int(row.get("residue_number") or 0))
            info.SetInsertionCode(str(row.get("insertion_code") or ""))
            info.SetChainId(str(row.get("chain") or ""))
            atom.SetPDBResidueInfo(info)
            editable.AddAtom(atom)
            serial_to_index[serial] = index
        for row in bonds:
            left = int(row["a"])
            right = int(row["b"])
            if left == right:
                return None, f"self bond in candidate graph at serial {left}"
            if left not in serial_to_index or right not in serial_to_index:
                return None, (
                    "candidate graph bond references an unknown serial: "
                    f"{left}-{right}"
                )
            order = float(row["order"])
            aromatic = bool(row.get("is_aromatic")) or order == 1.5
            if aromatic:
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = {
                    1.0: Chem.BondType.SINGLE,
                    2.0: Chem.BondType.DOUBLE,
                    3.0: Chem.BondType.TRIPLE,
                }.get(order)
            if bond_type is None:
                return None, (
                    "unsupported inferred bond order "
                    f"{order} at serials {left}-{right}"
                )
            left_index = serial_to_index[left]
            right_index = serial_to_index[right]
            editable.AddBond(left_index, right_index, bond_type)
            if aromatic:
                editable.GetAtomWithIdx(left_index).SetIsAromatic(True)
                editable.GetAtomWithIdx(right_index).SetIsAromatic(True)
                editable.GetBondBetweenAtoms(
                    left_index, right_index
                ).SetIsAromatic(True)
        molecule = editable.GetMol()
        conformer = Chem.Conformer(len(atoms))
        for index, row in enumerate(atoms):
            xyz = row.get("xyz")
            if (
                not isinstance(xyz, (list, tuple))
                or len(xyz) != 3
                or not all(math.isfinite(float(value)) for value in xyz)
            ):
                return None, (
                    "candidate graph has invalid coordinates at serial "
                    f"{row.get('serial')}"
                )
            conformer.SetAtomPosition(
                index, tuple(float(value) for value in xyz)
            )
        conformer.Set3D(True)
        molecule.AddConformer(conformer, assignId=True)
        Chem.SanitizeMol(molecule)
        Chem.RemoveStereochemistry(molecule)
    except Exception as exc:
        return None, f"result-first candidate graph materialization failed: {exc}"

    expected_inchikey = Chem.MolToInchiKey(expected)
    observed_inchikey = Chem.MolToInchiKey(molecule)
    expected_connectivity = (
        expected_inchikey.split("-", 1)[0] if expected_inchikey else None
    )
    observed_connectivity = (
        observed_inchikey.split("-", 1)[0] if observed_inchikey else None
    )
    if (
        not expected_connectivity
        or observed_connectivity != expected_connectivity
    ):
        # Typed divergence for the outer layer: never materialize from here.
        return None, (
            "GRAPH_SMILES_DIVERGENCE: expected "
            f"{expected_connectivity} observed {observed_connectivity}"
        )
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except Exception as exc:
        return None, f"result-first candidate kekulization failed: {exc}"
    return molecule, None


def _result_first_binding_error(
    result,
    *,
    coordinate_input_evidence,
    chain_id,
    minimum_macrocycle_ring_size,
    require_empty_persistent_overlay,
) -> str | None:
    """Fail closed unless the result is source-bound to THIS export request.

    Upstream contract (explicit): ``result_first.reconstruct_structure``
    installs ``provenance['request_binding']`` on every internally
    reconstructed result, and any caller that materializes a precomputed
    result must run ``result_first.bind_reconstruction_result`` against the
    same prepared input before handing it to ``pdb_to_mol2``.  A result
    whose binding is missing, or whose binding names another source file,
    chain, option set, or monomer-registry epoch, is refused: export must
    never rematerialize a reconstruction across inputs, chains, options, or
    registry states.
    """
    provenance = getattr(result, "provenance", None)
    binding = (
        provenance.get("request_binding")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(binding, dict):
        return (
            "result-first source binding is missing: the reconstruction was "
            "not bound to this request (reconstruct_structure, or "
            "bind_reconstruction_result on a prepared input, must install "
            "request_binding before export)"
        )
    schema = str(binding.get("schema_version") or "")
    if not schema.startswith("1."):
        return (
            "result-first source binding schema is unsupported: "
            f"{schema or 'missing'}"
        )
    requested_chain_id = binding.get("requested_chain_id")
    if not isinstance(requested_chain_id, str) or not requested_chain_id.strip():
        return "result-first source binding has no requested_chain_id"
    evidence = (
        coordinate_input_evidence
        if isinstance(coordinate_input_evidence, dict)
        else {}
    )
    from ..paths._map_utils import registry_epoch

    expected = {
        "source_sha256": evidence.get("source_sha256"),
        "normalized_sha256": evidence.get("normalized_sha256"),
        "normalized_chain_id": str(chain_id),
        "minimum_macrocycle_ring_size": int(
            minimum_macrocycle_ring_size
        ),
        "require_empty_persistent_overlay": bool(
            require_empty_persistent_overlay
        ),
        "infer_bond_orders": True,
        "registry_epoch": int(registry_epoch()),
    }
    mismatches = [
        f"{name}: bound {binding.get(name)!r} != this request {value!r}"
        for name, value in expected.items()
        if value is not None and binding.get(name) != value
    ]
    if mismatches:
        return (
            "result-first source binding mismatch "
            "(possible cross-input/cross-chain/cross-option/cross-registry "
            "reuse): " + "; ".join(mismatches)
        )
    return None


def _smiles_only_to_mol2(
    candidate_smiles,
    output_path=None,
    *,
    fallback_origin="smiles_only",
    graph_divergence_detail=None,
):
    """Maximum acceptance: SMILES -> ETKDG -> MOL2 at X1 with tier header.

    This is the outer materialization layer's SMILES fallback.  The X1 tier
    header records actual one-based heavy-atom ids parsed back from the
    written artifact plus embedding/optimization provenance and the fallback
    origin, so downstream receipts never mistake regenerated coordinates
    for source evidence.
    """
    molecule = Chem.MolFromSmiles(str(candidate_smiles or ""))
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None, "candidate SMILES is not parseable"
    try:
        molecule = Chem.AddHs(molecule)
    except Exception:
        return None, "hydrogen addition failed"

    etkdg_attempts = 1
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 500
    params.useRandomCoords = False
    if AllChem.EmbedMolecule(molecule, params) != 0:
        # retry with random coordinates
        etkdg_attempts = 2
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = 99
        params2.maxIterations = 500
        params2.useRandomCoords = True
        if AllChem.EmbedMolecule(molecule, params2) != 0:
            return None, "ETKDG embedding failed"
    mmff_available = True
    optimization = "mmff"
    try:
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)
    except Exception:
        mmff_available = False
        optimization = "uff"
        try:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
        except Exception:
            optimization = "none"
    produced, error = mol_to_mol2(molecule, output_path)
    if error or produced is None:
        return produced, error
    try:
        if output_path:
            with open(produced, encoding="utf-8") as handle:
                content = handle.read()
        else:
            content = produced
    except Exception:
        return None, "MOL2 roundtrip read failed"
    heavy_ids = _mol2_one_based_heavy_atom_ids(content)
    note = _coordinate_tier_note(
        "X1",
        mapped_one_based=[],
        generated_one_based=heavy_ids,
        fallback_origin=fallback_origin,
        graph_smiles_divergence=bool(graph_divergence_detail),
        graph_divergence_detail=graph_divergence_detail,
        etkdg_attempts=etkdg_attempts,
        mmff_available=mmff_available,
        optimization=optimization,
    )
    produced = _prepend_mol2_tier_note(produced, note, output_path)
    rik, _ = _mol2_roundtrip_full_inchikey(note + content)
    exp = Chem.MolFromSmiles(candidate_smiles)
    eik = Chem.MolToInchiKey(exp) if exp else None
    ec = eik.split("-", 1)[0] if eik else None
    rc = rik.split("-", 1)[0] if rik else None
    if not ec or not rc or rc != ec:
        if output_path and produced:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, f"connectivity mismatch: {ec} vs {rc}"
    return produced, None


def _result_first_candidate_to_mol2(result, output_path=None):
    """Write an explicitly selected inferred candidate with source coordinates.

    Layering contract: the graph helper only ever returns a molecule.  When
    the source-bound graph diverges from the candidate SMILES, this outer
    materialization layer alone decides to rematerialize from SMILES at X1,
    and the divergence is recorded in the artifact header rather than being
    silently substituted.  The graph-success route emits an X3 tier header
    whose atom ids are parsed back from the written artifact.
    """
    candidate_smiles = getattr(result, "candidate_smiles", None)
    candidate_graph = getattr(result, "candidate_graph", None)
    if not candidate_smiles:
        candidate_smiles = getattr(result, "smiles", None)
    if not candidate_smiles:
        return None, "result-first reconstruction produced no candidate SMILES"
    if not isinstance(candidate_graph, dict):
        # Maximum acceptance: no source-bound graph; regenerate at X1.
        return _smiles_only_to_mol2(
            candidate_smiles,
            output_path,
            fallback_origin="smiles_only_no_source_graph",
        )
    molecule, error = _candidate_graph_to_mol(
        candidate_graph, candidate_smiles
    )
    if error or molecule is None:
        if error and error.startswith("GRAPH_SMILES_DIVERGENCE"):
            produced, fallback_error = _smiles_only_to_mol2(
                candidate_smiles,
                output_path,
                fallback_origin="smiles_after_graph_divergence",
                graph_divergence_detail=error,
            )
            if fallback_error is None and produced is not None:
                return produced, None
            return None, (
                f"{error}; SMILES fallback also failed: {fallback_error}"
            )
        return None, error
    try:
        molecule, _generated_hydrogen_count = _add_v6_export_hydrogens(
            molecule
        )
    except Exception as exc:
        return None, (
            "result-first candidate protonation materialization failed: "
            f"{exc}"
        )
    produced, error = mol_to_mol2(molecule, output_path)
    if error or produced is None:
        return produced, error
    try:
        if output_path:
            with open(produced, encoding="utf-8") as handle:
                content = handle.read()
        else:
            content = produced
    except Exception as exc:
        if output_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, f"result-first MOL2 roundtrip read failed: {exc}"
    heavy_ids = _mol2_one_based_heavy_atom_ids(content)
    note = _coordinate_tier_note(
        "X3",
        mapped_one_based=heavy_ids,
        generated_one_based=[],
        coordinate_source="result_first_candidate_graph",
    )
    produced = _prepend_mol2_tier_note(produced, note, output_path)
    content = note + content
    roundtrip_inchikey, roundtrip_error = _mol2_roundtrip_full_inchikey(
        content
    )
    expected = Chem.MolFromSmiles(candidate_smiles)
    expected_inchikey = Chem.MolToInchiKey(expected) if expected else None
    expected_connectivity = (
        expected_inchikey.split("-", 1)[0] if expected_inchikey else None
    )
    roundtrip_connectivity = (
        roundtrip_inchikey.split("-", 1)[0] if roundtrip_inchikey else None
    )
    if (
        roundtrip_error
        or not expected_connectivity
        or roundtrip_connectivity != expected_connectivity
    ):
        if output_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, (
            roundtrip_error
            or "result-first MOL2 connectivity InChIKey mismatch: "
            f"{roundtrip_connectivity} != {expected_connectivity}"
        )
    return produced, None


def extract_pdb_coords(pdb_path, chain_id='L'):
    """Extract atom serial → (x, y, z) mapping from a PDB file."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            if line[21] != chain_id:
                continue
            try:
                serial = int(line[6:11].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords[serial] = (x, y, z)
            except (ValueError, IndexError):
                continue
    return coords


def _extract_pdb_atom_metadata(pdb_path, chain_id='L'):
    records = {}
    with open(pdb_path, encoding='ascii', errors='replace') as handle:
        for line in handle:
            if not line.startswith(('ATOM', 'HETATM')) or len(line) < 54:
                continue
            if line[21] != chain_id:
                continue
            try:
                serial = int(line[6:11])
                residue_number = int(line[22:26])
            except ValueError:
                continue
            records[serial] = {
                'atom_name': line[12:16].strip(),
                'residue_name': line[17:20].strip(),
                'chain_id': line[21].strip(),
                'residue_number': residue_number,
                'insertion_code': line[26:27].strip(),
            }
    return records


def _add_v6_export_hydrogens(mol):
    """Add deterministic hydrogen coordinates and inherit residue metadata."""
    original_atom_count = mol.GetNumAtoms()
    result = Chem.AddHs(mol, addCoords=True)
    generated_count = result.GetNumAtoms() - original_atom_count
    for atom_index in range(original_atom_count, result.GetNumAtoms()):
        atom = result.GetAtomWithIdx(atom_index)
        neighbors = list(atom.GetNeighbors())
        if atom.GetSymbol() != 'H' or len(neighbors) != 1:
            raise ValueError(
                "generated MOL2 atom is not a singly attached hydrogen"
            )
        parent = neighbors[0]
        info = parent.GetPDBResidueInfo()
        if info is not None:
            residue_name = info.GetResidueName().strip() or 'RES'
            chain_id = info.GetChainId().strip()
            residue_number = info.GetResidueNumber()
            insertion_code = info.GetInsertionCode().strip()
        else:
            residue_name = (
                parent.GetProp('_TriposResidueName')
                if parent.HasProp('_TriposResidueName') else 'RES'
            )
            chain_id = (
                parent.GetProp('_TriposChainId')
                if parent.HasProp('_TriposChainId') else ''
            )
            residue_number = (
                parent.GetIntProp('_TriposResidueNumber')
                if parent.HasProp('_TriposResidueNumber') else 1
            )
            insertion_code = (
                parent.GetProp('_TriposInsertionCode')
                if parent.HasProp('_TriposInsertionCode') else ''
            )
        atom.SetProp('_TriposAtomName', f"H{atom_index + 1}")
        atom.SetProp('_TriposResidueName', residue_name)
        atom.SetProp('_TriposChainId', chain_id)
        atom.SetIntProp('_TriposResidueNumber', residue_number)
        atom.SetProp('_TriposInsertionCode', insertion_code)
    result.SetIntProp('CYCPEP_GENERATED_HYDROGEN_COUNT', generated_count)
    return result, generated_count


def _source_component_count(
    pdb_path, chain_id, residue_keys, serial_residue_keys
):
    """Count author-declared covalent components among source residues.

    Only declared connectivity is trusted: atoms within one residue form a
    single component, CONECT record pairs join residues, and LINK records
    join residue pairs.  Implicit peptide bonds between consecutive
    residues are deliberately NOT inferred, so an under-declared file
    yields a component count at least as large as the true one; the
    downstream fragment-loss gate stays biased against false failures.
    Returns ``None`` when the residue list is empty or the file cannot be
    read (gate is then skipped, not guessed).
    """
    if not residue_keys:
        return None
    key_index = {key: index for index, key in enumerate(residue_keys)}
    residue_lookup = {}
    for key in residue_keys:
        icode = key[3].strip() if len(key) == 4 else ""
        residue_lookup[(str(key[0]), int(key[1]), icode)] = key_index[key]
    parent = list(range(len(residue_keys)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    try:
        with open(pdb_path, encoding="ascii", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    for line in lines:
        record = line[:6].strip()
        if record == "CONECT":
            try:
                center = int(line[6:11])
            except ValueError:
                continue
            center_key = serial_residue_keys.get(center)
            if center_key is None:
                continue
            for start in (11, 16, 21, 26):
                try:
                    partner = int(line[start:start + 5])
                except ValueError:
                    continue
                partner_key = serial_residue_keys.get(partner)
                if partner_key is None or partner_key == center_key:
                    continue
                union(key_index[center_key], key_index[partner_key])
        elif record == "LINK":
            if len(line) < 26:
                continue
            try:
                first = (
                    line[17:20].strip(),
                    line[21],
                    int(line[22:26]),
                    line[26:27].strip(),
                )
            except ValueError:
                continue
            second = None
            if len(line) >= 56:
                try:
                    second = (
                        line[47:50].strip(),
                        line[51],
                        int(line[52:56]),
                        line[56:57].strip(),
                    )
                except ValueError:
                    second = None
            if second is None:
                continue
            if first[1] != chain_id or second[1] != chain_id:
                continue
            left = residue_lookup.get((first[0], first[2], first[3]))
            right = residue_lookup.get((second[0], second[2], second[3]))
            if left is None or right is None or left == right:
                continue
            union(left, right)
    return len({find(index) for index in range(len(residue_keys))})


def pdb_to_mol2(
    pdb_path,
    output_path=None,
    chain_id='L',
    path='a',
    *,
    _prepared_coordinate_input=False,
    _coordinate_input_evidence=None,
    _embedded_chem_comp_templates=None,
    monomer_context=None,
    fallback_policy="strict_v6",
    _strict_result=None,
    _result_first_result=None,
    _minimum_macrocycle_ring_size=8,
    _require_empty_persistent_overlay=True,
):
    """Convert a PDB file to MOL2 preserving original 3D coordinates.

    Modes ``a`` and ``e`` expose their atom mapping directly. Mode ``v6``
    first requires fail-closed evidence qualification and then independently
    verifies that the mapped export graph has the same full InChIKey. Explicit
    mode ``result_first`` enables the audited bond-order candidate portfolio
    and materializes only a source-bound candidate graph. Coordinate
    regeneration is never silently substituted. ``_strict_result`` and
    ``_result_first_result`` let an internal caller materialize the exact
    reconstruction it already selected instead of rerunning the portfolio.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
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
                pdb_path, kind="coordinate"
            ),
        ):
            return pdb_to_mol2(
                pdb_path,
                output_path=output_path,
                chain_id=chain_id,
                path=path,
                _prepared_coordinate_input=_prepared_coordinate_input,
                _coordinate_input_evidence=_coordinate_input_evidence,
                _embedded_chem_comp_templates=(
                    _embedded_chem_comp_templates
                ),
                fallback_policy=fallback_policy,
                _strict_result=_strict_result,
                _result_first_result=_result_first_result,
                _minimum_macrocycle_ring_size=(
                    _minimum_macrocycle_ring_size
                ),
                _require_empty_persistent_overlay=(
                    _require_empty_persistent_overlay
                ),
            )
    try:
        from ..max_coverage import coerce_policy

        fallback_policy = coerce_policy(fallback_policy)
    except Exception as exc:
        return None, f"invalid_input: invalid fallback policy: {exc}"
    alias_error = _reject_output_alias(pdb_path, output_path, 'MOL2')
    if alias_error:
        return alias_error
    clear_error = _clear_prior_output(output_path, 'MOL2')
    if clear_error:
        return clear_error
    from ..core.structure_io import CoordinateInputError, prepare_coordinate_input
    if not _prepared_coordinate_input:
        try:
            with prepare_coordinate_input(pdb_path, chain_id) as prepared:
                return pdb_to_mol2(
                    str(prepared.pdb_path),
                    output_path=output_path,
                    chain_id=prepared.chain_id,
                    path=path,
                    _prepared_coordinate_input=True,
                    _coordinate_input_evidence=prepared.audit,
                    _embedded_chem_comp_templates=prepared.audit.get(
                        "embedded_chem_comp_templates"
                    ),
                    fallback_policy=fallback_policy,
                    _strict_result=_strict_result,
                    _result_first_result=_result_first_result,
                    _minimum_macrocycle_ring_size=(
                        _minimum_macrocycle_ring_size
                    ),
                    _require_empty_persistent_overlay=(
                        _require_empty_persistent_overlay
                    ),
                )
        except CoordinateInputError as exc:
            support = "not supported: " if exc.not_supported else ""
            return None, (
                "coordinate input preparation failed: "
                f"{support}{exc.code}: {exc}"
            )
        except Exception as exc:
            return None, f"coordinate input preparation failed: {exc}"

    from ..paths.path_a import _build_combo
    from ..remediation_v5 import validate_pdb_reconstruction_input_v5

    try:
        if path == 'result_first':
            inferred = _result_first_result
            if inferred is None:
                from ..result_first import reconstruct_structure

                inferred = reconstruct_structure(
                    pdb_path,
                    chain_id,
                    minimum_macrocycle_ring_size=int(
                        _minimum_macrocycle_ring_size
                    ),
                    require_empty_persistent_overlay=bool(
                        _require_empty_persistent_overlay
                    ),
                    infer_bond_orders=True,
                )
            # Fail closed for BOTH caller-provided and internally
            # reconstructed results: no binding, no export.  This blocks
            # cross-input, cross-chain, cross-option, and cross-registry
            # reuse of a stale reconstruction.
            binding_error = _result_first_binding_error(
                inferred,
                coordinate_input_evidence=(
                    _coordinate_input_evidence
                ),
                chain_id=chain_id,
                minimum_macrocycle_ring_size=(
                    _minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    _require_empty_persistent_overlay
                ),
            )
            if binding_error:
                return None, binding_error
            if inferred.status != 'success':
                return None, (
                    "RESULT_FIRST_FAILED: "
                    + str(
                        inferred.provenance.get("failure_reason")
                        or inferred.provenance.get("error")
                        or inferred.status
                    )
                )
            if inferred.quality == 'exact':
                # The exact route already carries the qualified strict V6
                # result: thread it instead of paying for a second full
                # reconstruction inside the v6 branch.
                return pdb_to_mol2(
                    pdb_path,
                    output_path=output_path,
                    chain_id=chain_id,
                    path='v6',
                    fallback_policy=fallback_policy,
                    _prepared_coordinate_input=True,
                    _coordinate_input_evidence=_coordinate_input_evidence,
                    _embedded_chem_comp_templates=(
                        _embedded_chem_comp_templates
                    ),
                    _strict_result=inferred.strict_result,
                    _minimum_macrocycle_ring_size=(
                        _minimum_macrocycle_ring_size
                    ),
                    _require_empty_persistent_overlay=(
                        _require_empty_persistent_overlay
                    ),
                )
            return _result_first_candidate_to_mol2(
                inferred, output_path=output_path
            )
        validation = validate_pdb_reconstruction_input_v5(pdb_path, chain_id)
        if not validation.accepted:
            code = (
                validation.warning_codes[0]
                if validation.warning_codes
                else 'V5_INPUT_AUDIT_FAILED'
            )
            return None, f"{code}: {validation.reason or 'input PDB audit failed'}"
        expected_inchikey = None
        registry_context = None
        geometric = path == 'e'
        if path == 'v6':
            strict = _strict_result
            if strict is None:
                from ..remediation_v6 import (
                    reconstruct_pdb_fail_closed_v6,
                )

                strict = reconstruct_pdb_fail_closed_v6(
                    pdb_path,
                    chain_id,
                    minimum_macrocycle_ring_size=int(
                        _minimum_macrocycle_ring_size
                    ),
                    require_empty_persistent_overlay=bool(
                        _require_empty_persistent_overlay
                    ),
                    embedded_chem_comp_templates=(
                        _embedded_chem_comp_templates
                    ),
                    coordinate_input_evidence=_coordinate_input_evidence,
                )
            if strict.status != 'success':
                code = (
                    strict.warning_codes[0]
                    if strict.warning_codes
                    else strict.rejection_reason
                    or f"V6_{strict.status.upper()}"
                )
                return None, (
                    f"V6_{strict.status.upper()}: "
                    f"{code}: {strict.rejection_reason or ','.join(strict.warning_codes)}"
                )
            expected_inchikey = strict.output_inchikey
            geometric = strict.output_evidence.get('selected_route') == 'e'
            if strict.input_evidence.get('local_monomer_bootstrap'):
                from ..core.local_monomer_inference import bootstrap_unknown_monomers
                from ..paths._map_utils import isolated_monomer_registry
                bootstrap = bootstrap_unknown_monomers(
                    pdb_path,
                    chain_id,
                    embedded_chem_comp_templates=(
                        _embedded_chem_comp_templates
                    ),
                    source_identity_audit=(
                        (_coordinate_input_evidence or {}).get(
                            "source_sequence_identity_audit"
                        )
                        if isinstance(_coordinate_input_evidence, dict)
                        else None
                    ),
                )
                if not bootstrap.ready:
                    reason_codes = sorted({
                        str(code)
                        for result in bootstrap.inference_results
                        for code in result.reason_codes
                    })
                    if "SOURCE_IDENTITY_MAPPING_UNRESOLVED" in reason_codes:
                        return None, "SOURCE_IDENTITY_MAPPING_UNRESOLVED"
                    if "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE" in reason_codes:
                        return None, "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"
                    if "SOURCE_IDENTITY_CONSTRAINT_CONFLICT" in reason_codes:
                        return None, "SOURCE_IDENTITY_CONSTRAINT_CONFLICT"
                    return None, "V6 local monomer overlay could not be reproduced"
                registry_context = isolated_monomer_registry(
                    derived_rows=bootstrap.derived_rows,
                    pdb_aliases=bootstrap.pdb_aliases,
                    include_persistent_user=False,
                )
        elif path not in {'a', 'e'}:
            return None, (
                "coordinate-preserving MOL2 export supports only a, e, v6, "
                "or result_first; "
                "use smiles_to_mol2 explicitly for regenerated coordinates"
            )
        if registry_context is None:
            combo, pdb2g = _build_combo(
                pdb_path, chain_id, geometric_cyclization=geometric
            )
        else:
            with registry_context:
                combo, pdb2g = _build_combo(
                    pdb_path, chain_id, geometric_cyclization=geometric
                )
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"PDB-to-MOL2 assembly failed: {e}"

    # A coordinate-preserving export must represent every selected source
    # heavy atom. Alternate locations, terminal atoms, or other records that
    # Path A cannot map are rejected instead of being silently dropped.
    from ..core.pdb_parser import get_pdb_atoms, get_res_seq
    source_serials = set()
    serial_residue_keys = {}
    residue_keys_ordered = []
    for residue in get_res_seq(pdb_path, chain_id):
        residue_keys_ordered.append(residue['key'])
        for atom in get_pdb_atoms(pdb_path, residue['key'], chain_id):
            source_serials.add(atom['num'])
            serial_residue_keys[atom['num']] = residue['key']
    unmapped_source = source_serials - set(pdb2g)
    if unmapped_source:
        return None, f"incomplete PDB atom mapping: {len(unmapped_source)} source heavy atoms unmapped"
    # X3 additionally requires full source-component preservation: if the
    # export graph fragments beyond what the source declares (CONECT/LINK),
    # bonds were lost during assembly and the export fails closed instead
    # of shipping a silently disconnected graph.
    source_components = _source_component_count(
        pdb_path, chain_id, residue_keys_ordered, serial_residue_keys
    )
    output_fragment_count = len(Chem.GetMolFrags(combo.GetMol()))
    if (
        source_components is not None
        and source_components > 0
        and output_fragment_count > source_components
    ):
        return None, (
            "fragment loss during coordinate export: output graph has "
            f"{output_fragment_count} components but the source declares "
            f"{source_components}"
        )

    # Inject PDB coordinates into the RWMol (before sanitization)
    coords = extract_pdb_coords(pdb_path, chain_id)
    conf = Chem.Conformer(combo.GetNumAtoms())
    assigned_indices = set()
    for pdb_serial, rdkit_idx in pdb2g.items():
        if pdb_serial in coords and rdkit_idx < combo.GetNumAtoms():
            x, y, z = coords[pdb_serial]
            conf.SetAtomPosition(rdkit_idx, (x, y, z))
            assigned_indices.add(rdkit_idx)

    if not assigned_indices and fallback_policy != "max_coverage":
        return None, "no atom mapping found between PDB and SMILES"
    coordinate_tier = "X3"
    unmapped_output_atoms = []
    if len(assigned_indices) != combo.GetNumAtoms():
        missing_count = combo.GetNumAtoms() - len(assigned_indices)
        if fallback_policy != "max_coverage":
            return None, f"incomplete PDB coordinate mapping: {missing_count} output atoms unmapped"
        # Max-coverage fallback (V7 design): emit the MOL2 anyway. Mapped
        # atoms keep their source coordinates; every other heavy atom gets a
        # generated coordinate, and the receipt records the X2 tier with the
        # generated-atom list. Graph identity gates (C-axis) are unchanged -
        # only coordinate evidence is tiered down. Placement strategy: few-
        # atom gaps (the dominant case in this cohort) are filled by local
        # bond-geometry placement against already-mapped neighbors, which is
        # both fast and consistent with the source coordinate frame; only
        # when geometry cannot place an atom do we fall back to a full
        # ETKDG embedding of the molecule.
        def _vec_sub(a, b):
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

        def _vec_add(a, b):
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        def _vec_scale(a, s):
            return (a[0] * s, a[1] * s, a[2] * s)

        def _norm(a):
            length = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
            return (a[0] / length, a[1] / length, a[2] / length) if length else (1.0, 0.0, 0.0)

        placed: dict[int, tuple[float, float, float]] = {}
        fallback_origin = None
        fallback_etkdg_attempts = None

        def _position(index):
            if index in placed:
                return placed[index]
            point = conf.GetAtomPosition(index)
            return (point.x, point.y, point.z)

        pending = [i for i in range(combo.GetNumAtoms())
                   if i not in assigned_indices]
        progress = True
        while pending and progress:
            progress = False
            for idx in list(pending):
                atom = combo.GetAtomWithIdx(idx)
                mapped_neighbors = [
                    n.GetIdx() for n in atom.GetNeighbors()
                    if n.GetIdx() in assigned_indices or n.GetIdx() in placed
                ]
                if not mapped_neighbors:
                    continue
                anchor = mapped_neighbors[0]
                anchor_pos = _position(anchor)
                others = [n for n in mapped_neighbors[1:]]
                if others:
                    centroid = [0.0, 0.0, 0.0]
                    for other in others:
                        op = _position(other)
                        centroid[0] += op[0]
                        centroid[1] += op[1]
                        centroid[2] += op[2]
                    centroid = _vec_scale(centroid, 1.0 / len(others))
                    direction = _norm(_vec_sub(anchor_pos, tuple(centroid)))
                else:
                    # single neighbor: extend away from the neighbor's own
                    # mapped neighborhood
                    nbr_atom = combo.GetAtomWithIdx(anchor)
                    back = [n.GetIdx() for n in nbr_atom.GetNeighbors()
                            if n.GetIdx() != idx and
                            (n.GetIdx() in assigned_indices or
                             n.GetIdx() in placed)]
                    if back:
                        centroid = [0.0, 0.0, 0.0]
                        for other in back:
                            op = conf.GetAtomPosition(other)
                            centroid[0] += op.x
                            centroid[1] += op.y
                            centroid[2] += op.z
                        centroid = _vec_scale(centroid, 1.0 / len(back))
                        direction = _norm(_vec_sub(
                            anchor_pos, tuple(centroid)))
                    else:
                        direction = (1.0, 0.0, 0.0)
                bond_len = 1.5
                new_pos = _vec_add(anchor_pos, _vec_scale(direction, bond_len))
                placed[idx] = new_pos
                conf.SetAtomPosition(idx, new_pos)
                pending.remove(idx)
                progress = True

        if not pending:
            for idx, pos in placed.items():
                conf.SetAtomPosition(idx, pos)
                unmapped_output_atoms.append(idx)
            fallback_origin = "bond_geometry_completion"
        else:
            from rdkit.Chem import AllChem as _AllChem
            probe = combo.GetMol()
            fallback_etkdg_attempts = 1
            try:
                Chem.SanitizeMol(probe)
                probe = Chem.AddHs(probe)
                if _AllChem.EmbedMolecule(probe, randomSeed=42) != 0:
                    fallback_etkdg_attempts = 2
                    if _AllChem.EmbedMolecule(
                            probe, useRandomCoords=True,
                            randomSeed=42) != 0:
                        raise RuntimeError("ETKDG returned failure")
            except Exception as exc:
                return None, (
                    "incomplete PDB coordinate mapping: "
                    f"{missing_count} output atoms unmapped "
                    f"(fallback embedding failed: {exc})"
                )
            fallback_origin = "etkdgv3_embedding_aligned"
            if assigned_indices:
                try:
                    from rdkit.Chem import rdMolAlign as _rdMolAlign
                    reference = combo.GetMol()
                    reference.AddConformer(conf, assignId=True)
                    _rdMolAlign.AlignMol(
                        probe,
                        reference,
                        atomMap=[(index, index) for index in sorted(assigned_indices)],
                    )
                except Exception as exc:
                    return None, (
                        "incomplete PDB coordinate mapping: "
                        f"{missing_count} output atoms unmapped "
                        f"(fallback embedding alignment failed: {exc})"
                    )
            embedded = probe.GetConformer()
            for rdkit_idx in range(combo.GetNumAtoms()):
                if rdkit_idx in assigned_indices:
                    continue
                x, y, z = embedded.GetAtomPosition(rdkit_idx)
                conf.SetAtomPosition(rdkit_idx, (x, y, z))
                unmapped_output_atoms.append(rdkit_idx)
        coordinate_tier = "X2" if assigned_indices else "X1"
    conf.Set3D(True)

    # Convert to Mol and add conformer
    mol = combo.GetMol()
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, f"coordinate-preserving MOL2 sanitization failed: {exc}"

    if expected_inchikey is not None:
        observed_inchikey = Chem.MolToInchiKey(mol)
        if observed_inchikey != expected_inchikey:
            return None, (
                "coordinate export graph differs from V6 reconstruction: "
                f"{observed_inchikey} != {expected_inchikey}"
            )

    metadata = _extract_pdb_atom_metadata(pdb_path, chain_id)
    for serial, atom_index in pdb2g.items():
        record = metadata.get(serial)
        if record is None:
            continue
        atom = mol.GetAtomWithIdx(atom_index)
        atom.SetProp('_TriposAtomName', record['atom_name'])
        atom.SetProp('_TriposResidueName', record['residue_name'])
        atom.SetProp('_TriposChainId', record['chain_id'])
        atom.SetIntProp('_TriposResidueNumber', record['residue_number'])
        atom.SetProp('_TriposInsertionCode', record['insertion_code'])

    if path == 'v6':
        try:
            mol, _ = _add_v6_export_hydrogens(mol)
        except Exception as exc:
            return None, f"MOL2 protonation materialization failed: {exc}"
        explicit_h_inchikey = Chem.MolToInchiKey(mol)
        if explicit_h_inchikey != expected_inchikey:
            return None, (
                "explicit-hydrogen MOL2 graph differs from V6 reconstruction: "
                f"{explicit_h_inchikey} != {expected_inchikey}"
            )

    # Readback gate: v6 carries an explicit expected full InChIKey from the
    # strict reconstruction and writes explicit hydrogens, so its artifact
    # gates on full identity.  Paths a/e write hydrogen-less artifacts: the
    # export graph carries template hydrogen counts while the readback lets
    # InChI guess protonation (observed -N vs -P suffix on the same
    # artifact), so a full-InChIKey comparison is unsound there.  a/e
    # instead gate serialization fidelity on the atom/bond/charge ledger
    # plus a mandatory sanitized readback.
    produced, write_error = mol_to_mol2(mol, output_path)
    if write_error or produced is None:
        return produced, write_error
    try:
        if output_path:
            with open(produced, encoding="utf-8") as handle:
                content = handle.read()
        else:
            content = produced
    except Exception as exc:
        return None, f"MOL2 roundtrip read failed: {exc}"
    heavy_ids = _mol2_one_based_heavy_atom_ids(content)
    if coordinate_tier == "X3":
        tier_note = _coordinate_tier_note(
            "X3",
            mapped_one_based=heavy_ids,
            generated_one_based=[],
            coordinate_source=f"pdb_path_{path}",
            source_components=source_components,
            output_components=output_fragment_count,
        )
    else:
        # Max-coverage X2/X1: generated ids are the artifact one-based ids
        # of the atoms this export generated, cross-checked against the
        # in-memory unmapped set so the header can never drift from the
        # artifact.
        unmapped_indices = set(unmapped_output_atoms)
        generated_ids = [
            atom_id for atom_id in heavy_ids
            if atom_id - 1 in unmapped_indices
        ]
        if len(generated_ids) != len(unmapped_indices):
            return None, (
                "coordinate tier ledger could not be reconciled with the "
                f"written MOL2: {len(generated_ids)} generated ids parsed "
                f"for {len(unmapped_indices)} generated atoms"
            )
        mapped_ids = [
            atom_id for atom_id in heavy_ids
            if atom_id - 1 not in unmapped_indices
        ]
        tier_note = _coordinate_tier_note(
            coordinate_tier,
            mapped_one_based=mapped_ids,
            generated_one_based=generated_ids,
            coordinate_source=f"pdb_path_{path}",
            fallback_origin=fallback_origin,
            etkdg_attempts=fallback_etkdg_attempts,
            source_components=source_components,
            output_components=output_fragment_count,
        )
    produced = _prepend_mol2_tier_note(produced, tier_note, output_path)
    content = tier_note + content
    if expected_inchikey is not None:
        roundtrip_inchikey, roundtrip_error = (
            _mol2_roundtrip_full_inchikey(content)
        )
        if roundtrip_error or roundtrip_inchikey != expected_inchikey:
            if output_path:
                try:
                    os.remove(produced)
                except OSError:
                    pass
            return None, (
                roundtrip_error
                or "MOL2 full InChIKey roundtrip mismatch: "
                f"{roundtrip_inchikey} != {expected_inchikey}"
            )
        return produced, None

    # a/e graph-ledger readback (mandatory serialization validation).
    readback = Chem.MolFromMol2Block(
        content, sanitize=False, removeHs=False
    )
    if readback is None:
        if output_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, "MOL2 readback parse failed"
    for atom_index, charge in _mol2_unity_formal_charges(content).items():
        if atom_index < 0 or atom_index >= readback.GetNumAtoms():
            if output_path:
                try:
                    os.remove(produced)
                except OSError:
                    pass
            return None, (
                "MOL2 readback formal-charge index out of range: "
                f"{atom_index + 1}"
            )
        readback.GetAtomWithIdx(atom_index).SetFormalCharge(charge)
    try:
        Chem.SanitizeMol(readback)
    except Exception as exc:
        if output_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, f"MOL2 readback sanitization failed: {exc}"
    if not Chem.MolToInchiKey(readback):
        if output_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, "MOL2 readback produced no InChIKey"

    def _graph_ledger(molecule):
        molecule = Chem.Mol(molecule)
        try:
            Chem.Kekulize(molecule, clearAromaticFlags=True)
        except Exception:
            pass
        atoms = sorted(
            (atom.GetAtomicNum(), atom.GetFormalCharge())
            for atom in molecule.GetAtoms()
        )
        bonds = sorted(
            str(bond.GetBondType()) for bond in molecule.GetBonds()
        )
        return atoms, bonds

    expected_atoms, expected_bonds = _graph_ledger(mol)
    observed_atoms, observed_bonds = _graph_ledger(readback)
    if (
        observed_atoms != expected_atoms
        or observed_bonds != expected_bonds
    ):
        if output_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        return None, (
            "MOL2 readback graph ledger mismatch: "
            f"{len(observed_atoms)} atoms/{len(observed_bonds)} bonds "
            f"observed vs {len(expected_atoms)}/"
            f"{len(expected_bonds)} expected"
        )
    return produced, None

