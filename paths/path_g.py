"""Path G: special-residue-library-driven symbol-level assembly.

For peptides whose special residues (stapled hydrocarbons, lanthionine,
depsipeptide) are registered in ``special_residue_library.csv``, Path G builds
a MAP string in which each residue resolves to a library monomer fragment
(unified library by symbol, or special library by PDB code), then assembles via
the same multi-cycle MAP->SMILES engine as Path B. Unlike Path F it produces
correct bond orders and stereochemistry (taken from the registered CXSMILES);
unlike Paths A/B/C/E it does not drop non-standard residues.

Registration of special residues into the map_utils monomer dicts is lazy and
idempotent.  Each Path G invocation verifies the live registry because formal
entity isolation can rebuild that registry between calls.
"""
from ..core.pdb_parser import get_res_seq
from ..core.cyclization import detect_cyclization
from ..core import special_residues as _sr
from . import _map_utils as _mu
from ._map_utils import (
    get_smi_from_cxsmiles, helm_to_map, get_smi_from_map,
    monomers2smi_dict,
)
from .path_b import _AA_3TO1, _format_helm_element
from ..core.monomer_resolution import needs_monomer_resolution_scope

# D-amino-acid PDB component codes -> unified-library symbol (the chemistry is
# already curated under these symbols; only the 3-letter-code mapping is needed
# so Path G resolves D-residues in peptides like daptomycin instead of leaving
# them as unknown monomers).
_D_AA_3TO_SYMBOL = {
    'DAL': 'dA', 'DSN': 'dS', 'DSG': 'dN', 'DAS': 'dD', 'DGL': 'dE',
    'DTH': 'dT', 'DVA': 'dV', 'DLE': 'dL', 'DIL': 'dI', 'DPR': 'dP',
    'DTR': 'dW', 'DTY': 'dY', 'DPN': 'dF', 'DAR': 'dR', 'DLY': 'dK',
    'DHI': 'dH', 'DCY': 'dC', 'MED': 'dM', 'DGN': 'dQ', 'DBB': 'dAbu',
}

# L non-standard PDB component codes whose chemistry is already curated in the
# unified library under a differently-cased/abbreviated symbol.
_L_NONSTD_3TO_SYMBOL = {
    'ORN': 'Orn', 'ABU': 'Abu', 'AIB': 'Aib', 'NLE': 'Nle', 'NVA': 'Nva',
    'SAR': 'Sar', 'HYP': 'Hyp', 'PCA': 'Glp',
}

# Lanthionine / methyllanthionine bridge donors. In a (methyl)lanthionine
# thioether, an Ala/Abu-type residue's β-carbon (CB) is bonded to a Cys sulfur.
# The plain Ala/Abu library monomers have no R3 side-chain attachment, so the
# bridge cannot form. These runtime variants add an R3 on CB. They are selected
# CONTEXTUALLY (only for residues detected as thioether CB-donors), never by
# PDB code alone — DAL/DBB are ordinary D-Ala/D-Abu outside a lanthionine.
#   base symbol -> (variant symbol, CXSMILES with _R3 on CB)
_LANTHIONINE_VARIANT = {
    # Ala-derived lanthionine: CB methyl carbon carries the thioether.
    'A':    ('LanA', '[*]C[C@@H](N[*])C([*])=O |$_R3;;;;_R1;;_R2;$|'),
    'dA':   ('dLanA', '[*]C[C@H](N[*])C([*])=O |$_R3;;;;_R1;;_R2;$|'),
    # Abu-derived β-methyllanthionine: CB (bearing the extra methyl) carries it.
    'Abu':  ('bMeLan', 'CC([*])[C@H](N[*])C([*])=O |$;;_R3;;;_R1;;_R2;$|'),
    'dAbu': ('dBMeLan', 'CC([*])[C@@H](N[*])C([*])=O |$;;_R3;;;_R1;;_R2;$|'),
}


def _register_lanthionine_variants():
    """Register the lanthionine bridge-donor monomers into the runtime dicts."""
    for _base, (sym, cx) in _LANTHIONINE_VARIANT.items():
        if sym in monomers2smi_dict:
            continue
        try:
            get_smi_from_cxsmiles(cx)
        except Exception:
            continue
        try:
            _mu.register_user_monomer_record({
                "symbol": sym,
                "CXSMILES": cx,
                "R1": "H",
                "R2": "OH",
                "R3": "H",
                "smiles_original": "",
            })
        except (KeyError, ValueError):
            continue


def _register_special_residues():
    """Ensure special-residue CXSMILES are present in the live registry."""
    for code in _sr.all_codes():
        sym = _sr.get_symbol(code)
        cx = _sr.get_cxsmiles(code)
        if not sym or not cx:
            continue
        # Register under the library symbol AND the PDB code, without ever
        # clobbering an existing unified-library entry.
        try:
            get_smi_from_cxsmiles(cx)
        except Exception:
            continue
        rg = _sr.get_rgroups(code) or ('H', 'OH', '-')
        rg_map = {}
        for name, val in zip(('R1', 'R2', 'R3'), rg):
            if val and val != '-':
                rg_map[name] = val
        if sym not in monomers2smi_dict:
            try:
                _mu.register_user_monomer_record({
                    "symbol": sym,
                    "CXSMILES": cx,
                    "R1": str(rg_map.get("R1", "-")),
                    "R2": str(rg_map.get("R2", "-")),
                    "R3": str(rg_map.get("R3", "-")),
                    "smiles_original": "",
                })
            except (KeyError, ValueError):
                continue
        if code and code.upper() != str(sym).upper():
            try:
                _mu.register_pdb_alias(code, sym)
            except (KeyError, ValueError):
                pass


def _residue_symbol(name):
    """Map a PDB residue name to a monomer symbol.

    Standard AA -> one-letter; registered special residue -> its library
    symbol; otherwise the raw name (which will fail downstream with a clear
    'unknown monomer', not a silent drop).
    """
    if name in ('ACE', 'NME', 'NH2'):
        return {'ACE': 'ac', 'NME': 'nme', 'NH2': 'nh2'}[name]
    if name in _AA_3TO1:
        return _AA_3TO1[name]
    if name in _D_AA_3TO_SYMBOL:
        return _D_AA_3TO_SYMBOL[name]
    if name in _L_NONSTD_3TO_SYMBOL:
        return _L_NONSTD_3TO_SYMBOL[name]
    active_alias = _mu.resolve_pdb_alias(name)
    if active_alias:
        return active_alias
    if _sr.is_special_residue(name):
        return _sr.get_symbol(name)
    return name


def build_helm_with_special(pdb_path, chain_id='L',
                            *, allow_geometric_inference=True,
                            monomer_context=None):
    """Build a HELM string including special (HETATM) residues.

    Differs from path_b.build_helm_from_pdb by (a) reading HETATM residues
    (include_het=True) so non-standard residues are not dropped, and (b)
    resolving special residues through the special-residue library.
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
            return build_helm_with_special(
                pdb_path,
                chain_id,
                allow_geometric_inference=allow_geometric_inference,
            )
    pdb_path = str(pdb_path)
    try:
        residues = get_res_seq(pdb_path, chain_id, include_het=True)
    except (FileNotFoundError, OSError):
        return None
    if not residues:
        return None

    symbols = [_residue_symbol(r['name']) for r in residues]

    cyc_info = detect_cyclization(
        pdb_path,
        chain_id,
        allow_geometric_inference=allow_geometric_inference,
    )

    # Lanthionine handling: a thioether bond whose donor side (CB, R3) sits on
    # an Ala/Abu-type residue needs the R3-bearing variant so the bridge can
    # form. Swap those positions' symbols to the lanthionine variant. Done
    # before building the sequence string. Positions are 1-based in bonds.
    _register_lanthionine_variants()
    for bond in cyc_info.bonds:
        if bond.bond_type != 'thioether':
            continue
        for pos, rg, atom in ((bond.pos1, bond.rgroup1, bond.atom1),
                              (bond.pos2, bond.rgroup2, bond.atom2)):
            if rg != 'R3' or atom == 'SG':
                continue  # the Cys-sulfur side already has R3; skip it
            i = pos - 1
            if 0 <= i < len(symbols):
                variant = _LANTHIONINE_VARIANT.get(symbols[i])
                if variant:
                    symbols[i] = variant[0]

    seq_str = '.'.join(_format_helm_element(s) for s in symbols)

    connections = []
    for bond in cyc_info.bonds:
        connections.append(
            f'PEPTIDE1,PEPTIDE1,{bond.pos1}:{bond.rgroup1}-{bond.pos2}:{bond.rgroup2}'
        )
    conn_str = '|'.join(connections) if connections else ''
    return f'PEPTIDE1{{{seq_str}}}${conn_str}$$$'


def generate_g_with_artifacts(
    pdb_path,
    chain_id='L',
    *,
    allow_geometric_inference=True,
    monomer_context=None,
):
    """Generate Path G and retain the forward HELM/MAP intermediates."""
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
            return generate_g_with_artifacts(
                pdb_path,
                chain_id,
                allow_geometric_inference=allow_geometric_inference,
            )
    pdb_path = str(pdb_path)
    _register_special_residues()
    artifact = {
        "route": "g",
        "allow_geometric_inference": bool(allow_geometric_inference),
        "helm": None,
        "map_payload": None,
        "output_smiles": None,
    }
    try:
        helm = build_helm_with_special(
            pdb_path,
            chain_id,
            allow_geometric_inference=allow_geometric_inference,
        )
        artifact["helm"] = helm
        if not helm:
            return None, "empty sequence", artifact
        mapped = helm_to_map(helm)
        artifact["map_payload"] = mapped
        if not mapped or mapped.startswith('ERROR'):
            return None, f"helm_to_map: {mapped}", artifact
        smiles = get_smi_from_map(mapped)
        artifact["output_smiles"] = smiles
        if not smiles:
            return None, "get_smi_from_map returned None", artifact
        return smiles, None, artifact
    except Exception as ex:
        return None, str(ex), artifact


def generate_g(pdb_path, chain_id='L', *, allow_geometric_inference=True,
               monomer_context=None):
    """Path G: special-residue-library-driven SMILES generation."""
    smiles, error, _artifact = generate_g_with_artifacts(
        pdb_path,
        chain_id,
        allow_geometric_inference=allow_geometric_inference,
        monomer_context=monomer_context,
    )
    return smiles, error
