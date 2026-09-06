"""Path B: HELM→MAP→SMILES pipeline.

Builds HELM directly from PDB residues, then converts through MAP notation to SMILES.
Uses the MAP monomer library + unified monomer library for broad monomer coverage.
"""
from collections import defaultdict

from ..core.pdb_parser import get_res_seq
from ._map_utils import helm_to_map, get_smi_from_map
from . import _map_utils as _mu
from ..core.monomer_resolution import needs_monomer_resolution_scope

# 3-letter to 1-letter for standard amino acids
_AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def _pdb_name_to_helm_symbol(name: str) -> str:
    """Convert PDB residue name to HELM monomer symbol."""
    if name == 'ACE':
        return 'ac'
    if name == 'NME':
        return 'nme'
    if name == 'NH2':
        return 'nh2'
    standard = _AA_3TO1.get(name)
    if standard:
        return standard
    return _mu.resolve_pdb_alias(name) or name


def _format_helm_element(symbol: str) -> str:
    """Format a monomer symbol as a HELM element (multi-char symbols in brackets)."""
    if len(symbol) == 1:
        return symbol
    return f'[{symbol}]'


def build_helm_from_pdb(pdb_path: str, chain_id: str = 'L',
                        *, allow_geometric_inference: bool = True,
                        monomer_context=None) -> str:
    """Build HELM string from a PDB file's peptide chain residues.

    Uses SSBOND/LINK records for data-driven cyclization detection.
    Returns the full HELM string (e.g. PEPTIDE1{A.ORN.D_PHE.G}$$$$),
    or None if the file is missing, unreadable, or the chain has no residues.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ):
            return build_helm_from_pdb(
                pdb_path,
                chain_id,
                allow_geometric_inference=allow_geometric_inference,
            )
    from ..core.cyclization import detect_cyclization as _detect_cyc

    try:
        residues = get_res_seq(pdb_path, chain_id)
    except (FileNotFoundError, OSError):
        return None
    if not residues:
        return None

    symbols = [_pdb_name_to_helm_symbol(r['name']) for r in residues]
    seq_str = '.'.join(_format_helm_element(s) for s in symbols)

    # Use data-driven cyclization detection
    cyc_info = _detect_cyc(
        pdb_path,
        chain_id,
        allow_geometric_inference=allow_geometric_inference,
    )
    connections = []
    for bond in cyc_info.bonds:
        connections.append(
            f'PEPTIDE1,PEPTIDE1,{bond.pos1}:{bond.rgroup1}-{bond.pos2}:{bond.rgroup2}'
        )

    conn_str = '|'.join(connections) if connections else ''
    return f'PEPTIDE1{{{seq_str}}}${conn_str}$$$'


def generate_with_artifacts(
    pdb_path,
    chain_id='L',
    *,
    allow_geometric_inference=True,
    monomer_context=None,
):
    """Generate SMILES and retain the forward HELM/MAP intermediates."""
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ):
            return generate_with_artifacts(
                pdb_path,
                chain_id,
                allow_geometric_inference=allow_geometric_inference,
            )
    artifact = {
        "route": "b",
        "allow_geometric_inference": bool(allow_geometric_inference),
        "helm": None,
        "map_payload": None,
        "output_smiles": None,
    }
    try:
        helm = build_helm_from_pdb(
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


def generate(pdb_path, chain_id='L', *, allow_geometric_inference=True,
             monomer_context=None):
    """Generate SMILES from PDB via HELM→MAP→SMILES."""
    smiles, error, _artifact = generate_with_artifacts(
        pdb_path,
        chain_id,
        allow_geometric_inference=allow_geometric_inference,
        monomer_context=monomer_context,
    )
    return smiles, error

# ── Ported from mirror: multi-chain HELM assembly (Lite was missing these) ─

def _peptide_chain_ids(pdb_path):
    """Return chain IDs that contain at least one standard/peptide residue,
    in first-seen order (skips water/ion-only chains)."""
    ids = []
    skip = {'HOH', 'WAT', 'DOD', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE',
            'MN', 'CU', 'SO4', 'PO4', 'GOL', 'EDO', 'DMS', 'NAG', 'MAN'}
    residues = {}
    order = []
    try:
        f = open(pdb_path, encoding='utf-8', errors='replace')
    except OSError:
        return ids
    with f:
        for line in f:
            if line[:6] not in ('ATOM  ', 'HETATM'):
                continue
            ch = line[21]
            resn = line[17:20].strip()
            if resn in skip:
                continue
            if ch not in residues:
                residues[ch] = {}
                order.append(ch)
            key = (line[22:26], line[26:27], resn)
            row = residues[ch].setdefault(
                key, {"name": resn, "atoms": set()}
            )
            row["atoms"].add(line[12:16].strip().upper())
    from . import _map_utils

    registry = _map_utils.monomers2smi_dict
    for chain in order:
        rows = list(residues[chain].values())
        registered = [
            row["name"] in _AA_3TO1
            or row["name"] in registry
            or _map_utils.resolve_pdb_alias(row["name"]) in registry
            for row in rows
        ]
        backbone = [
            {"N", "CA", "C"}.issubset(row["atoms"]) for row in rows
        ]
        if (
            sum(backbone) >= 2
            or sum(registered) >= 2
            or any(
                has_backbone and is_registered
                for has_backbone, is_registered in zip(backbone, registered)
            )
        ):
            ids.append(chain)
    return ids


def build_helm_multichain(
    pdb_path, chain_ids=None, *, monomer_context=None
):
    """Build a multi-polymer HELM string from several PDB chains.

    Each chain becomes one PEPTIDEn{...} block; inter- and intra-chain
    disulfides (SSBOND records) become HELM connections referencing the chains'
    1-based residue positions. Independent chains are joined only by these
    explicit bonds (no inter-chain backbone bond).

    chain_ids: ordered list of chain IDs to include. If None, all peptide-
    bearing chains are used (water/ion-only chains skipped).

    Returns the HELM string, or None if no usable chain is found. For a single
    chain this still emits a one-block HELM (equivalent to build_helm_from_pdb
    for the disulfide case).
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ):
            return build_helm_multichain(pdb_path, chain_ids)
    from ..core.cyclization import (
        _BACKBONE_OTHER,
        _LINKABLE_ELEMS,
        _rgroup_for_atom,
        read_atoms,
        read_conect,
        read_link,
        read_ssbond,
    )

    if chain_ids is None:
        chain_ids = _peptide_chain_ids(pdb_path)
    if not chain_ids:
        return None

    # Per chain: HELM sequence + full PDB residue identity -> 1-based pos.
    blocks = []
    num_to_pos = {}        # chain_id -> {pdb_num: position}
    used_chains = []
    for ch in chain_ids:
        try:
            residues = get_res_seq(pdb_path, ch)
        except (FileNotFoundError, OSError):
            return None
        if not residues:
            continue
        symbols = [_pdb_name_to_helm_symbol(r['name']) for r in residues]
        seq_str = '.'.join(_format_helm_element(s) for s in symbols)
        blocks.append(seq_str)
        num_to_pos[ch] = {
            (r['num'], str(r.get('icode', ''))): i + 1
            for i, r in enumerate(residues)
        }
        used_chains.append(ch)
    if not blocks:
        return None

    # Chain id -> PEPTIDE index (1-based, matches block order).
    chain_polymer = {ch: i + 1 for i, ch in enumerate(used_chains)}

    connections = []
    connection_keys = set()
    occupied_ports = {}

    def add_connection(c1, p1, rg1, c2, p2, rg2, source):
        left = (chain_polymer[c1], p1, rg1)
        right = (chain_polymer[c2], p2, rg2)
        key = tuple(sorted((left, right)))
        if key in connection_keys:
            return
        for endpoint in (left, right):
            prior = occupied_ports.get(endpoint)
            if prior is not None and prior != key:
                raise ValueError(
                    "MULTICHAIN_CONNECTION_PORT_REUSED: "
                    f"{endpoint} is used by {prior} and {key} ({source})"
                )
        connection_keys.add(key)
        occupied_ports[left] = key
        occupied_ports[right] = key
        connections.append(
            f'PEPTIDE{left[0]},PEPTIDE{right[0]},'
            f'{left[1]}:{left[2]}-{right[1]}:{right[2]}'
        )

    atoms_by_chain = {chain: read_atoms(pdb_path, chain) for chain in used_chains}
    def residue_position(chain, atom):
        return num_to_pos[chain].get(
            (atom['resseq'], str(atom.get('icode', '')))
        )

    def add_atom_connection(c1, a1, c2, a2, source):
        if c1 == c2 and a1['resid'] == a2['resid']:
            return
        if a1['elem'] not in _LINKABLE_ELEMS or a2['elem'] not in _LINKABLE_ELEMS:
            return
        p1 = residue_position(c1, a1)
        p2 = residue_position(c2, a2)
        if p1 is None or p2 is None:
            raise ValueError(
                "MULTICHAIN_CONNECTION_RESIDUE_UNMAPPED: "
                f"{source} endpoint is outside the assembled residues"
            )
        if a1['name'] in _BACKBONE_OTHER or a2['name'] in _BACKBONE_OTHER:
            raise ValueError(
                "MULTICHAIN_CONNECTION_ATOM_NOT_REPRESENTABLE: "
                f"{source} uses {a1['name']}-{a2['name']}"
            )
        rg1 = _rgroup_for_atom(a1, p1, len(num_to_pos[c1]))
        rg2 = _rgroup_for_atom(a2, p2, len(num_to_pos[c2]))
        if c1 == c2:
            forward_backbone = (
                (a1['name'] == 'C' and a2['name'] == 'N' and p2 == p1 + 1)
                or (a2['name'] == 'C' and a1['name'] == 'N' and p1 == p2 + 1)
            )
            if forward_backbone:
                return
        add_connection(c1, p1, rg1, c2, p2, rg2, source)

    # Translate SSBOND records first, then add LINK/CONECT edges not already
    # represented by the same HELM ports.
    for ss in read_ssbond(pdb_path):
        c1, c2 = ss['chain1'], ss['chain2']
        included = (c1 in chain_polymer, c2 in chain_polymer)
        if any(included) and not all(included):
            raise ValueError(
                "MULTICHAIN_CONNECTION_CHAIN_EXCLUDED: selected chain "
                "participates in an SSBOND to an excluded chain"
            )
        if not all(included):
            continue
        p1 = num_to_pos[c1].get((ss['num1'], str(ss.get('icode1', ''))))
        p2 = num_to_pos[c2].get((ss['num2'], str(ss.get('icode2', ''))))
        if p1 is None or p2 is None:
            raise ValueError(
                "MULTICHAIN_CONNECTION_RESIDUE_UNMAPPED: SSBOND endpoint "
                "is outside the assembled residues"
            )
        add_connection(c1, p1, 'R3', c2, p2, 'R3', 'SSBOND')

    atom_lookup = defaultdict(list)
    for chain, atoms in atoms_by_chain.items():
        for atom in atoms.values():
            atom_lookup[(
                chain,
                atom['resseq'],
                str(atom.get('icode', '')),
                atom['name'],
            )].append(atom)
    for link in read_link(pdb_path):
        c1, c2 = link['chain1'], link['chain2']
        included = (c1 in chain_polymer, c2 in chain_polymer)
        if any(included) and not all(included):
            raise ValueError(
                "MULTICHAIN_CONNECTION_CHAIN_EXCLUDED: selected chain "
                "participates in a LINK to an excluded chain"
            )
        if not all(included):
            continue
        left = atom_lookup.get((
            c1, link['num1'], str(link.get('icode1', '')), link['atom1']
        ), [])
        right = atom_lookup.get((
            c2, link['num2'], str(link.get('icode2', '')), link['atom2']
        ), [])
        if len(left) != 1 or len(right) != 1:
            raise ValueError(
                "MULTICHAIN_CONNECTION_ATOM_UNMAPPED: LINK endpoints do not "
                "map uniquely to selected atoms"
            )
        add_atom_connection(c1, left[0], c2, right[0], 'LINK')

    all_serial_atoms = {}
    all_chain_ids = set(used_chains)
    for chain in _peptide_chain_ids(pdb_path):
        all_chain_ids.add(chain)
    for chain in all_chain_ids:
        for serial, atom in read_atoms(pdb_path, chain).items():
            all_serial_atoms.setdefault(serial, []).append((chain, atom))
    for left_serial, right_serial in read_conect(pdb_path):
        left = all_serial_atoms.get(left_serial, [])
        right = all_serial_atoms.get(right_serial, [])
        if bool(left) != bool(right):
            known = left or right
            if any(chain in chain_polymer for chain, _atom in known):
                raise ValueError(
                    "MULTICHAIN_CONNECTION_CHAIN_EXCLUDED: selected chain "
                    "participates in a CONECT edge to an excluded chain"
                )
            continue
        if not left:
            continue
        if len(left) != 1 or len(right) != 1:
            raise ValueError(
                "MULTICHAIN_CONNECTION_ATOM_UNMAPPED: CONECT serial does not "
                "map uniquely"
            )
        (c1, a1), (c2, a2) = left[0], right[0]
        included = (c1 in chain_polymer, c2 in chain_polymer)
        if any(included) and not all(included):
            raise ValueError(
                "MULTICHAIN_CONNECTION_CHAIN_EXCLUDED: selected chain "
                "participates in a CONECT edge to an excluded chain"
            )
        if all(included):
            add_atom_connection(c1, a1, c2, a2, 'CONECT')

    seq_part = '|'.join(f'PEPTIDE{i + 1}{{{blk}}}'
                        for i, blk in enumerate(blocks))
    conn_str = '|'.join(connections) if connections else ''
    return f'{seq_part}${conn_str}$$$'


def generate_multichain(
    pdb_path, chain_ids=None, *, monomer_context=None
):
    """Generate SMILES for a multi-chain peptide (e.g. insulin) from a PDB.

    Returns (smiles, error); error is None on success.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ):
            return generate_multichain(pdb_path, chain_ids)
    try:
        helm = build_helm_multichain(pdb_path, chain_ids)
        if not helm:
            return None, "no usable peptide chains"
        mapped = helm_to_map(helm)
        if not mapped or mapped.startswith('ERROR'):
            return None, f"helm_to_map: {mapped}"
        smiles = get_smi_from_map(mapped)
        if not smiles:
            return None, "get_smi_from_map returned None"
        return smiles, None
    except Exception as ex:
        return None, str(ex)
