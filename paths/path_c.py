"""Path C: merge explicit caps using Unified-library-backed templates."""
from rdkit import Chem

from ..core.pdb_parser import (
    get_res_seq,
    get_pdb_atoms,
    parse_backbone,
    read_conect,
)
from ..core.molecule import add_to_combo, apply_conect, remove_orphans, finish_mol
from ..core.monomer_resolution import needs_monomer_resolution_scope
from .path_a import (
    _has_head_to_tail_evidence,
    _materialize_explicit_topology_bonds,
    _materialize_free_c_terminal_hydroxyl,
)
from .residue_template_factory import (
    CappedTemplate,
    cap_atom_name_map,
    compose_capped_template,
    get_residue_template,
    map_pdb_atoms,
)


_CAP_RESIDUES = frozenset({"ACE", "NME", "NH2"})


def _build_atom2key(residues, pdb_path, chain_id='L'):
    atom2key = {}
    for r in residues:
        for pa in get_pdb_atoms(pdb_path, r['key'], chain_id):
            atom2key[pa['num']] = r['key']
    return atom2key


def _map_het_to_std(conect, atom2key):
    het_to_std = {}
    for src, targets in conect.items():
        if src not in atom2key:
            continue
        sk = atom2key[src]
        for tgt in targets:
            if tgt not in atom2key:
                continue
            tk = atom2key[tgt]
            if sk[0] in _CAP_RESIDUES and tk[0] not in _CAP_RESIDUES:
                het_to_std[sk] = tk
            elif tk[0] in _CAP_RESIDUES and sk[0] not in _CAP_RESIDUES:
                het_to_std[tk] = sk
    return het_to_std


def _cap_links(pdb_path, residues, chain_id):
    """Resolve cap attachments from selected-chain LINK records."""
    from ..core.cyclization import read_link

    by_identity = {}
    for residue in residues:
        by_identity.setdefault((
            residue["name"], residue["num"], str(residue.get("icode", ""))
        ), []).append(
            residue["key"]
        )
    resolved = {}
    for link in read_link(pdb_path):
        if link["chain1"] != chain_id or link["chain2"] != chain_id:
            continue
        keys_1 = by_identity.get((
            link["res1"], link["num1"], str(link.get("icode1", ""))
        ), [])
        keys_2 = by_identity.get((
            link["res2"], link["num2"], str(link.get("icode2", ""))
        ), [])
        if len(keys_1) != 1 or len(keys_2) != 1:
            raise ValueError("LINK cap endpoint does not resolve uniquely")
        key_1, key_2 = keys_1[0], keys_2[0]
        if key_1[0] in _CAP_RESIDUES and key_2[0] not in _CAP_RESIDUES:
            cap, standard = key_1, key_2
        elif key_2[0] in _CAP_RESIDUES and key_1[0] not in _CAP_RESIDUES:
            cap, standard = key_2, key_1
        else:
            continue
        previous = resolved.get(cap)
        if previous is not None and previous != standard:
            raise ValueError("LINK records assign one cap to multiple residues")
        resolved[cap] = standard
    return resolved


def _resolve_cap_attachments(pdb_path, residues, conect, atom2key, chain_id):
    conect_map = _map_het_to_std(conect, atom2key)
    link_map = _cap_links(pdb_path, residues, chain_id)
    for cap in set(conect_map) & set(link_map):
        if conect_map[cap] != link_map[cap]:
            raise ValueError("LINK/CONECT cap attachment disagreement")
    combined = dict(conect_map)
    combined.update(link_map)
    return combined


def _build_ext_residues(residues, pdb_path, std_het, chain_id='L'):
    ext_residues = []
    merged_caps = set(std_het.values())
    for r in residues:
        if r['key'] in merged_caps:
            continue
        std_key = r['key']
        std_pats = get_pdb_atoms(pdb_path, std_key, chain_id)
        het_name = None
        ext_pats = list(std_pats)
        if std_key in std_het:
            het_key = std_het[std_key]
            het_name = het_key[0]
            het_pats = get_pdb_atoms(pdb_path, het_key, chain_id)
            ext_pats.extend(het_pats)
        ext_residues.append((r, ext_pats, het_name))
    return ext_residues


def _choose_template(rname, het_name):
    if het_name == 'ACE':
        return compose_capped_template(rname, 'ACE')
    elif het_name in {'NME', 'NH2'}:
        return compose_capped_template(rname, het_name)
    return get_residue_template(rname)


def _generate(pdb_path, chain_id='L'):
    residues = get_res_seq(pdb_path, chain_id)
    conect = read_conect(pdb_path)
    atom2key = _build_atom2key(residues, pdb_path, chain_id)
    het_to_std = _resolve_cap_attachments(
        pdb_path, residues, conect, atom2key, chain_id
    )
    std_het = {v: k for k, v in het_to_std.items()}
    unmerged_caps = [
        residue['name']
        for residue in residues
        if residue['name'] in _CAP_RESIDUES and residue['key'] not in het_to_std
    ]
    if unmerged_caps:
        raise ValueError(
            f"Unified cap(s) lack an explicit residue attachment: {sorted(unmerged_caps)}"
        )

    ext_residues = _build_ext_residues(residues, pdb_path, std_het, chain_id)

    rnames_all = [r['name'] for r in residues]
    has_caps = any(name in _CAP_RESIDUES for name in rnames_all)

    combo = Chem.RWMol()
    offs = []
    n_idxs = []
    c_idxs = []
    selected_templates = []

    for r, pats, het_name in ext_residues:
        rname = r['name']
        selected = _choose_template(rname, het_name)
        selected_templates.append(selected)
        smi = selected.smiles
        off = add_to_combo(combo, smi)
        offs.append(off)

        tmol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(tmol)
        ni, _, _, ci, _ = parse_backbone(tmol)
        n_idxs.append(ni)
        c_idxs.append(ci)

    # Peptide backbone bonds
    for i in range(len(ext_residues) - 1):
        if c_idxs[i] is not None and n_idxs[i + 1] is not None:
            ci_g = offs[i] + c_idxs[i]
            ni_g = offs[i + 1] + n_idxs[i + 1]
            if not combo.GetBondBetweenAtoms(ci_g, ni_g):
                combo.AddBond(ci_g, ni_g, Chem.BondType.SINGLE)

    # Cyclic backbone requires selected-chain explicit connection evidence.
    has_head_to_tail = len(ext_residues) >= 2 and _has_head_to_tail_evidence(
        pdb_path,
        chain_id,
        len(ext_residues),
        allow_geometry=False,
    )
    if not has_caps and has_head_to_tail:
        if n_idxs[0] is not None and c_idxs[-1] is not None:
            ni_g = offs[0] + n_idxs[0]
            ci_g = offs[-1] + c_idxs[-1]
            if not combo.GetBondBetweenAtoms(ni_g, ci_g):
                combo.AddBond(ni_g, ci_g, Chem.BondType.SINGLE)

    # Atom mapping
    pdb2g = {}
    pdb2r = {}
    assigned_globals = set()
    connected_serials = set(conect)
    connected_serials.update(target for targets in conect.values() for target in targets)
    serials_by_position_and_name = {}
    for position, residue in enumerate(residues, start=1):
        for atom in get_pdb_atoms(pdb_path, residue["key"], chain_id):
            serials_by_position_and_name.setdefault(
                (position, str(atom["name"]).strip().upper()), []
            ).append(atom["num"])
    from ..core.cyclization import detect_cyclization
    consumed_r3_serials = set()
    for bond in detect_cyclization(
        pdb_path, chain_id, allow_geometric_inference=False
    ).bonds:
        for position, atom_name, rgroup in (
            (
                int(bond.pos1),
                str(bond.atom1 or "").strip().upper(),
                str(bond.rgroup1 or "").strip().upper(),
            ),
            (
                int(bond.pos2),
                str(bond.atom2 or "").strip().upper(),
                str(bond.rgroup2 or "").strip().upper(),
            ),
        ):
            matches = serials_by_position_and_name.get((position, atom_name), [])
            if len(matches) == 1:
                connected_serials.add(matches[0])
                if rgroup == "R3":
                    consumed_r3_serials.add(matches[0])

    for ri, ((r, pats, het_name), selected) in enumerate(
        zip(ext_residues, selected_templates)
    ):
        off = offs[ri]

        std_key = r['key']
        het_key = std_het.get(std_key) if std_key in std_het else None

        std_pat_nums = {pa['num'] for pa in get_pdb_atoms(pdb_path, std_key, chain_id)}
        het_pat_nums = set()
        if het_key:
            het_pat_nums = {pa['num'] for pa in get_pdb_atoms(pdb_path, het_key, chain_id)}

        std_pats = [a for a in pats if a['num'] in std_pat_nums]
        het_pats = [a for a in pats if a['num'] in het_pat_nums]

        if isinstance(selected, CappedTemplate):
            base_mapping = map_pdb_atoms(
                selected.base,
                std_pats,
                connected_serials,
                consumed_r3_serials=consumed_r3_serials,
            )
            translated = {
                serial: selected.base_to_combined[index]
                for serial, index in base_mapping.items()
            }
            local_cap_map = cap_atom_name_map(selected.cap)
            for atom in het_pats:
                if atom['name'] in local_cap_map:
                    translated[atom['num']] = selected.cap_to_combined[
                        local_cap_map[atom['name']]
                    ]
            if len(translated) != len(std_pats) + len(het_pats):
                raise ValueError(f"Incomplete Unified capped-template mapping for {r['name']}")
        else:
            translated = map_pdb_atoms(
                selected,
                std_pats,
                connected_serials,
                consumed_r3_serials=consumed_r3_serials,
            )

        for serial, template_index in translated.items():
            global_index = off + template_index
            pdb2g[serial] = global_index
            pdb2r[serial] = ri
            assigned_globals.add(global_index)

    remove_orphans(combo, assigned_globals, pdb2g)
    apply_conect(combo, pdb2g, pdb2r, conect)
    _materialize_explicit_topology_bonds(
        combo,
        pdb2g,
        residues,
        serials_by_position_and_name,
        pdb_path,
        chain_id,
    )
    from ..core.cyclization import detect_cyclization
    explicit_topology = detect_cyclization(
        pdb_path, chain_id, allow_geometric_inference=False
    )
    occupied_ports = {
        (int(position), str(rgroup).upper())
        for bond in explicit_topology.bonds
        for position, rgroup in (
            (bond.pos1, bond.rgroup1), (bond.pos2, bond.rgroup2)
        )
    }
    if not has_caps:
        _materialize_free_c_terminal_hydroxyl(
            combo,
            pdb2g,
            residues,
            pdb_path,
            chain_id,
            has_head_to_tail=has_head_to_tail,
            occupied_ports=occupied_ports,
        )
    return finish_mol(combo)


def generate(pdb_path, chain_id='L', *, monomer_context=None):
    """Generate a Path C candidate or an explicit fail-closed error tuple."""
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
            return generate(pdb_path, chain_id)
    try:
        return _generate(pdb_path, chain_id)
    except Exception as exc:
        return None, str(exc)
