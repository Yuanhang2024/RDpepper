"""Opt-in automatic standalone-CCD resolution of unknown residue codes.

RDPEPPER_AUTO_CCD is disabled by default; these tests pin the three
contracted behaviors: off-by-default (the loader is never consulted),
offline cache miss (silent fallback to the existing ladder, nothing
persisted), and cache hit (resolution through the authoritative
component path with the "standalone_ccd_component" evidence vocabulary).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core import monomer_resolution
from cycpep_master.core.local_monomer_inference import bootstrap_unknown_monomers
from cycpep_master.paths.residue_template_factory import (
    get_residue_template,
    standard_pdb_atom_name_map,
)

_ALANINE_INCHIKEY = Chem.MolToInchiKey(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))

_ALANINE_COMPONENT_CIF = """data_ZQA
loop_
_chem_comp.id
_chem_comp.type
_chem_comp.name
_chem_comp.formula
ZQA 'L-peptide linking' 'alanine-like fixture' 'C3 H7 N1 O2'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_stereo_config
ZQA N N N N
ZQA H H N N
ZQA H2 H N N
ZQA CA C N S
ZQA HA H N N
ZQA C C N N
ZQA O O N N
ZQA OXT O N N
ZQA HXT H N N
ZQA CB C N N
ZQA HB1 H N N
ZQA HB2 H N N
ZQA HB3 H N N
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
ZQA N H sing N
ZQA N H2 sing N
ZQA N CA sing N
ZQA CA HA sing N
ZQA CA C sing N
ZQA CA CB sing N
ZQA C O doub N
ZQA C OXT sing N
ZQA OXT HXT sing N
ZQA CB HB1 sing N
ZQA CB HB2 sing N
ZQA CB HB3 sing N
"""


def _pdb_line(serial, name, resname, xyz, element, *, resseq=1):
    return (
        f"HETATM{serial:5d} {name:>4s} {resname:>3s} A{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          "
        f"{element:>2s}"
    )


def _write_unknown_alanine(tmp_path, *, resname="ZQA"):
    """One Unified-unknown residue carrying exact L-alanine chemistry."""
    template = get_residue_template("ALA")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=19) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "ALA", template.smiles
        ).items()
    }
    conformer = molecule.GetConformer()
    lines = []
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(_pdb_line(
            atom.GetIdx() + 1,
            names[atom.GetIdx()],
            resname,
            (point.x, point.y, point.z),
            atom.GetSymbol(),
        ))
    path = tmp_path / "unknown-alanine.pdb"
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")
    return path


def _clear_env(monkeypatch):
    for name in (
        "RDPEPPER_AUTO_CCD",
        "RDPEPPER_CCD_CACHE_DIR",
        "RDPEPPER_CCD_ALLOW_NETWORK",
    ):
        monkeypatch.delenv(name, raising=False)


def test_disabled_by_default_never_consults_ccd(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    cache = tmp_path / "ccd-cache"
    cache.mkdir()
    (cache / "ZQA.cif").write_text(_ALANINE_COMPONENT_CIF, encoding="ascii")
    path = _write_unknown_alanine(tmp_path)

    def _forbidden(*args, **kwargs):
        raise AssertionError("CCD loader must not run when feature is off")

    monkeypatch.setattr(
        monomer_resolution, "_load_requested_components", _forbidden
    )
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.ready
    assert bootstrap.inference_results[0].evidence["resolution_mode"] == (
        "unified_library_match"
    )
    assert "standalone_ccd_provenance" not in (
        bootstrap.inference_results[0].evidence
    )


def test_offline_cache_miss_falls_back_silently(tmp_path, monkeypatch):
    cache = tmp_path / "empty-ccd-cache"
    cache.mkdir()
    path = _write_unknown_alanine(tmp_path)

    _clear_env(monkeypatch)
    baseline = bootstrap_unknown_monomers(path, "A")
    monkeypatch.setenv("RDPEPPER_AUTO_CCD", "1")
    monkeypatch.setenv("RDPEPPER_CCD_CACHE_DIR", str(cache))

    feature_on = bootstrap_unknown_monomers(path, "A")
    assert feature_on.status == baseline.status
    assert feature_on.pdb_aliases == baseline.pdb_aliases
    assert feature_on.inference_results[0].evidence["resolution_mode"] == (
        "unified_library_match"
    )
    attempt = feature_on.inference_results[0].evidence.get(
        "standalone_ccd_attempt"
    )
    assert attempt["lookup"] == "unavailable"
    assert attempt["allow_network"] is False
    assert list(cache.iterdir()) == []  # no negative lookups persisted


def test_cached_component_resolves_unknown_code(tmp_path, monkeypatch):
    cache = tmp_path / "warm-ccd-cache"
    cache.mkdir()
    (cache / "ZQA.cif").write_text(_ALANINE_COMPONENT_CIF, encoding="ascii")
    path = _write_unknown_alanine(tmp_path)

    _clear_env(monkeypatch)
    monkeypatch.setenv("RDPEPPER_AUTO_CCD", "1")
    monkeypatch.setenv("RDPEPPER_CCD_CACHE_DIR", str(cache))

    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.ready
    result = bootstrap.inference_results[0]
    assert result.unique
    assert result.status == "unique"
    assert result.evidence["resolution_mode"] == "standalone_ccd_component"
    provenance = result.evidence["standalone_ccd_provenance"]
    assert provenance["component_id"] == "ZQA"
    assert provenance["network_fetch"] is False
    resolution = result.evidence["embedded_chem_comp_resolution"]
    assert resolution["full_inchikey"] == _ALANINE_INCHIKEY
    assert resolution["observed_connectivity_exact"] is True
    # Identical chemistry collapses to the existing Unified alias, now
    # attributed to the standalone-CCD resolution mode.
    assert bootstrap.pdb_aliases[0]["pdb_resname"] == "ZQA"
    assert bootstrap.pdb_aliases[0]["target_symbol"] == "A"
    assert bootstrap.pdb_aliases[0]["resolution_mode"] == (
        "standalone_ccd_component"
    )
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(result.candidate_smiles)) == (
        _ALANINE_INCHIKEY
    )
