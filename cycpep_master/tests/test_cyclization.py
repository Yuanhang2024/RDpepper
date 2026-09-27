"""Tests for cyclization detection and cross-path SMILES consistency.

These run against the gold-standard CPBind PDBs under
TestFiles_Example/CPBind_Examples. If those files are absent the whole module
is skipped (so the suite still passes in a minimal checkout).

The strongest correctness signal in this package is that four independent
SMILES generators agree:
  A  per-residue template assembly (atom-level)
  B  HELM -> MAP -> SMILES (symbol-level)
  C  A + HETATM-cap merge
  E  A + geometric covalent-radius cyclization
"""
import pytest

from cycpep_master.paths import (
    generate_a, generate_b, generate_c, generate_e, build_helm_from_pdb,
)
from cycpep_master.paths._map_utils import helm_to_map, get_smi_from_map
from cycpep_master.core.cyclization import detect_cyclization
from .conftest import canonical, has_dummy, count_smarts

ALDEHYDE = "[CX3H1]=O"


def _smi(result):
    return result[0] if isinstance(result, tuple) else result


def _all_paths(pdb):
    """Canonical SMILES from all four paths for one PDB."""
    a = canonical(_smi(generate_a(pdb, "L")))
    c = canonical(_smi(generate_c(pdb, "L")))
    e = canonical(_smi(generate_e(pdb, "L")))
    helm = build_helm_from_pdb(pdb, "L")
    mp = helm_to_map(helm) if helm else None
    b = canonical(get_smi_from_map(mp)) if mp else None
    return {"a": a, "b": b, "c": c, "e": e}


def test_cross_path_agreement(pdb_files):
    """A == B == C == E on every gold PDB, with no dummies and no aldehydes."""
    disagreements = []
    for pdb in pdb_files:
        res = _all_paths(pdb)
        if not all(res.values()):
            disagreements.append((pdb, "a path returned None", res))
            continue
        if len(set(res.values())) != 1:
            disagreements.append((pdb, "paths disagree", res))
            continue
        smi = res["a"]
        if has_dummy(smi):
            disagreements.append((pdb, "residual dummy", smi))
        if count_smarts(smi, ALDEHYDE) != 0:
            disagreements.append((pdb, "aldehyde present", smi))
    assert not disagreements, (
        f"{len(disagreements)}/{len(pdb_files)} PDBs failed:\n"
        + "\n".join(f"  {d[0]}: {d[1]}" for d in disagreements[:10]))


def test_cyclization_detected_on_each_pdb(pdb_files):
    """detect_cyclization returns a valid topology for every gold PDB."""
    valid = {"linear", "monocyclic", "bicyclic", "tricyclic", "polycyclic"}
    bad = []
    for pdb in pdb_files:
        info = detect_cyclization(pdb, "L")
        if info.topology not in valid:
            bad.append((pdb, info.topology))
    assert not bad, f"unexpected topologies: {bad[:10]}"


def test_geometric_recovery_when_no_records(pdb_files, tmp_path):
    """Path E recovers a disulfide geometrically only when records are absent.

    Strips CONECT from a disulfide-bridged gold PDB: with no explicit records,
    the geometric scan must recover the S-S bond. (When records ARE present,
    apply_geometric_crosslinks defers to them and adds nothing — covered by the
    cross-path agreement test, where Path E never over-adds.)
    """
    from cycpep_master.core.cyclization import detect_cyclization as _detect
    disulfide_pdb = None
    for pdb in pdb_files:
        if "disulfide" in _detect(pdb, "L").description:
            disulfide_pdb = pdb
            break
    if disulfide_pdb is None:
        import pytest
        pytest.skip("no disulfide-bridged gold PDB available")

    stripped = tmp_path / "no_conect.pdb"
    with open(disulfide_pdb) as fin, open(stripped, "w") as fout:
        for line in fin:
            if not line.startswith("CONECT"):
                fout.write(line)

    from rdkit import Chem
    smi = canonical(_smi(generate_e(str(stripped), "L")))
    assert smi, "Path E returned nothing on record-less disulfide PDB"
    mol = Chem.MolFromSmiles(smi)
    n_ss = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16X2]-[#16X2]")))
    assert n_ss >= 1, "geometric recovery failed to restore the disulfide"

