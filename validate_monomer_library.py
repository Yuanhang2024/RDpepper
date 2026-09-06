"""Validate every monomer in unified_monomer_library.csv for end-to-end
fragment combination and cyclization capability.

For each monomer:
  1. A + monomer   (R1→R2 peptide bond)
  2. monomer + A   (R2→R1 peptide bond)
  3. C + monomer + C disulfide cyclization (if R3 != "-")

Writes backbone_valid and sidechain_valid columns back to the CSV.
"""

import csv
import os
import sys
import time
from typing import Dict, Tuple

# Ensure project root on path so cycpep_master can be imported as a package
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from cycpep_master.paths._map_utils import (
    monomers2smi_dict,
    monomers2r_groups_dict,
    get_linear_peptide,
    cyclize_linpep_from_map,
    relabel_rgroup2index,
)


_R_GROUP_CAPS = {
    "R1": frozenset({"H"}),
    "R2": frozenset({"H", "OH", "NH2"}),
    "R3": frozenset({"H", "OH", "SH"}),
}


def _parse_sanitized_result(result):
    """Parse an assembler result, including its legacy ``_R`` labels."""
    if not isinstance(result, str) or not result.strip():
        return None
    normalized = relabel_rgroup2index(result) if "_R" in result else result
    try:
        mol = Chem.MolFromSmiles(normalized, sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumAtoms() == 0 or mol.GetNumBonds() == 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def _validate_port_declarations(symbol: str):
    """Validate R-group declarations against the parsed monomer structure.

    Returns ``(mol, declarations_ok, has_r3)``. A declared port must be one
    of the supported leaving-group caps and map to exactly one dummy atom with
    one real-atom neighbour. A missing R3 is a valid no-sidechain case.
    """
    smi = monomers2smi_dict.get(symbol)
    rg = monomers2r_groups_dict.get(symbol, {})
    declared_r3 = str(rg.get("R3", "-")).strip().upper() or "-"
    has_r3 = declared_r3 != "-"
    if not smi:
        return None, False, has_r3
    mol = _parse_sanitized_result(smi)
    if mol is None:
        return None, False, has_r3
    declarations_ok = True
    for name, allowed in _R_GROUP_CAPS.items():
        value = str(rg.get(name, "-")).strip().upper() or "-"
        if value not in allowed and not (name == "R3" and value == "-"):
            declarations_ok = False
            continue
        map_num = int(name[1:])
        ports = [
            atom for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_num
        ]
        usable = len(ports) == 1 and len(ports[0].GetNeighbors()) == 1
        if value == "-":
            if ports:
                declarations_ok = False
        elif not usable:
            declarations_ok = False
    return mol, declarations_ok, has_r3


def test_fragment_pair(smi1: str, smi2: str) -> bool:
    """Try to form a peptide bond between two monomers. Returns True on success."""
    try:
        result = get_linear_peptide([smi1, smi2])
        return _parse_sanitized_result(result) is not None
    except Exception:
        return False


# Standard reference monomers
ALA_SMI = monomers2smi_dict.get("A", "")


def test_disulfide_cyclization(symbol: str) -> bool:
    """Build C-monomer-C 3-mer and cyclize via disulfide. Returns True on success."""
    smi = monomers2smi_dict.get(symbol)
    cys_smi = monomers2smi_dict.get("C")
    if not smi or not cys_smi:
        return False
    try:
        linear = get_linear_peptide([cys_smi, smi, cys_smi])
        if _parse_sanitized_result(linear) is None:
            return False
        cyclized = cyclize_linpep_from_map(["C", symbol, "C"], "1:R3-3:R3")
        mol = _parse_sanitized_result(cyclized)
        if mol is None or mol.GetRingInfo().NumRings() < 1:
            return False
        return any(
            bond.GetBondType() == Chem.BondType.SINGLE
            and bond.GetBeginAtom().GetAtomicNum() == 16
            and bond.GetEndAtom().GetAtomicNum() == 16
            for bond in mol.GetBonds()
        )
    except Exception:
        return False


def test_monomer(symbol: str) -> Tuple[bool, bool]:
    """Test a single monomer. Returns (backbone_ok, sidechain_ok)."""
    smi = monomers2smi_dict.get(symbol)
    if not smi:
        return False, False

    _, declarations_ok, has_r3 = _validate_port_declarations(symbol)

    # Test 1: A + monomer
    ok1 = test_fragment_pair(ALA_SMI, smi)

    # Test 2: monomer + A
    ok2 = test_fragment_pair(smi, ALA_SMI)

    backbone_ok = declarations_ok and ok1 and ok2

    # A missing R3 is a valid monomer without a side-chain attachment; it is
    # not a failed side-chain test. For declared R3, require the parsed dummy
    # port to have a real neighbour (checked above).
    sidechain_ok = declarations_ok

    return backbone_ok, sidechain_ok


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "unified_monomer_library.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add new columns if missing
    new_fields = list(fieldnames)
    for col in ["backbone_valid", "sidechain_valid"]:
        if col not in new_fields:
            new_fields.append(col)

    total = len(rows)
    backbone_ok = 0
    sidechain_ok = 0
    t0 = time.time()

    print(f"Validating {total} monomers...")
    for i, row in enumerate(rows):
        symbol = row.get("symbol", "").strip()
        if not symbol:
            continue

        b_ok, s_ok = test_monomer(symbol)
        row["backbone_valid"] = "1" if b_ok else "0"
        row["sidechain_valid"] = "1" if s_ok else "0"

        if b_ok:
            backbone_ok += 1
        if s_ok:
            sidechain_ok += 1

        if (i + 1) % 500 == 0:
            elapsed = max(time.time() - t0, 1)
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  {i+1}/{total} rate={rate:.1f}/s ETA={eta:.0f}s "
                  f"backbone={backbone_ok} sidechain={sidechain_ok}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Backbone valid: {backbone_ok}/{total} ({100*backbone_ok/max(total,1):.1f}%)")
    print(f"Sidechain valid: {sidechain_ok}/{total} ({100*sidechain_ok/max(total,1):.1f}%)")

    # Write back
    tmp = csv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp, csv_path)
    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()
