#!/usr/bin/env python3
"""RDpepper Python API example: end-to-end reconstruction + export + readback.

Run from the repository root:
    python case_studies/api_example.py
"""
import json
import sys
from pathlib import Path

# Add the package to the path (adjust if running from elsewhere)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cycpep_master import application


def main():
    pdb = Path(__file__).parent / "data" / "PRD_000227.pdb"

    # --- Step 1: Reconstruct and export MOL2 ---
    print("=== Step 1: Export MOL2 ===")
    mol2_path = pdb.parent.parent / "outputs" / "api_example.mol2"
    mol2_path.parent.mkdir(exist_ok=True)

    result = application.export_best_available(
        str(pdb),
        str(mol2_path),
        source_kind="coordinate",
        output_format="mol2",
        chain_id="L",
    )
    data = result.get("data") or {}
    print(f"  status: {result.get('status')}")
    print(f"  format: {data.get('requested_format_status')}")
    print(f"  SMILES: {(data.get('smiles') or '')[:60]}...")

    # Check the validation receipt
    vr_path = mol2_path.parent / (mol2_path.name + ".validation.json")
    if vr_path.exists():
        vr = json.loads(vr_path.read_text(encoding="utf-8"))
        print(f"  coordinate_level: {vr.get('coordinate_level')}")
        print(f"  formal_charge: {vr.get('formal_charge')}")
        print(f"  InChIKey: {vr.get('full_inchikey', '')[:30]}...")
        print(f"  heavy atoms: {vr.get('heavy_atom_count')}")

    # --- Step 2: Read back with charge awareness ---
    print("\n=== Step 2: Charge-aware readback ===")
    read_result = application.read_mol2(
        str(mol2_path),
        compatibility="rdkit_charge_aware",
    )
    read_data = read_result.get("data") or {}
    print(f"  status: {read_result.get('status')}")
    print(f"  reader_mode: {read_data.get('reader_mode')}")
    print(f"  total_formal_charge: {read_data.get('total_formal_charge')}")
    print(f"  heavy_atom_count: {read_data.get('heavy_atom_count')}")
    print(f"  coordinates_verified: {read_data.get('coordinates_verified')}")

    receipt_check = read_data.get("receipt_verification") or {}
    print(f"  receipt: {receipt_check.get('status')}")

    # --- Step 3: SMILES input ---
    print("\n=== Step 3: SMILES direct input ===")
    smiles = "CC(C)C[C@H](NC(=O)[C@@H](N)CCCNC(N)=N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(O)=O"
    smiles_mol2 = mol2_path.parent / "api_example_smiles.mol2"
    smi_result = application.export_best_available(
        smiles,
        str(smiles_mol2),
        source_kind="smiles",
        output_format="mol2",
    )
    smi_data = smi_result.get("data") or {}
    print(f"  status: {smi_result.get('status')}")
    print(f"  format: {smi_data.get('requested_format_status')}")

    print("\nDone. Outputs in:", mol2_path.parent)


if __name__ == "__main__":
    main()
