#!/usr/bin/env python3
"""Run all RDpepper case studies end-to-end."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

CASES = [
    {
        "name": "PRD_000227 (NNAA cyclic peptide, code 004)",
        "pdb": DATA / "PRD_000227.pdb",
        "chain": "L",
    },
    {
        "name": "PRD_000331 (thioether, cystine-knot)",
        "pdb": DATA / "PRD_000331.pdb",
        "chain": "A",
    },
    {
        "name": "PRD_000807 (TPO-containing cyclic peptide)",
        "pdb": DATA / "PRD_000807.pdb",
        "chain": "L",
    },
]

for case in CASES:
    pdb = case["pdb"]
    if not pdb.exists():
        print(f"[SKIP] {case['name']}: {pdb.name} not found")
        continue
    mol2 = OUTPUTS / f"{pdb.stem}.mol2"
    print(f"\n{'='*60}")
    print(f"[RUN ] {case['name']}")
    print(f"       input: {pdb.name}, chain {case['chain']}")
    print(f"{'='*60}")

    # reconstruct + export
    r = subprocess.run(
        [sys.executable, "-m", "cycpep_master.cli.main",
         "export", str(pdb),
         "--chain", case["chain"],
         "--format", "mol2",
         "--output", str(mol2),
         "--best-available"],
        capture_output=True, text=True, cwd=str(HERE.parent),
    )
    if mol2.exists():
        vr = mol2.parent / f"{mol2.name}.validation.json"
        print(f"  MOL2: {mol2.stat().st_size} bytes")
        if vr.exists():
            import json
            v = json.loads(vr.read_text(encoding="utf-8"))
            print(f"  coordinate_level: {v.get('coordinate_level')}")
            print(f"  formal_charge: {v.get('formal_charge')}")
            print(f"  identity: {v.get('full_inchikey', 'N/A')[:30]}...")
        # charge-aware readback
        r2 = subprocess.run(
            [sys.executable, "-c",
             f"from cycpep_master import application; "
             f"r = application.read_mol2(r'{mol2}', "
             f"compatibility='rdkit_charge_aware'); "
             f"print('readback:', r.get('status'), "
             f"r.get('data', {{}}).get('total_formal_charge'))"],
            capture_output=True, text=True, cwd=str(HERE.parent),
        )
        print(f"  readback: {r2.stdout.strip()}")
    else:
        print(f"  FAILED: {r.stderr[-200:]}")

print(f"\n{'='*60}")
print("All case studies complete.")
print(f"Outputs in: {OUTPUTS}")
