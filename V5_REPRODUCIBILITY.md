# V5 reproducibility

Install the package with docking support and run:

```bash
cycpep prepare-sequence ACDEFG ./v5_smoke \
  --cyclization head-to-tail \
  --conformers 4 \
  --flexibility-mode balanced \
  --seed 42 \
  --threads 1
```

For a MOL2-only check that does not require Meeko:

```bash
cycpep-v5-reproduce ./v5_reproduction --no-pdbqt
```

The reproduction receipt records the package/runtime versions, resource
hashes, operation status, logical artifact IDs and emitted file hashes.
Re-running in a different empty output directory with the same chemistry,
resource files, seed and thread count must reproduce logical artifact IDs.

`balanced` and `thorough` consume an already materialized
`ensemble_manifest.json`. They do not generate coordinates in the PDBQT
stage. Unknown or low-confidence torsions remain flexible.

The command refuses a nonempty destination and never mutates packaged
resources, frozen benchmark results or the unified monomer library.

