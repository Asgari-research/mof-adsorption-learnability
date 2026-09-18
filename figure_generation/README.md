# Figure generation

This directory contains the current plotting implementation and the saved source values directly required by it.

## What this workflow does

- reads the CSV/TXT inputs under `data/`;
- regenerates Figures 1–5 and S1–S2 into `outputs/final/`;
- optionally redraws S3 into `outputs/held_s3/`;
- writes source/output hash manifests and checks that source files were not modified;
- verifies output dimensions and embedded Arial fonts when `pdffonts` is available.

It does **not** run the upstream model-fitting pipeline, select new candidates, recompute conformal intervals, or rerun external-domain calculations.

## WSL2 setup using the current Python environment

```bash
bash setup_wsl.sh
bash RUN_ALL_WSL.sh
```

`setup_wsl.sh` uses the Python already active in the shell. It creates **no** Conda environment and no venv. If dependencies are missing, it reports the optional installation command for the current environment.

Final exports require Arial. On Windows + WSL2, `setup_wsl.sh` links the four Arial styles from `/mnt/c/Windows/Fonts/` into a user font directory. Font files are not included in the repository.

## Publication assets

The authoritative manuscript PDFs live in `../figures/`, outside this directory. Plotting scripts write only to `outputs/`; do not copy regenerated files into `../figures/` without a deliberate review and manifest update.

## Figure S3

S3 is regenerated only when `--include-held-s3` is requested. Its historical external-geometry mapping remains provenance-limited; see `../docs/figures/S3_PROVENANCE_STATUS.md`.
