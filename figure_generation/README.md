# Figure generation

This directory contains the current plotting-only workflow for the finalized FewShot figures.

## Scope

`code/final_publication_figures.py` reads saved CSV values under `data/source_data/` and regenerates Figures 2-5 and S1-S3. It does not fit models, alter shortlist membership, recompute conformal intervals, or rerun adsorption calculations. Figure 1 is a locked static publication asset and is not regenerated.

The authoritative manuscript PDFs are stored under `../figures/`. Regeneration writes only to `outputs/`; a rerun must never overwrite the locked PDFs automatically.

## Windows / Anaconda

Activate the intended environment, then run:

```text
conda activate mofenv
cd figure_generation
python check_environment.py
python run_all.py
python verify_outputs.py
```

To regenerate one figure:

```text
python run_figure.py 3
```

Arial is required for final regeneration. `--allow-font-fallback-for-preview` is available only for layout testing.

## WSL2

The existing `setup_wsl.sh` helper can expose Windows Arial to Matplotlib in WSL2. After setup, run `python run_all.py`.

## Final layouts

- Figure 2: 2 x 3
- Figure 3: 3 x 2
- Figure 4: 2 x 3
- Figure 5: 3 x 2
- Figure S1: 2 x 2
- Figures S2-S3: 1 x 2
