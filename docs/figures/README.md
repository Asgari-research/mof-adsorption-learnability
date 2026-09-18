# Figure provenance and regeneration

This directory documents the publication figure assets and the source values used by the plotting workflow.

## Publication assets

The eight PDFs under `figures/` are the publication-authority assets for the current manuscript state. Their SHA-256 values are recorded in `FINAL_FIGURE_MANIFEST.csv`.

The plotting workflow under `figure_generation/` is kept separate from those PDFs. Running the plotting code writes to `figure_generation/outputs/`; it does not overwrite `figures/`.

## Source values

`FINAL_FIGURE_SOURCE_MANIFEST.csv` lists the saved tables directly read by the current plotting code. The plotting workflow is a read-only transformation of those saved values. It does not fit models, select new candidates, recompute conformal intervals, or rerun adsorption simulations.

## Important interpretation notes

- `B = 10-1000` is the model-fitting budget, not the total information cost of the workflow.
- Figure 2 bands are the 10th-90th percentile spread across configuration means, not bootstrap confidence intervals.
- Figure 4 reports empirical interval-calibration summaries; the split-conformal interval uses one `q` per run.
- Figure 5 separates the displayed top-25 cohort from the broader top-250 aggregate summary.
- Figure S3 is retained because it is part of the current SI figure set, but its external-geometry mapping remains provenance-limited. See `S3_PROVENANCE_STATUS.md`.
