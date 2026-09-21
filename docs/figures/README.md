# Figure provenance and regeneration

The eight PDFs under `figures/` are the publication-authority assets for the finalized figure state dated 2026-09-21. Their SHA-256 values are recorded in `FINAL_FIGURE_MANIFEST.csv`.

Figure 1 is a locked static asset. The current plotting implementation regenerates Figures 2-5 and S1-S3 from saved tables under `figure_generation/data/source_data/`, writing only to `figure_generation/outputs/`. The plotting workflow does not refit models or recompute scientific results.

The locked PDFs remain authoritative even if a future rerun differs in PDF metadata or font embedding. Replacing a locked figure requires explicit scientific review and a manifest update.

`FINAL_FIGURE_SOURCE_MANIFEST.csv` records the saved figure-source files. `CURRENT_CODE_MANIFEST.csv` records the current plotting and helper code.

Figure S3 is part of the locked SI figure set; the historical external-geometry mapping limitation documented in `S3_PROVENANCE_STATUS.md` still applies to scientific interpretation.
