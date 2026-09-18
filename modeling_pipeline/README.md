# Historical modelling pipeline

This directory preserves the computational modelling source supplied for the MOF adsorption learnability study. It is separated from the current publication-figure workflow so that numerical provenance and figure regeneration are not conflated.

## Release policy

The main modelling source is included **byte-for-byte as supplied**. Phase 2 release preparation did not refit models, recompute candidates, modify thresholds, alter random seeds, repair the external geometry mapping, or rewrite the historical override chain.

Primary historical source:

- `src/fewshot_mof_risk_controlled_external_pipeline_v4_8_supplied.py`
- SHA-256: `1553d2b27118b400b53b17d436e76b33f10629445b62c311d70e2f9e907b3c3d`

Historical publication renderer retained for provenance:

- `legacy_publication_renderer/regenerate_publication_figures_and_tables_v4_9.py`
- SHA-256: `c5f1fdfe8f7d758834124996bffeead5e46cbf4eccb33650cdb2abfc079a59ae`

The renderer above is **not** the current figure-generation route. Current manuscript figures and their saved source tables are under `../figure_generation/` and `../figures/`.

## What the historical pipeline implements

The supplied code constructs four 298 K gravimetric adsorption targets, geometry and RAC descriptor representations, random/grouped train-calibration-test splits, restricted model-fitting budgets, multiple regressors, top-k ranking metrics, empirical conformal intervals, candidate collection/consensus stages, and optional external-domain annotations.

The saved results audited for the manuscript contain geometry-only and geometry+RAC representations. TabPFN is an optional code branch but is not present in the supplied evaluated result tables. The nominal RDF branch is likewise not an independently evaluated representation in the supplied aggregate results.

## Important execution semantics

The tested budget is a **model-fitting budget**, not the full number of labels touched by the retrospective workflow. Calibration and test reservoirs remain outside that fitting subset, and the fitting subset uses target-stratified sampling across the tested budget grid.

The module also has import-time side effects: it configures runtime state and creates output directories before the final `if __name__ == "__main__"` guard. Do not import it merely to inspect configuration. Use the static maps under `../docs/modeling/` instead.

## Inputs and paths

At runtime the script requires at least:

- `geometric_properties.csv`
- `post_comb_vsa-CO2.csv`
- `methane.csv`

Additional cluster, RAC, topology, process, and external database files are optional or stage-specific. The exact path aliases embedded in the historical source are recorded in `../docs/modeling/INPUT_PATH_ALIASES.csv`.

The script accepts `--data-root` and also contains historical machine-specific fallback locations. Those fallbacks are preserved rather than silently rewritten because changing data resolution can change which files are selected. Prefer an explicit `--data-root` when intentionally rerunning the historical pipeline.

Example only:

```bash
python src/fewshot_mof_risk_controlled_external_pipeline_v4_8_supplied.py \
  --data-root /path/to/project_data \
  --n-jobs 2
```

A full run can be expensive. **No full run is required to use the finalized manuscript figures in this repository.**

## Dependencies

`requirements-historical.txt` is the unpinned dependency list supplied with the historical code. It is not an environment lock and should not be interpreted as the exact environment used for the saved results. No exact environment export was available in the supplied audit bundle.

Core scientific imports include NumPy, pandas, SciPy, scikit-learn, Matplotlib, and joblib. LightGBM and XGBoost are optional model dependencies. TabPFN is an optional code path but is not part of the supplied evaluated benchmark rows.

## Static audit and known limitations

See `../docs/modeling/` for:

- exact source hashes;
- configuration defaults;
- CLI arguments;
- input-path aliases;
- import/dependency classification;
- the complete top-level definition map;
- the active override map for repeated function names;
- known scientific/provenance limitations.

The external-domain `Di`/`Df` mapping issue documented in the manuscript audit is intentionally **not repaired here**. Historical external-overlap outputs must not be presented as corrected by this release step.
