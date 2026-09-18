# Modelling-code scientific and provenance notes

This file records limitations that matter for a public computational release. It is not a substitute for the manuscript Methods section.

## Preserved scientific semantics

Phase 2 does not change any of the following historical implementation choices:

- target-stratified sampling across tested model-fitting budgets;
- train/calibration/test split algorithms and grouped-split fallbacks;
- descriptor construction or feature-selection behavior;
- model constructors, hyperparameters, random seeds, or per-budget logic;
- conformal interval construction;
- run-level candidate eligibility thresholds;
- prediction-row compaction;
- consensus score construction;
- external geometry matching or `Di`/`Df` mapping;
- cached-stage logic or filename generation.

## Interpretation boundaries from the audit

1. **Model-fitting budget is not total label cost.** Calibration and test reservoirs exist outside the restricted fitting subset, and fitting indices are selected retrospectively using target values.
2. **Split-conformal intervals have one global `q` per run.** Within a run, subtracting the same `q` does not change point-prediction ranking, and the theoretical split interval width is constant.
3. **Compacted candidate rows are a preselected retrospective pool.** Some compaction routes can retain large-reference-error rows using `y_true`; downstream row-retention curves are not library-wide unique-MOF yield curves.
4. **External geometry mapping remains unresolved.** The supplied source maps `Di`/`Df` in a way that conflicts with the usual cavity/aperture convention. No historical external-distance output is silently recomputed or relabelled here.
5. **Grouped transfer penalties are benchmark-specific.** They should not be interpreted as causal effects of topology or another held-out grouping variable.
6. **TabPFN and RDF should not be advertised as evaluated benchmark results.** They are optional/unrealized branches relative to the supplied result tables.

## Historical source structure

The modelling file contains a sequence of appended publication/plotting layers and repeated top-level definitions. The final end-of-file definitions are active for repeated global names when `main()` executes. `TOP_LEVEL_OVERRIDE_MAP.csv` records those repetitions. The exact source is preserved rather than refactored because a large refactor would be a behavior-changing release task unless separately validated.

## Reproducibility limits

The release contains source code and static provenance maps but not a verified raw-input snapshot, historical environment lock, split-index archive, or complete cache/manifests for every historical execution. Therefore this repository should not claim exact end-to-end numerical reproduction from a fresh environment solely from the files in this release.
