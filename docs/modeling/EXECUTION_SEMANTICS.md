# Historical execution semantics

This note documents historical behavior that matters for intentional reruns and interpretation. It does not modify the preserved source.

## 1. Budget-dependent model settings

The historical HGB constructor is budget-aware:

```text
max_iter = min(hgb_max_iter, max(20, budget // 5))
```

With the default `hgb_max_iter = 100`, the tested grid gives:

| Fitting budget | HGB max_iter |
|---:|---:|
| 10 | 20 |
| 20 | 20 |
| 50 | 20 |
| 100 | 20 |
| 200 | 40 |
| 500 | 100 |
| 1000 | 100 |

Accordingly, an HGB-specific learning curve changes both fitting-set size and allowed boosting iterations.

The MLP constructor also changes regime at budget 50:

| Fitting budget | Hidden layers | Early stopping |
|---:|---|---|
| < 50 | `(32,)` | disabled |
| >= 50 | `(128, 64)` | enabled |

These are historical implementation choices and are preserved rather than silently standardized.

## 2. Restart/cache semantics

A stage is considered complete when its `.done.json` file exists and `force_recompute` is false. The marker records stage metadata but is not a cryptographic/full configuration fingerprint.

Per-experiment cache names encode:

- target;
- feature set;
- split;
- model;
- fitting budget;
- seed.

They do not encode every model hyperparameter, RAM profile, source-file hash, candidate-gate setting, or target-transform option.

Therefore, **changing configuration while reusing an old output directory can reuse incompatible cached results**.

For an intentional rerun with changed settings, use a clean output directory or deliberately force/rebuild the affected stages and retain the run configuration/environment alongside the outputs.

## 3. RAM profiles are not purely memory controls

The `standard`/`normal` profiles mainly preserve full descriptor breadth, but the low-memory profiles can change scientific outputs.

For example:

- `ultra_light` can cap RACs at 80, disable local conformal, reduce candidate-table/external-neighbor limits, and cap bootstrap repetitions;
- `light` can cap RACs at 160 and change several uncertainty/candidate limits.

The audited saved benchmark uses 148 RAC descriptors in the chemistry representation, consistent with the uncapped standard/full-width representation. Nevertheless, RAM mode should be recorded as part of the scientific run configuration.

## 4. Identifier normalization

The historical `normalize_mof_id` function strips:

- a terminal `.cif`;
- a terminal `_repeat`.

Geometry, RAC and RDF tables then use deduplication on the normalized key in several merge paths, while target rows are aggregated by normalized key.

The current release does not contain the raw input inventories needed to prove whether distinct historical identifiers collided under this normalization. This is therefore an unresolved input-provenance check, not evidence that collisions occurred.

## 5. Full-table descriptor screening

RAC columns are screened for global non-constancy before train/test partitioning. If a RAC feature cap is enabled, variance ranking is also computed on the full RAC table before splitting.

The audited saved benchmark representation contains all 148 RAC descriptors, so the optional low-memory variance cap does not define the supplied main representation. Alternative reruns using capped profiles should still record this preprocessing behavior.

## 6. Non-finite saved diagnostics

The saved quality-gate table contains genuine non-finite diagnostic outcomes. They should not be silently deleted from a release simply to make the benchmark appear cleaner.

The repository verifier summarizes these values from the included Table S3 as an integrity/diagnostic check. Their presence is compatible with a benchmark that explicitly records failed or degenerate configurations and then applies documented population restrictions/QC downstream.

## 7. Historical source versus current figure workflow

The historical source should not be imported merely for inspection because it has import-time side effects. Use the static maps under `docs/modeling/`.

The current figure workflow under `figure_generation/` does not execute the historical model-fitting pipeline.
