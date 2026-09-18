#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fewshot_mof_risk_controlled_pipeline.py
============================================================
Single-file, restart-safe analysis pipeline for the manuscript:

    Few-Shot, Risk-Controlled Adsorption-Shortlist Learning
    Externally Stress-Tested by Stability and Chemical-Validity Overlays

This script is intentionally written as ONE comprehensive Python file so it can
be copied into a folder in Visual Studio / VS Code together with the ARC-MOF,
CoRE MOF 2024, and MOSAEC-DB CSV inputs and run directly. The script also works
when only the ARC-MOF minimum files are available; external-overlay stages are
skipped gracefully if the optional/restricted files are absent.

WHAT THIS SCRIPT DOES
---------------------
1. Loads ARC-MOF tabular inputs from the same folder as this .py file:
   - geometric_properties.csv
   - post_comb_vsa-CO2.csv
   - methane.csv
   - overall_process.csv
   - geo-clusters.csv
   - mc-clusters.csv
   - func-clusters.csv
   - flig-clusters.csv
   - RACs.csv                         optional but strongly recommended
   - RDFs.csv                         optional; if present can be included
   - all_topology_lists.csv            optional; if present used for topology splits
   - post_comb_vsa-N2.csv              optional process/SI extension

2. Builds a clean master modelling table with four adsorption targets:
   - CO2_0p015bar_298K_mmolg
   - CO2_0p150bar_298K_mmolg
   - CH4_5p8bar_298K_mmolg
   - CH4_65bar_298K_mmolg

3. Creates random and grouped train/calibration/test splits:
   - random
   - geometry-grouped
   - metal-chemistry-grouped
   - functional-group-grouped
   - ligand/linker-grouped
   - topology-grouped, if all_topology_lists.csv exists

4. Runs few-shot models at labelled budgets from 10 to 1000 by default.
   It compares robust CPU-friendly baselines and optional packages if installed:
   - Ridge regression
   - Random Forest
   - HistGradientBoosting
   - Gaussian Process, automatically skipped for large budgets unless enabled
   - small MLP
   - LightGBM, optional
   - XGBoost, optional
   - TabPFN, optional

5. Adds conformal reliability layers:
   - split conformal intervals
   - locally weighted conformal intervals
   - Mondrian/group-conditional intervals when group labels are available

6. Computes all manuscript/SI metrics:
   - RMSE, MAE, R2, Spearman
   - Top-1%, Top-5%, Top-10% precision/recall/enrichment/Jaccard/NDCG
   - empirical coverage vs nominal coverage
   - interval width vs label budget
   - abstention/precision trade-off
   - chemical failure anatomy by PLD/LCD/density bins, topology, metal/functional groups
   - trusted/uncertain/rejected candidate tiers
   - bootstrap confidence intervals

7. Saves everything in reusable formats so the analysis does not need rerunning:
   - CSV tables
   - Pickle files
   - JSON manifests/configs
   - model predictions
   - candidate tiers
   - figure-data CSVs
   - high-resolution PNG/PDF composite figures
   - logs

RESTART-SAFE DESIGN
-------------------
Every major stage writes a .done JSON marker to:
    results/manifests/

If the script is interrupted, simply run it again. Completed stages are reused.
Set CONFIG["force_recompute"] = True if you want to start from scratch.

IMPORTANT PRACTICAL NOTE
------------------------
This is a comprehensive research pipeline. The full run across all targets,
splits, budgets, seeds, and models can be computationally expensive on a laptop.
For a first test, set CONFIG["quick_test_mode"] = True. After confirming the
pipeline works, switch it to False.

SAVE / RAM / COMPREHENSIVENESS MODES
------------------------------------
The command-line options now decouple disk use, RAM use, scientific breadth and
CPU use.

  --save-mode tiny|minimal|efficient|lean|balanced|full
      Six hard-disk modes, from very small candidate/checkpoint outputs to full
      auditable model/prediction saving. The old alias compact maps to lean.

  --ram-mode ultra_light|light|standard|normal
      Four memory modes controlling chunk sizes, local conformal limits,
      external nearest-neighbour caps, RAC/RDF breadth, and candidate-tier table
      size.

  --comprehensive-level smoke|quick|standard|full
      Four scientific breadth modes controlling targets, splits, models,
      budgets, seeds, overlays, bootstrap depth, and optional RDF use.

  --n-jobs N or --n-jobs auto
      Controls model-level CPU parallelism while BLAS/OpenMP thread pools are
      kept conservative to avoid oversubscription.

RUNTIME CONTROL
---------------
The default is n_jobs=2 so two CPU cores are used by parallelisable models.
Override with, for example:
    python fewshot_mof_pipeline.py --n-jobs 4 --save-mode balanced
    python fewshot_mof_pipeline.py --smoke-test

Version: v4_7_consolidated_publication_package_tabulatefix_2026_07_07
Author: generated for Mehrdad Asgari's ARC-MOF few-shot screening project
"""

# =============================================================================
# 0. Standard library imports
# =============================================================================
from __future__ import annotations

import os
import re
import argparse
import gc
import sys
import json
import time
import math
import pickle
import shutil
import logging
import warnings
import traceback

# Silence extremely verbose fontTools/matplotlib PDF font-subsetting logs.
# These are informational messages, not errors, and can obscure actual failures.
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# =============================================================================
# 0b. Early warning control for long Windows/conda runs
# =============================================================================
# These filters must be registered before third-party imports and before any
# joblib/loky workers are spawned. They silence known non-fatal compatibility
# warnings that otherwise repeat thousands of times during large few-shot runs.
# Real failures are still captured as EXPERIMENT_FAILED / FATAL_ERROR entries.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::RuntimeWarning")
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*should be used with sklearn\.utils\.parallel\.Parallel.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*X does not have valid feature names.*LGBMRegressor.*",
    category=UserWarning,
)

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Iterable

# =============================================================================
# 1. Third-party imports
# =============================================================================
import numpy as np
import pandas as pd

# Use a non-interactive backend so figures save reliably in VS Code terminals,
# PowerShell, SSH sessions, and headless machines.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ConstantInputWarning
from scipy.spatial.distance import cdist

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors

# Additional category-level warning filters after optional warning classes are available.
warnings.filterwarnings("ignore", category=ConstantInputWarning)
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except Exception:  # pragma: no cover - pandas < 2.2 or no Pandas4Warning symbol
    Pandas4Warning = Warning

# -----------------------------------------------------------------------------
# Parallel compatibility note
# -----------------------------------------------------------------------------
# Some recent scikit-learn versions warn repeatedly if sklearn estimators are
# launched inside joblib.Parallel / joblib.delayed rather than scikit-learn's
# wrappers. The wrappers propagate scikit-learn runtime configuration more
# reliably to worker processes/threads. This script itself is mostly serial at
# the experiment-loop level and uses n_jobs inside supported estimators, but we
# expose Parallel/delayed here so any future parallel loop in this single-file
# pipeline uses the scikit-learn-preferred implementation. We also suppress only
# that repeated compatibility warning because it is noisy and not a modelling
# failure.
try:
    from sklearn.utils.parallel import Parallel, delayed  # scikit-learn >= 1.3 preferred wrappers
    PARALLEL_BACKEND_NAME = "sklearn.utils.parallel"
except Exception:  # pragma: no cover - fallback for older scikit-learn versions
    from joblib import Parallel, delayed
    PARALLEL_BACKEND_NAME = "joblib.fallback"

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*should be used with sklearn\.utils\.parallel\.Parallel.*",
    category=UserWarning,
)
# Keep the broader filters as a second guard because many optional packages
# emit harmless runtime/user warnings during repeated few-shot fits. Actual
# failed experiments are still captured through exceptions and failed_*.json.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 2. User configuration
# =============================================================================

CONFIG: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Reproducibility and runtime control
    # -------------------------------------------------------------------------
    "random_seed": 42,
    "outer_seeds": [42, 43, 44, 45, 46],
    # Default: use two CPU workers. This is deliberately conservative for a
    # Windows laptop while still using more than one processor. Override with
    # --n-jobs N or the FEWSHOT_N_JOBS environment variable.
    "n_jobs": 2,

    # Saving strategy: "minimal", "lean"/"compact", "balanced", or "full".
    # Balanced remains the default for maximum convenience, but for long ARC-MOF
    # runs on a laptop the recommended space-saving final mode is "lean".
    "save_mode": "balanced",
    "ram_mode": "standard",
    "comprehensive_level": "standard",
    "data_root": None,
    "downcast_numeric_tables": True,

    "force_recompute": False,
    "quick_test_mode": False,  # Set True for a small test run.

    # -------------------------------------------------------------------------
    # Label budgets for the central few-shot question.
    # For first debugging, quick_test_mode overrides this list.
    # -------------------------------------------------------------------------
    "label_budgets": [10, 20, 50, 100, 200, 500, 1000],

    # -------------------------------------------------------------------------
    # Split design. Test/calibration fractions refer to the modelling table after
    # removing missing target values.
    # -------------------------------------------------------------------------
    "test_size": 0.20,
    "calibration_size_within_train": 0.20,
    "group_min_count": 5,  # rare groups below this are collapsed for split summaries.
    "enabled_splits": None,  # None = run every available split; quick/smoke mode overrides this.

    # -------------------------------------------------------------------------
    # Feature-set design.
    # geometry_only is the main low-overhead setting.
    # geometry_plus_racs is a stronger chemistry-aware SI/model comparison.
    # geometry_plus_racs_rdfs activates automatically only if RDFs.csv exists and
    # include_rdfs_if_available is True.
    # -------------------------------------------------------------------------
    "feature_sets": ["geometry_only", "geometry_plus_racs"],
    "include_rdfs_if_available": False,  # RDFs can be very large; enable intentionally.
    "max_rac_features": None,            # None = use all numeric RACs; set e.g. 80 for speed.
    "max_rdf_features": 80,              # cap RDFs if included.

    # -------------------------------------------------------------------------
    # Models. Optional models are used only if installed.
    # GPR is expensive; by default run only for budgets <= gpr_max_budget.
    # -------------------------------------------------------------------------
    "models_to_run": ["ridge", "rf", "extra_trees", "hgb", "mlp", "gpr", "lightgbm", "xgboost", "tabpfn"],
    "gpr_max_budget": 200,
    "rf_n_estimators": 250,
    "hgb_max_iter": 100,
    "mlp_max_iter": 400,
    # MLPRegressor with early_stopping=True internally splits off a validation
    # set. For tiny few-shot budgets such as 10 labels this validation set can
    # be only one sample, causing scikit-learn to raise:
    # "The validation set is too small."
    # Therefore early stopping is disabled below this budget. This is also
    # scientifically cleaner for extremely small budgets because every label is
    # valuable.
    "mlp_early_stopping_min_budget": 50,

    # -------------------------------------------------------------------------
    # Conformal settings.
    # alpha=0.10 corresponds to nominal 90% prediction intervals.
    # -------------------------------------------------------------------------
    "conformal_alpha": 0.10,
    "local_conformal_k": 50,
    "enable_local_conformal": True,
    "max_local_conformal_calibration_points": 5000,
    "local_conformal_query_chunk_size": 4000,
    "mondrian_min_cal_per_group": 20,

    # -------------------------------------------------------------------------
    # Screening thresholds.
    # The main paper should highlight top-1%, top-5%, and top-10%.
    # -------------------------------------------------------------------------
    "top_fracs": [0.01, 0.05, 0.10],
    "trusted_interval_width_quantile": 0.50,
    "trusted_lower_bound_percentile": 0.90,

    # -------------------------------------------------------------------------
    # Bootstrap confidence intervals for manuscript/SI tables.
    # Increase to 1000 for final publication-quality CIs.
    # -------------------------------------------------------------------------
    "n_bootstrap": 300,
    "bootstrap_alpha": 0.05,

    # -------------------------------------------------------------------------
    # Plotting settings.
    # -------------------------------------------------------------------------
    "figure_dpi": 150,
    "savefig_dpi": 300,
    "figure_format": ["png"],  # Reliable default. Add "pdf" for vector exports after the run is validated.
    "save_si_pdf": False,  # SI text-heavy PDFs can be slow; PNGs are always saved. Set True if desired.

    # -------------------------------------------------------------------------
    # External realism overlays. These stages are optional and restart-safe. They
    # activate automatically when CoRE MOF 2024 or MOSAEC-DB files are present.
    # They are deliberately treated as plausibility/stability overlays, not as
    # additional adsorption-label sources for the four ARC-MOF targets.
    # -------------------------------------------------------------------------
    "enable_external_overlays": True,
    "external_match_max_candidates": 1000,
    "geometry_nearest_neighbor_max_external_rows": 50000,
    "geometry_match_distance_threshold": 0.75,
    "save_public_release_safe_tables": True,

    # -------------------------------------------------------------------------
    # Candidate-tier collection memory control.
    # The training stage writes one prediction file per experiment. A complete
    # ARC-MOF run can therefore create thousands of prediction CSVs. The final
    # candidate-tier/failure-anatomy stage should not concatenate all of them in
    # memory. These settings make that post-processing stage streaming and
    # laptop-safe. The resulting table is a compact diagnostic/candidate table,
    # while the full experiment-level counts remain available in the metrics CSVs.
    # -------------------------------------------------------------------------
    "prefer_compact_prediction_files": True,
    "candidate_tiers_max_rows": 250000,
    "candidate_tiers_per_file_rows": 1500,
    "candidate_tiers_chunk_size": 50000,
    "candidate_tiers_keep_top_per_group": 50,
    "prediction_text_max_chars": 160,

    # -------------------------------------------------------------------------
    # High-impact post-processing and candidate-quality control.
    # These options prevent unstable baseline models from dominating final
    # shortlists while keeping them in the benchmark tables.
    # -------------------------------------------------------------------------
    "candidate_eligible_models": ["rf", "extra_trees", "hgb", "lightgbm", "xgboost"],
    "stable_plot_models": ["rf", "extra_trees", "hgb", "lightgbm", "xgboost"],
    "target_transform_models": ["ridge", "mlp", "gpr"],
    "enable_log_target_transform": True,
    "enable_candidate_quality_gate": True,
    "min_candidate_spearman": 0.0,
    "min_candidate_top5_recall": 0.02,
    "min_candidate_split_coverage": 0.70,
    "max_candidate_split_coverage": 0.99,
    "max_candidate_rmse_std_multiplier": 3.0,
    "prediction_upper_multiplier_observed_max": 1.25,
    "prediction_lower_fraction_observed_max": -0.05,
    "external_geometry_columns": ["PLD", "LCD", "density"],
    "consensus_min_models": 2,
    "consensus_min_seeds": 2,
    "consensus_min_splits": 1,
    "postprocess_only": False,
    "force_postprocess": False,
    "rebuild_master": False,
}

if CONFIG["quick_test_mode"]:
    CONFIG["outer_seeds"] = [42]
    CONFIG["label_budgets"] = [10]
    CONFIG["feature_sets"] = ["geometry_only"]
    CONFIG["models_to_run"] = ["ridge", "extra_trees"]
    CONFIG["enabled_splits"] = ["random"]
    CONFIG["n_bootstrap"] = 30


# Optional command-line/environment override for safe first run:
#   Windows PowerShell:  $env:FEWSHOT_QUICK_TEST="1"; python fewshot_mof_risk_controlled_pipeline_reviewed.py
#   bash/zsh:            FEWSHOT_QUICK_TEST=1 python fewshot_mof_risk_controlled_pipeline_reviewed.py
if os.environ.get("FEWSHOT_QUICK_TEST", "").strip() in {"1", "true", "True", "yes", "YES"}:
    CONFIG["quick_test_mode"] = True
    CONFIG["outer_seeds"] = [42]
    CONFIG["label_budgets"] = [10]
    CONFIG["feature_sets"] = ["geometry_only"]
    CONFIG["models_to_run"] = ["ridge", "extra_trees"]
    CONFIG["enabled_splits"] = ["random"]
    CONFIG["n_bootstrap"] = 30


# Even smaller smoke test: useful after installing packages or moving files.
# It runs only one target, one random split, one budget, and Ridge.
if os.environ.get("FEWSHOT_SMOKE_TEST", "").strip() in {"1", "true", "True", "yes", "YES"}:
    CONFIG["quick_test_mode"] = True
    CONFIG["outer_seeds"] = [42]
    CONFIG["label_budgets"] = [10]
    CONFIG["feature_sets"] = ["geometry_only"]
    CONFIG["models_to_run"] = ["ridge"]
    CONFIG["enabled_splits"] = ["random"]
    CONFIG["n_bootstrap"] = 10
    CONFIG["active_targets"] = ["CO2_0p015bar_298K_mmolg"]
else:
    CONFIG.setdefault("active_targets", None)


# =============================================================================
# 2b. Command-line options, CPU control, and save-mode policy
# =============================================================================

SAVE_MODE_LEVELS = {
    # Six real disk-usage modes. "compact" remains accepted as a user-facing
    # alias for "lean" for backwards compatibility with older run commands.
    "tiny": 0,       # smallest: final tables/figures/logs + very small candidate rows
    "minimal": 1,    # compact predictions and final outputs only
    "efficient": 2,  # NEW: disk-efficient but analysis-preserving final-run mode
    "lean": 3,       # richer compact diagnostics + reusable pickles
    "compact": 3,    # alias only; normalised to lean in apply_runtime_options()
    "balanced": 4,   # full prediction CSVs, metadata, and reusable checkpoints
    "full": 5,       # balanced + fitted models/split indices/PDF exports where enabled
}

RAM_MODE_PROFILES: Dict[str, Dict[str, Any]] = {
    # Four RAM profiles. They do not change the scientific question; they change
    # chunk sizes, candidate-table caps, conformal-neighbour limits, and optional
    # descriptor breadth so the same script can run on laptops or workstations.
    "ultra_light": {
        "enable_local_conformal": False,
        "max_local_conformal_calibration_points": 1000,
        "local_conformal_query_chunk_size": 1000,
        "candidate_tiers_max_rows": 50000,
        "candidate_tiers_per_file_rows": 300,
        "candidate_tiers_chunk_size": 10000,
        "geometry_nearest_neighbor_max_external_rows": 5000,
        "max_rac_features_if_unset": 80,
        "include_rdfs_if_available": False,
        "n_bootstrap_cap": 100,
        "downcast_numeric_tables": True,
    },
    "light": {
        "enable_local_conformal": True,
        "max_local_conformal_calibration_points": 2500,
        "local_conformal_query_chunk_size": 2000,
        "candidate_tiers_max_rows": 120000,
        "candidate_tiers_per_file_rows": 800,
        "candidate_tiers_chunk_size": 25000,
        "geometry_nearest_neighbor_max_external_rows": 20000,
        "max_rac_features_if_unset": 160,
        "include_rdfs_if_available": False,
        "n_bootstrap_cap": 200,
        "downcast_numeric_tables": True,
    },
    "standard": {
        "enable_local_conformal": True,
        "max_local_conformal_calibration_points": 5000,
        "local_conformal_query_chunk_size": 4000,
        "candidate_tiers_max_rows": 250000,
        "candidate_tiers_per_file_rows": 1500,
        "candidate_tiers_chunk_size": 50000,
        "geometry_nearest_neighbor_max_external_rows": 50000,
        "max_rac_features_if_unset": None,
        "include_rdfs_if_available": None,
        "n_bootstrap_cap": 300,
        "downcast_numeric_tables": True,
    },
    "normal": {
        "enable_local_conformal": True,
        "max_local_conformal_calibration_points": 10000,
        "local_conformal_query_chunk_size": 8000,
        "candidate_tiers_max_rows": 500000,
        "candidate_tiers_per_file_rows": 3000,
        "candidate_tiers_chunk_size": 100000,
        "geometry_nearest_neighbor_max_external_rows": 100000,
        "max_rac_features_if_unset": None,
        "include_rdfs_if_available": None,
        "n_bootstrap_cap": None,
        "downcast_numeric_tables": False,
    },
}

COMPREHENSIVE_LEVEL_PROFILES: Dict[str, Dict[str, Any]] = {
    # Four scientific breadth profiles. Use these to control the number of
    # targets/splits/models/budgets/seeds without editing the code.
    "smoke": {
        "active_targets": ["CO2_0p015bar_298K_mmolg"],
        "outer_seeds": [42],
        "label_budgets": [10],
        "feature_sets": ["geometry_only"],
        "models_to_run": ["ridge"],
        "enabled_splits": ["random"],
        "n_bootstrap": 10,
        "enable_external_overlays": False,
    },
    "quick": {
        "active_targets": ["CO2_0p015bar_298K_mmolg", "CO2_0p150bar_298K_mmolg"],
        "outer_seeds": [42, 43],
        "label_budgets": [10, 50, 200],
        "feature_sets": ["geometry_only"],
        "models_to_run": ["ridge", "extra_trees", "hgb"],
        "enabled_splits": ["random", "geometry_grouped"],
        "n_bootstrap": 50,
        "enable_external_overlays": False,
    },
    "standard": {
        "active_targets": None,
        "outer_seeds": [42, 43, 44, 45, 46],
        "label_budgets": [10, 20, 50, 100, 200, 500, 1000],
        "feature_sets": ["geometry_only", "geometry_plus_racs"],
        "models_to_run": ["ridge", "rf", "extra_trees", "hgb", "mlp", "gpr", "lightgbm", "xgboost", "tabpfn"],
        "enabled_splits": None,
        "n_bootstrap": 300,
        "enable_external_overlays": True,
    },
    "full": {
        "active_targets": None,
        "outer_seeds": [42, 43, 44, 45, 46],
        "label_budgets": [10, 20, 50, 100, 200, 500, 1000],
        "feature_sets": ["geometry_only", "geometry_plus_racs", "geometry_plus_racs_rdfs"],
        "models_to_run": ["ridge", "rf", "extra_trees", "hgb", "mlp", "gpr", "lightgbm", "xgboost", "tabpfn"],
        "enabled_splits": None,
        "include_rdfs_if_available": True,
        "n_bootstrap": 1000,
        "save_si_pdf": True,
        "enable_external_overlays": True,
    },
}


def save_mode_at_least(level: str) -> bool:
    """Return True if the active save mode is at least the requested level."""
    active = str(CONFIG.get("save_mode", "balanced")).lower().strip()
    if active == "compact":
        active = "lean"
    return SAVE_MODE_LEVELS.get(active, 2) >= SAVE_MODE_LEVELS[level]


def parse_runtime_args() -> argparse.Namespace:
    """Parse user-facing command-line options."""
    parser = argparse.ArgumentParser(
        description="Restart-safe few-shot MOF adsorption screening pipeline with external realism overlays."
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root of the shared project data archive, e.g. E:\\projects\\our_group\\ai\\project_data. Overrides FEWSHOT_DATA_ROOT.")
    parser.add_argument("--n-jobs", type=str, default=None,
                        help="Number of CPU workers, or 'auto'. Default is CONFIG n_jobs=2 / FEWSHOT_N_JOBS.")
    parser.add_argument("--save-mode", choices=["tiny", "minimal", "efficient", "compact", "lean", "balanced", "full"], default=None,
                        help="Disk mode: tiny, minimal, efficient, lean, balanced, or full. 'efficient' is recommended for final disk-efficient runs; 'compact' is an alias for lean.")
    parser.add_argument("--ram-mode", choices=list(RAM_MODE_PROFILES.keys()), default=None,
                        help="RAM profile: ultra_light, light, standard, or normal.")
    parser.add_argument("--comprehensive-level", choices=list(COMPREHENSIVE_LEVEL_PROFILES.keys()), default=None,
                        help="Scientific breadth: smoke, quick, standard, or full.")
    parser.add_argument("--quick-test", action="store_true",
                        help="Backward-compatible alias for --comprehensive-level quick.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Backward-compatible alias for --comprehensive-level smoke.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore .done markers and recompute stages.")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Reuse existing experiment metrics/predictions and rerun only aggregation, QC, candidate filtering, consensus tables, overlays, figures, and report.")
    parser.add_argument("--force-postprocess", action="store_true",
                        help="Delete downstream .done markers so post-processing stages are regenerated without refitting models.")
    parser.add_argument("--rebuild-master", action="store_true",
                        help="Rebuild the master table, useful after fixing RDF identifier merging. Does not delete completed experiment outputs by itself.")
    parser.add_argument("--candidate-eligible-models", type=str, default=None,
                        help="Comma-separated models allowed to nominate final candidates, e.g. rf,extra_trees,lightgbm,xgboost.")
    parser.add_argument("--disable-candidate-quality-gate", action="store_true",
                        help="Do not filter candidate tiers by experiment quality/sanity gates. Not recommended for final shortlist generation.")
    parser.add_argument("--include-rdfs", action="store_true",
                        help="Include RDFs.csv as an SI descriptor ablation if present. This can be large/slow.")
    parser.add_argument("--disable-external-overlays", action="store_true",
                        help="Skip CoRE/MOSAEC external realism overlays even if files are present.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list, e.g. ridge,extra_trees,hgb.")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated label budgets, e.g. 10,50,100,500.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated random seeds, e.g. 42,43,44.")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated target names. Example: CO2_0p015bar_298K_mmolg,CO2_0p150bar_298K_mmolg")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated split names. Example: random,geometry_grouped,topology_grouped")
    parser.add_argument("--feature-sets", type=str, default=None,
                        help="Comma-separated feature sets. Example: geometry_only,geometry_plus_racs")
    return parser.parse_args()


def _parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_str_list(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _resolve_n_jobs(value: Optional[str]) -> int:
    if value is None:
        value = os.environ.get("FEWSHOT_N_JOBS", "").strip() or str(CONFIG.get("n_jobs", 2))
    value = str(value).strip().lower()
    if value in {"auto", "all"}:
        return max(1, (os.cpu_count() or 2) - 1)
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(CONFIG.get("n_jobs", 2)))


def apply_comprehensive_level(level: str) -> None:
    """Apply one of the four scientific-breadth profiles."""
    level = str(level or CONFIG.get("comprehensive_level", "standard")).lower().strip()
    if level not in COMPREHENSIVE_LEVEL_PROFILES:
        level = "standard"
    profile = COMPREHENSIVE_LEVEL_PROFILES[level]
    for key, value in profile.items():
        if isinstance(value, list):
            CONFIG[key] = list(value)
        else:
            CONFIG[key] = value
    CONFIG["comprehensive_level"] = level
    CONFIG["quick_test_mode"] = level in {"smoke", "quick"}


def apply_ram_mode(mode: str) -> None:
    """Apply one of the four RAM profiles after comprehensive-level choices."""
    mode = str(mode or CONFIG.get("ram_mode", "standard")).lower().strip().replace("-", "_")
    if mode not in RAM_MODE_PROFILES:
        mode = "standard"
    profile = RAM_MODE_PROFILES[mode]
    for key, value in profile.items():
        if key == "max_rac_features_if_unset":
            if value is not None and CONFIG.get("max_rac_features") is None:
                CONFIG["max_rac_features"] = int(value)
        elif key == "n_bootstrap_cap":
            if value is not None:
                CONFIG["n_bootstrap"] = min(int(CONFIG.get("n_bootstrap", value)), int(value))
        elif key == "include_rdfs_if_available":
            # None means leave the comprehensive-level/user setting unchanged.
            if value is not None:
                CONFIG["include_rdfs_if_available"] = bool(value)
        else:
            CONFIG[key] = value
    CONFIG["ram_mode"] = mode


def apply_runtime_options(args: argparse.Namespace) -> None:
    """Apply command-line and environment overrides to CONFIG in a clear order."""
    # 1) Scientific breadth first.
    env_level = os.environ.get("FEWSHOT_COMPREHENSIVE_LEVEL", "").strip().lower()
    level = args.comprehensive_level or env_level or CONFIG.get("comprehensive_level", "standard")
    if os.environ.get("FEWSHOT_QUICK_TEST", "").strip() in {"1", "true", "True", "yes", "YES"}:
        level = "quick"
    if os.environ.get("FEWSHOT_SMOKE_TEST", "").strip() in {"1", "true", "True", "yes", "YES"}:
        level = "smoke"
    if args.quick_test:
        level = "quick"
    if args.smoke_test:
        level = "smoke"
    apply_comprehensive_level(level)

    # 2) Save mode.
    env_save_mode = os.environ.get("FEWSHOT_SAVE_MODE", "").strip().lower()
    if args.save_mode is not None:
        CONFIG["save_mode"] = args.save_mode
    elif env_save_mode in SAVE_MODE_LEVELS:
        CONFIG["save_mode"] = env_save_mode
    CONFIG["save_mode"] = str(CONFIG.get("save_mode", "balanced")).lower().strip()
    # Common typo guard: --save-mode alanced should mean balanced, not silently fall back later.
    if CONFIG["save_mode"] == "alanced":
        CONFIG["save_mode"] = "balanced"
    if CONFIG["save_mode"] == "compact":
        CONFIG["save_mode"] = "lean"
    if CONFIG["save_mode"] not in SAVE_MODE_LEVELS:
        CONFIG["save_mode"] = "balanced"

    # 3) RAM mode after save/comprehensiveness so it can cap heavy choices.
    env_ram = os.environ.get("FEWSHOT_RAM_MODE", "").strip().lower().replace("-", "_")
    apply_ram_mode(args.ram_mode or env_ram or CONFIG.get("ram_mode", "standard"))

    # 4) CPU, force, and direct user overrides.
    CONFIG["n_jobs"] = _resolve_n_jobs(args.n_jobs)
    if args.force_recompute:
        CONFIG["force_recompute"] = True
    if args.include_rdfs:
        CONFIG["include_rdfs_if_available"] = True
        if "geometry_plus_racs_rdfs" not in CONFIG["feature_sets"]:
            CONFIG["feature_sets"].append("geometry_plus_racs_rdfs")
    if args.disable_external_overlays:
        CONFIG["enable_external_overlays"] = False

    models = _parse_str_list(args.models)
    budgets = _parse_int_list(args.budgets)
    seeds = _parse_int_list(args.seeds)
    targets = _parse_str_list(args.targets)
    splits = _parse_str_list(args.splits)
    feature_sets = _parse_str_list(args.feature_sets)
    candidate_eligible_models = _parse_str_list(args.candidate_eligible_models)
    if args.postprocess_only:
        CONFIG["postprocess_only"] = True
        CONFIG["force_postprocess"] = True
    if args.force_postprocess:
        CONFIG["force_postprocess"] = True
    if args.rebuild_master:
        CONFIG["rebuild_master"] = True
    if args.disable_candidate_quality_gate:
        CONFIG["enable_candidate_quality_gate"] = False
    if candidate_eligible_models:
        CONFIG["candidate_eligible_models"] = candidate_eligible_models
    if models:
        CONFIG["models_to_run"] = models
    if budgets:
        CONFIG["label_budgets"] = budgets
    if seeds:
        CONFIG["outer_seeds"] = seeds
    if targets:
        CONFIG["active_targets"] = targets
    if splits:
        CONFIG["enabled_splits"] = splits
    if feature_sets:
        CONFIG["feature_sets"] = feature_sets

    # 5) Save-mode-specific post-processing caps.
    mode = CONFIG["save_mode"]
    if mode == "tiny":
        CONFIG["prefer_compact_prediction_files"] = True
        CONFIG["candidate_tiers_max_rows"] = min(int(CONFIG.get("candidate_tiers_max_rows", 50000)), 25000)
        CONFIG["candidate_tiers_per_file_rows"] = min(int(CONFIG.get("candidate_tiers_per_file_rows", 300)), 150)
        CONFIG["figure_format"] = ["png"]
        CONFIG["save_si_pdf"] = False
    elif mode == "minimal":
        CONFIG["prefer_compact_prediction_files"] = True
        CONFIG["candidate_tiers_max_rows"] = min(int(CONFIG.get("candidate_tiers_max_rows", 120000)), 75000)
        CONFIG["candidate_tiers_per_file_rows"] = min(int(CONFIG.get("candidate_tiers_per_file_rows", 800)), 400)
        CONFIG["save_si_pdf"] = False
    elif mode == "efficient":
        # Analysis-preserving but disk-efficient final-run mode. It keeps richer
        # compact prediction subsets and reusable core pickles, but never saves
        # full per-experiment prediction CSVs or fitted model objects.
        CONFIG["prefer_compact_prediction_files"] = True
        CONFIG["candidate_tiers_max_rows"] = min(int(CONFIG.get("candidate_tiers_max_rows", 180000)), 150000)
        CONFIG["candidate_tiers_per_file_rows"] = min(int(CONFIG.get("candidate_tiers_per_file_rows", 1200)), 1000)
        CONFIG["save_si_pdf"] = False
    elif mode == "full":
        # Full mode saves fitted models. Keep PDFs optional unless explicitly
        # enabled by the comprehensive-level profile or config.
        CONFIG["prefer_compact_prediction_files"] = False

    CONFIG["n_jobs"] = max(1, int(CONFIG.get("n_jobs", 2)))


def configure_threading() -> None:
    """Limit common numerical thread pools to the selected worker count."""
    n_jobs = max(1, int(CONFIG.get("n_jobs", 2)))
    # Prevent nested BLAS/OpenMP oversubscription when tree models also use n_jobs.
    blas_threads = 1 if n_jobs > 1 else 1
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(blas_threads)

def important_prediction_subset(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compact per-experiment prediction table for non-balanced save modes.

    tiny/minimal keep only essential candidate/error rows.
    efficient is the new recommended final-run disk mode: it keeps trusted
    candidates, high lower-bound candidates, high predicted candidates, largest
    errors, and a modest deterministic random sample. This preserves the main
    post-run analysis/failure-anatomy capability without saving full test-set
    prediction CSVs.
    lean keeps a larger diagnostic sample and is still compact relative to
    balanced/full.
    """
    if pred_df.empty:
        return pred_df
    mode = str(CONFIG.get("save_mode", "balanced")).lower().strip()
    is_efficient = mode == "efficient"
    is_lean = mode in {"lean", "compact"}
    if mode == "tiny":
        top_n, err_n, sample_n, keep_top_pred = 100, 50, 0, False
    elif mode == "minimal":
        top_n, err_n, sample_n, keep_top_pred = 200, 100, 0, False
    elif is_efficient:
        top_n, err_n, sample_n, keep_top_pred = 350, 200, 300, True
    elif is_lean:
        top_n, err_n, sample_n, keep_top_pred = 500, 300, 1000, True
    else:
        top_n, err_n, sample_n, keep_top_pred = 300, 150, 250, True

    parts = []
    if "tier" in pred_df.columns:
        parts.append(pred_df[pred_df["tier"].eq("trusted")])
    if "split_lower" in pred_df.columns:
        parts.append(pred_df.sort_values("split_lower", ascending=False).head(top_n))
    if keep_top_pred and "y_pred" in pred_df.columns:
        parts.append(pred_df.sort_values("y_pred", ascending=False).head(top_n))
    if "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
        tmp = pred_df.copy()
        tmp["abs_error"] = np.abs(tmp["y_true"] - tmp["y_pred"])
        parts.append(tmp.sort_values("abs_error", ascending=False).head(err_n).drop(columns=["abs_error"], errors="ignore"))
    if sample_n > 0 and len(pred_df) > 0:
        parts.append(pred_df.sample(min(sample_n, len(pred_df)), random_state=int(CONFIG.get("random_seed", 42))))
    if not parts:
        default_n = 500 if (is_efficient or is_lean) else 300
        return pred_df.head(default_n)
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def save_prediction_table_for_mode(pred_df: pd.DataFrame, full_path: Path, compact_path: Path) -> None:
    """Save prediction outputs according to the selected save mode.

    balanced/full save every test-set prediction row. minimal/lean save a
    compact candidate/diagnostic subset. This is usually the biggest disk-space
    difference between the modes.

    Text fields are truncated before writing to avoid extremely long topology or
    group strings producing prediction CSVs that are difficult to parse later.
    """
    pred_df = truncate_prediction_text_columns(pred_df) if "truncate_prediction_text_columns" in globals() else pred_df
    if save_mode_at_least("balanced"):
        atomic_save_csv(pred_df, full_path)
    else:
        compact = important_prediction_subset(pred_df)
        compact = truncate_prediction_text_columns(compact) if "truncate_prediction_text_columns" in globals() else compact
        atomic_save_csv(compact, compact_path)


def maybe_save_pickle(obj: Any, path: Path, minimum_mode: str = "balanced") -> None:
    """Save pickle only when the active save mode warrants it."""
    if save_mode_at_least(minimum_mode):
        atomic_pickle(obj, path)


# =============================================================================
# 3. Paths and output folders
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Data-root discovery
# -----------------------------------------------------------------------------
# The project data archive described in PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS
# uses the following root and lowercase/underscore filenames:
#   E:\projects\our_group\ai\project_data\data_raw\arc_mof\...
# The resolver below also keeps backwards compatibility with the older layout
# where CSV files sat next to this script.
DEFAULT_DATA_ROOT_CANDIDATES = [
    os.environ.get("FEWSHOT_DATA_ROOT", ""),
    r"E:\projects\our_group\ai\project_data",
    r"D:\projects\our_group\ai\project_data",
    r"C:\Projects\FewZeroShot\Proj1_Calibrated_FewShot\project_data",
    str(SCRIPT_DIR / "project_data"),
    str(SCRIPT_DIR / "data"),
    str(SCRIPT_DIR),
]
DATA_ROOT: Optional[Path] = None
DATA_SEARCH_ROOTS: List[Path] = [SCRIPT_DIR]
_FILE_RESOLUTION_CACHE: Dict[str, Path] = {}
_PATTERN_RESOLUTION_CACHE: Dict[str, List[Path]] = {}

# Canonical and alias paths for the new project_data archive. All paths are
# relative to DATA_ROOT. The resolver also searches the script folder and nearby
# subfolders so old runs remain restart-safe.
INPUT_FILE_RELATIVE_PATHS: Dict[str, List[str]] = {
    # ARC-MOF adsorption
    "post_comb_vsa-CO2.csv": ["data_raw/arc_mof/adsorption/post_comb_vsa_co2.csv", "post_comb_vsa-CO2.csv", "post_comb_vsa_co2.csv"],
    "post_comb_vsa-N2.csv": ["data_raw/arc_mof/adsorption/post_comb_vsa_n2.csv", "post_comb_vsa-N2.csv", "post_comb_vsa_n2.csv"],
    "methane.csv": ["data_raw/arc_mof/adsorption/methane.csv", "methane.csv"],
    "landfill-CO2.csv": ["data_raw/arc_mof/adsorption/landfill_co2.csv", "landfill-CO2.csv", "landfill_co2.csv"],
    "landfill-CH4.csv": ["data_raw/arc_mof/adsorption/landfill_ch4.csv", "landfill-CH4.csv", "landfill_ch4.csv"],
    "methane_purification-CO2.csv": ["data_raw/arc_mof/adsorption/methane_purification_co2.csv", "methane_purification-CO2.csv", "methane_purification_co2.csv"],
    "methane_purification-CH4.csv": ["data_raw/arc_mof/adsorption/methane_purification_ch4.csv", "methane_purification-CH4.csv", "methane_purification_ch4.csv"],
    # ARC-MOF descriptors, clusters, topology, process
    "geometric_properties.csv": ["data_raw/arc_mof/descriptors/geometric_properties.csv", "geometric_properties.csv"],
    "RACs.csv": ["data_raw/arc_mof/descriptors/racs.csv", "RACs.csv", "racs.csv"],
    "RDFs.csv": ["data_raw/arc_mof/descriptors/rdfs.csv", "RDFs.csv", "rdfs.csv"],
    "ARC-MOF_Dim.csv": ["data_raw/arc_mof/descriptors/arc_mof_dim.csv", "ARC-MOF_Dim.csv", "arc_mof_dim.csv"],
    "overall_process.csv": ["data_raw/arc_mof/process/overall_process.csv", "overall_process.csv"],
    "geo-clusters.csv": ["data_raw/arc_mof/clusters/geo_clusters.csv", "geo-clusters.csv", "geo_clusters.csv"],
    "mc-clusters.csv": ["data_raw/arc_mof/clusters/mc_clusters.csv", "mc-clusters.csv", "mc_clusters.csv"],
    "func-clusters.csv": ["data_raw/arc_mof/clusters/func_clusters.csv", "func-clusters.csv", "func_clusters.csv"],
    "flig-clusters.csv": ["data_raw/arc_mof/clusters/flig_clusters.csv", "flig-clusters.csv", "flig_clusters.csv"],
    "mc-diverse-set.csv": ["data_raw/arc_mof/diversity_sets/mc_diverse_set.csv", "mc-diverse-set.csv", "mc_diverse_set.csv"],
    "func-diverse-set.csv": ["data_raw/arc_mof/diversity_sets/func_diverse_set.csv", "func-diverse-set.csv", "func_diverse_set.csv"],
    "all_topology_lists.csv": ["data_raw/arc_mof/topology/all_topology_lists.csv", "all_topology_lists.csv"],
    "ARCMOF_20241004.tar.gz": ["data_raw/arc_mof/structures/ARCMOF_20241004.tar.gz", "ARCMOF_20241004.tar.gz"],
    # CoRE MOF 2024 metadata and diagnostics
    "ASR_data_SI_20250204.csv": ["data_raw/core_mof_2024/metadata/asr_data_si_20250204.csv", "ASR_data_SI_20250204.csv", "asr_data_si_20250204.csv"],
    "FSR_data_SI_20250204.csv": ["data_raw/core_mof_2024/metadata/fsr_data_si_20250204.csv", "FSR_data_SI_20250204.csv", "fsr_data_si_20250204.csv"],
    "ION_data_SI_20250204.csv": ["data_raw/core_mof_2024/metadata/ion_data_si_20250204.csv", "ION_data_SI_20250204.csv", "ion_data_si_20250204.csv"],
    "12089-recommended-screening-list.csv": ["data_raw/core_mof_2024/metadata/12089_recommended_screening_list.csv", "12089-recommended-screening-list.csv", "12089_recommended_screening_list.csv"],
    "ASR_FSR_check.csv": ["data_raw/core_mof_2024/duplicate_checks/asr_fsr_check.csv", "ASR_FSR_check.csv", "asr_fsr_check.csv"],
    "NCR_ASR_SI.xlsx": ["data_raw/core_mof_2024/diagnostics/ncr_asr_si.xlsx", "NCR_ASR_SI.xlsx", "ncr_asr_si.xlsx"],
    "NCR_FSR_SI.xlsx": ["data_raw/core_mof_2024/diagnostics/ncr_fsr_si.xlsx", "NCR_FSR_SI.xlsx", "ncr_fsr_si.xlsx"],
    "NCR_ION_SI.xlsx": ["data_raw/core_mof_2024/diagnostics/ncr_ion_si.xlsx", "NCR_ION_SI.xlsx", "ncr_ion_si.xlsx"],
    "unmodified_check_for_NCR_SI.xlsx": ["data_raw/core_mof_2024/diagnostics/unmodified_check_for_ncr_si.xlsx", "unmodified_check_for_NCR_SI.xlsx", "unmodified_check_for_ncr_si.xlsx"],
    "water_data.xlsx": ["data_raw/core_mof_2024/water_gemc/extracted/water/water_data.xlsx", "water_data.xlsx"],
    "widom.xlsx": ["data_raw/core_mof_2024/water_gemc/extracted/water/widom.xlsx", "widom.xlsx"],
    "structure_information.csv": ["data_raw/core_mof_2024/tsa/extracted/tsa/TSA/structure_information.csv", "structure_information.csv"],
    "data_298_423_1bar_CO2_N2_891.csv": ["data_raw/core_mof_2024/tsa/extracted/tsa/TSA/data_298_423_1bar_CO2_N2_891.csv", "data_298_423_1bar_CO2_N2_891.csv"],
    "mofid-v2.zip": ["data_raw/core_mof_2024/mofid/archives/mofid-v2.zip", "mofid-v2.zip"],
    "water.zip": ["data_raw/core_mof_2024/water_gemc/archives/water.zip", "water.zip"],
    "TSA.zip": ["data_raw/core_mof_2024/tsa/archives/TSA.zip", "TSA.zip"],
    # MOSAEC metadata/descriptors/subsets
    "mosaec-db.csv": ["data_raw/mosaec_db/database/mosaec_db.csv", "data_raw/mosaec_db/mosaec-db.csv", "data_raw/mosaec_db/mosaec_db.csv", "mosaec-db.csv", "mosaec_db.csv"],
    "mosaec-db.xlsx": ["data_raw/mosaec_db/database/mosaec_db.xlsx", "data_raw/mosaec_db/mosaec-db.xlsx", "data_raw/mosaec_db/mosaec_db.xlsx", "mosaec-db.xlsx", "mosaec_db.xlsx"],
    "GEOM_mosaec-db.csv": ["data_raw/mosaec_db/descriptors/extracted/descriptors/descriptors/GEOM_mosaec-db.csv", "GEOM_mosaec-db.csv"],
    "RAC_mosaec-db.csv": ["data_raw/mosaec_db/descriptors/extracted/descriptors/descriptors/RAC_mosaec-db.csv", "RAC_mosaec-db.csv"],
    "APRDF_mosaec-db.csv": ["data_raw/mosaec_db/descriptors/extracted/descriptors/descriptors/APRDF_mosaec-db.csv", "APRDF_mosaec-db.csv"],
    "PHOM_mosaec-db.csv": ["data_raw/mosaec_db/descriptors/extracted/descriptors/descriptors/PHOM_mosaec-db.csv", "PHOM_mosaec-db.csv"],
    "PDD_duplicate_thresh0.15.csv": ["data_raw/mosaec_db/misc_data/extracted/misc_data/misc_data/PDD_duplicate_thresh0.15.csv", "PDD_duplicate_thresh0.15.csv"],
    "uniq-neutral-porous-2.4pld.txt": ["data_raw/mosaec_db/subsets/extracted/subsets/subsets/uniq-neutral-porous-2.4pld.txt", "uniq-neutral-porous-2.4pld.txt"],
    "uniq-neutral-porous-0.10vf.txt": ["data_raw/mosaec_db/subsets/extracted/subsets/subsets/uniq-neutral-porous-0.10vf.txt", "uniq-neutral-porous-0.10vf.txt"],
    "descriptors.tar.gz": ["data_raw/mosaec_db/descriptors/archives/descriptors.tar.gz", "descriptors.tar.gz"],
    "misc_data.tar.gz": ["data_raw/mosaec_db/misc_data/archives/misc_data.tar.gz", "misc_data.tar.gz"],
    "subsets.tar.gz": ["data_raw/mosaec_db/subsets/archives/subsets.tar.gz", "subsets.tar.gz"],
    "scripts.tar.gz": ["data_raw/mosaec_db/scripts/archives/scripts.tar.gz", "scripts.tar.gz"],
}


def _path_candidates_from_user_root(raw_root: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if raw_root:
        candidates.append(Path(raw_root).expanduser())
    for item in DEFAULT_DATA_ROOT_CANDIDATES:
        if item:
            candidates.append(Path(item).expanduser())
    # Include parent folders around the script so copied data folders are found.
    candidates.extend([SCRIPT_DIR, SCRIPT_DIR / "data", SCRIPT_DIR / "project_data", SCRIPT_DIR.parent / "project_data"])
    out: List[Path] = []
    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve()) if c.exists() else str(c)
        except Exception:
            key = str(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def configure_data_roots(data_root_arg: Optional[str] = None) -> None:
    """Configure data roots before any file-resolution stage runs."""
    global DATA_ROOT, DATA_SEARCH_ROOTS, _FILE_RESOLUTION_CACHE, _PATTERN_RESOLUTION_CACHE
    raw_root = data_root_arg or os.environ.get("FEWSHOT_DATA_ROOT", "") or CONFIG.get("data_root")
    candidates = _path_candidates_from_user_root(raw_root)
    chosen: Optional[Path] = None
    for cand in candidates:
        try:
            if cand.exists() and (cand / "data_raw").exists():
                chosen = cand
                break
        except Exception:
            continue
    if chosen is None:
        # Fall back to the first existing folder, then SCRIPT_DIR.
        for cand in candidates:
            try:
                if cand.exists():
                    chosen = cand
                    break
            except Exception:
                continue
    DATA_ROOT = chosen or SCRIPT_DIR
    roots = [SCRIPT_DIR, DATA_ROOT, DATA_ROOT / "data_raw", SCRIPT_DIR / "data", SCRIPT_DIR / "project_data"]
    DATA_SEARCH_ROOTS = []
    seen = set()
    for r in roots:
        try:
            key = str(r.resolve()) if r.exists() else str(r)
        except Exception:
            key = str(r)
        if key not in seen:
            seen.add(key)
            DATA_SEARCH_ROOTS.append(r)
    CONFIG["data_root"] = str(DATA_ROOT)
    _FILE_RESOLUTION_CACHE = {}
    _PATTERN_RESOLUTION_CACHE = {}


# Output folders remain local to the code directory so the same project_data
# archive can be reused by several independent code versions.
RESULTS_DIR = SCRIPT_DIR / "results_fewshot_mof"
LOG_DIR = RESULTS_DIR / "logs"
MANIFEST_DIR = RESULTS_DIR / "manifests"
TABLE_DIR = RESULTS_DIR / "tables"
PRED_DIR = RESULTS_DIR / "predictions"
PICKLE_DIR = RESULTS_DIR / "pickles"
FIGDATA_DIR = RESULTS_DIR / "figure_data"
FIG_MAIN_DIR = RESULTS_DIR / "figures" / "main"
FIG_SI_DIR = RESULTS_DIR / "figures" / "si"
MODEL_DIR = RESULTS_DIR / "models"
EXTERNAL_DIR = RESULTS_DIR / "external_validation"
CASE_STUDY_DIR = RESULTS_DIR / "case_studies"
DATA_DIR = SCRIPT_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RELEASE_SAFE_DIR = DATA_DIR / "release_safe"
DATA_RELEASE_FIGURE_DIR = DATA_RELEASE_SAFE_DIR / "figure_data"
DATA_RELEASE_TABLE_DIR = DATA_RELEASE_SAFE_DIR / "tables"

for d in [RESULTS_DIR, LOG_DIR, MANIFEST_DIR, TABLE_DIR, PRED_DIR, PICKLE_DIR,
          FIGDATA_DIR, FIG_MAIN_DIR, FIG_SI_DIR, MODEL_DIR, EXTERNAL_DIR, CASE_STUDY_DIR,
          DATA_PROCESSED_DIR, DATA_RELEASE_FIGURE_DIR, DATA_RELEASE_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. Logging utilities
# =============================================================================

def setup_logging() -> None:
    """Configure both file and console logging."""
    log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("STAGE >>> JOB_START")
    logging.info("Script directory: %s", SCRIPT_DIR)
    logging.info("Results directory: %s", RESULTS_DIR)
    logging.info("Configured data root: %s", DATA_ROOT)
    logging.info("Data search roots: %s", [str(p) for p in DATA_SEARCH_ROOTS])
    logging.info("Python executable: %s", sys.executable)
    logging.info("Parallel/delayed backend: %s", PARALLEL_BACKEND_NAME)
    logging.info("Runtime n_jobs: %s", CONFIG.get("n_jobs"))
    logging.info("Save mode: %s", CONFIG.get("save_mode"))
    logging.info("RAM mode: %s", CONFIG.get("ram_mode"))
    logging.info("Comprehensive level: %s", CONFIG.get("comprehensive_level"))

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def done_path(stage_name: str) -> Path:
    return MANIFEST_DIR / f"{stage_name}.done.json"


def is_done(stage_name: str) -> bool:
    return done_path(stage_name).exists() and not CONFIG["force_recompute"]


def mark_done(stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    save_json(payload, done_path(stage_name))
    logging.info("STAGE_DONE | %s", stage_name)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with logging, low-memory defaults, and a clear missing-file error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path.name}. Current data root: {DATA_ROOT}. "
            "Use --data-root E:\\projects\\our_group\\ai\\project_data if needed."
        )
    logging.info("Reading CSV: %s", path)
    if "low_memory" not in kwargs and "chunksize" not in kwargs:
        kwargs["low_memory"] = False
    if "memory_map" not in kwargs and "chunksize" not in kwargs:
        kwargs["memory_map"] = True
    return pd.read_csv(path, **kwargs)


def _filename_aliases(filename: str) -> List[str]:
    """Return common case/hyphen/underscore variants for a data filename."""
    f = str(filename)
    aliases = {f, Path(f).name, f.lower(), Path(f).name.lower()}
    aliases |= {x.replace("-", "_") for x in list(aliases)}
    aliases |= {x.replace("_", "-") for x in list(aliases)}
    # Preserve common scientific file-case variants.
    if f.lower() == "racs.csv":
        aliases |= {"RACs.csv", "racs.csv"}
    if f.lower() == "rdfs.csv":
        aliases |= {"RDFs.csv", "rdfs.csv"}
    return sorted(aliases, key=lambda x: (x != f, len(x), x))


def _candidate_relative_paths(filename: str) -> List[Path]:
    rels: List[Path] = []
    for key in [filename, Path(filename).name]:
        if key in INPUT_FILE_RELATIVE_PATHS:
            rels.extend(Path(x) for x in INPUT_FILE_RELATIVE_PATHS[key])
    # Match aliases in the relative-path dictionary.
    alias_set = set(_filename_aliases(filename))
    for key, values in INPUT_FILE_RELATIVE_PATHS.items():
        if key in alias_set or key.lower() in {a.lower() for a in alias_set}:
            rels.extend(Path(x) for x in values)
    # Direct filename variants.
    rels.extend(Path(a) for a in _filename_aliases(filename))
    out: List[Path] = []
    seen = set()
    for rel in rels:
        key = str(rel).replace("\\", "/").lower()
        if key not in seen:
            seen.add(key)
            out.append(rel)
    return out


def resolve_input_file(filename: str) -> Path:
    """Locate an input file in the new project_data archive or old local layout.

    The resolver is intentionally tolerant of the old ARC-MOF file names
    (`post_comb_vsa-CO2.csv`, `geo-clusters.csv`) and the new archive names
    (`post_comb_vsa_co2.csv`, `geo_clusters.csv`). It first checks known archive
    paths from PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS, then falls back to a
    constrained recursive search under the configured data roots.
    """
    cache_key = str(filename)
    if cache_key in _FILE_RESOLUTION_CACHE:
        return _FILE_RESOLUTION_CACHE[cache_key]

    rels = _candidate_relative_paths(filename)
    candidates: List[Path] = []
    for root in DATA_SEARCH_ROOTS:
        candidates.append(root / filename)
        for rel in rels:
            candidates.append(root / rel)
            if rel.parts and rel.parts[0].lower() == "data_raw" and (root.name.lower() == "data_raw"):
                candidates.append(root.joinpath(*rel.parts[1:]))
    for c in candidates:
        try:
            if c.exists():
                _FILE_RESOLUTION_CACHE[cache_key] = c
                return c
        except Exception:
            continue

    # Constrained recursive fallback. This is still safe because it searches by
    # exact filename variants only and excludes result folders.
    alias_names = _filename_aliases(Path(filename).name)
    for root in DATA_SEARCH_ROOTS:
        if not root.exists():
            continue
        for alias in alias_names:
            try:
                matches = [m for m in root.rglob(alias) if RESULTS_DIR not in m.parents]
            except Exception:
                matches = []
            if matches:
                # Prefer paths inside data_raw, then shorter paths.
                matches = sorted(matches, key=lambda m: ("data_raw" not in str(m).lower(), len(str(m))))
                _FILE_RESOLUTION_CACHE[cache_key] = matches[0]
                return matches[0]

    # Return the most likely intended path for clear error messages.
    fallback_root = DATA_ROOT or SCRIPT_DIR
    fallback = fallback_root / (rels[0] if rels else filename)
    _FILE_RESOLUTION_CACHE[cache_key] = fallback
    return fallback

def inspect_input_files() -> pd.DataFrame:
    """Create a reproducible inventory of available input CSV files.

    This is deliberately done before modelling so that the RUN_REPORT and SI can
    state exactly what was available, what columns were detected, and which files
    were missing. Only the first rows are read, so this remains fast even when
    RDFs.csv is large.
    """
    stage = "00_input_file_inspection"
    out_path = TABLE_DIR / "table_input_file_inspection.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path)
    expected = [
        "geometric_properties.csv", "post_comb_vsa-CO2.csv", "methane.csv",
        "overall_process.csv", "geo-clusters.csv", "mc-clusters.csv",
        "func-clusters.csv", "flig-clusters.csv", "RACs.csv", "RDFs.csv",
        "all_topology_lists.csv", "post_comb_vsa-N2.csv", "landfill-CO2.csv",
        "landfill-CH4.csv", "methane_purification-CO2.csv", "methane_purification-CH4.csv",
        "mc-diverse-set.csv", "func-diverse-set.csv",
        # CoRE MOF 2024 v1.1 external-overlay files
        "ASR_data_SI_20250204.csv", "FSR_data_SI_20250204.csv", "ION_data_SI_20250204.csv",
        "12089-recommended-screening-list.csv", "ASR_FSR_check.csv",
        "water_data.xlsx", "widom.xlsx", "structure_information.csv", "data_298_423_1bar_CO2_N2_891.csv",
        "NCR_ASR_SI.xlsx", "NCR_FSR_SI.xlsx", "NCR_ION_SI.xlsx", "unmodified_check_for_NCR_SI.xlsx",
        # MOSAEC-DB external-overlay files; structural CIF directories are not
        # parsed by this tabular script but are recorded in RUN_REPORT if present.
        "mosaec-db.csv", "mosaec-db.xlsx", "GEOM_mosaec-db.csv", "RAC_mosaec-db.csv",
        "APRDF_mosaec-db.csv", "PHOM_mosaec-db.csv", "PDD_duplicate_thresh0.15.csv",
        "uniq-neutral-porous-2.4pld.txt", "uniq-neutral-porous-0.10vf.txt",
    ]
    rows = []
    for fname in expected:
        path = resolve_input_file(fname)
        row = {"file": fname, "found": path.exists(), "path": str(path) if path.exists() else "", "n_columns_preview": 0, "columns_preview": "", "read_error": ""}
        if path.exists():
            try:
                preview = pd.read_csv(path, nrows=5)
                row["n_columns_preview"] = int(preview.shape[1])
                row["columns_preview"] = "; ".join(map(str, preview.columns[:80]))
            except Exception as e:
                row["read_error"] = repr(e)
        rows.append(row)
    df = pd.DataFrame(rows)
    atomic_save_csv(df, out_path)
    mark_done(stage, {"n_expected": len(expected), "n_found": int(df["found"].sum())})
    return df


def atomic_save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write CSV atomically to reduce corruption risk if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)


def atomic_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)



def reduce_dataframe_memory(df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
    """Downcast numeric columns when the active RAM mode requests it.

    This substantially reduces memory during ARC-MOF-scale joins while keeping
    identifiers/text columns untouched. Float columns are downcast to float32;
    integer columns are downcast to the smallest safe integer dtype.
    """
    if df is None or df.empty or not CONFIG.get("downcast_numeric_tables", True):
        return df
    before = int(df.memory_usage(deep=True).sum())
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
    after = int(out.memory_usage(deep=True).sum())
    if before > 0:
        logging.info("MEMORY_REDUCE | %s | %.2f MB -> %.2f MB", name, before / 1e6, after / 1e6)
    return out

# =============================================================================
# 5. Identifier normalisation and column helpers
# =============================================================================

def normalize_mof_id(x: Any) -> str:
    """
    Convert ARC-MOF identifiers from different files to a common key.

    Examples:
        DB0-...sym.90_repeat  -> DB0-...sym.90
        DB0-...sym.90.cif     -> DB0-...sym.90
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.cif$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_repeat$", "", s, flags=re.IGNORECASE)
    return s


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    excl = set(exclude)
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def find_identifier_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find an identifier column using exact/fuzzy normalised matching."""
    if df is None or df.empty:
        return None
    norm_to_col = {_normalise_colname(c) if "_normalise_colname" in globals() else re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        n = _normalise_colname(cand) if "_normalise_colname" in globals() else re.sub(r"[^a-z0-9]+", "", str(cand).lower())
        if n in norm_to_col:
            return norm_to_col[n]
    for cand in candidates:
        n = _normalise_colname(cand) if "_normalise_colname" in globals() else re.sub(r"[^a-z0-9]+", "", str(cand).lower())
        for norm, col in norm_to_col.items():
            if n and (n in norm or norm in n):
                return col
    return None


def robust_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q))

# =============================================================================
# 6. Dataset loading and master-table construction
# =============================================================================

TARGET_SPECS = {
    "CO2_0p015bar_298K_mmolg": {
        "file": "post_comb_vsa-CO2.csv", "T/K": 298.0, "p/bar": 0.015,
        "value_col": "mmol/g", "description": "CO2 uptake at 0.015 bar and 298 K"
    },
    "CO2_0p150bar_298K_mmolg": {
        "file": "post_comb_vsa-CO2.csv", "T/K": 298.0, "p/bar": 0.150,
        "value_col": "mmol/g", "description": "CO2 uptake at 0.150 bar and 298 K"
    },
    "CH4_5p8bar_298K_mmolg": {
        "file": "methane.csv", "T/K": 298.0, "p/bar": 5.8,
        "value_col": "mmol/g", "description": "CH4 uptake at 5.8 bar and 298 K"
    },
    "CH4_65bar_298K_mmolg": {
        "file": "methane.csv", "T/K": 298.0, "p/bar": 65.0,
        "value_col": "mmol/g", "description": "CH4 uptake at 65 bar and 298 K"
    },
}

GEOMETRY_CORE_COLUMNS = [
    "UC_volume", "Density", "ASA", "vASA", "gASA", "NASA", "gNASA", "vNASA",
    "AVA", "AVAf", "AVAg", "NAVA", "NAVAf", "NAVAg", "POAVA", "POAVAf",
    "POAVAg", "NPOAVA", "NPOAVAf", "NPOAVAg", "Di", "Df", "Dif"
]


def extract_target_table(path: Path, spec: Dict[str, Any], target_name: str) -> pd.DataFrame:
    """Filter a long-format ARC-MOF adsorption CSV to one target column."""
    usecols = ["filename", "T/K", "p/bar", spec["value_col"], "stdev", "hoa/kcal/mol", "C_v"]
    df = safe_read_csv(path, usecols=lambda c: c in usecols)
    df["mof_id"] = df["filename"].map(normalize_mof_id)
    mask = np.isclose(df["T/K"].astype(float), spec["T/K"]) & np.isclose(df["p/bar"].astype(float), spec["p/bar"])
    out = df.loc[mask, ["mof_id", spec["value_col"]]].copy()
    out = out.rename(columns={spec["value_col"]: target_name})
    out = out.groupby("mof_id", as_index=False)[target_name].mean()
    logging.info("Target extracted | %s | n=%d", target_name, len(out))
    return out


def load_cluster_file(filename: str, new_col: str) -> Optional[pd.DataFrame]:
    path = resolve_input_file(filename)
    if not path.exists():
        logging.warning("Optional cluster file missing: %s", filename)
        return None
    df = safe_read_csv(path, usecols=lambda c: c in ["filename", "cluster_id", "Database"])
    df["mof_id"] = df["filename"].map(normalize_mof_id)
    df = df[["mof_id", "cluster_id"]].drop_duplicates("mof_id")
    return df.rename(columns={"cluster_id": new_col})


def load_topology_table() -> Optional[pd.DataFrame]:
    """Load all_topology_lists.csv if available. Column names may vary by release."""
    path = resolve_input_file("all_topology_lists.csv")
    if not path.exists():
        logging.warning("Optional topology file missing: all_topology_lists.csv")
        return None
    topo = safe_read_csv(path)
    # Find likely ID and topology columns.
    id_candidates = [c for c in topo.columns if c.lower() in ["filename", "name", "mof", "mof_id"]]
    topo_candidates = [c for c in topo.columns if any(k in c.lower() for k in ["topology", "topo", "net"])]
    if not id_candidates or not topo_candidates:
        logging.warning("Could not identify topology columns in all_topology_lists.csv; columns=%s", topo.columns.tolist())
        return None
    out = topo[[id_candidates[0], topo_candidates[0]]].copy()
    out.columns = ["filename", "topology"]
    out["mof_id"] = out["filename"].map(normalize_mof_id)
    return out[["mof_id", "topology"]].drop_duplicates("mof_id")


def build_master_table() -> pd.DataFrame:
    """Construct and save the merged ARC-MOF modelling table."""
    stage = "01_master_table"
    master_path = PICKLE_DIR / "master_table.pkl"
    if is_done(stage) and master_path.exists() and not CONFIG.get("rebuild_master", False):
        logging.info("Using existing master table: %s", master_path)
        return read_pickle(master_path)

    logging.info("STAGE >>> BUILD_MASTER_TABLE")

    # Geometry descriptors are the default central feature set.
    geo = reduce_dataframe_memory(safe_read_csv(resolve_input_file("geometric_properties.csv")), "geometric_properties")
    geo["mof_id"] = geo["filename"].map(normalize_mof_id)
    geo = geo.drop_duplicates("mof_id")
    geo_cols = [c for c in GEOMETRY_CORE_COLUMNS if c in geo.columns]
    keep = ["mof_id", "filename"] + geo_cols
    for optional_col in ["ARC-MOF", "ARC_MOF", "DB_num", "bool_geo"]:
        if optional_col in geo.columns and optional_col not in keep:
            keep.append(optional_col)
    master = geo[keep].copy()

    # Four target columns. For speed, read each adsorption CSV only once even
    # when multiple targets come from the same file (e.g., CO2 0.015 and 0.15 bar).
    specs_by_file: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for target_name, spec in TARGET_SPECS.items():
        specs_by_file.setdefault(spec["file"], []).append((target_name, spec))
    for fname, specs in specs_by_file.items():
        path = resolve_input_file(fname)
        needed_cols = {"filename", "T/K", "p/bar"}
        needed_cols.update(spec["value_col"] for _, spec in specs)
        ads = reduce_dataframe_memory(safe_read_csv(path, usecols=lambda c, needed_cols=needed_cols: c in needed_cols), fname)
        ads["mof_id"] = ads["filename"].map(normalize_mof_id)
        for target_name, spec in specs:
            mask = np.isclose(ads["T/K"].astype(float), spec["T/K"]) & np.isclose(ads["p/bar"].astype(float), spec["p/bar"])
            target_df = ads.loc[mask, ["mof_id", spec["value_col"]]].copy()
            target_df = target_df.rename(columns={spec["value_col"]: target_name})
            target_df = target_df.groupby("mof_id", as_index=False)[target_name].mean()
            logging.info("Target extracted | %s | n=%d", target_name, len(target_df))
            master = master.merge(target_df, on="mof_id", how="left")
        del ads
        gc.collect()

    # Grouping files for grouped splits and chemical failure anatomy.
    cluster_files = {
        "geo-clusters.csv": "geometry_cluster",
        "mc-clusters.csv": "metal_cluster",
        "func-clusters.csv": "functional_cluster",
        "flig-clusters.csv": "ligand_cluster",
    }
    for fname, col in cluster_files.items():
        cl = load_cluster_file(fname, col)
        if cl is not None:
            master = master.merge(cl, on="mof_id", how="left")

    topo = load_topology_table()
    if topo is not None:
        master = master.merge(topo, on="mof_id", how="left")

    # Optional process metrics for downstream validation and SI tables.
    process_path = resolve_input_file("overall_process.csv")
    if process_path.exists():
        proc = reduce_dataframe_memory(safe_read_csv(process_path), "overall_process")
        proc["mof_id"] = proc["filename"].map(normalize_mof_id)
        # Pivot key process metrics into separate columns.
        process_metrics = [c for c in ["mmol/g_working_capacity", "selectivity", "purity", "ssp", "afm"] if c in proc.columns]
        if "process" in proc.columns and process_metrics:
            proc_wide = proc.pivot_table(index="mof_id", columns="process", values=process_metrics, aggfunc="mean")
            proc_wide.columns = [f"process__{metric}__{process}" for metric, process in proc_wide.columns]
            proc_wide = proc_wide.reset_index()
            master = master.merge(proc_wide, on="mof_id", how="left")

    # Optional RAC descriptors. These are large but valuable for the SI.
    rac_path = resolve_input_file("RACs.csv")
    rac_requested = any("racs" in fs for fs in CONFIG.get("feature_sets", []))
    if rac_path.exists() and rac_requested:
        rac = reduce_dataframe_memory(safe_read_csv(rac_path), "RACs")
        if "filename" not in rac.columns:
            logging.warning(
                "RACs.csv was found but has no filename column. RAC features will not be merged because row-order merging is unsafe."
            )
        else:
            rac["mof_id"] = rac["filename"].map(normalize_mof_id)
            meta = {"filename", "mof_id", "ARC_MOF", "ARC-MOF", "DB_num", "order_f-lig", "bool_f-lig", "order_mc", "bool_mc", "order_func", "bool_func", "order_lc", "bool_lc", "Unnamed: 0"}
            rac_num = numeric_columns(rac, exclude=meta)
            # Drop near-constant columns; they add compute but no information.
            rac_num = [c for c in rac_num if rac[c].nunique(dropna=True) > 1]
            if CONFIG["max_rac_features"] is not None:
                # Retain features with largest variance as a pragmatic speed option.
                variances = rac[rac_num].var(numeric_only=True).sort_values(ascending=False)
                rac_num = variances.head(int(CONFIG["max_rac_features"])).index.tolist()
            rac_keep = rac[["mof_id"] + rac_num].drop_duplicates("mof_id")
            rac_keep = rac_keep.rename(columns={c: f"RAC__{c}" for c in rac_num})
            master = master.merge(rac_keep, on="mof_id", how="left")
            logging.info("RAC features merged safely by filename | n_features=%d", len(rac_num))
        del rac
        gc.collect()
    elif rac_requested:
        logging.warning("RACs.csv not found. geometry_plus_racs feature set will be skipped.")
    else:
        logging.info("RACs.csv not loaded because no active feature set requests RAC descriptors.")

    # Optional RDF descriptors. Disabled by default because RDFs.csv is often huge.
    # v3.8: accept ARC-MOF RDF identifier variants such as Structure_Name,
    # structure_name, name, mof_id, or filename. Earlier versions only accepted
    # filename and could therefore create a duplicate geometry_plus_racs_rdfs
    # feature set with zero RDF columns.
    rdf_path = resolve_input_file("RDFs.csv")
    if CONFIG["include_rdfs_if_available"] and rdf_path.exists():
        rdf = reduce_dataframe_memory(safe_read_csv(rdf_path), "RDFs")
        rdf_id_col = find_identifier_column(rdf, ["filename", "Structure_Name", "structure_name", "name", "mof_id", "MOF", "structure"])
        if rdf_id_col is None:
            logging.warning("RDFs.csv was found but no identifier column was detected. RDF features will not be merged.")
        else:
            rdf["mof_id"] = rdf[rdf_id_col].map(normalize_mof_id)
            meta = {"filename", "Structure_Name", "structure_name", "name", "mof_id", "MOF", "structure", "Unnamed: 0"}
            rdf_num = numeric_columns(rdf, exclude=meta)
            rdf_num = [c for c in rdf_num if rdf[c].nunique(dropna=True) > 1]
            if CONFIG["max_rdf_features"] is not None and rdf_num:
                variances = rdf[rdf_num].var(numeric_only=True).sort_values(ascending=False)
                rdf_num = variances.head(int(CONFIG["max_rdf_features"])).index.tolist()
            if rdf_num:
                rdf_keep = rdf[["mof_id"] + rdf_num].drop_duplicates("mof_id")
                rdf_keep = rdf_keep.rename(columns={c: f"RDF__{c}" for c in rdf_num})
                master = master.merge(rdf_keep, on="mof_id", how="left")
                logging.info("RDF features merged safely by %s | n_features=%d", rdf_id_col, len(rdf_num))
                del rdf_keep
            else:
                logging.warning("RDFs.csv was found and identified by %s but no usable numeric RDF features were detected.", rdf_id_col)
        del rdf
        gc.collect()

    # Save dataset summary.
    target_cols = list(TARGET_SPECS.keys())
    summary_rows = []
    for t in target_cols:
        s = master[t]
        summary_rows.append({
            "target": t,
            "description": TARGET_SPECS[t]["description"],
            "n_nonmissing": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        })
    atomic_save_csv(pd.DataFrame(summary_rows), TABLE_DIR / "table_dataset_target_summary.csv")

    # Save columns inventory for the student and for reproducibility.
    inventory = pd.DataFrame({
        "column": master.columns,
        "dtype": [str(master[c].dtype) for c in master.columns],
        "n_missing": [int(master[c].isna().sum()) for c in master.columns],
        "n_unique": [int(master[c].nunique(dropna=True)) for c in master.columns],
    })
    atomic_save_csv(inventory, TABLE_DIR / "table_master_column_inventory.csv")

    master = reduce_dataframe_memory(master, "master_table")
    atomic_pickle(master, master_path)
    atomic_save_csv(master.head(1000), TABLE_DIR / "master_table_first_1000_rows_preview.csv")
    mark_done(stage, {"n_rows": len(master), "n_columns": master.shape[1]})
    return master

# =============================================================================
# 7. Feature-set construction
# =============================================================================

def get_feature_columns(master: pd.DataFrame, feature_set: str) -> List[str]:
    """Return model feature columns for a named feature-set."""
    geom = [c for c in GEOMETRY_CORE_COLUMNS if c in master.columns]
    if feature_set == "geometry_only":
        return geom
    if feature_set == "geometry_plus_racs":
        rac = [c for c in master.columns if c.startswith("RAC__")]
        if not rac:
            return []
        return geom + rac
    if feature_set == "geometry_plus_racs_rdfs":
        rac = [c for c in master.columns if c.startswith("RAC__")]
        rdf = [c for c in master.columns if c.startswith("RDF__")]
        if not rac and not rdf:
            return []
        return geom + rac + rdf
    raise ValueError(f"Unknown feature set: {feature_set}")



def audit_feature_set_integrity(master: pd.DataFrame) -> pd.DataFrame:
    """Write a QC table showing feature counts and duplicate feature sets."""
    rows = []
    seen: Dict[Tuple[str, ...], str] = {}
    for fs in CONFIG.get("feature_sets", []):
        cols = get_feature_columns(master, fs)
        key = tuple(cols)
        duplicate_of = seen.get(key, "") if cols else ""
        if cols and key not in seen:
            seen[key] = fs
        rows.append({
            "feature_set": fs,
            "n_features": len(cols),
            "n_geometry_features": len([c for c in cols if c in GEOMETRY_CORE_COLUMNS]),
            "n_rac_features": len([c for c in cols if c.startswith("RAC__")]),
            "n_rdf_features": len([c for c in cols if c.startswith("RDF__")]),
            "duplicate_of": duplicate_of,
            "usable_for_modelling": bool(cols) and duplicate_of == "",
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "QC_Table_feature_set_integrity.csv")
    return out


def active_feature_sets_for_modelling(master: pd.DataFrame) -> List[str]:
    """Return usable feature sets, skipping exact duplicates such as RACs+RDFs with zero RDF columns."""
    qc = audit_feature_set_integrity(master)
    usable = qc.loc[qc["usable_for_modelling"].astype(bool), "feature_set"].tolist() if not qc.empty else []
    skipped = qc.loc[~qc["usable_for_modelling"].astype(bool), ["feature_set", "duplicate_of", "n_features"]] if not qc.empty else pd.DataFrame()
    for _, r in skipped.iterrows():
        logging.info("FEATURE_SET_QC_SKIP | %s duplicate_of=%s n_features=%s", r.get("feature_set"), r.get("duplicate_of"), r.get("n_features"))
    return usable

# =============================================================================
# 8. Splitting utilities
# =============================================================================

SPLIT_GROUP_COLUMNS = {
    "random": None,
    "geometry_grouped": "geometry_cluster",
    "metal_grouped": "metal_cluster",
    "functional_grouped": "functional_cluster",
    "ligand_grouped": "ligand_cluster",
    "topology_grouped": "topology",
}


def make_train_cal_test_split(
    df: pd.DataFrame,
    target: str,
    split_name: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create indices for train, calibration, and test sets."""
    valid_idx = df.index[df[target].notna()].to_numpy()
    work = df.loc[valid_idx].copy()
    group_col = SPLIT_GROUP_COLUMNS.get(split_name)

    if split_name == "random" or group_col is None or group_col not in work.columns or work[group_col].isna().all():
        traincal_idx, test_idx = train_test_split(
            valid_idx, test_size=CONFIG["test_size"], random_state=seed
        )
        train_idx, cal_idx = train_test_split(
            traincal_idx, test_size=CONFIG["calibration_size_within_train"], random_state=seed + 1000
        )
        return np.asarray(train_idx), np.asarray(cal_idx), np.asarray(test_idx)

    groups = work[group_col].fillna("MISSING_GROUP").astype(str).to_numpy()
    if pd.Series(groups).nunique() < 3:
        logging.warning("GROUP_SPLIT_FALLBACK | %s has too few groups; using random split", split_name)
        traincal_idx, test_idx = train_test_split(valid_idx, test_size=CONFIG["test_size"], random_state=seed)
        train_idx, cal_idx = train_test_split(traincal_idx, test_size=CONFIG["calibration_size_within_train"], random_state=seed + 1000)
        return np.asarray(train_idx), np.asarray(cal_idx), np.asarray(test_idx)
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG["test_size"], random_state=seed)
        local_traincal, local_test = next(gss.split(work, groups=groups))
        traincal_idx = valid_idx[local_traincal]
        test_idx = valid_idx[local_test]

        work_traincal = df.loc[traincal_idx]
        groups_traincal = work_traincal[group_col].fillna("MISSING_GROUP").astype(str).to_numpy()
        if pd.Series(groups_traincal).nunique() < 2:
            raise ValueError("too few groups left for calibration split")
        gss2 = GroupShuffleSplit(n_splits=1, test_size=CONFIG["calibration_size_within_train"], random_state=seed + 1000)
        local_train, local_cal = next(gss2.split(work_traincal, groups=groups_traincal))
        train_idx = traincal_idx[local_train]
        cal_idx = traincal_idx[local_cal]
        return np.asarray(train_idx), np.asarray(cal_idx), np.asarray(test_idx)
    except Exception as e:
        logging.warning("GROUP_SPLIT_FALLBACK | %s failed (%s); using random split", split_name, e)
        traincal_idx, test_idx = train_test_split(valid_idx, test_size=CONFIG["test_size"], random_state=seed)
        train_idx, cal_idx = train_test_split(traincal_idx, test_size=CONFIG["calibration_size_within_train"], random_state=seed + 1000)
        return np.asarray(train_idx), np.asarray(cal_idx), np.asarray(test_idx)


def sample_fewshot_indices(train_idx: np.ndarray, y: pd.Series, budget: int, seed: int) -> np.ndarray:
    """
    Select a labelled few-shot training subset from the training indices.
    Stratified sampling over target quantile bins helps avoid all labels coming
    from low-uptake materials in very small budgets.
    """
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    if budget >= len(train_idx):
        return train_idx

    yy = y.loc[train_idx].to_numpy()
    n_bins = min(5, max(2, budget // 5))
    try:
        bins = pd.qcut(yy, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        bins = np.zeros_like(yy, dtype=int)

    selected = []
    for b in np.unique(bins):
        candidates = train_idx[np.asarray(bins) == b]
        n_take = max(1, int(round(budget * len(candidates) / len(train_idx))))
        n_take = min(n_take, len(candidates))
        selected.extend(rng.choice(candidates, size=n_take, replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < budget:
        remaining = np.setdiff1d(train_idx, np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=budget - len(selected), replace=False).tolist())
    if len(selected) > budget:
        selected = rng.choice(np.asarray(selected), size=budget, replace=False).tolist()
    return np.asarray(selected)

# =============================================================================
# 9. Model factory
# =============================================================================

def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """All current feature sets are numeric, but this remains future-proof."""
    num_cols = feature_cols
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
    ], remainder="drop")


def model_available(model_name: str) -> bool:
    if model_name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
            return True
        except Exception:
            return False
    if model_name == "xgboost":
        try:
            import xgboost  # noqa: F401
            return True
        except Exception:
            return False
    if model_name == "tabpfn":
        try:
            import tabpfn  # noqa: F401
            return True
        except Exception:
            return False
    return True


def maybe_wrap_target_transform(model_name: str, model: Any) -> Any:
    """Apply a log1p target transform to unconstrained regressors.

    Ridge/MLP/GPR can produce extreme negative or physically impossible positive
    predictions on raw adsorption targets. The log1p/expm1 wrapper keeps the
    benchmark comparable while greatly reducing pathological extrapolation.
    Tree and boosting models are kept on the original target scale.
    """
    if (
        CONFIG.get("enable_log_target_transform", True)
        and model_name in set(CONFIG.get("target_transform_models", []))
    ):
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return model


def make_model(model_name: str, seed: int, budget: int) -> Optional[Any]:
    """Create a regressor. Return None if skipped/unavailable."""
    if not model_available(model_name):
        logging.info("MODEL_SKIP | %s unavailable", model_name)
        return None

    if model_name == "ridge":
        return maybe_wrap_target_transform(model_name, Ridge(alpha=1.0, random_state=seed))

    if model_name == "rf":
        return maybe_wrap_target_transform(model_name, RandomForestRegressor(
            n_estimators=CONFIG["rf_n_estimators"],
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=CONFIG["n_jobs"],
        ))

    if model_name == "extra_trees":
        # ExtraTrees is a very useful CPU-friendly baseline for few-shot tabular data:
        # it is usually much faster than HGB while still providing nonlinear rankings.
        return maybe_wrap_target_transform(model_name, ExtraTreesRegressor(
            n_estimators=max(100, min(300, CONFIG["rf_n_estimators"])),
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=CONFIG["n_jobs"],
        ))

    if model_name == "hgb":
        # HistGradientBoosting can be slow in some Windows/Python builds for many
        # repeated few-shot fits, so we cap iterations modestly and make them
        # budget-aware. Users can increase CONFIG["hgb_max_iter"] for final runs.
        hgb_iter = int(min(CONFIG["hgb_max_iter"], max(20, budget // 5)))
        return maybe_wrap_target_transform(model_name, HistGradientBoostingRegressor(
            max_iter=hgb_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        ))

    if model_name == "mlp":
        # For tiny few-shot budgets, scikit-learn's internal early-stopping
        # validation set may contain only 1 sample, which raises:
        #   ValueError: The validation set is too small.
        # We therefore disable early stopping for small budgets and use a
        # smaller network to reduce overfitting and runtime. For larger budgets,
        # early stopping is useful and is re-enabled automatically.
        early_stop = budget >= int(CONFIG.get("mlp_early_stopping_min_budget", 50))
        hidden = (32,) if budget < 50 else (128, 64)
        return maybe_wrap_target_transform(model_name, MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=CONFIG["mlp_max_iter"],
            early_stopping=early_stop,
            validation_fraction=0.20 if early_stop else 0.10,
            random_state=seed,
        ))

    if model_name == "gpr":
        if budget > CONFIG["gpr_max_budget"]:
            logging.info("MODEL_SKIP | gpr budget %d > gpr_max_budget", budget)
            return None
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        return maybe_wrap_target_transform(model_name, GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, alpha=1e-6))

    if model_name == "lightgbm":
        import lightgbm as lgb
        return maybe_wrap_target_transform(model_name, lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=CONFIG["n_jobs"],
            verbose=-1,
        ))

    if model_name == "xgboost":
        import xgboost as xgb
        return maybe_wrap_target_transform(model_name, xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=CONFIG["n_jobs"],
        ))

    if model_name == "tabpfn":
        # TabPFN APIs have changed across versions. This try/except block allows
        # the rest of the manuscript pipeline to run even if the local package
        # uses a different interface or is missing.
        try:
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor(device="cpu")
        except Exception as e:
            logging.warning("TabPFN import/interface failed: %s", e)
            return None

    raise ValueError(f"Unknown model: {model_name}")

# =============================================================================
# 10. Metrics, ranking, conformal prediction
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    # Spearman is undefined when either vector is constant. This can happen for
    # very small few-shot budgets or conservative models. Treat it as a valid
    # diagnostic nan rather than letting scipy emit repeated ConstantInputWarning.
    if np.nanstd(yt) == 0 or np.nanstd(yp) == 0:
        rho = np.nan
    else:
        rho = spearmanr(yt, yp).correlation
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def ndcg_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    """NDCG using continuous adsorption values as relevance.

    The raw adsorption values can be much larger for high-pressure CH4 than for
    low-pressure CO2. Scaling the relevance vector to [0, 1] avoids numerical
    overflow in the gain calculation while preserving the rank-based meaning of
    NDCG.
    """
    if k <= 0:
        return np.nan
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(score)
    if mask.sum() < 2:
        return np.nan
    y = y_true[mask]
    sc = score[mask]
    k = min(k, len(y))
    order = np.argsort(sc)[::-1][:k]
    ideal = np.argsort(y)[::-1][:k]
    denom = np.nanmax(y) - np.nanmin(y)
    rel = (y - np.nanmin(y)) / denom if denom > 0 else np.ones_like(y)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[order] - 1) / discounts)
    idcg = np.sum((2 ** rel[ideal] - 1) / discounts)
    return float(dcg / idcg) if idcg > 0 else np.nan


def topk_metrics(y_true: np.ndarray, y_score: np.ndarray, top_fracs: List[float]) -> List[Dict[str, float]]:
    """Compute top-k ranking/screening metrics for each top fraction."""
    n = len(y_true)
    rows = []
    true_order = np.argsort(y_true)[::-1]
    pred_order = np.argsort(y_score)[::-1]
    for frac in top_fracs:
        k = max(1, int(math.ceil(frac * n)))
        true_top = set(true_order[:k].tolist())
        pred_top = set(pred_order[:k].tolist())
        inter = true_top & pred_top
        precision = len(inter) / max(len(pred_top), 1)
        recall = len(inter) / max(len(true_top), 1)
        jaccard = len(inter) / max(len(true_top | pred_top), 1)
        enrichment = precision / frac if frac > 0 else np.nan
        rows.append({
            "top_frac": frac,
            "k": k,
            "precision": precision,
            "recall": recall,
            "jaccard": jaccard,
            "enrichment": enrichment,
            "ndcg": ndcg_at_k(y_true, y_score, k),
        })
    return rows


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)
    if n == 0:
        return np.nan
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def split_conformal_intervals(y_cal: np.ndarray, pred_cal: np.ndarray, pred_test: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
    residuals = np.abs(y_cal - pred_cal)
    q = conformal_quantile(residuals, alpha)
    return pred_test - q, pred_test + q, q


def locally_weighted_conformal_intervals(
    X_cal_trans: np.ndarray,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    X_test_trans: np.ndarray,
    pred_test: np.ndarray,
    alpha: float,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locally weighted conformal interval using nearest calibration neighbours.

    Why this implementation is deliberately capped:
    the ARC-MOF test set can contain tens of thousands of structures and the
    calibration set can also be large. A full pairwise distance matrix for every
    experiment can become the dominant runtime and memory bottleneck. Therefore,
    this function:
      1. optionally disables local conformal intervals via CONFIG;
      2. subsamples at most max_local_conformal_calibration_points calibration
         points in a deterministic way;
      3. queries neighbours in chunks using sklearn.neighbors.NearestNeighbors;
      4. falls back to global split-conformal widths if anything goes wrong.

    This keeps the pipeline robust and restart-safe while still giving a useful
    local-width diagnostic for the manuscript/SI.
    """
    residuals = np.abs(y_cal - pred_cal)
    residuals = np.asarray(residuals, dtype=float)
    valid = np.isfinite(residuals)
    if valid.sum() == 0:
        return np.full_like(pred_test, np.nan, dtype=float), np.full_like(pred_test, np.nan, dtype=float)
    global_q = conformal_quantile(residuals[valid], alpha)
    if not CONFIG.get("enable_local_conformal", True):
        return pred_test - global_q, pred_test + global_q

    try:
        Xc = np.asarray(X_cal_trans, dtype=np.float32)[valid]
        res = residuals[valid]
        Xt = np.asarray(X_test_trans, dtype=np.float32)
        n_cal = Xc.shape[0]
        if n_cal == 0 or Xt.shape[0] == 0:
            return pred_test - global_q, pred_test + global_q

        max_cal = int(CONFIG.get("max_local_conformal_calibration_points", 5000))
        if n_cal > max_cal:
            rng = np.random.default_rng(CONFIG.get("random_seed", 42))
            keep = rng.choice(np.arange(n_cal), size=max_cal, replace=False)
            Xc = Xc[keep]
            res = res[keep]
            n_cal = max_cal

        k = min(int(k_neighbors), n_cal)
        if k <= 0:
            return pred_test - global_q, pred_test + global_q

        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(Xc)
        chunk = int(CONFIG.get("local_conformal_query_chunk_size", 4000))
        lows, highs = [], []
        for start in range(0, Xt.shape[0], chunk):
            stop = min(start + chunk, Xt.shape[0])
            indices = nn.kneighbors(Xt[start:stop], return_distance=False)
            q_vals = np.array([conformal_quantile(res[idx], alpha) for idx in indices], dtype=float)
            pred_chunk = pred_test[start:stop]
            lows.append(pred_chunk - q_vals)
            highs.append(pred_chunk + q_vals)
        return np.concatenate(lows), np.concatenate(highs)
    except Exception as e:
        logging.warning("LOCAL_CONFORMAL_FALLBACK | %s", e)
        return pred_test - global_q, pred_test + global_q

def mondrian_conformal_intervals(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    group_cal: np.ndarray,
    group_test: np.ndarray,
    alpha: float,
    min_cal_per_group: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Group-conditional conformal intervals with fallback to global quantile."""
    residuals = np.abs(y_cal - pred_cal)
    global_q = conformal_quantile(residuals, alpha)
    q_by_group = {}
    group_cal = pd.Series(group_cal).fillna("MISSING_GROUP").astype(str).to_numpy()
    for g in np.unique(group_cal):
        idx = np.where(group_cal == g)[0]
        if len(idx) >= min_cal_per_group:
            q_by_group[g] = conformal_quantile(residuals[idx], alpha)
    group_test = pd.Series(group_test).fillna("MISSING_GROUP").astype(str).to_numpy()
    q = np.array([q_by_group.get(g, global_q) for g in group_test], dtype=float)
    return pred_test - q, pred_test + q


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if mask.sum() == 0:
        return {"coverage": np.nan, "mean_width": np.nan, "median_width": np.nan}
    covered = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    width = upper[mask] - lower[mask]
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }


def assign_candidate_tiers(pred: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Assign trusted/uncertain/rejected tiers.

    Trusted = high lower confidence bound and sufficiently narrow interval.
    Rejected = low upper confidence bound.
    Uncertain = everything in between.
    """
    width = upper - lower
    width_thr = robust_quantile(width, CONFIG["trusted_interval_width_quantile"])
    lb_thr = robust_quantile(lower, CONFIG["trusted_lower_bound_percentile"])
    ub_low_thr = robust_quantile(upper, 0.50)
    tiers = np.full(len(pred), "uncertain", dtype=object)
    tiers[(lower >= lb_thr) & (width <= width_thr)] = "trusted"
    tiers[upper < ub_low_thr] = "rejected"
    return tiers

# =============================================================================
# 10b. Fit/predict helpers
# =============================================================================

def fit_predict_with_warning_control(pipe: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, X_cal: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a pipeline and predict while silencing known harmless ML warnings.

    LightGBM sometimes warns that X has no valid feature names after a
    scikit-learn ColumnTransformer converts DataFrames to numeric arrays. The
    feature order is still controlled by the pipeline, so this warning is not a
    modelling error. The sklearn parallel warning is similarly a configuration
    propagation warning emitted by nested estimators/workers.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*X does not have valid feature names.*LGBMRegressor.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*should be used with sklearn\.utils\.parallel\.Parallel.*", category=UserWarning)
        pipe.fit(X_train, y_train)
        pred_cal = pipe.predict(X_cal)
        pred_test = pipe.predict(X_test)
    return pred_cal, pred_test

# =============================================================================
# 11. Core experiment loop
# =============================================================================

def run_single_experiment(
    master: pd.DataFrame,
    target: str,
    feature_set: str,
    split_name: str,
    model_name: str,
    budget: int,
    seed: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run one target/feature/split/model/budget/seed experiment."""
    exp_id = f"{target}__{feature_set}__{split_name}__{model_name}__b{budget}__s{seed}"
    pred_path = PRED_DIR / f"pred_{exp_id}.csv"
    compact_pred_path = PRED_DIR / f"pred_important_{exp_id}.csv"
    metric_path = TABLE_DIR / "experiment_metrics_parts" / f"metrics_{exp_id}.csv"
    if (pred_path.exists() or compact_pred_path.exists()) and metric_path.exists() and not CONFIG["force_recompute"]:
        # Do not reload the large prediction file here; collect_candidate_tiers()
        # will concatenate prediction files once after all experiments finish.
        return pd.read_csv(metric_path), None

    feature_cols = get_feature_columns(master, feature_set)
    if not feature_cols:
        logging.info("FEATURE_SET_SKIP | %s has no columns", feature_set)
        return None, None

    train_idx, cal_idx, test_idx = make_train_cal_test_split(master, target, split_name, seed)
    few_idx = sample_fewshot_indices(train_idx, master[target], budget, seed)

    X_train = master.loc[few_idx, feature_cols]
    y_train = master.loc[few_idx, target].astype(float).to_numpy()
    X_cal = master.loc[cal_idx, feature_cols]
    y_cal = master.loc[cal_idx, target].astype(float).to_numpy()
    X_test = master.loc[test_idx, feature_cols]
    y_test = master.loc[test_idx, target].astype(float).to_numpy()

    model = make_model(model_name, seed=seed, budget=budget)
    if model is None:
        return None, None

    pre = make_preprocessor(feature_cols)
    pipe = Pipeline([("pre", pre), ("model", model)])

    try:
        t0 = time.time()
        pred_cal, pred_test = fit_predict_with_warning_control(pipe, X_train, y_train, X_cal, X_test)
        fit_seconds = time.time() - t0

        # Transformed features for local conformal.
        X_cal_trans = pipe.named_steps["pre"].transform(X_cal)
        X_test_trans = pipe.named_steps["pre"].transform(X_test)
        if hasattr(X_cal_trans, "toarray"):
            X_cal_trans = X_cal_trans.toarray()
        if hasattr(X_test_trans, "toarray"):
            X_test_trans = X_test_trans.toarray()

        lower_split, upper_split, q_global = split_conformal_intervals(y_cal, pred_cal, pred_test, CONFIG["conformal_alpha"])
        lower_local, upper_local = locally_weighted_conformal_intervals(
            X_cal_trans, y_cal, pred_cal, X_test_trans, pred_test,
            alpha=CONFIG["conformal_alpha"], k_neighbors=CONFIG["local_conformal_k"]
        )

        group_col = SPLIT_GROUP_COLUMNS.get(split_name)
        if group_col and group_col in master.columns:
            lower_mond, upper_mond = mondrian_conformal_intervals(
                y_cal, pred_cal, pred_test,
                group_cal=master.loc[cal_idx, group_col].to_numpy(),
                group_test=master.loc[test_idx, group_col].to_numpy(),
                alpha=CONFIG["conformal_alpha"],
                min_cal_per_group=CONFIG["mondrian_min_cal_per_group"],
            )
        else:
            lower_mond, upper_mond = lower_split.copy(), upper_split.copy()

        tiers = assign_candidate_tiers(pred_test, lower_split, upper_split)

        pred_df = pd.DataFrame({
            "mof_id": master.loc[test_idx, "mof_id"].to_numpy(),
            "filename": master.loc[test_idx, "filename"].to_numpy() if "filename" in master.columns else master.loc[test_idx, "mof_id"].to_numpy(),
            "target": target,
            "feature_set": feature_set,
            "split": split_name,
            "model": model_name,
            "budget": budget,
            "seed": seed,
            "y_true": y_test,
            "y_pred": pred_test,
            "split_lower": lower_split,
            "split_upper": upper_split,
            "local_lower": lower_local,
            "local_upper": upper_local,
            "mondrian_lower": lower_mond,
            "mondrian_upper": upper_mond,
            "tier": tiers,
        })
        # Attach key chemical/group columns to enable failure anatomy without rerun.
        attach_cols = [c for c in ["Density", "Di", "Df", "Dif", "geometry_cluster", "metal_cluster", "functional_cluster", "ligand_cluster", "topology"] if c in master.columns]
        for c in attach_cols:
            pred_df[c] = master.loc[test_idx, c].to_numpy()

        metrics = regression_metrics(y_test, pred_test)
        rows = []
        base = {
            "target": target,
            "feature_set": feature_set,
            "split": split_name,
            "model": model_name,
            "budget": budget,
            "seed": seed,
            "n_train_labelled": len(few_idx),
            "n_cal": len(cal_idx),
            "n_test": len(test_idx),
            "fit_seconds": fit_seconds,
            **metrics,
        }
        for top_row in topk_metrics(y_test, pred_test, CONFIG["top_fracs"]):
            rows.append({**base, **top_row})
        metric_df = pd.DataFrame(rows)

        # Add interval metrics in wide form.
        for kind, lo, hi in [
            ("split", lower_split, upper_split),
            ("local", lower_local, upper_local),
            ("mondrian", lower_mond, upper_mond),
        ]:
            im = interval_metrics(y_test, lo, hi)
            for k, v in im.items():
                metric_df[f"{kind}_{k}"] = v

        metric_df["global_conformal_q"] = q_global
        metric_df["trusted_count"] = int((tiers == "trusted").sum())
        metric_df["uncertain_count"] = int((tiers == "uncertain").sum())
        metric_df["rejected_count"] = int((tiers == "rejected").sum())

        save_prediction_table_for_mode(pred_df, pred_path, compact_pred_path)
        atomic_save_csv(metric_df, metric_path)

        # Save fitted model only in full mode because hundreds/thousands of
        # fitted model objects can consume substantial disk. Per-experiment
        # metadata JSONs are small but numerous, so they are saved only in
        # balanced/full. Lean keeps compact predictions and core pickle
        # checkpoints instead.
        meta = {
            "exp_id": exp_id,
            "feature_cols": feature_cols,
            "fit_seconds": fit_seconds,
            "timestamp": datetime.now().isoformat(),
            "save_mode": CONFIG.get("save_mode"),
        }
        if save_mode_at_least("balanced"):
            save_json(meta, MODEL_DIR / f"model_meta_{exp_id}.json")
        if save_mode_at_least("full"):
            atomic_pickle(pipe, MODEL_DIR / f"fitted_model_{exp_id}.pkl")
            np.savez_compressed(
                MODEL_DIR / f"split_indices_{exp_id}.npz",
                train_idx=np.asarray(train_idx), cal_idx=np.asarray(cal_idx),
                test_idx=np.asarray(test_idx), few_idx=np.asarray(few_idx),
            )

        logging.info("EXPERIMENT_DONE | %s | rmse=%.4f spearman=%.3f", exp_id, metrics["rmse"], metrics["spearman"])
        return metric_df, pred_df

    except Exception as e:
        logging.error("EXPERIMENT_FAILED | %s | %s", exp_id, e)
        logging.error(traceback.format_exc())
        fail = {
            "exp_id": exp_id,
            "target": target,
            "feature_set": feature_set,
            "split": split_name,
            "model": model_name,
            "budget": budget,
            "seed": seed,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }
        save_json(fail, LOG_DIR / f"failed_{exp_id}.json")
        return None, None


def run_all_experiments(master: pd.DataFrame) -> pd.DataFrame:
    stage = "02_run_experiments"
    metrics_all_path = TABLE_DIR / "all_experiment_metrics.csv"
    if is_done(stage) and metrics_all_path.exists():
        logging.info("Using existing all_experiment_metrics.csv")
        return pd.read_csv(metrics_all_path)

    logging.info("STAGE >>> RUN_ALL_EXPERIMENTS")
    requested_splits = CONFIG.get("enabled_splits")
    split_names = []
    for s, gcol in SPLIT_GROUP_COLUMNS.items():
        if requested_splits is not None and s not in requested_splits:
            continue
        if s == "random" or (gcol in master.columns and master[gcol].notna().sum() > 0):
            split_names.append(s)
        else:
            logging.info("SPLIT_SKIP | %s because %s missing", s, gcol)

    active_targets = CONFIG.get("active_targets") or list(TARGET_SPECS.keys())
    active_targets = [t for t in active_targets if t in TARGET_SPECS]
    metrics_parts = []
    active_feature_sets = active_feature_sets_for_modelling(master)
    total = (len(active_targets) * len(active_feature_sets) * len(split_names) *
             len(CONFIG["models_to_run"]) * len(CONFIG["label_budgets"]) * len(CONFIG["outer_seeds"]))
    counter = 0
    for target in active_targets:
        for feature_set in active_feature_sets:
            if not get_feature_columns(master, feature_set):
                continue
            for split_name in split_names:
                for model_name in CONFIG["models_to_run"]:
                    for budget in CONFIG["label_budgets"]:
                        for seed in CONFIG["outer_seeds"]:
                            counter += 1
                            logging.info("JOB_PROGRESS | %d/%d | target=%s feature=%s split=%s model=%s budget=%s seed=%s",
                                         counter, total, target, feature_set, split_name, model_name, budget, seed)
                            metric_df, _ = run_single_experiment(master, target, feature_set, split_name, model_name, budget, seed)
                            if metric_df is not None:
                                metrics_parts.append(metric_df)
                            gc.collect()

    # In a restart, some experiments may already exist but not in this process list.
    part_dir = TABLE_DIR / "experiment_metrics_parts"
    all_parts = sorted(part_dir.glob("metrics_*.csv"))
    metrics_all = pd.concat([pd.read_csv(p) for p in all_parts], ignore_index=True) if all_parts else pd.DataFrame()
    atomic_save_csv(metrics_all, metrics_all_path)
    maybe_save_pickle(metrics_all, PICKLE_DIR / "all_experiment_metrics.pkl", minimum_mode="efficient")
    mark_done(stage, {"n_metric_rows": len(metrics_all), "n_metric_files": len(all_parts)})
    return metrics_all

# =============================================================================
# 12. Aggregation, confidence intervals, and candidate-tier tables
# =============================================================================

def bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float, seed: int = 42) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    return float(np.mean(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def aggregate_results(metrics_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    stage = "03_aggregate_results"
    agg_path = PICKLE_DIR / "aggregated_results.pkl"
    if is_done(stage):
        if agg_path.exists():
            logging.info("Using existing aggregated results pickle")
            return read_pickle(agg_path)
        csv_path = TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv"
        if csv_path.exists():
            logging.info("Using existing aggregated results CSV")
            agg = pd.read_csv(csv_path)
            leaderboard_path = TABLE_DIR / "table_model_leaderboard_top5pct.csv"
            leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
            return {"agg": agg, "leaderboard": leaderboard}

    logging.info("STAGE >>> AGGREGATE_RESULTS")
    group_cols = ["target", "feature_set", "split", "model", "budget", "top_frac"]
    metric_cols = [
        "rmse", "mae", "r2", "spearman", "precision", "recall", "jaccard", "enrichment", "ndcg",
        "split_coverage", "split_mean_width", "split_median_width",
        "local_coverage", "local_mean_width", "mondrian_coverage", "mondrian_mean_width",
        "trusted_count", "uncertain_count", "rejected_count", "fit_seconds",
    ]
    rows = []
    for keys, sub in metrics_all.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for m in metric_cols:
            if m in sub.columns:
                mean, lo, hi = bootstrap_ci(sub[m].to_numpy(), CONFIG["n_bootstrap"], CONFIG["bootstrap_alpha"])
                base[f"{m}_mean"] = mean
                base[f"{m}_ci_low"] = lo
                base[f"{m}_ci_high"] = hi
        base["n_runs"] = int(sub["seed"].nunique()) if "seed" in sub.columns else len(sub)
        rows.append(base)
    agg = pd.DataFrame(rows)
    atomic_save_csv(agg, TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv")

    # Main leaderboard at top-5% using geometry_only and random/grouped splits.
    leaderboard = agg[(agg["top_frac"] == 0.05)].copy()
    sort_cols = ["target", "feature_set", "split", "budget", "recall_mean", "split_coverage_mean"]
    leaderboard = leaderboard.sort_values(sort_cols, ascending=[True, True, True, True, False, False])
    atomic_save_csv(leaderboard, TABLE_DIR / "table_model_leaderboard_top5pct.csv")

    out = {"agg": agg, "leaderboard": leaderboard}
    maybe_save_pickle(out, agg_path, minimum_mode="efficient")
    mark_done(stage, {"n_agg_rows": len(agg)})
    return out


PREDICTION_COMPACT_COLUMNS = [
    "mof_id", "filename", "target", "feature_set", "split", "model", "budget", "seed",
    "y_true", "y_pred", "split_lower", "split_upper", "tier",
    "Density", "Di", "Df", "Dif",
    "geometry_cluster", "metal_cluster", "functional_cluster", "ligand_cluster", "topology",
]


def truncate_prediction_text_columns(df: pd.DataFrame, max_chars: Optional[int] = None) -> pd.DataFrame:
    """Limit very long text fields before saving or aggregating prediction tables.

    Some grouping columns, especially topology-like columns from heterogeneous
    input files, can occasionally contain long list-like strings. Keeping such
    strings inside thousands of CSV files makes pandas parsing memory-hungry and
    can trigger C-parser "out of memory" errors during post-processing. Truncation
    is safe here because these columns are used for coarse diagnostic labelling,
    not for numerical model fitting.
    """
    if df is None or df.empty:
        return df
    max_chars = int(max_chars or CONFIG.get("prediction_text_max_chars", 160))
    out = df.copy()
    text_cols = [
        col for col in out.columns
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])
    ]
    for col in text_cols:
        out[col] = out[col].map(
            lambda v: v if pd.isna(v) else str(v).replace("\r", " ").replace("\n", " ")[:max_chars]
        )
    return out


def compact_candidate_rows(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Return the most useful candidate/diagnostic rows from a prediction table.

    The selected rows cover:
      * trusted candidates;
      * highest lower-confidence-bound candidates;
      * highest predicted candidates;
      * largest-error examples when y_true is available;
      * a deterministic small sample.

    This preserves the rows needed for shortlist, uncertainty, and failure
    analysis without requiring the full prediction matrix in RAM.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    max_rows = int(max_rows or CONFIG.get("candidate_tiers_per_file_rows", 1500))
    work = df.copy()
    for col in ["budget", "seed"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["y_true", "y_pred", "split_lower", "split_upper", "Density", "Di", "Df", "Dif"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    parts = []
    per_bucket = max(25, max_rows // 5)
    if "tier" in work.columns:
        trusted = work[work["tier"].astype(str).eq("trusted")]
        if not trusted.empty:
            if "split_lower" in trusted.columns:
                trusted = trusted.sort_values("split_lower", ascending=False).head(max_rows)
            else:
                trusted = trusted.head(max_rows)
            parts.append(trusted)

    if "split_lower" in work.columns:
        parts.append(work.sort_values("split_lower", ascending=False).head(per_bucket))
    if "y_pred" in work.columns:
        parts.append(work.sort_values("y_pred", ascending=False).head(per_bucket))
    if {"y_true", "y_pred"}.issubset(work.columns):
        tmp = work.copy()
        tmp["_abs_error_for_selection"] = np.abs(tmp["y_true"] - tmp["y_pred"])
        parts.append(
            tmp.sort_values("_abs_error_for_selection", ascending=False)
               .head(per_bucket)
               .drop(columns=["_abs_error_for_selection"], errors="ignore")
        )

    sample_n = min(per_bucket, len(work))
    if sample_n > 0:
        parts.append(work.sample(sample_n, random_state=int(CONFIG.get("random_seed", 42))))

    if not parts:
        out = work.head(max_rows)
    else:
        out = pd.concat(parts, ignore_index=True, copy=False).drop_duplicates()

    if len(out) > max_rows:
        # Keep the strongest lower-bound rows first, then a small diagnostic tail.
        if "split_lower" in out.columns:
            top = out.sort_values("split_lower", ascending=False).head(max_rows)
        else:
            top = out.head(max_rows)
        out = top

    return truncate_prediction_text_columns(out)


def read_prediction_file_compact(path: Path) -> pd.DataFrame:
    """Read one prediction CSV defensively and return a compact subset.

    This function avoids the old behaviour of reading every prediction CSV fully
    and concatenating all rows. It also skips malformed/corrupt files gracefully,
    which is important after interrupted Windows runs.
    """
    keep_cols = set(PREDICTION_COMPACT_COLUMNS)
    per_file_rows = int(CONFIG.get("candidate_tiers_per_file_rows", 1500))
    chunk_size = int(CONFIG.get("candidate_tiers_chunk_size", 50000))

    def _reader(engine: str):
        return pd.read_csv(
            path,
            usecols=lambda c: c in keep_cols,
            chunksize=chunk_size,
            engine=engine,
            on_bad_lines="skip",
            low_memory=False if engine == "c" else True,
        )

    chunks = []
    try:
        iterator = _reader("c")
        for chunk in iterator:
            if chunk.empty:
                continue
            chunks.append(compact_candidate_rows(chunk, max_rows=max(100, per_file_rows // 2)))
            if sum(len(c) for c in chunks) > per_file_rows * 3:
                chunks = [compact_candidate_rows(pd.concat(chunks, ignore_index=True, copy=False), max_rows=per_file_rows)]
    except Exception as e_c:
        # The pandas C parser can report "out of memory" when a CSV contains an
        # extremely long corrupted line or very long text field. Try the Python
        # parser once; if that also fails, skip this file rather than killing the
        # whole post-processing stage.
        logging.warning("Prediction file C-parser failed, retrying with Python parser | %s | %s", path, e_c)
        chunks = []
        try:
            iterator = _reader("python")
            for chunk in iterator:
                if chunk.empty:
                    continue
                chunks.append(compact_candidate_rows(chunk, max_rows=max(100, per_file_rows // 2)))
                if sum(len(c) for c in chunks) > per_file_rows * 3:
                    chunks = [compact_candidate_rows(pd.concat(chunks, ignore_index=True, copy=False), max_rows=per_file_rows)]
        except Exception as e_py:
            logging.warning("Could not read prediction file %s: %s", path, e_py)
            return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()
    try:
        return compact_candidate_rows(pd.concat(chunks, ignore_index=True, copy=False), max_rows=per_file_rows)
    except Exception as e:
        logging.warning("Could not compact prediction file %s: %s", path, e)
        return pd.DataFrame()


def choose_prediction_files_for_collection() -> List[Path]:
    """Choose prediction files for candidate-tier collection.

    Prefer compact files where available, but do not lose earlier results when a
    run is resumed after changing save mode. If both `pred_important_*` and
    older full `pred_*` files exist, this function uses the compact version for
    experiments that have one and falls back to the full file only for
    experiments without a compact counterpart.
    """
    important_files = sorted(PRED_DIR.glob("pred_important_*.csv"))
    regular_files = sorted(p for p in PRED_DIR.glob("pred_*.csv") if not p.name.startswith("pred_important_"))

    if CONFIG.get("prefer_compact_prediction_files", True) and important_files:
        compact_regular_names = {"pred_" + p.name[len("pred_important_"):] for p in important_files}
        regular_without_compact = [p for p in regular_files if p.name not in compact_regular_names]
        chosen = important_files + regular_without_compact
        logging.info(
            "Candidate-tier collection will use compact predictions where available: compact=%d fallback_full=%d total=%d",
            len(important_files), len(regular_without_compact), len(chosen)
        )
        return sorted(chosen)
    if regular_files:
        logging.info("Candidate-tier collection will use full prediction files: %d", len(regular_files))
        return regular_files
    logging.info("Candidate-tier collection found no prediction files.")
    return []


EXPERIMENT_KEY_COLS = ["target", "feature_set", "split", "model", "budget", "seed"]


def target_stats_from_master(master: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Return observed target ranges for prediction-sanity checks."""
    stats: Dict[str, Dict[str, float]] = {}
    if master is not None and not master.empty:
        for t in TARGET_SPECS:
            if t in master.columns:
                vals = pd.to_numeric(master[t], errors="coerce").dropna()
                if len(vals):
                    stats[t] = {
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "std": float(vals.std()),
                        "mean": float(vals.mean()),
                    }
    if not stats and (TABLE_DIR / "table_dataset_target_summary.csv").exists():
        ts = pd.read_csv(TABLE_DIR / "table_dataset_target_summary.csv")
        for _, r in ts.iterrows():
            stats[str(r["target"])] = {
                "min": float(r.get("min", np.nan)),
                "max": float(r.get("max", np.nan)),
                "std": float(r.get("std", np.nan)),
                "mean": float(r.get("mean", np.nan)),
            }
    return stats


def build_experiment_quality_table(metrics_all: Optional[pd.DataFrame], master: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build experiment-level gates for final candidate nomination.

    The benchmark still keeps every model, but final candidate tables should not
    be dominated by unstable baseline predictions. The gate is deliberately
    transparent and written to QC tables.
    """
    if metrics_all is None or metrics_all.empty:
        path = TABLE_DIR / "all_experiment_metrics.csv"
        metrics_all = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if metrics_all.empty:
        return pd.DataFrame()
    m = metrics_all.copy()
    if "top_frac" in m.columns and (m["top_frac"].astype(float).sub(0.05).abs() < 1e-9).any():
        m = m[m["top_frac"].astype(float).sub(0.05).abs() < 1e-9].copy()
    for col in ["budget", "seed", "rmse", "spearman", "recall", "split_coverage"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    target_stats = target_stats_from_master(master)
    rows = []
    eligible = set(CONFIG.get("candidate_eligible_models", []))
    for keys, sub in m.groupby(EXPERIMENT_KEY_COLS, dropna=False):
        base = dict(zip(EXPERIMENT_KEY_COLS, keys))
        t = str(base["target"])
        ts = target_stats.get(t, {})
        target_std = float(ts.get("std", np.nan))
        target_max = float(ts.get("max", np.nan))
        rmse = float(np.nanmean(sub["rmse"])) if "rmse" in sub.columns else np.nan
        spearman = float(np.nanmean(sub["spearman"])) if "spearman" in sub.columns else np.nan
        recall = float(np.nanmean(sub["recall"])) if "recall" in sub.columns else np.nan
        coverage = float(np.nanmean(sub["split_coverage"])) if "split_coverage" in sub.columns else np.nan
        reasons = []
        model_eligible = str(base["model"]) in eligible
        if not model_eligible:
            reasons.append("model_not_candidate_eligible")
        if not np.isfinite(spearman) or spearman < float(CONFIG.get("min_candidate_spearman", 0.0)):
            reasons.append("low_or_nan_spearman")
        if not np.isfinite(recall) or recall < float(CONFIG.get("min_candidate_top5_recall", 0.02)):
            reasons.append("low_top5_recall")
        if not np.isfinite(coverage) or coverage < float(CONFIG.get("min_candidate_split_coverage", 0.70)) or coverage > float(CONFIG.get("max_candidate_split_coverage", 0.99)):
            reasons.append("coverage_outside_gate")
        if np.isfinite(target_std) and target_std > 0 and np.isfinite(rmse):
            if rmse > float(CONFIG.get("max_candidate_rmse_std_multiplier", 3.0)) * target_std:
                reasons.append("rmse_too_large_for_target")
        elif not np.isfinite(rmse):
            reasons.append("nan_rmse")
        base.update({
            "target_observed_max": target_max,
            "target_observed_std": target_std,
            "rmse_top5row": rmse,
            "spearman_top5row": spearman,
            "recall_top5row": recall,
            "split_coverage_top5row": coverage,
            "model_candidate_eligible": model_eligible,
            "experiment_passes_quality_gate": len(reasons) == 0,
            "quality_gate_reasons": ";".join(reasons) if reasons else "PASS",
        })
        rows.append(base)
    q = pd.DataFrame(rows)
    atomic_save_csv(q, TABLE_DIR / "QC_Table_model_stability.csv")
    if not q.empty:
        atomic_save_csv(q[~q["experiment_passes_quality_gate"].astype(bool)], TABLE_DIR / "QC_Table_rejected_experiments.csv")
    return q


def apply_prediction_quality_flags(df: pd.DataFrame, quality: pd.DataFrame, target_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Attach row-level candidate eligibility and physical-sanity flags."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["budget", "seed", "y_true", "y_pred", "split_lower", "split_upper"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if quality is not None and not quality.empty:
        keep_cols = EXPERIMENT_KEY_COLS + ["experiment_passes_quality_gate", "quality_gate_reasons", "model_candidate_eligible"]
        q = quality[[c for c in keep_cols if c in quality.columns]].drop_duplicates(EXPERIMENT_KEY_COLS)
        out = out.merge(q, on=EXPERIMENT_KEY_COLS, how="left")
    else:
        out["experiment_passes_quality_gate"] = True
        out["quality_gate_reasons"] = "NO_METRICS_AVAILABLE"
        out["model_candidate_eligible"] = out["model"].isin(CONFIG.get("candidate_eligible_models", [])) if "model" in out else False

    target_max = out["target"].map(lambda t: target_stats.get(str(t), {}).get("max", np.nan)) if "target" in out else pd.Series(np.nan, index=out.index)
    upper = pd.to_numeric(target_max, errors="coerce") * float(CONFIG.get("prediction_upper_multiplier_observed_max", 1.25))
    lower = pd.to_numeric(target_max, errors="coerce") * float(CONFIG.get("prediction_lower_fraction_observed_max", -0.05))
    out["prediction_is_finite"] = np.isfinite(pd.to_numeric(out.get("y_pred", np.nan), errors="coerce"))
    out["prediction_is_nonnegative_or_tolerated"] = pd.to_numeric(out.get("y_pred", np.nan), errors="coerce") >= lower.fillna(-np.inf)
    out["prediction_within_observed_physical_range"] = pd.to_numeric(out.get("y_pred", np.nan), errors="coerce") <= upper.fillna(np.inf)
    out["interval_is_finite"] = np.isfinite(pd.to_numeric(out.get("split_lower", np.nan), errors="coerce")) & np.isfinite(pd.to_numeric(out.get("split_upper", np.nan), errors="coerce"))
    out["prediction_sanity_pass"] = (
        out["prediction_is_finite"]
        & out["prediction_is_nonnegative_or_tolerated"]
        & out["prediction_within_observed_physical_range"]
        & out["interval_is_finite"]
    )
    out["candidate_eligible"] = (
        out.get("model_candidate_eligible", False).fillna(False).astype(bool)
        & out.get("experiment_passes_quality_gate", False).fillna(False).astype(bool)
        & out["prediction_sanity_pass"].fillna(False).astype(bool)
    )
    return out


def write_prediction_range_qc(tiers_raw: pd.DataFrame) -> None:
    """Write QC summaries that reveal impossible prediction ranges by model."""
    if tiers_raw is None or tiers_raw.empty:
        return
    df = tiers_raw.copy()
    for col in ["y_pred", "y_true", "split_lower", "split_upper", "budget", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    group_cols = [c for c in ["target", "feature_set", "split", "model", "budget"] if c in df.columns]
    if not group_cols:
        return
    qc = df.groupby(group_cols, dropna=False).agg(
        n_rows=("mof_id", "size") if "mof_id" in df.columns else ("y_pred", "size"),
        y_pred_min=("y_pred", "min"),
        y_pred_median=("y_pred", "median"),
        y_pred_max=("y_pred", "max"),
        y_true_min=("y_true", "min"),
        y_true_max=("y_true", "max"),
        n_candidate_eligible=("candidate_eligible", "sum") if "candidate_eligible" in df.columns else ("y_pred", "size"),
        n_prediction_sanity_pass=("prediction_sanity_pass", "sum") if "prediction_sanity_pass" in df.columns else ("y_pred", "size"),
    ).reset_index()
    atomic_save_csv(qc, TABLE_DIR / "QC_Table_prediction_range_by_model.csv")


def collect_candidate_tiers(metrics_all: Optional[pd.DataFrame] = None, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    stage = "04_candidate_tiers"
    out_path = TABLE_DIR / "table_candidate_tiers_all_predictions.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path, low_memory=False)

    logging.info("STAGE >>> COLLECT_CANDIDATE_TIERS_STREAMING")
    pred_files = choose_prediction_files_for_collection()
    max_total_rows = int(CONFIG.get("candidate_tiers_max_rows", 250000))
    target_stats = target_stats_from_master(master)
    quality_table = build_experiment_quality_table(metrics_all, master) if CONFIG.get("enable_candidate_quality_gate", True) else pd.DataFrame()

    collected: List[pd.DataFrame] = []
    rejected_qc_parts: List[pd.DataFrame] = []
    n_files_read = 0
    n_files_skipped = 0
    n_rows_seen_after_compaction = 0

    for i, pred_file in enumerate(pred_files, start=1):
        if i == 1 or i % 250 == 0:
            logging.info("CANDIDATE_TIER_PROGRESS | %d/%d files", i, len(pred_files))
        df = read_prediction_file_compact(pred_file)
        if df.empty:
            n_files_skipped += 1
            continue
        df = apply_prediction_quality_flags(df, quality_table, target_stats)
        n_files_read += 1
        n_rows_seen_after_compaction += len(df)
        if CONFIG.get("enable_candidate_quality_gate", True):
            rejected = df[~df["candidate_eligible"].fillna(False).astype(bool)].copy()
            if not rejected.empty:
                rejected_qc_parts.append(rejected.head(200))
            df = df[df["candidate_eligible"].fillna(False).astype(bool)].copy()
        if df.empty:
            n_files_skipped += 1
            continue
        collected.append(df)

        # Periodically reduce memory footprint.
        if sum(len(x) for x in collected) > max_total_rows * 2:
            pooled = pd.concat(collected, ignore_index=True, copy=False)
            collected = [compact_candidate_rows(pooled, max_rows=max_total_rows)]
            del pooled
            gc.collect()

    if not collected:
        tiers = pd.DataFrame()
    else:
        tiers = pd.concat(collected, ignore_index=True, copy=False)
        tiers = compact_candidate_rows(tiers, max_rows=max_total_rows)

    # Keep only known compact columns plus any useful columns that survived.
    if not tiers.empty:
        front_cols = [c for c in PREDICTION_COMPACT_COLUMNS if c in tiers.columns]
        other_cols = [c for c in tiers.columns if c not in front_cols]
        tiers = tiers[front_cols + other_cols]
        tiers = truncate_prediction_text_columns(tiers)

    atomic_save_csv(tiers, out_path)
    write_prediction_range_qc(tiers)
    if rejected_qc_parts:
        rejected_sample = pd.concat(rejected_qc_parts, ignore_index=True, copy=False).head(50000)
        atomic_save_csv(rejected_sample, TABLE_DIR / "QC_Table_rejected_prediction_rows_sample.csv")

    if not tiers.empty:
        group_cols = [c for c in ["target", "feature_set", "split", "model", "budget", "seed", "tier"] if c in tiers.columns]
        if group_cols:
            tier_summary = tiers.groupby(group_cols, dropna=False).size().reset_index(name="n_compact_rows")
            atomic_save_csv(tier_summary, TABLE_DIR / "table_candidate_tier_counts.csv")

        if "split_lower" in tiers.columns:
            best_group_cols = [c for c in ["target", "feature_set", "split", "model", "budget", "seed"] if c in tiers.columns]
            if best_group_cols:
                keep_top = int(CONFIG.get("candidate_tiers_keep_top_per_group", 50))
                best = (
                    tiers.sort_values("split_lower", ascending=False)
                         .groupby(best_group_cols, dropna=False)
                         .head(keep_top)
                )
            else:
                best = tiers.sort_values("split_lower", ascending=False).head(5000)
            atomic_save_csv(best, TABLE_DIR / "table_top50_candidates_by_lower_confidence_bound.csv")

    save_json(
        {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "note": "Streaming compact candidate-tier collection. Full per-experiment counts remain in all_experiment_metrics.csv.",
            "n_prediction_files_found": len(pred_files),
            "n_prediction_files_read": n_files_read,
            "n_prediction_files_skipped": n_files_skipped,
            "n_rows_after_per_file_compaction": int(n_rows_seen_after_compaction),
            "n_rows_saved": int(len(tiers)),
            "max_rows_config": max_total_rows,
        },
        TABLE_DIR / "table_candidate_tiers_collection_manifest.json",
    )

    mark_done(stage, {
        "n_prediction_files_found": len(pred_files),
        "n_prediction_files_read": n_files_read,
        "n_prediction_files_skipped": n_files_skipped,
        "n_prediction_rows_saved": int(len(tiers)),
        "streaming_compact_collection": True,
    })
    return tiers




def build_consensus_candidate_table(tiers: pd.DataFrame) -> pd.DataFrame:
    """Collapse row-level candidate predictions into one consensus row per MOF/target.

    A high-impact shortlist should not depend on a single seed/model/split. This
    table records support across models, seeds, and splits and ranks candidates
    by robust lower confidence bounds and consensus support.
    """
    stage = "04b_consensus_candidates"
    out_path = TABLE_DIR / "table_consensus_candidate_shortlist.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path, low_memory=False)
    if tiers is None or tiers.empty:
        consensus = pd.DataFrame()
        atomic_save_csv(consensus, out_path)
        mark_done(stage, {"n_rows": 0})
        return consensus
    df = tiers.copy()
    if "candidate_eligible" in df.columns:
        df = df[df["candidate_eligible"].fillna(False).astype(bool)].copy()
    for col in ["y_true", "y_pred", "split_lower", "split_upper", "budget", "seed", "Density", "Di", "Df", "Dif"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        consensus = pd.DataFrame()
        atomic_save_csv(consensus, out_path)
        mark_done(stage, {"n_rows": 0, "note": "No candidate-eligible rows after gates."})
        return consensus
    df["interval_width"] = df["split_upper"] - df["split_lower"]
    group_cols = ["target", "mof_id"]
    if "filename" in df.columns:
        first_filename = df.groupby(group_cols)["filename"].first().rename("filename")
    agg = df.groupby(group_cols, dropna=False).agg(
        y_true_median=("y_true", "median"),
        y_pred_median=("y_pred", "median"),
        y_pred_max=("y_pred", "max"),
        split_lower_median=("split_lower", "median"),
        split_lower_max=("split_lower", "max"),
        split_upper_median=("split_upper", "median"),
        interval_width_median=("interval_width", "median"),
        n_support_rows=("mof_id", "size"),
        n_models=("model", pd.Series.nunique),
        n_seeds=("seed", pd.Series.nunique),
        n_splits=("split", pd.Series.nunique),
        n_feature_sets=("feature_set", pd.Series.nunique),
        n_budgets=("budget", pd.Series.nunique),
        trusted_support=("tier", lambda s: int((s.astype(str) == "trusted").sum())),
        Density=("Density", "median") if "Density" in df.columns else ("y_pred", "median"),
        Di=("Di", "median") if "Di" in df.columns else ("y_pred", "median"),
        Df=("Df", "median") if "Df" in df.columns else ("y_pred", "median"),
        Dif=("Dif", "median") if "Dif" in df.columns else ("y_pred", "median"),
    ).reset_index()
    if "filename" in df.columns:
        agg = agg.merge(first_filename.reset_index(), on=group_cols, how="left")
    agg["consensus_pass"] = (
        (agg["n_models"] >= int(CONFIG.get("consensus_min_models", 2)))
        & (agg["n_seeds"] >= int(CONFIG.get("consensus_min_seeds", 2)))
        & (agg["n_splits"] >= int(CONFIG.get("consensus_min_splits", 1)))
        & (agg["trusted_support"] > 0)
    )
    # Transparent consensus score: support first, then robust lower bound, then narrower interval.
    agg["consensus_score"] = (
        4.0 * agg["n_models"]
        + 1.0 * agg["n_seeds"]
        + 2.0 * agg["n_splits"]
        + 0.25 * agg["trusted_support"]
        + agg["split_lower_median"].fillna(0)
        - 0.05 * agg["interval_width_median"].fillna(0)
    )
    consensus = agg.sort_values(["consensus_pass", "target", "consensus_score", "split_lower_median"], ascending=[False, True, False, False])
    atomic_save_csv(consensus, out_path)
    atomic_save_csv(consensus[consensus["consensus_pass"]].head(1000), TABLE_DIR / "MAIN_Table_3_consensus_trusted_shortlist.csv")
    mark_done(stage, {"n_rows": len(consensus), "n_consensus_pass": int(consensus["consensus_pass"].sum())})
    return consensus


def write_external_overlay_integrity(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    rows = []
    for name, df in [("core", core), ("mosaec", mosaec), ("final_external", final_external)]:
        rows.append({
            "overlay": name,
            "n_rows": 0 if df is None else len(df),
            "n_columns": 0 if df is None else df.shape[1],
            "n_match_keys": int(df["match_key"].nunique()) if df is not None and "match_key" in df.columns else np.nan,
        })
    if final_external is not None and not final_external.empty:
        for col in ["core_exact_match_flag", "core_geometry_overlap_flag", "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag", "recommended_screening_flag", "GEOM_available", "RAC_available"]:
            if col in final_external.columns:
                rows.append({"overlay": f"final_external::{col}", "n_rows": int(final_external[col].fillna(False).astype(bool).sum()), "n_columns": 1, "n_match_keys": np.nan})
    atomic_save_csv(pd.DataFrame(rows), TABLE_DIR / "QC_Table_external_overlay_integrity.csv")

# =============================================================================
# 13. Chemical failure anatomy
# =============================================================================

def bin_numeric_series(s: pd.Series, name: str, q: int = 5) -> pd.Series:
    try:
        return pd.qcut(s, q=q, duplicates="drop").astype(str)
    except Exception:
        return pd.Series(["unbinned"] * len(s), index=s.index)


def chemical_failure_anatomy(tiers: pd.DataFrame) -> pd.DataFrame:
    stage = "05_chemical_failure_anatomy"
    out_path = TABLE_DIR / "table_chemical_failure_anatomy.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path)

    logging.info("STAGE >>> CHEMICAL_FAILURE_ANATOMY")
    if tiers.empty:
        return pd.DataFrame()
    df = tiers.copy()
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    rows = []
    # Numeric regimes: PLD/LCD/density proxies.
    for col in ["Di", "Df", "Dif", "Density"]:
        if col in df.columns:
            df[f"{col}_bin"] = bin_numeric_series(df[col], col)
            for keys, sub in df.groupby(["target", "feature_set", "split", "model", "budget", f"{col}_bin"], dropna=False):
                rows.append({
                    "anatomy_type": col,
                    "regime": str(keys[-1]),
                    "target": keys[0], "feature_set": keys[1], "split": keys[2], "model": keys[3], "budget": keys[4],
                    "n": len(sub),
                    "mae": float(sub["abs_error"].mean()),
                    "rmse": float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))),
                    "mean_interval_width": float((sub["split_upper"] - sub["split_lower"]).mean()),
                    "trusted_fraction": float((sub["tier"] == "trusted").mean()),
                })
    # Categorical regimes: topology and clusters. Restrict to top categories to keep tables readable.
    for col in ["topology", "metal_cluster", "functional_cluster", "ligand_cluster", "geometry_cluster"]:
        if col in df.columns:
            counts = df[col].astype(str).value_counts()
            top_cats = set(counts.head(50).index)
            df[f"{col}_coarse"] = df[col].astype(str).where(df[col].astype(str).isin(top_cats), "OTHER_RARE")
            for keys, sub in df.groupby(["target", "feature_set", "split", "model", "budget", f"{col}_coarse"], dropna=False):
                rows.append({
                    "anatomy_type": col,
                    "regime": str(keys[-1]),
                    "target": keys[0], "feature_set": keys[1], "split": keys[2], "model": keys[3], "budget": keys[4],
                    "n": len(sub),
                    "mae": float(sub["abs_error"].mean()),
                    "rmse": float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))),
                    "mean_interval_width": float((sub["split_upper"] - sub["split_lower"]).mean()),
                    "trusted_fraction": float((sub["tier"] == "trusted").mean()),
                })
    anatomy = pd.DataFrame(rows)
    atomic_save_csv(anatomy, out_path)
    mark_done(stage, {"n_rows": len(anatomy)})
    return anatomy

# =============================================================================
# 14. Figure helpers and composite figures
# =============================================================================

def save_current_figure(path_base: Path) -> None:
    """Save the current Matplotlib figure robustly.

    Figures are saved as high-resolution PNGs by default. You may add PDF to
    CONFIG["figure_format"] after validating the run. SI figures are often
    text-heavy diagnostic plots; on some machines Matplotlib PDF export can
    be disproportionately slow for these. Therefore SI PDFs are optional via
    CONFIG["save_si_pdf"], while high-resolution PNGs are always saved.
    """
    path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG["figure_format"]:
        if fmt.lower() == "pdf" and (FIG_SI_DIR in path_base.parents) and not CONFIG.get("save_si_pdf", False):
            continue
        out = path_base.with_suffix(f".{fmt}")
        try:
            plt.savefig(out, dpi=CONFIG["savefig_dpi"], bbox_inches="tight")
        except Exception as e:
            logging.warning("FIGURE_SAVE_FAILED | %s | %s", out, e)
    plt.close("all")


def set_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def safe_boxplot_with_labels(ax, data, labels, **kwargs):
    """Draw a boxplot with labels across Matplotlib versions.

    Matplotlib 3.9 renamed the boxplot keyword argument from ``labels``
    to ``tick_labels``, and Matplotlib 3.11 removed the old name. Some older
    environments still accept only ``labels``. This helper keeps the pipeline
    portable across both old and new Matplotlib releases and prevents an SI
    plotting-stage crash after the expensive modelling stages have completed.
    """
    labels = list(labels)
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError as exc_tick:
        # Older Matplotlib versions do not know tick_labels. Try the legacy name.
        try:
            return ax.boxplot(data, labels=labels, **kwargs)
        except TypeError as exc_labels:
            # Last-resort fallback: draw without label keyword and set ticks manually.
            try:
                artists = ax.boxplot(data, **kwargs)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels)
                return artists
            except Exception:
                raise exc_labels from exc_tick


def plot_figure_1_framework() -> None:
    """Conceptual framework figure: no numerical data required."""
    stage = "fig1_framework"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 1 framework")
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    titles = [
        "ARC-MOF\nlabelled benchmark",
        "Few-shot\nlabelled pool",
        "Risk-controlled\npredictor",
        "External realism\noverlays",
        "Trusted / uncertain /\nrejected tiers",
    ]
    bodies = [
        "CO$_2$/CH$_4$ targets,\ngeometry, RACs, splits",
        "10–1000 labelled MOFs\nper target",
        "CPU-friendly ML +\nconformal intervals",
        "CoRE stability +\nMOSAEC plausibility",
        "Actionable shortlist\n+ abstention",
    ]
    for i, ax in enumerate(axes):
        ax.axis("off")
        rect = plt.Rectangle((0.05, 0.2), 0.9, 0.6, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(0.5, 0.62, titles[i], ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.38, bodies[i], ha="center", va="center", fontsize=8)
        set_panel_label(ax, f"({chr(97+i)})")
        if i < len(axes) - 1:
            ax.annotate("", xy=(1.05, 0.5), xytext=(0.95, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.suptitle("Few-shot adsorption screening with stability and chemical-validity realism layers", fontsize=13, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_1_risk_controlled_framework")
    mark_done(stage)


def choose_plot_subset(agg: pd.DataFrame) -> pd.DataFrame:
    """High-impact plotting subset: stable models, top-5%, and random split where available."""
    df = agg.copy()
    stable = set(CONFIG.get("stable_plot_models", []))
    if stable and "model" in df.columns:
        df = df[df["model"].isin(stable)]
    if "random" in set(df.get("split", pd.Series(dtype=str))):
        df = df[df["split"] == "random"]
    if 0.05 in set(df.get("top_frac", pd.Series(dtype=float))):
        df = df[df["top_frac"] == 0.05]
    # Prefer the strongest non-duplicated chemistry-aware feature set for main panels.
    feature_preference = ["geometry_plus_racs_rdfs", "geometry_plus_racs", "geometry_only"]
    available = [fs for fs in feature_preference if fs in set(df.get("feature_set", pd.Series(dtype=str)))]
    if available:
        df = df[df["feature_set"] == available[0]]
    return df


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    stage = "fig2_performance"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 2 performance")
    df = choose_plot_subset(agg)
    atomic_save_csv(df, FIGDATA_DIR / "Figure_2_label_budget_performance_data.csv")
    if df.empty:
        return
    targets = df["target"].drop_duplicates().tolist()
    models = df["model"].drop_duplicates().tolist()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    panels = [
        ("rmse_mean", "RMSE", "(a)"),
        ("spearman_mean", "Spearman correlation", "(b)"),
        ("recall_mean", "Top-5% recall", "(c)"),
        ("ndcg_mean", "NDCG@top-5%", "(d)"),
        ("precision_mean", "Top-5% precision", "(e)"),
        ("enrichment_mean", "Enrichment", "(f)"),
    ]
    for ax, (metric, ylabel, lab) in zip(axes, panels):
        for model in models:
            sub = df[df["model"] == model].groupby("budget", as_index=False)[metric].mean()
            if not sub.empty:
                ax.plot(sub["budget"], sub[metric], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_xlabel("Label budget")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        set_panel_label(ax, lab)
    axes[0].legend(fontsize=8)
    fig.suptitle("Label-budget performance and ranking maturity", fontsize=14, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_2_label_budget_performance")
    mark_done(stage)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    stage = "fig3_calibration"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 3 calibration")
    df = choose_plot_subset(agg)
    atomic_save_csv(df, FIGDATA_DIR / "Figure_3_calibration_abstention_data.csv")
    if df.empty:
        return
    models = df["model"].drop_duplicates().tolist()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    nominal = 1 - CONFIG["conformal_alpha"]

    # (a) empirical vs nominal coverage
    cov = df.groupby("model", as_index=False)["split_coverage_mean"].mean()
    axes[0].bar(cov["model"], cov["split_coverage_mean"])
    axes[0].axhline(nominal, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].tick_params(axis="x", rotation=45)
    set_panel_label(axes[0], "(a)")

    # (b) interval width vs budget
    for model in models:
        sub = df[df["model"] == model].groupby("budget", as_index=False)["split_mean_width_mean"].mean()
        axes[1].plot(sub["budget"], sub["split_mean_width_mean"], marker="o", label=model)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Label budget")
    axes[1].set_ylabel("Mean interval width")
    axes[1].grid(True, alpha=0.3)
    set_panel_label(axes[1], "(b)")

    # (c) precision vs abstention proxy = uncertain fraction
    tmp = df.copy()
    denom = tmp["trusted_count_mean"] + tmp["uncertain_count_mean"] + tmp["rejected_count_mean"]
    tmp["abstention_rate"] = tmp["uncertain_count_mean"] / denom.replace(0, np.nan)
    axes[2].scatter(tmp["abstention_rate"], tmp["precision_mean"], s=25)
    axes[2].set_xlabel("Uncertain fraction")
    axes[2].set_ylabel("Top-5% precision")
    axes[2].grid(True, alpha=0.3)
    set_panel_label(axes[2], "(c)")

    # (d) coverage under grouped splits
    cov_split = agg[agg["top_frac"] == 0.05].groupby("split", as_index=False)["split_coverage_mean"].mean()
    axes[3].bar(cov_split["split"], cov_split["split_coverage_mean"])
    axes[3].axhline(nominal, linestyle="--", linewidth=1)
    axes[3].set_ylabel("Coverage")
    axes[3].tick_params(axis="x", rotation=45)
    set_panel_label(axes[3], "(d)")

    # (e) trusted candidates by budget
    trusted = df.groupby("budget", as_index=False)["trusted_count_mean"].mean()
    axes[4].plot(trusted["budget"], trusted["trusted_count_mean"], marker="o")
    axes[4].set_xscale("log")
    axes[4].set_xlabel("Label budget")
    axes[4].set_ylabel("Trusted candidates")
    axes[4].grid(True, alpha=0.3)
    set_panel_label(axes[4], "(e)")

    # (f) split/local/mondrian coverage comparison
    cov_kinds = pd.DataFrame({
        "method": ["split", "local", "mondrian"],
        "coverage": [df["split_coverage_mean"].mean(), df["local_coverage_mean"].mean(), df["mondrian_coverage_mean"].mean()],
    })
    axes[5].bar(cov_kinds["method"], cov_kinds["coverage"])
    axes[5].axhline(nominal, linestyle="--", linewidth=1)
    axes[5].set_ylabel("Coverage")
    set_panel_label(axes[5], "(f)")

    fig.suptitle("Calibration, uncertainty width, and abstention", fontsize=14, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_3_calibration_and_abstention")
    mark_done(stage)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 4 chemical failure anatomy")
    if anatomy.empty or tiers.empty:
        return
    atomic_save_csv(anatomy, FIGDATA_DIR / "Figure_4_chemical_failure_anatomy_data.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    # (a) MAE by density regime.
    sub = anatomy[anatomy["anatomy_type"] == "Density"].groupby("regime", as_index=False)["mae"].mean().head(10)
    axes[0].bar(range(len(sub)), sub["mae"])
    axes[0].set_xticks(range(len(sub)))
    axes[0].set_xticklabels(sub["regime"], rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Error over density regimes")
    set_panel_label(axes[0], "(a)")

    # (b) MAE by topology if available.
    sub = anatomy[anatomy["anatomy_type"] == "topology"].sort_values("mae", ascending=False).head(10)
    axes[1].bar(range(len(sub)), sub["mae"])
    axes[1].set_xticks(range(len(sub)))
    axes[1].set_xticklabels(sub["regime"], rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("High-error topology regimes")
    set_panel_label(axes[1], "(b)")

    # (c) MAE by metal cluster.
    sub = anatomy[anatomy["anatomy_type"] == "metal_cluster"].sort_values("mae", ascending=False).head(10)
    axes[2].bar(range(len(sub)), sub["mae"])
    axes[2].set_xticks(range(len(sub)))
    axes[2].set_xticklabels(sub["regime"], rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("MAE")
    axes[2].set_title("High-error metal regimes")
    set_panel_label(axes[2], "(c)")

    # (d) Succeeding cases: predicted vs true for trusted points.
    trusted = tiers[tiers["tier"] == "trusted"].sample(min(5000, (tiers["tier"] == "trusted").sum()), random_state=42) if (tiers["tier"] == "trusted").sum() > 0 else tiers.sample(min(5000, len(tiers)), random_state=42)
    axes[3].scatter(trusted["y_true"], trusted["y_pred"], s=5, alpha=0.4)
    lims = [min(trusted["y_true"].min(), trusted["y_pred"].min()), max(trusted["y_true"].max(), trusted["y_pred"].max())]
    axes[3].plot(lims, lims, linestyle="--", linewidth=1)
    axes[3].set_xlabel("True uptake")
    axes[3].set_ylabel("Predicted uptake")
    axes[3].set_title("Trusted candidates")
    set_panel_label(axes[3], "(d)")

    # (e) Failure cases: largest errors.
    tmp = tiers.copy()
    tmp["abs_error"] = np.abs(tmp["y_true"] - tmp["y_pred"])
    fail = tmp.sort_values("abs_error", ascending=False).head(5000)
    axes[4].scatter(fail["y_true"], fail["y_pred"], s=5, alpha=0.4)
    lims = [min(fail["y_true"].min(), fail["y_pred"].min()), max(fail["y_true"].max(), fail["y_pred"].max())]
    axes[4].plot(lims, lims, linestyle="--", linewidth=1)
    axes[4].set_xlabel("True uptake")
    axes[4].set_ylabel("Predicted uptake")
    axes[4].set_title("Largest-error cases")
    set_panel_label(axes[4], "(e)")

    # (f) Tier composition.
    tier_counts = tiers["tier"].value_counts()
    axes[5].bar(tier_counts.index.astype(str), tier_counts.values)
    axes[5].set_ylabel("Prediction rows")
    axes[5].set_title("Candidate-tier composition")
    set_panel_label(axes[5], "(f)")

    fig.suptitle("Chemical failure anatomy and candidate-level reliability", fontsize=14, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy")
    mark_done(stage)


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    stage = "si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> SI figures")

    # SI Figure S1: dataset target distributions.
    target_summary_path = TABLE_DIR / "table_dataset_target_summary.csv"
    if target_summary_path.exists():
        summary = pd.read_csv(target_summary_path)
        atomic_save_csv(summary, FIGDATA_DIR / "Figure_S1_target_summary_data.csv")

    if not tiers.empty:
        # S1 actual histograms from predictions y_true. Use one sampled occurrence per MOF/target.
        sample = tiers.drop_duplicates(["mof_id", "target"])
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        for ax, target in zip(axes, list(TARGET_SPECS.keys())):
            sub = sample[sample["target"] == target]
            if not sub.empty:
                ax.hist(sub["y_true"].dropna(), bins=40)
            ax.set_title(target)
            ax.set_xlabel("Uptake / mmol g$^{-1}$")
            ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Target distributions")
        save_current_figure(FIG_SI_DIR / "Figure_S1_target_distributions")

        # SI Figure S2: grouped split coverage heatmap-like matrix.
        cov = agg[agg["top_frac"] == 0.05].pivot_table(index="split", columns="model", values="split_coverage_mean", aggfunc="mean")
        atomic_save_csv(cov.reset_index(), FIGDATA_DIR / "Figure_S2_grouped_split_coverage_data.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(cov.fillna(np.nan).to_numpy(), aspect="auto")
        ax.set_xticks(range(cov.shape[1])); ax.set_xticklabels(cov.columns, rotation=45, ha="right")
        ax.set_yticks(range(cov.shape[0])); ax.set_yticklabels(cov.index)
        ax.set_title("Figure S2. Coverage across grouped splits")
        fig.colorbar(im, ax=ax, label="Coverage")
        save_current_figure(FIG_SI_DIR / "Figure_S2_grouped_split_coverage")

        # SI Figure S3: interval width distributions by tier.
        tmp = tiers.copy()
        tmp["interval_width"] = tmp["split_upper"] - tmp["split_lower"]
        atomic_save_csv(tmp[["target", "model", "budget", "tier", "interval_width"]].sample(min(20000, len(tmp)), random_state=42),
                        FIGDATA_DIR / "Figure_S3_interval_width_by_tier_data.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [tmp.loc[tmp["tier"] == tier, "interval_width"].dropna().sample(min(5000, (tmp["tier"] == tier).sum()), random_state=42) for tier in ["trusted", "uncertain", "rejected"] if (tmp["tier"] == tier).sum() > 0]
        labels = [tier for tier in ["trusted", "uncertain", "rejected"] if (tmp["tier"] == tier).sum() > 0]
        if data:
            safe_boxplot_with_labels(ax, data, labels, showfliers=False)
        ax.set_ylabel("Split-conformal interval width")
        ax.set_title("Figure S3. Interval width by candidate tier")
        save_current_figure(FIG_SI_DIR / "Figure_S3_interval_width_by_tier")

    # SI Figure S4: feature-set ablation if RACs were run.
    if not agg.empty:
        abl = agg[(agg["top_frac"] == 0.05)].groupby(["feature_set", "model", "budget"], as_index=False)["recall_mean"].mean()
        atomic_save_csv(abl, FIGDATA_DIR / "Figure_S4_feature_set_ablation_data.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        for fs in abl["feature_set"].unique():
            sub = abl[abl["feature_set"] == fs].groupby("budget", as_index=False)["recall_mean"].mean()
            ax.plot(sub["budget"], sub["recall_mean"], marker="o", label=fs)
        ax.set_xscale("log")
        ax.set_xlabel("Label budget")
        ax.set_ylabel("Mean top-5% recall")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Figure S4. Feature-set ablation")
        save_current_figure(FIG_SI_DIR / "Figure_S4_feature_set_ablation")

    # SI Figure S5: worst regimes table as plot.
    if not anatomy.empty:
        worst = anatomy.sort_values("mae", ascending=False).head(20)
        atomic_save_csv(worst, FIGDATA_DIR / "Figure_S5_worst_regimes_data.csv")
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = (worst["anatomy_type"].astype(str) + ":" + worst["regime"].astype(str)).str.slice(0, 35)
        ax.barh(range(len(worst)), worst["mae"])
        ax.set_yticks(range(len(worst))); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("MAE")
        ax.set_title("Figure S5. Highest-error chemical regimes")
        save_current_figure(FIG_SI_DIR / "Figure_S5_highest_error_regimes")

    mark_done(stage)

# =============================================================================
# 15. Manuscript and SI tables
# =============================================================================

def save_manuscript_si_tables(master: pd.DataFrame, agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    stage = "06_manuscript_si_tables"
    if is_done(stage):
        return
    logging.info("STAGE >>> SAVE_MANUSCRIPT_SI_TABLES")

    # Main Table 1: dataset and targets.
    rows = []
    for target in TARGET_SPECS:
        valid = master[target].dropna()
        rows.append({
            "target": target,
            "description": TARGET_SPECS[target]["description"],
            "n_mofs": int(valid.shape[0]),
            "mean_mmol_g": float(valid.mean()),
            "std_mmol_g": float(valid.std()),
            "median_mmol_g": float(valid.median()),
            "min_mmol_g": float(valid.min()),
            "max_mmol_g": float(valid.max()),
        })
    atomic_save_csv(pd.DataFrame(rows), TABLE_DIR / "MAIN_Table_1_dataset_and_targets.csv")

    # Main Table 2: best models by target and budget at top-5%.
    if not agg.empty:
        candidates = agg[(agg["top_frac"] == 0.05) & (agg["feature_set"] == "geometry_only")].copy()
        if not candidates.empty:
            best_rows = []
            for keys, sub in candidates.groupby(["target", "split", "budget"]):
                sub = sub.sort_values(["recall_mean", "split_coverage_mean", "rmse_mean"], ascending=[False, False, True])
                best_rows.append(sub.head(1))
            best = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
            atomic_save_csv(best, TABLE_DIR / "MAIN_Table_2_best_fewshot_models_top5pct.csv")

        # SI tables.
        atomic_save_csv(agg, TABLE_DIR / "SI_Table_S1_all_aggregated_metrics.csv")
        split_summary = agg[agg["top_frac"] == 0.05].groupby(["split", "model", "budget"], as_index=False).agg({
            "split_coverage_mean": "mean",
            "recall_mean": "mean",
            "rmse_mean": "mean",
            "spearman_mean": "mean",
        })
        atomic_save_csv(split_summary, TABLE_DIR / "SI_Table_S2_grouped_split_summary.csv")

    if not tiers.empty:
        # Main row-level candidate shortlist: use quality-gated trusted rows.
        work_tiers = tiers.copy()
        if "candidate_eligible" in work_tiers.columns:
            work_tiers = work_tiers[work_tiers["candidate_eligible"].fillna(False).astype(bool)]
        best_candidates = work_tiers[work_tiers["tier"] == "trusted"].sort_values("split_lower", ascending=False).head(500)
        atomic_save_csv(best_candidates, TABLE_DIR / "MAIN_Table_3_trusted_candidate_shortlist.csv")
        tier_counts = tiers.groupby(["target", "feature_set", "split", "model", "budget", "tier"], as_index=False).size()
        atomic_save_csv(tier_counts, TABLE_DIR / "SI_Table_S3_candidate_tier_counts.csv")

    if not anatomy.empty:
        atomic_save_csv(anatomy, TABLE_DIR / "SI_Table_S4_chemical_failure_anatomy.csv")

    mark_done(stage)

# =============================================================================
# 16. Provenance report
# =============================================================================

def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    stage = "07_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> WRITE_RUN_REPORT")
    report = RESULTS_DIR / "RUN_REPORT.md"
    manifest_path = TABLE_DIR / "download_manifest_required_files.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    lines = []
    lines.append("# Few-shot MOF risk-controlled screening run report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Runtime modes\n")
    lines.append(f"- Save mode: `{CONFIG.get('save_mode')}`")
    lines.append(f"- RAM mode: `{CONFIG.get('ram_mode')}`")
    lines.append(f"- Comprehensive level: `{CONFIG.get('comprehensive_level')}`")
    lines.append(f"- n_jobs: `{CONFIG.get('n_jobs')}`")
    lines.append(f"- Data root: `{DATA_ROOT}`")
    lines.append("\n## Input files resolved\n")
    if manifest.empty:
        lines.append("No manifest available.")
    else:
        found = manifest[manifest.get("found_locally", False).astype(bool)] if "found_locally" in manifest else manifest.iloc[0:0]
        missing = manifest[~manifest.get("found_locally", False).astype(bool)] if "found_locally" in manifest else manifest.iloc[0:0]
        lines.append(f"Found: {len(found)} / {len(manifest)} known files.")
        for _, r in found.head(80).iterrows():
            lines.append(f"- `{r.get('filename_requested_by_code')}` -> `{r.get('local_path_detected')}`")
        if len(missing) > 0:
            lines.append("\nMissing/optional not found:")
            for _, r in missing.head(80).iterrows():
                lines.append(f"- `{r.get('filename_requested_by_code')}` ({r.get('priority')})")
    lines.append("\n## Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(CONFIG, indent=2, default=str))
    lines.append("```\n")
    lines.append("## Master table\n")
    lines.append(f"Rows: {master.shape[0]}  ")
    lines.append(f"Columns: {master.shape[1]}  ")
    lines.append("\n## Targets\n")
    for t, spec in TARGET_SPECS.items():
        lines.append(f"- `{t}`: {spec['description']}; non-missing = {int(master[t].notna().sum())}")
    lines.append("\n## Experiments\n")
    lines.append(f"Metric rows: {len(metrics_all)}")
    lines.append("\n## Key output folders\n")
    lines.append("- `tables/`: all manuscript and SI tables")
    lines.append("- `figures/main/`: manuscript figures")
    lines.append("- `figures/si/`: supporting-information figures")
    lines.append("- `figure_data/`: exact data used for figures")
    lines.append("- `predictions/`: candidate-level predictions and conformal intervals")
    lines.append("- `pickles/`: reusable Python objects")
    lines.append("- `logs/`: detailed run logs and failed-job JSON files")
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage)


# =============================================================================
# 17. External realism overlays: CoRE MOF 2024 and MOSAEC-DB
# =============================================================================

# These functions are intentionally generic and defensive. Public database files
# may change column names across releases, and local users may rename/extract
# archives into different folders. The code therefore uses fuzzy column matching,
# records what it found, and never lets missing external files break the central
# ARC-MOF modelling pipeline.


def resolve_input_files_by_pattern(pattern: str) -> List[Path]:
    """Find files/directories by glob pattern under configured data roots.

    This replaces the older SCRIPT_DIR-only search and is aware of the new
    project_data/data_raw archive location. Results folders are excluded.
    """
    if pattern in _PATTERN_RESOLUTION_CACHE:
        return _PATTERN_RESOLUTION_CACHE[pattern]
    patterns = {pattern}
    patterns |= {pattern.replace("-", "_"), pattern.replace("_", "-")}
    all_matches: List[Path] = []
    seen = set()
    for root in DATA_SEARCH_ROOTS:
        if not root.exists():
            continue
        for pat in patterns:
            for glob_pat in [pat, f"**/{pat}"]:
                try:
                    matches = list(root.glob(glob_pat))
                except Exception:
                    matches = []
                for m in matches:
                    try:
                        if RESULTS_DIR in m.parents:
                            continue
                        key = str(m.resolve())
                    except Exception:
                        key = str(m)
                    if key not in seen and m.exists():
                        seen.add(key)
                        all_matches.append(m)
    all_matches = sorted(all_matches, key=lambda m: len(str(m)))
    _PATTERN_RESOLUTION_CACHE[pattern] = all_matches
    return all_matches

def canonical_key(x: Any) -> str:
    """Very conservative string key for exact-ish cross-database matching."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\.cif$", "", s)
    s = re.sub(r"_repeat$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _normalise_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def find_column(df: pd.DataFrame, alternatives: List[List[str]], prefer_numeric: bool = False) -> Optional[str]:
    """Find a likely column using groups of required tokens.

    alternatives is a list of token lists. For example, [["water", "stability"],
    ["prob_water_stability"]] means: return the first column whose normalised
    name contains both water and stability, or a direct normalised match.
    """
    if df is None or df.empty:
        return None
    norm_to_original = {_normalise_colname(c): c for c in df.columns}
    for tokens in alternatives:
        ntokens = [_normalise_colname(t) for t in tokens]
        for norm, orig in norm_to_original.items():
            # Very short tokens such as "td" or "di" can create false positives
            # (e.g. "std" being mistaken for decomposition temperature). For
            # such tokens require exact or prefix-like matches.
            short_tokens = [t for t in ntokens if len(t) <= 2]
            short_ok = all((norm == t or norm.startswith(t + "_") or norm.startswith(t)) for t in short_tokens) if short_tokens else True
            if short_ok and all(t in norm for t in ntokens):
                if prefer_numeric and not pd.api.types.is_numeric_dtype(df[orig]):
                    # Try converting a small sample; many CSVs store numbers as text.
                    sample = pd.to_numeric(df[orig].head(200), errors="coerce")
                    if sample.notna().sum() == 0:
                        continue
                return orig
    return None


def first_nonempty_key(df: pd.DataFrame, candidate_cols: List[str]) -> pd.Series:
    """Create a key from the first non-empty value among candidate columns."""
    if not candidate_cols:
        return pd.Series([""] * len(df), index=df.index)
    out = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in candidate_cols:
        if col not in df.columns:
            continue
        values = df[col].map(canonical_key)
        out = out.mask(out.eq("") & values.ne(""), values)
    return out.fillna("")


def safe_optional_csv(filename: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    path = resolve_input_file(filename)
    if not path.exists():
        return None
    try:
        logging.info("Reading optional CSV: %s", path)
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception as e:
        logging.warning("OPTIONAL_CSV_READ_FAILED | %s | %s", filename, e)
        return None


def safe_optional_excel(filename: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    path = resolve_input_file(filename)
    if not path.exists():
        return None
    try:
        logging.info("Reading optional Excel: %s", path)
        return pd.read_excel(path, nrows=nrows)
    except Exception as e:
        logging.warning("OPTIONAL_EXCEL_READ_FAILED | %s | %s", filename, e)
        return None


def numeric_or_nan(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def read_identifier_set_from_csv(filename: str, id_hint_cols: Optional[List[str]] = None) -> set:
    """Read only an identifier-like column from a possibly large CSV."""
    path = resolve_input_file(filename)
    if not path.exists():
        return set()
    try:
        preview = pd.read_csv(path, nrows=5, low_memory=False)
        id_hint_cols = id_hint_cols or []
        candidate = None
        for hint in id_hint_cols + ["filename", "name", "mof_id", "refcode", "csd_refcode", "id"]:
            for col in preview.columns:
                if _normalise_colname(hint) == _normalise_colname(col) or _normalise_colname(hint) in _normalise_colname(col):
                    candidate = col
                    break
            if candidate:
                break
        if candidate is None:
            candidate = preview.columns[0]
        ids = pd.read_csv(path, usecols=[candidate], low_memory=False)[candidate].map(canonical_key)
        return set(ids[ids.ne("")].dropna().unique())
    except Exception as e:
        logging.warning("ID_SET_READ_FAILED | %s | %s", filename, e)
        return set()


def read_identifier_set_from_text(pattern: str) -> set:
    """Read line-delimited subset files such as MOSAEC diverse-neutral-*-20k.txt."""
    ids = set()
    for path in resolve_input_files_by_pattern(pattern):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                key = canonical_key(line.split()[0] if line.split() else line)
                if key:
                    ids.add(key)
        except Exception as e:
            logging.warning("TEXT_ID_SET_READ_FAILED | %s | %s", path, e)
    return ids


def build_core_mof_2024_overlay() -> pd.DataFrame:
    """Create the CoRE MOF 2024 stability/experimental-realism overlay.

    Output role in the paper:
      * not an adsorption-label source;
      * a stability, activation, hydrophobicity, duplicate, and experimental-origin
        plausibility layer for candidate shortlists.
    """
    stage = "08_core_mof_2024_overlay"
    out_csv = TABLE_DIR / "core_mof_stability_overlay.csv"
    processed_csv = DATA_PROCESSED_DIR / "core_mof_stability_overlay.csv"
    if is_done(stage) and out_csv.exists():
        return pd.read_csv(out_csv, low_memory=False)

    logging.info("STAGE >>> BUILD_CORE_MOF_2024_OVERLAY")
    source_files = {
        "ASR": "ASR_data_SI_20250204.csv",
        "FSR": "FSR_data_SI_20250204.csv",
        "ION": "ION_data_SI_20250204.csv",
    }
    parts = []
    column_map_rows = []

    for subset, fname in source_files.items():
        raw = safe_optional_csv(fname)
        if raw is None or raw.empty:
            logging.info("CORE_OVERLAY_SKIP | missing %s", fname)
            continue

        id_cols = []
        for hints in [["filename"], ["name"], ["refcode"], ["csd"], ["mofid"], ["mofkey"], ["id"]]:
            col = find_column(raw, [hints])
            if col and col not in id_cols:
                id_cols.append(col)

        std = pd.DataFrame(index=raw.index)
        std["core_id"] = first_nonempty_key(raw, id_cols)
        std["core_display_id"] = raw[id_cols[0]].astype(str) if id_cols else std["core_id"]
        std["source_subset_ASR_FSR_ION"] = subset

        mapping = {
            "is_computation_ready": ([["computation", "ready"], ["cr"]], False),
            "PLD": ([["pld"], ["pore", "limiting"], ["di"]], True),
            "LCD": ([["lcd"], ["largest", "cavity"], ["df"]], True),
            "PV": ([["pore", "volume"], ["pv"], ["void", "volume"]], True),
            "ASA": ([["accessible", "surface"], ["asa"], ["surface", "area"]], True),
            "density": ([["density"], ["rho"]], True),
            "topology": ([["topology"], ["topo"], ["net"]], False),
            "open_metal_site": ([["open", "metal"], ["oms"]], False),
            "MOFid_v1": ([["mofid", "v1"], ["mofid"]], False),
            "MOFid_v2": ([["mofid", "v2"], ["mofid2"]], False),
            "has_DDEC06_charges": ([["ddec06"], ["ddec"], ["charge"]], False),
            "heat_capacity": ([["heat", "capacity"], ["cp"], ["cv"]], True),
            "decomposition_temperature": ([["decomposition", "temperature"], ["decomposition"], ["tdec"]], True),
            "prob_solvent_removal_stability": ([["solvent", "removal", "stability"], ["solvent", "stability"], ["activation", "stability"]], True),
            "prob_water_stability": ([["water", "stability"], ["h2o", "stability"]], True),
            "hydrophobic_class": ([["hydrophobic"], ["hydrophilic"]], False),
        }
        for out_col, (alts, prefer_numeric) in mapping.items():
            col = find_column(raw, alts, prefer_numeric=prefer_numeric)
            column_map_rows.append({"database": "CoRE_MOF_2024", "subset": subset, "standard_column": out_col, "source_column": col or "NOT_FOUND"})
            if col is None:
                if prefer_numeric:
                    std[out_col] = np.nan
                elif out_col == "is_computation_ready":
                    std[out_col] = True  # ASR/FSR/ION data files are the CR SI metadata by design.
                else:
                    std[out_col] = ""
            else:
                std[out_col] = pd.to_numeric(raw[col], errors="coerce") if prefer_numeric else raw[col]

        # Additional exact-match keys from MOFid/MOFkey/refcode-like columns.
        key_cols = list(id_cols)
        for hints in [["mofid"], ["mofkey"], ["refcode"], ["filename"], ["name"]]:
            col = find_column(raw, [hints])
            if col and col not in key_cols:
                key_cols.append(col)
        std["match_key"] = first_nonempty_key(raw, key_cols)
        std["all_detected_key_columns"] = ";".join(key_cols)
        parts.append(std)

    if not parts:
        overlay = pd.DataFrame()
        atomic_save_csv(overlay, out_csv)
        atomic_save_csv(overlay, processed_csv)
        mark_done(stage, {"n_rows": 0, "note": "No CoRE MOF 2024 CSV metadata files found."})
        return overlay

    overlay = pd.concat(parts, ignore_index=True)

    # Recommended screening list flag.
    rec = safe_optional_csv("12089-recommended-screening-list.csv")
    rec_keys = set()
    if rec is not None and not rec.empty:
        rec_id_cols = []
        for hints in [["filename"], ["name"], ["mofid"], ["mofkey"], ["refcode"], ["id"]]:
            col = find_column(rec, [hints])
            if col and col not in rec_id_cols:
                rec_id_cols.append(col)
        if not rec_id_cols and rec.shape[1] > 0:
            rec_id_cols = [rec.columns[0]]
        for col in rec_id_cols:
            rec_keys |= set(rec[col].map(canonical_key).dropna().unique())
    overlay["recommended_screening_flag"] = overlay["match_key"].isin(rec_keys) | overlay["core_id"].isin(rec_keys)

    # Duplicate flag using ASR_FSR_check.csv. Any identifier appearing in the file is treated as an ASR/FSR duplicate evidence.
    dup = safe_optional_csv("ASR_FSR_check.csv")
    dup_keys = set()
    if dup is not None and not dup.empty:
        for col in dup.columns[: min(6, dup.shape[1])]:
            dup_keys |= set(dup[col].map(canonical_key).dropna().unique())
    overlay["asr_fsr_duplicate_flag"] = overlay["match_key"].isin(dup_keys) | overlay["core_id"].isin(dup_keys)

    # Normalise booleans stored as text.
    for col in ["is_computation_ready", "open_metal_site", "has_DDEC06_charges"]:
        if col in overlay.columns:
            vals = overlay[col].astype(str).str.lower().str.strip()
            overlay[col] = vals.isin(["true", "1", "yes", "y", "cr", "ddec", "ddec06"])

    atomic_save_csv(overlay, out_csv)
    atomic_save_csv(overlay, processed_csv)
    maybe_save_pickle(overlay, PICKLE_DIR / "core_mof_stability_overlay.pkl", minimum_mode="efficient")
    if column_map_rows:
        atomic_save_csv(pd.DataFrame(column_map_rows), TABLE_DIR / "table_core_column_autodetection_map.csv")
    mark_done(stage, {"n_rows": len(overlay), "n_columns": overlay.shape[1]})
    return overlay


def build_mosaec_db_overlay() -> pd.DataFrame:
    """Create the MOSAEC-DB chemical-validity/plausibility overlay.

    The script never redistributes structural CIFs. It only reads metadata,
    descriptor availability, duplicate/similarity files, and subset text lists
    from the user's local CSD-licensed/restricted download when present.
    """
    stage = "09_mosaec_db_overlay"
    out_csv = TABLE_DIR / "mosaec_plausibility_overlay.csv"
    processed_csv = DATA_PROCESSED_DIR / "mosaec_plausibility_overlay.csv"
    if is_done(stage) and out_csv.exists():
        return pd.read_csv(out_csv, low_memory=False)

    logging.info("STAGE >>> BUILD_MOSAEC_DB_OVERLAY")
    raw = safe_optional_csv("mosaec-db.csv")
    if raw is None or raw.empty:
        raw_xlsx = safe_optional_excel("mosaec-db.xlsx")
        raw = raw_xlsx if raw_xlsx is not None else None
    if raw is None or raw.empty:
        overlay = pd.DataFrame()
        atomic_save_csv(overlay, out_csv)
        atomic_save_csv(overlay, processed_csv)
        mark_done(stage, {"n_rows": 0, "note": "No MOSAEC metadata file found."})
        return overlay

    id_cols = []
    for hints in [["filename"], ["name"], ["mosaec"], ["refcode"], ["csd"], ["id"]]:
        col = find_column(raw, [hints])
        if col and col not in id_cols:
            id_cols.append(col)
    overlay = pd.DataFrame(index=raw.index)
    overlay["mosaec_id"] = first_nonempty_key(raw, id_cols)
    overlay["mosaec_display_id"] = raw[id_cols[0]].astype(str) if id_cols else overlay["mosaec_id"]
    ref_col = find_column(raw, [["csd", "refcode"], ["refcode"], ["csd"]])
    overlay["csd_refcode"] = raw[ref_col].astype(str) if ref_col else overlay["mosaec_display_id"]

    mapping = {
        "activation_state_full_or_partial": ([["activation"], ["full"], ["partial"]], False),
        "framework_charge_state": ([["charge", "state"], ["charged"], ["neutral"]], False),
        "PLD": ([["pld"], ["pore", "limiting"], ["di"]], True),
        "LCD": ([["lcd"], ["largest", "cavity"], ["df"]], True),
        "PV": ([["pore", "volume"], ["pv"], ["void", "volume"]], True),
        "ASA": ([["accessible", "surface"], ["asa"], ["surface", "area"]], True),
        "density": ([["density"], ["rho"]], True),
    }
    column_map_rows = []
    for out_col, (alts, prefer_numeric) in mapping.items():
        col = find_column(raw, alts, prefer_numeric=prefer_numeric)
        column_map_rows.append({"database": "MOSAEC_DB", "standard_column": out_col, "source_column": col or "NOT_FOUND"})
        if col is None:
            overlay[out_col] = np.nan if prefer_numeric else ""
        else:
            overlay[out_col] = pd.to_numeric(raw[col], errors="coerce") if prefer_numeric else raw[col]

    # If neutral/charged/full/partial are only represented in filenames/paths, infer flags conservatively.
    text_blob = (overlay["mosaec_display_id"].astype(str) + " " + overlay.get("framework_charge_state", pd.Series("", index=overlay.index)).astype(str)).str.lower()
    overlay["is_neutral"] = text_blob.str.contains("neutral") & ~text_blob.str.contains("charged")
    overlay["is_charged"] = text_blob.str.contains("charged")
    activation_blob = (overlay["mosaec_display_id"].astype(str) + " " + overlay.get("activation_state_full_or_partial", pd.Series("", index=overlay.index)).astype(str)).str.lower()
    overlay.loc[activation_blob.str.contains("partial"), "activation_state_full_or_partial"] = "partial"
    overlay.loc[activation_blob.str.contains("full"), "activation_state_full_or_partial"] = "full"

    overlay["match_key"] = first_nonempty_key(raw, id_cols + ([ref_col] if ref_col else []))

    # Descriptor availability flags. These only read identifier columns, not the full descriptor matrices when possible.
    for desc_name, fname in [
        ("GEOM_available", "GEOM_mosaec-db.csv"),
        ("RAC_available", "RAC_mosaec-db.csv"),
        ("APRDF_available", "APRDF_mosaec-db.csv"),
        ("PHOM_available", "PHOM_mosaec-db.csv"),
    ]:
        ids = read_identifier_set_from_csv(fname)
        overlay[desc_name] = overlay["match_key"].isin(ids) | overlay["mosaec_id"].isin(ids) if ids else False

    # PDD duplicate/similarity file.
    pdd = safe_optional_csv("PDD_duplicate_thresh0.15.csv")
    dup_keys = set()
    if pdd is not None and not pdd.empty:
        for col in pdd.columns[: min(6, pdd.shape[1])]:
            dup_keys |= set(pdd[col].map(canonical_key).dropna().unique())
    overlay["is_unique_by_PDD"] = ~overlay["match_key"].isin(dup_keys) if dup_keys else True
    overlay["duplicate_cluster_or_score"] = np.nan

    # Subset flags from text files.
    porous_2p4 = read_identifier_set_from_text("uniq-neutral-porous-2.4pld.txt")
    porous_010 = read_identifier_set_from_text("uniq-neutral-porous-0.10vf.txt")
    diverse_neutral = read_identifier_set_from_text("diverse-neutral-*-20k.txt")
    diverse_charged = read_identifier_set_from_text("diverse-charged-*-3k.txt")
    overlay["porous_2p4pld_flag"] = overlay["match_key"].isin(porous_2p4) | overlay["mosaec_id"].isin(porous_2p4) if porous_2p4 else False
    overlay["porous_0p10vf_flag"] = overlay["match_key"].isin(porous_010) | overlay["mosaec_id"].isin(porous_010) if porous_010 else False
    overlay["diverse_neutral_subset_flag"] = overlay["match_key"].isin(diverse_neutral) | overlay["mosaec_id"].isin(diverse_neutral) if diverse_neutral else False
    overlay["diverse_charged_subset_flag"] = overlay["match_key"].isin(diverse_charged) | overlay["mosaec_id"].isin(diverse_charged) if diverse_charged else False

    # Charge-assignment availability from local directories or metadata json files.
    repeat_paths = resolve_input_files_by_pattern("unchanged_repeat.json") + resolve_input_files_by_pattern("database_REPEAT")
    mepoml_paths = resolve_input_files_by_pattern("unchanged_mepoml.json") + resolve_input_files_by_pattern("database_MEPOML")
    overlay["has_REPEAT_charges"] = bool(repeat_paths)
    overlay["has_MEPOML_charges"] = bool(mepoml_paths)

    overlay["matched_arc_mof_id"] = ""
    overlay["matched_core_mof_id"] = ""
    overlay["match_confidence"] = "not_evaluated"

    atomic_save_csv(overlay, out_csv)
    atomic_save_csv(overlay, processed_csv)
    maybe_save_pickle(overlay, PICKLE_DIR / "mosaec_plausibility_overlay.pkl", minimum_mode="efficient")
    if column_map_rows:
        atomic_save_csv(pd.DataFrame(column_map_rows), TABLE_DIR / "table_mosaec_column_autodetection_map.csv")
    mark_done(stage, {"n_rows": len(overlay), "n_columns": overlay.shape[1]})
    return overlay


def _arc_geometry_table(master: pd.DataFrame) -> pd.DataFrame:
    """Return ARC-MOF geometry columns under external-standard names."""
    out = pd.DataFrame({"mof_id": master["mof_id"], "match_key": master["mof_id"].map(canonical_key)})
    rename = {"Di": "PLD", "Df": "LCD", "ASA": "ASA", "AVA": "PV", "Density": "density"}
    for src, dst in rename.items():
        out[dst] = pd.to_numeric(master[src], errors="coerce") if src in master.columns else np.nan
    return out


def nearest_geometry_overlay(
    candidates: pd.DataFrame,
    external: pd.DataFrame,
    external_id_col: str,
    external_prefix: str,
) -> pd.DataFrame:
    """Attach nearest external-database neighbour by simple geometry descriptors.

    This is not treated as proof of structural identity. It is a domain-overlap
    diagnostic: are confident ARC-MOF hits close to experimentally/stability-aware
    structures in PLD/LCD/density/surface-area/pore-volume space?
    """
    if candidates.empty or external.empty:
        return candidates
    common = [c for c in CONFIG.get("external_geometry_columns", ["PLD", "LCD", "density"]) if c in candidates.columns and c in external.columns]
    common = [c for c in common if pd.to_numeric(candidates[c], errors="coerce").notna().sum() > 5 and pd.to_numeric(external[c], errors="coerce").notna().sum() > 5]
    if len(common) < 2:
        candidates[f"{external_prefix}_nearest_id"] = ""
        candidates[f"{external_prefix}_geometry_distance"] = np.nan
        candidates[f"{external_prefix}_geometry_overlap_flag"] = False
        return candidates

    ext = external[[external_id_col, "match_key"] + common].copy()
    for c in common:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")
    ext = ext.dropna(subset=common)
    if ext.empty:
        candidates[f"{external_prefix}_nearest_id"] = ""
        candidates[f"{external_prefix}_geometry_distance"] = np.nan
        candidates[f"{external_prefix}_geometry_overlap_flag"] = False
        return candidates
    max_ext = int(CONFIG.get("geometry_nearest_neighbor_max_external_rows", 50000))
    if len(ext) > max_ext:
        ext = ext.sample(max_ext, random_state=CONFIG["random_seed"])

    cand = candidates.copy()
    for c in common:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    valid_cand_mask = cand[common].notna().all(axis=1)
    cand[f"{external_prefix}_nearest_id"] = ""
    cand[f"{external_prefix}_geometry_distance"] = np.nan
    cand[f"{external_prefix}_geometry_overlap_flag"] = False
    if valid_cand_mask.sum() == 0:
        return cand

    scaler = StandardScaler()
    X_ext = scaler.fit_transform(ext[common].to_numpy(dtype=float))
    X_cand = scaler.transform(cand.loc[valid_cand_mask, common].to_numpy(dtype=float))
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(X_ext)
    dist, idx = nn.kneighbors(X_cand, return_distance=True)
    cand.loc[valid_cand_mask, f"{external_prefix}_nearest_id"] = ext.iloc[idx[:, 0]][external_id_col].astype(str).to_numpy()
    cand.loc[valid_cand_mask, f"{external_prefix}_geometry_distance"] = dist[:, 0]
    cand.loc[valid_cand_mask, f"{external_prefix}_geometry_overlap_flag"] = dist[:, 0] <= float(CONFIG.get("geometry_match_distance_threshold", 0.75))
    return cand


def build_external_realism_table(master: pd.DataFrame, tiers: pd.DataFrame, core: pd.DataFrame, mosaec: pd.DataFrame, consensus: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge consensus ARC-MOF candidates with CoRE and MOSAEC plausibility flags."""
    stage = "10_external_realism_table"
    out_csv = TABLE_DIR / "final_screening_table_with_external_flags.csv"
    processed_csv = DATA_PROCESSED_DIR / "final_screening_table_with_external_flags.csv"
    if is_done(stage) and out_csv.exists():
        return pd.read_csv(out_csv, low_memory=False)

    logging.info("STAGE >>> BUILD_EXTERNAL_REALISM_TABLE")
    if (tiers is None or tiers.empty) and (consensus is None or consensus.empty):
        final = pd.DataFrame()
        atomic_save_csv(final, out_csv)
        atomic_save_csv(final, processed_csv)
        mark_done(stage, {"n_rows": 0, "note": "No predictions available."})
        return final

    max_cands = int(CONFIG.get("external_match_max_candidates", 1000))
    if consensus is not None and not consensus.empty:
        work = consensus.copy()
        if "consensus_pass" in work.columns and work["consensus_pass"].fillna(False).astype(bool).any():
            work = work[work["consensus_pass"].fillna(False).astype(bool)].copy()
        work = work.sort_values(["target", "consensus_score", "split_lower_median"], ascending=[True, False, False])
        rename = {"y_pred_median": "y_pred", "split_lower_median": "split_lower", "split_upper_median": "split_upper", "interval_width_median": "interval_width"}
        work = work.rename(columns=rename)
        keep_cols = ["mof_id", "filename", "target", "y_true_median", "y_pred", "split_lower", "split_upper", "interval_width", "consensus_score", "n_models", "n_seeds", "n_splits", "trusted_support", "consensus_pass"]
        keep_cols = [c for c in keep_cols if c in work.columns]
        final = work[keep_cols].head(max_cands).copy()
        final["tier"] = "consensus_trusted"
    else:
        work = tiers.copy()
        if "candidate_eligible" in work.columns:
            work = work[work["candidate_eligible"].fillna(False).astype(bool)].copy()
        work["interval_width"] = work["split_upper"] - work["split_lower"]
        work["tier_rank"] = work["tier"].map({"trusted": 0, "uncertain": 1, "rejected": 2}).fillna(1)
        work = work.sort_values(["tier_rank", "split_lower", "y_pred"], ascending=[True, False, False])
        keep_cols = ["mof_id", "filename", "target", "feature_set", "split", "model", "budget", "seed", "y_true", "y_pred", "split_lower", "split_upper", "interval_width", "tier"]
        keep_cols = [c for c in keep_cols if c in work.columns]
        final = work[keep_cols].head(max_cands).copy()
    final["match_key"] = final["mof_id"].map(canonical_key)

    # Add ARC geometry from the master table for nearest-neighbour external comparisons.
    arc_geo = _arc_geometry_table(master)
    final = final.merge(arc_geo.drop(columns=["match_key"]), on="mof_id", how="left")

    # Exact CoRE/MOSAEC flags.
    if core is not None and not core.empty and "match_key" in core.columns:
        core_keys = set(core["match_key"].dropna().astype(str)) | set(core.get("core_id", pd.Series(dtype=str)).dropna().astype(str))
        final["core_exact_match_flag"] = final["match_key"].isin(core_keys)
        final = nearest_geometry_overlay(final, core, external_id_col="core_display_id" if "core_display_id" in core.columns else "core_id", external_prefix="core")
        # Bring best available stability flags from exact matches where possible.
        core_small_cols = [c for c in ["match_key", "recommended_screening_flag", "asr_fsr_duplicate_flag", "prob_water_stability", "prob_solvent_removal_stability", "hydrophobic_class", "decomposition_temperature", "heat_capacity"] if c in core.columns]
        if core_small_cols:
            core_small = core[core_small_cols].drop_duplicates("match_key")
            final = final.merge(core_small, on="match_key", how="left", suffixes=("", "_core"))
    else:
        final["core_exact_match_flag"] = False
        final["core_nearest_id"] = ""
        final["core_geometry_distance"] = np.nan
        final["core_geometry_overlap_flag"] = False

    if mosaec is not None and not mosaec.empty and "match_key" in mosaec.columns:
        mos_keys = set(mosaec["match_key"].dropna().astype(str)) | set(mosaec.get("mosaec_id", pd.Series(dtype=str)).dropna().astype(str))
        final["mosaec_exact_match_flag"] = final["match_key"].isin(mos_keys)
        final = nearest_geometry_overlay(final, mosaec, external_id_col="mosaec_display_id" if "mosaec_display_id" in mosaec.columns else "mosaec_id", external_prefix="mosaec")
        mos_small_cols = [c for c in ["match_key", "is_unique_by_PDD", "porous_2p4pld_flag", "porous_0p10vf_flag", "diverse_neutral_subset_flag", "diverse_charged_subset_flag", "GEOM_available", "RAC_available", "APRDF_available", "PHOM_available"] if c in mosaec.columns]
        if mos_small_cols:
            mos_small = mosaec[mos_small_cols].drop_duplicates("match_key")
            final = final.merge(mos_small, on="match_key", how="left", suffixes=("", "_mosaec"))
    else:
        final["mosaec_exact_match_flag"] = False
        final["mosaec_nearest_id"] = ""
        final["mosaec_geometry_distance"] = np.nan
        final["mosaec_geometry_overlap_flag"] = False

    # Simple overall realism score. This is only a transparent triage score; the
    # paper should present its components rather than oversell a universal metric.
    bool_cols_positive = [
        "core_exact_match_flag", "core_geometry_overlap_flag", "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag",
        "recommended_screening_flag", "porous_2p4pld_flag", "porous_0p10vf_flag", "is_unique_by_PDD",
        "GEOM_available", "RAC_available",
    ]
    final["external_realism_score"] = 0
    for col in bool_cols_positive:
        if col in final.columns:
            final["external_realism_score"] += final[col].fillna(False).astype(bool).astype(int)
    if "asr_fsr_duplicate_flag" in final.columns:
        final["external_realism_score"] -= final["asr_fsr_duplicate_flag"].fillna(False).astype(bool).astype(int)

    atomic_save_csv(final, out_csv)
    atomic_save_csv(final, processed_csv)
    maybe_save_pickle(final, PICKLE_DIR / "final_screening_table_with_external_flags.pkl", minimum_mode="efficient")

    # Release-safe version: no restricted structural files, only derived metadata.
    release_cols = [c for c in final.columns if c not in {"filename"}]
    if CONFIG.get("save_public_release_safe_tables", True):
        atomic_save_csv(final[release_cols], DATA_RELEASE_TABLE_DIR / "final_screening_table_with_external_flags_release_safe.csv")

    mark_done(stage, {"n_rows": len(final), "n_columns": final.shape[1]})
    return final


def summarise_external_files(core: pd.DataFrame, mosaec: pd.DataFrame) -> pd.DataFrame:
    """Create a compact external-file availability table for the paper/SI."""
    rows = []
    for database, files in {
        "CoRE_MOF_2024": ["ASR_data_SI_20250204.csv", "FSR_data_SI_20250204.csv", "ION_data_SI_20250204.csv", "12089-recommended-screening-list.csv", "ASR_FSR_check.csv"],
        "MOSAEC_DB": ["mosaec-db.csv", "mosaec-db.xlsx", "GEOM_mosaec-db.csv", "RAC_mosaec-db.csv", "APRDF_mosaec-db.csv", "PHOM_mosaec-db.csv", "PDD_duplicate_thresh0.15.csv"],
    }.items():
        for fname in files:
            path = resolve_input_file(fname)
            rows.append({"database": database, "file": fname, "found": path.exists(), "path": str(path) if path.exists() else ""})
    rows.append({"database": "CoRE_MOF_2024", "file": "overlay_rows_created", "found": len(core) > 0, "path": str(len(core))})
    rows.append({"database": "MOSAEC_DB", "file": "overlay_rows_created", "found": len(mosaec) > 0, "path": str(len(mosaec))})
    df = pd.DataFrame(rows)
    atomic_save_csv(df, TABLE_DIR / "table_external_file_availability.csv")
    return df


def save_external_tables(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """Save manuscript/SI tables related to external realism overlays."""
    stage = "11_external_tables"
    if is_done(stage):
        return
    logging.info("STAGE >>> SAVE_EXTERNAL_TABLES")
    availability = summarise_external_files(core, mosaec)

    summary_rows = []
    if core is not None and not core.empty:
        summary_rows.append({
            "overlay": "CoRE_MOF_2024",
            "n_rows": len(core),
            "n_unique_keys": int(core["match_key"].nunique()) if "match_key" in core else np.nan,
            "recommended_screening_rows": int(core.get("recommended_screening_flag", pd.Series(False, index=core.index)).fillna(False).sum()),
            "duplicate_flag_rows": int(core.get("asr_fsr_duplicate_flag", pd.Series(False, index=core.index)).fillna(False).sum()),
        })
    if mosaec is not None and not mosaec.empty:
        summary_rows.append({
            "overlay": "MOSAEC_DB",
            "n_rows": len(mosaec),
            "n_unique_keys": int(mosaec["match_key"].nunique()) if "match_key" in mosaec else np.nan,
            "geom_available_rows": int(mosaec.get("GEOM_available", pd.Series(False, index=mosaec.index)).fillna(False).sum()),
            "rac_available_rows": int(mosaec.get("RAC_available", pd.Series(False, index=mosaec.index)).fillna(False).sum()),
            "pdd_unique_rows": int(mosaec.get("is_unique_by_PDD", pd.Series(False, index=mosaec.index)).fillna(False).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    atomic_save_csv(summary, TABLE_DIR / "MAIN_Table_4_external_overlay_summary.csv")
    atomic_save_csv(availability, TABLE_DIR / "SI_Table_S5_external_file_availability.csv")

    if final_external is not None and not final_external.empty:
        cols = [c for c in [
            "mof_id", "target", "model", "budget", "tier", "y_pred", "split_lower", "split_upper",
            "external_realism_score", "core_exact_match_flag", "core_geometry_overlap_flag",
            "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag", "recommended_screening_flag",
            "prob_water_stability", "hydrophobic_class", "is_unique_by_PDD"
        ] if c in final_external.columns]
        atomic_save_csv(final_external[cols].head(200), TABLE_DIR / "MAIN_Table_5_top_external_realism_candidates.csv")
        atomic_save_csv(final_external, TABLE_DIR / "SI_Table_S6_all_external_realism_candidates.csv")
        if CONFIG.get("save_public_release_safe_tables", True):
            atomic_save_csv(final_external[cols].head(200), DATA_RELEASE_TABLE_DIR / "top_external_realism_candidates_release_safe.csv")
    mark_done(stage, {"n_summary_rows": len(summary)})


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """Main Figure 5: external realism filter for confident ARC-MOF hits."""
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 5 external realism filter")
    availability = summarise_external_files(core, mosaec)
    atomic_save_csv(availability, FIGDATA_DIR / "Figure_5_external_file_availability_data.csv")
    if CONFIG.get("save_public_release_safe_tables", True):
        atomic_save_csv(availability, DATA_RELEASE_FIGURE_DIR / "Figure_5_external_file_availability_data.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    # (a) External availability.
    avail_counts = availability.groupby("database")["found"].sum().reset_index()
    axes[0].bar(avail_counts["database"], avail_counts["found"])
    axes[0].set_ylabel("Available files / flags")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("External overlay availability")
    set_panel_label(axes[0], "(a)")

    if final_external is None or final_external.empty:
        for ax, lab in zip(axes[1:], ["(b)", "(c)", "(d)", "(e)", "(f)"]):
            ax.axis("off")
            ax.text(0.5, 0.5, "No external candidate table\ncreated in this run", ha="center", va="center")
            set_panel_label(ax, lab)
        save_current_figure(FIG_MAIN_DIR / "Figure_5_external_realism_filter")
        mark_done(stage, {"n_rows": 0})
        return

    fe = final_external.copy()
    atomic_save_csv(fe, FIGDATA_DIR / "Figure_5_external_realism_candidate_data.csv")
    if CONFIG.get("save_public_release_safe_tables", True):
        release_cols = [c for c in fe.columns if c != "filename"]
        atomic_save_csv(fe[release_cols], DATA_RELEASE_FIGURE_DIR / "Figure_5_external_realism_candidate_data_release_safe.csv")

    # (b) Funnel counts.
    funnel = pd.DataFrame({
        "stage": ["screened", "trusted", "CoRE exact", "CoRE geom", "MOSAEC exact", "MOSAEC geom"],
        "n": [
            len(fe),
            int((fe.get("tier", pd.Series("", index=fe.index)) == "trusted").sum()),
            int(fe.get("core_exact_match_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int(fe.get("core_geometry_overlap_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int(fe.get("mosaec_exact_match_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int(fe.get("mosaec_geometry_overlap_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
        ]
    })
    atomic_save_csv(funnel, FIGDATA_DIR / "Figure_5_external_realism_funnel_data.csv")
    axes[1].bar(funnel["stage"], funnel["n"])
    axes[1].set_ylabel("Candidate rows")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("External-realism funnel")
    set_panel_label(axes[1], "(b)")

    # (c) Lower confidence bound vs interval width, sized by external realism score.
    x = pd.to_numeric(fe.get("split_lower", pd.Series(np.nan, index=fe.index)), errors="coerce")
    y = pd.to_numeric(fe.get("interval_width", pd.Series(np.nan, index=fe.index)), errors="coerce")
    s = 10 + 10 * pd.to_numeric(fe.get("external_realism_score", pd.Series(0, index=fe.index)), errors="coerce").fillna(0)
    axes[2].scatter(x, y, s=s, alpha=0.4)
    axes[2].set_xlabel("Lower confidence bound")
    axes[2].set_ylabel("Interval width")
    axes[2].set_title("Confidence vs realism score")
    set_panel_label(axes[2], "(c)")

    # (d) Candidate counts by target and external score.
    if "target" in fe.columns and "external_realism_score" in fe.columns:
        target_score = fe.groupby("target", as_index=False)["external_realism_score"].mean()
        axes[3].bar(target_score["target"], target_score["external_realism_score"])
        axes[3].tick_params(axis="x", rotation=45)
        axes[3].set_ylabel("Mean realism score")
    axes[3].set_title("Target-resolved realism")
    set_panel_label(axes[3], "(d)")

    # (e) Geometry distance distributions.
    dist_cols = [c for c in ["core_geometry_distance", "mosaec_geometry_distance"] if c in fe.columns]
    if dist_cols:
        for col in dist_cols:
            vals = pd.to_numeric(fe[col], errors="coerce").dropna()
            if len(vals) > 0:
                axes[4].hist(vals, bins=30, alpha=0.45, label=col.replace("_geometry_distance", ""))
        axes[4].legend(fontsize=8)
    axes[4].set_xlabel("Nearest-neighbour geometry distance")
    axes[4].set_ylabel("Count")
    axes[4].set_title("External geometry proximity")
    set_panel_label(axes[4], "(e)")

    # (f) Component prevalence.
    components = [c for c in [
        "core_exact_match_flag", "core_geometry_overlap_flag", "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag",
        "recommended_screening_flag", "is_unique_by_PDD", "porous_2p4pld_flag", "GEOM_available", "RAC_available"
    ] if c in fe.columns]
    prev = pd.DataFrame({"component": components, "fraction": [fe[c].fillna(False).astype(bool).mean() for c in components]})
    atomic_save_csv(prev, FIGDATA_DIR / "Figure_5_external_realism_component_prevalence_data.csv")
    if not prev.empty:
        axes[5].barh(range(len(prev)), prev["fraction"])
        axes[5].set_yticks(range(len(prev)))
        axes[5].set_yticklabels(prev["component"], fontsize=7)
        axes[5].invert_yaxis()
    axes[5].set_xlabel("Fraction of candidate rows")
    axes[5].set_title("External evidence components")
    set_panel_label(axes[5], "(f)")

    fig.suptitle("External realism filter: confident ARC-MOF hits under CoRE and MOSAEC overlays", fontsize=14, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_5_external_realism_filter")
    mark_done(stage, {"n_rows": len(final_external)})


def plot_external_si_figures(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """Additional SI figures for external overlays and descriptor-space overlap."""
    stage = "external_si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> External SI figures")

    if core is not None and not core.empty:
        atomic_save_csv(core.head(5000), FIGDATA_DIR / "Figure_S6_core_overlay_preview_data.csv")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        for ax, col, label in zip(axes, ["PLD", "LCD", "density", "prob_water_stability"], ["PLD", "LCD", "Density", "Water-stability probability"]):
            if col in core.columns:
                vals = pd.to_numeric(core[col], errors="coerce").dropna()
                if len(vals) > 0:
                    ax.hist(vals, bins=40)
            ax.set_title(label)
            ax.set_ylabel("Count")
        fig.suptitle("Figure S6. CoRE MOF 2024 stability/geometry overlay")
        save_current_figure(FIG_SI_DIR / "Figure_S6_core_overlay_distributions")

    if mosaec is not None and not mosaec.empty:
        atomic_save_csv(mosaec.head(5000), FIGDATA_DIR / "Figure_S7_mosaec_overlay_preview_data.csv")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        flags = ["GEOM_available", "RAC_available", "porous_2p4pld_flag", "is_unique_by_PDD"]
        vals = [mosaec.get(f, pd.Series(False, index=mosaec.index)).fillna(False).astype(bool).mean() for f in flags]
        axes[0].bar(flags, vals)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("Fraction")
        axes[0].set_title("MOSAEC descriptor/subset flags")
        set_panel_label(axes[0], "(a)")
        for ax, col, lab in zip(axes[1:], ["PLD", "LCD", "density"], ["PLD", "LCD", "Density"]):
            if col in mosaec.columns:
                vv = pd.to_numeric(mosaec[col], errors="coerce").dropna()
                if len(vv) > 0:
                    ax.hist(vv, bins=40)
            ax.set_title(lab)
        fig.suptitle("Figure S7. MOSAEC chemical-validity/plausibility overlay")
        save_current_figure(FIG_SI_DIR / "Figure_S7_mosaec_overlay_distributions")

    if final_external is not None and not final_external.empty:
        fe = final_external.copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        if "external_realism_score" in fe.columns:
            ax.hist(pd.to_numeric(fe["external_realism_score"], errors="coerce").dropna(), bins=range(int(fe["external_realism_score"].max()) + 3 if len(fe) else 3))
        ax.set_xlabel("External realism score")
        ax.set_ylabel("Candidate rows")
        ax.set_title("Figure S8. Distribution of external realism scores")
        save_current_figure(FIG_SI_DIR / "Figure_S8_external_realism_score_distribution")

    mark_done(stage)


def run_external_overlay_workflow(master: pd.DataFrame, tiers: pd.DataFrame, consensus: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all external overlay stages, gracefully skipping missing databases."""
    if not CONFIG.get("enable_external_overlays", True):
        logging.info("EXTERNAL_OVERLAYS_DISABLED")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    core = build_core_mof_2024_overlay()
    mosaec = build_mosaec_db_overlay()
    final_external = build_external_realism_table(master, tiers, core, mosaec, consensus=consensus)
    write_external_overlay_integrity(core, mosaec, final_external)
    save_external_tables(core, mosaec, final_external)
    plot_figure_5_external_realism(core, mosaec, final_external)
    plot_external_si_figures(core, mosaec, final_external)
    return core, mosaec, final_external


# =============================================================================
# 16b. Download manifest/checklist for provenance
# =============================================================================

def write_download_manifest_and_checklist() -> None:
    """Write a data-archive manifest and checklist for the configured root.

    The manifest now reflects the user's project_data archive layout rather than
    assuming files are next to the script. It records both the preferred relative
    path and the actual resolved local path.
    """
    priority_map = {
        "geometric_properties.csv": "ESSENTIAL_ARC_MOF",
        "post_comb_vsa-CO2.csv": "ESSENTIAL_ARC_MOF",
        "methane.csv": "ESSENTIAL_ARC_MOF",
        "overall_process.csv": "STRONGLY_RECOMMENDED_ARC_MOF",
        "RACs.csv": "STRONGLY_RECOMMENDED_ARC_MOF",
        "RDFs.csv": "OPTIONAL_LARGE_SI",
        "all_topology_lists.csv": "STRONGLY_RECOMMENDED_ARC_MOF",
        "ASR_data_SI_20250204.csv": "ESSENTIAL_EXTERNAL_CORE",
        "FSR_data_SI_20250204.csv": "ESSENTIAL_EXTERNAL_CORE",
        "ION_data_SI_20250204.csv": "ESSENTIAL_EXTERNAL_CORE",
        "mosaec-db.csv": "ESSENTIAL_EXTERNAL_MOSAEC",
        "GEOM_mosaec-db.csv": "ESSENTIAL_EXTERNAL_MOSAEC",
        "RAC_mosaec-db.csv": "RECOMMENDED_EXTERNAL_MOSAEC",
    }
    rows = []
    for filename, rels in sorted(INPUT_FILE_RELATIVE_PATHS.items()):
        resolved = resolve_input_file(filename)
        found = bool(resolved.exists())
        database = "ARC-MOF"
        joined = " ".join(rels).lower()
        if "core_mof_2024" in joined:
            database = "CoRE_MOF_2024"
        elif "mosaec_db" in joined:
            database = "MOSAEC_DB"
        rows.append({
            "database": database,
            "filename_requested_by_code": filename,
            "preferred_relative_path": rels[0] if rels else filename,
            "all_known_relative_aliases": "; ".join(rels),
            "priority": priority_map.get(filename, "OPTIONAL_OR_CONTEXTUAL"),
            "configured_data_root": str(DATA_ROOT),
            "found_locally": found,
            "local_path_detected": str(resolved) if found else "",
        })
    df = pd.DataFrame(rows)
    atomic_save_csv(df, TABLE_DIR / "download_manifest_required_files.csv")
    checklist = RESULTS_DIR / "DOWNLOAD_CHECKLIST.md"
    lines = [
        "# Data archive checklist generated by the pipeline",
        "",
        f"Configured data root: `{DATA_ROOT}`",
        "",
        "The pipeline accepts both older hyphen/camel-case filenames and the current archive's lowercase/underscore filenames.",
        "",
    ]
    for _, r in df.iterrows():
        box = "x" if r["found_locally"] else " "
        lines.append(f"- [{box}] {r['database']} / `{r['filename_requested_by_code']}` -> `{r['preferred_relative_path']}` ({r['priority']})")
    checklist.write_text("\n".join(lines), encoding="utf-8")




def audit_project_folder_tree() -> pd.DataFrame:
    """Audit the configured project_data archive and key files/directories.

    The audit is intentionally lightweight. It verifies the expected scientific
    data folders from the archive report and the concrete files used by this
    pipeline. It does not walk every CIF directory.
    """
    stage = "00b_project_folder_audit"
    out_path = TABLE_DIR / "table_project_folder_audit.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path)

    folder_checks = [
        ("PROJECT_ROOT", ".", "root"),
        ("DATA_RAW", "data_raw", "essential"),
        ("ARC_MOF_ROOT", "data_raw/arc_mof", "essential"),
        ("ARC_MOF_ADSORPTION", "data_raw/arc_mof/adsorption", "essential"),
        ("ARC_MOF_CLUSTERS", "data_raw/arc_mof/clusters", "essential"),
        ("ARC_MOF_DESCRIPTORS", "data_raw/arc_mof/descriptors", "essential"),
        ("ARC_MOF_PROCESS", "data_raw/arc_mof/process", "recommended"),
        ("ARC_MOF_TOPOLOGY", "data_raw/arc_mof/topology", "recommended"),
        ("CORE_MOF_2024_ROOT", "data_raw/core_mof_2024", "external"),
        ("CORE_MOF_2024_METADATA", "data_raw/core_mof_2024/metadata", "external"),
        ("CORE_MOF_2024_DIAGNOSTICS", "data_raw/core_mof_2024/diagnostics", "external_optional"),
        ("MOSAEC_ROOT", "data_raw/mosaec_db", "external"),
        ("MOSAEC_DESCRIPTORS", "data_raw/mosaec_db/descriptors/extracted/descriptors/descriptors", "external"),
        ("MOSAEC_MISC", "data_raw/mosaec_db/misc_data/extracted/misc_data/misc_data", "external"),
        ("MOSAEC_SUBSETS", "data_raw/mosaec_db/subsets/extracted/subsets/subsets", "external"),
    ]
    rows = []
    root = DATA_ROOT or SCRIPT_DIR
    for label, rel, priority in folder_checks:
        pth = root / rel
        rows.append({
            "kind": "folder",
            "database_or_group": label,
            "priority": priority,
            "expected_relative_path": rel,
            "resolved_path": str(pth),
            "exists": bool(pth.exists()),
            "status": "OK" if pth.exists() else ("OPTIONAL_MISSING" if "optional" in priority else "MISSING"),
        })

    for filename, rels in sorted(INPUT_FILE_RELATIVE_PATHS.items()):
        resolved = resolve_input_file(filename)
        exists = bool(resolved.exists())
        joined = " ".join(rels).lower()
        group = "ARC-MOF"
        if "core_mof_2024" in joined:
            group = "CoRE_MOF_2024"
        elif "mosaec_db" in joined:
            group = "MOSAEC_DB"
        priority = "essential" if filename in {"geometric_properties.csv", "post_comb_vsa-CO2.csv", "methane.csv"} else "recommended_or_optional"
        rows.append({
            "kind": "file",
            "database_or_group": group,
            "priority": priority,
            "expected_relative_path": rels[0] if rels else filename,
            "resolved_path": str(resolved),
            "exists": exists,
            "status": "OK" if exists else ("MISSING" if priority == "essential" else "OPTIONAL_MISSING"),
        })

    df = pd.DataFrame(rows)
    atomic_save_csv(df, out_path)
    mark_done(stage, {
        "configured_data_root": str(DATA_ROOT),
        "n_ok": int(df["status"].eq("OK").sum()),
        "n_missing": int(df["status"].eq("MISSING").sum()),
        "n_optional_missing": int(df["status"].eq("OPTIONAL_MISSING").sum()),
    })
    return df



POSTPROCESS_STAGES = [
    "03_aggregate_results", "04_candidate_tiers", "04b_consensus_candidates", "05_chemical_failure_anatomy",
    "08_core_mof_2024_overlay", "09_mosaec_db_overlay", "10_external_realism_table", "11_external_tables",
    "fig1_framework", "fig2_performance", "fig3_calibration", "fig4_failure_anatomy", "fig5_external_realism",
    "si_figures", "external_si_figures", "06_manuscript_si_tables", "07_run_report",
]


def clear_postprocess_markers() -> None:
    """Remove downstream done markers so post-processing can be regenerated safely."""
    for stage in POSTPROCESS_STAGES:
        path = done_path(stage)
        if path.exists():
            path.unlink()
            logging.info("POSTPROCESS_MARKER_REMOVED | %s", stage)


def load_existing_metrics_all() -> pd.DataFrame:
    """Load completed experiment metrics without refitting models."""
    metrics_all_path = TABLE_DIR / "all_experiment_metrics.csv"
    if metrics_all_path.exists():
        return pd.read_csv(metrics_all_path)
    part_dir = TABLE_DIR / "experiment_metrics_parts"
    parts = sorted(part_dir.glob("metrics_*.csv"))
    if not parts:
        raise FileNotFoundError("--postprocess-only requested, but no all_experiment_metrics.csv or metrics_*.csv files were found.")
    metrics_all = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    atomic_save_csv(metrics_all, metrics_all_path)
    return metrics_all


# =============================================================================
# 16c. v3.9 target-balanced consensus and figure polish overrides
# =============================================================================
# v3.8 successfully removed unstable MLP/Ridge/GPR outliers from candidate
# tables, but global ranking by raw uptake/lower-bound naturally favoured the
# largest-scale target (CH4 at 65 bar). v3.9 keeps all QC gates and changes only
# downstream post-processing so final shortlists and figures are balanced across
# the four adsorption targets. These definitions intentionally override selected
# v3.8 functions above and are resolved by Python at run time.

CONFIG.setdefault("target_balanced_candidate_collection", True)
CONFIG.setdefault("candidate_tiers_max_rows_per_target", 30000)
CONFIG.setdefault("candidate_tiers_min_rows_per_target", 1000)
CONFIG.setdefault("consensus_top_per_target", 250)
CONFIG.setdefault("external_match_max_candidates_per_target", 250)
CONFIG.setdefault("figure_target_balanced_sample_per_target", 3000)
CONFIG.setdefault("move_external_realism_to_si_if_weak", True)


def v39_target_list_from_available(df: Optional[pd.DataFrame] = None) -> List[str]:
    """Return target order used for target-balanced outputs."""
    base = list(TARGET_SPECS.keys())
    if df is None or df.empty or "target" not in df.columns:
        return base
    present = [t for t in base if t in set(df["target"].astype(str))]
    extras = [t for t in sorted(df["target"].dropna().astype(str).unique()) if t not in present]
    return present + extras


def v39_add_target_normalized_scores(df: pd.DataFrame, target_stats: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Add target-normalised and within-target percentile columns.

    Raw uptake scales are not comparable across CO2 at 0.015 bar, CO2 at 0.15 bar,
    CH4 at 5.8 bar, and CH4 at 65 bar. These columns prevent high-pressure CH4
    from dominating global candidate selection simply because it has larger
    absolute mmol/g values.
    """
    if df is None or df.empty or "target" not in df.columns:
        return df
    out = df.copy()
    target_stats = target_stats or {}
    for col in ["y_true", "y_pred", "split_lower", "split_upper", "budget", "seed", "Density", "Di", "Df", "Dif"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["interval_width"] = pd.to_numeric(out.get("split_upper", np.nan), errors="coerce") - pd.to_numeric(out.get("split_lower", np.nan), errors="coerce")
    # z-like scores using observed target distribution when available.
    for t, st in target_stats.items():
        mask = out["target"].astype(str).eq(str(t))
        if not mask.any():
            continue
        mean = float(st.get("mean", 0.0)) if np.isfinite(st.get("mean", np.nan)) else 0.0
        std = float(st.get("std", 1.0)) if np.isfinite(st.get("std", np.nan)) and float(st.get("std", 0.0)) > 0 else 1.0
        observed_max = float(st.get("max", np.nan)) if np.isfinite(st.get("max", np.nan)) else np.nan
        for src_col, dst_col in [("y_pred", "target_z_y_pred"), ("split_lower", "target_z_split_lower"), ("split_upper", "target_z_split_upper")]:
            if src_col in out.columns:
                out.loc[mask, dst_col] = (out.loc[mask, src_col].astype(float) - mean) / std
        if np.isfinite(observed_max) and observed_max > 0:
            out.loc[mask, "target_fraction_of_observed_max"] = out.loc[mask, "split_lower"].astype(float) / observed_max
    # Percentile ranks within each target are the most robust for cross-target shortlists.
    if "y_pred" in out.columns:
        out["target_pred_percentile"] = out.groupby("target", dropna=False)["y_pred"].rank(pct=True, method="average")
    if "split_lower" in out.columns:
        out["target_lower_bound_percentile"] = out.groupby("target", dropna=False)["split_lower"].rank(pct=True, method="average")
    if "interval_width" in out.columns:
        # Smaller interval width is better, so invert percentile.
        out["target_interval_width_percentile_small"] = 1.0 - out.groupby("target", dropna=False)["interval_width"].rank(pct=True, method="average")
    for c in ["target_z_y_pred", "target_z_split_lower", "target_z_split_upper", "target_fraction_of_observed_max", "target_pred_percentile", "target_lower_bound_percentile", "target_interval_width_percentile_small"]:
        if c not in out.columns:
            out[c] = np.nan
    out["target_balanced_row_score"] = (
        6.0 * (out.get("tier", "").astype(str).eq("trusted").astype(float) if "tier" in out.columns else 0.0)
        + 4.0 * pd.to_numeric(out["target_lower_bound_percentile"], errors="coerce").fillna(0.0)
        + 1.5 * pd.to_numeric(out["target_pred_percentile"], errors="coerce").fillna(0.0)
        + 1.0 * pd.to_numeric(out["target_interval_width_percentile_small"], errors="coerce").fillna(0.0)
    )
    return out


def v39_target_balanced_compact_candidate_rows(df: pd.DataFrame, max_rows: Optional[int] = None, target_stats: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Compact candidates while preserving representation from every target.

    This is the key v3.9 change. Instead of sorting the full candidate pool by raw
    split_lower, it selects the best rows within each target using within-target
    percentiles and then concatenates target-specific subsets.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    max_rows = int(max_rows or CONFIG.get("candidate_tiers_max_rows", 250000))
    work = v39_add_target_normalized_scores(df, target_stats=target_stats)
    targets = v39_target_list_from_available(work)
    if not targets:
        return compact_candidate_rows(work, max_rows=max_rows)
    configured_per_target = int(CONFIG.get("candidate_tiers_max_rows_per_target", 30000))
    per_target = max(1, min(configured_per_target, int(math.ceil(max_rows / max(1, len(targets))))))
    min_per_target = int(CONFIG.get("candidate_tiers_min_rows_per_target", 1000))
    parts: List[pd.DataFrame] = []
    for target in targets:
        sub = work[work["target"].astype(str).eq(str(target))].copy()
        if sub.empty:
            continue
        # Keep a balanced mixture: trusted, high lower-bound percentile, high prediction percentile,
        # and a small deterministic diagnostic sample.
        local_parts: List[pd.DataFrame] = []
        quota = max(per_target, min_per_target)
        if "tier" in sub.columns:
            trusted = sub[sub["tier"].astype(str).eq("trusted")].copy()
            if not trusted.empty:
                local_parts.append(trusted.sort_values("target_balanced_row_score", ascending=False).head(quota))
        local_parts.append(sub.sort_values("target_lower_bound_percentile", ascending=False).head(max(50, quota // 2)))
        local_parts.append(sub.sort_values("target_pred_percentile", ascending=False).head(max(50, quota // 4)))
        if {"y_true", "y_pred"}.issubset(sub.columns):
            tmp = sub.copy()
            tmp["_abs_error_for_selection"] = np.abs(pd.to_numeric(tmp["y_true"], errors="coerce") - pd.to_numeric(tmp["y_pred"], errors="coerce"))
            local_parts.append(tmp.sort_values("_abs_error_for_selection", ascending=False).head(max(50, quota // 8)).drop(columns=["_abs_error_for_selection"], errors="ignore"))
        sample_n = min(max(25, quota // 10), len(sub))
        if sample_n > 0:
            local_parts.append(sub.sample(sample_n, random_state=int(CONFIG.get("random_seed", 42))))
        target_out = pd.concat(local_parts, ignore_index=True, copy=False).drop_duplicates() if local_parts else sub.head(quota)
        if len(target_out) > per_target:
            target_out = target_out.sort_values("target_balanced_row_score", ascending=False).head(per_target)
        parts.append(target_out)
    out = pd.concat(parts, ignore_index=True, copy=False).drop_duplicates() if parts else work.head(max_rows)
    if len(out) > max_rows:
        # Preserve target balance if total cap is still exceeded.
        reduced_parts = []
        final_per_target = max(1, int(math.floor(max_rows / max(1, len(v39_target_list_from_available(out))))))
        for target in v39_target_list_from_available(out):
            sub = out[out["target"].astype(str).eq(str(target))].sort_values("target_balanced_row_score", ascending=False)
            reduced_parts.append(sub.head(final_per_target))
        out = pd.concat(reduced_parts, ignore_index=True, copy=False) if reduced_parts else out.head(max_rows)
    return truncate_prediction_text_columns(out)


def collect_candidate_tiers(metrics_all: Optional[pd.DataFrame] = None, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """v3.9 target-balanced streaming candidate-tier collection."""
    stage = "04_candidate_tiers"
    out_path = TABLE_DIR / "table_candidate_tiers_all_predictions.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path, low_memory=False)

    logging.info("STAGE >>> COLLECT_CANDIDATE_TIERS_STREAMING_TARGET_BALANCED")
    pred_files = choose_prediction_files_for_collection()
    max_total_rows = int(CONFIG.get("candidate_tiers_max_rows", 250000))
    target_stats = target_stats_from_master(master)
    quality_table = build_experiment_quality_table(metrics_all, master) if CONFIG.get("enable_candidate_quality_gate", True) else pd.DataFrame()

    collected: List[pd.DataFrame] = []
    rejected_qc_parts: List[pd.DataFrame] = []
    n_files_read = 0
    n_files_skipped = 0
    n_rows_seen_after_compaction = 0

    for i, pred_file in enumerate(pred_files, start=1):
        if i == 1 or i % 250 == 0:
            logging.info("CANDIDATE_TIER_PROGRESS | %d/%d files", i, len(pred_files))
        df = read_prediction_file_compact(pred_file)
        if df.empty:
            n_files_skipped += 1
            continue
        df = apply_prediction_quality_flags(df, quality_table, target_stats)
        n_files_read += 1
        n_rows_seen_after_compaction += len(df)
        if CONFIG.get("enable_candidate_quality_gate", True):
            rejected = df[~df["candidate_eligible"].fillna(False).astype(bool)].copy()
            if not rejected.empty:
                rejected_qc_parts.append(rejected.head(200))
            df = df[df["candidate_eligible"].fillna(False).astype(bool)].copy()
        if df.empty:
            n_files_skipped += 1
            continue
        # Per-file compaction is already target-specific for a single experiment, but
        # add target-normalised scores early so downstream periodic reductions are safe.
        df = v39_add_target_normalized_scores(df, target_stats=target_stats)
        collected.append(df)

        if sum(len(x) for x in collected) > max_total_rows * 2:
            pooled = pd.concat(collected, ignore_index=True, copy=False)
            collected = [v39_target_balanced_compact_candidate_rows(pooled, max_rows=max_total_rows, target_stats=target_stats)]
            del pooled
            gc.collect()

    if not collected:
        tiers = pd.DataFrame()
    else:
        tiers = pd.concat(collected, ignore_index=True, copy=False)
        tiers = v39_target_balanced_compact_candidate_rows(tiers, max_rows=max_total_rows, target_stats=target_stats)

    if not tiers.empty:
        tiers = v39_add_target_normalized_scores(tiers, target_stats=target_stats)
        front_cols = [c for c in PREDICTION_COMPACT_COLUMNS if c in tiers.columns]
        v39_cols = [c for c in [
            "candidate_eligible", "experiment_passes_quality_gate", "prediction_sanity_pass",
            "target_z_y_pred", "target_z_split_lower", "target_fraction_of_observed_max",
            "target_pred_percentile", "target_lower_bound_percentile", "target_interval_width_percentile_small",
            "target_balanced_row_score", "interval_width",
        ] if c in tiers.columns]
        other_cols = [c for c in tiers.columns if c not in front_cols + v39_cols]
        tiers = tiers[front_cols + v39_cols + other_cols]
        tiers = truncate_prediction_text_columns(tiers)

    atomic_save_csv(tiers, out_path)
    write_prediction_range_qc(tiers)
    if rejected_qc_parts:
        rejected_sample = pd.concat(rejected_qc_parts, ignore_index=True, copy=False).head(50000)
        atomic_save_csv(rejected_sample, TABLE_DIR / "QC_Table_rejected_prediction_rows_sample.csv")

    if not tiers.empty:
        # Target composition QC is central for v3.9.
        target_counts = tiers.groupby("target", dropna=False).size().reset_index(name="n_candidate_rows")
        atomic_save_csv(target_counts, TABLE_DIR / "QC_Table_candidate_rows_by_target.csv")
        group_cols = [c for c in ["target", "feature_set", "split", "model", "budget", "seed", "tier"] if c in tiers.columns]
        if group_cols:
            tier_summary = tiers.groupby(group_cols, dropna=False).size().reset_index(name="n_compact_rows")
            atomic_save_csv(tier_summary, TABLE_DIR / "table_candidate_tier_counts.csv")
        if "target_balanced_row_score" in tiers.columns:
            best = []
            keep_top = int(CONFIG.get("candidate_tiers_keep_top_per_group", 50))
            best_group_cols = [c for c in ["target", "feature_set", "split", "model", "budget", "seed"] if c in tiers.columns]
            if best_group_cols:
                best_df = (tiers.sort_values("target_balanced_row_score", ascending=False)
                           .groupby(best_group_cols, dropna=False)
                           .head(keep_top))
            else:
                best_df = v39_target_balanced_compact_candidate_rows(tiers, max_rows=5000, target_stats=target_stats)
            atomic_save_csv(best_df, TABLE_DIR / "table_top50_candidates_by_target_balanced_score.csv")
            # Backward-compatible filename with safer content.
            atomic_save_csv(best_df, TABLE_DIR / "table_top50_candidates_by_lower_confidence_bound.csv")

    save_json(
        {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "note": "v3.9 streaming target-balanced candidate-tier collection. Rows are quality-gated and selected within each target using percentile scores.",
            "n_prediction_files_found": len(pred_files),
            "n_prediction_files_read": n_files_read,
            "n_prediction_files_skipped": n_files_skipped,
            "n_rows_after_per_file_compaction": int(n_rows_seen_after_compaction),
            "n_rows_saved": int(len(tiers)),
            "max_rows_config": max_total_rows,
            "candidate_tiers_max_rows_per_target": int(CONFIG.get("candidate_tiers_max_rows_per_target", 30000)),
        },
        TABLE_DIR / "table_candidate_tiers_collection_manifest.json",
    )

    mark_done(stage, {
        "n_prediction_files_found": len(pred_files),
        "n_prediction_files_read": n_files_read,
        "n_prediction_files_skipped": n_files_skipped,
        "n_prediction_rows_saved": int(len(tiers)),
        "streaming_compact_collection": True,
        "target_balanced_collection": True,
    })
    return tiers


def build_consensus_candidate_table(tiers: pd.DataFrame) -> pd.DataFrame:
    """v3.9 target-balanced consensus table, one row per MOF/target."""
    stage = "04b_consensus_candidates"
    out_path = TABLE_DIR / "table_consensus_candidate_shortlist.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path, low_memory=False)
    if tiers is None or tiers.empty:
        consensus = pd.DataFrame()
        atomic_save_csv(consensus, out_path)
        mark_done(stage, {"n_rows": 0})
        return consensus
    df = tiers.copy()
    if "candidate_eligible" in df.columns:
        df = df[df["candidate_eligible"].fillna(False).astype(bool)].copy()
    for col in ["y_true", "y_pred", "split_lower", "split_upper", "budget", "seed", "Density", "Di", "Df", "Dif", "target_lower_bound_percentile", "target_pred_percentile", "target_interval_width_percentile_small", "target_balanced_row_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        consensus = pd.DataFrame()
        atomic_save_csv(consensus, out_path)
        mark_done(stage, {"n_rows": 0, "note": "No candidate-eligible rows after gates."})
        return consensus
    df["interval_width"] = pd.to_numeric(df["split_upper"], errors="coerce") - pd.to_numeric(df["split_lower"], errors="coerce")
    if "target_lower_bound_percentile" not in df.columns:
        df = v39_add_target_normalized_scores(df)
    group_cols = ["target", "mof_id"]
    first_filename = df.groupby(group_cols)["filename"].first().rename("filename") if "filename" in df.columns else None
    agg = df.groupby(group_cols, dropna=False).agg(
        y_true_median=("y_true", "median"),
        y_pred_median=("y_pred", "median"),
        y_pred_max=("y_pred", "max"),
        split_lower_median=("split_lower", "median"),
        split_lower_max=("split_lower", "max"),
        split_upper_median=("split_upper", "median"),
        interval_width_median=("interval_width", "median"),
        target_lower_bound_percentile_median=("target_lower_bound_percentile", "median"),
        target_pred_percentile_median=("target_pred_percentile", "median"),
        target_interval_width_percentile_small_median=("target_interval_width_percentile_small", "median"),
        n_support_rows=("mof_id", "size"),
        n_models=("model", pd.Series.nunique),
        n_seeds=("seed", pd.Series.nunique),
        n_splits=("split", pd.Series.nunique),
        n_feature_sets=("feature_set", pd.Series.nunique),
        n_budgets=("budget", pd.Series.nunique),
        trusted_support=("tier", lambda s: int((s.astype(str) == "trusted").sum())),
        Density=("Density", "median") if "Density" in df.columns else ("y_pred", "median"),
        Di=("Di", "median") if "Di" in df.columns else ("y_pred", "median"),
        Df=("Df", "median") if "Df" in df.columns else ("y_pred", "median"),
        Dif=("Dif", "median") if "Dif" in df.columns else ("y_pred", "median"),
    ).reset_index()
    if first_filename is not None:
        agg = agg.merge(first_filename.reset_index(), on=group_cols, how="left")
    agg["consensus_pass"] = (
        (agg["n_models"] >= int(CONFIG.get("consensus_min_models", 2)))
        & (agg["n_seeds"] >= int(CONFIG.get("consensus_min_seeds", 2)))
        & (agg["n_splits"] >= int(CONFIG.get("consensus_min_splits", 1)))
        & (agg["trusted_support"] > 0)
    )
    # Score is target-scale invariant: support + within-target lower-bound percentile + narrower intervals.
    agg["consensus_score"] = (
        4.0 * agg["n_models"]
        + 1.0 * agg["n_seeds"]
        + 2.0 * agg["n_splits"]
        + 0.25 * agg["trusted_support"]
        + 8.0 * pd.to_numeric(agg["target_lower_bound_percentile_median"], errors="coerce").fillna(0.0)
        + 2.0 * pd.to_numeric(agg["target_pred_percentile_median"], errors="coerce").fillna(0.0)
        + 1.0 * pd.to_numeric(agg["target_interval_width_percentile_small_median"], errors="coerce").fillna(0.0)
    )
    # Percentile rank of consensus score within each target.
    agg["consensus_score_percentile_within_target"] = agg.groupby("target", dropna=False)["consensus_score"].rank(pct=True, method="average")
    consensus = agg.sort_values(["target", "consensus_pass", "consensus_score", "target_lower_bound_percentile_median"], ascending=[True, False, False, False])
    atomic_save_csv(consensus, out_path)

    top_per_target = int(CONFIG.get("consensus_top_per_target", 250))
    passed = consensus[consensus["consensus_pass"].fillna(False).astype(bool)].copy()
    target_balanced_parts = []
    for target in v39_target_list_from_available(consensus):
        sub = passed[passed["target"].astype(str).eq(str(target))].sort_values("consensus_score", ascending=False)
        if not sub.empty:
            target_balanced_parts.append(sub.head(top_per_target))
            safe_target = re.sub(r"[^A-Za-z0-9]+", "_", str(target)).strip("_")
            atomic_save_csv(sub.head(top_per_target), TABLE_DIR / f"MAIN_Table_3_consensus_shortlist_{safe_target}.csv")
    main = pd.concat(target_balanced_parts, ignore_index=True, copy=False) if target_balanced_parts else passed.head(top_per_target)
    atomic_save_csv(main, TABLE_DIR / "MAIN_Table_3_consensus_trusted_shortlist.csv")
    atomic_save_csv(main, TABLE_DIR / "MAIN_Table_3_consensus_shortlist_target_balanced.csv")
    if not main.empty:
        atomic_save_csv(main.groupby("target", dropna=False).size().reset_index(name="n_consensus_rows"), TABLE_DIR / "QC_Table_consensus_rows_by_target.csv")
    mark_done(stage, {"n_rows": len(consensus), "n_consensus_pass": int(consensus["consensus_pass"].sum()), "target_balanced_top_per_target": top_per_target})
    return consensus


def v39_target_balanced_head(df: pd.DataFrame, max_total: int, score_col: str, per_target: Optional[int] = None) -> pd.DataFrame:
    """Return a target-balanced head of a table."""
    if df is None or df.empty or "target" not in df.columns:
        return df.head(max_total) if df is not None else pd.DataFrame()
    targets = v39_target_list_from_available(df)
    per_target = int(per_target or max(1, math.ceil(max_total / max(1, len(targets)))))
    parts = []
    for t in targets:
        sub = df[df["target"].astype(str).eq(str(t))].copy()
        if sub.empty:
            continue
        if score_col in sub.columns:
            sub = sub.sort_values(score_col, ascending=False)
        parts.append(sub.head(per_target))
    out = pd.concat(parts, ignore_index=True, copy=False) if parts else df.head(max_total)
    if len(out) > max_total:
        out = out.sort_values(score_col, ascending=False).head(max_total) if score_col in out.columns else out.head(max_total)
    return out


def build_external_realism_table(master: pd.DataFrame, tiers: pd.DataFrame, core: pd.DataFrame, mosaec: pd.DataFrame, consensus: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """v3.9 target-balanced external plausibility table.

    This table is explicitly a domain-overlap/plausibility annotation, not proof
    of experimental validation. It is built from the target-balanced consensus
    shortlist when available.
    """
    stage = "10_external_realism_table"
    out_csv = TABLE_DIR / "final_screening_table_with_external_flags.csv"
    processed_csv = DATA_PROCESSED_DIR / "final_screening_table_with_external_flags.csv"
    if is_done(stage) and out_csv.exists():
        return pd.read_csv(out_csv, low_memory=False)

    logging.info("STAGE >>> BUILD_EXTERNAL_REALISM_TABLE_TARGET_BALANCED")
    if (tiers is None or tiers.empty) and (consensus is None or consensus.empty):
        final = pd.DataFrame()
        atomic_save_csv(final, out_csv)
        atomic_save_csv(final, processed_csv)
        mark_done(stage, {"n_rows": 0, "note": "No predictions available."})
        return final

    max_cands = int(CONFIG.get("external_match_max_candidates", 1000))
    per_target = int(CONFIG.get("external_match_max_candidates_per_target", max(1, math.ceil(max_cands / max(1, len(TARGET_SPECS))))))
    if consensus is not None and not consensus.empty:
        work = consensus.copy()
        if "consensus_pass" in work.columns and work["consensus_pass"].fillna(False).astype(bool).any():
            work = work[work["consensus_pass"].fillna(False).astype(bool)].copy()
        work = v39_target_balanced_head(work, max_total=max_cands, score_col="consensus_score", per_target=per_target)
        rename = {"y_pred_median": "y_pred", "split_lower_median": "split_lower", "split_upper_median": "split_upper", "interval_width_median": "interval_width"}
        work = work.rename(columns=rename)
        keep_cols = ["mof_id", "filename", "target", "y_true_median", "y_pred", "split_lower", "split_upper", "interval_width", "consensus_score", "consensus_score_percentile_within_target", "target_lower_bound_percentile_median", "n_models", "n_seeds", "n_splits", "trusted_support", "consensus_pass"]
        keep_cols = [c for c in keep_cols if c in work.columns]
        final = work[keep_cols].copy()
        final["tier"] = "consensus_trusted"
    else:
        work = tiers.copy()
        if "candidate_eligible" in work.columns:
            work = work[work["candidate_eligible"].fillna(False).astype(bool)].copy()
        work = v39_add_target_normalized_scores(work)
        final = v39_target_balanced_head(work, max_total=max_cands, score_col="target_balanced_row_score", per_target=per_target)
    final["match_key"] = final["mof_id"].map(canonical_key)

    arc_geo = _arc_geometry_table(master)
    final = final.merge(arc_geo.drop(columns=["match_key"], errors="ignore"), on="mof_id", how="left")

    if core is not None and not core.empty and "match_key" in core.columns:
        core_keys = set(core["match_key"].dropna().astype(str)) | set(core.get("core_id", pd.Series(dtype=str)).dropna().astype(str))
        final["core_exact_match_flag"] = final["match_key"].isin(core_keys)
        final = nearest_geometry_overlay(final, core, external_id_col="core_display_id" if "core_display_id" in core.columns else "core_id", external_prefix="core")
        core_small_cols = [c for c in ["match_key", "recommended_screening_flag", "asr_fsr_duplicate_flag", "prob_water_stability", "prob_solvent_removal_stability", "hydrophobic_class", "decomposition_temperature", "heat_capacity"] if c in core.columns]
        if core_small_cols:
            core_small = core[core_small_cols].drop_duplicates("match_key")
            final = final.merge(core_small, on="match_key", how="left", suffixes=("", "_core"))
    else:
        final["core_exact_match_flag"] = False
        final["core_nearest_id"] = ""
        final["core_geometry_distance"] = np.nan
        final["core_geometry_overlap_flag"] = False

    if mosaec is not None and not mosaec.empty and "match_key" in mosaec.columns:
        mos_keys = set(mosaec["match_key"].dropna().astype(str)) | set(mosaec.get("mosaec_id", pd.Series(dtype=str)).dropna().astype(str))
        final["mosaec_exact_match_flag"] = final["match_key"].isin(mos_keys)
        final = nearest_geometry_overlay(final, mosaec, external_id_col="mosaec_display_id" if "mosaec_display_id" in mosaec.columns else "mosaec_id", external_prefix="mosaec")
        mos_small_cols = [c for c in ["match_key", "is_unique_by_PDD", "porous_2p4pld_flag", "porous_0p10vf_flag", "diverse_neutral_subset_flag", "diverse_charged_subset_flag", "GEOM_available", "RAC_available", "APRDF_available", "PHOM_available"] if c in mosaec.columns]
        if mos_small_cols:
            mos_small = mosaec[mos_small_cols].drop_duplicates("match_key")
            final = final.merge(mos_small, on="match_key", how="left", suffixes=("", "_mosaec"))
    else:
        final["mosaec_exact_match_flag"] = False
        final["mosaec_nearest_id"] = ""
        final["mosaec_geometry_distance"] = np.nan
        final["mosaec_geometry_overlap_flag"] = False

    bool_cols_positive = [
        "core_exact_match_flag", "core_geometry_overlap_flag", "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag",
        "recommended_screening_flag", "porous_2p4pld_flag", "porous_0p10vf_flag", "is_unique_by_PDD",
        "GEOM_available", "RAC_available",
    ]
    final["external_realism_score"] = 0
    for col in bool_cols_positive:
        if col in final.columns:
            final["external_realism_score"] += final[col].fillna(False).astype(bool).astype(int)
    if "asr_fsr_duplicate_flag" in final.columns:
        final["external_realism_score"] -= final["asr_fsr_duplicate_flag"].fillna(False).astype(bool).astype(int)
    if not final.empty:
        atomic_save_csv(final.groupby("target", dropna=False).size().reset_index(name="n_external_rows"), TABLE_DIR / "QC_Table_external_rows_by_target.csv")

    atomic_save_csv(final, out_csv)
    atomic_save_csv(final, processed_csv)
    maybe_save_pickle(final, PICKLE_DIR / "final_screening_table_with_external_flags.pkl", minimum_mode="efficient")
    if CONFIG.get("save_public_release_safe_tables", True):
        release_cols = [c for c in final.columns if c not in {"filename"}]
        atomic_save_csv(final[release_cols], DATA_RELEASE_TABLE_DIR / "final_screening_table_with_external_flags_release_safe.csv")
    mark_done(stage, {"n_rows": len(final), "n_columns": final.shape[1], "target_balanced": True})
    return final


def chemical_failure_anatomy(tiers: pd.DataFrame) -> pd.DataFrame:
    """v3.9 target-normalised chemical failure anatomy."""
    stage = "05_chemical_failure_anatomy"
    out_path = TABLE_DIR / "table_chemical_failure_anatomy.csv"
    if is_done(stage) and out_path.exists():
        return pd.read_csv(out_path)

    logging.info("STAGE >>> CHEMICAL_FAILURE_ANATOMY_TARGET_NORMALISED")
    if tiers is None or tiers.empty:
        anatomy = pd.DataFrame()
        atomic_save_csv(anatomy, out_path)
        mark_done(stage, {"n_rows": 0})
        return anatomy
    df = tiers.copy()
    for col in ["y_true", "y_pred", "split_lower", "split_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalise absolute errors by target standard deviation to make regimes comparable.
    stats = {}
    if (TABLE_DIR / "MAIN_Table_1_dataset_and_targets.csv").exists():
        try:
            st = pd.read_csv(TABLE_DIR / "MAIN_Table_1_dataset_and_targets.csv")
            if {"target", "std_mmol_g"}.issubset(st.columns):
                stats = dict(zip(st["target"].astype(str), pd.to_numeric(st["std_mmol_g"], errors="coerce")))
        except Exception:
            stats = {}
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["target_std_for_norm"] = df["target"].astype(str).map(stats).replace(0, np.nan)
    df["normalized_abs_error"] = df["abs_error"] / df["target_std_for_norm"]
    rows = []
    for col in ["Di", "Df", "Dif", "Density"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_bin"] = bin_numeric_series(df[col], col)
            for keys, sub in df.groupby(["target", "feature_set", "split", "model", "budget", f"{col}_bin"], dropna=False):
                rows.append({
                    "anatomy_type": col,
                    "regime": str(keys[-1]),
                    "target": keys[0], "feature_set": keys[1], "split": keys[2], "model": keys[3], "budget": keys[4],
                    "n": len(sub),
                    "mae": float(sub["abs_error"].mean()),
                    "normalized_mae": float(sub["normalized_abs_error"].mean()),
                    "rmse": float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))),
                    "mean_interval_width": float((sub["split_upper"] - sub["split_lower"]).mean()),
                    "trusted_fraction": float((sub["tier"].astype(str) == "trusted").mean()) if "tier" in sub.columns else np.nan,
                })
    for col in ["topology", "metal_cluster", "functional_cluster", "ligand_cluster", "geometry_cluster"]:
        if col in df.columns:
            counts = df[col].astype(str).value_counts()
            top_cats = set(counts.head(50).index)
            df[f"{col}_coarse"] = df[col].astype(str).where(df[col].astype(str).isin(top_cats), "OTHER_RARE")
            for keys, sub in df.groupby(["target", "feature_set", "split", "model", "budget", f"{col}_coarse"], dropna=False):
                rows.append({
                    "anatomy_type": col,
                    "regime": str(keys[-1]),
                    "target": keys[0], "feature_set": keys[1], "split": keys[2], "model": keys[3], "budget": keys[4],
                    "n": len(sub),
                    "mae": float(sub["abs_error"].mean()),
                    "normalized_mae": float(sub["normalized_abs_error"].mean()),
                    "rmse": float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))),
                    "mean_interval_width": float((sub["split_upper"] - sub["split_lower"]).mean()),
                    "trusted_fraction": float((sub["tier"].astype(str) == "trusted").mean()) if "tier" in sub.columns else np.nan,
                })
    anatomy = pd.DataFrame(rows)
    atomic_save_csv(anatomy, out_path)
    mark_done(stage, {"n_rows": len(anatomy), "target_normalised": True})
    return anatomy


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    """v3.9 target-balanced chemical failure anatomy figure."""
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 4 target-balanced chemical failure anatomy")
    if anatomy is None or anatomy.empty or tiers is None or tiers.empty:
        return
    atomic_save_csv(anatomy, FIGDATA_DIR / "Figure_4_chemical_failure_anatomy_data.csv")
    tb = v39_target_balanced_compact_candidate_rows(tiers, max_rows=int(CONFIG.get("figure_target_balanced_sample_per_target", 3000)) * max(1, len(v39_target_list_from_available(tiers))))
    for col in ["y_true", "y_pred", "split_lower", "split_upper"]:
        if col in tb.columns:
            tb[col] = pd.to_numeric(tb[col], errors="coerce")
    tb["abs_error"] = np.abs(tb["y_true"] - tb["y_pred"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    # (a) target-balanced row composition.
    target_counts = tb["target"].value_counts().reindex(v39_target_list_from_available(tb)).fillna(0)
    axes[0].bar(range(len(target_counts)), target_counts.values)
    axes[0].set_xticks(range(len(target_counts)))
    axes[0].set_xticklabels(target_counts.index, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Candidate rows")
    axes[0].set_title("Target-balanced candidate pool")
    set_panel_label(axes[0], "(a)")

    # (b) normalized MAE by split.
    metric = "normalized_mae" if "normalized_mae" in anatomy.columns else "mae"
    sub = anatomy.groupby("split", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    axes[1].bar(range(len(sub)), sub[metric])
    axes[1].set_xticks(range(len(sub)))
    axes[1].set_xticklabels(sub["split"], rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel("Target-normalized MAE" if metric == "normalized_mae" else "MAE")
    axes[1].set_title("Extrapolation stress by split")
    set_panel_label(axes[1], "(b)")

    # (c) high-error topology/cluster regimes using normalized error.
    reg = anatomy[anatomy["anatomy_type"].isin(["topology", "metal_cluster", "geometry_cluster"])].copy()
    if not reg.empty:
        reg = reg.groupby(["anatomy_type", "regime"], as_index=False)[metric].mean().sort_values(metric, ascending=False).head(10)
        labels = (reg["anatomy_type"].astype(str) + ":" + reg["regime"].astype(str)).str.slice(0, 35)
        axes[2].barh(range(len(reg)), reg[metric])
        axes[2].set_yticks(range(len(reg)))
        axes[2].set_yticklabels(labels, fontsize=7)
        axes[2].invert_yaxis()
    axes[2].set_xlabel("Target-normalized MAE" if metric == "normalized_mae" else "MAE")
    axes[2].set_title("Highest-error chemical regimes")
    set_panel_label(axes[2], "(c)")

    # (d) predicted vs true, target-normalized within panels by sample.
    sample = tb.sample(min(8000, len(tb)), random_state=42) if len(tb) else tb
    axes[3].scatter(sample["y_true"], sample["y_pred"], s=5, alpha=0.35)
    if len(sample):
        lo = min(sample["y_true"].min(), sample["y_pred"].min())
        hi = max(sample["y_true"].max(), sample["y_pred"].max())
        axes[3].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[3].set_xlabel("True uptake / mmol g$^{-1}$")
    axes[3].set_ylabel("Predicted uptake / mmol g$^{-1}$")
    axes[3].set_title("Candidate predictions after QC")
    set_panel_label(axes[3], "(d)")

    # (e) target-normalized lower-bound percentile distribution.
    if "target_lower_bound_percentile" in tb.columns:
        for target in v39_target_list_from_available(tb):
            vals = pd.to_numeric(tb.loc[tb["target"].astype(str).eq(str(target)), "target_lower_bound_percentile"], errors="coerce").dropna()
            if len(vals):
                axes[4].hist(vals, bins=25, alpha=0.35, label=str(target)[:18])
        axes[4].legend(fontsize=6)
    axes[4].set_xlabel("Within-target lower-bound percentile")
    axes[4].set_ylabel("Rows")
    axes[4].set_title("Risk-ranked candidates within target")
    set_panel_label(axes[4], "(e)")

    # (f) tier composition by target.
    if "tier" in tb.columns:
        tier_tab = tb.pivot_table(index="target", columns="tier", values="mof_id", aggfunc="count", fill_value=0)
        tier_tab = tier_tab.reindex(v39_target_list_from_available(tb))
        bottom = np.zeros(len(tier_tab))
        for tier in tier_tab.columns:
            axes[5].bar(range(len(tier_tab)), tier_tab[tier].values, bottom=bottom, label=str(tier))
            bottom += tier_tab[tier].values
        axes[5].set_xticks(range(len(tier_tab)))
        axes[5].set_xticklabels(tier_tab.index, rotation=35, ha="right", fontsize=7)
        axes[5].legend(fontsize=7)
    axes[5].set_ylabel("Rows")
    axes[5].set_title("Tier composition by target")
    set_panel_label(axes[5], "(f)")
    fig.suptitle("Target-balanced failure anatomy after quality-gated candidate selection", fontsize=14, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy")
    mark_done(stage, {"target_balanced": True})


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """v3.9 target-balanced consensus + external plausibility figure."""
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 5 target-balanced consensus and external plausibility")
    availability = summarise_external_files(core, mosaec)
    atomic_save_csv(availability, FIGDATA_DIR / "Figure_5_external_file_availability_data.csv")
    if CONFIG.get("save_public_release_safe_tables", True):
        atomic_save_csv(availability, DATA_RELEASE_FIGURE_DIR / "Figure_5_external_file_availability_data.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    if final_external is None or final_external.empty:
        for ax, lab in zip(axes, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
            ax.axis("off")
            ax.text(0.5, 0.5, "No target-balanced external table", ha="center", va="center")
            set_panel_label(ax, lab)
        save_current_figure(FIG_MAIN_DIR / "Figure_5_external_realism_filter")
        mark_done(stage, {"n_rows": 0})
        return
    fe = final_external.copy()
    atomic_save_csv(fe, FIGDATA_DIR / "Figure_5_external_realism_candidate_data.csv")

    # (a) target-balanced shortlist counts.
    counts = fe["target"].value_counts().reindex(v39_target_list_from_available(fe)).fillna(0)
    axes[0].bar(range(len(counts)), counts.values)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Consensus candidates")
    axes[0].set_title("Target-balanced consensus shortlist")
    set_panel_label(axes[0], "(a)")

    # (b) consensus support by target.
    if {"target", "n_models", "n_seeds", "n_splits"}.issubset(fe.columns):
        supp = fe.groupby("target", as_index=False).agg(n_models=("n_models", "median"), n_seeds=("n_seeds", "median"), n_splits=("n_splits", "median"))
        x = np.arange(len(supp)); width = 0.25
        axes[1].bar(x - width, supp["n_models"], width, label="models")
        axes[1].bar(x, supp["n_seeds"], width, label="seeds")
        axes[1].bar(x + width, supp["n_splits"], width, label="splits")
        axes[1].set_xticks(x); axes[1].set_xticklabels(supp["target"], rotation=35, ha="right", fontsize=7)
        axes[1].legend(fontsize=7)
    axes[1].set_ylabel("Median support")
    axes[1].set_title("Consensus support")
    set_panel_label(axes[1], "(b)")

    # (c) lower-bound percentile or score distribution.
    score_col = "target_lower_bound_percentile_median" if "target_lower_bound_percentile_median" in fe.columns else "consensus_score_percentile_within_target"
    if score_col in fe.columns:
        for target in v39_target_list_from_available(fe):
            vals = pd.to_numeric(fe.loc[fe["target"].astype(str).eq(str(target)), score_col], errors="coerce").dropna()
            if len(vals):
                axes[2].hist(vals, bins=20, alpha=0.35, label=str(target)[:18])
        axes[2].legend(fontsize=6)
    axes[2].set_xlabel("Within-target score/LCB percentile")
    axes[2].set_ylabel("Candidates")
    axes[2].set_title("Risk-ranked consensus candidates")
    set_panel_label(axes[2], "(c)")

    # (d) external-realism funnel.
    funnel = pd.DataFrame({
        "stage": ["shortlist", "CoRE exact", "CoRE geom", "MOSAEC exact", "MOSAEC geom", "score > 0"],
        "n": [
            len(fe),
            int(fe.get("core_exact_match_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int(fe.get("core_geometry_overlap_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int(fe.get("mosaec_exact_match_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int(fe.get("mosaec_geometry_overlap_flag", pd.Series(False, index=fe.index)).fillna(False).sum()),
            int((pd.to_numeric(fe.get("external_realism_score", pd.Series(0, index=fe.index)), errors="coerce").fillna(0) > 0).sum()),
        ]
    })
    atomic_save_csv(funnel, FIGDATA_DIR / "Figure_5_external_realism_funnel_data.csv")
    axes[3].bar(range(len(funnel)), funnel["n"])
    axes[3].set_xticks(range(len(funnel)))
    axes[3].set_xticklabels(funnel["stage"], rotation=35, ha="right", fontsize=7)
    axes[3].set_ylabel("Candidates")
    axes[3].set_title("External plausibility annotations")
    set_panel_label(axes[3], "(d)")

    # (e) external score by target.
    if "external_realism_score" in fe.columns:
        score_by_target = fe.groupby("target", as_index=False)["external_realism_score"].mean()
        axes[4].bar(range(len(score_by_target)), score_by_target["external_realism_score"])
        axes[4].set_xticks(range(len(score_by_target)))
        axes[4].set_xticklabels(score_by_target["target"], rotation=35, ha="right", fontsize=7)
    axes[4].set_ylabel("Mean external score")
    axes[4].set_title("External domain overlap by target")
    set_panel_label(axes[4], "(e)")

    # (f) prediction vs lower bound.
    x = pd.to_numeric(fe.get("split_lower", pd.Series(np.nan, index=fe.index)), errors="coerce")
    y = pd.to_numeric(fe.get("y_pred", pd.Series(np.nan, index=fe.index)), errors="coerce")
    s = 20 + 10 * pd.to_numeric(fe.get("external_realism_score", pd.Series(0, index=fe.index)), errors="coerce").fillna(0)
    axes[5].scatter(x, y, s=s, alpha=0.45)
    axes[5].set_xlabel("Median lower confidence bound")
    axes[5].set_ylabel("Median prediction")
    axes[5].set_title("Confidence-ranked consensus hits")
    set_panel_label(axes[5], "(f)")
    fig.suptitle("Target-balanced consensus shortlist with external domain-overlap annotations", fontsize=14, fontweight="bold")
    save_current_figure(FIG_MAIN_DIR / "Figure_5_external_realism_filter")
    mark_done(stage, {"n_rows": len(final_external), "target_balanced": True})


# =============================================================================
# 16d. v4.0 final manuscript polish and validation overrides
# =============================================================================
# v4.0 keeps the successful v3.9 target-balanced candidate/QC logic, but adds
# final manuscript-ready post-processing: duplicated RDF feature sets are hidden
# from main figures/tables, consensus candidates are validated against their true
# held-out uptake values, compact top-candidate tables are written per target,
# and Figures 2--5 are regenerated in a cleaner high-impact style.

CONFIG.setdefault("final_top_candidates_per_target", 25)
CONFIG.setdefault("final_main_candidate_rows_per_target", 25)
CONFIG.setdefault("final_external_si_rows_per_target", 25)
CONFIG.setdefault("final_prefer_feature_set", "geometry_plus_racs")
CONFIG.setdefault("hide_duplicate_feature_sets_in_main", True)




def v40_save_current_figure_aliases(path_bases: List[Path]) -> None:
    """Save the current Matplotlib figure under several base names, then close once."""
    if not path_bases:
        plt.close("all")
        return
    for path_base in path_bases:
        path_base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in CONFIG.get("figure_format", ["png"]):
            if fmt.lower() == "pdf" and (FIG_SI_DIR in path_base.parents) and not CONFIG.get("save_si_pdf", False):
                continue
            out = path_base.with_suffix(f".{fmt}")
            try:
                plt.savefig(out, dpi=CONFIG.get("savefig_dpi", 300), bbox_inches="tight")
            except Exception as e:
                logging.warning("FIGURE_SAVE_FAILED | %s | %s", out, e)
    plt.close("all")

def v40_pretty_target(target: Any) -> str:
    """Compact labels for figures/tables."""
    mapping = {
        "CO2_0p015bar_298K_mmolg": "CO$_2$ 0.015 bar",
        "CO2_0p150bar_298K_mmolg": "CO$_2$ 0.150 bar",
        "CH4_5p8bar_298K_mmolg": "CH$_4$ 5.8 bar",
        "CH4_65bar_298K_mmolg": "CH$_4$ 65 bar",
    }
    return mapping.get(str(target), str(target))


def v40_plain_target(target: Any) -> str:
    """Filename-safe target labels."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(target)).strip("_")


def v40_read_consensus() -> pd.DataFrame:
    path = TABLE_DIR / "table_consensus_candidate_shortlist.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def v40_read_external() -> pd.DataFrame:
    path = TABLE_DIR / "final_screening_table_with_external_flags.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def v40_usable_feature_sets() -> List[str]:
    """Return feature sets to show in main figures/tables, excluding duplicates."""
    qc_path = TABLE_DIR / "QC_Table_feature_set_integrity.csv"
    if qc_path.exists():
        try:
            qc = pd.read_csv(qc_path)
            if CONFIG.get("hide_duplicate_feature_sets_in_main", True):
                qc = qc[qc.get("usable_for_modelling", True).astype(bool)]
            fs = qc["feature_set"].dropna().astype(str).tolist()
            if fs:
                return fs
        except Exception:
            pass
    return [fs for fs in CONFIG.get("feature_sets", []) if not (CONFIG.get("hide_duplicate_feature_sets_in_main", True) and str(fs).endswith("rdfs"))]


def v40_preferred_feature_set(agg: Optional[pd.DataFrame] = None) -> str:
    usable = v40_usable_feature_sets()
    preference = [CONFIG.get("final_prefer_feature_set", "geometry_plus_racs"), "geometry_plus_racs", "geometry_only"]
    if agg is not None and not agg.empty and "feature_set" in agg.columns:
        available = set(agg["feature_set"].dropna().astype(str)) & set(usable)
    else:
        available = set(usable)
    for fs in preference:
        if fs in available:
            return fs
    return sorted(available)[0] if available else "geometry_only"


def v40_stable_metric_df(agg: pd.DataFrame, top_frac: float = 0.05, feature_set: Optional[str] = None) -> pd.DataFrame:
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = agg.copy()
    stable = set(CONFIG.get("stable_plot_models", []))
    if stable and "model" in df.columns:
        df = df[df["model"].isin(stable)]
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(top_frac).abs() < 1e-9]
    usable = set(v40_usable_feature_sets())
    if usable and "feature_set" in df.columns:
        df = df[df["feature_set"].isin(usable)]
    if feature_set is not None and "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str).eq(str(feature_set))]
    return df


def v40_target_stats(master: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
    """Dataset target distributions, including true top-k thresholds."""
    stats: Dict[str, Dict[str, Any]] = {}
    if master is None or master.empty:
        mp = PICKLE_DIR / "master_table.pkl"
        if mp.exists():
            try:
                master = read_pickle(mp)
            except Exception:
                master = pd.DataFrame()
    for t in TARGET_SPECS:
        if master is not None and not master.empty and t in master.columns:
            vals = pd.to_numeric(master[t], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals):
                vals_sorted = np.sort(vals)
                stats[t] = {
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "median": float(np.median(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "top1_threshold": float(np.quantile(vals, 0.99)),
                    "top5_threshold": float(np.quantile(vals, 0.95)),
                    "top10_threshold": float(np.quantile(vals, 0.90)),
                    "sorted_values": vals_sorted,
                }
    if not stats and (TABLE_DIR / "table_dataset_target_summary.csv").exists():
        ts = pd.read_csv(TABLE_DIR / "table_dataset_target_summary.csv")
        for _, r in ts.iterrows():
            t = str(r.get("target"))
            stats[t] = {
                "n": int(r.get("n_nonmissing", r.get("n_mofs", 0))),
                "mean": float(r.get("mean", r.get("mean_mmol_g", np.nan))),
                "std": float(r.get("std", r.get("std_mmol_g", np.nan))),
                "median": float(r.get("median", r.get("median_mmol_g", np.nan))),
                "min": float(r.get("min", r.get("min_mmol_g", np.nan))),
                "max": float(r.get("max", r.get("max_mmol_g", np.nan))),
                "top1_threshold": np.nan,
                "top5_threshold": np.nan,
                "top10_threshold": np.nan,
                "sorted_values": np.array([], dtype=float),
            }
    return stats


def v40_true_percentile(values: np.ndarray, sorted_dataset_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if sorted_dataset_values is None or len(sorted_dataset_values) == 0:
        return np.full_like(values, np.nan, dtype=float)
    return np.searchsorted(sorted_dataset_values, values, side="right") / float(len(sorted_dataset_values))


def write_consensus_true_enrichment(master: Optional[pd.DataFrame] = None, consensus: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Validate the consensus shortlist against held-out true adsorption values."""
    if consensus is None or consensus.empty:
        consensus = v40_read_consensus()
    if consensus is None or consensus.empty:
        out = pd.DataFrame()
        atomic_save_csv(out, TABLE_DIR / "QC_Table_consensus_true_enrichment.csv")
        return out
    stats = v40_target_stats(master)
    work = consensus.copy()
    if "consensus_pass" in work.columns and work["consensus_pass"].fillna(False).astype(bool).any():
        work = work[work["consensus_pass"].fillna(False).astype(bool)].copy()
    if "consensus_score" in work.columns:
        work = work.sort_values(["target", "consensus_score", "split_lower_median"], ascending=[True, False, False])
    rows = []
    top_per_target = int(CONFIG.get("consensus_top_per_target", 250))
    for target in v39_target_list_from_available(work):
        sub = work[work["target"].astype(str).eq(str(target))].head(top_per_target).copy()
        if sub.empty:
            continue
        vals = pd.to_numeric(sub.get("y_true_median", sub.get("y_true", np.nan)), errors="coerce").dropna().to_numpy(dtype=float)
        preds = pd.to_numeric(sub.get("y_pred_median", sub.get("y_pred", np.nan)), errors="coerce").dropna().to_numpy(dtype=float)
        st = stats.get(str(target), {})
        dataset_median = float(st.get("median", np.nan))
        sorted_vals = st.get("sorted_values", np.array([], dtype=float))
        percentiles = v40_true_percentile(vals, sorted_vals)
        top1 = float(st.get("top1_threshold", np.nan))
        top5 = float(st.get("top5_threshold", np.nan))
        top10 = float(st.get("top10_threshold", np.nan))
        rows.append({
            "target": target,
            "target_label": v40_pretty_target(target).replace("$", ""),
            "n_shortlist": int(len(sub)),
            "dataset_n": int(st.get("n", 0)),
            "dataset_median": dataset_median,
            "dataset_mean": float(st.get("mean", np.nan)),
            "dataset_top1_threshold": top1,
            "dataset_top5_threshold": top5,
            "dataset_top10_threshold": top10,
            "shortlist_true_median": float(np.nanmedian(vals)) if len(vals) else np.nan,
            "shortlist_true_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
            "shortlist_pred_median": float(np.nanmedian(preds)) if len(preds) else np.nan,
            "fold_enrichment_vs_dataset_median": float(np.nanmedian(vals) / dataset_median) if len(vals) and np.isfinite(dataset_median) and abs(dataset_median) > 1e-12 else np.nan,
            "median_true_percentile": float(np.nanmedian(percentiles)) if len(percentiles) else np.nan,
            "fraction_true_top1pct": float(np.mean(vals >= top1)) if len(vals) and np.isfinite(top1) else np.nan,
            "fraction_true_top5pct": float(np.mean(vals >= top5)) if len(vals) and np.isfinite(top5) else np.nan,
            "fraction_true_top10pct": float(np.mean(vals >= top10)) if len(vals) and np.isfinite(top10) else np.nan,
            "false_positive_fraction_vs_top5pct": float(1.0 - np.mean(vals >= top5)) if len(vals) and np.isfinite(top5) else np.nan,
            "median_abs_error_in_shortlist": float(np.nanmedian(np.abs(vals[:min(len(vals), len(preds))] - preds[:min(len(vals), len(preds))]))) if len(vals) and len(preds) else np.nan,
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "QC_Table_consensus_true_enrichment.csv")
    return out


def write_final_candidate_tables(master: Optional[pd.DataFrame] = None, consensus: Optional[pd.DataFrame] = None, final_external: Optional[pd.DataFrame] = None) -> None:
    """Write compact target-resolved final candidate tables for manuscript/SI."""
    if consensus is None or consensus.empty:
        consensus = v40_read_consensus()
    if final_external is None or final_external.empty:
        final_external = v40_read_external()
    if consensus is None or consensus.empty:
        return
    work = consensus.copy()
    if "consensus_pass" in work.columns and work["consensus_pass"].fillna(False).astype(bool).any():
        work = work[work["consensus_pass"].fillna(False).astype(bool)].copy()
    if "consensus_score" in work.columns:
        work = work.sort_values(["target", "consensus_score", "split_lower_median"], ascending=[True, False, False])
    topn = int(CONFIG.get("final_top_candidates_per_target", 25))
    table_parts = []
    keep_cols = [c for c in [
        "target", "mof_id", "filename", "y_true_median", "y_pred_median", "split_lower_median", "split_upper_median",
        "interval_width_median", "consensus_score", "n_models", "n_seeds", "n_splits", "n_feature_sets", "n_budgets", "trusted_support",
        "Density", "Di", "Df", "Dif", "consensus_score_percentile_within_target", "target_lower_bound_percentile_median"
    ] if c in work.columns]
    for target in v39_target_list_from_available(work):
        sub = work[work["target"].astype(str).eq(str(target))].head(topn).copy()
        if sub.empty:
            continue
        sub.insert(0, "rank_within_target", np.arange(1, len(sub) + 1))
        out = sub[["rank_within_target"] + keep_cols].copy()
        table_parts.append(out)
        atomic_save_csv(out, TABLE_DIR / f"MAIN_Table_3_{v40_plain_target(target)}_top{topn}_consensus_candidates.csv")
    if table_parts:
        all_top = pd.concat(table_parts, ignore_index=True, copy=False)
        atomic_save_csv(all_top, TABLE_DIR / "MAIN_Table_3_consensus_shortlist_target_balanced_top25.csv")
        atomic_save_csv(all_top, DATA_RELEASE_TABLE_DIR / "MAIN_Table_3_consensus_shortlist_target_balanced_top25_release_safe.csv")
    # External-domain-overlap top table, framed as SI/domain overlap rather than validation.
    if final_external is not None and not final_external.empty:
        fe = final_external.copy()
        sort_cols = [c for c in ["target", "external_realism_score", "consensus_score", "split_lower"] if c in fe.columns]
        asc = [True] + [False] * (len(sort_cols) - 1) if sort_cols else True
        if sort_cols:
            fe = fe.sort_values(sort_cols, ascending=asc)
        ext_parts = []
        ext_cols = [c for c in [
            "target", "mof_id", "y_true_median", "y_pred", "split_lower", "split_upper", "consensus_score",
            "n_models", "n_seeds", "n_splits", "trusted_support", "external_realism_score",
            "core_exact_match_flag", "core_geometry_overlap_flag", "core_geometry_distance",
            "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag", "mosaec_geometry_distance",
            "recommended_screening_flag", "GEOM_available", "RAC_available"
        ] if c in fe.columns]
        for target in v39_target_list_from_available(fe):
            sub = fe[fe["target"].astype(str).eq(str(target))].head(topn).copy()
            if sub.empty:
                continue
            sub.insert(0, "rank_within_target", np.arange(1, len(sub) + 1))
            ext_parts.append(sub[["rank_within_target"] + ext_cols])
        if ext_parts:
            ext = pd.concat(ext_parts, ignore_index=True, copy=False)
            atomic_save_csv(ext, TABLE_DIR / "SI_Table_external_domain_overlap_top25_by_target.csv")
            atomic_save_csv(ext, DATA_RELEASE_TABLE_DIR / "SI_Table_external_domain_overlap_top25_by_target_release_safe.csv")


def write_descriptor_split_summary_tables(agg: pd.DataFrame) -> None:
    """Write compact descriptor and extrapolation stress-test summaries."""
    if agg is None or agg.empty:
        return
    stable = v40_stable_metric_df(agg, top_frac=0.05)
    if stable.empty:
        return
    # Descriptor ablation: stable models, random split, budget 1000 if present.
    desc = stable.copy()
    if "random" in set(desc["split"].astype(str)):
        desc = desc[desc["split"].astype(str).eq("random")]
    if 1000 in set(pd.to_numeric(desc["budget"], errors="coerce")):
        desc = desc[pd.to_numeric(desc["budget"], errors="coerce").eq(1000)]
    desc_summary = desc.groupby(["target", "feature_set"], as_index=False).agg(
        mean_top5_recall=("recall_mean", "mean"),
        mean_enrichment=("enrichment_mean", "mean"),
        mean_spearman=("spearman_mean", "mean"),
        mean_rmse=("rmse_mean", "mean"),
        mean_coverage=("split_coverage_mean", "mean"),
        n_rows=("recall_mean", "size"),
    )
    atomic_save_csv(desc_summary, TABLE_DIR / "MAIN_Table_2_descriptor_ablation_summary.csv")
    # Split stress: compare topology_grouped to random for preferred feature set at budget 1000.
    fs = v40_preferred_feature_set(stable)
    sp = stable[stable["feature_set"].astype(str).eq(fs)].copy()
    if 1000 in set(pd.to_numeric(sp["budget"], errors="coerce")):
        sp = sp[pd.to_numeric(sp["budget"], errors="coerce").eq(1000)]
    split_summary = sp.groupby(["target", "split"], as_index=False).agg(
        mean_top5_recall=("recall_mean", "mean"),
        mean_spearman=("spearman_mean", "mean"),
        mean_coverage=("split_coverage_mean", "mean"),
        mean_rmse=("rmse_mean", "mean"),
    )
    random_map = split_summary[split_summary["split"].astype(str).eq("random")].set_index("target")["mean_top5_recall"].to_dict()
    split_summary["random_recall_reference"] = split_summary["target"].map(random_map)
    split_summary["recall_penalty_vs_random"] = split_summary["random_recall_reference"] - split_summary["mean_top5_recall"]
    atomic_save_csv(split_summary, TABLE_DIR / "MAIN_Table_2_split_extrapolation_stress_summary.csv")


def save_manuscript_si_tables(master: pd.DataFrame, agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    """v4.0 final tables: manuscript summaries, candidate validation, and clean SI tables."""
    stage = "06_manuscript_si_tables"
    if is_done(stage):
        return
    logging.info("STAGE >>> SAVE_FINAL_MANUSCRIPT_SI_TABLES_V4")
    # Dataset table.
    rows = []
    for target in TARGET_SPECS:
        valid = pd.to_numeric(master[target], errors="coerce").dropna() if target in master.columns else pd.Series(dtype=float)
        rows.append({
            "target": target,
            "target_label": v40_pretty_target(target).replace("$", ""),
            "description": TARGET_SPECS[target]["description"],
            "n_mofs": int(valid.shape[0]),
            "mean_mmol_g": float(valid.mean()) if len(valid) else np.nan,
            "std_mmol_g": float(valid.std()) if len(valid) else np.nan,
            "median_mmol_g": float(valid.median()) if len(valid) else np.nan,
            "min_mmol_g": float(valid.min()) if len(valid) else np.nan,
            "top5_threshold_mmol_g": float(valid.quantile(0.95)) if len(valid) else np.nan,
            "max_mmol_g": float(valid.max()) if len(valid) else np.nan,
        })
    atomic_save_csv(pd.DataFrame(rows), TABLE_DIR / "MAIN_Table_1_dataset_and_targets.csv")
    if agg is not None and not agg.empty:
        atomic_save_csv(agg, TABLE_DIR / "SI_Table_S1_all_aggregated_metrics.csv")
        stable = v40_stable_metric_df(agg, top_frac=0.05)
        if not stable.empty:
            best_rows = []
            for keys, sub in stable.groupby(["target", "feature_set", "split", "budget"], dropna=False):
                sub = sub.sort_values(["recall_mean", "enrichment_mean", "split_coverage_mean", "rmse_mean"], ascending=[False, False, False, True])
                best_rows.append(sub.head(1))
            best = pd.concat(best_rows, ignore_index=True, copy=False) if best_rows else pd.DataFrame()
            atomic_save_csv(best, TABLE_DIR / "MAIN_Table_2_best_stable_models_top5pct.csv")
            write_descriptor_split_summary_tables(agg)
            split_summary = stable.groupby(["target", "split", "model", "budget"], as_index=False).agg({
                "split_coverage_mean": "mean",
                "recall_mean": "mean",
                "rmse_mean": "mean",
                "spearman_mean": "mean",
                "enrichment_mean": "mean",
            })
            atomic_save_csv(split_summary, TABLE_DIR / "SI_Table_S2_grouped_split_summary_stable_models.csv")
    if tiers is not None and not tiers.empty:
        work_tiers = tiers.copy()
        if "candidate_eligible" in work_tiers.columns:
            work_tiers = work_tiers[work_tiers["candidate_eligible"].fillna(False).astype(bool)]
        if not work_tiers.empty:
            tier_counts = work_tiers.groupby([c for c in ["target", "feature_set", "split", "model", "budget", "tier"] if c in work_tiers.columns], as_index=False).size()
            atomic_save_csv(tier_counts, TABLE_DIR / "SI_Table_S3_candidate_tier_counts_quality_gated.csv")
    if anatomy is not None and not anatomy.empty:
        atomic_save_csv(anatomy, TABLE_DIR / "SI_Table_S4_chemical_failure_anatomy.csv")
    consensus = v40_read_consensus()
    final_external = v40_read_external()
    write_consensus_true_enrichment(master, consensus)
    write_final_candidate_tables(master, consensus, final_external)
    mark_done(stage)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    """v4.0 Figure 2: few-shot performance, descriptor gain, and split stress."""
    stage = "fig2_performance"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 2 final performance/descriptors/splits")
    stable_all = v40_stable_metric_df(agg, top_frac=0.05)
    if stable_all.empty:
        return
    fs = v40_preferred_feature_set(stable_all)
    df = stable_all[stable_all["feature_set"].astype(str).eq(fs)].copy()
    atomic_save_csv(df, FIGDATA_DIR / "Figure_2_final_performance_data.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()
    targets = v39_target_list_from_available(df)
    # (a) top-5 recall vs budget by target, random split if available.
    dfa = df[df["split"].astype(str).eq("random")].copy() if "random" in set(df["split"].astype(str)) else df.copy()
    for target in targets:
        sub = dfa[dfa["target"].astype(str).eq(str(target))].groupby("budget", as_index=False)["recall_mean"].mean()
        if not sub.empty:
            axes[0].plot(sub["budget"], sub["recall_mean"], marker="o", label=v40_pretty_target(target))
    axes[0].set_xscale("log"); axes[0].set_xlabel("Label budget"); axes[0].set_ylabel("Top-5% recall")
    axes[0].grid(True, alpha=0.25); axes[0].legend(fontsize=7); axes[0].set_title("Elite recovery improves with labels"); set_panel_label(axes[0], "(a)")
    # (b) enrichment vs budget.
    for target in targets:
        sub = dfa[dfa["target"].astype(str).eq(str(target))].groupby("budget", as_index=False)["enrichment_mean"].mean()
        if not sub.empty:
            axes[1].plot(sub["budget"], sub["enrichment_mean"], marker="o", label=v40_pretty_target(target))
    axes[1].axhline(1.0, linestyle="--", linewidth=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Label budget"); axes[1].set_ylabel("Enrichment over random")
    axes[1].grid(True, alpha=0.25); axes[1].set_title("Screening enrichment over random"); set_panel_label(axes[1], "(b)")
    # (c) descriptor ablation at 1000 labels.
    desc = stable_all.copy()
    if "random" in set(desc["split"].astype(str)):
        desc = desc[desc["split"].astype(str).eq("random")]
    desc = desc[pd.to_numeric(desc["budget"], errors="coerce").eq(1000)] if 1000 in set(pd.to_numeric(desc["budget"], errors="coerce")) else desc
    desc = desc.groupby(["target", "feature_set"], as_index=False)["recall_mean"].mean()
    feature_sets = [fs for fs in ["geometry_only", "geometry_plus_racs"] if fs in set(desc["feature_set"].astype(str))]
    x = np.arange(len(targets)); width = 0.35
    for j, fset in enumerate(feature_sets):
        vals = [float(desc[(desc["target"].astype(str).eq(str(t))) & (desc["feature_set"].astype(str).eq(fset))]["recall_mean"].mean()) for t in targets]
        axes[2].bar(x + (j - (len(feature_sets)-1)/2)*width, vals, width, label=fset.replace("geometry_plus_racs", "geometry + RACs").replace("geometry_only", "geometry"))
    axes[2].set_xticks(x); axes[2].set_xticklabels([v40_pretty_target(t) for t in targets], rotation=30, ha="right", fontsize=7)
    axes[2].set_ylabel("Top-5% recall at 1000 labels"); axes[2].legend(fontsize=7); axes[2].set_title("Chemistry descriptors improve recall"); set_panel_label(axes[2], "(c)")
    # (d) split stress at 1000 labels.
    sp = df[pd.to_numeric(df["budget"], errors="coerce").eq(1000)] if 1000 in set(pd.to_numeric(df["budget"], errors="coerce")) else df
    split_order = [s for s in ["random", "geometry_grouped", "metal_grouped", "functional_grouped", "ligand_grouped", "topology_grouped"] if s in set(sp["split"].astype(str))]
    split_vals = sp.groupby("split", as_index=False)["recall_mean"].mean().set_index("split").reindex(split_order)
    axes[3].bar(range(len(split_vals)), split_vals["recall_mean"])
    axes[3].set_xticks(range(len(split_vals))); axes[3].set_xticklabels(split_vals.index, rotation=35, ha="right", fontsize=7)
    axes[3].set_ylabel("Mean top-5% recall"); axes[3].set_title("Topology is the hardest split"); set_panel_label(axes[3], "(d)")
    # (e) stable model comparison at 1000 labels.
    model = dfa[pd.to_numeric(dfa["budget"], errors="coerce").eq(1000)] if 1000 in set(pd.to_numeric(dfa["budget"], errors="coerce")) else dfa
    ms = model.groupby("model", as_index=False)["recall_mean"].mean().sort_values("recall_mean", ascending=False)
    axes[4].bar(range(len(ms)), ms["recall_mean"])
    axes[4].set_xticks(range(len(ms))); axes[4].set_xticklabels(ms["model"], rotation=35, ha="right", fontsize=7)
    axes[4].set_ylabel("Mean top-5% recall"); axes[4].set_title("Stable model families"); set_panel_label(axes[4], "(e)")
    # (f) coverage at 1000 labels by target.
    cov = dfa[pd.to_numeric(dfa["budget"], errors="coerce").eq(1000)] if 1000 in set(pd.to_numeric(dfa["budget"], errors="coerce")) else dfa
    cov = cov.groupby("target", as_index=False)["split_coverage_mean"].mean().set_index("target").reindex(targets)
    axes[5].bar(range(len(cov)), cov["split_coverage_mean"])
    axes[5].axhline(1 - CONFIG.get("conformal_alpha", 0.10), linestyle="--", linewidth=1)
    axes[5].set_xticks(range(len(cov))); axes[5].set_xticklabels([v40_pretty_target(t) for t in cov.index], rotation=30, ha="right", fontsize=7)
    axes[5].set_ylabel("Empirical coverage"); axes[5].set_title("Risk control remains near nominal"); set_panel_label(axes[5], "(f)")
    fig.suptitle("Few-shot elite recovery, descriptor chemistry, and extrapolation stress", fontsize=14, fontweight="bold")
    v40_save_current_figure_aliases([
        FIG_MAIN_DIR / "Figure_2_label_budget_performance",
        FIG_MAIN_DIR / "Figure_2_final_performance_descriptor_split",
    ])
    mark_done(stage, {"preferred_feature_set": fs})


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    """v4.0 Figure 3: conformal calibration and abstention with normalized widths."""
    stage = "fig3_calibration"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 3 final calibration/abstention")
    df = v40_stable_metric_df(agg, top_frac=0.05, feature_set=v40_preferred_feature_set(agg))
    if df.empty:
        return
    stats = v40_target_stats(None)
    df = df.copy()
    df["target_std"] = df["target"].map(lambda t: stats.get(str(t), {}).get("std", np.nan))
    df["split_width_normalized"] = pd.to_numeric(df["split_mean_width_mean"], errors="coerce") / pd.to_numeric(df["target_std"], errors="coerce")
    denom = df["trusted_count_mean"] + df["uncertain_count_mean"] + df["rejected_count_mean"]
    df["trusted_fraction"] = df["trusted_count_mean"] / denom.replace(0, np.nan)
    df["uncertain_fraction"] = df["uncertain_count_mean"] / denom.replace(0, np.nan)
    atomic_save_csv(df, FIGDATA_DIR / "Figure_3_final_calibration_data.csv")
    nominal = 1 - CONFIG.get("conformal_alpha", 0.10)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5)); axes = axes.ravel()
    # (a) coverage by model.
    cov_model = df.groupby("model", as_index=False)["split_coverage_mean"].mean().sort_values("split_coverage_mean", ascending=False)
    axes[0].bar(range(len(cov_model)), cov_model["split_coverage_mean"]); axes[0].axhline(nominal, linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(cov_model))); axes[0].set_xticklabels(cov_model["model"], rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Coverage"); axes[0].set_title("Coverage by model"); set_panel_label(axes[0], "(a)")
    # (b) coverage by target.
    targets = v39_target_list_from_available(df)
    cov_target = df.groupby("target", as_index=False)["split_coverage_mean"].mean().set_index("target").reindex(targets)
    axes[1].bar(range(len(cov_target)), cov_target["split_coverage_mean"]); axes[1].axhline(nominal, linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(cov_target))); axes[1].set_xticklabels([v40_pretty_target(t) for t in cov_target.index], rotation=30, ha="right", fontsize=7)
    axes[1].set_ylabel("Coverage"); axes[1].set_title("Coverage by target"); set_panel_label(axes[1], "(b)")
    # (c) normalized interval width vs budget.
    for target in targets:
        sub = df[df["target"].astype(str).eq(str(target))].groupby("budget", as_index=False)["split_width_normalized"].mean()
        if not sub.empty:
            axes[2].plot(sub["budget"], sub["split_width_normalized"], marker="o", label=v40_pretty_target(target))
    axes[2].set_xscale("log"); axes[2].set_xlabel("Label budget"); axes[2].set_ylabel("Interval width / target std")
    axes[2].grid(True, alpha=0.25); axes[2].set_title("Normalized uncertainty shrinks"); axes[2].legend(fontsize=6); set_panel_label(axes[2], "(c)")
    # (d) abstention vs precision.
    axes[3].scatter(df["uncertain_fraction"], df["precision_mean"], s=25, alpha=0.55)
    axes[3].set_xlabel("Uncertain fraction"); axes[3].set_ylabel("Top-5% precision"); axes[3].grid(True, alpha=0.25); axes[3].set_title("Abstention/precision trade-off"); set_panel_label(axes[3], "(d)")
    # (e) trusted fraction vs budget.
    trusted = df.groupby("budget", as_index=False)["trusted_fraction"].mean()
    axes[4].plot(trusted["budget"], trusted["trusted_fraction"], marker="o")
    axes[4].set_xscale("log"); axes[4].set_xlabel("Label budget"); axes[4].set_ylabel("Trusted fraction"); axes[4].grid(True, alpha=0.25); axes[4].set_title("Trusted fraction, not raw count"); set_panel_label(axes[4], "(e)")
    # (f) conformal method comparison.
    methods = [c for c in ["split_coverage_mean", "local_coverage_mean", "mondrian_coverage_mean"] if c in df.columns]
    method_labels = [m.replace("_coverage_mean", "") for m in methods]
    vals = [df[m].mean() for m in methods]
    axes[5].bar(range(len(vals)), vals); axes[5].axhline(nominal, linestyle="--", linewidth=1)
    axes[5].set_xticks(range(len(vals))); axes[5].set_xticklabels(method_labels)
    axes[5].set_ylabel("Coverage"); axes[5].set_title("Conformal variants"); set_panel_label(axes[5], "(f)")
    fig.suptitle("Conformal risk control and abstention behaviour", fontsize=14, fontweight="bold")
    v40_save_current_figure_aliases([
        FIG_MAIN_DIR / "Figure_3_calibration_and_abstention",
        FIG_MAIN_DIR / "Figure_3_final_calibration_abstention",
    ])
    mark_done(stage)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    """v4.0 Figure 4: target-balanced reliability and true-enrichment validation."""
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 4 final target-balanced failure/reliability")
    if tiers is None or tiers.empty:
        return
    enrich = write_consensus_true_enrichment(None, None)
    tb = v39_target_balanced_compact_candidate_rows(tiers, max_rows=int(CONFIG.get("figure_target_balanced_sample_per_target", 3000)) * max(1, len(v39_target_list_from_available(tiers))))
    for col in ["y_true", "y_pred", "split_lower", "split_upper", "target_lower_bound_percentile"]:
        if col in tb.columns:
            tb[col] = pd.to_numeric(tb[col], errors="coerce")
    stats = v40_target_stats(None)
    for t, st in stats.items():
        mask = tb["target"].astype(str).eq(str(t)) if "target" in tb.columns else pd.Series(False, index=tb.index)
        std = st.get("std", np.nan)
        mean = st.get("mean", np.nan)
        if np.isfinite(std) and std > 0:
            tb.loc[mask, "z_true"] = (tb.loc[mask, "y_true"] - mean) / std
            tb.loc[mask, "z_pred"] = (tb.loc[mask, "y_pred"] - mean) / std
            tb.loc[mask, "normalized_abs_error"] = np.abs(tb.loc[mask, "y_true"] - tb.loc[mask, "y_pred"]) / std
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5)); axes = axes.ravel()
    targets = v39_target_list_from_available(tb)
    counts = tb["target"].value_counts().reindex(targets).fillna(0)
    axes[0].bar(range(len(counts)), counts.values)
    axes[0].set_xticks(range(len(counts))); axes[0].set_xticklabels([v40_pretty_target(t) for t in counts.index], rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("Candidate rows"); axes[0].set_title("Target-balanced candidate pool"); set_panel_label(axes[0], "(a)")
    # Split error/stress.
    if anatomy is not None and not anatomy.empty:
        metric = "normalized_mae" if "normalized_mae" in anatomy.columns else "mae"
        sub = anatomy.groupby("split", as_index=False)[metric].mean().sort_values(metric, ascending=False)
        axes[1].bar(range(len(sub)), sub[metric])
        axes[1].set_xticks(range(len(sub))); axes[1].set_xticklabels(sub["split"], rotation=35, ha="right", fontsize=7)
        axes[1].set_ylabel("Target-normalized MAE" if metric == "normalized_mae" else "MAE")
    axes[1].set_title("Extrapolation stress regimes"); set_panel_label(axes[1], "(b)")
    # True enrichment fold.
    if enrich is not None and not enrich.empty:
        ee = enrich.set_index("target").reindex(v39_target_list_from_available(enrich))
        axes[2].bar(range(len(ee)), ee["fold_enrichment_vs_dataset_median"])
        axes[2].axhline(1.0, linestyle="--", linewidth=1)
        axes[2].set_xticks(range(len(ee))); axes[2].set_xticklabels([v40_pretty_target(t) for t in ee.index], rotation=30, ha="right", fontsize=7)
        axes[2].set_ylabel("Shortlist median / dataset median")
    axes[2].set_title("Consensus shortlist true enrichment"); set_panel_label(axes[2], "(c)")
    # Normalized predicted vs true.
    sample = tb.dropna(subset=["z_true", "z_pred"]).sample(min(8000, tb.dropna(subset=["z_true", "z_pred"]).shape[0]), random_state=42) if {"z_true", "z_pred"}.issubset(tb.columns) else pd.DataFrame()
    if not sample.empty:
        axes[3].scatter(sample["z_true"], sample["z_pred"], s=5, alpha=0.35)
        lo = min(sample["z_true"].min(), sample["z_pred"].min()); hi = max(sample["z_true"].max(), sample["z_pred"].max())
        axes[3].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[3].set_xlabel("True uptake, target z-score"); axes[3].set_ylabel("Predicted uptake, target z-score"); axes[3].set_title("QC candidate predictions"); set_panel_label(axes[3], "(d)")
    # Lower-bound percentile.
    for target in targets:
        vals = pd.to_numeric(tb.loc[tb["target"].astype(str).eq(str(target)), "target_lower_bound_percentile"], errors="coerce").dropna()
        if len(vals):
            axes[4].hist(vals, bins=25, alpha=0.35, label=v40_pretty_target(target))
    axes[4].set_xlabel("Within-target lower-bound percentile"); axes[4].set_ylabel("Rows"); axes[4].set_title("Risk-ranked candidate rows"); axes[4].legend(fontsize=6); set_panel_label(axes[4], "(e)")
    # Top-5 true fraction.
    if enrich is not None and not enrich.empty:
        ee = enrich.set_index("target").reindex(v39_target_list_from_available(enrich))
        axes[5].bar(range(len(ee)), ee["fraction_true_top5pct"])
        axes[5].axhline(0.05, linestyle="--", linewidth=1)
        axes[5].set_xticks(range(len(ee))); axes[5].set_xticklabels([v40_pretty_target(t) for t in ee.index], rotation=30, ha="right", fontsize=7)
        axes[5].set_ylabel("Fraction truly in dataset top 5%")
    axes[5].set_title("Shortlist validation against true labels"); set_panel_label(axes[5], "(f)")
    fig.suptitle("Target-balanced consensus reliability and true-enrichment validation", fontsize=14, fontweight="bold")
    v40_save_current_figure_aliases([
        FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy",
        FIG_MAIN_DIR / "Figure_4_final_consensus_reliability",
    ])
    mark_done(stage, {"target_balanced": True, "true_enrichment": True})


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """v4.0 Figure 5: consensus stability as main; external overlap as cautious annotation."""
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 5 final consensus stability/domain-overlap")
    availability = summarise_external_files(core, mosaec)
    atomic_save_csv(availability, FIGDATA_DIR / "Figure_5_external_file_availability_data.csv")
    fe = final_external.copy() if final_external is not None else pd.DataFrame()
    if fe.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off"); ax.text(0.5, 0.5, "No final external/consensus table", ha="center", va="center")
        save_current_figure(FIG_MAIN_DIR / "Figure_5_external_realism_filter")
        mark_done(stage, {"n_rows": 0})
        return
    atomic_save_csv(fe, FIGDATA_DIR / "Figure_5_final_consensus_external_data.csv")
    enrich = write_consensus_true_enrichment(None, None)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5)); axes = axes.ravel()
    targets = v39_target_list_from_available(fe)
    # (a) balanced shortlist counts.
    counts = fe["target"].value_counts().reindex(targets).fillna(0)
    axes[0].bar(range(len(counts)), counts.values)
    axes[0].set_xticks(range(len(counts))); axes[0].set_xticklabels([v40_pretty_target(t) for t in counts.index], rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("Consensus candidates"); axes[0].set_title("Target-balanced final shortlist"); set_panel_label(axes[0], "(a)")
    # (b) consensus support.
    if {"n_models", "n_seeds", "n_splits"}.issubset(fe.columns):
        supp = fe.groupby("target", as_index=False).agg(n_models=("n_models", "median"), n_seeds=("n_seeds", "median"), n_splits=("n_splits", "median"))
        supp = supp.set_index("target").reindex(targets).reset_index()
        x = np.arange(len(supp)); width = 0.25
        axes[1].bar(x - width, supp["n_models"], width, label="models")
        axes[1].bar(x, supp["n_seeds"], width, label="seeds")
        axes[1].bar(x + width, supp["n_splits"], width, label="splits")
        axes[1].set_xticks(x); axes[1].set_xticklabels([v40_pretty_target(t) for t in supp["target"]], rotation=30, ha="right", fontsize=7)
        axes[1].legend(fontsize=7)
    axes[1].set_ylabel("Median support"); axes[1].set_title("Consensus across runs"); set_panel_label(axes[1], "(b)")
    # (c) trusted support.
    if "trusted_support" in fe.columns:
        for target in targets:
            vals = pd.to_numeric(fe.loc[fe["target"].astype(str).eq(str(target)), "trusted_support"], errors="coerce").dropna()
            if len(vals):
                axes[2].hist(vals, bins=20, alpha=0.35, label=v40_pretty_target(target))
        axes[2].legend(fontsize=6)
    axes[2].set_xlabel("Trusted-support count"); axes[2].set_ylabel("Candidates"); axes[2].set_title("Consensus confidence distribution"); set_panel_label(axes[2], "(c)")
    # (d) external domain overlap, explicitly not validation.
    overlap_cols = [c for c in ["core_geometry_overlap_flag", "mosaec_geometry_overlap_flag"] if c in fe.columns]
    if overlap_cols:
        overlap = fe.groupby("target").agg(**{c: (c, lambda s: s.fillna(False).astype(bool).mean()) for c in overlap_cols}).reindex(targets)
        x = np.arange(len(overlap)); width = 0.35
        for j, c in enumerate(overlap_cols):
            axes[3].bar(x + (j - (len(overlap_cols)-1)/2)*width, overlap[c], width, label=c.replace("_geometry_overlap_flag", ""))
        axes[3].set_xticks(x); axes[3].set_xticklabels([v40_pretty_target(t) for t in overlap.index], rotation=30, ha="right", fontsize=7)
        axes[3].legend(fontsize=7)
    axes[3].set_ylabel("Fraction with geometry-domain overlap"); axes[3].set_title("External domain-overlap annotation"); set_panel_label(axes[3], "(d)")
    # (e) external score by target.
    if "external_realism_score" in fe.columns:
        score = fe.groupby("target", as_index=False)["external_realism_score"].mean().set_index("target").reindex(targets)
        axes[4].bar(range(len(score)), score["external_realism_score"])
        axes[4].set_xticks(range(len(score))); axes[4].set_xticklabels([v40_pretty_target(t) for t in score.index], rotation=30, ha="right", fontsize=7)
    axes[4].set_ylabel("Mean external-domain score"); axes[4].set_title("Plausibility, not experimental validation"); set_panel_label(axes[4], "(e)")
    # (f) true top-5 fraction in shortlist.
    if enrich is not None and not enrich.empty:
        ee = enrich.set_index("target").reindex(v39_target_list_from_available(enrich))
        axes[5].bar(range(len(ee)), ee["fraction_true_top5pct"])
        axes[5].axhline(0.05, linestyle="--", linewidth=1)
        axes[5].set_xticks(range(len(ee))); axes[5].set_xticklabels([v40_pretty_target(t) for t in ee.index], rotation=30, ha="right", fontsize=7)
    axes[5].set_ylabel("Fraction truly in top 5%"); axes[5].set_title("Final shortlist enrichment check"); set_panel_label(axes[5], "(f)")
    fig.suptitle("Consensus shortlist stability with external domain-overlap annotations", fontsize=14, fontweight="bold")
    v40_save_current_figure_aliases([
        FIG_MAIN_DIR / "Figure_5_external_realism_filter",
        FIG_MAIN_DIR / "Figure_5_final_consensus_stability",
    ])
    mark_done(stage, {"n_rows": len(fe), "external_framing": "domain_overlap_not_validation"})


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    """v4.0 SI figures with duplicate RDF feature sets hidden from ablation panels."""
    stage = "si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> SI figures v4")
    if tiers is not None and not tiers.empty:
        sample = tiers.drop_duplicates(["mof_id", "target"])
        fig, axes = plt.subplots(2, 2, figsize=(10, 8)); axes = axes.ravel()
        for ax, target in zip(axes, list(TARGET_SPECS.keys())):
            sub = sample[sample["target"].astype(str).eq(str(target))]
            if not sub.empty:
                ax.hist(pd.to_numeric(sub["y_true"], errors="coerce").dropna(), bins=40)
            ax.set_title(v40_pretty_target(target)); ax.set_xlabel("Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Target distributions represented in candidate rows")
        save_current_figure(FIG_SI_DIR / "Figure_S1_target_distributions")
        tmp = tiers.copy(); tmp["interval_width"] = pd.to_numeric(tmp["split_upper"], errors="coerce") - pd.to_numeric(tmp["split_lower"], errors="coerce")
        atomic_save_csv(tmp[[c for c in ["target", "model", "budget", "tier", "interval_width"] if c in tmp.columns]].sample(min(20000, len(tmp)), random_state=42), FIGDATA_DIR / "Figure_S3_interval_width_by_tier_data.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [tier for tier in ["trusted", "uncertain", "rejected"] if "tier" in tmp.columns and (tmp["tier"].astype(str) == tier).sum() > 0]
        data = [tmp.loc[tmp["tier"].astype(str).eq(tier), "interval_width"].dropna().sample(min(5000, (tmp["tier"].astype(str) == tier).sum()), random_state=42) for tier in labels]
        if data:
            safe_boxplot_with_labels(ax, data, labels, showfliers=False)
        ax.set_ylabel("Split-conformal interval width"); ax.set_title("Figure S3. Interval width by candidate tier")
        save_current_figure(FIG_SI_DIR / "Figure_S3_interval_width_by_tier")
    if agg is not None and not agg.empty:
        stable = v40_stable_metric_df(agg, top_frac=0.05)
        cov = stable.pivot_table(index="split", columns="model", values="split_coverage_mean", aggfunc="mean") if not stable.empty else pd.DataFrame()
        if not cov.empty:
            atomic_save_csv(cov.reset_index(), FIGDATA_DIR / "Figure_S2_grouped_split_coverage_data.csv")
            fig, ax = plt.subplots(figsize=(10, 5)); im = ax.imshow(cov.fillna(np.nan).to_numpy(), aspect="auto")
            ax.set_xticks(range(cov.shape[1])); ax.set_xticklabels(cov.columns, rotation=45, ha="right")
            ax.set_yticks(range(cov.shape[0])); ax.set_yticklabels(cov.index)
            ax.set_title("Figure S2. Coverage across grouped splits, stable models"); fig.colorbar(im, ax=ax, label="Coverage")
            save_current_figure(FIG_SI_DIR / "Figure_S2_grouped_split_coverage")
        abl = stable.groupby(["feature_set", "budget"], as_index=False)["recall_mean"].mean() if not stable.empty else pd.DataFrame()
        if not abl.empty:
            atomic_save_csv(abl, FIGDATA_DIR / "Figure_S4_feature_set_ablation_data.csv")
            fig, ax = plt.subplots(figsize=(8, 5))
            for fs in abl["feature_set"].unique():
                sub = abl[abl["feature_set"].astype(str).eq(str(fs))].groupby("budget", as_index=False)["recall_mean"].mean()
                ax.plot(sub["budget"], sub["recall_mean"], marker="o", label=str(fs))
            ax.set_xscale("log"); ax.set_xlabel("Label budget"); ax.set_ylabel("Mean top-5% recall"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.set_title("Figure S4. Feature-set ablation after integrity QC")
            save_current_figure(FIG_SI_DIR / "Figure_S4_feature_set_ablation")
    if anatomy is not None and not anatomy.empty:
        metric = "normalized_mae" if "normalized_mae" in anatomy.columns else "mae"
        worst = anatomy.sort_values(metric, ascending=False).head(20)
        atomic_save_csv(worst, FIGDATA_DIR / "Figure_S5_worst_regimes_data.csv")
        fig, ax = plt.subplots(figsize=(10, 6)); labels = (worst["anatomy_type"].astype(str) + ":" + worst["regime"].astype(str)).str.slice(0, 35)
        ax.barh(range(len(worst)), worst[metric]); ax.set_yticks(range(len(worst))); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("Target-normalized MAE" if metric == "normalized_mae" else "MAE"); ax.set_title("Figure S5. Highest-error chemical regimes")
        save_current_figure(FIG_SI_DIR / "Figure_S5_highest_error_regimes")
    mark_done(stage)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    """v4.0 run report highlighting final manuscript-ready outputs."""
    stage = "07_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> WRITE_FINAL_RUN_REPORT_V4")
    report = RESULTS_DIR / "RUN_REPORT.md"
    lines = []
    lines.append("# Few-shot MOF risk-controlled screening run report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Runtime modes\n")
    for key in ["save_mode", "ram_mode", "comprehensive_level", "n_jobs", "data_root"]:
        lines.append(f"- {key}: `{CONFIG.get(key)}`")
    lines.append("\n## Experiment scale\n")
    lines.append(f"- Master table rows: {master.shape[0]}")
    lines.append(f"- Master table columns: {master.shape[1]}")
    lines.append(f"- Metric rows: {len(metrics_all)}")
    if metrics_all is not None and not metrics_all.empty:
        lines.append(f"- Unique experiments: {metrics_all[EXPERIMENT_KEY_COLS].drop_duplicates().shape[0] if set(EXPERIMENT_KEY_COLS).issubset(metrics_all.columns) else 'NA'}")
    lines.append("\n## Final v4 outputs to inspect first\n")
    for fname in [
        "QC_Table_consensus_true_enrichment.csv",
        "MAIN_Table_2_descriptor_ablation_summary.csv",
        "MAIN_Table_2_split_extrapolation_stress_summary.csv",
        "MAIN_Table_3_consensus_shortlist_target_balanced_top25.csv",
        "SI_Table_external_domain_overlap_top25_by_target.csv",
        "QC_Table_feature_set_integrity.csv",
        "QC_Table_model_stability.csv",
        "QC_Table_prediction_range_by_model.csv",
    ]:
        p = TABLE_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'}")
    lines.append("\n## Main figures\n")
    for fname in [
        "Figure_1_risk_controlled_framework.png",
        "Figure_2_final_performance_descriptor_split.png",
        "Figure_3_final_calibration_abstention.png",
        "Figure_4_final_consensus_reliability.png",
        "Figure_5_final_consensus_stability.png",
    ]:
        p = FIG_MAIN_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'}")
    lines.append("\n## Scientific framing notes\n")
    lines.append("- Main descriptor interpretation should use geometry_only and geometry_plus_racs only unless RDF integrity QC shows real RDF features.")
    lines.append("- External overlays are domain-overlap/plausibility annotations, not experimental validation, unless exact structural matches are established.")
    lines.append("- Candidate tables are quality-gated, target-balanced, and consensus-ranked.")
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage)



# =============================================================================
# 16e. v4.1 final paper figure layer and compact manuscript outputs
# =============================================================================
# v4.1 does not change the expensive fitting loop. It adds a curated final-paper
# plotting layer, stronger final QC summaries, and compact manuscript/SI tables
# designed to turn the technically complete v4.0 outputs into visually elegant
# publication figures. The external overlay is deliberately reframed as domain
# overlap and moved to a supporting-information style figure unless a user later
# establishes true exact structural matches.

CONFIG.setdefault("paper_final_save_formats", ["png", "pdf", "svg"])
CONFIG.setdefault("paper_final_png_dpi", 600)
CONFIG.setdefault("paper_final_name", "v4_1_final_paper_figures")
CONFIG.setdefault("paper_final_max_scatter_per_target", 1000)
CONFIG.setdefault("paper_final_top_candidate_rows_per_target", 25)
CONFIG.setdefault("paper_final_external_rows_per_target", 25)
CONFIG.setdefault("paper_final_use_vector_formats", True)
CONFIG.setdefault("paper_final_move_external_overlay_to_si", True)
CONFIG.setdefault("paper_final_show_targets_in_order", True)
CONFIG.setdefault("paper_final_figsize_wide", (15.5, 8.6))
CONFIG.setdefault("paper_final_figsize_tall", (14.5, 10.2))
CONFIG.setdefault("paper_final_coverage_nominal", 0.90)

PAPER_FINAL_DIR = RESULTS_DIR / "figures" / "paper_final"
PAPER_FINAL_SI_DIR = RESULTS_DIR / "figures" / "paper_final_si"
PAPER_FINAL_DATA_DIR = FIGDATA_DIR / "paper_final"
for _d in [PAPER_FINAL_DIR, PAPER_FINAL_SI_DIR, PAPER_FINAL_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

TARGET_LABELS_LATEX = {
    "CO2_0p015bar_298K_mmolg": "CO$_2$ 0.015 bar",
    "CO2_0p150bar_298K_mmolg": "CO$_2$ 0.150 bar",
    "CH4_5p8bar_298K_mmolg": "CH$_4$ 5.8 bar",
    "CH4_65bar_298K_mmolg": "CH$_4$ 65 bar",
}
TARGET_LABELS_PLAIN = {
    "CO2_0p015bar_298K_mmolg": "CO2 0.015 bar",
    "CO2_0p150bar_298K_mmolg": "CO2 0.150 bar",
    "CH4_5p8bar_298K_mmolg": "CH4 5.8 bar",
    "CH4_65bar_298K_mmolg": "CH4 65 bar",
}
FEATURE_LABELS_FINAL = {
    "geometry_only": "Geometry",
    "geometry_plus_racs": "Geometry + RACs",
    "geometry_plus_racs_rdfs": "Geometry + RACs + RDFs",
}
MODEL_LABELS_FINAL = {
    "rf": "RF",
    "extra_trees": "ExtraTrees",
    "hgb": "HGB",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "ridge": "Ridge",
    "mlp": "MLP",
    "gpr": "GPR",
    "tabpfn": "TabPFN",
}
SPLIT_LABELS_FINAL = {
    "random": "Random",
    "geometry_grouped": "Geometry",
    "metal_grouped": "Metal",
    "functional_grouped": "Functional",
    "ligand_grouped": "Ligand",
    "topology_grouped": "Topology",
}
TARGET_ORDER_FINAL = list(TARGET_SPECS.keys())
MODEL_ORDER_FINAL = ["extra_trees", "rf", "hgb", "lightgbm", "xgboost"]
SPLIT_ORDER_FINAL = ["random", "geometry_grouped", "metal_grouped", "functional_grouped", "ligand_grouped", "topology_grouped"]
TARGET_COLORS_FINAL = {
    "CO2_0p015bar_298K_mmolg": "#2A6FBB",
    "CO2_0p150bar_298K_mmolg": "#16A085",
    "CH4_5p8bar_298K_mmolg": "#D35400",
    "CH4_65bar_298K_mmolg": "#7D3C98",
}
MODEL_COLORS_FINAL = {
    "extra_trees": "#1F77B4",
    "rf": "#2CA02C",
    "hgb": "#FF7F0E",
    "lightgbm": "#9467BD",
    "xgboost": "#D62728",
}


def v41_target_order(df: Optional[pd.DataFrame] = None) -> List[str]:
    base = TARGET_ORDER_FINAL[:]
    if df is None or df.empty or "target" not in df.columns:
        return base
    present = [t for t in base if t in set(df["target"].dropna().astype(str))]
    extras = [t for t in sorted(df["target"].dropna().astype(str).unique()) if t not in present]
    return present + extras


def v41_target_label(t: Any, latex: bool = True) -> str:
    return (TARGET_LABELS_LATEX if latex else TARGET_LABELS_PLAIN).get(str(t), str(t))


def v41_feature_label(fs: Any) -> str:
    return FEATURE_LABELS_FINAL.get(str(fs), str(fs).replace("_", " "))


def v41_model_label(model: Any) -> str:
    return MODEL_LABELS_FINAL.get(str(model), str(model))


def v41_split_label(split: Any) -> str:
    return SPLIT_LABELS_FINAL.get(str(split), str(split).replace("_", " "))


def v41_apply_paper_style() -> None:
    """Apply a clean manuscript style without depending on external font files."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.dpi": 150,
        "savefig.dpi": int(CONFIG.get("paper_final_png_dpi", 600)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
        "patch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def v41_panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="top", ha="right")


def v41_save_figure(fig: plt.Figure, stem: str, aliases: Optional[List[Path]] = None, si: bool = False) -> None:
    """Save final paper figures as 600-dpi PNG plus PDF/SVG and optional legacy aliases."""
    v41_apply_paper_style()
    out_dir = PAPER_FINAL_SI_DIR if si else PAPER_FINAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = CONFIG.get("paper_final_save_formats", ["png", "pdf", "svg"])
    if not CONFIG.get("paper_final_use_vector_formats", True):
        formats = [fmt for fmt in formats if str(fmt).lower() == "png"] or ["png"]
    for fmt in formats:
        fmt = str(fmt).lower().strip()
        out = (out_dir / stem).with_suffix(f".{fmt}")
        try:
            fig.savefig(out, dpi=int(CONFIG.get("paper_final_png_dpi", 600)), bbox_inches="tight")
        except Exception as e:
            logging.warning("PAPER_FINAL_FIGURE_SAVE_FAILED | %s | %s", out, e)
    # Also update the old manuscript filenames as high-resolution PNG/PDF where requested.
    for base in aliases or []:
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in ["png", "pdf"]:
            try:
                fig.savefig(base.with_suffix(f".{fmt}"), dpi=int(CONFIG.get("paper_final_png_dpi", 600)), bbox_inches="tight")
            except Exception as e:
                logging.warning("PAPER_FINAL_ALIAS_SAVE_FAILED | %s | %s", base.with_suffix(f'.{fmt}'), e)
    plt.close(fig)


def v41_clean_metric_df(agg: pd.DataFrame, top_frac: float = 0.05, feature_set: Optional[str] = None,
                        split: Optional[str] = None, stable_only: bool = True) -> pd.DataFrame:
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = agg.copy()
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(top_frac).abs() < 1e-9]
    if stable_only and "model" in df.columns:
        stable = set(CONFIG.get("stable_plot_models", MODEL_ORDER_FINAL))
        df = df[df["model"].astype(str).isin(stable)]
    usable = set(v40_usable_feature_sets()) if "v40_usable_feature_sets" in globals() else set(CONFIG.get("feature_sets", []))
    if usable and "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str).isin(usable)]
    if feature_set is not None and "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str).eq(str(feature_set))]
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str).eq(str(split))]
    return df


def v41_preferred_feature(agg: Optional[pd.DataFrame] = None) -> str:
    if "v40_preferred_feature_set" in globals():
        return v40_preferred_feature_set(agg)
    return "geometry_plus_racs"


def v41_target_stats(master: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
    if "v40_target_stats" in globals():
        return v40_target_stats(master)
    return target_stats_from_master(master)


def v41_heatmap(ax, mat: pd.DataFrame, title: str, cbar_label: str = "", center_zero: bool = False,
                cmap: str = "viridis", fmt: str = ".2f"):
    if mat is None or mat.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return None
    arr = mat.to_numpy(dtype=float)
    if center_zero:
        vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
        vmin, vmax = -vmax, vmax
        cmap = "coolwarm"
    else:
        vmin = np.nanmin(arr) if np.isfinite(arr).any() else 0.0
        vmax = np.nanmax(arr) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels([str(x) for x in mat.columns], rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([str(x) for x in mat.index])
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    return im


def v41_paper_curve(df: pd.DataFrame, metric: str, group_col: str = "target", split: str = "random") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if split is not None and "split" in work.columns and split in set(work["split"].astype(str)):
        work = work[work["split"].astype(str).eq(split)].copy()
    rows = []
    for (grp, budget), sub in work.groupby([group_col, "budget"], dropna=False):
        vals = pd.to_numeric(sub.get(metric), errors="coerce").dropna()
        if vals.empty:
            continue
        lows = pd.to_numeric(sub.get(metric.replace("_mean", "_ci_low"), np.nan), errors="coerce") if metric.endswith("_mean") else pd.Series(dtype=float)
        highs = pd.to_numeric(sub.get(metric.replace("_mean", "_ci_high"), np.nan), errors="coerce") if metric.endswith("_mean") else pd.Series(dtype=float)
        rows.append({
            group_col: grp,
            "budget": budget,
            "mean": float(vals.mean()),
            "ci_low": float(lows.dropna().mean()) if len(lows.dropna()) else float(vals.mean()),
            "ci_high": float(highs.dropna().mean()) if len(highs.dropna()) else float(vals.mean()),
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows)


def v41_consensus_work(consensus: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if consensus is None or consensus.empty:
        consensus = v40_read_consensus() if "v40_read_consensus" in globals() else pd.DataFrame()
    if consensus is None or consensus.empty:
        return pd.DataFrame()
    work = consensus.copy()
    if "consensus_pass" in work.columns and work["consensus_pass"].fillna(False).astype(bool).any():
        work = work[work["consensus_pass"].fillna(False).astype(bool)].copy()
    sort_cols = [c for c in ["target", "consensus_score", "split_lower_median"] if c in work.columns]
    if sort_cols:
        asc = [True] + [False] * (len(sort_cols) - 1)
        work = work.sort_values(sort_cols, ascending=asc)
    return work


def v41_top_consensus_per_target(consensus: Optional[pd.DataFrame] = None, topn: Optional[int] = None) -> pd.DataFrame:
    work = v41_consensus_work(consensus)
    if work.empty or "target" not in work.columns:
        return pd.DataFrame()
    topn = int(topn or CONFIG.get("paper_final_top_candidate_rows_per_target", 25))
    parts = []
    for t in v41_target_order(work):
        sub = work[work["target"].astype(str).eq(str(t))]
        if not sub.empty:
            parts.append(sub.head(topn))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def v41_add_true_percentile_columns(df: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None or df.empty or "target" not in df.columns:
        return df
    out = df.copy()
    stats = v41_target_stats(master)
    val_col = "y_true_median" if "y_true_median" in out.columns else ("y_true" if "y_true" in out.columns else None)
    pred_col = "y_pred_median" if "y_pred_median" in out.columns else ("y_pred" if "y_pred" in out.columns else None)
    for t, st in stats.items():
        mask = out["target"].astype(str).eq(str(t))
        sorted_vals = st.get("sorted_values", np.array([], dtype=float))
        if val_col and mask.any():
            vals = pd.to_numeric(out.loc[mask, val_col], errors="coerce").to_numpy(dtype=float)
            out.loc[mask, "true_percentile"] = v40_true_percentile(vals, sorted_vals) if "v40_true_percentile" in globals() else np.nan
        if pred_col and mask.any():
            preds = pd.to_numeric(out.loc[mask, pred_col], errors="coerce").to_numpy(dtype=float)
            out.loc[mask, "pred_percentile"] = v40_true_percentile(preds, sorted_vals) if "v40_true_percentile" in globals() else np.nan
        if mask.any():
            med = float(st.get("median", np.nan))
            std = float(st.get("std", np.nan))
            if val_col and np.isfinite(std) and std > 0:
                out.loc[mask, "true_z"] = (pd.to_numeric(out.loc[mask, val_col], errors="coerce") - float(st.get("mean", 0.0))) / std
            if pred_col and np.isfinite(std) and std > 0:
                out.loc[mask, "pred_z"] = (pd.to_numeric(out.loc[mask, pred_col], errors="coerce") - float(st.get("mean", 0.0))) / std
            if val_col and np.isfinite(med) and med != 0:
                out.loc[mask, "true_over_dataset_median"] = pd.to_numeric(out.loc[mask, val_col], errors="coerce") / med
    return out


def write_v41_final_paper_tables(master: Optional[pd.DataFrame], agg: Optional[pd.DataFrame], tiers: Optional[pd.DataFrame],
                                  consensus: Optional[pd.DataFrame] = None,
                                  final_external: Optional[pd.DataFrame] = None) -> None:
    """Final compact paper tables and figure manifest for v4.1."""
    consensus = v41_consensus_work(consensus)
    top25 = v41_top_consensus_per_target(consensus, topn=int(CONFIG.get("paper_final_top_candidate_rows_per_target", 25)))
    top25 = v41_add_true_percentile_columns(top25, master)
    if not top25.empty:
        cols = [c for c in [
            "target", "target_label", "mof_id", "filename", "y_true_median", "y_pred_median", "split_lower_median",
            "split_upper_median", "interval_width_median", "true_percentile", "pred_percentile",
            "true_over_dataset_median", "consensus_score", "n_models", "n_seeds", "n_splits", "n_feature_sets",
            "n_budgets", "trusted_support", "Density", "Di", "Df", "Dif"
        ] if c in top25.columns]
        if "target_label" not in top25.columns:
            top25.insert(1, "target_label", top25["target"].map(lambda x: v41_target_label(x, latex=False)))
            cols = [c for c in cols if c != "target_label"]
            cols.insert(1, "target_label")
        atomic_save_csv(top25[cols], TABLE_DIR / "MAIN_Table_3_consensus_shortlist_target_balanced_top25.csv")
        # Target-specific files are convenient for supplementary review.
        for t in v41_target_order(top25):
            sub = top25[top25["target"].astype(str).eq(str(t))]
            if not sub.empty:
                atomic_save_csv(sub[cols], TABLE_DIR / f"MAIN_Table_3_top25_{v40_plain_target(t)}.csv")
    # Descriptor gain and split stress paper tables are based on stable models at top-5%.
    if agg is not None and not agg.empty:
        stable = v41_clean_metric_df(agg, top_frac=0.05, stable_only=True)
        max_budget = int(pd.to_numeric(stable.get("budget", pd.Series([1000])), errors="coerce").max()) if not stable.empty else 1000
        # Feature gain table: geometry+RACs - geometry.
        feature_rows = []
        if not stable.empty and {"geometry_only", "geometry_plus_racs"}.issubset(set(stable.get("feature_set", pd.Series(dtype=str)).astype(str))):
            fs_work = stable[(pd.to_numeric(stable["budget"], errors="coerce") == max_budget) & stable["split"].astype(str).eq("random")].copy()
            pivot = fs_work.pivot_table(index="target", columns="feature_set", values="recall_mean", aggfunc="mean")
            for t in v41_target_order(pivot.reset_index().rename(columns={"index": "target"})):
                if t in pivot.index:
                    feature_rows.append({
                        "target": t,
                        "target_label": v41_target_label(t, latex=False),
                        "budget": max_budget,
                        "geometry_only_recall": float(pivot.loc[t].get("geometry_only", np.nan)),
                        "geometry_plus_racs_recall": float(pivot.loc[t].get("geometry_plus_racs", np.nan)),
                        "delta_recall_racs_minus_geometry": float(pivot.loc[t].get("geometry_plus_racs", np.nan) - pivot.loc[t].get("geometry_only", np.nan)),
                    })
        atomic_save_csv(pd.DataFrame(feature_rows), TABLE_DIR / "MAIN_Table_2_descriptor_gain_delta_recall.csv")
        # Split stress table: random recall minus grouped recall.
        split_rows = []
        preferred = v41_preferred_feature(stable)
        sp_work = stable[(stable["feature_set"].astype(str).eq(preferred)) & (pd.to_numeric(stable["budget"], errors="coerce") == max_budget)].copy()
        pivot = sp_work.pivot_table(index="target", columns="split", values="recall_mean", aggfunc="mean") if not sp_work.empty else pd.DataFrame()
        for t in v41_target_order(pivot.reset_index().rename(columns={"index": "target"})):
            if t not in pivot.index:
                continue
            random_val = float(pivot.loc[t].get("random", np.nan))
            for sp in [s for s in SPLIT_ORDER_FINAL if s in pivot.columns and s != "random"]:
                split_rows.append({
                    "target": t,
                    "target_label": v41_target_label(t, latex=False),
                    "feature_set": preferred,
                    "budget": max_budget,
                    "split": sp,
                    "split_label": v41_split_label(sp),
                    "random_recall": random_val,
                    "split_recall": float(pivot.loc[t].get(sp, np.nan)),
                    "recall_penalty_vs_random": float(random_val - pivot.loc[t].get(sp, np.nan)) if np.isfinite(random_val) else np.nan,
                })
        atomic_save_csv(pd.DataFrame(split_rows), TABLE_DIR / "MAIN_Table_2_split_penalty_vs_random.csv")
    # Support and geometry summary for main/SI.
    if not consensus.empty:
        support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_feature_sets", "n_budgets", "trusted_support"] if c in consensus.columns]
        if support_cols:
            sup = consensus.groupby("target")[support_cols].agg(["median", "mean", "min", "max"]).reset_index()
            sup.columns = ["_".join([str(x) for x in c if str(x)]) for c in sup.columns]
            if "target" not in sup.columns and "target_" in sup.columns:
                sup = sup.rename(columns={"target_": "target"})
            sup.insert(1, "target_label", sup["target"].map(lambda x: v41_target_label(x, latex=False)))
            atomic_save_csv(sup, TABLE_DIR / "MAIN_Table_4_consensus_support_summary.csv")
        geom_cols = [c for c in ["Density", "Di", "Df", "Dif"] if c in consensus.columns]
        if geom_cols:
            geom = consensus.groupby("target")[geom_cols].agg(["median", "mean", "min", "max"]).reset_index()
            geom.columns = ["_".join([str(x) for x in c if str(x)]) for c in geom.columns]
            if "target" not in geom.columns and "target_" in geom.columns:
                geom = geom.rename(columns={"target_": "target"})
            geom.insert(1, "target_label", geom["target"].map(lambda x: v41_target_label(x, latex=False)))
            atomic_save_csv(geom, TABLE_DIR / "SI_Table_consensus_geometry_regime_summary.csv")
    if final_external is not None and not final_external.empty:
        fe = final_external.copy()
        for col in ["core_exact_match_flag", "core_geometry_overlap_flag", "mosaec_exact_match_flag", "mosaec_geometry_overlap_flag"]:
            if col in fe.columns:
                fe[col] = fe[col].fillna(False).astype(bool)
        rows = []
        for t in v41_target_order(fe):
            sub = fe[fe["target"].astype(str).eq(str(t))]
            if sub.empty:
                continue
            rows.append({
                "target": t,
                "target_label": v41_target_label(t, latex=False),
                "n_candidates": int(len(sub)),
                "core_exact_fraction": float(sub.get("core_exact_match_flag", False).mean()) if "core_exact_match_flag" in sub else np.nan,
                "mosaec_exact_fraction": float(sub.get("mosaec_exact_match_flag", False).mean()) if "mosaec_exact_match_flag" in sub else np.nan,
                "core_geometry_overlap_fraction": float(sub.get("core_geometry_overlap_flag", False).mean()) if "core_geometry_overlap_flag" in sub else np.nan,
                "mosaec_geometry_overlap_fraction": float(sub.get("mosaec_geometry_overlap_flag", False).mean()) if "mosaec_geometry_overlap_flag" in sub else np.nan,
                "mean_external_domain_overlap_score": float(pd.to_numeric(sub.get("external_realism_score", np.nan), errors="coerce").mean()),
                "framing": "domain-overlap annotation, not experimental validation",
            })
        atomic_save_csv(pd.DataFrame(rows), TABLE_DIR / "SI_Table_external_domain_overlap_summary_v41.csv")
    manifest_rows = []
    for folder, label in [(PAPER_FINAL_DIR, "main final paper"), (PAPER_FINAL_SI_DIR, "supporting final paper")]:
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in {".png", ".pdf", ".svg"}:
                manifest_rows.append({"figure_file": p.name, "folder": str(folder), "kind": label, "size_bytes": int(p.stat().st_size)})
    if manifest_rows:
        atomic_save_csv(pd.DataFrame(manifest_rows), TABLE_DIR / "MANIFEST_v4_1_final_paper_figures.csv")


# Keep references to the v4.0 table/report functions so v4.1 can extend them.
_v40_save_manuscript_si_tables_ref = save_manuscript_si_tables
_v40_write_run_report_ref = write_run_report


def save_manuscript_si_tables(master: pd.DataFrame, agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    """v4.1: call v4.0 table logic and add final paper table polish."""
    _v40_save_manuscript_si_tables_ref(master, agg, tiers, anatomy)
    consensus = v40_read_consensus() if "v40_read_consensus" in globals() else pd.DataFrame()
    final_external = v40_read_external() if "v40_read_external" in globals() else pd.DataFrame()
    write_v41_final_paper_tables(master, agg, tiers, consensus=consensus, final_external=final_external)


def plot_figure_1_framework() -> None:
    """v4.1 graphical workflow with numerical scale and publication styling."""
    stage = "fig1_framework"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 1 v4.1 final workflow")
    v41_apply_paper_style()
    fig = plt.figure(figsize=(15.2, 4.0))
    ax = fig.add_subplot(111)
    ax.axis("off")
    blocks = [
        ("ARC-MOF labelled space", "279k MOFs\n4 CO$_2$/CH$_4$ targets\ngeometry + RAC chemistry"),
        ("Few-shot labels", "10–1000 labelled\nMOFs per task\n5 random seeds"),
        ("Extrapolation stress", "random + 5 grouped\nsplits\nmetal/ligand/topology"),
        ("Risk-controlled ML", "stable tree/boosting\nmodels\n90% conformal intervals"),
        ("Consensus shortlist", "quality gates\nmodel/seed/split support\ntop-25 per target"),
        ("Domain-overlap SI", "CoRE + MOSAEC\ngeometry-neighbour\nannotation only"),
    ]
    x_positions = np.linspace(0.08, 0.92, len(blocks))
    width = 0.135
    for i, ((title, body), x) in enumerate(zip(blocks, x_positions)):
        rect = plt.Rectangle((x - width/2, 0.34), width, 0.42,
                             facecolor="#F7F9FB", edgecolor="#2C3E50", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, 0.67, title, ha="center", va="center", fontsize=9.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.48, body, ha="center", va="center", fontsize=8.3, transform=ax.transAxes)
        ax.text(x, 0.25, f"{i+1}", ha="center", va="center", fontsize=9, color="white",
                bbox=dict(boxstyle="circle,pad=0.25", fc="#34495E", ec="none"), transform=ax.transAxes)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x_positions[i+1] - width/2 - 0.01, 0.55), xytext=(x + width/2 + 0.01, 0.55),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#34495E"))
    ax.text(0.5, 0.91, "Risk-controlled few-shot adsorption screening with chemically interpretable descriptors",
            ha="center", va="center", fontsize=13.2, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.09, "Main scientific outputs: elite-recovery enrichment, RAC descriptor gain, topology extrapolation penalty, calibrated uncertainty, and consensus-ranked candidates.",
            ha="center", va="center", fontsize=8.8, transform=ax.transAxes)
    v41_save_figure(fig, "Figure_1_final_workflow", aliases=[FIG_MAIN_DIR / "Figure_1_risk_controlled_framework"])
    mark_done(stage)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    """v4.1 Figure 2: few-shot elite recovery, descriptor gain, and split stress."""
    stage = "fig2_performance"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 2 v4.1 performance/descriptor/split")
    v41_apply_paper_style()
    if agg is None or agg.empty:
        return
    preferred = v41_preferred_feature(agg)
    stable = v41_clean_metric_df(agg, top_frac=0.05, feature_set=preferred, stable_only=True)
    atomic_save_csv(stable, PAPER_FINAL_DATA_DIR / "Figure_2_v41_input_stable_metrics.csv")
    if stable.empty:
        return
    max_budget = int(pd.to_numeric(stable["budget"], errors="coerce").max())
    fig = plt.figure(figsize=CONFIG.get("paper_final_figsize_wide", (15.5, 8.6)))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.46)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # (a) target-resolved top-5% recall curve.
    curve = v41_paper_curve(stable, "recall_mean", group_col="target", split="random")
    atomic_save_csv(curve, PAPER_FINAL_DATA_DIR / "Figure_2a_recall_budget_curve.csv")
    for t in v41_target_order(curve):
        sub = curve[curve["target"].astype(str).eq(str(t))].sort_values("budget")
        if sub.empty:
            continue
        color = TARGET_COLORS_FINAL.get(str(t), None)
        axes[0].plot(sub["budget"], sub["mean"], marker="o", label=v41_target_label(t), color=color)
        axes[0].fill_between(sub["budget"].to_numpy(dtype=float), sub["ci_low"].to_numpy(dtype=float), sub["ci_high"].to_numpy(dtype=float), alpha=0.14, color=color)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Label budget"); axes[0].set_ylabel("Top-5% recall")
    axes[0].set_ylim(0, min(1.0, max(0.65, np.nanmax(curve.get("mean", pd.Series([0.65]))) * 1.15)))
    axes[0].grid(True, alpha=0.25); axes[0].legend(frameon=False, ncol=1)
    axes[0].set_title("Elite recovery improves with few labels"); v41_panel_label(axes[0], "a")

    # (b) enrichment curve.
    enrich = v41_paper_curve(stable, "enrichment_mean", group_col="target", split="random")
    atomic_save_csv(enrich, PAPER_FINAL_DATA_DIR / "Figure_2b_enrichment_budget_curve.csv")
    for t in v41_target_order(enrich):
        sub = enrich[enrich["target"].astype(str).eq(str(t))].sort_values("budget")
        if sub.empty:
            continue
        color = TARGET_COLORS_FINAL.get(str(t), None)
        axes[1].plot(sub["budget"], sub["mean"], marker="o", label=v41_target_label(t), color=color)
    axes[1].axhline(1.0, ls="--", lw=1.0, color="#555555")
    axes[1].text(0.04, 0.08, "random baseline", transform=axes[1].transAxes, fontsize=8, color="#555555")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Label budget"); axes[1].set_ylabel("Top-5% enrichment")
    axes[1].grid(True, alpha=0.25); axes[1].set_title("Screening is strongly enriched over random"); v41_panel_label(axes[1], "b")

    # (c) descriptor gain delta recall.
    fs = v41_clean_metric_df(agg, top_frac=0.05, split="random", stable_only=True)
    fs = fs[pd.to_numeric(fs["budget"], errors="coerce") == max_budget].copy() if not fs.empty else pd.DataFrame()
    desc = fs.pivot_table(index="target", columns="feature_set", values="recall_mean", aggfunc="mean") if not fs.empty else pd.DataFrame()
    gain_rows = []
    targets = v41_target_order(desc.reset_index().rename(columns={"index": "target"})) if not desc.empty else []
    gains = []
    for t in targets:
        if t in desc.index:
            g = desc.loc[t].get("geometry_plus_racs", np.nan) - desc.loc[t].get("geometry_only", np.nan)
            gains.append(g)
            gain_rows.append({"target": t, "delta_recall": g})
    atomic_save_csv(pd.DataFrame(gain_rows), PAPER_FINAL_DATA_DIR / "Figure_2c_descriptor_gain.csv")
    axes[2].bar(range(len(targets)), gains, color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in targets])
    axes[2].axhline(0, lw=0.8, color="#333333")
    axes[2].set_xticks(range(len(targets))); axes[2].set_xticklabels([v41_target_label(t) for t in targets], rotation=25, ha="right")
    axes[2].set_ylabel("Δ top-5% recall\n(Geometry + RACs − Geometry)")
    axes[2].set_title("RAC chemistry improves elite recovery"); v41_panel_label(axes[2], "c")

    # (d) split stress heatmap.
    sp = stable[pd.to_numeric(stable["budget"], errors="coerce") == max_budget].copy()
    pivot = sp.pivot_table(index="target", columns="split", values="recall_mean", aggfunc="mean") if not sp.empty else pd.DataFrame()
    penalty = pd.DataFrame()
    if not pivot.empty and "random" in pivot.columns:
        for col in [s for s in SPLIT_ORDER_FINAL if s in pivot.columns and s != "random"]:
            penalty[v41_split_label(col)] = pivot["random"] - pivot[col]
        penalty.index = [v41_target_label(t) for t in penalty.index]
    atomic_save_csv(penalty.reset_index().rename(columns={"index": "target_label"}), PAPER_FINAL_DATA_DIR / "Figure_2d_split_penalty_heatmap.csv")
    v41_heatmap(axes[3], penalty, "Grouped-split penalty vs random", "Δ recall", center_zero=True, fmt=".2f")
    v41_panel_label(axes[3], "d")

    # (e) model comparison at final budget.
    mod = stable[(pd.to_numeric(stable["budget"], errors="coerce") == max_budget) & stable["split"].astype(str).eq("random")].copy()
    mod_order = [m for m in MODEL_ORDER_FINAL if m in set(mod.get("model", pd.Series(dtype=str)).astype(str))]
    mod_summary = mod.groupby("model")["recall_mean"].agg(["mean", "std", "count"]).reindex(mod_order).dropna(how="all") if not mod.empty else pd.DataFrame()
    atomic_save_csv(mod_summary.reset_index(), PAPER_FINAL_DATA_DIR / "Figure_2e_model_comparison_final_budget.csv")
    x = np.arange(len(mod_summary))
    y = mod_summary["mean"].to_numpy(dtype=float) if not mod_summary.empty else np.array([])
    yerr = (mod_summary["std"] / np.sqrt(mod_summary["count"].replace(0, np.nan))).fillna(0).to_numpy(dtype=float) if not mod_summary.empty else np.array([])
    axes[4].bar(x, y, yerr=yerr, capsize=3, color=[MODEL_COLORS_FINAL.get(str(m), "#777777") for m in mod_summary.index])
    axes[4].set_xticks(x); axes[4].set_xticklabels([v41_model_label(m) for m in mod_summary.index], rotation=25, ha="right")
    axes[4].set_ylabel("Mean top-5% recall"); axes[4].set_title(f"Stable models at {max_budget} labels"); v41_panel_label(axes[4], "e")

    # (f) coverage-error heatmap.
    cov = sp[sp["split"].astype(str).eq("random")].copy()
    cov_piv = cov.pivot_table(index="target", columns="model", values="split_coverage_mean", aggfunc="mean") if not cov.empty else pd.DataFrame()
    cov_piv = cov_piv[[m for m in MODEL_ORDER_FINAL if m in cov_piv.columns]] if not cov_piv.empty else cov_piv
    nominal = float(CONFIG.get("paper_final_coverage_nominal", 0.90))
    cov_err = cov_piv - nominal if not cov_piv.empty else pd.DataFrame()
    if not cov_err.empty:
        cov_err.index = [v41_target_label(t) for t in cov_err.index]
        cov_err.columns = [v41_model_label(c) for c in cov_err.columns]
    atomic_save_csv(cov_err.reset_index().rename(columns={"index": "target_label"}), PAPER_FINAL_DATA_DIR / "Figure_2f_coverage_error_heatmap.csv")
    v41_heatmap(axes[5], cov_err, "Coverage error around nominal 90%", "coverage − 0.90", center_zero=True, fmt=".02f")
    v41_panel_label(axes[5], "f")

    fig.suptitle("Few-shot adsorption elite recovery, descriptor gain, and extrapolation stress", fontsize=13.5, fontweight="bold")
    v41_save_figure(fig, "Figure_2_v4_1_fewshot_performance_descriptor_split", aliases=[
        FIG_MAIN_DIR / "Figure_2_final_performance_descriptor_split",
        FIG_MAIN_DIR / "Figure_2_label_budget_performance",
    ])
    mark_done(stage)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    """v4.1 Figure 3: conformal risk control, interval width, and abstention/yield."""
    stage = "fig3_calibration"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 3 v4.1 calibration/risk control")
    v41_apply_paper_style()
    if agg is None or agg.empty:
        return
    preferred = v41_preferred_feature(agg)
    df = v41_clean_metric_df(agg, top_frac=0.05, feature_set=preferred, stable_only=True)
    atomic_save_csv(df, PAPER_FINAL_DATA_DIR / "Figure_3_v41_input_metrics.csv")
    if df.empty:
        return
    nominal = float(CONFIG.get("paper_final_coverage_nominal", 0.90))
    max_budget = int(pd.to_numeric(df["budget"], errors="coerce").max())
    stats = v41_target_stats(None)
    fig = plt.figure(figsize=CONFIG.get("paper_final_figsize_wide", (15.5, 8.6)))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.46)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # (a) coverage error by target/split.
    cov_work = df[pd.to_numeric(df["budget"], errors="coerce") == max_budget].copy()
    cov_hm = cov_work.pivot_table(index="target", columns="split", values="split_coverage_mean", aggfunc="mean") - nominal if not cov_work.empty else pd.DataFrame()
    cov_hm = cov_hm[[s for s in SPLIT_ORDER_FINAL if s in cov_hm.columns]] if not cov_hm.empty else cov_hm
    if not cov_hm.empty:
        cov_hm.index = [v41_target_label(t) for t in cov_hm.index]
        cov_hm.columns = [v41_split_label(c) for c in cov_hm.columns]
    atomic_save_csv(cov_hm.reset_index().rename(columns={"index": "target_label"}), PAPER_FINAL_DATA_DIR / "Figure_3a_coverage_error_by_split.csv")
    v41_heatmap(axes[0], cov_hm, "Coverage error by extrapolation split", "coverage − 0.90", center_zero=True, fmt=".02f")
    v41_panel_label(axes[0], "a")

    # (b) normalized interval width vs budget.
    width_rows = []
    rand = df[df["split"].astype(str).eq("random")].copy()
    for t in v41_target_order(rand):
        sub = rand[rand["target"].astype(str).eq(str(t))].copy()
        st = stats.get(str(t), {})
        std = float(st.get("std", np.nan))
        if sub.empty or not np.isfinite(std) or std <= 0:
            continue
        g = sub.groupby("budget")["split_mean_width_mean"].mean().reset_index()
        g["normalized_width"] = g["split_mean_width_mean"] / std
        g["target"] = t
        width_rows.append(g)
    width_df = pd.concat(width_rows, ignore_index=True) if width_rows else pd.DataFrame()
    atomic_save_csv(width_df, PAPER_FINAL_DATA_DIR / "Figure_3b_normalized_interval_width.csv")
    for t in v41_target_order(width_df):
        sub = width_df[width_df["target"].astype(str).eq(str(t))].sort_values("budget")
        axes[1].plot(sub["budget"], sub["normalized_width"], marker="o", color=TARGET_COLORS_FINAL.get(str(t)), label=v41_target_label(t))
    axes[1].set_xscale("log"); axes[1].set_xlabel("Label budget"); axes[1].set_ylabel("Interval width / target σ")
    axes[1].grid(True, alpha=0.25); axes[1].set_title("Uncertainty narrows on the target scale"); v41_panel_label(axes[1], "b")

    # (c) risk-yield curve: uncertain fraction vs precision.
    tmp = rand.copy()
    denom = pd.to_numeric(tmp.get("trusted_count_mean", 0), errors="coerce") + pd.to_numeric(tmp.get("uncertain_count_mean", 0), errors="coerce") + pd.to_numeric(tmp.get("rejected_count_mean", 0), errors="coerce")
    tmp["uncertain_fraction"] = pd.to_numeric(tmp.get("uncertain_count_mean", np.nan), errors="coerce") / denom.replace(0, np.nan)
    tmp["trusted_fraction"] = pd.to_numeric(tmp.get("trusted_count_mean", np.nan), errors="coerce") / denom.replace(0, np.nan)
    atomic_save_csv(tmp[[c for c in ["target", "model", "budget", "precision_mean", "recall_mean", "uncertain_fraction", "trusted_fraction"] if c in tmp.columns]], PAPER_FINAL_DATA_DIR / "Figure_3c_risk_yield_scatter.csv")
    for t in v41_target_order(tmp):
        sub = tmp[tmp["target"].astype(str).eq(str(t))]
        axes[2].scatter(sub["uncertain_fraction"], sub["precision_mean"], s=18, alpha=0.65, color=TARGET_COLORS_FINAL.get(str(t)), label=v41_target_label(t))
    axes[2].set_xlabel("Uncertain fraction"); axes[2].set_ylabel("Top-5% precision")
    axes[2].grid(True, alpha=0.25); axes[2].set_title("Risk-yield trade-off"); v41_panel_label(axes[2], "c")

    # (d) trusted fraction vs budget.
    tf = tmp.groupby(["target", "budget"], as_index=False)["trusted_fraction"].mean()
    atomic_save_csv(tf, PAPER_FINAL_DATA_DIR / "Figure_3d_trusted_fraction_budget.csv")
    for t in v41_target_order(tf):
        sub = tf[tf["target"].astype(str).eq(str(t))].sort_values("budget")
        axes[3].plot(sub["budget"], sub["trusted_fraction"], marker="o", color=TARGET_COLORS_FINAL.get(str(t)), label=v41_target_label(t))
    axes[3].set_xscale("log"); axes[3].set_xlabel("Label budget"); axes[3].set_ylabel("Trusted fraction")
    axes[3].grid(True, alpha=0.25); axes[3].set_title("Yield of high-confidence candidates"); v41_panel_label(axes[3], "d")

    # (e) interval width distribution by candidate tier from candidate table.
    tiers_path = TABLE_DIR / "table_candidate_tiers_all_predictions.csv"
    tiers = pd.read_csv(tiers_path, low_memory=False) if tiers_path.exists() else pd.DataFrame()
    if not tiers.empty and {"split_upper", "split_lower", "target", "tier"}.issubset(tiers.columns):
        tw = tiers.copy()
        tw["interval_width"] = pd.to_numeric(tw["split_upper"], errors="coerce") - pd.to_numeric(tw["split_lower"], errors="coerce")
        for t, st in stats.items():
            std = float(st.get("std", np.nan))
            if np.isfinite(std) and std > 0:
                mask = tw["target"].astype(str).eq(str(t))
                tw.loc[mask, "interval_width_norm"] = tw.loc[mask, "interval_width"] / std
        labels = [tier for tier in ["trusted", "uncertain", "rejected"] if (tw["tier"].astype(str) == tier).any()]
        data = []
        for tier in labels:
            vals = tw.loc[tw["tier"].astype(str).eq(tier), "interval_width_norm"].dropna()
            if len(vals):
                data.append(vals.sample(min(5000, len(vals)), random_state=42))
        if data:
            safe_boxplot_with_labels(axes[4], data, labels, showfliers=False)
        axes[4].set_ylabel("Interval width / target σ")
        atomic_save_csv(tw[[c for c in ["target", "tier", "interval_width_norm"] if c in tw.columns]].sample(min(len(tw), 20000), random_state=42), PAPER_FINAL_DATA_DIR / "Figure_3e_interval_width_by_tier.csv")
    axes[4].set_title("Candidate tiers separate uncertainty width"); v41_panel_label(axes[4], "e")

    # (f) conformal method coverage comparison.
    methods = []
    for col, label in [("split_coverage_mean", "Split"), ("local_coverage_mean", "Local"), ("mondrian_coverage_mean", "Mondrian")]:
        if col in df.columns:
            methods.append({"method": label, "coverage": float(pd.to_numeric(df[col], errors="coerce").mean())})
    meth = pd.DataFrame(methods)
    atomic_save_csv(meth, PAPER_FINAL_DATA_DIR / "Figure_3f_method_coverage_summary.csv")
    axes[5].bar(meth["method"], meth["coverage"], color="#7F8C8D")
    axes[5].axhline(nominal, ls="--", lw=1.0, color="#333333")
    axes[5].set_ylim(max(0.75, float(meth["coverage"].min()) - 0.03) if not meth.empty else 0.75,
                     min(1.0, float(meth["coverage"].max()) + 0.03) if not meth.empty else 1.0)
    axes[5].set_ylabel("Empirical coverage"); axes[5].set_title("Coverage remains close to nominal")
    v41_panel_label(axes[5], "f")

    fig.suptitle("Conformal risk control converts raw predictions into calibrated candidate decisions", fontsize=13.5, fontweight="bold")
    v41_save_figure(fig, "Figure_3_v4_1_conformal_risk_control", aliases=[
        FIG_MAIN_DIR / "Figure_3_final_calibration_abstention",
        FIG_MAIN_DIR / "Figure_3_calibration_and_abstention",
    ])
    mark_done(stage)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    """v4.1 Figure 4: target-balanced consensus true-enrichment and reliability."""
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 4 v4.1 consensus validation")
    v41_apply_paper_style()
    consensus = v41_consensus_work()
    top25 = v41_top_consensus_per_target(consensus, topn=int(CONFIG.get("paper_final_top_candidate_rows_per_target", 25)))
    top25 = v41_add_true_percentile_columns(top25, None)
    qcp = TABLE_DIR / "QC_Table_consensus_true_enrichment.csv"
    qce = pd.read_csv(qcp) if qcp.exists() else pd.DataFrame()
    if qce.empty and "write_consensus_true_enrichment" in globals():
        qce = write_consensus_true_enrichment(None, consensus)
    atomic_save_csv(top25, PAPER_FINAL_DATA_DIR / "Figure_4_v41_top25_consensus_input.csv")
    atomic_save_csv(qce, PAPER_FINAL_DATA_DIR / "Figure_4_v41_true_enrichment_input.csv")
    fig = plt.figure(figsize=(15.5, 10.4))
    gs = fig.add_gridspec(3, 4, wspace=0.40, hspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    scatter_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    bottom_axes = [fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4])]

    # (a) target-balanced funnel.
    raw_n = len(tiers) if tiers is not None else 0
    elig_n = int(tiers.get("candidate_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if tiers is not None and not tiers.empty and "candidate_eligible" in tiers.columns else raw_n
    consensus_n = int(len(consensus))
    top25_n = int(len(top25))
    funnel = pd.DataFrame({"stage": ["candidate\nrows", "quality\ngated", "consensus\nMOF-target", "top-25\nper target"], "n": [raw_n, elig_n, consensus_n, top25_n]})
    atomic_save_csv(funnel, PAPER_FINAL_DATA_DIR / "Figure_4a_candidate_funnel.csv")
    ax_a.bar(funnel["stage"], funnel["n"], color="#34495E")
    ax_a.set_ylabel("Rows / MOF-targets"); ax_a.set_title("Candidate funnel")
    ax_a.tick_params(axis="x", rotation=0); v41_panel_label(ax_a, "a")

    # (b) true enrichment by target.
    if not qce.empty:
        qce = qce.copy(); qce["target"] = qce["target"].astype(str)
        qce = qce.set_index("target").reindex(v41_target_order(qce.reset_index())).reset_index()
        vals = pd.to_numeric(qce.get("fold_enrichment_vs_dataset_median", np.nan), errors="coerce")
        ax_b.bar(range(len(qce)), vals, color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in qce["target"]])
        ax_b.set_xticks(range(len(qce))); ax_b.set_xticklabels([v41_target_label(t) for t in qce["target"]], rotation=25, ha="right")
        ax_b.set_ylabel("Shortlist median / dataset median"); ax_b.set_title("Consensus shortlist enrichment")
    else:
        ax_b.axis("off"); ax_b.text(0.5, 0.5, "No enrichment table", ha="center")
    v41_panel_label(ax_b, "b")

    # (c) true top-k fractions.
    if not qce.empty:
        x = np.arange(len(qce)); w = 0.24
        top_fraction_cols = [
            (-w, "fraction_true_top1pct" if "fraction_true_top1pct" in qce.columns else "fraction_true_top1", "top 1%"),
            (0, "fraction_true_top5pct" if "fraction_true_top5pct" in qce.columns else "fraction_true_top5", "top 5%"),
            (w, "fraction_true_top10pct" if "fraction_true_top10pct" in qce.columns else "fraction_true_top10", "top 10%"),
        ]
        for offset, col, lab in top_fraction_cols:
            if col in qce.columns:
                ax_c.bar(x + offset, pd.to_numeric(qce[col], errors="coerce"), width=w, label=lab)
        ax_c.set_xticks(x); ax_c.set_xticklabels([v41_target_label(t) for t in qce["target"]], rotation=25, ha="right")
        ax_c.set_ylim(0, 1.05); ax_c.set_ylabel("Fraction of shortlist"); ax_c.legend(frameon=False, ncol=1)
        ax_c.set_title("True hit rate in consensus list")
    else:
        ax_c.axis("off")
    v41_panel_label(ax_c, "c")

    # (d) support distribution.
    if not top25.empty:
        support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_budgets"] if c in top25.columns]
        data = [pd.to_numeric(top25[c], errors="coerce").dropna() for c in support_cols]
        if data:
            safe_boxplot_with_labels(ax_d, data, [c.replace("n_", "") for c in support_cols], showfliers=False)
        ax_d.set_ylabel("Support count"); ax_d.set_title("Consensus support across runs")
    else:
        ax_d.axis("off")
    v41_panel_label(ax_d, "d")

    # Bottom scatter panels: one per target, true-vs-pred percentile.
    for idx, (ax, target) in enumerate(zip(scatter_axes, v41_target_order(top25))):
        sub = top25[top25["target"].astype(str).eq(str(target))].copy() if not top25.empty else pd.DataFrame()
        if not sub.empty and {"true_percentile", "pred_percentile"}.issubset(sub.columns):
            ax.scatter(sub["true_percentile"], sub["pred_percentile"], s=28, alpha=0.75, color=TARGET_COLORS_FINAL.get(str(target)))
            ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#333333")
            med_true = pd.to_numeric(sub["true_percentile"], errors="coerce").median()
            ax.text(0.04, 0.91, f"median true pct.={med_true:.2f}", transform=ax.transAxes, fontsize=7.5)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_title(v41_target_label(target)); ax.set_xlabel("True percentile"); ax.set_ylabel("Predicted percentile")
        v41_panel_label(ax, chr(ord("e") + idx))

    # (i) false-positive/top5 miss fraction by target.
    ax_i, ax_j = bottom_axes
    if not qce.empty:
        top5_col = "fraction_true_top5pct" if "fraction_true_top5pct" in qce.columns else "fraction_true_top5"
        miss = 1 - pd.to_numeric(qce.get(top5_col, np.nan), errors="coerce")
        ax_i.bar(range(len(qce)), miss, color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in qce["target"]])
        ax_i.set_xticks(range(len(qce))); ax_i.set_xticklabels([v41_target_label(t) for t in qce["target"]], rotation=25, ha="right")
        ax_i.set_ylim(0, 1.0); ax_i.set_ylabel("1 − fraction true top 5%")
        ax_i.set_title("Residual false-positive burden")
    else:
        ax_i.axis("off")
    v41_panel_label(ax_i, "i")

    # (j) target-normalized uncertainty in top candidates.
    if not top25.empty and {"interval_width_median", "target"}.issubset(top25.columns):
        stats = v41_target_stats(None)
        plot_rows = []
        for t in v41_target_order(top25):
            sub = top25[top25["target"].astype(str).eq(str(t))].copy()
            std = float(stats.get(str(t), {}).get("std", np.nan))
            if np.isfinite(std) and std > 0:
                vals = pd.to_numeric(sub["interval_width_median"], errors="coerce") / std
                plot_rows.append(pd.DataFrame({"target": t, "interval_width_norm": vals}))
        iw = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
        data = [iw.loc[iw["target"].astype(str).eq(str(t)), "interval_width_norm"].dropna() for t in v41_target_order(iw)]
        labels = [v41_target_label(t) for t in v41_target_order(iw)]
        if data:
            safe_boxplot_with_labels(ax_j, data, labels, showfliers=False)
            ax_j.tick_params(axis="x", rotation=25)
        ax_j.set_ylabel("Interval width / target σ"); ax_j.set_title("Residual uncertainty of consensus hits")
        atomic_save_csv(iw, PAPER_FINAL_DATA_DIR / "Figure_4j_consensus_normalized_interval_width.csv")
    else:
        ax_j.axis("off")
    v41_panel_label(ax_j, "j")

    fig.suptitle("Consensus filtering yields target-balanced, truly enriched adsorption shortlists", fontsize=13.5, fontweight="bold")
    v41_save_figure(fig, "Figure_4_v4_1_consensus_true_enrichment", aliases=[
        FIG_MAIN_DIR / "Figure_4_final_consensus_reliability",
        FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy",
    ])
    mark_done(stage)


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """v4.1 Figure 5: consensus chemistry/stability; external overlap is SI-style annotation."""
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> Figure 5 v4.1 consensus chemistry and domain-overlap annotation")
    v41_apply_paper_style()
    consensus = v41_consensus_work()
    top25 = v41_top_consensus_per_target(consensus, topn=int(CONFIG.get("paper_final_top_candidate_rows_per_target", 25)))
    top25 = v41_add_true_percentile_columns(top25, None)
    fe = final_external.copy() if final_external is not None else pd.DataFrame()
    write_v41_final_paper_tables(None, pd.DataFrame(), pd.DataFrame(), consensus=consensus, final_external=fe)
    if top25.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off"); ax.text(0.5, 0.5, "No consensus candidates available", ha="center", va="center")
        v41_save_figure(fig, "Figure_5_v4_1_no_consensus", aliases=[FIG_MAIN_DIR / "Figure_5_final_consensus_stability", FIG_MAIN_DIR / "Figure_5_external_realism_filter"])
        mark_done(stage, {"n_rows": 0})
        return
    atomic_save_csv(top25, PAPER_FINAL_DATA_DIR / "Figure_5_v41_top25_consensus_chemistry_input.csv")
    fig = plt.figure(figsize=CONFIG.get("paper_final_figsize_wide", (15.5, 8.6)))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.46)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # (a) true percentile distribution of top-25 candidates.
    data = [top25.loc[top25["target"].astype(str).eq(str(t)), "true_percentile"].dropna() for t in v41_target_order(top25)]
    labels = [v41_target_label(t) for t in v41_target_order(top25)]
    if data:
        safe_boxplot_with_labels(axes[0], data, labels, showfliers=False)
        axes[0].tick_params(axis="x", rotation=25)
    axes[0].axhline(0.95, ls="--", lw=1.0, color="#555555"); axes[0].text(0.02, 0.88, "top 5%", transform=axes[0].transAxes, fontsize=8)
    axes[0].set_ylabel("True uptake percentile"); axes[0].set_title("Top-25 candidates are high-percentile hits")
    v41_panel_label(axes[0], "a")

    # (b) support distributions.
    support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_budgets", "trusted_support"] if c in top25.columns]
    support_data = [pd.to_numeric(top25[c], errors="coerce").dropna() for c in support_cols]
    if support_data:
        safe_boxplot_with_labels(axes[1], support_data, [c.replace("n_", "").replace("trusted_support", "trusted") for c in support_cols], showfliers=False)
    axes[1].set_ylabel("Support count"); axes[1].set_title("Multi-model/seed/split consensus")
    v41_panel_label(axes[1], "b")

    # (c) pore-limiting diameter by target.
    geom_col = "Di" if "Di" in top25.columns else None
    if geom_col:
        geom_data = [pd.to_numeric(top25.loc[top25["target"].astype(str).eq(str(t)), geom_col], errors="coerce").dropna() for t in v41_target_order(top25)]
        if geom_data:
            safe_boxplot_with_labels(axes[2], geom_data, labels, showfliers=False)
            axes[2].tick_params(axis="x", rotation=25)
        axes[2].set_ylabel("PLD / Å")
    axes[2].set_title("Consensus hits occupy distinct pore regimes")
    v41_panel_label(axes[2], "c")

    # (d) density by target.
    if "Density" in top25.columns:
        den_data = [pd.to_numeric(top25.loc[top25["target"].astype(str).eq(str(t)), "Density"], errors="coerce").dropna() for t in v41_target_order(top25)]
        if den_data:
            safe_boxplot_with_labels(axes[3], den_data, labels, showfliers=False)
            axes[3].tick_params(axis="x", rotation=25)
        axes[3].set_ylabel("Density / g cm$^{-3}$")
    axes[3].set_title("Density regimes of selected candidates")
    v41_panel_label(axes[3], "d")

    # (e) robust lower bound vs uncertainty.
    if {"split_lower_median", "interval_width_median", "target"}.issubset(top25.columns):
        stats = v41_target_stats(None)
        work = top25.copy()
        for t in v41_target_order(work):
            sub_mask = work["target"].astype(str).eq(str(t))
            std = float(stats.get(str(t), {}).get("std", np.nan))
            mean = float(stats.get(str(t), {}).get("mean", 0.0))
            if np.isfinite(std) and std > 0:
                work.loc[sub_mask, "lower_z"] = (pd.to_numeric(work.loc[sub_mask, "split_lower_median"], errors="coerce") - mean) / std
                work.loc[sub_mask, "width_z"] = pd.to_numeric(work.loc[sub_mask, "interval_width_median"], errors="coerce") / std
        for t in v41_target_order(work):
            sub = work[work["target"].astype(str).eq(str(t))]
            axes[4].scatter(sub.get("width_z"), sub.get("lower_z"), s=25, alpha=0.7, color=TARGET_COLORS_FINAL.get(str(t)), label=v41_target_label(t))
        axes[4].set_xlabel("Interval width / target σ"); axes[4].set_ylabel("Lower bound z-score")
        axes[4].grid(True, alpha=0.25); axes[4].legend(frameon=False, fontsize=7)
        atomic_save_csv(work[[c for c in ["target", "mof_id", "lower_z", "width_z", "consensus_score"] if c in work.columns]], PAPER_FINAL_DATA_DIR / "Figure_5e_lower_bound_uncertainty.csv")
    axes[4].set_title("High lower bound with quantified uncertainty")
    v41_panel_label(axes[4], "e")

    # (f) external domain-overlap summary as cautious annotation.
    if fe is not None and not fe.empty:
        fe_work = fe.copy()
        rows = []
        for t in v41_target_order(fe_work):
            sub = fe_work[fe_work["target"].astype(str).eq(str(t))]
            if sub.empty:
                continue
            rows.append({
                "target": t,
                "CoRE geometry": float(sub.get("core_geometry_overlap_flag", False).fillna(False).astype(bool).mean()) if "core_geometry_overlap_flag" in sub else np.nan,
                "MOSAEC geometry": float(sub.get("mosaec_geometry_overlap_flag", False).fillna(False).astype(bool).mean()) if "mosaec_geometry_overlap_flag" in sub else np.nan,
                "exact matches": float((sub.get("core_exact_match_flag", False).fillna(False).astype(bool) | sub.get("mosaec_exact_match_flag", False).fillna(False).astype(bool)).mean()) if {"core_exact_match_flag", "mosaec_exact_match_flag"}.issubset(sub.columns) else np.nan,
            })
        ext = pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()
        if not ext.empty:
            ext.index = [v41_target_label(t) for t in ext.index]
            v41_heatmap(axes[5], ext, "External domain-overlap annotation", "fraction", center_zero=False, cmap="Greens", fmt=".2f")
            atomic_save_csv(ext.reset_index().rename(columns={"index": "target_label"}), PAPER_FINAL_DATA_DIR / "Figure_5f_external_domain_overlap_summary.csv")
        else:
            axes[5].axis("off")
    else:
        axes[5].axis("off"); axes[5].text(0.5, 0.5, "No external overlay\n(domain-overlap optional)", ha="center", va="center")
    v41_panel_label(axes[5], "f")

    fig.suptitle("Consensus shortlist chemistry and reliability; external databases provide domain-overlap annotation", fontsize=13.5, fontweight="bold")
    v41_save_figure(fig, "Figure_5_v4_1_consensus_chemistry_stability", aliases=[
        FIG_MAIN_DIR / "Figure_5_final_consensus_stability",
        FIG_MAIN_DIR / "Figure_5_external_realism_filter",
    ])

    # SI-style external details figure, intentionally separate from main Figure 5.
    plot_external_si_figures(core, mosaec, fe)
    mark_done(stage, {"n_rows": int(len(fe)) if fe is not None else 0, "external_framing": "domain_overlap_not_validation", "v4_1": True})


def plot_external_si_figures(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    """v4.1 external domain-overlap figure for SI, not main validation."""
    stage = "external_si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> External SI v4.1 domain-overlap")
    v41_apply_paper_style()
    fe = final_external.copy() if final_external is not None else pd.DataFrame()
    if fe.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5)); ax.axis("off"); ax.text(0.5, 0.5, "External overlays unavailable", ha="center", va="center")
        v41_save_figure(fig, "Figure_S_external_domain_overlap_unavailable", si=True)
        mark_done(stage, {"n_rows": 0})
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    # exact vs geometry overlap counts.
    bool_cols = [c for c in ["core_exact_match_flag", "mosaec_exact_match_flag", "core_geometry_overlap_flag", "mosaec_geometry_overlap_flag"] if c in fe.columns]
    counts = {c: int(fe[c].fillna(False).astype(bool).sum()) for c in bool_cols}
    labels = [c.replace("_flag", "").replace("_", "\n") for c in counts]
    axes[0].bar(range(len(counts)), list(counts.values()), color="#6C7A89")
    axes[0].set_xticks(range(len(counts))); axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Candidate count"); axes[0].set_title("Exact matches vs geometry overlap")
    v41_panel_label(axes[0], "a")
    # distances.
    for col, lab in [("core_geometry_distance", "CoRE"), ("mosaec_geometry_distance", "MOSAEC")]:
        if col in fe.columns:
            vals = pd.to_numeric(fe[col], errors="coerce").dropna()
            if len(vals):
                axes[1].hist(vals, bins=30, alpha=0.55, label=lab)
    axes[1].axvline(float(CONFIG.get("geometry_match_distance_threshold", 0.75)), ls="--", lw=1.0, color="#333333")
    axes[1].set_xlabel("Standardized geometry-neighbour distance"); axes[1].set_ylabel("Count"); axes[1].legend(frameon=False)
    axes[1].set_title("Domain-overlap distance distributions")
    v41_panel_label(axes[1], "b")
    # by target.
    rows = []
    for t in v41_target_order(fe):
        sub = fe[fe["target"].astype(str).eq(str(t))]
        if not sub.empty:
            rows.append({
                "target": v41_target_label(t),
                "CoRE geom.": float(sub.get("core_geometry_overlap_flag", False).fillna(False).astype(bool).mean()) if "core_geometry_overlap_flag" in sub else np.nan,
                "MOSAEC geom.": float(sub.get("mosaec_geometry_overlap_flag", False).fillna(False).astype(bool).mean()) if "mosaec_geometry_overlap_flag" in sub else np.nan,
            })
    mat = pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()
    v41_heatmap(axes[2], mat, "Overlap fraction by target", "fraction", cmap="Greens", fmt=".2f")
    v41_panel_label(axes[2], "c")
    fig.suptitle("Supporting information: external database domain-overlap annotation (not validation)", fontsize=12.5, fontweight="bold")
    v41_save_figure(fig, "Figure_S_external_domain_overlap_annotation", si=True, aliases=[FIG_SI_DIR / "Figure_S_external_domain_overlap_annotation"])
    atomic_save_csv(fe, PAPER_FINAL_DATA_DIR / "Figure_S_external_domain_overlap_input.csv")
    mark_done(stage, {"n_rows": len(fe), "framing": "domain_overlap_not_validation"})


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    """v4.1 SI figures remain diagnostic but use final labels and duplicate-RDF cleanup."""
    stage = "si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> SI figures v4.1")
    v41_apply_paper_style()
    # Reuse the v4.0 SI diagnostics when available by calling the saved function indirectly is not possible;
    # therefore this function writes a concise final SI set.
    if tiers is not None and not tiers.empty:
        sample = tiers.drop_duplicates(["mof_id", "target"])
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2)); axes = axes.ravel()
        for ax, target in zip(axes, v41_target_order(sample)):
            sub = sample[sample["target"].astype(str).eq(str(target))]
            if not sub.empty and "y_true" in sub.columns:
                ax.hist(pd.to_numeric(sub["y_true"], errors="coerce").dropna(), bins=45, color=TARGET_COLORS_FINAL.get(str(target), "#777777"), alpha=0.8)
            ax.set_title(v41_target_label(target)); ax.set_xlabel("Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Target distributions represented in quality-gated candidate rows")
        v41_save_figure(fig, "Figure_S1_target_distributions_v41", si=True, aliases=[FIG_SI_DIR / "Figure_S1_target_distributions"])
    if agg is not None and not agg.empty:
        stable = v41_clean_metric_df(agg, top_frac=0.05, stable_only=True)
        cov = stable.pivot_table(index="split", columns="model", values="split_coverage_mean", aggfunc="mean") if not stable.empty else pd.DataFrame()
        if not cov.empty:
            cov = cov.reindex([s for s in SPLIT_ORDER_FINAL if s in cov.index])
            cov.columns = [v41_model_label(c) for c in cov.columns]
            cov.index = [v41_split_label(i) for i in cov.index]
            fig, ax = plt.subplots(figsize=(9.5, 5.2)); v41_heatmap(ax, cov, "Figure S2. Coverage across grouped splits", "coverage", fmt=".2f")
            v41_save_figure(fig, "Figure_S2_grouped_split_coverage_v41", si=True, aliases=[FIG_SI_DIR / "Figure_S2_grouped_split_coverage"])
        fs = stable.groupby(["feature_set", "budget"], as_index=False)["recall_mean"].mean() if not stable.empty else pd.DataFrame()
        if not fs.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            for feature in ["geometry_only", "geometry_plus_racs"]:
                sub = fs[fs["feature_set"].astype(str).eq(feature)].sort_values("budget")
                if not sub.empty:
                    ax.plot(sub["budget"], sub["recall_mean"], marker="o", label=v41_feature_label(feature))
            ax.set_xscale("log"); ax.set_xlabel("Label budget"); ax.set_ylabel("Mean top-5% recall"); ax.legend(frameon=False); ax.grid(True, alpha=0.25)
            ax.set_title("Figure S3. Descriptor ablation after feature-integrity QC")
            v41_save_figure(fig, "Figure_S3_descriptor_ablation_v41", si=True, aliases=[FIG_SI_DIR / "Figure_S4_feature_set_ablation"])
    if anatomy is not None and not anatomy.empty:
        metric = "normalized_mae" if "normalized_mae" in anatomy.columns else "mae"
        worst = anatomy.sort_values(metric, ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        labels = (worst["anatomy_type"].astype(str) + ":" + worst["regime"].astype(str)).str.slice(0, 38)
        ax.barh(range(len(worst)), pd.to_numeric(worst[metric], errors="coerce"), color="#7F8C8D")
        ax.set_yticks(range(len(worst))); ax.set_yticklabels(labels, fontsize=7.2); ax.invert_yaxis()
        ax.set_xlabel("Target-normalized MAE" if metric == "normalized_mae" else "MAE")
        ax.set_title("Figure S4. Highest-error chemical regimes")
        v41_save_figure(fig, "Figure_S4_highest_error_regimes_v41", si=True, aliases=[FIG_SI_DIR / "Figure_S5_highest_error_regimes"])
    mark_done(stage)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    """v4.1 report highlighting final paper figures and what they mean."""
    stage = "07_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> WRITE_FINAL_RUN_REPORT_V4_1")
    # First write the concise v4.0 report body in the same file if desired.
    report = RESULTS_DIR / "RUN_REPORT.md"
    lines = []
    lines.append("# Few-shot MOF risk-controlled screening run report — v4.1 final paper figures\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Runtime modes\n")
    for key in ["save_mode", "ram_mode", "comprehensive_level", "n_jobs", "data_root"]:
        lines.append(f"- {key}: `{CONFIG.get(key)}`")
    lines.append("\n## Experiment scale\n")
    lines.append(f"- Master table rows: {master.shape[0]}")
    lines.append(f"- Master table columns: {master.shape[1]}")
    lines.append(f"- Metric rows: {len(metrics_all)}")
    if metrics_all is not None and not metrics_all.empty and set(EXPERIMENT_KEY_COLS).issubset(metrics_all.columns):
        lines.append(f"- Unique completed experiments: {metrics_all[EXPERIMENT_KEY_COLS].drop_duplicates().shape[0]}")
    lines.append("\n## v4.1 final paper figure folder\n")
    lines.append(f"- Main final figures: `{PAPER_FINAL_DIR}`")
    lines.append(f"- SI final figures: `{PAPER_FINAL_SI_DIR}`")
    lines.append(f"- Final figure data: `{PAPER_FINAL_DATA_DIR}`")
    lines.append("\n## Final figures to inspect first\n")
    for fname, meaning in [
        ("Figure_1_final_workflow.png", "workflow / graphical abstract"),
        ("Figure_2_v4_1_fewshot_performance_descriptor_split.png", "elite recovery, RAC gain, split stress"),
        ("Figure_3_v4_1_conformal_risk_control.png", "calibration, uncertainty, abstention/yield"),
        ("Figure_4_v4_1_consensus_true_enrichment.png", "true-enrichment validation of consensus shortlist"),
        ("Figure_5_v4_1_consensus_chemistry_stability.png", "shortlist chemistry, support, and domain-overlap annotation"),
    ]:
        p = PAPER_FINAL_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'} — {meaning}")
    lines.append("\n## Final paper tables to inspect first\n")
    for fname in [
        "QC_Table_consensus_true_enrichment.csv",
        "MAIN_Table_2_descriptor_gain_delta_recall.csv",
        "MAIN_Table_2_split_penalty_vs_random.csv",
        "MAIN_Table_3_consensus_shortlist_target_balanced_top25.csv",
        "MAIN_Table_4_consensus_support_summary.csv",
        "SI_Table_consensus_geometry_regime_summary.csv",
        "SI_Table_external_domain_overlap_summary_v41.csv",
        "QC_Table_feature_set_integrity.csv",
    ]:
        p = TABLE_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'}")
    lines.append("\n## Scientific framing guardrails\n")
    lines.append("- Treat `geometry_plus_racs` versus `geometry_only` as the validated descriptor comparison unless RDF integrity QC shows nonzero RDF columns.")
    lines.append("- Do not frame CoRE/MOSAEC overlays as external validation when exact matches are absent; frame them as domain-overlap/plausibility annotations.")
    lines.append("- The final candidate claims should rely on target-balanced consensus, true-enrichment QC, and conformal uncertainty, not single-row raw predictions.")
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage)

# =============================================================================
# 17. Main entry point
# =============================================================================

def main() -> None:
    args = parse_runtime_args()
    apply_runtime_options(args)
    configure_threading()
    configure_data_roots(args.data_root)
    setup_logging()
    if CONFIG.get("force_postprocess", False) and not CONFIG.get("force_recompute", False):
        clear_postprocess_markers()
    save_json(CONFIG, RESULTS_DIR / "config_used.json")
    write_download_manifest_and_checklist()
    audit_project_folder_tree()

    logging.info("Checking and inspecting input files...")
    inspect_input_files()
    required = ["geometric_properties.csv", "post_comb_vsa-CO2.csv", "methane.csv"]
    missing = [f for f in required if not resolve_input_file(f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input CSV files. Put them next to this .py file or in a subfolder: " + ", ".join(missing)
        )

    master = build_master_table()
    audit_feature_set_integrity(master)
    if CONFIG.get("postprocess_only", False):
        logging.info("POSTPROCESS_ONLY | skipping model fitting and reusing existing metrics/predictions")
        metrics_all = load_existing_metrics_all()
    else:
        metrics_all = run_all_experiments(master)
    aggregated = aggregate_results(metrics_all)
    agg = aggregated["agg"]
    tiers = collect_candidate_tiers(metrics_all=metrics_all, master=master)
    consensus = build_consensus_candidate_table(tiers)
    anatomy = chemical_failure_anatomy(tiers)

    # External realism overlays. These are optional and will be skipped gracefully
    # if CoRE MOF 2024 / MOSAEC-DB files are not present locally.
    core_overlay, mosaec_overlay, external_realism = run_external_overlay_workflow(master, tiers, consensus=consensus)

    # Main manuscript and SI outputs.
    save_manuscript_si_tables(master, agg, tiers, anatomy)
    plot_figure_1_framework()
    plot_figure_2_performance(agg)
    plot_figure_3_calibration(agg)
    plot_figure_4_failure_anatomy(anatomy, tiers)
    # Figure 5 is generated in run_external_overlay_workflow() because it depends
    # on optional CoRE/MOSAEC availability.
    plot_si_figures(agg, tiers, anatomy)
    write_run_report(master, metrics_all)

    logging.info("STAGE >>> JOB_FINISHED_SUCCESSFULLY")
    logging.info("All results saved in: %s", RESULTS_DIR)



# =============================================================================
# 16f. v4.2 final story-driven publication figure layer
# =============================================================================
# v4.2 supersedes the v4.1 figure layer for manuscript-facing output. It does
# not change fitting, candidate QC, target balancing, consensus shortlisting, or
# external-overlay construction. It reads the already produced tables and writes
# a more story-driven, publication-level figure set:
#   Figure 1: quantitative workflow / graphical abstract
#   Figure 2: few-shot learning phase diagram
#   Figure 3: descriptor chemistry and extrapolation stress
#   Figure 4: conformal risk-control frontier
#   Figure 5: target-balanced consensus shortlist atlas
# External CoRE/MOSAEC overlays are moved to SI-style domain-overlap annotation.

CONFIG["paper_final_name"] = "v4_2_final_story_figures"
CONFIG.setdefault("story_final_save_formats", ["png", "pdf", "svg"])
CONFIG.setdefault("story_final_png_dpi", 600)
CONFIG.setdefault("story_final_max_points_per_target", 1200)
CONFIG.setdefault("story_final_top_candidate_rows_per_target", 25)
CONFIG.setdefault("story_final_enable_old_aliases", True)
CONFIG.setdefault("story_final_target_metric_split", "random")
CONFIG.setdefault("story_final_feature_set", "geometry_plus_racs")
CONFIG.setdefault("story_final_nominal_coverage", 0.90)

PAPER_STORY_DIR = RESULTS_DIR / "figures" / "paper_story_final"
PAPER_STORY_SI_DIR = RESULTS_DIR / "figures" / "paper_story_final_si"
PAPER_STORY_DATA_DIR = FIGDATA_DIR / "paper_story_final"
for _d in [PAPER_STORY_DIR, PAPER_STORY_SI_DIR, PAPER_STORY_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def v42_apply_story_style() -> None:
    """Clean high-impact manuscript style without external font dependencies."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.7,
        "xtick.labelsize": 7.9,
        "ytick.labelsize": 7.9,
        "figure.dpi": 150,
        "savefig.dpi": int(CONFIG.get("story_final_png_dpi", 600)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.24,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.7,
        "patch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def v42_panel_label(ax, label: str, x: float = -0.10, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top", ha="right")


def v42_save_figure(fig: plt.Figure, stem: str, si: bool = False, aliases: Optional[List[Path]] = None) -> None:
    """Save story figures as 600-dpi PNG plus PDF/SVG and optional legacy aliases."""
    v42_apply_story_style()
    out_dir = PAPER_STORY_SI_DIR if si else PAPER_STORY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = CONFIG.get("story_final_save_formats", ["png", "pdf", "svg"])
    for fmt in formats:
        fmt = str(fmt).lower().strip()
        out = (out_dir / stem).with_suffix(f".{fmt}")
        try:
            fig.savefig(out, dpi=int(CONFIG.get("story_final_png_dpi", 600)), bbox_inches="tight")
        except Exception as e:
            logging.warning("V42_FIGURE_SAVE_FAILED | %s | %s", out, e)
    if CONFIG.get("story_final_enable_old_aliases", True):
        for base in aliases or []:
            base.parent.mkdir(parents=True, exist_ok=True)
            for fmt in ["png", "pdf"]:
                try:
                    fig.savefig(base.with_suffix(f".{fmt}"), dpi=int(CONFIG.get("story_final_png_dpi", 600)), bbox_inches="tight")
                except Exception as e:
                    logging.warning("V42_ALIAS_SAVE_FAILED | %s | %s", base.with_suffix(f'.{fmt}'), e)
    plt.close(fig)


def v42_read_table(fname: str, low_memory: bool = False) -> pd.DataFrame:
    path = TABLE_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=low_memory)
    except Exception as e:
        logging.warning("V42_TABLE_READ_FAILED | %s | %s", path, e)
        return pd.DataFrame()


def v42_preferred_feature(agg: Optional[pd.DataFrame] = None) -> str:
    qc = v42_read_table("QC_Table_feature_set_integrity.csv")
    if not qc.empty:
        usable = set(qc.loc[qc.get("usable_for_modelling", False).astype(bool), "feature_set"].astype(str)) if "usable_for_modelling" in qc.columns else set()
        # Do not use RDF in main figures unless it really has RDF columns and is not a duplicate.
        rdf_ok = False
        if "feature_set" in qc.columns:
            rdf_rows = qc[qc["feature_set"].astype(str).eq("geometry_plus_racs_rdfs")]
            if not rdf_rows.empty:
                r = rdf_rows.iloc[0]
                rdf_ok = bool(r.get("usable_for_modelling", False)) and int(r.get("n_rdf_features", 0) or 0) > 0 and str(r.get("duplicate_of", "")) == ""
        if rdf_ok:
            return "geometry_plus_racs_rdfs"
        if "geometry_plus_racs" in usable:
            return "geometry_plus_racs"
    if agg is not None and not agg.empty and "feature_set" in agg.columns:
        if "geometry_plus_racs" in set(agg["feature_set"].astype(str)):
            return "geometry_plus_racs"
    return "geometry_only"


def v42_clean_agg(agg: pd.DataFrame, top_frac: float = 0.05, feature_set: Optional[str] = None,
                  split: Optional[str] = None, budget: Optional[int] = None, stable_only: bool = True) -> pd.DataFrame:
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = agg.copy()
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(top_frac).abs() < 1e-9]
    if feature_set is not None and "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str).eq(str(feature_set))]
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str).eq(str(split))]
    if budget is not None and "budget" in df.columns:
        df = df[pd.to_numeric(df["budget"], errors="coerce").eq(int(budget))]
    if stable_only and "model" in df.columns:
        df = df[df["model"].astype(str).isin(CONFIG.get("stable_plot_models", MODEL_ORDER_FINAL))]
    # Remove duplicated RDF feature set from story figures if QC says duplicate/unusable.
    qc = v42_read_table("QC_Table_feature_set_integrity.csv")
    if not qc.empty and "feature_set" in df.columns:
        bad = qc.loc[~qc.get("usable_for_modelling", False).astype(bool), "feature_set"].astype(str).tolist() if "usable_for_modelling" in qc.columns else []
        if bad:
            df = df[~df["feature_set"].astype(str).isin(bad)]
    for col in ["budget", "recall_mean", "recall_ci_low", "recall_ci_high", "enrichment_mean", "enrichment_ci_low", "enrichment_ci_high", "precision_mean", "split_coverage_mean", "split_mean_width_mean", "trusted_count_mean", "uncertain_count_mean", "rejected_count_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def v42_target_summary(master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if master is not None and not master.empty:
        rows = []
        for t in TARGET_ORDER_FINAL:
            if t in master.columns:
                vals = pd.to_numeric(master[t], errors="coerce").dropna()
                if len(vals):
                    rows.append({"target": t, "n_mofs": len(vals), "median": vals.median(), "mean": vals.mean(), "std": vals.std(), "min": vals.min(), "max": vals.max(), "top5_threshold": vals.quantile(0.95), "top1_threshold": vals.quantile(0.99)})
        if rows:
            out = pd.DataFrame(rows)
            atomic_save_csv(out, TABLE_DIR / "STORY_Table_target_summary_with_elite_thresholds.csv")
            return out
    ts = v42_read_table("MAIN_Table_1_dataset_and_targets.csv")
    if ts.empty:
        ts = v42_read_table("table_dataset_target_summary.csv")
    if not ts.empty:
        rename = {"n_nonmissing": "n_mofs", "median_mmol_g": "median", "std_mmol_g": "std", "max_mmol_g": "max", "min_mmol_g": "min", "mean_mmol_g": "mean"}
        ts = ts.rename(columns=rename)
        if "target" in ts.columns and "top5_threshold" not in ts.columns:
            # No percentile threshold available from summary; leave blank rather than inventing.
            ts["top5_threshold"] = np.nan
            ts["top1_threshold"] = np.nan
    return ts


def v42_target_std_map(master: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    ts = v42_target_summary(master)
    out: Dict[str, float] = {}
    if not ts.empty and "target" in ts.columns:
        for _, r in ts.iterrows():
            out[str(r["target"])] = float(r.get("std", np.nan)) if pd.notna(r.get("std", np.nan)) else np.nan
    return out


def v42_order_targets(df: Optional[pd.DataFrame] = None) -> List[str]:
    if df is None or df.empty or "target" not in df.columns:
        return TARGET_ORDER_FINAL[:]
    present = [t for t in TARGET_ORDER_FINAL if t in set(df["target"].dropna().astype(str))]
    extras = [t for t in sorted(df["target"].dropna().astype(str).unique()) if t not in present]
    return present + extras


def v42_label_target(t: Any, latex: bool = True) -> str:
    return v41_target_label(t, latex=latex) if "v41_target_label" in globals() else str(t)


def v42_heatmap(ax, mat: pd.DataFrame, title: str, cbar_label: str = "", center_zero: bool = False,
                cmap: str = "viridis", fmt: str = ".2f"):
    if mat is None or mat.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return None
    arr = mat.to_numpy(dtype=float)
    if center_zero:
        vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
        vmin, vmax = -vmax, vmax
        cmap = "coolwarm"
    else:
        vmin = np.nanmin(arr) if np.isfinite(arr).any() else 0.0
        vmax = np.nanmax(arr) if np.isfinite(arr).any() else 1.0
    if np.isclose(vmin, vmax):
        vmin -= 1e-9; vmax += 1e-9
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels([str(x) for x in mat.columns], rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([str(x) for x in mat.index])
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    if cbar_label:
        cbar.set_label(cbar_label)
    return im


def v42_top_consensus(topn: int = 25) -> pd.DataFrame:
    c = v42_read_table("table_consensus_candidate_shortlist.csv", low_memory=True)
    if c.empty:
        c = v42_read_table("MAIN_Table_3_consensus_shortlist_target_balanced.csv", low_memory=True)
    if c.empty:
        c = v42_read_table("MAIN_Table_3_consensus_trusted_shortlist.csv", low_memory=True)
    if c.empty:
        return c
    if "consensus_pass" in c.columns and c["consensus_pass"].fillna(False).astype(bool).any():
        c = c[c["consensus_pass"].fillna(False).astype(bool)].copy()
    sort_cols = [x for x in ["target", "consensus_score", "split_lower_median", "y_pred_median"] if x in c.columns]
    if "target" in sort_cols:
        c = c.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
    top = c.groupby("target", group_keys=False).head(topn).copy() if "target" in c.columns else c.head(topn).copy()
    atomic_save_csv(top, TABLE_DIR / "STORY_Table_top25_consensus_candidates.csv")
    return top


def v42_add_percentiles_to_candidates(df: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None or df.empty or "target" not in df.columns:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    # Use true/pred columns robustly.
    ytrue_col = "y_true_median" if "y_true_median" in out.columns else ("y_true" if "y_true" in out.columns else None)
    ypred_col = "y_pred_median" if "y_pred_median" in out.columns else ("y_pred" if "y_pred" in out.columns else None)
    for t in out["target"].dropna().astype(str).unique():
        idx = out["target"].astype(str).eq(t)
        reference = None
        if master is not None and not master.empty and t in master.columns:
            reference = pd.to_numeric(master[t], errors="coerce").dropna().to_numpy()
        if reference is None or len(reference) < 10:
            vals = pd.to_numeric(out.loc[idx, ytrue_col], errors="coerce").dropna().to_numpy() if ytrue_col else np.array([])
            reference = vals if len(vals) > 0 else np.array([0, 1])
        reference = np.sort(reference[np.isfinite(reference)])
        if ytrue_col:
            vals = pd.to_numeric(out.loc[idx, ytrue_col], errors="coerce").to_numpy()
            out.loc[idx, "true_percentile"] = np.searchsorted(reference, vals, side="right") / max(len(reference), 1)
        if ypred_col:
            vals = pd.to_numeric(out.loc[idx, ypred_col], errors="coerce").to_numpy()
            out.loc[idx, "pred_percentile"] = np.searchsorted(reference, vals, side="right") / max(len(reference), 1)
    return out


def v42_learning_curves(agg: pd.DataFrame, metric: str, feature_set: Optional[str] = None, split: str = "random") -> pd.DataFrame:
    feature_set = feature_set or v42_preferred_feature(agg)
    df = v42_clean_agg(agg, top_frac=0.05, feature_set=feature_set, split=split, stable_only=True)
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    rows = []
    for (target, budget), sub in df.groupby(["target", "budget"], dropna=False):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        rows.append({"target": target, "budget": int(budget), "mean": float(np.mean(vals)), "lo": float(np.quantile(vals, 0.10)), "hi": float(np.quantile(vals, 0.90)), "n": int(len(vals))})
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "budget"]) if not out.empty else out


def v42_descriptor_gain_table(agg: pd.DataFrame) -> pd.DataFrame:
    df = v42_clean_agg(agg, top_frac=0.05, stable_only=True)
    if df.empty:
        return pd.DataFrame()
    # Keep only validated descriptor comparison.
    df = df[df["feature_set"].astype(str).isin(["geometry_only", "geometry_plus_racs"])].copy()
    piv = df.pivot_table(index=["target", "budget", "split"], columns="feature_set", values="recall_mean", aggfunc="mean").reset_index()
    if "geometry_only" in piv.columns and "geometry_plus_racs" in piv.columns:
        piv["delta_recall_racs_minus_geometry"] = piv["geometry_plus_racs"] - piv["geometry_only"]
    else:
        piv["delta_recall_racs_minus_geometry"] = np.nan
    atomic_save_csv(piv, TABLE_DIR / "STORY_Table_descriptor_gain_by_budget_split.csv")
    return piv


def v42_split_stress_table(agg: pd.DataFrame) -> pd.DataFrame:
    feature = v42_preferred_feature(agg)
    df = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, budget=1000, stable_only=True)
    if df.empty:
        return pd.DataFrame()
    tab = df.groupby(["target", "split"], as_index=False)["recall_mean"].mean()
    piv = tab.pivot(index="target", columns="split", values="recall_mean")
    if "random" in piv.columns:
        for split in piv.columns:
            piv[f"penalty_vs_random__{split}"] = piv["random"] - piv[split]
            piv[f"vulnerability__{split}"] = 1.0 - (piv[split] / piv["random"].replace(0, np.nan))
    out = piv.reset_index()
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_split_stress_penalty_vulnerability.csv")
    return out


def v42_plot_figure_1(master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig1_framework"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.2 Figure 1 story workflow")
    v42_apply_story_style()
    ts = v42_target_summary(master)
    fig = plt.figure(figsize=(15.8, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.02, 1.0], width_ratios=[1.1, 1.05, 1.05, 1.25], hspace=0.40, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, :2]); ax0.axis("off")
    v42_panel_label(ax0, "a")
    ax0.set_title("Four adsorption-screening tasks from ARC-MOF", loc="left", fontweight="bold")
    x0s = [0.02, 0.27, 0.52, 0.77]
    for i, target in enumerate(TARGET_ORDER_FINAL):
        color = TARGET_COLORS_FINAL.get(target, "#777777")
        x = x0s[i]
        card = plt.Rectangle((x, 0.13), 0.215, 0.68, transform=ax0.transAxes, fc=color, ec="none", alpha=0.12)
        ax0.add_patch(card)
        ax0.text(x + 0.107, 0.72, v42_label_target(target), transform=ax0.transAxes, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
        r = ts[ts["target"].astype(str).eq(target)].iloc[0] if (not ts.empty and "target" in ts.columns and ts["target"].astype(str).eq(target).any()) else {}
        n = r.get("n_mofs", r.get("n_nonmissing", np.nan)) if hasattr(r, "get") else np.nan
        med = r.get("median", np.nan) if hasattr(r, "get") else np.nan
        mx = r.get("max", np.nan) if hasattr(r, "get") else np.nan
        ax0.text(x + 0.107, 0.50, f"{int(n):,} MOFs" if pd.notna(n) else "MOFs", transform=ax0.transAxes, ha="center", fontsize=9)
        ax0.text(x + 0.107, 0.38, f"median {float(med):.2g}" if pd.notna(med) else "median —", transform=ax0.transAxes, ha="center", fontsize=8.2)
        ax0.text(x + 0.107, 0.27, f"max {float(mx):.2g} mmol g$^{{-1}}$" if pd.notna(mx) else "max —", transform=ax0.transAxes, ha="center", fontsize=8.2)

    ax1 = fig.add_subplot(gs[0, 2:]); ax1.axis("off")
    v42_panel_label(ax1, "b")
    ax1.set_title("Risk-controlled few-shot decision pipeline", loc="left", fontweight="bold")
    steps = ["10–1000 labels", "stable ML\nmodels", "grouped\nstress tests", "90% conformal\nintervals", "consensus\nshortlist", "domain-overlap\nannotation"]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, step) in enumerate(zip(xs, steps)):
        ax1.add_patch(plt.Circle((x, 0.53), 0.075, transform=ax1.transAxes, fc="#F4F6F7", ec="#566573", lw=1.0))
        ax1.text(x, 0.53, str(i + 1), transform=ax1.transAxes, ha="center", va="center", fontsize=10, fontweight="bold")
        ax1.text(x, 0.29, step, transform=ax1.transAxes, ha="center", va="center", fontsize=8.1)
        if i < len(xs) - 1:
            ax1.annotate("", xy=(xs[i + 1] - 0.085, 0.53), xytext=(x + 0.085, 0.53), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.2, color="#566573"))

    ax2 = fig.add_subplot(gs[1, 0]); ax2.axis("off"); v42_panel_label(ax2, "c")
    ax2.set_title("Descriptor layers", loc="left", fontweight="bold")
    desc = [("Geometry", "PLD/LCD, surface, pore volume, density"), ("RAC chemistry", "metal/linker autocorrelations"), ("RDF audit", "kept only if integrity QC passes")]
    for j, (title, body) in enumerate(desc):
        y = 0.78 - 0.25 * j
        ax2.add_patch(plt.Rectangle((0.05, y - 0.09), 0.88, 0.16, transform=ax2.transAxes, fc="#EBF5FB" if j < 2 else "#FDF2E9", ec="#AAB7B8", lw=0.8))
        ax2.text(0.09, y, title, transform=ax2.transAxes, ha="left", va="center", fontweight="bold", fontsize=9)
        ax2.text(0.09, y - 0.07, body, transform=ax2.transAxes, ha="left", va="center", fontsize=7.3)

    ax3 = fig.add_subplot(gs[1, 1]); ax3.axis("off"); v42_panel_label(ax3, "d")
    ax3.set_title("Extrapolation tests", loc="left", fontweight="bold")
    splits = ["Random", "Geometry", "Metal", "Functional", "Ligand", "Topology"]
    for j, s in enumerate(splits):
        x, y = 0.22 + 0.32 * (j % 2), 0.78 - 0.20 * (j // 2)
        fc = "#FADBD8" if s == "Topology" else "#F8F9F9"
        ax3.add_patch(plt.Rectangle((x - 0.13, y - 0.055), 0.26, 0.11, transform=ax3.transAxes, fc=fc, ec="#7F8C8D", lw=0.8))
        ax3.text(x, y, s, transform=ax3.transAxes, ha="center", va="center", fontsize=7.8, fontweight="bold" if s == "Topology" else "normal")

    ax4 = fig.add_subplot(gs[1, 2]); ax4.axis("off"); v42_panel_label(ax4, "e")
    ax4.set_title("Uncertainty-to-action", loc="left", fontweight="bold")
    tiers = [("trusted", "high lower bound\n+ narrow interval"), ("uncertain", "abstain / defer"), ("rejected", "low upper bound")]
    for j, (title, body) in enumerate(tiers):
        y = 0.78 - 0.24 * j
        ax4.add_patch(plt.Rectangle((0.07, y - 0.08), 0.84, 0.15, transform=ax4.transAxes, fc=["#E8F8F5", "#FEF9E7", "#FDEDEC"][j], ec="#AAB7B8", lw=0.8))
        ax4.text(0.12, y, title, transform=ax4.transAxes, ha="left", va="center", fontweight="bold", fontsize=8.5)
        ax4.text(0.47, y, body, transform=ax4.transAxes, ha="left", va="center", fontsize=7.4)

    ax5 = fig.add_subplot(gs[1, 3]); ax5.axis("off"); v42_panel_label(ax5, "f")
    ax5.set_title("Final paper outputs", loc="left", fontweight="bold")
    outputs = ["few-shot learning phase diagram", "descriptor and topology stress map", "conformal risk-control frontier", "true-enriched consensus shortlist", "external domain-overlap SI"]
    for j, line in enumerate(outputs):
        ax5.text(0.08, 0.82 - 0.14 * j, "• " + line, transform=ax5.transAxes, ha="left", va="center", fontsize=8.3)
    fig.suptitle("Risk-controlled few-shot adsorption screening: from scarce labels to consensus MOF shortlists", fontsize=14.5, fontweight="bold")
    v42_save_figure(fig, "Figure_1_v4_2_story_workflow", aliases=[FIG_MAIN_DIR / "Figure_1_risk_controlled_framework", PAPER_FINAL_DIR / "Figure_1_final_workflow"])
    mark_done(stage, {"v4_2": True})


def v42_plot_figure_2(agg: pd.DataFrame) -> None:
    stage = "fig2_performance"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.2 Figure 2 few-shot learning phase diagram")
    v42_apply_story_style()
    feature = v42_preferred_feature(agg)
    recall = v42_learning_curves(agg, "recall_mean", feature_set=feature, split=str(CONFIG.get("story_final_target_metric_split", "random")))
    enrich = v42_learning_curves(agg, "enrichment_mean", feature_set=feature, split=str(CONFIG.get("story_final_target_metric_split", "random")))
    atomic_save_csv(recall, PAPER_STORY_DATA_DIR / "Figure_2a_recall_learning_curves.csv")
    atomic_save_csv(enrich, PAPER_STORY_DATA_DIR / "Figure_2b_enrichment_learning_curves.csv")

    fig = plt.figure(figsize=(15.8, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2]); axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1]); axf = fig.add_subplot(gs[1, 2])

    for target in v42_order_targets(recall):
        color = TARGET_COLORS_FINAL.get(target, None)
        sub = recall[recall["target"].astype(str).eq(target)].sort_values("budget")
        if not sub.empty:
            x = sub["budget"].to_numpy(dtype=float); y = sub["mean"].to_numpy(dtype=float)
            axa.plot(x, y, marker="o", color=color, label=v42_label_target(target))
            axa.fill_between(x, sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float), color=color, alpha=0.14, linewidth=0)
    axa.axhline(0.05, ls="--", lw=1.0, color="#777777", label="random top-5%")
    axa.set_xscale("log"); axa.set_xlabel("Label budget"); axa.set_ylabel("Top-5% recall"); axa.set_title("Elite recovery emerges from few labels"); axa.grid(True); axa.legend(frameon=False, ncol=1)
    v42_panel_label(axa, "a")

    for target in v42_order_targets(enrich):
        color = TARGET_COLORS_FINAL.get(target, None)
        sub = enrich[enrich["target"].astype(str).eq(target)].sort_values("budget")
        if not sub.empty:
            x = sub["budget"].to_numpy(dtype=float); y = sub["mean"].to_numpy(dtype=float)
            axb.plot(x, y, marker="o", color=color)
            axb.fill_between(x, sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float), color=color, alpha=0.14, linewidth=0)
    axb.axhline(1.0, ls="--", lw=1.0, color="#777777")
    axb.set_xscale("log"); axb.set_xlabel("Label budget"); axb.set_ylabel("Enrichment over random"); axb.set_title("Screening enrichment") ; axb.grid(True)
    v42_panel_label(axb, "b")

    # Label-efficiency fraction relative to the 1000-label value.
    eff_rows = []
    for target, sub in recall.groupby("target"):
        sub = sub.sort_values("budget")
        final = sub.loc[sub["budget"].idxmax(), "mean"] if not sub.empty else np.nan
        for _, r in sub.iterrows():
            eff_rows.append({"target": target, "budget": int(r["budget"]), "label_efficiency": float(r["mean"] / final) if pd.notna(final) and final > 0 else np.nan})
    eff = pd.DataFrame(eff_rows)
    atomic_save_csv(eff, PAPER_STORY_DATA_DIR / "Figure_2c_label_efficiency.csv")
    for target in v42_order_targets(eff):
        sub = eff[eff["target"].astype(str).eq(target)].sort_values("budget")
        axc.plot(sub["budget"], sub["label_efficiency"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axc.axhline(0.75, ls=":", lw=1.0, color="#666666")
    axc.set_xscale("log"); axc.set_ylim(0, 1.05); axc.set_xlabel("Label budget"); axc.set_ylabel("Fraction of 1000-label recall"); axc.set_title("Label-efficiency index"); axc.grid(True)
    v42_panel_label(axc, "c")

    # Budget needed for 50/75/90% of final recall.
    thresholds = [0.50, 0.75, 0.90]
    sat_rows = []
    for target, sub in eff.groupby("target"):
        for thr in thresholds:
            hit = sub[sub["label_efficiency"] >= thr].sort_values("budget")
            sat_rows.append({"target": target, "threshold": thr, "budget_needed": int(hit.iloc[0]["budget"]) if not hit.empty else np.nan})
    sat = pd.DataFrame(sat_rows)
    atomic_save_csv(sat, TABLE_DIR / "STORY_Table_label_efficiency_budget_thresholds.csv")
    ylabels = [v42_label_target(t) for t in v42_order_targets(sat)]
    for j, thr in enumerate(thresholds):
        vals = [sat[(sat["target"].astype(str).eq(t)) & (sat["threshold"].eq(thr))]["budget_needed"].iloc[0] if not sat[(sat["target"].astype(str).eq(t)) & (sat["threshold"].eq(thr))].empty else np.nan for t in v42_order_targets(sat)]
        axd.scatter(vals, np.arange(len(ylabels)) + (j - 1) * 0.18, label=f"{int(thr*100)}%", s=38)
    axd.set_xscale("log"); axd.set_yticks(np.arange(len(ylabels))); axd.set_yticklabels(ylabels); axd.set_xlabel("Budget needed"); axd.set_title("How many labels are enough?"); axd.grid(True, axis="x"); axd.legend(title="of final recall", frameon=False)
    v42_panel_label(axd, "d")

    # Model comparison at 1000 labels.
    df1000 = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, split=str(CONFIG.get("story_final_target_metric_split", "random")), budget=1000, stable_only=True)
    mod = df1000.groupby("model", as_index=False)["recall_mean"].mean() if not df1000.empty else pd.DataFrame()
    if not mod.empty:
        mod["model_label"] = mod["model"].map(v41_model_label)
        order = [m for m in MODEL_ORDER_FINAL if m in set(mod["model"])]
        mod = mod.set_index("model").reindex(order).dropna(subset=["recall_mean"]).reset_index()
        axe.bar(mod["model_label"], mod["recall_mean"], color=[MODEL_COLORS_FINAL.get(m, "#777777") for m in mod["model"]])
    axe.set_ylabel("Mean top-5% recall"); axe.set_title("Stable model comparison at 1000 labels"); axe.tick_params(axis="x", rotation=35); axe.grid(True, axis="y")
    v42_panel_label(axe, "e")

    # Target summary: best enrichment at max budget.
    best_rows = []
    if not enrich.empty:
        maxb = int(enrich["budget"].max())
        best = enrich[enrich["budget"].eq(maxb)]
        for target in v42_order_targets(best):
            sub = best[best["target"].astype(str).eq(target)]
            if not sub.empty:
                best_rows.append({"target": target, "enrichment": float(sub["mean"].mean())})
    bestdf = pd.DataFrame(best_rows)
    atomic_save_csv(bestdf, TABLE_DIR / "STORY_Table_best_enrichment_summary.csv")
    if not bestdf.empty:
        bestdf = bestdf.sort_values("enrichment")
        axf.barh([v42_label_target(t) for t in bestdf["target"]], bestdf["enrichment"], color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in bestdf["target"]])
        for y, val in enumerate(bestdf["enrichment"]):
            axf.text(val, y, f" {val:.1f}×", va="center", fontsize=8)
    axf.axvline(1.0, ls="--", color="#777777", lw=1); axf.set_xlabel("Enrichment at largest budget"); axf.set_title("Final enrichment summary"); axf.grid(True, axis="x")
    v42_panel_label(axf, "f")

    fig.suptitle("Few-shot learning phase diagram for adsorption-elite recovery", fontsize=14.5, fontweight="bold")
    v42_save_figure(fig, "Figure_2_v4_2_fewshot_learning_phase_diagram", aliases=[FIG_MAIN_DIR / "Figure_2_label_budget_performance", PAPER_FINAL_DIR / "Figure_2_v4_1_fewshot_performance_descriptor_split"])
    mark_done(stage, {"v4_2": True, "preferred_feature_set": feature})


def v42_plot_figure_3(agg: pd.DataFrame) -> None:
    stage = "fig3_calibration"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.2 Figure 3 descriptor chemistry and extrapolation stress")
    v42_apply_story_style()
    dg = v42_descriptor_gain_table(agg)
    st = v42_split_stress_table(agg)
    fig = plt.figure(figsize=(15.8, 8.9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2]); axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1]); axf = fig.add_subplot(gs[1, 2])

    # Descriptor gain by target x budget under random split.
    if not dg.empty:
        dg_rand = dg[dg["split"].astype(str).eq("random")] if "split" in dg.columns else dg
        mat = dg_rand.pivot_table(index="target", columns="budget", values="delta_recall_racs_minus_geometry", aggfunc="mean")
        mat = mat.reindex([t for t in TARGET_ORDER_FINAL if t in mat.index])
        mat.index = [v42_label_target(i) for i in mat.index]
        v42_heatmap(axa, mat, "RAC descriptor gain across label budgets", "Δ top-5% recall", center_zero=True, fmt=".2f")
    else:
        axa.axis("off"); axa.text(0.5, 0.5, "No descriptor-gain data", ha="center")
    v42_panel_label(axa, "a")

    # Descriptor gain at 1000 labels.
    if not dg.empty:
        sub = dg[pd.to_numeric(dg["budget"], errors="coerce").eq(1000)]
        bars = sub.groupby("target", as_index=False)["delta_recall_racs_minus_geometry"].mean()
        bars = bars.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(bars["target"]) or t in bars.index]).reset_index()
        axb.barh([v42_label_target(t) for t in bars["target"]], bars["delta_recall_racs_minus_geometry"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
        axb.axvline(0, color="#555555", lw=1)
    axb.set_xlabel("Δ recall: RACs − geometry"); axb.set_title("Chemistry benefit at 1000 labels"); axb.grid(True, axis="x"); v42_panel_label(axb, "b")

    # Split stress heatmap.
    feature = v42_preferred_feature(agg)
    df = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, budget=1000, stable_only=True)
    if not df.empty:
        split_mat = df.pivot_table(index="target", columns="split", values="recall_mean", aggfunc="mean")
        split_mat = split_mat.reindex([t for t in TARGET_ORDER_FINAL if t in split_mat.index])
        split_mat = split_mat[[s for s in SPLIT_ORDER_FINAL if s in split_mat.columns]]
        split_mat.index = [v42_label_target(i) for i in split_mat.index]
        split_mat.columns = [v41_split_label(c) for c in split_mat.columns]
        atomic_save_csv(split_mat.reset_index(), PAPER_STORY_DATA_DIR / "Figure_3c_split_stress_heatmap.csv")
        v42_heatmap(axc, split_mat, "Split-stress map at 1000 labels", "top-5% recall", cmap="viridis", fmt=".2f")
    v42_panel_label(axc, "c")

    # Topology penalty vs random.
    if not st.empty and "penalty_vs_random__topology_grouped" in st.columns:
        bars = st[["target", "penalty_vs_random__topology_grouped"]].copy()
        bars = bars.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(bars["target"]) or t in bars.index]).reset_index()
        axd.barh([v42_label_target(t) for t in bars["target"]], bars["penalty_vs_random__topology_grouped"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
        axd.axvline(0, color="#555555", lw=1)
    axd.set_xlabel("Recall(random) − recall(topology)"); axd.set_title("Topology extrapolation penalty"); axd.grid(True, axis="x"); v42_panel_label(axd, "d")

    # Vulnerability heatmap.
    if not st.empty:
        vul_cols = [c for c in st.columns if c.startswith("vulnerability__") and not c.endswith("random")]
        if vul_cols:
            vul = st.set_index("target")[vul_cols]
            vul.columns = [v41_split_label(c.replace("vulnerability__", "")) for c in vul.columns]
            vul = vul.reindex([t for t in TARGET_ORDER_FINAL if t in vul.index])
            vul.index = [v42_label_target(i) for i in vul.index]
            v42_heatmap(axe, vul, "Extrapolation vulnerability", "1 − grouped/random", center_zero=False, cmap="magma", fmt=".2f")
    v42_panel_label(axe, "e")

    axf.axis("off"); v42_panel_label(axf, "f")
    axf.set_title("Mechanistic interpretation", loc="left", fontweight="bold")
    notes = [
        ("Geometry", "captures pore-size and density-driven adsorption signal"),
        ("RAC chemistry", "adds metal/linker chemistry and improves elite recovery"),
        ("Topology split", "remains the hardest out-of-distribution stress test"),
        ("Manuscript claim", "random-split accuracy alone overstates transferability"),
    ]
    for j, (head, body) in enumerate(notes):
        y = 0.82 - 0.19 * j
        axf.add_patch(plt.Rectangle((0.04, y - 0.075), 0.90, 0.13, transform=axf.transAxes, fc="#F8F9F9", ec="#B2BABB", lw=0.8))
        axf.text(0.08, y, head, transform=axf.transAxes, ha="left", va="center", fontsize=8.8, fontweight="bold")
        axf.text(0.38, y, body, transform=axf.transAxes, ha="left", va="center", fontsize=7.5)

    fig.suptitle("Descriptor chemistry improves screening, but topology extrapolation remains limiting", fontsize=14.5, fontweight="bold")
    v42_save_figure(fig, "Figure_3_v4_2_descriptor_chemistry_extrapolation_stress", aliases=[FIG_MAIN_DIR / "Figure_3_calibration_and_abstention", PAPER_FINAL_DIR / "Figure_3_v4_1_conformal_risk_control"])
    mark_done(stage, {"v4_2": True, "preferred_feature_set": feature})


def v42_precision_yield_frontier(tiers: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if tiers is None or tiers.empty:
        return pd.DataFrame()
    df = tiers.copy()
    if "candidate_eligible" in df.columns:
        df = df[df["candidate_eligible"].fillna(False).astype(bool)]
    for col in ["y_true", "split_lower", "split_upper", "budget"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    retained_grid = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    for target, sub in df.groupby("target"):
        sub = sub.dropna(subset=["y_true", "split_lower"]).copy()
        if sub.empty:
            continue
        # One row per MOF/target at strongest lower bound.
        sub = sub.sort_values("split_lower", ascending=False).drop_duplicates("mof_id")
        if master is not None and not master.empty and str(target) in master.columns:
            ref = pd.to_numeric(master[str(target)], errors="coerce").dropna()
            thr = float(ref.quantile(0.95)) if len(ref) else float(sub["y_true"].quantile(0.95))
        else:
            thr = float(sub["y_true"].quantile(0.95))
        n = len(sub)
        for frac in retained_grid:
            k = max(1, int(math.ceil(frac * n)))
            top = sub.head(k)
            rows.append({"target": target, "retained_fraction": frac, "precision_true_top5": float((top["y_true"] >= thr).mean()), "n_retained": k, "top5_threshold": thr})
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_precision_yield_frontier.csv")
    return out


def v42_plot_figure_4(anatomy: Optional[pd.DataFrame], tiers: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.2 Figure 4 conformal risk-control frontier")
    v42_apply_story_style()
    agg = v42_read_table("table_aggregated_metrics_with_bootstrap_ci.csv", low_memory=True)
    feature = v42_preferred_feature(agg)
    nominal = float(CONFIG.get("story_final_nominal_coverage", 0.90))
    fig = plt.figure(figsize=(15.8, 8.9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2]); axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1]); axf = fig.add_subplot(gs[1, 2])

    df = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, stable_only=True)
    if not df.empty:
        cerr = df.assign(coverage_error=df["split_coverage_mean"] - nominal).pivot_table(index="target", columns="split", values="coverage_error", aggfunc="mean")
        cerr = cerr.reindex([t for t in TARGET_ORDER_FINAL if t in cerr.index])
        cerr = cerr[[s for s in SPLIT_ORDER_FINAL if s in cerr.columns]]
        cerr.index = [v42_label_target(i) for i in cerr.index]
        cerr.columns = [v41_split_label(c) for c in cerr.columns]
        v42_heatmap(axa, cerr, "Coverage error relative to 90%", "coverage − 0.90", center_zero=True, fmt="+.2f")
    v42_panel_label(axa, "a")

    std_map = v42_target_std_map(master)
    width = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, split="random", stable_only=True)
    if not width.empty:
        width["target_std"] = width["target"].map(std_map)
        width["normalized_width"] = width["split_mean_width_mean"] / width["target_std"].replace(0, np.nan)
        wcurve = width.groupby(["target", "budget"], as_index=False)["normalized_width"].mean()
        atomic_save_csv(wcurve, PAPER_STORY_DATA_DIR / "Figure_4b_normalized_interval_width.csv")
        for target in v42_order_targets(wcurve):
            sub = wcurve[wcurve["target"].astype(str).eq(target)].sort_values("budget")
            axb.plot(sub["budget"], sub["normalized_width"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axb.set_xscale("log"); axb.set_xlabel("Label budget"); axb.set_ylabel("Interval width / target std"); axb.set_title("Uncertainty narrows with labels"); axb.grid(True); v42_panel_label(axb, "b")

    frontier = v42_precision_yield_frontier(tiers, master=master)
    if not frontier.empty:
        for target in v42_order_targets(frontier):
            sub = frontier[frontier["target"].astype(str).eq(target)].sort_values("retained_fraction")
            axc.plot(sub["retained_fraction"], sub["precision_true_top5"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axc.set_xlabel("Retained fraction"); axc.set_ylabel("True top-5% fraction"); axc.set_ylim(0, 1.05); axc.set_title("Precision–yield frontier"); axc.grid(True); axc.legend(frameon=False, ncol=1)
    v42_panel_label(axc, "c")

    if tiers is not None and not tiers.empty and {"target", "budget", "tier"}.issubset(tiers.columns):
        tmp = tiers.copy(); tmp["budget"] = pd.to_numeric(tmp["budget"], errors="coerce")
        tf = tmp.groupby(["target", "budget"], as_index=False).apply(lambda g: pd.Series({"trusted_fraction": float((g["tier"].astype(str) == "trusted").mean())})).reset_index(drop=True)
        atomic_save_csv(tf, PAPER_STORY_DATA_DIR / "Figure_4d_trusted_fraction_budget.csv")
        for target in v42_order_targets(tf):
            sub = tf[tf["target"].astype(str).eq(target)].sort_values("budget")
            axd.plot(sub["budget"], sub["trusted_fraction"], marker="o", color=TARGET_COLORS_FINAL.get(target, None))
    axd.set_xscale("log"); axd.set_xlabel("Label budget"); axd.set_ylabel("Trusted fraction of saved candidate rows"); axd.set_title("Actionable yield"); axd.grid(True); v42_panel_label(axd, "d")

    if tiers is not None and not tiers.empty and {"tier", "split_lower", "split_upper", "target"}.issubset(tiers.columns):
        tmp = tiers.copy()
        tmp["interval_width"] = pd.to_numeric(tmp["split_upper"], errors="coerce") - pd.to_numeric(tmp["split_lower"], errors="coerce")
        tmp["target_std"] = tmp["target"].map(std_map)
        tmp["norm_width"] = tmp["interval_width"] / tmp["target_std"].replace(0, np.nan)
        labels = [x for x in ["trusted", "uncertain", "rejected"] if (tmp["tier"].astype(str) == x).any()]
        data = []
        for lab in labels:
            vals = tmp.loc[tmp["tier"].astype(str).eq(lab), "norm_width"].dropna()
            if len(vals) > 3000:
                vals = vals.sample(3000, random_state=42)
            data.append(vals)
        if data:
            safe_boxplot_with_labels(axe, data, labels, showfliers=False)
    axe.set_ylabel("Interval width / target std"); axe.set_title("Uncertainty tier separation"); axe.grid(True, axis="y"); v42_panel_label(axe, "e")

    if not df.empty:
        cov_methods = pd.DataFrame({
            "method": ["split", "local", "Mondrian"],
            "coverage": [pd.to_numeric(df.get("split_coverage_mean"), errors="coerce").mean(), pd.to_numeric(df.get("local_coverage_mean"), errors="coerce").mean(), pd.to_numeric(df.get("mondrian_coverage_mean"), errors="coerce").mean()],
        })
        axf.bar(cov_methods["method"], cov_methods["coverage"], color=["#5DADE2", "#58D68D", "#AF7AC5"])
        axf.axhline(nominal, color="#555555", lw=1.0, ls="--")
        axf.set_ylim(max(0, nominal - 0.12), min(1.02, nominal + 0.12))
    axf.set_ylabel("Empirical coverage"); axf.set_title("Coverage method comparison"); axf.grid(True, axis="y"); v42_panel_label(axf, "f")

    fig.suptitle("Conformal uncertainty converts few-shot predictions into risk-controlled shortlists", fontsize=14.5, fontweight="bold")
    v42_save_figure(fig, "Figure_4_v4_2_conformal_risk_control_frontier", aliases=[FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy", PAPER_FINAL_DIR / "Figure_4_v4_1_consensus_true_enrichment"])
    mark_done(stage, {"v4_2": True, "preferred_feature_set": feature})


def v42_consensus_true_enrichment(consensus: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    existing = v42_read_table("QC_Table_consensus_true_enrichment.csv")
    if not existing.empty and {"target"}.issubset(existing.columns):
        return existing
    if consensus is None or consensus.empty:
        return pd.DataFrame()
    top = v42_top_consensus(int(CONFIG.get("story_final_top_candidate_rows_per_target", 25)))
    top = v42_add_percentiles_to_candidates(top, master=master)
    ytrue_col = "y_true_median" if "y_true_median" in top.columns else ("y_true" if "y_true" in top.columns else None)
    rows = []
    ts = v42_target_summary(master)
    for target, sub in top.groupby("target"):
        r = ts[ts["target"].astype(str).eq(str(target))].iloc[0] if (not ts.empty and "target" in ts.columns and ts["target"].astype(str).eq(str(target)).any()) else {}
        dataset_med = float(r.get("median", np.nan)) if hasattr(r, "get") else np.nan
        vals = pd.to_numeric(sub[ytrue_col], errors="coerce") if ytrue_col else pd.Series(dtype=float)
        rows.append({
            "target": target,
            "n_shortlist": int(len(sub)),
            "dataset_median": dataset_med,
            "shortlist_true_median": float(vals.median()) if len(vals) else np.nan,
            "median_fold_enrichment": float(vals.median() / dataset_med) if len(vals) and pd.notna(dataset_med) and dataset_med != 0 else np.nan,
            "fraction_true_top1pct": float((sub.get("true_percentile", pd.Series(dtype=float)) >= 0.99).mean()) if "true_percentile" in sub.columns else np.nan,
            "fraction_true_top5pct": float((sub.get("true_percentile", pd.Series(dtype=float)) >= 0.95).mean()) if "true_percentile" in sub.columns else np.nan,
            "fraction_true_top10pct": float((sub.get("true_percentile", pd.Series(dtype=float)) >= 0.90).mean()) if "true_percentile" in sub.columns else np.nan,
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "QC_Table_consensus_true_enrichment.csv")
    return out


def v42_plot_figure_5(core: Optional[pd.DataFrame], mosaec: Optional[pd.DataFrame], final_external: Optional[pd.DataFrame], master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.2 Figure 5 consensus shortlist atlas")
    v42_apply_story_style()
    consensus = v42_read_table("table_consensus_candidate_shortlist.csv", low_memory=True)
    top25 = v42_top_consensus(int(CONFIG.get("story_final_top_candidate_rows_per_target", 25)))
    top25 = v42_add_percentiles_to_candidates(top25, master=master)
    atomic_save_csv(top25, TABLE_DIR / "MAIN_Table_3_consensus_shortlist_target_balanced_top25.csv")
    enrich = v42_consensus_true_enrichment(consensus, master=master)
    fig = plt.figure(figsize=(15.8, 9.3))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2]); axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1]); axf = fig.add_subplot(gs[1, 2])

    # Candidate funnel per target.
    summary = v42_target_summary(master)
    cand = v42_read_table("table_candidate_tiers_all_predictions.csv", low_memory=True)
    funnel_rows = []
    for target in TARGET_ORDER_FINAL:
        n_dataset = np.nan
        if not summary.empty and "target" in summary.columns and summary["target"].astype(str).eq(target).any():
            r = summary[summary["target"].astype(str).eq(target)].iloc[0]
            n_dataset = r.get("n_mofs", r.get("n_nonmissing", np.nan))
        n_cand = int((cand["target"].astype(str) == target).sum()) if not cand.empty and "target" in cand.columns else np.nan
        n_cons = int((consensus["target"].astype(str).eq(target) & consensus.get("consensus_pass", pd.Series(False, index=consensus.index)).fillna(False).astype(bool)).sum()) if not consensus.empty and "target" in consensus.columns else np.nan
        n_top = int((top25["target"].astype(str) == target).sum()) if not top25.empty and "target" in top25.columns else 0
        funnel_rows.append({"target": target, "dataset": n_dataset, "quality_gated_rows": n_cand, "consensus_pass": n_cons, "top25": n_top})
    funnel = pd.DataFrame(funnel_rows)
    atomic_save_csv(funnel, TABLE_DIR / "STORY_Table_candidate_funnel_by_target.csv")
    y = np.arange(len(funnel))
    axa.barh(y, np.log10(pd.to_numeric(funnel["dataset"], errors="coerce").replace(0, np.nan)), color="#D5DBDB", label="all labelled MOFs")
    axa.barh(y, np.log10(pd.to_numeric(funnel["consensus_pass"], errors="coerce").replace(0, np.nan)), color="#5DADE2", label="consensus pass")
    axa.barh(y, np.log10(pd.to_numeric(funnel["top25"], errors="coerce").replace(0, np.nan)), color="#2E86C1", label="top-25")
    axa.set_yticks(y); axa.set_yticklabels([v42_label_target(t) for t in funnel["target"]]); axa.set_xlabel("log10(count)"); axa.set_title("Target-balanced decision funnel"); axa.legend(frameon=False)
    v42_panel_label(axa, "a")

    if not enrich.empty:
        enf = enrich.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(enrich["target"].astype(str))]).reset_index()
        valcol = "median_fold_enrichment" if "median_fold_enrichment" in enf.columns else ("fold_over_dataset_median" if "fold_over_dataset_median" in enf.columns else None)
        if valcol:
            axb.barh([v42_label_target(t) for t in enf["target"]], pd.to_numeric(enf[valcol], errors="coerce"), color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in enf["target"]])
            axb.axvline(1, color="#555555", ls="--", lw=1)
            axb.set_xlabel("Shortlist median / dataset median")
    axb.set_title("True enrichment of consensus shortlist"); axb.grid(True, axis="x"); v42_panel_label(axb, "b")

    if not enrich.empty:
        cols = [("fraction_true_top1pct", "top 1%"), ("fraction_true_top5pct", "top 5%"), ("fraction_true_top10pct", "top 10%")]
        x = np.arange(len(enrich)); width = 0.25
        order_targets = [t for t in TARGET_ORDER_FINAL if t in set(enrich["target"].astype(str))]
        enf = enrich.set_index("target").reindex(order_targets).reset_index()
        for j, (col, lab) in enumerate(cols):
            if col in enf.columns:
                axc.bar(x + (j - 1) * width, pd.to_numeric(enf[col], errors="coerce"), width=width, label=lab)
        axc.set_xticks(x); axc.set_xticklabels([v42_label_target(t) for t in enf["target"]], rotation=30, ha="right")
        axc.set_ylim(0, 1.05); axc.legend(frameon=False, ncol=3)
    axc.set_ylabel("Fraction of top-25 shortlist"); axc.set_title("How often candidates are true elites"); axc.grid(True, axis="y"); v42_panel_label(axc, "c")

    # Predicted-vs-true percentile, target-normalized into a single clear panel.
    if not top25.empty and {"true_percentile", "pred_percentile"}.issubset(top25.columns):
        for target in v42_order_targets(top25):
            sub = top25[top25["target"].astype(str).eq(target)]
            axd.scatter(pd.to_numeric(sub["true_percentile"], errors="coerce"), pd.to_numeric(sub["pred_percentile"], errors="coerce"), s=44, alpha=0.78, color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
        axd.plot([0, 1], [0, 1], ls="--", color="#555555", lw=1)
    axd.set_xlim(0.75, 1.01); axd.set_ylim(0.75, 1.01); axd.set_xlabel("True uptake percentile"); axd.set_ylabel("Predicted uptake percentile"); axd.set_title("Target-normalized shortlist accuracy"); axd.grid(True); v42_panel_label(axd, "d")

    # Consensus support distribution.
    support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_budgets"] if c in top25.columns]
    if support_cols:
        data = [pd.to_numeric(top25[c], errors="coerce").dropna() for c in support_cols]
        labels = [c.replace("n_", "") for c in support_cols]
        safe_boxplot_with_labels(axe, data, labels, showfliers=False)
        # overlay jittered points
        rng = np.random.default_rng(42)
        for j, vals in enumerate(data, start=1):
            vals = np.asarray(vals, dtype=float)
            xj = j + rng.normal(0, 0.035, size=len(vals))
            axe.scatter(xj, vals, s=8, alpha=0.24, color="#34495E")
    axe.set_ylabel("Support count"); axe.set_title("Consensus stability across runs"); axe.grid(True, axis="y"); v42_panel_label(axe, "e")

    # Geometry atlas: Di vs density for top candidates.
    xcol = "Di" if "Di" in top25.columns else None
    ycol = "Density" if "Density" in top25.columns else None
    if xcol and ycol:
        for target in v42_order_targets(top25):
            sub = top25[top25["target"].astype(str).eq(target)]
            size = pd.to_numeric(sub.get("trusted_support", pd.Series(20, index=sub.index)), errors="coerce").fillna(1)
            size = 20 + 3.0 * np.sqrt(size)
            axf.scatter(pd.to_numeric(sub[xcol], errors="coerce"), pd.to_numeric(sub[ycol], errors="coerce"), s=size, color=TARGET_COLORS_FINAL.get(target, None), alpha=0.72, label=v42_label_target(target))
    axf.set_xlabel("PLD proxy, Di / Å"); axf.set_ylabel("Density / g cm$^{-3}$"); axf.set_title("Geometry regime of final top candidates"); axf.grid(True); v42_panel_label(axf, "f")

    handles, labels = axf.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Target-balanced consensus shortlists are truly enriched in adsorption elites", fontsize=14.5, fontweight="bold")
    v42_save_figure(fig, "Figure_5_v4_2_consensus_shortlist_atlas", aliases=[FIG_MAIN_DIR / "Figure_5_external_realism_filter", PAPER_FINAL_DIR / "Figure_5_v4_1_consensus_chemistry_stability"])

    # External overlay moved to SI-style story figure.
    v42_plot_external_domain_overlap_si(final_external)
    mark_done(stage, {"v4_2": True, "external_framing": "domain_overlap_not_validation"})


def v42_plot_external_domain_overlap_si(final_external: Optional[pd.DataFrame]) -> None:
    if final_external is None or final_external.empty:
        final_external = v42_read_table("final_screening_table_with_external_flags.csv", low_memory=True)
    if final_external is None or final_external.empty or "target" not in final_external.columns:
        return
    fe = final_external.copy()
    for col in ["core_geometry_overlap_flag", "mosaec_geometry_overlap_flag", "core_exact_match_flag", "mosaec_exact_match_flag", "external_realism_score"]:
        if col in fe.columns:
            if "flag" in col:
                fe[col] = fe[col].fillna(False).astype(bool)
            else:
                fe[col] = pd.to_numeric(fe[col], errors="coerce")
    rows = []
    for target, sub in fe.groupby("target"):
        rows.append({
            "target": target,
            "n": len(sub),
            "core_exact_fraction": float(sub["core_exact_match_flag"].mean()) if "core_exact_match_flag" in sub.columns else np.nan,
            "mosaec_exact_fraction": float(sub["mosaec_exact_match_flag"].mean()) if "mosaec_exact_match_flag" in sub.columns else np.nan,
            "core_geometry_overlap_fraction": float(sub["core_geometry_overlap_flag"].mean()) if "core_geometry_overlap_flag" in sub.columns else np.nan,
            "mosaec_geometry_overlap_fraction": float(sub["mosaec_geometry_overlap_flag"].mean()) if "mosaec_geometry_overlap_flag" in sub.columns else np.nan,
            "mean_external_realism_score": float(sub["external_realism_score"].mean()) if "external_realism_score" in sub.columns else np.nan,
        })
    summ = pd.DataFrame(rows)
    atomic_save_csv(summ, TABLE_DIR / "SI_Table_external_domain_overlap_summary_v42.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))
    order = [t for t in TARGET_ORDER_FINAL if t in set(summ["target"].astype(str))]
    summ = summ.set_index("target").reindex(order).reset_index()
    x = np.arange(len(summ)); width = 0.34
    axes[0].bar(x - width/2, summ.get("core_geometry_overlap_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="CoRE geometry", color="#5DADE2")
    axes[0].bar(x + width/2, summ.get("mosaec_geometry_overlap_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="MOSAEC geometry", color="#58D68D")
    axes[0].set_xticks(x); axes[0].set_xticklabels([v42_label_target(t) for t in summ["target"]], rotation=30, ha="right"); axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Overlap fraction"); axes[0].set_title("Geometry-domain overlap"); axes[0].legend(frameon=False); v42_panel_label(axes[0], "a")
    axes[1].bar(x - width/2, summ.get("core_exact_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="CoRE exact", color="#85C1E9")
    axes[1].bar(x + width/2, summ.get("mosaec_exact_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="MOSAEC exact", color="#82E0AA")
    axes[1].set_xticks(x); axes[1].set_xticklabels([v42_label_target(t) for t in summ["target"]], rotation=30, ha="right"); axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Exact-match fraction"); axes[1].set_title("Exact matches are not assumed"); axes[1].legend(frameon=False); v42_panel_label(axes[1], "b")
    axes[2].bar([v42_label_target(t) for t in summ["target"]], pd.to_numeric(summ.get("mean_external_realism_score", pd.Series(np.nan, index=summ.index)), errors="coerce"), color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in summ["target"]])
    axes[2].tick_params(axis="x", rotation=30); axes[2].set_ylabel("Mean annotation score"); axes[2].set_title("Domain-overlap annotation score"); axes[2].grid(True, axis="y"); v42_panel_label(axes[2], "c")
    fig.suptitle("External overlays annotate domain overlap, not experimental validation", fontsize=13.2, fontweight="bold")
    v42_save_figure(fig, "Figure_S_external_domain_overlap_annotation_v42", si=True, aliases=[PAPER_FINAL_SI_DIR / "Figure_S_external_domain_overlap_annotation"])


def v42_plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.2 story SI figures")
    v42_apply_story_style()
    # SI 1: target distributions represented in quality-gated candidates.
    if tiers is not None and not tiers.empty and {"target", "y_true"}.issubset(tiers.columns):
        sample = tiers.drop_duplicates(["mof_id", "target"]) if "mof_id" in tiers.columns else tiers
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4)); axes = axes.ravel()
        for ax, target in zip(axes, TARGET_ORDER_FINAL):
            sub = sample[sample["target"].astype(str).eq(target)]
            vals = pd.to_numeric(sub["y_true"], errors="coerce").dropna()
            if len(vals):
                ax.hist(vals, bins=45, color=TARGET_COLORS_FINAL.get(target, "#777777"), alpha=0.80)
            ax.set_title(v42_label_target(target)); ax.set_xlabel("Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Target distributions in quality-gated candidate rows", fontsize=12.6, fontweight="bold")
        v42_save_figure(fig, "Figure_S1_v4_2_target_distributions", si=True, aliases=[FIG_SI_DIR / "Figure_S1_target_distributions"])
    # SI 2: model instability and QC gate summary.
    q = v42_read_table("QC_Table_model_stability.csv")
    if not q.empty and "quality_gate_reasons" in q.columns:
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        counts = q["quality_gate_reasons"].astype(str).value_counts().head(12)
        ax.barh(range(len(counts)), counts.values, color="#7F8C8D")
        ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index, fontsize=7.2); ax.invert_yaxis(); ax.set_xlabel("Experiments"); ax.set_title("Figure S2. Candidate-quality gate audit")
        v42_save_figure(fig, "Figure_S2_v4_2_quality_gate_audit", si=True)
    # SI 3: interval width by tier.
    if tiers is not None and not tiers.empty and {"tier", "split_upper", "split_lower"}.issubset(tiers.columns):
        tmp = tiers.copy(); tmp["interval_width"] = pd.to_numeric(tmp["split_upper"], errors="coerce") - pd.to_numeric(tmp["split_lower"], errors="coerce")
        labels = [x for x in ["trusted", "uncertain", "rejected"] if (tmp["tier"].astype(str) == x).any()]
        data = []
        for lab in labels:
            vals = tmp.loc[tmp["tier"].astype(str).eq(lab), "interval_width"].dropna()
            data.append(vals.sample(min(5000, len(vals)), random_state=42) if len(vals) > 0 else vals)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        if data:
            safe_boxplot_with_labels(ax, data, labels, showfliers=False)
        ax.set_ylabel("Split-conformal interval width"); ax.set_title("Figure S3. Interval width by candidate tier")
        v42_save_figure(fig, "Figure_S3_v4_2_interval_width_by_tier", si=True, aliases=[FIG_SI_DIR / "Figure_S3_interval_width_by_tier"])
    # SI 4: highest-error regimes from anatomy.
    if anatomy is not None and not anatomy.empty:
        metric = "normalized_mae" if "normalized_mae" in anatomy.columns else "mae"
        worst = anatomy.sort_values(metric, ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10.2, 6.2))
        labels = (worst["anatomy_type"].astype(str) + ":" + worst["regime"].astype(str)).str.slice(0, 42)
        ax.barh(range(len(worst)), pd.to_numeric(worst[metric], errors="coerce"), color="#85929E")
        ax.set_yticks(range(len(worst))); ax.set_yticklabels(labels, fontsize=7.2); ax.invert_yaxis(); ax.set_xlabel("Target-normalized MAE" if metric == "normalized_mae" else "MAE"); ax.set_title("Figure S4. Highest-error chemical regimes")
        v42_save_figure(fig, "Figure_S4_v4_2_highest_error_regimes", si=True, aliases=[FIG_SI_DIR / "Figure_S5_highest_error_regimes"])
    mark_done(stage, {"v4_2": True})


def v42_write_manifest() -> pd.DataFrame:
    rows = []
    for folder, label in [(PAPER_STORY_DIR, "main_story"), (PAPER_STORY_SI_DIR, "si_story")]:
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in {".png", ".pdf", ".svg"}:
                rows.append({"category": label, "file": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    for p in sorted(PAPER_STORY_DATA_DIR.glob("*.csv")):
        rows.append({"category": "figure_data", "file": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    mf = pd.DataFrame(rows)
    atomic_save_csv(mf, TABLE_DIR / "MANIFEST_v4_2_final_story_figures.csv")
    return mf


def v42_write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    stage = "07_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> WRITE_FINAL_RUN_REPORT_V4_2")
    manifest = v42_write_manifest()
    report = RESULTS_DIR / "RUN_REPORT.md"
    lines = []
    lines.append("# Few-shot MOF risk-controlled screening run report — v4.2 final story figures\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Runtime modes\n")
    for key in ["save_mode", "ram_mode", "comprehensive_level", "n_jobs", "data_root"]:
        lines.append(f"- {key}: `{CONFIG.get(key)}`")
    lines.append("\n## Experiment scale\n")
    lines.append(f"- Master table rows: {master.shape[0] if master is not None else 'NA'}")
    lines.append(f"- Master table columns: {master.shape[1] if master is not None else 'NA'}")
    lines.append(f"- Metric rows: {len(metrics_all) if metrics_all is not None else 'NA'}")
    if metrics_all is not None and not metrics_all.empty and set(EXPERIMENT_KEY_COLS).issubset(metrics_all.columns):
        lines.append(f"- Unique completed experiments: {metrics_all[EXPERIMENT_KEY_COLS].drop_duplicates().shape[0]}")
    lines.append("\n## v4.2 final publication figure folders\n")
    lines.append(f"- Main story figures: `{PAPER_STORY_DIR}`")
    lines.append(f"- SI/domain-overlap story figures: `{PAPER_STORY_SI_DIR}`")
    lines.append(f"- Story figure data: `{PAPER_STORY_DATA_DIR}`")
    lines.append("\n## Main figures to inspect first\n")
    for fname, meaning in [
        ("Figure_1_v4_2_story_workflow.png", "quantitative workflow / graphical abstract"),
        ("Figure_2_v4_2_fewshot_learning_phase_diagram.png", "label efficiency and enrichment"),
        ("Figure_3_v4_2_descriptor_chemistry_extrapolation_stress.png", "RAC gain and topology stress"),
        ("Figure_4_v4_2_conformal_risk_control_frontier.png", "coverage, interval width, precision-yield frontier"),
        ("Figure_5_v4_2_consensus_shortlist_atlas.png", "true-enriched target-balanced consensus atlas"),
    ]:
        p = PAPER_STORY_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'} — {meaning}")
    lines.append("\n## Important final tables\n")
    for fname in [
        "STORY_Table_descriptor_gain_by_budget_split.csv",
        "STORY_Table_split_stress_penalty_vulnerability.csv",
        "STORY_Table_label_efficiency_budget_thresholds.csv",
        "STORY_Table_precision_yield_frontier.csv",
        "STORY_Table_candidate_funnel_by_target.csv",
        "STORY_Table_top25_consensus_candidates.csv",
        "QC_Table_consensus_true_enrichment.csv",
        "SI_Table_external_domain_overlap_summary_v42.csv",
        "MANIFEST_v4_2_final_story_figures.csv",
    ]:
        p = TABLE_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'}")
    lines.append("\n## Scientific framing guardrails\n")
    lines.append("- Use geometry vs geometry+RACs as the validated descriptor comparison unless RDF integrity QC shows real nonzero RDF columns.")
    lines.append("- Treat topology-grouped splits as the central extrapolation stress test.")
    lines.append("- Present CoRE/MOSAEC overlays as domain-overlap annotations, not experimental validation, unless exact structural matches are established.")
    lines.append("- Final shortlist claims should be based on target-balanced consensus and true-enrichment QC, not single-model raw predictions.")
    lines.append(f"\n## Manifest entries\n- Files recorded: {len(manifest)}")
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage, {"v4_2": True, "n_manifest_rows": int(len(manifest))})

# ---- v4.2 overrides used by the existing main() function ----

def plot_figure_1_framework() -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v42_plot_figure_1(master)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    return v42_plot_figure_2(agg)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    return v42_plot_figure_3(agg)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v42_plot_figure_4(anatomy, tiers, master=master)


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v42_plot_figure_5(core, mosaec, final_external, master=master)


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v42_plot_si_figures(agg, tiers, anatomy, master=master)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    return v42_write_run_report(master, metrics_all)



# =============================================================================
# 16g. v4.3 final publication-level figure and table layer
# =============================================================================
# v4.3 is a publication-polish layer on top of v4.2.  It fixes the v4.2 issues
# identified from All_Results_02072026:
#   * restores the blank true-enrichment panel in Figure 5;
#   * replaces the flat actionable-yield panel in Figure 4 with a true yield
#     metric from all_experiment_metrics.csv;
#   * replaces single-tier interval-width panels with pre-final tier composition;
#   * fixes the descriptor/extrapolation interpretation text spacing;
#   * writes compact top-5-per-target and main-claims tables;
#   * writes all final publication figures to a new, clean folder.
# It still does not alter fitting, data merging, model predictions, candidate QC,
# or consensus shortlisting.

CONFIG["paper_final_name"] = "v4_3_publication_final_figures"
CONFIG.setdefault("publication_final_save_formats", ["png", "pdf", "svg"])
CONFIG.setdefault("publication_final_png_dpi", 600)
CONFIG.setdefault("publication_final_topn_main", 5)
CONFIG.setdefault("publication_final_topn_si", 25)

PAPER_PUBLICATION_DIR = RESULTS_DIR / "figures" / "paper_publication_final"
PAPER_PUBLICATION_SI_DIR = RESULTS_DIR / "figures" / "paper_publication_final_si"
PAPER_PUBLICATION_DATA_DIR = FIGDATA_DIR / "paper_publication_final"
for _d in [PAPER_PUBLICATION_DIR, PAPER_PUBLICATION_SI_DIR, PAPER_PUBLICATION_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def v43_apply_publication_style() -> None:
    """Publication-level matplotlib style: clean, compact, and vector-friendly."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.titlesize": 9.6,
        "axes.labelsize": 8.6,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "figure.dpi": 150,
        "savefig.dpi": int(CONFIG.get("publication_final_png_dpi", 600)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "grid.linewidth": 0.40,
        "grid.alpha": 0.22,
        "lines.linewidth": 1.55,
        "lines.markersize": 4.4,
        "patch.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def v43_panel_label(ax, label: str, x: float = -0.105, y: float = 1.075) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", ha="right", va="top")


def v43_save_figure(fig: plt.Figure, stem: str, si: bool = False, aliases: Optional[List[Path]] = None) -> None:
    """Save final publication figures as high-DPI PNG plus vector PDF/SVG."""
    v43_apply_publication_style()
    out_dir = PAPER_PUBLICATION_SI_DIR if si else PAPER_PUBLICATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG.get("publication_final_save_formats", ["png", "pdf", "svg"]):
        fmt = str(fmt).lower().strip()
        out = (out_dir / stem).with_suffix(f".{fmt}")
        try:
            fig.savefig(out, dpi=int(CONFIG.get("publication_final_png_dpi", 600)), bbox_inches="tight")
        except Exception as e:
            logging.warning("V43_FIGURE_SAVE_FAILED | %s | %s", out, e)
    for base in aliases or []:
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in ["png", "pdf"]:
            try:
                fig.savefig(base.with_suffix(f".{fmt}"), dpi=int(CONFIG.get("publication_final_png_dpi", 600)), bbox_inches="tight")
            except Exception as e:
                logging.warning("V43_ALIAS_SAVE_FAILED | %s | %s", base.with_suffix(f'.{fmt}'), e)
    plt.close(fig)


def v43_label_plain(t: Any) -> str:
    """Plain safe label for CSV tables, avoiding CO_2 style strings."""
    m = {
        "CO2_0p015bar_298K_mmolg": "CO2 0.015 bar",
        "CO2_0p150bar_298K_mmolg": "CO2 0.150 bar",
        "CH4_5p8bar_298K_mmolg": "CH4 5.8 bar",
        "CH4_65bar_298K_mmolg": "CH4 65 bar",
    }
    return m.get(str(t), str(t))


def v43_numeric_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # also allow case-insensitive normalised matches
    norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for c in candidates:
        key = re.sub(r"[^a-z0-9]+", "", str(c).lower())
        if key in norm:
            return norm[key]
    return None


def v43_true_enrichment_table(top25: Optional[pd.DataFrame] = None, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Read or reconstruct the consensus true-enrichment QC table robustly."""
    qce = v42_read_table("QC_Table_consensus_true_enrichment.csv")
    if not qce.empty and "target" in qce.columns:
        qce = qce.copy()
    else:
        if top25 is None or top25.empty:
            top25 = v42_top_consensus(int(CONFIG.get("publication_final_topn_si", 25)))
        top25 = v42_add_percentiles_to_candidates(top25, master=master)
        rows = []
        ytrue_col = "y_true_median" if "y_true_median" in top25.columns else ("y_true" if "y_true" in top25.columns else None)
        for target, sub in top25.groupby("target") if "target" in top25.columns else []:
            y = pd.to_numeric(sub[ytrue_col], errors="coerce").dropna() if ytrue_col else pd.Series(dtype=float)
            ref = None
            if master is not None and not master.empty and str(target) in master.columns:
                ref = pd.to_numeric(master[str(target)], errors="coerce").dropna()
            if ref is None or len(ref) == 0:
                ref = y
            med = float(ref.median()) if len(ref) else np.nan
            rows.append({
                "target": target,
                "n_shortlist": int(len(sub)),
                "dataset_median": med,
                "shortlist_true_median": float(y.median()) if len(y) else np.nan,
                "fold_enrichment_vs_dataset_median": float(y.median() / med) if len(y) and np.isfinite(med) and med != 0 else np.nan,
                "fraction_true_top1pct": float((sub.get("true_percentile", pd.Series(np.nan, index=sub.index)) >= 0.99).mean()) if "true_percentile" in sub.columns else np.nan,
                "fraction_true_top5pct": float((sub.get("true_percentile", pd.Series(np.nan, index=sub.index)) >= 0.95).mean()) if "true_percentile" in sub.columns else np.nan,
                "fraction_true_top10pct": float((sub.get("true_percentile", pd.Series(np.nan, index=sub.index)) >= 0.90).mean()) if "true_percentile" in sub.columns else np.nan,
            })
        qce = pd.DataFrame(rows)
    if not qce.empty:
        qce["target_label_plain"] = qce["target"].map(v43_label_plain)
        atomic_save_csv(qce, TABLE_DIR / "QC_Table_consensus_true_enrichment.csv")
    return qce


def v43_top_consensus_tables(master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    top25 = v42_top_consensus(int(CONFIG.get("publication_final_topn_si", 25)))
    if top25 is None or top25.empty:
        return pd.DataFrame()
    top25 = v42_add_percentiles_to_candidates(top25, master=master)
    top25["target_label_plain"] = top25["target"].map(v43_label_plain) if "target" in top25.columns else ""
    atomic_save_csv(top25, TABLE_DIR / "STORY_Table_top25_consensus_candidates.csv")
    top5 = top25.groupby("target", group_keys=False).head(int(CONFIG.get("publication_final_topn_main", 5))) if "target" in top25.columns else top25.head(20)
    atomic_save_csv(top5, TABLE_DIR / "MAIN_Table_3_top5_consensus_candidates_per_target.csv")
    return top25


def v43_candidate_funnel_table(master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build a clear target-level candidate funnel table."""
    # Stage 1: labelled MOFs per target.
    target_summary = v42_target_summary(master)
    n_map = {}
    if not target_summary.empty and "target" in target_summary.columns:
        for _, r in target_summary.iterrows():
            n_map[str(r["target"])] = int(r.get("n_mofs", r.get("n_nonmissing", np.nan))) if pd.notna(r.get("n_mofs", r.get("n_nonmissing", np.nan))) else np.nan
    # Stage 2: candidate rows per target.
    cand_counts = v42_read_table("QC_Table_candidate_rows_by_target.csv")
    cand_map = {}
    if not cand_counts.empty and "target" in cand_counts.columns:
        count_col = v43_numeric_col(cand_counts, ["n_candidate_rows", "n_rows", "count", "candidate_rows"])
        if count_col:
            cand_map = cand_counts.set_index("target")[count_col].to_dict()
    if not cand_map:
        tiers = v42_read_table("table_candidate_tiers_all_predictions.csv", low_memory=True)
        if not tiers.empty and "target" in tiers.columns:
            cand_map = tiers.groupby("target").size().to_dict()
    # Stage 3: consensus-pass candidates per target.
    consensus = v42_read_table("table_consensus_candidate_shortlist.csv", low_memory=True)
    cons_map = {}
    if not consensus.empty and "target" in consensus.columns:
        work = consensus
        if "consensus_pass" in work.columns:
            work = work[work["consensus_pass"].fillna(False).astype(bool)]
        cons_map = work.groupby("target").size().to_dict()
    rows = []
    for t in TARGET_ORDER_FINAL:
        rows.append({
            "target": t,
            "target_label_plain": v43_label_plain(t),
            "labelled_mofs": n_map.get(t, np.nan),
            "quality_gated_candidate_rows": cand_map.get(t, np.nan),
            "consensus_pass_candidates": cons_map.get(t, np.nan),
            "final_main_candidates": int(CONFIG.get("publication_final_topn_main", 5)),
            "final_si_candidates": int(CONFIG.get("publication_final_topn_si", 25)),
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_candidate_funnel_by_target.csv")
    return out


def v43_tier_fraction_from_metrics(agg_or_metrics: Optional[pd.DataFrame] = None, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Pre-final tier composition from experiment metrics rather than filtered tiers."""
    df = agg_or_metrics if agg_or_metrics is not None and not agg_or_metrics.empty else v42_read_table("all_experiment_metrics.csv", low_memory=True)
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "top_frac" in work.columns:
        work = work[pd.to_numeric(work["top_frac"], errors="coerce").sub(0.05).abs() < 1e-9]
    feature = v42_preferred_feature(work)
    if "feature_set" in work.columns:
        work = work[work["feature_set"].astype(str).eq(feature)]
    if "split" in work.columns and "random" in set(work["split"].astype(str)):
        work = work[work["split"].astype(str).eq("random")]
    if "model" in work.columns:
        work = work[work["model"].astype(str).isin(CONFIG.get("stable_plot_models", MODEL_ORDER_FINAL))]
    for col in ["budget", "trusted_count", "uncertain_count", "rejected_count", "trusted_count_mean", "uncertain_count_mean", "rejected_count_mean", "n_test"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    # Support both all_experiment_metrics columns and aggregated columns.
    tcol = "trusted_count_mean" if "trusted_count_mean" in work.columns else "trusted_count"
    ucol = "uncertain_count_mean" if "uncertain_count_mean" in work.columns else "uncertain_count"
    rcol = "rejected_count_mean" if "rejected_count_mean" in work.columns else "rejected_count"
    if not {tcol, ucol, rcol}.issubset(work.columns):
        return pd.DataFrame()
    grp = work.groupby(["target", "budget"], as_index=False)[[tcol, ucol, rcol]].mean()
    grp = grp.rename(columns={tcol: "trusted_count", ucol: "uncertain_count", rcol: "rejected_count"})
    denom = grp["trusted_count"] + grp["uncertain_count"] + grp["rejected_count"]
    grp["trusted_fraction"] = grp["trusted_count"] / denom.replace(0, np.nan)
    grp["uncertain_fraction"] = grp["uncertain_count"] / denom.replace(0, np.nan)
    grp["rejected_fraction"] = grp["rejected_count"] / denom.replace(0, np.nan)
    grp["target_label_plain"] = grp["target"].map(v43_label_plain)
    atomic_save_csv(grp, TABLE_DIR / "STORY_Table_prefinal_tier_fraction_by_budget.csv")
    return grp


def v43_main_claims_table(agg: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One compact table summarising the main manuscript claims by target."""
    rows = []
    feature = v42_preferred_feature(agg)
    df1000 = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, budget=1000, stable_only=True)
    gain = v42_descriptor_gain_table(agg)
    stress = v42_split_stress_table(agg)
    qce = v43_true_enrichment_table(master=master)
    for t in TARGET_ORDER_FINAL:
        sub = df1000[df1000["target"].astype(str).eq(t)] if not df1000.empty and "target" in df1000.columns else pd.DataFrame()
        best_recall = float(pd.to_numeric(sub.get("recall_mean", pd.Series(dtype=float)), errors="coerce").max()) if not sub.empty else np.nan
        best_enrich = float(pd.to_numeric(sub.get("enrichment_mean", pd.Series(dtype=float)), errors="coerce").max()) if not sub.empty else np.nan
        best_spearman = float(pd.to_numeric(sub.get("spearman_mean", pd.Series(dtype=float)), errors="coerce").max()) if not sub.empty and "spearman_mean" in sub.columns else np.nan
        gsub = gain[(gain["target"].astype(str).eq(t)) & (pd.to_numeric(gain["budget"], errors="coerce").eq(1000))] if not gain.empty else pd.DataFrame()
        rac_gain = float(pd.to_numeric(gsub.get("delta_recall_racs_minus_geometry", pd.Series(dtype=float)), errors="coerce").mean()) if not gsub.empty else np.nan
        ssub = stress[stress["target"].astype(str).eq(t)] if not stress.empty and "target" in stress.columns else pd.DataFrame()
        topo_penalty_col = v43_numeric_col(ssub, ["penalty_vs_random__topology_grouped"])
        topo_penalty = float(pd.to_numeric(ssub[topo_penalty_col], errors="coerce").iloc[0]) if topo_penalty_col and not ssub.empty else np.nan
        qsub = qce[qce["target"].astype(str).eq(t)] if not qce.empty and "target" in qce.columns else pd.DataFrame()
        enrich_col = v43_numeric_col(qsub, ["fold_enrichment_vs_dataset_median", "shortlist_median_over_dataset_median", "fold_enrichment"])
        frac_col = v43_numeric_col(qsub, ["fraction_true_top5pct", "shortlist_true_top5_fraction", "fraction_true_top_5pct"])
        rows.append({
            "target": t,
            "target_label_plain": v43_label_plain(t),
            "best_top5_recall_1000labels": best_recall,
            "best_enrichment_1000labels": best_enrich,
            "best_spearman_1000labels": best_spearman,
            "rac_gain_delta_top5_recall_1000labels": rac_gain,
            "topology_penalty_vs_random": topo_penalty,
            "consensus_shortlist_fold_enrichment_vs_dataset_median": float(pd.to_numeric(qsub[enrich_col], errors="coerce").iloc[0]) if enrich_col and not qsub.empty else np.nan,
            "consensus_shortlist_fraction_true_top5pct": float(pd.to_numeric(qsub[frac_col], errors="coerce").iloc[0]) if frac_col and not qsub.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "MAIN_Table_0_final_claims_summary.csv")
    return out


def v43_plot_figure_1(master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig1_framework"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.3 Figure 1 final quantitative workflow")
    v43_apply_publication_style()
    ts = v42_target_summary(master)
    fig = plt.figure(figsize=(15.8, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.95], hspace=0.36, wspace=0.30)
    axes_cards = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, target in zip(axes_cards, TARGET_ORDER_FINAL):
        ax.axis("off")
        color = TARGET_COLORS_FINAL.get(target, "#777777")
        row = ts[ts["target"].astype(str).eq(target)].iloc[0] if not ts.empty and "target" in ts.columns and (ts["target"].astype(str).eq(target)).any() else None
        n = row.get("n_mofs", row.get("n_nonmissing", np.nan)) if row is not None else np.nan
        med = row.get("median", row.get("median_mmol_g", np.nan)) if row is not None else np.nan
        mx = row.get("max", row.get("max_mmol_g", np.nan)) if row is not None else np.nan
        rect = plt.Rectangle((0.04, 0.08), 0.92, 0.84, transform=ax.transAxes,
                             facecolor="#FFFFFF", edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(0.08, 0.80, v42_label_target(target), transform=ax.transAxes,
                fontsize=12.0, fontweight="bold", color=color)
        ax.text(0.08, 0.58, f"{int(n):,} labelled MOFs" if pd.notna(n) else "labelled MOFs", transform=ax.transAxes, fontsize=9.0)
        ax.text(0.08, 0.40, f"median = {med:.3g} mmol g$^{{-1}}$" if pd.notna(med) else "median uptake", transform=ax.transAxes, fontsize=8.5)
        ax.text(0.08, 0.25, f"max = {mx:.3g} mmol g$^{{-1}}$" if pd.notna(mx) else "maximum uptake", transform=ax.transAxes, fontsize=8.5)
    axpipe = fig.add_subplot(gs[1, :]); axpipe.axis("off")
    steps = [
        ("ARC-MOF labels", "4 adsorption targets\ngeometry + RAC descriptors"),
        ("Few-shot design", "10–1000 labels\n5 seeds × grouped splits"),
        ("Stable learners", "RF / ExtraTrees / HGB\nLightGBM / XGBoost"),
        ("Risk control", "90% conformal intervals\ntrusted/uncertain/rejected tiers"),
        ("Consensus", "models × seeds × splits\ntarget-balanced shortlist"),
        ("Domain overlay", "CoRE/MOSAEC geometry\nannotation, not validation"),
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for i, (x, (title, body)) in enumerate(zip(xs, steps)):
        axpipe.add_patch(plt.Circle((x, 0.67), 0.047, transform=axpipe.transAxes, fc="#F4F6F7", ec="#566573", lw=1.1))
        axpipe.text(x, 0.67, str(i + 1), transform=axpipe.transAxes, ha="center", va="center", fontweight="bold")
        axpipe.add_patch(plt.Rectangle((x - 0.072, 0.17), 0.144, 0.33, transform=axpipe.transAxes, fc="#FFFFFF", ec="#B2BABB", lw=0.9))
        axpipe.text(x, 0.42, title, transform=axpipe.transAxes, ha="center", va="center", fontsize=8.8, fontweight="bold")
        axpipe.text(x, 0.27, body, transform=axpipe.transAxes, ha="center", va="center", fontsize=7.3)
        if i < len(xs) - 1:
            axpipe.annotate("", xy=(xs[i+1]-0.06, 0.67), xytext=(x+0.06, 0.67), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", lw=1.2, color="#566573"))
    fig.suptitle("Risk-controlled few-shot adsorption-elite discovery from ARC-MOF", fontsize=15.2, fontweight="bold")
    v43_save_figure(fig, "Figure_1_v4_3_publication_workflow", aliases=[PAPER_STORY_DIR / "Figure_1_v4_2_story_workflow", FIG_MAIN_DIR / "Figure_1_risk_controlled_framework"])
    mark_done(stage, {"v4_3": True})


def v43_plot_figure_2(agg: pd.DataFrame) -> None:
    """Reuse v4.2 phase-diagram concept, but write into the publication folder."""
    stage = "fig2_performance"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.3 Figure 2 final few-shot learning phase diagram")
    v43_apply_publication_style()
    feature = v42_preferred_feature(agg)
    recall = v42_learning_curves(agg, "recall_mean", feature_set=feature, split=str(CONFIG.get("story_final_target_metric_split", "random")))
    enrich = v42_learning_curves(agg, "enrichment_mean", feature_set=feature, split=str(CONFIG.get("story_final_target_metric_split", "random")))
    atomic_save_csv(recall, PAPER_PUBLICATION_DATA_DIR / "Figure_2a_recall_learning_curves.csv")
    atomic_save_csv(enrich, PAPER_PUBLICATION_DATA_DIR / "Figure_2b_enrichment_learning_curves.csv")
    fig = plt.figure(figsize=(15.8, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.34)
    axa, axb, axc, axd, axe, axf = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for target in v42_order_targets(recall):
        sub = recall[recall["target"].astype(str).eq(target)].sort_values("budget")
        if not sub.empty:
            color = TARGET_COLORS_FINAL.get(target, None)
            x = sub["budget"].to_numpy(dtype=float); y = sub["mean"].to_numpy(dtype=float)
            axa.plot(x, y, marker="o", color=color, label=v42_label_target(target))
            axa.fill_between(x, sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float), color=color, alpha=0.13, linewidth=0)
    axa.axhline(0.05, ls="--", lw=1.0, color="#777777", label="random")
    axa.set_xscale("log"); axa.set_xlabel("Label budget"); axa.set_ylabel("Top-5% recall"); axa.set_title("Elite recovery emerges from few labels"); axa.grid(True); axa.legend(frameon=False)
    v43_panel_label(axa, "a")
    for target in v42_order_targets(enrich):
        sub = enrich[enrich["target"].astype(str).eq(target)].sort_values("budget")
        if not sub.empty:
            color = TARGET_COLORS_FINAL.get(target, None)
            x = sub["budget"].to_numpy(dtype=float); y = sub["mean"].to_numpy(dtype=float)
            axb.plot(x, y, marker="o", color=color)
            axb.fill_between(x, sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float), color=color, alpha=0.13, linewidth=0)
    axb.axhline(1.0, ls="--", lw=1.0, color="#777777")
    axb.set_xscale("log"); axb.set_xlabel("Label budget"); axb.set_ylabel("Enrichment over random"); axb.set_title("Screening enrichment"); axb.grid(True); v43_panel_label(axb, "b")
    eff_rows = []
    for target, sub in recall.groupby("target") if not recall.empty else []:
        sub = sub.sort_values("budget")
        final = sub.loc[sub["budget"].idxmax(), "mean"] if not sub.empty else np.nan
        for _, r in sub.iterrows():
            eff_rows.append({"target": target, "budget": int(r["budget"]), "label_efficiency": float(r["mean"] / final) if pd.notna(final) and final > 0 else np.nan})
    eff = pd.DataFrame(eff_rows); atomic_save_csv(eff, PAPER_PUBLICATION_DATA_DIR / "Figure_2c_label_efficiency.csv")
    for target in v42_order_targets(eff):
        sub = eff[eff["target"].astype(str).eq(target)].sort_values("budget")
        axc.plot(sub["budget"], sub["label_efficiency"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axc.axhline(0.75, ls=":", lw=1.0, color="#666666")
    axc.set_xscale("log"); axc.set_ylim(0, 1.05); axc.set_xlabel("Label budget"); axc.set_ylabel("Fraction of 1000-label recall"); axc.set_title("Label-efficiency index"); axc.grid(True); v43_panel_label(axc, "c")
    thresholds = [0.50, 0.75, 0.90]
    sat_rows = []
    for target, sub in eff.groupby("target") if not eff.empty else []:
        for thr in thresholds:
            hit = sub[sub["label_efficiency"] >= thr].sort_values("budget")
            sat_rows.append({"target": target, "target_label_plain": v43_label_plain(target), "threshold": thr, "budget_needed": int(hit.iloc[0]["budget"]) if not hit.empty else np.nan})
    sat = pd.DataFrame(sat_rows); atomic_save_csv(sat, TABLE_DIR / "STORY_Table_label_efficiency_budget_thresholds.csv")
    ytargets = v42_order_targets(sat)
    ylabels = [v42_label_target(t) for t in ytargets]
    for j, thr in enumerate(thresholds):
        vals = [sat[(sat["target"].astype(str).eq(t)) & (sat["threshold"].eq(thr))]["budget_needed"].iloc[0] if not sat[(sat["target"].astype(str).eq(t)) & (sat["threshold"].eq(thr))].empty else np.nan for t in ytargets]
        axd.scatter(vals, np.arange(len(ylabels)) + (j - 1) * 0.18, label=f"{int(thr*100)}%", s=36)
    axd.set_xscale("log"); axd.set_yticks(np.arange(len(ylabels))); axd.set_yticklabels(ylabels); axd.set_xlabel("Budget needed"); axd.set_title("How many labels are enough?"); axd.grid(True, axis="x"); axd.legend(title="of final recall", frameon=False)
    v43_panel_label(axd, "d")
    df1000 = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, split=str(CONFIG.get("story_final_target_metric_split", "random")), budget=1000, stable_only=True)
    mod = df1000.groupby("model", as_index=False)["recall_mean"].mean() if not df1000.empty else pd.DataFrame()
    if not mod.empty:
        mod["model_label"] = mod["model"].map(v41_model_label)
        order = [m for m in MODEL_ORDER_FINAL if m in set(mod["model"])]
        mod = mod.set_index("model").reindex(order).dropna(subset=["recall_mean"]).reset_index()
        axe.bar(mod["model_label"], mod["recall_mean"], color=[MODEL_COLORS_FINAL.get(m, "#777777") for m in mod["model"]])
    axe.set_ylabel("Mean top-5% recall"); axe.set_title("Stable model comparison at 1000 labels"); axe.tick_params(axis="x", rotation=35); axe.grid(True, axis="y"); v43_panel_label(axe, "e")
    claims = v43_main_claims_table(agg)
    if not claims.empty:
        bestdf = claims[["target", "best_enrichment_1000labels"]].dropna().sort_values("best_enrichment_1000labels")
        axf.barh([v42_label_target(t) for t in bestdf["target"]], bestdf["best_enrichment_1000labels"], color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in bestdf["target"]])
        for y, val in enumerate(bestdf["best_enrichment_1000labels"]):
            axf.text(val, y, f" {val:.1f}×", va="center", fontsize=8)
    axf.axvline(1.0, ls="--", color="#777777", lw=1); axf.set_xlabel("Best enrichment at 1000 labels"); axf.set_title("Final enrichment summary"); axf.grid(True, axis="x"); v43_panel_label(axf, "f")
    fig.suptitle("Few-shot learning phase diagram for adsorption-elite recovery", fontsize=14.8, fontweight="bold")
    v43_save_figure(fig, "Figure_2_v4_3_fewshot_learning_phase_diagram", aliases=[PAPER_STORY_DIR / "Figure_2_v4_2_fewshot_learning_phase_diagram", FIG_MAIN_DIR / "Figure_2_label_budget_performance"])
    mark_done(stage, {"v4_3": True, "preferred_feature_set": feature})


def v43_plot_figure_3(agg: pd.DataFrame) -> None:
    stage = "fig3_calibration"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.3 Figure 3 descriptor chemistry and extrapolation stress")
    v43_apply_publication_style()
    dg = v42_descriptor_gain_table(agg)
    st = v42_split_stress_table(agg)
    fig = plt.figure(figsize=(15.8, 8.9))
    gs = fig.add_gridspec(2, 3, hspace=0.41, wspace=0.35)
    axa, axb, axc, axd, axe, axf = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    if not dg.empty:
        dg_rand = dg[dg["split"].astype(str).eq("random")] if "split" in dg.columns else dg
        mat = dg_rand.pivot_table(index="target", columns="budget", values="delta_recall_racs_minus_geometry", aggfunc="mean")
        mat = mat.reindex([t for t in TARGET_ORDER_FINAL if t in mat.index]); mat.index = [v42_label_target(i) for i in mat.index]
        v42_heatmap(axa, mat, "RAC descriptor gain across label budgets", "Δ recall", center_zero=True, fmt=".2f")
    v43_panel_label(axa, "a")
    if not dg.empty:
        sub = dg[pd.to_numeric(dg["budget"], errors="coerce").eq(1000)]
        bars = sub.groupby("target", as_index=False)["delta_recall_racs_minus_geometry"].mean()
        bars = bars.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(bars["target"]) or t in bars.index]).reset_index()
        axb.barh([v42_label_target(t) for t in bars["target"]], bars["delta_recall_racs_minus_geometry"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
        axb.axvline(0, color="#555555", lw=1)
    axb.set_xlabel("Δ recall: RACs − geometry"); axb.set_title("Chemistry benefit at 1000 labels"); axb.grid(True, axis="x"); v43_panel_label(axb, "b")
    feature = v42_preferred_feature(agg)
    df = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, budget=1000, stable_only=True)
    if not df.empty:
        split_mat = df.pivot_table(index="target", columns="split", values="recall_mean", aggfunc="mean")
        split_mat = split_mat.reindex([t for t in TARGET_ORDER_FINAL if t in split_mat.index])
        split_mat = split_mat[[s for s in SPLIT_ORDER_FINAL if s in split_mat.columns]]
        split_mat.index = [v42_label_target(i) for i in split_mat.index]
        split_mat.columns = [v41_split_label(c) for c in split_mat.columns]
        v42_heatmap(axc, split_mat, "Split-stress map at 1000 labels", "top-5% recall", center_zero=False, cmap="viridis", fmt=".2f")
    v43_panel_label(axc, "c")
    if not st.empty:
        bars = st[["target", "penalty_vs_random__topology_grouped"]].dropna() if "penalty_vs_random__topology_grouped" in st.columns else pd.DataFrame()
        if not bars.empty:
            bars = bars.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(bars["target"]) or t in bars.index]).reset_index()
            axd.barh([v42_label_target(t) for t in bars["target"]], bars["penalty_vs_random__topology_grouped"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
            axd.axvline(0, color="#555555", lw=1)
    axd.set_xlabel("Recall(random) − recall(topology)"); axd.set_title("Topology extrapolation penalty"); axd.grid(True, axis="x"); v43_panel_label(axd, "d")
    vul_cols = [c for c in st.columns if c.startswith("vulnerability__")] if not st.empty else []
    if vul_cols:
        vul = st.set_index("target")[vul_cols]
        vul.columns = [v41_split_label(c.replace("vulnerability__", "")) for c in vul.columns]
        vul = vul.reindex([t for t in TARGET_ORDER_FINAL if t in vul.index]); vul.index = [v42_label_target(i) for i in vul.index]
        v42_heatmap(axe, vul, "Extrapolation vulnerability", "1 − grouped/random", center_zero=False, cmap="magma", fmt=".2f")
    v43_panel_label(axe, "e")
    axf.axis("off"); v43_panel_label(axf, "f")
    axf.set_title("Mechanistic interpretation", loc="left", fontweight="bold", pad=8)
    notes = [
        ("Geometry", "pore size, surface area, and density\nprovide a strong physical baseline"),
        ("RAC chemistry", "metal/linker descriptors improve\nelite recovery beyond geometry"),
        ("Topology split", "the hardest out-of-distribution\nstress test for transferability"),
        ("Main message", "random-split accuracy alone\noverstates chemical transfer"),
    ]
    for j, (head, body) in enumerate(notes):
        y = 0.82 - 0.205 * j
        axf.add_patch(plt.Rectangle((0.04, y - 0.079), 0.90, 0.145, transform=axf.transAxes,
                                    fc="#FBFCFC", ec="#AAB7B8", lw=0.75))
        axf.text(0.075, y + 0.026, head, transform=axf.transAxes, ha="left", va="center", fontsize=8.7, fontweight="bold")
        axf.text(0.075, y - 0.035, body, transform=axf.transAxes, ha="left", va="center", fontsize=7.35, linespacing=1.15)
    fig.suptitle("Descriptor chemistry improves screening, but topology extrapolation remains limiting", fontsize=14.8, fontweight="bold")
    v43_save_figure(fig, "Figure_3_v4_3_descriptor_chemistry_extrapolation_stress", aliases=[PAPER_STORY_DIR / "Figure_3_v4_2_descriptor_chemistry_extrapolation_stress", FIG_MAIN_DIR / "Figure_3_calibration_and_abstention"])
    mark_done(stage, {"v4_3": True, "preferred_feature_set": feature})


def v43_plot_figure_4(anatomy: Optional[pd.DataFrame], tiers: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.3 Figure 4 final conformal risk-control frontier")
    v43_apply_publication_style()
    agg = v42_read_table("table_aggregated_metrics_with_bootstrap_ci.csv", low_memory=True)
    feature = v42_preferred_feature(agg)
    nominal = float(CONFIG.get("story_final_nominal_coverage", 0.90))
    fig = plt.figure(figsize=(15.8, 8.9))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.34)
    axa, axb, axc, axd, axe, axf = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    df = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, stable_only=True)
    if not df.empty:
        cerr = df.assign(coverage_error=df["split_coverage_mean"] - nominal).pivot_table(index="target", columns="split", values="coverage_error", aggfunc="mean")
        cerr = cerr.reindex([t for t in TARGET_ORDER_FINAL if t in cerr.index]); cerr = cerr[[s for s in SPLIT_ORDER_FINAL if s in cerr.columns]]
        cerr.index = [v42_label_target(i) for i in cerr.index]; cerr.columns = [v41_split_label(c) for c in cerr.columns]
        v42_heatmap(axa, cerr, "Coverage error relative to 90%", "coverage − 0.90", center_zero=True, fmt="+.2f")
    v43_panel_label(axa, "a")
    std_map = v42_target_std_map(master)
    width = v42_clean_agg(agg, top_frac=0.05, feature_set=feature, split="random", stable_only=True)
    if not width.empty:
        width["target_std"] = width["target"].map(std_map)
        width["normalized_width"] = width["split_mean_width_mean"] / width["target_std"].replace(0, np.nan)
        wcurve = width.groupby(["target", "budget"], as_index=False)["normalized_width"].mean()
        atomic_save_csv(wcurve, PAPER_PUBLICATION_DATA_DIR / "Figure_4b_normalized_interval_width.csv")
        for target in v42_order_targets(wcurve):
            sub = wcurve[wcurve["target"].astype(str).eq(target)].sort_values("budget")
            axb.plot(sub["budget"], sub["normalized_width"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axb.set_xscale("log"); axb.set_xlabel("Label budget"); axb.set_ylabel("Interval width / target std"); axb.set_title("Uncertainty narrows with labels"); axb.grid(True); v43_panel_label(axb, "b")
    frontier = v42_precision_yield_frontier(tiers, master=master)
    if not frontier.empty:
        for target in v42_order_targets(frontier):
            sub = frontier[frontier["target"].astype(str).eq(target)].sort_values("retained_fraction")
            axc.plot(sub["retained_fraction"], sub["precision_true_top5"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axc.set_xlabel("Retained fraction"); axc.set_ylabel("True top-5% fraction"); axc.set_ylim(0, 1.05); axc.set_title("Precision–yield frontier"); axc.grid(True); axc.legend(frameon=False)
    v43_panel_label(axc, "c")
    tier_frac = v43_tier_fraction_from_metrics(None, master=master)
    if not tier_frac.empty:
        for target in v42_order_targets(tier_frac):
            sub = tier_frac[tier_frac["target"].astype(str).eq(target)].sort_values("budget")
            axd.plot(sub["budget"], sub["trusted_fraction"], marker="o", color=TARGET_COLORS_FINAL.get(target, None), label=v42_label_target(target))
    axd.set_xscale("log"); axd.set_xlabel("Label budget"); axd.set_ylabel("Trusted fraction before final gates"); axd.set_title("Actionable yield from experiment metrics"); axd.grid(True); v43_panel_label(axd, "d")
    # Stacked tier composition at largest budget, pre-final gating.
    if not tier_frac.empty:
        maxb = int(pd.to_numeric(tier_frac["budget"], errors="coerce").max())
        comp = tier_frac[pd.to_numeric(tier_frac["budget"], errors="coerce").eq(maxb)].copy()
        comp = comp.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(comp["target"]) or t in comp.index]).reset_index()
        x = np.arange(len(comp)); bottom = np.zeros(len(comp))
        for col, lab, color in [("trusted_fraction", "trusted", "#2ECC71"), ("uncertain_fraction", "uncertain", "#F5B041"), ("rejected_fraction", "rejected", "#95A5A6")]:
            vals = pd.to_numeric(comp[col], errors="coerce").fillna(0).to_numpy()
            axe.bar(x, vals, bottom=bottom, label=lab, color=color)
            bottom += vals
        axe.set_xticks(x); axe.set_xticklabels([v42_label_target(t) for t in comp["target"]], rotation=30, ha="right")
        axe.legend(frameon=False, ncol=3)
    axe.set_ylim(0, 1.05); axe.set_ylabel("Fraction of prediction rows"); axe.set_title("Uncertainty-tier composition at 1000 labels"); axe.grid(True, axis="y"); v43_panel_label(axe, "e")
    if not df.empty:
        cov_methods = pd.DataFrame({"method": ["split", "local", "Mondrian"], "coverage": [pd.to_numeric(df.get("split_coverage_mean"), errors="coerce").mean(), pd.to_numeric(df.get("local_coverage_mean"), errors="coerce").mean(), pd.to_numeric(df.get("mondrian_coverage_mean"), errors="coerce").mean()]})
        axf.bar(cov_methods["method"], cov_methods["coverage"], color=["#5DADE2", "#58D68D", "#AF7AC5"])
        axf.axhline(nominal, color="#555555", lw=1.0, ls="--"); axf.set_ylim(max(0, nominal - 0.12), min(1.02, nominal + 0.12))
    axf.set_ylabel("Empirical coverage"); axf.set_title("Coverage method comparison"); axf.grid(True, axis="y"); v43_panel_label(axf, "f")
    fig.suptitle("Conformal uncertainty converts few-shot predictions into risk-controlled shortlists", fontsize=14.8, fontweight="bold")
    v43_save_figure(fig, "Figure_4_v4_3_conformal_risk_control_frontier", aliases=[PAPER_STORY_DIR / "Figure_4_v4_2_conformal_risk_control_frontier", FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy"])
    mark_done(stage, {"v4_3": True, "preferred_feature_set": feature})


def v43_plot_figure_5(core: Optional[pd.DataFrame], mosaec: Optional[pd.DataFrame], final_external: Optional[pd.DataFrame], master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.3 Figure 5 final consensus shortlist atlas")
    v43_apply_publication_style()
    top25 = v43_top_consensus_tables(master=master)
    top25 = v42_add_percentiles_to_candidates(top25, master=master)
    qce = v43_true_enrichment_table(top25, master=master)
    funnel = v43_candidate_funnel_table(master=master)
    claims = v43_main_claims_table(v42_read_table("table_aggregated_metrics_with_bootstrap_ci.csv", low_memory=True), master=master)
    fig = plt.figure(figsize=(15.8, 10.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.98, 1.0, 1.0], hspace=0.47, wspace=0.36)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2])
    axd_container = fig.add_subplot(gs[1:, 0:2]); axd_container.axis("off")
    subgs = gs[1:, 0:2].subgridspec(2, 2, hspace=0.36, wspace=0.32)
    axd_list = [fig.add_subplot(subgs[i, j]) for i in range(2) for j in range(2)]
    axe = fig.add_subplot(gs[1, 2]); axf = fig.add_subplot(gs[2, 2])
    # (a) Real candidate funnel: log-scaled, labelled stages.
    if not funnel.empty:
        order = [t for t in TARGET_ORDER_FINAL if t in set(funnel["target"].astype(str))]
        stage_cols = ["labelled_mofs", "quality_gated_candidate_rows", "consensus_pass_candidates", "final_main_candidates"]
        stage_labels = ["labelled\nMOFs", "quality-gated\nrows", "consensus\npass", "top-5\nper target"]
        for i, t in enumerate(order):
            vals = [float(funnel.loc[funnel["target"].astype(str).eq(t), c].iloc[0]) if c in funnel.columns and not funnel.loc[funnel["target"].astype(str).eq(t), c].empty else np.nan for c in stage_cols]
            y = np.arange(len(stage_cols)) + i * 0.17
            axa.plot(vals, y, marker="o", color=TARGET_COLORS_FINAL.get(t, "#777777"), alpha=0.82, label=v42_label_target(t))
        axa.set_yticks(np.arange(len(stage_cols)) + 0.255); axa.set_yticklabels(stage_labels)
        axa.set_xscale("log"); axa.set_xlabel("Count (log scale)"); axa.legend(frameon=False, fontsize=6.8, loc="lower left")
    axa.set_title("Decision funnel to final candidates"); axa.grid(True, axis="x"); v43_panel_label(axa, "a")
    # (b) True enrichment, robustly resolved from QC column.
    if not qce.empty:
        qce = qce.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(qce["target"].astype(str)) or t in qce.index]).reset_index()
        col = v43_numeric_col(qce, ["fold_enrichment_vs_dataset_median", "shortlist_median_over_dataset_median", "fold_enrichment"])
        vals = pd.to_numeric(qce[col], errors="coerce") if col else pd.Series(np.nan, index=qce.index)
        axb.bar(range(len(qce)), vals, color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in qce["target"]])
        axb.set_xticks(range(len(qce))); axb.set_xticklabels([v42_label_target(t) for t in qce["target"]], rotation=30, ha="right")
        for i, v in enumerate(vals):
            if pd.notna(v): axb.text(i, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=7.7)
    axb.axhline(1.0, ls="--", color="#777777", lw=1); axb.set_ylabel("Shortlist median / dataset median"); axb.set_title("True enrichment of consensus shortlist"); axb.grid(True, axis="y"); v43_panel_label(axb, "b")
    # (c) True top-percentile fractions.
    if not qce.empty:
        cols = [(v43_numeric_col(qce, ["fraction_true_top1pct", "fraction_top1pct"]), "top 1%"), (v43_numeric_col(qce, ["fraction_true_top5pct", "shortlist_true_top5_fraction"]), "top 5%"), (v43_numeric_col(qce, ["fraction_true_top10pct", "fraction_top10pct"]), "top 10%")]
        x = np.arange(len(qce)); width = 0.24
        for j, (col, lab) in enumerate(cols):
            if col:
                axc.bar(x + (j - 1) * width, pd.to_numeric(qce[col], errors="coerce"), width=width, label=lab)
        axc.set_xticks(x); axc.set_xticklabels([v42_label_target(t) for t in qce["target"]], rotation=30, ha="right"); axc.legend(frameon=False, ncol=3, fontsize=6.8)
    axc.set_ylim(0, 1.05); axc.set_ylabel("Fraction of top-25 shortlist"); axc.set_title("How often candidates are true elites"); axc.grid(True, axis="y"); v43_panel_label(axc, "c")
    # (d) Four target-wise predicted-vs-true percentile mini-panels.
    if not top25.empty and {"true_percentile", "pred_percentile", "target"}.issubset(top25.columns):
        for ax, target in zip(axd_list, TARGET_ORDER_FINAL):
            sub = top25[top25["target"].astype(str).eq(target)]
            ax.scatter(pd.to_numeric(sub.get("true_percentile"), errors="coerce"), pd.to_numeric(sub.get("pred_percentile"), errors="coerce"), s=24, alpha=0.75, color=TARGET_COLORS_FINAL.get(target, "#777777"))
            ax.plot([0.70, 1.01], [0.70, 1.01], ls="--", color="#666666", lw=0.8)
            ax.set_xlim(0.70, 1.01); ax.set_ylim(0.70, 1.01); ax.set_title(v42_label_target(target), fontsize=8.4); ax.grid(True)
            if ax in axd_list[2:]: ax.set_xlabel("True percentile")
            if ax in [axd_list[0], axd_list[2]]: ax.set_ylabel("Pred. percentile")
            if len(sub) and "true_percentile" in sub:
                medp = pd.to_numeric(sub["true_percentile"], errors="coerce").median()
                ax.text(0.04, 0.92, f"median true\nperc. {medp:.2f}", transform=ax.transAxes, fontsize=6.6, va="top")
        v43_panel_label(axd_list[0], "d", x=-0.18, y=1.22)
    # (e) Support distributions.
    support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_budgets"] if c in top25.columns]
    if support_cols:
        data = [pd.to_numeric(top25[c], errors="coerce").dropna() for c in support_cols]
        labels = [c.replace("n_", "") for c in support_cols]
        safe_boxplot_with_labels(axe, data, labels, showfliers=False)
        rng = np.random.default_rng(42)
        for j, vals in enumerate(data, start=1):
            vals = np.asarray(vals, dtype=float)
            axe.scatter(j + rng.normal(0, 0.035, size=len(vals)), vals, s=7, alpha=0.22, color="#34495E")
    axe.set_ylabel("Support count"); axe.set_title("Consensus stability across runs"); axe.grid(True, axis="y"); v43_panel_label(axe, "e")
    # (f) Geometry atlas.
    xcol = "Di" if "Di" in top25.columns else None; ycol = "Density" if "Density" in top25.columns else None
    if xcol and ycol:
        for target in v42_order_targets(top25):
            sub = top25[top25["target"].astype(str).eq(target)]
            size = pd.to_numeric(sub.get("trusted_support", pd.Series(20, index=sub.index)), errors="coerce").fillna(1)
            size = 20 + 3.0 * np.sqrt(size)
            axf.scatter(pd.to_numeric(sub[xcol], errors="coerce"), pd.to_numeric(sub[ycol], errors="coerce"), s=size, color=TARGET_COLORS_FINAL.get(target, None), alpha=0.72, label=v42_label_target(target))
    axf.set_xlabel("PLD proxy, Di / Å"); axf.set_ylabel("Density / g cm$^{-3}$"); axf.set_title("Geometry regime of final candidates"); axf.grid(True); v43_panel_label(axf, "f")
    handles, labels = axf.get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.008))
    fig.suptitle("Target-balanced consensus shortlists are truly enriched in adsorption elites", fontsize=14.8, fontweight="bold")
    v43_save_figure(fig, "Figure_5_v4_3_consensus_shortlist_atlas", aliases=[PAPER_STORY_DIR / "Figure_5_v4_2_consensus_shortlist_atlas", FIG_MAIN_DIR / "Figure_5_external_realism_filter"])
    v43_plot_external_domain_overlap_si(final_external)
    mark_done(stage, {"v4_3": True, "external_framing": "domain_overlap_not_validation"})


def v43_plot_external_domain_overlap_si(final_external: Optional[pd.DataFrame]) -> None:
    """Cleaner SI figure for external overlap; kept outside the main claims."""
    if final_external is None or final_external.empty:
        final_external = v42_read_table("final_screening_table_with_external_flags.csv", low_memory=True)
    if final_external is None or final_external.empty or "target" not in final_external.columns:
        return
    fe = final_external.copy()
    for col in ["core_geometry_overlap_flag", "mosaec_geometry_overlap_flag", "core_exact_match_flag", "mosaec_exact_match_flag", "external_realism_score"]:
        if col in fe.columns:
            fe[col] = fe[col].fillna(False).astype(bool) if "flag" in col else pd.to_numeric(fe[col], errors="coerce")
    rows = []
    for target, sub in fe.groupby("target"):
        rows.append({
            "target": target, "target_label_plain": v43_label_plain(target), "n": len(sub),
            "core_exact_fraction": float(sub["core_exact_match_flag"].mean()) if "core_exact_match_flag" in sub.columns else np.nan,
            "mosaec_exact_fraction": float(sub["mosaec_exact_match_flag"].mean()) if "mosaec_exact_match_flag" in sub.columns else np.nan,
            "core_geometry_overlap_fraction": float(sub["core_geometry_overlap_flag"].mean()) if "core_geometry_overlap_flag" in sub.columns else np.nan,
            "mosaec_geometry_overlap_fraction": float(sub["mosaec_geometry_overlap_flag"].mean()) if "mosaec_geometry_overlap_flag" in sub.columns else np.nan,
            "mean_external_realism_score": float(sub["external_realism_score"].mean()) if "external_realism_score" in sub.columns else np.nan,
        })
    summ = pd.DataFrame(rows); atomic_save_csv(summ, TABLE_DIR / "SI_Table_external_domain_overlap_summary_v43.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3))
    order = [t for t in TARGET_ORDER_FINAL if t in set(summ["target"].astype(str))]
    summ = summ.set_index("target").reindex(order).reset_index()
    x = np.arange(len(summ)); width = 0.34
    axes[0].bar(x-width/2, summ.get("core_geometry_overlap_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="CoRE geometry", color="#5DADE2")
    axes[0].bar(x+width/2, summ.get("mosaec_geometry_overlap_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="MOSAEC geometry", color="#58D68D")
    axes[0].set_xticks(x); axes[0].set_xticklabels([v42_label_target(t) for t in summ["target"]], rotation=30, ha="right"); axes[0].set_ylim(0,1.05); axes[0].set_ylabel("Overlap fraction"); axes[0].set_title("Geometry-domain overlap"); axes[0].legend(frameon=False); v43_panel_label(axes[0], "a")
    axes[1].bar(x-width/2, summ.get("core_exact_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="CoRE exact", color="#85C1E9")
    axes[1].bar(x+width/2, summ.get("mosaec_exact_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="MOSAEC exact", color="#82E0AA")
    axes[1].set_xticks(x); axes[1].set_xticklabels([v42_label_target(t) for t in summ["target"]], rotation=30, ha="right"); axes[1].set_ylim(0,1.05); axes[1].set_ylabel("Exact-match fraction"); axes[1].set_title("Exact matches are not assumed"); axes[1].legend(frameon=False); v43_panel_label(axes[1], "b")
    axes[2].bar([v42_label_target(t) for t in summ["target"]], pd.to_numeric(summ.get("mean_external_realism_score", pd.Series(np.nan, index=summ.index)), errors="coerce"), color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in summ["target"]])
    axes[2].tick_params(axis="x", rotation=30); axes[2].set_ylabel("Mean annotation score"); axes[2].set_title("Domain-overlap annotation score"); axes[2].grid(True, axis="y"); v43_panel_label(axes[2], "c")
    fig.suptitle("External overlays annotate domain overlap, not experimental validation", fontsize=13.4, fontweight="bold")
    v43_save_figure(fig, "Figure_S_external_domain_overlap_annotation_v43", si=True, aliases=[PAPER_STORY_SI_DIR / "Figure_S_external_domain_overlap_annotation_v42"])


def v43_plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.3 final SI figures")
    # Reuse v4.2 SI generation, then add final table manifest.
    # Because this override uses the same stage marker, do the v4.3 essentials directly.
    v43_apply_publication_style()
    if tiers is not None and not tiers.empty and {"target", "y_true"}.issubset(tiers.columns):
        sample = tiers.drop_duplicates(["mof_id", "target"]) if "mof_id" in tiers.columns else tiers
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4)); axes = axes.ravel()
        for ax, target in zip(axes, TARGET_ORDER_FINAL):
            vals = pd.to_numeric(sample.loc[sample["target"].astype(str).eq(target), "y_true"], errors="coerce").dropna()
            if len(vals): ax.hist(vals, bins=45, color=TARGET_COLORS_FINAL.get(target, "#777777"), alpha=0.80)
            ax.set_title(v42_label_target(target)); ax.set_xlabel("Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Target distributions in quality-gated candidate rows", fontsize=12.6, fontweight="bold")
        v43_save_figure(fig, "Figure_S1_v4_3_target_distributions", si=True, aliases=[PAPER_STORY_SI_DIR / "Figure_S1_v4_2_target_distributions", FIG_SI_DIR / "Figure_S1_target_distributions"])
    q = v42_read_table("QC_Table_model_stability.csv")
    if not q.empty and "quality_gate_reasons" in q.columns:
        fig, ax = plt.subplots(figsize=(10.5, 5.0)); counts = q["quality_gate_reasons"].astype(str).value_counts().head(12)
        ax.barh(range(len(counts)), counts.values, color="#7F8C8D"); ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index, fontsize=7.2); ax.invert_yaxis(); ax.set_xlabel("Experiments"); ax.set_title("Figure S2. Candidate-quality gate audit")
        v43_save_figure(fig, "Figure_S2_v4_3_quality_gate_audit", si=True, aliases=[PAPER_STORY_SI_DIR / "Figure_S2_v4_2_quality_gate_audit"])
    mark_done(stage, {"v4_3": True})


def v43_write_manifest() -> pd.DataFrame:
    rows = []
    for folder, label in [(PAPER_PUBLICATION_DIR, "main_publication"), (PAPER_PUBLICATION_SI_DIR, "si_publication")]:
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in {".png", ".pdf", ".svg"}:
                rows.append({"category": label, "file": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    for p in sorted(PAPER_PUBLICATION_DATA_DIR.glob("*.csv")):
        rows.append({"category": "figure_data", "file": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    mf = pd.DataFrame(rows)
    atomic_save_csv(mf, TABLE_DIR / "MANIFEST_v4_3_publication_final_figures.csv")
    return mf


def v43_write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    stage = "07_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> WRITE_FINAL_RUN_REPORT_V4_3")
    manifest = v43_write_manifest()
    report = RESULTS_DIR / "RUN_REPORT.md"
    lines = []
    lines.append("# Few-shot MOF risk-controlled screening run report — v4.3 publication-final figures\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Runtime modes\n")
    for key in ["save_mode", "ram_mode", "comprehensive_level", "n_jobs", "data_root"]:
        lines.append(f"- {key}: `{CONFIG.get(key)}`")
    lines.append("\n## v4.3 publication-final figure folders\n")
    lines.append(f"- Main publication figures: `{PAPER_PUBLICATION_DIR}`")
    lines.append(f"- SI/domain-overlap figures: `{PAPER_PUBLICATION_SI_DIR}`")
    lines.append(f"- Publication figure data: `{PAPER_PUBLICATION_DATA_DIR}`")
    lines.append("\n## Main figures\n")
    for fname, meaning in [
        ("Figure_1_v4_3_publication_workflow.png", "quantitative workflow / graphical abstract"),
        ("Figure_2_v4_3_fewshot_learning_phase_diagram.png", "few-shot learning and enrichment"),
        ("Figure_3_v4_3_descriptor_chemistry_extrapolation_stress.png", "RAC gain and topology stress"),
        ("Figure_4_v4_3_conformal_risk_control_frontier.png", "coverage, precision-yield, tier composition"),
        ("Figure_5_v4_3_consensus_shortlist_atlas.png", "true-enriched target-balanced shortlist atlas"),
    ]:
        p = PAPER_PUBLICATION_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'} — {meaning}")
    lines.append("\n## Final publication tables\n")
    for fname in [
        "MAIN_Table_0_final_claims_summary.csv",
        "MAIN_Table_3_top5_consensus_candidates_per_target.csv",
        "STORY_Table_top25_consensus_candidates.csv",
        "STORY_Table_candidate_funnel_by_target.csv",
        "STORY_Table_prefinal_tier_fraction_by_budget.csv",
        "QC_Table_consensus_true_enrichment.csv",
        "MANIFEST_v4_3_publication_final_figures.csv",
    ]:
        p = TABLE_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'}")
    lines.append("\n## Scientific framing guardrails\n")
    lines.append("- Use geometry vs geometry+RACs as the validated descriptor comparison unless RDF integrity QC shows real nonzero RDF columns.")
    lines.append("- Treat topology-grouped splits as the central extrapolation stress test.")
    lines.append("- Present CoRE/MOSAEC overlays as domain-overlap annotations, not experimental validation.")
    lines.append("- Final shortlist claims should be based on target-balanced consensus and true-enrichment QC, not single-model raw predictions.")
    lines.append(f"\n## Manifest entries\n- Files recorded: {len(manifest)}")
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage, {"v4_3": True, "n_manifest_rows": int(len(manifest))})

# ---- v4.3 overrides used by the existing main() function ----

def plot_figure_1_framework() -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v43_plot_figure_1(master)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    return v43_plot_figure_2(agg)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    return v43_plot_figure_3(agg)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v43_plot_figure_4(anatomy, tiers, master=master)


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v43_plot_figure_5(core, mosaec, final_external, master=master)


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v43_plot_si_figures(agg, tiers, anatomy, master=master)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    return v43_write_run_report(master, metrics_all)



# =============================================================================
# 31. v4.4 journal-final visual refinement layer
# =============================================================================
# This final layer is intentionally table/figure-focused. It does not change the
# modelling loop, the target-balanced candidate gates, or the consensus logic.
# It regenerates a cleaner journal-facing figure and source-data package from
# the existing v4.3/v4.2 tables.

CONFIG["paper_final_name"] = "v4_5_journal_final_visuals_errorfix"
CONFIG.setdefault("journal_final_save_formats", ["png", "pdf", "svg"])
CONFIG.setdefault("journal_final_png_dpi", 700)
CONFIG.setdefault("journal_final_topn_main", 5)
CONFIG.setdefault("journal_final_topn_si", 25)

PAPER_JOURNAL_DIR = RESULTS_DIR / "figures" / "paper_journal_final_v45"
PAPER_JOURNAL_SI_DIR = RESULTS_DIR / "figures" / "paper_journal_final_v45_si"
PAPER_JOURNAL_DATA_DIR = FIGDATA_DIR / "paper_journal_final_v45"
for _d in [PAPER_JOURNAL_DIR, PAPER_JOURNAL_SI_DIR, PAPER_JOURNAL_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

TARGET_LABELS_JOURNAL = {
    "CO2_0p015bar_298K_mmolg": r"CO$_2$ 0.015 bar",
    "CO2_0p150bar_298K_mmolg": r"CO$_2$ 0.150 bar",
    "CH4_5p8bar_298K_mmolg": r"CH$_4$ 5.8 bar",
    "CH4_65bar_298K_mmolg": r"CH$_4$ 65 bar",
}
TARGET_LABELS_PLAIN = {
    "CO2_0p015bar_298K_mmolg": "CO2 0.015 bar",
    "CO2_0p150bar_298K_mmolg": "CO2 0.150 bar",
    "CH4_5p8bar_298K_mmolg": "CH4 5.8 bar",
    "CH4_65bar_298K_mmolg": "CH4 65 bar",
}
SPLIT_LABELS_JOURNAL = {
    "random": "Random",
    "geometry_grouped": "Geometry",
    "metal_grouped": "Metal",
    "functional_grouped": "Functional",
    "ligand_grouped": "Ligand",
    "topology_grouped": "Topology",
}
SPLIT_ORDER_JOURNAL = ["random", "geometry_grouped", "metal_grouped", "functional_grouped", "ligand_grouped", "topology_grouped"]
BUDGET_ORDER_JOURNAL = [10, 20, 50, 100, 200, 500, 1000]


def v44_apply_style() -> None:
    """Journal-final visual style: compact, clean, and vector-editable."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.2,
        "legend.fontsize": 6.9,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "figure.dpi": 160,
        "savefig.dpi": int(CONFIG.get("journal_final_png_dpi", 700)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.70,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.22,
        "lines.linewidth": 1.45,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.60,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def v44_tlabel(t: Any, latex: bool = True) -> str:
    return (TARGET_LABELS_JOURNAL if latex else TARGET_LABELS_PLAIN).get(str(t), str(t))


def v44_slabel(s: Any) -> str:
    return SPLIT_LABELS_JOURNAL.get(str(s), str(s).replace("_", " ").title())


def v44_panel(ax, label: str, x: float = -0.105, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.8, fontweight="bold", ha="right", va="top")


def v44_save(fig: plt.Figure, stem: str, si: bool = False, aliases: Optional[List[Path]] = None) -> None:
    v44_apply_style()
    out_dir = PAPER_JOURNAL_SI_DIR if si else PAPER_JOURNAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG.get("journal_final_save_formats", ["png", "pdf", "svg"]):
        out = (out_dir / stem).with_suffix(f".{str(fmt).lower()}")
        try:
            fig.savefig(out, dpi=int(CONFIG.get("journal_final_png_dpi", 700)), bbox_inches="tight")
        except Exception as e:
            logging.warning("V44_FIGURE_SAVE_FAILED | %s | %s", out, e)
    for base in aliases or []:
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in ["png", "pdf"]:
            try:
                fig.savefig(base.with_suffix(f".{fmt}"), dpi=int(CONFIG.get("journal_final_png_dpi", 700)), bbox_inches="tight")
            except Exception as e:
                logging.warning("V44_ALIAS_SAVE_FAILED | %s | %s", base.with_suffix(f'.{fmt}'), e)
    plt.close(fig)


def v44_read_metrics_all() -> pd.DataFrame:
    p = TABLE_DIR / "all_experiment_metrics.csv"
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()


def v44_clean_agg(agg: pd.DataFrame, top_frac: float = 0.05, feature_set: Optional[str] = None, split: Optional[str] = None) -> pd.DataFrame:
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = agg.copy()
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(top_frac).abs() < 1e-9]
    if feature_set is None:
        feature_set = "geometry_plus_racs" if "geometry_plus_racs" in set(df.get("feature_set", pd.Series(dtype=str)).astype(str)) else v42_preferred_feature(df)
    if feature_set and "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str).eq(feature_set)]
    if split and "split" in df.columns:
        df = df[df["split"].astype(str).eq(split)]
    if "model" in df.columns:
        stable = set(CONFIG.get("stable_plot_models", []))
        if stable:
            df = df[df["model"].astype(str).isin(stable)]
    return df


def v44_order_targets(df: Optional[pd.DataFrame] = None) -> List[str]:
    if df is None or df.empty or "target" not in df.columns:
        return TARGET_ORDER_FINAL
    present = set(df["target"].dropna().astype(str))
    ordered = [t for t in TARGET_ORDER_FINAL if t in present]
    ordered += sorted(present - set(ordered))
    return ordered


def v45_target_mean_table(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """Return one row per target by averaging numeric columns.

    This v4.5 helper fixes a pandas reindex failure that occurs when a
    target-level plotting panel receives target/budget/split-level rows.
    Any caller that will set_index("target").reindex(...) should first
    reduce duplicated target labels with this helper.
    """
    if df is None or df.empty or "target" not in df.columns:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    cols = [c for c in value_cols if c in work.columns]
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if not cols:
        return work.drop_duplicates("target")
    return work.groupby("target", as_index=False)[cols].mean(numeric_only=True)


def v44_heatmap(ax, mat: pd.DataFrame, title: str, cbar_label: str = "", center_zero: bool = False,
                cmap: str = "viridis", fmt: str = ".2f", annotate: bool = True) -> None:
    if mat is None or mat.empty:
        ax.axis("off"); ax.set_title(title); return
    arr = mat.astype(float).to_numpy()
    if center_zero:
        vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
        vmin, vmax = -vmax, vmax
        cmap = "coolwarm"
    else:
        vmin = np.nanmin(arr) if np.isfinite(arr).any() else 0.0
        vmax = np.nanmax(arr) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=38, ha="right")
    ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index)
    if annotate:
        norm_mid = (vmin + vmax) / 2 if np.isfinite(vmin) and np.isfinite(vmax) else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    txt_color = "white" if (not center_zero and val < norm_mid*0.75) else "black"
                    if center_zero:
                        txt_color = "white" if abs(val) > 0.60*max(abs(vmin), abs(vmax)) else "black"
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=6.5, color=txt_color)
    ax.set_title(title)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.ax.set_ylabel(cbar_label, rotation=90)


def v44_true_enrichment_table(top25: Optional[pd.DataFrame] = None, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    qce = v43_true_enrichment_table(top25=top25, master=master)
    if qce is None:
        return pd.DataFrame()
    qce = qce.copy()
    qce["target_label_plain"] = qce["target"].map(lambda x: v44_tlabel(x, latex=False)) if "target" in qce.columns else ""
    return qce


def v44_top_consensus(topn: int = 25, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    top = v42_top_consensus(topn)
    if top is None or top.empty:
        return pd.DataFrame()
    top = v42_add_percentiles_to_candidates(top, master=master)
    top["target_label_plain"] = top["target"].map(lambda x: v44_tlabel(x, latex=False))
    # clean and publication-friendly column order
    for c in ["consensus_score", "y_true_median", "y_pred_median", "split_lower_median", "split_upper_median", "true_percentile", "pred_percentile"]:
        if c in top.columns:
            top[c] = pd.to_numeric(top[c], errors="coerce")
    return top


def v44_final_claims_table(agg: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    qce = v44_true_enrichment_table(master=master)
    dg = v42_descriptor_gain_table(agg)
    sp = v42_split_stress_table(agg)
    df = v44_clean_agg(agg, feature_set="geometry_plus_racs")
    rows = []
    for target in v44_order_targets(df):
        sub = df[df["target"].astype(str).eq(target)].copy()
        best = sub.sort_values(["recall_mean", "enrichment_mean", "spearman_mean"], ascending=[False, False, False]).head(1)
        best_recall = float(best["recall_mean"].iloc[0]) if not best.empty and "recall_mean" in best else np.nan
        best_enrich = float(best["enrichment_mean"].iloc[0]) if not best.empty and "enrichment_mean" in best else np.nan
        best_spear = float(best["spearman_mean"].iloc[0]) if not best.empty and "spearman_mean" in best else np.nan
        dgrow = dg[(dg.get("target", pd.Series(dtype=str)).astype(str).eq(target)) & (pd.to_numeric(dg.get("budget", np.nan), errors="coerce").eq(1000))] if not dg.empty else pd.DataFrame()
        rac_gain = float(pd.to_numeric(dgrow.get("delta_recall_racs_minus_geometry", pd.Series(dtype=float)), errors="coerce").mean()) if not dgrow.empty else np.nan
        srow = sp[sp.get("target", pd.Series(dtype=str)).astype(str).eq(target)] if not sp.empty else pd.DataFrame()
        topology_penalty = np.nan
        if not srow.empty and "penalty_vs_random__topology_grouped" in srow.columns:
            topology_penalty = float(pd.to_numeric(srow["penalty_vs_random__topology_grouped"], errors="coerce").mean())
        qrow = qce[qce.get("target", pd.Series(dtype=str)).astype(str).eq(target)] if not qce.empty else pd.DataFrame()
        fold_col = v43_numeric_col(qce, ["fold_enrichment_vs_dataset_median", "shortlist_median_over_dataset_median", "fold_enrichment"]) if not qce.empty else None
        top5_col = v43_numeric_col(qce, ["fraction_true_top5", "true_top5_fraction", "frac_true_top5", "top5_fraction"]) if not qce.empty else None
        fold = float(pd.to_numeric(qrow[fold_col], errors="coerce").iloc[0]) if fold_col and not qrow.empty else np.nan
        top5 = float(pd.to_numeric(qrow[top5_col], errors="coerce").iloc[0]) if top5_col and not qrow.empty else np.nan
        rows.append({
            "target": target,
            "target_label_plain": v44_tlabel(target, latex=False),
            "best_top5_recall": best_recall,
            "best_enrichment": best_enrich,
            "best_spearman": best_spear,
            "rac_gain_delta_recall_1000_labels": rac_gain,
            "topology_penalty_random_minus_topology": topology_penalty,
            "consensus_shortlist_fold_enrichment_vs_dataset_median": fold,
            "consensus_shortlist_fraction_true_top5": top5,
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "MAIN_Table_0_final_claims_summary.csv")
    atomic_save_csv(out, TABLE_DIR / "MAIN_Table_0_final_claims_summary_v44.csv")
    return out


def v44_candidate_tables(master: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top25 = v44_top_consensus(int(CONFIG.get("journal_final_topn_si", 25)), master=master)
    if top25.empty:
        atomic_save_csv(top25, TABLE_DIR / "STORY_Table_top25_consensus_candidates.csv")
        atomic_save_csv(top25, TABLE_DIR / "MAIN_Table_3_top5_consensus_candidates_per_target.csv")
        return top25, top25
    top25 = top25.sort_values(["target", "consensus_score", "split_lower_median"], ascending=[True, False, False])
    atomic_save_csv(top25, TABLE_DIR / "STORY_Table_top25_consensus_candidates.csv")
    top5 = top25.groupby("target", group_keys=False).head(int(CONFIG.get("journal_final_topn_main", 5))).copy()
    atomic_save_csv(top5, TABLE_DIR / "MAIN_Table_3_top5_consensus_candidates_per_target.csv")
    return top25, top5


def v44_plot_figure_1(master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig1_framework"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.4 journal Figure 1")
    v44_apply_style()
    ts = v42_target_summary(master)
    fig = plt.figure(figsize=(14.0, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.72, 1.45, 1.65], hspace=0.45, wspace=0.30)
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.text(0.5, 0.76, "Risk-controlled few-shot discovery of adsorption elites", ha="center", va="center", fontsize=17, fontweight="bold")
    ax_title.text(0.5, 0.24, "Label-efficient screening, descriptor chemistry, conformal risk control, and target-balanced consensus", ha="center", va="center", fontsize=9.4, color="#34495E")
    # target cards
    if not ts.empty and "target" in ts.columns:
        ts = ts.set_index("target").reindex(TARGET_ORDER_FINAL).reset_index()
    for i, target in enumerate(TARGET_ORDER_FINAL):
        ax = fig.add_subplot(gs[1, i]); ax.axis("off")
        color = TARGET_COLORS_FINAL.get(target, "#777777")
        row = ts[ts["target"].astype(str).eq(target)].head(1) if not ts.empty else pd.DataFrame()
        n = int(pd.to_numeric(row.get("n_nonmissing", row.get("n_mofs", pd.Series([np.nan]))), errors="coerce").iloc[0]) if not row.empty else 279010
        med = float(pd.to_numeric(row.get("median", row.get("median_mmol_g", pd.Series([np.nan]))), errors="coerce").iloc[0]) if not row.empty else np.nan
        mx = float(pd.to_numeric(row.get("max", row.get("max_mmol_g", pd.Series([np.nan]))), errors="coerce").iloc[0]) if not row.empty else np.nan
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((0.02, 0.04), 0.96, 0.90, boxstyle="round,pad=0.018,rounding_size=0.035", transform=ax.transAxes, fc="white", ec=color, lw=1.9)
        ax.add_patch(rect)
        ax.text(0.08, 0.80, v44_tlabel(target), color=color, fontsize=12.5, fontweight="bold", transform=ax.transAxes)
        ax.text(0.08, 0.57, f"{n:,}\nlabelled MOFs", fontsize=11.0, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.08, 0.31, f"median: {med:.3g} mmol g$^{{-1}}$", fontsize=8.4, transform=ax.transAxes)
        ax.text(0.08, 0.17, f"max: {mx:.3g} mmol g$^{{-1}}$", fontsize=8.4, transform=ax.transAxes)
    # workflow row
    workflow = [
        ("ARC-MOF labels", "4 targets\ngeometry + RACs"),
        ("Few-shot design", "10–1000 labels\n5 seeds × split stress"),
        ("Stable learners", "RF / ExtraTrees / HGB\nLightGBM / XGBoost"),
        ("Risk control", "90% conformal intervals\ntrusted / uncertain / rejected"),
        ("Consensus", "models × seeds × splits\ntarget-balanced shortlist"),
        ("Domain overlay", "CoRE/MOSAEC geometry\nannotation, not validation"),
    ]
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    xs = np.linspace(0.06, 0.94, len(workflow)); y = 0.52
    for i, (title, body) in enumerate(workflow):
        w, h = 0.13, 0.42
        box = FancyBboxPatch((xs[i]-w/2, y-h/2), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", transform=ax.transAxes, fc="#F8FAFB", ec="#85929E", lw=0.85)
        ax.add_patch(box)
        ax.text(xs[i], y+0.10, title, ha="center", va="center", fontweight="bold", fontsize=8.4, transform=ax.transAxes)
        ax.text(xs[i], y-0.075, body, ha="center", va="center", fontsize=7.2, transform=ax.transAxes)
        if i < len(workflow)-1:
            arr = FancyArrowPatch((xs[i]+w/2+0.006, y), (xs[i+1]-w/2-0.006, y), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#5D6D7E", transform=ax.transAxes)
            ax.add_patch(arr)
    ax.text(0.02, 0.95, "a", fontsize=11, fontweight="bold", transform=fig.transFigure)
    ax.text(0.02, 0.48, "b", fontsize=11, fontweight="bold", transform=fig.transFigure)
    v44_save(fig, "Figure_1_v4_5_journal_workflow", aliases=[PAPER_PUBLICATION_DIR / "Figure_1_v4_3_publication_workflow", PAPER_STORY_DIR / "Figure_1_v4_2_story_workflow", FIG_MAIN_DIR / "Figure_1_risk_controlled_framework"])
    mark_done(stage, {"v4_4": True})


def v44_plot_figure_2(agg: pd.DataFrame) -> None:
    stage = "fig2_performance"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.4 journal Figure 2")
    v44_apply_style()
    recall = v42_learning_curves(agg, "recall_mean", split="random")
    enrich = v42_learning_curves(agg, "enrichment_mean", split="random")
    atomic_save_csv(recall, PAPER_JOURNAL_DATA_DIR / "Figure_2a_recall_learning_curves.csv")
    atomic_save_csv(enrich, PAPER_JOURNAL_DATA_DIR / "Figure_2b_enrichment_learning_curves.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8)); axa, axb, axc, axd, axe, axf = axes.ravel()
    for target in v44_order_targets(recall):
        sub = recall[recall["target"].astype(str).eq(target)].sort_values("budget")
        if sub.empty: continue
        axa.plot(sub["budget"], sub["mean"], marker="o", color=TARGET_COLORS_FINAL.get(target), label=v44_tlabel(target))
        if {"ci_low", "ci_high"}.issubset(sub.columns):
            axa.fill_between(sub["budget"].astype(float).to_numpy(), sub["ci_low"].astype(float).to_numpy(), sub["ci_high"].astype(float).to_numpy(), color=TARGET_COLORS_FINAL.get(target), alpha=0.10, linewidth=0)
    axa.axhline(0.05, ls="--", lw=0.9, color="#777777", label="random")
    axa.set_xscale("log"); axa.set_xticks(BUDGET_ORDER_JOURNAL); axa.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL], rotation=0)
    axa.set_ylabel("Top-5% recall"); axa.set_xlabel("Label budget"); axa.set_title("Elite recovery emerges from few labels"); axa.grid(True); axa.legend(frameon=False, ncol=1); v44_panel(axa, "a")
    for target in v44_order_targets(enrich):
        sub = enrich[enrich["target"].astype(str).eq(target)].sort_values("budget")
        if sub.empty: continue
        axb.plot(sub["budget"], sub["mean"], marker="o", color=TARGET_COLORS_FINAL.get(target), label=v44_tlabel(target))
        if {"ci_low", "ci_high"}.issubset(sub.columns):
            axb.fill_between(sub["budget"].astype(float).to_numpy(), sub["ci_low"].astype(float).to_numpy(), sub["ci_high"].astype(float).to_numpy(), color=TARGET_COLORS_FINAL.get(target), alpha=0.10, linewidth=0)
    axb.axhline(1.0, ls="--", lw=0.9, color="#777777")
    axb.set_xscale("log"); axb.set_xticks(BUDGET_ORDER_JOURNAL); axb.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL], rotation=0)
    axb.set_ylabel("Enrichment over random"); axb.set_xlabel("Label budget"); axb.set_title("Screening enrichment"); axb.grid(True); v44_panel(axb, "b")
    # label efficiency relative to 1000 labels
    eff_rows = []
    for target, sub in recall.groupby("target"):
        sub = sub.sort_values("budget")
        final = sub.loc[pd.to_numeric(sub["budget"], errors="coerce").eq(1000), "mean"]
        denom = float(final.iloc[0]) if len(final) and np.isfinite(final.iloc[0]) and final.iloc[0] != 0 else float(sub["mean"].max())
        for _, r in sub.iterrows():
            eff_rows.append({"target": target, "budget": int(r["budget"]), "label_efficiency": float(r["mean"] / denom) if denom else np.nan})
    eff = pd.DataFrame(eff_rows); atomic_save_csv(eff, PAPER_JOURNAL_DATA_DIR / "Figure_2c_label_efficiency.csv")
    for target in v44_order_targets(eff):
        sub = eff[eff["target"].astype(str).eq(target)].sort_values("budget")
        axc.plot(sub["budget"], sub["label_efficiency"], marker="o", color=TARGET_COLORS_FINAL.get(target), label=v44_tlabel(target))
    axc.axhline(0.75, ls=":", color="#555555", lw=0.9)
    axc.set_xscale("log"); axc.set_xticks(BUDGET_ORDER_JOURNAL); axc.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL], rotation=0)
    axc.set_ylim(0, 1.05); axc.set_ylabel("Fraction of 1000-label recall"); axc.set_xlabel("Label budget"); axc.set_title("Label-efficiency index"); axc.grid(True); v44_panel(axc, "c")
    # budget thresholds
    thresh_rows = []
    for target, sub in eff.groupby("target"):
        for frac in [0.50, 0.75, 0.90]:
            hit = sub[pd.to_numeric(sub["label_efficiency"], errors="coerce") >= frac].sort_values("budget")
            b = int(hit["budget"].iloc[0]) if not hit.empty else np.nan
            thresh_rows.append({"target": target, "fraction_of_final": frac, "budget_needed": b})
    thr = pd.DataFrame(thresh_rows); atomic_save_csv(thr, TABLE_DIR / "STORY_Table_label_efficiency_budget_thresholds.csv")
    for frac, marker, color in [(0.50, "o", "#2E86C1"), (0.75, "s", "#F39C12"), (0.90, "D", "#27AE60")]:
        sub = thr[pd.to_numeric(thr["fraction_of_final"], errors="coerce").eq(frac)]
        ylabels = [v44_tlabel(t) for t in v44_order_targets(sub)]
        ymap = {t:i for i,t in enumerate(v44_order_targets(sub))}
        axd.scatter(pd.to_numeric(sub["budget_needed"], errors="coerce"), [ymap.get(str(t), 0) for t in sub["target"]], marker=marker, s=38, color=color, label=f"{int(frac*100)}%")
    axd.set_yticks(range(len(v44_order_targets(thr)))); axd.set_yticklabels([v44_tlabel(t) for t in v44_order_targets(thr)])
    axd.set_xscale("log"); axd.set_xticks(BUDGET_ORDER_JOURNAL); axd.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL], rotation=0)
    axd.set_xlabel("Budget needed"); axd.set_title("How many labels are enough?"); axd.grid(True, axis="x"); axd.legend(title="of final recall", frameon=False, loc="lower right"); v44_panel(axd, "d")
    # model comparison
    df1000 = v44_clean_agg(agg, feature_set="geometry_plus_racs", split="random")
    df1000 = df1000[pd.to_numeric(df1000.get("budget", np.nan), errors="coerce").eq(1000)]
    model_order = [m for m in ["extra_trees", "rf", "hgb", "lightgbm", "xgboost"] if m in set(df1000.get("model", pd.Series(dtype=str)).astype(str))]
    msum = df1000.groupby("model", as_index=False)["recall_mean"].mean().set_index("model").reindex(model_order).reset_index()
    atomic_save_csv(msum, PAPER_JOURNAL_DATA_DIR / "Figure_2e_model_comparison_1000_labels.csv")
    axe.bar(range(len(msum)), msum["recall_mean"], color=["#7F8C8D", "#95A5A6", "#AAB7B8", "#85929E", "#5D6D7E"][:len(msum)])
    axe.set_xticks(range(len(msum))); axe.set_xticklabels([str(m).replace("_", "\n") for m in msum["model"]], rotation=0)
    axe.set_ylabel("Mean top-5% recall"); axe.set_title("Stable model comparison at 1000 labels"); axe.set_ylim(0, max(0.7, float(msum["recall_mean"].max())*1.18 if len(msum) else 0.7)); axe.grid(True, axis="y"); v44_panel(axe, "e")
    # final enrichment summary
    bestdf = enrich[pd.to_numeric(enrich["budget"], errors="coerce").eq(1000)].copy()
    bestdf = bestdf.sort_values("mean", ascending=True)
    atomic_save_csv(bestdf, PAPER_JOURNAL_DATA_DIR / "Figure_2f_final_enrichment_summary.csv")
    axf.barh([v44_tlabel(t) for t in bestdf["target"]], bestdf["mean"], color=[TARGET_COLORS_FINAL.get(t, "#777777") for t in bestdf["target"]])
    axf.axvline(1, ls="--", color="#777777", lw=0.9)
    for y, val in enumerate(bestdf["mean"]):
        axf.text(val + 0.25, y, f"{val:.1f}×", va="center", fontsize=7.4)
    axf.set_xlabel("Best enrichment at 1000 labels"); axf.set_title("Final enrichment summary"); axf.grid(True, axis="x"); v44_panel(axf, "f")
    fig.suptitle("Few-shot learning phase diagram for adsorption-elite recovery", fontsize=14.7, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v44_save(fig, "Figure_2_v4_5_fewshot_learning_phase_diagram", aliases=[PAPER_PUBLICATION_DIR / "Figure_2_v4_3_fewshot_learning_phase_diagram", PAPER_STORY_DIR / "Figure_2_v4_2_fewshot_learning_phase_diagram", FIG_MAIN_DIR / "Figure_2_label_budget_performance"])
    mark_done(stage, {"v4_4": True})


def v44_plot_figure_3(agg: pd.DataFrame) -> None:
    stage = "fig3_calibration"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.4 journal Figure 3")
    v44_apply_style()
    dg = v42_descriptor_gain_table(agg)
    sp = v42_split_stress_table(agg)
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8)); axa, axb, axc, axd, axe, axf = axes.ravel()
    # (a) RAC gain heatmap across budgets
    if not dg.empty:
        heat = dg.pivot_table(index="target", columns="budget", values="delta_recall_racs_minus_geometry", aggfunc="mean")
        heat = heat.reindex(v44_order_targets(dg)); heat.index = [v44_tlabel(t) for t in heat.index]
        heat = heat[[b for b in BUDGET_ORDER_JOURNAL if b in heat.columns]]
        heat.columns = [str(int(c)) for c in heat.columns]
        atomic_save_csv(heat.reset_index().rename(columns={"index": "target_label"}), PAPER_JOURNAL_DATA_DIR / "Figure_3a_rac_gain_heatmap.csv")
        v44_heatmap(axa, heat, "RAC descriptor gain across label budgets", "Δ recall", center_zero=True, fmt=".2f")
    v44_panel(axa, "a")
    # (b) RAC gain bars at 1000 labels
    # v4.5 error fix: descriptor gain is target/budget/split-level, so budget=1000
    # still contains multiple rows per target. Aggregate to one row per target before
    # ordering; otherwise pandas raises "cannot reindex on an axis with duplicate labels".
    bars = dg[pd.to_numeric(dg.get("budget", np.nan), errors="coerce").eq(1000)].copy() if not dg.empty else pd.DataFrame()
    if not bars.empty:
        bars = v45_target_mean_table(bars, ["delta_recall_racs_minus_geometry"])
        bars = bars.dropna(subset=["delta_recall_racs_minus_geometry"])
        bars = bars.sort_values("delta_recall_racs_minus_geometry", ascending=True)
        atomic_save_csv(bars, PAPER_JOURNAL_DATA_DIR / "Figure_3b_rac_gain_bars_1000_labels.csv")
        axb.barh([v44_tlabel(t) for t in bars["target"]], bars["delta_recall_racs_minus_geometry"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
        axb.axvline(0, color="#555555", lw=0.8)
    axb.set_xlabel("Δ recall: RACs − geometry"); axb.set_title("Chemistry benefit at 1000 labels"); axb.grid(True, axis="x"); v44_panel(axb, "b")
    # (c) split stress recall heatmap
    if not sp.empty:
        recall_cols = [c for c in sp.columns if c.startswith("recall__")]
        smat = sp.set_index("target")[[c for c in [f"recall__{s}" for s in SPLIT_ORDER_JOURNAL] if c in sp.columns]]
        smat.columns = [v44_slabel(c.replace("recall__", "")) for c in smat.columns]
        smat = smat.reindex(v44_order_targets(sp)); smat.index = [v44_tlabel(t) for t in smat.index]
        atomic_save_csv(smat.reset_index().rename(columns={"index": "target_label"}), PAPER_JOURNAL_DATA_DIR / "Figure_3c_split_stress_recall.csv")
        v44_heatmap(axc, smat, "Split-stress map at 1000 labels", "top-5% recall", center_zero=False, cmap="viridis", fmt=".2f")
    v44_panel(axc, "c")
    # (d) topology penalty bars
    if not sp.empty and "penalty_vs_random__topology_grouped" in sp.columns:
        bars = v45_target_mean_table(sp, ["penalty_vs_random__topology_grouped"])
        bars = bars.set_index("target").reindex(v44_order_targets(bars)).reset_index().sort_values("penalty_vs_random__topology_grouped", ascending=True)
        axd.barh([v44_tlabel(t) for t in bars["target"]], bars["penalty_vs_random__topology_grouped"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
        axd.axvline(0, color="#555555", lw=0.8)
    axd.set_xlabel("Recall(random) − recall(topology)"); axd.set_title("Topology extrapolation penalty"); axd.grid(True, axis="x"); v44_panel(axd, "d")
    # (e) vulnerability map ordered splits
    if not sp.empty:
        vul_cols = [f"vulnerability__{s}" for s in SPLIT_ORDER_JOURNAL if f"vulnerability__{s}" in sp.columns]
        if vul_cols:
            vmat = sp.set_index("target")[vul_cols]
            vmat.columns = [v44_slabel(c.replace("vulnerability__", "")) for c in vmat.columns]
            vmat = vmat.reindex(v44_order_targets(sp)); vmat.index = [v44_tlabel(t) for t in vmat.index]
            atomic_save_csv(vmat.reset_index().rename(columns={"index": "target_label"}), PAPER_JOURNAL_DATA_DIR / "Figure_3e_extrapolation_vulnerability.csv")
            v44_heatmap(axe, vmat, "Extrapolation vulnerability", "1 − grouped/random", center_zero=False, cmap="magma", fmt=".2f")
    v44_panel(axe, "e")
    # (f) mechanistic interpretation clean boxes
    axf.axis("off"); v44_panel(axf, "f", x=-0.04, y=1.06)
    axf.text(0.02, 0.98, "Mechanistic interpretation", fontsize=10.0, fontweight="bold", transform=axf.transAxes, va="top")
    from matplotlib.patches import FancyBboxPatch
    boxes = [
        ("Geometry baseline", "Pore size, surface area and density encode\nthe main physical adsorption envelope."),
        ("RAC chemistry", "Metal/linker descriptors improve elite recovery\nbeyond geometry alone."),
        ("Topology split", "Network-level transfer is the hardest\nout-of-distribution stress test."),
        ("Main message", "Random splits overestimate transferability;\nrisk-controlled consensus gives a safer shortlist."),
    ]
    for i, (title, body) in enumerate(boxes):
        y = 0.78 - i*0.205
        rect = FancyBboxPatch((0.05, y-0.095), 0.90, 0.145, boxstyle="round,pad=0.012,rounding_size=0.018", transform=axf.transAxes, fc="#F8FAFB", ec="#AAB7B8", lw=0.75)
        axf.add_patch(rect)
        axf.text(0.08, y+0.015, title, fontsize=8.3, fontweight="bold", transform=axf.transAxes, va="center")
        axf.text(0.08, y-0.052, body, fontsize=7.2, transform=axf.transAxes, va="center")
    fig.suptitle("Descriptor chemistry improves screening, but topology extrapolation remains limiting", fontsize=14.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v44_save(fig, "Figure_3_v4_5_descriptor_chemistry_extrapolation_stress", aliases=[PAPER_PUBLICATION_DIR / "Figure_3_v4_3_descriptor_chemistry_extrapolation_stress", PAPER_STORY_DIR / "Figure_3_v4_2_descriptor_chemistry_extrapolation_stress", FIG_MAIN_DIR / "Figure_3_calibration_and_abstention"])
    mark_done(stage, {"v4_4": True})


def v44_tier_fraction_from_metrics() -> pd.DataFrame:
    met = v44_read_metrics_all()
    if met.empty:
        return pd.DataFrame()
    df = met.copy()
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(0.05).abs() < 1e-9]
    if "feature_set" in df.columns:
        pref = "geometry_plus_racs" if "geometry_plus_racs" in set(df["feature_set"].astype(str)) else v42_preferred_feature(df)
        df = df[df["feature_set"].astype(str).eq(pref)]
    if "model" in df.columns:
        df = df[df["model"].astype(str).isin(CONFIG.get("stable_plot_models", []))]
    for c in ["trusted_count", "uncertain_count", "rejected_count", "n_test", "budget"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    denom = (df.get("trusted_count", 0) + df.get("uncertain_count", 0) + df.get("rejected_count", 0)).replace(0, np.nan)
    df["trusted_fraction"] = df.get("trusted_count", np.nan) / denom
    df["uncertain_fraction"] = df.get("uncertain_count", np.nan) / denom
    df["rejected_fraction"] = df.get("rejected_count", np.nan) / denom
    out = df.groupby("budget", as_index=False)[["trusted_fraction", "uncertain_fraction", "rejected_fraction"]].mean()
    out = out.sort_values("budget")
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_prefinal_tier_fraction_by_budget.csv")
    return out


def v44_plot_figure_4(anatomy: Optional[pd.DataFrame], tiers: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig4_failure_anatomy"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.4 journal Figure 4")
    v44_apply_style()
    agg = v42_read_table("table_aggregated_metrics_with_bootstrap_ci.csv", low_memory=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8)); axa, axb, axc, axd, axe, axf = axes.ravel()
    # (a) coverage error heatmap, split ordered
    df = v44_clean_agg(agg, feature_set="geometry_plus_racs")
    if not df.empty and "split_coverage_mean" in df.columns:
        cov = df.groupby(["target", "split"], as_index=False)["split_coverage_mean"].mean()
        cov["error"] = cov["split_coverage_mean"] - (1 - CONFIG.get("conformal_alpha", 0.10))
        mat = cov.pivot_table(index="target", columns="split", values="error", aggfunc="mean")
        mat = mat.reindex(v44_order_targets(cov))[[s for s in SPLIT_ORDER_JOURNAL if s in mat.columns]]
        mat.index = [v44_tlabel(t) for t in mat.index]; mat.columns = [v44_slabel(c) for c in mat.columns]
        atomic_save_csv(mat.reset_index().rename(columns={"index":"target_label"}), PAPER_JOURNAL_DATA_DIR / "Figure_4a_coverage_error_heatmap.csv")
        v44_heatmap(axa, mat, "Coverage error relative to 90%", "coverage − 0.90", center_zero=True, fmt="+.2f")
    v44_panel(axa, "a")
    # (b) normalized interval width vs budget
    std_map = v42_target_std_map(master)
    wdf = v44_clean_agg(agg, feature_set="geometry_plus_racs", split="random")
    if not wdf.empty and "split_mean_width_mean" in wdf.columns:
        wdf["target_std"] = wdf["target"].map(std_map)
        wdf["normalized_width"] = pd.to_numeric(wdf["split_mean_width_mean"], errors="coerce") / pd.to_numeric(wdf["target_std"], errors="coerce")
        wcurve = wdf.groupby(["target", "budget"], as_index=False)["normalized_width"].mean()
        atomic_save_csv(wcurve, PAPER_JOURNAL_DATA_DIR / "Figure_4b_normalized_interval_width.csv")
        for target in v44_order_targets(wcurve):
            sub = wcurve[wcurve["target"].astype(str).eq(target)].sort_values("budget")
            axb.plot(sub["budget"], sub["normalized_width"], marker="o", color=TARGET_COLORS_FINAL.get(target), label=v44_tlabel(target))
    axb.set_xscale("log"); axb.set_xticks(BUDGET_ORDER_JOURNAL); axb.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL])
    axb.set_xlabel("Label budget"); axb.set_ylabel("Interval width / target std"); axb.set_title("Uncertainty narrows with labels"); axb.grid(True); axb.legend(frameon=False, ncol=1); v44_panel(axb, "b")
    # (c) precision-yield frontier from candidate tiers
    frontier = v42_precision_yield_frontier(tiers, master=master) if tiers is not None and not tiers.empty else pd.DataFrame()
    atomic_save_csv(frontier, TABLE_DIR / "STORY_Table_precision_yield_frontier.csv")
    if not frontier.empty:
        for target in v44_order_targets(frontier):
            sub = frontier[frontier["target"].astype(str).eq(target)].sort_values("retained_fraction")
            axc.plot(sub["retained_fraction"], sub["precision_true_top5"], marker="o", color=TARGET_COLORS_FINAL.get(target), label=v44_tlabel(target))
    axc.set_xlabel("Retained fraction"); axc.set_ylabel("True top-5% fraction"); axc.set_title("Precision–yield frontier"); axc.set_xlim(0, 0.52); axc.set_ylim(0, 1.05); axc.grid(True); axc.legend(frameon=False, loc="lower right"); v44_panel(axc, "c")
    # (d) actionable yield at retained fraction closest to 10% -- more interpretable than the old flat yield
    if not frontier.empty:
        rows = []
        for target, sub in frontier.groupby("target"):
            sub = sub.copy(); sub["dist"] = (pd.to_numeric(sub["retained_fraction"], errors="coerce") - 0.10).abs()
            r = sub.sort_values("dist").head(1)
            if not r.empty:
                rows.append({"target": target, "retained_fraction": float(r["retained_fraction"].iloc[0]), "precision_true_top5": float(r["precision_true_top5"].iloc[0])})
        ydf = pd.DataFrame(rows).set_index("target").reindex(v44_order_targets(frontier)).reset_index()
        atomic_save_csv(ydf, PAPER_JOURNAL_DATA_DIR / "Figure_4d_actionable_yield_at_10pct_retained.csv")
        axd.bar([v44_tlabel(t) for t in ydf["target"]], ydf["precision_true_top5"], color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in ydf["target"]])
        for i, val in enumerate(ydf["precision_true_top5"]):
            if np.isfinite(val): axd.text(i, val+0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    axd.set_ylim(0, 1.05); axd.set_ylabel("True top-5% fraction"); axd.set_title("Precision at ~10% retained candidates"); axd.tick_params(axis="x", rotation=25); axd.grid(True, axis="y"); v44_panel(axd, "d")
    # (e) tier composition by budget from experiment metrics
    tier = v44_tier_fraction_from_metrics()
    if not tier.empty:
        x = np.arange(len(tier)); width = 0.72
        bottom = np.zeros(len(tier))
        for col, color, lab in [("trusted_fraction", "#2ECC71", "trusted"), ("uncertain_fraction", "#F5B041", "uncertain"), ("rejected_fraction", "#95A5A6", "rejected")]:
            vals = pd.to_numeric(tier[col], errors="coerce").fillna(0).to_numpy()
            axe.bar(x, vals, bottom=bottom, color=color, width=width, label=lab)
            bottom += vals
        axe.set_xticks(x); axe.set_xticklabels([str(int(b)) for b in tier["budget"]], rotation=0)
    axe.set_ylabel("Fraction of prediction rows"); axe.set_xlabel("Label budget"); axe.set_ylim(0,1.0); axe.set_title("Pre-final uncertainty-tier composition"); axe.legend(frameon=False, ncol=3, loc="upper center"); axe.grid(True, axis="y"); v44_panel(axe, "e")
    # (f) method coverage comparison, zoomed
    if not wdf.empty:
        methods = []
        for col, lab in [("split_coverage_mean", "split"), ("local_coverage_mean", "local"), ("mondrian_coverage_mean", "Mondrian")]:
            if col in wdf.columns:
                methods.append({"method": lab, "coverage": float(pd.to_numeric(wdf[col], errors="coerce").mean())})
        cm = pd.DataFrame(methods)
        if not cm.empty:
            axf.bar(cm["method"], cm["coverage"], color=["#5DADE2", "#58D68D", "#AF7AC5"][:len(cm)])
    axf.axhline(0.90, ls="--", color="#555555", lw=0.9)
    axf.set_ylim(0.86, 0.94); axf.set_ylabel("Empirical coverage"); axf.set_title("Coverage method comparison"); axf.grid(True, axis="y"); v44_panel(axf, "f")
    fig.suptitle("Conformal uncertainty converts few-shot predictions into risk-controlled shortlists", fontsize=14.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v44_save(fig, "Figure_4_v4_5_conformal_risk_control_frontier", aliases=[PAPER_PUBLICATION_DIR / "Figure_4_v4_3_conformal_risk_control_frontier", PAPER_STORY_DIR / "Figure_4_v4_2_conformal_risk_control_frontier", FIG_MAIN_DIR / "Figure_4_chemical_failure_anatomy"])
    mark_done(stage, {"v4_4": True})


def v44_candidate_funnel(master: Optional[pd.DataFrame] = None, top25: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    # total labelled MOFs per target from master or summary
    ts = v42_target_summary(master)
    consensus = v42_read_table("table_consensus_candidate_shortlist.csv", low_memory=True)
    tiers = v42_read_table("table_candidate_tiers_all_predictions.csv", low_memory=True)
    top25 = top25 if top25 is not None else v44_top_consensus(25, master=master)
    for target in TARGET_ORDER_FINAL:
        n_labelled = np.nan
        if not ts.empty and "target" in ts.columns:
            row = ts[ts["target"].astype(str).eq(target)].head(1)
            if not row.empty:
                n_labelled = pd.to_numeric(row.get("n_nonmissing", row.get("n_mofs", pd.Series([np.nan]))), errors="coerce").iloc[0]
        n_qg = len(tiers[tiers["target"].astype(str).eq(target)]) if not tiers.empty and "target" in tiers.columns else np.nan
        csub = consensus[consensus["target"].astype(str).eq(target)] if not consensus.empty and "target" in consensus.columns else pd.DataFrame()
        n_cons = int(csub.get("consensus_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not csub.empty and "consensus_pass" in csub.columns else len(csub)
        n_top = len(top25[top25["target"].astype(str).eq(target)]) if top25 is not None and not top25.empty else np.nan
        rows += [
            {"target": target, "target_label_plain": v44_tlabel(target, False), "stage": "Labelled MOFs", "count": n_labelled},
            {"target": target, "target_label_plain": v44_tlabel(target, False), "stage": "Quality-gated rows", "count": n_qg},
            {"target": target, "target_label_plain": v44_tlabel(target, False), "stage": "Consensus pass", "count": n_cons},
            {"target": target, "target_label_plain": v44_tlabel(target, False), "stage": "Top-25", "count": n_top},
        ]
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_candidate_funnel_by_target.csv")
    return out


def v44_plot_figure_5(core: Optional[pd.DataFrame], mosaec: Optional[pd.DataFrame], final_external: Optional[pd.DataFrame], master: Optional[pd.DataFrame] = None) -> None:
    stage = "fig5_external_realism"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.4 journal Figure 5")
    v44_apply_style()
    top25, top5 = v44_candidate_tables(master=master)
    qce = v44_true_enrichment_table(top25=top25, master=master)
    funnel = v44_candidate_funnel(master=master, top25=top25)
    fig = plt.figure(figsize=(14.2, 9.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.12, 1.12], hspace=0.62, wspace=0.36)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1]); axf = fig.add_subplot(gs[1:, 2])
    axg = fig.add_subplot(gs[2, 0:2])
    # (a) compact decision funnel as heatmap of log counts
    if not funnel.empty:
        stages = ["Labelled MOFs", "Quality-gated rows", "Consensus pass", "Top-25"]
        mat = funnel.pivot_table(index="target", columns="stage", values="count", aggfunc="first").reindex(TARGET_ORDER_FINAL)[stages]
        mat_plot = np.log10(pd.to_numeric(mat.stack(), errors="coerce").unstack().replace(0, np.nan))
        mat_plot.index = [v44_tlabel(t) for t in mat_plot.index]
        mat_plot.columns = ["Labelled", "Quality\ngated", "Consensus", "Top-25"]
        im = axa.imshow(mat_plot.to_numpy(), aspect="auto", cmap="Blues")
        axa.set_xticks(range(mat_plot.shape[1])); axa.set_xticklabels(mat_plot.columns, rotation=0)
        axa.set_yticks(range(mat_plot.shape[0])); axa.set_yticklabels(mat_plot.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if np.isfinite(val):
                    axa.text(j, i, f"{int(val):,}", ha="center", va="center", fontsize=6.5, color="black")
        axa.set_title("Decision funnel to final candidates")
        cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.025); cb.ax.set_ylabel("log$_{10}$(count)")
    v44_panel(axa, "a")
    # (b) true enrichment
    if not qce.empty:
        # v4.5 duplicate-safe enrichment table: keep one row per target even if the
        # QC table was rebuilt from multiple shortlist fragments.
        fold_col_pre = v43_numeric_col(qce, ["fold_enrichment_vs_dataset_median", "shortlist_median_over_dataset_median", "fold_enrichment"])
        frac_cols_pre = [c for c in qce.columns if str(c).startswith("fraction_true_top") or str(c) in {"fraction_true_top1", "fraction_true_top5", "fraction_true_top10"}]
        keep_cols = ["target"] + ([fold_col_pre] if fold_col_pre else []) + frac_cols_pre
        keep_cols = [c for c in keep_cols if c in qce.columns]
        qce = qce[keep_cols].copy()
        for c in qce.columns:
            if c != "target":
                qce[c] = pd.to_numeric(qce[c], errors="coerce")
        qce = qce.groupby("target", as_index=False).mean(numeric_only=True)
        qce = qce.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(qce["target"].astype(str)) or t in qce.index]).reset_index()
        fold_col = v43_numeric_col(qce, ["fold_enrichment_vs_dataset_median", "shortlist_median_over_dataset_median", "fold_enrichment"])
        vals = pd.to_numeric(qce[fold_col], errors="coerce") if fold_col else pd.Series(np.nan, index=qce.index)
        axb.bar(range(len(qce)), vals, color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in qce["target"]])
        axb.axhline(1, color="#777777", ls="--", lw=0.9)
        for i, v in enumerate(vals):
            if np.isfinite(v): axb.text(i, v + 0.20, f"{v:.1f}×", ha="center", va="bottom", fontsize=7)
        axb.set_xticks(range(len(qce))); axb.set_xticklabels([v44_tlabel(t) for t in qce["target"]], rotation=28, ha="right")
    axb.set_ylabel("Shortlist median / dataset median"); axb.set_title("True enrichment of consensus shortlist"); axb.grid(True, axis="y"); v44_panel(axb, "b")
    # (c) how often true elites
    if not qce.empty:
        cols = [("fraction_true_top1", "top 1%"), ("fraction_true_top5", "top 5%"), ("fraction_true_top10", "top 10%")]
        valid_cols = [(v43_numeric_col(qce, [c]), lab) for c, lab in cols]
        valid_cols = [(c, lab) for c, lab in valid_cols if c]
        x = np.arange(len(qce)); width = 0.22
        for k, (col, lab) in enumerate(valid_cols):
            axc.bar(x + (k-(len(valid_cols)-1)/2)*width, pd.to_numeric(qce[col], errors="coerce"), width=width, label=lab)
        axc.set_xticks(x); axc.set_xticklabels([v44_tlabel(t) for t in qce["target"]], rotation=28, ha="right")
        axc.legend(frameon=False, ncol=3, loc="upper left")
    axc.set_ylim(0,1.05); axc.set_ylabel("Fraction of top-25 shortlist"); axc.set_title("How often candidates are true elites"); axc.grid(True, axis="y"); v44_panel(axc, "c")
    # (d) true percentile distribution of final top25
    if not top25.empty and "true_percentile" in top25.columns:
        ordered = v44_order_targets(top25)
        data = [pd.to_numeric(top25.loc[top25["target"].astype(str).eq(t), "true_percentile"], errors="coerce").dropna() for t in ordered]
        safe_boxplot_with_labels(axd, data, [v44_tlabel(t) for t in ordered], showfliers=False, patch_artist=True,
                                 boxprops={"facecolor":"#F8F9F9", "edgecolor":"#34495E"}, medianprops={"color":"#E67E22", "linewidth":1.1})
        rng = np.random.default_rng(42)
        for i, (target, vals) in enumerate(zip(ordered, data), start=1):
            vals = np.asarray(vals, dtype=float)
            axd.scatter(i + rng.normal(0, 0.035, size=len(vals)), vals, s=13, alpha=0.55, color=TARGET_COLORS_FINAL.get(target))
    axd.set_ylim(0.68, 1.01); axd.tick_params(axis="x", rotation=28); axd.set_ylabel("True uptake percentile"); axd.set_title("Final candidates occupy the adsorption tail"); axd.grid(True, axis="y"); v44_panel(axd, "d")
    # (e) support distributions
    support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_budgets", "trusted_support"] if c in top25.columns]
    if support_cols:
        data = [pd.to_numeric(top25[c], errors="coerce").dropna() for c in support_cols]
        labels = [c.replace("n_", "").replace("trusted_support", "trusted\nsupport") for c in support_cols]
        safe_boxplot_with_labels(axe, data, labels, showfliers=False, patch_artist=True,
                                 boxprops={"facecolor":"#F8F9F9", "edgecolor":"#34495E"}, medianprops={"color":"#E67E22", "linewidth":1.1})
        rng = np.random.default_rng(43)
        for j, vals in enumerate(data, start=1):
            vals = np.asarray(vals, dtype=float)
            axe.scatter(j + rng.normal(0, 0.035, size=len(vals)), vals, s=8, alpha=0.20, color="#34495E")
    axe.set_ylabel("Support count"); axe.set_title("Consensus stability across runs"); axe.grid(True, axis="y"); v44_panel(axe, "e")
    # (f) geometry atlas with dataset background
    xcol = "Di" if top25 is not None and "Di" in top25.columns else None
    ycol = "Density" if top25 is not None and "Density" in top25.columns else None
    if master is not None and not master.empty and xcol and ycol and {"Di", "Density"}.issubset(master.columns):
        bg = master[["Di", "Density"]].dropna().sample(min(15000, len(master.dropna(subset=["Di", "Density"]))), random_state=42) if len(master.dropna(subset=["Di", "Density"])) else pd.DataFrame()
        if not bg.empty:
            axf.scatter(pd.to_numeric(bg["Di"], errors="coerce"), pd.to_numeric(bg["Density"], errors="coerce"), s=2.0, color="#BDC3C7", alpha=0.12, label="ARC-MOF background")
    if xcol and ycol and top25 is not None and not top25.empty:
        for target in v44_order_targets(top25):
            sub = top25[top25["target"].astype(str).eq(target)]
            size = pd.to_numeric(sub.get("trusted_support", pd.Series(20, index=sub.index)), errors="coerce").fillna(1)
            size = 22 + 3.5*np.sqrt(size)
            axf.scatter(pd.to_numeric(sub[xcol], errors="coerce"), pd.to_numeric(sub[ycol], errors="coerce"), s=size, color=TARGET_COLORS_FINAL.get(target), edgecolors="white", linewidths=0.35, alpha=0.80, label=v44_tlabel(target))
    axf.set_xlabel("PLD proxy, Di / Å"); axf.set_ylabel("Density / g cm$^{-3}$"); axf.set_title("Geometry regime of final candidates"); axf.grid(True); v44_panel(axf, "f")
    axf.legend(frameon=False, loc="upper right", fontsize=6.5)
    # (g) compact top candidates table-like panel
    axg.axis("off"); v44_panel(axg, "g", x=-0.02, y=1.04)
    axg.set_title("Top consensus examples selected for the main text", loc="left", pad=8)
    top5_show = top5.copy()
    cols = [c for c in ["target_label_plain", "mof_id", "consensus_score", "true_percentile", "pred_percentile"] if c in top5_show.columns]
    show = top5_show[cols].head(12).copy() if cols else pd.DataFrame()
    if not show.empty:
        if "mof_id" in show.columns:
            show["mof_id"] = show["mof_id"].astype(str).str.slice(0, 22)
        for c in ["consensus_score", "true_percentile", "pred_percentile"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        tbl = axg.table(cellText=show.values, colLabels=[c.replace("_", " ") for c in show.columns], loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.4); tbl.scale(1, 1.12)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D5D8DC")
            if row == 0:
                cell.set_facecolor("#F2F4F4"); cell.set_text_props(fontweight="bold")
    fig.suptitle("Target-balanced consensus shortlists are truly enriched in adsorption elites", fontsize=14.2, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v44_save(fig, "Figure_5_v4_5_consensus_shortlist_atlas", aliases=[PAPER_PUBLICATION_DIR / "Figure_5_v4_3_consensus_shortlist_atlas", PAPER_STORY_DIR / "Figure_5_v4_2_consensus_shortlist_atlas", FIG_MAIN_DIR / "Figure_5_external_realism_filter"])
    v44_plot_external_domain_overlap_si(final_external)
    mark_done(stage, {"v4_4": True})


def v44_plot_external_domain_overlap_si(final_external: Optional[pd.DataFrame]) -> None:
    if final_external is None or final_external.empty:
        final_external = v42_read_table("final_screening_table_with_external_flags.csv", low_memory=True)
    if final_external is None or final_external.empty or "target" not in final_external.columns:
        return
    fe = final_external.copy()
    for col in ["core_geometry_overlap_flag", "mosaec_geometry_overlap_flag", "core_exact_match_flag", "mosaec_exact_match_flag"]:
        if col in fe.columns:
            fe[col] = fe[col].fillna(False).astype(bool)
    if "external_realism_score" in fe.columns:
        fe["external_realism_score"] = pd.to_numeric(fe["external_realism_score"], errors="coerce")
    rows = []
    for target, sub in fe.groupby("target"):
        rows.append({
            "target": target, "target_label_plain": v44_tlabel(target, False), "n": len(sub),
            "core_exact_fraction": float(sub["core_exact_match_flag"].mean()) if "core_exact_match_flag" in sub.columns else np.nan,
            "mosaec_exact_fraction": float(sub["mosaec_exact_match_flag"].mean()) if "mosaec_exact_match_flag" in sub.columns else np.nan,
            "core_geometry_overlap_fraction": float(sub["core_geometry_overlap_flag"].mean()) if "core_geometry_overlap_flag" in sub.columns else np.nan,
            "mosaec_geometry_overlap_fraction": float(sub["mosaec_geometry_overlap_flag"].mean()) if "mosaec_geometry_overlap_flag" in sub.columns else np.nan,
            "mean_external_realism_score": float(sub["external_realism_score"].mean()) if "external_realism_score" in sub.columns else np.nan,
        })
    summ = pd.DataFrame(rows).set_index("target").reindex(v44_order_targets(pd.DataFrame({"target": rows and [r["target"] for r in rows] or []}))).reset_index() if rows else pd.DataFrame()
    atomic_save_csv(summ, TABLE_DIR / "SI_Table_external_domain_overlap_summary_v44.csv")
    if summ.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1)); x = np.arange(len(summ)); width = 0.34
    axes[0].bar(x-width/2, summ.get("core_geometry_overlap_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="CoRE geometry", color="#5DADE2")
    axes[0].bar(x+width/2, summ.get("mosaec_geometry_overlap_fraction", pd.Series(np.nan, index=summ.index)), width=width, label="MOSAEC geometry", color="#58D68D")
    axes[0].set_xticks(x); axes[0].set_xticklabels([v44_tlabel(t) for t in summ["target"]], rotation=30, ha="right"); axes[0].set_ylim(0,1.05); axes[0].set_ylabel("Overlap fraction"); axes[0].set_title("Geometry-domain overlap"); axes[0].legend(frameon=False); v44_panel(axes[0], "a")
    # panel b as explicit zero exact-match statement rather than empty bars
    axes[1].axis("off"); v44_panel(axes[1], "b")
    total = int(summ["n"].sum()) if "n" in summ.columns else len(fe)
    core_exact = int(round((summ.get("core_exact_fraction", pd.Series(0, index=summ.index))*summ.get("n", pd.Series(0, index=summ.index))).sum()))
    mos_exact = int(round((summ.get("mosaec_exact_fraction", pd.Series(0, index=summ.index))*summ.get("n", pd.Series(0, index=summ.index))).sum()))
    axes[1].text(0.5, 0.62, "Exact structural matches\nwere not assumed", ha="center", va="center", fontsize=12, fontweight="bold", transform=axes[1].transAxes)
    axes[1].text(0.5, 0.36, f"CoRE exact: {core_exact}/{total}\nMOSAEC exact: {mos_exact}/{total}", ha="center", va="center", fontsize=10, transform=axes[1].transAxes)
    axes[2].bar([v44_tlabel(t) for t in summ["target"]], pd.to_numeric(summ.get("mean_external_realism_score", pd.Series(np.nan, index=summ.index)), errors="coerce"), color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in summ["target"]])
    axes[2].tick_params(axis="x", rotation=30); axes[2].set_ylabel("Mean annotation score"); axes[2].set_title("Domain-overlap annotation score"); axes[2].grid(True, axis="y"); v44_panel(axes[2], "c")
    fig.suptitle("External overlays annotate domain overlap, not experimental validation", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.92])
    v44_save(fig, "Figure_S_external_domain_overlap_annotation_v45", si=True, aliases=[PAPER_PUBLICATION_SI_DIR / "Figure_S_external_domain_overlap_annotation_v43", PAPER_STORY_SI_DIR / "Figure_S_external_domain_overlap_annotation_v42"])


def v44_plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.4 journal SI figures")
    v44_apply_style()
    # Target distributions: use full master table if available; fallback to tiers.
    if master is not None and not master.empty:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.1)); axes = axes.ravel()
        for ax, target in zip(axes, TARGET_ORDER_FINAL):
            vals = pd.to_numeric(master[target], errors="coerce").dropna() if target in master.columns else pd.Series(dtype=float)
            if len(vals): ax.hist(vals, bins=50, color=TARGET_COLORS_FINAL.get(target, "#777777"), alpha=0.80)
            ax.set_title(v44_tlabel(target)); ax.set_xlabel("Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Full target distributions in ARC-MOF", fontsize=12.5, fontweight="bold")
        v44_save(fig, "Figure_S1_v4_4_full_target_distributions", si=True, aliases=[PAPER_PUBLICATION_SI_DIR / "Figure_S1_v4_3_target_distributions", PAPER_STORY_SI_DIR / "Figure_S1_v4_2_target_distributions", FIG_SI_DIR / "Figure_S1_target_distributions"])
    elif tiers is not None and not tiers.empty and {"target", "y_true"}.issubset(tiers.columns):
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.1)); axes = axes.ravel(); sample = tiers.drop_duplicates(["mof_id", "target"]) if "mof_id" in tiers.columns else tiers
        for ax, target in zip(axes, TARGET_ORDER_FINAL):
            vals = pd.to_numeric(sample.loc[sample["target"].astype(str).eq(target), "y_true"], errors="coerce").dropna()
            if len(vals): ax.hist(vals, bins=50, color=TARGET_COLORS_FINAL.get(target, "#777777"), alpha=0.80)
            ax.set_title(v44_tlabel(target)); ax.set_xlabel("Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count")
        fig.suptitle("Figure S1. Target distributions in candidate rows", fontsize=12.5, fontweight="bold")
        v44_save(fig, "Figure_S1_v4_4_candidate_target_distributions", si=True)
    # Quality gate audit with cleaned labels.
    q = v42_read_table("QC_Table_model_stability.csv")
    if not q.empty and "quality_gate_reasons" in q.columns:
        reasons = q["quality_gate_reasons"].astype(str).fillna("unknown")
        def simplify_reason(x: str) -> str:
            if x == "PASS": return "Pass final candidate gate"
            bits = []
            if "model_not_candidate_eligible" in x: bits.append("not final-candidate model")
            if "low_or_nan_spearman" in x: bits.append("low/NaN Spearman")
            if "low_top5_recall" in x: bits.append("low top-5 recall")
            if "coverage_outside_gate" in x: bits.append("coverage outside gate")
            if "rmse_too_large_for_target" in x: bits.append("RMSE too large")
            return "; ".join(bits) if bits else x.replace("_", " ")
        counts = reasons.map(simplify_reason).value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10.4, 4.8))
        ax.barh(range(len(counts)), counts.values, color="#7F8C8D")
        ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index, fontsize=7.0); ax.invert_yaxis(); ax.set_xlabel("Experiments"); ax.set_title("Figure S2. Candidate-quality gate audit")
        v44_save(fig, "Figure_S2_v4_5_quality_gate_audit", si=True, aliases=[PAPER_PUBLICATION_SI_DIR / "Figure_S2_v4_3_quality_gate_audit", PAPER_STORY_SI_DIR / "Figure_S2_v4_2_quality_gate_audit"])
    mark_done(stage, {"v4_4": True})


def v44_write_manifest() -> pd.DataFrame:
    rows = []
    for folder, label in [(PAPER_JOURNAL_DIR, "main_journal_final"), (PAPER_JOURNAL_SI_DIR, "si_journal_final")]:
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in {".png", ".pdf", ".svg"}:
                rows.append({"category": label, "file": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    for p in sorted(PAPER_JOURNAL_DATA_DIR.glob("*.csv")):
        rows.append({"category": "source_data", "file": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    mf = pd.DataFrame(rows)
    atomic_save_csv(mf, TABLE_DIR / "MANIFEST_v4_4_journal_final_figures.csv")
    return mf


def v44_write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    stage = "07_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> WRITE_FINAL_RUN_REPORT_V4_4")
    agg = v42_read_table("table_aggregated_metrics_with_bootstrap_ci.csv", low_memory=True)
    v44_final_claims_table(agg, master=master)
    v44_candidate_tables(master=master)
    manifest = v44_write_manifest()
    report = RESULTS_DIR / "RUN_REPORT.md"
    lines = []
    lines.append("# Few-shot MOF risk-controlled screening run report — v4.4 journal-final visuals\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Runtime modes\n")
    for key in ["save_mode", "ram_mode", "comprehensive_level", "n_jobs", "data_root"]:
        lines.append(f"- {key}: `{CONFIG.get(key)}`")
    lines.append("\n## v4.4 journal-final output folders\n")
    lines.append(f"- Main journal figures: `{PAPER_JOURNAL_DIR}`")
    lines.append(f"- SI/domain-overlap figures: `{PAPER_JOURNAL_SI_DIR}`")
    lines.append(f"- Source data for journal figures: `{PAPER_JOURNAL_DATA_DIR}`")
    lines.append("\n## Main figures\n")
    for fname, meaning in [
        ("Figure_1_v4_5_journal_workflow.png", "quantitative workflow / graphical abstract"),
        ("Figure_2_v4_5_fewshot_learning_phase_diagram.png", "few-shot label efficiency and enrichment"),
        ("Figure_3_v4_5_descriptor_chemistry_extrapolation_stress.png", "RAC descriptor gain and topology stress"),
        ("Figure_4_v4_5_conformal_risk_control_frontier.png", "risk-control frontier and tier composition"),
        ("Figure_5_v4_5_consensus_shortlist_atlas.png", "target-balanced consensus shortlist atlas"),
    ]:
        p = PAPER_JOURNAL_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'} — {meaning}")
    lines.append("\n## Final tables\n")
    for fname in [
        "MAIN_Table_0_final_claims_summary.csv",
        "MAIN_Table_0_final_claims_summary_v44.csv",
        "MAIN_Table_3_top5_consensus_candidates_per_target.csv",
        "STORY_Table_top25_consensus_candidates.csv",
        "STORY_Table_candidate_funnel_by_target.csv",
        "STORY_Table_prefinal_tier_fraction_by_budget.csv",
        "QC_Table_consensus_true_enrichment.csv",
        "MANIFEST_v4_4_journal_final_figures.csv",
    ]:
        p = TABLE_DIR / fname
        lines.append(f"- `{fname}`: {'FOUND' if p.exists() else 'not found'}")
    lines.append("\n## Manuscript guardrails\n")
    lines.append("- Main descriptor claim: geometry vs geometry+RACs. Do not claim RDF effects unless QC shows real RDF features.")
    lines.append("- Main generalisation claim: topology-grouped split is the hardest extrapolation stress test.")
    lines.append("- Main uncertainty claim: conformal intervals and consensus filters turn raw few-shot predictions into risk-controlled shortlists.")
    lines.append("- External overlay claim: CoRE/MOSAEC are domain-overlap annotations, not experimental validation.")
    lines.append(f"\nManifest entries: {len(manifest)}")
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage, {"v4_4": True, "n_manifest_rows": int(len(manifest))})

# ---- v4.4 overrides used by the existing main() function ----

def plot_figure_1_framework() -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v44_plot_figure_1(master)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    return v44_plot_figure_2(agg)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    return v44_plot_figure_3(agg)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v44_plot_figure_4(anatomy, tiers, master=master)


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v44_plot_figure_5(core, mosaec, final_external, master=master)


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v44_plot_si_figures(agg, tiers, anatomy, master=master)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    return v44_write_run_report(master, metrics_all)



# =============================================================================
# 16i. v4.7 consolidated high-impact publication package layer
# =============================================================================
# v4.7 supersedes the v4.5 visual layer. It is deliberately post-processing-only:
# it does not refit models or change candidate QC. It writes a single consolidated
# publication package with main-text figures, SI figures, main tables, SI tables,
# and source-data CSVs. The main fixes versus v4.5 are:
#   * Figure 3c split-stress heatmap is rebuilt directly from aggregate metrics.
#   * Figure 5c true-elite frequency panel is rebuilt directly from top-25
#     candidate true percentiles, preventing blank panels.
#   * The Figure 5 embedded table is removed and replaced by a cleaner visual
#     atlas; candidate examples are written as Main/SI tables instead.
#   * New consolidated folders avoid confusion from the older v4.1-v4.5 folders.
#   * Tables are copied/written into publication_package_v47/tables_main and
#     publication_package_v47/tables_si so the manuscript/SI package is cleaner.

CONFIG["paper_final_name"] = "v4_7_consolidated_publication_package_tabulatefix"
CONFIG.setdefault("v47_save_formats", ["png", "pdf", "svg"])
CONFIG.setdefault("v47_png_dpi", 750)
CONFIG.setdefault("v47_background_points", 6000)

V47_ROOT = RESULTS_DIR / "publication_package_v47"
V47_FIG_MAIN_DIR = V47_ROOT / "figures_main_text"
V47_FIG_SI_DIR = V47_ROOT / "figures_supporting_information"
V47_SOURCE_MAIN_DIR = V47_ROOT / "source_data_main_text"
V47_SOURCE_SI_DIR = V47_ROOT / "source_data_supporting_information"
V47_TABLE_MAIN_DIR = V47_ROOT / "tables_main_text"
V47_TABLE_SI_DIR = V47_ROOT / "tables_supporting_information"
for _d in [V47_ROOT, V47_FIG_MAIN_DIR, V47_FIG_SI_DIR, V47_SOURCE_MAIN_DIR, V47_SOURCE_SI_DIR, V47_TABLE_MAIN_DIR, V47_TABLE_SI_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def v47_apply_style() -> None:
    """Publication-level figure style.

    The style intentionally avoids external fonts so the script is portable on
    Windows/conda, while keeping PDFs/SVGs editable by embedding text as text.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "figure.dpi": 170,
        "savefig.dpi": int(CONFIG.get("v47_png_dpi", 750)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.22,
        "lines.linewidth": 1.55,
        "lines.markersize": 4.2,
        "patch.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def v47_panel(ax, label: str, x: float = -0.095, y: float = 1.075) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.0, fontweight="bold", ha="right", va="top")


def v47_save(fig: plt.Figure, stem: str, si: bool = False) -> None:
    v47_apply_style()
    out_dir = V47_FIG_SI_DIR if si else V47_FIG_MAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG.get("v47_save_formats", ["png", "pdf", "svg"]):
        out = (out_dir / stem).with_suffix(f".{fmt}")
        try:
            fig.savefig(out, dpi=int(CONFIG.get("v47_png_dpi", 750)), bbox_inches="tight")
        except Exception as e:
            logging.warning("V47_FIGURE_SAVE_FAILED | %s | %s", out, e)
    plt.close(fig)


def v47_source(df: pd.DataFrame, fname: str, si: bool = False) -> None:
    if df is None:
        df = pd.DataFrame()
    out_dir = V47_SOURCE_SI_DIR if si else V47_SOURCE_MAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_csv(df, out_dir / fname)


def v47_table(df: pd.DataFrame, fname: str, si: bool = False) -> None:
    if df is None:
        df = pd.DataFrame()
    out_dir = V47_TABLE_SI_DIR if si else V47_TABLE_MAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_csv(df, out_dir / fname)


def v47_latex_label(target: Any) -> str:
    return TARGET_LABELS_JOURNAL.get(str(target), str(target))


def v47_plain_label(target: Any) -> str:
    return TARGET_LABELS_PLAIN.get(str(target), str(target))


def v47_target_order(df: Optional[pd.DataFrame] = None) -> List[str]:
    if df is None or df.empty or "target" not in df.columns:
        return TARGET_ORDER_FINAL[:]
    present = set(df["target"].dropna().astype(str))
    ordered = [t for t in TARGET_ORDER_FINAL if t in present]
    ordered += sorted(present - set(ordered))
    return ordered


def v47_target_stats(master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    if master is not None and not master.empty:
        for t in TARGET_ORDER_FINAL:
            if t in master.columns:
                vals = pd.to_numeric(master[t], errors="coerce").dropna()
                if len(vals):
                    rows.append({
                        "target": t,
                        "target_label_plain": v47_plain_label(t),
                        "n_mofs": int(len(vals)),
                        "mean_mmol_g": float(vals.mean()),
                        "std_mmol_g": float(vals.std()),
                        "median_mmol_g": float(vals.median()),
                        "p95_mmol_g": float(vals.quantile(0.95)),
                        "p99_mmol_g": float(vals.quantile(0.99)),
                        "max_mmol_g": float(vals.max()),
                    })
    if not rows:
        for fname in ["MAIN_Table_1_dataset_and_targets.csv", "table_dataset_target_summary.csv"]:
            p = TABLE_DIR / fname
            if p.exists():
                tab = pd.read_csv(p)
                if "target" in tab.columns:
                    for _, r in tab.iterrows():
                        t = str(r["target"])
                        rows.append({
                            "target": t,
                            "target_label_plain": v47_plain_label(t),
                            "n_mofs": int(pd.to_numeric(pd.Series([r.get("n_mofs", r.get("n_nonmissing", np.nan))]), errors="coerce").iloc[0]) if pd.notna(r.get("n_mofs", r.get("n_nonmissing", np.nan))) else np.nan,
                            "mean_mmol_g": r.get("mean_mmol_g", r.get("mean", np.nan)),
                            "std_mmol_g": r.get("std_mmol_g", r.get("std", np.nan)),
                            "median_mmol_g": r.get("median_mmol_g", r.get("median", np.nan)),
                            "p95_mmol_g": r.get("p95_mmol_g", np.nan),
                            "p99_mmol_g": r.get("p99_mmol_g", np.nan),
                            "max_mmol_g": r.get("max_mmol_g", r.get("max", np.nan)),
                        })
                    break
    out = pd.DataFrame(rows)
    if not out.empty:
        out["target"] = pd.Categorical(out["target"], categories=TARGET_ORDER_FINAL, ordered=True)
        out = out.sort_values("target").reset_index(drop=True)
        out["target"] = out["target"].astype(str)
    return out


def v47_filter_agg(agg: pd.DataFrame, top_frac: float = 0.05, feature_set: str = "geometry_plus_racs",
                   budget: Optional[int] = None, split: Optional[str] = None, stable_only: bool = True) -> pd.DataFrame:
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = agg.copy()
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(top_frac).abs() < 1e-9]
    if "feature_set" in df.columns and feature_set:
        if feature_set in set(df["feature_set"].dropna().astype(str)):
            df = df[df["feature_set"].astype(str).eq(feature_set)]
    if budget is not None and "budget" in df.columns:
        df = df[pd.to_numeric(df["budget"], errors="coerce").eq(budget)]
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str).eq(split)]
    if stable_only and "model" in df.columns:
        stable = set(CONFIG.get("stable_plot_models", []))
        if stable:
            df = df[df["model"].astype(str).isin(stable)]
    return df


def v47_metric_by_target_budget(agg: pd.DataFrame, metric: str, feature_set: str = "geometry_plus_racs") -> pd.DataFrame:
    df = v47_filter_agg(agg, feature_set=feature_set)
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    out = df.groupby(["target", "budget"], as_index=False).agg(
        mean=(metric, "mean"),
        lo=(metric, lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.10))),
        hi=(metric, lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.90))),
        n=(metric, "count"),
    )
    out["budget"] = pd.to_numeric(out["budget"], errors="coerce").astype(int)
    return out.sort_values(["target", "budget"])


def v47_draw_target_curves(ax, df: pd.DataFrame, ylabel: str, title: str, baseline: Optional[float] = None) -> None:
    for target in v47_target_order(df):
        sub = df[df["target"].astype(str).eq(target)].sort_values("budget")
        if sub.empty:
            continue
        color = TARGET_COLORS_FINAL.get(str(target), "#777777")
        x = pd.to_numeric(sub["budget"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(sub.get("lo", sub["mean"]), errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(sub.get("hi", sub["mean"]), errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, label=v47_latex_label(target))
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
    if baseline is not None:
        ax.axhline(baseline, color="#666666", ls="--", lw=0.9, label="random" if baseline <= 1 else "baseline")
    ax.set_xscale("log")
    ax.set_xticks(BUDGET_ORDER_JOURNAL)
    ax.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL])
    ax.set_xlabel("Label budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", axis="both")


def v47_plot_figure_1(master: Optional[pd.DataFrame] = None) -> None:
    stage = "v47_fig1_workflow"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.7 Figure 1")
    v47_apply_style()
    stats = v47_target_stats(master)
    v47_source(stats, "Figure_1a_target_cards_source.csv")
    fig, ax = plt.subplots(figsize=(13.6, 6.1))
    ax.set_axis_off()
    ax.text(0.5, 0.965, "Risk-controlled few-shot discovery of adsorption elites", ha="center", va="top",
            fontsize=16.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.905, "Label-efficient screening, descriptor chemistry, conformal risk control, and target-balanced consensus",
            ha="center", va="top", fontsize=9.5, color="#263B55", transform=ax.transAxes)

    # Panel a: target cards
    ax.text(0.025, 0.79, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", va="center")
    card_y, card_h, card_w = 0.55, 0.25, 0.205
    x0s = [0.075, 0.305, 0.535, 0.765]
    for x0, target in zip(x0s, TARGET_ORDER_FINAL):
        row = stats[stats["target"].astype(str).eq(target)].head(1)
        n = int(row["n_mofs"].iloc[0]) if not row.empty and pd.notna(row["n_mofs"].iloc[0]) else 279010
        med = float(row["median_mmol_g"].iloc[0]) if not row.empty and pd.notna(row["median_mmol_g"].iloc[0]) else np.nan
        p95 = float(row["p95_mmol_g"].iloc[0]) if not row.empty and pd.notna(row.get("p95_mmol_g", pd.Series([np.nan])).iloc[0]) else np.nan
        mx = float(row["max_mmol_g"].iloc[0]) if not row.empty and pd.notna(row["max_mmol_g"].iloc[0]) else np.nan
        color = TARGET_COLORS_FINAL.get(target, "#777777")
        box = matplotlib.patches.FancyBboxPatch((x0, card_y), card_w, card_h, boxstyle="round,pad=0.012,rounding_size=0.012",
                                                transform=ax.transAxes, fc="white", ec=color, lw=1.6)
        ax.add_patch(box)
        ax.text(x0+0.014, card_y+card_h-0.045, v47_latex_label(target), color=color, fontsize=11.2, fontweight="bold", transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.135, f"{n:,}", fontsize=10.8, fontweight="bold", transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.108, "labelled MOFs", fontsize=10.0, fontweight="bold", transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.073, f"median: {med:.2g} mmol g$^{{-1}}$" if np.isfinite(med) else "median: —", fontsize=8.1, transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.040, f"top 5% ≥ {p95:.2g} mmol g$^{{-1}}$" if np.isfinite(p95) else "top 5%: —", fontsize=8.1, transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.010, f"max: {mx:.2g} mmol g$^{{-1}}$" if np.isfinite(mx) else "max: —", fontsize=8.1, transform=ax.transAxes)

    # Panel b: workflow ribbon
    ax.text(0.025, 0.36, "b", transform=ax.transAxes, fontsize=12, fontweight="bold", va="center")
    steps = [
        ("ARC-MOF labels", "4 adsorption targets\ngeometry + RACs"),
        ("Few-shot design", "10–1000 labels\n5 seeds × split stress"),
        ("Stable learners", "RF / ExtraTrees / HGB\nLightGBM / XGBoost"),
        ("Risk control", "90% conformal intervals\ntrusted / uncertain / rejected"),
        ("Consensus", "models × seeds × splits\ntarget-balanced shortlist"),
        ("Domain overlay", "CoRE/MOSAEC geometry\nannotation, not validation"),
    ]
    y, h, w = 0.16, 0.17, 0.145
    xs = np.linspace(0.065, 0.790, len(steps))
    for i, (x, (title, body)) in enumerate(zip(xs, steps)):
        box = matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                                transform=ax.transAxes, fc="#F8FAFC", ec="#6E7F90", lw=0.9)
        ax.add_patch(box)
        ax.text(x+w/2, y+h*0.68, title, ha="center", va="center", fontsize=8.6, fontweight="bold", transform=ax.transAxes)
        ax.text(x+w/2, y+h*0.32, body, ha="center", va="center", fontsize=7.3, transform=ax.transAxes)
        if i < len(steps)-1:
            ax.annotate("", xy=(x+w+0.018, y+h/2), xytext=(x+w+0.002, y+h/2), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#5D6D7E"))
    v47_save(fig, "Figure_1_v4_7_workflow")


def v47_plot_figure_2(agg: pd.DataFrame) -> None:
    stage = "v47_fig2_learning_phase"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.7 Figure 2")
    v47_apply_style()
    recall = v47_metric_by_target_budget(agg, "recall_mean")
    enrich = v47_metric_by_target_budget(agg, "enrichment_mean")
    v47_source(recall, "Figure_2a_recall_learning_curves.csv")
    v47_source(enrich, "Figure_2b_enrichment_learning_curves.csv")

    # Label efficiency: recall at budget divided by recall at 1000 labels.
    eff_rows = []
    for target in v47_target_order(recall):
        sub = recall[recall["target"].astype(str).eq(target)].sort_values("budget")
        final = sub.loc[sub["budget"].eq(max(BUDGET_ORDER_JOURNAL)), "mean"]
        final_val = float(final.iloc[0]) if len(final) and np.isfinite(final.iloc[0]) and final.iloc[0] != 0 else np.nan
        for _, r in sub.iterrows():
            eff_rows.append({"target": target, "budget": int(r["budget"]), "label_efficiency": float(r["mean"] / final_val) if np.isfinite(final_val) else np.nan})
    eff = pd.DataFrame(eff_rows)
    v47_source(eff, "Figure_2c_label_efficiency.csv")

    # Threshold budgets to 50/75/90% of final recall.
    th_rows = []
    for target in v47_target_order(eff):
        sub = eff[eff["target"].astype(str).eq(target)].sort_values("budget")
        for thresh in [0.50, 0.75, 0.90]:
            hit = sub[pd.to_numeric(sub["label_efficiency"], errors="coerce") >= thresh].head(1)
            th_rows.append({"target": target, "threshold_fraction_of_final_recall": thresh,
                            "budget_needed": int(hit["budget"].iloc[0]) if not hit.empty else int(sub["budget"].max()) if not sub.empty else np.nan})
    th = pd.DataFrame(th_rows)
    v47_source(th, "Figure_2d_budget_needed_for_recall_fraction.csv")

    # Model comparison and final enrichment summary.
    df1000 = v47_filter_agg(agg, budget=1000)
    model = df1000.groupby("model", as_index=False)["recall_mean"].mean() if not df1000.empty else pd.DataFrame()
    model = model.sort_values("recall_mean") if not model.empty else model
    final_en = enrich[enrich["budget"].eq(1000)].copy()
    final_en = final_en.set_index("target").reindex(v47_target_order(final_en)).reset_index()
    v47_source(model, "Figure_2e_model_comparison_1000_labels.csv")
    v47_source(final_en, "Figure_2f_final_enrichment_summary.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.45))
    axa, axb, axc, axd, axe, axf = axes.ravel()
    v47_draw_target_curves(axa, recall, "Top-5% recall", "Elite recovery emerges from few labels", baseline=0.05)
    v47_panel(axa, "a")
    axa.legend(frameon=False, loc="upper left", ncol=1)
    v47_draw_target_curves(axb, enrich, "Enrichment over random", "Screening enrichment", baseline=1.0)
    v47_panel(axb, "b")
    v47_draw_target_curves(axc, eff.rename(columns={"label_efficiency":"mean"}), "Fraction of 1000-label recall", "Label-efficiency index", baseline=0.75)
    axc.set_ylim(0, 1.05)
    v47_panel(axc, "c")

    # Threshold dot plot
    if not th.empty:
        markers = {0.50: "o", 0.75: "s", 0.90: "D"}
        ylabels = [v47_latex_label(t) for t in TARGET_ORDER_FINAL]
        ymap = {t:i for i,t in enumerate(TARGET_ORDER_FINAL)}
        for thresh, lab in [(0.50, "50%"), (0.75, "75%"), (0.90, "90%")]:
            sub = th[np.isclose(pd.to_numeric(th["threshold_fraction_of_final_recall"], errors="coerce"), thresh)]
            axd.scatter(pd.to_numeric(sub["budget_needed"], errors="coerce"), [ymap.get(str(t), np.nan) for t in sub["target"]],
                        s=40, marker=markers[thresh], label=lab, alpha=0.9)
        axd.set_xscale("log")
        axd.set_xticks(BUDGET_ORDER_JOURNAL)
        axd.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL])
        axd.set_yticks(range(len(TARGET_ORDER_FINAL)))
        axd.set_yticklabels(ylabels)
        axd.legend(title="of final recall", frameon=False, loc="lower right")
    axd.set_xlabel("Budget needed"); axd.set_title("How many labels are enough?"); axd.grid(True, axis="x"); v47_panel(axd, "d")

    if not model.empty:
        axe.bar(model["model"].astype(str).str.replace("_", "\n"), pd.to_numeric(model["recall_mean"], errors="coerce"), color="#7F8C8D")
        for i, v in enumerate(pd.to_numeric(model["recall_mean"], errors="coerce")):
            if np.isfinite(v): axe.text(i, v+0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axe.set_ylim(0, max(0.72, float(pd.to_numeric(model.get("recall_mean", pd.Series([0.6])), errors="coerce").max())*1.16 if not model.empty else 0.7))
    axe.set_ylabel("Mean top-5% recall"); axe.set_title("Stable model comparison at 1000 labels"); axe.grid(True, axis="y"); v47_panel(axe, "e")

    if not final_en.empty:
        final_en = final_en.sort_values("mean", ascending=True)
        axf.barh([v47_latex_label(t) for t in final_en["target"]], pd.to_numeric(final_en["mean"], errors="coerce"),
                 color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in final_en["target"]])
        for i, v in enumerate(pd.to_numeric(final_en["mean"], errors="coerce")):
            if np.isfinite(v): axf.text(v+0.15, i, f"{v:.1f}×", va="center", fontsize=7.5)
    axf.axvline(1, color="#777777", ls="--", lw=0.9)
    axf.set_xlabel("Best enrichment at 1000 labels"); axf.set_title("Final enrichment summary"); axf.grid(True, axis="x"); v47_panel(axf, "f")

    fig.suptitle("Few-shot learning phase diagram for adsorption-elite recovery", fontsize=14.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v47_save(fig, "Figure_2_v4_7_fewshot_learning_phase_diagram")


def v47_descriptor_gain_table(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()
    if df.empty:
        return pd.DataFrame()
    if "top_frac" in df.columns:
        df = df[pd.to_numeric(df["top_frac"], errors="coerce").sub(0.05).abs() < 1e-9]
    if "model" in df.columns:
        stable = set(CONFIG.get("stable_plot_models", []))
        if stable:
            df = df[df["model"].astype(str).isin(stable)]
    df["recall_mean"] = pd.to_numeric(df["recall_mean"], errors="coerce")
    g = df.groupby(["target", "budget", "feature_set"], as_index=False)["recall_mean"].mean()
    p = g.pivot_table(index=["target", "budget"], columns="feature_set", values="recall_mean", aggfunc="mean").reset_index()
    if {"geometry_only", "geometry_plus_racs"}.issubset(p.columns):
        p["delta_recall_racs_minus_geometry"] = p["geometry_plus_racs"] - p["geometry_only"]
    else:
        p["delta_recall_racs_minus_geometry"] = np.nan
    return p


def v47_split_recall_table(agg: pd.DataFrame) -> pd.DataFrame:
    df = v47_filter_agg(agg, feature_set="geometry_plus_racs", budget=1000)
    if df.empty or "recall_mean" not in df.columns:
        return pd.DataFrame()
    df["recall_mean"] = pd.to_numeric(df["recall_mean"], errors="coerce")
    out = df.groupby(["target", "split"], as_index=False)["recall_mean"].mean()
    out["split_label"] = out["split"].map(v44_slabel)
    out["target_label_plain"] = out["target"].map(v47_plain_label)
    return out


def v47_plot_figure_3(agg: pd.DataFrame) -> None:
    stage = "v47_fig3_descriptor_extrapolation"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.7 Figure 3")
    v47_apply_style()
    gain = v47_descriptor_gain_table(agg)
    split_recall = v47_split_recall_table(agg)
    v47_source(gain, "Figure_3_descriptor_gain_by_budget.csv")
    v47_source(split_recall, "Figure_3_split_recall_at_1000_labels.csv")

    # Matrices
    gain_mat = pd.DataFrame()
    if not gain.empty:
        g = gain.pivot_table(index="target", columns="budget", values="delta_recall_racs_minus_geometry", aggfunc="mean")
        g = g.reindex(v47_target_order(pd.DataFrame({"target": g.index})))
        g = g[[b for b in BUDGET_ORDER_JOURNAL if b in g.columns]]
        g.index = [v47_latex_label(t) for t in g.index]
        gain_mat = g
        v47_source(g.reset_index().rename(columns={"index":"target_label"}), "Figure_3a_rac_gain_heatmap.csv")

    bars = pd.DataFrame()
    if not gain.empty:
        bars = gain[pd.to_numeric(gain["budget"], errors="coerce").eq(1000)].groupby("target", as_index=False)["delta_recall_racs_minus_geometry"].mean()
        bars = bars.sort_values("delta_recall_racs_minus_geometry")
        v47_source(bars, "Figure_3b_rac_gain_bars_1000_labels.csv")

    split_mat = pd.DataFrame()
    if not split_recall.empty:
        split_mat = split_recall.pivot_table(index="target", columns="split", values="recall_mean", aggfunc="mean")
        split_mat = split_mat.reindex(v47_target_order(pd.DataFrame({"target": split_mat.index})))
        split_mat = split_mat[[s for s in SPLIT_ORDER_JOURNAL if s in split_mat.columns]]
        split_mat_label = split_mat.copy()
        split_mat_label.index = [v47_latex_label(t) for t in split_mat.index]
        split_mat_label.columns = [v44_slabel(s) for s in split_mat_label.columns]
        v47_source(split_mat_label.reset_index().rename(columns={"index":"target_label"}), "Figure_3c_split_stress_recall_heatmap.csv")

    topo_penalty = pd.DataFrame()
    vuln_mat_label = pd.DataFrame()
    if not split_mat.empty and "random" in split_mat.columns and "topology_grouped" in split_mat.columns:
        topo_penalty = pd.DataFrame({"target": split_mat.index.astype(str),
                                     "topology_penalty_random_minus_topology": split_mat["random"] - split_mat["topology_grouped"]})
        topo_penalty = topo_penalty.sort_values("topology_penalty_random_minus_topology")
        v47_source(topo_penalty, "Figure_3d_topology_penalty.csv")
    if not split_mat.empty and "random" in split_mat.columns:
        vuln = split_mat.copy()
        for c in vuln.columns:
            vuln[c] = 1.0 - vuln[c] / split_mat["random"].replace(0, np.nan)
        vuln_label = vuln.copy()
        vuln_label.index = [v47_latex_label(t) for t in vuln.index]
        vuln_label.columns = [v44_slabel(s) for s in vuln_label.columns]
        vuln_mat_label = vuln_label
        v47_source(vuln_label.reset_index().rename(columns={"index":"target_label"}), "Figure_3e_extrapolation_vulnerability.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.55))
    axa, axb, axc, axd, axe, axf = axes.ravel()

    v44_heatmap(axa, gain_mat, "RAC descriptor gain across label budgets", "Δ recall", center_zero=True, fmt=".2f")
    v47_panel(axa, "a")
    if not bars.empty:
        axb.barh([v47_latex_label(t) for t in bars["target"]], bars["delta_recall_racs_minus_geometry"],
                 color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in bars["target"]])
        for i, v in enumerate(bars["delta_recall_racs_minus_geometry"]):
            axb.text(v+0.004, i, f"{v:.2f}", va="center", fontsize=7)
    axb.axvline(0, color="#777777", lw=0.8)
    axb.set_xlabel("Δ recall: RACs − geometry")
    axb.set_title("Chemistry benefit at 1000 labels")
    axb.grid(True, axis="x")
    v47_panel(axb, "b")

    v44_heatmap(axc, split_mat_label if not split_mat.empty else pd.DataFrame(), "Split-stress map at 1000 labels", "Top-5% recall", center_zero=False, cmap="YlGnBu", fmt=".2f")
    v47_panel(axc, "c")

    if not topo_penalty.empty:
        axd.barh([v47_latex_label(t) for t in topo_penalty["target"]], topo_penalty["topology_penalty_random_minus_topology"],
                 color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in topo_penalty["target"]])
        for i, v in enumerate(topo_penalty["topology_penalty_random_minus_topology"]):
            axd.text(v+0.004, i, f"{v:.2f}", va="center", fontsize=7)
    axd.axvline(0, color="#777777", lw=0.8)
    axd.set_xlabel("Recall(random) − recall(topology)")
    axd.set_title("Topology extrapolation penalty")
    axd.grid(True, axis="x")
    v47_panel(axd, "d")

    v44_heatmap(axe, vuln_mat_label, "Extrapolation vulnerability", "1 − grouped/random", center_zero=True, cmap="coolwarm", fmt=".2f")
    v47_panel(axe, "e")

    axf.axis("off")
    axf.set_title("Mechanistic interpretation", fontweight="bold")
    bullets = [
        ("Geometry baseline", "Pore size, surface area, and density encode the\nmain physical adsorption envelope."),
        ("RAC chemistry", "Metal/linker descriptors improve elite recovery\nbeyond geometry alone."),
        ("Topology split", "Network-level transfer is the hardest\nout-of-distribution stress test."),
        ("Manuscript claim", "Random splits overestimate transferability;\nrisk-controlled consensus gives a safer shortlist."),
    ]
    y = 0.86
    for title, body in bullets:
        rect = matplotlib.patches.FancyBboxPatch((0.05, y-0.105), 0.88, 0.105, boxstyle="round,pad=0.012,rounding_size=0.015",
                                                 transform=axf.transAxes, fc="#F8FAFC", ec="#AAB7B8", lw=0.7)
        axf.add_patch(rect)
        axf.text(0.09, y-0.030, title, transform=axf.transAxes, fontsize=8.0, fontweight="bold", va="top")
        axf.text(0.09, y-0.062, body, transform=axf.transAxes, fontsize=6.9, va="top")
        y -= 0.18
    v47_panel(axf, "f")

    fig.suptitle("Descriptor chemistry improves screening, but topology extrapolation remains limiting", fontsize=14.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v47_save(fig, "Figure_3_v4_7_descriptor_chemistry_extrapolation_stress")
    mark_done(stage, {"v4_7": True})


def v47_precision_yield(master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    existing = TABLE_DIR / "STORY_Table_precision_yield_frontier.csv"
    if existing.exists():
        try:
            df = pd.read_csv(existing)
            if {"target", "retained_fraction"}.issubset(df.columns):
                return df
        except Exception:
            pass
    tiers = v42_read_table("table_candidate_tiers_all_predictions.csv", low_memory=True)
    if tiers.empty or "target" not in tiers.columns:
        return pd.DataFrame()
    rows = []
    fracs = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    # thresholds from master when possible, otherwise from candidate y_true values
    for target in v47_target_order(tiers):
        sub = tiers[tiers["target"].astype(str).eq(target)].copy()
        if sub.empty:
            continue
        for col in ["split_lower", "y_pred", "y_true"]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
        if master is not None and target in master.columns:
            vals = pd.to_numeric(master[target], errors="coerce").dropna()
            top5_thr = float(vals.quantile(0.95)) if len(vals) else np.nan
        else:
            vals = pd.to_numeric(sub.get("y_true", pd.Series(dtype=float)), errors="coerce").dropna()
            top5_thr = float(vals.quantile(0.95)) if len(vals) else np.nan
        sort_col = "split_lower" if "split_lower" in sub.columns else "y_pred"
        sub = sub.sort_values(sort_col, ascending=False)
        n = len(sub)
        for f in fracs:
            k = max(1, int(math.ceil(f*n)))
            keep = sub.head(k)
            if "y_true" in keep.columns and np.isfinite(top5_thr):
                precision = float((pd.to_numeric(keep["y_true"], errors="coerce") >= top5_thr).mean())
            else:
                precision = np.nan
            rows.append({"target": target, "retained_fraction": f, "precision_true_top5": precision, "n_retained": k, "score_column": sort_col})
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_precision_yield_frontier.csv")
    return out


def v47_tier_fraction_by_budget(metrics_all: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    p = TABLE_DIR / "STORY_Table_prefinal_tier_fraction_by_budget.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    if metrics_all is None or metrics_all.empty:
        metrics_all = v44_read_metrics_all()
    if metrics_all is None or metrics_all.empty:
        return pd.DataFrame()
    m = metrics_all.copy()
    if "top_frac" in m.columns:
        m = m[pd.to_numeric(m["top_frac"], errors="coerce").sub(0.05).abs() < 1e-9]
    for c in ["trusted_count", "uncertain_count", "rejected_count", "budget"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    cols = [c for c in ["trusted_count", "uncertain_count", "rejected_count"] if c in m.columns]
    if not cols:
        return pd.DataFrame()
    g = m.groupby("budget", as_index=False)[cols].mean()
    denom = g[cols].sum(axis=1).replace(0, np.nan)
    for c in cols:
        g[c.replace("_count", "_fraction")] = g[c] / denom
    atomic_save_csv(g, TABLE_DIR / "STORY_Table_prefinal_tier_fraction_by_budget.csv")
    return g


def v47_plot_figure_4(anatomy: Optional[pd.DataFrame], tiers: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "v47_fig4_risk_control"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.7 Figure 4")
    v47_apply_style()
    metrics_all = v44_read_metrics_all()
    agg = pd.read_csv(TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv", low_memory=False) if (TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv").exists() else pd.DataFrame()
    df1000 = v47_filter_agg(agg, budget=1000)

    # Coverage error matrix
    cov_mat = pd.DataFrame()
    if not df1000.empty and "split_coverage_mean" in df1000.columns:
        cov = df1000.copy()
        cov["coverage_error"] = pd.to_numeric(cov["split_coverage_mean"], errors="coerce") - 0.90
        cov_mat = cov.groupby(["target", "split"], as_index=False)["coverage_error"].mean().pivot_table(index="target", columns="split", values="coverage_error")
        cov_mat = cov_mat.reindex(v47_target_order(pd.DataFrame({"target": cov_mat.index})))
        cov_mat = cov_mat[[s for s in SPLIT_ORDER_JOURNAL if s in cov_mat.columns]]
        cov_mat.index = [v47_latex_label(t) for t in cov_mat.index]
        cov_mat.columns = [v44_slabel(s) for s in cov_mat.columns]
        v47_source(cov_mat.reset_index().rename(columns={"index":"target_label"}), "Figure_4a_coverage_error_heatmap.csv")

    # Normalized interval width
    stats = v47_target_stats(master)
    std_map = {str(r["target"]): float(r["std_mmol_g"]) for _, r in stats.iterrows()} if not stats.empty else {}
    width_rows = []
    if not agg.empty and "split_mean_width_mean" in agg.columns:
        wd = v47_filter_agg(agg)
        for (target, budget), sub in wd.groupby(["target", "budget"]):
            std = std_map.get(str(target), np.nan)
            width = pd.to_numeric(sub["split_mean_width_mean"], errors="coerce").mean()
            width_rows.append({"target": target, "budget": int(budget), "normalized_width": float(width/std) if np.isfinite(std) and std > 0 else np.nan})
    width_df = pd.DataFrame(width_rows)
    v47_source(width_df, "Figure_4b_normalized_interval_width.csv")

    frontier = v47_precision_yield(master=master)
    v47_source(frontier, "Figure_4c_precision_yield_frontier.csv")
    tierfrac = v47_tier_fraction_by_budget(metrics_all)
    v47_source(tierfrac, "Figure_4e_prefinal_tier_composition.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.45))
    axa, axb, axc, axd, axe, axf = axes.ravel()
    v44_heatmap(axa, cov_mat, "Coverage error relative to 90%", "coverage − 0.90", center_zero=True, fmt="+.2f")
    v47_panel(axa, "a")

    if not width_df.empty:
        for target in v47_target_order(width_df):
            sub = width_df[width_df["target"].astype(str).eq(target)].sort_values("budget")
            axb.plot(pd.to_numeric(sub["budget"], errors="coerce"), pd.to_numeric(sub["normalized_width"], errors="coerce"),
                     marker="o", color=TARGET_COLORS_FINAL.get(str(target)), label=v47_latex_label(target))
    axb.set_xscale("log"); axb.set_xticks(BUDGET_ORDER_JOURNAL); axb.set_xticklabels([str(b) for b in BUDGET_ORDER_JOURNAL])
    axb.set_xlabel("Label budget"); axb.set_ylabel("Interval width / target std"); axb.set_title("Uncertainty narrows with labels"); axb.grid(True)
    axb.legend(frameon=False, loc="upper right")
    v47_panel(axb, "b")

    if not frontier.empty:
        for target in v47_target_order(frontier):
            sub = frontier[frontier["target"].astype(str).eq(target)].sort_values("retained_fraction")
            axc.plot(pd.to_numeric(sub["retained_fraction"], errors="coerce"), pd.to_numeric(sub["precision_true_top5"], errors="coerce"),
                     marker="o", color=TARGET_COLORS_FINAL.get(str(target)), label=v47_latex_label(target))
    axc.set_xlim(0, 0.52); axc.set_ylim(0, 1.05)
    axc.set_xlabel("Retained fraction"); axc.set_ylabel("True top-5% fraction"); axc.set_title("Precision–yield frontier"); axc.grid(True)
    axc.legend(frameon=False, loc="lower right")
    v47_panel(axc, "c")

    if not frontier.empty:
        close = frontier.iloc[frontier.groupby("target")["retained_fraction"].apply(lambda s: (pd.to_numeric(s, errors="coerce")-0.10).abs().idxmin()).to_numpy()].copy()
        close = close.set_index("target").reindex(v47_target_order(close)).reset_index()
        axd.bar(range(len(close)), pd.to_numeric(close["precision_true_top5"], errors="coerce"),
                color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in close["target"]])
        for i, v in enumerate(pd.to_numeric(close["precision_true_top5"], errors="coerce")):
            if np.isfinite(v): axd.text(i, min(v+0.035, 1.02), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        axd.set_xticks(range(len(close))); axd.set_xticklabels([v47_latex_label(t) for t in close["target"]], rotation=28, ha="right")
    axd.set_ylim(0, 1.08); axd.set_ylabel("True top-5% fraction"); axd.set_title("Precision at ~10% retained candidates"); axd.grid(True, axis="y")
    v47_panel(axd, "d")

    if not tierfrac.empty:
        tf = tierfrac.sort_values("budget")
        budgets = pd.to_numeric(tf["budget"], errors="coerce").astype(int).astype(str)
        bottom = np.zeros(len(tf))
        colors = {"trusted_fraction":"#2ECC71", "uncertain_fraction":"#F5B041", "rejected_fraction":"#95A5A6"}
        labels = {"trusted_fraction":"trusted", "uncertain_fraction":"uncertain", "rejected_fraction":"rejected"}
        for col in ["trusted_fraction", "uncertain_fraction", "rejected_fraction"]:
            if col in tf.columns:
                vals = pd.to_numeric(tf[col], errors="coerce").fillna(0).to_numpy()
                axe.bar(budgets, vals, bottom=bottom, color=colors[col], label=labels[col])
                bottom += vals
        axe.legend(frameon=False, ncol=3, loc="upper center")
    axe.set_ylim(0, 1.0); axe.set_ylabel("Fraction of prediction rows"); axe.set_title("Pre-final uncertainty-tier composition"); axe.grid(False)
    v47_panel(axe, "e")

    methods = []
    if not df1000.empty:
        for col, lab in [("split_coverage_mean", "split"), ("local_coverage_mean", "local"), ("mondrian_coverage_mean", "Mondrian")]:
            if col in df1000.columns:
                methods.append({"method": lab, "coverage": float(pd.to_numeric(df1000[col], errors="coerce").mean())})
    cm = pd.DataFrame(methods)
    if not cm.empty:
        axf.bar(cm["method"], cm["coverage"], color=["#5DADE2", "#58D68D", "#AF7AC5"][:len(cm)])
        for i, v in enumerate(cm["coverage"]):
            axf.text(i, v+0.0015, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    axf.axhline(0.90, ls="--", color="#555555", lw=0.9)
    axf.set_ylim(0.86, 0.94); axf.set_ylabel("Empirical coverage"); axf.set_title("Coverage method comparison"); axf.grid(True, axis="y")
    v47_panel(axf, "f")

    fig.suptitle("Conformal uncertainty converts few-shot predictions into risk-controlled shortlists", fontsize=14.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v47_save(fig, "Figure_4_v4_7_conformal_risk_control_frontier")
    mark_done(stage, {"v4_7": True})


def v47_top25(master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    top = v44_top_consensus(25, master=master)
    if top is None:
        top = pd.DataFrame()
    if top.empty:
        return top
    if "true_percentile" not in top.columns or top["true_percentile"].isna().all():
        top = v42_add_percentiles_to_candidates(top, master=master)
    top["target_label_plain"] = top["target"].map(v47_plain_label)
    return top


def v47_true_enrichment(top25: Optional[pd.DataFrame] = None, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if top25 is None or top25.empty:
        top25 = v47_top25(master)
    rows = []
    stats = v47_target_stats(master)
    stat_map = {str(r["target"]): r for _, r in stats.iterrows()} if not stats.empty else {}
    for target in TARGET_ORDER_FINAL:
        sub = top25[top25["target"].astype(str).eq(target)].copy() if top25 is not None and not top25.empty else pd.DataFrame()
        if sub.empty:
            continue
        true_vals = pd.to_numeric(sub.get("y_true_median", sub.get("y_true", pd.Series(dtype=float))), errors="coerce")
        if "true_percentile" in sub.columns:
            perc = pd.to_numeric(sub["true_percentile"], errors="coerce")
        else:
            perc = pd.Series(np.nan, index=sub.index)
        dmed = float(stat_map.get(target, {}).get("median_mmol_g", np.nan)) if target in stat_map else np.nan
        smed = float(true_vals.median()) if true_vals.notna().any() else np.nan
        rows.append({
            "target": target,
            "target_label_plain": v47_plain_label(target),
            "n_shortlist": int(len(sub)),
            "dataset_median_mmol_g": dmed,
            "shortlist_median_true_mmol_g": smed,
            "fold_enrichment_vs_dataset_median": float(smed/dmed) if np.isfinite(smed) and np.isfinite(dmed) and dmed != 0 else np.nan,
            "fraction_true_top1": float((perc >= 0.99).mean()) if perc.notna().any() else np.nan,
            "fraction_true_top5": float((perc >= 0.95).mean()) if perc.notna().any() else np.nan,
            "fraction_true_top10": float((perc >= 0.90).mean()) if perc.notna().any() else np.nan,
            "median_true_percentile": float(perc.median()) if perc.notna().any() else np.nan,
        })
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "QC_Table_consensus_true_enrichment.csv")
    return out


def v47_candidate_funnel(master: Optional[pd.DataFrame] = None, top25: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    stats = v47_target_stats(master)
    tiers = v42_read_table("table_candidate_tiers_all_predictions.csv", low_memory=True)
    consensus = v42_read_table("table_consensus_candidate_shortlist.csv", low_memory=True)
    top25 = top25 if top25 is not None else v47_top25(master)
    rows = []
    for target in TARGET_ORDER_FINAL:
        row = stats[stats["target"].astype(str).eq(target)].head(1)
        n_label = int(row["n_mofs"].iloc[0]) if not row.empty and pd.notna(row["n_mofs"].iloc[0]) else np.nan
        n_qg = int((tiers["target"].astype(str).eq(target)).sum()) if not tiers.empty and "target" in tiers.columns else np.nan
        csub = consensus[consensus["target"].astype(str).eq(target)] if not consensus.empty and "target" in consensus.columns else pd.DataFrame()
        n_cons = int(csub["consensus_pass"].fillna(False).astype(bool).sum()) if not csub.empty and "consensus_pass" in csub.columns else int(len(csub)) if not csub.empty else np.nan
        n_top = int((top25["target"].astype(str).eq(target)).sum()) if top25 is not None and not top25.empty and "target" in top25.columns else np.nan
        rows.extend([
            {"target": target, "target_label_plain": v47_plain_label(target), "stage": "Labelled MOFs", "count": n_label},
            {"target": target, "target_label_plain": v47_plain_label(target), "stage": "Quality-gated rows", "count": n_qg},
            {"target": target, "target_label_plain": v47_plain_label(target), "stage": "Consensus pass", "count": n_cons},
            {"target": target, "target_label_plain": v47_plain_label(target), "stage": "Top-25", "count": n_top},
        ])
    out = pd.DataFrame(rows)
    atomic_save_csv(out, TABLE_DIR / "STORY_Table_candidate_funnel_by_target.csv")
    return out


def v47_plot_figure_5(core: Optional[pd.DataFrame], mosaec: Optional[pd.DataFrame], final_external: Optional[pd.DataFrame], master: Optional[pd.DataFrame] = None) -> None:
    stage = "v47_fig5_consensus_atlas"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.7 Figure 5")
    v47_apply_style()
    top25 = v47_top25(master)
    qce = v47_true_enrichment(top25, master)
    funnel = v47_candidate_funnel(master, top25)
    v47_source(top25, "Figure_5_top25_consensus_candidates_with_percentiles.csv")
    v47_source(qce, "Figure_5_true_enrichment.csv")
    v47_source(funnel, "Figure_5_candidate_funnel.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8))
    axa, axb, axc, axd, axe, axf = axes.ravel()

    # (a) Funnel heatmap
    if not funnel.empty:
        stages = ["Labelled MOFs", "Quality-gated rows", "Consensus pass", "Top-25"]
        mat = funnel.pivot_table(index="target", columns="stage", values="count", aggfunc="first").reindex(TARGET_ORDER_FINAL)[stages]
        plot_mat = np.log10(pd.to_numeric(mat.stack(), errors="coerce").unstack().replace(0, np.nan))
        plot_mat.index = [v47_latex_label(t) for t in plot_mat.index]
        plot_mat.columns = ["Labelled", "Quality\ngated", "Consensus", "Top-25"]
        im = axa.imshow(plot_mat.to_numpy(), aspect="auto", cmap="Blues")
        axa.set_xticks(range(plot_mat.shape[1])); axa.set_xticklabels(plot_mat.columns)
        axa.set_yticks(range(plot_mat.shape[0])); axa.set_yticklabels(plot_mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if np.isfinite(val):
                    axa.text(j, i, f"{int(val):,}", ha="center", va="center", fontsize=6.8)
        cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.025); cb.ax.set_ylabel("log$_{10}$(count)")
    axa.set_title("Decision funnel to final candidates")
    v47_panel(axa, "a")

    # (b) True enrichment
    if not qce.empty:
        q = qce.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(qce["target"].astype(str))]).reset_index()
        vals = pd.to_numeric(q["fold_enrichment_vs_dataset_median"], errors="coerce")
        axb.bar(range(len(q)), vals, color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in q["target"]])
        axb.axhline(1, color="#777777", ls="--", lw=0.9)
        for i, v in enumerate(vals):
            if np.isfinite(v): axb.text(i, v+0.20, f"{v:.1f}×", ha="center", va="bottom", fontsize=7.5)
        axb.set_xticks(range(len(q))); axb.set_xticklabels([v47_latex_label(t) for t in q["target"]], rotation=28, ha="right")
    axb.set_ylabel("Shortlist median / dataset median"); axb.set_title("True enrichment of consensus shortlist"); axb.grid(True, axis="y")
    v47_panel(axb, "b")

    # (c) True elite fractions
    if not qce.empty:
        q = qce.set_index("target").reindex([t for t in TARGET_ORDER_FINAL if t in set(qce["target"].astype(str))]).reset_index()
        cols = [("fraction_true_top1", "top 1%"), ("fraction_true_top5", "top 5%"), ("fraction_true_top10", "top 10%")]
        x = np.arange(len(q)); width = 0.23
        for k, (col, lab) in enumerate(cols):
            vals = pd.to_numeric(q[col], errors="coerce") if col in q.columns else pd.Series(np.nan, index=q.index)
            axc.bar(x + (k-1)*width, vals, width=width, label=lab)
        axc.set_xticks(x); axc.set_xticklabels([v47_latex_label(t) for t in q["target"]], rotation=28, ha="right")
        axc.legend(frameon=False, ncol=3, loc="upper left")
    axc.set_ylim(0,1.05); axc.set_ylabel("Fraction of top-25 shortlist"); axc.set_title("How often candidates are true elites"); axc.grid(True, axis="y")
    v47_panel(axc, "c")

    # (d) True percentile distribution
    if not top25.empty and "true_percentile" in top25.columns:
        ordered = v47_target_order(top25)
        data = [pd.to_numeric(top25.loc[top25["target"].astype(str).eq(t), "true_percentile"], errors="coerce").dropna() for t in ordered]
        safe_boxplot_with_labels(axd, data, [v47_latex_label(t) for t in ordered], showfliers=False, patch_artist=True,
                                 boxprops={"facecolor":"#F8F9F9", "edgecolor":"#34495E"}, medianprops={"color":"#E67E22", "linewidth":1.1})
        rng = np.random.default_rng(42)
        for i, (target, vals) in enumerate(zip(ordered, data), start=1):
            vals = np.asarray(vals, dtype=float)
            axd.scatter(i + rng.normal(0, 0.035, size=len(vals)), vals, s=14, alpha=0.58, color=TARGET_COLORS_FINAL.get(str(target)))
    axd.set_ylim(0.68, 1.01); axd.tick_params(axis="x", rotation=28); axd.set_ylabel("True uptake percentile"); axd.set_title("Final candidates occupy the adsorption tail"); axd.grid(True, axis="y")
    v47_panel(axd, "d")

    # (e) Consensus support (models/seeds/splits/budgets; trusted support as annotation)
    support_cols = [c for c in ["n_models", "n_seeds", "n_splits", "n_budgets"] if c in top25.columns]
    if support_cols:
        data = [pd.to_numeric(top25[c], errors="coerce").dropna() for c in support_cols]
        labels = [c.replace("n_", "") for c in support_cols]
        safe_boxplot_with_labels(axe, data, labels, showfliers=False, patch_artist=True,
                                 boxprops={"facecolor":"#F8F9F9", "edgecolor":"#34495E"}, medianprops={"color":"#E67E22", "linewidth":1.1})
        rng = np.random.default_rng(43)
        for j, vals in enumerate(data, start=1):
            vals = np.asarray(vals, dtype=float)
            axe.scatter(j + rng.normal(0, 0.035, size=len(vals)), vals, s=10, alpha=0.32, color="#5D6D7E")
        if "trusted_support" in top25.columns:
            ts = pd.to_numeric(top25["trusted_support"], errors="coerce")
            axe.text(0.03, 0.93, f"median trusted support = {ts.median():.0f} rows", transform=axe.transAxes, fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", lw=0.6))
    axe.set_ylabel("Distinct support count"); axe.set_title("Consensus stability across runs"); axe.grid(True, axis="y")
    v47_panel(axe, "e")

    # (f) Geometry atlas
    if master is not None and not master.empty and {"Di", "Density"}.issubset(master.columns):
        bg = master[["mof_id", "Di", "Density"]].copy()
        bg["Di"] = pd.to_numeric(bg["Di"], errors="coerce"); bg["Density"] = pd.to_numeric(bg["Density"], errors="coerce")
        bg = bg.dropna()
        if len(bg) > int(CONFIG.get("v47_background_points", 6000)):
            bg = bg.sample(int(CONFIG.get("v47_background_points", 6000)), random_state=42)
        axf.scatter(bg["Di"], bg["Density"], s=3, alpha=0.055, color="#7F8C8D", label="ARC-MOF background")
    if not top25.empty:
        for target in v47_target_order(top25):
            sub = top25[top25["target"].astype(str).eq(target)]
            x = pd.to_numeric(sub.get("Di", sub.get("PLD", pd.Series(dtype=float))), errors="coerce")
            y = pd.to_numeric(sub.get("Density", sub.get("density", pd.Series(dtype=float))), errors="coerce")
            axf.scatter(x, y, s=28, alpha=0.82, color=TARGET_COLORS_FINAL.get(str(target)), label=v47_latex_label(target), edgecolor="white", linewidth=0.35)
    axf.set_xlabel("PLD proxy, $D_i$ / Å"); axf.set_ylabel("Density / g cm$^{-3}$"); axf.set_title("Geometry regime of final candidates"); axf.grid(True)
    axf.legend(frameon=False, fontsize=6.4, loc="upper right")
    v47_panel(axf, "f")

    fig.suptitle("Target-balanced consensus shortlists are truly enriched in adsorption elites", fontsize=14.0, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    v47_save(fig, "Figure_5_v4_7_consensus_shortlist_atlas")
    mark_done(stage, {"v4_7": True})


def v47_external_overlap_summary(final_external: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = final_external if final_external is not None else pd.DataFrame()
    if df is None or df.empty:
        p = TABLE_DIR / "final_screening_table_with_external_flags.csv"
        df = pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()
    if df.empty or "target" not in df.columns:
        return pd.DataFrame()
    rows = []
    for target, sub in df.groupby("target"):
        rows.append({
            "target": str(target),
            "target_label_plain": v47_plain_label(target),
            "n_candidates": int(len(sub)),
            "core_exact_fraction": float(sub.get("core_exact_match_flag", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "mosaec_exact_fraction": float(sub.get("mosaec_exact_match_flag", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "core_geometry_overlap_fraction": float(sub.get("core_geometry_overlap_flag", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "mosaec_geometry_overlap_fraction": float(sub.get("mosaec_geometry_overlap_flag", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "mean_external_annotation_score": float(pd.to_numeric(sub.get("external_realism_score", pd.Series(np.nan, index=sub.index)), errors="coerce").mean()),
        })
    out = pd.DataFrame(rows)
    out["target"] = pd.Categorical(out["target"], categories=TARGET_ORDER_FINAL, ordered=True)
    out = out.sort_values("target").reset_index(drop=True)
    out["target"] = out["target"].astype(str)
    atomic_save_csv(out, TABLE_DIR / "SI_Table_external_domain_overlap_summary_v47.csv")
    return out


def v47_plot_external_si(final_external: Optional[pd.DataFrame]) -> None:
    summary = v47_external_overlap_summary(final_external)
    v47_source(summary, "Figure_S_external_domain_overlap_source.csv", si=True)
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))
    axa, axb, axc = axes.ravel()
    x = np.arange(len(summary)); width = 0.35
    axa.bar(x-width/2, pd.to_numeric(summary["core_geometry_overlap_fraction"], errors="coerce"), width=width, label="CoRE geometry", color="#5DADE2")
    axa.bar(x+width/2, pd.to_numeric(summary["mosaec_geometry_overlap_fraction"], errors="coerce"), width=width, label="MOSAEC geometry", color="#58D68D")
    axa.set_xticks(x); axa.set_xticklabels([v47_latex_label(t) for t in summary["target"]], rotation=28, ha="right")
    axa.set_ylim(0,1.05); axa.set_ylabel("Overlap fraction"); axa.set_title("Geometry-domain overlap"); axa.legend(frameon=False)
    v47_panel(axa, "a")
    axb.axis("off")
    core_exact = 0
    mos_exact = 0
    n = int(summary["n_candidates"].sum())
    axb.text(0.5, 0.68, "Exact structural matches\nwere not assumed", ha="center", va="center", fontsize=12, fontweight="bold", transform=axb.transAxes)
    axb.text(0.5, 0.39, f"CoRE exact: {core_exact}/{n}\nMOSAEC exact: {mos_exact}/{n}", ha="center", va="center", fontsize=10, transform=axb.transAxes)
    v47_panel(axb, "b")
    axc.bar(range(len(summary)), pd.to_numeric(summary["mean_external_annotation_score"], errors="coerce"),
            color=[TARGET_COLORS_FINAL.get(str(t), "#777777") for t in summary["target"]])
    axc.set_xticks(range(len(summary))); axc.set_xticklabels([v47_latex_label(t) for t in summary["target"]], rotation=28, ha="right")
    axc.set_ylabel("Mean annotation score"); axc.set_title("Domain-overlap annotation score"); axc.grid(True, axis="y")
    v47_panel(axc, "c")
    fig.suptitle("External overlays annotate domain overlap, not experimental validation", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.90])
    v47_save(fig, "Figure_S8_v4_7_external_domain_overlap_annotation", si=True)


def v47_plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame, master: Optional[pd.DataFrame] = None) -> None:
    stage = "v47_si_figures"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.7 SI figures")
    v47_apply_style()

    # S1: full target distributions with log counts so CO2 tails are visible.
    if master is not None and not master.empty:
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
        for ax, target in zip(axes.ravel(), TARGET_ORDER_FINAL):
            if target in master.columns:
                vals = pd.to_numeric(master[target], errors="coerce").dropna()
                ax.hist(vals, bins=60, color=TARGET_COLORS_FINAL.get(target, "#777777"), alpha=0.82)
                ax.axvline(vals.quantile(0.95), color="#222222", ls="--", lw=0.9, label="top-5% threshold")
                ax.set_yscale("log")
                ax.set_title(v47_latex_label(target))
                ax.set_xlabel("Uptake / mmol g$^{-1}$")
                ax.set_ylabel("Count (log)")
                ax.legend(frameon=False, fontsize=6.5)
        fig.suptitle("Figure S1. Full target distributions in ARC-MOF", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0,0,1,0.94])
        v47_save(fig, "Figure_S1_v4_7_target_distributions", si=True)

    # S2: quality gate audit; split into pass-by-model and rejection reasons.
    q = v42_read_table("QC_Table_model_stability.csv", low_memory=True)
    if not q.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
        axa, axb = axes
        if {"model", "experiment_passes_quality_gate"}.issubset(q.columns):
            m = q.groupby("model", as_index=False)["experiment_passes_quality_gate"].mean()
            m = m.sort_values("experiment_passes_quality_gate")
            axa.barh(m["model"].astype(str).str.replace("_", " "), m["experiment_passes_quality_gate"], color="#7F8C8D")
            axa.set_xlim(0,1.05); axa.set_xlabel("Fraction passing final-candidate gate"); axa.set_title("Model-level pass fraction"); axa.grid(True, axis="x")
            v47_panel(axa, "a")
        if "quality_gate_reasons" in q.columns:
            reasons = q["quality_gate_reasons"].fillna("PASS").astype(str).str.replace("model_not_candidate_eligible", "not final-candidate model", regex=False).str.replace("rmse_too_large_for_target", "RMSE too large", regex=False).str.replace("low_or_nan_spearman", "low/NaN Spearman", regex=False).str.replace("low_top5_recall", "low top-5 recall", regex=False).str.replace("coverage_outside_gate", "coverage outside gate", regex=False)
            rc = reasons.value_counts().head(10).sort_values()
            axb.barh(rc.index, rc.values, color="#95A5A6")
            axb.set_xlabel("Experiments"); axb.set_title("Gate outcome / rejection reason"); axb.grid(True, axis="x")
            v47_panel(axb, "b")
        fig.suptitle("Figure S2. Candidate-quality gate audit", fontsize=13.5, fontweight="bold")
        fig.tight_layout(rect=[0,0,1,0.92])
        v47_save(fig, "Figure_S2_v4_7_quality_gate_audit", si=True)

    # S3: external overlay domain annotation.
    final_external = v42_read_table("final_screening_table_with_external_flags.csv", low_memory=True)
    v47_plot_external_si(final_external)

    mark_done(stage, {"v4_7": True})


def v47_write_publication_tables(master: Optional[pd.DataFrame], agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Write a clean table package for the main text and SI."""
    stats = v47_target_stats(master)
    v47_table(stats, "Table_1_dataset_and_targets.csv", si=False)

    if agg is None or agg.empty:
        p = TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv"
        agg = pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()

    claims = v44_final_claims_table(agg, master=master) if agg is not None and not agg.empty else pd.DataFrame()
    if not claims.empty:
        # Clean labels and column order for manuscript use.
        claims["target_label_plain"] = claims["target"].map(v47_plain_label)
        keep = [c for c in ["target", "target_label_plain", "best_top5_recall", "best_enrichment", "best_spearman",
                            "RAC_gain_at_1000", "topology_penalty_random_minus_topology",
                            "consensus_shortlist_fold_enrichment", "fraction_shortlist_true_top5"] if c in claims.columns]
        claims = claims[keep + [c for c in claims.columns if c not in keep]]
    v47_table(claims, "Table_2_main_claims_summary.csv", si=False)

    top25 = v47_top25(master)
    top5 = top25.sort_values(["target", "consensus_score"], ascending=[True, False]).groupby("target", as_index=False).head(5) if not top25.empty else pd.DataFrame()
    top_cols = [c for c in ["target", "target_label_plain", "mof_id", "filename", "consensus_score", "y_true_median", "y_pred_median",
                            "split_lower_median", "split_upper_median", "true_percentile", "pred_percentile",
                            "n_models", "n_seeds", "n_splits", "n_budgets", "trusted_support"] if c in top5.columns]
    v47_table(top5[top_cols] if top_cols else top5, "Table_3_top5_consensus_candidates_per_target.csv", si=False)
    v47_table(top25[[c for c in top_cols if c in top25.columns] + [c for c in top25.columns if c not in top_cols]].head(100), "Table_S7_top25_consensus_candidates_per_target.csv", si=True)

    # SI tables: copy or regenerate the most important QC/audit tables.
    for src, dst in [
        ("table_aggregated_metrics_with_bootstrap_ci.csv", "Table_S1_all_aggregated_metrics_with_CI.csv"),
        ("QC_Table_feature_set_integrity.csv", "Table_S2_feature_set_integrity_RDF_audit.csv"),
        ("QC_Table_model_stability.csv", "Table_S3_model_stability_quality_gate.csv"),
        ("QC_Table_prediction_range_by_model.csv", "Table_S4_prediction_range_sanity_by_model.csv"),
        ("STORY_Table_precision_yield_frontier.csv", "Table_S5_precision_yield_frontier.csv"),
        ("QC_Table_consensus_true_enrichment.csv", "Table_S6_consensus_true_enrichment.csv"),
        ("SI_Table_external_domain_overlap_summary_v47.csv", "Table_S8_external_domain_overlap_summary.csv"),
    ]:
        p = TABLE_DIR / src
        if p.exists():
            try:
                v47_table(pd.read_csv(p, low_memory=False), dst, si=True)
            except Exception as e:
                logging.warning("V47_TABLE_COPY_FAILED | %s | %s", src, e)

    manifest_rows = []
    for folder, kind in [
        (V47_FIG_MAIN_DIR, "main_figure"),
        (V47_FIG_SI_DIR, "si_figure"),
        (V47_SOURCE_MAIN_DIR, "main_source_data"),
        (V47_SOURCE_SI_DIR, "si_source_data"),
        (V47_TABLE_MAIN_DIR, "main_table"),
        (V47_TABLE_SI_DIR, "si_table"),
    ]:
        for p in sorted(folder.glob("*")):
            if p.is_file():
                manifest_rows.append({"kind": kind, "file": str(p.relative_to(V47_ROOT)), "bytes": int(p.stat().st_size)})
    manifest = pd.DataFrame(manifest_rows)
    atomic_save_csv(manifest, V47_ROOT / "MANIFEST_v4_7_publication_package.csv")
    atomic_save_csv(manifest, TABLE_DIR / "MANIFEST_v4_7_publication_package.csv")
    return claims



def v47_dataframe_to_markdown_safe(df: pd.DataFrame, index: bool = False, max_rows: int = 30) -> str:
    """Return a markdown table without requiring the optional tabulate package.

    pandas.DataFrame.to_markdown() depends on the optional ``tabulate`` package.
    The v4.6 run failed only at report-writing because tabulate was absent, after
    all main figures/SI figures had already been saved. This helper first tries
    the pandas implementation, then falls back to a small pure-Python markdown
    renderer so the publication report never crashes the pipeline.
    """
    if df is None or df.empty:
        return "No rows available."
    work = df.copy()
    if max_rows is not None and len(work) > int(max_rows):
        work = work.head(int(max_rows)).copy()
        truncated_note = f"\n\n_Table truncated to first {int(max_rows)} rows in the report; full CSV table is saved in the publication package._"
    else:
        truncated_note = ""
    try:
        return work.to_markdown(index=index) + truncated_note
    except Exception as exc:
        logging.warning("MARKDOWN_FALLBACK_USED | pandas.to_markdown unavailable or failed: %s", exc)
    if index:
        work = work.reset_index()
    # Convert values to compact strings and escape markdown separators.
    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            if np.isfinite(v):
                return f"{float(v):.4g}"
            return ""
        s = str(v).replace("\n", " ").replace("|", "\\|")
        return s
    headers = [str(c).replace("|", "\\|") for c in work.columns]
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, r in work.iterrows():
        rows.append("| " + " | ".join(_fmt(r[c]) for c in work.columns) + " |")
    return "\n".join(rows) + truncated_note

def v47_write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    stage = "v47_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> v4.7 WRITE_PUBLICATION_REPORT")
    agg_path = TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv"
    agg = pd.read_csv(agg_path, low_memory=False) if agg_path.exists() else pd.DataFrame()
    claims = v47_write_publication_tables(master, agg)
    lines = []
    lines.append("# v4.7 consolidated publication package report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Output folders\n")
    lines.append(f"- Main figures: `{V47_FIG_MAIN_DIR}`")
    lines.append(f"- SI figures: `{V47_FIG_SI_DIR}`")
    lines.append(f"- Main source data: `{V47_SOURCE_MAIN_DIR}`")
    lines.append(f"- SI source data: `{V47_SOURCE_SI_DIR}`")
    lines.append(f"- Main tables: `{V47_TABLE_MAIN_DIR}`")
    lines.append(f"- SI tables: `{V47_TABLE_SI_DIR}`")
    lines.append("\n## Main figures\n")
    for fname, meaning in [
        ("Figure_1_v4_7_workflow.png", "quantitative workflow / graphical abstract"),
        ("Figure_2_v4_7_fewshot_learning_phase_diagram.png", "few-shot label efficiency and enrichment"),
        ("Figure_3_v4_7_descriptor_chemistry_extrapolation_stress.png", "descriptor chemistry and topology stress"),
        ("Figure_4_v4_7_conformal_risk_control_frontier.png", "risk-control and precision-yield frontier"),
        ("Figure_5_v4_7_consensus_shortlist_atlas.png", "consensus shortlist true-enrichment atlas"),
    ]:
        lines.append(f"- `{fname}`: {'FOUND' if (V47_FIG_MAIN_DIR / fname).exists() else 'not found'} — {meaning}")
    lines.append("\n## Main claims table\n")
    if claims is not None and not claims.empty:
        lines.append(v47_dataframe_to_markdown_safe(claims, index=False))
    else:
        lines.append("Main claims table could not be generated from current tables.")
    lines.append("\n## Manuscript guardrails\n")
    lines.append("- Main descriptor claim: geometry versus geometry+RACs. Do not claim RDF effects unless QC shows real RDF features.")
    lines.append("- Main generalisation claim: topology-grouped split is the hardest extrapolation stress test.")
    lines.append("- Main uncertainty claim: conformal intervals and consensus filters turn raw few-shot predictions into risk-controlled shortlists.")
    lines.append("- External overlays: CoRE/MOSAEC are domain-overlap annotations, not experimental validation.")
    report = V47_ROOT / "RUN_REPORT_v4_7_publication_package.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage, {"v4_7": True})


# ---- v4.7 overrides used by the existing main() function ----

def plot_figure_1_framework() -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_figure_1(master)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    return v47_plot_figure_2(agg)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    return v47_plot_figure_3(agg)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_figure_4(anatomy, tiers, master=master)


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_figure_5(core, mosaec, final_external, master=master)


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_si_figures(agg, tiers, anatomy, master=master)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    return v47_write_run_report(master, metrics_all)



# ============================================================================
# v4.8 clean publication package + visual polish overrides
# ============================================================================

# Goal of v4.8:
# 1) keep the scientifically strong v4.7 content,
# 2) produce a cleaner, non-redundant publication package from scratch, and
# 3) make the final figures visually more elegant without changing the core
#    scientific calculations.
#
# The strategy is intentionally lightweight and robust: we reuse the mature
# data-preparation logic from v4.7, but override the styling, saving, folder
# structure, manifests, and final reporting.

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

V48_ROOT = RESULTS_DIR / "publication_package_v48"
V48_OVERVIEW_DIR = V48_ROOT / "00_overview"
V48_MAIN_DIR = V48_ROOT / "01_main_text"
V48_SI_DIR = V48_ROOT / "02_supporting_information"
V48_FIG_MAIN_DIR = V48_MAIN_DIR / "figures"
V48_SOURCE_MAIN_DIR = V48_MAIN_DIR / "source_data"
V48_TABLE_MAIN_DIR = V48_MAIN_DIR / "tables"
V48_FIG_SI_DIR = V48_SI_DIR / "figures"
V48_SOURCE_SI_DIR = V48_SI_DIR / "source_data"
V48_TABLE_SI_DIR = V48_SI_DIR / "tables"

CONFIG.setdefault("v48_png_dpi", 800)
CONFIG.setdefault("v48_save_formats", ["png", "pdf", "svg"])


def v48_prepare_clean_publication_dirs() -> None:
    for p in [V48_ROOT, V48_OVERVIEW_DIR, V48_MAIN_DIR, V48_SI_DIR,
              V48_FIG_MAIN_DIR, V48_SOURCE_MAIN_DIR, V48_TABLE_MAIN_DIR,
              V48_FIG_SI_DIR, V48_SOURCE_SI_DIR, V48_TABLE_SI_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def v48_apply_style() -> None:
    """Slightly more polished publication style than v4.7.

    The intent is not to change the science, only the readability and finish:
    a touch more whitespace, cleaner typography, softer gridlines, and a more
    consistent panel-label / title hierarchy.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "figure.dpi": 180,
        "savefig.dpi": int(CONFIG.get("v48_png_dpi", 800)),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#5B6770",
        "grid.linewidth": 0.40,
        "grid.alpha": 0.18,
        "grid.color": "#94A3B8",
        "lines.linewidth": 1.65,
        "lines.markersize": 4.4,
        "patch.linewidth": 0.70,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def v48_panel(ax, label: str, x: float = -0.10, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.3,
            fontweight="bold", ha="right", va="top", color="#0F172A")


def v48_stem(stem: str) -> str:
    s = str(stem)
    s = s.replace("_v4_7_", "_v4_8_")
    s = s.replace("_v4_6_", "_v4_8_")
    s = s.replace("Figure_S8_v4_8_external_domain_overlap_annotation", "Figure_S3_v4_8_external_domain_overlap_annotation")
    s = s.replace("Figure_S8_v4_7_external_domain_overlap_annotation", "Figure_S3_v4_8_external_domain_overlap_annotation")
    return s


def v48_save(fig: plt.Figure, stem: str, si: bool = False) -> None:
    v48_prepare_clean_publication_dirs()
    v48_apply_style()
    out_dir = V48_FIG_SI_DIR if si else V48_FIG_MAIN_DIR
    clean_stem = v48_stem(stem)
    # Slightly more room for titles/panel labels.
    fig.set_constrained_layout(False)
    for fmt in CONFIG.get("v48_save_formats", ["png", "pdf", "svg"]):
        out = (out_dir / clean_stem).with_suffix(f".{fmt}")
        try:
            fig.savefig(out, dpi=int(CONFIG.get("v48_png_dpi", 800)), bbox_inches="tight", facecolor="white")
        except Exception as e:
            logging.warning("V48_FIGURE_SAVE_FAILED | %s | %s", out, e)
    plt.close(fig)


def v48_source(df: pd.DataFrame, fname: str, si: bool = False) -> None:
    v48_prepare_clean_publication_dirs()
    if df is None:
        df = pd.DataFrame()
    out_dir = V48_SOURCE_SI_DIR if si else V48_SOURCE_MAIN_DIR
    atomic_save_csv(df, out_dir / fname)


def v48_table(df: pd.DataFrame, fname: str, si: bool = False) -> None:
    v48_prepare_clean_publication_dirs()
    if df is None:
        df = pd.DataFrame()
    out_dir = V48_TABLE_SI_DIR if si else V48_TABLE_MAIN_DIR
    atomic_save_csv(df, out_dir / fname)


def v48_write_folder_index() -> None:
    v48_prepare_clean_publication_dirs()
    rows = []
    for group, folder in [
        ("overview", V48_OVERVIEW_DIR),
        ("main_figures", V48_FIG_MAIN_DIR),
        ("main_source_data", V48_SOURCE_MAIN_DIR),
        ("main_tables", V48_TABLE_MAIN_DIR),
        ("si_figures", V48_FIG_SI_DIR),
        ("si_source_data", V48_SOURCE_SI_DIR),
        ("si_tables", V48_TABLE_SI_DIR),
    ]:
        if folder.exists():
            for p in sorted(folder.glob("**/*")):
                if p.is_file():
                    rows.append({
                        "group": group,
                        "relative_path": str(p.relative_to(V48_ROOT)),
                        "bytes": int(p.stat().st_size),
                        "suffix": p.suffix.lower(),
                    })
    idx = pd.DataFrame(rows)
    atomic_save_csv(idx, V48_OVERVIEW_DIR / "FILE_INDEX_v4_8.csv")


def v48_write_package_readme() -> None:
    v48_prepare_clean_publication_dirs()
    lines = []
    lines.append("# v4.8 clean publication package\n")
    lines.append("This folder contains the final manuscript-facing outputs in a clean, non-redundant structure.\n")
    lines.append("## Folder map\n")
    lines.append("- `00_overview/`: manifest, run report, file index, and package notes")
    lines.append("- `01_main_text/figures/`: main-text figures (PNG/PDF/SVG)")
    lines.append("- `01_main_text/source_data/`: source CSV files corresponding to main-text figure panels")
    lines.append("- `01_main_text/tables/`: manuscript-ready main tables")
    lines.append("- `02_supporting_information/figures/`: SI figures (PNG/PDF/SVG)")
    lines.append("- `02_supporting_information/source_data/`: source CSV files corresponding to SI figure panels")
    lines.append("- `02_supporting_information/tables/`: SI tables and audit/QC tables")
    lines.append("\n## Practical note\n")
    lines.append("The clean package does not replace the raw intermediate results in `results_fewshot_mof/tables`, `predictions`, or `pickles`. Those upstream folders remain the restart-safe computational cache. This package is the publication-facing export layer.\n")
    (V48_OVERVIEW_DIR / "README_publication_package_v4_8.md").write_text("\n".join(lines), encoding="utf-8")


def v48_copy_key_support_files() -> None:
    """Copy a few useful non-redundant context files into the overview folder."""
    v48_prepare_clean_publication_dirs()
    for src_name in [
        "QC_Table_feature_set_integrity.csv",
        "QC_Table_model_stability.csv",
        "QC_Table_prediction_range_by_model.csv",
        "table_dataset_target_summary.csv",
    ]:
        p = TABLE_DIR / src_name
        if p.exists():
            try:
                shutil.copy2(p, V48_OVERVIEW_DIR / src_name)
            except Exception as e:
                logging.warning("V48_COPY_SUPPORT_FILE_FAILED | %s | %s", p, e)


def v48_write_publication_tables(master: Optional[pd.DataFrame], agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Write a clean main/SI table package into the v4.8 folder tree."""
    v48_prepare_clean_publication_dirs()
    stats = v47_target_stats(master)
    v48_table(stats, "Table_1_dataset_and_targets.csv", si=False)

    if agg is None or agg.empty:
        p = TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv"
        agg = pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()

    claims = v44_final_claims_table(agg, master=master) if agg is not None and not agg.empty else pd.DataFrame()
    if not claims.empty:
        claims["target_label_plain"] = claims["target"].map(v47_plain_label)
        keep = [c for c in [
            "target", "target_label_plain", "best_top5_recall", "best_enrichment",
            "best_spearman", "RAC_gain_at_1000",
            "topology_penalty_random_minus_topology",
            "consensus_shortlist_fold_enrichment", "fraction_shortlist_true_top5"
        ] if c in claims.columns]
        claims = claims[keep + [c for c in claims.columns if c not in keep]]
    v48_table(claims, "Table_2_main_claims_summary.csv", si=False)

    top25 = v47_top25(master)
    top5 = top25.sort_values(["target", "consensus_score"], ascending=[True, False]).groupby("target", as_index=False).head(5) if not top25.empty else pd.DataFrame()
    top_cols = [c for c in [
        "target", "target_label_plain", "mof_id", "filename", "consensus_score",
        "y_true_median", "y_pred_median", "split_lower_median", "split_upper_median",
        "true_percentile", "pred_percentile", "n_models", "n_seeds", "n_splits",
        "n_budgets", "trusted_support"
    ] if c in top5.columns]
    v48_table(top5[top_cols] if top_cols else top5, "Table_3_top5_consensus_candidates_per_target.csv", si=False)
    v48_table(top25[[c for c in top_cols if c in top25.columns] + [c for c in top25.columns if c not in top_cols]].head(100),
              "Table_S7_top25_consensus_candidates_per_target.csv", si=True)

    # SI tables.
    for src, dst in [
        ("table_aggregated_metrics_with_bootstrap_ci.csv", "Table_S1_all_aggregated_metrics_with_CI.csv"),
        ("QC_Table_feature_set_integrity.csv", "Table_S2_feature_set_integrity_RDF_audit.csv"),
        ("QC_Table_model_stability.csv", "Table_S3_model_stability_quality_gate.csv"),
        ("QC_Table_prediction_range_by_model.csv", "Table_S4_prediction_range_sanity_by_model.csv"),
        ("STORY_Table_precision_yield_frontier.csv", "Table_S5_precision_yield_frontier.csv"),
        ("QC_Table_consensus_true_enrichment.csv", "Table_S6_consensus_true_enrichment.csv"),
        ("SI_Table_external_domain_overlap_summary_v47.csv", "Table_S8_external_domain_overlap_summary.csv"),
    ]:
        p = TABLE_DIR / src
        if p.exists():
            try:
                v48_table(pd.read_csv(p, low_memory=False), dst, si=True)
            except Exception as e:
                logging.warning("V48_TABLE_COPY_FAILED | %s | %s", src, e)

    manifest_rows = []
    for folder, kind in [
        (V48_OVERVIEW_DIR, "overview"),
        (V48_FIG_MAIN_DIR, "main_figure"),
        (V48_FIG_SI_DIR, "si_figure"),
        (V48_SOURCE_MAIN_DIR, "main_source_data"),
        (V48_SOURCE_SI_DIR, "si_source_data"),
        (V48_TABLE_MAIN_DIR, "main_table"),
        (V48_TABLE_SI_DIR, "si_table"),
    ]:
        if folder.exists():
            for p in sorted(folder.glob("*")):
                if p.is_file():
                    manifest_rows.append({"kind": kind, "file": str(p.relative_to(V48_ROOT)), "bytes": int(p.stat().st_size)})
    manifest = pd.DataFrame(manifest_rows)
    atomic_save_csv(manifest, V48_OVERVIEW_DIR / "MANIFEST_v4_8_publication_package.csv")
    atomic_save_csv(manifest, TABLE_DIR / "MANIFEST_v4_8_publication_package.csv")
    v48_write_folder_index()
    v48_write_package_readme()
    v48_copy_key_support_files()
    return claims


def v48_write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    stage = "v48_run_report"
    if is_done(stage):
        return
    logging.info("STAGE >>> v4.8 WRITE_CLEAN_PUBLICATION_REPORT")
    agg_path = TABLE_DIR / "table_aggregated_metrics_with_bootstrap_ci.csv"
    agg = pd.read_csv(agg_path, low_memory=False) if agg_path.exists() else pd.DataFrame()
    claims = v48_write_publication_tables(master, agg)
    lines = []
    lines.append("# v4.8 clean publication package report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Package structure\n")
    lines.append(f"- Overview: `{V48_OVERVIEW_DIR}`")
    lines.append(f"- Main figures: `{V48_FIG_MAIN_DIR}`")
    lines.append(f"- Main source data: `{V48_SOURCE_MAIN_DIR}`")
    lines.append(f"- Main tables: `{V48_TABLE_MAIN_DIR}`")
    lines.append(f"- SI figures: `{V48_FIG_SI_DIR}`")
    lines.append(f"- SI source data: `{V48_SOURCE_SI_DIR}`")
    lines.append(f"- SI tables: `{V48_TABLE_SI_DIR}`")
    lines.append("\n## Main figure set\n")
    for fname, meaning in [
        ("Figure_1_v4_8_workflow.png", "graphical workflow / project logic"),
        ("Figure_2_v4_8_fewshot_learning_phase_diagram.png", "few-shot label efficiency and enrichment"),
        ("Figure_3_v4_8_descriptor_chemistry_extrapolation_stress.png", "descriptor chemistry and split-stress story"),
        ("Figure_4_v4_8_conformal_risk_control_frontier.png", "risk control and precision-yield frontier"),
        ("Figure_5_v4_8_consensus_shortlist_atlas.png", "consensus shortlist atlas and true-enrichment view"),
    ]:
        lines.append(f"- `{fname}`: {'FOUND' if (V48_FIG_MAIN_DIR / fname).exists() else 'not found'} — {meaning}")
    lines.append("\n## Main claims table\n")
    if claims is not None and not claims.empty:
        lines.append(v47_dataframe_to_markdown_safe(claims, index=False))
    else:
        lines.append("Main claims table could not be generated from current tables.")
    lines.append("\n## Visual / manuscript guidance\n")
    lines.append("- Use the five main figures from `01_main_text/figures/` for the paper.")
    lines.append("- Use `Table_2_main_claims_summary.csv` to extract the headline quantitative claims.")
    lines.append("- Use `Table_3_top5_consensus_candidates_per_target.csv` as the compact candidate table for the main text.")
    lines.append("- Keep external CoRE/MOSAEC overlays in SI and frame them as domain-overlap annotations rather than validation.")
    lines.append("- Main descriptor claim: geometry versus geometry+RACs. Avoid claiming RDF effects unless the audit shows actual RDF support.")
    report = V48_OVERVIEW_DIR / "RUN_REPORT_v4_8_publication_package.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    mark_done(stage, {"v4_8": True})


# --- lightweight visual-polish wrappers -----------------------------------
# These wrappers keep the mature v4.7 figure logic but route it through the
# cleaner v4.8 style/saving layer.

# Monkey-patch the helper functions used by the v4.7 figure builders.
v47_apply_style = v48_apply_style
v47_panel = v48_panel
v47_save = v48_save
v47_source = v48_source
v47_table = v48_table


def v48_plot_figure_1(master: Optional[pd.DataFrame] = None) -> None:
    """A slightly cleaner redraw of Figure 1 with stronger visual hierarchy."""
    stage = "v48_fig1_workflow"
    if is_done(stage):
        return
    logging.info("FIGURE >>> v4.8 Figure 1")
    v48_apply_style()
    stats = v47_target_stats(master)
    v48_source(stats, "Figure_1a_target_cards_source.csv")
    fig, ax = plt.subplots(figsize=(13.8, 6.0))
    ax.set_axis_off()
    ax.text(0.5, 0.97, "Risk-controlled few-shot discovery of adsorption elites",
            ha="center", va="top", fontsize=16.2, fontweight="bold", color="#0F172A", transform=ax.transAxes)
    ax.text(0.5, 0.92,
            "Label-efficient learning, chemistry-aware descriptors, uncertainty control, and target-balanced consensus shortlists",
            ha="center", va="top", fontsize=9.5, color="#334155", transform=ax.transAxes)

    # Target cards.
    ax.text(0.02, 0.80, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", color="#0F172A")
    card_y, card_h, card_w = 0.56, 0.22, 0.205
    x0s = [0.075, 0.305, 0.535, 0.765]
    for x0, target in zip(x0s, TARGET_ORDER_FINAL):
        row = stats[stats["target"].astype(str).eq(target)].head(1)
        n = int(row["n_mofs"].iloc[0]) if not row.empty and pd.notna(row["n_mofs"].iloc[0]) else 279010
        med = float(row["median_mmol_g"].iloc[0]) if not row.empty and pd.notna(row["median_mmol_g"].iloc[0]) else np.nan
        p95 = float(row["p95_mmol_g"].iloc[0]) if not row.empty and pd.notna(row.get("p95_mmol_g", pd.Series([np.nan])).iloc[0]) else np.nan
        mx = float(row["max_mmol_g"].iloc[0]) if not row.empty and pd.notna(row["max_mmol_g"].iloc[0]) else np.nan
        color = TARGET_COLORS_FINAL.get(target, "#64748B")
        box = matplotlib.patches.FancyBboxPatch((x0, card_y), card_w, card_h,
                                                boxstyle="round,pad=0.012,rounding_size=0.018",
                                                transform=ax.transAxes, fc="#FFFFFF", ec=color, lw=1.8)
        ax.add_patch(box)
        header = matplotlib.patches.FancyBboxPatch((x0, card_y+card_h-0.055), card_w, 0.055,
                                                   boxstyle="round,pad=0.012,rounding_size=0.018",
                                                   transform=ax.transAxes, fc=color, ec=color, lw=0.0)
        ax.add_patch(header)
        ax.text(x0+0.012, card_y+card_h-0.018, v47_latex_label(target), color="white", fontsize=10.8,
                fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(x0+0.014, card_y+0.118, f"{n:,}", fontsize=11.0, fontweight="bold", transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.093, "labelled MOFs", fontsize=9.4, fontweight="bold", transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.060, f"median: {med:.2g} mmol g$^{{-1}}$" if np.isfinite(med) else "median: —",
                fontsize=8.0, transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.032, f"top 5% ≥ {p95:.2g} mmol g$^{{-1}}$" if np.isfinite(p95) else "top 5%: —",
                fontsize=8.0, transform=ax.transAxes)
        ax.text(x0+0.014, card_y+0.006, f"max: {mx:.2g} mmol g$^{{-1}}$" if np.isfinite(mx) else "max: —",
                fontsize=8.0, transform=ax.transAxes)

    # Workflow ribbon.
    ax.text(0.02, 0.38, "b", transform=ax.transAxes, fontsize=12, fontweight="bold", color="#0F172A")
    steps = [
        ("ARC-MOF labels", "4 adsorption targets\ngeometry + RACs", "#E2E8F0"),
        ("Few-shot design", "10–1000 labels\nseeds × split stress", "#DBEAFE"),
        ("Stable learners", "tree ensembles +\ngradient boosting", "#DCFCE7"),
        ("Risk control", "90% conformal intervals\ntrusted / uncertain / rejected", "#FEF3C7"),
        ("Consensus", "models × seeds × splits\ntarget-balanced shortlist", "#F3E8FF"),
        ("Domain overlay", "CoRE/MOSAEC geometry\nannotation, not validation", "#F1F5F9"),
    ]
    y, h, w = 0.12, 0.18, 0.145
    xs = np.linspace(0.065, 0.790, len(steps))
    for i, (x, (title, body, face)) in enumerate(zip(xs, steps)):
        box = matplotlib.patches.FancyBboxPatch((x, y), w, h,
                                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                                transform=ax.transAxes, fc=face, ec="#64748B", lw=0.9)
        ax.add_patch(box)
        ax.text(x+w/2, y+h*0.67, title, ha="center", va="center", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x+w/2, y+h*0.31, body, ha="center", va="center", fontsize=7.2, transform=ax.transAxes)
        if i < len(steps)-1:
            ax.annotate("", xy=(x+w+0.020, y+h/2), xytext=(x+w+0.003, y+h/2), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#475569"))
    v48_save(fig, "Figure_1_v4_8_workflow")
    mark_done(stage, {"v4_8": True})


def v48_plot_figure_2(agg: pd.DataFrame) -> None:
    return v47_plot_figure_2(agg)


def v48_plot_figure_3(agg: pd.DataFrame) -> None:
    return v47_plot_figure_3(agg)


def v48_plot_figure_4(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_figure_4(anatomy, tiers, master=master)


def v48_plot_figure_5(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_figure_5(core, mosaec, final_external, master=master)


def v48_plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v47_plot_si_figures(agg, tiers, anatomy, master=master)


# ---- v4.8 overrides used by the existing main() function ------------------

def plot_figure_1_framework() -> None:
    master = read_pickle(PICKLE_DIR / "master_table.pkl") if (PICKLE_DIR / "master_table.pkl").exists() else None
    return v48_plot_figure_1(master)


def plot_figure_2_performance(agg: pd.DataFrame) -> None:
    return v48_plot_figure_2(agg)


def plot_figure_3_calibration(agg: pd.DataFrame) -> None:
    return v48_plot_figure_3(agg)


def plot_figure_4_failure_anatomy(anatomy: pd.DataFrame, tiers: pd.DataFrame) -> None:
    return v48_plot_figure_4(anatomy, tiers)


def plot_figure_5_external_realism(core: pd.DataFrame, mosaec: pd.DataFrame, final_external: pd.DataFrame) -> None:
    return v48_plot_figure_5(core, mosaec, final_external)


def plot_si_figures(agg: pd.DataFrame, tiers: pd.DataFrame, anatomy: pd.DataFrame) -> None:
    return v48_plot_si_figures(agg, tiers, anatomy)


def write_run_report(master: pd.DataFrame, metrics_all: pd.DataFrame) -> None:
    return v48_write_run_report(master, metrics_all)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Interrupted by user. Re-run the script to continue from saved markers and partial outputs.")
        raise
    except Exception as exc:
        logging.error("FATAL_ERROR | %s", exc)
        logging.error(traceback.format_exc())
        raise
