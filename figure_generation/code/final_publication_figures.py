#!/usr/bin/env python3
"""Cosmetic-only regeneration of the finalized original figure layouts.

Scientific content and panel membership are preserved from the saved
publication renderer. Only presentation is changed: Arial typography, readable
sizes, restrained palettes, cleaner spacing, and improved legend placement.

Figures regenerated: 2, 3, 4, 5, S1, S2, S3. Figure 1 is intentionally excluded.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = ROOT / "data" / "source_data" / "main"
SI_SOURCE = ROOT / "data" / "source_data" / "SI"
MAIN_FIG = ROOT / "outputs" / "main"
SI_FIG = ROOT / "outputs" / "si"

TARGETS = [
    "CO2_0p015bar_298K_mmolg",
    "CO2_0p150bar_298K_mmolg",
    "CH4_5p8bar_298K_mmolg",
    "CH4_65bar_298K_mmolg",
]
TARGET_LABEL = {
    TARGETS[0]: r"CO$_2$ 0.015 bar",
    TARGETS[1]: r"CO$_2$ 0.150 bar",
    TARGETS[2]: r"CH$_4$ 5.8 bar",
    TARGETS[3]: r"CH$_4$ 65 bar",
}
# Target palette.
TARGET_COLOR = {
    TARGETS[0]: "#274C5E",   # deep blue-teal
    TARGETS[1]: "#2F9D8F",   # teal
    TARGETS[2]: "#D08A24",   # amber
    TARGETS[3]: "#7E5AA6",   # purple
}
BUDGETS = [10, 20, 50, 100, 200, 500, 1000]
DARK = "#24313C"
MID = "#5F6F7B"
LIGHTGRID = "#D7DEE3"
NEUTRAL = "#788791"
PAPER_BG = "#FFFFFF"

# Final main-figure layouts.
DEFAULT_MAIN_GRID = {
    "2": "2x3",
    "3": "3x2",
    "4": "2x3",
    "5": "3x2",
}
DEFAULT_MAIN_SIZE = {
    "2x3": (11.9, 7.0),
    "3x2": (9.5, 11.7),
}
# Figure-specific canvas sizes.
FIGSIZE_BY_FIGURE = {
    "2": (12.6, 7.5),
    "3": (10.2, 12.3),
    "4": (13.8, 8.0),
    "5": (10.0, 12.2),
}

CURRENT_LAYOUTS: dict[str, str] = {}


def _pick_font(strict: bool = True) -> str:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except Exception:
        if strict:
            raise SystemExit(
                "Arial was not found by Matplotlib. On Windows, Arial should normally be installed.\n"
                "Close/reopen Anaconda Prompt and run CHECK_WINDOWS_ENV.bat.\n"
                "For layout-only testing, add --allow-font-fallback-for-preview."
            )
        try:
            font_manager.findfont("Arimo", fallback_to_default=False)
            return "Arimo"
        except Exception:
            return "DejaVu Sans"


def style(strict_arial: bool = True) -> str:
    font = _pick_font(strict_arial)
    plt.rcParams.update({
        "font.family": font,
        "font.size": 11.1,
        "axes.titlesize": 12.3,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.2,
        "axes.labelcolor": DARK,
        "axes.edgecolor": "#8C98A2",
        "axes.linewidth": 0.78,
        "xtick.labelsize": 10.1,
        "ytick.labelsize": 10.1,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "legend.fontsize": 9.1,
        "legend.frameon": False,
        "figure.facecolor": PAPER_BG,
        "axes.facecolor": PAPER_BG,
        "savefig.facecolor": PAPER_BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "lines.linewidth": 1.9,
        "lines.markersize": 5.0,
        "mathtext.fontset": "stixsans",
    })
    return font


def panel(ax, letter: str, x: float = -0.11, y: float = 1.155) -> None:
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=13.8,
            fontweight="bold", color=DARK, ha="right", va="top")


def polish_ax(ax, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8C98A2")
    ax.spines["bottom"].set_color("#8C98A2")
    if grid:
        ax.grid(True, axis=grid, color=LIGHTGRID, linewidth=0.65, alpha=0.82, zorder=0)
    ax.set_axisbelow(True)


def tune_fonts(ax, title=12.6, label=11.4, tick=10.2):
    ax.title.set_fontsize(title)
    ax.title.set_fontweight("semibold")
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)
    ax.tick_params(axis="both", labelsize=tick)


def save(fig, stem: str, si: bool, formats: list[str], dpi: int) -> None:
    outdir = SI_FIG if si else MAIN_FIG
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if fmt.lower() == "png":
            kwargs["dpi"] = dpi
        fig.savefig(outdir / f"{stem}.{fmt}", **kwargs)
    plt.close(fig)


def read_main(name: str) -> pd.DataFrame:
    return pd.read_csv(MAIN_SOURCE / name, low_memory=False)


def read_si(name: str) -> pd.DataFrame:
    return pd.read_csv(SI_SOURCE / name, low_memory=False)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns:
        return df
    work = df.copy()
    work["_order"] = work["target"].astype(str).map({t: i for i, t in enumerate(TARGETS)})
    return work.sort_values("_order").drop(columns="_order")


def heatmap(ax, data: pd.DataFrame, title: str, cbar_label: str, center_zero=False,
            fmt=".2f", cmap=None, cbar_pad: float = 0.025):
    vals = data.to_numpy(dtype=float)
    if cmap is None:
        cmap = "RdBu_r" if center_zero else "YlGnBu"
    if center_zero:
        vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=33, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8.0,
                        color="white" if lum < 0.48 else DARK)
    ax.set_title(title, pad=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=cbar_pad)
    cb.set_label(cbar_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8.0, width=0.6, length=2.8)
    cb.outline.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def target_curves(ax, df, value="mean", lo="lo", hi="hi", ylabel="", title="",
                  baseline=None):
    handles, labels = [], []
    for t in TARGETS:
        sub = df[df["target"].astype(str).eq(t)].sort_values("budget")
        if sub.empty:
            continue
        x = pd.to_numeric(sub["budget"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub[value], errors="coerce").to_numpy(float)
        line, = ax.plot(
            x, y, marker="o", color=TARGET_COLOR[t], label=TARGET_LABEL[t], zorder=3,
            markeredgecolor="white", markeredgewidth=0.55
        )
        if lo in sub and hi in sub:
            yl = pd.to_numeric(sub[lo], errors="coerce").to_numpy(float)
            yh = pd.to_numeric(sub[hi], errors="coerce").to_numpy(float)
            ax.fill_between(x, yl, yh, color=TARGET_COLOR[t], alpha=0.10, linewidth=0, zorder=1)
        handles.append(line)
        labels.append(TARGET_LABEL[t])
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Label budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    if baseline is not None:
        ax.axhline(baseline, color="#8A959D", linestyle=(0, (3, 2)), linewidth=1.0, zorder=0)
    polish_ax(ax, "both")
    return handles, labels


def nicer_model_name(x: str) -> str:
    mapping = {
        "rf": "Random Forest",
        "extra trees": "Extra Trees",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "hgb": "HGB",
        "ridge": "Ridge",
        "mlp": "MLP",
        "gpr": "GPR",
    }
    key = str(x).strip().lower()
    return mapping.get(key, str(x).replace("_", " ").title())


def wrap_label(x: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(x), width=width, break_long_words=False))


def main_panel_grid(fig_key: str):
    layout = CURRENT_LAYOUTS.get(fig_key, DEFAULT_MAIN_GRID.get(fig_key, "3x2"))
    figsize = FIGSIZE_BY_FIGURE.get(fig_key, DEFAULT_MAIN_SIZE[layout])
    if layout == "2x3":
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(3, 2, figsize=figsize)
    return fig, axes.ravel(), layout


def add_shared_target_legend(fig, handles, labels, anchor=(0.5, 0.955), ncol=4, fontsize=9.0):
    fig.legend(handles, labels, ncol=ncol, loc="upper center", bbox_to_anchor=anchor,
               columnspacing=1.3, handlelength=2.0, frameon=False, prop={"size": fontsize})


def figure2(formats, dpi):
    recall = read_main("Figure_2a_recall_learning_curves.csv")
    enrich = read_main("Figure_2b_enrichment_learning_curves.csv")
    eff = read_main("Figure_2c_label_efficiency.csv")
    th = read_main("Figure_2d_budget_needed_for_recall_fraction.csv")
    model = read_main("Figure_2e_model_comparison_1000_labels.csv")
    final_en = ordered(read_main("Figure_2f_final_enrichment_summary.csv"))
    fig, axes, layout = main_panel_grid("2")
    a, b, c, d, e, f = axes

    handles, labels = target_curves(a, recall, ylabel="Top-5% recall",
                                    title="Elite recovery from few labels", baseline=0.05)
    tune_fonts(a, title=13.0, label=11.9, tick=10.9)
    panel(a, "A")

    target_curves(b, enrich, ylabel="Enrichment over random",
                  title="Screening enrichment", baseline=1.0)
    tune_fonts(b, title=13.0, label=11.9, tick=10.9)
    panel(b, "B")

    target_curves(c, eff.rename(columns={"label_efficiency": "mean"}),
                  ylabel="Fraction of 1000-label recall",
                  title="Label-efficiency index", baseline=0.75)
    c.set_ylim(0, 1.05)
    tune_fonts(c, title=13.0, label=11.9, tick=10.9)
    panel(c, "C")

    ymap = {t: i for i, t in enumerate(TARGETS)}
    markers = {0.5: "o", 0.75: "s", 0.9: "D"}
    marker_colors = {0.5: "#3C7FB1", 0.75: "#F58518", 0.9: "#4CAF50"}
    for threshold, label in [(0.5, "50%"), (0.75, "75%"), (0.9, "90%")]:
        sub = th[np.isclose(th["threshold_fraction_of_final_recall"], threshold)]
        d.scatter(sub["budget_needed"], [ymap[t] for t in sub["target"]], s=72,
                  marker=markers[threshold], label=label, alpha=0.95,
                  color=marker_colors[threshold])
    d.set_xscale("log")
    d.set_xticks(BUDGETS)
    d.set_xticklabels([str(x) for x in BUDGETS])
    d.set_yticks(range(4))
    d.set_yticklabels([TARGET_LABEL[t] for t in TARGETS])
    d.set_xlabel("Budget needed")
    d.set_title("How many labels are enough?", pad=10)
    polish_ax(d, "x")
    tune_fonts(d, title=13.0, label=11.9, tick=10.8)
    d.legend(title="of final recall", loc="upper center", bbox_to_anchor=(0.5, -0.24),
             ncol=3, title_fontsize=9.4, fontsize=9.2, handletextpad=0.6, columnspacing=1.2)
    panel(d, "D")

    model = model.sort_values("recall_mean")
    e.bar(range(len(model)), model["recall_mean"], color="#718593", width=0.46, zorder=2)
    e.set_xticks(range(len(model)))
    e.set_xticklabels([nicer_model_name(x) for x in model["model"]], rotation=45, ha="right")
    for i, v in enumerate(model["recall_mean"]):
        e.text(i, v + 0.014, f"{v:.2f}", ha="center", va="bottom", fontsize=9.4, color=DARK)
    e.set_ylim(0, max(0.68, model["recall_mean"].max() * 1.18))
    e.set_ylabel("Mean top-5% recall")
    e.set_title("Stable models at 1000 labels", pad=10)
    polish_ax(e, "y")
    tune_fonts(e, title=13.0, label=11.9, tick=10.7)
    panel(e, "E")

    final_en = final_en.sort_values("mean")
    f.barh(range(len(final_en)), final_en["mean"],
           color=[TARGET_COLOR[t] for t in final_en["target"]], height=0.46, zorder=2)
    f.set_yticks(range(len(final_en)))
    f.set_yticklabels([TARGET_LABEL[t] for t in final_en["target"]])
    f.axvline(1, color="#8A959D", ls="--", lw=1.0)
    for i, v in enumerate(final_en["mean"]):
        f.text(v + 0.18, i, f"{v:.1f}×", va="center", fontsize=8.1, color=DARK)
    f.set_xlabel("Enrichment at 1000 labels")
    f.set_title("Final enrichment summary", pad=10)
    polish_ax(f, "x")
    tune_fonts(f, title=13.0, label=11.9, tick=10.7)
    panel(f, "F")

    add_shared_target_legend(fig, handles, labels, anchor=(0.50, 0.955), ncol=4, fontsize=10.2)
    if layout == "2x3":
        fig.subplots_adjust(left=0.075, right=0.988, top=0.875, bottom=0.13, wspace=0.46, hspace=0.70)
    else:
        fig.subplots_adjust(left=0.09, right=0.985, top=0.87, bottom=0.07, wspace=0.40, hspace=0.64)
    save(fig, "Figure_2", False, formats, dpi)


def figure3(formats, dpi):
    hm = read_main("Figure_3a_rac_gain_heatmap.csv").set_index("target_label")
    bars = ordered(read_main("Figure_3b_rac_gain_bars_1000_labels.csv"))
    split = read_main("Figure_3c_split_stress_recall_heatmap.csv").set_index("target_label")
    topo = ordered(read_main("Figure_3d_topology_penalty.csv"))
    vuln = read_main("Figure_3e_extrapolation_vulnerability.csv").set_index("target_label")
    mech = read_main("Figure_3f_mechanistic_interpretation.csv")
    fig, axes, layout = main_panel_grid("3")
    a, b, c, d, e, f = axes

    heatmap(a, hm, "RAC descriptor gain across budgets", r"Δ top-5% recall", True, "+.02f", cmap="BrBG")
    tune_fonts(a, title=12.8, label=11.7, tick=10.7)
    panel(a, "A")

    bars = bars.sort_values("delta_recall_racs_minus_geometry")
    b.barh(range(len(bars)), bars["delta_recall_racs_minus_geometry"],
           color=[TARGET_COLOR[t] for t in bars["target"]], height=0.44)
    b.set_yticks(range(len(bars)))
    b.set_yticklabels([TARGET_LABEL[t] for t in bars["target"]])
    b.axvline(0, color="#8A959D", lw=1.0)
    for i, v in enumerate(bars["delta_recall_racs_minus_geometry"]):
        b.text(v + 0.004, i, f"{v:+.2f}", va="center", fontsize=8.0)
    b.set_xlabel("Δ recall: RACs - geometry")
    b.set_title("Chemistry gain at 1000 labels", pad=10)
    polish_ax(b, "x")
    tune_fonts(b, title=12.8, label=11.7, tick=10.6)
    panel(b, "B")

    heatmap(c, split, "Split-stress map at 1000 labels", "Top-5% recall", False, ".2f", "viridis")
    tune_fonts(c, title=12.8, label=11.7, tick=10.7)
    panel(c, "C")

    topo = topo.sort_values("topology_penalty_random_minus_topology")
    d.barh(range(len(topo)), topo["topology_penalty_random_minus_topology"],
           color=[TARGET_COLOR[t] for t in topo["target"]], height=0.44)
    d.set_yticks(range(len(topo)))
    d.set_yticklabels([TARGET_LABEL[t] for t in topo["target"]])
    for i, v in enumerate(topo["topology_penalty_random_minus_topology"]):
        d.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=8.0)
    d.set_xlabel("Random recall - topology recall")
    d.set_title("Topology extrapolation penalty", pad=10)
    polish_ax(d, "x")
    tune_fonts(d, title=12.8, label=11.7, tick=10.6)
    panel(d, "D")

    heatmap(e, vuln, "Extrapolation vulnerability", "Relative recall loss vs random", True, "+.02f", cmap="PuOr_r")
    tune_fonts(e, title=12.8, label=11.7, tick=10.7)
    panel(e, "E")

    f.set_axis_off()
    panel(f, "F", x=-0.08, y=1.13)
    f.set_title("Mechanistic interpretation", pad=10, fontsize=12.8, fontweight="semibold")
    cards = [
        ("Geometry saturates", "Pore size, surface area, and density encode the first-order adsorption envelope.", "#EFF4F6"),
        ("RAC chemistry", "Local chemical descriptors improve elite recovery, particularly at intermediate and high-label budgets.", "#EEF8F6"),
        ("Topology shift", "Topology-grouped splitting remains the hardest extrapolation stress test for all four targets.", "#FCF5E8"),
        ("Risk control", "Conformal intervals turn ranking uncertainty into an explicit abstention/retention decision.", "#F5F0F8"),
    ]
    y0 = 0.935
    h = 0.195
    gap = 0.028
    for title, body, fc in cards:
        y1 = y0 - h
        box = FancyBboxPatch((0.045, y1), 0.91, h, transform=f.transAxes,
                             boxstyle="round,pad=0.012,rounding_size=0.014",
                             facecolor=fc, edgecolor="#C8D1D8", linewidth=0.78)
        f.add_patch(box)
        f.text(0.085, y0 - 0.038, title, transform=f.transAxes, fontsize=10.1,
               fontweight="bold", color=DARK, va="top")
        f.text(0.085, y0 - 0.090, wrap_label(body, 44), transform=f.transAxes,
               fontsize=8.9, color="#41505A", va="top")
        y0 = y1 - gap

    fig.suptitle("Descriptor chemistry improves screening, but topology extrapolation remains limiting",
                 fontsize=12.6, fontweight="bold", color=DARK, y=0.992)
    if layout == "3x2":
        fig.subplots_adjust(left=0.078, right=0.987, top=0.895, bottom=0.055, wspace=0.52, hspace=0.68)
    else:
        fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.08, wspace=0.50, hspace=0.64)
    save(fig, "Figure_3", False, formats, dpi)


def figure4(formats, dpi):
    cov = read_main("Figure_4a_coverage_error_heatmap.csv").set_index("target_label")
    width = read_main("Figure_4b_normalized_interval_width.csv")
    front = read_main("Figure_4c_precision_yield_frontier.csv")
    p10 = ordered(read_main("Figure_4d_precision_at_10pct_retained.csv"))
    tiers = read_main("Figure_4e_prefinal_tier_composition.csv").sort_values("budget")
    methods = read_main("Figure_4f_coverage_method_comparison.csv")
    fig, axes, layout = main_panel_grid("4")
    a, b, c, d, e, f = axes

    heatmap(a, cov, "Coverage error relative to 90%", "coverage - 0.90", True, "+.02f", cmap="BrBG")
    tune_fonts(a, title=12.8, label=11.7, tick=10.6)
    panel(a, "A")

    handles, labels = target_curves(
        b, width.rename(columns={"normalized_width": "mean"}),
        ylabel="Interval width / target std",
        title="Uncertainty narrows with labels"
    )
    tune_fonts(b, title=12.8, label=11.7, tick=10.6)
    panel(b, "B")

    for t in TARGETS:
        sub = front[front["target"].astype(str).eq(t)].sort_values("retained_fraction")
        c.plot(sub["retained_fraction"], sub["precision_true_top5"], marker="o",
               color=TARGET_COLOR[t], label=TARGET_LABEL[t], markeredgecolor="white",
               markeredgewidth=0.55)
    c.set_xlim(0, 0.52)
    c.set_ylim(0, 1.04)
    c.set_xlabel("Retained fraction")
    c.set_ylabel("True top-5% fraction")
    c.set_title("Precision-yield frontier", pad=10)
    polish_ax(c, "both")
    tune_fonts(c, title=12.8, label=11.7, tick=10.6)
    panel(c, "C")

    d.bar(range(len(p10)), p10["precision_true_top5"],
          color=[TARGET_COLOR[t] for t in p10["target"]], width=0.46)
    d.set_xticks(range(len(p10)))
    d.set_xticklabels([TARGET_LABEL[t] for t in p10["target"]], rotation=25, ha="right")
    for i, v in enumerate(p10["precision_true_top5"]):
        d.text(i, min(v + 0.03, 1.03), f"{v:.2f}", ha="center", fontsize=8.0)
    d.set_ylim(0, 1.08)
    d.set_ylabel("True top-5% fraction")
    d.set_title("Precision at ~10% retained", pad=10)
    polish_ax(d, "y")
    tune_fonts(d, title=12.8, label=11.7, tick=10.6)
    panel(d, "D")

    x = np.arange(len(tiers))
    bottom = np.zeros(len(tiers))
    cols = [
        ("trusted_fraction", "trusted", "#2A9D8F"),
        ("uncertain_fraction", "uncertain", "#D8A441"),
        ("rejected_fraction", "rejected", "#9CA6AD"),
    ]
    for col, label, color in cols:
        vals = tiers[col].to_numpy(float)
        e.bar(x, vals, bottom=bottom, label=label, color=color, width=0.60)
        bottom += vals
    e.set_xticks(x)
    e.set_xticklabels(tiers["budget"].astype(int).astype(str))
    e.set_ylim(0, 1)
    e.set_xlabel("Label budget")
    e.set_ylabel("Fraction of prediction rows")
    e.set_title("Pre-final uncertainty-tier composition", pad=10)
    polish_ax(e, "y")
    tune_fonts(e, title=12.8, label=11.7, tick=10.5)
    e.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=9.3)
    panel(e, "E")

    method_colors = ["#4C78A8", "#2A9D8F", "#7A5195"]
    f.bar(range(len(methods)), methods["coverage"], color=method_colors[:len(methods)], width=0.46)
    f.set_xticks(range(len(methods)))
    f.set_xticklabels(methods["method"])
    f.axhline(0.90, color="#7F8991", ls="--", lw=1.0)
    for i, v in enumerate(methods["coverage"]):
        f.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=8.0)
    f.set_ylim(0.86, 0.94)
    f.set_ylabel("Empirical coverage")
    f.set_title("Coverage method comparison", pad=10)
    polish_ax(f, "y")
    tune_fonts(f, title=12.8, label=11.7, tick=10.5)
    panel(f, "F")

    b.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.26),
             fontsize=9.2, handlelength=1.8, columnspacing=1.2)
    c.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.26),
             fontsize=9.2, handlelength=1.8, columnspacing=1.2)

    fig.suptitle("Conformal uncertainty converts few-shot predictions into risk-controlled shortlists",
                 fontsize=12.6, fontweight="bold", color=DARK, y=0.992)
    if layout == "2x3":
        fig.subplots_adjust(left=0.06, right=0.99, top=0.868, bottom=0.15, wspace=0.40, hspace=0.82)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.08, wspace=0.42, hspace=0.60)
    save(fig, "Figure_4", False, formats, dpi)


def figure5(formats, dpi):
    funnel = read_main("Figure_5_candidate_funnel.csv")
    qce = ordered(read_main("Figure_5_true_enrichment.csv"))
    top = read_main("Figure_5_top25_consensus_candidates_with_percentiles.csv")
    fig, axes, layout = main_panel_grid("5")
    a, b, c, d, e, f = axes

    stages = ["Labelled MOFs", "Quality-gated rows", "Consensus pass", "Top-25"]
    mat = funnel.pivot_table(index="target", columns="stage", values="count", aggfunc="first").reindex(TARGETS)[stages]
    pmat = np.log10(mat.replace(0, np.nan))
    im = a.imshow(pmat.to_numpy(), aspect="auto", cmap="Blues")
    a.set_xticks(range(4))
    a.set_xticklabels(["Labelled", "Quality\ngated", "Consensus", "Top-25"])
    a.set_yticks(range(4))
    a.set_yticklabels([TARGET_LABEL[t] for t in TARGETS])
    for i in range(4):
        for j in range(4):
            txt = f"{int(mat.iloc[i, j]):,}"
            a.text(j, i, txt, ha="center", va="center", fontsize=8.6,
                   color=DARK if pmat.iloc[i, j] < 4.5 else "white")
    cb = fig.colorbar(im, ax=a, fraction=0.044, pad=0.030)
    cb.set_label(r"log$_{10}$(count)", fontsize=8.6)
    cb.ax.tick_params(labelsize=8.0, width=0.6, length=2.8)
    cb.outline.set_visible(False)
    a.set_title("Decision funnel to final candidates", pad=10)
    tune_fonts(a, title=12.2, label=11.0, tick=9.8)
    panel(a, "A")

    vals = qce["fold_enrichment_vs_dataset_median"].to_numpy(float)
    b.bar(range(4), vals, color=[TARGET_COLOR[t] for t in qce["target"]], width=0.48)
    b.axhline(1, color="#89959E", ls="--", lw=1.0)
    b.set_xticks(range(4))
    b.set_xticklabels([TARGET_LABEL[t] for t in qce["target"]], rotation=23, ha="right")
    for i, v in enumerate(vals):
        b.text(i, v + 0.18, f"{v:.1f}×", ha="center", fontsize=8.2)
    b.set_ylabel("Shortlist median / dataset median")
    b.set_title("True enrichment of consensus shortlist", pad=10)
    polish_ax(b, "y")
    tune_fonts(b, title=12.2, label=11.0, tick=9.8)
    panel(b, "B")

    x = np.arange(4)
    bw = 0.16
    elite_colors = ["#537C8C", "#D08A24", "#7E5AA6"]
    for k, (col, label) in enumerate([
        ("fraction_true_top1", "top 1%"),
        ("fraction_true_top5", "top 5%"),
        ("fraction_true_top10", "top 10%"),
    ]):
        c.bar(x + (k - 1) * bw, qce[col], width=bw, label=label, color=elite_colors[k])
    c.set_xticks(x)
    c.set_xticklabels([TARGET_LABEL[t] for t in qce["target"]], rotation=23, ha="right")
    c.set_ylim(0, 1.05)
    c.set_ylabel("Fraction of top-25 shortlist")
    c.set_title("How often candidates are true elites", pad=10)
    c.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=9.3)
    polish_ax(c, "y")
    tune_fonts(c, title=12.2, label=11.0, tick=9.8)
    panel(c, "C")

    data = [pd.to_numeric(top.loc[top["target"].astype(str).eq(t), "true_percentile"], errors="coerce").dropna().to_numpy() for t in TARGETS]
    bp = d.boxplot(
        data,
        tick_labels=[TARGET_LABEL[t] for t in TARGETS],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#C98324", "linewidth": 1.1},
        widths=0.46,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="#F5F7F8", edgecolor="#66747E", linewidth=0.82)
    rng = np.random.default_rng(42)
    for i, (t, v) in enumerate(zip(TARGETS, data), 1):
        d.scatter(i + rng.normal(0, 0.035, len(v)), v, s=18, alpha=0.60,
                  color=TARGET_COLOR[t], edgecolors="none")
    d.set_ylim(0.68, 1.01)
    d.tick_params(axis="x", rotation=23)
    d.set_ylabel("True uptake percentile")
    d.set_title("Final candidates occupy the adsorption tail", pad=10)
    polish_ax(d, "y")
    tune_fonts(d, title=12.2, label=11.0, tick=9.8)
    panel(d, "D")

    support = ["n_models", "n_seeds", "n_splits", "n_budgets"]
    sdata = [pd.to_numeric(top[x], errors="coerce").dropna().to_numpy() for x in support]
    bp = e.boxplot(
        sdata,
        tick_labels=[x.replace("n_", "") for x in support],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#C98324", "linewidth": 1.1},
        widths=0.46,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="#F5F7F8", edgecolor="#66747E", linewidth=0.82)
    rng = np.random.default_rng(43)
    for i, v in enumerate(sdata, 1):
        e.scatter(i + rng.normal(0, 0.035, len(v)), v, s=10, alpha=0.30,
                  color=NEUTRAL, edgecolors="none")
    e.text(0.03, 0.95, f"median trusted support = {pd.to_numeric(top['trusted_support'], errors='coerce').median():.0f} rows",
           transform=e.transAxes, fontsize=7.6, va="top",
           bbox=dict(boxstyle="round,pad=.24", fc="white", ec="#CCD4D9", lw=.6))
    e.set_ylabel("Distinct support count")
    e.set_title("Consensus stability across runs", pad=10)
    polish_ax(e, "y")
    tune_fonts(e, title=12.8, label=11.7, tick=10.5)
    panel(e, "E")

    for t in TARGETS:
        sub = top[top["target"].astype(str).eq(t)]
        f.scatter(sub["Di"], sub["Density"], s=34, alpha=0.80, color=TARGET_COLOR[t],
                  label=TARGET_LABEL[t], edgecolor="white", linewidth=0.40)
    f.set_xlabel(r"PLD proxy, $D_i$ / Å")
    f.set_ylabel(r"Density / g cm$^{-3}$")
    f.set_title("Geometry regime of final candidates", pad=10)
    polish_ax(f, "both")
    tune_fonts(f, title=12.8, label=11.7, tick=10.5)
    f.legend(loc="upper right", fontsize=9.3)
    panel(f, "F")

    fig.suptitle("Target-balanced consensus shortlists are truly enriched in adsorption elites",
                 fontsize=12.6, fontweight="bold", color=DARK, y=0.992)
    if layout == "3x2":
        fig.subplots_adjust(left=0.082, right=0.985, top=0.914, bottom=0.06, wspace=0.34, hspace=0.60)
    else:
        fig.subplots_adjust(left=0.055, right=0.985, top=0.89, bottom=0.08, wspace=0.42, hspace=0.56)
    save(fig, "Figure_5", False, formats, dpi)


def si1(formats, dpi):
    hist = read_si("Figure_S1_histogram_bin_counts_recovered_from_v48_svg.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    for idx, (ax, t) in enumerate(zip(axes.ravel(), TARGETS)):
        sub = hist[hist["target"].astype(str).eq(t)].sort_values("bin_index")
        width = (sub["bin_right"] - sub["bin_left"]).to_numpy(float)
        ax.bar(sub["bin_left"], sub["count"], width=width, align="edge",
               color=TARGET_COLOR[t], alpha=0.88, linewidth=0)
        p95 = float(sub["p95_threshold"].iloc[0])
        ax.axvline(p95, color=DARK, ls=(0, (3, 2)), lw=1.0, label="top-5% threshold")
        ax.set_yscale("log")
        ax.set_title(TARGET_LABEL[t], pad=8)
        ax.set_xlabel(r"Uptake / mmol g$^{-1}$")
        ax.set_ylabel("Count (log)")
        ax.legend(loc="upper right", fontsize=7.6)
        polish_ax(ax, None)
        panel(ax, chr(ord("A") + idx), x=-0.08, y=1.09)
    fig.suptitle("Figure S1. Full target distributions in ARC-MOF",
                 fontsize=12.0, fontweight="bold", color=DARK, y=0.982)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.08, wspace=0.30, hspace=0.34)
    save(fig, "Figure_S1", True, formats, dpi)


def si2(formats, dpi):
    mp = read_si("Figure_S2a_model_pass_fraction.csv").copy()
    rc = read_si("Figure_S2b_gate_outcome_rejection_reason.csv").copy()

    mp["model"] = mp["model"].map(nicer_model_name)

    label_map = {
        "PASS": "PASS",
        "not final-candidate model": "Not final-candidate model",
        "not final-candidate model;RMSE too large": "Not final-candidate model + RMSE too large",
        "low/NaN Spearman": "Low/NaN Spearman",
        "not final-candidate model;nan_rmse": "Not final-candidate model + NaN RMSE",
        "coverage outside gate": "Coverage outside gate",
        "not final-candidate model;low top-5 recall": "Not final-candidate model + low top-5 recall",
        "not final-candidate model;low/NaN Spearman;RMSE too large": "Not final-candidate model + low/NaN Spearman + RMSE too large",
        "low top-5 recall": "Low top-5 recall",
        "not final-candidate model;coverage outside gate;RMSE too large": "Not final-candidate model + coverage outside gate + RMSE too large",
    }
    rc["nice_label"] = rc["gate_outcome_or_rejection_reason"].map(label_map).fillna(rc["gate_outcome_or_rejection_reason"])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.2), gridspec_kw={"wspace": 1.04, "width_ratios": [0.88, 1.42]})
    a, b = axes
    a.barh(mp["model"], mp["experiment_passes_quality_gate"], color="#6E8799", height=0.50)
    a.set_xlim(0, 1.05)
    a.set_xlabel("Fraction passing final-candidate gate")
    a.set_title("Model-level pass fraction", pad=10)
    polish_ax(a, "x")
    tune_fonts(a, title=12.0, label=10.8, tick=9.6)
    panel(a, "A", x=-0.08, y=1.06)

    wrapped_labels = [wrap_label(lbl, 21) for lbl in rc["nice_label"]]
    ypos = np.arange(len(rc))
    b.barh(ypos, rc["experiments"], color="#9EABB5", height=0.40)
    b.set_yticks(ypos)
    b.set_yticklabels(wrapped_labels)
    b.tick_params(axis="y", pad=6)
    b.set_xlabel("Experiments")
    b.set_title("Gate outcome / rejection reason", pad=10)
    polish_ax(b, "x")
    tune_fonts(b, title=12.0, label=10.8, tick=8.9)
    panel(b, "B", x=-0.08, y=1.06)

    fig.suptitle("Figure S2. Candidate-quality gate audit",
                 fontsize=12.0, fontweight="bold", color=DARK, y=0.992)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.09, wspace=1.04)
    save(fig, "Figure_S2", True, formats, dpi)


def si3(formats, dpi):
    s = ordered(read_si("Figure_S_external_domain_overlap_source.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.95), gridspec_kw={"wspace": 0.38, "width_ratios": [1.06, 0.94]})
    a, b = axes
    x = np.arange(len(s))
    bw = 0.22
    a.bar(x - bw/2, s["core_geometry_overlap_fraction"], width=bw, label="CoRE geometry", color="#4C78A8")
    a.bar(x + bw/2, s["mosaec_geometry_overlap_fraction"], width=bw, label="MOSAEC geometry", color="#2A9D8F")
    a.set_xticks(x)
    a.set_xticklabels([TARGET_LABEL[t] for t in s["target"]], rotation=24, ha="right")
    a.set_ylim(0, 1.05)
    a.set_ylabel("Overlap fraction")
    a.set_title("Geometry-domain overlap", pad=10)
    a.legend(loc="upper center", bbox_to_anchor=(0.52, 1.07), ncol=2, fontsize=8.5)
    polish_ax(a, "y")
    panel(a, "A")

    b.bar(range(len(s)), s["mean_external_annotation_score"],
          color=[TARGET_COLOR[t] for t in s["target"]], width=0.34)
    b.set_xticks(range(len(s)))
    b.set_xticklabels([TARGET_LABEL[t] for t in s["target"]], rotation=24, ha="right")
    b.set_ylabel("Mean annotation score")
    b.set_title("Domain-overlap annotation score", pad=10)
    polish_ax(b, "y")
    tune_fonts(b, title=11.8, label=10.7, tick=9.4)
    panel(b, "B")

    fig.suptitle("External overlays annotate domain overlap, not experimental validation",
                 fontsize=11.4, fontweight="bold", color=DARK, y=0.992)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.20, wspace=0.38)
    save(fig, "Figure_S3", True, formats, dpi)


FIGURE_FUNCTIONS = {"2": figure2, "3": figure3, "4": figure4, "5": figure5, "S1": si1, "S2": si2, "S3": si3}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", nargs="+", default=list(FIGURE_FUNCTIONS), choices=list(FIGURE_FUNCTIONS))
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--allow-font-fallback-for-preview", action="store_true")
    ap.add_argument(
        "--main-grid", choices=["3x2", "2x3", "mixed"], default="mixed",
        help="Use one layout for Figures 2-5, or the finalized mixed layout."
    )
    args = ap.parse_args()

    global CURRENT_LAYOUTS
    if args.main_grid == "mixed":
        CURRENT_LAYOUTS = DEFAULT_MAIN_GRID.copy()
    else:
        CURRENT_LAYOUTS = {key: args.main_grid for key in ["2", "3", "4", "5"]}

    font = style(strict_arial=not args.allow_font_fallback_for_preview)
    for key in args.figures:
        FIGURE_FUNCTIONS[key](args.formats, args.dpi)
    print(f"Done. Font used: {font}")
    print(f"Main outputs: {MAIN_FIG}")
    print(f"SI outputs:   {SI_FIG}")
    print(f"Main layout mode: {args.main_grid}")


if __name__ == "__main__":
    main()
