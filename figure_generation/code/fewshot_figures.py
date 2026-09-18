from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from figure_style import (
    FULL_WIDTH_IN, TARGETS, TARGET_LABEL, TARGET_PLAIN, TARGET_COLOR, TARGET_MARKER,
    BUDGETS, DARK, MID, GRID, LIGHT, configure_matplotlib, panel_label, clean_axis,
    save_figure,
)


def _read(root: Path, rel: str) -> pd.DataFrame:
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing required source file: {p}")
    return pd.read_csv(p, low_memory=False)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "target" in out.columns:
        out["_order"] = out["target"].astype(str).map({t: i for i, t in enumerate(TARGETS)})
        out = out.sort_values("_order").drop(columns="_order")
    return out


def _target_legend_handles():
    return [
        Line2D([0], [0], color=TARGET_COLOR[t], marker=TARGET_MARKER[t], lw=1.7,
               markerfacecolor=TARGET_COLOR[t], markeredgecolor="white", markeredgewidth=0.4,
               label=TARGET_LABEL[t])
        for t in TARGETS
    ]


def figure1(source_root: Path, output_root: Path, dpi: int = 200):
    """Render Figure 1 from the saved benchmark workflow and target summaries."""
    stats = _ordered(_read(source_root, "source_data/main/Figure_1a_target_cards_source.csv"))
    _workflow_provenance = _read(source_root, "source_data/main/Figure_1b_workflow_steps.csv")

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.78))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.84, 1.16], hspace=0.08)

    # ------------------------- Panel A -------------------------
    axa = fig.add_subplot(gs[0])
    axa.set_axis_off()
    panel_label(axa, "A", x=-0.025, y=1.02)
    axa.text(0.02, 0.99, "From adsorption condition to screening evidence", transform=axa.transAxes,
             fontsize=10.1, fontweight="semibold", va="top")

    cards = [
        (0.045, 0.49, 0.255, 0.31, "Adsorption conditions",
         "interaction-sensitive\n→ pore-filling regimes", "#EEF4F8", "#4F7188"),
        (0.3725, 0.49, 0.255, 0.31, "Information needed",
         "geometry narrows the search\nchemistry refines the order", "#EEF8F5", "#3F8E82"),
        (0.700, 0.49, 0.255, 0.31, "Where support weakens",
         "unfamiliar topology\nlimited cross-run support", "#F5EFF9", "#7B5B9B"),
    ]
    for x, y, w, h, title, body, fc, ec in cards:
        axa.add_patch(FancyBboxPatch((x, y), w, h, transform=axa.transAxes,
                                     boxstyle="round,pad=0.010,rounding_size=0.018",
                                     facecolor=fc, edgecolor=ec, linewidth=1.05))
        axa.text(x+w/2, y+h*0.67, title, transform=axa.transAxes,
                 ha="center", va="center", fontsize=8.4, fontweight="bold")
        axa.text(x+w/2, y+h*0.30, body, transform=axa.transAxes,
                 ha="center", va="center", fontsize=7.15, color="#3E4851", linespacing=1.12)

    for x1, x2 in [(0.300, 0.3725), (0.6275, 0.700)]:
        axa.add_patch(FancyArrowPatch((x1, 0.645), (x2, 0.645), transform=axa.transAxes,
                                     arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#5F6972"))

    axa.add_patch(FancyBboxPatch((0.085, 0.205), 0.83, 0.145, transform=axa.transAxes,
                                 boxstyle="round,pad=0.009,rounding_size=0.014",
                                 facecolor="#FFF7E8", edgecolor="#C79645", linewidth=0.9))
    axa.text(0.50, 0.278,
             "Useful screening requires ranking signal, transfer behavior, and empirical calibration",
             transform=axa.transAxes, ha="center", va="center", fontsize=7.75, fontweight="semibold")
    axa.annotate("", xy=(0.84, 0.105), xytext=(0.16, 0.105), xycoords=axa.transAxes,
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#5F6972"))
    axa.text(0.50, 0.055, "more model-fitting evidence", transform=axa.transAxes,
             ha="center", va="center", fontsize=7.05, color=MID)

    # ------------------------- Panel B -------------------------
    axb = fig.add_subplot(gs[1])
    axb.set_axis_off()
    panel_label(axb, "B", x=-0.025, y=1.02)
    axb.text(0.02, 0.99, "Retrospective benchmark design and label roles", transform=axb.transAxes,
             fontsize=10.1, fontweight="semibold", va="top")

    # Target summary cards.
    x0s = [0.10, 0.31, 0.52, 0.73]
    fills = ["#EEF3F5", "#ECF7F5", "#FBF4E8", "#F4EFF7"]
    for x, fc, (_, r) in zip(x0s, fills, stats.iterrows()):
        t = str(r["target"]); c = TARGET_COLOR[t]
        axb.add_patch(FancyBboxPatch((x, 0.77), 0.17, 0.13, transform=axb.transAxes,
                                     boxstyle="round,pad=0.007,rounding_size=0.013",
                                     facecolor=fc, edgecolor=c, linewidth=1.0))
        axb.text(x+0.085, 0.842, TARGET_LABEL[t], transform=axb.transAxes,
                 ha="center", va="center", fontsize=7.45, fontweight="bold", color=c)
        axb.text(x+0.085, 0.797, f"n = {int(r['n_mofs']):,}", transform=axb.transAxes,
                 ha="center", va="center", fontsize=6.65, color=MID)

    def box(x, y, w, h, title, body, face, edge, title_fs=7.05, body_fs=5.75,
            title_rel=0.79, body_rel=0.40, body_linesp=1.10):
        axb.add_patch(FancyBboxPatch((x, y), w, h, transform=axb.transAxes,
                                     boxstyle="round,pad=0.009,rounding_size=0.013",
                                     facecolor=face, edgecolor=edge, linewidth=0.95, zorder=2))
        axb.text(x+w/2, y+h*title_rel, title, transform=axb.transAxes,
                 fontsize=title_fs, fontweight="bold", ha="center", va="center")
        axb.text(x+w/2, y+h*body_rel, body, transform=axb.transAxes,
                 fontsize=body_fs, color="#3E4851", ha="center", va="center", linespacing=body_linesp)

    # Panel B layout.
    # Reference, calibration, evidence, and consensus all share the same midline.
    ref_x, ref_y, ref_w, ref_h = 0.055, 0.1925, 0.190, 0.400
    mid_x, mid_w, mid_h = 0.325, 0.165, 0.135
    cal_center = 0.3925
    delta = 0.175
    model_y = cal_center + delta - mid_h/2      # 0.5000
    calib_y = cal_center - mid_h/2              # 0.3250
    held_y  = cal_center - delta - mid_h/2      # 0.1500
    ev_x, ev_y, ev_w, ev_h = 0.590, 0.2075, 0.205, 0.370
    con_x, con_y, con_w, con_h = 0.825, 0.2075, 0.145, 0.370

    box(ref_x, ref_y, ref_w, ref_h, "Reference library",
        "279,010 / target\n23 geometry + 148 RACs\nreference uptake retained\nfor retrospective evaluation",
        "#F5F7F8", "#8C98A1", title_fs=6.70, body_fs=5.05,
        title_rel=0.80, body_rel=0.39, body_linesp=1.05)
    box(mid_x, model_y, mid_w, mid_h, "Model fitting",
        "B = 10–1000\nstratified sampling", "#EEF4FA", "#6C8FA7",
        title_fs=6.05, body_fs=4.65, title_rel=0.82, body_rel=0.22, body_linesp=1.02)
    box(mid_x, calib_y, mid_w, mid_h, "Calibration",
        "separate reservoir\none q per run", "#EDF7F3", "#6DA295",
        title_fs=6.05, body_fs=4.65, title_rel=0.82, body_rel=0.22, body_linesp=1.02)
    box(mid_x, held_y, mid_w, mid_h, "Held-out test",
        "recall • transfer\nempirical coverage", "#F4EFF7", "#8D74A0",
        title_fs=6.05, body_fs=4.65, title_rel=0.82, body_rel=0.22, body_linesp=1.02)
    box(ev_x, ev_y, ev_w, ev_h, "Evidence summaries",
        "learning curves\nrepresentation gain\ntransfer stress\ninterval calibration",
        "#FFF7E8", "#C49B50", title_fs=6.55, body_fs=4.95,
        title_rel=0.80, body_rel=0.38, body_linesp=1.04)
    box(con_x, con_y, con_w, con_h, "Consensus atlas",
        "cross-run support\ntop-25 displayed\ntop-250 aggregate",
        "#F8F0E7", "#B78B5A", title_fs=6.30, body_fs=4.80,
        title_rel=0.80, body_rel=0.38, body_linesp=1.04)

    ref_right, mid_left, mid_right = ref_x + ref_w, mid_x, mid_x + mid_w
    ev_left, ev_right, con_left = ev_x, ev_x + ev_w, con_x
    ref_center_y = ref_y + ref_h/2               # 0.3925
    mid_centers = [model_y + mid_h/2, calib_y + mid_h/2, held_y + mid_h/2]
    ev_targets = [0.515, 0.3925, 0.270]
    for yy, ey in zip(mid_centers, ev_targets):
        axb.add_patch(FancyArrowPatch((ref_right, ref_center_y), (mid_left, yy), transform=axb.transAxes,
                                     arrowstyle="-|>", mutation_scale=8.5, lw=0.85, color="#7D8790"))
        axb.add_patch(FancyArrowPatch((mid_right, yy), (ev_left, ey), transform=axb.transAxes,
                                     arrowstyle="-|>", mutation_scale=8.5, lw=0.85, color="#7D8790"))
    axb.add_patch(FancyArrowPatch((ev_right, cal_center), (con_left, cal_center), transform=axb.transAxes,
                                 arrowstyle="-|>", mutation_scale=8.5, lw=0.85, color="#7D8790"))


    return save_figure(fig, output_root, "Figure_1_redesign_hybrid_screening_evidence", dpi)


def figure2(source_root: Path, output_root: Path, dpi: int = 200):
    recall = _read(source_root, "source_data/main/Figure_2a_recall_learning_curves.csv")
    norm = _read(source_root, "source_data/main/Figure_2c_label_efficiency.csv")
    thresh = _read(source_root, "source_data/main/Figure_2d_budget_needed_for_recall_fraction.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.20), gridspec_kw={"wspace": 0.30})
    axa, axb = axes
    for t in TARGETS:
        sub = recall[recall["target"].astype(str).eq(t)].sort_values("budget")
        x = sub["budget"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        lo = sub["lo"].to_numpy(float)
        hi = sub["hi"].to_numpy(float)
        axa.fill_between(x, lo, hi, color=TARGET_COLOR[t], alpha=0.12, linewidth=0, zorder=1)
        axa.plot(x, y, color=TARGET_COLOR[t], marker=TARGET_MARKER[t], markeredgecolor="white",
                 markeredgewidth=0.45, label=TARGET_LABEL[t], zorder=3)
    axa.axhline(0.05, color="#7E878F", ls=(0, (3, 2)), lw=0.9)
    axa.text(11, 0.055, "random top-5% baseline", fontsize=7.2, color=MID, va="bottom")
    axa.set_xscale("log")
    axa.set_xticks(BUDGETS); axa.set_xticklabels([str(b) for b in BUDGETS])
    axa.set_xlabel("Model-fitting budget, B")
    axa.set_ylabel("Top-5% recall")
    axa.set_ylim(0.0, 0.92)
    axa.set_title("Absolute top-5% recall")
    clean_axis(axa, "both")
    panel_label(axa, "A")

    for t in TARGETS:
        sub = norm[norm["target"].astype(str).eq(t)].sort_values("budget")
        axb.plot(sub["budget"], sub["label_efficiency"], color=TARGET_COLOR[t],
                 marker=TARGET_MARKER[t], markeredgecolor="white", markeredgewidth=0.45)
    for level, ls in [(0.5, (0, (1, 2))), (0.75, (0, (3, 2))), (0.90, (0, (5, 2)))]:
        axb.axhline(level, color="#A4ACB3", ls=ls, lw=0.8, zorder=0)
        axb.text(10.5, level+0.012, f"{int(level*100)}%", fontsize=6.9, color=MID, va="bottom")
    axb.set_xscale("log")
    axb.set_xticks(BUDGETS); axb.set_xticklabels([str(b) for b in BUDGETS])
    axb.set_xlabel("Model-fitting budget, B")
    axb.set_ylabel(r"Recall / recall at $B=1000$")
    axb.set_ylim(0.18, 1.05)
    axb.set_title("Relative recall vs B = 1000")
    clean_axis(axb, "both")
    panel_label(axb, "B")

    th90 = _ordered(thresh[np.isclose(thresh["threshold_fraction_of_final_recall"], 0.90)])
    budgets90 = th90["budget_needed"].astype(int).tolist()
    if len(set(budgets90)) == 1:
        txt = f"All targets: ≥90% of B=1000 endpoint by B={budgets90[0]}"
    else:
        txt = "90% threshold: " + "; ".join(
            f"{TARGET_PLAIN[str(t)]}: B={int(b)}" for t, b in zip(th90["target"], budgets90)
        )
    axb.text(0.97, 0.045, txt, transform=axb.transAxes, fontsize=6.85, color=MID,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#D4D9DD", lw=0.6))

    fig.legend(handles=_target_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=2, frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.095, right=0.985)
    return save_figure(fig, output_root, "Figure_2_redesign_learning_curves", dpi)


def _annotated_heatmap(ax, df: pd.DataFrame, center_zero: bool, cbar_label: str, fmt: str, cmap: str, show_cbar: bool = True):
    vals = df.to_numpy(float)
    if center_zero:
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=34, ha="right")
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = vals[i, j]
            rgba = im.cmap(im.norm(v))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.55,
                    color="white" if lum < 0.48 else DARK)
    for s in ax.spines.values(): s.set_visible(False)
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label(cbar_label, fontsize=7.5)
        cb.ax.tick_params(labelsize=7.0)
    return im

def figure3(source_root: Path, output_root: Path, dpi: int = 200):
    gain = _read(source_root, "source_data/main/Figure_3_descriptor_gain_by_budget.csv")
    split = _read(source_root, "source_data/main/Figure_3c_split_stress_recall_heatmap.csv").set_index("target_label")
    topo = _ordered(_read(source_root, "source_data/main/Figure_3d_topology_penalty.csv"))

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.92))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.04, 1.0], width_ratios=[1.0, 1.03], hspace=0.43, wspace=0.42)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    for t in TARGETS:
        sub = gain[gain["target"].astype(str).eq(t)].sort_values("budget")
        axa.plot(sub["budget"], sub["delta_recall_racs_minus_geometry"], color=TARGET_COLOR[t],
                 marker=TARGET_MARKER[t], markeredgecolor="white", markeredgewidth=0.45, label=TARGET_LABEL[t])
    axa.axhline(0, color="#7F8991", lw=0.8)
    axa.set_xscale("log"); axa.set_xticks(BUDGETS); axa.set_xticklabels([str(b) for b in BUDGETS])
    axa.set_xlabel("Model-fitting budget, B")
    axa.set_ylabel("Δ top-5% recall\n(geometry + RACs − geometry)")
    axa.set_title("RAC contribution across fitting budget")
    clean_axis(axa, "both")
    axa.legend(handles=_target_legend_handles(), ncol=2, loc="upper left", frameon=False,
               bbox_to_anchor=(0.0, 0.98), handlelength=1.5)

    _annotated_heatmap(axb, split, center_zero=False, cbar_label="", fmt=".2f", cmap="YlGnBu", show_cbar=False)
    axb.set_title("Grouped-split recall at B = 1000")

    y = np.arange(len(topo))
    vals = topo["topology_penalty_random_minus_topology"].to_numpy(float)
    for i, (t, v) in enumerate(zip(topo["target"], vals)):
        axc.hlines(i, 0, v, color="#C9CED3", lw=1.4, zorder=1)
        axc.scatter(v, i, s=50, color=TARGET_COLOR[str(t)], marker=TARGET_MARKER[str(t)],
                    edgecolor="white", linewidth=0.5, zorder=3)
        axc.text(v+0.006, i, f"{v:.3f}", va="center", fontsize=7.2)
    axc.set_yticks(y); axc.set_yticklabels([TARGET_LABEL[str(t)] for t in topo["target"]])
    axc.invert_yaxis(); axc.tick_params(axis='y', labelsize=7.8)
    axc.set_xlim(0, max(vals)*1.32)
    axc.set_xlabel("Random - topology-split recall")
    axc.set_title("Topology-split penalty")
    clean_axis(axc, "x")

    fig.subplots_adjust(top=0.95, bottom=0.11, left=0.15, right=0.985)
    fig.text(0.040, 0.975, "A", fontsize=11.6, fontweight="bold", ha="left", va="top")
    fig.text(0.040, 0.482, "B", fontsize=11.6, fontweight="bold", ha="left", va="top")
    fig.text(0.530, 0.482, "C", fontsize=11.6, fontweight="bold", ha="left", va="top")
    return save_figure(fig, output_root, "Figure_3_redesign_descriptor_transfer", dpi)


def figure4(source_root: Path, output_root: Path, dpi: int = 200):
    cov = _read(source_root, "source_data/main/Figure_4a_coverage_error_heatmap.csv").set_index("target_label")
    width = _read(source_root, "source_data/main/Figure_4b_normalized_interval_width.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.48),
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.34})
    axa, axb = axes
    _annotated_heatmap(axa, cov, center_zero=True, cbar_label="", fmt="+.3f", cmap="RdBu_r", show_cbar=False)
    axa.set_title("Coverage deviation from 0.90 at B = 1000")

    for t in TARGETS:
        sub = width[width["target"].astype(str).eq(t)].sort_values("budget")
        axb.plot(sub["budget"], sub["normalized_width"], color=TARGET_COLOR[t], marker=TARGET_MARKER[t],
                 markeredgecolor="white", markeredgewidth=0.45, label=TARGET_LABEL[t])
    axb.set_xscale("log"); axb.set_xticks(BUDGETS); axb.set_xticklabels([str(b) for b in BUDGETS])
    axb.set_xlabel("Model-fitting budget, B")
    axb.set_ylabel(r"Mean interval width / target $\sigma$")
    axb.set_title("Normalized interval width")
    clean_axis(axb, "both")
    axb.legend(handles=_target_legend_handles(), ncol=1, loc="upper right", frameon=False,
               handlelength=1.5, borderaxespad=0.2)

    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.13, right=0.985)
    fig.text(0.060, 0.965, "A", fontsize=11.6, fontweight="bold", ha="left", va="top")
    fig.text(0.545, 0.965, "B", fontsize=11.6, fontweight="bold", ha="left", va="top")
    return save_figure(fig, output_root, "Figure_4_redesign_empirical_calibration", dpi)


def figure5(source_root: Path, output_root: Path, dpi: int = 200):
    top = _read(source_root, "source_data/main/Figure_5_top25_consensus_candidates_with_percentiles.csv")
    top25 = _ordered(_read(source_root, "source_data/main/Figure_5_true_enrichment.csv"))
    top250 = _ordered(_read(source_root, "tables/si/Table_S6_consensus_true_enrichment.csv"))

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.03, 1.0], width_ratios=[1.0, 1.0], hspace=0.49, wspace=0.40)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    # A: show every supplied top-25 candidate percentile. No median bars or random summaries.
    rng = np.random.default_rng(314159)
    axa.axhspan(0.95, 0.99, color="#F3F5F6", zorder=0)
    axa.axhspan(0.99, 1.005, color="#E8EDF0", zorder=0)
    for i, t in enumerate(TARGETS):
        vals = pd.to_numeric(top.loc[top["target"].astype(str).eq(t), "true_percentile"], errors="coerce").dropna().to_numpy()
        jitter = rng.normal(0, 0.050, len(vals))
        axa.scatter(np.full(len(vals), i) + jitter, vals, s=28, color=TARGET_COLOR[t], alpha=0.82,
                    edgecolor="white", linewidth=0.40, marker=TARGET_MARKER[t], zorder=3)
    axa.axhline(0.95, color="#8E989F", ls=(0, (4, 2)), lw=0.85, zorder=1)
    axa.axhline(0.99, color="#AAB2B8", ls=(0, (1.5, 2)), lw=0.85, zorder=1)
    axa.set_xlim(-0.25, 3.34)
    label_box = dict(facecolor="#F7F7F7", edgecolor="none", pad=0.20)
    axa.text(3.12, 0.962, "top 5%", fontsize=6.8, color=MID, va="center", ha="left", bbox=label_box)
    axa.text(3.12, 0.998, "top 1%", fontsize=6.8, color=MID, va="center", ha="left", bbox=label_box)
    axa.set_xticks(range(4)); axa.set_xticklabels([TARGET_LABEL[t] for t in TARGETS])
    axa.set_ylim(0.40, 1.008)
    axa.set_ylabel("Reference uptake percentile")
    axa.set_title("Reference-tail position of the 25 displayed candidates per target")
    clean_axis(axa, "y")
    axa.legend(handles=_target_legend_handles(), ncol=4, loc="center",
               bbox_to_anchor=(0.50, 0.69), frameon=False, handlelength=1.4, columnspacing=1.2)

    # B: candidate geometry only; marker area encodes stored support rows.
    support_min = float(pd.to_numeric(top["n_support_rows"], errors="coerce").min())
    support_max = float(pd.to_numeric(top["n_support_rows"], errors="coerce").max())
    def size_from_support(v):
        if support_max <= support_min:
            return 34
        return 22 + 48 * (float(v)-support_min)/(support_max-support_min)
    for t in TARGETS:
        sub = top[top["target"].astype(str).eq(t)]
        sizes = [size_from_support(v) for v in sub["n_support_rows"]]
        axb.scatter(sub["Di"], sub["Density"], s=sizes, color=TARGET_COLOR[t], alpha=0.80,
                    edgecolor="white", linewidth=0.45, marker=TARGET_MARKER[t], label=TARGET_LABEL[t])
    axb.set_xlabel(r"Largest cavity diameter, $D_i$ (Å)")
    axb.set_ylabel(r"Density (g cm$^{-3}$)")
    axb.set_title("Candidate pore-size / density regime", fontsize=8.9)
    clean_axis(axb, "both")
    axb.legend(handles=_target_legend_handles(), loc="upper right", ncol=1, frameon=False, handlelength=1.4)

    # C: explicitly separate top-25 and top-250 cohort summaries.
    y = np.arange(4)
    a = top25.set_index("target").reindex(TARGETS)["fraction_true_top5"].to_numpy(float)
    b = top250.set_index("target").reindex(TARGETS)["fraction_true_top5pct"].to_numpy(float)
    for i, t in enumerate(TARGETS):
        axc.plot([b[i], a[i]], [i, i], color="#C6CCD1", lw=1.4, zorder=1)
        axc.scatter(b[i], i, s=42, facecolor="white", edgecolor=TARGET_COLOR[t], linewidth=1.2, marker="o", zorder=3)
        axc.scatter(a[i], i, s=48, facecolor=TARGET_COLOR[t], edgecolor="white", linewidth=0.45, marker="o", zorder=4)
    axc.set_yticks(y); axc.set_yticklabels([TARGET_LABEL[t] for t in TARGETS]); axc.invert_yaxis()
    axc.set_xlim(0.58, 1.02)
    axc.set_xlabel("Fraction in reference top 5%")
    axc.set_title("Top-5% membership by cohort", fontsize=8.9)
    clean_axis(axc, "x")
    axc.legend(handles=[
        Line2D([0],[0], marker="o", color="none", markerfacecolor=DARK, markeredgecolor="white", label="top-25 displayed"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="white", markeredgecolor=DARK, label="top-250 aggregate"),
    ], loc="lower right", frameon=False, fontsize=7.2)

    fig.subplots_adjust(top=0.91, bottom=0.17, left=0.105, right=0.985)
    fig.text(0.060, 0.968, "A", fontsize=11.6, fontweight="bold", ha="left", va="top")
    fig.text(0.060, 0.488, "B", fontsize=11.6, fontweight="bold", ha="left", va="top")
    fig.text(0.545, 0.488, "C", fontsize=11.6, fontweight="bold", ha="left", va="top")
    # Requested note is kept only where it is needed: directly below Panel B.
    fig.text(0.105, 0.072, "Marker area proportional to stored support rows (support count is not a probability).",
             fontsize=6.55, color=MID, ha="left")
    return save_figure(fig, output_root, "Figure_5_redesign_retrospective_candidates", dpi)


def figure_s1(source_root: Path, output_root: Path, dpi: int = 200):
    hist = _read(source_root, "source_data/SI/Figure_S1_histogram_bin_counts_recovered_from_v48_svg.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH_IN, 5.55), gridspec_kw={"hspace": 0.56, "wspace": 0.38})
    for idx, (ax, t) in enumerate(zip(axes.ravel(), TARGETS)):
        sub = hist[hist["target"].astype(str).eq(t)].sort_values("bin_index")
        widths = (sub["bin_right"] - sub["bin_left"]).to_numpy(float)
        ax.bar(sub["bin_left"], sub["count"], width=widths, align="edge", color=TARGET_COLOR[t],
               alpha=0.88, linewidth=0)
        p95 = float(sub["p95_threshold"].iloc[0])
        ax.axvline(p95, color=DARK, ls=(0, (3, 2)), lw=0.95)
        ax.set_yscale("log")
        ax.set_xlabel(r"Uptake (mmol g$^{-1}$)", fontsize=9.7)
        ax.set_ylabel("Count (log scale)", fontsize=9.7)
        ax.set_title(TARGET_LABEL[t], fontsize=10.4, pad=6)
        ax.tick_params(axis="both", labelsize=9.0)
        clean_axis(ax, None)
        ax.text(-0.14, 1.08, chr(ord('A') + idx), transform=ax.transAxes,
                fontsize=12.2, fontweight="bold", ha="left", va="top", clip_on=False)
        ax.text(0.97, 0.93, f"p95 = {p95:.3g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.2, color=MID)
    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.10, right=0.985)
    return save_figure(fig, output_root, "Figure_S1_redesign_target_distributions", dpi)


def figure_s2(source_root: Path, output_root: Path, dpi: int = 200):
    mp = _read(source_root, "source_data/SI/Figure_S2a_model_pass_fraction.csv").copy()
    rc = _read(source_root, "source_data/SI/Figure_S2b_gate_outcome_rejection_reason.csv").copy()
    q = _read(source_root, "tables/si/Table_S3_model_stability_quality_gate.csv")
    other = int(len(q) - int(rc["experiments"].sum()))
    if other < 0:
        raise ValueError("Displayed QC combinations exceed the full experiment table.")
    rc2 = rc.copy()
    if other:
        rc2 = pd.concat([rc2, pd.DataFrame({
            "gate_outcome_or_rejection_reason": ["other combinations (not shown in original top-10)"],
            "experiments": [other],
        })], ignore_index=True)

    label_map = {
        "not final-candidate model;coverage outside gate;RMSE too large": "Ineligible + coverage + RMSE",
        "low top-5 recall": "Low top-5 recall",
        "not final-candidate model;low/NaN Spearman;RMSE too large": "Ineligible + rank + RMSE",
        "not final-candidate model;low top-5 recall": "Ineligible + low recall",
        "coverage outside gate": "Coverage outside gate",
        "not final-candidate model;nan_rmse": "Ineligible + NaN RMSE",
        "low/NaN Spearman": "Low/NaN Spearman",
        "not final-candidate model;RMSE too large": "Ineligible + RMSE",
        "not final-candidate model": "Ineligible model family",
        "PASS": "PASS",
        "other combinations (not shown in original top-10)": "Other combinations",
    }
    rc2["short_label"] = rc2["gate_outcome_or_rejection_reason"].map(label_map)
    model_name = {
        "rf": "Random Forest",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "hgb": "HGB",
        "ridge": "Ridge",
        "mlp": "MLP",
        "gpr": "GPR",
    }
    mp["display_model"] = mp["model"].map(model_name).fillna(mp["model"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 4.10),
                                   gridspec_kw={"width_ratios": [0.80, 1.30], "wspace": 0.74})

    mp = mp.sort_values("experiment_passes_quality_gate", ascending=True)
    y = np.arange(len(mp))
    cols = ["#C8CED3" if v == 0 else "#607D8B" for v in mp["experiment_passes_quality_gate"]]
    axa.barh(y, mp["experiment_passes_quality_gate"], color=cols, height=0.60)
    axa.set_yticks(y); axa.set_yticklabels(mp["display_model"], fontsize=8.2)
    axa.set_xlim(0, 1.08)
    axa.set_xlabel("Experiment pass fraction", fontsize=9.2)
    axa.set_title("Model-family pass fraction", fontsize=9.8)
    clean_axis(axa, "x")
    for yi, v in zip(y, mp["experiment_passes_quality_gate"]):
        axa.text(v+0.016, yi, f"{v:.3f}", va="center", fontsize=7.2)

    rc2 = rc2.sort_values("experiments", ascending=True)
    y2 = np.arange(len(rc2))
    colors = ["#344A5E" if lab == "PASS" else "#A8B1B8" for lab in rc2["short_label"]]
    axb.barh(y2, rc2["experiments"], color=colors, height=0.60)
    axb.set_yticks(y2); axb.set_yticklabels(rc2["short_label"], fontsize=7.45)
    axb.set_xscale("log")
    axb.set_xlim(30, 12000)
    axb.set_xlabel("Experiments (log scale)", fontsize=9.2)
    axb.set_title("Mutually exclusive gate outcomes", fontsize=9.8)
    clean_axis(axb, "x")
    for yi, v in zip(y2, rc2["experiments"]):
        axb.text(v*1.07, yi, f"{int(v):,}", va="center", fontsize=7.0)

    fig.subplots_adjust(top=0.89, bottom=0.14, left=0.18, right=0.97)
    fig.text(0.060, 0.965, "A", fontsize=11.8, fontweight="bold", ha="left", va="top")
    fig.text(0.555, 0.965, "B", fontsize=11.8, fontweight="bold", ha="left", va="top")
    return save_figure(fig, output_root, "Figure_S2_redesign_quality_gate_audit", dpi)


def figure_s3(source_root: Path, output_root: Path, dpi: int = 200):
    """Redraw supplied S3 summary values only; provenance status is documented outside the artwork."""
    df = _ordered(_read(source_root, "source_data/SI/Figure_S_external_domain_overlap_source.csv"))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.38),
                                   gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.54})

    y = np.arange(len(df))
    off = 0.12
    a = df["core_geometry_overlap_fraction"].to_numpy(float)
    b = df["mosaec_geometry_overlap_fraction"].to_numpy(float)
    for yi in y:
        axa.plot([0, max(a[yi], b[yi])], [yi, yi], color="#E1E5E8", lw=1.0, zorder=0)
    axa.scatter(a, y-off, s=42, marker="o", color="#4E95C3", edgecolor="white", linewidth=0.5,
                label="Resource A geometry", zorder=3)
    axa.scatter(b, y+off, s=42, marker="s", color="#4CBF91", edgecolor="white", linewidth=0.5,
                label="Resource B geometry", zorder=3)
    axa.set_yticks(y); axa.set_yticklabels([TARGET_LABEL[str(t)] for t in df["target"]], fontsize=8.8)
    axa.invert_yaxis()
    axa.set_xlim(-0.02, 1.02)
    axa.set_xlabel("Geometry-domain overlap fraction", fontsize=9.5)
    axa.set_title("Stored external-domain overlap", fontsize=10.1)
    axa.tick_params(axis="x", labelsize=8.6)
    clean_axis(axa, "x")
    axa.legend(frameon=False, loc="lower right", fontsize=7.6)
    for vals, yy in [(a, y-off), (b, y+off)]:
        for xv, yv in zip(vals, yy):
            axa.text(min(xv+0.025, 0.96), yv, f"{xv:.3f}", va="center", fontsize=7.3, color=MID)

    score = df["mean_external_annotation_score"].to_numpy(float)
    for yi, (t, s) in enumerate(zip(df["target"], score)):
        c = TARGET_COLOR[str(t)]
        axb.plot([0, s], [yi, yi], color="#C8CED3", lw=1.3, zorder=1)
        axb.scatter([s], [yi], s=56, color=c, marker=TARGET_MARKER[str(t)],
                    edgecolor="white", linewidth=0.55, zorder=3)
        axb.text(s+0.045, yi, f"{s:.3f}", va="center", fontsize=7.45)
    axb.set_yticks(y); axb.set_yticklabels([TARGET_LABEL[str(t)] for t in df["target"]], fontsize=8.8)
    axb.invert_yaxis()
    axb.set_xlim(0, 1.78)
    axb.set_xlabel("Mean external-annotation score", fontsize=9.5)
    axb.set_title("Stored annotation score", fontsize=10.1)
    axb.tick_params(axis="x", labelsize=8.6)
    clean_axis(axb, "x")

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.18, right=0.985)
    fig.text(0.060, 0.955, "A", fontsize=12.0, fontweight="bold", ha="left", va="top")
    fig.text(0.555, 0.955, "B", fontsize=12.0, fontweight="bold", ha="left", va="top")
    return save_figure(fig, output_root, "Figure_S3_redesign_external_domain_overlap_HOLD", dpi)



FIGURE_FUNCTIONS = {
    "1": figure1,
    "2": figure2,
    "3": figure3,
    "4": figure4,
    "5": figure5,
    "S1": figure_s1,
    "S2": figure_s2,
    "S3": figure_s3,
}
