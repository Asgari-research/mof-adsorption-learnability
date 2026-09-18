#!/usr/bin/env python3
"""Regenerate the publication-facing v4.9 figures and key tables from packaged source data only.

This script is intentionally independent of the expensive upstream few-shot modelling run.
It reads only CSV files contained in this publication package and recreates all main/SI
figures in PNG, PDF, and SVG. It also repairs/regenerates the compact main claims table
from the packaged SI enrichment table.

Usage
-----
python regenerate_publication_figures_and_tables_v4_9.py
python regenerate_publication_figures_and_tables_v4_9.py --formats png pdf svg --dpi 600
python regenerate_publication_figures_and_tables_v4_9.py --main-only
python regenerate_publication_figures_and_tables_v4_9.py --si-only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
MAIN_SOURCE = ROOT / "01_main_text" / "source_data"
MAIN_FIG = ROOT / "01_main_text" / "figures"
MAIN_TABLE = ROOT / "01_main_text" / "tables"
SI_SOURCE = ROOT / "02_supporting_information" / "source_data"
SI_FIG = ROOT / "02_supporting_information" / "figures"
SI_TABLE = ROOT / "02_supporting_information" / "tables"

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
TARGET_PLAIN = {
    TARGETS[0]: "CO2 0.015 bar",
    TARGETS[1]: "CO2 0.150 bar",
    TARGETS[2]: "CH4 5.8 bar",
    TARGETS[3]: "CH4 65 bar",
}
TARGET_COLOR = {
    TARGETS[0]: "#2A6FBB",
    TARGETS[1]: "#16A085",
    TARGETS[2]: "#D35400",
    TARGETS[3]: "#7D3C98",
}
BUDGETS = [10, 20, 50, 100, 200, 500, 1000]
DARK = "#17212B"
MID = "#5B6770"
LIGHTGRID = "#DCE3E8"
NEUTRAL = "#7A8791"
PAPER_BG = "#FFFFFF"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 9.6,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.9,
        "axes.labelcolor": DARK,
        "axes.edgecolor": "#AAB4BC",
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "xtick.color": "#46515A",
        "ytick.color": "#46515A",
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "figure.facecolor": PAPER_BG,
        "axes.facecolor": PAPER_BG,
        "savefig.facecolor": PAPER_BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
    })


def panel(ax, letter: str, x: float = -0.10, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=DARK, ha="right", va="top")


def polish_ax(ax, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=LIGHTGRID, linewidth=0.55, alpha=0.72, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem: str, si: bool, formats: list[str], dpi: int) -> None:
    outdir = SI_FIG if si else MAIN_FIG
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None,
                    bbox_inches="tight", pad_inches=0.06)
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


def target_curves(ax, df, value="mean", lo="lo", hi="hi", ylabel="", title="", baseline=None, legend=False):
    for t in TARGETS:
        sub = df[df["target"].astype(str).eq(t)].sort_values("budget")
        if sub.empty:
            continue
        x = pd.to_numeric(sub["budget"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub[value], errors="coerce").to_numpy(float)
        ax.plot(x, y, marker="o", color=TARGET_COLOR[t], label=TARGET_LABEL[t], zorder=3)
        if lo in sub and hi in sub:
            yl = pd.to_numeric(sub[lo], errors="coerce").to_numpy(float)
            yh = pd.to_numeric(sub[hi], errors="coerce").to_numpy(float)
            ax.fill_between(x, yl, yh, color=TARGET_COLOR[t], alpha=0.11, linewidth=0, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Label budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if baseline is not None:
        ax.axhline(baseline, color="#9AA5AD", linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
    polish_ax(ax, "both")
    if legend:
        ax.legend(loc="best", handlelength=1.6)


def heatmap(ax, data: pd.DataFrame, title: str, cbar_label: str, center_zero=False, fmt=".2f", cmap=None):
    vals = data.to_numpy(dtype=float)
    if cmap is None:
        cmap = "RdBu_r" if center_zero else "YlGnBu"
    if center_zero:
        vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0])); ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                rgba = im.cmap(im.norm(v))
                lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.7,
                        color="white" if lum < 0.48 else DARK)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(cbar_label, fontsize=7.5)
    cb.ax.tick_params(labelsize=6.8)
    for sp in ax.spines.values(): sp.set_visible(False)


def figure1(formats, dpi):
    stats = ordered(read_main("Figure_1a_target_cards_source.csv"))
    steps = read_main("Figure_1b_workflow_steps.csv")
    fig, ax = plt.subplots(figsize=(14.2, 6.0))
    ax.set_axis_off()
    ax.text(0.5, 0.965, "Risk-controlled few-shot discovery of adsorption elites",
            transform=ax.transAxes, ha="center", va="top", fontsize=16.8, fontweight="bold", color=DARK)
    ax.text(0.5, 0.915,
            "Label-efficient learning • chemistry-aware descriptors • uncertainty control • target-balanced consensus",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.6, color=MID)
    ax.text(0.018, 0.80, "a", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=DARK)
    card_y, card_h, card_w = 0.565, 0.215, 0.205
    xs = [0.055, 0.285, 0.515, 0.745]
    for x, (_, r) in zip(xs, stats.iterrows()):
        t = str(r["target"]); c = TARGET_COLOR[t]
        box = FancyBboxPatch((x, card_y), card_w, card_h, transform=ax.transAxes,
                             boxstyle="round,pad=0.012,rounding_size=0.016",
                             facecolor="#FBFCFD", edgecolor=c, linewidth=1.35)
        ax.add_patch(box)
        head = FancyBboxPatch((x, card_y+card_h-0.055), card_w, 0.055, transform=ax.transAxes,
                              boxstyle="round,pad=0.012,rounding_size=0.016",
                              facecolor=c, edgecolor=c, linewidth=0)
        ax.add_patch(head)
        ax.text(x+0.012, card_y+card_h-0.019, TARGET_LABEL[t], transform=ax.transAxes,
                color="white", fontsize=10.5, fontweight="bold", va="center")
        ax.text(x+0.014, card_y+0.116, f"{int(r['n_mofs']):,}", transform=ax.transAxes,
                fontsize=12.0, fontweight="bold", color=DARK)
        ax.text(x+0.014, card_y+0.090, "labelled MOFs", transform=ax.transAxes,
                fontsize=8.8, fontweight="semibold", color=MID)
        ax.text(x+0.014, card_y+0.058, f"median  {r['median_mmol_g']:.3g} mmol g$^{{-1}}$",
                transform=ax.transAxes, fontsize=8.0, color=DARK)
        ax.text(x+0.014, card_y+0.031, f"top 5%  ≥ {r['p95_mmol_g']:.3g} mmol g$^{{-1}}$",
                transform=ax.transAxes, fontsize=8.0, color=DARK)
        ax.text(x+0.014, card_y+0.005, f"maximum  {r['max_mmol_g']:.3g} mmol g$^{{-1}}$",
                transform=ax.transAxes, fontsize=8.0, color=DARK)
    ax.text(0.018, 0.39, "b", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=DARK)
    faces = ["#EEF3F7", "#EAF2FB", "#EAF7F2", "#FFF5DE", "#F4EEFB", "#EFF2F5"]
    y, h, w = 0.115, 0.19, 0.142
    xs = np.linspace(0.055, 0.795, len(steps))
    for i, (x, (_, r)) in enumerate(zip(xs, steps.iterrows())):
        box = FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                             boxstyle="round,pad=0.011,rounding_size=0.015",
                             facecolor=faces[i], edgecolor="#9AA8B2", linewidth=0.85)
        ax.add_patch(box)
        ax.text(x+w/2, y+h*0.68, str(r["title"]), transform=ax.transAxes,
                ha="center", va="center", fontsize=8.4, fontweight="bold", color=DARK)
        desc = str(r["description"])
        ax.text(x+w/2, y+h*0.31, desc, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.1, color="#34414B")
        if i < len(steps)-1:
            ax.annotate("", xy=(x+w+0.020, y+h/2), xytext=(x+w+0.004, y+h/2), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#74828C"))
    save(fig, "Figure_1_v4_9_workflow", False, formats, dpi)


def figure2(formats, dpi):
    recall=read_main("Figure_2a_recall_learning_curves.csv")
    enrich=read_main("Figure_2b_enrichment_learning_curves.csv")
    eff=read_main("Figure_2c_label_efficiency.csv")
    th=read_main("Figure_2d_budget_needed_for_recall_fraction.csv")
    model=read_main("Figure_2e_model_comparison_1000_labels.csv")
    final_en=ordered(read_main("Figure_2f_final_enrichment_summary.csv"))
    fig, axes=plt.subplots(2,3,figsize=(14.0,7.7)); a,b,c,d,e,f=axes.ravel()
    target_curves(a,recall,ylabel="Top-5% recall",title="Elite recovery from few labels",baseline=0.05,legend=True); panel(a,"a")
    target_curves(b,enrich,ylabel="Enrichment over random",title="Screening enrichment",baseline=1.0); panel(b,"b")
    target_curves(c,eff.rename(columns={"label_efficiency":"mean"}),ylabel="Fraction of 1000-label recall",title="Label-efficiency index",baseline=0.75)
    c.set_ylim(0,1.05); panel(c,"c")
    ymap={t:i for i,t in enumerate(TARGETS)}; markers={0.5:"o",0.75:"s",0.9:"D"}
    for threshold,label in [(0.5,"50%"),(0.75,"75%"),(0.9,"90%")]:
        sub=th[np.isclose(th["threshold_fraction_of_final_recall"],threshold)]
        d.scatter(sub["budget_needed"],[ymap[t] for t in sub["target"]],s=42,marker=markers[threshold],label=label,alpha=.92)
    d.set_xscale("log"); d.set_xticks(BUDGETS); d.set_xticklabels([str(x) for x in BUDGETS]); d.set_yticks(range(4)); d.set_yticklabels([TARGET_LABEL[t] for t in TARGETS])
    d.set_xlabel("Budget needed"); d.set_title("How many labels are enough?"); polish_ax(d,"x"); d.legend(title="of final recall",loc="lower right"); panel(d,"d")
    model=model.sort_values("recall_mean")
    e.bar(range(len(model)),model["recall_mean"],color="#77848D",width=.68,zorder=2)
    e.set_xticks(range(len(model))); e.set_xticklabels(model["model"].str.replace("_","\n",regex=False))
    for i,v in enumerate(model["recall_mean"]): e.text(i,v+.012,f"{v:.2f}",ha="center",va="bottom",fontsize=7.2,color=DARK)
    e.set_ylim(0,max(.68,model["recall_mean"].max()*1.15)); e.set_ylabel("Mean top-5% recall"); e.set_title("Stable models at 1000 labels"); polish_ax(e,"y"); panel(e,"e")
    final_en=final_en.sort_values("mean")
    f.barh(range(len(final_en)),final_en["mean"],color=[TARGET_COLOR[t] for t in final_en["target"]],height=.62,zorder=2)
    f.set_yticks(range(len(final_en))); f.set_yticklabels([TARGET_LABEL[t] for t in final_en["target"]]); f.axvline(1,color="#9AA5AD",ls="--",lw=.9)
    for i,v in enumerate(final_en["mean"]): f.text(v+.16,i,f"{v:.1f}×",va="center",fontsize=7.5,color=DARK)
    f.set_xlabel("Enrichment at 1000 labels"); f.set_title("Final enrichment summary"); polish_ax(f,"x"); panel(f,"f")
    fig.suptitle("Few-shot learning phase diagram for adsorption-elite recovery",fontsize=14.6,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.955],w_pad=2.2,h_pad=2.2)
    save(fig,"Figure_2_v4_9_fewshot_learning_phase_diagram",False,formats,dpi)


def figure3(formats,dpi):
    hm=read_main("Figure_3a_rac_gain_heatmap.csv").set_index("target_label")
    bars=ordered(read_main("Figure_3b_rac_gain_bars_1000_labels.csv"))
    split=read_main("Figure_3c_split_stress_recall_heatmap.csv").set_index("target_label")
    topo=ordered(read_main("Figure_3d_topology_penalty.csv"))
    vuln=read_main("Figure_3e_extrapolation_vulnerability.csv").set_index("target_label")
    mech=read_main("Figure_3f_mechanistic_interpretation.csv")
    fig,axes=plt.subplots(2,3,figsize=(14.0,7.7)); a,b,c,d,e,f=axes.ravel()
    heatmap(a,hm,"RAC descriptor gain across budgets",r"Δ top-5% recall",True,"+.02f"); panel(a,"a")
    bars=bars.sort_values("delta_recall_racs_minus_geometry")
    b.barh(range(len(bars)),bars["delta_recall_racs_minus_geometry"],color=[TARGET_COLOR[t] for t in bars["target"]],height=.62)
    b.set_yticks(range(len(bars))); b.set_yticklabels([TARGET_LABEL[t] for t in bars["target"]]); b.axvline(0,color="#9AA5AD",lw=.9)
    for i,v in enumerate(bars["delta_recall_racs_minus_geometry"]): b.text(v+.004,i,f"{v:+.2f}",va="center",fontsize=7.2)
    b.set_xlabel("Δ recall: RACs − geometry"); b.set_title("Chemistry gain at 1000 labels"); polish_ax(b,"x"); panel(b,"b")
    heatmap(c,split,"Split-stress map at 1000 labels","Top-5% recall",False,".2f","YlGnBu"); panel(c,"c")
    topo=topo.sort_values("topology_penalty_random_minus_topology")
    d.barh(range(len(topo)),topo["topology_penalty_random_minus_topology"],color=[TARGET_COLOR[t] for t in topo["target"]],height=.62)
    d.set_yticks(range(len(topo))); d.set_yticklabels([TARGET_LABEL[t] for t in topo["target"]])
    for i,v in enumerate(topo["topology_penalty_random_minus_topology"]): d.text(v+.004,i,f"{v:.2f}",va="center",fontsize=7.2)
    d.set_xlabel("Random recall − topology recall"); d.set_title("Topology extrapolation penalty"); polish_ax(d,"x"); panel(d,"d")
    heatmap(e,vuln,"Extrapolation vulnerability","Relative recall loss vs random",True,"+.02f"); panel(e,"e")
    f.set_axis_off(); panel(f,"f",x=-.07,y=1.06)
    y=.89
    accents=["#EAF2FB","#EAF7F2","#FFF3E0","#F4EEFB"]
    for i,(_,r) in enumerate(mech.iterrows()):
        h=.19; box=FancyBboxPatch((.04,y-h),.92,h-.018,transform=f.transAxes,boxstyle="round,pad=.012,rounding_size=.012",facecolor=accents[i],edgecolor="#CBD4DB",linewidth=.7)
        f.add_patch(box); f.text(.075,y-.055,r["concept"],transform=f.transAxes,fontweight="bold",fontsize=8.5,color=DARK,va="top")
        f.text(.075,y-.10,r["interpretation"],transform=f.transAxes,fontsize=7.2,color="#3F4B54",va="top",wrap=True)
        y-=.215
    f.set_title("Mechanistic interpretation",pad=10)
    fig.suptitle("Descriptor chemistry improves screening, but topology extrapolation remains limiting",fontsize=14.3,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.955],w_pad=2.0,h_pad=2.15)
    save(fig,"Figure_3_v4_9_descriptor_chemistry_extrapolation_stress",False,formats,dpi)


def figure4(formats,dpi):
    cov=read_main("Figure_4a_coverage_error_heatmap.csv").set_index("target_label")
    width=read_main("Figure_4b_normalized_interval_width.csv")
    front=read_main("Figure_4c_precision_yield_frontier.csv")
    p10=ordered(read_main("Figure_4d_precision_at_10pct_retained.csv"))
    tiers=read_main("Figure_4e_prefinal_tier_composition.csv").sort_values("budget")
    methods=read_main("Figure_4f_coverage_method_comparison.csv")
    fig,axes=plt.subplots(2,3,figsize=(14.0,7.7)); a,b,c,d,e,f=axes.ravel()
    heatmap(a,cov,"Coverage error relative to 90%","coverage − 0.90",True,"+.02f"); panel(a,"a")
    target_curves(b,width.rename(columns={"normalized_width":"mean"}),ylabel="Interval width / target std",title="Uncertainty narrows with labels",legend=True); panel(b,"b")
    for t in TARGETS:
        sub=front[front["target"].astype(str).eq(t)].sort_values("retained_fraction")
        c.plot(sub["retained_fraction"],sub["precision_true_top5"],marker="o",color=TARGET_COLOR[t],label=TARGET_LABEL[t])
    c.set_xlim(0,.52); c.set_ylim(0,1.04); c.set_xlabel("Retained fraction"); c.set_ylabel("True top-5% fraction"); c.set_title("Precision–yield frontier"); polish_ax(c,"both"); c.legend(loc="lower right"); panel(c,"c")
    d.bar(range(len(p10)),p10["precision_true_top5"],color=[TARGET_COLOR[t] for t in p10["target"]],width=.64)
    d.set_xticks(range(len(p10))); d.set_xticklabels([TARGET_LABEL[t] for t in p10["target"]],rotation=28,ha="right")
    for i,v in enumerate(p10["precision_true_top5"]): d.text(i,min(v+.03,1.03),f"{v:.2f}",ha="center",fontsize=7.3)
    d.set_ylim(0,1.08); d.set_ylabel("True top-5% fraction"); d.set_title("Precision at ~10% retained"); polish_ax(d,"y"); panel(d,"d")
    x=np.arange(len(tiers)); bottom=np.zeros(len(tiers))
    cols=[("trusted_fraction","trusted","#2EAA72"),("uncertain_fraction","uncertain","#E7A53B"),("rejected_fraction","rejected","#9AA4AA")]
    for col,label,color in cols:
        vals=tiers[col].to_numpy(float); e.bar(x,vals,bottom=bottom,label=label,color=color,width=.72); bottom+=vals
    e.set_xticks(x); e.set_xticklabels(tiers["budget"].astype(int).astype(str)); e.set_ylim(0,1); e.set_xlabel("Label budget"); e.set_ylabel("Fraction of prediction rows"); e.set_title("Pre-final uncertainty-tier composition"); e.legend(ncol=3,loc="upper center"); panel(e,"e")
    f.bar(range(len(methods)),methods["coverage"],color=["#4EA5D9","#48C78E","#9B6BC8"][:len(methods)],width=.6)
    f.set_xticks(range(len(methods))); f.set_xticklabels(methods["method"]); f.axhline(.90,color="#7F8991",ls="--",lw=.9)
    for i,v in enumerate(methods["coverage"]): f.text(i,v+.0015,f"{v:.3f}",ha="center",fontsize=7.3)
    f.set_ylim(.86,.94); f.set_ylabel("Empirical coverage"); f.set_title("Coverage method comparison"); polish_ax(f,"y"); panel(f,"f")
    fig.suptitle("Conformal uncertainty converts few-shot predictions into risk-controlled shortlists",fontsize=14.3,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.955],w_pad=2.1,h_pad=2.2)
    save(fig,"Figure_4_v4_9_conformal_risk_control_frontier",False,formats,dpi)


def figure5(formats,dpi):
    funnel=read_main("Figure_5_candidate_funnel.csv")
    qce=ordered(read_main("Figure_5_true_enrichment.csv"))
    top=read_main("Figure_5_top25_consensus_candidates_with_percentiles.csv")
    fig,axes=plt.subplots(2,3,figsize=(14.0,7.8)); a,b,c,d,e,f=axes.ravel()
    stages=["Labelled MOFs","Quality-gated rows","Consensus pass","Top-25"]
    mat=funnel.pivot_table(index="target",columns="stage",values="count",aggfunc="first").reindex(TARGETS)[stages]
    pmat=np.log10(mat.replace(0,np.nan)); im=a.imshow(pmat.to_numpy(),aspect="auto",cmap="Blues")
    a.set_xticks(range(4)); a.set_xticklabels(["Labelled","Quality\ngated","Consensus","Top-25"]); a.set_yticks(range(4)); a.set_yticklabels([TARGET_LABEL[t] for t in TARGETS])
    for i in range(4):
        for j in range(4): a.text(j,i,f"{int(mat.iloc[i,j]):,}",ha="center",va="center",fontsize=7.0,color=DARK if pmat.iloc[i,j]<4.5 else "white")
    cb=fig.colorbar(im,ax=a,fraction=.046,pad=.025); cb.set_label(r"log$_{10}$(count)",fontsize=7.5); a.set_title("Decision funnel to final candidates"); panel(a,"a")
    vals=qce["fold_enrichment_vs_dataset_median"].to_numpy(float); b.bar(range(4),vals,color=[TARGET_COLOR[t] for t in qce["target"]],width=.64); b.axhline(1,color="#9AA5AD",ls="--",lw=.9)
    b.set_xticks(range(4)); b.set_xticklabels([TARGET_LABEL[t] for t in qce["target"]],rotation=28,ha="right")
    for i,v in enumerate(vals): b.text(i,v+.20,f"{v:.1f}×",ha="center",fontsize=7.5)
    b.set_ylabel("Shortlist median / dataset median"); b.set_title("True enrichment of consensus shortlist"); polish_ax(b,"y"); panel(b,"b")
    x=np.arange(4); bw=.23
    for k,(col,label) in enumerate([("fraction_true_top1","top 1%"),("fraction_true_top5","top 5%"),("fraction_true_top10","top 10%")]):
        c.bar(x+(k-1)*bw,qce[col],width=bw,label=label)
    c.set_xticks(x); c.set_xticklabels([TARGET_LABEL[t] for t in qce["target"]],rotation=28,ha="right"); c.set_ylim(0,1.05); c.set_ylabel("Fraction of top-25 shortlist"); c.set_title("How often candidates are true elites"); c.legend(ncol=3,loc="upper left"); polish_ax(c,"y"); panel(c,"c")
    data=[pd.to_numeric(top.loc[top["target"].astype(str).eq(t),"true_percentile"],errors="coerce").dropna().to_numpy() for t in TARGETS]
    bp=d.boxplot(data,tick_labels=[TARGET_LABEL[t] for t in TARGETS],showfliers=False,patch_artist=True,medianprops={"color":"#D35400","linewidth":1.2})
    for patch in bp["boxes"]: patch.set(facecolor="#F5F7F8",edgecolor="#66747E")
    rng=np.random.default_rng(42)
    for i,(t,v) in enumerate(zip(TARGETS,data),1): d.scatter(i+rng.normal(0,.035,len(v)),v,s=13,alpha=.58,color=TARGET_COLOR[t],edgecolors="none")
    d.set_ylim(.68,1.01); d.tick_params(axis="x",rotation=28); d.set_ylabel("True uptake percentile"); d.set_title("Final candidates occupy the adsorption tail"); polish_ax(d,"y"); panel(d,"d")
    support=["n_models","n_seeds","n_splits","n_budgets"]; sdata=[pd.to_numeric(top[x],errors="coerce").dropna().to_numpy() for x in support]
    bp=e.boxplot(sdata,tick_labels=[x.replace("n_","") for x in support],showfliers=False,patch_artist=True,medianprops={"color":"#D35400","linewidth":1.2})
    for patch in bp["boxes"]: patch.set(facecolor="#F5F7F8",edgecolor="#66747E")
    rng=np.random.default_rng(43)
    for i,v in enumerate(sdata,1): e.scatter(i+rng.normal(0,.035,len(v)),v,s=9,alpha=.28,color=NEUTRAL,edgecolors="none")
    e.text(.03,.95,f"median trusted support = {pd.to_numeric(top['trusted_support'],errors='coerce').median():.0f} rows",transform=e.transAxes,fontsize=7.4,va="top",bbox=dict(boxstyle="round,pad=.25",fc="white",ec="#CCD4D9",lw=.6))
    e.set_ylabel("Distinct support count"); e.set_title("Consensus stability across runs"); polish_ax(e,"y"); panel(e,"e")
    # Fully reproducible candidate-only geometry atlas (no hidden master-table background dependency).
    for t in TARGETS:
        sub=top[top["target"].astype(str).eq(t)]; f.scatter(sub["Di"],sub["Density"],s=32,alpha=.80,color=TARGET_COLOR[t],label=TARGET_LABEL[t],edgecolor="white",linewidth=.35)
    f.set_xlabel(r"PLD proxy, $D_i$ / Å"); f.set_ylabel(r"Density / g cm$^{-3}$"); f.set_title("Geometry regime of final candidates"); polish_ax(f,"both"); f.legend(loc="upper right",fontsize=6.7); panel(f,"f")
    fig.suptitle("Target-balanced consensus shortlists are truly enriched in adsorption elites",fontsize=14.3,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.955],w_pad=2.15,h_pad=2.2)
    save(fig,"Figure_5_v4_9_consensus_shortlist_atlas",False,formats,dpi)


def si1(formats,dpi):
    hist=read_si("Figure_S1_histogram_bin_counts_recovered_from_v48_svg.csv")
    fig,axes=plt.subplots(2,2,figsize=(10.9,8.15))
    for ax,t in zip(axes.ravel(),TARGETS):
        sub=hist[hist["target"].astype(str).eq(t)].sort_values("bin_index")
        width=(sub["bin_right"]-sub["bin_left"]).to_numpy(float)
        ax.bar(sub["bin_left"],sub["count"],width=width,align="edge",color=TARGET_COLOR[t],alpha=.86,linewidth=0)
        p95=float(sub["p95_threshold"].iloc[0]); ax.axvline(p95,color=DARK,ls=(0,(3,2)),lw=.9,label="top-5% threshold")
        ax.set_yscale("log"); ax.set_title(TARGET_LABEL[t]); ax.set_xlabel(r"Uptake / mmol g$^{-1}$"); ax.set_ylabel("Count (log)"); ax.legend(loc="upper right",fontsize=6.8); polish_ax(ax,None)
    fig.suptitle("Figure S1. Full target distributions in ARC-MOF",fontsize=14.0,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.955],w_pad=2.0,h_pad=2.0)
    save(fig,"Figure_S1_v4_9_target_distributions",True,formats,dpi)


def si2(formats,dpi):
    mp=read_si("Figure_S2a_model_pass_fraction.csv"); rc=read_si("Figure_S2b_gate_outcome_rejection_reason.csv")
    fig,axes=plt.subplots(1,2,figsize=(12.9,4.65)); a,b=axes
    a.barh(mp["model"].astype(str).str.replace("_"," ",regex=False),mp["experiment_passes_quality_gate"],color="#7C8991",height=.64); a.set_xlim(0,1.05); a.set_xlabel("Fraction passing final-candidate gate"); a.set_title("Model-level pass fraction"); polish_ax(a,"x"); panel(a,"a",x=-.07)
    b.barh(rc["gate_outcome_or_rejection_reason"],rc["experiments"],color="#9BA6AD",height=.64); b.set_xlabel("Experiments"); b.set_title("Gate outcome / rejection reason"); polish_ax(b,"x"); panel(b,"b",x=-.07)
    fig.suptitle("Figure S2. Candidate-quality gate audit",fontsize=13.7,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.92],w_pad=2.8)
    save(fig,"Figure_S2_v4_9_quality_gate_audit",True,formats,dpi)


def si3(formats,dpi):
    s=ordered(read_si("Figure_S_external_domain_overlap_source.csv"))
    fig,axes=plt.subplots(1,3,figsize=(13.7,3.85)); a,b,c=axes
    x=np.arange(len(s)); bw=.34
    a.bar(x-bw/2,s["core_geometry_overlap_fraction"],width=bw,label="CoRE geometry",color="#4EA5D9")
    a.bar(x+bw/2,s["mosaec_geometry_overlap_fraction"],width=bw,label="MOSAEC geometry",color="#48C78E")
    a.set_xticks(x); a.set_xticklabels([TARGET_LABEL[t] for t in s["target"]],rotation=28,ha="right"); a.set_ylim(0,1.05); a.set_ylabel("Overlap fraction"); a.set_title("Geometry-domain overlap"); a.legend(); polish_ax(a,"y"); panel(a,"a")
    b.set_axis_off(); n=int(s["n_candidates"].sum()); core_exact=int(round((s["core_exact_fraction"]*s["n_candidates"]).sum())); mos_exact=int(round((s["mosaec_exact_fraction"]*s["n_candidates"]).sum()))
    b.text(.5,.67,"Exact structural matches\nwere not assumed",transform=b.transAxes,ha="center",va="center",fontsize=11.8,fontweight="bold",color=DARK)
    b.text(.5,.40,f"CoRE exact: {core_exact}/{n}\nMOSAEC exact: {mos_exact}/{n}",transform=b.transAxes,ha="center",va="center",fontsize=9.8,color="#46515A"); panel(b,"b",x=-.06)
    c.bar(range(len(s)),s["mean_external_annotation_score"],color=[TARGET_COLOR[t] for t in s["target"]],width=.64); c.set_xticks(range(len(s))); c.set_xticklabels([TARGET_LABEL[t] for t in s["target"]],rotation=28,ha="right"); c.set_ylabel("Mean annotation score"); c.set_title("Domain-overlap annotation score"); polish_ax(c,"y"); panel(c,"c")
    fig.suptitle("External overlays annotate domain overlap, not experimental validation",fontsize=13.6,fontweight="bold",color=DARK,y=.995)
    fig.tight_layout(rect=[0,0,1,.90],w_pad=2.2)
    save(fig,"Figure_S3_v4_9_external_domain_overlap_annotation",True,formats,dpi)


def repair_main_claims_table() -> None:
    path=MAIN_TABLE/"Table_2_main_claims_summary.csv"; si=SI_TABLE/"Table_S6_consensus_true_enrichment.csv"
    if not path.exists() or not si.exists(): return
    df=pd.read_csv(path); en=pd.read_csv(si)
    if "fraction_true_top5pct" in en.columns:
        mp=en.set_index("target")["fraction_true_top5pct"]
        df["consensus_shortlist_fraction_true_top5"]=df["target"].map(mp)
    df.to_csv(path,index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--formats",nargs="+",default=["png","pdf","svg"],choices=["png","pdf","svg"])
    ap.add_argument("--dpi",type=int,default=600)
    ap.add_argument("--main-only",action="store_true")
    ap.add_argument("--si-only",action="store_true")
    args=ap.parse_args()
    if args.main_only and args.si_only: ap.error("Choose only one of --main-only or --si-only")
    style(); repair_main_claims_table()
    if not args.si_only:
        figure1(args.formats,args.dpi); figure2(args.formats,args.dpi); figure3(args.formats,args.dpi); figure4(args.formats,args.dpi); figure5(args.formats,args.dpi)
    if not args.main_only:
        si1(args.formats,args.dpi); si2(args.formats,args.dpi); si3(args.formats,args.dpi)
    print(f"Publication figures regenerated under: {ROOT}")

if __name__ == "__main__":
    main()
