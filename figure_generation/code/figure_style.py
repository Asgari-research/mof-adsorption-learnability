from __future__ import annotations
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

MM = 1 / 25.4
FULL_WIDTH_IN = 178 * MM

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
    TARGETS[0]: "#264653",   # navy
    TARGETS[1]: "#2A9D8F",   # teal
    TARGETS[2]: "#C98B2E",   # ochre
    TARGETS[3]: "#7A5195",   # muted plum
}
TARGET_MARKER = {
    TARGETS[0]: "o",
    TARGETS[1]: "s",
    TARGETS[2]: "^",
    TARGETS[3]: "D",
}
BUDGETS = [10, 20, 50, 100, 200, 500, 1000]
DARK = "#202A33"
MID = "#59636D"
GRID = "#D9DEE3"
LIGHT = "#F5F7F8"


def _arial_candidates() -> list[Path]:
    roots = []
    if os.environ.get("FEWSHOT_ARIAL_DIR"):
        roots.append(Path(os.environ["FEWSHOT_ARIAL_DIR"]).expanduser())
    roots.extend([
        Path.home() / ".local/share/fonts/fewshot_windows_arial",
        Path("/mnt/c/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    ])
    names = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf", "Arial.ttf"]
    out: list[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            if p.exists() and p not in out:
                out.append(p)
    return out


def register_arial(strict: bool = True) -> list[Path]:
    fonts = _arial_candidates()
    for p in fonts:
        try:
            font_manager.fontManager.addfont(str(p))
        except Exception:
            pass
    regular = [p for p in fonts if p.name.lower() in {"arial.ttf"}]
    if strict and not regular:
        raise RuntimeError(
            "Actual Arial regular was not found. Run ./setup_wsl.sh first. "
            "The plotting code will not silently substitute another font."
        )
    return fonts


def resolve_arial(strict: bool = True) -> Path:
    fonts = register_arial(strict=strict)
    for p in fonts:
        if p.name.lower() == "arial.ttf":
            return p
    if strict:
        raise RuntimeError("Arial regular font could not be resolved.")
    return Path(font_manager.findfont("DejaVu Sans"))


def assert_arial_family() -> dict[str, str]:
    """Resolve the four normal/italic/bold combinations without fallback."""
    register_arial(strict=True)
    resolved = {}
    for key, weight, style in [
        ("regular", "normal", "normal"),
        ("bold", "bold", "normal"),
        ("italic", "normal", "italic"),
        ("bold_italic", "bold", "italic"),
    ]:
        path = font_manager.findfont(
            FontProperties(family="Arial", weight=weight, style=style),
            fallback_to_default=False,
        )
        resolved[key] = path
    return resolved

def configure_matplotlib(strict_font: bool = True) -> None:
    if strict_font:
        register_arial(strict=True)
        assert_arial_family()
        family = "Arial"
    else:
        family = os.environ.get("FEWSHOT_PREVIEW_FONT", "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family],
        "font.size": 9.0,
        "axes.titlesize": 9.6,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#8D969E",
        "axes.linewidth": 0.75,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "mathtext.fontset": "custom" if strict_font else "dejavusans",
        "mathtext.rm": "Arial" if strict_font else "DejaVu Sans",
        "mathtext.it": "Arial:italic" if strict_font else "DejaVu Sans:italic",
        "mathtext.bf": "Arial:bold" if strict_font else "DejaVu Sans:bold",
        "mathtext.sf": "Arial" if strict_font else "DejaVu Sans",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "lines.linewidth": 1.65,
        "lines.markersize": 4.8,
    })


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11.6, fontweight="bold",
            ha="left", va="top", clip_on=False)


def clean_axis(ax, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, output_root: Path, stem: str, dpi: int = 200) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    pdf = output_root / f"{stem}.pdf"
    png = output_root / f"{stem}.png"
    # Preserve the exact physical canvas. Do not use bbox_inches="tight":
    # it expands page width around legends/panel labels and forces later rescaling.
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return [pdf, png]
