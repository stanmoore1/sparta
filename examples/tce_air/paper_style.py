"""Shared publication style for all article figures.

Conventions: serif text with Computer Modern math (matches the LaTeX body),
Okabe-Ito colorblind-safe palette, no in-plot titles (PoF captions carry the
description; multi-panel figures use (a)/(b) labels inside the axes), inward
ticks on all four sides, light grid. Two standard widths: SINGLE (~0.55
textwidth) and DOUBLE (full textwidth, two panels).
"""
import matplotlib

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"      # vermillion (use instead of red)
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#000000"
GRAY = "0.35"

SINGLE = (4.4, 3.2)
SINGLE_TALL = (4.4, 3.5)
DOUBLE = (7.0, 3.0)


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.85",
        "legend.fancybox": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # TrueType, not Type 3, for the publisher
        "ps.fonttype": 42,
    })


def panel(ax, label, loc="nw"):
    x, y, ha, va = {"nw": (0.03, 0.965, "left", "top"),
                    "ne": (0.97, 0.965, "right", "top"),
                    "sw": (0.03, 0.04, "left", "bottom"),
                    "se": (0.97, 0.04, "right", "bottom")}[loc]
    ax.text(x, y, label, transform=ax.transAxes, ha=ha, va=va,
            fontsize=10, fontweight="bold")
