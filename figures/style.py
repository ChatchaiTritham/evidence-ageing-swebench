"""Shared figure style (Okabe-Ito palette, final-size fonts) and result loader."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

# One palette for every figure: strategy -> (label, colour, marker)
BLUE, GREEN, VERM, GREY = "#0072B2", "#009E73", "#D55E00", "#9E9E9E"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


