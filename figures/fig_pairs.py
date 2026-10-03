"""Paper A, Fig. 4: per-pair regression rate and directional shift by change category (archived tier).

    python figures/fig_pairs.py
"""
import random

from style import BLUE, GREEN, GREY, HERE, load, plt

CATS = [("N-regrade", "Re-graded"), ("N-rerun", "Re-run"), ("C", "Config."), ("H", "Harness"), ("M", "Model")]
COL = {"N-regrade": GREY, "N-rerun": "#E69F00", "C": "#CC79A7", "H": BLUE, "M": GREEN}


def asym(r):
    return abs(r["improvements"] - r["regressions"]) / max(1, r["improvements"] + r["regressions"])


def main():
    pairs = load("swebench_pairs.json")["pairs"]
    rnd = random.Random(7)
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7))
    panels = ((axs[0], lambda r: r["regression_rate"], "Share of solved tasks lost", "(a) Regression rate", 1.08),
              (axs[1], asym, "|improv. − regr.| / flips", "(b) Directional shift", 1.08))
    for ax, f, ylabel, title, ymax in panels:
        for x, (c, _) in enumerate(CATS):
            rs = [r for r in pairs if r["category"] == c]
            vals = [min(f(r), ymax - 0.01) for r in rs]
            sig = [r.get("p_holm", 1.0) < 0.05 for r in rs]
            xs = [x + rnd.uniform(-0.2, 0.2) for _ in vals]
            ax.scatter([a for a, g in zip(xs, sig) if not g], [v for v, g in zip(vals, sig) if not g], s=10,
                       facecolors="none", edgecolors=COL[c], linewidths=0.7, zorder=3)
            ax.scatter([a for a, g in zip(xs, sig) if g], [v for v, g in zip(vals, sig) if g], s=10, color=COL[c], zorder=3)
            med = sorted(f(r) for r in rs)[len(rs) // 2]
            ax.plot([x - 0.32, x + 0.32], [med, med], color="#1F2A37", lw=1.4, zorder=4)
        counts = {c: sum(r["category"] == c for r in pairs) for c, _ in CATS}
        ax.set_xticks(range(len(CATS)), [lab + chr(10) + f"(n={counts[c]})" for c, lab in CATS], fontsize=7)
        ax.set(ylabel=ylabel, ylim=(-0.02, ymax)); ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "figure3_pairs.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
