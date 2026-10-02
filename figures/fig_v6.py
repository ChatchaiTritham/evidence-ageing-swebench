"""Paper A, RQ5 figure: recall of change-induced regressions by strategy and budget (results/v6_selection.json)."""
from style import BLUE, GREEN, GREY, HERE, VERM, load, plt

STR = [("random", "Random", GREY, "o"), ("coarse", "Coarse (repository)", "#E69F00", "^"),
       ("flaky", "Instability", "#CC79A7", "s"), ("informed", "Informed sentinel (Eq. 4)", VERM, "v"),
       ("predictive", "Regression history", BLUE, "D"), ("fragile", "Historical difficulty", GREEN, "P")]


def main():
    d = load("v6_selection.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    for ax, subset, title in ((axs[0], "all", f"(a) All pairs (n={d['pairs_evaluated']})"),
                              (axs[1], "significant_shift", f"(b) Significant shift (n={d['pairs_significant_shift']})")):
        res = d["results"][subset]
        bs = [float(b) for b in res]
        for key, lab, col, mk in STR:
            m = [res[str(b)][key]["mean_recall"] for b in bs]
            lo = [res[str(b)][key]["ci95"][0] for b in bs]
            hi = [res[str(b)][key]["ci95"][1] for b in bs]
            ax.plot(bs, m, marker=mk, color=col, label=lab, ms=4)
            ax.fill_between(bs, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set(xlabel="Budget (share of solved tasks re-run)", xticks=bs, xticklabels=[f"{b:.0%}" for b in bs],
               ylim=(0, 0.9))
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("Recall of regressions")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(HERE / "figure5_selection.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
