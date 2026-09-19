"""Create the paper's result overview from the frozen reported numbers.

This script is deliberately data-only: it does not read predictions or refit a
model.  The values below are copied from the finalized receipt-bound reports
named in the paper's reproducibility section.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    blue = "#4477AA"
    orange = "#EE7733"
    green = "#228833"
    red = "#CC3311"

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))

    # Representation Autopsy V1, complete repeated-frame controls.
    labels = ["TALL\nFF++", "TALL\nCeleb", "FCG\nFF++", "FCG\nCeleb"]
    retention = np.array([102.386, 119.809, 93.267, 102.536])
    axes[0].bar(np.arange(4), retention, color=[blue, blue, orange, orange], width=0.68)
    axes[0].axhline(100, color="black", lw=0.8, ls="--")
    axes[0].set_ylim(0, 130)
    axes[0].set_xticks(np.arange(4), labels)
    axes[0].set_ylabel("Excess-AUC retention (%)")
    axes[0].set_title("(a) Repeating one frame")
    for index, value in enumerate(retention):
        axes[0].text(index, value + 2.2, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    # Pipeline Controls V2, FF++ test source-cluster bootstrap.
    prep_labels = ["Codec /\ncontainer", "Detector\ngeometry", "All observed\n(centered)"]
    prep_auc = np.array([64.552, 70.620, 71.741])
    prep_low = np.array([60.676, 66.949, 68.348])
    prep_high = np.array([69.055, 74.418, 75.293])
    axes[1].bar(np.arange(3), prep_auc, color=[green, green, green], width=0.62)
    axes[1].errorbar(
        np.arange(3),
        prep_auc,
        yerr=np.vstack((prep_auc - prep_low, prep_high - prep_auc)),
        fmt="none",
        ecolor="black",
        capsize=2.5,
        lw=0.8,
    )
    axes[1].axhline(50, color="black", lw=0.8, ls="--")
    axes[1].set_ylim(45, 80)
    axes[1].set_xticks(np.arange(3), prep_labels)
    axes[1].set_ylabel("FF++ test AUC (%)")
    axes[1].set_title("(b) Pixels excluded")
    for index, value in enumerate(prep_auc):
        axes[1].text(index, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    # CODS multi-seed confirmation and processed-DFDCP report.
    seeds = ["42", "123", "3407"]
    source_delta = np.array([0.2887, 0.3072, 0.2226])
    target_delta = np.array([-0.1466, -0.1953, -0.3057])
    x = np.arange(3)
    width = 0.34
    axes[2].bar(x - width / 2, source_delta, width, color=blue, label="FF++ sealed codecs")
    axes[2].bar(x + width / 2, target_delta, width, color=red, label="Processed DFDCP")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x, seeds)
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("Decision $-$ Average (AUC pp)")
    axes[2].set_title("(c) Robustness reverses")
    axes[2].legend(loc="lower left", frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        axis.set_axisbelow(True)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "result_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "result_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
