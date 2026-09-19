"""Analysis and visualization of DynaMoFE routing behavior across degradation environments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Flexible root resolution: works standalone in DynaMoFE repo or inside deepfake-research
REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_WORKSPACE = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else REPO_ROOT

if (REPO_ROOT / "outputs/dynamofe_results.json").exists() or (REPO_ROOT / "outputs/figures").exists():
    PROJECT_ROOT = REPO_ROOT
    OUTPUT_ROOT = REPO_ROOT / "outputs"
    FIGURES_DIR = OUTPUT_ROOT / "figures"
    PAPER_FIGURES_DIR = REPO_ROOT / "paper/figures"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTPUT_ROOT = PROJECT_ROOT / "outputs/dynamofe"
    FIGURES_DIR = OUTPUT_ROOT / "figures"
    PAPER_FIGURES_DIR = PROJECT_ROOT / "paper/figures"

ENV_NAMES = [
    "canonical",
    "jpeg40",
    "webp50",
    "h264_crf35",
    "h265_crf32",
    "resize075_then_h264_crf30",
    "h264_crf30_then_resize075",
]

ENV_LABELS = [
    "Clean\n(Canonical)",
    "JPEG\n(Q=40)",
    "WebP\n(Q=50)",
    "H.264\n(CRF 35)",
    "H.265\n(CRF 32)",
    "Resize $\\rightarrow$\nH.264",
    "H.264 $\\rightarrow$\nResize",
]


def plot_routing_distributions():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_file = OUTPUT_ROOT / "dynamofe_results.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    experts = data["experts"]
    env_data = data["per_environment"]

    # Extract mean weights per environment
    weights_matrix = []
    for item in env_data:
        w_dict = item["mean_weights"]
        weights_matrix.append([w_dict[exp] for exp in experts])

    weights_arr = np.array(weights_matrix)  # (7, M)

    # Style configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Stacked Bar Chart of Expert Weights across Environments
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(ENV_LABELS))
    width = 0.55

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    expert_display = {
        "fcg": "Semantic Component (FCG)",
        "tall": "Spatiotemporal Thumbnail (TALL)",
        "f3net": "Frequency Decomposition (F3Net)",
        "xception": "Spatial Texture (Xception)",
        "forensics_adapter": "Forensics Adapter",
    }

    bottom = np.zeros(len(ENV_LABELS))
    for i, exp in enumerate(experts):
        vals = weights_arr[:, i]
        ax.bar(
            x, vals, width, bottom=bottom,
            label=expert_display.get(exp, exp),
            color=colors[i % len(colors)],
            edgecolor="white", linewidth=1.2,
        )
        bottom += vals

    ax.set_ylabel("Dynamic Gating Weight", fontsize=11, fontweight="bold")
    ax.set_title("DynaMoFE Adaptive Expert Allocation Across Transmission Degradations", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_LABELS, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0 / len(experts), color="gray", linestyle="--", alpha=0.6, label="Uniform Baseline")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_dynamic_routing_weights.pdf")
    fig.savefig(FIGURES_DIR / "figure_dynamic_routing_weights.png")
    fig.savefig(PAPER_FIGURES_DIR / "figure_dynamic_routing_weights.pdf")
    fig.savefig(PAPER_FIGURES_DIR / "figure_dynamic_routing_weights.png")
    plt.close(fig)
    print("Saved figure_dynamic_routing_weights")


def plot_benchmark_comparison():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    auc_file = OUTPUT_ROOT / "dynamofe_main_comparison_auc.csv"
    if not auc_file.exists():
        print(f"AUC file not found: {auc_file}")
        return

    df = pd.read_csv(auc_file, index_col=0)

    # Key models to plot
    key_models = [
        "TALL (Local Seed 42)",
        "F3Net",
        "FCG (Official CVPR 2025)",
        "Static Quad-Expert",
        "DynaMoFE (Ours)",
    ]
    model_colors = {
        "TALL (Local Seed 42)": "#8c564b",
        "F3Net": "#7570b3",
        "FCG (Official CVPR 2025)": "#2b5c8f",
        "Static Quad-Expert": "#7f7f7f",
        "DynaMoFE (Ours)": "#d95f02",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    x = np.arange(len(ENV_LABELS))
    n_models = len(key_models)
    width = 0.15

    for i, m in enumerate(key_models):
        if m not in df.index:
            continue
        vals = [df.loc[m, env] for env in ENV_NAMES]
        offset = (i - (n_models - 1) / 2) * width
        is_ours = "DynaMoFE" in m
        bars = ax.bar(
            x + offset, vals, width,
            label=m,
            color=model_colors.get(m, "#333333"),
            edgecolor="black" if is_ours else "none",
            linewidth=1.2 if is_ours else 0.0,
            alpha=1.0 if is_ours else 0.85,
        )

    ax.set_ylabel("ROC-AUC (%)", fontsize=11, fontweight="bold")
    ax.set_title("Matched Deepfake Detection Performance Across 7 Transmission Codec Environments", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_LABELS, fontsize=9.5)
    ax.set_ylim(70, 101)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_codec_auc_comparison.pdf")
    fig.savefig(FIGURES_DIR / "figure_codec_auc_comparison.png")
    fig.savefig(PAPER_FIGURES_DIR / "figure_codec_auc_comparison.pdf")
    fig.savefig(PAPER_FIGURES_DIR / "figure_codec_auc_comparison.png")
    plt.close(fig)
    print("Saved figure_codec_auc_comparison")


def plot_bootstrap_forest():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    boot_file = OUTPUT_ROOT / "dynamofe_paired_bootstraps.csv"
    if not boot_file.exists():
        print(f"Bootstrap file not found: {boot_file}")
        return

    df = pd.read_csv(boot_file)

    # Focus on Canonical and H.264 CRF-35 across baselines
    baselines = [
        "TALL (Local Seed 42)",
        "ForensicsAdapter",
        "Xception",
        "F3Net",
        "FCG (Official CVPR 2025)",
        "Static Quad-Expert",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, sharey=True)

    for ax, env, title in [(ax1, "canonical", "Clean / Canonical"), (ax2, "h264_crf35", "H.264 (CRF 35)")]:
        sub = df[df["environment"] == env].set_index("baseline")
        y_pos = np.arange(len(baselines))

        for idx, b in enumerate(baselines):
            if b not in sub.index:
                continue
            row = sub.loc[b]
            diff = row["point_diff"]
            low = row["ci_95_low"]
            high = row["ci_95_high"]
            color = "#1b9e77" if low > 0 else "#d95f02"

            ax.errorbar(diff, idx, xerr=[[diff - low], [high - diff]], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, markersize=7)

        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(baselines, fontsize=9.5)
        ax.set_xlabel(r"$\Delta$ AUC (\% points, DynaMoFE $-$ Baseline)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    plt.suptitle("10,000-Replicate Paired Cluster Bootstrap (95% CI vs Baselines)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_bootstrap_forest.pdf")
    fig.savefig(FIGURES_DIR / "figure_bootstrap_forest.png")
    fig.savefig(PAPER_FIGURES_DIR / "figure_bootstrap_forest.pdf")
    fig.savefig(PAPER_FIGURES_DIR / "figure_bootstrap_forest.png")
    plt.close(fig)
    print("Saved figure_bootstrap_forest")


if __name__ == "__main__":
    plot_routing_distributions()
    plot_benchmark_comparison()
    plot_bootstrap_forest()
