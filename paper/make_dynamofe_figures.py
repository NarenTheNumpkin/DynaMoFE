"""Generate publication figures for DynaMoFE paper."""

import sys
from pathlib import Path

# Add TALL4Deepfake to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code/TALL4Deepfake"))

from dynamofe.analyze_routing import (
    plot_routing_distributions,
    plot_benchmark_comparison,
    plot_bootstrap_forest,
)


if __name__ == "__main__":
    print("Generating DynaMoFE publication figures...")
    plot_routing_distributions()
    plot_benchmark_comparison()
    plot_bootstrap_forest()
    print("All publication figures generated successfully in paper/figures/ and outputs/dynamofe/figures/.")
