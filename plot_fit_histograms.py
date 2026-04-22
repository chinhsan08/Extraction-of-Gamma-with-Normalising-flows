import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(".mplconfig")
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRUE_VALUES = {
    "results_rB": 0.10,
    "results_delta": 130.0,
    "results_gamma": 70.0,
}

DEFAULT_EXACT_VALUES = {
    "results_rB": 0.0991,
    "results_delta": 129.50,
    "results_gamma": 69.57,
}


def load_fit_results(path):
    data = np.load(path)
    required_keys = [
        "results_rB",
        "results_delta",
        "results_gamma",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return data


def make_histograms(data, output_path, title, true_values, exact_values, bins):
    label_map = {
        "results_rB": r"$r_B$",
        "results_delta": r"$\delta_B$ [deg]",
        "results_gamma": r"$\gamma$ [deg]",
    }
    colors = {
        "results_rB": "#4c78a8",
        "results_delta": "#f58518",
        "results_gamma": "#54a24b",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    for ax, key in zip(axes, ["results_rB", "results_delta", "results_gamma"]):
        values = data[key]
        ax.hist(values, bins=bins, color=colors[key], alpha=0.75, edgecolor="black")
        ax.axvline(true_values[key], color="black", linestyle="--", linewidth=2, label="Injected")
        ax.axvline(exact_values[key], color="crimson", linestyle=":", linewidth=2, label="Exact fit")
        ax.axvline(np.mean(values), color="navy", linestyle="-", linewidth=1.8, label="Ensemble mean")
        ax.set_title(label_map[key])
        ax.set_xlabel(label_map[key])
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot histograms of fitted r_B, delta_B, and gamma.")
    parser.add_argument(
        "--input",
        default="fit_results_symmetric.npz",
        help="Input .npz file containing results_rB, results_delta, and results_gamma.",
    )
    parser.add_argument(
        "--output",
        default="fit_parameter_histograms.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Ensemble fit parameter histograms",
        help="Figure title.",
    )
    parser.add_argument("--bins", type=int, default=12, help="Number of histogram bins.")
    parser.add_argument("--true-rb", type=float, default=DEFAULT_TRUE_VALUES["results_rB"])
    parser.add_argument("--true-delta", type=float, default=DEFAULT_TRUE_VALUES["results_delta"])
    parser.add_argument("--true-gamma", type=float, default=DEFAULT_TRUE_VALUES["results_gamma"])
    parser.add_argument("--exact-rb", type=float, default=DEFAULT_EXACT_VALUES["results_rB"])
    parser.add_argument("--exact-delta", type=float, default=DEFAULT_EXACT_VALUES["results_delta"])
    parser.add_argument("--exact-gamma", type=float, default=DEFAULT_EXACT_VALUES["results_gamma"])
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_fit_results(args.input)
    true_values = {
        "results_rB": args.true_rb,
        "results_delta": args.true_delta,
        "results_gamma": args.true_gamma,
    }
    exact_values = {
        "results_rB": args.exact_rb,
        "results_delta": args.exact_delta,
        "results_gamma": args.exact_gamma,
    }
    make_histograms(data, args.output, args.title, true_values, exact_values, args.bins)


if __name__ == "__main__":
    main()
