import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import config
from metrics.performance_profile import performance_profile


def load_cost_table(metrics):
    cost_table = {}
    pattern = os.path.join(config.RESULTS_DIR, "*", "*", "*.json")
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)

        algorithm = data["algorithm"]
        mode = data["mode"]
        method_key = f"{algorithm}_{mode}"

        track = path.split(os.sep)[-3]
        dataset = path.split(os.sep)[-2]

        for metric_key in metrics:
            values = [record[metric_key] for record in data["records"]]
            finite_values = [v for v in values if np.isfinite(v)]
            cost = float(np.mean(finite_values)) if finite_values else float("inf")
            problem_key = f"{track}/{dataset}:{metric_key}"
            cost_table.setdefault(method_key, {})[problem_key] = cost

    return cost_table


def plot(profiles, taus, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, curve in sorted(profiles.items()):
        ax.plot(taus, curve, label=method, linewidth=1.2)
    ax.set_xlabel("tau")
    ax.set_ylabel("rho_s(tau)")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=6, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default=None, help="if omitted, uses target_1/target_2/target_3 together")
    parser.add_argument("--tau-max", type=float, default=32.0)
    parser.add_argument("--n-taus", type=int, default=200)
    parser.add_argument("--out-dir", default=os.path.join(config.RESULTS_DIR, "plots"))
    args = parser.parse_args()

    metrics = [args.metric] if args.metric else ["target_1", "target_2", "target_3"]
    cost_table = load_cost_table(metrics)

    os.makedirs(args.out_dir, exist_ok=True)

    taus_linear = np.linspace(1.0, args.tau_max, args.n_taus)
    profiles_linear = performance_profile(cost_table, taus_linear)
    plot(profiles_linear, taus_linear, "Performance profile (linear)", os.path.join(args.out_dir, "profile_linear.png"))

    taus_log2 = np.geomspace(1.0, args.tau_max, args.n_taus)
    profiles_log2 = performance_profile(cost_table, taus_log2)
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, curve in sorted(profiles_log2.items()):
        ax.plot(np.log2(taus_log2), curve, label=method, linewidth=1.2)
    ax.set_xlabel("log2(tau)")
    ax.set_ylabel("rho_s(tau)")
    ax.set_title("Performance profile (log2)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=6, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "profile_log2.png"), dpi=150)
    plt.close(fig)

    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
