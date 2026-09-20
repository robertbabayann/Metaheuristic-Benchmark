import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import config
from metrics import load_cost_table, performance_profile


def plot(profiles, x_values, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, curve in sorted(profiles.items()):
        ax.plot(x_values, curve, label=method, linewidth=1.2)
    ax.set_xlabel(xlabel)
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
    cost_table = load_cost_table(config.RESULTS_DIR, metrics)
    if not cost_table:
        print("no results found yet")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    taus_linear = np.linspace(1.0, args.tau_max, args.n_taus)
    profiles_linear = performance_profile(cost_table, taus_linear)
    plot(profiles_linear, taus_linear, "tau", "Performance profile (linear)", os.path.join(args.out_dir, "profile_linear.png"))

    taus_log2 = np.geomspace(1.0, args.tau_max, args.n_taus)
    profiles_log2 = performance_profile(cost_table, taus_log2)
    plot(profiles_log2, np.log2(taus_log2), "log2(tau)", "Performance profile (log2)", os.path.join(args.out_dir, "profile_log2.png"))

    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
