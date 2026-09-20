import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from metrics import load_cost_table, performance_profile
from optimizers import ALGORITHM_POOL

ALGORITHM_ORDER = list(ALGORITHM_POOL.keys())


def _color_map():
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    return {name: palette[i % len(palette)] for i, name in enumerate(ALGORITHM_ORDER)}


def _linestyle(name):
    return "--" if ALGORITHM_ORDER.index(name) % 2 else "-"


def _filter_by_mode(cost_table, mode):
    suffix = f"_{mode}"
    return {
        method[: -len(suffix)]: problems
        for method, problems in cost_table.items()
        if method.endswith(suffix)
    }


def plot_mode(cost_table, mode, x_values, xlabel, taus, title, out_path, colors):
    by_mode = _filter_by_mode(cost_table, mode)
    if not by_mode:
        return
    profiles = performance_profile(by_mode, taus)

    fig, ax = plt.subplots(figsize=(11, 7))
    for name in sorted(profiles.keys(), key=lambda n: (ALGORITHM_ORDER.index(n) if n in ALGORITHM_ORDER else 999, n)):
        ax.plot(x_values, profiles[name], label=name, linewidth=1.6, color=colors.get(name), linestyle=_linestyle(name) if name in ALGORITHM_ORDER else "-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\rho_s(\tau)$")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id (bench_...); defaults to the latest run")
    parser.add_argument("--metric", default=None, help="if omitted, uses target_1/target_2/target_3 together")
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--n-taus", type=int, default=200)
    args = parser.parse_args()

    import storage

    run_id = args.run or (storage.list_runs()[0] if storage.list_runs() else None)
    if run_id is None:
        print("no runs found")
        return

    run_dir = storage.run_dir(run_id)
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    metrics = [args.metric] if args.metric else ["target_1", "target_2", "target_3"]
    cost_table = load_cost_table(run_dir, metrics)
    if not cost_table:
        print("no results found yet")
        return

    colors = _color_map()

    taus_linear = np.linspace(1.0, args.tau_max, args.n_taus)
    taus_log2 = np.geomspace(1.0, args.tau_max, args.n_taus)

    for mode in ("default", "tuned"):
        plot_mode(cost_table, mode, taus_linear, r"$\tau$", taus_linear, f"Performance profile — {mode} (linear)",
                  os.path.join(out_dir, f"profile_{mode}_linear.png"), colors)
        plot_mode(cost_table, mode, np.log2(taus_log2), r"$\log_2(\tau)$", taus_log2, f"Performance profile — {mode} (log2)",
                  os.path.join(out_dir, f"profile_{mode}_log2.png"), colors)

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
