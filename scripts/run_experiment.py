import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from experiment import ProgressTracker, build_benchmark, get_task_targets, run_algorithm_on_task, track_progress
from optimizers import ALGORITHM_POOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["hpo", "nas"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modes", nargs="+", default=list(config.MODES))
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHM_POOL.keys()))
    args = parser.parse_args()

    benchmark = build_benchmark(args.track, args.dataset)
    track_progress(benchmark)

    targets_cache_path = os.path.join(config.CACHE_DIR, "targets", args.track, f"{args.dataset}.json")
    targets = get_task_targets(
        benchmark,
        targets_cache_path,
        config.SEARCH_BUDGET,
        config.RANDOM_SEARCH_REPEATS,
        config.RANDOM_BASELINE_SEED,
        config.TARGET_LEVELS,
    )

    output_dir = os.path.join(config.RESULTS_DIR, args.track, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    total_runs = len(args.algorithms) * len(args.modes)
    run_index = 0
    task_label = f"{args.track}/{args.dataset}"
    for algorithm_name in args.algorithms:
        for mode in args.modes:
            run_index += 1
            result = run_algorithm_on_task(
                algorithm_name,
                ALGORITHM_POOL[algorithm_name],
                benchmark,
                targets,
                mode,
                config.SEARCH_BUDGET,
                config.FINAL_SEEDS,
                config.TUNING_SEED,
                config.TUNING_N_TRIALS,
                config.TUNING_PROXY_FRACTION,
                run_index=run_index,
                total_runs=total_runs,
                task_label=task_label,
                progress=True,
            )
            ProgressTracker.close()
            output_path = os.path.join(output_dir, f"{algorithm_name}_{mode}.json")
            with open(output_path, "w") as f:
                json.dump({"targets": targets, **result}, f, indent=2)
            print(f"[{run_index}/{total_runs}] saved {output_path}")


if __name__ == "__main__":
    main()
