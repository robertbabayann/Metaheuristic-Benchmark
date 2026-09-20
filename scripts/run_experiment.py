import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import storage
from experiment import ProgressTracker, build_benchmark, get_task_targets, run_algorithm_on_task, track_progress
from optimizers import ALGORITHM_POOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id (bench_...); a new one is created if omitted")
    parser.add_argument("--track", required=True, choices=["hpo", "nas"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modes", nargs="+", default=list(config.MODES))
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHM_POOL.keys()))
    args = parser.parse_args()

    run_id = args.run or storage.new_run_id()
    os.makedirs(storage.run_dir(run_id), exist_ok=True)

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

    output_dir = os.path.join(storage.run_dir(run_id), args.track, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    task_label = f"{args.track}/{args.dataset}"
    total_budget = 0
    proxy_budget = max(1, int(config.SEARCH_BUDGET * config.TUNING_PROXY_FRACTION))
    for mode in args.modes:
        tuning_budget = config.TUNING_N_TRIALS * proxy_budget if mode == "tuned" else 0
        total_budget += len(args.algorithms) * (config.SEARCH_BUDGET * len(config.FINAL_SEEDS) + tuning_budget)
    ProgressTracker.start(total_budget, desc=run_id)

    for algorithm_name in args.algorithms:
        for mode in args.modes:
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
                task_label=task_label,
            )
            output_path = os.path.join(output_dir, f"{algorithm_name}_{mode}.json")
            with open(output_path, "w") as f:
                json.dump({"targets": targets, **result}, f, indent=2)
            ProgressTracker.log(f"{task_label} {algorithm_name} {mode} -> {output_path}")

    ProgressTracker.close()


if __name__ == "__main__":
    main()
