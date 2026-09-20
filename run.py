import dataclasses
import json
import os

import config
from experiment import ProgressTracker, build_benchmark, get_task_targets, run_algorithm_on_task, track_progress
from optimizers import ALGORITHM_POOL

TRUE_WORDS = {"y", "yes", "true", "1"}
FALSE_WORDS = {"n", "no", "false", "0"}


@dataclasses.dataclass
class RunSettings:
    search_budget: int
    final_seeds: tuple
    tuning_seed: int
    tuning_n_trials: int
    tuning_proxy_fraction: float
    random_search_repeats: int
    random_baseline_seed: int


DEFAULT_SETTINGS = RunSettings(
    search_budget=config.SEARCH_BUDGET,
    final_seeds=tuple(config.FINAL_SEEDS),
    tuning_seed=config.TUNING_SEED,
    tuning_n_trials=config.TUNING_N_TRIALS,
    tuning_proxy_fraction=config.TUNING_PROXY_FRACTION,
    random_search_repeats=config.RANDOM_SEARCH_REPEATS,
    random_baseline_seed=config.RANDOM_BASELINE_SEED,
)


def parse_value(current, raw):
    if isinstance(current, bool):
        lowered = raw.lower()
        if lowered in TRUE_WORDS:
            return True
        if lowered in FALSE_WORDS:
            return False
        raise ValueError(raw)
    if isinstance(current, tuple):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        parsed = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError:
                parsed.append(part)
        return tuple(parsed)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def edit_settings(settings):
    print("Press Enter to keep the current value")
    for f in dataclasses.fields(settings):
        current = getattr(settings, f.name)
        while True:
            raw = input(f"  {f.name} [{current}]: ").strip()
            if not raw:
                break
            try:
                setattr(settings, f.name, parse_value(current, raw))
                break
            except (ValueError, json.JSONDecodeError):
                print("    invalid value, try again")


def prepare(settings):
    answer = input("\nEdit settings? [Y/N]: ").strip().lower()
    if answer in TRUE_WORDS:
        edit_settings(settings)


def parse_indices(raw, limit):
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(part)
        value = int(part)
        if not 1 <= value <= limit:
            raise ValueError(part)
        indices.append(value)
    return indices


def task_select_menu(title, tasks, all_label):
    full_index = len(tasks) + 1
    print(f"\n{title}")
    for index, name in enumerate(tasks, 1):
        print(f"  {index}) {name}")
    print(f"  {full_index}) {all_label}")
    print("  0) Back")

    raw = input(f"Select tasks (e.g. 1,3 or {full_index}): ").strip()
    if raw == "0":
        return None
    try:
        indices = parse_indices(raw, full_index)
    except ValueError:
        print("Invalid choice")
        return None
    if full_index in indices:
        return ()
    return tuple(tasks[index - 1] for index in indices)


def choose_one(title, options, prompt="Select"):
    print(f"\n{title}")
    for index, name in enumerate(options, 1):
        print(f"  {index}) {name}")
    print("  0) Back")

    raw = input(f"{prompt}: ").strip()
    if raw == "0":
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= len(options):
        print("Invalid choice")
        return None
    return options[int(raw) - 1]


def _mean_and_hits(values):
    finite = [v for v in values if v != float("inf")]
    mean = float(sum(finite) / len(finite)) if finite else float("inf")
    return mean, len(finite), len(values)


def _fmt(value):
    return "inf" if value == float("inf") else f"{value:.3g}"


def summarize(run_index, total_runs, task_label, result, output_path):
    params = result["params"]
    fits = [r["final_fitness"] for r in result["records"]]
    target_2 = [r.get("target_2") for r in result["records"]]
    total_q = [r["total_queries"] for r in result["records"]]
    fit_mean, fit_hits, n_seeds = _mean_and_hits(fits)
    t2_mean, t2_hits, _ = _mean_and_hits(target_2)
    return " | ".join(
        [
            f"[{run_index}/{total_runs}] {task_label} {result['algorithm']} {result['mode']}",
            f"pop={params['pop_size']}",
            f"q={sum(total_q)}",
            f"fit={_fmt(fit_mean)}({fit_hits}/{n_seeds})",
            f"t2={_fmt(t2_mean)} hit={t2_hits}/{n_seeds}",
            os.path.relpath(output_path, config.RESULTS_DIR),
        ]
    )


def run_experiments():
    track = choose_one("Track:", ["hpo", "nas"])
    if track is None:
        return
    dataset_options = config.HPO_DATASETS if track == "hpo" else config.NAS_DATASETS
    datasets = task_select_menu(f"Dataset ({track}):", dataset_options, "all datasets")
    if datasets is None:
        return
    mode = choose_one("Mode:", ["default", "tuned", "both"])
    if mode is None:
        return
    algorithms = task_select_menu("Algorithms:", list(ALGORITHM_POOL.keys()), "all algorithms")
    if algorithms is None:
        return

    settings = dataclasses.replace(DEFAULT_SETTINGS)
    prepare(settings)

    if not datasets:
        datasets = tuple(dataset_options)
    if not algorithms:
        algorithms = tuple(ALGORITHM_POOL.keys())
    modes = ("default", "tuned") if mode == "both" else (mode,)

    print("\nPlan:")
    print(f"  track:      {track}")
    print(f"  datasets:   {', '.join(datasets)}")
    print(f"  modes:      {', '.join(modes)}")
    print(f"  algorithms: {', '.join(algorithms)} ({len(algorithms)})")
    print(
        "  settings:  "
        f"search_budget={settings.search_budget} "
        f"final_seeds={list(settings.final_seeds)} "
        f"tuning_n_trials={settings.tuning_n_trials} "
        f"tuning_proxy_fraction={settings.tuning_proxy_fraction} "
        f"random_search_repeats={settings.random_search_repeats}"
    )
    if input("\nRun? [Y/N]: ").strip().lower() not in TRUE_WORDS:
        return

    total_runs = len(datasets) * len(algorithms) * len(modes)
    run_index = 0
    try:
        for dataset in datasets:
            task_label = f"{track}/{dataset}"
            ProgressTracker.write(f"\n=== {task_label}: prepare")
            benchmark = build_benchmark(track, dataset)
            track_progress(benchmark)

            targets_cache_path = os.path.join(config.CACHE_DIR, "targets", track, f"{dataset}.json")
            if not os.path.exists(targets_cache_path):
                reference_total = settings.search_budget * settings.random_search_repeats
                ProgressTracker.phase(reference_total, f"[reference] {task_label}")
            targets = get_task_targets(
                benchmark,
                targets_cache_path,
                settings.search_budget,
                settings.random_search_repeats,
                settings.random_baseline_seed,
                config.TARGET_LEVELS,
            )

            output_dir = os.path.join(config.RESULTS_DIR, track, dataset)
            os.makedirs(output_dir, exist_ok=True)

            for algorithm_name in algorithms:
                for run_mode in modes:
                    run_index += 1
                    result = run_algorithm_on_task(
                        algorithm_name,
                        ALGORITHM_POOL[algorithm_name],
                        benchmark,
                        targets,
                        run_mode,
                        settings.search_budget,
                        settings.final_seeds,
                        settings.tuning_seed,
                        settings.tuning_n_trials,
                        settings.tuning_proxy_fraction,
                        run_index=run_index,
                        total_runs=total_runs,
                        task_label=task_label,
                        progress=True,
                    )
                    ProgressTracker.close()
                    output_path = os.path.join(output_dir, f"{algorithm_name}_{run_mode}.json")
                    with open(output_path, "w") as f:
                        json.dump({"targets": targets, **result}, f, indent=2)
                    ProgressTracker.write(summarize(run_index, total_runs, task_label, result, output_path))
    except KeyboardInterrupt:
        ProgressTracker.close()
        print("\ninterrupted")


def analyze_results():
    try:
        from scripts.build_performance_profile import main

        main()
    except KeyboardInterrupt:
        print("\ninterrupted")


def show_menu():
    print("\nmeta-benchmark:")
    print("  1) Run Experiments")
    print("  2) Analyze Results")
    print("  0) Exit")


def main():
    while True:
        show_menu()
        raw = input("Select: ").strip()
        if raw == "0":
            break
        if raw == "1":
            run_experiments()
        elif raw == "2":
            analyze_results()


if __name__ == "__main__":
    main()
