import json
import os

from mealpy import FloatVar # type: ignore

from experiment.progress import ProgressTracker
from metrics.queries_to_target import queries_to_targets
from metrics.targets import compute_targets, random_baseline_median
from optimizers.registry import build_optimizer, default_pop_size, fit_pop_size, resolve_schedule, sanitize_pop_size
from tuning.tpe_tuner import tune_internal_params


def _run_single_seed(class_name, params, pop_size, benchmark, epoch, seed, desc_prefix=""):
    best = float("inf")

    def obj_func(vector):
        nonlocal best
        value = benchmark.evaluate(vector)
        if value < best:
            best = value
        if desc_prefix:
            ProgressTracker.desc(f"{desc_prefix} | best={best:.4g}")
        return value

    problem = {
        "obj_func": obj_func,
        "bounds": FloatVar(lb=[0.0] * benchmark.encoding_dim, ub=[1.0] * benchmark.encoding_dim),
        "minmax": "min",
        "log_to": None,
    }
    try:
        optimizer = build_optimizer(class_name, epoch, pop_size, params)
        optimizer.solve(problem, seed=seed)
    except Exception:
        return [], float("inf")
    return list(optimizer.history.list_global_best_fit), float(optimizer.g_best.target.fitness)


def get_task_targets(benchmark, cache_path, random_baseline_samples, random_baseline_seed, target_levels):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    median = random_baseline_median(benchmark, random_baseline_samples, random_baseline_seed)
    targets = compute_targets(benchmark, median, target_levels)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(targets, f)

    return targets


def run_algorithm_on_task(
    algorithm_name,
    algorithm_spec,
    benchmark,
    targets,
    mode,
    budget,
    final_seeds,
    tuning_seed,
    tuning_n_trials,
    tuning_proxy_fraction,
    run_index=None,
    total_runs=None,
    task_label=None,
    progress=False,
):
    class_name = algorithm_spec["class_name"]
    defaults = algorithm_spec.get("defaults", {})

    prefix_parts = []
    if run_index or total_runs:
        prefix_parts.append(f"[{run_index or 0}/{total_runs}]")
    if task_label:
        prefix_parts.append(task_label)
    prefix_parts.append(f"{algorithm_name} {mode}")
    prefix = " ".join(part for part in prefix_parts if part)

    if mode == "tuned":
        proxy_budget = max(1, int(budget * tuning_proxy_fraction))
        if progress:
            ProgressTracker.phase(tuning_n_trials * proxy_budget, f"{prefix} | tuning")
        tuned_params, tuning_queries = tune_internal_params(
            class_name,
            algorithm_spec,
            benchmark,
            budget,
            tuning_proxy_fraction,
            tuning_n_trials,
            tuning_seed,
            prefix=f"{prefix} | tuning",
        )
        best_params = {**defaults, **tuned_params}
    else:
        best_params, tuning_queries = dict(defaults), 0

    internal_params = dict(best_params)
    if "pop_size" in internal_params:
        pop_size = int(internal_params.pop("pop_size"))
    else:
        pop_size = default_pop_size(class_name)
    pop_size = fit_pop_size(algorithm_spec, pop_size, budget)
    pop_size = sanitize_pop_size(algorithm_spec, pop_size, internal_params)

    epoch, queries_per_epoch = resolve_schedule(algorithm_spec, pop_size, budget)
    search_queries = epoch * queries_per_epoch

    if progress:
        ProgressTracker.phase(len(final_seeds) * search_queries, f"{prefix} | seed 1/{len(final_seeds)}")

    records = []
    for i, seed in enumerate(final_seeds, 1):
        desc_prefix = f"{prefix} | seed {i}/{len(final_seeds)}" if progress else ""
        history, final_fitness = _run_single_seed(class_name, internal_params, pop_size, benchmark, epoch, seed, desc_prefix)
        q = queries_to_targets(history, queries_per_epoch, targets)
        records.append(
            {
                "seed": seed,
                "pop_size": pop_size,
                "epoch": epoch,
                "final_fitness": final_fitness,
                "search_queries": search_queries,
                "tuning_queries": tuning_queries,
                "total_queries": search_queries + tuning_queries,
                **q,
            }
        )

    return {
        "algorithm": algorithm_name,
        "mode": mode,
        "params": {**internal_params, "pop_size": pop_size},
        "records": records,
    }