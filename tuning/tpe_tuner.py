import optuna

from mealpy import FloatVar

from experiment.progress import ProgressTracker
from optimizers.registry import build_optimizer, default_pop_size, fit_pop_size, resolve_schedule, sanitize_pop_size

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_internal_params(class_name, algorithm_spec, benchmark, budget, proxy_fraction, n_trials, seed, prefix=""):
    tuning_space = algorithm_spec["tuning_space"]
    defaults = algorithm_spec.get("defaults", {})
    if not tuning_space:
        return {}, 0

    proxy_budget = max(1, int(budget * proxy_fraction))

    def objective(trial):
        sampled = dict(defaults)
        for name, spec in tuning_space.items():
            kind, low, high = spec
            if kind == "float":
                sampled[name] = trial.suggest_float(name, low, high)
            elif kind == "int":
                sampled[name] = trial.suggest_int(name, low, high)
            else:
                raise ValueError(f"unsupported parameter kind: {kind}")

        pop_size = int(sampled.pop("pop_size")) if "pop_size" in sampled else default_pop_size(class_name)
        pop_size = fit_pop_size(algorithm_spec, pop_size, proxy_budget)
        pop_size = sanitize_pop_size(algorithm_spec, pop_size, sampled)
        epoch, _ = resolve_schedule(algorithm_spec, pop_size, proxy_budget)

        problem = {
            "obj_func": benchmark.evaluate,
            "bounds": FloatVar(lb=[0.0] * benchmark.encoding_dim, ub=[1.0] * benchmark.encoding_dim),
            "minmax": "min",
            "log_to": None,
        }
        try:
            optimizer = build_optimizer(class_name, epoch, pop_size, sampled)
            optimizer.solve(problem, seed=seed)
        except Exception:
            return float("inf")
        if prefix:
            ProgressTracker.desc(f"{prefix} | trial {trial.number + 1}/{n_trials}")
        return optimizer.g_best.target.fitness

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    tuning_queries = n_trials * proxy_budget
    return study.best_params, tuning_queries