import optuna
from mealpy import FloatVar

from metrics import BudgetExceeded, QueryBudget
from optimizers import build_optimizer, default_pop_size, fit_pop_size, sanitize_pop_size

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial, name, spec):
    kind, low, high = spec
    if kind == "int":
        return trial.suggest_int(name, low, high)
    if kind == "float":
        return trial.suggest_float(name, low, high)
    raise ValueError(f"unsupported tuning-space kind: {kind}")


def tune_internal_params(class_name, algorithm_spec, benchmark, proxy_budget, n_trials, seed, prefix=""):
    tuning_space = algorithm_spec.get("tuning_space", {})
    defaults = algorithm_spec.get("defaults", {})
    total_queries = 0

    def objective(trial):
        nonlocal total_queries
        params = {name: _suggest(trial, name, spec) for name, spec in tuning_space.items()}

        internal_params = dict(defaults)
        internal_params.update(params)
        if "pop_size" in internal_params:
            pop_size = int(internal_params.pop("pop_size"))
        else:
            pop_size = default_pop_size(class_name)
        pop_size = fit_pop_size(algorithm_spec, pop_size, proxy_budget)
        pop_size = sanitize_pop_size(algorithm_spec, pop_size, internal_params)

        tracker = QueryBudget(benchmark, proxy_budget)

        problem = {
            "obj_func": tracker,
            "bounds": FloatVar(lb=[0.0] * benchmark.encoding_dim, ub=[1.0] * benchmark.encoding_dim),
            "minmax": "min",
            "log_to": None,
        }
        trial_seed = seed + trial.number
        try:
            optimizer = build_optimizer(class_name, proxy_budget, pop_size, internal_params)
            optimizer.solve(problem, seed=trial_seed)
        except BudgetExceeded:
            pass
        except Exception:
            pass
        finally:
            total_queries += tracker.count

        return tracker.best

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, total_queries