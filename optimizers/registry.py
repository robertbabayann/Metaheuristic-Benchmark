import inspect
import math

import numpy as np

import mealpy

_OPTIMIZER_CLASSES = None


def _patch_mealpy_vcs():
    cls = mealpy.get_all_optimizers(verbose=False).get("OriginalVCS")
    if cls is None or getattr(cls, "_meta_v2_patched", False):
        return

    def calculate_xmean__(self, pop):
        pop = self.get_sorted_population(pop, self.problem.minmax)
        pos_list = [agent.solution for agent in pop[: self.n_best]]
        factor_down = self.n_best * math.log1p(self.n_best + 1) - math.lgamma(self.n_best + 1)
        weight = math.log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        return weight * np.sum(pos_list, axis=0)

    cls.calculate_xmean__ = calculate_xmean__
    cls._meta_v2_patched = True


_patch_mealpy_vcs()


def _all_optimizers():
    global _OPTIMIZER_CLASSES
    if _OPTIMIZER_CLASSES is None:
        _OPTIMIZER_CLASSES = mealpy.get_all_optimizers(verbose=False)
    return _OPTIMIZER_CLASSES


def get_optimizer_class(class_name):
    optimizers = _all_optimizers()
    if class_name not in optimizers:
        raise KeyError(f"mealpy has no optimizer class named {class_name}")
    return optimizers[class_name]


def default_pop_size(class_name):
    cls = get_optimizer_class(class_name)
    parameter = inspect.signature(cls.__init__).parameters.get("pop_size")
    if parameter is not None and parameter.default is not inspect.Parameter.empty:
        return int(parameter.default)
    return 100


def sanitize_pop_size(algorithm_spec, pop_size, params):
    pop_size = int(pop_size)
    step = algorithm_spec.get("pop_size_step")
    multiple_of = algorithm_spec.get("pop_size_multiple_of")
    low = int(algorithm_spec.get("pop_size_min", 1))

    multiple = None
    if multiple_of:
        multiple = int(params.get(multiple_of, algorithm_spec.get("defaults", {}).get(multiple_of, 1)))

    if step:
        pop_size = (pop_size // step) * step
    if multiple:
        pop_size = (pop_size // multiple) * multiple

    if pop_size < low:
        if multiple:
            pop_size = ((low + multiple - 1) // multiple) * multiple
        elif step:
            pop_size = ((low + step - 1) // step) * step
        else:
            pop_size = low

    return pop_size


def resolve_schedule(algorithm_spec, pop_size, budget):
    queries_per_epoch = 1 if algorithm_spec.get("single_based") else int(pop_size)
    epoch = max(1, budget // queries_per_epoch)
    return epoch, queries_per_epoch


def fit_pop_size(algorithm_spec, pop_size, budget):
    if algorithm_spec.get("single_based"):
        return int(pop_size)
    low = int(algorithm_spec.get("pop_size_min", 1))
    return max(low, min(int(pop_size), int(budget)))


def build_optimizer(class_name, epoch, pop_size, params=None):
    cls = get_optimizer_class(class_name)
    params = params or {}
    return cls(epoch=epoch, pop_size=pop_size, **params)