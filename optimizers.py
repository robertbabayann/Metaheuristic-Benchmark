import inspect
import math

import numpy as np

import mealpy

ALGORITHM_POOL = {
    "SA": {
        "class_name": "OriginalSA",
        "single_based": True,
        "tuning_space": {
            "temp_init": ("float", 50.0, 500.0),
            "step_size": ("float", 0.01, 0.5),
        },
    },
    "GA": {
        "class_name": "BaseGA",
        "pop_size_min": 20,
        "pop_size_step": 2,
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "pc": ("float", 0.5, 0.99),
            "pm": ("float", 0.001, 0.3),
        },
    },
    "PSO": {
        "class_name": "OriginalPSO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "c1": ("float", 0.5, 4.0),
            "c2": ("float", 0.5, 4.0),
            "w": ("float", 0.1, 0.9),
        },
    },
    "DE": {
        "class_name": "OriginalDE",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "wf": ("float", 0.1, 1.0),
            "cr": ("float", 0.1, 0.99),
        },
    },
    "IWO": {
        "class_name": "OriginalIWO",
        "pop_size_min": 20,
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "exponent": ("int", 2, 4),
            "sigma_start": ("float", 0.5, 4.0),
            "sigma_end": ("float", 0.001, 0.49),
        },
    },
    "TLO": {
        "class_name": "OriginalTLO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "BSO": {
        "class_name": "OriginalBSO",
        "defaults": {"m_clusters": 4},
        "pop_size_min": 20,
        "pop_size_multiple_of": "m_clusters",
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "m_clusters": ("int", 2, 4),
            "p1": ("float", 0.05, 0.5),
            "p2": ("float", 0.5, 0.95),
            "p3": ("float", 0.1, 0.6),
            "p4": ("float", 0.2, 0.8),
            "slope": ("int", 10, 50),
        },
    },
    "WDO": {
        "class_name": "OriginalWDO",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
            "RT": ("int", 1, 4),
            "g_c": ("float", 0.1, 0.9),
            "alp": ("float", 0.1, 0.9),
            "c_e": ("float", 0.1, 0.9),
            "max_v": ("float", 0.1, 0.9),
        },
    },
    "GWO": {
        "class_name": "OriginalGWO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "WOA": {
        "class_name": "OriginalWOA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "HHO": {
        "class_name": "OriginalHHO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "AEO": {
        "class_name": "OriginalAEO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "SSA": {
        "class_name": "OriginalSSA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "ST": ("float", 0.4, 0.95),
            "PD": ("float", 0.1, 0.9),
            "SD": ("float", 0.05, 0.5),
        },
    },
    "GBO": {
        "class_name": "OriginalGBO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "pr": ("float", 0.2, 0.9),
            "beta_min": ("float", 0.05, 1.5),
            "beta_max": ("float", 0.6, 4.0),
        },
    },
    "AOA": {
        "class_name": "OriginalAOA",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
            "alpha": ("int", 2, 10),
            "miu": ("float", 0.2, 1.5),
            "moa_min": ("float", 0.05, 0.4),
            "moa_max": ("float", 0.45, 0.95),
        },
    },
    "VCS": {
        "class_name": "OriginalVCS",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "lamda": ("float", 0.1, 0.9),
            "sigma": ("float", 0.5, 4.0),
        },
    },
    "GCO": {
        "class_name": "OriginalGCO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "cr": ("float", 0.1, 0.99),
            "wf": ("float", 0.5, 2.0),
        },
    },
    "HGS": {
        "class_name": "OriginalHGS",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "PUP": ("float", 0.01, 0.3),
            "LH": ("int", 1000, 20000),
        },
    },
    "CDO": {
        "class_name": "OriginalCDO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "ESO": {
        "class_name": "OriginalESOA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "CMA-ES": {
        "class_name": "Simple_CMA_ES",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "ES": {
        "class_name": "OriginalES",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "lamda": ("float", 0.5, 0.95),
        },
    },
    "ABC": {
        "class_name": "OriginalABC",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
            "n_limits": ("int", 5, 50),
        },
    },
    "L-SHADE": {
        "class_name": "L_SHADE",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
            "miu_f": ("float", 0.1, 0.9),
            "miu_cr": ("float", 0.1, 0.9),
        },
    },
    "JA": {
        "class_name": "OriginalJA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "RUN": {
        "class_name": "OriginalRUN",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "INFO": {
        "class_name": "OriginalINFO",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
        },
    },
    "ArchOA": {
        "class_name": "OriginalArchOA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "c1": ("float", 1.0, 3.0),
            "c2": ("float", 2.0, 6.0),
            "c3": ("float", 1.0, 3.0),
        },
    },
    "MPA": {
        "class_name": "OriginalMPA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "GSKA": {
        "class_name": "OriginalGSKA",
        "pop_size_min": 20,
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "pb": ("float", 0.05, 0.3),
            "kf": ("float", 0.1, 0.9),
            "kr": ("float", 0.1, 0.9),
        },
    },
    "SMA": {
        "class_name": "OriginalSMA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "p_t": ("float", 0.01, 0.1),
        },
    },
    "FFA": {
        "class_name": "OriginalFFA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "gamma": ("float", 0.0001, 0.01),
            "beta_base": ("float", 0.5, 2.9),
            "alpha": ("float", 0.05, 0.5),
        },
    },
    "HBA": {
        "class_name": "OriginalHBA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
}

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


def fit_pop_size(algorithm_spec, pop_size, budget):
    if algorithm_spec.get("single_based"):
        return int(pop_size)
    low = int(algorithm_spec.get("pop_size_min", 1))
    return max(low, min(int(pop_size), int(budget)))


def build_optimizer(class_name, epoch, pop_size, params=None):
    cls = get_optimizer_class(class_name)
    params = params or {}
    return cls(epoch=epoch, pop_size=pop_size, **params)