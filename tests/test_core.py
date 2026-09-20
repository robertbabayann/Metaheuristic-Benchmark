import numpy as np
import pytest
from mealpy import FloatVar

from benchmarks import Benchmark, nearest_choice_index, onehot_argmax_decode
from metrics import BudgetExceeded, QueryBudget, performance_profile, queries_to_targets
from optimizers import ALGORITHM_POOL, build_optimizer, default_pop_size, fit_pop_size, sanitize_pop_size


class SphereBenchmark(Benchmark):
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim

    @property
    def encoding_dim(self):
        return self.dim

    def evaluate(self, vector):
        return float(np.sum((np.asarray(vector) - 0.5) ** 2))

    def _compute_global_optimum(self):
        return 0.0


def test_nearest_choice_index_edges():
    assert nearest_choice_index(0.0, 4) == 0
    assert nearest_choice_index(1.0, 4) == 3
    assert nearest_choice_index(0.99, 4) == 3
    assert nearest_choice_index(0.26, 4) == 1


def test_onehot_argmax_decode():
    vector = [0.1, 0.9, 0.2, 0.7, 0.3, 0.0]
    idx = onehot_argmax_decode(vector, n_slots=2, n_choices=3)
    assert [int(i) for i in idx] == [1, 0]


def test_sanitize_pop_size_ga_even_step():
    spec = ALGORITHM_POOL["GA"]
    assert sanitize_pop_size(spec, 21, {}) % 2 == 0
    assert sanitize_pop_size(spec, 5, {}) >= spec["pop_size_min"]


def test_sanitize_pop_size_bso_multiple_of_clusters():
    spec = ALGORITHM_POOL["BSO"]
    result = sanitize_pop_size(spec, 23, {"m_clusters": 4})
    assert result % 4 == 0
    assert result >= spec["pop_size_min"]


@pytest.mark.parametrize("class_name", ["OriginalSA", "BaseGA", "OriginalVCS", "OriginalAEO", "OriginalSSA", "OriginalHHO"])
def test_query_budget_stops_exactly_at_budget(class_name):
    bench = SphereBenchmark(dim=5)
    budget = 37
    tracker = QueryBudget(bench, budget)

    def obj_func(vector):
        return tracker(vector)

    problem = {
        "obj_func": obj_func,
        "bounds": FloatVar(lb=[0.0] * bench.encoding_dim, ub=[1.0] * bench.encoding_dim),
        "minmax": "min",
        "log_to": None,
    }
    try:
        optimizer = build_optimizer(class_name, epoch=budget, pop_size=10)
        optimizer.solve(problem, seed=1)
    except BudgetExceeded:
        pass

    assert tracker.count == budget
    assert len(tracker.best_history) == tracker.count


def test_queries_to_targets_exact_index():
    best_history = [5.0, 4.0, 3.0, 2.0, 1.0]
    targets = {"target_1": 3.5, "target_2": 1.5, "target_3": 0.1}
    result = queries_to_targets(best_history, targets)
    assert result["target_1"] == 3
    assert result["target_2"] == 5
    assert result["target_3"] == float("inf")


def test_performance_profile_does_not_hide_partial_failures():
    cost_table = {
        "reliable": {"t1": 10, "t2": 10, "t3": 10},
        "unreliable": {"t1": 5, "t2": float("inf"), "t3": float("inf")},
    }
    taus = np.array([1.0, 2.0])
    profiles = performance_profile(cost_table, taus)
    assert profiles["unreliable"][0] < profiles["reliable"][0]
    assert profiles["unreliable"][-1] <= 1.0 / 3.0 + 1e-9


def test_default_pop_size_matches_mealpy_default():
    assert default_pop_size("OriginalSA") == 2
