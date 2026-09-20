import numpy as np
from scipy import stats

from metrics import load_cost_table, performance_profile


def _auc(curve, taus):
    return float(np.trapz(curve, taus) / (taus[-1] - taus[0]))


def _split_method_key(method_key):
    algorithm, _, mode = method_key.rpartition("_")
    return algorithm, mode


def _collect_records(results_dir):
    import glob
    import json
    import os

    records_by_method = {}
    pattern = os.path.join(results_dir, "*", "*", "*.json")
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)
        method_key = f"{data['algorithm']}_{data['mode']}"
        records_by_method.setdefault(method_key, []).extend(data["records"])
    return records_by_method


def build_ranking_table(results_dir, metric_key="target_2", tau_max=10.0, n_taus=200):
    cost_table = load_cost_table(results_dir, [metric_key])
    taus = np.linspace(1.0, tau_max, n_taus)
    profiles = performance_profile(cost_table, taus)
    auc = {method: _auc(curve, taus) for method, curve in profiles.items()}

    records_by_method = _collect_records(results_dir)

    rows = {}
    for method_key, records in records_by_method.items():
        algorithm, mode = _split_method_key(method_key)
        if mode not in ("default", "tuned"):
            continue
        row = rows.setdefault(algorithm, {"algorithm": algorithm})

        finite = [r for r in records if np.isfinite(r.get(metric_key, float("inf")))]
        success_rate = len(finite) / len(records) if records else 0.0
        median_queries = float(np.median([r["total_queries"] for r in finite])) if finite else float("inf")
        mean_wall_clock = float(np.mean([r["wall_clock_seconds"] for r in records])) if records else 0.0

        row[f"auc_{mode}"] = auc.get(method_key, 0.0)
        row[f"success_rate_{mode}"] = success_rate
        row[f"median_queries_{mode}"] = median_queries
        row[f"mean_wall_clock_{mode}"] = mean_wall_clock

    for row in rows.values():
        auc_default = row.get("auc_default", 0.0)
        auc_tuned = row.get("auc_tuned", 0.0)
        row["best_auc"] = max(auc_default, auc_tuned)
        row["tuning_gain"] = auc_tuned - auc_default if "auc_tuned" in row and "auc_default" in row else float("nan")

    ordered = sorted(rows.values(), key=lambda r: r["best_auc"], reverse=True)
    for i, row in enumerate(ordered, 1):
        row["rank"] = i

    return ordered


def _common_matrix(cost_table, methods):
    problems = None
    for method in methods:
        keys = set(cost_table.get(method, {}))
        problems = keys if problems is None else (problems & keys)
    problems = sorted(problems or [])

    rows = []
    for problem in problems:
        values = [cost_table[method][problem] for method in methods]
        if all(not np.isfinite(v) for v in values):
            continue
        rows.append(values)
    if not rows:
        return np.empty((0, len(methods)))

    matrix = np.array(rows, dtype=float)
    finite = matrix[np.isfinite(matrix)]
    cap = float(finite.max() * 10) if finite.size else 1.0
    matrix = np.where(np.isfinite(matrix), matrix, cap)
    return matrix


def friedman_report(cost_table, min_methods=3):
    """Friedman test across every algorithm x mode combination that shares at
    least one (dataset, seed) problem with all the others. Non-convergent
    runs (metric == inf) are treated as the worst outcome on that problem
    (capped at 10x the largest finite value actually observed) rather than
    dropped, so failing to reach a target is not silently excluded."""
    methods = sorted(cost_table.keys())
    if len(methods) < min_methods:
        return None
    matrix = _common_matrix(cost_table, methods)
    if matrix.shape[0] < 2:
        return None
    stat, p_value = stats.friedmanchisquare(*matrix.T)
    return {
        "methods": methods,
        "n_tasks": matrix.shape[0],
        "statistic": float(stat),
        "p_value": float(p_value),
    }


def tuned_vs_default_report(cost_table, alpha=0.05):
    """Paired Wilcoxon signed-rank test, per algorithm, comparing the tuned
    and default regimes on the (dataset, seed) problems they share. Answers
    whether internal-parameter tuning made a statistically significant
    difference for that specific algorithm, separately from AUC/tuning_gain,
    which only report the direction and magnitude, not significance."""
    algorithms = sorted({_split_method_key(m)[0] for m in cost_table})
    results = []
    for algorithm in algorithms:
        default_key = f"{algorithm}_default"
        tuned_key = f"{algorithm}_tuned"
        if default_key not in cost_table or tuned_key not in cost_table:
            continue
        common = sorted(set(cost_table[default_key]) & set(cost_table[tuned_key]))
        if len(common) < 6:
            results.append({"algorithm": algorithm, "n_tasks": len(common), "statistic": None, "p_value": None, "note": "too few paired tasks"})
            continue

        default_vals = np.array([cost_table[default_key][p] for p in common], dtype=float)
        tuned_vals = np.array([cost_table[tuned_key][p] for p in common], dtype=float)
        finite = np.concatenate([default_vals[np.isfinite(default_vals)], tuned_vals[np.isfinite(tuned_vals)]])
        cap = float(finite.max() * 10) if finite.size else 1.0
        default_vals = np.where(np.isfinite(default_vals), default_vals, cap)
        tuned_vals = np.where(np.isfinite(tuned_vals), tuned_vals, cap)

        if np.all(default_vals == tuned_vals):
            results.append({"algorithm": algorithm, "n_tasks": len(common), "statistic": None, "p_value": None, "note": "identical, no variation"})
            continue

        stat, p_value = stats.wilcoxon(default_vals, tuned_vals)
        direction = "tuned better" if np.median(tuned_vals) < np.median(default_vals) else "default better"
        results.append({
            "algorithm": algorithm,
            "n_tasks": len(common),
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "direction": direction,
        })
    return results


def format_significance_report(cost_table, alpha=0.05, metric_key="target_2"):
    lines = [f"Significance tests on '{metric_key}' (alpha={alpha}):", ""]

    friedman = friedman_report(cost_table)
    if friedman is None:
        lines.append("Friedman test: not enough shared (dataset, seed) tasks across methods yet.")
    else:
        verdict = "significant" if friedman["p_value"] < alpha else "not significant"
        lines.append(
            f"Friedman test across {len(friedman['methods'])} algorithm x mode combinations, "
            f"{friedman['n_tasks']} shared tasks: chi2={friedman['statistic']:.3f}, "
            f"p={friedman['p_value']:.4g} ({verdict})."
        )
    lines.append("")

    lines.append("Tuned vs. default, per algorithm (paired Wilcoxon signed-rank):")
    for row in tuned_vs_default_report(cost_table, alpha=alpha):
        if row.get("p_value") is None:
            lines.append(f"  {row['algorithm']:<10} n={row['n_tasks']:<4} {row.get('note', 'skipped')}")
            continue
        verdict = "significant" if row["significant"] else "not significant"
        lines.append(
            f"  {row['algorithm']:<10} n={row['n_tasks']:<4} W={row['statistic']:.1f} "
            f"p={row['p_value']:.4g} ({verdict}, {row['direction']})"
        )
    return "\n".join(lines)


def format_ranking_table(rows):
    headers = [
        "Rank", "Algorithm", "AUC(D)", "AUC(T)", "Succ%(D)", "Succ%(T)",
        "MedQ(D)", "MedQ(T)", "WallClk(D)s", "WallClk(T)s", "TuneGain",
    ]

    def fmt(value, spec):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "-"
        return format(value, spec)

    lines = []
    col_widths = [6, 10, 7, 7, 9, 9, 8, 8, 11, 11, 9]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        cells = [
            str(row["rank"]),
            row["algorithm"],
            fmt(row.get("auc_default"), ".3f"),
            fmt(row.get("auc_tuned"), ".3f"),
            fmt(row.get("success_rate_default", 0.0) * 100, ".0f"),
            fmt(row.get("success_rate_tuned", 0.0) * 100, ".0f"),
            fmt(row.get("median_queries_default"), ".0f"),
            fmt(row.get("median_queries_tuned"), ".0f"),
            fmt(row.get("mean_wall_clock_default"), ".2f"),
            fmt(row.get("mean_wall_clock_tuned"), ".2f"),
            fmt(row.get("tuning_gain"), "+.3f"),
        ]
        lines.append(" | ".join(c.ljust(w) for c, w in zip(cells, col_widths)))

    return "\n".join(lines)