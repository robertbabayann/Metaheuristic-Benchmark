import numpy as np

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
