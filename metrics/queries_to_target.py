def queries_to_targets(history_best_fit, queries_per_epoch, targets):
    result = {}
    for key, threshold in targets.items():
        if not key.startswith("target_"):
            continue
        hit_epoch = None
        for i, value in enumerate(history_best_fit):
            if value <= threshold:
                hit_epoch = i
                break
        result[key] = (hit_epoch + 1) * queries_per_epoch if hit_epoch is not None else float("inf")
    return result