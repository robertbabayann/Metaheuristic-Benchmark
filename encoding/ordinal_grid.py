def nearest_choice_index(value, n_choices):
    clipped = min(max(value, 0.0), 1.0)
    idx = int(clipped * n_choices)
    return min(idx, n_choices - 1)
