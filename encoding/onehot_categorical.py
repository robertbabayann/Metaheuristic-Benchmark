import numpy as np


def onehot_argmax_decode(vector, n_slots, n_choices):
    matrix = np.asarray(vector).reshape(n_slots, n_choices)
    return matrix.argmax(axis=1)
