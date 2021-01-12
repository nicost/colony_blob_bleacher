import numpy as np
from skimage.morphology import remove_small_objects

def remove_small(ar, min_size = 10):
    ar_bool = np.array(ar, bool)
    ar_mask = remove_small_objects(ar_bool, min_size)
    ar_filtered = np.zeros_like(ar)
    ar_filtered[ar_mask] = 1

    return ar_filtered

def remove_large(ar, max_size = 1000):
    ar_bool = np.array(ar, bool)
    ar_mask = remove_small_objects(ar_bool, max_size)
    ar_filtered = ar.copy()
    ar_filtered[ar_mask] = 0

    return ar_filtered
