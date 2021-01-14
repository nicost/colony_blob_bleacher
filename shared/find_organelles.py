# ------------------------------------------
# FUNCTIONS for ORGANELLE DETECTION
# ------------------------------------------

from skimage.filters import threshold_otsu, threshold_yen
from skimage.segmentation import clear_border

from shared.find_blobs import find_blobs, get_binary_global
from shared.objects import remove_small, remove_large
import shared.warning as warn
import numpy as np


def find_nucleoli(pixels: np.array, global_thresholding='na', extreme_val=500, bg_val=200,
                  min_size=10, max_size=1000):
    """
    Find nucleoli from a given image.

    Expects pixels to be an array, and finds nucleoli objects using watershed by
    flooding approach with indicated global thresholding methods (supports 'na',
    'otsu' and 'yen').  Founded nucleoli are filtered by default location filter
    (filter out nucleoli located at the boundary of the image) and size filter.

    :param pixels: np.array (non-negative int type)
                Image pixel
    :param global_thresholding: only accepts 'na', 'otsu', 'yen' or 'local',
                optional (default: 'local')
                Whether or not ('na') to apply global thresholding method and
                which method ('otsu' or 'yen') to apply
    :param extreme_val: int, optional (default: 500)
                Used in shared.find_blobs.segment_watershed to find local maxima
    :param bg_val: int, optional (default: 200)
                Used in shared.find_blobs.segment_watershed to define background
                for watershed
    :param min_size: int, optional (default: 10)
                The smallest allowable nucleoli size.
    :param max_size: int, optional (default: 1000)
                The largest allowable nucleoli size.
    :returns nucleoli_filtered: 0-and-1 ndarray, same shape and type as input img
                Binary array with found nucleoli labeled with 1.
    """
    # Raise type error if not int
    warn.check_img_supported(pixels)

    # Check global thresholding options
    # Raise value error if not 'na', 'otsu' or 'yen'
    warn.check_input_supported(global_thresholding, ['na', 'otsu', 'yen', 'local'])

    # find nucleoli
    nucleoli = find_blobs(pixels, get_binary_global(pixels, global_thresholding), extreme_val, bg_val)

    # Nucleoli filters:
    # Location filter: remove artifacts connected to image border
    nucleoli_filtered = clear_border(nucleoli)
    # Size filter: default [10,1000]
    nucleoli_filtered = remove_small(nucleoli_filtered, min_size)
    nucleoli_filtered = remove_large(nucleoli_filtered, max_size)

    return nucleoli_filtered
