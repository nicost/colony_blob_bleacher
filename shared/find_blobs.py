import numpy as np
from skimage import segmentation
from skimage.filters import sobel
from skimage.morphology import extrema, binary_dilation, binary_erosion
from skimage.filters import threshold_otsu, threshold_yen, threshold_local

import shared.objects as obj


def segment_watershed(pixels: np.array, extreme_val: int, bg_val: int):
    """
    Returns an np.array that is the segmented version of the input

        Finds local maxima using extreme_val as the minimal height for the maximum,
        Then uses the local maxima and background pixels (pixels that are
        smaller than bg_val) to execute a watershed segmentation
    """
    maxima = extrema.h_maxima(pixels, extreme_val)
    elevation_map = sobel(pixels)
    markers = np.zeros_like(pixels)
    markers[pixels < bg_val] = 1
    markers[maxima == 1] = 2
    return segmentation.watershed(elevation_map, markers)


def find_blobs(pixels: np.array, binary_global: np.array, extreme_val: int, bg_val: int):
    """
    Find "blobs" in image.  Current strategy: use a segmentation.watershed on the input pixes,
    using extreme_val to find local maxima, adn bg_val to find background.  Combine the
    watershed with a globally thresholded image (using logical OR) binary_global.

    :param pixels: input image
    :param binary_global: binary threshold image gain from global thresholding
    :param extreme_val: used to find local maxima
    :param bg_val: used to define background for watershed
    :return: segmented image of same size as input
    """
    seg_wat = segment_watershed(pixels, extreme_val, bg_val)
    merge = np.zeros_like(pixels)
    merge[seg_wat == 2] = 1
    merge[binary_global == 1] = 1

    return merge


def get_binary_global(pixels: np.array, threshold_method='na'):
    """
    Calculate binary global thresholding image

    :param pixels: np.array
    :param threshold_method: method used to perform global thresholding, enable 'na',
                'otsu', 'yen' and 'local'
    :return: out: 0-and-1 np.array, binary global thresholding image
    """
    if threshold_method == 'na':
        out = np.zeros_like(pixels)
    elif (threshold_method == 'otsu')|(threshold_method == 'yen'):
        if threshold_method == 'otsu':
            global_threshold_val = threshold_otsu(pixels)
            # Threshold value to create global threshold.  try: threshold_otsu(pixels)
            # 0: does not apply global thresholding
        else:
            global_threshold_val = threshold_yen(pixels)
        out = pixels > global_threshold_val
        # one round of erosion/dilation to clear out boundary
        out = binary_erosion(out)
        out = binary_dilation(out)
    elif threshold_method == 'local':
        # generate background mask
        bg = np.zeros_like(pixels)
        bg[pixels > 200] = 1
        for i in range(10):  # 10: specific for nucleoli
            bg = binary_erosion(bg)
        # apply local thresholding
        local = threshold_local(pixels, 21)  # 21: specific for nucleoli
        out = pixels > local
        # remove large connected areas
        out = obj.remove_large(out, 1000)  # 1000: specific for nucleoli
        # avoid areas close to nuclear boundary
        # intensity tends to connected and generate fake blobs
        # might miss something
        out[bg == 0] = 0
        # two rounds of erosion/dilation and remove_small to clear out background
        out = binary_erosion(out)
        out = binary_erosion(out)
        out = obj.remove_small(out, 10)  # 10: specific for nucleoli
        out = binary_dilation(out)
        out = binary_dilation(out)

    return out


def select(in_list: list, key, in_min: int, in_max: int):
    """
    Selects subset of input dict (which should contain keys of type str, and vals of type int)
    and returns these as a dict.  Included are entries where val > in_min and val < in_max

    :param in_list:  list that contains keys of type str, and vals of type int
    :param key: key to use for selection
    :param in_min: val should be > min_val for the entry to be included
    :param in_max: vale should < max_val for the entry to be included
    :return: list that contains a subset of the entries in the input
    """
    out = []
    for test in in_list:
        if isinstance(in_list, list):
            if type(test[key]) == np.int32:
                if in_min < test[key] < in_max:
                    out.append(test)
            elif type(test[key]) == tuple:
                if in_min < test[key][0] < in_max and in_min < test[key][1] < in_max:
                    out.append(test)
    return out


# canny filter and fill
#hist, hist_centers = histogram(pixels)
#edges = canny(pixels/(hist_centers.argmax() * 0.4), sigma=3)
#filled = ndi.binary_fill_holes(edges)

#seg_500_200 = find_ps_organelles(pixels, 500, 200)
#binary_global = pixels > threshold_otsu(pixels)
##merge = np.zeros_like(pixels)
#merge[seg_500_200 == 2] = 1
#merge |= binary_global


