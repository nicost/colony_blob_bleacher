import numpy as np
from skimage import segmentation
from skimage.filters import sobel
from skimage.morphology import extrema, binary_dilation, binary_erosion
from skimage.filters import threshold_otsu, threshold_yen, threshold_local
import shared.objects as obj
from shared.objects import remove_large
from scipy import ndimage

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for BLOB IDENTIFICATION
# ---------------------------------------------------------------------------------------------------

segment_watershed
    FUNCTION: returns an np.array that is the segmented version of the input
    SYNTAX:   segment_watershed(pixels: np.array, extreme_val: int, bg_val: int)

find_blobs
    FUNCTION: find blobs in image
    SYNTAX:   find_blobs(pixels: np.array, binary_global: np.array, extreme_val: int, bg_val=200, 
              max_size=1000)

get_binary_global
    FUNCTION: calculate binary global thresholding image
    SYNTAX:   get_binary_global(pixels: np.array, threshold_method='na', min_size=5, max_size=1000)

select
    FUNCTION: selects subset of input dict
    SYNTAX:   select(in_list: list, key, in_min: int, in_max: int)
    
"""


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


def find_blobs(pixels: np.array, binary_global: np.array, extreme_val: int, bg_val=200, max_size=1000):
    """
    Find "blobs" in image.  Current strategy: use a segmentation.watershed on the input pixes,
    using extreme_val to find local maxima, adn bg_val to find background.  Combine the
    watershed with a globally threshold image (using logical OR) binary_global.

    :param pixels: input image
    :param binary_global: binary threshold image gain from global thresholding
    :param extreme_val: used to find local maxima
    :param bg_val: used to define background for watershed
    :param max_size: maximum size of the blobs
    :return: segmented image of same size as input

    """
    if np.amax(pixels) < 1000:
        merge = np.zeros_like(pixels)
    else:
        seg_wat = segment_watershed(pixels, extreme_val, bg_val)
        merge = np.zeros_like(pixels)
        merge[seg_wat == 2] = 1
        merge = remove_large(merge, max_size)
        binary_global = remove_large(binary_global, max_size)
        merge[binary_global == 1] = 1

    return merge


def get_binary_global(pixels: np.array, threshold_method='na', min_size=5, max_size=1000, local_param=21):
    """
    Calculate binary global thresholding image

    :param pixels: np.array
    :param threshold_method: method used to perform global thresholding, enable 'na',
                'otsu', 'yen', 'local-nucleoli' and 'local-sg', 'local-sg1'
                'na': not applied, return a black image
                'otsu': otsu thresholding + one round of erosion/dilation
                'yen': yen thresholding + one round of erosion/dilation
                'local-nucleoli': otsu & local thresholding for nucleoli identification
                'local-sg': otsu & local thresholding for stress granule identification
    :param min_size: minimum size of blobs
    :param max_size: maximum size of blobs
    :param local_param: parameter for local thresholding
    :return: out: 0-and-1 np.array, binary global thresholding image

    """

    check_lst = ['na', 'otsu', 'yen', 'local-nucleoli', 'local-sg', 'local-sg1']
    if threshold_method not in check_lst:
        raise ValueError("global thresholding method only accepts %s. Got %s" % (check_lst, threshold_method))

    elif (threshold_method == 'otsu') | (threshold_method == 'yen'):
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

    elif threshold_method == 'local-nucleoli':
        # use otsu thresholding to determine background region
        global_threshold_val = threshold_otsu(pixels)
        bg = pixels > global_threshold_val
        # apply local thresholding
        local = threshold_local(pixels, local_param)  # 21: specific for nucleoli
        out = pixels > local
        # remove large connected areas
        out = obj.remove_large(out, max_size)
        # combine with otsu thresholding to determine background region
        out[bg == 0] = 0
        # two rounds of erosion/dilation and remove_small to clear out background
        out = binary_erosion(out)
        out = obj.remove_small(out, min_size)
        out = binary_dilation(out)

    elif threshold_method == 'local-sg':
        # use otsu thresholding to determine background region
        global_threshold_val = threshold_otsu(pixels)
        bg = pixels > global_threshold_val
        # apply local thresholding
        local = threshold_local(pixels, local_param)  # 21: specific for nucleoli
        out = pixels > local
        # remove large connected areas
        out = obj.remove_large(out, max_size)
        # combine with otsu thresholding to determine background region
        out[bg == 0] = 0
        out = ndimage.binary_fill_holes(out)

    elif threshold_method == 'local-sg1':
        # use otsu thresholding to determine background region
        global_threshold_val = threshold_otsu(pixels)
        global_threshold_val1 = threshold_yen(pixels)
        bg = pixels > global_threshold_val
        bg1 = pixels > global_threshold_val1
        bg2 = pixels > 2500
        # apply local thresholding
        local = threshold_local(pixels, 51)
        out = pixels > local
        # remove large connected areas
        out = obj.remove_large(out, max_size)
        # combine with otsu thresholding to determine background region
        out[bg == 0] = 0
        out[bg1 == 0] = 0
        out[bg2 == 0] = 0

    else:
        out = np.zeros_like(pixels)

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
# hist, hist_centers = histogram(pixels)
# edges = canny(pixels/(hist_centers.argmax() * 0.4), sigma=3)
# filled = ndi.binary_fill_holes(edges)

# seg_500_200 = find_ps_organelles(pixels, 500, 200)
# binary_global = pixels > threshold_otsu(pixels)
# merge = np.zeros_like(pixels)
# merge[seg_500_200 == 2] = 1
# merge |= binary_global
