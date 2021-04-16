import numpy as np
import shared.warning as warn
from skimage.morphology import remove_small_objects, medial_axis, extrema, binary_dilation, dilation
from skimage.measure import label, regionprops
from skimage.filters import sobel
from skimage import segmentation
import math
import random

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for 0-AND-1 NP.ARRAY (BINARY IMAGE)
# ---------------------------------------------------------------------------------------------------

remove_small
    FUNCTION: remove objects smaller than the specified size
    SYNTAX:   remove_small(obj: np.array, min_size=10)

remove_large
    FUNCTION: remove objects larger than the specified size
    SYNTAX:   remove_large(obj: np.array, max_size=1000)

obj_display_in_eccentricity
    FUNCTION: generate color image based on eccentricity value
    SYNTAX:   obj_display_in_eccentricity(label_obj: np.array)

obj_display_in_circularity
    FUNCTION: generate color image based on circularity value
    SYNTAX:   obj_display_in_circularity(label_obj: np.array)

obj_display_in_intensity
    FUNCTION: generate color image based on intensity value
    SYNTAX:   obj_display_in_intensity(label_obj: np.array, pixels: np.array, int_range)

points_in_objects
    FUNCTION: correlate points with objects' labeled numbers from label(obj)
    SYNTAX:   points_in_objects(label_obj: np.array, points_x: list, points_y: list)

object_count
    FUNCTION: count the number of objects in given image
    SYNTAX:   object_count(obj: np.array)

label_watershed
    FUNCTION: separate touching objects based on distance map (similar to imageJ watershed)
    SYNTAX:   label_watershed(obj: np.array)

label_remove_small
    FUNCTION: remove objects smaller than the specified size from labeled image
    SYNTAX:   label_remove_small(label_obj: np.array, min_size: int)

label_remove_large
    FUNCTION: remove objects larger than the specified size from labeled image
    SYNTAX:   label_remove_large(label_obj: np.array, max_size: int)
    
label_resort
    FUNCTION: re-label random labeled image into sequential labeled image
    SYNTAX:   label_resort(label_obj: np.array)

label_fill_holes
    FUNCTION: fill background holes in labeled images
    SYNTAX:   label_fill_holes(label_obj: np.array)
    
select_random_in_label
    FUNCTION: select random positions in label image
    SYNTAX:   select_random_in_label(label_obj: np.array, num: int)
"""


def remove_small(obj: np.array, min_size=10):
    """
    Remove objects smaller than the specified size.

    Expects obj to be an integer image array with labeled objects, and removes objects
    smaller than min_size.

    :param obj: np.array, 0-and-1
    :param min_size: int, optional (default: 10)
                The smallest allowable object size.
    :return: out: nd.array, 0-and-1, same shape and type as input obj

    """
    # Raise type error if not int
    warn.check_img_supported(obj)

    obj_bool = np.array(obj, bool)
    obj_mask = remove_small_objects(obj_bool, min_size)
    out = np.zeros_like(obj)
    out[obj_mask] = 1

    return out


def remove_large(obj: np.array, max_size=1000):
    """
    Remove objects larger than the specified size.

    Expects ar to be an integer image array with labeled objects, and removes objects
    larger than max_size.

    :param obj: np.array, 0-and-1
    :param max_size: int, optional (default: 1000)
                The largest allowable object size.
    :return: out: np.array, 0-and-1, same shape and type as input obj
    """
    # Raise type error if not int
    warn.check_img_supported(obj)

    obj_bool = np.array(obj, bool)
    obj_mask = remove_small_objects(obj_bool, max_size)
    out = obj.copy()
    out[obj_mask] = 0

    return out


def obj_display_in_eccentricity(label_obj: np.array):
    """
    generate color image based on eccentricity value

    :param label_obj: np.array, grey scale labeled image
    :return: out: np.array, 0 - 255 int8 format, same shape and type as input obj

    """
    obj_prop = regionprops(label_obj)
    out = np.zeros_like(label_obj, dtype=float)
    for i in obj_prop:
        ecce = i.eccentricity
        # convert values to 0 - 255 int8 format with (0,1.0) as color dynamic range
        top_ecce = 1.0
        if ecce > top_ecce:
            out[label_obj == i.label] = 255
        else:
            out[label_obj == i.label] = ecce * 255 / top_ecce

    out = out.astype('uint8')

    return out


def obj_display_in_circularity(label_obj: np.array):
    """
    generate color image based on circularity value

    :param label_obj: np.array, grey scale labeled image
    :return: out: np.array, 0 - 255 int8 format, same shape and type as input obj

    """
    obj_prop = regionprops(label_obj)
    out = np.zeros_like(label_obj, dtype=float)
    for i in obj_prop:
        # convert values to 0 - 255 int8 format with (0,1.0) as color dynamic range
        if i.area >= 50:  # only calculate for area>50 organelle
            circ = (4 * math.pi * i.area) / (i.perimeter ** 2)
            top_circ = 1.0
            if circ > top_circ:
                out[label_obj == i.label] = 255
            else:
                out[label_obj == i.label] = circ * 255 / top_circ

    out = out.astype('uint8')

    return out


def obj_display_in_intensity(label_obj: np.array, pixels: np.array, int_range):
    """
    generate color image based on intensity value

    :param label_obj: np.array, grey scale labeled image
    :param pixels: np.array, corresponding grey scale image
    :param int_range: intensity range
    :return: out: np.array, 0 - 255 int8 format, same shape and type as input obj

    """
    obj_prop = regionprops(label_obj, pixels)
    out = np.zeros_like(label_obj, dtype=float)
    for i in obj_prop:
        mean_int = np.log(i.mean_intensity)
        # convert values to 0 - 255 int8 format with (0,1.0) as color dynamic range
        top_int = int_range[1]
        if mean_int > top_int:
            out[label_obj == i.label] = 255
        elif mean_int < int_range[0]:
            out[label_obj == i.label] = 0
        else:
            out[label_obj == i.label] = mean_int * 255 / top_int

    out = out.astype('uint8')

    return out


def points_in_objects(label_obj: np.array, points_x: list, points_y: list):
    """
    Correlate points with objects' labeled numbers from label(obj).

    :param label_obj: np.array, grey scale
    :param points_x: list of points' x coordinates
    :param points_y: list of points' y coordinates
    :return: out: list of correlated objects' label.
    """
    out = []
    if len(points_x) == len(points_y):
        for i in range(len(points_x)):
            out.append(label_obj[points_y[i], points_x[i]])
    else:
        raise ValueError("Length of x: %d and y: %d does not match" % (len(points_x), len(points_y)))

    return out


def object_count(obj: np.array):
    """
    Count the number of objects in given image.

    :param obj: np.array, 0-and-1
    :return: count_obj: number of objects.
    """
    label_obj = label(obj, connectivity=1)
    obj_prop = regionprops(label_obj)
    count_obj = len(obj_prop)

    return count_obj


def label_watershed(obj: np.array, maxima_threshold):
    """
    Separate touching objects based on distance map (similar to imageJ watershed)

    :param obj: np.array, 0-and-1
    :param maxima_threshold: threshold for identify maxima
    :return: seg: np.array, grey scale with different objects labeled with different numbers
    """
    _, dis = medial_axis(obj, return_distance=True)
    maxima = extrema.h_maxima(dis, maxima_threshold)
    # maxima_threshold for Jess data = 1
    # maxima_threshold for Jose 60x data = 10
    # maxima_threshold for Jose 40x data = 20
    maxima_mask = binary_dilation(maxima)
    for i in range(6):
        maxima_mask = binary_dilation(maxima_mask)

    label_maxima = label(maxima_mask, connectivity=2)
    markers = label_maxima.copy()
    markers[obj == 0] = np.amax(label_maxima) + 1
    elevation_map = sobel(obj)
    label_obj = segmentation.watershed(elevation_map, markers)
    label_obj[label_obj == np.amax(label_maxima) + 1] = 0

    return label_obj


def label_remove_small(label_obj: np.array, min_size: int):
    """
    Remove objects smaller than the specified size from labeled image

    :param label_obj: np.array, grey scale labeled image
    :param min_size: int
                The smallest allowable object size
    :return: out: np.array, grey scale labeled image with small objects removed
    """
    out = np.zeros_like(label_obj)
    obj_prop = regionprops(label_obj)
    for i in range(len(obj_prop)):
        if obj_prop[i].area >= min_size:
            out[label_obj == obj_prop[i].label] = obj_prop[i].label

    return out


def label_remove_large(label_obj: np.array, max_size: int):
    """
    Remove objects larger than the specified size from labeled image

    :param label_obj: np.array, grey scale labeled image
    :param max_size: int
                The largest allowable object size
    :return: out: np.array, grey scale labeled image with large objects removed
    """
    out = np.zeros_like(label_obj)
    obj_prop = regionprops(label_obj)
    for i in range(len(obj_prop)):
        if obj_prop[i].area <= max_size:
            out[label_obj == obj_prop[i].label] = obj_prop[i].label

    return out


def label_resort(label_obj: np.array):
    """
    Re-label random labeled image into sequential labeled image.

    :param label_obj: np.array, grey scale labeled image
    :return: label_out: np.array, sorted grey scale labeled image
    """
    count = 1
    label_out = np.zeros_like(label_obj)
    for i in range(np.amax(label_obj)):
        temp = np.zeros_like(label_obj)
        temp[label_obj == i + 1] = 1
        if object_count(temp) > 0:
            label_out[label_obj == i + 1] = count
            count = count + 1

    return label_out


def label_fill_holes(label_obj: np.array):
    """
    Fill background holes in labeled images

    Algorithm:
        when the boundary of a given hole belong to the same indexed object, assign the index to the hole

    :param label_obj: np.array, labeled image
    :return: out: label_image with holes filled
    """
    out = label_obj.copy()

    # identify background
    bg = np.zeros_like(label_obj)
    bg[label_obj == 0] = 1
    bg = segmentation.clear_border(bg)
    label_bg = label(bg)

    # identify border
    label_border = dilation(label_bg)
    for i in range(3):
        label_border = dilation(label_border)
    label_border[bg == 1] = 0

    border_prop = regionprops(label_border, label_obj)
    for i in range(np.amax(label_border)):
        index = border_prop[i].label
        mean_index = border_prop[i].mean_intensity
        diff = mean_index - int(mean_index)
        if diff == 0:
            out[label_bg == index] = int(mean_index)

    return out


def select_random_in_label(label_obj: np.array, num: int):
    """
    Select random positions in label image

    :param label_obj: np.array, label image
    :param num: number of random points to be selected
    :return: x_lst: list, list of x coordinates
             y_lst: list, list of y coordinates
    """
    y_lst = []
    x_lst = []
    for i in range(len(np.unique(label_obj))-1):
        y_idx, x_idx = np.where(label_obj == i+1)
        label_idx = range(len(y_idx))
        rand_idx = random.sample(label_idx, num)
        for j in range(len(rand_idx)):
            y_lst.append(y_idx[rand_idx[j]])
            x_lst.append(x_idx[rand_idx[j]])
    return x_lst, y_lst
