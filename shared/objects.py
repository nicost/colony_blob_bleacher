# ---------------------------------------------
# FUNCTIONS for 0-AND-1 NP.ARRAY (BINARY IMAGE)
# ---------------------------------------------

import numpy as np
import shared.warning as warn
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import math


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


def obj_display_in_eccentricity(obj: np.array):
    """
    generate color image based on eccentricity value

    :param obj: np.array, 0-and-1
    :return: out: np.array, 0 - 255 int8 format, same shape and type as input obj
    """
    label_obj = label(obj, connectivity=1)
    obj_prop = regionprops(label_obj)
    out = np.zeros_like(obj, dtype=float)
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


def obj_display_in_circularity(obj: np.array):
    """
    generate color image based on circularity value

    :param obj: np.array, 0-and-1
    :return: out: np.array, 0 - 255 int8 format, same shape and type as input obj
    """
    label_obj = label(obj, connectivity=1)
    obj_prop = regionprops(label_obj)
    out = np.zeros_like(obj, dtype=float)
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


def obj_display_in_intensity(obj: np.array, pixels: np.array, int_range):
    """
    generate color image based on intensity value

    :param obj: np.array, 0-and-1
    :param pixels: np.array, corresponding grey scale image
    :param int_range: intensity range
    :return: out: np.array, 0 - 255 int8 format, same shape and type as input obj
    """
    label_obj = label(obj, connectivity=1)
    obj_prop = regionprops(label_obj, pixels)
    out = np.zeros_like(obj, dtype=float)
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


def points_in_objects(obj: np.array, points_x: list, points_y: list):
    """
    Correlate points with objects' labeled numbers from label(obj).

    :param obj: np.array, 0-and-1
    :param points_x: list of points' x coordinates
    :param points_y: list of points' y coordinates
    :return: out: list of correlated objects' numbers.
    """
    out = []
    label_obj = label(obj, connectivity=1)
    if len(points_x) == len(points_y):
        for i in range(len(points_x)):
            out.append(label_obj[points_y[i], points_x[i]] - 1)
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
