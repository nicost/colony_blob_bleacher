# ---------------------------------------------
# FUNCTIONS for 0-AND-1 NP.ARRAY (BINARY IMAGE)
# ---------------------------------------------

import numpy as np
import shared.warning as warn
import shared.dataframe as dat
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import math


def remove_small(obj: np.array, min_size=10):
    """
    Remove objects smaller than the specified size.

    Expects ar to be an integer image array with labeled objects, and removes objects
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


def filter_eccentricity(obj: np.array, filter_min, filter_max):
    """
    filter objects based on corresponding eccentricity

    :param obj: np.array, 0-and-1
    :param filter_min: minimum allowable value for eccentricity
    :param filter_max: maximum allowable value for eccentricity
    :return: out: np.array, 0-and-1, same shape and type as input obj
    """
    label_obj = label(obj)
    obj_prop = regionprops(label_obj)
    out = obj.copy()
    for i in obj_prop:
        if (i.eccentricity <= filter_min) | (i.eccentricity > filter_max):
            out[label_obj == i.label] = 0

    return out


def group_label_eccentricity(obj: np.array, lst):
    label_obj = label(obj)
    obj_prop = regionprops(label_obj)
    out = obj.copy()
    for i in obj_prop:
        pos = dat.find_pos(i.eccentricity, lst)
        out[label_obj == i.label] = pos+1

    return out


def filter_circularity(obj: np.array, filter_min, filter_max):
    """
        filter objects based on corresponding circularity

        :param obj: np.array, 0-and-1
        :param filter_min: minimum allowable value for circularity
        :param filter_max: maximum allowable value for circularity
        :return: out: np.array, 0-and-1, same shape and type as input obj
        """
    label_obj = label(obj)
    obj_prop = regionprops(label_obj)
    out = obj.copy()
    for i in obj_prop:
        circ = (4 * math.pi * i.area)/(i.perimeter ** 2)
        if (circ <= filter_min) | (circ > filter_max):
            out[label_obj == i.label] = 0

    return out


def group_label_circularity(obj: np.array, lst):
    label_obj = label(obj)
    obj_prop = regionprops(label_obj)
    out = obj.copy()
    for i in obj_prop:
        circ = (4 * math.pi * i.area)/(i.perimeter ** 2)
        pos = dat.find_pos(circ, lst)
        out[label_obj == i.label] = pos+1

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
    label_obj = label(obj)
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
    label_obj = label(obj)
    obj_prop = regionprops(label_obj)
    count_obj = len(obj_prop)

    return count_obj
