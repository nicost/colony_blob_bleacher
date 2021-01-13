# ---------------------------------------------
# FUNCTIONS for 0-AND-1 NP.ARRAY (BINARY IMAGE)
# ---------------------------------------------

import numpy as np
import shared.warning as warn
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


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

        :param obj: nd.array, 0-and-1
        :param max_size: int, optional (default: 1000)
                    The largest allowable object size.
        :return: out: nd.array, 0-and-1, same shape and type as input obj
        """
    # Raise type error if not int
    warn.check_img_supported(obj)

    obj_bool = np.array(obj, bool)
    obj_mask = remove_small_objects(obj_bool, max_size)
    out = obj.copy()
    out[obj_mask] = 0

    return out


def get_centroid(obj: np.array):
    """
    Get list of centroids' coordinates of given image with objects.

    :param obj: np.array, 0-and-1
    :return: obj_centroid_x: list of x coordinates
             obj_centroid_y: list of y coordinates
    """
    obj_centroid_x = []
    obj_centroid_y = []
    label_obj = label(obj)
    obj_prop = regionprops(label_obj)
    for i in range(len(obj_prop)):
        obj_centroid_x.append(obj_prop[i].centroid[0])
        obj_centroid_y.append(obj_prop[i].centroid[1])

    return obj_centroid_x, obj_centroid_y


def get_size(obj: np.array):
    """
    Get list of 2D areas of given image with objects.

    :param obj: np.array, 0-and-1
    :return: obj_areas: list of areas
    """
    label_obj = label(obj)
    obj_areas = np.bincount(label_obj.ravel())[1:]

    return obj_areas


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
