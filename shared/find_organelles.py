# ------------------------------------------
# FUNCTIONS for ORGANELLE DETECTION
# ------------------------------------------

from skimage.segmentation import clear_border

from shared.find_blobs import find_blobs, get_binary_global
from shared.objects import remove_small, remove_large
from skimage.measure import label, regionprops
import shared.warning as warn
import numpy as np
import pandas as pd
import math


def find_organelle(pixels: np.array, global_thresholding='na', extreme_val=500, bg_val=200,
                   min_size=5, max_size=1000):
    """
    Find organelle(nucleoli or SG) from a given image.

    Expects pixels to be an array, and finds nucleoli objects using watershed by
    flooding approach with indicated global thresholding methods (supports 'na',
    'otsu', 'yen', 'local-nucleoli' and 'local-sg').  Founded organelles are 
    filtered by default location filter (filter out organelles located at the 
    boundary of the image) and size filter.

    :param pixels: np.array (non-negative int type)
                Image pixel
    :param global_thresholding: only accepts 'na', 'otsu', 'yen', 'local-nucleoli'
                or 'local-sg'
                optional (default: 'na')
                Whether or not ('na') to apply global thresholding method and
                which method to apply
    :param extreme_val: int, optional (default: 500)
                Used in shared.find_blobs.segment_watershed to find local maxima
    :param bg_val: int, optional (default: 200)
                Used in shared.find_blobs.segment_watershed to define background
                for watershed
    :param min_size: int, optional (default: 5)
                The smallest allowable organelle size.
    :param max_size: int, optional (default: 1000)
                The largest allowable organelle size.
    :returns nucleoli_filtered: 0-and-1 np.array, same shape and type as input img
                Binary array with found nucleoli labeled with 1.
    """
    # Raise type error if not int
    warn.check_img_supported(pixels)

    # Check global thresholding options
    # Raise value error if not 'na', 'otsu' or 'yen'
    check_lst = ['na', 'otsu', 'yen', 'local-nucleoli', 'local-sg']
    if global_thresholding not in check_lst:
        raise ValueError("global thresholding method only accepts %s. Got %s" % (check_lst, global_thresholding))

    # find nucleoli
    organelle = find_blobs(pixels, get_binary_global(pixels, global_thresholding, min_size, max_size),
                           extreme_val, bg_val, max_size)

    # Nucleoli filters:
    # Location filter: remove artifacts connected to image border
    organelle_filtered = clear_border(organelle)
    # Size filter: default [10,1000]
    organelle_filtered = remove_small(organelle_filtered, min_size)
    organelle_filtered = remove_large(organelle_filtered, max_size)

    return organelle_filtered


def sg_analysis(pixels: np.array, sg, pos=0):
    """
    Analyze SG properties and return a pd.DataFrame table

    :param pixels: np.array, grey scale image
    :param sg: np.array, 0-and-1, SG mask
    :param pos: position of pixels (for FOV distinction and multi-image stitch)
    :return: sg_pd: pd.DataFrame describes SG features including position in uManager dataset, SG number,
        x, y coordinate, size, mean intensity, circularity and eccentricity
    """
    label_sg = label(sg, connectivity=1)
    sg_prop = regionprops(label_sg)
    sg_prop_int = regionprops(label_sg, pixels)
    sg_areas = [p.area for p in sg_prop]
    sg_x = [p.centroid[0] for p in sg_prop]
    sg_y = [p.centroid[1] for p in sg_prop]
    sg_mean_int = [p.mean_intensity for p in sg_prop_int]
    sg_label = [p.label for p in sg_prop]
    sg_circ = [(4 * math.pi * p.area) / (p.perimeter ** 2) for p in sg_prop]
    # Eccentricity of the ellipse that has the same second-moments as the region.
    # The eccentricity is the ratio of the focal distance (distance between focal points) over the major
    # axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    sg_eccentricity = [p.eccentricity for p in sg_prop]

    # sg pd dataset
    sg_pd = pd.DataFrame({'pos': [pos]*len(sg_prop), 'sg': sg_label, 'x': sg_x, 'y': sg_y, 'size': sg_areas,
                          'int': sg_mean_int, 'circ': sg_circ, 'eccentricity': sg_eccentricity})

    return sg_pd


def nucleoli_analysis(pixels: np.array, nucleoli, pos=0):
    """
    Analyze nucleoli properties and return a pd.DataFrame table

    :param: pixels: np.array, grey scale image
    :param: nucleoli: np.array, 0-and-1, nucleoli mask
    :param: pos: position of pixels (for FOV distinction and multi-image stitch)
    :return: nucleoli_pd: pd.DataFrame describes nucleoli features including nucleoli number, size, x, y
        coordinates of the centroid
    """
    # get the size and centroid of each nucleoli
    nucleoli_label = label(nucleoli, connectivity=1)
    nucleoli_prop = regionprops(nucleoli_label)
    nucleoli_prop_int = regionprops(nucleoli_label, pixels)
    nucleoli_areas = [p.area for p in nucleoli_prop]
    nucleoli_centroid_x = [p.centroid[0] for p in nucleoli_prop]
    nucleoli_centroid_y = [p.centroid[1] for p in nucleoli_prop]
    nucleoli_label = [p.label for p in nucleoli_prop]
    nucleoli_mean_int = [p.mean_intensity for p in nucleoli_prop_int]
    nucleoli_circ = [(4 * math.pi * p.area) / (p.perimeter ** 2) for p in nucleoli_prop]

    # nucleoli pd dataset
    nucleoli_pd = pd.DataFrame({'pos': [pos]*len(nucleoli_prop), 'nucleoli': nucleoli_label, 'size': nucleoli_areas,
                                'centroid_x': nucleoli_centroid_x, 'centroid_y': nucleoli_centroid_y,
                                'mean_int': nucleoli_mean_int, 'circ': nucleoli_circ})

    return nucleoli_pd
