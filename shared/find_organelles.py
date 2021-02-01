from skimage.segmentation import clear_border, random_walker
from shared.find_blobs import find_blobs, get_binary_global
from shared.objects import remove_small, remove_large
from skimage.measure import label, regionprops
import shared.warning as warn
import numpy as np
import pandas as pd
import math
from scipy import ndimage
import shared.objects as obj

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for ORGANELLE IDENTIFICATION/ANALYSIS
# ---------------------------------------------------------------------------------------------------

find_organelle
    FUNCTION: find organelle (nucleoli or SG) from a given image
    SYNTAX:   find_organelle(pixels: np.array, global_thresholding='na', extreme_val=500, bg_val=200,
              min_size=5, max_size=1000)

organelle_analysis
    FUNCTION: analyze object properties and return a pd.DataFrame table, for SG/nucleoli
    SYNTAX:   organelle_analysis(pixels: np.array, organelle: np.array, organelle_name: str, pos=0)

find_nuclear
    FUNCTION: detect nuclear in nucleoli stain images
    SYNTAX:   find_nuclear(pixels: np.array)

nuclear_analysis
    FUNCTION: analyze nuclear properties and return a pd.DataFrame table
    SYNTAX:   nuclear_analysis(label_nuclear: np.array, nucleoli_pd: pd.DataFrame, pos=0)

"""


def find_organelle(pixels: np.array, global_thresholding='na', extreme_val=500, bg_val=200,
                   min_size=5, max_size=1000):
    """
    Find organelle (nucleoli or SG) from a given image.

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
                nucleoli default: 'local-nucleoli'
                SG default: 'local-sg'
    :param extreme_val: int, optional (default: 500)
                Used in shared.find_blobs.segment_watershed to find local maxima
    :param bg_val: int, optional (default: 200)
                Used in shared.find_blobs.segment_watershed to define background
                for watershed
    :param min_size: int, optional (default: 5)
                The smallest allowable organelle size.
                nucleoli default: 10
                SG default: 5
    :param max_size: int, optional (default: 1000)
                The largest allowable organelle size.
                nucleoli default: 1000
                SG default: 350
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

    # find organelle
    organelle = find_blobs(pixels, get_binary_global(pixels, global_thresholding, min_size, max_size),
                           extreme_val, bg_val, max_size)

    # Filters:
    # Location filter: remove artifacts connected to image border
    organelle_filtered = clear_border(organelle)

    # Size filter: default [10,1000]
    organelle_filtered = remove_small(organelle_filtered, min_size)
    organelle_filtered = remove_large(organelle_filtered, max_size)

    return organelle_filtered


def organelle_analysis(pixels: np.array, organelle: np.array, organelle_name: str, pos=0):
    """
    Analyze object properties and return a pd.DataFrame table, for SG/nucleoli

    :param pixels: np.array, grey scale image
    :param organelle: np.array, 0-and-1, SG mask
    :param organelle_name: str
    :param pos: position of pixels (for FOV distinction and multi-image stitch)
    :return: organelle_pd: pd.DataFrame describes organelle features, includes 'pos', organelle_name, 'x', 'y',
                'size', 'raw_int', 'circ', 'eccentricity'

                'pos': position of pixels
                 organelle_name: organelle label index
                'x': x coordinate
                'y': y coordinate
                'size': area
                'raw_int': raw mean intensity
                'circ': circularity
                'eccentricity': eccentricity

    """
    label_organelle = label(organelle, connectivity=1)
    organelle_props = regionprops(label_organelle, pixels)

    organelle_areas = [p.area for p in organelle_props]
    organelle_x = [p.centroid[0] for p in organelle_props]
    organelle_y = [p.centroid[1] for p in organelle_props]
    organelle_label = [p.label for p in organelle_props]
    organelle_circ = [(4 * math.pi * p.area) / (p.perimeter ** 2) for p in organelle_props]
    # Eccentricity of the ellipse that has the same second-moments as the region.
    # The eccentricity is the ratio of the focal distance (distance between focal points) over the major
    # axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    organelle_eccentricity = [p.eccentricity for p in organelle_props]
    organelle_mean_int = [p.mean_intensity for p in organelle_props]

    # organelle pd dataset
    organelle_pd = pd.DataFrame({'pos': [pos]*len(organelle_props), organelle_name: organelle_label,
                                 'x': organelle_x, 'y': organelle_y, 'size': organelle_areas,
                                 'raw_int': organelle_mean_int, 'circ': organelle_circ,
                                 'eccentricity': organelle_eccentricity})

    return organelle_pd


def find_nuclear(pixels: np.array):
    """
    Detect nuclear in nucleoli stain images

    :param pixels: np.array, nucleoli stain image
    :return: label_nuclear_sort: np.array, grey scale labeled nuclear image
    """
    # nuclear detection
    markers = np.zeros_like(pixels)
    markers[pixels < 220] = 1
    markers[pixels > 450] = 2
    seg = random_walker(pixels, markers)
    # nuclear binary mask
    nuclear = np.zeros_like(pixels)
    nuclear[seg == 2] = 1
    # fill holes
    nuclear_fill = ndimage.binary_fill_holes(nuclear)
    # separate touching nuclei
    label_nuclear = obj.label_watershed(nuclear_fill)
    # filter out:
    # 1) nuclear size < 1500
    # 2) nuclear touches boundary
    label_nuclear_ft = clear_border(label_nuclear)
    label_nuclear_ft = obj.label_remove_small(label_nuclear_ft, 1500)
    label_nuclear_sort = obj.label_resort(label_nuclear_ft)

    return label_nuclear_sort


def nuclear_analysis(label_nuclear: np.array, nucleoli_pd: pd.DataFrame, pos=0):
    """
    Analyze nuclear properties and return a pd.DataFrame table

    :param label_nuclear: np.array, grey scale labeled nuclear image
    :param nucleoli_pd: pd.DataFrame, nucleoli table
    :param pos: FOV position
    :return: nuclear_pd: pd.DataFrame, nuclear table
    """
    nuclear_prop = regionprops(label_nuclear)
    nuclear_centroid_x = [p.centroid[0] for p in nuclear_prop]
    nuclear_centroid_y = [p.centroid[1] for p in nuclear_prop]
    nuclear_index = [p.label for p in nuclear_prop]

    nuclear_pd = pd.DataFrame({'pos': [pos]*len(nuclear_prop), 'nuclear': nuclear_index,
                               'x': nuclear_centroid_x, 'y': nuclear_centroid_y})

    num_nucleoli = []
    for i in nuclear_pd['nuclear']:
        nucleoli_pd_temp = nucleoli_pd[nucleoli_pd['nuclear'] == i].reset_index(drop=True)
        num_nucleoli.append(len(nucleoli_pd_temp))

    nuclear_pd['num_nucleoli'] = num_nucleoli

    return nuclear_pd
