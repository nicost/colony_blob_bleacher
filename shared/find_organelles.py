# ------------------------------------------
# FUNCTIONS for ORGANELLE DETECTION
# ------------------------------------------

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
import shared.analysis as ana


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


def nucleoli_analysis(pixels: np.array, nucleoli, label_nuclear, pos=0):
    """
    Analyze nucleoli properties and return a pd.DataFrame table

    :param: pixels: np.array, grey scale image
    :param: nucleoli: np.array, 0-and-1, nucleoli mask
    :param: pos: position of pixels (for FOV distinction and multi-image stitch)
    :return: nucleoli_pd: pd.DataFrame describes nucleoli features including nucleoli number, size, x, y
        coordinates of the centroid
    """
    # get nucleoli properties
    nucleoli_label = label(nucleoli, connectivity=1)
    nucleoli_prop = regionprops(nucleoli_label)
    nucleoli_prop_int = regionprops(nucleoli_label, pixels)
    # export nucleoli information about area, centroid, label, circularity
    nucleoli_areas = [p.area for p in nucleoli_prop]
    nucleoli_centroid_x = [p.centroid[0] for p in nucleoli_prop]
    nucleoli_centroid_y = [p.centroid[1] for p in nucleoli_prop]
    nucleoli_index = [p.label for p in nucleoli_prop]
    nucleoli_circ = [(4 * math.pi * p.area) / (p.perimeter ** 2) for p in nucleoli_prop]
    # measure mean_int and export background corrected value
    # for time 0, does not require photobleaching correction
    nucleoli_mean_int = [p.mean_intensity for p in nucleoli_prop_int]
    bg_int = ana.get_bg_int([pixels])[0]
    nucleoli_mean_int_cor = [x - bg_int for x in nucleoli_mean_int]

    # nucleoli pd dataset
    nucleoli_pd = pd.DataFrame({'pos': [pos]*len(nucleoli_prop), 'nucleoli': nucleoli_index, 'size': nucleoli_areas,
                                'centroid_x': nucleoli_centroid_x, 'centroid_y': nucleoli_centroid_y,
                                'mean_int': nucleoli_mean_int_cor, 'circ': nucleoli_circ})

    # link nucleoli with corresponding nuclear
    round_x = [round(num) for num in nucleoli_centroid_x]
    round_y = [round(num) for num in nucleoli_centroid_y]
    nucleoli_pd['nuclear'] = obj.points_in_objects(label_nuclear, round_y, round_x)

    return nucleoli_pd


def find_nuclear(pixels: np.array):
    """
    Detect nuclear in nucleoli stain images

    :param pixels: np.array, nucleoli stain image
    :return: label_nuclear_sort: np.array, grey scale with each nuclear labeled
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


def nuclear_analysis(label_nuclear: np.array, nucleoli_pd, pos):
    nuclear_prop = regionprops(label_nuclear)
    nuclear_centroid_x = [p.centroid[0] for p in nuclear_prop]
    nuclear_centroid_y = [p.centroid[1] for p in nuclear_prop]
    nuclear_index = [p.label for p in nuclear_prop]

    nuclear_pd = pd.DataFrame({'pos': [pos]*len(nuclear_prop), 'nuclear': nuclear_index,
                               'centroid_x': nuclear_centroid_x, 'centroid_y': nuclear_centroid_y})

    num_nucleoli = []
    total_nucleoli_int = []
    for i in nuclear_pd['nuclear']:
        nucleoli_pd_temp = nucleoli_pd[nucleoli_pd['nuclear'] == i].reset_index(drop=True)
        num_nucleoli.append(len(nucleoli_pd_temp))
        total_nucleoli_int_temp = 0
        if len(nucleoli_pd_temp) != 0:
            for j in range(len(nucleoli_pd_temp)):
                total_nucleoli_int_temp += nucleoli_pd_temp['mean_int'][j] * nucleoli_pd_temp['size'][j]
        total_nucleoli_int.append(total_nucleoli_int_temp)

    nuclear_pd['num_nucleoli'] = num_nucleoli
    nuclear_pd['total_nucleoli_int '] = total_nucleoli_int

    return nuclear_pd

