from skimage.segmentation import clear_border, random_walker, watershed
from shared.find_blobs import find_blobs, get_binary_global
from shared.objects import remove_small, remove_large
from skimage.measure import label, regionprops
from skimage.filters import threshold_local, sobel, threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion
import shared.analysis as ana
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

find_nuclear_nucleoli
    FUNCTION: detect nuclear in nucleoli stain images
    SYNTAX:   find_nuclear_nucleoli(pixels: np.array)

nuclear_analysis
    FUNCTION: analyze nuclear properties and return a pd.DataFrame table
    SYNTAX:   nuclear_analysis(label_nuclear: np.array, nucleoli_pd: pd.DataFrame, pos=0)

find_cell
    FUNCTION: cell segmentation based on membrane/nuclear staining
    SYNTAX:   find_cell(bg_pixels: np.array, mem_pixels: np.array, nuclear_pixels: np.array, with_boundary='N',
              nuclear_local_threshold_size=701, min_size_nuclear=15000, max_size_nuclear=130000,
              min_size_cell=50000, max_size_cell=300000)

find_nuclear
    FUNCTION: detect nuclear in nuclear stain image
    SYNTAX:   find_nuclear(pixels: np.array, local_thresholding_size: int, min_size: int, max_size: int)

"""


def find_organelle(pixels: np.array, global_thresholding='na', extreme_val=500, bg_val=200,
                   min_size=5, max_size=1000, local_param=21):
    """
    Find organelle (nucleoli or SG) from a given image.

    Expects pixels to be an array, and finds nucleoli objects using watershed by
    flooding approach with indicated global thresholding methods (supports 'na',
    'otsu', 'yen', 'local-nucleoli' and 'local-sg', 'local-sg1').  Founded organelles
    are filtered by default location filter (filter out organelles located at the
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
    :param local_param: int, optional (default: 21)
                parameter for local thresholding
    :returns nucleoli_filtered: 0-and-1 np.array, same shape and type as input img
                Binary array with found nucleoli labeled with 1.
    """
    # Raise type error if not int
    warn.check_img_supported(pixels)

    # Check global thresholding options
    # Raise value error if not 'na', 'otsu' or 'yen'
    check_lst = ['na', 'otsu', 'yen', 'local-nucleoli', 'local-sg', 'local-sg1']
    if global_thresholding not in check_lst:
        raise ValueError("global thresholding method only accepts %s. Got %s" % (check_lst, global_thresholding))

    # find organelle
    organelle = find_blobs(pixels, get_binary_global(pixels, global_thresholding, min_size, max_size, local_param),
                           extreme_val, bg_val, max_size)

    # Filters:
    # Location filter: remove artifacts connected to image border
    organelle_filtered = clear_border(organelle)

    # Size filter: default [10,1000]
    organelle_filtered = remove_small(organelle_filtered, min_size)
    organelle_filtered = remove_large(organelle_filtered, max_size)

    return organelle, organelle_filtered


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


def find_nuclear_nucleoli(pixels: np.array):
    """
    Detect nuclear in nucleoli stain images

    :param pixels: np.array, nucleoli stain image
    :return: label_nuclear_sort: np.array, grey scale labeled nuclear image
    """
    bg_int = ana.get_bg_int([pixels])[0]
    # nuclear detection
    markers = np.zeros_like(pixels)
    markers[pixels < (1.7 * bg_int)] = 1
    markers[pixels > (3 * bg_int)] = 2
    seg = random_walker(pixels, markers)
    # nuclear binary mask
    nuclear = np.zeros_like(pixels)
    nuclear[seg == 2] = 1
    # fill holes
    nuclear_fill = ndimage.binary_fill_holes(nuclear)
    # separate touching nuclei
    label_nuclear = obj.label_watershed(nuclear_fill, 1)
    # filter out:
    # 1) nuclear size < 1500
    label_nuclear_ft = obj.label_remove_small(label_nuclear, 1500)
    label_nuclear_sort = obj.label_resort(label_nuclear_ft)
    # 2) nuclear touches boundary
    label_nuclear_exclude_boundary = clear_border(label_nuclear_sort)
    label_nuclear_exclude_boundary_sort = obj.label_resort(label_nuclear_exclude_boundary)

    return label_nuclear_sort, label_nuclear_exclude_boundary_sort


def nuclear_analysis(label_nuclear: np.array, organelle_pd: pd.DataFrame, pos=0):
    """
    Analyze nuclear properties and return a pd.DataFrame table

    :param label_nuclear: np.array, grey scale labeled nuclear image
    :param organelle_pd: pd.DataFrame, nucleoli table
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
        nucleoli_pd_temp = organelle_pd[organelle_pd['nuclear'] == i].reset_index(drop=True)
        num_nucleoli.append(len(nucleoli_pd_temp))

    nuclear_pd['num_nucleoli'] = num_nucleoli

    return nuclear_pd


def find_cell(bg_pixels: np.array, mem_pixels: np.array, nuclear_pixels: np.array,
              nuclear_local_threshold_size=701, min_size_nuclear=15000, max_size_nuclear=130000,
              min_size_cell=50000, max_size_cell=300000, low_stain_mode='off', watershed_mode='on'):
    """
    Cell segmentation based on membrane/nuclear staining

    :param bg_pixels: np.array, image for true background identification
    :param mem_pixels: np.array, membrane staining image
    :param nuclear_pixels: np.array, nuclear staining image
    :param nuclear_local_threshold_size: running size for nuclear local thresholding (default = 701)
    :param min_size_nuclear: minimum allowable nuclear size (default = 15000)
    :param max_size_nuclear: maximum allowable nuclear size (default = 130000)
    :param min_size_cell: minimum allowable cell size (default = 50000)
    :param max_size_cell: maximum allowable cell size (default = 300000)
    :param low_stain_mode: 'on' or 'off', if use background to capture more nuclei
    :param watershed_mode: 'on' or 'off', if use watershed to cut neighbour nuclei
    :return: seg: np.array, label image of segmented cells
             seg_ft: np.array, label image of segmented cells excluding cells touching border
    """
    # identify true background
    bg = ana.find_background(bg_pixels)

    # identify nuclear
    label_nuclear = find_nuclear(nuclear_pixels, nuclear_local_threshold_size, min_size_nuclear, max_size_nuclear,
                                 low_stain_mode, watershed_mode)[1]

    """# identify aggregates in membrane staining
    bg_mem = ana.get_bg_int([mem_pixels])[0]
    aggregates = mem_pixels > threshold_otsu(mem_pixels)
    for i in range(3):
        aggregates = binary_erosion(aggregates)
    for i in range(3):
        aggregates = binary_dilation(aggregates)
    aggregates = obj.remove_small(aggregates, 1500)
    label_aggregates = label(aggregates)
    aggregates_prop = regionprops(label_aggregates)
    for i in range(len(aggregates_prop)):
        if (aggregates_prop[i].eccentricity > 0.95) | (aggregates_prop[i].convex_area > (2*aggregates_prop[i].area)):
            aggregates[label_aggregates == aggregates_prop[i].label] = bg_mem

    # identify cell boundary
    mem_pixels_wo_aggregates = mem_pixels.copy()
    mem_pixels_wo_aggregates[aggregates == 1] = 0
    elevation_map = sobel(mem_pixels_wo_aggregates)"""

    # identify cell boundary
    elevation_map = sobel(mem_pixels)

    # generate markers
    markers = label_nuclear.copy()
    markers[(bg == 1) & (label_nuclear == 0)] = np.amax(label_nuclear) + 1

    # segment cells
    seg = watershed(elevation_map, markers)
    seg[seg == np.amax(label_nuclear) + 1] = 0

    # fill holes
    seg = obj.label_fill_holes(seg)

    # filter cells based on size
    seg = obj.label_resort(obj.label_remove_small(seg, min_size_cell))
    seg = obj.label_resort(obj.label_remove_large(seg, max_size_cell))
    seg_ft = obj.label_resort(clear_border(seg))

    return seg, seg_ft


def find_nuclear(pixels: np.array, local_thresholding_size: int, min_size: int, max_size: int, low_stain_mode: str,
                 watershed_mode: str):
    """
    Detect nuclear in nuclear stain image

    :param pixels: np.array, nuclear stain image
    :param local_thresholding_size: int, running size for local thresholding, varies for different magnification etc.
    :param min_size: minimum allowable nuclear size
    :param max_size: maximum allowable nuclear size
    :param low_stain_mode: 'on' or 'off', if use background to capture more nuclei
    :param watershed_mode: 'on' or 'off', if use watershed to cut neighbour nuclei
    :return: nuclear: np.array, 0-and-1, nuclear mask with nuclei identified
             label_nuclear: np.array, labeled image of nuclear mask
    """
    # nuclear identification using local thresholding
    local = threshold_local(pixels, local_thresholding_size)
    # for Jose data under 60x objective, local_thresholding_size = 701
    nuclei_local = pixels > local
    for i in range(2):
        nuclei_local = binary_erosion(nuclei_local)
    for i in range(2):
        nuclei_local = binary_dilation(nuclei_local)
    nuclei_local = ndimage.binary_fill_holes(nuclei_local)
    nuclei_local = remove_small(nuclei_local, min_size)
    nuclei_local = remove_large(nuclei_local, max_size)
    # for Jose data under 60x objective, min_size = 15000

    if low_stain_mode == 'on':
        # nuclear identification using background intensity
        bg_int = ana.get_bg_int([pixels])[0]
        nuclei_bg = pixels > 1.05*bg_int
        nuclei_bg = ndimage.binary_fill_holes(nuclei_bg)
        nuclei_bg = binary_erosion(nuclei_bg)
        nuclei_bg = binary_dilation(nuclei_bg)
        nuclei_bg = remove_small(nuclei_bg, min_size)
        nuclei_bg = remove_large(nuclei_bg, max_size)

        # nuclear mask
        nuclear = np.zeros_like(pixels)
        nuclear[nuclei_local == 1] = 1
        nuclear[nuclei_bg == 1] = 1
    else:
        nuclear = nuclei_local

    if watershed_mode == 'off':
        label_nuclear = label(nuclear)
    else:
        label_nuclear = obj.label_resort(obj.label_watershed(nuclear, 20))

    return nuclear, label_nuclear
