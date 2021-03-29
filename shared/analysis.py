import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import rank, threshold_triangle
from skimage.morphology import disk, opening, dilation, binary_dilation, binary_erosion
from skimage.measure import label, regionprops_table, regionprops
import shared.dataframe as dat
import shared.objects as obj
import pandas as pd
from scipy import ndimage

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for NON 0-AND-1 NP.ARRAY (IMAGE)
# ---------------------------------------------------------------------------------------------------

bleach_location
    FUNCTION:
    SYNTAX:   bleach_location(pre_pixels: np.array, post_pixels: np.array, expected_position=DEFAULT, 
              half_roi_size=DEFAULT)
    
central_pixel_without_cells
    FUNCTION: 
    SYNTAX:   central_pixel_without_cells(pixels: np.array)

analysis_mask
    FUNCTION: generates a dilated mask from given points
    SYNTAX:   analysis_mask(x: list, y: list, pixels_same_size: np.array, num_dilation=3)

get_intensity
    FUNCTION: measure mean intensity time series for all given objects
    SYNTAX:   get_intensity(obj: np.array, pixels_tseries: list)
    
get_bg_int
    FUNCTION: measure background intensities from a given movie
    SYNTAX:   get_bg_int(pixels_tseries: list)
        
bg_correction
    FUNCTION: background correction of time series intensities for multiple points
    SYNTAX:   bg_correction(int_tseries_multiple_points: list, bg_int_tseries: list)
     
get_pb_factor
    FUNCTION: measure photobleaching factor from given time series intensities of multiple 
              control points
    SYNTAX:   get_pb_factor(int_tseries_ctrl_spots: list)
    
pb_correction
    FUNCTION: photobleaching correction of time series intensities
    SYNTAX:   pb_correction(int_tseries_multiple_points: list, pb_factor: list)

pix_stitch_same_row
    FUNCTION: stitch same size images in a row as a single image (empty space filled with same size 
              black images)
    SYNTAX:   pix_stitch_same_row(pixels_pd: pd.DataFrame, num_col: int)

pix_stitch
    FUNCTION: stitch same size images into a single image based on provided location (empty space 
              filled with same size black images)
    SYNTAX:   pix_stitch(pixels_pd: pd.DataFrame, num_col: int, num_row: int)
    
find_background
    FUNCTION: detect true background from given image
    SYNTAX:   find_background(pixels: np.array)
"""

DEFAULT = object()


def bleach_location(pre_pixels: np.array,
                    post_pixels: np.array,
                    expected_position=DEFAULT,
                    half_roi_size=DEFAULT):
    """
    Finds the location of a bright spot in the post_pixels image
    The pre_pixel image will be subtracted from the pos_pixel image (after adding
    an offset to account for noise), mean filtered, and the position of the maximum in the image will
    be returned.  To speed up finding the location provide the estimated location and an roi
    within which the real maximum should be located
    :param pre_pixels:
    :param post_pixels:
    :param expected_position: tuple[int, int] with expected position
    :param half_roi_size: tuple[int, int] area around expected_position to be searched for spot
    :return:
    """
    # assume pre and post are the same size
    ep_rc = expected_position
    if expected_position == DEFAULT or half_roi_size == DEFAULT:
        pre_roi = pre_pixels
        post_roi = post_pixels
    else:
        if (expected_position[0] - half_roi_size[0] < 0) or \
                (expected_position[1] - half_roi_size[1] < 0) or \
                (expected_position[0] + half_roi_size[0] > post_pixels.shape[0]) or \
                (expected_position[1] + half_roi_size[1] > post_pixels.shape[1]):
            pre_roi = pre_pixels
            post_roi = post_pixels
        else:
            cc = post_pixels.shape[1] - expected_position[1]
            ep_rc = [expected_position[0], cc]
            pre_roi = pre_pixels[
                      ep_rc[0] - half_roi_size[0]:ep_rc[0] + half_roi_size[0],
                      ep_rc[1] - half_roi_size[1]:ep_rc[1] + half_roi_size[1]]
            post_roi = post_pixels[
                       ep_rc[0] - half_roi_size[0]:ep_rc[0] + half_roi_size[0],
                       ep_rc[1] - half_roi_size[1]:ep_rc[1] + half_roi_size[1]]
    subtracted = post_roi + 100 - pre_roi
    selem = disk(2)
    subtracted_mean = rank.mean(subtracted, selem=selem)
    peaks_rc_roi = peak_local_max(subtracted_mean, min_distance=20, threshold_rel=0.6, num_peaks=1, indices=True)
    peaks_rc = peaks_rc_roi + [ep_rc[0] - half_roi_size[0], ep_rc[1] - half_roi_size[1]]
    if len(peaks_rc) < 1:
        return expected_position
    peaks = [peaks_rc[0][0], peaks_rc[0][1]]
    return peaks


def central_pixel_without_cells(pixels: np.array):
    """
    Find the location closest to the center that does not have an object (cell) near by
    :param pixels: Input image as n.array
    :return: location as a tuple, or False if not found
    """
    s2 = disk(2)
    s15 = disk(15)
    binary = pixels > threshold_triangle(pixels)
    opened = opening(binary, s2)
    dilated = dilation(opened, s15)

    # for FOV full of cells, filled up holes within cells (might only be needed for SG staining)
    dilated = ndimage.binary_fill_holes(dilated)

    location = [pixels.shape[0] // 2, pixels.shape[1] // 2]
    center = [pixels.shape[0] // 2, pixels.shape[1] // 2]
    distance = 1
    test = dilated[center[0], center[1]]
    while test and distance < pixels.shape[0] // 2:
        subarray = dilated[center[0] - distance: center[0] + distance, center[1] - distance: center[1] + distance]
        if False in subarray:
            rows, cols = np.where(subarray == False)
            location = [rows[0] + center[0] - distance, cols[0] + center[1] - distance]
            test = False
        else:
            distance += 1

    if not test:
        return location

    return False


def analysis_mask(x: list, y: list, pixels_same_size: np.array, num_dilation=3):
    """
    Generates a dilated mask from given points

    Algorithm description:
    Performs serial dilation (total rounds = num_dilation) from the provided x,y coordinates and
    exports the dilated image as a binary mask using the image size provided from pixels_same_size

    Usage examples:
    1) used to generate bleach spots mask for FRAP analysis

    :param x: list
                list of x coordinates of the points, requires len(x) = len(y), otherwise raises
                ValueError
    :param y: list
                list of y coordinates of the points, requires len(x) = len(y), otherwise raises
                ValueError
    :param pixels_same_size: nd.array
                requires to be the same size as the expected output
                img_same_size will only be used to get the width/height information to determine
                output size, but not for any analysis purposes
    :param num_dilation: int, optional (default: 3)
                number of dilation applied from the coordinates
                determines the size of the generated mask for each point
                default number was determined for FRAP analysis
    :return: out: nd.array, binary image
                same size as img_same_size
                1: points regions
                0: background

    """
    out = np.zeros_like(pixels_same_size)

    if len(x) == len(y):
        for i in range(len(x)):
            out[x[i], y[i]] = 1
        for i in range(num_dilation):
            out = binary_dilation(out)
    else:
        raise ValueError("Length of x: %d and y: %d does not match" % (len(x), len(y)))

    return out


def get_intensity(label_obj: np.array, pixels_tseries: list):
    """
    Measure mean intensity time series for all given objects

    Usage examples:
    1) measure bleach spots/ctrl spots intensity series

    :param label_obj: np.array, 0-and-1 object mask
    :param pixels_tseries: list, pixels time series
                e.g. [pixels_t0, pixels_t1, pixels_t2, ...]
    :return: obj_int_tseries: list
                list of intensity time series
    """

    max_t = len(pixels_tseries)
    obj_int_tseries = [[] for _ in range(len(np.unique(label_obj))-1)]

    for t in range(0, max_t):
        # measure mean intensity for objects
        obj_props = regionprops(label_obj, pixels_tseries[t])
        for i in range(len(obj_props)):
            obj_int_tseries[i].append(obj_props[i].mean_intensity)

    return obj_int_tseries


def get_bg_int(pixels_tseries: list):
    """
    Measure background intensities from a given movie

    Algorithm description:
    For each time frame from the movie pixels_tseries, generate binary image of regions that
    pixel intensity < 300, remove regions whose area < 50 and return the mean intensity of the
    largest area as background intensity for this given time frame.

    Usage examples:
    1) used for background correction

    Note:
    1) function was originally designed for getting background series from a movie, but can also
        be applied to get background intensity from a single image. If so, please do:
        bg_int = get_bg_int([pixels])[0]

    :param pixels_tseries: list
                time series of np.array (movie), e.g. [pixel1, pixel2, ...]
                pixels_tseries[i]: np.array, pixels at time frame i
    :return: bg_int_tseries: list, 0 indicates background intensity detection failure
                list of background intensities, e.g. [bg_1, bg_2, ...]
                t_bg_int[i]: float, bg_int at frame i

    """
    bg_int_tseries = []
    for i in range(len(pixels_tseries)):
        # get regions whose pixel intensity < 300
        bg = np.zeros_like(pixels_tseries[i])
        bg[pixels_tseries[i] < 300] = 1
        # smooth region
        bg = binary_dilation(bg)
        bg = binary_dilation(bg)
        bg = binary_erosion(bg)
        bg = binary_erosion(bg)
        # remove regions < 50
        bg = obj.remove_small(bg, 50)
        # measure bg object properties
        bg_props = regionprops_table(label(bg), pixels_tseries[i], properties=['area', 'mean_intensity'])
        bg_prop_pd = pd.DataFrame(bg_props)

        # high intensity image, do not have pixel intensity of any connected 50 pixels < 300
        if len(bg_prop_pd) == 0:
            # set bg_int as 0 to get the script going without interruption
            # 0 should not affect bg intensity curve fitting
            bg_int_tseries.append(0)
        elif len(bg_prop_pd) == 1:
            bg_int_tseries.append(bg_prop_pd['mean_intensity'][0])
        else:
            # find the mean_intensity of the largest area
            max_area = 0
            bg_int_temp = 0
            for j in range(len(bg_prop_pd)):
                if bg_prop_pd['area'][j] > max_area:
                    max_area = bg_prop_pd['area'][j]
                    bg_int_temp = bg_prop_pd['mean_intensity'][j]
            bg_int_tseries.append(bg_int_temp)

    return bg_int_tseries


def bg_correction(int_tseries_multiple_points: list, bg_int_tseries_multiple_points: list):
    """
    Background correction of time series intensities for multiple points

    Algorithm description:
    corrected intensity = measured intensity - background intensity

    Usage examples:
    1) used for background correction

    :param int_tseries_multiple_points: list
                list of time series intensities of multiple points
                e.g. [[point1_t0, point1_t1 ...], [point2_t0, point2_t1, ...], ...]
                int_tseries_multiple_points[i]: list, intensity series of point i
    :param bg_int_tseries_multiple_points: list
                list of background intensities of multiple points
                e.g. [[bg_1_t0, bg_1_t1 ...], [bg_2_t0, bg_2_t1 ...], ...]
                t_bg_int[i]: list, background intensity series for point i
    :return: out: list
                background corrected time series intensities of multiple points
                corrected intensities <0 were set to 0

    """
    out = []
    # for each point
    num_points = len(int_tseries_multiple_points)
    for i in range(num_points):
        # calculate corrected intensities
        int_tseries_cor = dat.list_subtraction(int_tseries_multiple_points[i], bg_int_tseries_multiple_points[i])
        # set any negative value to 0
        int_tseries_cor = [0 if i < 0 else i for i in int_tseries_cor]
        # store corrected intensity
        out.append(int_tseries_cor)

    return out


def get_pb_factor(int_tseries_ctrl_spots: list):
    """
    Measure photobleaching factor from given time series intensities of multiple control points

    Algorithm description:
    pb_factor[i] = mean(ctrl_t[i]/ctrl_t[0])

    Usage examples:
    1) used for photobleaching correction

    :param int_tseries_ctrl_spots: list
                list of time series intensities of ctrl points
                e.g. [[ctrl1_t0, ctrl1_t1 ...], [ctrl2_t0, ctrl2_t1, ...], ...]
                int_tseries_ctrl_points[i]: list, intensity series of ctrl point i
    :return: pb_factor: list
                list of photobleaching factors, e.g. [pb_factor_t0, pb_factor_t1, ...]
                pb_factor[i]: float, (0,1]

    """
    pb_factor_tseries = []
    # for each time frame
    for t in range(len(int_tseries_ctrl_spots[0])):
        pb_ratio = []
        # calculate pb_ratio ctrl_t(t)/ctrl_t(0) for each single ctrl spots
        for i in range(len(int_tseries_ctrl_spots)):
            pb_ratio.append(int_tseries_ctrl_spots[i][t] / (int_tseries_ctrl_spots[i][0] + 0.0001))
        # calculate pb_factor
        pb_factor = np.mean(pb_ratio)
        pb_factor_tseries.append(pb_factor)

    return pb_factor_tseries


def pb_correction(int_tseries_multiple_points: list, pb_factor: list):
    """
    Photobleaching correction of time series intensities

    Algorithm description:
    corrected intensity = measured intensity / pb_factor

    Usage examples:
    1) used for photobleaching correction

    :param int_tseries_multiple_points: list
                list of time series intensities of multiple points
                e.g. [[point1_t0, point1_t1 ...], [point2_t0, point2_t1, ...], ...]
                int_tseries_multiple_points[i]: list, intensity series of point i
    :param pb_factor: list
                list of photobleaching factors, e.g. [pb_factor_t0, pb_factor_t1, ...]
                pb_factor[i]: float, (0,1]
    :return: out: list
                photobleaching corrected time series intensities of multiple points

    """
    out = []
    num_points = len(int_tseries_multiple_points)
    for i in range(num_points):
        out.append(np.divide(int_tseries_multiple_points[i], pb_factor))

    return out


def pix_stitch_same_row(pixels_pd: pd.DataFrame, num_col: int):
    """
    Stitch same size images in a row as a single image (empty space filled with same size black images)

    Usage examples:
    1) > print(pixels_pd)
       > col     pix
         0       pixel_0
         1       pixel_1
         3       pixel_2
       > stitched_image = pix_stitch_same_row(pixels_pd, 5)
       > print(stitched_image)
       > pixel_0 - pixel_1 - empty_img - pixel_2 - empty_img

    :param pixels_pd: pd.DataFrame, columns includes 'col', 'pix'
                'col': int, column number of the image in the stitched image, start from 0
                'pix': np.array, pixels of the image
    :param num_col: int
    :return: out: np.array, stitched images

    """
    # start with first image
    if 0 in pixels_pd['col'].tolist():
        # if image exists, assign first image to out
        out = pixels_pd[pixels_pd['col'] == 0].iloc[0]['pix']
    else:
        # if not exist, assign empty image to out
        out = np.zeros_like(pixels_pd.iloc[0]['pix'])

    # concatenate the other images sequentially on the right side of the first image
    for i in range(num_col - 1):
        # if image exists, assign corresponding image
        if (i + 1) in pixels_pd['col'].tolist():
            out = np.concatenate((out, pixels_pd[pixels_pd['col'] == i + 1].iloc[0]['pix']), axis=1, out=None)
        # if not exist, assign empty image
        else:
            out = np.concatenate((out, np.zeros_like(pixels_pd.iloc[0]['pix'])), axis=1, out=None)

    return out


def pix_stitch(pixels_pd: pd.DataFrame, num_col: int, num_row: int):
    """
    Stitch same size images into a single image based on provided location
    (empty space filled with same size black images)

    Usage examples:
    1) > print(pixels_pd)
       > row    col     pix
         0      0       pixel_0
         0      1       pixel_1
         0      3       pixel_2
         1      0       pixel_3
         1      2       pixel_4
         1      4       pixel_5
       > stitched_image = pix_stitch(pixels_pd, 5, 2)
       > print(stitched_image)
       > pixel_0 -  pixel_1  - empty_img -  pixel_2  - empty_img
         pixel_3 - empty_img -  pixel_4  - empty_img -  pixel_5

    :param pixels_pd: pd.DataFrame, columns includes 'row', 'col', 'pix'
                'row': int, row number of the image in the stitched image, start from 0
                'col': int, column number of the image in the stitched image, start from 0
                'pix': np.array, pixels of the image
    :param num_col: number of columns of stitched images
    :param num_row: number of rows of stitched images
    :return: out: np.array, stitched images

    """
    # first row
    out = pix_stitch_same_row(pixels_pd[pixels_pd['row'] == 0], num_col)
    # stitch the following rows beneath the first row
    for i in range(num_row - 1):
        out = np.concatenate((out, pix_stitch_same_row(pixels_pd[pixels_pd['row'] == i + 1], num_col)),
                             axis=0, out=None)

    return out


def find_background(pixels: np.array):
    """
    Detect true background from given image

    :param pixels: np.array, image for background identification
    :return: bg: np.array, 0-and-1, background image
    """

    bg_int = get_bg_int([pixels])[0]
    bg = pixels < 0.9 * bg_int
    for i in range(2):
        bg = binary_erosion(bg)
    for i in range(2):
        bg = binary_dilation(bg)

    return bg
