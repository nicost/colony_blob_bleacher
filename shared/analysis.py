# ------------------------------------------
# FUNCTIONS for NON 0-AND-1 NP.ARRAY (IMAGE)
# ------------------------------------------

import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import rank, threshold_triangle
from skimage.morphology import disk, opening, dilation, binary_dilation
from skimage.measure import label, regionprops


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
            pre_roi = pre_pixels[ep_rc[0] - half_roi_size[0]:ep_rc[0] + half_roi_size[0],
                      ep_rc[1] - half_roi_size[1]:ep_rc[1] + half_roi_size[1]]
            post_roi = post_pixels[ep_rc[0] - half_roi_size[0]:ep_rc[0] + half_roi_size[0],
                       ep_rc[1] - half_roi_size[1]:ep_rc[1] + half_roi_size[1]]
    subtracted = post_roi + 100 - pre_roi
    selem = disk(2)
    subtracted_mean = rank.mean(subtracted, selem=selem)
    peaks_rc_roi = peak_local_max(subtracted_mean, min_distance=20, threshold_rel=0.6, num_peaks=1, indices=True)
    peaks_rc = peaks_rc_roi + [ep_rc[0] - half_roi_size[0], ep_rc[1] - half_roi_size[1]]
    peaks = [peaks_rc[0][0], post_pixels.shape[1] - peaks_rc[0][1]]
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


def analysis_mask(pixels: np.array, x: list, y: list, num_dilation=3):
    """
    Generates an analysis mask from all the points' x,y positions.

    Expects img_example to be the same size as the expected output. With the given x,y
    positions, a 0-and-1 binary mask is generated after dilating certain rounds from the
    given coordinates.

    :param pixels: nd.array
                Requires to be the same size as the output.
    :param x: list
                List of x coordinates.
    :param y: list
                List of y coordinates
    :param num_dilation: int, optional (default: 3)
                Number of dilation applied from the coordinates.
    :return: out: nd.array, 0-and-1, same shape and type as input pixels
    """
    out = np.zeros_like(pixels)

    if len(x) == len(y):
        for i in range(len(x)):
            out[x[i], y[i]] = 1
        for i in range(num_dilation):
            out = binary_dilation(out)
    else:
        raise ValueError("Length of x: %d and y: %d does not match" % (len(x), len(y)))

    return out


def get_bg_int(t_pixels: list):
    """
    Measure background intensities of a given movie.

    :param t_pixels: list, time series of np.array (movie)
    :return: t_bg_int: list of background intensities
    """
    t_bg_int = []
    for i in range(len(t_pixels)):
        bg = np.zeros_like(t_pixels[i])
        bg[t_pixels[i] < 300] = 1
        bg_prop = regionprops(label(bg), t_pixels[i])
        t_bg_int.append(bg_prop[0].mean_intensity)

    return t_bg_int


def bg_correction(t_int: list, t_bg_int: list):
    """
    Background correction of time series intensities.

    :param t_int: list of time series intensities of multiple points
    :param t_bg_int: list of background intensities
    :return: out: list of background corrected time series intensities of multiple points
    """
    out = [[] for _ in range(len(t_int))]
    for i in range(len(t_int)):
        if len(t_int[i]) == len(t_bg_int):
            for t in range(len(t_int[i])):
                if t_int[i][t]-t_bg_int[t] > 0:
                    out[i].append(t_int[i][t]-t_bg_int[t])
                else:
                    out[i].append(0)
        else:
            raise ValueError("Length of intensities: %d does not match with background intensities: %d"
                             % (len(t_int[i]), len(t_bg_int)))

    return out


def get_pb_factor(t_int: list):
    """
    Measure photobleaching factor from given time series intensities of multiple control points.

    :param t_int: list of time series intensities of multiple points
    :return: pb_factor: list of photobleaching factors
    """
    pb_factor = []
    for t in range(len(t_int[0])):
        pb_ratio = []
        for i in range(len(t_int)):
            pb_ratio.append(np.mean(t_int[i][t])/np.mean(t_int[i][0]))
        pb_factor.append(np.mean(pb_ratio))

    return pb_factor


def pb_correction(t_int: list, pb_factor: list):
    """
    Photobleaching correction of time series intensities.

    :param t_int: list of time series intensities of multiple points
    :param pb_factor: list of photobleaching factors
    :return: out: list of photobleaching corrected time series intensities of multiple points
    """
    out = []
    for i in range(len(t_int)):
        out.append(np.divide(t_int[i], pb_factor))

    return out


def pix_stitch_same_row(pix_pd, num_grid):
    out = pix_pd[pix_pd['col'] == 0].iloc[0]['pix']
    for i in range(num_grid-1):
        out = np.concatenate((out, pix_pd[pix_pd['col'] == i+1].iloc[0]['pix']), axis=1, out=None)

    return out


def pix_stitch(pix_pd, num_grid):
    out = pix_stitch_same_row(pix_pd[pix_pd['row'] == 0], num_grid)
    for i in range(num_grid-1):
        out = np.concatenate((out, pix_stitch_same_row(pix_pd[pix_pd['row'] == i+1], num_grid)), axis=0, out=None)

    return out
