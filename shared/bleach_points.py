import shared.dataframe as dat
import shared.objects as obj
import shared.analysis as ana
import shared.math_functions as mat
import collections
from skimage.measure import label, regionprops
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
import pandas as pd

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for FRAP ANALYSIS
# ---------------------------------------------------------------------------------------------------

pd.DataFrame related:
    
    get_bleach_spots_coordinates
        FUNCTION: get coordinates of bleach spots
        SYNTAX:   get_bleach_spots_coordinates(log_pd: pd.DataFrame, store, cb, data_c: int, mode: str)
    
    get_bleach_spots
        FUNCTION: generate bleach spots mask and corresponding pd.DataFrame table
        SYNTAX:   get_bleach_spots(log_pd: pd.DataFrame, label_nucleoli: np.array, num_dilation=3)
    
    get_spots
        FUNCTION: generate spots mask and corresponding pd.DataFrame table
        SYNTAX:   get_spots(x_lst: list, y_lst: list, pixels_same_size: np.array, num_dilation=3)
    
    filter_bleach_spots
        FUNCTION: filter bleach spots
        SYNTAX:   filter_bleach_spots(log_pd: pd.DataFrame)
    
    frap_analysis
        FUNCTION: analyze FRAP curve
        SYNTAX:   frap_analysis(pointer_pd: pd.DataFrame, max_t: int, acquire_time_tseries: list, real_time: list)
    
    frap_filter
        FUNCTION: filter FRAP curves
        SYNTAX:   frap_filter(pointer_pd: pd.DataFrame, f: str)
    
    filter_ctrl
        FUNCTION: filter control spots for FRAP photobleaching correction
        SYNTAX:   filter_ctrl(df: pd.DataFrame)
    
    frap_pb_correction
        FUNCTION: photobleaching correction for FRAP
        SYNTAX:   frap_pb_correction(pointer_pd: pd.DataFrame, ctrl_pd: pd.DataFrame)
    
    frap_curve_fitting
        FUNCTION: FRAP curve fitting
        SYNTAX:   frap_curve_fitting(pointer_pd: pd.DataFrame)
    
"""


def get_bleach_spots_coordinates(log_pd: pd.DataFrame, store, cb, data_c: int, mode: str, frap_start_delay: int):
    """
    Get coordinates of bleach spots

    Algorithm description:
    single-raw:
        use single coordinates from raw log file to track FRAP for the whole movie
    single-offset:
        use single coordinates detected from intensity change around photobleaching for the whole movie.
        for each pointer (center coordinates of potential bleach spot), its minimum intensity after
        photobleaching was generally detected at bleach frame + 4. this image was subtracted from image
        at bleach frame (before photobleaching) and add 100 to smooth background noise.  otsu global
        thresholding was then applied to find the bright spots (detected bleach spots). two rounds of
        erosion/dilation was applied to clean background and the closest round of the centroid coordinates
        of the closest detected bleach spots was used as the new coordinates of given bleach spot/pointer.

    Usage examples:
    1) used for detecting bleach spots

    :param log_pd: pd.DataFrame, requires columns 'aim_x', 'aim_y', 'bleach_frame'
                'aim_x': x coordinates of aim during photobleaching experiment
                'aim_y': y coordinates of aim during photobleaching experiment
                'bleach_frame': photobleaching frame
    :param store: store: store = mm.data().load_data(data_path, True)
    :param cb: cb = mm.data().get_coords_builder()
    :param data_c: channel to be analyzed
    :param mode: bleach spot detection mode, currently only accepts 'single-raw' and 'single-offset'
    :param frap_start_delay: delay compared with bleach_frame to determine the starting point of frap
    :return: coordinate_pd: pd.DataFrame, same index as log_pd with columns 'x', 'y', 'x_diff', 'y_diff'

    """
    if mode == 'single-raw':
        coordinate_pd = pd.DataFrame({'x': log_pd['aim_x'],
                                      'y': log_pd['aim_y'],
                                      'x_diff': [0] * len(log_pd),
                                      'y_diff': [0] * len(log_pd)})

    elif mode == 'single-offset':
        x_lst = []
        y_lst = []
        for i in range(len(log_pd)):
            aim_x = log_pd['aim_x'][i]
            aim_y = log_pd['aim_y'][i]
            bleach_frame = int(log_pd['bleach_frame'][i])

            # subtract minimum intensity image pix(bleach_frame+4) and image before photobleaching
            post = store.get_image(cb.p(0).z(0).c(data_c).t(bleach_frame+frap_start_delay).build())
            pix_post = np.reshape(post.get_raw_pixels(), newshape=[post.get_height(), post.get_width()])
            pre = store.get_image(cb.p(0).z(0).c(data_c).t(bleach_frame).build())
            pix_pre = np.reshape(pre.get_raw_pixels(), newshape=[pre.get_height(), pre.get_width()])
            pix = pix_post - pix_pre + 100

            # otsu global thresholding to find bright spots
            otsu_val = threshold_otsu(pix)
            otsu = pix > otsu_val
            # two rounds of erosion/dilation to clear out noises
            bleach = binary_erosion(otsu)
            bleach = binary_erosion(bleach)
            bleach = binary_dilation(bleach)
            bleach = binary_dilation(bleach)

            # tried to use local maxima to find bleach spots, not good
            # local structure will interfere with intensity detection
            """if mode == 'local_maxima':
                label_bleach = label(bleach)
                for m in range(np.amax(label_bleach)):
                    mask = np.zeros_like(pix)
                    mask[label_bleach == m+1] = 1
                    bleach_pix = pix.copy()
                    bleach_pix[mask == 0] = 0
                    peaks = peak_local_max(bleach_pix, min_distance=20)
                    if len(peaks) > 0:
                        for n in peaks:
                            peak_x.append(n[1])
                            peak_y.append(n[0])"""

            # use centroid to find bleach spots
            # local maxima is largely affected by intensity from other cellular feature
            bleach_prop = regionprops(label(bleach))
            peak_x = [round(p.centroid[1]) for p in bleach_prop]
            peak_y = [round(p.centroid[0]) for p in bleach_prop]

            if len(peak_x) == 1:  # found one bright spot
                x_lst.append(peak_x[0])
                y_lst.append(peak_y[0])
            elif len(peak_x) > 1:  # found more than one bright spots
                x_closest, y_closest = dat.find_closest(aim_x, aim_y, peak_x, peak_y)
                x_lst.append(x_closest)
                y_lst.append(y_closest)
            else:  # do not find any bright spots, use the coordinates from log file
                x_lst.append(0)
                y_lst.append(0)

        coordinate_pd = pd.DataFrame({'x': x_lst, 'y': y_lst})
        coordinate_pd['x_diff'] = coordinate_pd['x'] - log_pd['aim_x']
        coordinate_pd['y_diff'] = log_pd['aim_y'] - coordinate_pd['y']

    else:
        coordinate_pd = pd.DataFrame()

    return coordinate_pd


def get_bleach_spots(log_pd: pd.DataFrame, label_nucleoli: np.array, analyze_organelle: str, num_dilation=3):
    """
    Generate bleach spots mask and corresponding pd.DataFrame table

    :param log_pd: pd.DataFrame, requires columns 'x', 'y', 'nucleoli'
                'x': x coordinates of all possible bleach spots
                'y': y coordinates of all possible bleach spots
                'nucleoli': pointer corresponding nucleoli label index
                originally imports from .log file
    :param label_nucleoli: np.array, grey scale labeled nucleoli image
    :param analyze_organelle: str, 'sg' or 'nucleoli'
    :param num_dilation: int, optional (default: 3)
                number of dilation applied from the coordinates
                determines the size of the generated mask for each point
                default number was determined for FRAP analysis
    :return: bleach_spots_ft: np.array, 0-and-1
                bleach spots mask
             pointer_pd: pd.DataFrame
                pointer dataframe, add corresponding bleach spots/nucleoli information
                sorted based on bleach spots, ascending

    """
    # link pointer with corresponding detected bleach spots
    bleach_spots = ana.analysis_mask(log_pd['y'], log_pd['x'], label_nucleoli, num_dilation)
    label_bleach_spots = label(bleach_spots, connectivity=1)
    log_pd['bleach_spots'] = obj.points_in_objects(label_bleach_spots, log_pd['x'], log_pd['y'])

    # filter bleach spots
    pointer_pd = filter_bleach_spots(log_pd, analyze_organelle)

    # generate bleach spots mask (after filtering)
    bleach_spots_ft = ana.analysis_mask(pointer_pd['y'], pointer_pd['x'], label_nucleoli, num_dilation)
    label_bleach_spots_ft = label(bleach_spots_ft, connectivity=1)

    # link pointer with corresponding filtered bleach spots
    pointer_pd['bleach_spots'] = obj.points_in_objects(label_bleach_spots_ft, pointer_pd['x'], pointer_pd['y'])
    pointer_pd = pointer_pd.sort_values(by='bleach_spots').reset_index(drop=True)

    return bleach_spots_ft, pointer_pd


def get_spots(x_lst: list, y_lst: list, pixels_same_size: np.array, num_dilation=3):
    """
    Generate spots mask and corresponding pd.DataFrame table

    :param x_lst: list, list of x coordinates
    :param y_lst: list, list of y coordinates
    :param pixels_same_size: np.array, any image of the same size
    :param num_dilation: int, number of dilation applied, default: 3
    :return:
    """
    spots = ana.analysis_mask(y_lst, x_lst, pixels_same_size, num_dilation)
    label_spots = label(spots, connectivity=1)
    pd_spots = pd.DataFrame({'x': x_lst, 'y': y_lst})
    pd_spots['spots'] = obj.points_in_objects(label_spots, pd_spots['x'], pd_spots['y'])

    # filter spots
    pointer_same_spots = [item for item, count in collections.Counter(pd_spots['spots'].tolist()).items() if count > 1]
    pd_spots = pd_spots[~pd_spots['spots'].isin(pointer_same_spots)]
    del pd_spots['spots']  # delete previous bleach_spots information
    pd_spots = pd_spots.reset_index(drop=True)  # reset index

    spots_ft = ana.analysis_mask(pd_spots['y'], pd_spots['x'], pixels_same_size, num_dilation)
    label_spots_ft = label(spots_ft, connectivity=1)
    # link pointer with corresponding filtered bleach spots
    pd_spots['spots'] = obj.points_in_objects(label_spots_ft, pd_spots['x'], pd_spots['y'])
    pd_spots = pd_spots.sort_values(by='spots').reset_index(drop=True)
    return spots_ft, pd_spots


def filter_bleach_spots(log_pd: pd.DataFrame, analyze_organelle: str):
    """
    Filter bleach spots

    Filter out bleach spots:
    1) aim outside of nucleoli (bleach_spots_check_organelle = 'Y')
    2) bleach the same nucleoli (bleach_spots_check_organelle = 'Y')
    3) too close to merge as a single bleach spots
    4) (0,0)
    5) too far away from the aim point (>20 any direction)

    :param log_pd: pd.DataFrame, requires columns 'nucleoli', 'bleach_spots'
                'nucleoli': corresponding nucleoli label index
                'bleach_spots': corresponding bleach spots label index
    :param analyze_organelle: str, 'sg' or 'nucleoli'
    :return: pointer_pd: pd.DataFrame
                pointer dataframe of all filtered bleach spots

    """
    # mask2
    pointer_target_same_organelle = \
        [item for item, count in collections.Counter(log_pd['%s' % analyze_organelle].tolist()).items() if count > 1]
    # mask3
    pointer_same_analysis_spots = \
        [item for item, count in collections.Counter(log_pd['bleach_spots'].tolist()).items() if count > 1]

    # filter all the pointers from log to generate real pointer_pd
    pointer_pd = log_pd[(log_pd['%s' % analyze_organelle] > 0)
                        & (~log_pd['%s' % analyze_organelle].isin(pointer_target_same_organelle))
                        & (~log_pd['bleach_spots'].isin(pointer_same_analysis_spots))
                        & (log_pd['x'] != 0) & (log_pd['y'] != 0) & (np.abs(log_pd['x_diff']) <= 10)
                        & (np.abs(log_pd['y_diff'] <= 10))]

    del pointer_pd['bleach_spots']  # delete previous bleach_spots information
    pointer_pd = pointer_pd.reset_index(drop=True)  # reset index

    # print number of pointers failed to pass each filter
    # filters applied later will not count the ones that fail from the previous filters
    num_filter1 = len(log_pd[(log_pd['%s' % analyze_organelle] == 0)])
    print("%d bleach spots aim outside of %s." % (num_filter1, analyze_organelle))
    num_filter2 = len(log_pd[(log_pd['%s' % analyze_organelle] > 0)
                             & (log_pd['%s' % analyze_organelle].isin(pointer_target_same_organelle))])
    print("%d bleach spots aim to the same %s." % (num_filter2, analyze_organelle))
    num_filter3 = len(log_pd[(log_pd['%s' % analyze_organelle] > 0)
                             & (~log_pd['%s' % analyze_organelle].isin(pointer_target_same_organelle))
                             & (log_pd['bleach_spots'].isin(pointer_same_analysis_spots))])
    print("%d bleach spots aim too close." % num_filter3)
    num_filter4 = len(log_pd[(log_pd['%s' % analyze_organelle] > 0)
                             & (~log_pd['%s' % analyze_organelle].isin(pointer_target_same_organelle))
                             & (~log_pd['bleach_spots'].isin(pointer_same_analysis_spots))
                             & ((log_pd['x'] == 0) | (log_pd['y'] == 0))])
    print("%d bleach spots did not find." % num_filter4)
    num_filter5 = len(log_pd[(log_pd['%s' % analyze_organelle] > 0)
                             & (~log_pd['%s' % analyze_organelle].isin(pointer_target_same_organelle))
                             & (~log_pd['bleach_spots'].isin(pointer_same_analysis_spots))
                             & (log_pd['x'] != 0) & (log_pd['y'] != 0)
                             & ((np.abs(log_pd['x_diff']) > 10) | (np.abs(log_pd['y_diff']) > 10))])
    print("%d bleach spots too far away from aim points." % num_filter5)

    return pointer_pd


def get_t_half(half_int: float or int, int_tseries: list, real_time_post: list):
    """
    Calculate t_half from half intensity

    :param half_int:
    :param int_tseries:
    :param real_time_post:
    :return:
    """
    half_frame = dat.find_pos(half_int, int_tseries)
    if half_frame == len(real_time_post):
        t_half = np.nan
    else:
        t_half = real_time_post[half_frame]

    return t_half


def frap_analysis(pointer_pd: pd.DataFrame, max_t: int, acquire_time_tseries: list, real_time: list,
                  frap_start_delay: int, frap_start_mode: str):
    """
    Analyze FRAP curve

    :param pointer_pd: pd.DataFrame, requires columns 'bleach_frame', 'mean_int'
                'bleach_frame': frame of or right after photobleaching
                'mean_int': FRAP double corrected intensity series
    :param max_t: int
                number of total frame
    :param acquire_time_tseries: list
                list of acquisition time displayed in 'hour:min:sec' format
                e.g. ['17:30:38.360', '17:30:38.455', '17:30:38.536', '17:30:38.615', ...]
    :param real_time: list
                list of time in sec with first frame set as 0
                e.g. [0.0, 0.09499999999999886, 0.17600000000000193, 0.25500000000000256, ...]
    :param frap_start_delay: int
                delay compared with bleach_frame to determine the starting point of frap
    :param frap_start_mode: str
                only accepts 'delay' or 'min'
    :return: frap_pd: pd.DataFrame
                add information: 'int_curve_nor', 'frap_start_frame', 'imaging_length', 'int_curve_pre',
                'int_curve_post', 'int_curve_post_nor', 'real_time_post', 'pre_bleach_int', 'frap_start_int',
                'plateau_int', 'mobile_fraction', 'immobile_fraction', 'half_int', 'half_int_nor',
                'half_frame', 't_half', 'ini_slope'

                'int_curve_nor': intensity series normalized with pre_bleach_int and frap_start_int (min_int)
                'frap_start_frame': bleach_frame + 4, generally the frame of minimum intensity
                'imaging_length': number of frames of FRAP curves
                'int_curve_pre': intensity series before photobleaching (without bleach_frame, before spike)
                'int_curve_post': FRAP curve, intensity series after minimum intensity (includes frap_start_frame)
                'int_curve_post_nor': t_int_post normalized with pre_bleach_int and frap_start_int (min_int)
                'real_time_post': time series represents in second
                'pre_bleach_int': mean intensity before photobleaching; pre-bleach intensity
                'frap_start_int': frap start intensity after photobleaching, generally minimum intensity
                'plateau_int': plateau level intensity calculated from last 10 frames
                'mobile_fraction': int_plateau normalized with pre_bleach_int and frap_start_int (min_int)
                'immobile_fraction': 1 - mobile_fraction
                'half_int': half intensity
                'half_int_nor': int_half normalized with pre_bleach_int and frap_start_int (min_int)
                'half_frame': number of frames it takes to reach half intensity (frap_start_frame, half_int_frame]
                't_half': t-half
                'ini_slope': initial slope of the recovery curve (relative intensity)

    """
    # for all the bleach spots
    frap_start_frame = []  # bleach_frame + frap_start_delay or minimum intensity frame if delay = 0
    min_int = []
    min_int_frame = [] # frame number of the minimum intensity
    t_int_post = []  # intensity series after minimum intensity (includes min_int_frame, frap recovery curve)
    imaging_length = []  # number of frames of t_int_post
    t_int_pre = []  # intensity series before photobleaching (without bleach_frame, before spike)
    sigma_lst = []  # constant measurement error: standard deviation of pre_bleach intensities
    pre_bleach_int = []  # mean intensity before photobleaching; pre-bleach intensity
    frap_start_int = []  # frap start intensity after photobleaching
    t_int_post_nor = []  # t_int_post normalized with pre_bleach_int and min_int
    real_time_post = []  # time series represents in second
    mean_int_nor = []  # intensity series normalized with pre_bleach_int and min_int
    plateau_int = []  # plateau level intensity
    plateau_int_nor = []  # int_plateau normalized with pre_bleach_int and min_int; mobile fraction
    t_half = []  # t-half
    slope = []  # initial slope of the recovery curve (relative intensity)
    t_diff = []

    for i in range(len(pointer_pd)):
        # number of first frame after photobleaching (num_pre)
        num_pre = pointer_pd['bleach_frame'][i]
        # minimum intensity
        min_int_temp = min(pointer_pd['mean_int'][i])
        min_int.append(min_int_temp)
        # minimum intensity frame
        min_int_frame_temp = pointer_pd['mean_int'][i].tolist().index(min_int_temp)
        min_int_frame.append(min_int_frame_temp)
        # frap curve starting point
        if frap_start_mode == 'delay':
            frap_start_frame_temp = pointer_pd['bleach_frame'][i] + frap_start_delay
        else:
            if min_int_frame_temp < (len(pointer_pd['mean_int'][i])-10):
                frap_start_frame_temp = min_int_frame_temp
            else:
                frap_start_frame_temp = pointer_pd['bleach_frame'][i] + frap_start_delay
        frap_start_frame.append(frap_start_frame_temp)
        # imaging length of the frap curve after min_int_frame
        num_post = max_t - frap_start_frame_temp
        imaging_length.append(num_post)
        # intensities before photobleaching and intensities after frap_start_int
        int_post = pointer_pd['mean_int'][i][-num_post:]
        int_pre = pointer_pd['mean_int'][i][:num_pre]
        t_int_pre.append(int_pre)
        t_int_post.append(int_post)
        # time series represents in sec
        real_time_post_temp = [x - real_time[-num_post] for x in real_time[-num_post:]]
        real_time_post.append(real_time_post_temp)
        # mean intensity before photobleaching
        pre_bleach_int_temp = np.mean(int_pre)
        pre_bleach_int.append(pre_bleach_int_temp)
        # frap_start intensity after photobleaching
        frap_start_int_temp = int_post[0]
        frap_start_int.append(frap_start_int_temp)
        # normalized intensities after min_intensity based on pre_bleach_int and min_int
        full_range_int = pre_bleach_int_temp - frap_start_int_temp
        int_post_nor = [(x - frap_start_int_temp) / full_range_int for x in int_post]
        t_int_post_nor.append(int_post_nor)
        # constant measurement error: standard deviation of pre_bleach intensities
        int_pre_nor = [(x - frap_start_int_temp) / full_range_int for x in int_pre]
        sigma = np.std(int_pre_nor)
        sigma_lst.append(sigma)
        # intensity normalized based on pre_bleach_int and min_int
        mean_int_nor_temp = [(x - frap_start_int_temp) / full_range_int for x in pointer_pd['mean_int'][i]]
        mean_int_nor.append(mean_int_nor_temp)
        # plateau level intensity calculated from last 10 frames of the frap curve
        plateau_int_temp = np.mean(pointer_pd['mean_int'][i][-10:])
        plateau_int.append(plateau_int_temp)
        plateau_int_nor_temp = (plateau_int_temp - frap_start_int_temp) / full_range_int
        plateau_int_nor.append(plateau_int_nor_temp)
        # t_half (sec)
        t_half_temp = get_t_half(plateau_int_nor_temp/2, int_post_nor, real_time_post_temp)
        t_half.append(t_half_temp)
        # initial slope calculated based on first 5 frames
        if frap_start_frame_temp + 5 < len(pointer_pd['mean_int']):
            int_change = (pointer_pd['mean_int'][i][frap_start_frame_temp + 5] - frap_start_int_temp) / full_range_int
            t_change = dat.get_time_length(frap_start_frame_temp, frap_start_frame_temp + 5, acquire_time_tseries)
            slope_temp = 1.0 * (int_change / t_change)
        else:
            slope_temp = np.nan
        slope.append(slope_temp)
        # real time difference between photobleaching and next acquisition
        time = pointer_pd['time'][i].split(' ')[1]
        t_bleach = [float(time.split(':')[0]), float(time.split(':')[1]), float(time.split(':')[2])]
        t_diff_temp = dat.get_time_diff(t_bleach, pointer_pd['bleach_frame'][i], acquire_time_tseries)
        t_diff.append(t_diff_temp)

    frap_pd = pd.DataFrame({'int_curve_nor': mean_int_nor,
                            'min_int_frame': min_int_frame,
                            'frap_start_frame': frap_start_frame,
                            'imaging_length': imaging_length,
                            'int_curve_pre': t_int_pre,
                            'int_curve_post': t_int_post,
                            'int_curve_post_nor': t_int_post_nor,
                            'real_time_post': real_time_post,
                            'sigma': sigma_lst,
                            'pre_bleach_int': pre_bleach_int,
                            'min_int': min_int,
                            'frap_start_int': frap_start_int,
                            'plateau_int': plateau_int,
                            'mobile_fraction': plateau_int_nor,
                            't_half': t_half,
                            'ini_slope': slope,
                            't_diff': t_diff})

    return frap_pd


def frap_filter(pointer_pd: pd.DataFrame, f: str):
    """
    Filter FRAP curves

    filter frap curves:
    1) number of pre_bleach frame < 5
    2) total imaging length < 100
    3) does not find optional fit (single exponential)
    4) mobile fraction < 0 or mobile fraction > 1.05
    5) r2 of fit < 0.7

    :param pointer_pd: pd.DataFrame
    :param f: filter based on which function
    :return: pointer_pd: pd.DataFrame, add one column 'frap_filter'
                'frap_filter': FRAP curve filtering result
                    1: passed
                    0: not passed

    """
    frap_flt = []
    for i in range(len(pointer_pd)):
        if (pointer_pd['bleach_frame'][i] < 5) \
                | (pointer_pd['imaging_length'][i] < 100) \
                | (np.isnan(pointer_pd['%s_r2' % f][i])) \
                | (pointer_pd['%s_mobile_fraction' % f][i] < 0) \
                | (pointer_pd['%s_mobile_fraction' % f][i] >= 1.05) \
                | (pointer_pd['%s_r2' % f][i] < 0.7):
            frap_flt.append(0)
        else:
            frap_flt.append(1)

    return frap_flt


def filter_ctrl(df: pd.DataFrame):
    """
    Filter control spots for FRAP curve photobleaching correction
    :param df: pd.DataFrame, control spots dataframe
    :return: df_ft: pd.DataFrame, sorted control spots dataframe
    """
    flt_ctrl = []
    for i in range(len(df)):
        ctrl_int = df['bg_cor_int'][i]
        if (max(ctrl_int) - min(ctrl_int)) / (max(ctrl_int) + 0.0001) > 0.4:
            flt_ctrl.append(0)
        else:
            flt_ctrl.append(1)
    df['filter'] = flt_ctrl
    df_ft = df[df['filter'] == 1].reset_index()
    return df_ft


def frap_pb_correction(pointer_pd: pd.DataFrame, ctrl_pd: pd.DataFrame):
    """
    Photobleaching correction for FRAP
    :param pointer_pd: pd.DataFrame, dataframe of all the bleach spots
    :param ctrl_pd: pd.DataFrame, dataframe of control spots
    :return:
    """
    # calculate photobleaching factor
    pb_factor = ana.get_pb_factor(ctrl_pd['bg_cor_int'])

    pointer_pd['pb_factor'] = [pb_factor] * len(pointer_pd)

    # pb_factor fitting with single exponential decay
    pb_fit = mat.fitting_single_exp_decay(np.arange(0, len(pb_factor), 1), pb_factor)
    pointer_pd = dat.add_columns(pointer_pd, ['pb_single_exp_decay_fit', 'pb_single_exp_decay_r2',
                                              'pb_single_exp_decay_a', 'pb_single_exp_decay_b',
                                              'pb_single_exp_decay_c'],
                                 [[pb_fit[0]] * len(pointer_pd), [pb_fit[1]] * len(pointer_pd),
                                  [pb_fit[2]] * len(pointer_pd), [pb_fit[3]] * len(pointer_pd),
                                  [pb_fit[4]] * len(pointer_pd)])

    # photobleaching correction
    if np.isnan(pb_fit[2]):
        pb = pb_factor
    else:
        pb = pb_fit[0]
    pointer_pd['mean_int'] = ana.pb_correction(pointer_pd['bg_cor_int'], pb)
    return pointer_pd


def frap_curve_fitting(pointer_pd: pd.DataFrame):
    """
    FRAP curve fitting
    :param pointer_pd: pd.DataFrame, dataframe of all the bleach spots
    :return:
    """
    # curve fitting with linear to determine initial slope
    linear_fit_pd = mat.frap_fitting_linear(pointer_pd['real_time_post'], pointer_pd['int_curve_post_nor'])
    pointer_pd = pd.concat([pointer_pd, linear_fit_pd], axis=1)

    # curve fitting with single exponential function
    single_exp_fit_pd = mat.frap_fitting_single_exp(pointer_pd['real_time_post'],
                                                    pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, single_exp_fit_pd], axis=1)

    # curve fitting with soumpasis function
    soumpasis_fit_pd = mat.frap_fitting_soumpasis(pointer_pd['real_time_post'],
                                                  pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, soumpasis_fit_pd], axis=1)

    # curve fitting with double exponential function
    double_exp_fit_pd = mat.frap_fitting_double_exp(pointer_pd['real_time_post'],
                                                    pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, double_exp_fit_pd], axis=1)

    # curve fitting with ellenberg function
    ellenberg_fit_pd = mat.frap_fitting_ellenberg(pointer_pd['real_time_post'],
                                                  pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, ellenberg_fit_pd], axis=1)

    # find optimal fitting
    optimal_fit_pd = mat.find_optimal_fitting(pointer_pd, ['single_exp', 'soumpasis', 'ellenberg', 'double_exp'])
    pointer_pd = pd.concat([pointer_pd, optimal_fit_pd], axis=1)

    # filter frap curves
    pointer_pd['frap_filter_single_exp'] = frap_filter(pointer_pd, 'single_exp')
    pointer_pd['frap_filter_soumpasis'] = frap_filter(pointer_pd, 'soumpasis')
    pointer_pd['frap_filter_double_exp'] = frap_filter(pointer_pd, 'double_exp')
    pointer_pd['frap_filter_ellenberg'] = frap_filter(pointer_pd, 'ellenberg')
    pointer_pd['frap_filter_optimal'] = frap_filter(pointer_pd, 'optimal')

    return pointer_pd
