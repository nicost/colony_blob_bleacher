import shared.dataframe as dat
import shared.objects as obj
import shared.analysis as ana
import collections
from skimage.measure import label, regionprops
import numpy as np
from scipy.optimize import curve_fit
import shared.math_functions as mat
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
import pandas as pd


def get_bleach_spots(log_pd, label_nucleoli, nucleoli_pd, num_dilation=3):
    """
    Generate bleach spots mask and corresponding pd.DataFrame table.

    :param log_pd: pd.DataFrame imports from .log file
    :param label_nucleoli: np.array, grey scale
    :param nucleoli_pd:
    :param num_dilation: number of dilation from the coordinate; determines analysis size of the
        analysis spots; default = 3
    :return: bleach_spots_ft: np.array, 0-and-1, bleach spots mask
             pointer_ft_pd: pd.DataFrame of pointer information after filter
    """

    # link pointer with corresponding nucleoli
    log_pd['nucleoli'] = obj.points_in_objects(label_nucleoli, log_pd['x'], log_pd['y'])

    # create analysis mask for all analysis points
    bleach_spots = ana.analysis_mask(log_pd['y'], log_pd['x'], label_nucleoli, num_dilation)
    label_bleach_spots = label(bleach_spots, connectivity=1)

    # link pointer with corresponding bleach spots
    log_pd['bleach_spots'] = obj.points_in_objects(label_bleach_spots, log_pd['x'], log_pd['y'])

    # filter out analysis spots:
    # 1) aim outside of nucleoli
    # 2) bleach the same nucleoli
    # 3) too close to merge as a single analysis spots
    pointer_target_same_nucleoli = \
        [item for item, count in collections.Counter(log_pd['nucleoli'].tolist()).items() if count > 1]
    pointer_same_analysis_spots = \
        [item for item, count in collections.Counter(log_pd['bleach_spots'].tolist()).items() if count > 1]
    pointer_ft_pd = log_pd[(log_pd['nucleoli'] > 0)
                           & (~log_pd['nucleoli'].isin(pointer_target_same_nucleoli))
                           & (~log_pd['bleach_spots'].isin(pointer_same_analysis_spots))]
    del pointer_ft_pd['bleach_spots']
    pointer_ft_pd = pointer_ft_pd.reset_index(drop=True)
    print("%d bleach spots aim outside of nucleoli." % len(log_pd[(log_pd['nucleoli'] == 0)]))
    print("%d bleach spots aim to the same nucleoli." %
          len(log_pd[(log_pd['nucleoli'] > 0) & (log_pd['nucleoli'].isin(pointer_target_same_nucleoli))]))
    print("%d bleach spots aim too close." % len(log_pd[(log_pd['nucleoli'] > 0)
                                                        & (~log_pd['nucleoli'].isin(pointer_target_same_nucleoli))
                                                        & (log_pd['bleach_spots'].isin(pointer_same_analysis_spots))]))

    bleach_spots_ft = ana.analysis_mask(pointer_ft_pd['y'], pointer_ft_pd['x'], label_nucleoli, num_dilation)
    label_bleach_spots_ft = label(bleach_spots_ft, connectivity=1)
    # link pointer with corresponding filtered bleach spots
    pointer_ft_pd['bleach_spots'] = obj.points_in_objects(label_bleach_spots_ft, pointer_ft_pd['x'], pointer_ft_pd['y'])

    # measure pointer corresponding nucleoli sizes
    pointer_nucleoli_x = []
    pointer_nucleoli_y = []
    pointer_nucleoli_size = []
    pointer_nucleoli_int = []
    pointer_nucleoli_circ = []
    for i in range(len(pointer_ft_pd)):
        pointer_nucleoli_x.append(nucleoli_pd['centroid_x'][pointer_ft_pd['nucleoli'][i]-1])
        pointer_nucleoli_y.append(nucleoli_pd['centroid_y'][pointer_ft_pd['nucleoli'][i]-1])
        pointer_nucleoli_size.append(nucleoli_pd['size'][pointer_ft_pd['nucleoli'][i]-1])
        pointer_nucleoli_int.append(nucleoli_pd['mean_int'][pointer_ft_pd['nucleoli'][i]-1])
        pointer_nucleoli_circ.append(nucleoli_pd['circ'][pointer_ft_pd['nucleoli'][i]-1])

    pointer_ft_pd = dat.add_columns(pointer_ft_pd, ['nucleoli_x', 'nucleoli_y', 'nucleoli_size',
                                                    'nucleoli_mean_int', 'nucleoli_circ'],
                                    [pointer_nucleoli_x, pointer_nucleoli_y, pointer_nucleoli_size,
                                     pointer_nucleoli_int, pointer_nucleoli_circ])

    return bleach_spots_ft, pointer_ft_pd


def get_frap(pointer_pd, store, cb, bleach_spots, nucleoli_pd, log_pd, num_dilation=3):

    img = store.get_image(cb.c(0).z(0).p(0).t(0).build())
    max_t = store.get_max_indices().get_t()
    pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

    # create analysis mask for control spots
    ctrl_nucleoli = ~nucleoli_pd.index.isin(log_pd['nucleoli'].tolist())
    ctrl_centroids_x = nucleoli_pd[ctrl_nucleoli]['centroid_x'].astype(int).tolist()
    ctrl_centroids_y = nucleoli_pd[ctrl_nucleoli]['centroid_y'].astype(int).tolist()
    ctrl_spots = ana.analysis_mask(ctrl_centroids_x, ctrl_centroids_y, pixels, num_dilation)

    pixels_tseries = []
    # measure mean intensity for bleach spots and control spots
    bleach_spots_int_tseries = [[] for _ in range(obj.object_count(bleach_spots))]
    ctrl_spots_int_tseries = [[] for _ in range(obj.object_count(ctrl_spots))]
    for t in range(0, max_t):
        img = store.get_image(cb.t(t).build())
        pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
        pixels_tseries.append(pixels)

        bleach_spots_pix = regionprops(label(bleach_spots, connectivity=1), pixels)
        ctrl_spots_pix = regionprops(label(ctrl_spots, connectivity=1), pixels)
        for i in range(len(bleach_spots_pix)):
            bleach_spots_int_tseries[i].append(bleach_spots_pix[i].mean_intensity)
        for i in range(len(ctrl_spots_pix)):
            ctrl_spots_int_tseries[i].append(ctrl_spots_pix[i].mean_intensity)
    pointer_pd['raw_int'] = bleach_spots_int_tseries

    # background intensity measurement
    bg_int_tseries = ana.get_bg_int(pixels_tseries)
    pointer_pd['bg_int'] = [bg_int_tseries] * len(pointer_pd)

    # background intensity fitting
    pointer_pd = bg_fitting_linear(pointer_pd)

    # background correction
    if np.isnan(pointer_pd['bg_linear_a'][0]):
        bg = bg_int_tseries
    else:
        bg = pointer_pd['bg_linear_fit'][0]
    bleach_spots_int_cor = ana.bg_correction(bleach_spots_int_tseries, bg)
    ctrl_spots_int_cor = ana.bg_correction(ctrl_spots_int_tseries, bg)
    pointer_pd['bg_cor_int'] = bleach_spots_int_cor
    num_ctrl_spots = obj.object_count(ctrl_spots)
    ctrl_pd = pd.DataFrame({'ctrl_spots': np.arange(0, num_ctrl_spots, 1), 'raw_int': ctrl_spots_int_tseries,
                            'bg_cor_int': ctrl_spots_int_cor})
    pointer_pd['num_ctrl_spots'] = [num_ctrl_spots] * len(pointer_pd)

    # photobleaching correction
    if num_ctrl_spots != 0:
        # calculate photobleaching factor
        pb_factor = ana.get_pb_factor(ctrl_spots_int_cor)
        pointer_pd['pb_factor'] = [pb_factor] * len(pointer_pd)
        print("%d ctrl points are used to correct photobleaching." % obj.object_count(ctrl_spots))
        # pb_factor fitting with single exponential decay
        pointer_pd = pb_factor_fitting_single_exp(pointer_pd)

        # photobleaching correction
        if np.isnan(pointer_pd['pb_single_exp_decay_a'][0]):
            pb = pb_factor
        else:
            pb = pointer_pd['pb_single_exp_decay_fit'][0]
        bleach_spots_int_dual_cor = ana.pb_correction(bleach_spots_int_cor, pb)
        # add corrected intensities into pointer_ft
        pointer_pd['mean_int'] = bleach_spots_int_dual_cor

    return pointer_pd, ctrl_pd


def get_bleach_frame(log_pd, store, cb):
    log_pd.columns = ['time', 'aim_x', 'aim_y']
    acquire_time_tseries, real_time = dat.get_time_tseries(store, cb)
    bleach_frame_pointer_fl = []  # frame number of or right after photobleaching
    for i in range(len(log_pd)):
        # number of first frame after photobleaching (num_pre)
        num_pre = dat.find_pos(log_pd['time'][i].split(' ')[1], acquire_time_tseries)
        bleach_frame_pointer_fl.append(num_pre)

    log_pd['bleach_frame'] = bleach_frame_pointer_fl

    return log_pd


def frap_analysis(pointer_pd, store, cb):
    max_t = store.get_max_indices().get_t()
    acquire_time_tseries, real_time = dat.get_time_tseries(store, cb)
    # for all the bleach spots
    frap_start_frame = []  # bleach_frame + 4
    # frame number of the minimum intensity
    t_int_post = []  # intensity series after minimum intensity (includes min_int_frame, frap recovery curve)
    imaging_length = []  # number of frames of t_int_post
    t_int_pre = []  # intensity series before photobleaching (without bleach_frame, before spike)
    pre_bleach_int = []  # mean intensity before photobleaching; pre-bleach intensity
    frap_start_int = []  # frap start intensity after photobleaching
    t_int_post_nor = []  # t_int_post normalized with pre_bleach_int and min_int
    real_time_post = []  # time series represents in second
    mean_int_nor = []  # intensity series normalized with pre_bleach_int and min_int
    plateau_int = []  # plateau level intensity
    plateau_int_nor = []  # int_plateau normalized with pre_bleach_int and min_int; mobile fraction
    immobile_fraction = []  # 1-plateau_int_nor
    half_int = []  # half intensity
    half_int_nor = []  # int_half normalized with pre_bleach_int and min_int
    half_frame = []  # number of frames it takes to reach half intensity (min_int_frame, half_int_frame]
    t_half = []  # t-half
    slope = []  # initial slope of the recovery curve (relative intensity)

    for i in range(len(pointer_pd)):
        # number of first frame after photobleaching (num_pre)
        num_pre = dat.find_pos(pointer_pd['time'][i].split(' ')[1], acquire_time_tseries)
        # frap curve starting point
        frap_start_frame_temp = pointer_pd['bleach_frame'][i] + 4
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
        real_time_post.append([x - real_time[-num_post] for x in real_time[-num_post:]])
        # mean intensity before photobleaching
        pre_bleach_int_temp = np.mean(int_pre)
        pre_bleach_int.append(pre_bleach_int_temp)
        # frap_start intensity after photobleaching
        frap_start_int_temp = int_post[0]
        frap_start_int.append(frap_start_int_temp)
        # normalized intensities after min_intensity based on pre_bleach_int and min_int
        full_range_int = pre_bleach_int_temp - frap_start_int_temp
        t_int_post_nor.append([(x - frap_start_int_temp) / full_range_int for x in int_post])
        # intensity normalized based on pre_bleach_int and min_int
        mean_int_nor_temp = [(x - frap_start_int_temp) / full_range_int for x in pointer_pd['mean_int'][i]]
        mean_int_nor.append(mean_int_nor_temp)
        # plateau level intensity calculated from last 10 frames of the frap curve
        plateau_int_temp = np.mean(pointer_pd['mean_int'][i][-10:])
        plateau_int.append(plateau_int_temp)
        plateau_int_nor_temp = (plateau_int_temp - frap_start_int_temp) / full_range_int
        plateau_int_nor.append(plateau_int_nor_temp)
        immobile_fraction_temp = 1 - plateau_int_nor_temp
        immobile_fraction.append(immobile_fraction_temp)
        # half intensity
        half_int_temp = 0.5 * (frap_start_int_temp + plateau_int_temp)
        half_int.append(half_int_temp)
        half_int_nor_temp = (half_int_temp - frap_start_int_temp) / full_range_int
        half_int_nor.append(half_int_nor_temp)
        # number of frames it take to reach half intensity
        half_frame_temp = dat.find_pos(half_int_temp, int_post)
        half_frame.append(half_frame_temp)
        # t_half (sec)
        t_half_temp = dat.get_time_length(frap_start_frame_temp,
                                          frap_start_frame_temp + half_frame_temp, acquire_time_tseries)
        t_half.append(t_half_temp)
        # initial slope calculated based on first 5 frames
        int_change = (pointer_pd['mean_int'][i][frap_start_frame_temp + 5] - frap_start_int_temp) / full_range_int
        t_change = dat.get_time_length(frap_start_frame_temp, frap_start_frame_temp + 5, acquire_time_tseries)
        slope_temp = 1.0 * (int_change / t_change)
        slope.append(slope_temp)

    pointer_pd = dat.add_columns(pointer_pd, ['int_curve_nor', 'frap_start_frame', 'imaging_length',
                                              'int_curve_pre', 'int_curve_post', 'int_curve_post_nor', 'real_time_post',
                                              'pre_bleach_int', 'frap_start_int', 'plateau_int', 'mobile_fraction',
                                              'immobile_fraction', 'half_int', 'half_int_nor', 'half_frame',
                                              't_half', 'ini_slope'],
                                 [mean_int_nor,  frap_start_frame, imaging_length,
                                  t_int_pre, t_int_post, t_int_post_nor, real_time_post,
                                  pre_bleach_int, frap_start_int, plateau_int, plateau_int_nor,
                                  immobile_fraction, half_int, half_int_nor, half_frame,
                                  t_half, slope])

    return pointer_pd


def bg_fitting_linear(pointer_pd):

    bg = pointer_pd['bg_int'][0]
    try:
        popt, _ = curve_fit(mat.linear, np.arange(0, len(bg), 1), bg)
        a, b = popt
    except RuntimeError:
        a = np.nan
        b = np.nan

    bg_fit = []
    for j in range(len(bg)):
        bg_fit.append(mat.linear(j, a, b))
    r2 = mat.r_square(bg, bg_fit)

    pointer_pd = dat.add_columns(pointer_pd, ['bg_linear_fit', 'bg_linear_r2', 'bg_linear_a', 'bg_linear_b'],
                                 [[bg_fit] * len(pointer_pd), [r2] * len(pointer_pd),
                                  [a] * len(pointer_pd), [b] * len(pointer_pd)])

    return pointer_pd


def pb_factor_fitting_single_exp(pointer_pd):
    pb = pointer_pd['pb_factor'][0]
    try:
        popt, _ = curve_fit(mat.single_exp_decay, np.arange(0, len(pb), 1), pb)
        a, b = popt
    except RuntimeError:
        a = np.nan
        b = np.nan

    pb_fit = []
    for j in range(len(pb)):
        pb_fit.append(mat.single_exp_decay(j, a, b))
    r2 = mat.r_square(pb, pb_fit)

    pointer_pd = dat.add_columns(pointer_pd, ['pb_single_exp_decay_fit', 'pb_single_exp_decay_r2',
                                              'pb_single_exp_decay_a', 'pb_single_exp_decay_b'],
                                 [[pb_fit] * len(pointer_pd), [r2] * len(pointer_pd),
                                  [a] * len(pointer_pd), [b] * len(pointer_pd)])

    return pointer_pd


def frap_fitting_single_exp(pointer_pd):

    single_exp_a = []
    single_exp_b = []
    single_exp_fit = []
    single_exp_r2 = []
    single_exp_t_half = []

    for i in range(len(pointer_pd)):
        try:
            popt, _ = curve_fit(mat.single_exp, pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i])
            a, b = popt
        except RuntimeError:
            a = np.nan
            b = np.nan

        y_fit = []
        for j in range(len(pointer_pd['real_time_post'][i])):
            y_fit.append(mat.single_exp(pointer_pd['real_time_post'][i][j], a, b))
        r2 = mat.r_square(pointer_pd['int_curve_post_nor'][i], y_fit)
        t_half_fit = np.log(0.5) / (-b)
        single_exp_a.append(a)
        single_exp_b.append(b)
        single_exp_fit.append(y_fit)
        single_exp_r2.append(r2)
        single_exp_t_half.append(t_half_fit)

    pointer_pd = dat.add_columns(pointer_pd, ['single_exp_fit', 'single_exp_r2', 'single_exp_a', 'single_exp_b',
                                              'single_exp_mobile_fraction', 'single_exp_t_half'],
                                 [single_exp_fit, single_exp_r2, single_exp_a, single_exp_b,
                                  single_exp_a, single_exp_t_half])

    return pointer_pd


def get_bleach_spots_coordinates(log_pd, store, cb, mode):
    if mode == 'single-raw':
        log_pd['x'] = log_pd['aim_x']
        log_pd['y'] = log_pd['aim_y']

    elif mode == 'single-offset':
        x_lst = []
        y_lst = []
        for i in range(len(log_pd)):
            aim_x = log_pd['aim_x'][i]
            aim_y = log_pd['aim_y'][i]
            bleach_frame = int(log_pd['bleach_frame'][i])

            post = store.get_image(cb.p(0).z(0).c(0).t(bleach_frame+4).build())
            pix_post = np.reshape(post.get_raw_pixels(), newshape=[post.get_height(), post.get_width()])
            pre = store.get_image(cb.p(0).z(0).c(0).t(bleach_frame).build())
            pix_pre = np.reshape(pre.get_raw_pixels(), newshape=[pre.get_height(), pre.get_width()])
            pix = pix_post - pix_pre + 100

            otsu_val = threshold_otsu(pix)
            otsu = pix > otsu_val
            bleach = binary_erosion(otsu)
            bleach = binary_erosion(bleach)
            bleach = binary_dilation(bleach)
            bleach = binary_dilation(bleach)

            # tried to use local maxima to find bleach spots, not good
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

            if len(peak_x) == 1:
                x_lst.append(peak_x[0])
                y_lst.append(peak_y[0])
            elif len(peak_x) > 1:
                x_closest, y_closest = dat.find_closest(aim_x, aim_y, peak_x, peak_y)
                x_lst.append(x_closest)
                y_lst.append(y_closest)
            else:
                x_lst.append(0)
                y_lst.append(0)

        log_pd['bleach_x'] = x_lst
        log_pd['bleach_y'] = y_lst
        log_pd['x_diff'] = log_pd['bleach_x'] - log_pd['aim_x']
        log_pd['y_diff'] = log_pd['aim_y'] - log_pd['bleach_y']

        log_pd['x'] = log_pd['bleach_x']
        log_pd['y'] = log_pd['bleach_y']
        log_pd.loc[log_pd.x == 0, 'x'] = log_pd[log_pd['x'] == 0]['aim_x']
        log_pd.loc[log_pd.y == 0, 'y'] = log_pd[log_pd['y'] == 0]['aim_y']

    return log_pd


def frap_filter(pointer_pd):
    # filter frap curves
    # 1) number of pre_bleach frame < 5
    # 2) total imaging length < 150
    # 3) does not find optional fit (single exponential)
    # 4) mobile fraction < 0 or mobile fraction > 1.5
    # 5) r2 of fit < 0.5

    frap_flt = []
    for i in range(len(pointer_pd)):
        if (pointer_pd['bleach_frame'][i] < 5) \
                | (pointer_pd['imaging_length'][i] < 150) \
                | (np.isnan(pointer_pd['single_exp_a'][i])) \
                | (pointer_pd['single_exp_a'][i] < 0) \
                | (pointer_pd['single_exp_a'][i] >= 1.5) \
                | (pointer_pd['single_exp_r2'][i] < 0.5):
            frap_flt.append(0)
        else:
            frap_flt.append(1)
    pointer_pd['frap_filter'] = frap_flt
    print("%d FRAP curves: less than 5 frames before photobleaching."
          % len(pointer_pd[pointer_pd['bleach_frame'] < 5]))
    print("%d FRAP curves: less than 150 frames in total."
          % len(pointer_pd[(pointer_pd['bleach_frame'] >= 5) & (pointer_pd['imaging_length'] < 150)]))
    print("%d FRAP curves: no optimal single exponential fit."
          % len(pointer_pd[(pointer_pd['bleach_frame'] >= 5) & (pointer_pd['imaging_length'] >= 150)
                           & (np.isnan(pointer_pd['single_exp_a']))]))
    print("%d FRAP curves: mobile fraction < 0 or >= 1.5."
          % len(pointer_pd[(pointer_pd['bleach_frame'] >= 5) & (pointer_pd['imaging_length'] >= 150)
                           & (~np.isnan(pointer_pd['single_exp_a']))
                           & ((pointer_pd['single_exp_a'] < 0) | (pointer_pd['single_exp_a'] >= 1.5))]))
    print("%d FRAP curves: r square of single exponential fit < 0.5"
          % len(pointer_pd[(pointer_pd['bleach_frame'] >= 5) & (pointer_pd['imaging_length'] >= 150)
                           & (~np.isnan(pointer_pd['single_exp_a'])) & (pointer_pd['single_exp_a'] > 0)
                           & (pointer_pd['single_exp_a'] < 1.5) & (pointer_pd['single_exp_r2'] < 0.5)]))

    return pointer_pd

