import numpy as np
from matplotlib import cm
from vispy.color import Colormap
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for OUTPUT DISPLAY
# ---------------------------------------------------------------------------------------------------

General functions:

    num_color_colormap
        FUNCTION: generate num-color colormap from available matplotlib cmap
        SYNTAX:   num_color_colormap(cmap_name: str, num: int, bg_color=None)
    
    sorted_num_color_colormap
        FUNCTION: sort num-color colormap based on sort_name to display object obj_name
        SYNTAX:   sorted_num_color_colormap(num_color_rgba, dataframe: pd.DataFrame, sort_name: str, 
                  obj_name: str)
        NOTE:     not good, stop using this function, find time to replace all the used ones 
                  (frap_analysis.py)
    
    napari_movie
        FUNCTION: transform uManager movie for napari display
        SYNTAX:   napari_movie(store, cb)

Specific output/display (save space in the main script):
    
    plot_offset_map
        FUNCTION: plot and save the offset map for detected bleach spots (for FRAP analysis)
        SYNTAX:   plot_offset_map(pointer_pd: pd.DataFrame, storage_path: str)
    
    plot_raw_intensity
        FUNCTION: plot and save raw intensity measured from bleach spots, ctrl spots and background 
                  (for FRAP analysis)
        SYNTAX:   plot_raw_intensity(pointer_pd: pd.DataFrame, ctrl_pd: pd.DataFrame, storage_path: str)
    
    plot_pb_factor
        FUNCTION: plot and save photobleaching factor curve and single exponential decay fitting 
                  curve calculated from control spots (for FRAP analysis)
        SYNTAX:   plot_pb_factor(pointer_pd: pd.DataFrame, storage_path: str)
    
    plot_corrected_intensity
        FUNCTION: plot and save corrected intensity measured from bleach spots after both background and 
                  photobleaching correction (for FRAP analysis)
        SYNTAX:   plot_corrected_intensity(pointer_pd: pd.DataFrame, storage_path: str)

    plot_normalized_frap
        FUNCTION: plot and save normalized FRAP curves measured from bleach spots (for FRAP analysis)
        SYNTAX:   plot_normalized_frap(pointer_pd: pd.DataFrame, storage_path: str)

    plot_frap_fitting
        FUNCTION: plot and save normalized FRAP curves and corresponding single exponential fitting 
                  measured from good bleach spots (for FRAP analysis)
        SYNTAX:   plot_frap_fitting(pointer_pd: pd.DataFrame, storage_path: str)
    
    get_p
        FUNCTION: calculate pair wise KS test p-value for -ln(p) plot
        SYNTAX:   get_p(data1: pd.DataFrame, data2: pd.DataFrame, feature: str, inc: int, limit: int, repeat: int)
    
    get_x
        FUNCTION: create pair-wise x value for -ln(p) plot
        SYNTAX:   get_x(inc: int, limit: int, repeat: int, offset: float)
    
    get_phenotype
        FUNCTION: get phenotype value (average -ln(p) value)
        SYNTAX:   get_phenotype(data1: pd.DataFrame, data2: pd.DataFrame, feature: str, limit: int, repeat: int)
    
    plot_minus_ln_p
        FUNCTION: generate -ln(p) plot for given feature
        SYNTAX:   plot_minus_ln_p(inc: int, limit: int, repeat: int, feature: str, data_pd: pd.DataFrame, ctrl_lst: 
                  list, sample: str, save_path: str)
    
    plot_violin
        FUNCTION: generate violin plot for given feature
        SYNTAX:   plot_violin(feature: str, pd_data: pd.DataFrame, save_path: str, sample_name: str)
    
"""

# ---------------------------------------------------------------------------------------------------
# GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------


def num_color_colormap(cmap_name: str, num: int, bg_color=None):
    """
    Generate num-color colormap from available matplotlib cmap

    :param cmap_name: str, matplotlib cmap name
    :param num: int, positive, number of colors used for colormap generation
    :param bg_color: background color rgba value, default = transparent black
    :return: cmap_napari: generated num-color colormap for napari display
             cmap_plt: generated num-color colormap for matplotlib display
             rgba: colormap array, note that rgba[0] = bg_color
    """
    if bg_color is None:
        bg_color = [0.0, 0.0, 0.0, 0.0]

    cmap = cm.get_cmap(cmap_name)
    if num <= 0:
        raise ValueError("0 or negative values cannot be used to generate n-color colormap.")
    else:
        rgba = cmap(np.arange(0, 1, 1/num))
        rgba = np.insert(rgba, 0, bg_color, axis=0)
        cmap_napari = Colormap(rgba)
        cmap_plt = ListedColormap(rgba)

    return cmap_napari, cmap_plt, rgba


def sorted_num_color_colormap(num_color_rgba, dataframe: pd.DataFrame, sort_name: str, obj_name: str):
    """
    Sort num-color colormap based on sort_name to display object obj_name

    :param num_color_rgba: num-color colormap array
    :param dataframe: pd.DataFrame with the same length as the colormap number
    :param sort_name: name of the column in dataframe used for sorting
    :param obj_name: name of the column in dataframe used for plotting
    :return: cmap_napari: generated num-color colormap for napari display
             cmap_plt: generated num-color colormap for matplotlib display
             rgba: colormap array, note that rgba[0] = bg_color
    """
    pd_sort = dataframe.sort_values(by=sort_name).reset_index(drop=True)

    rgba = [num_color_rgba[0]]
    for i in pd_sort.sort_values(by=obj_name).index.tolist():
        rgba.append(num_color_rgba[i+1])
    cmap_napari = Colormap(rgba)
    cmap_plt = ListedColormap(rgba)

    return cmap_napari, cmap_plt, rgba


def napari_movie(store, cb):
    """
    Transform uManager movie for napari display

    :param store: store = mm.data().load_data(data_path, True)
    :param cb: cb = mm.data().get_coords_builder()
    :return: mov: movie used for napari display

    """
    # create stack for time series
    pixels_tseries = []
    max_t = store.get_max_indices().get_t()
    for t in range(0, max_t):
        img = store.get_image(cb.t(t).build())
        pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
        pixels_tseries.append(pixels)
    mov = np.stack(pixels_tseries, axis=0)

    return mov

# ---------------------------------------------------------------------------------------------------
# SPECIFIC FUNCTIONS
# ---------------------------------------------------------------------------------------------------


def plot_offset_map(pointer_pd: pd.DataFrame, fitting_mode: str, prefix: str, storage_path: str):
    """
    Plot and save the offset map for detected bleach spots (for FRAP analysis)

    Aim spots (coordinates get from .log file) are centered to (0,0), non (0,0) end of the lines indicate
    location of detected bleach spots relative to aim spots.

    Red: offset measured from good bleach spots
    Blue: offset measured from filtered bleach spots (didn't pass FRAP curve quality control)

    :param pointer_pd: pd.DataFrame, requires columns 'frap_filter', 'x_diff', 'y_diff'
                'frap_filter': if the FRAP curve passes the FRAP curve quality control or not
                'x_diff': difference in x coordinates between detected bleach spots and aim spots
                'y_diff': difference in y coordinates between detected bleach spots and aim spots
    :param fitting_mode: str, fitting functions
    :param prefix: str, storing prefix
    :param storage_path: str, directory to save image

    """
    m = 0
    n = 0
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter_%s' % fitting_mode][i] == 0:
            if m == 0:
                plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color='#1E90FF', alpha=0.5,
                         label='filtered ones')
                m = m + 1
            else:
                plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color='#1E90FF', alpha=0.5)
        else:
            if n == 0:
                plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color=(0.85, 0.35, 0.25),
                         alpha=0.5, label='good ones')
                n = n + 1
            else:
                plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color=(0.85, 0.35, 0.25),
                         alpha=0.5)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.xlabel('x offset (pixel)')
    plt.ylabel('y offset (pixel)')
    plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
    plt.savefig('%s/%s_offset_map.pdf' % (storage_path, prefix))
    plt.close()


def plot_raw_intensity(pointer_pd: pd.DataFrame, ctrl_pd: pd.DataFrame, fitting_mode: str, prefix: str,
                       storage_path: str):
    """
    Plot and save raw intensity measured from bleach spots, ctrl spots and background (for FRAP analysis)

    Red: raw intensity measured from good bleach spots
    Blue: raw intensity measured from filtered bleach spots (didn't pass FRAP curve quality control)
    Grey: raw intensity measured from control spots
    Black: raw intensity of background

    :param pointer_pd: pd.DataFrame, requires columns 'frap_filter', 'raw_int', 'bg_int',
                'bg_linear_fit', 'bg_linear_a'
                'frap_filter': if the FRAP curve passes the FRAP curve quality control or not
                'raw_int': raw intensity of each bleach spot
                'bg_int': background intensity
                'bg_linear_fit': values from linear fit (a * x + b) of background intensity
                'bg_linear_a': parameter a of linear fit (a * x + b) of background intensity
    :param ctrl_pd: pd.DataFrame, requires columns 'raw_int'
                'raw_int': raw intensity of each control spot
    :param fitting_mode: str, fitting functions
    :param prefix: str, storing prefix
    :param storage_path: str, directory to save image

    """
    m = 0
    n = 0
    j = 0
    plt.subplots(figsize=(6, 4))
    if len(pointer_pd) != 0:
        plt.plot(pointer_pd['bg_int'][0], color=(0, 0, 0), alpha=0.7, label='bg')
        if ~np.isnan(pointer_pd['bg_linear_a'][0]):
            plt.plot(pointer_pd['bg_linear_fit'][0], '--', color=(0, 0, 0), alpha=0.7)
        for i in range(len(ctrl_pd)):
            if j == 0:
                plt.plot(ctrl_pd['raw_int'][i], color=(0.7, 0.7, 0.7), alpha=0.5, label='ctrl')
                j = j + 1
            else:
                plt.plot(ctrl_pd['raw_int'][i], color=(0.7, 0.7, 0.7), alpha=0.5)
        for i in range(len(pointer_pd)):
            if pointer_pd['frap_filter_%s' % fitting_mode][i] == 0:
                if m == 0:
                    plt.plot(pointer_pd['raw_int'][i], color='#1E90FF', alpha=0.7, label='filtered ones')
                    m = m + 1
                else:
                    plt.plot(pointer_pd['raw_int'][i], color='#1E90FF', alpha=0.7)
            else:
                if n == 0:
                    plt.plot(pointer_pd['raw_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7, label='good ones')
                    n = n + 1
                else:
                    plt.plot(pointer_pd['raw_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7)
        plt.xlabel('time (frame)')
        plt.ylabel('raw intensity (AU)')
        plt.legend(loc=2, bbox_to_anchor=(0.65, 0.99))
        plt.savefig('%s/%s_raw_intensity.pdf' % (storage_path, prefix))
        plt.close()


def plot_pb_factor(pointer_pd: pd.DataFrame, prefix: str, storage_path: str):
    """
    Plot and save photobleaching factor curve and single exponential decay fitting curve calculated
    from control spots (for FRAP analysis)

    :param pointer_pd: pd.DataFrame, requires columns 'pb_factor', 'pb_single_exp_decay_fit',
                'pb_single_exp_decay_a'
                'pb_factor': photobleaching factor calculated from control spots
                'pb_single_exp_decay_fit': values from single exponential decay fit
                    (1 - a * (1 - np.exp(-b * x))) of photobleaching factor
                'pb_single_exp_decay_a': parameter a of single exponential decay fit
                    (1 - a * (1 - np.exp(-b * x))) of photobleaching factor
    :param prefix: str, storing prefix
    :param storage_path: str, directory to save image

    """
    plt.subplots(figsize=(6, 4))
    if len(pointer_pd) != 0:
        plt.plot(pointer_pd['pb_factor'][0], color=(0.8, 0.8, 0.8))
        if ~np.isnan(pointer_pd['pb_single_exp_decay_a'][0]):
            plt.plot(pointer_pd['pb_single_exp_decay_fit'][0], '--', color=(0.8, 0.8, 0.8))
        plt.xlabel('time (frame)')
        plt.ylabel('photobleaching factor')
        plt.savefig('%s/%s_pb_factor.pdf' % (storage_path, prefix))
        plt.close()


def plot_corrected_intensity(pointer_pd: pd.DataFrame, fitting_mode: str, prefix: str, storage_path: str):
    """
    Plot and save corrected intensity measured from bleach spots after both background and photobleaching
    correction (for FRAP analysis)

    Red: corrected intensity measured from good bleach spots
    Blue: corrected intensity measured from filtered bleach spots (didn't pass FRAP curve quality control)

    :param pointer_pd: pd.DataFrame, requires columns 'frap_filter', 'mean_int'
                'frap_filter': if the FRAP curve passes the FRAP curve quality control or not
                'mean_int': double corrected mean intensity of each bleach spot
    :param fitting_mode: str, fitting functions
    :param prefix: str, storing prefix
    :param storage_path: str, directory to save image

    """
    m = 0
    n = 0
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter_%s' % fitting_mode][i] == 0:
            if m == 0:
                plt.plot(pointer_pd['mean_int'][i], color='#1E90FF', alpha=0.7, label='filtered ones')
                m = m + 1
            else:
                plt.plot(pointer_pd['mean_int'][i], color='#1E90FF', alpha=0.7)
        else:
            if n == 0:
                plt.plot(pointer_pd['mean_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7, label='good ones')
                n = n + 1
            else:
                plt.plot(pointer_pd['mean_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7)
    plt.xlabel('time (frame)')
    plt.ylabel('bg/pb corrected intensity (AU)')
    plt.legend(loc=2, bbox_to_anchor=(0.65, 0.99))
    plt.savefig('%s/%s_double_corrected_intensity.pdf' % (storage_path, prefix))
    plt.close()


def plot_normalized_frap(pointer_pd: pd.DataFrame, fitting_mode: str, prefix: str, storage_path: str):
    """
    Plot and save normalized FRAP curves measured from bleach spots (for FRAP analysis)

    Red: normalized FRAP curves measured from good bleach spots
    Blue: normalized FRAP curves measured from filtered bleach spots (didn't pass FRAP curve quality control)

    :param pointer_pd: pd.DataFrame, requires columns 'frap_filter', 'real_time_post', 'int_curve_post_nor'
                'frap_filter': if the FRAP curve passes the FRAP curve quality control or not
                'real_time_post': time series after frap_start_frame (included) displayed in second
                'int_curve_post_nor': normalized double corrected intensity after frap_start_frame (included)
    :param fitting_mode: str, fitting functions
    :param prefix: str, storing prefix
    :param storage_path: str, directory to save image

    """
    m = 0
    n = 0
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter_%s' % fitting_mode][i] == 0:
            if m == 0:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color='#1E90FF', alpha=0.7, label='filtered ones')
                m = m + 1
            else:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color='#1E90FF', alpha=0.7)
        else:
            if n == 0:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color=(0.85, 0.35, 0.25), alpha=0.7, label='good ones')
                n = n + 1
            else:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color=(0.85, 0.35, 0.25), alpha=0.7)
    plt.xlabel('time (s)')
    plt.ylabel('normalized intensity (AU)')
    plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
    plt.savefig('%s/%s_normalized_frap_curves.pdf' % (storage_path, prefix))
    plt.close()


def plot_frap_fitting(pointer_pd: pd.DataFrame, fitting_mode: str, prefix: str, storage_path: str):
    """
    Plot and save normalized FRAP curves and corresponding single exponential fitting measured from good
    bleach spots (for FRAP analysis)

    Color: indicate each single curve (do not correspond to napari viewer)
    Solid line: normalized FRAP curves
    Dotted line: single exponential fitting curves

    :param pointer_pd: pd.DataFrame, requires columns 'frap_filter', 'real_time_post',
                'int_curve_post_nor', 'single_exp_fit'
                'frap_filter': if the FRAP curve passes the FRAP curve quality control or not
                'real_time_post': time series after frap_start_frame (included) displayed in second
                'int_curve_post_nor': normalized double corrected intensity after frap_start_frame (included)
                'single_exp_fit': values from single exponential fit (a * (1 - np.exp(-b * x))) of
                    int_curve_post_nor
    :param fitting_mode: str, fitting functions
    :param prefix: str, storing prefix
    :param storage_path: str, directory to save image

    """
    if len(pointer_pd) != 0:
        cmap1 = 'viridis'
        cmap1_rgba = num_color_colormap(cmap1, len(pointer_pd))[2]
        cmap2_rgba = num_color_colormap(cmap1, 6)[2]
        plt.subplots(figsize=(6, 4))
        for i in range(len(pointer_pd)):
            if pointer_pd['frap_filter_%s' % fitting_mode][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color=cmap1_rgba[i + 1], alpha=0.7)
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['%s_fit' % fitting_mode][i], '--',
                         color=cmap1_rgba[i + 1], alpha=0.7)
        plt.xlabel('time (s)')
        plt.ylabel('normalized intensity (AU)')
        plt.savefig('%s/%s_normalized_frap_curves_filtered.pdf' % (storage_path, prefix))
        plt.close()

        for i in range(len(pointer_pd)):
            plt.subplots(figsize=(6, 4))
            if pointer_pd['frap_filter_%s' % fitting_mode][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color=(0.85, 0.35, 0.25), alpha=0.7, label='data')
            else:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                         color='#1E90FF', alpha=0.7, label='data')
            if pointer_pd['frap_filter_single_exp'][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['single_exp_fit'][i], '--',
                         color=cmap2_rgba[1], alpha=0.7, label='single_exp')
            if pointer_pd['frap_filter_soumpasis'][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['soumpasis_fit'][i], '--',
                         color=cmap2_rgba[2], alpha=0.7, label='soumpasis')
            if pointer_pd['frap_filter_double_exp'][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['double_exp_fit'][i], '--',
                         color=cmap2_rgba[3], alpha=0.7, label='double_exp')
            if pointer_pd['frap_filter_ellenberg'][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['ellenberg_fit'][i], '--',
                         color=cmap2_rgba[4], alpha=0.7, label='ellenberg')
            if pointer_pd['frap_filter_optimal'][i] == 1:
                plt.plot(pointer_pd['real_time_post'][i], pointer_pd['optimal_fit'][i], '--',
                         color=cmap2_rgba[5], alpha=0.7, label='optimal')
            plt.xlabel('time (s)')
            plt.ylabel('normalized intensity (AU)')
            plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
            plt.savefig('%s/%s_frap_curves_filtered_%d.pdf' % (storage_path, prefix, i))
            plt.close()


def get_p(data1: pd.DataFrame, data2: pd.DataFrame, feature: str, inc: int, limit: int, repeat: int):
    """
    Calculate pair wise KS test p-value for -ln(p) plot

    :param data1: pd.DataFrame, data1
    :param data2: pd.DataFrame, data2
    :param feature: str, comparing feature, column name
    :param inc: int, increment (generally 5)
    :param limit: int, upper limit of the plot
    :param repeat: int, how many runs to be calculated per condition (generally 50)
    :return: out: list, list of p-value
    """
    out = []
    for i in np.arange(inc, limit, inc):
        for j in range(repeat):
            p = -np.log(ks_2samp(data1[feature].sample(n=i).tolist(), data2[feature].sample(n=i).tolist())[1])
            out.append(p)
    return out


def get_x(inc: int, limit: int, repeat: int, offset: float):
    """
    Create pair-wise x value for -ln(p) plot

    :param inc: int, increment (generally 5)
    :param limit: int, upper limit of the plot
    :param repeat: int, how many runs to be calculated per condition (generally 50)
    :param offset: float, offset applied to avoid different datasets overlapping
    :return: out: list, list of x values
    """
    out = []
    for i in np.arange(inc, limit, inc):
        for j in range(repeat):
            x = i+offset
            out.append(x)
    return out


def get_phenotype(data1: pd.DataFrame, data2: pd.DataFrame, feature: str, limit: int, repeat: int):
    """
    get phenotype value (average -ln(p) value)

    :param data1: pd.DataFrame, data1
    :param data2: pd.DataFrame, data2
    :param feature: str, comparing feature, column name
    :param limit: int, upper limit of the plot
    :param repeat: int, how many runs to be calculated per condition (generally 50)
    :return:
    """
    pd_data1 = data1 if feature != 'circ' else data1[data1['size'] > 50]
    pd_data2 = data2 if feature != 'circ' else data2[data2['size'] > 50]

    minimum = np.min([len(pd_data1), len(pd_data2)])
    limit = minimum if (minimum < limit) else limit

    p_lst = []
    for j in range(repeat):
        p = -np.log(ks_2samp(pd_data1[feature].sample(n=limit).tolist(), pd_data2[feature].sample(n=limit).tolist())[1])
        p_lst.append(p)

    return limit, np.mean(p_lst)


def plot_minus_ln_p(inc: int, limit: int, repeat: int, feature: str, data_pd: pd.DataFrame, ctrl_lst: list, sample: str,
                    save_path: str):
    """
    Generate -ln(p) plot for given feature

    :param inc: int, increment (generally 5)
    :param limit: int, upper limit of the plot
    :param repeat: int, how many runs to be calculated per condition (generally 50)
    :param feature: str, comparing feature, column name
    :param data_pd: pd.DataFrame, total sample
    :param ctrl_lst: list, list of control wells
    :param sample: str, sample name
    :param save_path: str, saving path
    :return:
    """
    sample_lst = ctrl_lst + [sample, 'WT']
    for i in sample_lst:
        n_curve = len(data_pd[data_pd['sample'] == i]) if feature != 'circ' else \
            len(data_pd[(data_pd['sample'] == i) & (data_pd['size'] > 50)])
        if n_curve < limit:
            limit = n_curve

    x = np.arange(limit+5)
    plt.figure(figsize=(15, 4))
    n_middle = (len(ctrl_lst) + 1) // 2
    pd_WT = data_pd[data_pd['sample'] == 'WT'] if feature != 'circ' else \
        data_pd[(data_pd['sample'] == 'WT') & (data_pd['size'] > 50)]
    for i in range(len(ctrl_lst)):
        pd_ctrl = data_pd[data_pd['sample'] == ctrl_lst[i]] if feature != 'circ' else \
            data_pd[(data_pd['sample'] == ctrl_lst[i]) & (data_pd['size'] > 50)]
        plt.scatter(get_x(inc, limit, repeat, (i-n_middle)/2.0),
                    get_p(pd_ctrl, pd_WT, feature, inc, limit, repeat),
                    alpha=0.5, s=5, c='#40E0D0', label=ctrl_lst[i])

    pd_sample = data_pd[data_pd['sample'] == sample] if feature != 'circ' else \
        data_pd[(data_pd['sample'] == sample) & (data_pd['size'] > 50)]
    plt.scatter(get_x(inc, limit, repeat, (len(ctrl_lst)-n_middle)/2.0),
                get_p(pd_sample, pd_WT, feature, inc, limit, repeat),
                alpha=0.5, s=5, c='#FF4500', label=sample)
    plt.plot(x, 0 * x + 5, linestyle='--', color='#696969')
    plt.xlabel('number of traces')
    plt.ylabel('%s -ln(KS p-value)' % feature)
    plt.savefig('%s%s_%s_p.pdf' % (save_path, sample, feature))
    plt.close()


def plot_violin(feature: str, pd_data: pd.DataFrame, save_path: str, sample_name: str):
    """
    Generate violin plot for given feature

    :param feature: str, comparing feature, column name
    :param pd_data: pd.DataFrame, all the data
    :param save_path: str, save location
    :param sample_name: str, treatment sample name
    :return:
    """
    plt.figure(figsize=(12, 4), dpi=80)
    sns.violinplot(x='sample', y=feature, data=pd_data, notch=False)
    plt.savefig('%s%s_%s.pdf' % (save_path, sample_name, feature))
    plt.close()
