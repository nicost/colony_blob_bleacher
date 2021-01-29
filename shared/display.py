# ----------------------------
# FUNCTIONS for OUTPUT DISPLAY
# ----------------------------

import numpy as np
from matplotlib import cm
from vispy.color import Colormap
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


def num_color_colormap(cmap_name, num: int, bg_color=None):
    """
    Generate num-color colormap from available matplotlib cmap.

    :param bg_color:
    :param cmap_name: matplotlib cmap name
    :param num: non negative int, can not be zero
    :return: cmap_napari: generated num-color colormap for napari display
             cmap_plt: generated num-color colormap for matplotlib display
             rgba: colormap array (without background)
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


def sorted_num_color_colormap(num_color_rgba, pd, sort_name, obj_name):
    """
    Sort num-color colormap based on sort_name to display object obj_name.

    :param num_color_rgba: num-color colormap array
    :param pd: pandas.dataFrame with the same length as the colormap number
    :param sort_name: name of the column in pd used for sorting
    :param obj_name: name of the column in pd used for plotting
    :return: cmap_napari: generated num-color colormap for napari display
             cmap_plt: generated num-color colormap for matplotlib display
             rgba: colormap array (without background)
    """
    pd_sort = pd.sort_values(by=sort_name).reset_index(drop=True)

    rgba = [num_color_rgba[0]]
    for i in pd_sort.sort_values(by=obj_name).index.tolist():
        rgba.append(num_color_rgba[i+1])
    cmap_napari = Colormap(rgba)
    cmap_plt = ListedColormap(rgba)

    return cmap_napari, cmap_plt, rgba


def plot_offset_map(pointer_pd, storage_path):
    """
    Plot and save the offset map for detected bleach spots (for FRAP analysis)

    Aim spots (coordinates get from .log file) are centered to (0,0), non (0,0) end of the lines indicate
    location of detected bleach spots relative to aim spots.

    Red: offset measured from good bleach spots
    Blue: offset measured from filtered bleach spots (didn't pass FRAP curve quality control)

    :param pointer_pd: pd.DataFrame(), requires columns 'frap_filter', 'x_diff', 'y_diff'
    :param storage_path: directory to save image
    :return:
    """
    m = 0
    n = 0
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter'][i] == 0:
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
    plt.savefig('%s/offset_map.pdf' % storage_path)


def plot_raw_intensity(pointer_pd, ctrl_pd, storage_path):
    """
    Plot and save raw intensity measured from bleach spots, ctrl spots and background (for FRAP analysis)

    Red: raw intensity measured from good bleach spots
    Blue: raw intensity measured from filtered bleach spots (didn't pass FRAP curve quality control)
    Grey: raw intensity measured from control spots
    Black: raw intensity of background

    :param pointer_pd: pd.DataFrame(), requires columns 'frap_filter', 'raw_int', 'bg_int',
        'bg_linear_fit', 'bg_linear_a'
    :param ctrl_pd: pd.DataFrame(), requires columns 'raw_int'
    :param storage_path: directory to save image
    :return:
    """
    m = 0
    n = 0
    j = 0
    plt.subplots(figsize=(6, 4))
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
        if pointer_pd['frap_filter'][i] == 0:
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
    plt.savefig('%s/raw_intensity.pdf' % storage_path)


def plot_pb_factor(pointer_pd, storage_path):
    """
    Plot and save photobleaching factor curve and single exponential decay fitting curve calculated
    from control spots (for FRAP analysis)

    :param pointer_pd: pd.DataFrame(), requires columns 'pb_factor', 'pb_single_exp_decay_fit',
        'pb_single_exp_decay_a'
    :param storage_path: directory to save image
    :return:
    """
    plt.subplots(figsize=(6, 4))
    plt.plot(pointer_pd['pb_factor'][0], color=(0.8, 0.8, 0.8))
    if ~np.isnan(pointer_pd['pb_single_exp_decay_a'][0]):
        plt.plot(pointer_pd['pb_single_exp_decay_fit'][0], '--', color=(0.8, 0.8, 0.8))
    plt.xlabel('time (frame)')
    plt.ylabel('photobleaching factor')
    plt.savefig('%s/pb_factor.pdf' % storage_path)


def plot_corrected_intensity(pointer_pd, storage_path):
    """
    Plot and save corrected intensity measured from bleach spots after both background and photobleaching
    correction (for FRAP analysis)

    Red: corrected intensity measured from good bleach spots
    Blue: corrected intensity measured from filtered bleach spots (didn't pass FRAP curve quality control)

    :param pointer_pd: pd.DataFrame(), requires columns 'frap_filter', 'mean_int'
    :param storage_path: directory to save image
    :return:
    """
    m = 0
    n = 0
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter'][i] == 0:
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
    plt.savefig('%s/double_corrected_intensity.pdf' % storage_path)


def plot_normalized_frap(pointer_pd, storage_path):
    """
    Plot and save normalized FRAP curves measured from bleach spots (for FRAP analysis)

    Red: normalized FRAP curves measured from good bleach spots
    Blue: normalized FRAP curves measured from filtered bleach spots (didn't pass FRAP curve quality control)

    :param pointer_pd: pd.DataFrame(), requires columns 'frap_filter', 'real_time_post', 'int_curve_post_nor'
    :param storage_path: directory to save image
    :return:
    """
    m = 0
    n = 0
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter'][i] == 0:
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
    plt.savefig('%s/normalized_frap_curves.pdf' % storage_path)


def plot_fitting(pointer_pd, storage_path):
    """
    Plot and save normalized FRAP curves and corresponding single exponential fitting measured from good
    bleach spots (for FRAP analysis)

    Color: indicate each single curve (do not correspond to napari viewer)
    Solid line: normalized FRAP curves
    Dotted line: single exponential fitting curves

    :param pointer_pd: pd.DataFrame(), requires columns 'frap_filter', 'real_time_post',
        'int_curve_post_nor', 'single_exp_fit'
    :param storage_path: directory to save image
    :return:
    """
    cmap1 = 'viridis'
    cmap1_rgba = num_color_colormap(cmap1, len(pointer_pd))[2]
    plt.subplots(figsize=(6, 4))
    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter'][i] == 1:
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                     color=cmap1_rgba[i + 1], alpha=0.7)
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['single_exp_fit'][i], '--',
                     color=cmap1_rgba[i + 1], alpha=0.7)
    plt.xlabel('time (s)')
    plt.ylabel('normalized intensity (AU)')
    plt.savefig('%s/normalized_frap_curves_filtered.pdf' % storage_path)

    for i in range(len(pointer_pd)):
        if pointer_pd['frap_filter'][i] == 1:
            plt.subplots(figsize=(6, 4))
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                     color=cmap1_rgba[i + 1], alpha=0.7)
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['single_exp_fit'][i], '--',
                     color=cmap1_rgba[i + 1], alpha=0.7)
            plt.xlabel('time (s)')
            plt.ylabel('normalized intensity (AU)')
            plt.savefig('%s/frap_curves_filtered_%d.pdf' % (storage_path, i))