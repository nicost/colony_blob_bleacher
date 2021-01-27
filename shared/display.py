# ----------------------------
# FUNCTIONS for OUTPUT DISPLAY
# ----------------------------

import numpy as np
from matplotlib import cm
from vispy.color import Colormap
from matplotlib.colors import ListedColormap


def num_color_colormap(cmap_name, num: int, bg_color=[0.0, 0.0, 0.0, 0.0]):
    """
    Generate num-color colormap from available matplotlib cmap.

    :param bg_color:
    :param cmap_name: matplotlib cmap name
    :param num: non negative int, can not be zero
    :return: cmap_napari: generated num-color colormap for napari display
             cmap_plt: generated num-color colormap for matplotlib display
             rgba: colormap array (without background)
    """
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
    :param obj_name: name of the column in pd used for ploting
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



