# ----------------------------
# FUNCTIONS for OUTPUT DISPLAY
# ----------------------------

import numpy as np
from matplotlib import cm
from vispy.color import Colormap


def num_color_colormap(cmap_name, num: int):
    """
    Generate num-color colormap from available matplotlib cmap.

    :param cmap_name: matplotlib cmap name
    :param num: non negative int, can not be zero
    :return: num_color_cmap: generated num-color colormap
             rgba: colormap array (without background)
    """
    cmap = cm.get_cmap(cmap_name)
    if num <= 0:
        raise ValueError("0 or negative values cannot be used to generate n-color colormap.")
    else:
        rgba = cmap(np.arange(0, 1, 1/num))
        rgba = np.insert(rgba, 0, [0.0, 0.0, 0.0, 0.0], axis=0)
        num_color_cmap = Colormap(rgba)

    return num_color_cmap, rgba


def sorted_num_color_colormap(num_color_rgba, pd, sort_name, obj_name):
    """
    Sort num-color colormap based on sort_name to display object obj_name.

    :param num_color_rgba: num-color colormap array
    :param pd: pandas.dataFrame with the same length as the colormap number
    :param sort_name: name of the column in pd used for sorting
    :param obj_name: name of the column in pd used for ploting
    :return: num_color_cmap: sorted colormap
             rgba: colormap array (without background)
    """
    pd_sort = pd.sort_values(by=sort_name).reset_index(drop=True)

    rgba = [num_color_rgba[0]]
    for i in pd_sort.sort_values(by=obj_name).index.tolist():
        rgba.append(num_color_rgba[i+1])
    num_color_cmap = Colormap(rgba)

    return num_color_cmap, rgba



