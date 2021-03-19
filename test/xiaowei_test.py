import numpy as np
import pandas as pd
import napari
from pycromanager import Bridge
from matplotlib.backends.qt_compat import QtCore, QtWidgets

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import FigureCanvas
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure
from vispy.color import Colormap
from shared.find_organelles import find_organelle, organelle_analysis, find_nuclear_nucleoli, nuclear_analysis
from skimage.measure import label, regionprops_table, regionprops
from shared.find_blobs import select
from skimage.morphology import medial_axis
import shared.analysis as ana
import shared.dataframe as dat
import shared.display as dis
import shared.objects as obj
import shared.bleach_points as ble
import shared.math_functions as mat
import os

# --------------------------
# PARAMETERS allow change
# --------------------------
# Please changes
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/" \
                "20210203_CBB_nucleoliArsAndHeatshockTreatment/data/WT1/C2-Site_15_1"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/" \
                "20210203_CBB_nucleoliArsAndHeatshockTreatment/dataAnalysis1/WT1/C2-Site_15_1"
analyze_organelle = 'nucleoli'  # only accepts 'sg' or 'nucleoli'
frap_start_delay = 6  # 50ms default = 4; 100ms default = 5; 200ms default = 6
display_mode = 'Y'  # only accepts 'N' or 'Y'
display_sort = 'pre_bleach_int'  # accepts 'na' or other features like 'sg_size'
display_data = 'local'  # only accepts 'bg' or 'local'

# values for analysis
data_c = 0
pos = 0
num_dilation = 3  # number of dilation from the coordinate;
# determines analysis size of the analysis spots; default = 3

# presets
if analyze_organelle == 'sg':
    thresholding = 'na'
    # global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
    min_size = 5  # minimum size; sg default = 5
    max_size = 200  # maximum size; sg default = 200
else:  # for 'nucleoli'
    thresholding = 'local-nucleoli'
    # global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
    min_size = 10  # minimum size; nucleoli default = 10
    max_size = 1000  # maximum size; nucleoli default = 1000;
    # larger ones are generally cells without nucleoli

# modes
mode_bleach_detection = 'single-offset'  # only accepts 'single-raw' or 'single-offset'
frap_start_mode = 'min'  # only accepts 'delay' or 'min'
fitting_mode = 'single_exp'  # accepts 'single_exp', 'double_exp', 'soumpasis', 'ellenberg', 'optimal'

"""
# ---------------------------------------------------------------------------------------------------
# PLEASE DO NOT CHANGE AFTER THIS
# ---------------------------------------------------------------------------------------------------
"""

# --------------------------
# LOAD MOVIE
# --------------------------
print("### Load movie ...")
data_log = pd.DataFrame({'pos': [pos]})

# build up pycromanager bridge
# first start up Micro-Manager (needs to be compatible version)
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
# load time series data
store = mm.data().load_data(data_path, True)
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)
# get max_t and acquisition time
max_t = store.get_max_indices().get_t()
pixels_tseries = dat.get_pixels_tseries(store, cb, data_c)
acquire_time_tseries, real_time = dat.get_time_tseries(store, cb)
data_log['acquire_time'] = [acquire_time_tseries]
data_log['real_time'] = [real_time]

# --------------------------------------
# ORGANELLE ANALYSIS based on time 0
# --------------------------------------
print("### Image analysis: %s detection based on time 0 ..." % analyze_organelle)

# reference image of time 0
# if decide to use other image as ref_image
# be sure to check photobleaching correction for all reported intensities
temp = store.get_image(cb.c(data_c).t(0).build())
pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

_, organelle = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
label_organelle = label(organelle, connectivity=1)
blobs = regionprops(label_organelle)
selected = select(blobs, 'centroid', temp.get_width() / 10, 0.9 * temp.get_width())
print(selected)

spots = obj.select_random_in_label(label_organelle, 1)

if display_mode == 'Y':
    print("### Output display ...")

    with napari.gui_qt():
        # embed mpl widget in napari viewer
        mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        [ax1, ax2, ax3] = mpl_widget.figure.subplots(nrows=1, ncols=3)
        viewer = napari.Viewer()
        viewer.window.add_dock_widget(mpl_widget)

        # napari display
        # Layer1: data
        # display time series movies in napari main viewer
        viewer.add_image(pix, name='data')

        points = np.column_stack((spots[1], spots[0]))
        size = [3] * len(points)
        viewer.add_points(points, name='aim points', size=size, edge_color='r', face_color='r')
