# --------------------------
# IMPORTS
# --------------------------
import numpy as np
import pandas as pd
import napari
from pycromanager import Bridge

from matplotlib.backends.qt_compat import QtCore, QtWidgets
if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure
from vispy.color import Colormap
import collections

from skimage.measure import label, regionprops

from shared.find_organelles import find_nucleoli
import shared.analysis as ana
import shared.display as dis
import shared.dataframe as dat
import shared.objects as obj

# --------------------------
# PARAMETERS
# --------------------------
# data source
# data_path = "C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoili-bleaching-4x\\PythonAcq1\\AutoBleach_15"
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/TestedData/20201216/Ctrl-2DG-CCCP-36pos_partial/exp_37"

# values
thresholding = 'local'  # global thresholding method; choose in between 'na','otsu' and 'yen'; default = 'na'
min_size = 10  # minimum nucleoli size; default = 10
max_size = 1000  # maximum nucleoli size; default = 1000;
                 # larger ones are generally cells without nucleoli
num_dilation = 3  # number of dilation from the coordinate;
                  # determines analysis size of the analysis spots; default = 3
x_shift = 0  # positive: right; default = 0
y_shift = 0  # positive: up; default = 0

# colormap
violet_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129 / 255, 55 / 255, 114 / 255, 1.0]])
red_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])

# --------------------------
# LOAD MOVIE
# --------------------------
print("### Load movie ...")
# build up pycromanager bridge
# first start up Micro-Manager (needs to be compatible version)
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()

# load time series data
store = mm.data().load_data(data_path, True)
max_t = store.get_max_indices().get_t()
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

# ------------------------------
# IMAGE ANALYSIS based on time 0
# ------------------------------
print("### Image analysis based on time 0 ...")

# test image of time 0
t0 = store.get_image(cb.t(0).build())
t0_pix = np.reshape(t0.get_raw_pixels(), newshape=[t0.get_height(), t0.get_width()])

# find nucleoli
nucleoli = find_nucleoli(t0_pix, thresholding, min_size=min_size, max_size=max_size)
print("Found %d nucleoli." % obj.object_count(nucleoli))

# --------------------------
# OUTPUT DISPLAY
# --------------------------
print("### Output display ...")
with napari.gui_qt():
    # embed mpl widget in napari viewer
    mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
    [ax1, ax2] = mpl_widget.figure.subplots(nrows=1, ncols=2)
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(mpl_widget)

    viewer.add_image(t0_pix, name='data')

    viewer.add_image(nucleoli, name='nucleoli', contrast_limits=[0, 1],
                     colormap=('violet woBg', violet_woBg))