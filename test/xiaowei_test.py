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
from skimage.measure import label
from shared.find_organelles import find_organelle, nucleoli_analysis
import shared.analysis as ana
import shared.display as dis
import shared.objects as obj
import shared.bleach_points as ble
from matplotlib import pyplot as plt

# --------------------------
# PARAMETERS
# --------------------------
# data source
# data_path = "C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoli-bleaching-4x\\PythonAcq1\\AutoBleach_15"
# data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/AutoBleach_15"
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/" \
            "20201216_CBB_nucleoliBleachingTest_drugTreatment/Ctrl-2DG-CCCP-36pos_partial/exp_112/"
# values
data_z = 0
data_c = 0
data_p = 0
ref_t = 0
thresholding = 'local-nucleoli'
mode_bleach_detection = 'single-raw'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
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

# --------------------------------------
# IMAGE ANALYSIS based on reference time
# --------------------------------------
print("### Image analysis based on reference time %s ..." % ref_t)

# reference image of ref_t
temp = store.get_image(cb.p(data_p).z(data_z).c(data_c).t(ref_t).build())
pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

# find nucleoli
nucleoli = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
print("Found %d nucleoli." % obj.object_count(nucleoli))

# nucleoli pd dataset
nucleoli_pd = nucleoli_analysis(nucleoli)

# ----------------------------------
# BLEACH SPOTS DETECTION
# ----------------------------------
print("### Bleach spots detection ...")

# load point_and_shoot log file
log_pd = pd.read_csv('%s/PointAndShoot.log' % data_path, na_values=['.'], sep='\t', header=None)
print("Aim to photobleach %d spots." % len(log_pd))
log_pd = ble.get_bleach_frame(log_pd, store, cb)

# get bleach spot coordinate
log_pd = ble.get_bleach_spots_coordinates(log_pd, store, cb, mode_bleach_detection)

# generate bleach spot mask
bleach_spots, pointer_pd = ble.get_bleach_spots(log_pd, nucleoli, nucleoli_pd, store, cb, num_dilation)
print("%d spots passed filter for analysis." % obj.object_count(bleach_spots))

plt.subplots(figsize=(8, 6))
for i in range(len(pointer_pd)):
    plt.plot(pointer_pd['mean_int'][i])
plt.savefig('%s/test.pdf' % data_path)
