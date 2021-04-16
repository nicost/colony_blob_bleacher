import numpy as np
import napari
from pycromanager import Bridge
from matplotlib import pyplot as plt
from shared.find_organelles import organelle_analysis, find_organelle
import shared.display as dis
from skimage.measure import label, regionprops_table
import shared.objects as obj
import pandas as pd
import shared.analysis as ana
import shared.dataframe as dat
from matplotlib.figure import Figure
from vispy.color import Colormap
from matplotlib.backends.qt_compat import QtCore, QtWidgets

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import FigureCanvas
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvas
import seaborn as sns
import os

# --------------------------
# PARAMETERS ALLOW CHANGE
# --------------------------
# paths
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/20210224_SG_Top10_ArsTreatments/"\
            "1235678910_1"

# values for analysis
name = 'SF1-N'
data_p = 128
data_c_G3BP1 = 1  # channel for G3BP1-mScarlet  1:G3BP1-mScarlet channel
data_c_sample = 0  # channel for sample  0:GFP channel
thresholding = 'na'  # only accepts 'na', 'otsu', 'yen', 'local-nucleoli' and 'local-sg'
min_size = 5
max_size = 200

# modes
display_mode = 'Y'

"""
# ---------------------------------------------------------------------------------------------------
# PLEASE DO NOT CHANGE AFTER THIS
# ---------------------------------------------------------------------------------------------------
"""

# --------------------------
# LOAD DATA
# --------------------------
print("### Load data ...")
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
# IMAGE ANALYSIS based on position
# ------------------------------
print("### Image analysis: calculate SG mask/pd ...")
# test image of position
num = []
size = []
raw_int_G3BP1 = []
bg_G3BP1 = []
raw_int_sample = []
bg_sample = []
pix_tseries = []  # G3BP1
pix1_tseries = []  # sample
sg_tseries = []

raw_int_G3BP1_full = []
raw_int_sample_full = []
bg_G3BP1_full = []
bg_sample_full = []
x_frame = []

for i in range(max_t+1):
    temp = store.get_image(cb.t(i).c(data_c_G3BP1).z(0).p(data_p).build())
    pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])
    temp1 = store.get_image(cb.t(i).c(data_c_sample).z(0).p(data_p).build())
    pix1 = np.reshape(temp1.get_raw_pixels(), newshape=[temp1.get_height(), temp1.get_width()])
    sg = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
    # cell =

    pix_tseries.append(pix)
    pix1_tseries.append(pix1)
    sg_tseries.append(sg)

    if 1 in sg:
        label_sg = label(sg, connectivity=1)
        sg_props = regionprops_table(label_sg, pix, properties=('label', 'area', 'mean_intensity'))
        sg_props1 = regionprops_table(label_sg, pix1, properties=('label', 'area', 'mean_intensity'))
        sg_pd = pd.DataFrame(sg_props)
        sg_pd1 = pd.DataFrame(sg_props1)

        num_temp = len(sg_pd)
        num.append(num_temp)
        size_temp = np.mean(sg_pd['area'])
        size.append(size_temp)
        raw_int_G3BP1_temp = np.mean(sg_pd['mean_intensity'])
        raw_int_G3BP1.append(raw_int_G3BP1_temp)
        bg_G3BP1_temp = ana.get_bg_int([pix])[0]
        bg_G3BP1.append(bg_G3BP1_temp)
        raw_int_sample_temp = np.mean(sg_pd1['mean_intensity'])
        raw_int_sample.append(raw_int_sample_temp)
        bg_sample_temp = ana.get_bg_int([pix1])[0]
        bg_sample.append(bg_sample_temp)

        raw_int_G3BP1_full = raw_int_G3BP1_full + sg_pd['mean_intensity'].tolist()
        raw_int_sample_full = raw_int_sample_full + sg_pd1['mean_intensity'].tolist()
        x_frame = x_frame + [i] * len(sg_pd)
        bg_G3BP1_full = bg_G3BP1_full + [bg_G3BP1_temp] * len(sg_pd)
        bg_sample_full = bg_sample_full + [bg_sample_temp] * len(sg_pd1)
    else:
        num.append(0)
        size.append(0)
        raw_int_G3BP1.append(0)
        bg_G3BP1.append(0)
        raw_int_sample.append(0)
        bg_sample.append(0)

mov = np.stack(pix_tseries, axis=0)
mov1 = np.stack(pix1_tseries, axis=0)
mov_sg = np.stack(sg_tseries, axis=0)

ana_pd = pd.DataFrame({'number': num, 'size': size, 'raw_int_G3BP1': raw_int_G3BP1, 'bg_G3BP1': bg_G3BP1,
                       'raw_int_sample': raw_int_sample, 'bg_sample': bg_sample})
ana_pd['int_G3BP1'] = ana_pd['raw_int_G3BP1'] - ana_pd['bg_G3BP1']
ana_pd['int_sample'] = ana_pd['raw_int_sample'] - ana_pd['bg_sample']
ana_pd[ana_pd < 0] = 0
ana_pd['int_ratio'] = ana_pd['int_sample']/(ana_pd['int_G3BP1']+0.0001)
ana_pd = dat.get_normalized(ana_pd, 'number', 'number_norm')
ana_pd = dat.get_normalized(ana_pd, 'int_G3BP1', 'int_G3BP1_norm')
ana_pd = dat.get_normalized(ana_pd, 'int_sample', 'int_sample_norm')
ana_pd = dat.get_normalized(ana_pd, 'int_ratio', 'int_ratio_norm')

print(ana_pd)

int_G3BP1_full = dat.list_subtraction(raw_int_G3BP1_full, bg_G3BP1_full)
int_sample_full = dat.list_subtraction(raw_int_sample_full, bg_sample_full)
int_ratio_full = [list1_i / (list2_i+0.0001) for list1_i, list2_i in zip(int_sample_full, int_G3BP1_full)]

full_pd = pd.DataFrame({'frame': x_frame, 'int_G3BP1': int_G3BP1_full, 'int_sample': int_sample_full,
                        'int_ratio': int_ratio_full})

# --------------------------
# OUTPUT DISPLAY
# --------------------------
if display_mode == 'Y':
    print("### Output display ...")

    with napari.gui_qt():
        # embed mpl widget in napari viewer
        mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        [(ax1, ax2), (ax3, ax4)] = mpl_widget.figure.subplots(nrows=2, ncols=2)
        viewer = napari.Viewer()
        viewer.window.add_dock_widget(mpl_widget)

        # napari display
        # display time series movies in napari main viewer
        viewer.add_image(mov, name='G3BP1-mScarlet', colormap='red', blending='additive')
        viewer.add_image(mov1, name='sample-GFP', colormap='green', blending='additive')
        violet_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129 / 255, 55 / 255, 114 / 255, 1.0]])
        viewer.add_image(mov_sg, name='SG', contrast_limits=[0, 1], colormap=('violet woBg', violet_woBg))
