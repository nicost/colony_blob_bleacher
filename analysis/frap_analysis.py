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
# PARAMETERS allow change
# --------------------------
# paths
# data_path = "C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoli-bleaching-4x\\PythonAcq1\\AutoBleach_15"
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/" \
            "20201216_CBB_nucleoliBleachingTest_drugTreatment/Ctrl-2DG-CCCP-36pos_partial/exp_110/"

# values for analysis
data_z = 0
data_c = 0
data_p = 0
ref_t = 0  # reference frame t for nucleoli detection, bleach spots filtration
thresholding = 'local-nucleoli'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
min_size = 10  # minimum nucleoli size; default = 10
max_size = 1000  # maximum nucleoli size; default = 1000;
                 # larger ones are generally cells without nucleoli
num_dilation = 3  # number of dilation from the coordinate;
                  # determines analysis size of the analysis spots; default = 3

# modes
mode_bleach_detection = 'single-offset'  # only accepts 'single-raw' or 'single-offset'
display_mode = 'N'  # only accepts 'N' or 'Y'

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
print("### Image analysis: nucleoli detection based on reference time %s ..." % ref_t)
# reference image of ref_t
temp = store.get_image(cb.p(data_p).z(data_z).c(data_c).t(ref_t).build())
pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

# nucleoli detection
nucleoli = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
print("Found %d nucleoli." % obj.object_count(nucleoli))

# nucleoli pd dataset
nucleoli_pd = nucleoli_analysis(nucleoli)

# ----------------------------------
# BLEACH SPOTS DETECTION
# ----------------------------------
print("### Image analysis: bleach spots detection ...")

# load point_and_shoot log file
log_pd = pd.read_csv('%s/PointAndShoot.log' % data_path, na_values=['.'], sep='\t', header=None)
print("Aim to photobleach %d spots." % len(log_pd))
log_pd = ble.get_bleach_frame(log_pd, store, cb)

# get bleach spot coordinate
log_pd = ble.get_bleach_spots_coordinates(log_pd, store, cb, mode_bleach_detection)

# generate bleach spot mask
bleach_spots, pointer_pd = ble.get_bleach_spots(log_pd, nucleoli, nucleoli_pd, num_dilation)
print("%d spots passed filters for analysis." % obj.object_count(bleach_spots))

# --------------------------------------------------
# FRAP CURVE ANALYSIS from bleach spots
# --------------------------------------------------
print("### Image analysis: FRAP curve calculation ...")

# generate frap curve double corrected intensity ('mean_int') and added into pointer_pd
pointer_pd, ctrl_pd = ble.get_frap(pointer_pd, store, cb, bleach_spots, nucleoli_pd, log_pd, num_dilation)

# normalize frap curve and measure mobile fraction and t-half based on curve itself
pointer_pd = ble.frap_analysis(pointer_pd, store, cb)

# curve fitting with single exponential function
pointer_pd = ble.frap_fitting_single_exp(pointer_pd)

# filter frap curves
pointer_pd = ble.frap_filter(pointer_pd)
pointer_ft_pd = pointer_pd[pointer_pd['frap_filter'] == 1]
print("%d spots passed filters for FRAP curve quality control." % len(pointer_ft_pd))

# --------------------------
# OUTPUT
# --------------------------
print("### Export data ...")

# measurements
pointer_pd.to_csv('%s/data_full.txt' % data_path, index=False, sep='\t')
ctrl_pd.to_csv('%s/data_ctrl.txt' % data_path, index=False, sep='\t')

pointer_out = pd.DataFrame({'bleach_spots': pointer_ft_pd['bleach_spots'],
                            'x': pointer_ft_pd['x'],
                            'y': pointer_ft_pd['y'],
                            'nucleoli_size': pointer_ft_pd['nucleoli_size'],
                            'bleach_frame': pointer_ft_pd['bleach_frame'],
                            'pre_bleach_int': pointer_ft_pd['pre_bleach_int'],
                            'start_int': pointer_ft_pd['frap_start_int'],
                            'mobile_fraction': pointer_ft_pd['mobile_fraction'],
                            't_half (s)': pointer_ft_pd['t_half'],
                            'single_exp_r2': pointer_ft_pd['single_exp_r2'],
                            'single_exp_mobile_fraction': pointer_ft_pd['single_exp_mobile_fraction'],
                            'single_exp_t_half': pointer_ft_pd['single_exp_t_half']})
pointer_out.to_csv('%s/data.txt' % data_path, index=False, sep='\t')

# images
storage_path = data_path

# offset map
m = 0
n = 0
plt.subplots(figsize=(6, 4))
for i in range(len(pointer_pd)):
    if pointer_pd['frap_filter'][i] == 0:
        if m == 0:
            plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color='#1E90FF', alpha=0.5, label='filtered ones')
            m = m+1
        else:
            plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color='#1E90FF', alpha=0.5)
    else:
        if n == 0:
            plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color=(0.85, 0.35, 0.25), alpha=0.5, label='good ones')
            n = n+1
        else:
            plt.plot([0, pointer_pd['x_diff'][i]], [0, pointer_pd['y_diff'][i]], color=(0.85, 0.35, 0.25), alpha=0.5)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('x offset (pixel)')
plt.ylabel('y offset (pixel)')
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
plt.savefig('%s/offset_map.pdf' % storage_path)

# raw intensity
m = 0
n = 0
j = 0
plt.subplots(figsize=(6, 4))
plt.plot(pointer_pd['bg_int'][0], color=(0, 0, 0), label='bg')
for i in range(len(ctrl_pd)):
    if j == 0:
        plt.plot(ctrl_pd['raw_int'][i], color=(0.7, 0.7, 0.7), alpha=0.5, label='ctrl')
        j = j+1
    else:
        plt.plot(ctrl_pd['raw_int'][i], color=(0.7, 0.7, 0.7), alpha=0.5)
for i in range(len(pointer_pd)):
    if pointer_pd['frap_filter'][i] == 0:
        if m == 0:
            plt.plot(pointer_pd['raw_int'][i], color='#1E90FF', alpha=0.7, label='filtered ones')
            m = m+1
        else:
            plt.plot(pointer_pd['raw_int'][i], color='#1E90FF', alpha=0.7)
    else:
        if n == 0:
            plt.plot(pointer_pd['raw_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7, label='good ones')
            n = n+1
        else:
            plt.plot(pointer_pd['raw_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7)
plt.xlabel('time (frame)')
plt.ylabel('raw intensity (AU)')
plt.legend(loc=2, bbox_to_anchor=(0.65, 0.99))
plt.savefig('%s/raw_intensity.pdf' % storage_path)

# photobleaching factor
plt.subplots(figsize=(6, 4))
plt.plot(pointer_pd['pb_factor'][0], color=(0.8, 0.8, 0.8))
plt.xlabel('time (frame)')
plt.ylabel('photobleaching factor')
plt.savefig('%s/pb_factor.pdf' % storage_path)

# intensity after dual correction
m = 0
n = 0
plt.subplots(figsize=(6, 4))
for i in range(len(pointer_pd)):
    if pointer_pd['frap_filter'][i] == 0:
        if m == 0:
            plt.plot(pointer_pd['mean_int'][i], color='#1E90FF', alpha=0.7, label='filtered ones')
            m = m+1
        else:
            plt.plot(pointer_pd['mean_int'][i], color='#1E90FF', alpha=0.7)
    else:
        if n == 0:
            plt.plot(pointer_pd['mean_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7, label='good ones')
            n = n+1
        else:
            plt.plot(pointer_pd['mean_int'][i], color=(0.85, 0.35, 0.25), alpha=0.7)
plt.xlabel('time (frame)')
plt.ylabel('bg/pb corrected intensity (AU)')
plt.legend(loc=2, bbox_to_anchor=(0.65, 0.99))
plt.savefig('%s/double_corrected_intensity.pdf' % storage_path)

# normalized FRAP curves
m = 0
n = 0
plt.subplots(figsize=(6, 4))
for i in range(len(pointer_pd)):
    if pointer_pd['frap_filter'][i] == 0:
        if m == 0:
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                     color='#1E90FF', alpha=0.7, label='filtered ones')
            m = m+1
        else:
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                     color='#1E90FF', alpha=0.7)
    else:
        if n == 0:
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                     color=(0.85, 0.35, 0.25), alpha=0.7, label='good ones')
            n = n+1
        else:
            plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                     color=(0.85, 0.35, 0.25), alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel('normalized intensity (AU)')
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99))
plt.savefig('%s/normalized_frap_curves.pdf' % storage_path)

# normalized FRAP curves after filtering with fitting
cmap1 = 'viridis'
cmap1_rgba = dis.num_color_colormap(cmap1, len(pointer_pd))[2]
plt.subplots(figsize=(6, 4))
for i in range(len(pointer_pd)):
    if pointer_pd['frap_filter'][i] == 1:
        plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                 color=cmap1_rgba[i+1], alpha=0.7)
        plt.plot(pointer_pd['real_time_post'][i], pointer_pd['single_exp_fit'][i], '--',
                 color=cmap1_rgba[i+1], alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel('normalized intensity (AU)')
plt.savefig('%s/normalized_frap_curves_filtered.pdf' % storage_path)

# individual normalized FRAP curves with fitting
for i in range(len(pointer_pd)):
    if pointer_pd['frap_filter'][i] == 1:
        plt.subplots(figsize=(6, 4))
        plt.plot(pointer_pd['real_time_post'][i], pointer_pd['int_curve_post_nor'][i],
                 color=cmap1_rgba[i+1], alpha=0.7)
        plt.plot(pointer_pd['real_time_post'][i], pointer_pd['single_exp_fit'][i], '--',
                 color=cmap1_rgba[i+1], alpha=0.7)
        plt.xlabel('time (s)')
        plt.ylabel('normalized intensity (AU)')
        plt.savefig('%s/frap_curves_filtered_%d.pdf' % (storage_path, i))

# --------------------------
# OUTPUT DISPLAY
# --------------------------
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
        mov = ana.get_movie(store, cb)
        viewer.add_image(mov, name='data')

        # Layer2: nucleoli
        # display nucleoli mask (violet)
        violet_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129 / 255, 55 / 255, 114 / 255, 1.0]])
        viewer.add_image(nucleoli, name='nucleoli', contrast_limits=[0, 1], colormap=('violet woBg', violet_woBg))

        # Layer3: aim points
        # display aim points from .log file (red)
        points = np.column_stack((log_pd['aim_y'].tolist(), log_pd['aim_x'].tolist()))
        size = [3] * len(points)
        viewer.add_points(points, name='aim points', size=size, edge_color='r', face_color='r')

        # Layer4: analysis spots
        # display bleach spots, color sorted based on corresponding nucleoli size
        # sort colormap based on analysis spots filtered
        rgba_winter = dis.num_color_colormap('winter', len(pointer_pd))[2]
        winter_woBg = dis.sorted_num_color_colormap(rgba_winter, pointer_pd, 'nucleoli_size', 'bleach_spots')[0]
        if len(pointer_pd) != 0:
            viewer.add_image(label(bleach_spots), name='bleach spots', colormap=('winter woBg', winter_woBg))

        # matplotlib display
        # sorted based on nucleoli size (color coded)
        pointer_sort = pointer_pd.sort_values(by='nucleoli_size').reset_index(drop=True)  # from small to large

        # Plot-left: FRAP curves of filtered analysis spots after intensity correction (absolute intensity)
        for i in range(len(pointer_sort)):
            ax1.plot(pointer_sort['mean_int'][i], color=rgba_winter[i+1])
        ax1.set_title('FRAP curves')
        ax1.set_xlabel('time')
        ax1.set_ylabel('intensity')

        # Plot-middle: FRAP curves of filtered analysis spots after intensity correction
        # relative intensity, bleach time zero aligned
        for i in range(len(pointer_sort)):
            if pointer_sort['frap_filter'][i] == 1:
                ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['int_curve_post_nor'][i],
                         color=rgba_winter[i+1], alpha=0.5)
                ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['single_exp_fit'][i], '--',
                         color=rgba_winter[i+1])
        ax2.set_title('FRAP curves')
        ax2.set_xlabel('time (sec)')
        ax2.set_ylabel('intensity')

        # Plot-right: offset
        if mode_bleach_detection == 'single-offset':
            for i in range(len(pointer_sort)):
                ax3.plot([0, pointer_sort['x_diff'][i]], [0, pointer_sort['y_diff'][i]],
                         color=rgba_winter[i+1])
            ax3.set_xlim([-10, 10])
            ax3.set_ylim([-10, 10])
            ax3.set_title('Offset map')
            ax3.set_xlabel('x offset')
            ax3.set_ylabel('y offset')
