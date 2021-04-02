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
from skimage.measure import label
from skimage.morphology import medial_axis
import shared.analysis as ana
import shared.dataframe as dat
import shared.display as dis
import shared.objects as obj
import shared.bleach_points as ble
import shared.math_functions as mat
import os

"""
# ---------------------------------------------------------------------------------------------------
# FRAP ANALYSIS for NUCLEOLI (SINGLE-FOV)
# ---------------------------------------------------------------------------------------------------

EXPECTS 
    an uManager data (single z, single p),
SEGMENTS and ANALYZES
    nuclear properties (enables x, y position and number of nucleoli), nucleoli properties (enables 
    x, y position, size, mean intensity (without correction), circularity, eccentricity and 
    corresponding nuclear index,
DETECTS and ANALYZES
    bleach spots to measure FRAP curves. Intensities were background and photobleaching corrected and
    normalized based on pre-bleach intensity and minimum intensity, curves were then fitted with 
    single exponential function and mobile fraction and t-half were calculated based on fitting,
EXPORTS 
    data.txt: simplified bleach spots related information
    data_full.txt: full bleach spots related information
    data_log.txt: some information during analysis
    data_nuclear.txt: nuclear relevant information
    data_nucleoli.txt: nucleoli relevant information
    data_ctrl.txt: control spots relevant information
    offset_map.pdf: offset map, aim spots (coordinates get from .log file) are centered to (0,0), 
        non (0,0) end of the lines indicate location of detected bleach spots relative to aim spots
    raw_intensity.pdf: raw intensity curves of bleach spots, control spots and background and 
        background linear fit curve
    pb_factor.pdf: photobleach factor curve and its single exponential decay fit curve
    double_corrected_intensity.pdf: double corrected intensity curves of bleach spots
    normalized_frap_curves.pdf: normalized FRAP curves
    normalized_frap_curves_filtered.pdf: normalized FRAP curves after filtering and their 
        corresponding single exponential fitting curves
    frap_curves_filtered_NUMBER.pdf: each single normalized FRAP curve and its corresponding single 
        exponential fitting curve
DISPLAYS 
    images (raw image, nuclear, nucleoli, aim spots, color coded bleach spots) in napari, images 
    (double corrected intensity curves, normalized filtered FRAP curves and their fitting curves, 
    offset map) in matplotlib viewer.

# ----------------------------------
# PARAMETERS ALLOW CHANGE
# ----------------------------------

    # paths
    data_path: directory of uManager data
    save_path: primary directory for output saving

    # values for analysis
    data_c: channel to be analyzed
    pos: position of the given FOV in multi-image dataset, default = 0
    thresholding: global thresholding method used for nucleoli segmentation; only accepts 'na', 
        'otsu', 'yen', 'local-nucleoli' and 'local-sg'
    min_size: the smallest allowable nucleoli size
    max_size: the largest allowable nucleoli size
    num_dilation: number of dilation used to generate bleach spots, determines size of bleach spots
        default = 3

    # modes
    mode_bleach_detection: bleach spots detection modes; only accept 'single-raw' or 'single-offset'
    display_mode: displays stitched images in napari or not; only accepts 'N' or 'Y'

"""

# --------------------------
# PARAMETERS allow change
# --------------------------
# Please changes
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/"\
                "20210203_CBB_nucleoliArsAndHeatshockTreatment/data/WT1/C2-Site_0_1"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/"\
                "20210203_CBB_nucleoliArsAndHeatshockTreatment/dataAnalysis1/WT1/C2-Site_0_1"
analyze_organelle = 'nucleoli'  # only accepts 'sg' or 'nucleoli'
frap_start_delay = 6  # 50ms default = 4; 100ms default = 5; 200ms default = 6
display_mode = 'Y'  # only accepts 'N' or 'Y'
display_sort = 'pre_bleach_int'  # accepts 'na' or other features like 'sg_size'

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

if analyze_organelle == 'nucleoli':
    # nuclear detection (currently only doable for nucleoli staining image)
    label_nuclear, _ = find_nuclear_nucleoli(pix)
    data_log['num_nuclei_detected'] = [np.amax(label_nuclear)]
    print("Found %d nuclei." % data_log['num_nuclei_detected'][0])

# organelle detection
organelle_before_filter, organelle = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
label_organelle = label(organelle, connectivity=1)
data_log['num_%s_detected' % analyze_organelle] = [obj.object_count(organelle)]
print("Found %d %s." % (data_log['num_%s_detected' % analyze_organelle][0], analyze_organelle))

# organelle pd dataset
organelle_pd = organelle_analysis(pix, organelle, '%s' % analyze_organelle, pos)

if analyze_organelle == 'nucleoli':
    # link nucleoli with corresponding nuclear
    round_x = [round(num) for num in organelle_pd['x']]
    round_y = [round(num) for num in organelle_pd['y']]
    organelle_pd['nuclear'] = obj.points_in_objects(label_nuclear, round_y, round_x)

    # nuclear pd dataset
    nuclear_pd = nuclear_analysis(label_nuclear, organelle_pd, pos)

    data_log['num_nucleoli_in_nuclei'] = [len(organelle_pd[organelle_pd['nuclear'] != 0])]
    print("Found %d out of %d nucleoli within nuclei." % (data_log['num_nucleoli_in_nuclei'][0],
                                                          obj.object_count(organelle)))

# ----------------------------------
# BLEACH SPOTS DETECTION
# ----------------------------------
print("### Image analysis: bleach spots detection ...")

# load point_and_shoot log file
log_pd = pd.read_csv('%s/PointAndShoot.log' % data_path, na_values=['.'], sep='\t', header=None)
data_log['num_aim_spots'] = [len(log_pd)]
print("Aim to photobleach %d spots." % data_log['num_aim_spots'][0])
log_pd.columns = ['time', 'aim_x', 'aim_y']  # reformat log_pd

# get bleach_frame
log_pd['bleach_frame'] = dat.get_frame(log_pd['time'], acquire_time_tseries)

# get bleach spot coordinate
coordinate_pd = ble.get_bleach_spots_coordinates(log_pd, store, cb, data_c, mode_bleach_detection, frap_start_delay)
log_pd = pd.concat([log_pd, coordinate_pd], axis=1)

# link pointer with corresponding organelle
log_pd['%s' % analyze_organelle] = obj.points_in_objects(label_organelle, log_pd['x'], log_pd['y'])

# calculate distance to organelle boundary
_, distance_map = medial_axis(organelle, return_distance=True)
distance_lst = []
for i in range(len(log_pd)):
    distance_lst.append(distance_map[log_pd['y'][i]][log_pd['x'][i]])
log_pd['distance'] = distance_lst

# generate bleach spot mask and bleach spots dataframe (pointer_pd)
bleach_spots, pointer_pd = ble.get_bleach_spots(log_pd, label_organelle, analyze_organelle, num_dilation)
data_log['num_bleach_spots'] = [obj.object_count(bleach_spots)]
print("%d spots passed filters for analysis." % data_log['num_bleach_spots'][0])

# add bleach spots corresponding organelle measurements
pointer_pd = dat.copy_based_on_index(pointer_pd, organelle_pd, '%s' % analyze_organelle, '%s' % analyze_organelle,
                                     ['%s_x' % analyze_organelle, '%s_y' % analyze_organelle,
                                      '%s_size' % analyze_organelle, '%s_mean_int' % analyze_organelle,
                                      '%s_circ' % analyze_organelle],
                                     ['x', 'y', 'size', 'raw_int', 'circ'])

# --------------------------------------------------
# FRAP CURVE ANALYSIS from bleach spots
# --------------------------------------------------
print("### Image analysis: FRAP curve calculation ...")

# create control spots mask
ctrl_organelle = ~organelle_pd.index.isin(log_pd['%s' % analyze_organelle].tolist())
ctrl_x = organelle_pd[ctrl_organelle]['x'].astype(int).tolist()
ctrl_y = organelle_pd[ctrl_organelle]['y'].astype(int).tolist()
ctrl_spots = ana.analysis_mask(ctrl_x, ctrl_y, pix, num_dilation)
num_ctrl_spots = obj.object_count(ctrl_spots)
pointer_pd['num_ctrl_spots'] = [num_ctrl_spots] * len(pointer_pd)

# get raw intensities for bleach spots and control spots
pointer_pd['raw_int'] = ana.get_intensity(label(bleach_spots, connectivity=1), pixels_tseries)
ctrl_spots_int_tseries = ana.get_intensity(label(ctrl_spots, connectivity=1), pixels_tseries)
ctrl_pd = pd.DataFrame({'pos': [pos] * num_ctrl_spots, 'ctrl_spots': np.arange(0, num_ctrl_spots, 1),
                        'x': ctrl_y, 'y': ctrl_x, 'raw_int': ctrl_spots_int_tseries})

# link ctrl spots with corresponding organelle
ctrl_pd['%s' % analyze_organelle] = obj.points_in_objects(label_organelle, ctrl_pd['x'], ctrl_pd['y'])

print("### Image analysis: background correction ...")
# background intensity measurement
bg_int_tseries = ana.get_bg_int(pixels_tseries)
pointer_pd['bg_int'] = [bg_int_tseries] * len(pointer_pd)

# background intensity fitting
bg_fit = mat.fitting_linear(np.arange(0, len(bg_int_tseries), 1), bg_int_tseries)
pointer_pd = dat.add_columns(pointer_pd, ['bg_linear_fit', 'bg_linear_r2', 'bg_linear_a', 'bg_linear_b'],
                             [[bg_fit[0]] * len(pointer_pd), [bg_fit[1]] * len(pointer_pd),
                              [bg_fit[2]] * len(pointer_pd), [bg_fit[3]] * len(pointer_pd)])

# background correction
# use original measurement if fitting does not exist
if np.isnan(bg_fit[2]):
    bg = bg_int_tseries
else:
    bg = bg_fit[0]
pointer_pd['bg_cor_int'] = ana.bg_correction(pointer_pd['raw_int'], [bg]*len(pointer_pd))
ctrl_pd['bg_cor_int'] = ana.bg_correction(ctrl_pd['raw_int'], [bg]*len(ctrl_pd))

# filter control traces
ctrl_pd_ft = ble.filter_ctrl(ctrl_pd)
pointer_pd['num_ctrl_spots_ft'] = [len(ctrl_pd_ft)] * len(pointer_pd)
data_log['num_ctrl_spots'] = len(ctrl_pd_ft)

print("### Image analysis: photobleaching correction ...")
# photobleaching factor calculation
if len(ctrl_pd_ft) != 0:
    pointer_pd = ble.frap_pb_correction(pointer_pd, ctrl_pd_ft)
    # normalize frap curve and measure mobile fraction and t-half based on curve itself
    frap_pd = ble.frap_analysis(pointer_pd, max_t, acquire_time_tseries, real_time, frap_start_delay,
                                frap_start_mode)
    pointer_pd = pd.concat([pointer_pd, frap_pd], axis=1)

    # frap curve fitting
    print("### Imaging analysis: curve fitting ...")
    pointer_pd = ble.frap_curve_fitting(pointer_pd)
    pointer_pd['pos'] = [pos] * len(pointer_pd)
    pointer_ft_pd = pointer_pd[pointer_pd['frap_filter_%s' % fitting_mode] == 1]
    data_log['num_frap_curves'] = [len(pointer_ft_pd)]
    print("%d spots passed filters for FRAP curve quality control." % data_log['num_frap_curves'][0])

    # --------------------------
    # OUTPUT
    # --------------------------
    print("### Export data ...")

    storage_path = save_path
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # measurements
    # data_log
    data_log.to_csv('%s/data_log.txt' % storage_path, index=False, sep='\t')
    # full dataset of all bleach spots
    pointer_pd.to_csv('%s/data_full.txt' % storage_path, index=False, sep='\t')
    # dataset of control spots
    ctrl_pd.to_csv('%s/data_ctrl.txt' % storage_path, index=False, sep='\t')
    # dataset of organelle
    organelle_pd.to_csv('%s/data_%s.txt' % (storage_path, analyze_organelle), index=False, sep='\t')
    if analyze_organelle == 'nucleoli':
        nuclear_pd.to_csv('%s/data_nuclear.txt' % storage_path, index=False, sep='\t')

    # images
    dis.plot_offset_map(pointer_pd, fitting_mode, 'bg', storage_path)  # offset map
    dis.plot_raw_intensity(pointer_pd, ctrl_pd_ft, fitting_mode, 'bg', storage_path)  # raw intensity
    dis.plot_pb_factor(pointer_pd, 'bg', storage_path)  # photobleaching factor
    dis.plot_corrected_intensity(pointer_pd, fitting_mode, 'bg', storage_path)  # intensity after dual correction
    dis.plot_normalized_frap(pointer_pd, fitting_mode, 'bg', storage_path)  # normalized FRAP curves
    # normalized FRAP curves after filtering with fitting
    # individual normalized FRAP curves with fitting
    dis.plot_frap_fitting(pointer_pd, fitting_mode, 'bg', storage_path)

else:
    # --------------------------
    # OUTPUT
    # --------------------------
    print("### Export data ...")

    storage_path = save_path
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    # data_log
    data_log.to_csv('%s/data_log.txt' % storage_path, index=False, sep='\t')

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
        mov = dis.napari_movie(store, cb)
        viewer.add_image(mov, name='data')

        if (analyze_organelle == 'nucleoli') & (np.amax(label_nuclear) > 0):
            # Layer2: nuclear
            # display labeled nuclei
            cmap1 = 'winter'
            cmap1_woBg = dis.num_color_colormap(cmap1, np.amax(label_nuclear))[0]
            viewer.add_image(label_nuclear, name='nuclear', colormap=('winter woBg', cmap1_woBg))

        # Layer3: organelle
        # display organelle mask (violet)
        violet_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129 / 255, 55 / 255, 114 / 255, 1.0]])
        viewer.add_image(organelle, name=('%s' % analyze_organelle), contrast_limits=[0, 1],
                         colormap=('violet woBg', violet_woBg))

        # Layer3: aim points
        # display aim points from .log file (red)
        points = np.column_stack((log_pd['aim_y'].tolist(), log_pd['aim_x'].tolist()))
        size = [3] * len(points)
        viewer.add_points(points, name='aim points', size=size, edge_color='r', face_color='r')

        # Layer4: analysis spots
        # display bleach spots, color sorted based on corresponding nucleoli size
        # sort colormap based on analysis spots filtered
        if len(pointer_pd) != 0:
            cmap2 = 'winter'
            cmap2_rgba = dis.num_color_colormap(cmap2, len(pointer_pd))[2]
            if display_sort == 'na':
                cmap2_napari = dis.num_color_colormap(cmap2, len(pointer_pd))[0]
            else:
                cmap2_napari = dis.sorted_num_color_colormap(cmap2_rgba, pointer_pd,
                                                             '%s' % display_sort,
                                                             'bleach_spots')[0]
            viewer.add_image(label(bleach_spots), name='bleach spots', colormap=('winter woBg', cmap2_napari))

        # matplotlib display
        if len(ctrl_pd_ft) != 0:
            if display_sort == 'na':
                pointer_sort = pointer_pd
            else:
                # sorted based on feature (color coded)
                # from small to large
                pointer_sort = \
                    pointer_pd.sort_values(by='%s' % display_sort).reset_index(drop=True)

            # Plot-left: FRAP curves of filtered analysis spots after intensity correction (absolute intensity)
            for i in range(len(pointer_sort)):
                ax1.plot(pointer_sort['mean_int'][i], color=cmap2_rgba[i + 1])
            ax1.set_title('FRAP curves')
            ax1.set_xlabel('time')
            ax1.set_ylabel('intensity')

            # Plot-middle: FRAP curves of filtered analysis spots after intensity correction
            # relative intensity, bleach time zero aligned
            for i in range(len(pointer_sort)):
                if pointer_sort['frap_filter_%s' % fitting_mode][i] == 1:
                    ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['int_curve_post_nor'][i],
                             color=cmap2_rgba[i + 1], alpha=0.5)
                    ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['%s_fit' % fitting_mode][i], '--',
                             color=cmap2_rgba[i + 1])
            ax2.set_title('FRAP curves')
            ax2.set_xlabel('time (sec)')
            ax2.set_ylabel('intensity')

            # Plot-right: offset
            if mode_bleach_detection == 'single-offset':
                for i in range(len(pointer_sort)):
                    ax3.plot([0, pointer_sort['x_diff'][i]], [0, pointer_sort['y_diff'][i]],
                             color=cmap2_rgba[i + 1])
                ax3.set_xlim([-10, 10])
                ax3.set_ylim([-10, 10])
                ax3.set_title('Offset map')
                ax3.set_xlabel('x offset')
                ax3.set_ylabel('y offset')
