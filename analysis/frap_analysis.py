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
# paths
# data_path = "C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoli-bleaching-4x\\PythonAcq1\\AutoBleach_15"
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/" \
            "20210302_CBB_nucleoliWTcellDensityTest/50k/D4/D4-Site_0_1"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/" \
            "20210302_CBB_nucleoliWTcellDensityTest/50k/dataAnalysis1/D4/D4-Site_0_1"

# values for analysis
data_c = 0
pos = 0
thresholding = 'local-nucleoli'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
min_size = 10  # minimum nucleoli size; default = 10
max_size = 1000  # maximum nucleoli size; default = 1000;
# larger ones are generally cells without nucleoli
num_dilation = 3  # number of dilation from the coordinate;
# determines analysis size of the analysis spots; default = 3
frap_start_delay = 4

# modes
mode_bleach_detection = 'single-offset'  # only accepts 'single-raw' or 'single-offset'
frap_start_mode = 'min'  # only accepts 'delay' or 'min'
display_mode = 'Y'  # only accepts 'N' or 'Y'

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
print("### Image analysis: nucleoli detection based on time 0 ...")

# reference image of time 0
# if decide to use other image as ref_image
# be sure to check photobleaching correction for all reported intensities
temp = store.get_image(cb.c(data_c).t(0).build())
pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

# nuclear detection
label_nuclear = find_nuclear_nucleoli(pix)
data_log['num_nuclei_detected'] = [np.amax(label_nuclear)]
print("Found %d nuclei." % data_log['num_nuclei_detected'][0])

# nucleoli detection
nucleoli = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
label_nucleoli = label(nucleoli, connectivity=1)
data_log['num_nucleoli_detected'] = [obj.object_count(nucleoli)]
print("Found %d nucleoli." % data_log['num_nucleoli_detected'][0])

# nucleoli pd dataset
nucleoli_pd = organelle_analysis(pix, nucleoli, 'nucleoli', pos)
# link nucleoli with corresponding nuclear
round_x = [round(num) for num in nucleoli_pd['x']]
round_y = [round(num) for num in nucleoli_pd['y']]
nucleoli_pd['nuclear'] = obj.points_in_objects(label_nuclear, round_y, round_x)

data_log['num_nucleoli_in_nuclei'] = [len(nucleoli_pd[nucleoli_pd['nuclear'] != 0])]
print("Found %d out of %d nucleoli within nuclei." % (data_log['num_nucleoli_in_nuclei'][0],
                                                      obj.object_count(nucleoli)))

# nuclear pd dataset
nuclear_pd = nuclear_analysis(label_nuclear, nucleoli_pd, pos)

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

# link pointer with corresponding nucleoli
log_pd['nucleoli'] = obj.points_in_objects(label_nucleoli, log_pd['x'], log_pd['y'])

# generate bleach spot mask and bleach spots dataframe (pointer_pd)
bleach_spots, pointer_pd = ble.get_bleach_spots(log_pd, label_nucleoli, num_dilation)
data_log['num_bleach_spots'] = [obj.object_count(bleach_spots)]
print("%d spots passed filters for analysis." % data_log['num_bleach_spots'][0])

# add bleach spots corresponding nucleoli measurements
pointer_pd = dat.copy_based_on_index(pointer_pd, nucleoli_pd, 'nucleoli', 'nucleoli',
                                     ['nucleoli_x', 'nucleoli_y', 'nucleoli_size',
                                      'nucleoli_mean_int', 'nucleoli_circ'],
                                     ['x', 'y', 'size', 'raw_int', 'circ'])

# --------------------------------------------------
# FRAP CURVE ANALYSIS from bleach spots
# --------------------------------------------------
print("### Image analysis: FRAP curve calculation ...")

# create control spots mask
ctrl_nucleoli = ~nucleoli_pd.index.isin(log_pd['nucleoli'].tolist())
ctrl_x = nucleoli_pd[ctrl_nucleoli]['x'].astype(int).tolist()
ctrl_y = nucleoli_pd[ctrl_nucleoli]['y'].astype(int).tolist()
ctrl_spots = ana.analysis_mask(ctrl_x, ctrl_y, pix, num_dilation)
num_ctrl_spots = obj.object_count(ctrl_spots)
pointer_pd['num_ctrl_spots'] = [num_ctrl_spots] * len(pointer_pd)

# get raw intensities for bleach spots and control spots
pointer_pd['raw_int'] = ana.get_intensity(bleach_spots, pixels_tseries)
ctrl_spots_int_tseries = ana.get_intensity(ctrl_spots, pixels_tseries)
ctrl_pd = pd.DataFrame({'ctrl_spots': np.arange(0, num_ctrl_spots, 1), 'raw_int': ctrl_spots_int_tseries})

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
pointer_pd['bg_cor_int'] = ana.bg_correction(pointer_pd['raw_int'], bg)
ctrl_pd['bg_cor_int'] = ana.bg_correction(ctrl_pd['raw_int'], bg)

# filter control traces
filter_ctrl = []
for i in range(len(ctrl_pd)):
    ctrl_int = ctrl_pd['bg_cor_int'][i]
    if (max(ctrl_int)-min(ctrl_int))/max(ctrl_int) > 0.4:
        filter_ctrl.append(0)
    else:
        filter_ctrl.append(1)
ctrl_pd['filter'] = filter_ctrl
ctrl_pd_ft = ctrl_pd[ctrl_pd['filter'] == 1].reset_index()
pointer_pd['num_ctrl_spots_ft'] = [len(ctrl_pd_ft)] * len(pointer_pd)
data_log['num_ctrl_spots'] = len(ctrl_pd_ft)

print("### Image analysis: photobleaching correction ...")
# photobleaching factor calculation
if len(ctrl_pd_ft) != 0:
    # calculate photobleaching factor
    pb_factor = ana.get_pb_factor(ctrl_pd_ft['bg_cor_int'])

    pointer_pd['pb_factor'] = [pb_factor] * len(pointer_pd)
    print("%d ctrl points are used to correct photobleaching." % len(ctrl_pd_ft))

    # pb_factor fitting with single exponential decay
    pb_fit = mat.fitting_single_exp_decay(np.arange(0, len(pb_factor), 1), pb_factor)
    pointer_pd = dat.add_columns(pointer_pd, ['pb_single_exp_decay_fit', 'pb_single_exp_decay_r2',
                                              'pb_single_exp_decay_a', 'pb_single_exp_decay_b',
                                              'pb_single_exp_decay_c'],
                                 [[pb_fit[0]] * len(pointer_pd), [pb_fit[1]] * len(pointer_pd),
                                  [pb_fit[2]] * len(pointer_pd), [pb_fit[3]] * len(pointer_pd),
                                  [pb_fit[4]] * len(pointer_pd)])

    # photobleaching correction
    if np.isnan(pb_fit[2]):
        pb = pb_factor
    else:
        pb = pb_fit[0]
    pointer_pd['mean_int'] = ana.pb_correction(pointer_pd['bg_cor_int'], pb)

    # normalize frap curve and measure mobile fraction and t-half based on curve itself
    frap_pd = ble.frap_analysis(pointer_pd, max_t, acquire_time_tseries, real_time, frap_start_delay, frap_start_mode)
    pointer_pd = pd.concat([pointer_pd, frap_pd], axis=1)

    # --------------------------------------------------
    # FRAP CURVE FITTING
    # --------------------------------------------------
    print("### Imaging analysis: curve fitting ...")

    # curve fitting with linear to determine initial slope
    linear_fit_pd = mat.frap_fitting_linear(pointer_pd['real_time_post'], pointer_pd['int_curve_post_nor'])
    pointer_pd = pd.concat([pointer_pd, linear_fit_pd], axis=1)

    # curve fitting with single exponential function
    single_exp_fit_pd = mat.frap_fitting_single_exp(pointer_pd['real_time_post'],
                                                    pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, single_exp_fit_pd], axis=1)

    # curve fitting with soumpasis function
    soumpasis_fit_pd = mat.frap_fitting_soumpasis(pointer_pd['real_time_post'],
                                                  pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, soumpasis_fit_pd], axis=1)

    # curve fitting with double exponential function
    double_exp_fit_pd = mat.frap_fitting_double_exp(pointer_pd['real_time_post'],
                                                    pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, double_exp_fit_pd], axis=1)

    # curve fitting with ellenberg function
    ellenberg_fit_pd = mat.frap_fitting_ellenberg(pointer_pd['real_time_post'],
                                                  pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, ellenberg_fit_pd], axis=1)

    # find optimal fitting
    optimal_fit_pd = mat.find_optimal_fitting(pointer_pd, ['single_exp', 'soumpasis', 'ellenberg', 'double_exp'])
    pointer_pd = pd.concat([pointer_pd, optimal_fit_pd], axis=1)

    # filter frap curves
    pointer_pd['frap_filter_single_exp'] = ble.frap_filter(pointer_pd, 'single_exp')
    pointer_pd['frap_filter_soumpasis'] = ble.frap_filter(pointer_pd, 'soumpasis')
    pointer_pd['frap_filter_double_exp'] = ble.frap_filter(pointer_pd, 'double_exp')
    pointer_pd['frap_filter_ellenberg'] = ble.frap_filter(pointer_pd, 'ellenberg')
    pointer_pd['frap_filter_optimal'] = ble.frap_filter(pointer_pd, 'optimal')

    pointer_pd['pos'] = [0] * len(pointer_pd)
    pointer_ft_pd = pointer_pd[pointer_pd['frap_filter_optimal'] == 1]
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
    # dataset of nuclear
    nuclear_pd.to_csv('%s/data_nuclear.txt' % storage_path, index=False, sep='\t')
    # dataset of nucleoli
    nucleoli_pd.to_csv('%s/data_nucleoli.txt' % storage_path, index=False, sep='\t')

    # simplified dataset of bleach spots after FRAP curve quality control
    pointer_out = pd.DataFrame({'pos': pointer_ft_pd['pos'],
                                'bleach_spots': pointer_ft_pd['bleach_spots'],
                                'x': pointer_ft_pd['x'],
                                'y': pointer_ft_pd['y'],
                                'nucleoli': pointer_ft_pd['nucleoli'],
                                'nucleoli_size': pointer_ft_pd['nucleoli_size'],
                                'nucleoli_mean_int': pointer_ft_pd['nucleoli_mean_int'],
                                'bleach_frame': pointer_ft_pd['bleach_frame'],
                                'pre_bleach_int': pointer_ft_pd['pre_bleach_int'],
                                'start_int': pointer_ft_pd['frap_start_int'],
                                'mobile_fraction': pointer_ft_pd['mobile_fraction'],
                                't_half': pointer_ft_pd['t_half'],
                                'ini_slope': pointer_ft_pd['ini_slope'],
                                'linear_slope': pointer_ft_pd['linear_slope'],
                                'optimal_r2': pointer_ft_pd['optimal_r2'],
                                'optimal_mobile_fraction': pointer_ft_pd['optimal_mobile_fraction'],
                                'optimal_t_half': pointer_ft_pd['optimal_t_half'],
                                'optimal_slope': pointer_ft_pd['optimal_slope']})
    pointer_out.to_csv('%s/data.txt' % storage_path, index=False, sep='\t')

    # images
    dis.plot_offset_map(pointer_pd, storage_path)  # offset map
    dis.plot_raw_intensity(pointer_pd, ctrl_pd_ft, storage_path)  # raw intensity
    dis.plot_pb_factor(pointer_pd, storage_path)  # photobleaching factor
    dis.plot_corrected_intensity(pointer_pd, storage_path)  # intensity after dual correction
    dis.plot_normalized_frap(pointer_pd, storage_path)  # normalized FRAP curves
    dis.plot_frap_fitting(pointer_pd, storage_path)  # normalized FRAP curves after filtering with fitting
    # individual normalized FRAP curves with fitting
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

        # Layer2: nuclear
        # display labeled nuclei
        cmap1 = 'winter'
        cmap1_woBg = dis.num_color_colormap(cmap1, np.amax(label_nuclear))[0]
        viewer.add_image(label_nuclear, name='nuclear', colormap=('winter woBg', cmap1_woBg))

        # Layer3: nucleoli
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
        if len(pointer_pd) != 0:
            cmap2 = 'winter'
            cmap2_rgba = dis.num_color_colormap(cmap2, len(pointer_pd))[2]
            cmap2_napari = dis.sorted_num_color_colormap(cmap2_rgba, pointer_pd, 'nucleoli_size', 'bleach_spots')[0]
            viewer.add_image(label(bleach_spots), name='bleach spots', colormap=('winter woBg', cmap2_napari))

        # matplotlib display
        if len(ctrl_pd_ft) != 0:
            # sorted based on nucleoli size (color coded)
            pointer_sort = pointer_pd.sort_values(by='nucleoli_size').reset_index(drop=True)  # from small to large

            # Plot-left: FRAP curves of filtered analysis spots after intensity correction (absolute intensity)
            for i in range(len(pointer_sort)):
                ax1.plot(pointer_sort['mean_int'][i], color=cmap2_rgba[i + 1])
            ax1.set_title('FRAP curves')
            ax1.set_xlabel('time')
            ax1.set_ylabel('intensity')

            # Plot-middle: FRAP curves of filtered analysis spots after intensity correction
            # relative intensity, bleach time zero aligned
            for i in range(len(pointer_sort)):
                if pointer_sort['frap_filter_optimal'][i] == 1:
                    ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['int_curve_post_nor'][i],
                             color=cmap2_rgba[i + 1], alpha=0.5)
                    ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['optimal_fit'][i], '--',
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