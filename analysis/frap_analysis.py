# --------------------------
# IMPORTS
# --------------------------
import numpy as np
import pandas as pd
import napari
from scipy.optimize import curve_fit

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

from shared.find_organelles import find_organelle
import shared.analysis as ana
import shared.display as dis
import shared.dataframe as dat
import shared.objects as obj
import shared.math_functions as mat

# --------------------------
# PARAMETERS
# --------------------------
# data source
# data_path = "C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoili-bleaching-4x\\PythonAcq1\\AutoBleach_15"
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/AutoBleach_15"

# values
thresholding = 'local-nucleoli'
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

# ------------------------------
# IMAGE ANALYSIS based on time 0
# ------------------------------
print("### Image analysis based on time 0 ...")

# test image of time 0
t0 = store.get_image(cb.t(0).build())
t0_pix = np.reshape(t0.get_raw_pixels(), newshape=[t0.get_height(), t0.get_width()])

# find nucleoli
nucleoli = find_organelle(t0_pix, thresholding, min_size=min_size, max_size=max_size)
print("Found %d nucleoli." % obj.object_count(nucleoli))

# get the size and centroid of each nucleoli
nucleoli_label = label(nucleoli)
nucleoli_prop = regionprops(nucleoli_label)
nucleoli_areas = [p.area for p in nucleoli_prop]
nucleoli_centroid_x = [p.centroid[0] for p in nucleoli_prop]
nucleoli_centroid_y = [p.centroid[1] for p in nucleoli_prop]

# nucleoli pd dataset
nucleoli_pd = pd.DataFrame({'size': nucleoli_areas, 'centroid_x': nucleoli_centroid_x,
                            'centroid_y': nucleoli_centroid_y})

# ----------------------------------
# POINTER ANALYSIS based on log file
# ----------------------------------
print("### Pointer analysis based on log file ...")

# load point_and_shoot log file
pointer = pd.read_csv('%s/PointAndShoot.log' % data_path, na_values=['.'], sep='\t', header=None)
print("Aim to photobleach %d spots." % len(pointer))
pointer = dat.add_columns(pointer, ['aim_x', 'aim_y', 'x', 'y'],
                          [pointer[1], pointer[2], pointer[1]+x_shift, pointer[2]-y_shift])
del pointer[1]
del pointer[2]

# link pointer with corresponding nucleoli
pointer['nucleoli'] = obj.points_in_objects(nucleoli, pointer['x'], pointer['y'])

# create analysis mask for all analysis points
analysis_spots = ana.analysis_mask(t0_pix, pointer['y'], pointer['x'], num_dilation)

# create analysis mask for control spots
ctrl_nucleoli = ~nucleoli_pd.index.isin(pointer['nucleoli'].tolist())
ctrl_centroids_x = nucleoli_pd[ctrl_nucleoli]['centroid_x'].astype(int).tolist()
ctrl_centroids_y = nucleoli_pd[ctrl_nucleoli]['centroid_y'].astype(int).tolist()
ctrl_spots = ana.analysis_mask(t0_pix, ctrl_centroids_x, ctrl_centroids_y, num_dilation)

# link pointer with corresponding analysis spots
pointer['analysis_spots'] = obj.points_in_objects(analysis_spots, pointer['x'], pointer['y'])

# filter out analysis spots:
# 1) aim outside of nucleoli
# 2) bleach the same nucleoli
# 3) too close to merge as a single analysis spots
pointer_target_same_nucleoli = \
    [item for item, count in collections.Counter(pointer['nucleoli'].tolist()).items() if count > 1]
pointer_same_analysis_spots = \
    [item for item, count in collections.Counter(pointer['analysis_spots'].tolist()).items() if count > 1]
pointer_ft = pointer[(pointer['nucleoli'] > 0)
                     & (~pointer['nucleoli'].isin(pointer_target_same_nucleoli))
                     & (~pointer['analysis_spots'].isin(pointer_same_analysis_spots))].reset_index(drop=True)

# --------------------------------------------------
# FRAP CURVE GENERATION from filtered analysis spots
# --------------------------------------------------
print("### FRAP curve generation from filtered analysis spots ...")
# create analysis mask for filtered analysis spots
analysis_spots_ft = ana.analysis_mask(t0_pix, pointer_ft['y'], pointer_ft['x'], num_dilation)
print("%d spots passed filter for analysis." % obj.object_count(analysis_spots_ft))

# link pointer with corresponding filtered analysis spots
pointer_ft['analysis_spots_ft'] = obj.points_in_objects(analysis_spots_ft, pointer_ft['x'], pointer_ft['y'])

# measure pointer corresponding nucleoli sizes
pointer_nucleoli_sizes = []
for i in range(len(pointer_ft)):
    pointer_nucleoli_sizes.append(nucleoli_pd['size'][pointer_ft['nucleoli'][i]])
pointer_ft['size'] = pointer_nucleoli_sizes

# create stack for time series
t_pixels = []
# get acquisition time
t_time = []
# measure mean intensity for filtered analysis spots and control spots
t_int_analysis_spots_ft = [[] for _ in range(obj.object_count(analysis_spots_ft))]
t_int_ctrl_spots = [[] for _ in range(obj.object_count(ctrl_spots))]
for t in range(0, max_t):
    img = store.get_image(cb.t(t).build())
    acq_time = img.get_metadata().get_received_time().split(' ')[1]
    t_time.append(acq_time)

    pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
    t_pixels.append(pixels)
    analysis_spots_ft_pix = regionprops(label(analysis_spots_ft), pixels)
    ctrl_spots_pix = regionprops(label(ctrl_spots), pixels)
    for i in range(len(analysis_spots_ft_pix)):
        t_int_analysis_spots_ft[i].append(analysis_spots_ft_pix[i].mean_intensity)
    for i in range(len(ctrl_spots_pix)):
        t_int_ctrl_spots[i].append(ctrl_spots_pix[i].mean_intensity)

movies = np.stack(t_pixels, axis=0)

real_time = []
for i in range(max_t):
    real_time.append(dat.get_time_length(0,i,t_time))

# --------------------------
# FRAP CURVE CORRECTION
# --------------------------
print("### FRAP curve correction ...")
# background correction
t_bg_int = ana.get_bg_int(t_pixels)
t_int_analysis_spots_ft_cor = ana.bg_correction(t_int_analysis_spots_ft, t_bg_int)
t_int_ctrl_spots_cor = ana.bg_correction(t_int_ctrl_spots, t_bg_int)
# calculate photobleaching factor
pb_factor = ana.get_pb_factor(t_int_ctrl_spots_cor)
print("%d ctrl points are used to correct photobleaching." % obj.object_count(ctrl_spots))
# photobleaching correction
t_int_analysis_spots_ft_cor = ana.pb_correction(t_int_analysis_spots_ft_cor, pb_factor)
# add corrected intensities into pointer_ft
pointer_ft = dat.add_object_measurements(pointer_ft, 'mean_int', 'analysis_spots_ft', t_int_analysis_spots_ft_cor)

# --------------------------
# MEASUREMENTS
# --------------------------
# for all the filtered analysis spots
bleach_frame_pointer_fl = []  # frame number of or right after photobleaching
min_int_frame = []  # frame number of the minimum intensity
t_int_post = []  # intensity series after minimum intensity (includes min_int_frame, frap recovery curve)
imaging_length = []  # number of frames of t_int_post
t_int_pre = []  # intensity series before photobleaching (without bleach_frame, before spike)
pre_bleach_int = []  # mean intensity before photobleaching; pre-bleach intensity
min_int = []  # minimum intensity after photobleaching
t_int_post_nor = []  # t_int_post normalized with pre_bleach_int and min_int
real_time_post = []  # time series represents in second
mean_int_nor = []  # intensity series normalized with pre_bleach_int and min_int
plateau_int = []  # plateau level intensity
plateau_int_nor = []  # int_plateau normalized with pre_bleach_int and min_int; mobile fraction
immobile_fraction = []  # 1-plateau_int_nor
half_int = []  # half intensity
half_int_nor = []  # int_half normalized with pre_bleach_int and min_int
half_frame = []  # number of frames it takes to reach half intensity (min_int_frame, half_int_frame]
t_half = []  # t-half
slope = []  # initial slope of the recovery curve (relative intensity)

for i in range(len(pointer_ft)):
    # number of first frame after photobleaching (num_pre)
    num_pre = dat.find_pos(pointer_ft[0][i].split(' ')[1], t_time)
    bleach_frame_pointer_fl.append(num_pre)
    # the frame of min intensity
    min_int_frame_temp = pointer_ft['mean_int'][i].tolist().index(min(pointer_ft['mean_int'][i]))
    min_int_frame.append(min_int_frame_temp)
    # imaging length of the frap curve after min_int_frame
    num_post = max_t - min_int_frame_temp
    imaging_length.append(num_post)
    # intensities before photobleaching and intensities after min_intensity
    int_post = pointer_ft['mean_int'][i][-num_post:]
    int_pre = pointer_ft['mean_int'][i][:num_pre]
    t_int_pre.append(int_pre)
    t_int_post.append(int_post)
    # time series represents in sec
    real_time_post.append([x - real_time[-num_post] for x in real_time[-num_post:]])
    # mean intensity before photobleaching
    pre_bleach_int_temp = np.mean(int_pre)
    pre_bleach_int.append(pre_bleach_int_temp)
    # minimum intensity after photobleaching
    min_int_temp = np.min(int_post)
    min_int.append(min_int_temp)
    # normalized intensities after min_intensity based on pre_bleach_int and min_int
    full_range_int = pre_bleach_int_temp-min_int_temp
    t_int_post_nor.append([(x - min_int_temp)/full_range_int for x in int_post])
    # intensity normalized based on pre_bleach_int and min_int
    mean_int_nor_temp = [(x - min_int_temp)/full_range_int for x in pointer_ft['mean_int'][i]]
    mean_int_nor.append(mean_int_nor_temp)
    # plateau level intensity calculated from last 10 frames of the frap curve
    plateau_int_temp = np.mean(pointer_ft['mean_int'][i][-10:])
    plateau_int.append(plateau_int_temp)
    plateau_int_nor_temp = (plateau_int_temp - min_int_temp) / full_range_int
    plateau_int_nor.append(plateau_int_nor_temp)
    immobile_fraction_temp = 1-plateau_int_nor_temp
    immobile_fraction.append(immobile_fraction_temp)
    # half intensity
    half_int_temp = 0.5 * (min_int_temp + plateau_int_temp)
    half_int.append(half_int_temp)
    half_int_nor_temp = (half_int_temp - min_int_temp) / full_range_int
    half_int_nor.append(half_int_nor_temp)
    # number of frames it take to reach half intensity
    half_frame_temp = dat.find_pos(half_int_temp, int_post)
    half_frame.append(half_frame_temp)
    # t_half (sec)
    t_half_temp = dat.get_time_length(min_int_frame_temp,
                                      min_int_frame_temp + half_frame_temp, t_time)
    t_half.append(t_half_temp)
    # initial slope calculated based on first 5 frames
    int_change = (pointer_ft['mean_int'][i][min_int_frame_temp + 5] - min_int_temp)/full_range_int
    t_change = dat.get_time_length(min_int_frame_temp, min_int_frame_temp + 5, t_time)
    slope_temp = 1.0 * (int_change / t_change)
    slope.append(slope_temp)

pointer_ft = dat.add_columns(pointer_ft, ['int_curve_nor', 'bleach_frame', 'min_int_frame', 'imaging_length',
                                          'int_curve_pre', 'int_curve_post', 'int_curve_post_nor', 'real_time_post',
                                          'pre_bleach_int', 'min_int', 'plateau_int', 'mobile_fraction',
                                          'immobile_fraction', 'half_int', 'half_int_nor', 'half_frame',
                                          't_half', 'ini_slope'],
                             [mean_int_nor, bleach_frame_pointer_fl, min_int_frame, imaging_length,
                              t_int_pre, t_int_post, t_int_post_nor, real_time_post,
                              pre_bleach_int, min_int, plateau_int, plateau_int_nor,
                              immobile_fraction, half_int, half_int_nor, half_frame,
                              t_half, slope])

# --------------------------
# CURVE FITTING
# --------------------------
single_exp_a = []
single_exp_b = []
single_exp_fit = []
single_exp_r2 = []
single_exp_t_half = []

for i in range(len(pointer_ft)):
    popt, _ = curve_fit(mat.single_exp, pointer_ft['real_time_post'][i], pointer_ft['int_curve_post_nor'][i])
    a, b = popt
    y_fit = []
    for j in range(len(pointer_ft['real_time_post'][i])):
        y_fit.append(mat.single_exp(pointer_ft['real_time_post'][i][j], a, b))
    r2 = mat.r_square(pointer_ft['int_curve_post_nor'][i], y_fit)
    t_half_fit = np.log(0.5)/(-b)
    single_exp_a.append(a)
    single_exp_b.append(b)
    single_exp_fit.append(y_fit)
    single_exp_r2.append(r2)
    single_exp_t_half.append(t_half_fit)

pointer_ft = dat.add_columns(pointer_ft, ['single_exp_fit', 'single_exp_r2', 'single_exp_a', 'single_exp_b',
                                          'single_exp_mobile_fraction', 'single_exp_t_half'],
                             [single_exp_fit, single_exp_r2, single_exp_a, single_exp_b,
                              single_exp_a, single_exp_t_half])

# --------------------------
# OUTPUT FILE
# --------------------------
pointer_ft.to_csv('%s/data_full.txt'% data_path, index=None, sep='\t')
pointer_out = pd.DataFrame({'x': pointer_ft['x'],
                            'y': pointer_ft['y'],
                            'corresponding_nucleoli_size': pointer_ft['size'],
                            'bleach_frame': pointer_ft['bleach_frame'],
                            'pre_bleach_int': pointer_ft['pre_bleach_int'],
                            'min_int': pointer_ft['min_int'],
                            'mobile_fraction': pointer_ft['mobile_fraction'],
                            't_half (s)': pointer_ft['t_half'],
                            'single_exp_r2': pointer_ft['single_exp_r2'],
                            'single_exp_mobile_fraction': pointer_ft['single_exp_mobile_fraction'],
                            'single_exp_t_half': pointer_ft['single_exp_t_half']})
pointer_out.to_csv('%s/data.txt'% data_path, index=None, sep='\t')

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

    # napari display
    # Layer1: data
    # display time series movies in napari main viewer
    viewer.add_image(movies, name='data')

    # Layer2: nucleoli
    # display nucleoli mask (violet)
    viewer.add_image(nucleoli, name='nucleoli', contrast_limits=[0, 1],
                     colormap=('violet woBg', violet_woBg))

    # Layer3: aim points
    # display aim points from .log file (red)
    points = np.column_stack((pointer['aim_y'].tolist(), pointer['aim_x'].tolist()))
    size = [3] * len(points)
    viewer.add_points(points, name='aim points', size=size, edge_color='r', face_color='r')

    # Layer4: analysis spots
    # display filtered analysis spots, color sorted based on corresponding nucleoli size
    # sort colormap based on analysis spots filtered
    rgba_winter = dis.num_color_colormap('winter', len(pointer_ft))[1]
    winter_woBg = dis.sorted_num_color_colormap(rgba_winter, pointer_ft, 'size', 'analysis_spots_ft')[0]
    if len(pointer_ft) != 0:
        viewer.add_image(label(analysis_spots_ft), name='analysis spots', colormap=('winter woBg', winter_woBg))

    """
    # nucleoli labels
    label_nucleoli = label(nucleoli)
    nucleoli_properties = {'size': ['none'] + list(nucleoli_areas)}  # background is size: none
    label_layer = viewer.add_labels(label_nucleoli, name='nucleoli label', properties=nucleoli_properties, num_colors=3)
    """

    """
    # display control points
    viewer.add_image(ctrl_spots, name='ctrl points', colormap=('red woBg',red_woBg))
    """

    # matplotlib display
    # sorted based on nucleoli size (color coded)
    pointer_sort = pointer_ft.sort_values(by='size').reset_index(drop=True)  # from small to large

    # Plot-left: FRAP curves of filtered analysis spots after intensity correction (absolute intensity)
    for i in range(len(pointer_sort)):
        ax1.plot(pointer_sort['mean_int'][i], color=rgba_winter[i])
    ax1.set_title('FRAP curves')
    ax1.set_xlabel('time')
    ax1.set_ylabel('intensity')

    """
    # Plot-left: FRAP curves of filtered analysis spots after intensity correction (relative intensity)
    for i in range(len(pointer_sort)):
        ax1.plot(pointer_sort['mean_int_nor'][i], color=rgba_winter[i])
    ax1.set_title('FRAP curves')
    ax1.set_xlabel('time')
    ax1.set_ylabel('intensity')
    """

    """
    # Plot-right: FRAP curves of filtered analysis spots after intensity correction 
    # absolute intensity, bleach time zero aligned
    for i in range(len(pointer_sort)):
        ax2.plot(np.arange(len(pointer_sort['mean_int_post'][i])), pointer_sort['mean_int_post'][i], color=rgba_winter[i])
    ax2.set_title('FRAP curves')
    ax2.set_xlabel('time')
    ax2.set_ylabel('intensity')
    """

    # Plot-right: FRAP curves of filtered analysis spots after intensity correction
    # relative intensity, bleach time zero aligned
    for i in range(len(pointer_sort)):
        ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['int_curve_post_nor'][i],
                 color=rgba_winter[i], alpha=0.5)
        ax2.plot(pointer_sort['real_time_post'][i], pointer_sort['single_exp_fit'][i], '--',
                 color=rgba_winter[i])
    ax2.set_title('FRAP curves')
    ax2.set_xlabel('time (sec)')
    ax2.set_ylabel('intensity')

    """
    # Plot-right: photobleaching curves of control spots before photobleaching correction
    for i in range(len(t_int_ctrl_spots)):
        ax2.plot(t_int_ctrl_spots[i], color=[1.0, 0.0, 0.0, 1.0])
    ax2.set_title('Photobleaching curves')
    ax2.set_xlabel('time')
    ax2.set_ylabel('intensity')
    """
