# main imports
import numpy as np
import pandas as pd
import napari
from pycromanager import Bridge

# for mpl widget embedding
from matplotlib.backends.qt_compat import QtCore, QtWidgets
if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import cm

# skimage
from skimage.filters import threshold_otsu, threshold_yen
from skimage.measure import label, regionprops
from skimage import morphology, segmentation

# .py
from shared.find_blobs import find_blobs
from shared.remove_objects import remove_small, remove_large

# others
from vispy.color import Colormap
import collections



# nomenclature notes
# aim points: all shoot points retrieved from .log file
# bleach points: filtered aim points
#       filters: 1) aim points that are too close (merge together when generating analysis mask)



# constant values
global_thresholding = 'na'  # choose in between 'na','otsu' and 'yen'; default = 'na'
min_nucleoli_size = 10  # minimum nucleoli size; default = 10
max_nucleoli_size = 1000    # maximum nucleoli size; default = 1000; larger ones are generally cells without nucleoli
dilation_round = 3  # analysis size of the bleach points; default = 3
x_shift = 0   # positive: right; default = 0
y_shift = 0   # positive: up; default = 0

# colormaps
dark_violetred_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129/255, 55/255, 114/255, 1.0]])
red_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
cmap_winter = cm.get_cmap('winter')

# data source
# data_path = "C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoili-bleaching-4x\\PythonAcq1\\AutoBleach_15"
#data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/TestedData/20201216/Ctrl-2DG-CCCP-36pos_partial/exp_37"
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/TestedData/20210109/B3-Site_0_1"



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

# test image of time 0
test1 = store.get_image(cb.t(0).build())
test1_pix = np.reshape(test1.get_raw_pixels(), newshape=[test1.get_height(), test1.get_width()])

# image analysis based on image of time 0
# find organelles using a combination of thresholding and watershed
if global_thresholding == 'na':
    nucleoli = find_blobs(test1_pix, 0, 500, 200)
elif global_thresholding == 'otsu':
    nucleoli = find_blobs(test1_pix, threshold_otsu(test1_pix), 500, 200)
elif global_thresholding == 'yen':
    nucleoli = find_blobs(test1_pix, threshold_yen(test1_pix), 500, 200)

# nucleoli filter:
# remove artifacts connected to image border
# size filter [10,1000]
nucleoli_filtered = segmentation.clear_border(nucleoli)
nucleoli_filtered = remove_small(nucleoli_filtered, min_nucleoli_size)
nucleoli_filtered = remove_large(nucleoli_filtered, max_nucleoli_size)
label_nucleoli_filtered = label(nucleoli_filtered)
nucleoli_prop = regionprops(label_nucleoli_filtered)
# get the size of each nucleoli
nucleoli_areas = np.bincount(label_nucleoli_filtered.ravel())[1:]
# get the centroid of each nucleoli
nucleoli_centroid_x = []
nucleoli_centroid_y = []
for i in range(len(nucleoli_prop)):
    nucleoli_centroid_x.append(nucleoli_prop[i].centroid[0])
    nucleoli_centroid_y.append(nucleoli_prop[i].centroid[1])

# nucleoli labels
nucleoli_properties = {'size': ['none'] + list(nucleoli_areas)}  # background is size: none

# nucleoli dataset
nucleoli_pd = pd.DataFrame()
nucleoli_pd['size'] = nucleoli_areas
nucleoli_pd['centroid_x'] = nucleoli_centroid_x
nucleoli_pd['centroid_y'] = nucleoli_centroid_y

# load point_and_shoot log file
pointer = pd.read_csv('%s/PointAndShoot.log'%data_path,na_values=['.'],sep='\t', header = None)
pointer['aim_x'] = pointer[1]
pointer['aim_y'] = pointer[2]
pointer[1] = pointer[1] + x_shift
pointer[2] = pointer[2] - y_shift

# link pointer with corresponding nucleoli
pointer_in_nucleoli = []
for i in range(len(pointer)):
    pointer_in_nucleoli.append(label_nucleoli_filtered[pointer[2][i], pointer[1][i]] - 1)
pointer['nucleoli'] = pointer_in_nucleoli


def analysis_mask(img_example,num_dilation,listx,listy):
    mask = np.zeros_like(img_example)
    if len(listx) == len(listy):
        for i in range(len(listx)):
            mask[listx[i],listy[i]] = 1
        for i in range(num_dilation):
            mask = morphology.binary_dilation(mask)
    else:
        print("Input error: length of x and y does not match")
    return mask


# create analysis mask for all aim points
aimpoints = analysis_mask(test1_pix, dilation_round, pointer[2].tolist(), pointer[1].tolist())
label_aimpoints = label(aimpoints)
aimpoints_prop = regionprops(label_aimpoints)

# create analysis mask for control points
ctrlpoints_x = nucleoli_pd[~nucleoli_pd.index.isin(pointer['nucleoli'].tolist())]['centroid_x'].astype(int).tolist()
ctrlpoints_y = nucleoli_pd[~nucleoli_pd.index.isin(pointer['nucleoli'].tolist())]['centroid_y'].astype(int).tolist()
ctrlpoints = analysis_mask(test1_pix, dilation_round, ctrlpoints_x, ctrlpoints_y)
label_ctrlpoints = label(ctrlpoints)
ctrlpoints_prop = regionprops(label_ctrlpoints)

# link pointer with corresponding aim points
pointer_in_aimpoints = []
for i in range(len(pointer)):
    pointer_in_aimpoints.append(label_aimpoints[pointer[2][i], pointer[1][i]] - 1)
pointer['aimpoints'] = pointer_in_aimpoints

# filter out bleach points:
# 1) aim outside of nucleoli
# 2) bleach the same nucleoli
# 3) too close to merge as a single bleach points
pointer_target_same_nucleoli = [item for item, count in collections.Counter(pointer['nucleoli'].tolist()).items() if count > 1]
pointer_merge_in_same_aimpoints = [item for item, count in collections.Counter(pointer['aimpoints'].tolist()).items() if count > 1]
pointer_filtered = pointer[(pointer['nucleoli'] > 0)
                           & (~pointer['nucleoli'].isin(pointer_target_same_nucleoli))
                           & (~pointer['aimpoints'].isin(pointer_merge_in_same_aimpoints))].reset_index()

# create analysis mask for filtered aim points, i.e. bleach points
bleachpoints = analysis_mask(test1_pix, dilation_round, pointer_filtered[2].tolist(), pointer_filtered[1].tolist())
label_bleachpoints = label(bleachpoints)

# link pointer with corresponding bleachpoints
pointer_in_bleachpoints = []
for i in range(len(pointer_filtered)):
    pointer_in_bleachpoints.append(label_bleachpoints[pointer_filtered[2][i], pointer_filtered[1][i]] - 1)
pointer_filtered['bleachpoints'] = pointer_in_bleachpoints

# measure pointer corresponding nucleoli sizes
pointer_nucleoli_sizes = []
for i in range(len(pointer_filtered)):
    pointer_nucleoli_sizes.append(nucleoli_pd['size'][pointer_filtered['nucleoli'][i]])
pointer_filtered['size'] = pointer_nucleoli_sizes

# sort pointer for plotting
pointer_sort = pointer_filtered.sort_values(by='size').reset_index()   # from small to large

# create stack for time series
t_pixels = []
# measure mean intensity for bleach points and control points
t_meanInt_bleachpoints = [[] for _ in range(len(pointer_filtered))]
t_meanInt_ctrlpoints = [[] for _ in range(len(ctrlpoints_prop))]
for t in range(0, max_t):
    img = store.get_image(cb.t(t).build())
    pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
    t_pixels.append(pixels)

    bleach_points = regionprops(label_bleachpoints, pixels)
    ctrl_points = regionprops(label_ctrlpoints, pixels)
    for i in range(len(bleach_points)):
        t_meanInt_bleachpoints[i].append(bleach_points[i].mean_intensity)
    for i in range(len(ctrl_points)):
        t_meanInt_ctrlpoints[i].append(ctrl_points[i].mean_intensity)
movies = np.stack(t_pixels, axis=0)

# calculate photobleaching factor
photobleaching_factor = []
for t in range(max_t):
    photobleaching_ratio_lst = []
    for i in range(len(t_meanInt_ctrlpoints)):
        photobleaching_ratio_lst.append(np.mean(t_meanInt_ctrlpoints[i][t])/np.mean(t_meanInt_ctrlpoints[i][0]))
    photobleaching_factor.append(np.mean(photobleaching_ratio_lst))

# correct photobleaching
t_meanInt_bleachpoints_pbcorrected = []
for i in range(len(t_meanInt_bleachpoints)):
    t_meanInt_bleachpoints_pbcorrected.append(np.divide(t_meanInt_bleachpoints[i],photobleaching_factor))

# image display
with napari.gui_qt():
    # embed mpl widget in napari viewer
    mpl_widget = FigureCanvas(Figure(figsize=(5,3)))
    [ax1,ax2] = mpl_widget.figure.subplots(nrows=1,ncols=2)
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(mpl_widget)

    # display time series movies in napari main viewer
    viewer.add_image(movies, name='data')

    # display nucleoli mask
    label_layer = viewer.add_labels(label_nucleoli_filtered, name='nucleoli label', properties=nucleoli_properties, num_colors=3)
    viewer.add_image(nucleoli_filtered, name='nucleoli', contrast_limits=[0, 1], colormap=('dark violetred woBg', dark_violetred_woBg))

    # generate colormap based on the number of bleach points
    if len(pointer_filtered) != 0:
        rgba_winter = cmap_winter(np.arange(0, 1, 1 / len(pointer_filtered)))
        rgba_winter_woBg = np.insert(rgba_winter, 0, [0.0, 0.0, 0.0, 0.0], axis=0)
        rgba_winter_sort = [rgba_winter_woBg[0]]
        for i in pointer_sort.sort_values(by='bleachpoints').index.tolist():
            rgba_winter_sort.append(rgba_winter_woBg[i + 1])
        winter_woBg = Colormap(rgba_winter_sort)
    else:
        print("no bleach points")

    # display control points
    #viewer.add_image(ctrlpoints, name='ctrl points', colormap=('red woBg',red_woBg))

    # display point_and_shoot aim points
    # create points for bleach points
    points = np.column_stack((pointer['aim_y'].tolist(), pointer['aim_x'].tolist()))
    size = [3]*len(points)
    viewer.add_points(points, name='aim points', size=size, edge_color='r', face_color='r')

    # display filtered bleach points
    if len(pointer_filtered) != 0:
        viewer.add_image(label_bleachpoints, name='bleach points', colormap=('winter woBg', winter_woBg))

    # plot FRAP curves of bleach points (photobleaching corrected)
    for i in range(len(t_meanInt_bleachpoints_pbcorrected)):
        ax1.plot(t_meanInt_bleachpoints_pbcorrected[pointer_sort['bleachpoints'][i]], color=rgba_winter[i])
    ax1.set_title('FRAP curves')
    ax1.set_xlabel('time')
    ax1.set_ylabel('intensity')

    # example of photobleaching correction effect
    #ax2.plot(t_meanInt_bleachpoints[0], color='r')
    #ax2.plot(t_meanInt_bleachpoints_pbcorrected[0], color='b')
    #ax2.set_title('Before (red) and after (blue) photobleaching correction')
    #ax2.set_xlabel('time')
    #ax2.set_ylabel('intensity')

    # plot photobleaching curves of control points before photobleaching correction
    for i in range(len(t_meanInt_ctrlpoints)):
        ax2.plot(t_meanInt_ctrlpoints[i], color=[1.0,0.0,0.0,1.0])
    ax2.set_title('Photobleaching curves')
    ax2.set_xlabel('time')
    ax2.set_ylabel('intensity')

