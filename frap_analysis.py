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
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import morphology, segmentation

# .py
import find_blobs

# others
from vispy.color import Colormap
import collections



# constant values
dilation_round = 3  # analysis size of the bleach points

# colormaps
dark_violetred_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129/255, 55/255, 114/255, 1.0]])
red_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
cmap_winter = cm.get_cmap('winter')

# data source
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/TestedData/data20201116/AutoBleach_15"



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
segmented = find_blobs.find_blobs(test1_pix, threshold_otsu(test1_pix), 500, 200)
# remove artifacts connected to image border and nucleoli less than 5
segmented_filtered = morphology.remove_small_objects(segmentation.clear_border(segmented), 5)
label_img = label(segmented_filtered)
nucleoli = regionprops(label_img)
# get the size of each nucleoli
label_areas = np.bincount(label_img.ravel())[1:]
# get the centroid of each nucleoli
label_centroid_x = []
label_centroid_y = []
for i in range(len(nucleoli)):
    label_centroid_x.append(nucleoli[i].centroid[0])
    label_centroid_y.append(nucleoli[i].centroid[1])

label_properties = {
    'size': ['none'] + list(label_areas),  # background is size: none
    }


# nucleoli dataset
nucleoli_pd = pd.DataFrame()
nucleoli_pd['size'] = label_areas
nucleoli_pd['centroid_x'] = label_centroid_x
nucleoli_pd['centroid_y'] = label_centroid_y

# load point_and_shoot log file
pointer = pd.read_csv('%s/PointAndShoot.log'%data_path,na_values=['.'],sep='\t', header = None)

# link pointer with corresponding nucleoli
pointer_in_nucleoli = []
for i in range(len(pointer)):
    pointer_in_nucleoli.append(label_img[pointer[2][i],pointer[1][i]]-1)
pointer['nucleoli'] = pointer_in_nucleoli
pointer_target_same_nucleoli = [item for item, count in collections.Counter(pointer['nucleoli'].tolist()).items() if count > 1]
# filter out bleach points:
# 1) aim outside of nucleoli
# 2) bleach the same nucleoli
pointer_filtered = pointer[(pointer['nucleoli'] > 0) & (~pointer['nucleoli'].isin(pointer_target_same_nucleoli))].reset_index()

# measure pointer corresponding nucleoli sizes
pointer_nucleoli_sizes = []
for i in range(len(pointer_filtered)):
    pointer_nucleoli_sizes.append(nucleoli_pd['size'][pointer_filtered['nucleoli'][i]])
pointer_filtered['size'] = pointer_nucleoli_sizes

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

# create analysis mask for filtered bleach points
bleachpoints = analysis_mask(test1_pix, dilation_round, pointer_filtered[2].tolist(), pointer_filtered[1].tolist())
label_bleachpoints = label(bleachpoints)

# create analysis mask for control points
ctrlpoints_x = nucleoli_pd[~nucleoli_pd.index.isin(pointer['nucleoli'].tolist())]['centroid_x'].astype(int).tolist()
ctrlpoints_y = nucleoli_pd[~nucleoli_pd.index.isin(pointer['nucleoli'].tolist())]['centroid_y'].astype(int).tolist()
ctrlpoints = analysis_mask(test1_pix, dilation_round, ctrlpoints_x, ctrlpoints_y)
label_ctrlpoints = label(ctrlpoints)

# link pointer with corresponding bleachpoints
pointer_in_bleachpoints = []
for i in range(len(pointer_filtered)):
    pointer_in_bleachpoints.append(label_bleachpoints[pointer_filtered[2][i],pointer_filtered[1][i]]-1)
pointer_filtered['bleachpoints'] = pointer_in_bleachpoints

# sort pointer for plotting
pointer_sort = pointer_filtered.sort_values(by='size').reset_index()   # from small to large

# create stack for time series
t_pixels = []
# measure mean intensity for bleach points and control points
t_meanInt_bleachpoints = [[] for _ in range(len(pointer_filtered))]
t_meanInt_ctrlpoints = [[] for _ in range(len(ctrlpoints_x))]
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
    label_layer = viewer.add_labels(label_img, name='nucleoli label', properties=label_properties, num_colors=3)
    viewer.add_image(segmented_filtered, name='nucleoli', contrast_limits=[0,1], colormap=('dark violetred woBg',dark_violetred_woBg))

    # generate colormap based on the number of bleach points
    rgba_winter = cmap_winter(np.arange(0, 1, 1 / len(pointer_filtered)))
    rgba_winter_woBg = np.insert(rgba_winter, 0, [0.0, 0.0, 0.0, 0.0], axis=0)
    rgba_winter_sort = [rgba_winter_woBg[0]]
    for i in pointer_sort.sort_values(by='bleachpoints').index.tolist():
        rgba_winter_sort.append(rgba_winter_woBg[i + 1])
    winter_woBg = Colormap(rgba_winter_sort)

    # display control points
    #viewer.add_image(ctrlpoints, name='ctrl points', colormap=('red woBg',red_woBg))

    # display point_and_shoot points
    # create points for bleach points
    # points = np.column_stack((log[2].tolist(), log[1].tolist()))
    # size = [3]*len(points)
    #bleach_point = viewer.add_points([140,107], name='test points', size=[3], edge_color='r', face_color='r')
    viewer.add_image(label_bleachpoints, name='bleach points', colormap=('winter woBg',winter_woBg))

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

