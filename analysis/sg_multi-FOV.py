import numpy as np
import pandas as pd
import napari
from pycromanager import Bridge
from matplotlib import pyplot as plt
from shared.find_organelles import organelle_analysis, find_organelle
import shared.dataframe as dat
import shared.analysis as ana
import shared.display as dis
from skimage.measure import label
import shared.objects as obj
import os

"""
# ---------------------------------------------------------------------------------------------------
# SG ANALYZER (MULTI-FOV)
# ---------------------------------------------------------------------------------------------------

EXPECTS 
    an uManager data, 
SEGMENTS and ANALYZES
    SG properties (enables size, mean intensity, circularity and eccentricity) for all positions with 
    the same time/channel/z-plane, 
EXPORTS 
    measurements (including position in uManager data, SG number, x, y position, size, mean intensity,
    circularity and eccentricity) in .txt format and stitched images (raw image, SG mask, color coded
    mean intensity/circularity/eccentricity data) in .pdf format, 
DISPLAYS 
    stitched images (raw image, SG mask, color coded mean_intensity/circularity/eccentricity data) in 
    napari. 

# ----------------------------------
# PARAMETERS ALLOW CHANGE
# ----------------------------------

    # paths
    data_path: directory of uManager data
    save_path: primary directory for output saving
    
    # values for analysis
    data_t: time/frame to be analyzed
    data_c: channel to be analyzed
    data_z: z-plane to be analyzed
    thresholding: global thresholding method used for SG segmentation; only accepts 'na', 'otsu', 
        'yen', 'local-nucleoli' and 'local-sg'
    min_size: the smallest allowable SG size
    max_size: the largest allowable SG size
    
    # modes
    analyze_boundary: if SG fall at the boundaries of two neighbour images after image stitch are 
        subject to analysis or not; only accepts 'N' and 'Y'
    export_mode: enables export or not; only accepts 'N' and 'Y'
        export_pd_pre_stitch: exports SG measurements (.txt) based on uManager position/SG number/
            x,y coordinates on original FOV or not; only accepts 'N' or 'Y'; 'Y' is functional only 
            while export_mode == 'Y'
        export_pd_post_stitch: exports SG measurements (.txt) based on uManager position/SG number/
            x,y coordinates on stitched image or not; only accepts 'N' or 'Y'; 'Y' is functional only 
            while export_mode == 'Y'
        export_img: exports stitched images (.pdf) or not; only accepts 'N' or 'Y'; 'Y' is functional 
            only while export_mode == 'Y'
    display_mode: displays stitched images in napari or not; only accepts 'N' or 'Y'
    
    # color-coded (cc) images calculation (added due to time concern)
    # Note: calculates color coded images for image export/display is the time limiting step of 
        current codes, please toggle off corresponding features if not needed
    cc_circ: calculates color coded circularity image or not; only accepts 'N' or 'Y'
    cc_ecce: calculates color coded eccentricity image or not; only accepts 'N' or 'Y'
    cc_int: calculates color coded mean intensity image or not; only accepts 'N' or 'Y'
"""

# --------------------------
# PARAMETERS ALLOW CHANGE
# --------------------------
# paths
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/20210100_SG_scoring/WT"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/20210100_SG_scoring/dataAnalysis1/"

# values for analysis
data_t = 0  # non-negative int, make sure within data range
data_c = 0  # non-negative int, make sure within data range
data_z = 0  # non-negative int, make sure within data range
thresholding = 'local-sg'  # only accepts 'na', 'otsu', 'yen', 'local-nucleoli' and 'local-sg'
min_size = 5  # non-negative int
max_size = 350  # non-negative int

# modes
analyze_boundary = 'N'  # only accepts 'N' or 'Y'
export_mode = 'Y'  # only accepts 'N' or 'Y'
export_pd_pre_stitch = 'Y'  # only accepts 'N' or 'Y'
export_pd_post_stitch = 'N'  # only accepts 'N' or 'Y'
export_img = 'Y'  # only accepts 'N' or 'Y'
display_mode = 'Y'  # only accepts 'N' or 'Y'

# color-coded (cc) images calculation (added due to time concern)
cc_circ = 'Y'  # only accepts 'N' or 'Y'
cc_ecce = 'N'  # only accepts 'N' or 'Y'
cc_int = 'Y'  # only accepts 'N' or 'Y'

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
max_p = store.get_max_indices().get_p()  # 24
num_grid = int(np.sqrt(max_p+1))
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

# ------------------------------
# IMAGE ANALYSIS
# ------------------------------
print("### Image analysis: calculate SG mask/pd for each FOV ...")
sg_fov_pd = pd.DataFrame()  # SG dataFrame of all FOVs based on each single FOV
row_lst = []  # row location of given FOV in multi-FOV-grid
col_lst = []  # column location of given FOV in multi-FOV-grid
pix_lst = []  # list of t(0).p(0) image for multi-FOV-display
sg_lst = []  # list of t(0).p(0) SG mask for multi-FOV-display
for i in range(max_p+1):
    temp = store.get_image(cb.t(data_t).c(data_c).z(data_z).p(i).build())
    temp_pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])
    temp_sg = find_organelle(temp_pix, thresholding, min_size=min_size, max_size=max_size)
    row, col = dat.get_grid_pos(i, num_grid)
    row_lst.append(row)
    col_lst.append(col)
    pix_lst.append(temp_pix)
    sg_lst.append(temp_sg)
    if export_pd_pre_stitch == 'Y':
        temp_sg_pd = organelle_analysis(temp_pix, temp_sg, 'sg', i)
        sg_fov_pd = pd.concat([sg_fov_pd, temp_sg_pd], ignore_index=True)

print("### Image analysis: Stitch image ...")
pix_pd = pd.DataFrame({'row': row_lst, 'col': col_lst, 'pix': pix_lst})  # dataFrame of t(0).p(0) pixel images
pix = ana.pix_stitch(pix_pd, num_grid, num_grid)  # stitched image

if analyze_boundary not in ['Y', 'N']:
    raise ValueError("analyze_boundary only accepts 'Y' and 'N'. Got %s" % analyze_boundary)
elif analyze_boundary == 'N':
    sg_pix_pd = pd.DataFrame({'row': row_lst, 'col': col_lst, 'pix': sg_lst})  # dataFrame of t(0).p(0) SG masks
    sg = ana.pix_stitch(sg_pix_pd, num_grid, num_grid)  # stitched SG masks (exclude boundary of each single FOV)
else:
    sg = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)

label_sg = label(sg, connectivity=1)
sg_pd = organelle_analysis(pix, sg, 'sg', 0)  # SG dataFrame based on multi-FOV stitched image

# --------------------------
# COLOR CODED IMAGES
# --------------------------
print("### Calculate color coded circ/ecce/int images ...")

cmap1 = 'YlOrRd'
cmap1_napari = dis.num_color_colormap(cmap1, 255)[0]
cmap1_plt = dis.num_color_colormap(cmap1, 255)[1]
if cc_circ == 'Y':
    sg_circ = obj.obj_display_in_circularity(label_sg)
else:
    sg_circ = np.zeros_like(sg)

cmap2 = 'Blues'
cmap2_napari = dis.num_color_colormap(cmap2, 255)[0]
cmap2_plt = dis.num_color_colormap(cmap2, 255)[1]
if cc_ecce == 'Y':
    sg_ecce = obj.obj_display_in_eccentricity(label_sg)
else:
    sg_ecce = np.zeros_like(sg)

cmap3 = 'viridis'
cmap3_napari = dis.num_color_colormap(cmap3, 255)[0]
cmap3_plt = dis.num_color_colormap(cmap3, 255)[1]
if cc_int == 'Y':
    sg_int = obj.obj_display_in_intensity(label_sg, pix, [6, 10])
else:
    sg_int = np.zeros_like(sg)

# --------------------------
# OUTPUT
# --------------------------
if export_mode == 'Y':
    print("### Export file ...")
    # check and create saving directory
    sample_name = data_path.split('"')[0].split('/')[-1]
    storage_path = '%s/%s/' % (save_path, sample_name)
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # SG dataFrame of all FOVs based on each single FOV
    if export_pd_pre_stitch == 'Y':
        sg_fov_pd.to_csv('%s/data_single-FOV_%s.txt' % (storage_path, sample_name), index=False, sep='\t')
    if export_pd_post_stitch == 'Y':
        sg_pd.to_csv('%s/data_multi-FOV_%s.txt' % (storage_path, sample_name), index=False, sep='\t')

    # Images
    if export_img == 'Y':
        # stitched pixel image
        plt.subplots(figsize=(8*num_grid, 8*num_grid))
        plt.imshow(pix, cmap='binary_r')
        plt.colorbar()
        plt.savefig('%s/pix_%s.pdf' % (storage_path, sample_name))

        # stitched SG mask
        plt.subplots(figsize=(8*num_grid, 8*num_grid))
        plt.imshow(sg, cmap='binary_r')
        plt.colorbar()
        plt.savefig('%s/sg_%s.pdf' % (storage_path, sample_name))

        # circularity
        if cc_circ == 'Y':
            plt.subplots(figsize=(8*num_grid, 8*num_grid))
            plt.imshow(sg_circ, cmap=cmap1_plt)
            plt.colorbar()
            plt.savefig('%s/circularity_%s.pdf' % (storage_path, sample_name))

        # eccentricity
        if cc_ecce == 'Y':
            plt.subplots(figsize=(8*num_grid, 8*num_grid))
            plt.imshow(sg_ecce, cmap=cmap2_plt)
            plt.colorbar()
            plt.savefig('%s/eccentricity_%s.pdf' % (storage_path, sample_name))

        # intensity
        if cc_int == 'Y':
            plt.subplots(figsize=(8 * num_grid, 8 * num_grid))
            plt.imshow(sg_int, cmap=cmap3_plt)
            plt.colorbar()
            plt.savefig('%s/intensity_%s.pdf' % (storage_path, sample_name))

# --------------------------
# OUTPUT DISPLAY
# --------------------------
if display_mode == 'Y':
    print("### Output display ...")
    with napari.gui_qt():
        viewer = napari.Viewer()

        # stitched pixel image
        viewer.add_image(pix, name='data')

        # stitched SG label with properties
        sg_properties = {
            'size': ['none'] + list(sg_pd['size']),  # background is size: none
            'int': ['none'] + list(sg_pd['raw_int']),
            'circ': ['none'] + list(sg_pd['circ']),
            'eccentricity': ['none'] + list(sg_pd['eccentricity'])
        }
        viewer.add_labels(label(sg), name='SG label', properties=sg_properties, num_colors=3)

        # circularity
        if cc_circ == 'Y':
            viewer.add_image(sg_circ, name='circ', colormap=('cmap1', cmap1_napari))

        # eccentricity
        if cc_ecce == 'Y':
            viewer.add_image(sg_ecce, name='ecce', colormap=('cmap2', cmap2_napari))

        # intensity
        if cc_int == 'Y':
            viewer.add_image(sg_int, name='int', colormap=('cmap3', cmap3_napari))
