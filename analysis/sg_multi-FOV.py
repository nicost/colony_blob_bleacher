import numpy as np
import pandas as pd
import napari
from pycromanager import Bridge
from matplotlib import pyplot as plt
from shared.find_organelles import sg_analysis, find_organelle
import shared.dataframe as dat
import shared.analysis as ana
import shared.display as dis
from skimage.measure import label
import shared.objects as obj
import os

# --------------------------
# PARAMETERS
# --------------------------
# paths
# data source
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/CX"
# storage path
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/dataAnalysis/"

# values
thresholding = 'local-sg'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
min_size = 5  # minimum SG size
max_size = 350  # maximum SG size

# analyzing mode
analyze_boundary = 'N'  # only accepts 'N' or 'Y'

# display mode
display_circ = 'Y'  # only accepts 'N' or 'Y'
display_ecce = 'N'  # only accepts 'N' or 'Y'
display_int = 'Y'  # only accepts 'N' or 'Y'

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
    temp = store.get_image(cb.t(0).p(i).build())
    temp_pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])
    temp_sg = find_organelle(temp_pix, thresholding, min_size=min_size, max_size=max_size)
    row, col = dat.get_grid_pos(i, num_grid)
    row_lst.append(row)
    col_lst.append(col)
    pix_lst.append(temp_pix)
    sg_lst.append(temp_sg)

    temp_sg_pd = sg_analysis(temp_pix, temp_sg, i)
    sg_fov_pd = pd.concat([sg_fov_pd, temp_sg_pd], ignore_index=True)

print("### Image analysis: Stitch image ...")
pix_pd = pd.DataFrame({'row': row_lst, 'col': col_lst, 'pix': pix_lst})  # dataFrame of t(0).p(0) pixel images
pix = ana.pix_stitch(pix_pd, num_grid)  # stitched image
if analyze_boundary == 'N':
    sg_pix_pd = pd.DataFrame({'row': row_lst, 'col': col_lst, 'pix': sg_lst})  # dataFrame of t(0).p(0) SG masks
    sg = ana.pix_stitch(sg_pix_pd, num_grid)  # stitched SG masks (exclude boundary of each single FOV)
elif analyze_boundary == 'Y':
    sg = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
sg_pd = sg_analysis(pix, sg, 0)  # SG dataFrame based on multi-FOV stitched image

# --------------------------
# OUTPUT IMAGES
# --------------------------
print("### Generating output images: Calculate group labeled circ/ecce image ...")
if display_circ == 'Y':
    cmap1 = 'YlOrRd'
    cmap1_napari = dis.num_color_colormap(cmap1, 255)[0]
    cmap1_plt = dis.num_color_colormap(cmap1, 255)[1]
    sg_circ = obj.obj_display_in_circularity(sg)

if display_ecce == 'Y':
    cmap2 = 'Blues'
    cmap2_napari = dis.num_color_colormap(cmap2, 255)[0]
    cmap2_plt = dis.num_color_colormap(cmap2, 255)[1]
    sg_ecce = obj.obj_display_in_eccentricity(sg)

if display_int == 'Y':
    cmap3 = 'viridis'
    cmap3_napari = dis.num_color_colormap(cmap3, 255)[0]
    cmap3_plt = dis.num_color_colormap(cmap3, 255)[1]
    sg_int = obj.obj_display_in_intensity(sg, pix, [6, 10])

# --------------------------
# OUTPUT DIR/NAMES
# --------------------------
# get sample name
sample_name = data_path.split('"')[0].split('/')[-1]
storage_path = '%s/%s/' % (save_path, sample_name)
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

# --------------------------
# OUTPUT FILE
# --------------------------
print("### Export file ...")
# SG dataFrame of all FOVs based on each single FOV
sg_fov_pd.to_csv('%s/data_single-FOV_%s.txt' % (storage_path, sample_name), index=None, sep='\t')
sg_pd.to_csv('%s/data_multi-FOV_%s.txt' % (storage_path, sample_name), index=None, sep='\t')

# Images
# stitched pixel image
fig, ax = plt.subplots(figsize=(8*num_grid, 8*num_grid))
ax = plt.imshow(pix, cmap='binary_r')
ax = plt.colorbar()
plt.savefig('%s/pix_%s.pdf' % (storage_path, sample_name))

# stitched SG mask
fig, ax = plt.subplots(figsize=(8*num_grid, 8*num_grid))
ax = plt.imshow(sg, cmap='binary_r')
ax = plt.colorbar()
plt.savefig('%s/sg_%s.pdf' % (storage_path, sample_name))

# circularity
if display_circ == 'Y':
    fig, ax = plt.subplots(figsize=(8*num_grid, 8*num_grid))
    ax = plt.imshow(sg_circ, cmap=cmap1_plt)
    ax = plt.colorbar()
    plt.savefig('%s/circularity_%s.pdf' % (storage_path, sample_name))

# eccentricity
if display_ecce == 'Y':
    fig, ax = plt.subplots(figsize=(8*num_grid, 8*num_grid))
    ax = plt.imshow(sg_ecce, cmap=cmap2_plt)
    ax = plt.colorbar()
    plt.savefig('%s/eccentricity_%s.pdf' % (storage_path, sample_name))

# intensity
if display_int == 'Y':
    fig, ax = plt.subplots(figsize=(8 * num_grid, 8 * num_grid))
    ax = plt.imshow(sg_int, cmap=cmap3_plt)
    ax = plt.colorbar()
    plt.savefig('%s/intensity_%s.pdf' % (storage_path, sample_name))

# --------------------------
# OUTPUT DISPLAY
# --------------------------
print("### Output display ...")
with napari.gui_qt():
    viewer = napari.Viewer()

    # stitched pixel image
    viewer.add_image(pix, name='data')

    # stitched SG label with properties
    sg_properties = {
        'size': ['none'] + list(sg_pd['size']),  # background is size: none
        'int': ['none'] + list(sg_pd['int']),
        'circ': ['none'] + list(sg_pd['circ']),
        'eccentricity': ['none'] + list(sg_pd['eccentricity'])
    }
    viewer.add_labels(label(sg), name='SG label', properties=sg_properties, num_colors=3)

    # circularity
    if display_circ == 'Y':
        viewer.add_image(sg_circ, name='circ', colormap=('cmap1', cmap1_napari))

    # eccentricity
    if display_ecce == 'Y':
        viewer.add_image(sg_ecce, name='ecce', colormap=('cmap2', cmap2_napari))

    # intensity
    if display_int == 'Y':
        viewer.add_image(sg_int, name='int', colormap=('cmap3', cmap3_napari))