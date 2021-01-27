import numpy as np
import napari
from pycromanager import Bridge
from matplotlib import pyplot as plt
from shared.find_organelles import sg_analysis, find_organelle
import shared.display as dis
from skimage.measure import label
import shared.objects as obj
import os

# --------------------------
# PARAMETERS
# --------------------------
# paths
# data source
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/WT"
# storage path
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/dataAnalysis"

# values
thresholding = 'local-sg'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
min_size = 5  # minimum SG size
max_size = 350  # maximum SG size
# analyzing position
pos = 2

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
temp = store.get_image(cb.t(0).p(pos).build())
pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

sg = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
sg_pd = sg_analysis(pix, sg, 0)

# --------------------------
# OUTPUT IMAGES
# --------------------------
print("### Generating output images: Calculate group labeled circ/ecce/int image ...")
# circ image
cmap1 = 'YlOrRd'
cmap1_napari = dis.num_color_colormap(cmap1, 255)[0]
cmap1_plt = dis.num_color_colormap(cmap1, 255)[1]
sg_circ = obj.obj_display_in_circularity(sg)

# ecce image
cmap2 = 'Blues'
cmap2_napari = dis.num_color_colormap(cmap2, 255)[0]
cmap2_plt = dis.num_color_colormap(cmap2, 255)[1]
sg_ecce = obj.obj_display_in_eccentricity(sg)

# int image
cmap3 = 'viridis'
cmap3_napari = dis.num_color_colormap(cmap3, 255)[0]
cmap3_plt = dis.num_color_colormap(cmap3, 255)[1]
sg_int = obj.obj_display_in_intensity(sg, pix, [6, 10])

# --------------------------
# OUTPUT DIR/NAMES
# --------------------------
"""
# get sample name
sample_name = data_path.split('"')[0].split('/')[-1]
storage_path = '%s/%s_pos%s/' % (save_path, sample_name, pos)
if not os.path.exists(storage_path):
    os.makedirs(storage_path)
"""

# --------------------------
# OUTPUT FILE
# --------------------------
"""
print("### Export file ...")
sg_pd.to_csv('%s/data_%s_pos%s.txt' % (storage_path, sample_name, pos), index=None, sep='\t')

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.imshow(pix, cmap='binary_r')
ax = plt.colorbar()
plt.savefig('%s/pix_%s_pos%s.pdf' % (storage_path, sample_name, pos))

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.imshow(sg, cmap='binary_r')
ax = plt.colorbar()
plt.savefig('%s/sg_%s_pos%s.pdf' % (storage_path, sample_name, pos))

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.imshow(sg_circ, cmap=cmap1_plt)
ax = plt.colorbar()
plt.savefig('%s/circularity_%s_pos%s.pdf' % (storage_path, sample_name, pos))

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.imshow(sg_ecce, cmap=cmap2_plt)
ax = plt.colorbar()
plt.savefig('%s/eccentricity_%s_pos%s.pdf' % (storage_path, sample_name, pos))

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.imshow(sg_int, cmap=cmap3_plt)
ax = plt.colorbar()
plt.savefig('%s/intensity_%s_pos%s.pdf' % (storage_path, sample_name, pos))
"""

# --------------------------
# OUTPUT DISPLAY
# --------------------------
print("### Output display ...")
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(pix, name='data')

    sg_properties = {
        'size': ['none'] + list(sg_pd['size']),  # background is size: none
        'circ': ['none'] + list(sg_pd['circ']),
        'eccentricity': ['none'] + list(sg_pd['eccentricity'])
    }
    viewer.add_labels(label(sg), name='SG label', properties=sg_properties, num_colors=3)

    viewer.add_image(sg_circ, name='circ', colormap=('cmap1', cmap1_napari))
    viewer.add_image(sg_ecce, name='ecce', colormap=('cmap2', cmap2_napari))
    viewer.add_image(sg_int, name='int',  colormap=('cmap3', cmap3_napari))

    """
    # color coded based on sequential sorting
    # eccentricity
    rgba_cmap = dis.num_color_colormap(cmap, len(sg_pd))[1]
    cmap_woBg = dis.sorted_num_color_colormap(rgba_cmap, sg_pd, 'eccentricity', 'sg')[0]
    if len(sg) != 0:
        viewer.add_image(label(sg), name='SG-ecce', colormap=('cmap woBg', cmap_woBg))

    # circularity
    cmap_woBg = dis.sorted_num_color_colormap(rgba_cmap, sg_pd, 'circ', 'sg')[0]
    if len(sg) != 0:
        viewer.add_image(label(sg), name='SG-circ', colormap=('cmap woBg', cmap_woBg))

    # solidity
    cmap_woBg = dis.sorted_num_color_colormap(rgba_cmap, sg_pd, 'solidity', 'sg')[0]
    if len(sg) != 0:
        viewer.add_image(label(sg), name='SG-soli', colormap=('cmap woBg', cmap_woBg))
    """
