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

# --------------------------
# PARAMETERS
# --------------------------
# data source
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/SG_scoring/WT"

# values
thresholding = 'local-sg'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
min_size = 5  # minimum SG size
max_size = 350  # maximum SG size

# --------------------------
# LOAD MOVIE
# --------------------------
# build up pycromanager bridge
# first start up Micro-Manager (needs to be compatible version)
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()

# load time series data
store = mm.data().load_data(data_path, True)
max_p = store.get_max_indices().get_p() #24
num_grid = int(np.sqrt(max_p+1))
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

# ------------------------------
# IMAGE ANALYSIS
# ------------------------------
sg_fov_pd = pd.DataFrame()
row_lst = []
col_lst = []
pix_lst = []
sg_lst = []
for i in range(max_p+1):
    temp = store.get_image(cb.t(0).p(i).build())
    temp_pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])
    temp_sg = find_organelle(temp_pix, thresholding, min_size=min_size, max_size=max_size)
    row, col = dat.get_grid_pos(i, num_grid)
    row_lst.append(row)
    col_lst.append(col)
    pix_lst.append(temp_pix)
    sg_lst.append(temp_sg)

    #temp_sg_pd = sg_analysis(temp_pix, temp_sg, i)
    #sg_fov_pd = pd.concat([sg_fov_pd, temp_sg_pd], ignore_index=True)

pix_pd = pd.DataFrame({'row': row_lst, 'col': col_lst, 'pix': pix_lst})
pix = ana.pix_stitch(pix_pd, num_grid)
sg_pix_pd = pd.DataFrame({'row': row_lst, 'col': col_lst, 'pix': sg_lst})
sg = ana.pix_stitch(sg_pix_pd, num_grid)
sg_pd = sg_analysis(pix, sg, 0)

num_interval = 5
range_lst = np.arange(0, 1 + 1 / num_interval, 1 / num_interval)
#sg_circ = obj.group_label_circularity(sg, range_lst)
bias_range_lst = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] #11
sg_ecce = obj.group_label_eccentricity(sg, bias_range_lst)

# --------------------------
# OUTPUT FILE
# --------------------------
#sg_fov_pd.to_csv('%s/data.txt' % data_path, index=None, sep='\t')

"""
fig, ax = plt.subplots(figsize=(8*num_grid, 8*num_grid))
ax = plt.imshow(pix, cmap='binary_r')
ax = plt.colorbar()
plt.savefig('%s/full_img.pdf' % data_path)
"""

# --------------------------
# OUTPUT DISPLAY
# --------------------------
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(pix, name='data')

    sg_properties = {
        'size': ['none'] + list(sg_pd['size']),  # background is size: none
        'circ': ['none'] + list(sg_pd['circ']),
        'eccentricity': ['none'] + list(sg_pd['eccentricity'])
    }
    viewer.add_labels(label(sg), name='SG label', properties=sg_properties, num_colors=3)

    cmap1 = 'YlOrRd'
    cmap2 = 'hot'
    cmap1_woBg = dis.num_color_colormap(cmap1, num_interval + 2)[0]
    cmap2_woBg = dis.num_color_colormap(cmap2, num_interval + 2)[0]
    #viewer.add_image(sg_circ, name='circ', colormap=('cmap1', cmap1_woBg))
    viewer.add_image(sg_ecce, name='ecce', colormap=('cmap2', cmap2_woBg))