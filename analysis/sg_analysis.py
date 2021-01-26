import numpy as np
import napari
from pycromanager import Bridge

from shared.find_organelles import sg_analysis, find_organelle
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
max_t = store.get_max_indices().get_t()
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

# ------------------------------
# IMAGE ANALYSIS based on position 0
# ------------------------------
# test image of position 0
p0 = store.get_image(cb.t(0).p(2).build())
p0_pix = np.reshape(p0.get_raw_pixels(), newshape=[p0.get_height(), p0.get_width()])

sg = find_organelle(p0_pix, thresholding, min_size=min_size, max_size=max_size)
sg_pd = sg_analysis(p0_pix, sg, 0)

num_interval = 10
range_lst = np.arange(0, 1 + 1 / num_interval, 1 / num_interval)
sg_circ = obj.group_label_circularity(sg, range_lst)
bias_range_lst = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] #11
sg_ecce = obj.group_label_eccentricity(sg, bias_range_lst)

# --------------------------
# OUTPUT FILE
# --------------------------
#sg_pd.to_csv('%s/data_full.txt' % data_path, index=None, sep='\t')

# --------------------------
# OUTPUT DISPLAY
# --------------------------
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(p0_pix, name='data')

    sg_properties = {
        'size': ['none'] + list(sg_pd['size']),  # background is size: none
        'circ': ['none'] + list(sg_pd['circ']),
        'eccentricity': ['none'] + list(sg_pd['eccentricity'])
    }
    viewer.add_labels(label(sg), name='SG label', properties=sg_properties, num_colors=3)

    cmap1 = 'YlOrRd'
    cmap2 = 'hot'
    cmap1_woBg = dis.num_color_colormap(cmap1, num_interval+2)[0]
    cmap2_woBg = dis.num_color_colormap(cmap2, num_interval+2)[0]
    viewer.add_image(sg_circ, name='circ', colormap=('cmap1', cmap1_woBg))
    viewer.add_image(sg_ecce, name='ecce', colormap=('cmap2', cmap2_woBg))

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
