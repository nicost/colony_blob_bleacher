import numpy as np
from pycromanager import Bridge
from shared.find_organelles import find_organelle
from skimage.measure import label, regionprops_table, regionprops
import pandas as pd
import shared.analysis as ana
import os

# --------------------------
# PARAMETERS ALLOW CHANGE
# --------------------------
# paths
data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/"\
            "20210323_SG_top10MonoclonalCells/95-5_5_7_8_1-2"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/20210323_SG_top10MonoclonalCells/" \
            "dataAnalysis/95-5_5_7_8_1-2"

# values for analysis
data_c_G3BP1 = 1  # channel for G3BP1-mScarlet  1:G3BP1-mScarlet channel
data_c_sample = 0  # channel for sample  0:GFP channel
thresholding = 'na'  # only accepts 'na', 'otsu', 'yen', 'local-nucleoli' and 'local-sg'
frame_offset = 0
min_size = 5
max_size = 200

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
max_t = store.get_max_indices().get_t()
max_p = store.get_max_indices().get_p()
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

# ------------------------------
# IMAGE ANALYSIS based on position
# ------------------------------
print("### Image analysis: calculate SG mask/pd ...")
# test image of position
pos_lst = []
well_lst = []
frame_lst = []
label_lst = []
size_lst = []
raw_int_G3BP1_lst = []
bg_G3BP1_lst = []
fov_G3BP1_lst = []
raw_int_sample_lst = []
bg_sample_lst = []
fov_sample_lst = []

for pos in range(max_p+1):
# for pos in np.arange(1000, 1550, 1):
    pos = int(pos)
    print('Analyzing pos: %s/%s' % (pos+1, max_p + 1))
    for frame in range(max_t+1):
        # get G3BP1 image
        temp = store.get_image(cb.t(frame).c(data_c_G3BP1).z(0).p(pos).build())
        pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])
        well = temp.get_metadata().get_position_name().split('-')[0]
        # get GFP channel image
        temp1 = store.get_image(cb.t(frame).c(data_c_sample).z(0).p(pos).build())
        pix1 = np.reshape(temp1.get_raw_pixels(), newshape=[temp1.get_height(), temp1.get_width()])
        # identify stress granule
        _, sg = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
        label_sg = label(sg, connectivity=1)
        # total image
        fov = np.ones_like(pix)
        # measure stress granule properties in both channel
        sg_props = regionprops_table(label_sg, pix, properties=('label', 'area', 'mean_intensity'))
        sg_props1 = regionprops_table(label_sg, pix1, properties=('label', 'area', 'mean_intensity'))
        sg_pd = pd.DataFrame(sg_props)
        sg_pd1 = pd.DataFrame(sg_props1)
        # measure total intensity
        fov_G3BP1 = regionprops(fov, pix)[0].mean_intensity
        fov_sample = regionprops(fov, pix1)[0].mean_intensity
        # measure background intensity
        bg_G3BP1 = ana.get_bg_int([pix])[0]
        bg_sample = ana.get_bg_int([pix1])[0]
        # add information into list
        pos_lst = pos_lst + [pos]*len(sg_pd)
        well_lst = well_lst + [well]*len(sg_pd)
        frame_lst = frame_lst + [frame+frame_offset]*len(sg_pd)
        label_lst = label_lst + sg_pd['label'].tolist()
        size_lst = size_lst + sg_pd['area'].tolist()
        raw_int_G3BP1_lst = raw_int_G3BP1_lst + sg_pd['mean_intensity'].tolist()
        bg_G3BP1_lst = bg_G3BP1_lst + [bg_G3BP1]*len(sg_pd)
        fov_G3BP1_lst = fov_G3BP1_lst + [fov_G3BP1]*len(sg_pd)
        raw_int_sample_lst = raw_int_sample_lst + sg_pd1['mean_intensity'].tolist()
        bg_sample_lst = bg_sample_lst + [bg_sample]*len(sg_pd)
        fov_sample_lst = fov_sample_lst + [fov_sample]*len(sg_pd)

ana_pd = pd.DataFrame({'pos': pos_lst, 'well': well_lst, 'frame': frame_lst, 'label': label_lst,
                       'size': size_lst, 'raw_int_G3BP1': raw_int_G3BP1_lst, 'bg_G3BP1': bg_G3BP1_lst,
                       'fov_G3BP1': fov_G3BP1_lst, 'raw_int_GFP': raw_int_sample_lst, 'bg_GFP': bg_sample_lst,
                       'fov_sample': fov_sample_lst})
ana_pd['int_G3BP1'] = ana_pd['raw_int_G3BP1'] - ana_pd['bg_G3BP1']
ana_pd['int_GFP'] = ana_pd['raw_int_GFP'] - ana_pd['bg_GFP']


print("### Export data ...")
storage_path = save_path
if not os.path.exists(storage_path):
    os.makedirs(storage_path)
ana_pd.to_csv('%s/data.txt' % storage_path, index=False, sep='\t')

print('DONE!')


