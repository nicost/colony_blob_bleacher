from tifffile import TiffFile
from pycromanager import Bridge
import numpy as np

data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/20210224_SG_Top10_ArsTreatments/"\
            "1235678910_1"

# with TiffFile(data_path) as tif:
#    img_metadata = tif.imagej_metadata

# print(img_metadata)

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()

# ds = mm.displays().get_active_data_viewer().get_data_provider()
ds = mm.data().load_data(data_path, True)
sm = ds.get_summary_metadata()
pos_list = sm.get_stage_position_list()
p_size = ds.get_any_image().get_metadata().get_pixel_size_um()


"""store = mm.data().load_data(data_path, True)
max_t = store.get_max_indices().get_t()
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

temp = store.get_image(cb.t(0).c(1).z(0).p(0).build())
pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])
p_size = temp.get_height()"""

print(p_size)
