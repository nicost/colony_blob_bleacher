from tifffile import TiffFile
from pycromanager import Bridge

data_path = '/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/test/' \
            'D2-Site_3_1/D2-Site_3_1_MMStack_Default.ome.tif'
data_path1 = '/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/test/data/D2/D2-Site_1_1'

#with TiffFile(data_path) as tif:
#    img_metadata = tif.imagej_metadata

#print(img_metadata)

# build up pycromanager bridge
bridge = Bridge()
mm = bridge.get_studio()

ds = mm.displays().get_active_data_viewer().get_data_provider()
ds = mm.data().load_data(data_path, False)

print(ds)
