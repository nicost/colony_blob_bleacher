from tifffile import TiffFile

data_path = '/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/test/' \
            'D2-Site_3_1/D2-Site_3_1_MMStack_Default.ome.tif'

with TiffFile(data_path) as tif:
    img_metadata = tif.imagej_metadata

print(img_metadata)
