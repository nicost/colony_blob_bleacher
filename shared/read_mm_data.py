import numpy as np

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import morphology

from pycromanager import Bridge
from shared.find_blobs import find_blobs
"""
    Example code showing how to read Micro-Manager data using the pycro-manager bridge.
    ALso reads the date/time stamp from the image metadata and runs the analysis
    code locating blobs in the images
"""
# build up pycromanager bridge
# first start up Micro-Manager (needs to be compatible version)
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()

data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/TestedData/20201116/AutoBleach_15"
#"C:\\Users\\NicoLocal\\Images\\Jess\\20201116-Nucleoili-bleaching-4x\\PythonAcq1\\AutoBleach_15"

store = mm.data().load_data(data_path, True)
max_t = store.get_max_indices().get_t()
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

for t in range(0, max_t):
    img = store.get_image(cb.t(t).build())
    pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

    # find organelles using a combination of thresholding and watershed
    segmented = find_blobs(pixels, threshold_otsu(pixels), 500, 200)
    label_img = label(segmented)
    label_img = morphology.remove_small_objects(label_img, 5)
    blobs = regionprops(label_img)

    print(img.get_metadata().get_received_time(), " ", len(blobs))




