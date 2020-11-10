import numpy as np
import random
import find_blobs

from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import morphology
from pycromanager import Bridge

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()

# get first image from active dataviewer
dv = mm.displays().get_active_data_viewer()
ds = dv.get_datastore()
cb = mm.data().get_coords_builder()
coord = cb.c(0).t(0).p(0).z(0).build()
img = ds.get_image(coord)

pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

# find organelles using a combination of thresholding and watershed
segmented = find_blobs.find_blobs(pixels, threshold_otsu(pixels), 500, 200)
label_img = label(segmented)
blobs = morphology.remove_small_objects(label_img, 5)

# select two sizes, small and large.  Select those that are away from the border
centered = find_blobs.select(blobs, 'centroid', img.get_width() / 10, 0.9 * img.get_width())
small = find_blobs.select(centered, 'area', 5, 45)
large = find_blobs.select(centered, 'area', 90, 600)

# for each, select up to 10, but no more than half of the spots, and send them to SLM
nr = 10
for region_list in [small, large]:
    nr_shots = nr if len(region_list) >= 2 * nr else len(region_list) / 2
    shots = random.sample(region_list, nr_shots)
    for shot in shots:
        projector.display_spot(projector_device, shot['centroid'][0], shot['centroid'][1])
    print("Shots ", len(shots))





