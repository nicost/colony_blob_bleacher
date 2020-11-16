import numpy as np
import random
import time
import find_blobs

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import morphology
from pycromanager import Bridge

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()

# Note: to use napari as a viewer, start the QT event loop using the IPython magic command:
# %gui qt5
# then, start napari:
# import naparai
# viewer = napari.Viewer()
# viewer.add_image(pixels)

# get first image from active dataviewer
#pos_list = mm.get_position_list()

#for i in np.arange(pos_list.get_number_of_positions()):
#    multi-stage_position = pos_list.get_position(i.item())

#mm.live().snap(True)
dv = mm.displays().get_active_data_viewer()
ds = dv.get_datastore()
cb = mm.data().get_coords_builder()
coord = cb.c(0).t(0).p(0).z(0).build()
img = ds.get_image(coord)

pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

# find organelles using a combination of thresholding and watershed
segmented = find_blobs.find_blobs(pixels, threshold_otsu(pixels), 500, 200)
label_img = label(segmented)
label_img = morphology.remove_small_objects(label_img, 5)
blobs = regionprops(label_img)

# select two sizes, small and large.  Select those that are away from the border
centered = find_blobs.select(blobs, 'centroid', img.get_width() / 10, 0.9 * img.get_width())
small = find_blobs.select(centered, 'area', 5, 45)
large = find_blobs.select(centered, 'area', 90, 600)

# for each, select up to 10, but no more than half of the spots, and send them to SLM
nr = 20
projector.enable_point_and_shoot_mode(True)
mm.acquisitions().run_acquisition_nonblocking()
# Trick to get timing right.  Wait for Core to report that Sequence is running
while not mmc.is_sequence_running(mmc.get_camera_device()):
    time.sleep(0.1)
time.sleep(0.5)
for region_list in [small, large]:
    nr_shots = nr if len(region_list) >= (2 * nr) else int(len(region_list) / 2)
    shots = random.sample(region_list, nr_shots)
    # shots = region_list[0:10]
    for shot in shots:
        # Note that MM has x-y coordinates, and Python uses row-column (equivalent to y-x)
        projector.add_point_to_point_and_shoot_queue(shot['centroid'][1], shot['centroid'][0])
        # print(shot['centroid'][1], " ", shot['centroid'][0])
        time.sleep(0.07)
    print("Shots ", len(shots))
    time.sleep(1)






