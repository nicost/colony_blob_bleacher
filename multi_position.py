
import numpy as np
import random
import time
from pycromanager import Bridge
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import morphology
import find_blobs

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()

#TODO We may want to configure the acquisition settings to ensure they are what we want

# Note currently, pm and mm are the same object, and both have the positionlistmanager methods
# Be ready for when this gets fixed
pm = mm.positions()
pos_list = pm.get_position_list()

for idx in range(pos_list.get_number_of_positions()):
    pos = pos_list.get_position(idx)
    pos.go_to_position(pos, mmc)
    time.sleep(0.1)
    img = mm.live().snap(False).get(0)
    pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
    # find organelles using a combination of thresholding and watershed
    segmented = find_blobs.find_blobs(pixels, threshold_otsu(pixels), 500, 200)
    label_img = label(segmented)
    label_img = morphology.remove_small_objects(label_img, 5)
    blobs = regionprops(label_img)
    centered = find_blobs.select(blobs, 'centroid', img.get_width() / 10, 0.9 * img.get_width())
    
    nr = 20
    projector.enable_point_and_shoot_mode(True)
    mm.acquisitions().run_acquisition_nonblocking()
    # Trick to get timing right.  Wait for Core to report that Sequence is running
    while not mmc.is_sequence_running(mmc.get_camera_device()):
        time.sleep(0.1)
    time.sleep(0.5)
    for region_list in [centered]:
        nr_shots = nr if len(region_list) >= (2 * nr) else int(len(region_list) / 2)
        shots = random.sample(region_list, nr_shots)
        # shots = region_list[0:10]
        for shot in shots:
            # Note that MM has x-y coordinates, and Python uses row-column (equivalent to y-x)
            projector.add_point_to_point_and_shoot_queue(shot['centroid'][1], shot['centroid'][0])
            # print(shot['centroid'][1], " ", shot['centroid'][0])
            time.sleep(0.07)
        print("Shots ", len(shots))
        while mmc.is_sequence_running(mmc.get_camera_device()):
            time.sleep(0.5)
        time.sleep(1)

