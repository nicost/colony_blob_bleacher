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

p_exposure = projector_device.get_exposure()

img = mm.live().snap(False).get(0)
pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

projector.enable_point_and_shoot_mode(True)
projector_device.set_exposure(400)
projector.add_point_to_point_and_shoot_queue(shot['centroid'][1], shot['centroid'][0])
img = mm.live().snap(False).get(0)
pixels2 = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

