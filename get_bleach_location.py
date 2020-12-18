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
mapping = projector.load_mapping(projector_device)
point_java = bridge.construct_java_object("java.awt.geom.Point2D.Double")

p_exposure = projector_device.get_exposure()
img = mm.live().snap(False).get(0)
pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

point_java = bridge.construct_java_object("java.awt.geom.Point2D.Double")
point = projector.transformPoint(mapping, point_java, null, 1)
projector_device.set_exposure(400)
projector_device.display_spot(point_java.x, point_java.y)