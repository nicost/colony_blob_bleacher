import numpy as np

from pycromanager import Bridge

from shared.analysis import bleach_location, central_pixel_without_cells

print("Hello")

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()
p_exposure = projector_device.get_exposure()
pre_img = mm.live().snap(False)
pre_np_img = np.reshape(pre_img.get_raw_pixels(), newshape=[pre_img.get_height(), pre_img.get_width()])
location = central_pixel_without_cells(pre_np_img)
if location:
    projector.add_point_to_point_and_shoot_queue(location[1], location[0])
    post_img = mm.live().snap(True)
    post_np_img = np.reshape(post_img.get_raw_pixels(), newshape=[post_img.get_height(), post_img.get_width()])
    measured_location = bleach_location(pre_np_img, post_np_img, location, [100, 100])
    offset = location - measured_location
    print(offset)
else:
    print("No location found to project bleach spot")
