import numpy as np

from pycromanager import Bridge

from shared.analysis import bleach_location

print("Hello")

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()
p_exposure = projector_device.get_exposure()

# note this is the position in image coordinated.  Need to invert y to go to numpy coordinates (below)
expected_position = (256, 256)
half_roi_size = (100, 100)

pre = "C://Users//NicoLocal//Images//Jess//20201217//Pre"
post = "C://Users//NicoLocal//Images//Jess//20201217//Post"

pre_data = mm.data().load_data(pre, False)
post_data = mm.data().load_data(post, False)

cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)

pre_img = pre_data.get_image(cb.build())
post_img = post_data.get_image(cb.build())

pre_pixels = np.reshape(pre_img.get_raw_pixels(), newshape=[pre_img.get_height(), pre_img.get_width()])
post_pixels = np.reshape(post_img.get_raw_pixels(), newshape=[post_img.get_height(), post_img.get_width()])
bleach_location = bleach_location(pre_pixels, post_pixels, expected_position, half_roi_size)

print(bleach_location)
offset = [bleach_location[0] - expected_position[0], bleach_location[1] - expected_position[1] ]
print(offset)
