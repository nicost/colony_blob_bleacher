import numpy as np

from pycromanager import Bridge

from shared.analysis import central_pixel_without_cells

print("Hello")

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()

dv = mm.displays().get_active_data_viewer()
ds = dv.get_data_provider()
cb = mm.data().get_coords_builder()
cb.t(0).p(0).c(0).z(0)
img = ds.get_image(cb.build())
np_img = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

print(central_pixel_without_cells(np_img))
