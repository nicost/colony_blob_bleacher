import numpy as np

from pycromanager import Bridge

from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.filters import rank


print("Hello")

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()
p_exposure = projector_device.get_exposure()

# note this is the position in image coordinated.  Need to invert y to go to numpy coordinates (below)
expected_position = [256, 256]
half_roi_size = [100, 100]

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
# assume pre and post are the same size
cc = post_pixels.shape[1] - expected_position[1]
ep_rc = [expected_position[0], cc]
pre_roi = pre_pixels[ep_rc[0] - half_roi_size[0]:ep_rc[0] + half_roi_size[0], ep_rc[1] - half_roi_size[1]:ep_rc[1] + half_roi_size[1]]
post_roi = post_pixels[ep_rc[0] - half_roi_size[0]:ep_rc[0] + half_roi_size[0], ep_rc[1] - half_roi_size[1]:ep_rc[1] + half_roi_size[1]]

subtracted = post_roi + 100 - pre_roi

selem = disk(2)
subtracted_mean = rank.mean(subtracted, selem=selem)
peaks_rc_roi = peak_local_max(subtracted_mean, min_distance=20, threshold_rel=0.6, num_peaks=1, indices=True)
peaks_rc = peaks_rc_roi + [ep_rc[0] - half_roi_size[0], ep_rc[1] - half_roi_size[1]]
peaks = [peaks_rc[0][0], post_pixels.shape[1] - peaks_rc[0][1]]

print(peaks_rc_roi)
print(peaks_rc)
print(peaks)










# img = mm.live().snap(False).get(0)
#pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

# projector.enable_point_and_shoot_mode(True)
# projector_device.set_exposure(400)
# projector.add_point_to_point_and_shoot_queue(shot['centroid'][1], shot['centroid'][0])
# img = mm.live().snap(False).get(0)
# pixels2 = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])

