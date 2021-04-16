import numpy as np
import time
from pycromanager import Bridge
from skimage.measure import label, regionprops

from shared.analysis import central_pixel_without_cells, bleach_location
from shared.find_blobs import select_lst

# variables
from shared.find_organelles import find_organelle
import shared.dataframe as dat
import shared.objects as obj

# SG FRAP analyzed with Photobleach-561-confocal

nr = 40
nr_between_projector_checks = 2
cal_exposure = 200
cal_offset = 5
n_curve = 300
organelle = 'sg'  # only accepts 'sg' or 'nucleoli'
photobleaching_mode = 'random'  # only accepts 'random' or 'centroid'
cell_detect_channel = "PhotoBleach-RFP-confocal"
analyze_channel = "PhotoBleach-RFP-confocal"  # also used as reference channel
analyze_save = 'Y'
n_acquire_channel = 2
acquisition_channel_lst = ["PhotoBleach-GFP-confocal", "PhotoBleach-RFP-confocal"]
prefix_channel_lst = ['GFP', 'RFP']

# build up pycromanager bridge
bridge = Bridge()
mmc = bridge.get_core()
mm = bridge.get_studio()
projector = bridge.construct_java_object("org.micromanager.projector.ProjectorAPI")
projector_device = projector.get_projection_device()


def snap_and_get_bleach_location(exposure, cutoff):
    """
    Takes an image with the current settings.  Finds a location close to the center where there are
    no objects (as defined in function central_picel_without_cells).  If no such location is found, returns -1.
    Targets bleacher to this location, exposes and takes an image of that exposure.  Finds the center of the
    actual bleach spot.  When the square of the distance between the intended target and the actual bleach spot is
    greater than provided offset, will execute a full calibration.
    :param exposure: exposure time to use for bleaching
    :param cutoff: square of distance.  When offset is higher than this code should execute a calibration
    :return: tuple with first Boolean indicating if a calibration took place, second variable the square of the offset distance
    """
    p_exposure = projector_device.get_exposure()
    c_exposure = mmc.get_exposure()

    # set analyze channel
    mmc.set_config("Channels", cell_detect_channel)

    test_img = mm.live().snap(True).get(0)
    test_np_img = np.reshape(test_img.get_raw_pixels(), newshape=[test_img.get_height(), test_img.get_width()])
    location = central_pixel_without_cells(test_np_img)
    if location:
        auto_shutter = mm.shutter().get_auto_shutter()
        mm.shutter().set_auto_shutter(False)
        projector.set_exposure(projector_device, exposure)
        mmc.set_exposure(exposure)
        projector.enable_point_and_shoot_mode(True)
        pre_img = mm.live().snap(True).get(0)
        pre_np_img = np.reshape(pre_img.get_raw_pixels(), newshape=[pre_img.get_height(), pre_img.get_width()])
        projector.add_point_to_point_and_shoot_queue(int(location[1]), int(location[0]))
        post_img = mm.live().snap(True).get(0)
        post_np_img = np.reshape(post_img.get_raw_pixels(), newshape=[post_img.get_height(), post_img.get_width()])
        measured_location = bleach_location(pre_np_img, post_np_img, location, [100, 100])
        offset = (measured_location[0] - location[0], measured_location[1] - location[1])
        print(offset)
        cal = False
        if offset[0] * offset[0] + offset[1] * offset[1] > cutoff:
            projector.calibrate(True)
            cal = True
            print("Calibrated")
        projector.set_exposure(projector_device, p_exposure)
        mmc.set_exposure(c_exposure)
        mm.shutter().set_auto_shutter(auto_shutter)
        return cal, offset[0] * offset[0] + offset[1] * offset[1]
    return False, -1


# TODO We may want to configure the acquisition settings to ensure they are what we want

pm = mm.positions()
pos_list = pm.get_position_list()
well = pos_list.get_position(0).get_label().split('-')[0]
well_count = 0
channel_count = 0
ds = mm.data().create_ram_datastore()
count = 0
acquisition_channel = acquisition_channel_lst[0]
prefix_channel = prefix_channel_lst[0]

for idx in range(pos_list.get_number_of_positions()):
    pos = pos_list.get_position(idx)

    well_temp = pos.get_label().split('-')[0]
    if well_temp == well:
        if well_count >= n_curve:
            if channel_count == n_acquire_channel-1:
                continue
            else:
                well_count = 0
                channel_count += 1
                acquisition_channel = acquisition_channel_lst[channel_count]
                prefix_channel = prefix_channel_lst[channel_count]
    else:
        well_count = 0
        well = well_temp
        channel_count = 0
        acquisition_channel = acquisition_channel_lst[0]
        prefix_channel = prefix_channel_lst[0]

    # Close DataViewer opened during previous run
    dv = mm.displays().close_displays_for(ds)
    pos.go_to_position(pos, mmc)

    time.sleep(0.1)
    if count >= nr_between_projector_checks:
        calibrated, error = snap_and_get_bleach_location(cal_exposure, cal_offset)
        if error < 0:
            count -= 1
        else:
            count = 0
        if calibrated:
            continue
    count += 1

    # set analyze channel
    mmc.set_config("Channels", analyze_channel)

    img = mm.live().snap(False).get(0)
    pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
    # find organelles using a combination of thresholding and watershed
    if organelle == 'sg':
        _, segmented = find_organelle(pixels, 'na', 500, 200, 5, 200)  # stress granule
    else:
        _, segmented = find_organelle(pixels, 'local-nucleoli', 500, 200, 10, 1000)  # nucleoli

    label_img = label(segmented)
    if photobleaching_mode == 'centroid':
        blobs = regionprops(label_img)
        centroid_x = [round(p.centroid[1]) for p in blobs]
        centroid_y = [round(p.centroid[0]) for p in blobs]
        selected = select_lst(centroid_x, centroid_y, img.get_width() / 10, 0.9 * img.get_width())
    elif photobleaching_mode == 'random':
        spots = obj.select_random_in_label(label_img, 1)
        selected = select_lst(spots[0], spots[1], img.get_width() / 10, 0.9 * img.get_width())
    else:
        raise TypeError("photobleaching_mode only accepts 'centroid' or 'random', get %s instead"
                        % photobleaching_mode)

    if len(selected[0]) > (nr // 2):
        projector.enable_point_and_shoot_mode(True)

        # acquisition of the single mScarlet image
        # https://valelab4.ucsf.edu/~MM/doc-2.0.0-gamma/mmstudio/org/micromanager/acquisition/SequenceSettings.Builder.
        # html#timeFirst-boolean-
        if analyze_save == 'Y':
            mmc.set_config("Channels", analyze_channel)
            ssb1 = mm.acquisitions().get_acquisition_settings().copy_builder()
            ssb1.num_frames(1)
            ssb1.prefix('%s-ref' % pos.get_label())
            mm.acquisitions().set_acquisition_settings(ssb1.build())
            ds1 = mm.acquisitions().run_acquisition()
            dv1 = mm.displays().close_displays_for(ds1)

        # acquire FRAP movies
        ssb = mm.acquisitions().get_acquisition_settings().copy_builder()
        mm.acquisitions().set_acquisition_settings(ssb.prefix(pos.get_label()).build())
        ds = mm.acquisitions().run_acquisition_nonblocking()
        # Trick to get timing right.  Wait for Core to report that Sequence is running
        while not mmc.is_sequence_running(mmc.get_camera_device()):
            time.sleep(0.1)
        time.sleep(1.5)

        nr_shots = nr if len(selected[0]) >= (2 * nr) else int(len(selected[0]) / 2)
        well_count += nr_shots
        shots = dat.select_multiple(selected[0], selected[1], nr_shots)

        for i in range(len(shots[0])):
            # Note that MM has x-y coordinates, and Python uses row-column (equivalent to y-x)
            projector.add_point_to_point_and_shoot_queue(shots[0][i], shots[1][i])
            time.sleep(0.07)
        print(pos.get_label(), ": Shots ", len(shots[0]))
        while mmc.is_sequence_running(mmc.get_camera_device()):
            time.sleep(0.5)
        time.sleep(1)

print("Done!")
