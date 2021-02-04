import numpy as np
import pandas as pd
from pycromanager import Bridge
from shared.find_organelles import find_organelle, organelle_analysis, find_nuclear, nuclear_analysis
from skimage.measure import label
import shared.analysis as ana
import shared.dataframe as dat
import shared.display as dis
import shared.objects as obj
import shared.bleach_points as ble
import shared.math_functions as mat
import os

# paths
data_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/" \
            "20201216_CBB_nucleoliBleachingTest_drugTreatment/Ctrl-2DG-CCCP-36pos_partial/WT/"

# values for analysis
data_c = 0
pos = 0
thresholding = 'local-nucleoli'
# global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
min_size = 10  # minimum nucleoli size; default = 10
max_size = 1000  # maximum nucleoli size; default = 1000;
# larger ones are generally cells without nucleoli
num_dilation = 3  # number of dilation from the coordinate;
# determines analysis size of the analysis spots; default = 3

# modes
mode_bleach_detection = 'single-offset'  # only accepts 'single-raw' or 'single-offset'

"""
# ---------------------------------------------------------------------------------------------------
# PLEASE DO NOT CHANGE AFTER THIS
# ---------------------------------------------------------------------------------------------------
"""

dirs = [x[0] for x in os.walk(data_source)]
dirs.pop(0)
num_dir = len(dirs)

for s in range(len(dirs)):
    print("### DATA PROCESSING: %d / %d" % (s+1, num_dir))
    data_path = dirs[s]
    save_path = dirs[s]

    # --------------------------
    # LOAD MOVIE
    # --------------------------
    print("### Load movie ...")
    data_log = pd.DataFrame({'pos': [s]})

    # build up pycromanager bridge
    # first start up Micro-Manager (needs to be compatible version)
    bridge = Bridge()
    mmc = bridge.get_core()
    mm = bridge.get_studio()
    # load time series data
    store = mm.data().load_data(data_path, True)
    cb = mm.data().get_coords_builder()
    cb.t(0).p(0).c(0).z(0)
    # get max_t and acquisition time
    max_t = store.get_max_indices().get_t()
    pixels_tseries = dat.get_pixels_tseries(store, cb, data_c)
    acquire_time_tseries, real_time = dat.get_time_tseries(store, cb)

    # --------------------------------------
    # ORGANELLE ANALYSIS based on time 0
    # --------------------------------------
    print("### Image analysis: nucleoli detection based on time 0 ...")

    # reference image of time 0
    # if decide to use other image as ref_image
    # be sure to check photobleaching correction for all reported intensities
    temp = store.get_image(cb.c(data_c).t(0).build())
    pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

    # nuclear detection
    label_nuclear = find_nuclear(pix)
    data_log['num_nuclei_detected'] = [np.amax(label_nuclear)]
    print("Found %d nuclei." % data_log['num_nuclei_detected'][0])

    # nucleoli detection
    nucleoli = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
    label_nucleoli = label(nucleoli, connectivity=1)
    data_log['num_nucleoli_detected'] = [obj.object_count(nucleoli)]
    print("Found %d nucleoli." % data_log['num_nucleoli_detected'][0])

    # nucleoli pd dataset
    nucleoli_pd = organelle_analysis(pix, nucleoli, 'nucleoli', s)
    # link nucleoli with corresponding nuclear
    round_x = [round(num) for num in nucleoli_pd['x']]
    round_y = [round(num) for num in nucleoli_pd['y']]
    nucleoli_pd['nuclear'] = obj.points_in_objects(label_nuclear, round_y, round_x)

    data_log['num_nucleoli_in_nuclei'] = [len(nucleoli_pd[nucleoli_pd['nuclear'] != 0])]
    print("Found %d out of %d nucleoli within nuclei." % (data_log['num_nucleoli_in_nuclei'][0],
                                                          obj.object_count(nucleoli)))

    # nuclear pd dataset
    nuclear_pd = nuclear_analysis(label_nuclear, nucleoli_pd, s)

    # ----------------------------------
    # BLEACH SPOTS DETECTION
    # ----------------------------------
    print("### Image analysis: bleach spots detection ...")

    # load point_and_shoot log file
    log_pd = pd.read_csv('%s/PointAndShoot.log' % data_path, na_values=['.'], sep='\t', header=None)
    data_log['num_aim_spots'] = [len(log_pd)]
    print("Aim to photobleach %d spots." % data_log['num_aim_spots'][0])
    log_pd.columns = ['time', 'aim_x', 'aim_y']  # reformat log_pd

    # get bleach_frame
    log_pd['bleach_frame'] = dat.get_frame(log_pd['time'], acquire_time_tseries)

    # get bleach spot coordinate
    coordinate_pd = ble.get_bleach_spots_coordinates(log_pd, store, cb, data_c, mode_bleach_detection)
    log_pd = pd.concat([log_pd, coordinate_pd], axis=1)

    # link pointer with corresponding nucleoli
    log_pd['nucleoli'] = obj.points_in_objects(label_nucleoli, log_pd['x'], log_pd['y'])

    # generate bleach spot mask and bleach spots dataframe (pointer_pd)
    bleach_spots, pointer_pd = ble.get_bleach_spots(log_pd, label_nucleoli, num_dilation)
    data_log['num_bleach_spots'] = [obj.object_count(bleach_spots)]
    print("%d spots passed filters for analysis." % data_log['num_bleach_spots'][0])

    # add bleach spots corresponding nucleoli measurements
    pointer_pd = dat.copy_based_on_index(pointer_pd, nucleoli_pd, 'nucleoli', 'nucleoli',
                                         ['nucleoli_x', 'nucleoli_y', 'nucleoli_size',
                                          'nucleoli_mean_int', 'nucleoli_circ'],
                                         ['x', 'y', 'size', 'raw_int', 'circ'])

    # --------------------------------------------------
    # FRAP CURVE ANALYSIS from bleach spots
    # --------------------------------------------------
    print("### Image analysis: FRAP curve calculation ...")

    # create control spots mask
    ctrl_nucleoli = ~nucleoli_pd.index.isin(log_pd['nucleoli'].tolist())
    ctrl_x = nucleoli_pd[ctrl_nucleoli]['x'].astype(int).tolist()
    ctrl_y = nucleoli_pd[ctrl_nucleoli]['y'].astype(int).tolist()
    ctrl_spots = ana.analysis_mask(ctrl_x, ctrl_y, pix, num_dilation)
    num_ctrl_spots = obj.object_count(ctrl_spots)
    pointer_pd['num_ctrl_spots'] = [num_ctrl_spots] * len(pointer_pd)

    # get raw intensities for bleach spots and control spots
    pointer_pd['raw_int'] = ana.get_intensity(bleach_spots, pixels_tseries)
    ctrl_spots_int_tseries = ana.get_intensity(ctrl_spots, pixels_tseries)
    ctrl_pd = pd.DataFrame({'pos': [s] * num_ctrl_spots, 'ctrl_spots': np.arange(0, num_ctrl_spots, 1),
                            'raw_int': ctrl_spots_int_tseries})

    print("### Image analysis: background correction ...")
    # background intensity measurement
    bg_int_tseries = ana.get_bg_int(pixels_tseries)
    pointer_pd['bg_int'] = [bg_int_tseries] * len(pointer_pd)

    # background intensity fitting
    bg_fit = mat.fitting_linear(np.arange(0, len(bg_int_tseries), 1), bg_int_tseries)
    pointer_pd = dat.add_columns(pointer_pd, ['bg_linear_fit', 'bg_linear_r2', 'bg_linear_a', 'bg_linear_b'],
                                 [[bg_fit[0]] * len(pointer_pd), [bg_fit[1]] * len(pointer_pd),
                                  [bg_fit[2]] * len(pointer_pd), [bg_fit[3]] * len(pointer_pd)])

    # background correction
    # use original measurement if fitting does not exist
    if np.isnan(bg_fit[2]):
        bg = bg_int_tseries
    else:
        bg = bg_fit[0]
    pointer_pd['bg_cor_int'] = ana.bg_correction(pointer_pd['raw_int'], bg)
    ctrl_pd['bg_cor_int'] = ana.bg_correction(ctrl_pd['raw_int'], bg)

    print("### Image analysis: photobleaching correction ...")
    # photobleaching factor calculation
    if num_ctrl_spots != 0:
        # calculate photobleaching factor
        pb_factor = ana.get_pb_factor(ctrl_pd['bg_cor_int'])
        pointer_pd['pb_factor'] = [pb_factor] * len(pointer_pd)
        print("%d ctrl points are used to correct photobleaching." % len(ctrl_pd))

        # pb_factor fitting with single exponential decay
        pb_fit = mat.fitting_single_exp_decay(np.arange(0, len(pb_factor), 1), pb_factor)
        pointer_pd = dat.add_columns(pointer_pd, ['pb_single_exp_decay_fit', 'pb_single_exp_decay_r2',
                                                  'pb_single_exp_decay_a', 'pb_single_exp_decay_b',
                                                  'pb_single_exp_decay_c'],
                                     [[pb_fit[0]] * len(pointer_pd), [pb_fit[1]] * len(pointer_pd),
                                      [pb_fit[2]] * len(pointer_pd), [pb_fit[3]] * len(pointer_pd),
                                      [pb_fit[4]] * len(pointer_pd)])

        # photobleaching correction
        if np.isnan(pb_fit[2]):
            pb = pb_factor
        else:
            pb = pb_fit[0]
        pointer_pd['mean_int'] = ana.pb_correction(pointer_pd['bg_cor_int'], pb)

    # normalize frap curve and measure mobile fraction and t-half based on curve itself
    frap_pd = ble.frap_analysis(pointer_pd, max_t, acquire_time_tseries, real_time)
    pointer_pd = pd.concat([pointer_pd, frap_pd], axis=1)

    # --------------------------------------------------
    # FRAP CURVE FITTING
    # --------------------------------------------------
    print("### Imaging analysis: curve fitting ...")

    # curve fitting with linear to determine initial slope
    linear_fit_pd = mat.frap_fitting_linear(pointer_pd['real_time_post'], pointer_pd['int_curve_post_nor'])
    pointer_pd = pd.concat([pointer_pd, linear_fit_pd], axis=1)

    # curve fitting with single exponential function
    single_exp_fit_pd = mat.frap_fitting_single_exp(pointer_pd['real_time_post'],
                                                    pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, single_exp_fit_pd], axis=1)

    # curve fitting with soumpasis function
    soumpasis_fit_pd = mat.frap_fitting_soumpasis(pointer_pd['real_time_post'],
                                                  pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, soumpasis_fit_pd], axis=1)

    # curve fitting with double exponential function
    double_exp_fit_pd = mat.frap_fitting_double_exp(pointer_pd['real_time_post'],
                                                    pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, double_exp_fit_pd], axis=1)

    # curve fitting with ellenberg function
    ellenberg_fit_pd = mat.frap_fitting_ellenberg(pointer_pd['real_time_post'],
                                                  pointer_pd['int_curve_post_nor'], pointer_pd['sigma'])
    pointer_pd = pd.concat([pointer_pd, ellenberg_fit_pd], axis=1)

    # find optimal fitting
    optimal_fit_pd = mat.find_optimal_fitting(pointer_pd, ['single_exp', 'soumpasis', 'ellenberg', 'double_exp'])
    pointer_pd = pd.concat([pointer_pd, optimal_fit_pd], axis=1)

    # filter frap curves
    pointer_pd['frap_filter_single_exp'] = ble.frap_filter(pointer_pd, 'single_exp')
    pointer_pd['frap_filter_soumpasis'] = ble.frap_filter(pointer_pd, 'soumpasis')
    pointer_pd['frap_filter_double_exp'] = ble.frap_filter(pointer_pd, 'double_exp')
    pointer_pd['frap_filter_ellenberg'] = ble.frap_filter(pointer_pd, 'ellenberg')
    pointer_pd['frap_filter_optimal'] = ble.frap_filter(pointer_pd, 'optimal')

    pointer_pd['pos'] = [s] * len(pointer_pd)
    pointer_ft_pd = pointer_pd[pointer_pd['frap_filter_optimal'] == 1]
    data_log['num_frap_curves'] = [len(pointer_ft_pd)]
    print("%d spots passed filters for FRAP curve quality control." % data_log['num_frap_curves'][0])

    # --------------------------
    # OUTPUT
    # --------------------------
    print("### Export data ...")

    storage_path = save_path
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # measurements
    # data_log
    data_log.to_csv('%s/data_log.txt' % storage_path, index=False, sep='\t')
    # full dataset of all bleach spots
    pointer_pd.to_csv('%s/data_full.txt' % storage_path, index=False, sep='\t')
    # dataset of control spots
    ctrl_pd.to_csv('%s/data_ctrl.txt' % storage_path, index=False, sep='\t')
    # dataset of nuclear
    nuclear_pd.to_csv('%s/data_nuclear.txt' % storage_path, index=False, sep='\t')
    # dataset of nucleoli
    nucleoli_pd.to_csv('%s/data_nucleoli.txt' % storage_path, index=False, sep='\t')

    # simplified dataset of bleach spots after FRAP curve quality control
    pointer_out = pd.DataFrame({'pos': pointer_ft_pd['pos'],
                                'bleach_spots': pointer_ft_pd['bleach_spots'],
                                'x': pointer_ft_pd['x'],
                                'y': pointer_ft_pd['y'],
                                'nucleoli': pointer_ft_pd['nucleoli'],
                                'nucleoli_size': pointer_ft_pd['nucleoli_size'],
                                'nucleoli_mean_int': pointer_ft_pd['nucleoli_mean_int'],
                                'bleach_frame': pointer_ft_pd['bleach_frame'],
                                'pre_bleach_int': pointer_ft_pd['pre_bleach_int'],
                                'start_int': pointer_ft_pd['frap_start_int'],
                                'mobile_fraction': pointer_ft_pd['mobile_fraction'],
                                't_half': pointer_ft_pd['t_half'],
                                'ini_slope': pointer_ft_pd['ini_slope'],
                                'linear_slope': pointer_ft_pd['linear_slope'],
                                'optimal_r2': pointer_ft_pd['optimal_r2'],
                                'optimal_mobile_fraction': pointer_ft_pd['optimal_mobile_fraction'],
                                'optimal_t_half': pointer_ft_pd['optimal_t_half'],
                                'optimal_slope': pointer_ft_pd['optimal_slope']})
    pointer_out.to_csv('%s/data.txt' % storage_path, index=False, sep='\t')

    # images
    dis.plot_offset_map(pointer_pd, storage_path)  # offset map
    dis.plot_raw_intensity(pointer_pd, ctrl_pd, storage_path)  # raw intensity
    dis.plot_pb_factor(pointer_pd, storage_path)  # photobleaching factor
    dis.plot_corrected_intensity(pointer_pd, storage_path)  # intensity after dual correction
    dis.plot_normalized_frap(pointer_pd, storage_path)  # normalized FRAP curves
    dis.plot_frap_fitting(pointer_pd, storage_path)  # normalized FRAP curves after filtering with fitting
    # individual normalized FRAP curves with fitting

print("DONE")