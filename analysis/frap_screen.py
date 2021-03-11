import os
import shutil
import numpy as np
import pandas as pd
from pycromanager import Bridge
from shared.find_organelles import find_organelle, organelle_analysis, find_nuclear_nucleoli, nuclear_analysis
from skimage.measure import label
import shared.analysis as ana
import shared.dataframe as dat
import shared.display as dis
import shared.objects as obj
import shared.bleach_points as ble
import shared.math_functions as mat

# paths
data_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/20210310_SGfrapTest"
save_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Exp/20210310_SGfrapTest"
sampletable_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Exp/20210311_SampletableAndWTFile/"\
                    "sampletable.txt"
WT_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Exp/"\
            "20210100_coding_FRAPscriptDevelopment/20210304_exampleSampletableAndWTFile/WT.txt"

# saving options
save_name = 'dataAnalysis'

# values for analysis
analyze_organelle = 'sg'  # only accepts 'sg' or 'nucleoli'
data_c = 0
pos = 0
num_dilation = 3  # number of dilation from the coordinate;
# determines analysis size of the analysis spots; default = 3
frap_start_delay = 4  # 50ms default = 4; 100ms default = 5; 200ms default = 6

# presets
if analyze_organelle == 'sg':
    thresholding = 'na'
    # global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
    min_size = 5  # minimum size; sg default = 5
    max_size = 200  # maximum size; sg default = 200
else:  # for 'nucleoli'
    thresholding = 'local-nucleoli'
    # global thresholding method; choose in between 'na','otsu','yen', 'local-nucleoli' and 'local-sg'
    min_size = 10  # minimum size; nucleoli default = 10
    max_size = 1000  # maximum size; nucleoli default = 1000;
    # larger ones are generally cells without nucleoli

# modes
mode_bleach_detection = 'single-offset'  # only accepts 'single-raw' or 'single-offset'
frap_start_mode = 'min'  # only accepts 'delay' or 'min'
fitting_mode = 'single_exp'  # accepts 'single_exp', 'double_exp', 'soumpasis', 'ellenberg', 'optimal'
analysis_mode = 'single_exp'

# display settings for data analysis
inc = 5
repeat = 50

# run_mode
folder_organization = 'N'
movie_analysis = 'Y'
file_processing = 'Y'
data_analysis = 'N'

"""
# ---------------------------------------------------------------------------------------------------
# PLEASE DO NOT CHANGE AFTER THIS
# ---------------------------------------------------------------------------------------------------
"""

# --------------------------
# Folder organization
# --------------------------
if folder_organization == 'Y':
    print("######## FOLDER ORGANIZING #######")
    dirs = [x[0] for x in os.walk(data_source)]
    dirs.pop(0)

    for s in range(len(dirs)):
        data_path = dirs[s]
        mf_name = dirs[s].split('/')[-1].split('-')[0]
        move_path = ("%s/data/%s/" % (data_source, mf_name))
        if not os.path.exists(move_path):
            os.makedirs(move_path)
        shutil.move(dirs[s], move_path)

# --------------------------
# Movie analysis
# --------------------------
if movie_analysis == 'Y':
    print("####### MOVIE ANALYSIS #######")
    data_store_source = ("%s/data/" % data_source)

    sampletable = pd.read_csv(sampletable_source, na_values=['.'], sep='\t')
    dirs = [x for x in os.listdir(data_store_source)]
    dirs = dat.remove_elements(dirs, sampletable['well'].tolist())

    for r in range(len(dirs)):
        data_folder = ("%s/%s" % (data_store_source, dirs[r]))
        sub_dirs = [x[0] for x in os.walk(data_folder)]
        sub_dirs.pop(0)

        for s in range(len(sub_dirs)):
            folder = sub_dirs[s].split('/')[-1]
            mf_name = sub_dirs[s].split('/')[-2]
            print("### DATA PROCESSING ...")
            print("well: %s (%d / %d)" % (mf_name, r + 1, len(dirs)))
            print("FOV: %s (%d / %d)" % (folder, s + 1, len(sub_dirs)))
            data_path = sub_dirs[s]
            save_path = ("%s/%s/%s/%s" % (data_source, save_name, mf_name, folder))
            pos = sub_dirs[s].split('/')[-1].split('_')[1]

            # --------------------------
            # LOAD MOVIE
            # --------------------------
            print("### Load movie ...")
            data_log = pd.DataFrame({'pos': [pos]})

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
            data_log['acquire_time'] = [acquire_time_tseries]
            data_log['real_time'] = [real_time]

            # --------------------------------------
            # ORGANELLE ANALYSIS based on time 0
            # --------------------------------------
            print("### Image analysis: %s detection based on time 0 ..." % analyze_organelle)

            # reference image of time 0
            # if decide to use other image as ref_image
            # be sure to check photobleaching correction for all reported intensities
            temp = store.get_image(cb.c(data_c).t(0).build())
            pix = np.reshape(temp.get_raw_pixels(), newshape=[temp.get_height(), temp.get_width()])

            if analyze_organelle == 'nucleoli':
                # nuclear detection (currently only doable for nucleoli staining image)
                label_nuclear = find_nuclear_nucleoli(pix)
                data_log['num_nuclei_detected'] = [np.amax(label_nuclear)]
                print("Found %d nuclei." % data_log['num_nuclei_detected'][0])

            # organelle detection
            organelle = find_organelle(pix, thresholding, min_size=min_size, max_size=max_size)
            label_organelle = label(organelle, connectivity=1)
            data_log['num_%s_detected' % analyze_organelle] = [obj.object_count(organelle)]
            print("Found %d %s." % (data_log['num_%s_detected' % analyze_organelle][0], analyze_organelle))

            # organelle pd dataset
            organelle_pd = organelle_analysis(pix, organelle, '%s' % analyze_organelle, pos)

            if analyze_organelle == 'nucleoli':
                # link nucleoli with corresponding nuclear
                round_x = [round(num) for num in organelle_pd['x']]
                round_y = [round(num) for num in organelle_pd['y']]
                organelle_pd['nuclear'] = obj.points_in_objects(label_nuclear, round_y, round_x)

                data_log['num_nucleoli_in_nuclei'] = [len(organelle_pd[organelle_pd['nuclear'] != 0])]
                print("Found %d out of %d nucleoli within nuclei." % (data_log['num_nucleoli_in_nuclei'][0],
                                                                      obj.object_count(organelle)))

                # nuclear pd dataset
                nuclear_pd = nuclear_analysis(label_nuclear, organelle_pd, pos)

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
            coordinate_pd = ble.get_bleach_spots_coordinates(log_pd, store, cb, data_c, mode_bleach_detection,
                                                             frap_start_delay)
            log_pd = pd.concat([log_pd, coordinate_pd], axis=1)

            # link pointer with corresponding organelle
            log_pd['%s' % analyze_organelle] = obj.points_in_objects(label_organelle, log_pd['x'], log_pd['y'])

            # generate bleach spot mask and bleach spots dataframe (pointer_pd)
            bleach_spots, pointer_pd = ble.get_bleach_spots(log_pd, label_organelle, analyze_organelle, num_dilation)
            data_log['num_bleach_spots'] = [obj.object_count(bleach_spots)]
            print("%d spots passed filters for analysis." % data_log['num_bleach_spots'][0])

            # add bleach spots corresponding organelle measurements
            pointer_pd = dat.copy_based_on_index(pointer_pd, organelle_pd, '%s' % analyze_organelle,
                                                 '%s' % analyze_organelle,
                                                 ['%s_x' % analyze_organelle, '%s_y' % analyze_organelle,
                                                  '%s_size' % analyze_organelle, '%s_mean_int' % analyze_organelle,
                                                  '%s_circ' % analyze_organelle],
                                                 ['x', 'y', 'size', 'raw_int', 'circ'])

            # --------------------------------------------------
            # FRAP CURVE ANALYSIS from bleach spots
            # --------------------------------------------------
            print("### Image analysis: FRAP curve calculation ...")

            # create control spots mask
            ctrl_organelle = ~organelle_pd.index.isin(log_pd['%s' % analyze_organelle].tolist())
            ctrl_x = organelle_pd[ctrl_organelle]['x'].astype(int).tolist()
            ctrl_y = organelle_pd[ctrl_organelle]['y'].astype(int).tolist()
            ctrl_spots = ana.analysis_mask(ctrl_x, ctrl_y, pix, num_dilation)
            num_ctrl_spots = obj.object_count(ctrl_spots)
            pointer_pd['num_ctrl_spots'] = [num_ctrl_spots] * len(pointer_pd)

            # get raw intensities for bleach spots and control spots
            pointer_pd['raw_int'] = ana.get_intensity(bleach_spots, pixels_tseries)
            ctrl_spots_int_tseries = ana.get_intensity(ctrl_spots, pixels_tseries)
            ctrl_pd = pd.DataFrame({'pos': [pos] * num_ctrl_spots, 'ctrl_spots': np.arange(0, num_ctrl_spots, 1),
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

            # filter control traces
            filter_ctrl = []
            for i in range(len(ctrl_pd)):
                ctrl_int = ctrl_pd['bg_cor_int'][i]
                if (max(ctrl_int) - min(ctrl_int)) / max(ctrl_int) > 0.4:
                    filter_ctrl.append(0)
                else:
                    filter_ctrl.append(1)
            ctrl_pd['filter'] = filter_ctrl
            ctrl_pd_ft = ctrl_pd[ctrl_pd['filter'] == 1].reset_index()
            pointer_pd['num_ctrl_spots_ft'] = [len(ctrl_pd_ft)] * len(pointer_pd)
            data_log['num_ctrl_spots'] = len(ctrl_pd_ft)

            print("### Image analysis: photobleaching correction ...")
            # photobleaching factor calculation
            if len(ctrl_pd_ft) != 0:
                # calculate photobleaching factor
                pb_factor = ana.get_pb_factor(ctrl_pd_ft['bg_cor_int'])

                pointer_pd['pb_factor'] = [pb_factor] * len(pointer_pd)
                print("%d ctrl points are used to correct photobleaching." % len(ctrl_pd_ft))

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
                frap_pd = ble.frap_analysis(pointer_pd, max_t, acquire_time_tseries, real_time, frap_start_delay,
                                            frap_start_mode)
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
                optimal_fit_pd = mat.find_optimal_fitting(pointer_pd,
                                                          ['single_exp', 'soumpasis', 'ellenberg', 'double_exp'])
                pointer_pd = pd.concat([pointer_pd, optimal_fit_pd], axis=1)

                # filter frap curves
                pointer_pd['frap_filter_single_exp'] = ble.frap_filter(pointer_pd, 'single_exp')
                pointer_pd['frap_filter_soumpasis'] = ble.frap_filter(pointer_pd, 'soumpasis')
                pointer_pd['frap_filter_double_exp'] = ble.frap_filter(pointer_pd, 'double_exp')
                pointer_pd['frap_filter_ellenberg'] = ble.frap_filter(pointer_pd, 'ellenberg')
                pointer_pd['frap_filter_optimal'] = ble.frap_filter(pointer_pd, 'optimal')

                pointer_pd['pos'] = [pos] * len(pointer_pd)
                pointer_ft_pd = pointer_pd[pointer_pd['frap_filter_%s' % fitting_mode] == 1]
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
                if analyze_organelle == 'nucleoli':
                    # dataset of nuclear
                    nuclear_pd.to_csv('%s/data_nuclear.txt' % storage_path, index=False, sep='\t')
                # dataset of organelle
                organelle_pd.to_csv('%s/data_%s.txt' % (storage_path, analyze_organelle), index=False, sep='\t')

                # images
                dis.plot_offset_map(pointer_pd, fitting_mode, storage_path)  # offset map
                dis.plot_raw_intensity(pointer_pd, ctrl_pd_ft, fitting_mode, storage_path)  # raw intensity
                dis.plot_pb_factor(pointer_pd, storage_path)  # photobleaching factor
                dis.plot_corrected_intensity(pointer_pd, fitting_mode, storage_path)  # intensity after dual correction
                dis.plot_normalized_frap(pointer_pd, fitting_mode, storage_path)  # normalized FRAP curves
                dis.plot_frap_fitting(pointer_pd, fitting_mode,
                                      storage_path)  # normalized FRAP curves after filtering with fitting
                # individual normalized FRAP curves with fitting
            else:
                # --------------------------
                # OUTPUT
                # --------------------------
                print("### Export data ...")

                storage_path = save_path
                if not os.path.exists(storage_path):
                    os.makedirs(storage_path)
                # data_log
                data_log.to_csv('%s/data_log.txt' % storage_path, index=False, sep='\t')

# --------------------------
# File processing
# --------------------------
if file_processing == 'Y':
    print("####### FILE PROCESSING #######")
    data_analysis_source = ("%s/%s/" % (data_source, save_name))
    dirs = [x for x in os.listdir(data_analysis_source)]
    sampletable = pd.read_csv(sampletable_source, na_values=['.'], sep='\t')
    dirs = dat.remove_elements(dirs, sampletable['well'].tolist())

    for r in range(len(dirs)):
        data_analysis_folder = ("%s/%s" % (data_analysis_source, dirs[r]))
        mf_name = dirs[r]
        if mf_name in sampletable['well'].tolist():
            name = sampletable[sampletable['well'] == mf_name]['name'].tolist()[0]
        else:
            name = mf_name

        sub_dirs = [x[0] for x in os.walk(data_analysis_folder)]
        sub_dirs.pop(0)
        num_sub_dir = len(sub_dirs)

        save_path = save_source

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f_full = open("%s/%s_data_full.txt" % (save_path, name), 'w+')
        f_log = open("%s/%s_data_log.txt" % (save_path, name), 'w+')
        f_ctrl = open("%s/%s_data_ctrl.txt" % (save_path, name), 'w+')
        if analyze_organelle == 'nucleoli':
            f_nuclear = open("%s/%s_data_nuclear.txt" % (save_path, name), 'w+')
        f_organelle = open("%s/%s_data_%s.txt" % (save_path, name, analyze_organelle), 'w+')

        for s in range(len(sub_dirs)):
            data_path = sub_dirs[s]
            if os.path.exists("%s/data_full.txt" % data_path):
                f1_full = open("%s/data_full.txt" % data_path, 'r')
                dat.append_data(f_full, f1_full, s)
                f1_full.close()
                f1_log = open("%s/data_log.txt" % data_path, 'r')
                dat.append_data(f_log, f1_log, s)
                f1_log.close()
                f1_ctrl = open("%s/data_ctrl.txt" % data_path, 'r')
                dat.append_data(f_ctrl, f1_ctrl, s)
                f1_ctrl.close()
                if analyze_organelle == 'nucleoli':
                    f1_nuclear = open("%s/data_nuclear.txt" % data_path, 'r')
                    dat.append_data(f_nuclear, f1_nuclear, s)
                    f1_nuclear.close()
                f1_organelle = open("%s/data_%s.txt" % (data_path, analyze_organelle), 'r')
                dat.append_data(f_organelle, f1_organelle, s)
                f1_organelle.close()

        f_full.close()
        f_log.close()
        f_ctrl.close()
        if analyze_organelle == 'nucleoli':
            f_nuclear.close()
        f_organelle.close()

# --------------------------
# Data analysis
# --------------------------
if data_analysis == 'Y':
    print("####### DATA ANALYSIS #######")
    WT = pd.read_csv(WT_source, na_values=['.'], sep='\t')
    WT_lst = []
    for i in range(WT['nSample_ctrl'][0]):
        WT_lst.append(WT['name_ctrl'][0].split(',')[i])

    # organize files
    files = [x for x in os.listdir(save_source)]
    num_files = len(files)

    if files[0].split('.')[-1] == 'txt':
        for s in range(num_files):
            name = files[s].split('_')[0]
            if name in WT_lst:
                move_path = ("%s/WT/%s/" % (save_source, name))
            else:
                move_path = ("%s/sample/%s/" % (save_source, name))
            if not os.path.exists(move_path):
                os.makedirs(move_path)
            shutil.move(("%s/%s" % (save_source, files[s])), move_path)

    print("### Create WT for comparison...")
    # create WT samples from all the controls
    WT_folders = [x for x in os.listdir("%s/WT/" % save_source)]
    WT_folders = dat.remove_elements(WT_folders, WT_lst)

    data_WT = pd.DataFrame()
    data_ctrl_WT = pd.DataFrame()
    data_log_WT = pd.DataFrame()
    data_nuclear_WT = pd.DataFrame()
    data_nucleoli_WT = pd.DataFrame()

    for i in range(len(WT_folders)):
        data_WT_temp = pd.read_csv(("%s/WT/%s/%s_data_full.txt" % (save_source, WT_folders[i], WT_folders[i])),
                                   na_values=['.'], sep='\t')
        data_ctrl_WT_temp = pd.read_csv(("%s/WT/%s/%s_data_ctrl.txt" % (save_source, WT_folders[i], WT_folders[i])),
                                        na_values=['.'], sep='\t')
        data_log_WT_temp = pd.read_csv(("%s/WT/%s/%s_data_log.txt" % (save_source, WT_folders[i], WT_folders[i])),
                                       na_values=['.'], sep='\t')
        data_nuclear_WT_temp = pd.read_csv(("%s/WT/%s/%s_data_nuclear.txt" % (save_source, WT_folders[i], WT_folders[i])),
                                           na_values=['.'], sep='\t')
        data_nucleoli_WT_temp = pd.read_csv(("%s/WT/%s/%s_data_nucleoli.txt" % (save_source, WT_folders[i], WT_folders[i])),
                                            na_values=['.'], sep='\t')

        data_WT = pd.concat([data_WT, data_WT_temp])
        data_ctrl_WT = pd.concat([data_ctrl_WT, data_ctrl_WT_temp])
        data_log_WT = pd.concat([data_log_WT, data_log_WT_temp])
        data_nuclear_WT = pd.concat([data_nuclear_WT, data_nuclear_WT_temp])
        data_nucleoli_WT = pd.concat([data_nucleoli_WT, data_nucleoli_WT_temp])

        if not os.path.exists('%s/WT/WT_sum/' % save_source):
            os.makedirs('%s/WT/WT_sum/' % save_source)
        data_WT.to_csv('%s/WT/WT_sum/WT_data_full.txt' % save_source, index=False, sep='\t')
        data_ctrl_WT.to_csv('%s/WT/WT_sum/WT_data_ctrl.txt' % save_source, index=False, sep='\t')
        data_log_WT.to_csv('%s/WT/WT_sum/WT_data_log.txt' % save_source, index=False, sep='\t')
        data_nuclear_WT.to_csv('%s/WT/WT_sum/WT_data_nuclear.txt' % save_source, index=False, sep='\t')
        data_nucleoli_WT.to_csv('%s/WT/WT_sum/WT_data_nucleoli.txt' % save_source, index=False, sep='\t')

    # do comparison
    WT_comparison_lst = []
    for i in range(WT['nSample_comparison'][0]):
        WT_comparison_lst.append(WT['name_comparison'][0].split(',')[i])

    data_WT1 = pd.read_csv(("%s/sample/%s/%s_data_full.txt" % (save_source, WT_comparison_lst[0], WT_comparison_lst[0])),
                           na_values=['.'], sep='\t')
    data_WT2 = pd.read_csv(("%s/sample/%s/%s_data_full.txt" % (save_source, WT_comparison_lst[1], WT_comparison_lst[1])),
                           na_values=['.'], sep='\t')
    data_WT3 = pd.read_csv(("%s/sample/%s/%s_data_full.txt" % (save_source, WT_comparison_lst[2], WT_comparison_lst[2])),
                           na_values=['.'], sep='\t')

    data_WT1_ft = data_WT1[data_WT1['frap_filter_%s' % analysis_mode] == 1]
    data_WT2_ft = data_WT2[data_WT2['frap_filter_%s' % analysis_mode] == 1]
    data_WT3_ft = data_WT3[data_WT3['frap_filter_%s' % analysis_mode] == 1]
    data_WT_ft = data_WT[data_WT['frap_filter_%s' % analysis_mode] == 1]

    sample_folders = [x for x in os.listdir("%s/sample/" % save_source)]
    sample_folders = dat.remove_elements(sample_folders, sampletable['name'].tolist()+sampletable['well'].tolist())

    for i in range(len(sample_folders)):
        print("### Calculate %s (%d / %d)" % (sample_folders[i], i+1, len(sample_folders)))
        if sample_folders[i] in WT_comparison_lst:
            continue
        else:
            sample_folder = ("%s/sample/%s/" % (save_source, sample_folders[i]))
            data_sample = pd.read_csv(("%s/%s_data_full.txt" % (sample_folder, sample_folders[i])), na_values=['.'],
                                      sep='\t')
            data_sample_ft = data_sample[data_sample['frap_filter_%s' % analysis_mode] == 1]
            limit = min([len(data_WT_ft), len(data_WT1_ft), len(data_WT2_ft), len(data_WT3_ft), len(data_sample_ft)])

            pd_lst = [data_WT_ft, data_WT1_ft, data_WT2_ft, data_WT3_ft, data_sample_ft]
            curve_mob = dat.make_full_lst(pd_lst, 'mobile_fraction')
            mob = dat.make_full_lst(pd_lst, '%s_mobile_fraction' % analysis_mode)
            curve_t_half = dat.make_full_lst(pd_lst, 't_half')
            t_half = dat.make_full_lst(pd_lst, '%s_t_half' % analysis_mode)
            curve_slope = dat.make_full_lst(pd_lst, 'linear_slope')
            slope = dat.make_full_lst(pd_lst, '%s_slope' % analysis_mode)
            nucleoli_size = dat.make_full_lst(pd_lst, 'nucleoli_size')
            sample = ['WT']*len(data_WT_ft) + ['WT1']*len(data_WT1_ft) + ['WT2']*len(data_WT2_ft) \
                     + ['WT3']*len(data_WT3_ft) + ['%s' % (sample_folders[i])]*len(data_sample_ft)
            data = pd.DataFrame(
                {'mob': mob, 'curve_mob': curve_mob, 't_half': t_half, 'curve_t_half': curve_t_half, 'slope': slope,
                 'curve_slope': curve_slope, 'nucleoli size': nucleoli_size, 'sample': sample})

            print("# Export mobile_fraction -ln(p) ...")
            dis.plot_minus_ln_p(inc, limit, repeat, 'mobile_fraction', data_WT_ft, data_WT1_ft, data_WT2_ft,
                                data_WT3_ft, data_sample_ft, sample_folder, sample_folders[i])
            dis.plot_minus_ln_p(inc, limit, repeat, '%s_mobile_fraction' % analysis_mode, data_WT_ft, data_WT1_ft,
                                data_WT2_ft, data_WT3_ft, data_sample_ft, sample_folder, sample_folders[i])
            print("# Export t_half -ln(p) ...")
            dis.plot_minus_ln_p(inc, limit, repeat, 't_half', data_WT_ft, data_WT1_ft, data_WT2_ft, data_WT3_ft,
                                data_sample_ft, sample_folder, sample_folders[i])
            dis.plot_minus_ln_p(inc, limit, repeat, '%s_t_half' % analysis_mode, data_WT_ft, data_WT1_ft, data_WT2_ft,
                                data_WT3_ft, data_sample_ft, sample_folder, sample_folders[i])
            print("# Export linear_slope -ln(p) ...")
            dis.plot_minus_ln_p(inc, limit, repeat, 'linear_slope', data_WT_ft, data_WT1_ft, data_WT2_ft, data_WT3_ft,
                                data_sample_ft, sample_folder, sample_folders[i])
            dis.plot_minus_ln_p(inc, limit, repeat, '%s_slope' % analysis_mode, data_WT_ft, data_WT1_ft, data_WT2_ft,
                                data_WT3_ft, data_sample_ft, sample_folder, sample_folders[i])
            print("# Export violin plots ...")
            dis.plot_violin('mob', data, sample_folder, sample_folders[i])
            dis.plot_violin('curve_mob', data, sample_folder, sample_folders[i])
            dis.plot_violin('t_half', data, sample_folder, sample_folders[i])
            dis.plot_violin('curve_t_half', data, sample_folder, sample_folders[i])
            dis.plot_violin('slope', data, sample_folder, sample_folders[i])
            dis.plot_violin('curve_slope', data, sample_folder, sample_folders[i])

print("DONE!")
