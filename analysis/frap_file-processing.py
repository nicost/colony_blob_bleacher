import os
import shared.dataframe as dat

data_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/" \
              "20210203_CBB_nucleoliArsAndHeatshockTreatment/WT3"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/" \
              "20210203_CBB_nucleoliArsAndHeatshockTreatment/dataAnalysis"

name = 'WT3'
dirs = [x[0] for x in os.walk(data_source)]
dirs.pop(0)
num_dir = len(dirs)

if not os.path.exists(save_path):
    os.makedirs(save_path)
f_full = open("%s/%s_data_full.txt" % (save_path, name), 'w+')
f_data = open("%s/%s_data.txt" % (save_path, name), 'w+')
f_log = open("%s/%s_data_log.txt" % (save_path, name), 'w+')
f_ctrl = open("%s/%s_data_ctrl.txt" % (save_path, name), 'w+')
f_nuclear = open("%s/%s_data_nuclear.txt" % (save_path, name), 'w+')
f_nucleoli = open("%s/%s_data_nucleoli.txt" % (save_path, name), 'w+')

for s in range(len(dirs)):
    data_path = dirs[s]
    f1_full = open("%s/data_full.txt" % data_path, 'r')
    dat.append_data(f_full, f1_full, s)
    f1_full.close()
    f1_data = open("%s/data.txt" % data_path, 'r')
    dat.append_data(f_data, f1_data, s)
    f1_data.close()
    f1_log = open("%s/data_log.txt" % data_path, 'r')
    dat.append_data(f_log, f1_log, s)
    f1_log.close()
    f1_ctrl = open("%s/data_ctrl.txt" % data_path, 'r')
    dat.append_data(f_ctrl, f1_ctrl, s)
    f1_ctrl.close()
    f1_nuclear = open("%s/data_nuclear.txt" % data_path, 'r')
    dat.append_data(f_nuclear, f1_nuclear, s)
    f1_nuclear.close()
    f1_nucleoli = open("%s/data_nucleoli.txt" % data_path, 'r')
    dat.append_data(f_nucleoli, f1_nucleoli, s)
    f1_nucleoli.close()

f_full.close()
f_data.close()
f_log.close()
f_ctrl.close()
f_nuclear.close()
f_nucleoli.close()

print('Done')
