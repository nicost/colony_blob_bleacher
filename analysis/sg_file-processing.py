import os
import shared.dataframe as dat

data_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/20210323_SG_top10MonoclonalCells/" \
            "dataAnalysis/"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/SG/Exp/" \
            "20201100_exp_proofTestOf10Genes_cellLineGeneration/02_CellLineGeneration/" \
            "20210324_dataAnalysis_monoclonalCells/"

name = '95-5_5_7_8'
dirs = [x[0] for x in os.walk(data_source)]
dirs.pop(0)
num_dir = len(dirs)

if not os.path.exists(save_path):
    os.makedirs(save_path)

f_full = open("%s/%s_data.txt" % (save_path, name), 'w+')

for s in range(len(dirs)):
    data_path = dirs[s]
    if os.path.exists("%s/data.txt" % data_path):
        f1_full = open("%s/data.txt" % data_path, 'r')
        dat.append_data(f_full, f1_full, s)
        f1_full.close()

f_full.close()

print('Done')
