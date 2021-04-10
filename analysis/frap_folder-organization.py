import os
import shutil

data_source = "D:/Xiaowei/data/20210407_SG_frap/"

dirs = [x[0] for x in os.walk(data_source)]
dirs.pop(0)
num_dir = len(dirs)

for s in range(len(dirs)):
    data_path = dirs[s]
    mf_name = dirs[s].split('/')[-1].split('-')[0]
    move_path = ("%s/data/%s/" % (data_source, mf_name))
    if not os.path.exists(move_path):
        os.makedirs(move_path)
    shutil.move(dirs[s], move_path)

print("Done")
