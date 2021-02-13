import nd2reader as nd2
from matplotlib.figure import Figure
from shared.find_organelles import find_organelle, find_cell, get_binary_global
import os

data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/20200212_Jose_SG/data"
save_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/20200212_Jose_SG/dataAnalysis"

thresholding = 'local-sg1'  # only accepts 'na', 'otsu', 'yen', 'local-nucleoli' and 'local-sg'
min_size_sg = 100
max_size_sg = 10000
nuclear_local_thresholding_size = 701
min_size_nuclear = 15000
max_size_nuclear = 130000
min_size_cell = 50000
max_size_cell = 1000000

dirs = [x for x in os.listdir(data_path)]
dirs.pop(0)
num_dir = len(dirs)

for s in range(len(dirs)):

    name = dirs[s][:-4]
    print("### DATA PROCESSING: %s (%d / %d)" % (name, s + 1, num_dir))

    imgs = nd2.ND2Reader('%s/%s.nd2' % (data_path, name))

    # 0: SG channel
    # 1: cell boundary channel
    # 2: nuclei channel

    # identify SG
    print("### SG IDENTIFICATION ...")
    sg = find_organelle(imgs[0], thresholding, 3000, 200, min_size_sg, max_size_sg)

    # identify cell
    print("### CELL SEGMENTATION ...")
    cell, cell_ft = find_cell(imgs[0], imgs[1], imgs[2], nuclear_local_thresholding_size, min_size_nuclear,
                              max_size_nuclear, min_size_cell, max_size_cell)

    # figures
    storage_path = save_path
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    print("### FIGURE EXPORTATION ...")
    fig = Figure(figsize=(15, 10), dpi=100)
    # adding the subplot
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    # plotting the graph
    ax1.imshow(imgs[0], cmap='binary')
    ax2.imshow(imgs[1], cmap='binary')
    ax3.imshow(imgs[2], cmap='binary')
    ax4.imshow(sg, cmap='binary')
    ax5.imshow(cell, cmap='viridis')
    ax6.imshow(cell_ft, cmap='viridis')
    fig.savefig('%s/%s.pdf' % (storage_path, name))

print('DONE')
