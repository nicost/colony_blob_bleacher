import nd2reader as nd2
import numpy as np
import napari
from vispy.color import Colormap
import shared.display as dis
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from shared.find_organelles import find_organelle, find_cell

data_path = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/20210212_Jose_SG/data/"

thresholding = 'local-sg1'  # only accepts 'na', 'otsu', 'yen', 'local-nucleoli' and 'local-sg', 'local-sg1'
min_size_sg = 100
max_size_sg = 10000
nuclear_local_thresholding_size = 701
min_size_nuclear = 15000
max_size_nuclear = 130000
min_size_cell = 50000
max_size_cell = 1000000

name = 'G3BP1_SA50uM_5WGA009'
imgs = nd2.ND2Reader('%s/%s.nd2' % (data_path, name))

# 0: SG channel
# 1: cell boundary channel
# 2: nuclei channel

# identify SG
sg = find_organelle(imgs[0], thresholding, 3000, 200, min_size_sg, max_size_sg)

# identify cell
cell, cell_ft = find_cell(imgs[0], imgs[1], imgs[2], nuclear_local_thresholding_size, min_size_nuclear,
                          max_size_nuclear, min_size_cell, max_size_cell)

with napari.gui_qt():
    # embed mpl widget in napari viewer
    viewer = napari.Viewer()
    viewer.add_image(np.array(imgs[1]), name='data_mem', colormap='red', blending='additive',
                     contrast_limits=(100, 1200))
    viewer.add_image(np.array(imgs[0]), name='data_sg', colormap='green', blending='additive')
    viewer.add_image(np.array(imgs[2]), name='data_nuclear', colormap='blue', blending='additive',
                     contrast_limits=(100, 250))

    cmap1 = 'viridis'
    cmap1_woBg = dis.num_color_colormap(cmap1, np.amax(cell))[0]
    viewer.add_image(cell, name='cell', colormap=('cmap1 woBg', cmap1_woBg), blending='additive', opacity=0.5)

    violet_woBg = Colormap([[0.0, 0.0, 0.0, 0.0], [129 / 255, 55 / 255, 114 / 255, 1.0]])
    viewer.add_image(sg, name='SG', contrast_limits=[0, 1], colormap=('violet woBg', violet_woBg))

"""# plotting the graph in tkinter window
# the main Tkinter window
window = Tk()
# setting the title
window.title('SG analysis')
# dimensions of the main window
window.geometry("1500x1000")
# figures
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
# creating the Tkinter canvas
# containing the Matplotlib figure
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
# placing the canvas on the Tkinter window
canvas.get_tk_widget().pack()
# run the gui
window.mainloop()"""
