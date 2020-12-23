import napari
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


with napari.gui_qt():
    mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
    static_ax = mpl_widget.figure.subplots()
    t = np.linspace(0, 10, 501)
    static_ax.plot(t, np.tan(t), ".")

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(mpl_widget)