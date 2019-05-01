
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
#               2015 Jens H Nielsen
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

from __future__ import unicode_literals
import sys
import os
import time
import random
import json
import ctypes

single_window = False
plot_windows  = []

import logging
logger = logging.getLogger('auspex_plot_client')
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(name)s-%(levelname)s: %(asctime)s ----> %(message)s')

from scipy.spatial import Delaunay

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon

import numpy as np
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.5"

import zmq

main_app_mdi = None

class DataListener(QtCore.QObject):

    message  = QtCore.pyqtSignal(tuple)
    finished = QtCore.pyqtSignal()

    def __init__(self, host, uuid, num_plots, port=7772):
        QtCore.QObject.__init__(self)

        self.uuid = uuid
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.num_plots = num_plots
        self.socket.connect("tcp://{}:{}".format(host, port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, uuid)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.running = True

    def loop(self):
        while self.running:
            done_plots = 0
            socks = dict(self.poller.poll(1000))
            if socks.get(self.socket) == zmq.POLLIN:
                msg = self.socket.recv_multipart()
                msg_type = msg[1].decode()
                uuid     = msg[0].decode()
                name     = msg[2].decode()
                if msg_type == "done":
                    done_plots += 1
                    if done_plots == self.num_plots:
                        self.finished.emit()
                        logger.debug(f"Data listener thread for {self.uuid} got done message.")
                elif msg_type == "data":
                    result = [name, uuid]
                    # How many pairs of metadata and data are there?
                    num_arrays = int((len(msg) - 3)/2)
                    for i in range(num_arrays):
                        md, data = msg[3+2*i:5+2*i]
                        md = json.loads(md.decode())
                        A = np.frombuffer(data, dtype=md['dtype'])
                        result.append(A)
                    self.message.emit(tuple(result))
        self.socket.close()
        self.context.term()
        logger.debug(f"Data listener thread for {self.uuid} exiting.")

class DescListener(QtCore.QObject):

    new_plot   = QtCore.pyqtSignal(tuple)
    first_plot = QtCore.pyqtSignal()

    def __init__(self, host, port=7771):
        QtCore.QObject.__init__(self)
        self.got_plot = False

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = "Matplotlib_Qt_Client".encode()
        self.socket.connect("tcp://{}:{}".format(host, port))
        self.socket.send_multipart([b"new_client"])
        self.socket.setsockopt(zmq.LINGER, 0)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        time.sleep(0.1)
        logger.debug("desc listener init")

        self.running = True

    def loop(self):
        logger.debug("desc loop")
        while self.running:
            evts = dict(self.poller.poll(1000))
            if self.socket in evts and evts[self.socket] == zmq.POLLIN:
                msg_type, uuid, desc = [e.decode() for e in self.socket.recv_multipart()]
                logger.debug(f"GOT: {msg_type}, {uuid}, {desc}")
                if msg_type == "new":
                    if not self.got_plot:
                        self.got_plot = True
                        self.first_plot.emit()
                    self.new_plot.emit(tuple([uuid, desc]))
        logger.debug("Desc listener at end")
        self.socket.close()
        self.context.term()

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100, plot_mode="quad"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.plots = []

        if plot_mode == "quad":
            self.real_axis  = self.fig.add_subplot(221)
            self.imag_axis  = self.fig.add_subplot(222)
            self.abs_axis   = self.fig.add_subplot(223)
            self.phase_axis = self.fig.add_subplot(224)
            self.axes = [self.real_axis, self.imag_axis, self.abs_axis, self.phase_axis]
            self.func_names = ["Real", "Imag", "Abs", "Phase"]
            self.plot_funcs = [np.real, np.imag, np.abs, np.angle]
        elif plot_mode == "real":
            self.real_axis  = self.fig.add_subplot(111)
            self.axes = [self.real_axis]
            self.func_names = ["Real"]
            self.plot_funcs = [np.real]
        elif plot_mode == "imag":
            self.imag_axis  = self.fig.add_subplot(111)
            self.axes = [self.imag_axis]
            self.func_names = ["Imag"]
            self.plot_funcs = [np.imag]
        elif plot_mode == "amp":
            self.abs_axis  = self.fig.add_subplot(111)
            self.axes = [self.abs_axis]
            self.func_names = ["Amp"]
            self.plot_funcs = [np.abs]
        elif plot_mode == "real/imag":
            self.real_axis  = self.fig.add_subplot(121)
            self.imag_axis  = self.fig.add_subplot(122)
            self.axes = [self.real_axis, self.imag_axis]
            self.func_names = ["Real", "Imag"]
            self.plot_funcs = [np.real, np.imag]
        elif plot_mode == "amp/phase":
            self.abs_axis  = self.fig.add_subplot(121)
            self.phase_axis  = self.fig.add_subplot(122)
            self.axes = [self.abs_axis, self.phase_axis]
            self.func_names = ["Amp", "Phase"]
            self.plot_funcs = [np.abs, np.angle]

        self.compute_initial_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class Canvas1D(MplCanvas):
    def compute_initial_figure(self):
        for ax in self.axes:
            plt, = ax.plot([0,0,0])
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
            self.plots.append(plt)

    def update_figure(self, data):
        x_data, y_data = data
        for plt, ax, f in zip(self.plots, self.axes, self.plot_funcs):
            plt.set_xdata(x_data)
            plt.set_ydata(f(y_data))
            ax.relim()
            ax.autoscale_view()
            self.draw()
            self.flush_events()

    def set_desc(self, desc):
        for ax, name in zip(self.axes, self.func_names):
            if 'x_label' in desc.keys():
                ax.set_xlabel(desc['x_label'])
            if 'y_label' in desc.keys():
                ax.set_ylabel(name + " " + desc['y_label'])
        for plt in self.plots:
            plt.set_xdata(np.linspace(desc['x_min'], desc['x_max'], desc['x_len']))
            plt.set_ydata(np.nan*np.linspace(desc['x_min'], desc['x_max'], desc['x_len']))
        self.fig.tight_layout()

class CanvasManual(MplCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, numplots=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = []
        for n in range(numplots):
            self.axes += [self.fig.add_subplot(100+10*(numplots)+n+1)]
            self.axes[n].ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
            self.axes[n].ticklabel_format(style='sci', axis='y', scilimits=(-3,3))

        self.traces = {}

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def update_trace(self, trace_name, x_data, y_data):
        self.traces[trace_name]['plot'].set_xdata(x_data)
        self.traces[trace_name]['plot'].set_ydata(y_data)
        curr_axis = self.axes[self.traces[trace_name]['axis_num']]
        curr_axis.relim()
        curr_axis.autoscale_view()
        if len(self.traces)>1:
            curr_axis.legend()
        self.draw()
        self.flush_events()

    def set_desc(self, desc):
        for k, ax in enumerate(self.axes):
            if 'x_label' in desc.keys():
                ax.set_xlabel(desc['x_label'][k])
            if 'y_label' in desc.keys():
                ax.set_ylabel(desc['y_label'][k])
            if 'y_lim' in desc.keys() and desc['y_lim']:
                ax.set_ylim(*desc['y_lim'])
        for trace in desc['traces']:  # relink traces and axes
            self.traces[trace['name']] = {'plot': self.axes[trace['axis_num']].plot([], label=trace['name'], **trace['matplotlib_kwargs'])[0], 'axis_num': trace['axis_num']}
        self.fig.tight_layout()

class Canvas2D(MplCanvas):
    def compute_initial_figure(self):
        for ax in self.axes:
            plt = ax.imshow(np.zeros((10,10)))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
            self.plots.append(plt)

    def update_figure(self, data):
        x_data, y_data, im_data = data
        im_data = im_data.reshape((len(y_data), len(x_data)), order='c')
        for plt, f in zip(self.plots, self.plot_funcs):
            plt.set_data(f(im_data))
            plt.autoscale()
        self.draw()
        self.flush_events()

    def set_desc(self, desc):
        self.aspect = (desc['x_max']-desc['x_min'])/(desc['y_max']-desc['y_min'])
        self.extent = (desc['x_min'], desc['x_max'], desc['y_min'], desc['y_max'])
        self.xlen = desc['x_len']
        self.ylen = desc['y_len']
        self.plots = []
        for ax in self.axes:
            ax.clear()
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
            plt = ax.imshow(np.zeros((self.xlen, self.ylen)),
                animated=True, aspect=self.aspect, extent=self.extent, origin="lower")
            self.plots.append(plt)
        for ax, name in zip(self.axes, self.func_names):
            if 'x_label' in desc.keys():
                ax.set_xlabel(desc['x_label'])
            if 'y_label' in desc.keys():
                ax.set_ylabel(name + " " + desc['y_label'])
        self.fig.tight_layout()

class CanvasMesh(MplCanvas):
    def compute_initial_figure(self):
        # data = np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]])
        # self.update_figure(np.array(data))
        pass

    def update_figure(self, data):
        # Expected xs, ys, zs coming in as
        # data = np.array([xs, ys, zs]).transpose()
        data = data.reshape((-1, 3), order='c')
        points = data[:,0:2]
        mesh = self.scaled_Delaunay(points)
        xs   = mesh.points[:,0]
        ys   = mesh.points[:,1]
        for ax, f in zip(self.axes, self.plot_funcs):
            ax.clear()
            ax.tripcolor(xs, ys, mesh.simplices, f(data[:,2]), cmap="RdGy", shading="flat")
            ax.autoscale()
        self.draw()
        self.flush_events()

    def set_desc(self, desc):

        self.plots = []
        for ax, name in zip(self.axes, self.func_names):
            if 'x_label' in desc.keys():
                ax.set_xlabel(desc['x_label'])
            if 'y_label' in desc.keys():
                ax.set_ylabel(name + " " + desc['y_label'])
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
        self.fig.tight_layout()

    def scaled_Delaunay(self, points):
        """ Return a scaled Delaunay mesh and scale factors """
        scale_factors = []
        points = np.array(points)
        for i in range(points.shape[1]):
            scale_factors.append(1.0/np.mean(points[:,i]))
            points[:,i] = points[:,i]*scale_factors[-1]
        mesh = Delaunay(points)
        for i in range(points.shape[1]):
            mesh.points[:,i] = mesh.points[:,i]/scale_factors[i]
        return mesh

class MatplotWindowMixin(object):
    def build_main_window(self, setMethod = None):
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setMinimumWidth(800)
        self.main_widget.setMinimumHeight(600)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.main_widget.setFocus()
        if setMethod:
            setMethod(self.main_widget)

    def init_comms(self):
        self.context = zmq.Context()
        self.uuid = None
        self.data_listener_thread = None

    def toggleAutoClose(self, state):
        global single_window
        single_window = state

    def listen_for_data(self, uuid, num_plots, address="localhost", data_port=7772):
        self.uuid = uuid
        self.data_listener_thread = QtCore.QThread()
        self.Datalistener = DataListener(address, uuid, num_plots, port=data_port)
        self.Datalistener.moveToThread(self.data_listener_thread)
        self.data_listener_thread.started.connect(self.Datalistener.loop)
        self.Datalistener.message.connect(self.data_signal_received)
        self.Datalistener.finished.connect(self.stop_listening)
        QtCore.QTimer.singleShot(0, self.data_listener_thread.start)

    def construct_plots(self, plot_desc):
        self.toolbars = []
        self.canvas_by_name = {}

        # Purge everything in the layout
        for i in reversed(range(self.layout.count())):
            widgetToRemove = self.layout.itemAt( i ).widget()
            self.layout.removeWidget( widgetToRemove )
            widgetToRemove.setParent( None )

        self.tabs  = QtWidgets.QTabWidget(self.main_widget)

        for name, desc in plot_desc.items():
            if desc['plot_type'] == "standard":
                if desc['plot_dims'] == 1:
                    canvas = Canvas1D(self.main_widget, width=5, height=4, dpi=100, plot_mode=desc['plot_mode'])
                if desc['plot_dims'] == 2:
                    canvas = Canvas2D(self.main_widget, width=5, height=4, dpi=100, plot_mode=desc['plot_mode'])
            elif desc['plot_type'] == "manual":
                canvas = CanvasManual(self.main_widget, width=5, height=4, dpi=100, numplots=desc['numplots'])
            elif desc['plot_type'] == "mesh":
                canvas = CanvasMesh(self.main_widget, width=5, height=4, dpi=100, plot_mode=desc['plot_mode'])
            nav    = NavigationToolbar(canvas, self)

            canvas.set_desc(desc)
            self.toolbars.append(nav)
            self.tabs.addTab(canvas, name)
            self.layout.addWidget(nav)

            self.canvas_by_name[name] = canvas

        self.layout.addWidget(self.tabs)
        self.switch_toolbar()
        self.tabs.currentChanged.connect(self.switch_toolbar)

    def data_signal_received(self, message):
        plot_name = message[0]
        uuid      = message[1]
        data      = message[2:]
        if uuid == self.uuid:
            try:
                # If we see a colon, then we must look for a named trace
                if ":" in plot_name:
                    plot_name, trace_name = plot_name.split(":")
                    self.canvas_by_name[plot_name].update_trace(trace_name, *data)
                else:
                    if isinstance(self.canvas_by_name[plot_name], CanvasMesh):
                        self.canvas_by_name[plot_name].update_figure(data[0])
                    else:
                        self.canvas_by_name[plot_name].update_figure(data)
            except Exception as e:
                self.statusBar().showMessage("Exception while plotting {}. Length of data: {}".format(e, len(data)), 1000)

    def switch_toolbar(self):
        if len(self.toolbars) > 0:
            for toolbar in self.toolbars:
                toolbar.setVisible(False)
            self.toolbars[self.tabs.currentIndex()].setVisible(True)

    def stop_listening(self):
        if self.data_listener_thread and self.Datalistener.running:
            # update status bar if possible
            try:
                self.statusBar().showMessage("Disconnecting from server.", 10000)
            except:
                pass
            self.Datalistener.running = False
            self.data_listener_thread.quit()
            self.data_listener_thread.wait()

    def closeEvent(self, event):
        self._quit()


class MatplotClientSubWindow(MatplotWindowMixin,QtWidgets.QMdiSubWindow):
    def __init__(self):
        global single_window
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Auspex Plotting")

        self.build_main_window(self.setWidget)
        self.init_comms()

    def _quit(self):
        self.stop_listening()
        self.close()


class MatplotClientWindow(MatplotWindowMixin, QtWidgets.QMainWindow):
    def __init__(self):
        global single_window
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Auspex Plotting")

        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu.addAction('&Quit', self._quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction('&Close All', close_all_plotters,
                                 QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_W)

        self.recent = self.file_menu.addMenu("Open Recent")

        self.settings_menu = self.menuBar().addMenu('&Settings')
        auto_close = QtWidgets.QAction('Auto Close Plots', self, checkable=True)
        auto_close.setChecked(single_window)
        self.settings_menu.addAction(auto_close)

        auto_close.triggered.connect(self.toggleAutoClose)

        self.build_main_window(self.setCentralWidget)
        self.init_comms()

    def toggleAutoClose(self, state):
        global single_window
        single_window = state

    def _quit(self):
        self.stop_listening()
        plotters = [w for w in QtWidgets.QApplication.topLevelWidgets() if isinstance(w, MatplotClientWindow)]
        if len(plotters) <= 1:
            # This is the last plotter window:
            wait_window.show()
        self.close()


def new_plotter_window(message):
    uuid, desc = message
    desc = json.loads(desc)

    pw = MatplotClientWindow()
    pw.setWindowTitle("%s" % progname)
    pw.show()
    pw.setWindowState(pw.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    pw.activateWindow()
    pw.construct_plots(desc)
    pw.listen_for_data(uuid, len(desc))

    if single_window and len(plot_windows) > 0:
        for w in plot_windows:
            w.closeEvent(0)
            plot_windows.remove(w)

    logger.info(f"Appending {pw}")
    plot_windows.append(pw)

def new_plotter_window_mdi(message):
    uuid, desc = message
    desc = json.loads(desc)

    pw = MatplotClientSubWindow()
  
    pw.setWindowTitle("%s" % progname)
    pw.construct_plots(desc)
    pw.listen_for_data(uuid, len(desc))

    if single_window:
        for window in main_app_mdi.subWindowList():
            window.close()

    main_app_mdi.addSubWindow(pw)
    pw.show()

def close_all_plotters():
    for w in plot_windows:
        w.closeEvent(0)
        time.sleep(0.01)
        plot_windows.remove(w)
    plotters = [w for w in QtWidgets.QApplication.topLevelWidgets() if isinstance(w, MatplotClientWindow)]
    for w in plotters:
        w.closeEvent(0)
        time.sleep(0.01)
    wait_window.show()

class ListenerMixin:
    def start_listener(self, new_plot_callback):
        # Start listener thread
        self.desc_listener_thread = QtCore.QThread()
        self.Desclistener = DescListener("localhost", 7771  )
        self.Desclistener.moveToThread(self.desc_listener_thread)
        self.desc_listener_thread.started.connect(self.Desclistener.loop)
        self.Desclistener.new_plot.connect(new_plot_callback)
        QtCore.QTimer.singleShot(0, self.desc_listener_thread.start)

    def stop_listening(self, _):
        self.Desclistener.running = False
        self.desc_listener_thread.quit()
        self.desc_listener_thread.wait()

    def closeEvent(self, ce):
        if self.desc_listener_thread:
            self.stop_listening(True)
        self.close()
        

class PlotMDI(ListenerMixin,QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        global main_app_mdi
        super(PlotMDI, self).__init__(parent)
        self.mdi = QtWidgets.QMdiArea()
        main_app_mdi = self.mdi
        self.setCentralWidget(self.mdi)
        self.setWindowTitle("Auspex Plots")

        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu.addAction('&Quit', self.close,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction('&Close All', self.close_all_windows,
                                 QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_W)

        self.settings_menu = self.menuBar().addMenu('&Settings')
        auto_close = QtWidgets.QAction('Auto Close Plots', self, checkable=True)
        auto_close.setChecked(single_window)
        self.settings_menu.addAction(auto_close)

        auto_close.triggered.connect(self.toggleAutoClose)

        self.windows_menu = self.menuBar().addMenu('&Windows')
        self.windows_menu.addAction("Cascade", self.mdi.cascadeSubWindows)
        self.windows_menu.addAction("Tiled", self.mdi.tileSubWindows)

        self.start_listener(new_plotter_window_mdi)

    def toggleAutoClose(self, state):
        global single_window
        single_window = state

    def close_all_windows(self):
        for window in self.mdi.subWindowList():
            window.close()


class WaitAndListenWidget(ListenerMixin,QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(WaitAndListenWidget, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        # Create a progress bar and a button and add them to the main layout
        self.progressBar = QtWidgets.QProgressBar(self)
        self.progressBar.setRange(0,0)
        self.progressBar.setValue(0)
        layout.addWidget(QtWidgets.QLabel("Waiting for an available Auspex plot..."))
        layout.addWidget(self.progressBar)
        button = QtWidgets.QPushButton("Quit", self)
        layout.addWidget(button)
        button.clicked.connect(self.closeEvent)

        self.start_listener(new_plotter_window)
        # Start listener thread
        self.Desclistener.new_plot.connect(self.done_waiting)

    def done_waiting(self, thing=None):
        self.hide()

if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)

    # Setup icon
    png_path = os.path.join(os.path.dirname(__file__), "../src/auspex/assets/plotter_icon.png")
    qApp.setWindowIcon(QIcon(png_path))

    # Convince windows that this is a separate application to get the task bar icon working
    # https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
    if (os.name == 'nt'):
        myappid = u'BBN.auspex.auspex-plot-client.0001' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    if '--mdi' in sys.argv:
        wait_window = PlotMDI()
    else:
        wait_window = WaitAndListenWidget()
    wait_window.show()
    sys.exit(qApp.exec_())
