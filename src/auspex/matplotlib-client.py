
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

from scipy.spatial import Delaunay

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

import numpy as np
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

import zmq

class DataListener(QtCore.QObject):

    message = QtCore.pyqtSignal(tuple)
    finished = QtCore.pyqtSignal(bool)

    def __init__(self, host, port=7772):
        QtCore.QObject.__init__(self)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://{}:{}".format(host, port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "data")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "done")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.running = True

    def loop(self):
        while self.running:
            evts = dict(self.poller.poll(50))
            if self.socket in evts and evts[self.socket] == zmq.POLLIN:
                msg = self.socket.recv_multipart()
                msg_type = msg[0].decode()
                name     = msg[1].decode()
                if msg_type == "done":
                    self.finished.emit(True)
                elif msg_type == "data":
                    result = [name]
                    # How many pairs of metadata and data are there?
                    num_arrays = int((len(msg) - 2)/2)
                    for i in range(num_arrays):
                        md, data = msg[2+2*i:4+2*i]
                        md = json.loads(md.decode())
                        A = np.frombuffer(data, dtype=md['dtype'])
                        result.append(A)
                        self.message.emit(tuple(result))
        self.socket.close()

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
            self.plots.append(plt)

    def update_figure(self, x_data, y_data):
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
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axis = self.fig.add_subplot(111)
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
        self.traces[trace_name].set_xdata(x_data)
        self.traces[trace_name].set_ydata(y_data)
        self.axis.relim()
        self.axis.autoscale_view()
        self.draw()
        self.flush_events()

    def set_desc(self, desc):
        if 'x_label' in desc.keys():
            self.axis.set_xlabel(desc['x_label'])
        if 'y_label' in desc.keys():
            self.axis.set_ylabel(desc['y_label'])
        for trace in desc['traces']:
            self.traces[trace['name']], = self.axis.plot([], **trace['matplotlib_kwargs'])
        self.fig.tight_layout()

class Canvas2D(MplCanvas):
    def compute_initial_figure(self):
        for ax in self.axes:
            plt = ax.imshow(np.zeros((10,10)))
            self.plots.append(plt)

    def update_figure(self, x_data, y_data, im_data):
        im_data = im_data.reshape((self.xlen, self.ylen), order='c').T
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

class MatplotClientWindow(QtWidgets.QMainWindow):
    def __init__(self, hostname=None):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Auspex Plotting")

        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction('&Open', self.open_connection_dialog,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Open Localhost', lambda: self.open_connection("localhost"),
                                 QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.recent = self.file_menu.addMenu("Open Recent")

        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setMinimumWidth(800)
        self.main_widget.setMinimumHeight(600)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.context = zmq.Context()

        self.listener_thread = None

        if hostname:
            self.open_connection(hostname)

    def open_connection(self, address):
        port = 7771
        self.statusBar().showMessage("Open session to {}:{}".format(address, port), 2000)
        socket = self.context.socket(zmq.DEALER)
        socket.identity = "Matplotlib_Qt_Client".encode()
        socket.connect("tcp://{}:{}".format(address, port))
        socket.send(b"WHATSUP")

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLOUT)
        time.sleep(0.1)

        evts = dict(poller.poll(100))
        if socket in evts:
            try:
                reply, desc = [e.decode() for e in socket.recv_multipart(flags=zmq.NOBLOCK)]
                desc = json.loads(desc)
                self.statusBar().showMessage("Connection established. Pulling plot information.", 2000)
            except:
                self.statusBar().showMessage("Could not connect to server.", 2000)
                return
        else:
            self.statusBar().showMessage("Server did not respond.", 2000)

        socket.close()
        self.construct_plots(desc)

        # Actual data listener
        if self.listener_thread:
            self.Datalistener.running = False
            self.listener_thread.quit()
            self.listener_thread.wait()

        self.listener_thread = QtCore.QThread()
        self.Datalistener = DataListener(address)
        self.Datalistener.moveToThread(self.listener_thread)
        self.listener_thread.started.connect(self.Datalistener.loop)
        self.Datalistener.message.connect(self.data_signal_received)
        self.Datalistener.finished.connect(self.stop_listening)
        
        QtCore.QTimer.singleShot(0, self.listener_thread.start)

    def open_connection_dialog(self):
        address, ok = QtWidgets.QInputDialog.getText(self, 'Open Connection', 
            'Resource Name:')
        if ok:
            self.open_connection(address)
    
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
                canvas = CanvasManual(self.main_widget, width=5, height=4, dpi=100)
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
        data      = message[1:]
        try:
            # If we see a colon, then we must look for a named trace
            if ":" in plot_name:
                plot_name, trace_name = plot_name.split(":")
                self.canvas_by_name[plot_name].update_trace(trace_name, *data)
            else:
                if isinstance(self.canvas_by_name[plot_name], CanvasMesh):
                    self.canvas_by_name[plot_name].update_figure(data[0])
                else:
                    self.canvas_by_name[plot_name].update_figure(*data)
        except:
            pass

    def switch_toolbar(self):
        for toolbar in self.toolbars:
            toolbar.setVisible(False)
        self.toolbars[self.tabs.currentIndex()].setVisible(True)

    def fileQuit(self):
        self.close()

    def stop_listening(self, _):
        self.statusBar().showMessage("Disconnecting from server.", 10000)
        self.Datalistener.running = False
        self.listener_thread.quit()
        self.listener_thread.wait()

    def closeEvent(self, ce):
        if self.listener_thread:
            self.stop_listening(True)
        self.fileQuit()

if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 1:
        aw = MatplotClientWindow(hostname=sys.argv[1])
    else:
        aw = MatplotClientWindow()
    aw.setWindowTitle("%s" % progname)
    aw.show()
    sys.exit(qApp.exec_())
