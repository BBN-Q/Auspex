
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
import random
import json
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
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.running = True

    def loop(self):
        while self.running:
            evts = dict(self.poller.poll(100))
            if self.socket in evts and evts[self.socket] == zmq.POLLIN:
                msg_type, name, md, data = self.socket.recv_multipart()
                if msg_type.decode() == "done":
                    self.finished.emit(True)
                else:
                    name = name.decode()
                    md   = json.loads(md.decode())
                    A    = np.frombuffer(data, dtype=md['dtype'])
                    self.message.emit((name, A.reshape(md['shape'])))
        self.socket.close()

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class StaticMplCanvas(MplCanvas):
    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.plt, = self.axes.plot(t, s)

    def update_figure(self, data):
        self.plt.set_xdata(np.arange(len(data)))
        self.plt.set_ydata(data)
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()
        self.flush_events()

    def set_desc(self, desc):
        if 'xlabel' in desc.keys():
            self.axes.set_xlabel(desc['xlabel'])
        if 'ylabel' in desc.keys():
            self.axes.set_ylabel(desc['ylabel'])

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
        poller.register(socket)

        evts = dict(poller.poll(2000))
        if socket in evts:
            reply, desc = [e.decode() for e in socket.recv_multipart()]
            desc = json.loads(desc)
            self.statusBar().showMessage("Connection established. Pulling plot information.", 2000)
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
            canvas = StaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
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
        plot_name, data = message
        self.canvas_by_name[plot_name].update_figure(data)

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
