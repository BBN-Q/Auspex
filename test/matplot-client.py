
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

def recv_array(socket, flags=0, copy=False, track=False):
    """recv a numpy array"""
    session = socket.recv_string()
    md      = socket.recv_json(flags=flags)
    msg     = socket.recv(flags=flags, copy=copy, track=track)
    
    A = np.frombuffer(msg, dtype=md['dtype'])
    return session, A.reshape(md['shape'])

class ZeroMQ_Listener(QtCore.QObject):

    message = QtCore.pyqtSignal(tuple)
    
    def __init__(self, session_name, port=5556):
        QtCore.QObject.__init__(self)
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect (f"tcp://localhost:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, session_name)
        self.running = True
    
    def loop(self):
        while self.running:
            mesg = recv_array(self.socket)
            self.message.emit(mesg)

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

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Auspex Plotting")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtWidgets.QWidget(self)

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.tabs   = QtWidgets.QTabWidget(self.main_widget)
        self.toolbars = []

        self.canvas_by_name = {}

        for i in range(3):
            sc = StaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
            # dc = DynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
            snav = NavigationToolbar(sc, self)
            # dnav = NavigationToolbar(dc, self)
            self.toolbars.extend([snav]) #, dnav])
            self.tabs.addTab(sc, f"Plot{i}")
            # self.tabs.addTab(dc, "Dynamic")
            self.layout.addWidget(snav)
            # self.layout.addWidget(dnav)
            self.canvas_by_name[f"Plot{i}"] = sc
        self.layout.addWidget(self.tabs)
        self.switch_toolbar()
        self.tabs.currentChanged.connect(self.switch_toolbar)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.thread = QtCore.QThread()
        self.zeromq_listener = ZeroMQ_Listener("buq123")
        self.zeromq_listener.moveToThread(self.thread)
        
        self.thread.started.connect(self.zeromq_listener.loop)
        self.zeromq_listener.message.connect(self.signal_received)
        
        QtCore.QTimer.singleShot(0, self.thread.start)

        # self.statusBar().showMessage("All hail matplotlib!", 2000)
    def signal_received(self, stuff): #message):
        message, data = stuff
        session, plot_name, subplot = message.split()
        self.statusBar().showMessage(message, 500)
        self.canvas_by_name[plot_name].update_figure(data)

    def switch_toolbar(self):
        for toolbar in self.toolbars:
            toolbar.setVisible(False)
        self.toolbars[self.tabs.currentIndex()].setVisible(True)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.zeromq_listener.running = False
        self.thread.quit()
        self.thread.wait()
        self.fileQuit()

qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())
