# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

try:
    from QGL import *
    from QGL import config as QGLconfig
except:
    print("Could not load QGL.")
# from QGL.BasicSequences.helpers import create_cal_seqs, time_descriptor, cal_descriptor
import auspex.config as config
from copy import deepcopy
import os, sys
import json
import time
import networkx as nx
import bbndb
import queue

from auspex.log import logger
from auspex.error import CalibrationError, PipelineError
from .qubit_exp import QubitExperiment
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.filters.singleshot import SingleShotMeasurement
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data
import auspex.config

class SingleShotFidelityExperiment(QubitExperiment):
    """Experiment to measure single-shot measurement fidelity of a qubit."""

    def __init__(self, qubit, output_nodes=None, meta_file=None, **kwargs):

        self.pdf_data = []
        self.qubit = qubit

        if meta_file:
            self.meta_file = meta_file
        else:
            self.meta_file = self._single_shot_sequence(self.qubit)

        super(SingleShotFidelityExperiment, self).__init__(self.meta_file, **kwargs)

    def guess_output_nodes(self, graph):
        output_nodes = []
        for qubit in self.qubits:
            ds = nx.descendants(graph, self.qubit_proxies[qubit.label])
            outputs = [d for d in ds if isinstance(d, (bbndb.auspex.Write, bbndb.auspex.Buffer))]
            if len(outputs) != 1:
                raise PipelineError(f"More than one output node found for {qubit}, please explicitly define output node using output_nodes argument.")
            output_nodes.append(outputs[0])
        return output_nodes

    def _single_shot_sequence(self, qubit):
        seqs = create_cal_seqs((qubit,), 1)
        return compile_to_hardware(seqs, 'SingleShot/SingleShot')

    def init_plots(self):
        self.re_plot = ManualPlotter("Fidelity - Real", x_label='Bins', y_label='Real Quadrature')
        self.im_plot = ManualPlotter("Fidelity - Imag", x_label='Bins', y_label='Imag Quadrature')
        self.re_plot.add_trace("Excited", matplotlib_kwargs={'color': 'r', 'linestyle': '-', 'linewidth': 2})
        self.re_plot.add_trace("Excited Gaussian Fit", matplotlib_kwargs={'color': 'r', 'linestyle': '--', 'linewidth': 2})
        self.re_plot.add_trace("Ground", matplotlib_kwargs={'color': 'b', 'linestyle': '-', 'linewidth': 2})
        self.re_plot.add_trace("Ground Gaussian Fit", matplotlib_kwargs={'color': 'b', 'linestyle': '--', 'linewidth': 2})
        self.im_plot.add_trace("Excited", matplotlib_kwargs={'color': 'r', 'linestyle': '-', 'linewidth': 2})
        self.im_plot.add_trace("Excited Gaussian Fit", matplotlib_kwargs={'color': 'r', 'linestyle': '--', 'linewidth': 2})
        self.im_plot.add_trace("Ground", matplotlib_kwargs={'color': 'b', 'linestyle': '-', 'linewidth': 2})
        self.im_plot.add_trace("Ground Gaussian Fit", matplotlib_kwargs={'color': 'b', 'linestyle': '--', 'linewidth': 2})
        self.add_manual_plotter(self.re_plot)
        self.add_manual_plotter(self.im_plot)

    def _update_histogram_plots(self):
        self.re_plot["Ground"] = (self.pdf_data[-1]["I Bins"], self.pdf_data[-1]["Ground I PDF"])
        self.re_plot["Ground Gaussian Fit"] = (self.pdf_data[-1]["I Bins"], self.pdf_data[-1]["Ground I Gaussian PDF"])
        self.re_plot["Excited"] = (self.pdf_data[-1]["I Bins"], self.pdf_data[-1]["Excited I PDF"])
        self.re_plot["Excited Gaussian Fit"] = (self.pdf_data[-1]["I Bins"], self.pdf_data[-1]["Excited I Gaussian PDF"])
        self.im_plot["Ground"] = (self.pdf_data[-1]["Q Bins"], self.pdf_data[-1]["Ground Q PDF"])
        self.im_plot["Ground Gaussian Fit"] = (self.pdf_data[-1]["Q Bins"], self.pdf_data[-1]["Ground Q Gaussian PDF"])
        self.im_plot["Excited"] = (self.pdf_data[-1]["Q Bins"], self.pdf_data[-1]["Excited Q PDF"])
        self.im_plot["Excited Gaussian Fit"] = (self.pdf_data[-1]["Q Bins"], self.pdf_data[-1]["Excited Q Gaussian PDF"])

    def run_sweeps(self):
        if not self.sweeper.axes:
            self.init_plots()
            self.start_manual_plotters()
        else:
            for f in self.filters:
                if isinstance(f, SingleShotMeasurement):
                    f.save_kernel.value = False
        super(SingleShotFidelityExperiment, self).run_sweeps()
        self.get_results()
        if not self.sweeper.axes:
            self._update_histogram_plots()
            self.stop_manual_plotters()

    def find_single_shot_filter(self):
        """Make sure there is one single shot measurement filter in the pipeline."""
        ssf = [x for x in self.filters if type(x) is SingleShotMeasurement]
        if len(ssf) > 1:
            raise NotImplementedError("Single shot fidelity for more than one qubit is not yet implemented.")
        elif len(ssf) == 0:
            raise PipelineError("There do not appear to be any single-shot measurements in your filter pipeline. Please add one!")
        return ssf

    def get_fidelity(self):
        if self.pdf_data is None:
            raise CalibrationError("Could not find single shot PDF data in results. Did you run the sweeps?")
        return [p["Max I Fidelity"] for p in self.pdf_data]

    def get_threshold(self):
        if self.pdf_data is None:
            raise CalibrationError("Could not find single shot PDF data in results. Did you run the sweeps?")
        return [p["I Threshold"] for p in self.pdf_data]

    def get_results(self):
        """Get the PDF and fidelity numbers from the filters. Returns a dictionary of PDF data with the
        filter names as keys."""
        if len(self.pdf_data) == 0:
            ssf = self.find_single_shot_filter()
            while True:
                try:
                    self.pdf_data.append(ssf[0].pdf_data_queue.get(False))
                except queue.Empty as e:
                    break
            if len(self.pdf_data) == 0:
                raise CalibrationError("Could not find single shot PDF data in results. Did you run the sweeps?")
