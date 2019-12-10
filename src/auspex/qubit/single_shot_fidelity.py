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
import time, datetime
import networkx as nx
import bbndb
import queue

from auspex.log import logger
# from .qubit_exp_factory import QubitExpFactory
from .qubit_exp import QubitExperiment
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.filters.singleshot import SingleShotMeasurement
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data
import auspex.config

class SingleShotFidelityExperiment(QubitExperiment):
    """Experiment to measure single-shot measurement fidelity of a qubit.

        Args:
            qubit:                          qubit object
            output_nodes (optional):        the output node of the filter pipeline to use for single-shot readout. The default is choses, if single output.
            meta_file (string, optional):   path to the QGL sequence meta_file. Default to standard SingleShot sequence
            optimize (bool, optional):      if True and a qubit_sweep is added, set the parameter corresponding to the maximum measured fidelity at the end of the sweep

    """
    def __init__(self, qubit, sample_name=None, output_nodes=None, meta_file=None, optimize=True, set_threshold = True, **kwargs):

        self.pdf_data = []
        self.qubit = qubit
        self.optimize = optimize
        self.set_threshold = set_threshold

        if meta_file:
            self.meta_file = meta_file
        else:
            self.meta_file = self._single_shot_sequence(self.qubit)

        super(SingleShotFidelityExperiment, self).__init__(self.meta_file, **kwargs)

        if not sample_name:
            sample_name = self.qubit.label
        if not bbndb.get_cl_session():
            raise Exception("Attempting to load Calibrations database, \
                but no database session is open! Have the ChannelLibrary and PipelineManager been created?")
        existing_samples = list(bbndb.get_cl_session().query(bbndb.calibration.Sample).filter_by(name=sample_name).all())
        if len(existing_samples) == 0:
            logger.info("Creating a new sample in the calibration database.")
            self.sample = bbndb.calibration.Sample(name=sample_name)
            bbndb.get_cl_session().add(self.sample)
        elif len(existing_samples) == 1:
            self.sample = existing_samples[0]
        else:
            raise Exception("Multiple samples found in calibration database with the same name! How?")

    def guess_output_nodes(self, graph):
        output_nodes = []
        for qubit in self.qubits:
            ds = nx.descendants(graph, self.qubit_proxies[qubit.label])
            outputs = [d for d in ds if isinstance(d, (bbndb.auspex.Write, bbndb.auspex.Buffer))]
            if len(outputs) != 1:
                raise Exception(f"More than one output node found for {qubit}, please explicitly define output node using output_nodes argument.")
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
            if self.set_threshold:
                self.stream_selectors[0].threshold = self.get_threshold()[0]
            if self.sample:
                c = bbndb.calibration.Calibration(value=self.get_fidelity()[0], sample=self.sample, name="Readout fid.", category="Readout")
                c.date = datetime.datetime.now()
                bbndb.get_cl_session().add(c)
                bbndb.get_cl_session().commit()
        elif self.optimize:
            fidelities = [f['Max I Fidelity'] for f in self.pdf_data]
            opt_ind = np.argmax(fidelities)
            for k, axis in enumerate(self.sweeper.axes):
                set_pair = axis.parameter.set_pair
                opt_value = axis.points[opt_ind]
                if set_pair[1] == 'amplitude' or set_pair[1] == "offset":
                    # special case for APS chans
                    param = [c for c in self.chan_db.channels if c.label == set_pair[0]][0]
                    attr = 'amp_factor' if set_pair[1] == 'amplitude' else 'offset'
                    setattr(param, f'I_channel_{attr}', opt_value)
                    setattr(param, f'Q_channel_{attr}', opt_value)
                else:
                    param = [c for c in self.chan_db.all_instruments() if c.label == set_pair[0]][0]
                    setattr(param, set_pair[1], opt_value)
            logger.info(f'Set {set_pair[0]} {set_pair[1]} to optimum value {opt_value}')
            if self.set_threshold:
                self.stream_selectors[0].threshold = self.get_threshold()[opt_ind]
                logger.info(f'Set threshold to {self.stream_selectors[0].threshold}')

    def find_single_shot_filter(self):
        """Make sure there is one single shot measurement filter in the pipeline."""
        ssf = [x for x in self.filters if type(x) is SingleShotMeasurement]
        if len(ssf) > 1:
            raise NotImplementedError("Single shot fidelity for more than one qubit is not yet implemented.")
        elif len(ssf) == 0:
            raise NameError("There do not appear to be any single-shot measurements in your filter pipeline. Please add one!")
        return ssf

    def get_fidelity(self):
        if not self.pdf_data:
            raise Exception("Could not find single shot PDF data in results. Did you run the sweeps?")
        return [p["Max I Fidelity"] for p in self.pdf_data]

    def get_threshold(self):
        if not self.pdf_data:
            raise Exception("Could not find single shot PDF data in results. Did you run the sweeps?")
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
                raise Exception("Could not find single shot PDF data in results. Did you run the sweeps?")
