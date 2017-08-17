# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from QGL import *
from QGL import config as QGLconfig
# from QGL.BasicSequences.helpers import create_cal_seqs, time_descriptor, cal_descriptor
import auspex.config as config
from copy import copy
import os
import json

from auspex.log import logger
from auspex.exp_factory import QubitExpFactory, QubitExperiment
from auspex.analysis.io import load_from_HDF5
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.filters.singleshot import SingleShotMeasurement
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data

class SingleShotFidelityExperiment(QubitExperiment):
    """Experiment to measure single-shot measurement fidelity of a qubit."""

    def __init__(self, qubit_names, num_shots=10000, expname=None, meta_file=None, save_data=False, stream_type = 'Raw'):
        """Create a single shot fidelity measurement experiment. Assumes that there is a single shot measurement
        filter in the filter pipeline.
        Arguments:
            qubit_names: Names of the qubits to be measured. (str)
            num_shots: Total number of 0 and 1 measurements used to reconstruct fidelity histograms (int)
            expname: Experiment name for data saving.
            meta_file: Meta file for defining custom single-shot fidelity experiment.
            save_data: If true, will save the raw or demodulated data."""

        super(SingleShotFidelityExperiment, self).__init__()
        self.qubit_names = qubit_names if isinstance(qubit_names, list) else [qubit_names]
        self.qubit     = [QubitFactory(qubit_name) for qubit_name in qubit_names] if isinstance(qubit_names, list) else QubitFactory(qubit_names)

        self.settings = config.yaml_load(config.configFile)
        self.save_data = save_data
        self.calibration = True
        self.name = expname
        self.cw_mode = False
        self.repeats = num_shots
        self.ss_stream_type = stream_type

        if meta_file is None:
            meta_file = SingleShot(self.qubit)

        self._squash_round_robins()

        QubitExpFactory.load_meta_info(self, meta_file)
        QubitExpFactory.load_instruments(self)
        QubitExpFactory.load_qubits(self)
        QubitExpFactory.load_filters(self)
        if 'sweeps' in self.settings:
            QubitExpFactory.load_parameter_sweeps(experiment)
        self.ssf = self.find_single_shot_filter()
        self.leave_plot_server_open = True

    def run_sweeps(self):
        #For now, only update histograms if we have a parameter sweep.
        if not self.sweeper.axes:
            self.init_plots()
            self.add_manual_plotter(self.re_plot)
            self.add_manual_plotter(self.im_plot)
        else:
            if any([x.save_kernel.value for x in self.filters.values() if type(x) is SingleShotMeasurement]):
                raise ValueError("Kernel saving is not supported if you have parameter sweeps!")

        super(SingleShotFidelityExperiment, self).run_sweeps()

        if not self.sweeper.axes:
            self._update_histogram_plots()

        self.plot_server.stop()

    def _update_histogram_plots(self):
        pdf_data = self.get_results()
        self.re_plot.set_data("Ground", pdf_data["I Bins"], pdf_data["Ground I PDF"])
        self.re_plot.set_data("Ground Gaussian Fit", pdf_data["I Bins"], pdf_data["Ground I Gaussian PDF"])
        self.re_plot.set_data("Excited", pdf_data["I Bins"], pdf_data["Excited I PDF"])
        self.re_plot.set_data("Excited Gaussian Fit", pdf_data["I Bins"], pdf_data["Excited I Gaussian PDF"])
        self.im_plot.set_data("Ground", pdf_data["Q Bins"], pdf_data["Ground Q PDF"])
        self.im_plot.set_data("Ground Gaussian Fit", pdf_data["Q Bins"], pdf_data["Ground Q Gaussian PDF"])
        self.im_plot.set_data("Excited", pdf_data["Q Bins"], pdf_data["Excited Q PDF"])
        self.im_plot.set_data("Excited Gaussian Fit", pdf_data["Q Bins"], pdf_data["Excited Q Gaussian PDF"])

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

    def _squash_round_robins(self):
        """Make it so that the round robins are set to 1."""
        digitizers =  [_ for _ in self.settings['instruments'].keys() if 'nbr_round_robins' in self.settings['instruments'][_].keys()]
        for d in digitizers:
            logger.info(f"Set digitizer {d} round robins to 1 for single shot experiment.")
            self.settings['instruments'][d]['nbr_round_robins'] = 1

    def find_single_shot_filter(self):
        """Make sure there is one single shot measurement filter in the pipeline."""
        ssf = [x for x in self.filters.values() if type(x) is SingleShotMeasurement]
        if len(ssf) > 1:
            raise NotImplementedError("Single shot fidelity for more than one qubit is not yet implemented.")
        elif len(ssf) == 0:
            raise NameError("There do not appear to be any single-shot measurements in your filter pipeline. Please add one!")
        return ssf

    def get_results(self):
        """Get the PDF and fidelity numbers from the filters. Returns a dictionary of PDF data with the
        filter names as keys."""
        ssf = self.find_single_shot_filter()
        if len(ssf) > 1:
            try:
                return {x.name: x.pdf_data for x in ssf}
            except AttributeError:
                raise AttributeError("Could not find single shot PDF data in results. Did you run the sweeps?")
        else:
            try:
                return ssf[0].pdf_data
            except AttributeError:
                raise AttributeError("Could not find single shot PDF data in results. Did you run the sweeps?")
