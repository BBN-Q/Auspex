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
import os
import json
import time

from auspex.log import logger
from auspex.exp_factory import QubitExpFactory, QubitExperiment
from auspex.parameter import FloatParameter
from auspex.filters.plot import ManualPlotter
from auspex.filters.singleshot import SingleShotMeasurement
from auspex.analysis.fits import *
from auspex.analysis.helpers import normalize_data
import auspex.config

class SingleShotFidelityExperiment(QubitExperiment):
    """Experiment to measure single-shot measurement fidelity of a qubit."""

    def __init__(self, qubit_names, num_shots=10000, expname=None, meta_file=None, save_data=False, optimize=False, set_threshold = True, stream_type = 'Raw', **kwargs):
        """Create a single shot fidelity measurement experiment. Assumes that there is a single shot measurement
        filter in the filter pipeline.
        Arguments:
            qubit_names: Names of the qubits to be measured. (str)
            num_shots: Total number of 0 and 1 measurements used to reconstruct fidelity histograms (int)
            expname: Experiment name for data saving.
            meta_file: Meta file for defining custom single-shot fidelity experiment.
            save_data: If true, will save the raw or demodulated data.
            optimize: if true, will set the swept parameters to their optimum values"""

        super(SingleShotFidelityExperiment, self).__init__()
        self.qubit_names = qubit_names if isinstance(qubit_names, list) else [qubit_names]
        self.qubit     = [QubitFactory(qubit_name) for qubit_name in qubit_names] if isinstance(qubit_names, list) else QubitFactory(qubit_names)
        # make a copy of the settings to restore default
        self.saved_settings = config.load_meas_file(config.meas_file)
        self.settings = deepcopy(self.saved_settings)
        self.save_data = save_data
        self.calibration = True
        self.optimize = optimize
        self.set_threshold = True
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

        self.pdf_data = None

    def run_sweeps(self):
        #For now, only update histograms if we don't have a parameter sweep.
        if not self.sweeper.axes:
            self.init_plots()
            self.add_manual_plotter(self.re_plot)
            self.add_manual_plotter(self.im_plot)
        else:
            if any([x.save_kernel.value for x in self.filters.values() if type(x) is SingleShotMeasurement]):
                logger.warning("Kernel saving is not supported if you have parameter sweeps!")

        super(SingleShotFidelityExperiment, self).run_sweeps()

        if not self.sweeper.axes:
            self._update_histogram_plots()

        if hasattr(self, 'extra_plot_server'):
            try:
                self.extra_plot_server.stop()
            except:
                pass

        if self.set_threshold:
            self.update_threshold()

        if self.sweeper.axes and self.optimize:
            #select the buffers/writers whose sources are singleshot filters
            fid_buffers = [buff for buff in self.buffers if self.settings['filters'][buff.name]['source'].strip().split()[1] == 'fidelity']
            if not fid_buffers:
                raise NameError("Please connect a buffer to the single-shot filter output in order to optimize fidelity.")
            #set sweep parameters to the values that maximize fidelity. Then update the saved_settings with the new values
            for buff in fid_buffers:
                dataset, descriptor = buff.get_data(), buff.get_descriptor()
                opt_ind = np.argmax(dataset['Data'])
                for k, axis in enumerate(self.sweeper.axes):
                    instr_tree = axis.parameter.instr_tree
                    param_key = self.saved_settings['instruments']
                    for key in instr_tree[:-1]:
                        # go through the tree
                        param_key = param_key[key]
                    opt_value = float(dataset[axis.name][opt_ind])
                    # special case to set APS ch12 amplitudes
                    if instr_tree[-1] == 'amplitude' and instr_tree[-2] in self.saved_settings['instruments'].keys():
                        param_key['tx_channels']['12']['1']['amplitude'] = round(float(opt_value), 5)
                        param_key['tx_channels']['12']['2']['amplitude'] = round(float(opt_value), 5)
                    else:
                        param_key[instr_tree[-1]] = opt_value
                    logger.info("Set{} to {}.".format(" ".join(str(x) for x in instr_tree),opt_value ))

        if self.set_threshold or (self.sweeper.axes and self.optimize): # update settings if something was calibrated
            config.dump_meas_file(self.saved_settings, config.meas_file)

    def _update_histogram_plots(self):
        self.get_results()
        self.re_plot.set_data("Ground", self.pdf_data["I Bins"], self.pdf_data["Ground I PDF"])
        self.re_plot.set_data("Ground Gaussian Fit", self.pdf_data["I Bins"], self.pdf_data["Ground I Gaussian PDF"])
        self.re_plot.set_data("Excited", self.pdf_data["I Bins"], self.pdf_data["Excited I PDF"])
        self.re_plot.set_data("Excited Gaussian Fit", self.pdf_data["I Bins"], self.pdf_data["Excited I Gaussian PDF"])
        self.im_plot.set_data("Ground", self.pdf_data["Q Bins"], self.pdf_data["Ground Q PDF"])
        self.im_plot.set_data("Ground Gaussian Fit", self.pdf_data["Q Bins"], self.pdf_data["Ground Q Gaussian PDF"])
        self.im_plot.set_data("Excited", self.pdf_data["Q Bins"], self.pdf_data["Excited Q PDF"])
        self.im_plot.set_data("Excited Gaussian Fit", self.pdf_data["Q Bins"], self.pdf_data["Excited Q Gaussian PDF"])

        time.sleep(0.2)
        try:
            self.extra_plot_server.shutdown()
            self.extra_plot_desc_server.shutdown()
        except:
            logger.info("Could not stop extra plot server gracefully...")

    def update_threshold(self):
        self.get_results()
        if 'I Threshold' in self.pdf_data:
            self.saved_settings['filters'][self.qubit_to_stream_sel[self.qubit.label]]['threshold'] = round(float(self.pdf_data['I Threshold']), 6)
        else:
            logger.warning(f"Tried to save filter {self.name} threshold but was not calculated!")

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
            logger.info("Set digitizer {} round robins to 1 for single shot experiment.".format(d))
            self.settings['instruments'][d]['nbr_round_robins'] = 1
        # disable averagers
        for _, f in self.settings['filters'].items():
            if f['type'] == 'Averager':
                f['enabled'] = False

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
        if self.pdf_data is None:
            ssf = self.find_single_shot_filter()
            if len(ssf) > 1: # not implemented
                try:
                    self.pdf_data =  {x.name: x.pdf_data_queue.get(False) for x in ssf}
                except AttributeError:
                    raise AttributeError("Could not find single shot PDF data in results. Did you run the sweeps?")
            else:
                try:
                    self.pdf_data = ssf[0].pdf_data_queue.get(False)
                except AttributeError:
                    raise AttributeError("Could not find single shot PDF data in results. Did you run the sweeps?")
