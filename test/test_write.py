# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest

import tempfile
import os, shutil
import glob
import time
import numpy as np

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters.debug import Print
from auspex.filters.io import WriteToFile
from auspex.log import logger
from auspex.data_format import AuspexDataContainer

class SweptTestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()
    current = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    # Complex?
    is_complex = False

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.data_name='voltage'
        if self.is_complex:
            descrip.dtype = np.complex128
        descrip.add_axis(DataAxis("samples", list(range(self.samples))))
        self.voltage.set_descriptor(descrip)

    def __repr__(self):
        return "<SweptTestExperiment>"

    def run(self):
        time_step = 0.1
        time.sleep(0.001)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.samples) + 0.1*np.random.random(self.samples)
        self.time_val += time_step
        if self.is_complex:
            self.voltage.push(np.array(data_row + 0.5j*data_row, dtype=np.complex128))
            self.current.push(np.array(np.sin(2*np.pi*self.time_val) + 0.1*np.random.random(1) + 0.5j*np.random.random(1), dtype=np.complex128))
        else:
            self.voltage.push(data_row)
            self.current.push(np.sin(2*np.pi*self.time_val) + 0.1*np.random.random(1))

class SweptTestExperimentMetadata(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()
    current = OutputConnector()

    # Constants
    samples  = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        self.voltage.add_axis(DataAxis("samples", np.array([0,1,2,np.nan,np.nan]), metadata=["data", "data", "data", "0", "1"]))

    def __repr__(self):
        return "<SweptTestExperimentMetadata>"

    def run(self):
        time_step = 0.1
        time.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.samples) + 0.1*np.random.random(self.samples)
        self.time_val += time_step
        self.voltage.push(data_row)
        self.current.push(np.sin(2*np.pi*self.time_val) + 0.1*np.random.random(1))

class SweptTestExperiment2(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")
    dur   = FloatParameter(default=5,unit="ns")

    # DataStreams
    voltage = OutputConnector()
    current = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: logger.debug("Field got value " + str(x)))
        self.freq.assign_method(lambda x: logger.debug("Freq got value " + str(x)))
        self.dur.assign_method(lambda x: logger.debug("Duration got value " + str(x)))

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        self.voltage.add_axis(DataAxis("samples", list(range(self.samples))))
        self.current.add_axis(DataAxis("samples", list(range(self.samples))))

    def __repr__(self):
        return "<SweptTestExperiment2>"

    def run(self):
        time_step = 0.1
        time.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.samples) + 0.1*np.random.random(self.samples)
        self.time_val += time_step
        self.voltage.push(data_row)
        self.current.push(-0.1*data_row)

class WriteTestCase(unittest.TestCase):

    def test_write(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            wr = WriteToFile(tmpdirname+"/test_write.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            exp.add_sweep(exp.field, np.linspace(0,100.0,4))
            exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
            exp.run_sweeps()
            self.assertTrue(os.path.exists(tmpdirname+"/test_write-0000.auspex"))
            container = AuspexDataContainer(tmpdirname+"/test_write-0000.auspex")
            data, desc = container.open_dataset('main', 'data')

            self.assertTrue(0.0 not in data)
            self.assertTrue(np.all(desc['field'] == np.linspace(0,100.0,4)))
            self.assertTrue(np.all(desc['freq'] == np.linspace(0,10.0,3)))
            self.assertTrue(np.all(desc['samples'] == np.linspace(0,4,5)))
            self.assertTrue(desc.axis('freq').unit == "Hz")

    def test_filename_increment(self):
        with tempfile.TemporaryDirectory() as tmpdirname:

            exp = SweptTestExperiment()
            wr = WriteToFile(tmpdirname+"/test_write.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            exp.add_sweep(exp.field, np.linspace(0,100.0,4))
            exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
            exp.run_sweeps()

            exp = SweptTestExperiment()
            wr = WriteToFile(tmpdirname+"/test_write.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            exp.add_sweep(exp.field, np.linspace(0,100.0,4))
            exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
            exp.run_sweeps()

            self.assertTrue(os.path.exists(tmpdirname+"/test_write-0000.auspex"))
            self.assertTrue(os.path.exists(tmpdirname+"/test_write-0001.auspex"))

    def test_write_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperimentMetadata()
            wr = WriteToFile(tmpdirname+"/test_write_metadata.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            exp.add_sweep(exp.field, np.linspace(0,100.0,4))
            exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
            exp.run_sweeps()

            self.assertTrue(os.path.exists(tmpdirname+"/test_write_metadata-0000.auspex"))
            container = AuspexDataContainer(tmpdirname+"/test_write_metadata-0000.auspex")
            data, desc = container.open_dataset('main', 'data')
            
            self.assertTrue(0.0 not in data)
            self.assertTrue(np.all(desc['field'] == np.linspace(0,100.0,4)))
            self.assertTrue(np.all(desc['freq'] == np.linspace(0,10.0,3)))
            self.assertTrue(np.all(desc['samples'][:3] == [0.0,1.0,2.0]))
            self.assertTrue(np.all(np.isnan(desc['samples'][3:])))
            self.assertTrue(np.all(desc.axis('samples').metadata == ["data", "data", "data", "0", "1"]))

    @unittest.skip("Need to update tests for new auspex data writer")
    def test_write_metadata_unstructured(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperimentMetadata()
            wr = WriteToFile(tmpdirname+"/test_write_metadata_unstructured.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)
            coords = [[ 0, 0.1],
                      [10, 4.0],
                      [15, 2.5],
                      [40, 4.4],
                      [50, 2.5],
                      [60, 1.4],
                      [65, 3.6],
                      [66, 3.5],
                      [67, 3.6],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan]]
            md = ["data"]*9 + ["a","b","c"]
            exp.add_sweep([exp.field, exp.freq], coords, metadata=md)
            exp.run_sweeps()
            self.assertTrue(os.path.exists(tmpdirname+"/test_write_metadata_unstructured-0000.auspex"))
            with h5py.File(tmpdirname+"/test_write_metadata_unstructured-0000.auspex", 'r') as f:
                self.assertTrue(0.0 not in f['main/data/voltage'])
                self.assertTrue(np.sum(np.isnan(f['main/data/field'])) == 3*5 )
                self.assertTrue(np.sum(np.isnan(f['main/data/freq'])) == 3*5 )
                self.assertTrue(np.sum(np.isnan(f['main/data/samples'])) == 3*4*2 )

                md_enum = f['main/field+freq_metadata_enum'][:]
                md = f['main/data/field+freq_metadata'][:]
                md = md_enum[md]

                self.assertTrue(np.sum(md == b'a') == 5)
                self.assertTrue(np.sum(md == b'b') == 5)
                self.assertTrue(np.sum(md == b'c') == 5)
                self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
                self.assertTrue(f['main/data'].attrs['time_val'] == 0)
                self.assertTrue(f['main/data'].attrs['unit_freq'] == "Hz")

    @unittest.skip("need to add metadata to adaptive sweeps")
    def test_write_metadata_unstructured_adaptive(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperimentMetadata()
            wr = WriteToFile(tmpdirname+"/test_write_metadata_unstructured_adaptive.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)
            coords = [[ 0, 0.1],
                      [10, 4.0],
                      [15, 2.5],
                      [40, 4.4],
                      [50, 2.5],
                      [60, 1.4],
                      [65, 3.6],
                      [66, 3.5],
                      [67, 3.6]]
            md = ["data"]*9

            def rf(sweep_axis, exp):
                logger.debug("Running refinement function.")
                if sweep_axis.num_points() >= 12:
                    return False
                # sweep_axis.set_metadata(np.append(sweep_axis.metadata_enum[sweep_axis.metadata],["a", "b", "c"]))
                # sweep_axis.add_points([
                #       [np.nan, np.nan],
                #       [np.nan, np.nan],
                      # [np.nan, np.nan]])
                logger.debug("Sweep points now: {}.".format(sweep_axis.points))
                return [[np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan]]

            exp.add_sweep([exp.field, exp.freq], coords, metadata=md, refine_func=rf)
            exp.run_sweeps()
            self.assertTrue(os.path.exists(tmpdirname+"/test_write_metadata_unstructured_adaptive-0000.auspex"))
            with h5py.File(tmpdirname+"/test_write_metadata_unstructured_adaptive-0000.auspex", 'r') as f:
                self.assertTrue(0.0 not in f['main/data']['voltage'])
                self.assertTrue(np.sum(np.isnan(f['main/data/field'])) == 3*5 )
                self.assertTrue(np.sum(np.isnan(f['main/data/freq'])) == 3*5 )
                self.assertTrue(np.sum(np.isnan(f['main/data/samples'])) == 3*4*2 )
                self.assertTrue("Here the run loop merely spews" in f.attrs['exp_src'])
                self.assertTrue(f['main/data'].attrs['time_val'] == 0)
                self.assertTrue(f['main/data'].attrs['unit_freq'] == "Hz")

    def test_write_samefile(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            wr1 = WriteToFile(tmpdirname+"/test_write_samefile.auspex", groupname="group1")
            wr2 = WriteToFile(tmpdirname+"/test_write_samefile.auspex", groupname="group2")

            edges = [(exp.voltage, wr1.sink), (exp.current, wr2.sink)]
            exp.set_graph(edges)

            exp.add_sweep(exp.field, np.linspace(0,100.0,4))
            exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
            exp.run_sweeps()

            container = AuspexDataContainer(tmpdirname+"/test_write_samefile-0000.auspex")
            data1, desc1 = container.open_dataset('group1', 'data')
            data2, desc2 = container.open_dataset('group2', 'data')
            self.assertTrue(os.path.exists(tmpdirname+"/test_write_samefile-0000.auspex"))
            self.assertTrue(0.0 not in data1)
            self.assertTrue(0.0 not in data2)
            self.assertTrue(np.all(desc1['field'] == np.linspace(0,100.0,4)))
            self.assertTrue(np.all(desc1['freq'] == np.linspace(0,10.0,3)))
            self.assertTrue(np.all(desc1['samples'] == np.linspace(0,4,5)))
            self.assertTrue(np.all(desc2['field'] == np.linspace(0,100.0,4)))
            self.assertTrue(np.all(desc2['freq'] == np.linspace(0,10.0,3)))
            self.assertTrue(desc1.axis('freq').unit == "Hz")
            self.assertTrue(desc1.axis('freq').unit == "Hz")

    def test_write_complex(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            exp.is_complex = True
            wr = WriteToFile(tmpdirname+"/test_write_complex.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)
            exp.voltage.descriptor.dtype = np.complex128
            exp.update_descriptors()
            exp.add_sweep(exp.field, np.linspace(0,100.0,4))
            exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
            exp.run_sweeps()

            self.assertTrue(os.path.exists(tmpdirname+"/test_write_complex-0000.auspex"))
            container = AuspexDataContainer(tmpdirname+"/test_write_complex-0000.auspex")
            data, desc = container.open_dataset('main', 'data')

            self.assertTrue(0.0 not in data)
            self.assertTrue(np.all(desc['field'] == np.linspace(0,100.0,4)))
            self.assertTrue(np.all(desc['freq'] == np.linspace(0,10.0,3)))
            self.assertTrue(np.all(desc['samples'] == np.linspace(0,4,5)))
            self.assertTrue(desc.axis('freq').unit == "Hz")
            self.assertTrue(data.dtype.type is np.complex128)

    @unittest.skip("Need to update tests for new auspex data writer")
    def test_write_adaptive_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            wr = WriteToFile(tmpdirname+"/test_write_adaptive.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            def rf(sweep_axis, exp):
                num_points = 5
                logger.debug("Running refinement function.")
                if sweep_axis.num_points() >= num_points:
                    return False
                return [sweep_axis.points[-1]*2]

            exp.add_sweep(exp.field, np.linspace(0,100.0,11))
            exp.add_sweep(exp.freq, [1.0, 2.0], refine_func=rf)
            exp.run_sweeps()
            self.assertTrue(os.path.exists(tmpdirname+"/test_write_adaptive-0000.auspex"))

            with h5py.File(tmpdirname+"/test_write_adaptive-0000.auspex", 'r') as f:
                self.assertTrue(len(f['main/data/freq'][:]) == 5*11*5)
                self.assertTrue(f['main/data/freq'][:].sum() == (55*(1+2+4+8+16)))

    @unittest.skip("Need to update tests for new auspex data writer")
    def test_write_unstructured_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            wr = WriteToFile(tmpdirname+"/test_write_unstructured.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            coords = [[ 0, 0.1],
                      [10, 4.0],
                      [15, 2.5],
                      [40, 4.4],
                      [50, 2.5],
                      [60, 1.4],
                      [65, 3.6],
                      [66, 3.5],
                      [67, 3.6],
                      [68, 1.2]]
            exp.add_sweep([exp.field, exp.freq], coords)
            exp.run_sweeps()
            self.assertTrue(os.path.exists(tmpdirname+"/test_write_unstructured-0000.auspex"))

            with h5py.File(tmpdirname+"/test_write_unstructured-0000.auspex", 'r') as f:
                self.assertTrue(len(f['main/data/voltage']) == 5*10)
                self.assertTrue(f[f['main/field+freq'][0]] == f['main/field'])
                self.assertTrue(f[f['main/field+freq'][1]] == f['main/freq'])

            data, desc = load_from_HDF5(tmpdirname+"/test_write_unstructured-0000.auspex", reshape=False)
            self.assertTrue(data['main']['field'][-5:].sum() == 5*68)

    @unittest.skip("Need to update tests for new auspex data writer")
    def test_write_adaptive_unstructured_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exp = SweptTestExperiment()
            wr = WriteToFile(tmpdirname+"/test_write_adaptive_unstructured.auspex")

            edges = [(exp.voltage, wr.sink)]
            exp.set_graph(edges)

            coords = [[ 0, 0.1],
                      [10, 4.0],
                      [15, 2.5],
                      [40, 4.4],
                      [50, 2.5],
                      [60, 1.4],
                      [65, 3.6],
                      [66, 3.5],
                      [67, 3.6],
                      [68, 1.2]]

            def rf(sweep_axis, exp):
                logger.debug("Running refinement function.")
                if sweep_axis.num_points() >= 30:
                    return False

                first_points = np.array(sweep_axis.points[-10:])
                new_points   = first_points.copy()
                new_points[:,0] += 100
                new_points[:,1] += 10

                logger.debug("Sweep points now: {}.".format(sweep_axis.points))
                return new_points

            exp.add_sweep([exp.field, exp.freq], coords, refine_func=rf)
            exp.run_sweeps()
            self.assertTrue(os.path.exists(tmpdirname+"/test_write_adaptive_unstructured-0000.auspex"))
            data, desc = load_from_HDF5(tmpdirname+"/test_write_adaptive_unstructured-0000.auspex", reshape=False)
            self.assertTrue(len(data['main']['field'])==10*5*3)

if __name__ == '__main__':
    unittest.main()
