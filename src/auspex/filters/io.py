# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['WriteToHDF5', 'H5Handler', 'DumbFileHandler', 'DataBuffer', 'ProgressBar']

import os
import sys

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    import threading as mp
    from threading import Thread as Process
    from queue import Queue
else:
    import multiprocessing as mp
    from multiprocessing import Process
    from multiprocessing import Queue

import itertools
import h5py
# import h5py_cache as h5c
import contextlib
import queue
import numpy as np
import os.path
import os, psutil
import time
import re
import datetime
import pandas as pd
from shutil import copyfile
import cProfile

from .filter import Filter
from auspex.parameter import Parameter, FilenameParameter, BoolParameter
from auspex.stream import InputConnector, OutputConnector
from auspex.log import logger
import auspex.config as config

from tqdm import tqdm, tqdm_notebook

class DumbFileHandler(Process):
    def __init__(self, filename, queue, return_queue):
        super(DumbFileHandler, self).__init__()
        self.queue = queue
        self.return_queue = return_queue
        self.exit = mp.Event()
        self.filename = filename
        self.filter_name = f"{self.filename}"
        self.p = psutil.Process(os.getpid())
        self.perf_queue = None
        self.beginning = datetime.datetime.now()
        self.processed = 0 

    def shutdown(self):
        logger.info("H5 handler shutting down")
        self.exit.set()

    def process_queue_item(self, args, file):
        if args[0] == "write":
            file.write(args[4].tobytes())
        elif args[0] == "done":
            self.exit.set()
        self.push_resource_usage()

    def run(self):
        with open(self.filename+".bin", 'wb') as file:

            self.last_performance_update = datetime.datetime.now()

            def thing():
                while not self.exit.is_set():
                    msgs = []
                    while not self.exit.is_set():
                        try:
                            msgs.append(self.queue.get(False))
                        except queue.Empty as e:
                            time.sleep(0.002)
                            break
                    for msg in msgs:
                        pass
                        self.process_queue_item(msg, file)
                        
            if config.profile:
                cProfile.runctx('thing()', globals(), locals(), 'prof-%s-%s.prof' % (self.__class__.__name__, self.filter_name))
        print(self.filter_name, "leaving main loop")


    def push_resource_usage(self):
        if self.perf_queue:
            if (datetime.datetime.now() - self.last_performance_update).seconds > 0.5:
                self.perf_queue.put((self.filter_name, datetime.datetime.now()-self.beginning, self.p.cpu_percent(), self.p.memory_info()))
                self.last_performance_update = datetime.datetime.now()

    def __repr__(self):
        return "<{} Process (filename={})>".format(self.__class__.__name__, self.filename)


class H5Handler(Process):
    def __init__(self, filename, queue, return_queue):
        super(H5Handler, self).__init__()
        self.queue = queue
        self.return_queue = return_queue
        self.exit = mp.Event()
        self.done = mp.Event()
        self.filename = filename
        self.filter_name = f"{self.filename}"
        self.p = psutil.Process(os.getpid())
        self.perf_queue = None
        self.beginning = datetime.datetime.now()
        self.dtype = None
        self.write_buf = None
        self.processed = 0

    def shutdown(self):
        logger.info("H5 handler shutting down")
        self.exit.set()

    def process_queue_item(self, args, file):
        if args[0] == "write":
            if self.dtype is None:
                # Sometimes we get a scalar, deal with it by converting to numpy array
                try:
                    len(args[4])
                except:
                    args = (args[0], args[1], args[2], args[3], np.array([args[4]]))
                # Set up datatype and write buffer if we haven't already
                self.dtype = args[4].dtype
                self.write_buf = np.zeros(len(args[4])*1000, dtype=self.dtype)
                # print(self.filename, "New buffer size", len(args[4]))
            file[args[1]][args[2]:args[3]] = args[4]
            self.processed += args[4].nbytes

        elif args[0] == "resize":
            file[args[1]].resize(args[2])
        elif args[0] == "resize_to":
            if args[2] > len(file[args[1]]):
                file[args[1]].resize((args[2],))
        elif args[0] == "set_attr":
            file[args[1]].attrs[args[2]] = args[3]
        elif args[0] == "get_data":
            self.return_queue.put((args[1], file[args[1]][:]))
        elif args[0] == "flush":
            file.flush()
        elif args[0] == "done":
            self.exit.set()
        self.push_resource_usage()

    def concat_msgs(self, msgs):
        # If receiving a bunch of consecutive writes, coalesce them into one
        # print(self.dtype, False not in [m[0]=='write' for m in msgs])
        if (len(msgs) > 2) and self.dtype is not None and (False not in [m[0]=='write' for m in msgs]):
            begs = [m[2] for m in msgs]
            ends = [m[3] for m in msgs]
            # print("*", begs[1:], ends[:-1])
            if begs[1:] == ends[:-1]:
                for m in msgs:
                    # print(f"{self.filename}, self.write_buf[{m[2]-begs[0]}:{m[3]-begs[0]}] = {m[4][0]}...")
                    try:
                        self.write_buf[m[2]-begs[0]:m[3]-begs[0]] = m[4]
                    except:
                        logger.error(f"Write buffer overrun in {self.filename} handler")
                # print(f"Catting {len(msgs)} msgs")
                return [('write', msgs[0][1], begs[0], ends[-1], self.write_buf[:ends[-1]-begs[0]])]
        return msgs

    def run(self):
        with h5py.File(self.filename, "r+", libver="latest") as file:
            self.last_performance_update = datetime.datetime.now()

            def thing():
                while not self.exit.is_set():
                    msgs = []
                    while not self.exit.is_set():
                        try:
                            msgs.append(self.queue.get(False))
                        except queue.Empty as e:
                            time.sleep(0.002)
                            break
                        
                    msgs = self.concat_msgs(msgs)
                    for msg in msgs:
                        self.process_queue_item(msg, file)

            if config.profile:
                cProfile.runctx('thing()', globals(), locals(), 'prof-%s-%s.prof' % (self.__class__.__name__, self.filter_name))
            else:
                thing()
            self.done.set()

    def push_resource_usage(self):
        if self.perf_queue:
            if (datetime.datetime.now() - self.last_performance_update).seconds > 0.5:
                self.perf_queue.put((self.filter_name, datetime.datetime.now()-self.beginning, self.p.cpu_percent(), self.p.memory_info(), self.processed))
                self.last_performance_update = datetime.datetime.now()

    def __repr__(self):
        return "<{} Process (filename={})>".format(self.__class__.__name__, self.filename)

class WriteToHDF5(Filter):
    """Writes data to file."""

    sink = InputConnector()
    filename = FilenameParameter()
    groupname = Parameter(default='main')
    add_date = BoolParameter(default = False)
    save_settings = BoolParameter(default = True)

    def __init__(self, filename=None, groupname=None, add_date=False, save_settings=True, compress=False, store_tuples=True, exp_log=True, **kwargs):
        super(WriteToHDF5, self).__init__(**kwargs)
        self.compress = compress
        if filename:
            self.filename.value = filename
        if groupname:
            self.groupname.value = groupname

        self.file = None
        self.group = None
        self.queue = None # For putting values in files
        self.ret_queue = None # For returning values to files

        self.data_write_paths  = {}
        self.tuple_write_paths = {}

        self.store_tuples = store_tuples
        self.up_to_date = False
        self.sink.max_input_streams = 100
        self.add_date.value = add_date
        self.save_settings.value = save_settings
        self.exp_log = exp_log
        self.quince_parameters = [self.filename, self.groupname, self.add_date, self.save_settings]

        self.last_flush = datetime.datetime.now()

    def final_init(self):
        if not self.filename.value and not self.file:
            raise Exception("File object or filename never supplied to writer.")
        # If self.file is still None, then we need to create
        # the file object. Otherwise, we presume someone has
        # already set it up for us.
        if not self.file:
            self.file = self.new_file()

        streams = self.sink.input_streams
        stream  = streams[0]

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to writer must have matching descriptors.")

        desc       = stream.descriptor
        axes       = desc.axes
        params     = desc.params
        self.axis_names = desc.axis_names(with_metadata=True)

        self.file.attrs['exp_src'] = desc._exp_src
        num_axes = len(axes)

        if desc.is_adaptive() and not self.store_tuples:
            raise Exception("Cannot omit writing tuples with an adaptive sweep... please enabled store_tuples.")

        if self.store_tuples:
            # All of the combinations for the present values of the sweep parameters only
            tuples = desc.expected_tuples(with_metadata=True, as_structured_array=True)
        expected_length = desc.expected_num_points()

        compression = 'gzip' if self.compress else None

        # If desired, create the group in which the dataset and axes will reside

        self.group = self.file.create_group(self.groupname.value)
        self.data_group = self.group.create_group("data")

        # If desired, push experimental metadata into the h5 file
        if self.save_settings.value and 'header' not in self.file.keys(): # only save header once for multiple writers
            self.save_yaml_h5()

        # Create datasets for each stream
        for stream in streams:
            dset = self.data_group.create_dataset(stream.descriptor.data_name, (expected_length,),
                                        dtype=stream.descriptor.dtype,
                                        chunks=True, maxshape=(None,),
                                        compression=compression)
            # print("***", dset.chunks)
            dset.attrs['is_data'] = True
            dset.attrs['store_tuples'] = self.store_tuples
            dset.attrs['name'] = stream.descriptor.data_name
            self.data_write_paths[stream] = "/{}/data/{}".format(self.groupname.value, stream.descriptor.data_name)

        # Write params into attrs
        for k,v in params.items():
            if k not in self.axis_names:
                self.data_group.attrs[k] = v

        # Create a table for the DataStreamDescriptor
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        self.descriptor = self.group.create_dataset("descriptor", (len(axes),), dtype=ref_dtype)
        for k,v in desc.metadata.items():
            self.descriptor.attrs[k] = v

        # Create axis data sets for storing the base axes as well as the
        # full set of tuples. For the former we add
        # references to the descriptor.
        self.tuple_write_paths = {}
        for i, a in enumerate(axes):
            if a.unstructured:
                name = "+".join(a.name)
            else:
                name = a.name

            if a.unstructured:
                # Create another reference table to refer to the constituent axes
                unstruc_ref_dset = self.group.create_dataset(name, (len(a.name),), dtype=ref_dtype)
                unstruc_ref_dset.attrs['unstructured'] = True

                for j, (col_name, col_unit) in enumerate(zip(a.name, a.unit)):
                    # Create table to store the axis value independently for each column
                    unstruc_dset = self.group.create_dataset(col_name, (a.num_points(),), dtype=a.dtype)
                    unstruc_ref_dset[j] = unstruc_dset.ref
                    unstruc_dset[:] = a.points[:,j]
                    unstruc_dset.attrs['unit'] = col_unit
                    unstruc_dset.attrs['name'] = col_name

                    # This stores the values taking during the experiment sweeps
                    if self.store_tuples:
                        dset = self.data_group.create_dataset(col_name, (expected_length,), dtype=a.dtype,
                                                             chunks=True, compression=compression, maxshape=(None,) )
                        dset.attrs['unit'] = col_unit
                        dset.attrs['is_data'] = False
                        dset.attrs['name'] = col_name
                        self.tuple_write_paths[col_name] = "/{}/data/{}".format(self.groupname.value, col_name)

                self.descriptor[i] = self.group[name].ref
            else:
                # This stores the axis values
                self.group.create_dataset(name, (a.num_points(),), dtype=a.dtype, maxshape=(None,) )
                self.group[name].attrs['unstructured'] = False
                self.group[name][:] = a.points
                self.group[name].attrs['unit'] = "None" if a.unit is None else a.unit
                self.group[name].attrs['name'] = a.name
                self.descriptor[i] = self.group[name].ref

                # This stores the values taking during the experiment sweeps
                if self.store_tuples:
                    dset = self.data_group.create_dataset(name, (expected_length,), dtype=a.dtype,
                                                          chunks=True, compression=compression, maxshape=(None,) )
                    dset.attrs['unit'] = "None" if a.unit is None else a.unit
                    dset.attrs['is_data'] = False
                    dset.attrs['name'] = name
                    self.tuple_write_paths[name] = "/{}/data/{}".format(self.groupname.value, name)

            # Give the reader some warning about the usefulness of these axes
            self.group[name].attrs['was_refined'] = False

            if a.metadata is not None:
                # Create the axis table for the metadata
                dset = self.group.create_dataset(name + "_metadata", (a.metadata.size,), dtype=np.uint8, maxshape=(None,) )
                dset[:] = a.metadata
                dset = self.group.create_dataset(name + "_metadata_enum", (a.metadata_enum.size,), dtype='S128', maxshape=(None,) )
                dset[:] = np.asarray(a.metadata_enum, dtype='S128')

                # Associate the metadata with the data axis
                self.group[name].attrs['metadata'] = self.group[name + "_metadata"].ref
                self.group[name].attrs['metadata_enum'] = self.group[name + "_metadata_enum"].ref
                self.group[name].attrs['name'] = name + "_metadata"

                # Create the dataset that stores the individual tuple values
                if self.store_tuples:
                    dset = self.data_group.create_dataset(name + "_metadata" , (expected_length,),
                                                          dtype=np.uint8, maxshape=(None,) )
                    dset.attrs['name'] = name + "_metadata"
                    self.tuple_write_paths[name + "_metadata"] = "/{}/data/{}".format(self.groupname.value, name + "_metadata")


        # Write all the tuples if this isn't adaptive
        if self.store_tuples:
            if not desc.is_adaptive():
                for i, a in enumerate(self.axis_names):
                    # import pdb; pdb.set_trace()
                    self.file[self.tuple_write_paths[a]][:] = tuples[a]

    def new_filename(self):
        filename = self.filename.value
        basename, ext = os.path.splitext(filename)
        if ext == "":
            logger.debug("Filename for writer {} does not have an extension -- using default '.h5'".format(self.name))
            ext = ".h5"

        dirname = os.path.dirname(os.path.abspath(filename))

        if self.add_date.value:
            date     = time.strftime("%y%m%d")
            dirname  = os.path.join(dirname, date)
            basename = os.path.join(dirname, os.path.basename(basename))

        # Set the file number to the maximum in the current folder + 1
        filenums = []
        # import pdb; pdb.set_trace()
        if os.path.exists(dirname):
            for f in os.listdir(dirname):
                if ext in f and os.path.isfile(os.path.join(dirname, f)):
                    nums = re.findall('-(\d{4})\.', f)
                    if len(nums) > 0:
                        filenums.append(int(nums[0]))
                    # filenums += [int(re.findall('-(\d{4})\.', f)[0])] if os.path.isfile(os.path.join(dirname, f)) else []

        i = max(filenums) + 1 if filenums else 0
        return "{}-{:04d}{}".format(basename,i,ext)

    def new_file(self):
        """ Open a new data file to write """
        # Close the current file, if any
        if self.file is not None:
            try:
                self.file.close()
            except Exception as e:
                logger.error("Encounter exception: {}".format(e))
                logger.error("Cannot close file '{}'. File may be damaged.".format(self.file.filename))
        # Get new file name
        self.filename.value = self.new_filename()
        head = os.path.dirname(self.filename.value)
        head = os.path.normpath(head)
        dirs = head.split(os.sep)
        # Check if path exists. If not, create new one(s).
        os.makedirs(head, exist_ok=True)
        logger.debug("Create new data file: %s." % self.filename.value)
        # Copy current settings to a folder with the file name
        if self.save_settings.value:
            # just move copies to a new directory
            self.save_yaml()
        if self.exp_log:
            self.write_to_log()
        return h5py.File(self.filename.value, 'w', libver='latest')

    def write_to_log(self):
        """ Record the experiment in a log file """
        if config.LogDir:
            logfile = os.path.join(config.LogDir, "experiment_log.tsv")
            if os.path.isfile(logfile):
                lf = pd.read_csv(logfile, sep="\t")
            else:
                logger.info("Experiment log file created.")
                lf = pd.DataFrame(columns = ["Filename", "Date", "Time"])
            lf = lf.append(pd.DataFrame([[self.filename.value, time.strftime("%y%m%d"), time.strftime("%H:%M:%S")]],columns=["Filename", "Date", "Time"]),ignore_index=True)
            lf.to_csv(logfile, sep = "\t", index = False)

    def save_yaml(self):
        """ Save a copy of current experiment settings """
        if config.meas_file:
            head = os.path.dirname(self.filename.value)
            fulldir = os.path.splitext(self.filename.value)[0]
            if not os.path.exists(fulldir):
                os.makedirs(fulldir)
                config.dump_meas_file(config.load_meas_file(config.meas_file), os.path.join(fulldir, os.path.split(config.meas_file)[1]), flatten = True)

    def save_yaml_h5(self):
        """ Save a copy of current experiment settings in the h5 metadata"""
        if config.meas_file:
            header = self.file.create_group("header")
            # load them dump to get the 'include' information
            header.attrs['settings'] = config.dump_meas_file(config.load_meas_file(config.meas_file), flatten = True)

    def get_data(self, data_path):
        """Request data back from the file handler. Data_path is given with respect to root: e.g. /data/voltage"""
        if self.done.is_set():
            logger.info("Not yet implemented. Please load the file directly")
        else:
            self.queue.put(("get_data", data_path))
            time.sleep(0.1)
            try:
                path, data = self.ret_queue.get(True, 5.0)
                if path != data_path:
                    logger.warning("Retrieved data path {} doesn't match requested path {}".format(path, data_path))
                return data
            except queue.Empty:
                logger.warning("Could not retrieve dataset within 5 seconds")

    def main(self):
        streams = self.sink.input_streams
        desc = streams[0].descriptor
        got_done_msg = {s: False for s in streams}

        # Write pointer
        w_idx = {s: 0 for s in streams}
        points_taken = {s: 0 for s in streams}
        i = 0
        stream_done = False

        while not self.exit.is_set():
            i +=1
            # Try to pull all messages in the queue. queue.empty() is not reliable, so we
            # ask for forgiveness rather than permission.
            msgs_by_stream = {s: [] for s in streams}

            for stream in streams[::-1]:
                while not self.exit.is_set():
                    try:
                        msgs_by_stream[stream].append(stream.queue.get(False))
                    except queue.Empty as e:
                        time.sleep(0.002)
                        break

            self.push_resource_usage()

            # Process many messages for each stream
            for stream, messages in msgs_by_stream.items():
                for message in messages:

                    # If we receive a message
                    if message['type'] == 'event':
                        # logger.info('%s "%s" received event of type "%s"', self.__class__.__name__, self.name, message['event_type'])
                        if message['event_type'] == 'done':
                            got_done_msg[stream] = True
                            self.queue.put(("done",))
                        elif message['event_type'] == 'refined':
                            message_data = message['data']
                            self.refine(message_data)
                        elif message['event_type'] == 'new_tuples':
                            num_new_points = self.process_new_tuples(desc, message['data'])
                            for stream in streams:
                                self.queue.put(("resize_to", self.data_write_paths[stream], len(desc.visited_tuples),))
                            if self.store_tuples:
                                for an in self.axis_names:
                                    self.queue.put(("resize_to", self.tuple_write_paths[an], len(desc.visited_tuples),))

                    elif message['type'] == 'data':

                        message_data = message['data']
                        self.processed += message_data.nbytes

                        if not hasattr(message_data, 'size'):
                            message_data = np.array([message_data])

                        # Write the data
                        try:
                            self.queue.put(("write", self.data_write_paths[stream], w_idx[stream], w_idx[stream]+message_data.size, message_data))
                        except queue.Full as e:
                            logger.warning("QUEUE FULLL!!!!!!!!!!!")

                        # Write the coordinate tuples
                        if self.store_tuples:
                            if desc.is_adaptive():
                                tuples = desc.tuples()
                                for axis_name in self.axis_names:
                                    self.queue.put(("write", self.tuple_write_paths[axis_name], w_idx[stream], w_idx[stream]+message_data.size,
                                                    tuples[axis_name][w_idx[stream]:w_idx[stream]+message_data.size]))

                        if (datetime.datetime.now() - self.last_flush).seconds > 3:
                            self.queue.put(("flush",))
                            self.last_flush = datetime.datetime.now()
                        w_idx[stream] += message_data.size
                        points_taken[stream] = w_idx[stream]

                        logger.debug("HDF5: Write index at %d", w_idx[stream])
                        logger.debug("HDF5: %s has written %d points", stream.name, w_idx[stream])

            # outputs = self.output_connectors.values()
            # if outputs:
            #     output_status = [v.done() for v in outputs]
            #     print('x2 dones: %s' % str(output_status))

            #     if not np.all(output_status):
            #         print('------------------->>>>>>>>> NOT ALL DONE YET')

            if np.all(list(got_done_msg.values())):
                stream_done = True
                self.done.set()
                break

        print(self.filter_name, "leaving main loop")


class DataBuffer(Filter):
    """Writes data to IO."""

    sink = InputConnector()

    def __init__(self, store_tuples=True, **kwargs):
        super(DataBuffer, self).__init__(**kwargs)
        self.quince_parameters = []
        self.sink.max_input_streams = 100
        self.store_tuples = store_tuples
        self._final_buffers = Queue()
        self.final_buffers = None

        self.out_queue = Queue() # This seems to work, somewhat surprisingly

    def final_init(self):
        self.w_idxs  = {s: 0 for s in self.sink.input_streams}
        self.descriptor = self.sink.input_streams[0].descriptor

    def main(self):
        self.done.clear()
        streams = self.sink.input_streams

        buffers = {s: np.empty(s.descriptor.expected_num_points(), dtype=s.descriptor.dtype) for s in self.sink.input_streams}

        for s in streams[1:]:
            if not np.all(s.descriptor.expected_tuples() == streams[0].descriptor.expected_tuples()):
                raise ValueError("Multiple streams connected to DataBuffer must have matching descriptors.")

        # Buffers for stream data
        stream_data = {s: np.zeros(0, dtype=self.sink.descriptor.dtype) for s in streams}

        # Store whether streams are done
        stream_done = {s: False for s in streams}

        while not self.exit.is_set():
            try:
                stream_results = {stream: stream.queue.get(True, 0.2) for stream in streams}
            except queue.Empty as e:
                continue

            # Add any new data to the buffers
            for stream, message in stream_results.items():
                message_type = message['type']
                message_data = message['data']
                message_data = message_data if hasattr(message_data, 'size') else np.array([message_data])
                if message_type == 'event':
                    if message['event_type'] == 'done':
                        stream_done[stream] = True
                    elif message['event_type'] == 'refined':
                        # Single we don't have much structure here we simply
                        # create a new buffer and paste the old buffer into it
                        old_buffer = buffers[stream]
                        new_size   = stream.descriptor.num_points()
                        buffers[stream] = np.empty(stream.descriptor.num_points(), dtype=stream.descriptor.dtype)
                        buffers[stream][:old_buffer.size] = old_buffer

                elif message_type == 'data':
                    stream_data[stream] = message_data.flatten()

            # if False not in stream_done.values():
            #     logger.debug('%s "%s" is done', self.__class__.__name__, self.name)
            #     break

            for stream in stream_results.keys():
                data = stream_data[stream]
                buffers[stream][self.w_idxs[stream]:self.w_idxs[stream]+data.size] = data
                self.w_idxs[stream] += data.size

            # if not np.all([v.done() for v in self.output_connectors.values()]):
            #     print('--------- IO NOT ALL DONE')

            # If we have gotten all our data and process_data has returned, then we are done!
            if np.all([v.done() for v in self.input_connectors.values()]):
                self.done.set()
                break

        for s in streams:
            self._final_buffers.put(buffers[s])

        # If we're using an mp Queue to return the results, put them in here
        if self.out_queue:
            self.out_queue.put(self.get_data())

    def get_rdata(self):
        if self.out_queue:
            data = self.get_data()
            self.out_queue.put(data)

    def get_data(self):
        streams = self.sink.input_streams
        desc = streams[0].descriptor

                # Get the data from the queue if necessary
        if not self.final_buffers:
            self.final_buffers = {s: self._final_buffers.get() for s in streams}

        # Set the dtype for the parameter columns
        if self.store_tuples:
            dtype = desc.axis_data_type(with_metadata=True)
        else:
            dtype = []

        # Extend the dtypes for each data column
        for stream in streams:
            dtype.append((stream.descriptor.data_name, stream.descriptor.dtype))
        data = np.empty(self.final_buffers[streams[0]].size, dtype=dtype)

        if self.store_tuples:
            tuples = desc.tuples(as_structured_array=True)
            for a in desc.axis_names(with_metadata=True):
                data[a] = tuples[a]
        for stream in streams:
            data[stream.descriptor.data_name] = self.final_buffers[stream]
        return data

    def get_descriptor(self):
        return self.sink.input_streams[0].descriptor

class ProgressBar(Filter):
    """ Display progress bar(s) on the terminal/notebook.

    num: number of progress bars to be display, \
    corresponding to the number of axes (counting from outer most)

        For running in Jupyter Notebook:
    Needs to open '_tqdm_notebook.py',\
    search for 'n = int(s[:npos])'\
    then replace it with 'n = float(s[:npos])'
    """
    sink = InputConnector()
    def __init__(self, num=0, notebook=False):
        super(ProgressBar,self).__init__()
        self.num    = num
        self.notebook = notebook
        self.bars   = []
        self.w_id   = 0

    def run(self):
        if config.profile:
            cProfile.runctx('self.main()', globals(), locals(), 'prof-%s.prof' % self.filter_name)
        else:
            self.main()

    def main(self):
        self.stream = self.sink.input_streams[0]
        axes = self.stream.descriptor.axes
        num_axes = len(axes)
        totals = [self.stream.descriptor.num_points_through_axis(axis) for axis in range(num_axes)]
        chunk_sizes = [max(1,self.stream.descriptor.num_points_through_axis(axis+1)) for axis in range(num_axes)]
        self.num = min(self.num, num_axes)

        self.bars   = []
        for i in range(self.num):
            if self.notebook:
                self.bars.append(tqdm_notebook(total=totals[i]/chunk_sizes[i]))
            else:
                self.bars.append(tqdm(total=totals[i]/chunk_sizes[i]))
        self.w_id   = 0
        while not self.exit.is_set():
            if self.stream.done() and self.w_id==self.stream.num_points():
                break

            new_data = np.array(self.stream.queue.get()).flatten()
            while self.stream.queue.qsize() > 0:
                new_data = np.append(new_data, np.array(self.stream.queue.get_nowait()).flatten())
            self.w_id += new_data.size
            num_data = self.stream.points_taken.value
            for i in range(self.num):
                if num_data == 0:
                    if self.notebook:
                        self.bars[i].sp(close=True)
                        # Reset the progress bar with a new one
                        self.bars[i] = tqdm_notebook(total=totals[i]/chunk_sizes[i])
                    else:
                        # Reset the progress bar with a new one
                        self.bars[i].close()
                        self.bars[i] = tqdm(total=totals[i]/chunk_sizes[i])
                pos = int(10*num_data / chunk_sizes[i])/10.0 # One decimal is good enough
                if pos > self.bars[i].n:
                    self.bars[i].update(pos - self.bars[i].n)
                num_data = num_data % chunk_sizes[i]
