from __future__ import print_function, division
import logging
import datetime
import signal
import sys

import numpy as np
import itertools
import time
import h5py

from bokeh.client import push_session
from bokeh.plotting import hplot
from bokeh.io import curdoc, curstate
from bokeh.util.session_id import generate_session_id
from bokeh.document import Document

from .plotting import BokehServerThread, Plotter, Plotter2D, MultiPlotter
from .procedure import Procedure, Parameter, Quantity

logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s - %(levelname)s: \t%(asctime)s: \t%(message)s')
logger.setLevel(logging.INFO)

class Writer(object):
    """Data structure for the written quantities"""
    def __init__(self, dataset, quantities):
        super(Writer, self).__init__()
        self.dataset = dataset
        self.quantities = quantities

class SweptParameter(object):
    """Data structure for a swept Parameters, contains the Parameter
    object rather than subclassing it since we just need to keep track
    of some values"""
    def __init__(self, parameter, values):
        self.parameter = parameter
        self.values = values
        self.length = len(values)
        self.indices = range(self.length)
        self.name = parameter.name
        self.unit = parameter.unit

    @property
    def value(self):
        return self.parameter.value

    @value.setter
    def value(self, value):
        self.parameter.value = value

    def push(self):
        for pph in self.parameter.pre_push_hooks:
            pph()
        self.parameter.push()
        for pph in self.parameter.post_push_hooks:
            pph()

class Sweep(object):
    """For controlling sweeps over arbitrary number of arbitrary parameters. The order of sweeps\
    is defined by the order of the parameters passed to the add_parameter method. The first\
    quantity varies the slowest, the final quantity the quickest.

    """
    def __init__(self, procedure):
        super(Sweep, self).__init__()

        if isinstance(procedure, Procedure):
            self._procedure = procedure
        else:
            raise TypeError("Must pass a Procedure subclass.")

        # Container for SweptParmeters
        self._swept_parameters =  []

        # Container for written Quantities
        self._quantities = []

        # Iterable the yields sweep values
        self._sweep_generator = None

        # Contains a list of tuples (dataset_object, [list of parameters and quantities] )
        self._filenames = []
        self._files = {}
        self._writers = []
        self._plotters = []

    def __del__(self):
        #Close the h5 files
        for fid in self._files.values():
            fid.close()

    def __iter__(self):
        return self

    def add_parameter(self, param, sweep_list):
        p = SweptParameter(param, sweep_list)
        self._swept_parameters.append(p)
        self.generate_sweep()

        # Set the value of the parameter to the initial value of the sweep
        param.value = sweep_list[0]
        return p

    def add_writer(self, filename, sample_name, dataset_name, *quants, **kwargs):
        """Add a dataset that updates based on the supplied quantities"""

        # Loop through and check the supplied quantities
        for q in quants:
            if not isinstance(q, Quantity):
                raise TypeError("Expecting Quantity, not %s" % str(type(q)) )

        # See if we've already made the file
        if filename not in self._filenames:
            self._filenames.append(filename)
            self._files[filename] = h5py.File(filename, 'a')
        f = self._files[filename]

        # See if there is already a group matching this sample
        if sample_name not in f.keys():
            f.create_group(sample_name)
        s = f[sample_name]

        # See if there is already a group matching today's date
        date_str = datetime.date.today().strftime('%Y-%m-%d')
        if date_str not in s.keys():
            s.create_group(date_str)
        g = s[date_str]

        # See if there is already a dataset with the same name
        # increment the actual dataset name by 1 and store this
        # as dataset_name-0001 dataset_name-0002, etc. First we
        # parse any filenames already in the group, then we make
        # sure we store a new file with the name incremented by
        # 1.

        files_with_same_prefix = ["-".join(k.split("-")[:-1]) for k in g.keys() if dataset_name == "-".join(k.split("-")[:-1])]
        if dataset_name not in files_with_same_prefix:
            dataset_name = "{:s}-{:04d}".format(dataset_name, 1)
        else:
            largest_index = max([int(k.split("-")[-1]) for k in g.keys() if dataset_name in k])
            dataset_name = "{:s}-{:04d}".format(dataset_name, largest_index + 1)

        # Determine the dataset dimensions
        sweep_dims = [ p.length for p in self._swept_parameters ]
        logger.debug("Sweep dims are %s for the list of swept parameters in the writer %s, %s." % (str(sweep_dims), filename, dataset_name) )

        data_dims = [len(quants)+len(self._swept_parameters)]
        dataset_dimensions = tuple(sweep_dims + data_dims)

        # Get the datatype, defaulting to float
        dtype = kwargs['dtype'] if 'dtype' in kwargs else 'f'

        # Create the data set
        dset = g.create_dataset(dataset_name, dataset_dimensions, dtype=dtype)

        # Create a new instances of the data structure and store it
        self._writers.append( Writer(dset, quants) )

    def write(self):
        indices = list(next(self._index_generator))

        for w in self._writers:
            current_p_values = [p.parameter.value for p in self._swept_parameters]

            logger.debug("Current indicies are: %s" % str(indices) )

            for i, p in enumerate(self._swept_parameters):
                coords = tuple( indices + [i] )
                logger.debug("Coords: %s" % str(coords) )
                w.dataset[coords] = p.parameter.value
            for i, q in enumerate(w.quantities):
                coords = tuple( indices + [len(self._swept_parameters) + i] )
                w.dataset[coords] = q.value

    def add_plotter(self, title, x, y, x_axis_type='auto', y_axis_type='auto', **kwargs):
        kwargs['x_axis_type'] = x_axis_type
        kwargs['y_axis_type'] = y_axis_type
        self._plotters.append(Plotter(title, x, y, **kwargs))
        return self._plotters[-1]

    def add_plotter2d(self, title, x, y, z, **kwargs):
        swept_param_dict = {sp.name: sp for sp in self._swept_parameters}
        if not x.name in swept_param_dict:
            print("Could not find parameter {} in the list of sweeps!".format(xs.name))
            raise Exception("Cannot plot over a non-swept parameter.")
        else:
            s_x = swept_param_dict[x.name]
        if not y.name in swept_param_dict:
            print("Could not find parameter {} in the list of sweeps!".format(xs.name))
            raise Exception("Cannot plot over a non-swept parameter.")
        else:
            s_y = swept_param_dict[y.name]
        self._plotters.append(Plotter2D(title, s_x, s_y, z, **kwargs))
        return self._plotters[-1]

    def add_multiplotter(self, title, xs, ys, **kwargs):
        self._plotters.append(MultiPlotter(title, xs, ys, **kwargs))
        return self._plotters[-1]

    def plot(self, force=False):
        for p in self._plotters:
            p.update(force=force)

    def generate_sweep(self):
        self._sweep_generator = itertools.product(*[sp.values for sp in self._swept_parameters])
        self._index_generator = itertools.product(*[sp.indices for sp in self._swept_parameters])

    def run(self, notebook=False):
        self._procedure.init_instruments()

        if len(self._plotters) > 0:
            t = BokehServerThread(notebook=notebook)
            t.start()
            #On some systems there is a possibility we try to `push_session` before the
            #the server on the BokehServerThread has started.
            time.sleep(1)
            h = hplot(*[p.figure for p in self._plotters])
            curdoc().clear()
            sid = generate_session_id()
            doc = Document()
            doc.add_root(h)
            session = push_session(doc, session_id=sid)
            
            if notebook:
                from bokeh.embed import autoload_server, components
                from bokeh.io import output_notebook
                from IPython.display import display, HTML

                output_notebook()
                script = autoload_server(model=None, session_id=sid)
                html = \
                        """
                        <html>
                        <head></head>
                        <body>
                        %s
                        </body>
                        </html>
                        """ % script
                display(HTML(html))
            else:
                session.show(doc)

        def shutdown():
            if len(self._plotters) > 0:
                time.sleep(0.5)
                t.join()
            self._procedure.shutdown_instruments()

        def catch_ctrl_c(signum, frame):
            logger.info("Caught SIGINT.  Shutting down.")
            shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, catch_ctrl_c)

        # Keep track of the previous values
        last_param_values = None

        for param_values in self._sweep_generator:

            # Update the parameter values. Unles set and push if there has been a change
            # in the value from the previous iteration.
            for i, sp in enumerate(self._swept_parameters):
                if last_param_values is None or param_values[i] != last_param_values[i]:
                    sp.value = param_values[i]
                    sp.push()
                    logger.debug("Updated {:s} to {:g} since the value changed.".format(sp.parameter.name, sp.value))
                else:
                    logger.debug("Didn't update {:s} since the value didn't change.".format(sp.parameter.name))

            # update previous values
            last_param_values = param_values

            # Run the procedure
            self._procedure.run()

            # Push values to file and update plots
            self.write()
            self.plot()

        self.plot(force=True)

        shutdown()
