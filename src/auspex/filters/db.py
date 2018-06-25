# Get these in global scope for module imports
Connection       = None
NodeProxy        = None
FilterProxy      = None
Demodulate       = None
Average          = None
Integrate        = None
OutputProxy      = None
Display          = None
Write            = None
QubiteProxy      = None
stream_hierarchy = None

def define_entities(db):
    class Connection(db.Entity):
        node1         = Required("NodeProxy")
        node2         = Required("NodeProxy")
        pipeline_name = Required(str)

    class NodeProxy(db.Entity):
        qubit_name      = Optional(str)
        connection_to   = Set(Connection, reverse='node1')
        connection_from = Set(Connection, reverse='node2')
        
    class FilterProxy(NodeProxy):
        """docstring for FilterProxy"""
        
        def __init__(self):
            super(FilterProxy, self).__init__()
            self.exp = None

        def add(self, filter_obj):
            if not self.exp:
                print("This filter may have been orphaned by a clear_pipeline call!")
                return
            self.exp.meas_graph.add_edge(self, filter_obj)
            filter_obj.exp = self.exp
            filter_obj.qubit_name = self.qubit_name
            return filter_obj

        def label(self):
            return f"{self.__class__.__name__}" # for {self.qubit_name}"
        
        def __repr__(self):
            return f"{self.__class__.__name__} for {self.qubit_name} at {hex(id(self))}"
        
    class Demodulate(FilterProxy):
        """Digital demodulation and filtering to select a signal at a particular frequency component. This 
        filter does the following:
            
            1. First stage decimating filter on data
            2. Take product of result with with reference signal at demodulation frequency
            3. Second stage decimating filter on result to boost n_bandwidth
            4. Final channel selecting filter at n_bandwidth/2
        
        If an axis name is supplied to `follow_axis` then the filter will demodulate at the freqency 
        `axis_frequency_value - follow_freq_offset` otherwise it will demodulate at `frequency`. Note that
        the filter coefficients are still calculated with respect to the `frequency` paramter, so it should
        be chosen accordingly when `follow_axis` is defined."""
        # Demodulation frequency
        frequency          = Optional(float, default=10e6, min=-10e9, max=10e9)
        # Filter bandwidth 
        bandwidth          = Optional(float, default=5e6, min=0.0, max=100e6)
        # Let the demodulation frequency follow an axis' value (useful for sweeps)
        follow_axis        = Optional(str) # Name of the axis to follow
        # Offset of the actual demodulation from the followed axis
        follow_freq_offset = Optional(float, default=0.0) # Offset
        # Maximum data reduction factor
        decimation_factor  = Optional(int, default=4, min=1, max=100)

    class Average(FilterProxy):
        """Takes data and collapses along the specified axis."""
        # Over which axis should averaging take place
        axis = Optional(str, default="round_robins")

    class Integrate(FilterProxy):
        """Integrate with a given kernel or using a simple boxcar.
        Kernel will be padded/truncated to match record length"""
        # Use a boxcar (simple) or use a kernel specified by the kernel_filename parameter
        simple_kernel   = Optional(bool, default=True)
        # File in which to find the kernel data
        kernel_filename = Optional(str)
        # DC bias
        bias            = Optional(float, default=0.0)
        # For a simple kernel, where does the boxcar start
        box_car_start   = Optional(float, default=0.0, min=0.0)
        # For a simple kernel, where does the boxcar stop
        box_car_stop    = Optional(float, default=100.0e-9, min=0.0)
        # Built in frequency for demodulation
        demod_frequency = Optional(float, default=0.0)

    class OutputProxy(FilterProxy):
        pass

    class Display(OutputProxy):
        """Create a plot tab within the plotting interface."""
        # Should we plot in 1D or 2D? 0 means choose the largest possible.
        plot_dims = Required(int, default=0, min=0, max=2) 
        # Allowed values are "real", "imag", "real/imag", "amp/phase", "quad"
        # TODO: figure out how to validate these in pony
        plot_mode = Required(str, default="quad")

    class Write(OutputProxy):
        """Writes data to file."""
        filename  = Optional(str)
        groupname = Optional(str)

    class QubitProxy(NodeProxy):
        """docstring for FilterProxy"""
        
        def __init__(self, exp, qubit_name):
            super(QubitProxy, self).__init__(qubit_name=qubit_name)
            self.exp = exp
            self.qubit_name = qubit_name
            self.digitizer_settings = None
            self.available_streams = None
            self.stream_type = None
        
        def add(self, filter_obj):
            if not self.exp:
                print("This qubit may have been orphaned!")
                return
            
            self.exp.meas_graph.add_edge(self, filter_obj)
            filter_obj.exp = self.exp
            filter_obj.qubit_name = self.qubit_name
            return filter_obj

        def set_stream_type(self, stream_type):
            if stream_type not in ["raw", "demodulated", "integrated", "averaged"]:
                raise ValueError(f"Stream type {stream_type} must be one of raw, demodulated, integrated, or result.")
            if stream_type not in self.available_streams:
                raise ValueError(f"Stream type {stream_type} is not avaible for {self.qubit_name}. Must be one of {self.available_streams}")
            self.stream_type = stream_type
            
        def clear_pipeline(self):
            """Remove all nodes coresponding to the qubit"""
            # import ipdb; ipdb.set_trace()
            # nodes_to_remove = [n for n in self.exp.meas_graph.nodes() if n.qubit_name == self.qubit_name]
            # self.exp.meas_graph.remove_nodes_from(nodes_to_remove)
            desc = nx.algorithms.dag.descendants(self.exp.meas_graph, self)
            for n in desc:
                n.exp = None
            self.exp.meas_graph.remove_nodes_from(desc)

        def auto_create_pipeline(self):
            if self.stream_type == "raw":
                self.add(Demodulate()).add(Integrate()).add(Average()).add(Write())
            if self.stream_type == "demodulated":
                self.add(Integrate()).add(Average()).add(Write())
            if self.stream_type == "integrated":
                self.add(Average()).add(Write())
            if self.stream_type == "averaged":
                self.add(Write())
            
        def show_pipeline(self):
            desc = list(nx.algorithms.dag.descendants(self.exp.meas_graph, self)) + [self]
            labels = {n: n.label() for n in desc}
            subgraph = self.exp.meas_graph.subgraph(desc)
            colors = ["#3182bd" if isinstance(n, QubitProxy) else "#ff9933" for n in subgraph.nodes()]
            plot_graph(subgraph, labels, colors=colors, prog='dot')
            
        def show_connectivity(self):
            pass
            
        def label(self):
            return self.__repr__()
            
        def __repr__(self):
            return f"Qubit {self.qubit_name}"

    globals()["Connection"] = Connection
    globals()["NodeProxy"] = NodeProxy
    globals()["FilterProxy"] = FilterProxy
    globals()["Demodulate"] = Demodulate
    globals()["Average"] = Average
    globals()["Integrate"] = Integrate
    globals()["OutputProxy"] = OutputProxy
    globals()["Display"] = Display
    globals()["Write"] = Write
    globals()["QubitProxy"] = QubitProxy
    globals()["stream_hierarchy"] = [Demodulate, Integrate, Average, OutputProxy]

