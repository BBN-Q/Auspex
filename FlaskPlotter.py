import threading

# flask
from flask import Flask, jsonify

# tornado
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

# bokeh
from bokeh.server.crossdomain import crossdomain
from bokeh.plotting import figure, show, output_file, hplot, vplot
from bokeh.models.sources import AjaxDataSource

class FlaskPlotter(object):
    def __init__(self, plotters):
        super(FlaskPlotter, self).__init__()
        self.data_lookup = {p.filename: p.data for p in plotters}
        self.filenames = [p.filename for p in plotters]
        self.plotter_lookup = {p.filename: p for p in plotters}
        self.setup_data_server()
        self.thread = threading.Thread(target=IOLoop.instance().start)
        self.thread.daemon = True
        self.thread.start()
        self.setup_bokeh_plots()

    def shutdown(self):
        def shutdown_callback():
            self.http_server.stop()
            IOLoop.instance().stop()

        IOLoop.instance().add_callback(shutdown_callback)
        self.thread.join()

    def setup_data_server(self):
        app = Flask(__name__)
        self.http_server = HTTPServer(WSGIContainer(app))
        self.http_server.listen(5050)

        @app.route('/<filename>', methods=['GET', 'OPTIONS'])
        @crossdomain(origin="*", methods=['GET', 'POST'], headers=None)
        def fetch_func(filename):
            p = self.plotter_lookup[filename]

            xs = []
            ys = [[] for i in range(p.num_ys)]
            while not self.data_lookup[filename].empty():
                data = self.data_lookup[filename].get_nowait()
                xs.append(data[0])
                for i in range(p.num_ys):
                    ys[i].append(data[i+1])
            kwargs = { 'y{:d}'.format(i+1): ys[i] for i in range(p.num_ys) }
            kwargs['x'] = xs
            return jsonify(**kwargs)

    def setup_bokeh_plots(self):
        output_file("main.html", title="Plotting Output")
        plots = []
        sources = []

        for f in self.filenames:
            p = self.plotter_lookup[f]
            source = AjaxDataSource(data_url='http://localhost:5050/'+f,
                                    polling_interval=750, mode="append")

            xlabel = p.x.name + (" ("+p.x.unit+")" if p.x.unit is not None else '')
            ylabel = p.ys[0].name + (" ("+p.ys[0].unit+")" if p.ys[0].unit is not None else '')
            plot = figure(webgl=True, title=p.title,
                          x_axis_label=xlabel, y_axis_label=ylabel,
                          tools="save,crosshair")
            plots.append(plot)
            sources.append(source)

            # plots[-1].line('x', 'y', source=sources[-1], color="firebrick", line_width=2)
            xargs = ['x' for i in range(p.num_ys)]
            yargs = ['y{:d}'.format(i+1) for i in range(p.num_ys)]

            if p.num_ys > 1:
                plots[-1].multi_line(xargs, yargs, source=sources[-1], **p.figure_args)
            else:
                plots[-1].line('x', 'y1', source=sources[-1], **p.figure_args)

        q = hplot(*plots)
        show(q)

if __name__ == "__main__":
    import signal, time
    fp = FlaskPlotter([])

    def sig_handler(sig, fname):
        fp.shutdown()

    signal.signal(signal.SIGINT, sig_handler)
    print "server started"
    while fp.thread.isAlive():
        time.sleep(.2)
    print "server shutdown"
