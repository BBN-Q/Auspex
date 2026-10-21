Plot Server
========================
Auspex plotting is facilitated by a plot server and plot clients. A single server can handle multiple running experiments, which publish their data with a unique UUID. Many clients can connect to the server and request data for a particular UUID.

Running the Plot Server
***********************
The plot server must currently be started manually with::

	python plot_server.py


Running the Plot Client
***********************
The plot client *matplotlib-client.py* should be run automatically whenever plotters are put in an experiment's filter pipeline. The code can be run manually and used to connect to a remote system, simply by running the exectuable with::

	python matplotlib-client.py