# PyControl #

Supports running physics experiments with:

1. Instrument control classes wrapping [PyVISA](https://github.com/hgrecco/pyvisa)
2. Sweeping instrument parameters.
3. Plotting real-time measured data with [Bokeh](http://bokeh.pydata.org/)
4. Saving data to [HDF5](https://www.hdfgroup.org/HDF5/) files


## Continuous Integration ##

The project is under continuous integration using Docker.  To test locally:

1. Install Anaconda3 Docker image.
	```shell
	cryan ~ $ docker pull continuumio/anaconda3
	Using default tag: latest
	latest: Pulling from continuumio/anaconda3
	d1464a6a95cd: Pull complete
	ef3ea792de03: Pull complete
	1322c83552fe: Pull complete
	f1352f3b5067: Pull complete
	b17dfd99777b: Pull complete
	740252606a3c: Pull complete
	9c942920d00c: Pull complete
	47ea262b7cc3: Pull complete
	Digest: sha256:84518655edbae7cf7008f71016773837b32d7d2287398b0f0a7fde1b342fea8c
	Status: Downloaded newer image for continuumio/anaconda3:latest
	```
1. Run an ephemeral container based off of the image and mount the local copy of the repository with changes to test.
	```shell
	docker run -it --rm -v /path/to/repo/:/pycontrol continuumio/anaconda3 /bin/bash
	```
1. In the container setup and run the tests
	```shell
	root@0c4b7fe64114:/# pip install pyvisa
	Collecting pyvisa
	  Downloading PyVISA-1.8.tar.gz (429kB)
	    100% |████████████████████████████████| 430kB 1.2MB/s
	Building wheels for collected packages: pyvisa
	  Running setup.py bdist_wheel for pyvisa
	  Stored in directory: /.cache/pip/wheels/50/4e/80/06f577a61ca17b0d50961a2d900f38c80f8b79afbcb2c9c7bd
	Successfully built pyvisa
	Installing collected packages: pyvisa
	Successfully installed pyvisa-1.8
	root@0c4b7fe64114:/# export PYTHONPATH=/pycontrol/
	root@0c4b7fe64114:/# python -m unittest discover -v /pycontrol/test/
	test_getters (test_instrument.InstrumentTestCase)
	Check that Commands with only `get_string` have only get methods ... ok
	test_properties (test_instrument.InstrumentTestCase)
	Check that property and setter/getter are implemented ... ok

	----------------------------------------------------------------------
	Ran 2 tests in 0.000s

	OK
	root@349fa096554f:/#
	```
