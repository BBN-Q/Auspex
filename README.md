# PyControl #

[![build status](https://qiplab.bbn.com/ci/projects/1/status.png?ref=master)](https://qiplab.bbn.com/ci/projects/1?ref=master)

Supports running physics experiments with:

1. Instrument control classes wrapping [PyVISA](https://github.com/hgrecco/pyvisa)
2. Sweeping instrument parameters.
3. Plotting real-time measured data with [Bokeh](http://bokeh.pydata.org/)
4. Saving data to [HDF5](https://www.hdfgroup.org/HDF5/) files


## Continuous Integration ##

The project is under continuous integration using Docker.  To test locally:

1. Build the image using the Dockerfile in test. The Dockerfile gives us Anaconda3 with PyVISA and the BBN certificate installed.

    ```shell
    cryan ~ $ cd /path/to/repo/test/docker
    cryan docker $ docker build -t cryan/pycontrol .
    ```

1. Run an ephemeral container based off of the image and mount the local copy of the repository with changes to test.

    ```shell
    docker run -it --rm -v /path/to/repo/:/pycontrol cryan/pycontrol /bin/bash
    ```

1. In the container setup and run the tests

    ```shell
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
