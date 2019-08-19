from auspex.data_format import AuspexDataContainer
import datetime
import os, re
from os import path
import numpy as np
from itertools import product

def get_file_name():
    """Helper function to get a filepath from a dialog box"""

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw() # remove edges

    filepath = [filedialog.askdirectory()]
    root.update() # fixes OSX hanging issue
    #https://stackoverflow.com/questions/21866537/what-could-cause-an-open-file-dialog-window-in-tkinter-python-to-be-really-slow

    return filepath

def open_data(num=None, folder=None, groupname="main", datasetname="data", date=None):
    """Convenience Load data from an `AuspexDataContainer` given a file number and folder.
        Assumes that files are named with the convention `ExperimentName-NNNNN.auspex`

    Parameters:
        num (int)
            File number to be loaded.
        folder (string)
            Base folder where file is stored. If the `date` parameter is not None, assumes file is a dated folder. If no folder is specified, open a dialogue box. Open the folder with the desired ExperimentName-NNNN.auspex, then press OK
        groupname (string)
            Group name of data to be loaded.
        datasetname (string, optional)
            Data set name to be loaded. Default is "data".
        date (string, optional)
            Date folder from which data is to be loaded. Format is "YYMMDD" Defaults to today's date.

    Returns:
        data (numpy.array)
            Data loaded from file.
        desc (DataSetDescriptor)
            Dataset descriptor loaded from file.

    Examples:
        Loading a data container

        >>> data, desc = open_data(42, '/path/to/my/data', "q1-main", date="190301")

    """
    if num is None or folder is None:
        # pull up dialog box
        data_file = get_file_name()
        folder = ""
    else:
        if date == None:
            date = datetime.date.today().strftime('%y%m%d')
        folder = path.join(folder, date)
        assert path.isdir(folder), f"Could not find data folder: {folder}"

        p = re.compile(r".+-(\d+).auspex")
        files = [x.name for x in os.scandir(folder) if x.is_dir()]
        data_file = [x for x in files if p.match(x) and int(p.match(x).groups()[0]) == num]

    if len(data_file) == 0:
        raise ValueError("Could not find file!")
    elif len(data_file) > 1:
        raise ValueError(f"Ambiguous file information: found {data_file}")

    data_container = AuspexDataContainer(path.join(folder, data_file[0]))
    return data_container.open_dataset(groupname, datasetname)


def normalize_data(data, zero_id = 0, one_id = 1):
    metadata_str = [f for f in data.dtype.fields.keys() if 'metadata' in f]
    if len(metadata_str)!=1:
        raise ValueError('Data format not valid')
    metadata = data[metadata_str[0]]
    #find indeces for calibration points
    zero_cal_idx = [i for i, x in enumerate(metadata) if x == zero_id]
    one_cal_idx = [i for i, x in enumerate(metadata) if x == one_id]
    #find values for calibrated 0 and 1
    zero_cal = np.mean(data['Data'][zero_cal_idx])
    one_cal = np.mean(data['Data'][one_cal_idx])

    #normalize
    scale_factor = (zero_cal - one_cal)/2
    norm_data = (data['Data']-zero_cal)/scale_factor + 1
    #remove calibration points
    norm_data = [d for ind, d in enumerate(norm_data) if metadata[ind] == max(metadata)]
    return norm_data

def normalize_buffer_data(data, desc, qubit_index, zero_id = 0, one_id = 1):
    # Qubit index gives the string offset of the qubit in the metadata
    metadata = [(i, int(v[qubit_index])) for i,v in enumerate(desc.axes[0].metadata) if v != "data"]
    
    #find indeces for calibration points
    zero_cal_idx = [i for i, x in metadata if x == zero_id]
    one_cal_idx = [i for i, x in metadata if x == one_id]
    #find values for calibrated 0 and 1
    zero_cal = np.mean(data[zero_cal_idx])
    one_cal = np.mean(data[one_cal_idx])

    #normalize
    scale_factor = (zero_cal - one_cal)/2
    norm_data = (data-zero_cal)/scale_factor + 1

    #remove calibration points
    return norm_data[:metadata[0][0]]

def cal_scale(data, bit=0, nqubits=1, repeats=2):
    """
    Scale data from calibration points.
    Parameters
    ----------
    data : Unscaled data with cal points.
    bit: Which qubit in the sequence is to be calibrated (0 for 1st, etc...). Default 0.
    nqubits: Number of qubits in the data. Default 1.
    repeats: Number of calibraiton repeats. Default 2.
    Returns
    -------
    data : scaled data array
    """
    ncals = 2**nqubits * 2
    calSet = [0, 1]
    codes = [p for p in product(calSet, repeat=nqubits) for _ in range(repeats)]
    assert (len(codes) == ncals), "Did not calculate the right number of calibration points."
    assert (bit < nqubits), "Please select a qubit that is in your data"

    cals = data[-ncals:]
    zero_cals = []
    one_cals = []

    for j in range(ncals):
        if codes[j][bit] == 0:
            zero_cals.append(cals[j])
        if codes[j][bit] == 1:
            one_cals.append(cals[j])

    scale = -(np.mean(one_cals) - np.mean(zero_cals))/2
    data = data[:-ncals]
    data = (data-np.mean(zero_cals))/scale +  1

    return data

def cal_data(data, quad=np.real, qubit_name="q1", group_name="main", \
        return_type=np.float32, key=""):
    """
    Rescale data to :math:`\\sigma_z`. expectation value based on calibration sequences.

    Parameters:
        data (numpy array)
            The data from the writer or buffer, which is a dictionary
            whose keys are typically in the format qubit_name-group_name, e.g.
            ({'q1-main'} : array([(0.0+0.0j, ...), (...), ...]))
        quad (numpy function)
            This should be the quadrature where most of
            the data can be found.  Options are: np.real, np.imag, np.abs
            and np.angle
        qubit_name (string)
            Name of the qubit in the data file. Default is 'q1'
        group_name (string)
            Name of the data group to calibrate. Default is 'main'
        return_type (numpy data type)
            Type of the returned data. Default is np.float32.
        key (string)
            In the case where the dictionary keys don't conform
            to the default format a string can be passed in specifying the
            data key to scale.
    Returns:
        numpy array (type ``return_type``)
            Returns the data rescaled to match the calibration results for the :math:`\\sigma_z` expectation value.


    Examples:
        Loading and calibrating data

        >>> exp = QubitExperiment(T1(q1),averages=500)
        >>> exp.run_sweeps()
        >>> data, desc = exp.writers[0].get_data()

    """
    if key:
        pass
    else:
        key = qubit_name + "-" + group_name

    fields = data[key].dtype.fields.keys()
    meta_field = [f for f in fields if 'metadata' in f][0]
    ind_axis = meta_field.replace("_metadata", "")

    ind0 = np.where(data[key][meta_field] == 0 )[0]
    ind1 = np.where(data[key][meta_field] == 1 )[0]

    dat = quad(data[key]["Data"])
    zero_cal = np.mean(dat[ind0])
    one_cal = np.mean(dat[ind1])

    scale_factor = -(one_cal - zero_cal)/2

    #assumes calibrations at the end only
    y_dat = dat[:-(len(ind0) + len(ind1))]
    x_dat = data[key][ind_axis][:-(len(ind0) + len(ind1))]
    y_dat = (y_dat - zero_cal)/scale_factor + 1
    return y_dat.astype(return_type), x_dat
