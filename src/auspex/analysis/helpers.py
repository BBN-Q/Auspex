from auspex.data_format import AuspexDataContainer
import datetime
import os, re
from os import path
import numpy as np

def open_data(num, folder, groupname, datasetname="data", date=datetime.date.today().strftime('%y%m%d')):
    
    if date is not None:
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
