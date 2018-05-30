import numpy as np
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
