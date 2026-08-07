'''
Detect burst and save recordings.
'''

from glob import glob
import time
import pickle

from mxtreme.recording import Recording
from mxtreme import utils
from mxtreme import constants

def burst_detection(exp_data, method="ISI_N", overwrite_burst_log=False, **kwargs):
    """
    Takes preprocessed experimental data, creates a recording object, and performs burst detection.

    :param exp_data: Dictionary containing experimental data.
    :type exp_data: dict

    :param method: Burst detection method to use, defaults to "ISI_N".
    :type method: str, optional

    :param overwrite_burst_log: Whether or not to add to or overwrite the burst log if one already exists, defaults to False.
    :type overwrite_burst_log: bool, optional
    """

    # Load data
    start_time = time.time()
    print(f'Detecting bursts...')

    # Make a Recording object
    rec = Recording(id=0, exp_data=exp_data)

    # Run burst detection on each phase

    rec.detect_bursts(**kwargs)

    burst_csv_savepath = rec.save_burst_data()

    burst_log_savepath = rec.save_burst_metadata(overwrite=overwrite_burst_log)

    end_time = time.time()-start_time
    print(f'Burst detection took {end_time:.2f} sec.')

    return burst_csv_savepath, burst_log_savepath

if __name__ == '__main__': 
    
    path_to_npz = constants.braintrix_linux+f'HALnalysis/data/preprocessed/m1BurstTrainer/P004722/well0/DIV33_111825_P004722_m1BurstTrainer_well0_exp_data.npz'

    burst_detection(utils.load_data(path_to_npz))
    
