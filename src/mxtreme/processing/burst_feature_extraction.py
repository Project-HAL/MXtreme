'''
Extract burst features and save to csv.
'''

import subprocess
from glob import glob
import time
import pandas as pd
import pickle
from pathlib import Path

from mxtreme.recording import Recording
from mxtreme import utils
from mxtreme import constants
from mxtreme import visualizations as viz

def extract_burst_features(exp_data, burst_csv_path, burst_log_path=None,
                           onset_thresh_pct=constants.ONSET_THRESH_PCT, offset_thresh_pct=constants.OFFSET_THRESH_PCT):

    print('Computing burst features...')
    start_time = time.time()

    rec = Recording(id, exp_data, burst_csv=burst_csv_path)

    # Burst characteristics
    rec.compute_burst_features(onset_thresh_pct=onset_thresh_pct, offset_thresh_pct=offset_thresh_pct) # adds columns to the burst csv
    rec.save_burst_data()
    rec.save_burst_metadata(burst_log_path, overwrite=True)

    end_time = time.time()-start_time
    print(f'Burst feature extraction took {end_time:.2f} sec.')
    

if __name__=='__main__':

    EXP_ID = 'waveTraining'
    chip = '*'
    well = '*'
    DIV = '*'
    
    constants.PARENT_DIR = Path(constants.braintrix_linux) / "Halnalysis" # set parent directory for saving
    
    path_to_exp_data = constants.PARENT_DIR / "data" / "preprocessed" / f"{EXP_ID}" / f"{chip}" / f"well{well}" / f"DIV{DIV}*_exp_data.npz"

    for i, filepath in enumerate(glob(str(path_to_exp_data))):

        if chip=='M07459' and well==0:
            continue
    
        exp_data = utils.load_data(filepath)
        
        # Paths to relevant data
        burst_csv_path = glob(str(Path(constants.PARENT_DIR) / "data" / "burst_data" / f"{exp_data['exp_id'].item()}" / f"{exp_data['chip'].item()}" / f"well{exp_data['well'].item()}" / f"DIV{exp_data['DIV'].item()}_*burst_data.csv"))[0]
        burst_log_path = Path(constants.PARENT_DIR) / "data" / "burst_data" / f"{exp_data['exp_id'].item()}_burst_log.csv" 
        
        extract_burst_features(exp_data, burst_csv_path, burst_log_path=burst_log_path)
