'''
Driver to apply ETL pipeline to raw experimental data.
'''

from glob import glob
import time

from mxtreme.etl import ETL
from mxtreme import constants

def preprocess(filepath, amplitude_thresh=constants.AMPLITUDE_THRESH, bin_size=constants.BIN_SIZE,
               refractory_period=constants.REFRACTORY_PERIOD, post_stim_period=constants.POST_STIM_PERIOD,
               exp_condition_dir=None, clean_data_dir=None, metadata=None, overwrite=True, wells=None):
    """
    Function to process raw data from an h5 file including extracting data from the h5, cleaning and transforming the data, and saving it out to an npz file. 

    :param filepath: Path to h5 file containing raw experimental data.
    :type filepath: str

    :param amplitude_thresh: Minimum spike amplitude. Defaults to constants.AMPLITUDE_THRESH.
    :type amplitude_thresh: float, optional

    :param bin_size: Size of bins in frames for spike binning. Defaults to constants.BIN_SIZE. 
    :type bin_size: float, optional

    :param refractory_frames: Minimum distance between spikes on the same channel. Spikes within refractory_frames of a previous spike are removed. Defaults to constants.REFRACTORY_FRAMES.
    :type refractory_frames: int, optional

    :param post_stim_frames: The number of frames after the end-stimulation event tag that we want to remove from the spike data due to artifacts. Defaults to constants.POST_STIM_FRAMES.
    :type post_stim_frames: int, optional

    :param exp_condition_dir: Directory specifying where to save the experimental conditions. If no path is provided, the directory will be set to 'constants.PARENT_DIR/HALnalysis/data/experimental_conditions/'. Defaults to None.
    :type exp_condition_dir: str, optional

    :param clean_data_dir: Directory in which to store the cleaned and transformed data (stored in npz files). If None, clean_data_dir will be set to 'constants.PARENT_DIR/HALnalysis/data/preprocessed/'. Defaults to None.
    :type clean_data_dir: str, optional

    :param metadata: If no metadata structure is included in the raw data metadata must be manually provided. This is the case for many of the older experiments, defaults to None. 
    The metadata must be a dictionary in the following format.
        metadata = {'Exp ID' : EXP_ID,
                  'Chip ID' : chip_id,
                  'Plate date' : int(plate_date),
                  'DIV' : int(DIV)
                  }
    :type metadata: dict, optional

    :param overwrite: Whether or not to overwrite an existing file if preprocessing has already been run for a file with the same name. 
    :type clean_data_dir: bool, optional

    :param wells: List of wells from the h5 that you want to process. If not provided, all wells in the h5 file will be processed.
    :type wells: list or int, optional

    :return: Returns a list of filepaths to npzs containing the cleaned and transformed data. There is one npz per culture per recording. 
    :rtype: list 
    """
    
    print(f"Extracting data from: {filepath}")

    start_time = time.time()

    etl = ETL(filepath, wells=wells)
    etl.extract(metadata)
    etl.write_experimental_conditions(datastore=exp_condition_dir)
    etl.transform(amplitude_cutoff=amplitude_thresh, bin_size=bin_size, refractory_period=refractory_period, post_stim_period=post_stim_period)
    clean_data_paths = etl.load(clean_data_dir, overwrite=overwrite)
    etl.load_HAL_burst_csvs()
    etl.register() # save info about what was processed to a csv 

    print(f"Preprocessing took {time.time()-start_time:.2f} seconds.")
        
    return clean_data_paths

if __name__=='__main__':

    path_to_h5 = ''
    preprocess(path_to_h5)




