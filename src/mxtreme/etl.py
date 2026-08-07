'''
Class to extract, transform, and load data from raw experimental output.
'''

import os
import h5py
import json
import numpy as np
import pandas as pd
import csv
import ast
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob

from mxtreme import constants

class ETL:
    """
    A class to extract raw experimental data from an h5 file, filter the data, remove stimulation periods from spike data, and bin spikes. 
    """

    def __init__(self, filepath:str, wells:list|int=None) -> None:
        """
        Creates an ETL object from a single h5 file. 

        :param filepath: Path to h5 file.
        :type filepath: str

        :param wells: List of wells from the h5 that you want to process. If not provided, all wells in the h5 file will be processed.
        :type wells: list or int, optional

        :raises FileNotFoundError
        """
        self.filepath = filepath

        if isinstance(wells, int):
            # Convert int to list
            self.wells_to_process = [wells]
        else:
            self.wells_to_process = wells
        
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"File does not exist: {self.filepath}")
        
    def extract(self, metadata=None):
        """
        Extracts relevant data from the raw h5 file.

        :param metadata: If no metadata structure is included in the raw data metadata must be manually provided. This is the case for many of the older experiments, defaults to None. 
        The metadata must be a dictionary in the following format.
            metadata = {'Exp ID' : EXP_ID,
                    'Chip ID' : chip_id,
                    'Plate date' : int(plate_date),
                    'DIV' : int(DIV)
                    }
        :type metadata: dict, optional

        :raises ValueError
        """

        self.data = {} # initialize empty dictionary to hold data for each well 

        with h5py.File(self.filepath, 'r') as f:
            
            # See all contents of h5 file
            # def print_h5(name, obj):
            #     print(name, type(obj))
            # f.visititems(print_h5)

            try: 
                raw_metadata = f[f'/assay/metadata'][:][0]
                decoded_metadata = raw_metadata.decode('utf-8').strip().replace("'",'"')
                h5_metadata = json.loads(decoded_metadata)

            except:
                if metadata is None:
                    raise ValueError("No metadata in h5 file. Metadata must be supplied in the format"+"\n"+ "metadata = {'Exp ID' : exp_id,'Chip ID' : chip_id, 'Plate date' : plate_date, 'DIV' : DIV}.")
                else:
                    h5_metadata = metadata

            h5_metadata['wells'] = [] # keeping track of how many and which wells have data recorded in this h5

            # Find well no.
            for i,well_long in enumerate(list(f[f'/wells/'].keys())):
                
                well = int(well_long[-1]) # well values range from 0-5

                if self.wells_to_process is not None:
                    if well not in self.wells_to_process:
                        continue

                self.data[well] = {} # initialize empty dictionary for well-specific data
                h5_metadata['wells'].append(well)

                # Extract experiment info
                    
                # Handle metadata here
                self.data[well]['exp_id'] = h5_metadata['Exp ID']
                self.data[well]['chip'] = h5_metadata['Chip ID']
                self.data[well]['plate_date'] = h5_metadata['Plate date']
                self.data[well]['DIV'] = h5_metadata['DIV'] 
                # self.data[well]['training_exp'] = h5_metadata['Training exp']
                self.data[well]['path_to_h5'] = self.filepath # if we want just the base filename it is os.path.basename(self.filepath).split('.raw')[0]

                # Extract spiking data, sample rate, channel mapping, least significant bit
                self.data[well]['data'] = f[f'/recordings/rec0000/well00{well}/spikes'][:] # [frame_number, channel, amplitude]

                self.data[well]['samp_rate'] = np.array([f[f'/recordings/rec0000/well00{well}/settings/sampling'][:][0]]) # value
                if isinstance(self.data[well]['samp_rate'], list) or isinstance(self.data[well]['samp_rate'], np.ndarray):
                    self.data[well]['samp_rate']=self.data[well]['samp_rate'][0] # TODO: sometimes samp_rate is [samp_rate] and sometimes it's [[samp_rate]]? why
                self.data[well]['mapping'] = f[f'/recordings/rec0000/well00{well}/settings/mapping'][:] # [channel, electrode, x, y]
                self.data[well]['lsb'] = np.array([f[f'/recordings/rec0000/well00{well}/settings/lsb'][:][0]]) # value

                # Extract stimulation electrodes
                self.data[well]['stim_elecs'] = None
                if 'stim_elecs' in f[f'/assay']:
                    stim_elecs_raw = f[f'/assay/stim_elecs'][:] # comma-separated string of stim electrodes
                    self.data[well]['stim_elecs'] = np.array(stim_elecs_raw[0].decode().split(', '), dtype=int)

                # Extract the start frame from the raw recording data (start of recording... not necessary when the first spike or event occurs)
                # TODO: Maxwell may have changed the raw start to start at frame number 0, if this is the case the following code still works but is unnecessary 
                recording_no = 0
                h5_object = f['wells']['well{0:0>3}'.format(well)]['rec{0:0>4}'.format(recording_no)]

                groups = h5_object['groups']
                
                if not groups.keys():
                    print(f"Warning: No raw data for {self.filepath} for well no {well}")
                    self.data[well]['raw_start'] = np.min(self.data[well]['data']['frameno'])  # No groups present, offset by first spike frame so we start at 0
                else:
                    group0 = groups[next(iter(groups))]
                    self.data[well]['raw_start'] = int(group0['frame_nos'][0])

                # Extract the event data 
                events = None
                self.data[well]['event_messages'] = []
                event_keys = []
                if 'events' in f[f'/recordings/rec0000/well00{well}']:
                    events = f[f'/recordings/rec0000/well00{well}/events'][:]
                    
                    for raw_message in events['eventmessage']:

                        try:
                            decoded = raw_message.decode('utf-8').strip()
                            parsed = json.loads(decoded)
                            key = list(parsed.keys())[0]
                            self.data[well]['event_messages'].append(parsed)
                            event_keys.append(key)
                        except Exception as e:
                            print(f"Failed to decode or parse event: {e}")
                
                # # Set training exp variable if not already set
                # if self.data[well]['training_exp'] is None:
                #     if 'closed_loop_start' in event_keys:
                #         self.data[well]['training_exp'] = True
                #     else:
                #         self.data[well]['training_exp'] = False

                self.data[well]['eventtime'] = events['frameno']

                # Extract experimental conditions
                if 'closed_loop_args' in f[f'/assay']:

                    closed_loop_args = f[f'/assay/closed_loop_args'][:][0]
                    closed_loop_args = closed_loop_args.decode('utf-8').strip()

                    if '--left-type' in closed_loop_args and '--right-type' in closed_loop_args:
                        left_stim_list = ast.literal_eval(closed_loop_args.split('--left-type')[-1].split('--')[0])
                        right_stim_list = ast.literal_eval(closed_loop_args.split('--right-type')[-1].split('--')[0])

                        try:
                            left_stim = int(left_stim_list[i])
                            right_stim = int(right_stim_list[i])
                        except: # if there's only one well the experimental conditions may not be in a list 
                            left_stim = int(left_stim_list)
                            right_stim = int(right_stim_list)

                        self.data[well]['experimental_condition'] = np.asarray({'left_stim':left_stim, 'right_stim':right_stim})
                    else:
                        self.data[well]['experimental_condition'] = np.asarray({'left_stim':0, 'right_stim':0}) # no stimulations (controls, network scans)

                    print("well",well, ":", self.data[well]['experimental_condition'])
                    
                    if self.data[well]['exp_id'] == 'stimRemoval':
                        self.data[well]['experimental_condition'] = np.asarray({'left_stim':0, 'right_stim':2}) # only for stim removal

                else:
                    if self.data[well]['exp_id'] == 'stimRemoval':
                        self.data[well]['experimental_condition'] = np.asarray({'left_stim':0, 'right_stim':2}) # only for stim removal
                    else:
                        self.data[well]['experimental_condition'] = np.asarray({'left_stim':0, 'right_stim':0}) # no stimulations (controls, network scans)

    def write_experimental_conditions(self, datastore=None):
        """
        Saves experimental conditions to a csv file for easy access. 

        :param datastore: Directory in with to save experimental conditions. If None, datastore is constants.PARENT_DIR/data/experimental_conditions/. Defaults to None. 
        :type datastore: str
        """

        if datastore is None:

            datastore = Path(constants.PARENT_DIR) / "data" / "experimental_conditions"

        for well in self.data:
        
            new_row = {'chip': self.data[well]['chip'], 'well' : well, 'DIV':self.data[well]['DIV'], 'experimental condition':self.data[well]['experimental_condition']}
            
            chip = self.data[well]['chip']
            savepath = datastore / self.data[well]['exp_id'] / f'{chip}_well{well}_exp_conditions.csv'
            
            if os.path.exists(savepath): # add row if it doesn't already exist
                df = pd.read_csv(savepath)

                new_row_df = pd.DataFrame([new_row])
                row_exists = df.apply(lambda row: all(str(row[col]) == str(new_row[col]) for col in new_row), axis=1).any()
                
                if not row_exists:
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    df.to_csv(savepath, index=False)

            else:
                os.makedirs(datastore / self.data[well]['exp_id'], exist_ok=True)
                df = pd.DataFrame([new_row], columns=['chip', 'well', 'DIV', 'experimental condition'])
                df.to_csv(savepath, mode="w", header=True, index=False)


    def transform(self, amplitude_cutoff=None, refractory_period=constants.REFRACTORY_PERIOD, post_stim_period=constants.POST_STIM_PERIOD, bin_size=constants.BIN_SIZE):
        """
        Performs data cleaning operations such as denoising, applying channel and spike filters, removing stimulation from spike data, and binning spikes.

        :param amplitude_thresh: Minimum spike amplitude. Defaults to constants.AMPLITUDE_THRESH.
        :type amplitude_thresh: float, optional

        :param bin_size: Size of bins in frames for spike binning. Defaults to constants.BIN_SIZE. 
        :type bin_size: float, optional

        :param refractory_frames: Minimum distance between spikes on the same channel. Spikes within refractory_frames of a previous spike are removed. Defaults to constants.REFRACTORY_FRAMES.
        :type refractory_frames: int, optional

        :param post_stim_frames: The number of frames after the end-stimulation event tag that we want to remove from the spike data due to artifacts. Defaults to constants.POST_STIM_FRAMES.
        :type post_stim_frames: int, optional
        """

        self.bin_size = bin_size
        self.amplitude_threshold = amplitude_cutoff
        self.refractory_period = refractory_period
        self.post_stim_period = post_stim_period

        ignore_wells = [] # wells to ignore from analysis if needed (this currently only happens if they have no spikes after applying the filters)

        for well in self.data:

            self.normalize_time(well)

            self.dac_to_voltage(well)   

            self.remove_positive_deflections(well)

            # Spike Filter 1 (remove noise based on amplitude threshold)
            if amplitude_cutoff is not None:
                self.spike_filter(well, amplitude_cutoff)  

            # Spike Filter 2 (remove spikes that occur too close to one another on the same channel)
            if refractory_period is not None:
                self.remove_spurious_spikes(well, refractory_period*self.data[well]["samp_rate"])
            
            # Channel Filter 1 (remove spurious channels)
            self.remove_spurious_channels(well)

            # Check if there are any spikes left after filtering
            if len(self.data[well]['data'])==0:
                print("Well " + str(well) + " has 0 spikes. Removing well " + str(well) + "...")
                ignore_wells.append(well)
                continue

            # TODO: Channel Filter 2 (remove redundant channels)
            # self.remove_redundant_channels()

            self.build_channel_map(well)

            self.remove_stim_frames(well, post_stim_period*self.data[well]["samp_rate"])

            self.bin_spikes(well)

            # Get the length of the recording
            rec_t_samp = np.max(self.data[well]['data']['frameno'])
            self.data[well]['rec_t_sec'] = rec_t_samp / self.data[well]['samp_rate'] # time in seconds from recording start to the last spike recorded

        # remove wells to be ignored
        for well in ignore_wells:
            del self.data[well]
    
    ########################################## PREPROCESSING FUNCTIONS ###############################################################

    def normalize_time(self, well):
        """
        Offset spike frames using the first frame from the raw recording

        :param well: Which well's data to process.
        :type well: int
        """
        # Normalize the time with the first frame from the raw recording
        self.data[well]['data']['frameno'] -= self.data[well]['raw_start']
        self.data[well]['eventtime'] -= self.data[well]['raw_start'] # normalize the event frames by raw recording start

    def dac_to_voltage(self, well):
        """
        Convert DAC amplitudes to voltage.
        Amplitudes are reported in DAC values passed through a finite impluse filter (FIR). Multiply by least significant bit to get voltage.

        :param well: Which well's data to process.
        :type well: int
        """
        self.data[well]['data']['amplitude'] = self.data[well]['data']['amplitude']*self.data[well]['lsb'] # amplitude in voltage
    
    def remove_positive_deflections(self, well):
        """
        Remove positive spikes from the spike data.

        :param well: Which well's data to process.
        :type well: int
        """

        self.data[well]['data'] = self.data[well]['data'][self.data[well]['data']['amplitude']<0] # keep negative deflections only
    
    def spike_filter(self, well, amp_thresh):
        """
        Denoising spike data by filtering out spikes with an amplitude below amp_thresh.

        :param well: Which well's data to process.
        :type well: int

        :param amp_thresh: Minimum required amplitude for spikes in voltage. 
        :type amp_thresh: float

        :raises ValueError: If the provided amplitude threshold is so high such that it removes all spikes a ValueError will be raised. 
        """
        # Spike Filter 
        # Threshold the spiking data using a min amplitude cutoff to remove noise

        print(f'Applying amplitude filter with threshold of {amp_thresh*1e6} µV...')
        spike_thresh_mask = np.abs(self.data[well]['data']['amplitude']) >= amp_thresh  # Account for both positive and negative spike amplitudes
        self.data[well]['data'] = self.data[well]['data'][spike_thresh_mask]

        if len(self.data[well]['data'])==0:
            raise ValueError("Amplitude threshold is too high.")
        
    def remove_spurious_spikes(self, well, refractory_frames):
        """
        Filters out spikes that are within refractory_frames of each other on the same channel. 

        :param well: Which well's data to process.
        :type well: int

        :param refractory_frames: Minimum distance between spikes on the same channel. Spikes within refractory_frames of a previous spike are removed. Defaults to constants.REFRACTORY_FRAMES.
        :type refractory_frames: int
        """

        df = pd.DataFrame(self.data[well]['data'])
        grouped = df.groupby('channel')
        
        groups = []


        for name, group in grouped:

            # print(name)
            # print(group)

            inds = np.where(np.diff(group['frameno'])<refractory_frames)[0]

            while len(inds)!=0:

                mask = np.ones(shape=len(group['frameno']), dtype=int)
                for ind in inds:
                    if np.array(group['amplitude'])[ind]<np.array(group['amplitude'])[ind+1]: # keep the spike with the smaller amplitude 
                        mask[ind+1]=0
                    else:
                        mask[ind]=0

                group = group[mask.astype(bool)]

                inds = np.where(np.diff(group['frameno'])<refractory_frames)[0]

            groups.append(group)

        filtered = pd.concat(groups).sort_index()
        
        # Convert back into a numpy structured array
        self.data[well]['data'] = filtered.to_records(index=False)
        
    def remove_spurious_channels(self, well):
        """
        Data is recorded from all channels whether or not they are included in the config but it's not clear which electrodes they are recording from. 
        We spikes on channels that are not included in the channelmap (specified by the config).

        :param well: Which well's data to process.
        :type well: int
        """        
        # Check channel map for duplicates - this should never happen
        if len(self.data[well]['mapping']['channel']) != len(np.unique(self.data[well]['mapping']['channel'])):
            print("Duplicate channels found.")

        # Apply channel filter - gets rid of data with channels that aren't included in the channel map (mapping of channel to electrode to x,y loc)
        data_mask = np.isin(self.data[well]['data']['channel'], self.data[well]['mapping']['channel'])
        self.data[well]['data'] = self.data[well]['data'][data_mask]

    def remove_redundant_channels(self):
        # Francesca's code here!
        pass

    def remove_stim_frames(self, well, post_stim_frames):

        print("Removing stimulation frames from spike data...")
        event_df = pd.DataFrame({'eventtime': self.data[well]['eventtime'], 'eventmessage': self.data[well]['event_messages']})

        if len(event_df)==0:
            self.data[well]['stim_frames'] = []
            return
        
        starts = event_df[event_df["eventmessage"].apply(lambda d: 'start_stimulation' in d)]["eventtime"].to_numpy()
        stops = event_df[event_df["eventmessage"].apply(lambda d: 'end_stimulation' in d)]["eventtime"].to_numpy()

        if len(starts)==0:
            self.data[well]['stim_frames'] = []
            return

        if stops[0]<starts[0]: # there's an end stim before the first start stim
            stops = stops[1:] # remove the first stop

        assert len(starts) == len(stops)
        
        self.data[well]['stim_frames'] = np.concatenate([
                        np.arange(s, e + post_stim_frames, dtype=np.int64)
                            for s, e in zip(starts, stops)
                        ])

        mask = ~np.isin(self.data[well]['data']['frameno'], self.data[well]['stim_frames'])
        self.data[well]['data'] = self.data[well]['data'][mask]

    def build_channel_map(self, well):
        """
        Builds a channel map array specifying the layout of electrodes and the electrode-to-channel mapping. 
        The resulting channel map is in the format (num_channels x 5) with columns ('index', 'channel' ,'electrode', 'x', 'y').
        X and y positions are given in micrometers.

        :param well: Which well's data to process.
        :type well: int
        """
        # Build a channel map array
        num_chan = len(self.data[well]['mapping']['channel'])
        self.data[well]['channelmap'] = np.zeros((num_chan, 5))
        self.data[well]['channelmap'][:, 0] = np.arange(0, num_chan)  # This will correspond to the row index in spike_bin
        self.data[well]['channelmap'][:, 1] = self.data[well]['mapping']['channel']  # Channel from h5 mapping
        self.data[well]['channelmap'][:, 2] = self.data[well]['mapping']['electrode'] # Electrode from h5 mapping
        self.data[well]['channelmap'][:, 3] = self.data[well]['mapping']['x']
        self.data[well]['channelmap'][:, 4] = self.data[well]['mapping']['y']

    def bin_spikes(self, well):
        """
        Bins spikes from clean data into the specified bin size.
        
        :param well: Which well's spike data to process.
        :type well: int
        """

        # Sets bin windows (in frames) based on desired bin size
        bin_win = self.bin_size * self.data[well]['samp_rate']  # In samples
        rec_t_samp = np.max(self.data[well]['data']['frameno'])

        num_bins = int((np.ceil(np.divide(rec_t_samp, bin_win))))

        # Spike_bin is the output
        num_chan = self.data[well]['channelmap'].shape[0]
        self.data[well]['spike_bin'] = np.zeros((num_chan, num_bins))
        which_bin = (self.data[well]['data']['frameno'] / bin_win).astype(int)
        which_bin = np.clip(which_bin, 0, num_bins - 1)  # Ensure indices are within bounds

        print("Binning Spikes...")
        # for i in np.arange(len(which_bin)): # iterate through the spikes
        #     if self.spike_data['channel'][i] in self.channelmap[:, 1]: 
        #         channel_id = self.channelmap[np.where(self.channelmap[:, 1] == self.spike_data['channel'][i])[0], 0][0].astype(int) # index in channelmap
        #         self.spike_bin[channel_id, which_bin[i]] = 1

        # Vectorized spike binning
        mapping = dict(zip(self.data[well]['channelmap'][:,1], self.data[well]['channelmap'][:,0])) # create a mapping
        channel_ids = pd.Series(self.data[well]['data']['channel']).map(mapping).to_numpy(dtype=int)

        assert len(channel_ids)==len(which_bin)

        self.data[well]['spike_bin'][channel_ids, which_bin] = 1


    ##################################################################################################################################

    def get_unique_path(self, path: Path):
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            if counter < 10:
                new_path = parent / f"{stem}_0{counter}{suffix}"
            else:
                new_path = parent / f"{stem}_{counter}{suffix}"

            if not new_path.exists():
                return new_path
            
            counter += 1

    def load(self, datastore=None, overwrite=True):
        """
        Saves cleaned and transformed data to the specified data storage location. Data is saved as a collection of npz files (one npz per well per recording).
        Each npz file contains a numpy structured array. 

        :param datastore: Directory in which to save the clean and transformed data. If None, datastore will be constants.PARENT_DIR/data/preprocessed. Defaults to None.
        :type datastore: str, optional

        :param overwrite: If the path to the cleaned and transformed data already exists for the well/recording 
        overwrite specifies whether or not to overwrite the existing file or to add a suffix indicating the number of files that exist. Defaults to True.
        :type overwrite: bool, optional

        :return: List of filepaths to the npz files. 
        :rtype: list
        """

        if not datastore:
            # Construct the full savepath
            datastore = Path(constants.PARENT_DIR) / "data" / "preprocessed"

            os.makedirs(datastore, exist_ok=True)  # Create directory if it doesn't exist

        clean_data_filepaths = []

        for well in self.data:

            exp_id = self.data[well]['exp_id']
            chip = self.data[well]['chip']
            DIV = self.data[well]['DIV']
            plate_date = self.data[well]['plate_date']

            # Construct the full savepath
            
            os.makedirs(datastore / exp_id / chip / f'well{well}', exist_ok=True)  # Create directory if it doesn't exist
            clean_data_path =  datastore / exp_id / chip / f'well{well}' / f'DIV{DIV}_{plate_date}_{chip}_{exp_id}_well{well}_exp_data.npz'

            # temp for elena data
            # clean_data_path =  datastore / f'{exp_id}_exp_data.npz'

            if not overwrite:
                clean_data_path = self.get_unique_path(Path(clean_data_path))


            np.savez_compressed(clean_data_path, spike_data=self.data[well]['data'], channelmap=self.data[well]['channelmap'], 
                                stim_elecs=self.data[well]['stim_elecs'], rec_t_sec=np.array([self.data[well]['rec_t_sec']]), 
                                samp_rate=np.array([self.data[well]['samp_rate']]), lsb=np.array(self.data[well]['lsb']), 
                                eventtime=self.data[well]['eventtime'], event_messages=self.data[well]['event_messages'], 
                                spike_bin=self.data[well]['spike_bin'], bin_size=self.bin_size, 
                                exp_condition=self.data[well]['experimental_condition'], DIV=DIV, plate_date=plate_date, 
                                well=well, chip=chip, exp_id=exp_id, stim_frames=self.data[well]['stim_frames'], 
                                raw_start=self.data[well]['raw_start'], path_to_h5=self.data[well]['path_to_h5'],
                                amplitude_threshold=self.amplitude_threshold, refractory_period=self.refractory_period,
                                post_stim_period=self.post_stim_period)
            
            print(f"Transformed data saved to: {clean_data_path}")
                        
            clean_data_filepaths.append(clean_data_path)

        return clean_data_filepaths
    

    def load_HAL_burst_csvs(self, datastore=None):
        """
        Filters and saves real-time HAL burst data for a recording. 

        :param datastore: Directory in which to save the HAL burst data. If None, datastore will be constants.PARENT_DIR/data/HAL_burst_data. Defaults to None. 
        :type datastore: str, optional
        """
   
        if not datastore:
            # Construct the full savepath
            datastore = Path(constants.PARENT_DIR) / "data" / "HAL_burst_data"

        os.makedirs(datastore, exist_ok=True)  # Create directory if it doesn't exist

        for well in self.data:

            if self.data[well]["exp_id"] == "waveTraining":

                try:
                    hal_burst_csv_path = glob(str(Path(self.filepath).parent / "*burst*.csv"))[0]
                    hal_bursts = pd.read_csv(hal_burst_csv_path)
                except:
                    print("No HAL burst data found.")
                    return
                
                sys_start_buffer = 50000 # frames
                channel_window = 200 # frames
                
                # take the PeakFrame and subtract 50200 and add the the train start frames (found from the event data)

                # also take the peak loc, and direction, burst start frame (don't think this exists), peak spikes, frames above threshold

                hal_bursts["PeakFrame"] = hal_bursts["PeakFrame"] - sys_start_buffer - channel_window
                
                hal_bursts_filtered = hal_bursts[['PeakFrame', 'PeakLoc', 'PeakSpikes', 'Direction', 'FramesAboveThreshold']]


            else:

                try:
                    hal_burst_csv_path = glob(str(Path(self.filepath).parent / f"DIV{self.data[well]['DIV']}*burst*.csv"))[0]
                    hal_bursts = pd.read_csv(hal_burst_csv_path)
                except:
                    print("No HAL burst data found.")
                    return
                
                # Filter by well
                hal_bursts = hal_bursts[hal_bursts['WellID']==well].copy().reset_index()
                
                # Subtract raw start
                hal_bursts["BurstStartFrame"] = hal_bursts["BurstStartFrame"] - self.data[well]['raw_start']
                hal_bursts["PeakFrame"] = hal_bursts["PeakFrame"] - self.data[well]['raw_start']

                hal_bursts_filtered = hal_bursts[['BurstStartFrame', 'PeakFrame', 'PeakLoc', 'PeakSpikes', 'Direction', 'FramesAboveThreshold']]


            # Save the hal burst csv

            exp_id = self.data[well]['exp_id']
            chip = self.data[well]['chip']
            DIV = self.data[well]['DIV']
            plate_date = self.data[well]['plate_date']

            os.makedirs(datastore / exp_id / chip / f'well{well}', exist_ok=True)  # Create directory if it doesn't exist

            hal_burst_savepath =  datastore / exp_id / chip / f'well{well}' / f'DIV{DIV}_{plate_date}_{chip}_{exp_id}_well{well}_HAL_bursts.csv'

            hal_bursts_filtered.to_csv(hal_burst_savepath)

    def register(self, registry_path=None):

        # TODO; this function needs to be tested the next time data is processed
        # TODO: add a status argument which tells you whether processing was successful or if it failed

        if registry_path is None:
            REGISTRY_PATH = Path(constants.PARENT_DIR)/"data"/"registry.csv"
        else:
            REGISTRY_PATH = registry_path

        if REGISTRY_PATH.exists():
            df = pd.read_csv(REGISTRY_PATH)
        else:
            df = pd.DataFrame()

        for well in self.data:
            new_row = {
                "exp_id": self.data[well]['exp_id'],
                "chip": self.data[well]['chip'],
                "well": well,
                "div": self.data[well]['DIV'],
                # "status": status,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            if not df.empty:
                mask = (
                    (df.exp_id == self.data[well]['exp_id']) &
                    (df.chip == self.data[well]['chip']) &
                    (df.well == well) &
                    (df.div == self.data[well]['DIV'])
                )
                df = df[~mask]
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row]) # overwrite the empty dataframe 

        df.to_csv(REGISTRY_PATH, index=False)
         
