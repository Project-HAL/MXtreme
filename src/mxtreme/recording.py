'''
Class for a single recording.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage import gaussian_filter1d

import pandas as pd
import pickle
import scipy
import seaborn as sns
import ast
from pathlib import Path
import os
import json
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation


from mxtreme.burst import BurstDetector, Burst, ISIThreshBurst, RateThreshBurst
from mxtreme import visualizations as viz
from mxtreme import utils
from mxtreme import constants

class Recording:
    """
    Class to handle a single recording session including burst detection, feature extraction, plotting, and more.
    """

    def __init__(self, id, exp_data, burst_list=None, burst_csv=None, smooth=False) -> None:
        """
        Generates a Recording object from experimental data from a numpy structured array (the output from the preprocessing pipeline).

        :param id: The recording id.
        :type id: int
        :param exp_data: Preprocessed experimental data.
        :type exp_data: dict
        :param burst_list: A list of Burst objects, defaults to None. If no burst_list or burst_csv has been supplied, then burst detection needs to be run. If a burst list is supplied, it can be used to generate the burst csv without needing to re-run burst detection. This is helpful when you want to run burst detection and burst feature extraction in to separate steps.
        :type burst_list: list, optional
        :param burst_csv: Path to a burst data csv, defaults to None. Supplies the detection bursts and their features for the recording. 
        :type burst_csv: str, optional
        :param smooth: Whether or not to smooth the array-wide binned spike signal with a Gaussian filter. Defaults to True. 
        :type smooth: bool, optional
        """
        self.id = id

        # Unpack experiment info
        self.exp_id = exp_data['exp_id'].item()
        self.chip = exp_data['chip'].item()
        self.well = exp_data['well'].item()
        self.DIV = exp_data['DIV'].item()
        self.plate_date = exp_data['plate_date'].item()
        self.raw_start = exp_data['raw_start'].item()
        self.path_to_h5 = exp_data['path_to_h5'].item()

        # Unpack experimental data
        self.spike_data = exp_data['spike_data']
        self.channelmap = exp_data['channelmap'] # index, channel, electrode, x, y
        self.stim_elecs = exp_data['stim_elecs']
        self.rec_t_sec = exp_data['rec_t_sec'][0]
        self.samp_rate = exp_data['samp_rate'][0]
        self.lsb = exp_data['lsb'][0]
        self.eventtime = exp_data['eventtime']
        self.event_messages = exp_data['event_messages']
        self.spike_bin = exp_data['spike_bin']
        self.bin_size = exp_data['bin_size']
        self.stim_frames = exp_data['stim_frames']
        self.amplitude_thresh=exp_data['amplitude_threshold'].item()
       
        try:
            self.refractory_period=exp_data['refractory_period'].item()
        except:
            self.refractory_period = exp_data['refractory_frames'].item()/self.samp_rate # older versions passed in the refractory period in frames   

        try:
            self.post_stim_period=exp_data['post_stim_period'].item()
        except:
            self.post_stim_period = exp_data['post_stim_frames'].item()/self.samp_rate

        # Handle bursts if provided
        self.bursts = burst_list # list of bursts is output of burst detection and the input for burst feature extraction
        if burst_csv is not None:
            if Path(burst_csv).exists():
                self.burst_df = pd.read_csv(burst_csv)
            else:
                raise FileNotFoundError()
        

        self.asdr = self.spike_bin.mean(axis=0) # array-wide spike detection rate

        # self.estimate_noise()

        self.event_df = pd.DataFrame({'eventtime': self.eventtime, 'eventmessage': self.event_messages})

        self.epoch_frames, self.epochs = self.find_epoch_transitions() # pre/training/post epochs

        # Other stuff
        self.burst_detection_params = None
        self.num_ignored = None
        self.burst_feature_params = None
        self.IBI = None
        self.bursting_rates = None
        self.channel_ISI = None
        self.channel_firing_rates = None
        self.pop_firing_rates = None
        self.pop_ISI = None

    # def estimate_noise(self):
    #     # Calculate the MAD of the entire signal
    #     mad_value = median_abs_deviation(self.asdr)

    #     # Convert MAD to an estimated standard deviation 
    #     # (for normally distributed noise, scale by ~1.4826)
    #     noise_estimate_mad_std = mad_value * 1.4826 

    #     self.noise = noise_estimate_mad_std

    def interpolate_stim_bins(self, signal):

        all_stim_bins = (self.stim_frames/self.samp_rate/self.bin_size).astype(int)

        stim_mask = np.zeros_like(signal, dtype=bool)
        stim_mask[all_stim_bins] = True

        x = np.arange(len(signal))

        f = interp1d(x[~stim_mask],
                    signal[~stim_mask],
                    kind="linear",     # or "cubic"
                    fill_value="extrapolate"
                )

        signal_filled = signal.copy()
        signal_filled[stim_mask] = f(x[stim_mask])
        return signal_filled


    def find_epoch_transitions(self) -> dict|None:
        """
        Find the frames were we transition between different periods (pre/train/post) of the experiment based on the event data.

        :return: A dictionary of the epochs and the start frame of the epoch.
        :rtype: dict|None
        """

        # TODO: Get rid of epoch_frames altogether but need to make sure everything works with out it

        all_keys = set().union(*self.event_df["eventmessage"])
        if {'pre_recording_start', 'closed_loop_start', 'post_recording_start'}.issubset(all_keys):

            pre_start_frame = self.event_df[self.event_df["eventmessage"].apply(lambda d: 'pre_recording_start' in d)]["eventtime"].iloc[0]
            train_start_frame = self.event_df[self.event_df["eventmessage"].apply(lambda d: 'closed_loop_start' in d)]["eventtime"].iloc[0]
            post_start_frame = self.event_df[self.event_df["eventmessage"].apply(lambda d: 'post_recording_start' in d)]["eventtime"].iloc[0]
            end_experiment_frame = self.event_df[self.event_df["eventmessage"].apply(lambda d: 'end_experiment' in d)]["eventtime"].iloc[0]
            
            epoch_frames = {'pre_start_frame':pre_start_frame,
                           'train_start_frame':train_start_frame,
                           'post_start_frame':post_start_frame
                           }        
            epochs = {'pre' : [pre_start_frame, train_start_frame],
                      'train':[train_start_frame, post_start_frame],
                      'post' : [post_start_frame, end_experiment_frame]}
        # else:
        #     epoch_frames = None # network scans, short recordings, etc.. 

        # The below is for stimRemoval but for all future experiments if there are no phase tags then we just leave the recording whole. 
        
        elif self.exp_id=='stimRemoval' or self.exp_id=='stimRemovalNull':
            # Split into 20 min increments starting from the end
            phase_length = 20*60*self.samp_rate # default is 20 mins per phase
            num_frames = np.max(self.spike_data['frameno'])

            # work backwards from end because recordings end pretty much when the post-training phase ends but may have more stuff going on in the beginning
            post_start_frame = int(num_frames-phase_length)
            train_start_frame = int(post_start_frame-phase_length)
            pre_start_frame = int(train_start_frame-phase_length)

            epoch_frames = {'pre_start_frame':pre_start_frame,
                'train_start_frame':train_start_frame,
                'post_start_frame':post_start_frame
                }      
            
            epochs = {'pre' : [pre_start_frame, train_start_frame],
                    'train':[train_start_frame, post_start_frame],
                    'post' : [[post_start_frame, end_experiment_frame]]}

        else:
            epoch_frames = None

            epochs = {'full':[np.min(self.spike_data["frameno"]), np.max(self.spike_data["frameno"])]}
            

        return epoch_frames, epochs

    
    def detect_bursts(self, method='ISI_N', **kwargs):
        
        bd = BurstDetector(method=method)
        self.bursts = bd.detect_bursts(self.spike_data, self.spike_bin, self.epoch_frames, 
                                       self.samp_rate, self.bin_size, **kwargs)
        self.burst_detection_params = bd.burst_detection_parameters

        # Create burst_df with the detection bursts
        if len(self.bursts)!=0:
            burst_dicts = [b.__dict__ for b in self.bursts]
            self.burst_df = pd.DataFrame(burst_dicts)
        else:
            # Create an empty burst to save the dataframe
            empty_burst = ISIThreshBurst(0, None, None, None, None, None, None, None)
            self.burst_df = pd.DataFrame(columns=empty_burst.__dict__.keys())

    def convert_burst_csv_to_burst_list(self):
        '''
        Takes the recording's burst CSV and generates a list of burst objects.
        '''

        if len(self.burst_df)==0: # no bursts
            self.bursts = []

        # Different types of bursts
        ISI_N_attrs = ['id', 't_peak_bin', 'peak_bin_amp', 'phase', # needed for ISI_N bursts 
                     'ISI_N_start_frame', 'ISI_N_stop_frame',
                     'ISI_N_size_in_spikes','type']
        
        RateThresh_attrs = ['id', 't_peak_bin', 'peak_bin_amp', 'phase'] # needed for rate thresh brusts

        # Convert bursts df to burst objects
        if set(ISI_N_attrs).issubset(set(self.burst_df.columns)):
            df = self.burst_df[ISI_N_attrs]
            self.bursts = [ISIThreshBurst(**d) for d in df.to_dict(orient='records')]
        elif set(RateThresh_attrs).issubset(set(self.burst_df.columns)):
            df = self.burst_df[RateThresh_attrs]
            self.bursts = [RateThreshBurst(**d) for d in df.to_dict(orient='records')]
        else:
            raise ValueError(f"Burst CSVs must include either ISI-N attributes: {ISI_N_attrs} or rate-thresh attributes: {RateThresh_attrs}")
        


    def compute_burst_features(self, onset_thresh_pct, offset_thresh_pct):

        self.burst_feature_params = {'onset_thresh_pct':onset_thresh_pct,
                                'offset_thresh_pct':offset_thresh_pct}
        
        if len(self.burst_df)==0: # no bursts
            self.num_ignored = 0
            return

        print(self.exp_id, self.chip, self.well, self.DIV)

        if self.stim_frames is not None:
            self.asdr_filled = self.interpolate_stim_bins(self.asdr) # useful for some of the features that get messed up with the stimulation artifacts
        else:
            self.asdr_filled=None

        if not self.bursts:
            self.convert_burst_csv_to_burst_list()

        # print("The recording has ", len(self.bursts), "bursts.")
        records = []

        self.num_ignored = 0

        for burst in self.bursts:

            feature_dict = burst.compute_burst_characteristics(self.asdr, self.spike_data, self.stim_frames, 
                                                self.channelmap, self.samp_rate, self.bin_size, onset_thresh_pct, 
                                                offset_thresh_pct, self.asdr_filled)
            
            if not burst.ignore:
                records.append(feature_dict)
            else:
                self.num_ignored += 1
        
        self.burst_df = pd.DataFrame(records)

    def save_burst_data(self, datastore=None):

        if not datastore:
            # Construct the full savepath
            datastore = Path(constants.PARENT_DIR) / "data" / "burst_data"

            os.makedirs(datastore, exist_ok=True)  # Create directory if it doesn't exist

        # Construct the full savepath

        os.makedirs(datastore / self.exp_id / self.chip / f'well{self.well}', exist_ok=True)  # Create directory if it doesn't exist

        burst_save_path =  datastore / self.exp_id / self.chip / f'well{self.well}' / f'DIV{self.DIV}_{self.plate_date}_{self.chip}_{self.exp_id}_well{self.well}_burst_data.csv'

        self.burst_df.to_csv(burst_save_path, index=False)

        return burst_save_path

    def save_burst_metadata(self, burst_log_path=None, overwrite=False):

        # Create a new row for the burst log 
        if self.burst_detection_params:
            if len(self.burst_df)==0:
                new_row = {'exp_id':self.exp_id, 'chip':self.chip, 'well':self.well, 'DIV':self.DIV, 
                    'plate_date':int(self.plate_date), 'num_total_bursts':len(self.bursts),
                    'num_mini_bs':0, 
                    'num_HAL_like':0, 
                    'median_HALlike_IBI':0,
                    'detection_params':json.dumps(self.burst_detection_params)}

            else:
                new_row = {'exp_id':self.exp_id, 'chip':self.chip, 'well':self.well, 'DIV':self.DIV, 
                'plate_date':int(self.plate_date), 'num_total_bursts':len(self.bursts),
                'num_mini_bs':np.sum(self.burst_df['type']=='mini_b'), 
                'num_HAL_like':np.sum(self.burst_df['type']=='HAL_like'), 
                'median_HALlike_IBI':np.median(np.diff(self.burst_df[self.burst_df["type"]=="HAL_like"]['t_peak_bin']*self.bin_size)),
                'detection_params':json.dumps(self.burst_detection_params)}
        else:
            new_row = None # don't update this info

        # Add to the burst log or create a new one if it doesn't exist yet

        if not burst_log_path:
            burst_log_path = Path(constants.PARENT_DIR) / "data" / "burst_data" / f"{self.exp_id}_burst_log.csv"
        
        print(burst_log_path)

        if burst_log_path.exists():
            print('path_exists')

            burst_log = pd.read_csv(burst_log_path)
            # Boolean mask for this specific row
            mask = (
                (burst_log['exp_id'] == self.exp_id) &
                (burst_log['plate_date'] == int(self.plate_date)) &
                (burst_log['chip'] == self.chip) &
                (burst_log['well'] == int(self.well)) &
                (burst_log['DIV'] == int(self.DIV))
            )

            if mask.any(): # if any of the rows are a match

                if new_row:
                    # Update existing row
                    for col, value in new_row.items():
                        burst_log.loc[mask, col] = value

                row_index = burst_log.index[mask][0]

            else:

                if new_row:
                    # Append new row
                    burst_log = pd.concat(
                        [burst_log, pd.DataFrame([new_row])],
                        ignore_index=True
                    )
                    row_index = len(burst_log) - 1
                else:
                    print("Burst detection has not been run on recording.")
                    return
        else:
            if new_row:
                # Create file with headers and first row
                burst_log = pd.DataFrame([new_row])
                row_index = 0
            else:
                print("Burst detection has not been run on recording.")
        
        if self.burst_feature_params:
            if "feature_extraction_params" not in burst_log.columns:
                burst_log["feature_extraction_params"] = pd.Series(index=burst_log.index, dtype="object")
                burst_log["num_ignored_bursts"] = pd.Series(index=burst_log.index, dtype="Int64")

            burst_log.loc[row_index, "num_ignored_bursts"] = int(self.num_ignored)
            burst_log.loc[row_index, "feature_extraction_params"] = json.dumps(self.burst_feature_params)    
        else:
            if new_row and "feature_extraction_params" in burst_log.columns:
                burst_log.loc[row_index, "num_ignored_bursts"] = pd.NA
                burst_log.loc[row_index, "feature_extraction_params"] = json.dumps({})


        # Save burst log
        burst_log.to_csv(burst_log_path, index= False)

    def pickle_bursts(self, datastore):
        # This is for checkpointing in case you want to detect bursts and compute burst features in two separate stages.
        with open(datastore, 'wb') as f:
            pickle.dump(f, self.bursts)
    
    def pickle_recording(self, datastore=None):

        if datastore is None and self.datastore is None:
            raise ValueError('Need to provide a save path.')
        if self.datastore is None: # store the path to the recording
            self.datastore = datastore 
        
        with open(self.datastore, 'wb') as f:
            pickle.dump(self, f)

    def compute_IBI_bursting_rates(self):
        '''
        Compute time (in seconds) between bursts split by phase if the recording was from an experiment with training. 
        '''

        self.IBI = {}
        self.bursting_rates = {}
 
        if self.training_exp:
            # compute IBI and bursting rate by phase
            phases = ['pre', 'train', 'post']
            phase_dur_frames = [self.train_start_frame-self.pre_start_frame, self.post_start_frame-self.train_start_frame, np.max(self.spike_data['frameno'])-self.post_start_frame]

            for i, phase in enumerate(phases):

                phase_df = self.burst_df[self.burst_df['phase']==phase]
                self.IBI[phase] = np.diff(phase_df['t_peak_sec'])

                self.bursting_rates[phase] = len(phase_df) / (phase_dur_frames[i]/self.samp_rate) # num bursts per sec


        else:
            # compute IBI and bursting rate over the entire recording 
            self.IBI['full'] = np.diff(self.burst_df['t_peak_sec'])
            
            self.bursting_rates['full'] = len(self.burst_df) / (np.max(self.spike_data['frameno'])/self.samp_rate)


    def compute_ISI_firing_rates(self):
        '''
        Compute population and channel-wise firing rate (in Hz) and ISI (sec) split by phase for training experiments. 
        '''

        self.pop_firing_rates = {}
        self.pop_ISI = {}

        self.channel_firing_rates = {}
        self.channel_ISI = {}
 
        if self.training_exp:
            # compute firing rates and ISI by phase

            ranges = {'pre':(self.pre_start_frame, self.train_start_frame), 'train':(self.train_start_frame, self.post_start_frame), 'post':(self.post_start_frame, np.max(self.spike_data['frameno']))}

            for phase in ['pre', 'train', 'post']:

                # population firing rate and ISI
                phase_spike_data = self.spike_data[(self.spike_data['frameno']>=ranges[phase][0])&(self.spike_data['frameno']<ranges[phase][1])]
                self.pop_firing_rates[phase] = len(phase_spike_data)/((ranges[phase][1]- ranges[phase][0])/self.samp_rate)
                self.pop_ISI[phase] = np.diff(phase_spike_data['frameno'])/self.samp_rate

                # channel-wise firing rate and ISI
                phase_spike_df = pd.DataFrame(phase_spike_data)
                grouped_by_chan = phase_spike_df.groupby('channel')

                self.channel_firing_rates[phase] = list(grouped_by_chan['frameno'].apply(lambda x: len(x)/((np.max(x)-np.min(x))/self.samp_rate) if len(x) > 1 else np.nan))
                # self.channel_ISI[phase] = list(grouped_by_chan['frameno'].apply(lambda x: np.median(np.diff(x))/self.samp_rate if len(x) > 1 else np.nan)) # this is unfiltered channel ISI 
                
                # filtering out ISIs > 200 ms
                self.channel_ISI[phase] = []
                for channel, group_df in grouped_by_chan:
                    ISI_ms = np.diff(group_df['frameno'])/self.samp_rate*1000
                    ISI_filtered = ISI_ms[ISI_ms<constants.ISI_threshold]
                    if len(ISI_filtered) > 0:
                        self.channel_ISI[phase].append(np.median(ISI_filtered))
        else:
            # population firing rate and ISI
            self.pop_firing_rates['full'] = len(self.spike_data)/((np.max(self.spike_data['frameno'])-np.min(self.spike_data['frameno']))/self.samp_rate)
            self.pop_ISI['full'] = np.diff(self.spike_data['frameno'])/self.samp_rate

            # average channel firing rate and ISI
            spike_df = pd.DataFrame(self.spike_data)
            grouped_by_chan = spike_df.groupby('channel')

            self.channel_firing_rates['full'] = list(grouped_by_chan.apply(lambda x: len(x)/((np.max(x['frameno'])-np.min(x['frameno']))/self.samp_rate) if len(x) > 1 else np.nan))
            # self.channel_ISI['full'] = list(grouped_by_chan['frameno'].apply(lambda x: np.median(np.diff(x))/self.samp_rate if len(x) > 1 else np.nan))

            # filtering out ISIs > 200 ms
            self.channel_ISI['full'] = []
            for channel, group_df in grouped_by_chan:
                ISI_ms = np.diff(group_df['frameno'])/self.samp_rate*1000
                ISI_filtered = ISI_ms[ISI_ms<constants.ISI_threshold]
                if len(ISI_filtered) > 0:
                    self.channel_ISI['full'].append(np.median(ISI_filtered))

    def compute_interStimInterval(self, phase=None):
        pass
    
    def compute_time_in_state(self, state, phase=None):
        pass

    def get_bursts_as_vectors(self, phase=None):

        chip_ht_um = constants.CHIP_HEIGHT * constants.ELEC_SIZE


        hal_burst_df = self.burst_df[self.burst_df['type']=='HAL_like']

        if len(hal_burst_df)==0:
            return []
            
        if self.epoch_frames and phase is not None:
            hal_burst_df = hal_burst_df[hal_burst_df['phase']==phase]

        vector_data = hal_burst_df[['origin_x', 'origin_y', 'peak_x', 'peak_y']].values

        return vector_data

    # VISUALIZATIONS
    # ------------------------------------------------------------------------------
            
    def plot_asdr(self, save_path, save_filename, burst_thresh=None):

        if save_path is not None:
            SAVE_PATH = Path(save_path) # make sure the save path is a pathlib path
        else:
            SAVE_PATH=None

        if save_filename is None:
            save_filename=''

        hal_burst_df = self.burst_df[self.burst_df['type']=='HAL_like']

        # Plot ASDR with annotated bursts
        if self.epoch_frames:
            zoompre = (utils.frame_to_bin(self.epoch_frames["pre_start_frame"], self.samp_rate, self.bin_size),utils.frame_to_bin(self.epoch_frames["train_start_frame"], self.samp_rate, self.bin_size))
            viz.plot_asdr(self.spike_bin, title='Pre - Binned Spikes with Bursts', zoom=zoompre, save_path=SAVE_PATH, savefilename=save_filename+'_pre_bursts.png', annotate_bursts=True, bursts=hal_burst_df, burst_threshold=burst_thresh)

            zoomtrain = (utils.frame_to_bin(self.epoch_frames["train_start_frame"], self.samp_rate, self.bin_size),utils.frame_to_bin(self.epoch_frames["post_start_frame"], self.samp_rate, self.bin_size))
            viz.plot_asdr(self.spike_bin, title='Train - Binned Spikes with Bursts', zoom=zoomtrain, save_path=SAVE_PATH, savefilename=save_filename+'_train_bursts.png', annotate_bursts=True, bursts=hal_burst_df, burst_threshold=burst_thresh)

            zoompost = (utils.frame_to_bin(self.epoch_frames["post_start_frame"], self.samp_rate, self.bin_size), None)
            viz.plot_asdr(self.spike_bin, title='Post - Binned Spikes with Bursts', zoom=zoompost, save_path=SAVE_PATH, savefilename=save_filename+'_post_bursts.png', annotate_bursts=True, bursts=hal_burst_df, burst_threshold=burst_thresh)
        else:
            viz.plot_asdr(self.spike_bin, title='Binned Spikes with Bursts', save_path=SAVE_PATH, savefilename=save_filename+'_bursts.png', annotate_bursts=True, bursts=hal_burst_df, burst_threshold=burst_thresh)


    def plot_origin_heatmap(self, save_path=None, save_filename=None, phase=None, c='r', title=None, kde_cmap='Blues', ret=False, plot_axes=True, weighted=False):
        # Origin heatmap

        SAVE_PATH = Path(save_path) # make sure the save path is a pathlib path

        # create own figure 
        fig, ax = plt.subplots(1, 1, figsize=(9,5))

        chip_ht_um = constants.CHIP_HEIGHT * constants.ELEC_SIZE

        # Plot heatmap using the electrodes that are spiking at the onset of the burst

        x_origin = []
        y_origin = []

        hal_burst_df = self.burst_df[self.burst_df['type']=='HAL_like']

        for idx, burst in hal_burst_df.iterrows():
            
            if self.epoch_frames and phase is not None:
                if burst['phase'] != phase:
                    continue
            
            try:
                # channels_spiking = [int(x.strip()) for x in burst['channels_spiking_at_origin'][1:-1].split(",")] # if csv is read in it will  be a list of strings
                channels_spiking = list(dict(ast.literal_eval(burst['origin_chan_counts'])).keys()) # if csv is read in it will  be a list of strings
                if len(channels_spiking)==0:
                    continue
            except: # TODO: how to handle instantaneous bursts that don't have an origin
                continue
            #     channels_spiking = list(burst['channels_spiking_at_origin_counts'].keys())

            mask = np.isin(self.channelmap[:,1], channels_spiking)
            chanmap_slice = self.channelmap[mask]

            xs = chanmap_slice[:,3]
            ys = chip_ht_um - np.array(chanmap_slice[:,4]) # invert y-values

            sns.kdeplot(x=xs, y=ys, fill=True, alpha=0.2, cmap=kde_cmap)

            if weighted:
                x_origin.append(int(burst['weighted_origin_x']))
                y_origin.append(int(burst['weighted_origin_y']))
            else:
                x_origin.append(int(burst['origin_x']))
                y_origin.append(int(burst['origin_y']))

        viz.MEA(ax, self.channelmap, self.stim_elecs, title='')

        # Plot origins
        if self.epoch_frames and phase is not None:
            title = f'MEA electrode layout - {phase} burst origins' if title is None else title
            if weighted:
                full_savepath = SAVE_PATH  / f'{save_filename}_origin_heatmap_{phase}_weighted.png'
            else:
                full_savepath = SAVE_PATH  / f'{save_filename}_origin_heatmap_{phase}.png'
        else:
            title = f'MEA electrode layout - burst origins' if title is None else title
            if weighted:
                full_savepath = SAVE_PATH  / f'{save_filename}_origin_heatmap_weighted.png'
            else:
                full_savepath = SAVE_PATH  / f'{save_filename}_origin_heatmap.png'

        ax.scatter(x_origin, chip_ht_um-np.array(y_origin), color=c, marker='x', s=100, linewidths=3)

        ax.set_xlim([0, constants.CHIP_WIDTH*constants.ELEC_SIZE])
        ax.set_ylim([0,chip_ht_um])
        ax.set_title(title)

        if not plot_axes:
            # Hide ticks & labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove axis labels
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Ensure all four spines (borders) are visible like a frame
            for spine in ax.spines.values():
                spine.set_visible(True)
                # spine.set_linewidth(2)       # optional: thicker
                # spine.set_edgecolor("black") # optional: color

        # ax.set_box_aspect(0.55)  # height / width

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove margins

        if save_path is not None and save_filename is not None:
            plt.savefig(full_savepath, dpi=300, bbox_inches='tight')
            plt.close()

        if ret:
            # Draw the figure to a canvas and get the RGBA buffer as a numpy array
            img = utils.fig_to_array(fig)
            return img
        
    def plot_x_origin_histogram(self, phase=None, title=None, plot_type='histogram', ax=None):

        if ax is None:
            # create own figure 
            fig, ax = plt.subplots(1, 1, figsize=(9,5))

        chip_width_um = constants.CHIP_WIDTH * constants.ELEC_SIZE

        # Plot histogram using the electrodes that are spiking at the onset of the burst

        x_locs = [] # for all electrodes

        for idx, burst in self.burst_df.iterrows():
            
            if self.training_exp and phase is not None:
                if burst['phase'] != phase:
                    continue
            
            try:
                channels_spiking = list(ast.literal_eval(burst['channels_spiking_at_origin_counts']).keys())
                # channels_spiking = [int(x.strip()) for x in burst['channels_spiking_at_origin_counts'][1:-1].split(",")] # if csv is read in it will  be a list of strings
            except: 
                channels_spiking = list(burst['channels_spiking_at_origin_counts'].keys())

            mask = np.isin(self.channelmap[:,1], channels_spiking)
            chanmap_slice = self.channelmap[mask]

            xs = chanmap_slice[:,3]
            
            x_locs+=list(xs) # combine them all 

        if plot_type=='histogram':
            n, _, _ = ax.hist(x_locs, bins=20)
            y = n
            ax.set_ylabel('# electrodes')
        elif plot_type=='kde':
            kde = scipy.stats.gaussian_kde(x_locs, bw_method=0.3)  # bandwidth controls smoothness
            x = np.linspace(min(x_locs), max(x_locs), 1000)
            pdf = kde(x)
            ax.plot(x, pdf, color='blue')
            ax.fill_between(x, pdf, color='blue', alpha=0.3)  # fills from y=0 to the curve
            y=pdf
            ax.set_ylabel('Probability Density')

        ax.set_xlim([0,chip_width_um])
        ax.vlines(x=chip_width_um/2, ymin=0, ymax=max(y), colors='r', linestyles='--')
        ax.set_xlabel('Chip width (µm)')
        ax.set_title(title)

        if ax is None:
            plt.show()


    def plot_origin_ellipses(self, rec, save_path, save_filename):
        pass

    def plot_duration_hist(self, burst_data, save_path, save_filename):
        # Duration histogram
        # viz.histogram(list(burst_data['duration']), title='Burst durations', xlabel='Burst Duration (sec)', save_path=save_path, savefilename=save_filename+'_burst_duration_hist.png')
        pass

    def plot_burst_size_hist(self, burst_data, save_path, save_filename):
        # Fraction of electrodes involved in burst
        # viz.histogram(list(burst_data['size_pct_elec']), title='Burst size onset to offset', xlabel='Frac. electrodes', save_path=save_path, savefilename=save_filename+'_burst_size_hist.png')
        pass

    def plot_firing_rates(self):
        pass

    def plot_ISI(self):
        pass


    def plot_bursts_vectors(self, ax=None, save_path=None, save_filename=None, phase=None, title=None, ret=False, plot_axes=False, show=True):
        # Burst plot 

        SAVE_PATH = None
        if save_path is not None:
            SAVE_PATH = Path(save_path) # make sure the save path is a pathlib path

        # create own figure 
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9,5))

        chip_ht_um = constants.CHIP_HEIGHT * constants.ELEC_SIZE

        # Plot heatmap using the electrodes that are spiking at the onset of the burst

        x_origin = []
        y_origin = []

        x_peak = []
        y_peak = []

        hal_burst_df = self.burst_df[self.burst_df['type']=='HAL_like']

        for idx, burst in hal_burst_df.iterrows():
            
            if self.epoch_frames and phase is not None:
                if burst['phase'] != phase:
                    continue
            
            # channels_spiking = [int(x.strip()) for x in burst['channels_spiking_at_origin'][1:-1].split(",")] # if csv is read in it will  be a list of strings
            origin_channels_spiking = list(dict(ast.literal_eval(burst['origin_chan_counts'])).keys()) # if csv is read in it will  be a list of strings
            if len(origin_channels_spiking)==0:
                print("No origin spikes on any channels.")
                continue

            peak_channels_spiking = list(dict(ast.literal_eval(burst['burst_chan_counts'])).keys())
            if len(peak_channels_spiking)==0:
                print("No peak spikes on any channels.")
                continue


            origin_mask = np.isin(self.channelmap[:,1], origin_channels_spiking)
            origin_chanmap_slice = self.channelmap[origin_mask]

            origin_xs = origin_chanmap_slice[:,3]
            origin_ys = chip_ht_um - np.array(origin_chanmap_slice[:,4]) # invert y-values

            sns.kdeplot(x=origin_xs, y=origin_ys, fill=True, alpha=0.2, cmap='Blues',warn_singular=False, ax=ax)

            x_origin.append(int(burst['origin_x']))
            y_origin.append(int(burst['origin_y']))

            peak_mask = np.isin(self.channelmap[:,1], peak_channels_spiking)
            peak_chanmap_slice = self.channelmap[peak_mask]

            peak_xs = peak_chanmap_slice[:,3]
            peak_ys = chip_ht_um - np.array(peak_chanmap_slice[:,4]) # invert y-values

            # sns.kdeplot(x=peak_xs, y=peak_ys, fill=True, alpha=0.2, cmap="Reds",warn_singular=False)

            x_peak.append(int(burst['peak_x']))
            y_peak.append(int(burst['peak_y']))

        viz.MEA(ax, self.channelmap, self.stim_elecs, title='')

        # Vector components
        dx = np.array(x_peak) - np.array(x_origin)
        dy = (chip_ht_um-np.array(y_peak)) - (chip_ht_um-np.array(y_origin))

        ax.quiver(x_origin, chip_ht_um-np.array(y_origin), dx, dy, angles='xy', scale_units='xy', scale=1, color='black',
                  width=0.002,      # thinner shaft
                    headwidth=3,      # smaller arrowhead width
                    headlength=4,     # smaller arrowhead length
                    headaxislength=3,  # smaller head base)
                    alpha = 0.6)

        ax.set_xlim([0, constants.CHIP_WIDTH*constants.ELEC_SIZE])
        ax.set_ylim([0,chip_ht_um])
        ax.set_title(title)

        if not plot_axes:
            # Hide ticks & labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove axis labels
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Ensure all four spines (borders) are visible like a frame
            for spine in ax.spines.values():
                spine.set_visible(True)
                # spine.set_linewidth(2)       # optional: thicker
                # spine.set_edgecolor("black") # optional: color

        # ax.set_box_aspect(0.55)  # height / width

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove margins

        if SAVE_PATH is not None and save_filename is not None:
            # Plot origins
            if self.epoch_frames and phase is not None:
                title = f'MEA electrode layout - {phase} burst origins' if title is None else title
                full_savepath = SAVE_PATH  / f'{save_filename}_origin_heatmap_{phase}.png'
            else:
                title = f'MEA electrode layout - burst origins' if title is None else title
                full_savepath = SAVE_PATH  / f'{save_filename}_origin_heatmap.png'

            plt.savefig(full_savepath, dpi=300, bbox_inches='tight')
            plt.close()

        if ret:
            # Draw the figure to a canvas and get the RGBA buffer as a numpy array
            img = utils.fig_to_array(fig)
            return img
        
        if show:
            plt.show()