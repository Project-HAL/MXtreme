'''
For all things bursts.

Includes a BurstDetector class for burst detection and a Burst class to compute burst characteristics such as...
- origin
- direction
- electrodes involved
- duration
- # of peaks
- IBI
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from scipy.stats import median_abs_deviation



from mxtreme import constants
from mxtreme import utils
from mxtreme import visualizations as viz

class BurstDetector:
    """
    Class to implement burst detection on array-wide spike-binned data. 
    """

    def __init__(self, method) -> None:
        self.method = method
        self.burst_detection_parameters = None

    def detect_bursts(self, *args, **kwargs):
        
        method_handlers = {
        'basic_rate_thresh': self.basic_rate_thresh,
        'ISI_N': self.ISI_N,
        'auto_rate_thresh':self.auto_rate_thresh
        }

        handler = method_handlers.get(self.method)

        if handler:
            try:
                # Unpack args and kwargs to call the specific handler
                return handler(*args, **kwargs)
            except TypeError as e:
                # Catch incorrect parameter counts
                print(f"Error in {self.method}. \n Details: {e}")
                return
        else:
            print(f"Error: Unknown method '{self.method}'")
            return

    def ISI_N(self, spike_data, spike_bin, epoch_frames, 
            samp_rate, bin_size, N=100, noise_thresh=0.05, burst_thresh=0.15, 
            plot=False, dist_btw_bursts=30):
        
        asdr = spike_bin.mean(axis=0)

        # N = 1 is standard ISI 
        
        # Generate the ISI-N histogram 
        array_spike_frames = spike_data['frameno']
        
        if len(array_spike_frames)<N:
            print("Too few spikes in recording to compute ISI-N.")
            return []

        ISI_N_frames = [array_spike_frames[N:]-array_spike_frames[:-N]]
        ISI_N_sec = ISI_N_frames/samp_rate
        ISI_N_sec = ISI_N_sec[ISI_N_sec>0] # filter out ISI's of 0

        # KDE
        kde = gaussian_kde(ISI_N_sec)
        

        # xs = np.linspace(ISI_N_sec.min(), ISI_N_sec.max(), 400)
        xs = np.geomspace(ISI_N_sec.min(), ISI_N_sec.max(), 400)
        ys = kde(xs)

        peaks, _ = scipy.signal.find_peaks(ys)

        if len(peaks)<2: # no bursts
            self.burst_detection_parameters = {"N" : N,
                                            "noise_thresh" : noise_thresh, 
                                            "burst_thesh" : burst_thresh
                                            }
            return []

        p1, p2 = peaks[:2]

        # convert valley back
        ISI_N_thresh = xs[np.argmin(ys[p1:p2]) + p1]

        print(ISI_N_thresh)

        # plot=True
        if plot:
            plt.figure(figsize=(8, 6))

            # number of bins
            n_bins = 30

                # compute log-spaced bin edges
            bins = np.logspace(
                np.log10(ISI_N_sec.min()),
                np.log10(ISI_N_sec.max()),
                n_bins + 1
            )

                
            eps = 1e-12
            ys_plot = np.clip(ys, eps, None)

            # Plot the histogram
            # 'density=True' normalizes the histogram so the total area under the bars is 1
            plt.hist(ISI_N_sec, density=True, bins=bins, color='skyblue', alpha=0.6)
            plt.plot(xs, ys_plot, color="black", lw=2, label="KDE")
            plt.axvline(x=ISI_N_thresh, color='red')

            plt.xlabel(f'ISI_{N}')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel('Density')
            # plt.legend()
            plt.grid(True)
            plt.show()

        # Use the ISI_N threshold to detect bursts
        # Look both directions from each spike

        # Chat's python implementation of the ISI_N MATLAB algorithm: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2013.00193/full#B35
        n_spikes = len(array_spike_frames)

        ISI_N_thresh_frames = ISI_N_thresh*samp_rate
        
        dT = np.full((N, n_spikes), np.inf)

        for j in range(N):
            dT[j, N-1 : n_spikes-(N-1)] = (
                array_spike_frames[(N-1+j) : n_spikes-(N-1)+j]
                - array_spike_frames[j : n_spikes-(N-1)*2 + j]
            )

        criteria = np.zeros(n_spikes, dtype=bool)
        criteria[np.min(dT, axis=0) <= ISI_N_thresh_frames] = True


        # windows = np.array([
        #     array_spike_frames[j + N - 1 : n_spikes - (N - 1) + j]
        #     - array_spike_frames[j : n_spikes - (N - 1) * 2 + j]
        #     for j in range(N)
        # ])

        # # Apply ISI_N threshold to each spike
        # criteria = (windows.min(axis=0) <= ISI_N_thresh).astype(int)
        
        # Assign burst numbers to each spike
        SpikeBurstNumber = np.full(n_spikes, -1, dtype=int)

        INBURST = 0          # In a burst (1) or not (0)
        NUM_ = 0             # Burst number iterator
        NUMBER = -1          # Burst number assigned
        BL = 0               # Burst length

        for i in range(N-1, n_spikes):
            if INBURST == 0:  # Was not in burst
                if criteria[i]:
                    INBURST = 1
                    NUM_ += 1
                    NUMBER = NUM_
                    BL = 1
                # else: remain outside burst
            else:  # Was in burst
                if not criteria[i]:
                    INBURST = 0
                    if BL < N: # if the burst was shorter that N spikes, discard
                        SpikeBurstNumber[SpikeBurstNumber == NUMBER] = -1
                        NUM_ -= 1
                    NUMBER = -1

                elif (array_spike_frames[i] - array_spike_frames[i-(N-1)] > ISI_N_thresh_frames) and (BL >= N):
                    # Split consecutive bursts not separated by tonic spikes
                    NUM_ += 1
                    NUMBER = NUM_
                    BL = 1
                else:
                    BL += 1

            SpikeBurstNumber[i] = NUMBER

        # Assign Burst information

        MaxBurstNumber = SpikeBurstNumber.max() # number of bursts

        print("# bursts detected after ISI-N:", MaxBurstNumber)

        bursts = []
        detected_bursts = 1
        previous_burst_loc = -dist_btw_bursts

        for i in range(1, MaxBurstNumber + 1):
            ID = np.where(SpikeBurstNumber == i)[0]

            T_start_frame = array_spike_frames[ID[0]]
            T_end_frame = array_spike_frames[ID[-1]]
            size = len(ID)
            # print(i, size)

            # Combine with a rate threshold

            t_start_bin = int(T_start_frame*bin_size)
            t_stop_bin = int(T_end_frame*bin_size)

            # Find local maxima 
            peaks_hal, properties_hal = scipy.signal.find_peaks(asdr[t_start_bin:t_stop_bin+1], height=burst_thresh, distance=dist_btw_bursts)
            peaks_mini, properties_mini = scipy.signal.find_peaks(asdr[t_start_bin:t_stop_bin+1], height=(noise_thresh, burst_thresh), distance=dist_btw_bursts)

            # Debugging 
            
            # if len(peaks)==0:
            #     print('skipping peak')
            #     continue

            # visualize asdr and raster 
            # viz.raster_plot(spike_data, zoom=[T_start_frame, T_end_frame])

            # viz.plot_asdr_raster(spike_bin, title=None, zoom=[t_start_bin, t_stop_bin], save_path=None, savefilename=None, annotate_bursts=False, bursts=None)

            for th, hh in zip(peaks_hal, properties_hal['peak_heights']):

                burst_type = 'HAL_like'

                peak_t_bin = th+t_start_bin

                phase = self.get_burst_phase(peak_t_bin, epoch_frames, bin_size, samp_rate)

                # store as list of Burst objects
                bursts.append(ISIThreshBurst(detected_bursts, T_start_frame, T_end_frame, size, peak_t_bin, hh, phase=phase, type=burst_type))
                detected_bursts+=1

            for tm, hm in zip(peaks_mini, properties_mini['peak_heights']):
                burst_type = 'mini_b'

                peak_t_bin = tm+t_start_bin 

                phase = self.get_burst_phase(peak_t_bin, epoch_frames, bin_size, samp_rate)

                # store as list of Burst objects
                bursts.append(ISIThreshBurst(detected_bursts, T_start_frame, T_end_frame, size, peak_t_bin, hm, phase=phase, type=burst_type))
                detected_bursts+=1

            # all_peaks = list(zip(peaks, properties['peak_heights'])) # (t_binned, height)

            # if len(all_peaks)==0:
            #     continue

            # # highest peak in the span
            # t_binned, peak = max(all_peaks, key=lambda x: x[1])
            # peak_t_bin = t_binned + t_start_bin

            # phase = self.get_burst_phase(peak_t_bin, epoch_frames, bin_size, samp_rate)

            # if peak<burst_thresh:
            #     burst_type = 'mini_b'
            # else:
            #     burst_type = 'HAL_like'

            # # store as list of Burst objects
            # bursts.append(ISIThreshBurst(detected_bursts, T_start_frame, T_end_frame, size, peak_t_bin, peak, phase=phase, type=burst_type))
            # detected_bursts+=1

            # for t_binned, pheight in all_peaks:
                
            #     peak_t_bin = t_binned+t_start_bin
                
            #     if pheight<burst_thresh:
            #         burst_type = 'mini_b'
                    
            #     else:
            #         burst_type = 'HAL_like'

            #         if (peak_t_bin-previous_burst_loc)>dist_btw_bursts: # check dist between burst threshold only for the big ones
            #             previous_burst_loc=peak_t_bin # update the location of the last burst
            #         else:
            #             continue # continue if the last big burst was less that 30 bins ago

            #     phase = self.get_burst_phase(peak_t_bin, epoch_frames, bin_size, samp_rate)

            #     # store as list of Burst objects
            #     bursts.append(ISIThreshBurst(detected_bursts, T_start_frame, T_end_frame, size, peak_t_bin, pheight, phase=phase, type=burst_type))
            #     detected_bursts+=1

            # print("num bursts detected:", detected_bursts)
            # plt.plot(asdr)
            # plt.xlim([t_start_bin,t_stop_bin+1])
            # plt.show()

            
        print(len(bursts), "bursts detected.")

        self.burst_detection_parameters = {"N" : N,
                                            "noise_thresh" : noise_thresh, 
                                            "burst_thesh" : burst_thresh,
                                            "dist_btw_bursts":dist_btw_bursts
                                            }

        return bursts

    def get_burst_phase(self, t_peak_bin, epoch_frames, bin_size, samp_rate):
        t_peak_frame = t_peak_bin*bin_size*samp_rate

        # get burst phase
        if epoch_frames:
            if t_peak_frame>=epoch_frames['pre_start_frame'] and t_peak_frame<epoch_frames['train_start_frame']:
                phase = 'pre'
            elif t_peak_frame>=epoch_frames['train_start_frame'] and t_peak_frame<epoch_frames['post_start_frame']:
                phase = 'train'
            elif t_peak_frame>=epoch_frames['post_start_frame']:
                phase = 'post'
            else:
                phase = np.nan
        else:
            phase = np.nan

        return phase
    

    def auto_rate_thresh(self, spike_data, spike_bin, epoch_frames, 
                        samp_rate, bin_size, dist_btw_bursts, prominence_percentile):
        # TODO: implement a rate-based threshold
        # Works on binned data. Generate histogram of firing rates (spike counts) in a given window (the bin size). Choose the threshold as the spike count that separates the separates peaks of the bimodal histogram. 
        pass

    def basic_rate_thresh(self, spike_data, spike_bin, epoch_frames, samp_rate, bin_size, 
                          K=constants.K, dist_btw_bursts=constants.DIST_BTW_BURSTS, 
                          prominence_percentile=constants.PROMINENCE_PERCENTILE, noise_thresh=0.01, 
                          detect_train_bursts=True, smooth=True, sigma=constants.GAUSSIAN_SIGMA):
        
        # TODO: compute asdr here and add a parameter for Gaussian smoothing instead of doing this in the Recording class. 
        asdr = spike_bin.mean(axis=0) # array-wide spike detection rate

        if smooth: # optionally smooth with a Gaussian filter
            asdr = gaussian_filter1d(asdr, sigma=sigma)

        if epoch_frames: # Recording has pre/train/post periods

            # Calculate pre/post thresholds
            pre_start_bin = int(epoch_frames['pre_start_frame']/samp_rate/bin_size)
            train_start_bin = int(epoch_frames['train_start_frame']/samp_rate/bin_size)
            post_start_bin = int(epoch_frames['post_start_frame']/samp_rate/bin_size)

            # Detect peaks on pre/post periods using only a lower threshold for nosie
            peak_inds_pre, properties_pre = scipy.signal.find_peaks(asdr[pre_start_bin:train_start_bin], height=noise_thresh)
            peak_inds_post, properties_post = scipy.signal.find_peaks(asdr[post_start_bin:], height=noise_thresh)

            prominences_pre = scipy.signal.peak_prominences(asdr[pre_start_bin:train_start_bin], peak_inds_pre)[0]
            prominences_post = scipy.signal.peak_prominences(asdr[post_start_bin:], peak_inds_post)[0]

            # Merge pre/post peaks
            peaks = list(properties_pre['peak_heights'])+list(properties_post['peak_heights'])
            peak_inds = list(peak_inds_pre)+list(peak_inds_post)
            prominences = list(prominences_pre)+list(prominences_post)

            # Compute pre/post thresholds
            # Mean
            if len(peaks)>0:
                mean = np.mean(peaks)
                std = np.std(peaks)
                burst_thresh = mean + K * std

                # Prominence threshold
                if len(prominences)>0:
                    prominence_thresh = np.percentile(prominences, prominence_percentile)
                
                # Compute burst_detection on pre/post epochs
                bursts_pre = self.find_local_maxima(asdr[pre_start_bin:train_start_bin], burst_thresh, dist_btw_bursts, prominence_thresh)
                for x in bursts_pre: # have to offset the indices by the epochs that come before 
                    x.t_peak_bin += pre_start_bin

                bursts_post = self.find_local_maxima(asdr[post_start_bin:], burst_thresh, dist_btw_bursts, prominence_thresh)
                for x in bursts_post: # have to offset the indices by the epochs that come before 
                    x.t_peak_bin += post_start_bin

                bursts = bursts_pre + bursts_post
            
            else:
                burst_thresh=noise_thresh
                bursts = [] # no bursts with more than 1% of neurons firing in pre/post --> no bursts 

            # Keep track of the hyperparameters in a dictionary
            self.burst_detection_parameters = {'burst_thresh' : burst_thresh,
                                               'dist_btw_bursts' : dist_btw_bursts,
                                               'prominence_thresh' : prominence_thresh,
                                               'K' : K,
                                               'prominence_percentile' : prominence_percentile,
                                               'noise_thresh' : noise_thresh}
            
            # Train periods
            if detect_train_bursts:
                # Detect peaks on train period using only a lower threshold for nosie
                peak_inds_train, properties_train = scipy.signal.find_peaks(asdr[train_start_bin:post_start_bin], height=noise_thresh)

                prominences_train = scipy.signal.peak_prominences(asdr[train_start_bin:post_start_bin], peak_inds_train)[0]

                peaks_train = list(properties_train['peak_heights'])

                # Compute train thresholds

                # Mean
                if len(peaks_train)>0:
                    mean = np.mean(peaks_train)
                    std = np.std(peaks_train)
                    burst_thresh_train = mean + K * std

                    # Prominence threshold
                    if len(prominences_train)>0:
                        prominence_thresh_train = np.percentile(prominences_train, prominence_percentile)
                    
                    # Compute burst_detection on train epoch
                    train_bursts = self.find_local_maxima(asdr[train_start_bin:post_start_bin], burst_thresh_train, dist_btw_bursts, prominence_thresh_train)

                    for x in train_bursts: # have to offset the indices by the epochs that come before 
                        x.t_peak_bin += train_start_bin
                
                else:
                    burst_thresh_train=noise_thresh
                    train_bursts = [] # no bursts with more than 1% of neurons firing in train period --> no bursts 

                bursts+=train_bursts
    
                # Add train parameters to the dictionary
                self.burst_detection_parameters['burst_thresh_train'] = burst_thresh_train
                self.burst_detection_parameters['prominence_thresh_train'] = prominence_thresh_train

        else:

            # Burst detection for a full recording 

            # Detect peaks using only a lower threshold for nosie
            peak_inds, properties = scipy.signal.find_peaks(asdr, height=noise_thresh)

            prominences = scipy.signal.peak_prominences(asdr, peak_inds)[0]

            peaks = list(properties['peak_heights'])

            # Compute train thresholds

            # Mean
            if len(peaks)>0:
                mean = np.mean(peaks)
                std = np.std(peaks)
                burst_thresh = mean + K * std

                # Prominence threshold
                if len(prominences)>0:
                    prominence_thresh = np.percentile(prominences, prominence_percentile)
                
                # Compute burst_detection on pre/post epochs
                bursts = self.find_local_maxima(asdr, burst_thresh, dist_btw_bursts, prominence_thresh)
            
            else:
                burst_thresh=noise_thresh
                bursts = [] # no bursts with more than 1% of neurons firing in entire recording --> no bursts 

            # Keep track of the hyperparameters in a dictionary
            self.burst_detection_parameters = {'burst_thresh' : burst_thresh,
                                               'dist_btw_bursts' : dist_btw_bursts,
                                               'prominence_thresh' : prominence_thresh,
                                               'K' : K,
                                               'prominence_percentile' : prominence_percentile,
                                               'noise_thresh' : noise_thresh}  

        print(len(bursts), "bursts detected.")

        return bursts


    def find_local_maxima(self, signal, burst_thresh,  dist_btw_bursts, prominence_tresh):
        """
        Implements the 'local_maxima' strategy for burst detection which uses scipy's find_peaks() function. 

        :return: A list of bursts.
        :rtype: list
        """
        
        # Find local maxima 
        peaks, properties = scipy.signal.find_peaks(signal, height=burst_thresh, distance=dist_btw_bursts, prominence=prominence_tresh)

        bursts = list(zip(peaks, properties['peak_heights'])) # (t_binned, height)

        # store as list of Burst objects
        return [Burst(i,peak_t_bin,peak) for i,(peak_t_bin,peak) in enumerate(bursts)]

class Burst:

    def __init__(self, id, t_peak_bin, peak, phase=None):
        self.id = id
        # All bursts have a peak in bin_space
        self.t_peak_bin = t_peak_bin
        self.peak_bin_amp = peak # amplitude (peak size in fraction of network active in the given window/bin)

        self.phase = phase



    def compute_onset_length(self, spike_bin):
        '''
        Computes the length in bins of the onset (between 0-peak of the average activity plot). For small onsets with only a 
        few bins, we may not have the granularity to accurately find the time of onset (point where the onset threshold is crossed, typically 20% of the peak height).
        '''
        avg_activity = spike_bin.mean(axis=0)

        # points where the avg activity is 0
        zero_points = np.where((avg_activity==0))[0]

        # find the zero point closest (but less than) the peak time
        onset_zero = zero_points[zero_points<self.t_peak_bin][-1]

        onset_len_bins = self.t_peak_bin-onset_zero

        return onset_len_bins
    
    def compute_origin(self, raw_spike_data, channelmap, bin_size):
        
        if self.t_onset_sec is None:
            raise ValueError("Need to compute onset time first.")
        
        if self.instantaneous:
            # # use the binned data to get channels that are spiking at the peak
            # chan_ids = np.where(spike_bin[:,self.t_peak_bin]==1)
            # mask = np.isin(channelmap[:,0], chan_ids)
            # chanmap_slice = channelmap[mask]
            # self.channels_spiking_at_origin = chanmap_slice[:,1]
            
            # No spikes between zero point and the peak -> use channels spiking at the peak
            slice_zon = raw_spike_data[(raw_spike_data['frameno'] >= self.t_peak_samp) & (raw_spike_data['frameno'] <= self.t_peak_samp+(1/bin_size))]
    
        else:
            # Zero -> Onset
            # Get slice of the raw data given zero and onset frames
            slice_zon = raw_spike_data[(raw_spike_data['frameno'] >= self.t_onset_zero_samp) & (raw_spike_data['frameno'] <= self.onset_in_samples)]

        # Find channels that have spiked between the 0->onset
        self.channels_spiking_at_origin = [int(x) for x in np.unique(slice_zon['channel'])]

        mask = np.isin(channelmap[:,1], self.channels_spiking_at_origin)
        chanmap_slice = channelmap[mask]


        xs = chanmap_slice[:,3] # electrode x values
        ys = chanmap_slice[:,4] # electrode y values

        self.origin = (np.mean(xs), np.mean(ys))


    def compute_weighted_origin(self, raw_spike_data, channelmap, bin_size):
        
        if self.t_onset_sec is None:
            raise ValueError("Need to compute onset time first.")
        
        if self.instantaneous:
            # use the binned data to get channels that are spiking at the peak
            # chan_ids = np.where(spike_bin[:,self.t_peak_bin]==1)
            # mask = np.isin(channelmap[:,0], chan_ids)
            # chanmap_slice = channelmap[mask]
            # self.channels_spiking_at_origin = chanmap_slice[:,1]

            # No spikes between zero point and the peak -> use channels spiking at the peak
            slice_zon = raw_spike_data[(raw_spike_data['frameno'] >= self.t_peak_samp) & (raw_spike_data['frameno'] <= self.t_peak_samp+(1/bin_size))]
    
        else:
            # Zero -> Onset
            # Get slice of the raw data given zero and onset frames
            slice_zon = raw_spike_data[(raw_spike_data['frameno'] >= self.t_onset_zero_samp) & (raw_spike_data['frameno'] <= self.onset_in_samples)]

        # channels = [int(x) for x in slice_zon['channel']]
        channels = np.array(slice_zon['channel'], dtype=int)
        if len(channels)==0:
            print('No spikes at zero onset bin')

        counter = Counter(channels)
        self.chans_spiking_at_origin_counts = self.chans_spiking_at_origin_counts = {int(k): int(v) for k, v in counter.items()}

        # Find channels that have spiked between the 0->onset
        self.channels_spiking_at_origin = [int(x) for x in np.unique(slice_zon['channel'])]

        mask = np.isin(channelmap[:,1], self.channels_spiking_at_origin)
        chanmap_slice = channelmap[mask]

        weights_ordered = [self.chans_spiking_at_origin_counts[x] for x in chanmap_slice[:,1]]

        total = np.sum(list(self.chans_spiking_at_origin_counts.values()))
        
        xs_weighted = chanmap_slice[:,3]*weights_ordered
        ys_weighted = chanmap_slice[:,4]*weights_ordered

        self.weighted_origin = (np.sum(xs_weighted)/total, np.sum(ys_weighted)/total)


    def compute_1D_direction(self):
        
        if self.origin[0] < self.peak_loc[0]:
            self.x_direction = 'right'
        else:
            self.x_direction = 'left'


    def compute_peak_location(self, raw_spike_data, channelmap):
        # TODO: get spikes from onset->offset to compute peak location

        # Onset -> Offset
        # Get slice of the raw data given onset and offset frames
        slice_on_off = raw_spike_data[(raw_spike_data['frameno'] >= self.onset_in_samples) & (raw_spike_data['frameno'] <= self.offset_in_samples)]

        # Find channels that have spiked between the 0->onset
        self.channels_spiking_at_peak = [int(x) for x in np.unique(slice_on_off['channel'])]

        mask = np.isin(channelmap[:,1], self.channels_spiking_at_peak)
        chanmap_slice = channelmap[mask]

        xs = chanmap_slice[:,3] # electrode x values
        ys = chanmap_slice[:,4] # electrode y values

        self.peak_loc = (float(np.mean(xs)), float(np.mean(ys)))


    def find_constituent_channels(self, raw_spike_data, channelmap, samp_rate):
        onset_in_samples = self.t_onset_sec*samp_rate
        offset_in_samples = self.t_offset_sec*samp_rate

        channels = [int(x) for x in raw_spike_data[(raw_spike_data['frameno']>=int(onset_in_samples)) & (raw_spike_data['frameno']<=int(offset_in_samples))]['channel']]

        # Get channels involved in burst and the number of times they spiked
        self.constituent_channels = dict(Counter(channels)) # unique channels and counts
        self.frac_constituent_channels = len(self.constituent_channels)/len(channelmap[:,1])

    def compute_duration(self):
        self.duration = self.t_offset_sec-self.t_onset_sec

    def plot(self, spike_bin, window_bin=200, annotate_onset=False, title=None, samp_rate=None, raw_spike_data=None, save_path=None, savefilename=None):

        avg_activity = spike_bin.mean(axis=0)

        burst_slice = avg_activity[self.t_peak_bin-window_bin//2: self.t_peak_bin+window_bin//2]

        fig, ax = plt.subplots(1, 1, figsize=(12,3))

        t_sec = np.arange(-window_bin//2, window_bin//2)*constants.BIN_SIZE # time in seconds centered around the burst peak
        ax.plot(t_sec, burst_slice) 

        ax.set_ylabel("Fraction channels active", fontsize=10)
        ax.set_xlabel("Time (seconds)", fontsize=10)

        # Annotate peaks with upside-down red triangles
        plt.scatter(0, self.peak+0.02, marker='v', color='red', s=10, label='Annotations') # +0.02 for annotation offset

        # Annotate onset
        if annotate_onset:

            # doesn't really make sense to plot onset computed from raw data on the binned plot...
            onset_sec = self.t_onset_sec-(self.t_peak_bin*constants.BIN_SIZE)
            offset_sec = self.t_offset_sec - (self.t_peak_bin*constants.BIN_SIZE)
            plt.vlines(onset_sec,0,self.peak,colors='orange')
            plt.vlines(offset_sec,0,self.peak,colors='orange')
    

        # Plot raw spike data
        if raw_spike_data is not None:
            # TODO: would have to reformat the raw spike data here into (channels x frames)
            # Convert windows to samps
            samp_start = int((self.t_peak_bin-window_bin//2)*constants.BIN_SIZE*samp_rate)
            samp_end = int((self.t_peak_bin+window_bin//2)*constants.BIN_SIZE*samp_rate)

            raw_burst_slice = raw_spike_data[:,samp_start:samp_end].mean(axis=0)
            t_sec_upsamp = np.linspace(-window_bin//2*constants.BIN_SIZE, window_bin//2*constants.BIN_SIZE, num=len(raw_burst_slice))

            ax.plot(t_sec_upsamp, raw_burst_slice, color='k') 

        if title is not None:
            plt.title(title+f" - burst {self.id}")
        else:
            plt.title(f"Burst {self.id}")

        if save_path is not None:
            plt.savefig(save_path+savefilename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class ISIThreshBurst(Burst):
    """
    A class for a burst detected using an ISI-based burst detection algorithm. Start/stop frames and burst size in spikes are detected using an ISI threshold. 
    """

    def __init__(self, id, ISI_N_start_frame, ISI_N_stop_frame, ISI_N_size_in_spikes, t_peak_bin, peak_bin_amp, phase, type):
        super().__init__(id, t_peak_bin, peak_bin_amp, phase)
        
        # ISI_N burst detection
        self.ISI_N_start_frame = ISI_N_start_frame
        self.ISI_N_stop_frame = ISI_N_stop_frame
        self.ISI_N_size_in_spikes = ISI_N_size_in_spikes # in number of spikes

        self.type = type

    def compute_burst_characteristics(self, asdr, raw_spike_data, stim_frames, channelmap, samp_rate, bin_size, onset_thresh_pct, offset_thresh_pct, asdr_filled, verbose=False):
        
        self.init_features()
        
        try:
            self.compute_onset_offset(asdr, raw_spike_data, samp_rate, bin_size, onset_thresh_pct, offset_thresh_pct, verbose)
            self.compute_duration()
            self.compute_weighted_origin(raw_spike_data, channelmap, bin_size)
            self.compute_origin(raw_spike_data, channelmap, bin_size)
            self.find_constituent_channels(raw_spike_data, channelmap, samp_rate)
            self.compute_peak_location(raw_spike_data, channelmap)
            self.compute_1D_direction()
            self.compute_num_peaks(asdr_filled, bin_size)
            self.stim_in_burst(stim_frames)

        except Exception as e: # skip but tag for further investigation
                print(f'Burst {self.id} feature computation failed: {e}')
                self.ignore=True

        # Return feature dict - these will be rows in a dataframe
        feature_dict = {'id' : self.id,
                        't_peak_bin' : self.t_peak_bin,
                        'peak_bin_amp' : self.peak_bin_amp,
                        'phase' : self.phase,
                        'ISI_N_start_frame' : self.ISI_N_start_frame,
                        'ISI_N_stop_frame' : self.ISI_N_stop_frame,
                        'ISI_N_size_in_spikes' : self.ISI_N_size_in_spikes,
                        'type' : self.type,
                        'has_stim' : self.has_stim,
                        'instantaneous' : self.instantaneous,
                        't_onset_sec' : self.t_onset_sec,
                        't_peak_sec' : self.t_peak_bin*bin_size,
                        't_offset_sec' : self.t_offset_sec,
                        'duration' : self.duration,
                        'size_pct_elec' : self.frac_constituent_channels,
                        'origin_x' : self.origin[0],
                        'origin_y' : self.origin[1],
                        'origin_chan_counts' : self.chans_spiking_at_origin_counts,
                        'burst_chan_counts' : self.constituent_channels,
                        'peak_x' : self.peak_loc[0],
                        'peak_y' : self.peak_loc[1],
                        'weighted_orign_x' : self.weighted_origin[0],
                        'weighted_origin_y' : self.weighted_origin[1],
                        'num_secondary_peaks' : self.num_secondary_peaks
                        } 
        
        return feature_dict


    def init_features(self):
        # Initialize burst features
        self.t_onset_sec = -1
        self.t_offset_sec = -1
        self.t_onset_zero_samp = -1 # nearest 0 point left of peak - useful for finding the burst slice in samples
        self.t_offset_zero_samp = -1 # nearest 0 point right of peak
        self.ignore = False
        self.instantaneous = False
        self.has_stim = False
        self.duration = -1
        self.frac_constituent_channels = -1
        self.origin = [-1,-1]
        self.chans_spiking_at_origin_counts = {}
        self.constituent_channels = {}
        self.peak_loc = [-1,-1]
        self.weighted_origin = [-1,-1]
        self.num_secondary_peaks = -1


    def compute_onset_offset(self, asdr, raw_spike_data, samp_rate, bin_size, onset_thresh_pct, offset_thresh_pct, verbose):
        """Computes the burst onset from the binned and raw spike data. 
        Finds the zero point and peak using the binned data, then uses the raw data from 0->peak 
        to find the point where the cumulative number of spikes surpasses 20% of the burst peak height. 
        """

        # Calculate the MAD of the local signal
        mad_value = median_abs_deviation(asdr[int(self.ISI_N_start_frame*bin_size):int(self.ISI_N_stop_frame*bin_size)])

        # Convert MAD to an estimated standard deviation 
        # (for normally distributed noise, scale by ~1.4826)
        noise_estimate_mad_std = mad_value * 1.4826 
    
        # Points where the avg activity is 0 (these act as anchor points around a burst)
        # zero_points = np.where((asdr==0))[0]
        zero_points = np.where((asdr<=noise_estimate_mad_std))[0]

        # Find the closest zero points on either side of the peak
        pre_burst_zero_points = zero_points[zero_points<self.t_peak_bin]
        post_burst_zero_points = zero_points[zero_points>self.t_peak_bin]

        #TODO: handle bursts right before end of recording (len(zero_points[zero_points>self.t_peak_bin])==0) - ignored for now
        if len(pre_burst_zero_points)==0:
            raise IndexError("Burst too close to start of recording to compute onset.")
        
        if len(post_burst_zero_points)==0:
            raise IndexError("Burst too close to end of recording to compute offset.")
        
        onset_zero = pre_burst_zero_points[-1]
        offset_zero = post_burst_zero_points[0]
                
        # Convert to samples for raw spike data
        self.t_onset_zero_samp = int(onset_zero*bin_size*samp_rate)
        self.t_offset_zero_samp = int(offset_zero*bin_size*samp_rate)

        self.t_peak_samp = int(self.t_peak_bin * bin_size * samp_rate)

        self.t_peak_samp_start = int(self.t_peak_bin * bin_size * samp_rate)
        self.t_peak_samp_end = self.t_peak_samp_start-(1/bin_size) # add the width of one bin in frames (start and stop of the peak frame in bins)

        # Get 0->peak slice of raw data for onset
        onset_slice = raw_spike_data[(raw_spike_data['frameno']>=self.t_onset_zero_samp) & (raw_spike_data['frameno']<=self.t_peak_samp_start)]

        # Get peak->0 slice of raw data for offset
        offset_slice = raw_spike_data[(raw_spike_data['frameno']<=self.t_offset_zero_samp) & (raw_spike_data['frameno']>=self.t_peak_samp_end)]

        # Onset, offset thresholds based total number of spikes from 0->peak for onset and peak->0 for offset
        onset_thresh = int(len(onset_slice) * onset_thresh_pct)
        offset_thresh = int(len(offset_slice) * (1-offset_thresh_pct)) # 1-onset_thresh because we work from peak outwards 

        # # Alternative: Onset, offset thresholds based on binned spike data (i.e. the number of spikes at burst peak determined with spike bin data)
        # onset_thresh = int(spike_bin[:,self.t_peak_bin].sum() * onset_thresh_pct)
        # offset_thresh = int(spike_bin[:,self.t_peak_bin].sum() * (1-onset_thresh_pct)) # 1-onset_thresh because we work from peak outwards 
                        
        # Onset occurs onset_thresh spikes from the zero point
        if len(onset_slice)==0: # TODO: check this... no spikes between left marker frame and peak (aka instantaneous onset). Ex. DIV33_stim_removal, well5, burst 167
            self.onset_in_samples = self.t_onset_zero_samp
            self.instantaneous = True # see origin detection
            print('Burst ' + str(self.id) + ': No spikes between onset zero frame and peak')
        else:
            self.onset_in_samples = onset_slice['frameno'][onset_thresh]

        if len(offset_slice)==0:
            self.offset_in_samples = self.t_offset_zero_samp
            print('Burst ' + str(self.id) + ': No spikes between peak and offset zero frame')
        else:
            self.offset_in_samples = offset_slice['frameno'][offset_thresh] # work forward from peak to zero until we hit 80%

        # Convert onset/offset time to seconds
        self.t_onset_sec = self.onset_in_samples/samp_rate
        self.t_offset_sec = self.offset_in_samples/samp_rate
        
        if verbose:
            print('Onset:', self.t_onset_sec, 'sec, Peak:', self.t_peak_bin*constants.BIN_SIZE,'sec, Offset:', self.t_offset_sec,'sec')

        # plt.plot(asdr)
        # plt.axvline(self.onset_in_samples*bin_size, label="onset", color='orange')
        # plt.axvline(self.offset_in_samples*bin_size, label="offset", color='red')
        # plt.axvline(self.t_peak_bin, label="peak", color='purple')
        # plt.xlim([onset_zero-10, offset_zero+10])
        # plt.title(self.type)
        # plt.legend()
        # plt.show()


    def compute_num_peaks(self, asdr, bin_size):

        t_start_bin = int(self.ISI_N_start_frame*bin_size)
        t_stop_bin = int(self.ISI_N_stop_frame*bin_size)

        # Find local maxima 
        peaks, properties = scipy.signal.find_peaks(asdr[t_start_bin:t_stop_bin+1], height=0, prominence=0)

        all_peaks = list(zip(peaks, properties['peak_heights'], properties['prominences'])) # (t_binned, prominence)

        t_peak, height_peak, prom_peak = max(all_peaks, key=lambda x: x[1])

        prom_thresh = prom_peak*0.25

        peaks, properties = scipy.signal.find_peaks(asdr[t_start_bin:t_stop_bin+1], height=(0,height_peak-1e-4), prominence=prom_thresh)

        # plt.plot(asdr[t_start_bin:t_stop_bin+1])
        # plt.axvline(t_peak, color='red')
        # plt.vlines(peaks, (0,)*len(peaks), (height_peak,)*len(peaks), color='orange')
        # plt.title(self.phase +", "+ self.type)
        # plt.show()
        
        self.num_secondary_peaks = len(peaks)

    def stim_in_burst(self, stim_frames):
        self.has_stim = True if np.any((stim_frames < self.ISI_N_stop_frame) & (stim_frames>self.ISI_N_start_frame)) else False

class RateThreshBurst(Burst):
    """
    A class for a burst detected using a rate-based burst detection algorithm. Burst peaks are detected in bin-space using a series of thresholds.  
    """

    def __init__(self, id, t_peak_bin, peak_bin_amp, phase) -> None:
        """
        Constructor for a Burst.

        :param id: Burst ID
        :type id: int
        :param t_peak_bin: Bin in the array-wide spike-bin data of the burst's peak.
        :type t_peak_bin: int
        :param peak: Amplitude of the burst peak. Can be interpretted as the fraction of the network spiking at the burst's peak.
        :type peak: float
        """
        super().__init__(id, t_peak_bin, peak_bin_amp , phase) # index in BurstDetector.bursts

        # Burst onset and offset
        self.t_onset_sec = None
        self.t_offset_sec = None
        self.t_onset_zero_samp = None # nearest 0 point left of peak - useful for finding the burst slice in samples
        self.t_offset_zero_samp = None # nearest 0 point right of peak

        self.ignore = False
        self.instantaneous = False


    def compute_burst_characteristics(self, asdr, spike_bin, raw_spike_data, stim_frames, channelmap, samp_rate, bin_size, onset_thresh_pct, offset_thresh_pct, verbose=False):
       
        try:
            self.compute_onset_offset(asdr, raw_spike_data, stim_frames, samp_rate, bin_size, onset_thresh_pct, offset_thresh_pct, verbose)
            self.compute_duration()
            self.compute_weighted_origin(spike_bin, raw_spike_data, channelmap)
            self.compute_origin(spike_bin, raw_spike_data, channelmap)
            self.find_constituent_channels(raw_spike_data, channelmap, samp_rate)
            self.compute_peak_location(raw_spike_data, channelmap)
            self.compute_1D_direction()

        except Exception as e: # skip but tag for further investigation
            print(f'Burst {self.id} feature computation failed: {e}')
            self.ignore=True

    def compute_onset_offset(self, asdr, raw_spike_data, samp_rate, bin_size, onset_thresh_pct, offset_thresh_pct, noise, verbose):
        """Computes the burst onset from the binned and raw spike data. 
        Finds the zero point and peak using the binned data, then uses the raw data from 0->peak 
        to find the point where the cumulative number of spikes surpasses 20% of the burst peak height. 
        """

        # Points where the avg activity is 0 (these act as anchor points around a burst)
        zero_points = np.where((asdr==0))[0]

        # Find the closest zero points on either side of the peak
        pre_burst_zero_points = zero_points[zero_points<self.t_peak_bin]
        post_burst_zero_points = zero_points[zero_points>self.t_peak_bin]

        #TODO: handle bursts right before end of recording (len(zero_points[zero_points>self.t_peak_bin])==0) - ignored for now
        if len(pre_burst_zero_points)==0:
            raise IndexError("Burst too close to start of recording to compute onset.")
        
        if len(post_burst_zero_points)==0:
            raise IndexError("Burst too close to end of recording to compute offset.")
        
        onset_zero = pre_burst_zero_points[-1]
        offset_zero = post_burst_zero_points[0]
                
        # Convert to samples for raw spike data
        self.t_onset_zero_samp = int(onset_zero*bin_size*samp_rate)
        self.t_offset_zero_samp = int(offset_zero*bin_size*samp_rate)

        self.t_peak_samp = int(self.t_peak_bin * bin_size * samp_rate)

        self.t_peak_samp_start = int(self.t_peak_bin * bin_size * samp_rate)
        self.t_peak_samp_end = self.t_peak_samp_start-(1/bin_size) # add the width of one bin in frames (start and stop of the peak frame in bins)

        # Get 0->peak slice of raw data for onset
        onset_slice = raw_spike_data[(raw_spike_data['frameno']>=self.t_onset_zero_samp) & (raw_spike_data['frameno']<=self.t_peak_samp_start)]

        # Get peak->0 slice of raw data for offset
        offset_slice = raw_spike_data[(raw_spike_data['frameno']<=self.t_offset_zero_samp) & (raw_spike_data['frameno']>=self.t_peak_samp_end)]

        # Onset, offset thresholds based total number of spikes from 0->peak for onset and peak->0 for offset
        onset_thresh = int(len(onset_slice) * onset_thresh_pct)
        offset_thresh = int(len(offset_slice) * (1-offset_thresh_pct)) # 1-onset_thresh because we work from peak outwards 

        # # Alternative: Onset, offset thresholds based on binned spike data (i.e. the number of spikes at burst peak determined with spike bin data)
        # onset_thresh = int(spike_bin[:,self.t_peak_bin].sum() * onset_thresh_pct)
        # offset_thresh = int(spike_bin[:,self.t_peak_bin].sum() * (1-onset_thresh_pct)) # 1-onset_thresh because we work from peak outwards 
                        
        # Onset occurs onset_thresh spikes from the zero point
        if len(onset_slice)==0: # TODO: check this... no spikes between left marker frame and peak (aka instantaneous onset). Ex. DIV33_stim_removal, well5, burst 167
            self.onset_in_samples = self.t_onset_zero_samp
            self.instantaneous = True # see origin detection
            print('Burst ' + str(self.id) + ': No spikes between onset zero frame and peak')
        else:
            self.onset_in_samples = onset_slice['frameno'][onset_thresh]

        if len(offset_slice)==0:
            self.offset_in_samples = self.t_offset_zero_samp
            print('Burst ' + str(self.id) + ': No spikes between peak and offset zero frame')
        else:
            self.offset_in_samples = offset_slice['frameno'][offset_thresh] # work forward from peak to zero until we hit 80%

        # Convert onset/offset time to seconds
        self.t_onset_sec = self.onset_in_samples/samp_rate
        self.t_offset_sec = self.offset_in_samples/samp_rate
        
        if verbose:
            print('Onset:', self.t_onset_sec, 'sec, Peak:', self.t_peak_bin*constants.BIN_SIZE,'sec, Offset:', self.t_offset_sec,'sec')

        # plt.plot(asdr)
        # plt.axvline(self.onset_in_samples*bin_size, label="onset", color='orange')
        # plt.axvline(self.offset_in_samples*bin_size, label="offset", color='red')
        # plt.axvline(self.t_peak_bin, label="peak", color='purple')
        # plt.xlim([onset_zero-10, offset_zero+10])
        # plt.title(self.type)
        # plt.legend()
        # plt.show()
