'''
Helper functions.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path


def load_data(filepath):
    """Loads preprocessed experimental data saved as an ``.npz``.

    Thin backwards-compatible re-export of :func:`mxtreme.io.load_preprocessed`, kept so existing
    imports (``from mxtreme.utils import load_data``) keep working.

    Args:
        filepath (string): path to cleaned data.

    Returns:
        dictionary: data associated with an experiment. Keys: ['spike_data', 'channelmap',
                            'stim_elecs', 'rec_t_sec', 'samp_rate', 'lsb', 'eventtime', 'event_messages'.]
    """
    from mxtreme.io import load_preprocessed

    return load_preprocessed(filepath)
 
def get_well_id_from_raw(path_to_raw, well_no=0, recording_no=0):
    with h5py.File(path_to_raw, 'r') as f:
        h5_object = f['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        well_id = h5_object['well_id'][0]
    return well_id

def load_raw_waveform(path_to_raw, well_no, recording_no, chan_id, start_frame, block_size):
    # max_allowed_block_size = 150000
    # assert(block_size<=max_allowed_block_size)

    with h5py.File(path_to_raw, 'r') as f:
        h5_object = f['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        
        # Load settings from file
        lsb = h5_object['settings']['lsb'][0]

        # Load raw data from file
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]

        return group0['raw'][chan_id, start_frame:start_frame+block_size].T * lsb
    
def waveform_generator(path_to_raw, well_no, recording_no, channels, start_frame, block_size):
    # max_allowed_block_size = 150000
    # assert(block_size<=max_allowed_block_size)

    for chan_id in channels:

        with h5py.File(path_to_raw, 'r') as f:
            h5_object = f['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
            
            # Load settings from file
            lsb = h5_object['settings']['lsb'][0]

            # Load raw data from file
            groups = h5_object['groups']
            group0 = groups[next(iter(groups))]

            yield chan_id, group0['raw'][chan_id, start_frame:start_frame+block_size].T * lsb
    
def load_raw_waveforms(path_to_raw, well_no, recording_no, channels, block_size, spike_data):
    t0 = time.time()
    with h5py.File(path_to_raw, 'r') as f:
        h5_object = f['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        
        # Load settings from file
        lsb = h5_object['settings']['lsb'][0]

        # Load raw data from file
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]

        for chan in channels:
            spikes_on_chan = spike_data[spike_data['channel'] == chan]
            n_spikes = len(spikes_on_chan)
            print(f'Working on Channel {chan} with {n_spikes} spikes')

            arr = np.zeros((n_spikes, block_size*2))

            #NS recordings only record 0-1019 in raw file. Skip 1020-1024 if present   
            n_chan = group0['raw'].shape[0]
            if (chan >= n_chan):
                print(f'chan={chan} outside of raw file range')
                continue
            
            for i,spike_frame in enumerate(spikes_on_chan['frameno']):
                start = spike_frame - block_size
                end = spike_frame + block_size
                        
                if start < 0 or end >  group0['raw'].shape[1]:
                    print(f"Skipping chan={chan}, spike={i}, waveform length out of bounds ({start}:{end})")
                    continue

                raw_waveform = group0['raw'][chan, start:end] * lsb
                arr[i,:] = raw_waveform

            print("Elapsed:", time.time() - t0)
            yield chan, arr

def get_n_colors(n, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    return colors

def fig_to_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    ncols, nrows = canvas.get_width_height()
    composite_img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
    # composite_img = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
    # composite_img = composite_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return composite_img

def frame_to_bin(t_frame, samp_rate, bin_size):
    return int(t_frame/samp_rate/bin_size)

def bin_to_frame(t_bin, samp_rate, bin_size):
    return int(t_bin*bin_size*samp_rate)

def frame_to_sec(t_frame, samp_rate):
    return t_frame/samp_rate

def sec_to_frame(t_sec, samp_rate):
    return int(t_sec*samp_rate)

def bin_to_sec(t_bin, bin_size):
    return t_bin*bin_size

def sec_to_bin(t_sec, bin_size):
    return int(np.ceil(t_sec/bin_size))

def get_DIV_from_yymmdd_dates(plate_date_str, curr_date_str):

    # Convert date strings to datetime objects
    date1 = datetime.strptime(plate_date_str, "%y%m%d")
    date2 = datetime.strptime(curr_date_str, "%y%m%d")

    # Calculate the difference
    difference = date2 - date1
    return difference.days # subtract 1 because first day is DIV 0


def get_yymmdd_dates_from_DIV(plate_date_str, DIV):

    # Convert date strings to datetime objects
    dt = datetime.strptime(plate_date_str, "%y%m%d") + timedelta(days=DIV)
    
    return dt.strftime("%y%m%d")


def get_yymmdd_platedate_from_DIV(date_str, DIV):

    # Convert date strings to datetime objects
    dt = datetime.strptime(date_str, "%y%m%d") - timedelta(days=DIV)
    
    return dt.strftime("%y%m%d")


def get_DIVs(path_to_preprocessed_data):

    filepaths = glob(str(Path(path_to_preprocessed_data)/"*"))

    DIVS = []

    for fp in filepaths:
        DIV = int(fp.split('DIV')[-1].split('_')[0])

        DIVS.append(DIV)

    return sorted(DIVS)
