# Hyperparameters

# Preprocessing
AMPLITUDE_THRESH = 2e-5 # V (20 uV)
BIN_SIZE = 0.01
REFRACTORY_PERIOD=0.002 # in seconds 
POST_STIM_PERIOD = 0 #500 # this is the number of frames after the end-stimulation event tag that we want to remove from the spike data due to artifacts
NOISE_THRESH = 0.01 # what we consider noise

# Burst Detection
GAUSSIAN_SIGMA = 1 # sigma for the 1D Gaussian filter to smooth spike bin
BURST_THRESH = 0.2
K = 0.25 # dynamics burst detection (k std/mad from mean/median)
DIST_BTW_BURSTS= 30 # in bins
PROMINENCE_PERCENTILE = 25
N=300 # ISI-N burst detection - min num spikes comprising a burst

# Burst Features
ONSET_THRESH_PCT = 0.1 #0.2
OFFSET_THRESH_PCT = 0.05

# Spike Features
ISI_threshold = 200 # ms - MAxLab threshold

# Channel Map / device geometry moved to mxtreme.device (CHIP_WIDTH, CHIP_HEIGHT, ELEC_SIZE).


# Original Experiments

waveTraining = {'plate_date':'250403'}
stimRemoval = {'plate_date':'240703'}
sydneyDec2024 = {'plate_date':'241114'}
stimRemovalNull = {'plate_date':'240703'}
control = {'plate_date':'2024'} # Batch 4 - spring 2024 (don't know the plate date and doesn't have date collected info either)
scans = {}

experiment_info = {'waveTraining' : waveTraining,
                   'stimRemoval' : stimRemoval,
                   'sydneyDec2024' : sydneyDec2024,
                    'stimRemovalNull' : stimRemovalNull,
                    'control' : control,
                    }  


# Data stores
braintrix_mac = '/Volumes/LevinLab_BRAINTRIX$/'
braintrix_linux = '/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=levinlab_braintrix$/'
braintrix_mxwbio_m2 = '/run/user/1000/gvfs/smb-share:server=rstore.it.tufts.edu,share=levinlab_braintrix$/'
wes_rstore_linux = '/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/'
wes_rstore_windows = r'\\rstore.it.tufts.edu\\as_rsch_levinlab_wclawson01$\\'

PARENT_DIR = braintrix_linux + 'HALnalysis'