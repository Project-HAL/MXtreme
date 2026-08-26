# Custom network scan 

import os, sys
import time

import maxlab as mx

from mxtreme.scans import electrode_selection, mx_setup

# User Input

# metadata 
PLATE_DATE = 250828
DIV = 25
CHIP = ""
EXP_ID = "Network Scan" # the task 
WELLS = [0,4] # list of wells between 0-5

# saving 
SAVE_DIR = ""
H5_FILENAME = ""

# network scan parameters
REC_LENGTH = 60 # recording length in seconds
PATH_TO_AS = "" # this holds the activity scan data for all wells for the specified chip
REC_ELECS = None # Alternative to PATH_TO_AS, supply recording electrodes as a dictionary {well:[electrode_list]}

# logging
DESCRIPTION = 'Testing AS->NS pipeline and metadata'

if __name__=='__main__':

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = mx_setup.get_system_type()
    if device==0:
        print("MaxOne")
    else:
        print("MaxTwo")

    if REC_ELECS is None and PATH_TO_AS!='':
        # Determine recording electrodes from activity scans
        REC_ELECS = electrode_selection.select_electrodes(PATH_TO_AS, SAVE_DIR)

    # initialize wells 
    for well in WELLS:
        mx_setup.init_well(well, REC_ELECS[well], stim_elecs=[])
    
    # Assay information
    metadata = {'Plate date' : PLATE_DATE,
                'DIV' : DIV,
                'Chip ID': CHIP,
                'Exp ID' : EXP_ID,
                'Well IDs' : WELLS, 
                'Conditions' :[]} # conditions decoded based on the particular task

    s = mx.Saving() # a single h5 with multiple wells as different groups
    s.open_directory(SAVE_DIR)
    s.start_file(H5_FILENAME)
    s.group_delete_all()

    # Write info to h5 assay/
    mx_setup.write_exp_description(s, DESCRIPTION)
    metadata = mx_setup.write_metadata(s, metadata)

    for well in WELLS:
        s.group_define(well, f"all_channels_{well}", list(range(1024)))

    mx.activate(WELLS) # activate all wells

    s.start_recording(WELLS)

    print("Start recording")
    time.sleep(REC_LENGTH)
    print("Stop recording")

    s.stop_recording()

    time.sleep(mx.Timing.waitAfterRecording)

    s.stop_file()
    s.group_delete_all()




    




