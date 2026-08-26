# Annotations are deferred so this module imports on the rig's Python: MaxLab Live ships an
# interpreter older than 3.10, where evaluating `List[int] | str` below raises TypeError.
from __future__ import annotations

import os
import datetime
import re # regular expressions
import shutil
from typing import List
import sys
import time

try:
    import maxlab as mx
except ModuleNotFoundError as exc:  # pragma: no cover -- depends on a rig-only install
    raise ModuleNotFoundError(
        "mxtreme.scans.mx_setup requires the 'maxlab' Python API, which ships with MaxWell's "
        "MaxLab Live software and is not available from PyPI. Install MaxLab Live on the rig "
        "machine and put its Python package on PYTHONPATH. Electrode selection "
        "(mxtreme.scans.electrode_selection) does not need maxlab."
    ) from exc

import mxtreme.scans.mx_config as cfg
from mxtreme import device

phase_event_id = 200

def get_system_type():
    """
    Queries the wellplate version

    :return: Returns 0 for MaxOne and 1 for MaxTwo
    :rtype: int
    """
    with mx.comm.api_context() as api:
        system_type = int(api.send("wellplate_query_version"))

    return system_type
    

def build_frequency_list(num_levels: int, lower: float, upper: float) -> List[float]:
    """creates a list of stimulation frequencies for frequency game. Given the 
    inclusive bounds and the number of levels, it automatically generates 
    a list of floats with corresponding frequencies.   

    :param num_levels: number of frequencies to generate
    :type num_levels: int
    :param lower: lower bound, in Hz
    :type lower: float
    :param upper: upper bound, in Hz
    :type upper: float
    :return: list of frequencies
    :rtype: List[float]
    """
    diff = upper - lower
    acceleration = diff / (num_levels - 1) 
    full_list = [lower]
    curr_level = lower

    for _ in range(1, num_levels):
        curr_level += acceleration
        full_list.append(curr_level)

    return full_list


def write_frequency_list(file_path: str, frequencies: List[float]):
    """Writes a list of floats, representing stimulation frequency levels for 
    frequency game, to a file.

    :param file_path: Path of file to write to
    :type file_path: str
    :param frequencies: List of frequencies
    :type frequencies: List[float]
    """
    freq_strs = list(map(str, frequencies))
    output = ",".join(freq_strs)
    
    with open(file_path, "w") as file:
        file.write(output)
        
    

def get_electrode_square(center: int, radius: int, step_width=1):
    """Generates a square of electrodes centered around a single electrode

    :param center: the target electrode to center the square around
    :type center: ``int``
    :param radius: half the length of the square, measured in steps/chosen 
                   electrodes
    :type radius: ``int``
    :param step_width: how far to "step" when choosing electrodes in the square, 
                       defaults to 1
    :type step_width: ``int``, optional
    :return: list of electrodes
    :rtype: ``List[int]``
    """
    chip_width = device.CHIP_WIDTH

    top_left = center

    center_coord = [center % chip_width, center // chip_width]

    top_left = [center_coord[0] - radius * step_width, center_coord[1] - radius * step_width]

    coords = [top_left]
    for row in range(radius * 2 + 1):
        left = [top_left[0] + step_width * row, top_left[1]]
        for col in range(radius * 2 + 1):
            curr = [left[0], left[1] + step_width * col]
            coords.append(curr)

    return [coord_to_elec(c) for c in coords if in_bounds(c)]


def in_bounds(coord):
    return (coord[0] >= 0 and coord[0] < device.CHIP_WIDTH
            and coord[1] >= 0 and coord[1] < device.CHIP_HEIGHT)

def coord_to_elec(coord):
    return coord[1] * device.CHIP_WIDTH + coord[0]
            
def get_file_names(well_no: int, experiment_name: str, trial_path: str):
    """Returns names and paths for recording directory, config, and 
    h5 file for sequential experiments.

    :param well_no: well number
    :type well_no: int
    :param experiment_name: name of the experiment
    :type experiment_name: str
    :param trial_path: full path of the trial
    :type trial_path: str

    :return: tuple of recording directory, experiment name, and config, 
             respectively
    :rtype: ``List[str]``
    """
    recording_dir = f"{trial_path}/well{well_no}"
    ret = (recording_dir,
           f"{experiment_name}_well_{str(well_no)}",
           f"{recording_dir}/{experiment_name}_well_{well_no}_config.cfg")

    return ret

def build_training_sequences(well_no: int):
    """Builds general event sequences: pre_recording_start (ID: 2), closed_loop_start
    (ID: 3), post_recording_start (ID: 4), and end_experiment (ID: 5)

    These are meant to be used to signal when the training actually happens in 
    the closed-loop program.

    :param well_no: well number
    :type well_no: ``int``
    """
    global phase_event_id

    start_seq = mx.Sequence(f"pre_recording_start_{well_no}", persistent = False)
    del start_seq
    start_seq = mx.Sequence(f"pre_recording_start_{well_no}", persistent = True)
    # print("Phase event number:", phase_event_id)
    start_seq.append(mx.Event(well_no, 1, phase_event_id, f"pre_recording_start {well_no}"))
    phase_event_id += 1

    train_seq = mx.Sequence(f"closed_loop_start_{well_no}", persistent = False)
    del train_seq
    train_seq = mx.Sequence(f"closed_loop_start_{well_no}", persistent = True)
    train_seq.append(mx.Event(well_no, 1, phase_event_id, f"closed_loop_start {well_no}"))
    # print("Phase event number:", phase_event_id)
    phase_event_id += 1

    post_seq = mx.Sequence(f"post_recording_start_{well_no}", persistent = False)
    del post_seq
    post_seq = mx.Sequence(f"post_recording_start_{well_no}", persistent = True)
    post_seq.append(mx.Event(well_no, 1, phase_event_id, f"post_recording_start {well_no}"))
    # print("Phase event number:", phase_event_id)
    phase_event_id += 1

    end_seq = mx.Sequence(f"end_experiment_{well_no}", persistent = False)
    del end_seq
    end_seq = mx.Sequence(f"end_experiment_{well_no}", persistent = True)
    end_seq.append(mx.Event(well_no, 1, phase_event_id, f"end_experiment {well_no}"))
    # print("Phase event number:", phase_event_id)
    phase_event_id += 1

def build_burst_detection_sequences(well_no: int):
    global phase_event_id

    left_burst_seq = mx.Sequence(f"left_burst_detected_{well_no}", persistent = False)
    del left_burst_seq
    left_burst_seq = mx.Sequence(f"left_burst_detected_{well_no}", persistent = True)
    # print("Phase event number:", phase_event_id)
    left_burst_seq.append(mx.Event(well_no, 1, phase_event_id, f"left_burst_detected {well_no}"))
    phase_event_id += 1

    right_burst_seq = mx.Sequence(f"right_burst_detected_{well_no}", persistent = False)
    del right_burst_seq
    right_burst_seq = mx.Sequence(f"right_burst_detected_{well_no}", persistent = True)
    # print("Phase event number:", phase_event_id)
    right_burst_seq.append(mx.Event(well_no, 1, phase_event_id, f"right_burst_detected {well_no}"))
    phase_event_id += 1


def init_well(well: int, rec_elecs: List[int] | str, stim_elecs: List[int], connect=True, power_up=True, dac_source=0) -> mx.Array:
    """ Initilizes the well recording electrodes and stimulation electrodes. 
    By default, it will connect the stimulation units and power them, but this 
    can be overridden by setting the last two arguments to False or by passing
    an empty list for ``stim_elecs``. This can be useful for setting 
    stimulations to different DAC channels, since this function sets all 
    stimulation electrodes to DAC 0 automatically. The stimulation electrodes 
    will also be selected for recording, so they don't need to be in the 
    recording electrodes list.

    **Important Note:** This function will pick recording electrodes by adding 
    ``stim_elecs`` and ``rec_elecs``, meaning that ``rec_elecs`` does not have 
    to contain everything in ``stim_elecs``. However, if the number of electrodes
    in both lists combined is greater than 1020, then some of the electrodes 
    *will not be chosen*. If that happens, ``stim_elecs`` will have preference 
    over ``rec_elecs``. 


    :param well_no: Well number
    :type well_no: ``int``
    :param rec_elecs: Recording electrodes, as a list of electrode numbers OR as a
                      full path to a MaxWell configuration file
    :type rec_elecs: ``List[int]`` | str
    :param stim_elecs: Stimulation electrodes, as a list of electrode numbers. 
                       If empty, connecting and powering up will not be 
                       attempted.
    :type stim_elecs: ``List[int]``
    :param connect: Indicates if the stimulation units should be connected, 
                    defaults to True
    :type connect: ``bool``, optional
    :param power_up: Indicates if the stimulation units should be powered up, 
                    defaults to True
    :type power_up: ``bool``, optional
    :return: a maxlab array with the configured electrodes
    :rtype: ``mx.Array``
    """

    if stim_elecs == []:
        connect = False
        power_up = False

    mx.activate([well])
    # mx.set_primary_well(well) 
    
    array = mx.Array(f"stimulation{well}", persistent=False)
    array.close()
    array = mx.Array(f"stimulation{well}", persistent=True)
    array.reset()
    array.clear_selected_electrodes()
    if type(rec_elecs) is str:
        elec_nums = cfg.read_config_elecs(rec_elecs)
    else:
        elec_nums = rec_elecs
    
    array.select_electrodes(stim_elecs + elec_nums)
    
    array.select_stimulation_electrodes(stim_elecs)
    
    array.route()

    if connect:
        stimulation_units = connect_stim_units_to_stim_electrodes(stim_elecs, array)
    else:
        stimulation_units = None
    
    array.download([well])
    time.sleep(mx.Timing.waitAfterDownload)

    if power_up:
        power_up_stim_units(stimulation_units, dac=dac_source)

    return array, stimulation_units


#TODO: move this to an Array class.
def connect_stim_units_to_stim_electrodes(stim_electrodes: List[int], array: mx.Array) -> List[int]:
    """Connects stimulation units to stimulation electrodes. This function 
    was taken from example code from MaxWell.

    :param stim_electrodes: list of stimulation electrodes to connect
    :type stim_electrodes: ``List[int]``
    :param array: MaxWell array generated from a list of recording/stimulation electrodes
    :type array: ``mx.Array``
    :raises RuntimeError: when a stimulation channel cannot connect to a stimulation electrode
    :raises RuntimeError: when two electrodes are connected to the same unit
    :return: a list of stimulation units
    :rtype: ``List[int]``
    """
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if len(stim) == 0:
            raise RuntimeError(
                f"No stimulation channel can connect to electrode: {str(stim_el)}"
            )
        stim_unit_int = int(stim)
        if stim_unit_int in stim_units:
            raise RuntimeError(
                f"Two electrodes connected to the same stim unit. This is not allowed. Please Select a neighboring electrode of {stim_el}!"
            )
        else:
            stim_units.append(stim_unit_int)
    return stim_units

def power_up_stim_units(stimulation_units: List[int], dac=0):
    """Powers up stimulation units for an experiment

    :param stimulation_units: stimulation units to power up
    :type stimulation_units: ``List[int]``
    """
    for stimulation_unit in stimulation_units:
            stimulation = mx.StimulationUnit(stimulation_unit).power_up(True).connect(True).set_voltage_mode().dac_source(dac)
            mx.send(stimulation)

def write_exp_description(s: mx.Saving, description: str) -> None:
    """Writes experiment description as a property to the experiment h5 file
    under key "description"

    :param s: Experiment to save the stimulation electrodes for
    :type s: ``mx.Saving``
    :param description: description of the experiment
    :type description: ``str``
    """
    s.write_assay_property("description", description)

def write_exp_notes(s: mx.Saving, notes: str) -> None:
    """Writes experiment notes as a property to the experiment h5 file
    under key "notes"

    :param s: Experiment to save the stimulation electrodes for
    :type s: ``mx.Saving``
    :param notes: notes of the experiment
    :type notes: ``str``
    """
    s.write_assay_property("notes", notes)

def write_metadata(s: mx.Saving, metadata: dict):
    """
    Writes metadata to experiment h5 file under the key metadata. 

    :param s: mx.Saving object associated with a particular experiment 
    :type s: mx.Saving
    :param metadata: Experiment metadata 
    :type metadata: dict
    """     

    # Format checks
    if not re.fullmatch(r"\d{6}", str(metadata['Plate date'])):
        raise ValueError("Plate date must be in the format 'YYMMDD'.")
    
    if not re.fullmatch(r"\d+", str(metadata['DIV'])):
        raise ValueError("DIV must be an integer.")
    
    if not isinstance(metadata["Well IDs"], list) and not all(isinstance(i,int) for i in metadata["Well IDs"]):
        raise ValueError("Well IDs must be a list of integers.")
    
    if not isinstance(metadata["Conditions"], list) or (len(metadata["Well IDs"]) != len(metadata["Conditions"]) and len(metadata["Conditions"])!=0):
        raise ValueError("Conditions must be a list the same length as the number of wells or length 0.")
    
    # Convert the experiment ID to camelCase so there's no underscores or spaces
    # e.g. "wave training" or "wave_training" --> "waveTraining"
    parts = re.split(r'[_\s]+', metadata["Exp ID"].strip())
    if parts:
        if len(parts)>1:
            metadata["Exp ID"] = parts[0].lower() + ''.join(word.capitalize() for word in parts[1:]) # replace experiment ID in the metadata dict
        else:
            metadata["Exp ID"] = parts[0][0].lower()+parts[0][1:] # make sure it starts lowercase
    else:
        raise ValueError("Enter an valid string Experiment ID.")
    
    s.write_assay_property("metadata", str(metadata))

    return metadata

    # TODO: read the python dict as a string back in with json.loads(json_string)

def write_stim_electrodes(s: mx.Saving, stim_electrodes: List[int]):
    """Writes the stimulation electrodes as a property to the experiment h5 file
    under key "stim_electrodes"

    Note that this function must be called after opening the file

    :param s: Experiment to save the stimulation electrodes for
    :type s: ``mx.Saving``
    :param stim_electrodes: stimulation electrodes
    :type stim_electrodes: ``List[int]``
    """
    val = ""
    for stim in stim_electrodes:
        val += str(stim) + ", "
    val = val[:len(val) - 2]
    
    s.write_assay_property("stim_elecs", ", ".join(str(x) for x in stim_electrodes))