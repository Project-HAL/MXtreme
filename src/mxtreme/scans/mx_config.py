import re
import os
from typing import List

from mxtreme import device

# if the buttons are electrode lists, 
# then they will be named "buttonA.cfg", "buttonB.cfg" etc. If one is already
# a string, then its basename will be taken and the corresponding letter will be
# skipped (assuming there are other List[int] type buttons)
def get_buttons(full_config, buttons, save_path):
    # keys and values are STRINGS
    full_config_map = read_config_electrode_to_channel(full_config)
    buttons_data = [read_config(b) if isinstance(b, str) else make_config_args(b) for b in buttons]

    for button in buttons_data:
        for data in button:
            try:
                data[0] = full_config_map[data[1]]
            except Exception as err:
                print(f"{data[1]} not routed")
                button.remove(data)

    new_buttons = []
    curr_button = "A"

    for i in range(len(buttons)):
        filename = f"button{curr_button}.cfg"
        curr_button = chr(ord(curr_button) + 1)
        if (isinstance(buttons[i], str)):
            filename = os.path.basename(buttons[i])
            
        new_path = save_path + "/" + filename
        print(f"writing {new_path}")

        make_config(buttons_data[i], new_path)
        new_buttons.append(new_path)

    return new_buttons

# makes a list of config args (channel no (0), elec no, elec coords) given a 
# list of electrode numbers
def make_config_args(elecs: List[int]):
    return [
        [-1,
         str(e),
         str((e // device.CHIP_WIDTH) * device.ELEC_SIZE),
         str((e % device.CHIP_WIDTH) * device.ELEC_SIZE)]
        for e in elecs
    ]
    
    
def make_config(data, file_name):
    f = open(file_name, "w")

    to_write = ""

    for d in data:
        if d[0] != -1:
            to_write += f"{d[0]}({d[1]}){d[2]}/{d[3]};"

    to_write += "\nH"

    f.write(to_write)

def read_config_elecs(config_path: str):
    """Gets electrode numbers from a MaxWell electrode configuration file.

    :param file: full path of config 
    :type file: ``str``
    :return: list of electrodes
    :rtype: ``List[int]``
    """
    data = read_config(config_path)
    
    elecs = [int(e[1]) for e in data]

    return elecs


def read_config(file: str):
    """Reads all electrode data from a MaxWell configuration file

    :param file: full path of the file
    :type file: ``str``
    :return: a list containing data for each electrode (channel, electrode, x, y)
    :rtype: ``List[List[str]]``
    """

    # Error checking
    if (not file.endswith('.cfg')):
        return False


    f = open(file, "r")
    contents = f.read()

    channels = contents.split(";")
    channels.pop(len(channels) - 1)

    all_electrodes = []
    for channel_data in channels:
        data = re.split("\\(|\\)|/", channel_data)
        all_electrodes.append(data)


    return all_electrodes

def read_config_electrode_to_channel(file: str):
    """Reads channel-electrode number mappings from a config file

    :param file: full path of the file
    :type file: str
    :return: a dictionary of strings, where the keys are channel numbers and the 
             values are electrode numbers
    :rtype: ``Dict[str]``
    """
    data = read_config(file)

    electrodes = {}

    for channel_data in data:
        electrodes[channel_data[1]] = channel_data[0]

    return electrodes