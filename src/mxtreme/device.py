"""Device geometry for the Maxwell Biosystems HD-MEA chips.

Physical layout of the electrode array, used to place channels/electrodes in real coordinates (e.g. for
the MEA spatial map). These describe the MaxOne/MaxTwo electrode grid; other device versions would supply
their own values here.
"""

CHIP_WIDTH = 220    # array width, in electrodes
CHIP_HEIGHT = 120   # array height, in electrodes
ELEC_SIZE = 17.5    # electrode pitch, in µm
