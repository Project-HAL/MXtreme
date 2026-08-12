"""MXtreme -- analysis toolkit for Maxwell Biosystems MaxOne/MaxTwo HD-MEA recordings.

Deliberately lightweight: only the pure-Python identity types are re-exported here. Importing
`Recording`, the preprocessing modules, or anything from `mxtreme.analysis` pulls in
matplotlib/seaborn/h5py, so those stay behind explicit submodule imports to keep `import mxtreme` fast:

    from mxtreme.recording import Recording
    from mxtreme import extract, clean            # raw .h5 -> cleaned npz
    from mxtreme.pipeline import Pipeline
    from mxtreme.bursting import BurstDetector     # cleaned npz -> bursts
    from mxtreme.analysis import activity
"""

__version__ = "0.1.0"

from mxtreme.identity import CultureID, CultureSelector, RecordingID

__all__ = ["CultureID", "CultureSelector", "RecordingID", "__version__"]
