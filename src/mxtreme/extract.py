"""Extraction: raw ``.raw.h5`` recording -> per-well data dictionaries.

:func:`extract` is the *E* of the old ETL pipeline. It opens a single raw Maxwell ``.h5`` file and
pulls out, for every requested well, the spike table plus the metadata and settings needed to clean,
bin, and later analyse it. The result is a plain ``dict`` keyed by well number; each value is itself a
dict that flows through the composable cleaning steps in :mod:`mxtreme.clean` and is finally written by
:func:`mxtreme.io.save_preprocessed`.

The newer file format carries an embedded ``/assay/metadata`` blob that is read automatically. Older
files (e.g. ``May2025_Wave``) lack it, so the caller must pass a ``metadata`` dict for those.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np


def extract(filepath: str, metadata: dict | None = None, wells: list | int | None = None) -> dict[int, dict]:
    """Extract per-well data from a single raw ``.h5`` file.

    :param filepath: Path to the raw ``.raw.h5`` file.
    :type filepath: str
    :param metadata: Fallback metadata for older files that lack an embedded ``/assay/metadata``
        blob. Ignored when the file carries its own metadata. Must be shaped as::

            {'Exp ID': exp_id, 'Chip ID': chip_id, 'Plate date': plate_date, 'DIV': div}

    :type metadata: dict, optional
    :param wells: Well number(s) to extract. ``None`` extracts every well present in the file.
    :type wells: list or int, optional
    :raises FileNotFoundError: If ``filepath`` does not exist.
    :raises ValueError: If the file has no embedded metadata and ``metadata`` was not supplied.
    :returns: Mapping of well number to that well's data dict.
    :rtype: dict[int, dict]
    """
    import h5py  # local import keeps `import mxtreme.extract` cheap for callers that only need types

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    if isinstance(wells, int):
        wells_to_process = [wells]
    else:
        wells_to_process = wells

    data: dict[int, dict] = {}

    with h5py.File(filepath, "r") as f:
        # Prefer the metadata embedded in the file; fall back to the caller-supplied dict for older
        # files that predate the embedded-metadata format.
        try:
            raw_metadata = f["/assay/metadata"][:][0]
            decoded_metadata = raw_metadata.decode("utf-8").strip().replace("'", '"')
            h5_metadata = json.loads(decoded_metadata)
        except Exception:
            if metadata is None:
                raise ValueError(
                    "No metadata in h5 file. Metadata must be supplied in the format\n"
                    "metadata = {'Exp ID': exp_id, 'Chip ID': chip_id, 'Plate date': plate_date, 'DIV': DIV}."
                )
            h5_metadata = metadata

        h5_metadata["wells"] = []  # track which wells actually carry data in this file

        for i, well_long in enumerate(list(f["/wells/"].keys())):
            well = int(well_long[-1])  # well values range from 0-5

            if wells_to_process is not None and well not in wells_to_process:
                continue

            well_data: dict = {}
            data[well] = well_data
            h5_metadata["wells"].append(well)

            # Experiment identity
            well_data["well"] = well
            well_data["exp_id"] = h5_metadata["Exp ID"]
            well_data["chip"] = h5_metadata["Chip ID"]
            well_data["plate_date"] = h5_metadata["Plate date"]
            well_data["DIV"] = h5_metadata["DIV"]
            well_data["path_to_h5"] = filepath

            # Spiking data, sample rate, channel mapping, least significant bit
            well_data["data"] = f[f"/recordings/rec0000/well00{well}/spikes"][:]  # [frameno, channel, amplitude]

            samp_rate = np.array([f[f"/recordings/rec0000/well00{well}/settings/sampling"][:][0]])
            if isinstance(samp_rate, (list, np.ndarray)):
                samp_rate = samp_rate[0]  # sometimes [samp_rate], sometimes [[samp_rate]]
            well_data["samp_rate"] = samp_rate
            well_data["mapping"] = f[f"/recordings/rec0000/well00{well}/settings/mapping"][:]  # [channel, electrode, x, y]
            well_data["lsb"] = np.array([f[f"/recordings/rec0000/well00{well}/settings/lsb"][:][0]])

            # Stimulation electrodes
            well_data["stim_elecs"] = None
            if "stim_elecs" in f["/assay"]:
                stim_elecs_raw = f["/assay/stim_elecs"][:]
                well_data["stim_elecs"] = np.array(stim_elecs_raw[0].decode().split(", "), dtype=int)

            # Start frame of the raw recording (used to normalise spike/event frames to start at 0)
            h5_object = f["wells"]["well{0:0>3}".format(well)]["rec{0:0>4}".format(0)]
            groups = h5_object["groups"]
            if not groups.keys():
                print(f"Warning: No raw data for {filepath} for well no {well}")
                well_data["raw_start"] = np.min(well_data["data"]["frameno"])
            else:
                group0 = groups[next(iter(groups))]
                well_data["raw_start"] = int(group0["frame_nos"][0])

            # Event data (stimulation start/stop markers, etc.)
            events = None
            well_data["event_messages"] = []
            event_keys = []
            if "events" in f[f"/recordings/rec0000/well00{well}"]:
                events = f[f"/recordings/rec0000/well00{well}/events"][:]
                for raw_message in events["eventmessage"]:
                    try:
                        decoded = raw_message.decode("utf-8").strip()
                        parsed = json.loads(decoded)
                        key = list(parsed.keys())[0]
                        well_data["event_messages"].append(parsed)
                        event_keys.append(key)
                    except Exception as e:
                        print(f"Failed to decode or parse event: {e}")

            well_data["eventtime"] = events["frameno"] if events is not None else np.array([], dtype=np.int64)

            # Experimental conditions (left/right stimulation types)
            well_data["experimental_condition"] = _extract_experimental_condition(f, i, well_data["exp_id"], well)

    return data


def _extract_experimental_condition(f, i: int, exp_id: str, well: int) -> np.ndarray:
    """Derive the ``{left_stim, right_stim}`` condition for one well.

    Parses ``/assay/closed_loop_args`` when present, honouring the ``stimRemoval`` special case.

    :param f: Open :class:`h5py.File` handle.
    :param i: Enumeration index of the well within the file (indexes into per-well arg lists).
    :type i: int
    :param exp_id: Experiment identifier for this well.
    :type exp_id: str
    :param well: Well number (used only for logging).
    :type well: int
    :returns: A 0-d object array wrapping ``{'left_stim': int, 'right_stim': int}``.
    :rtype: numpy.ndarray
    """
    if "closed_loop_args" in f["/assay"]:
        closed_loop_args = f["/assay/closed_loop_args"][:][0].decode("utf-8").strip()

        if "--left-type" in closed_loop_args and "--right-type" in closed_loop_args:
            left_stim_list = ast.literal_eval(closed_loop_args.split("--left-type")[-1].split("--")[0])
            right_stim_list = ast.literal_eval(closed_loop_args.split("--right-type")[-1].split("--")[0])
            try:
                left_stim = int(left_stim_list[i])
                right_stim = int(right_stim_list[i])
            except Exception:  # a single-well experiment may not store conditions in a list
                left_stim = int(left_stim_list)
                right_stim = int(right_stim_list)
            condition = {"left_stim": left_stim, "right_stim": right_stim}
        else:
            condition = {"left_stim": 0, "right_stim": 0}  # controls, network scans

        print("well", well, ":", condition)

        if exp_id == "stimRemoval":
            condition = {"left_stim": 0, "right_stim": 2}
        return np.asarray(condition)

    if exp_id == "stimRemoval":
        return np.asarray({"left_stim": 0, "right_stim": 2})
    return np.asarray({"left_stim": 0, "right_stim": 0})  # no stimulation
