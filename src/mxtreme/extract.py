"""Extraction: raw ``.raw.h5`` recording -> per-well data dictionaries.

:func:`extract` is the *E* of the old ETL pipeline. It opens a single raw Maxwell ``.h5`` file and
pulls out, for every requested well, the spike table plus the metadata and settings needed to clean,
bin, and later analyse it. The result is a plain ``dict`` keyed by well number; each value is itself a
dict that flows through the composable cleaning steps in :mod:`mxtreme.clean` and is finally written by
:func:`mxtreme.io.save_preprocessed`.

The newer file format carries an embedded ``/assay/metadata`` blob that is read automatically. Older
files (e.g. ``May2025_Wave``) lack it, so the caller must pass a ``metadata`` dict for those. A
caller-supplied ``metadata`` dict is merged *over* the embedded blob (per-key), so it can override or
fill in individual fields -- including an optional ``Conditions`` list of per-well conditions.
"""

from __future__ import annotations

import json
import os

import numpy as np


def extract(filepath: str, metadata: dict | None = None, wells: list | int | None = None) -> dict[int, dict]:
    """Extract per-well data from a single raw ``.h5`` file.

    :param filepath: Path to the raw ``.raw.h5`` file.
    :type filepath: str
    :param metadata: Metadata supplied by the caller. It is *merged over* any ``/assay/metadata``
        blob embedded in the file: fields present in both are taken from ``metadata``, and fields
        missing from the embedded blob are filled from ``metadata``. Older files that lack an
        embedded blob rely on ``metadata`` entirely. Typically shaped as::

            {'Exp ID': exp_id, 'Chip ID': chip_id, 'Plate date': plate_date, 'DIV': div}

        An optional ``'Conditions'`` key (a list with one ``[left, right]`` entry per well) supplies
        the per-well experimental condition; it is entirely optional and its absence is not an error.

        An optional ``'Phases'`` key supplies an experiment-level phase spec that is embedded in every
        preprocessed ``.npz`` and reconstructed automatically by :class:`~mxtreme.recording.Recording`
        (see :func:`mxtreme.phases.phases_from_spec`). It is shaped as
        ``{'starts': {name: tag_or_minutes, ...}, 'end': tag_or_minutes}`` where each boundary is
        either a maxlab event tag (``str``) or a number of minutes from the recording's first frame.
    :type metadata: dict, optional
    :param wells: Well number(s) to extract. ``None`` extracts every well present in the file.
    :type wells: list or int, optional
    :raises FileNotFoundError: If ``filepath`` does not exist.
    :raises ValueError: If the file has no embedded metadata and ``metadata`` was not supplied.
    :returns: Mapping of well number to that well's data dict.
    :rtype: dict[int, dict]
    """
    import h5py  # local import keeps `import mxtreme.extract` cheap for callers that only need types

    from mxtreme.utils import sort_spike_data  # local for the same reason (utils pulls matplotlib)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    if isinstance(wells, int):
        wells_to_process = [wells]
    else:
        wells_to_process = wells

    data: dict[int, dict] = {}

    with h5py.File(filepath, "r") as f:
        # Read the metadata embedded in the file, then merge the caller-supplied dict *over* it so
        # caller values win per-key and any fields the embedded blob is missing get filled in. Older
        # files predate the embedded-metadata format, so they rely on ``metadata`` entirely.
        try:
            raw_metadata = f["/assay/metadata"][:][0]
            decoded_metadata = raw_metadata.decode("utf-8").strip().replace("'", '"')
            embedded = json.loads(decoded_metadata)
        except Exception:
            embedded = None

        if embedded is None and metadata is None:
            raise ValueError(
                "No metadata in h5 file. Metadata must be supplied in the format\n"
                "metadata = {'Exp ID': exp_id, 'Chip ID': chip_id, 'Plate date': plate_date, 'DIV': DIV}."
            )

        h5_metadata = {**(embedded or {}), **(metadata or {})}

        if embedded is not None and metadata is not None:
            meta_source = "embedded /assay/metadata + caller override"
        elif embedded is not None:
            meta_source = "embedded /assay/metadata"
        else:
            meta_source = "caller-supplied"

        conditions = h5_metadata.get("Conditions")
        
        # Optional phase spec (experiment-level: applies to every well). Same shape as accepted by
        # ``mxtreme.phases.phases_from_spec`` -- {"starts": {name: tag|minutes}, "end": tag|minutes}.
        phase_spec = h5_metadata.get("Phases")

        h5_metadata["wells"] = []  # track which wells actually carry data in this file

        print("=" * 60)
        print(f"Extracting: {os.path.basename(filepath)}")
        print(f"  Metadata source : {meta_source}")
        print(f"  Conditions      : {'present' if conditions is not None else 'none provided'}")
        print(f"  Wells requested : {'all' if wells_to_process is None else wells_to_process}")
        print("-" * 60)

        well_ids = [int(well_long.split('well')[-1]) for well_long in list(f["/wells/"].keys())]

        for well_long in list(f["/wells/"].keys()):
            well = int(well_long.split('well')[-1])  # well values range from 0-5

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

            # Spiking data, sample rate, channel mapping, least significant bit.
            # The spike table can end with an out-of-order spike from the recording's final buffer
            # flush; burst feature windows are found by binary search, so order is restored at the
            # source. Every cleaning step downstream is order-preserving, so this holds to the .npz.
            spikes = f[f"/recordings/rec0000/well00{well}/spikes"][:]  # [frameno, channel, amplitude]
            well_data["data"] = sort_spike_data(spikes)

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
                if len(well_data["data"]) == 0:
                    raise ValueError(
                        f"{filepath} holds no data for well {well}: no raw frames and no spikes. "
                        "An empty or aborted recording cannot be extracted."
                    )
                print(f"Warning: No raw data for {filepath} for well no {well}")
                well_data["raw_start"] = np.min(well_data["data"]["frameno"])
            else:
                group0 = groups[next(iter(groups))]
                if group0["frame_nos"].shape[0] == 0:
                    raise ValueError(
                        f"{filepath} holds no data for well {well}: the recording has 0 frames. "
                        "An empty or aborted recording cannot be extracted."
                    )
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

            # Experimental condition for this well: the matching entry from the metadata ``Conditions``
            # list (one ``[left, right]`` per well). Optional -- a missing/short list yields ``None``.
            well_condition = None
            if conditions is not None:
                try:
                    idx = well_ids.index(well) if well_ids is not None else well
                    well_condition = conditions[idx]
                except (ValueError, IndexError, TypeError, KeyError):
                    well_condition = None
            well_data["experimental_condition"] = well_condition

            # Phase spec travels with every well so it is embedded in each preprocessed ``.npz`` and
            # picked up automatically by ``Recording`` at detection/analysis time. Optional (``None``).
            well_data["phase_spec"] = phase_spec

            print(f"  well {well} | {len(well_data['data']):>9,} spikes | condition {well_condition}")

    print("-" * 60)
    print(f"Done: extracted {len(data)} well(s) from {os.path.basename(filepath)}")
    print("=" * 60)

    return data
