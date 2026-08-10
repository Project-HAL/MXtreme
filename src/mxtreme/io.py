"""Reading and writing preprocessed data -- the *L* (load) of the old ETL pipeline.

All writers take an explicit output directory; none fall back to a lab-specific default. In the example
workflow the directories come from a :class:`mxtreme.config.Config` (e.g. ``config.preprocessed_dir``).

Contents:
- :func:`load_preprocessed` -- read a cleaned ``.npz`` back into a dict.
- :func:`save_preprocessed` -- write one well's cleaned data to an ``.npz``.
- :func:`write_experimental_conditions` -- append per-culture stimulation conditions to a CSV.
- :func:`register` -- record processed recordings in the registry CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_preprocessed(filepath: str | Path) -> dict:
    """Load a preprocessed ``.npz`` produced by :func:`save_preprocessed`.

    :param filepath: Path to the ``.npz`` file.
    :type filepath: str or Path
    :raises FileNotFoundError: If ``filepath`` does not exist.
    :returns: The saved arrays keyed by name (e.g. ``spike_data``, ``channelmap``, ``samp_rate``).
    :rtype: dict
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")
    return dict(np.load(filepath, allow_pickle=True))


def _unique_path(path: Path) -> Path:
    """Return ``path`` unchanged, or a suffixed variant (``_01``, ``_02``, ...) if it already exists.

    :param path: Desired output path.
    :type path: Path
    :returns: A path that does not yet exist.
    :rtype: Path
    """
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter:02d}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def save_preprocessed(datastore: str | Path, well: dict, *, overwrite: bool = True) -> Path:
    """Write one well's cleaned data to a compressed ``.npz`` under the managed store.

    The file lands at ``<datastore>/<exp_id>/<chip>/well<well>/DIV<DIV>_<plate_date>_<chip>_<exp_id>_well<well>_exp_data.npz``.
    Fields produced by optional cleaning steps (channel map, binning, stimulation removal) are written
    when present and defaulted otherwise, so the pipeline still saves if a user omits a step. The
    provenance dict ``well['preprocessing_params']`` is saved alongside the data.

    :param datastore: Directory under which to write (typically ``config.preprocessed_dir``).
    :type datastore: str or Path
    :param well: A well data dict after running the pipeline.
    :type well: dict
    :param overwrite: If ``False``, avoid clobbering an existing file by adding a numeric suffix.
    :type overwrite: bool
    :returns: The path the data was written to.
    :rtype: Path
    """
    datastore = Path(datastore)
    exp_id, chip, div, plate_date, well_no = (
        well["exp_id"], well["chip"], well["DIV"], well["plate_date"], well["well"],
    )

    out_dir = datastore / exp_id / chip / f"well{well_no}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"DIV{div}_{plate_date}_{chip}_{exp_id}_well{well_no}_exp_data.npz"
    if not overwrite:
        out_path = _unique_path(out_path)

    np.savez_compressed(
        out_path,
        spike_data=well["data"],
        channelmap=well.get("channelmap", np.zeros((0, 5))),
        stim_elecs=well.get("stim_elecs"),
        rec_t_sec=np.array([well.get("rec_t_sec", np.nan)]),
        samp_rate=np.array([well["samp_rate"]]),
        lsb=np.array(well["lsb"]),
        eventtime=well["eventtime"],
        event_messages=well["event_messages"],
        spike_bin=well.get("spike_bin", np.zeros((0, 0))),
        bin_size=well.get("bin_size", np.nan),
        exp_condition=well["experimental_condition"],
        DIV=div,
        plate_date=plate_date,
        well=well_no,
        chip=chip,
        exp_id=exp_id,
        stim_frames=well.get("stim_frames", []),
        raw_start=well["raw_start"],
        path_to_h5=well["path_to_h5"],
        preprocessing_params=np.asarray(well.get("preprocessing_params", {})),
    )

    print(f"Transformed data saved to: {out_path}")
    return out_path


def write_experimental_conditions(data: dict[int, dict], datastore: str | Path) -> None:
    """Append each well's stimulation condition to a per-culture CSV (deduplicated by row).

    :param data: Mapping of well number to well data dict (from :func:`mxtreme.extract.extract`).
    :type data: dict[int, dict]
    :param datastore: Directory under which to write (typically ``config.experimental_conditions_dir``).
    :type datastore: str or Path
    """
    datastore = Path(datastore)

    for well_no, well in data.items():
        new_row = {
            "chip": well["chip"],
            "well": well_no,
            "DIV": well["DIV"],
            "experimental condition": well["experimental_condition"],
        }
        savepath = datastore / well["exp_id"] / f'{well["chip"]}_well{well_no}_exp_conditions.csv'

        if savepath.exists():
            df = pd.read_csv(savepath)
            new_row_df = pd.DataFrame([new_row])
            row_exists = df.apply(
                lambda row: all(str(row[col]) == str(new_row[col]) for col in new_row), axis=1
            ).any()
            if not row_exists:
                df = pd.concat([df, new_row_df], ignore_index=True)
                df.to_csv(savepath, index=False)
        else:
            os.makedirs(savepath.parent, exist_ok=True)
            df = pd.DataFrame([new_row], columns=["chip", "well", "DIV", "experimental condition"])
            df.to_csv(savepath, mode="w", header=True, index=False)


def register(data: dict[int, dict], registry_path: str | Path) -> None:
    """Upsert one row per processed recording into the registry CSV.

    Rows are keyed by ``(exp_id, chip, well, div)``; an existing row for the same key is replaced. Each
    row is marked ``status="complete"`` so downstream path resolution can filter on completed recordings.

    :param data: Mapping of well number to well data dict.
    :type data: dict[int, dict]
    :param registry_path: Path to the registry CSV (typically ``config.registry_path``).
    :type registry_path: str or Path
    """
    registry_path = Path(registry_path)
    os.makedirs(registry_path.parent, exist_ok=True)

    df = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()

    for well_no, well in data.items():
        new_row = {
            "exp_id": well["exp_id"],
            "chip": well["chip"],
            "well": well_no,
            "div": well["DIV"],
            "status": "complete",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        if not df.empty:
            mask = (
                (df.exp_id == well["exp_id"])
                & (df.chip == well["chip"])
                & (df.well == well_no)
                & (df.div == well["DIV"])
            )
            df = df[~mask]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

    df.to_csv(registry_path, index=False)
