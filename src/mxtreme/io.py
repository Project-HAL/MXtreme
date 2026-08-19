"""Reading and writing preprocessed data -- the *L* (load) of the old ETL pipeline.

All writers take an explicit output directory; none fall back to a lab-specific default. In the example
workflow the directories come from a :class:`mxtreme.config.Config` (e.g. ``config.preprocessed_dir``).

Contents:
- :func:`load_preprocessed` -- read a cleaned ``.npz`` back into a dict.
- :func:`save_preprocessed` -- write one well's cleaned data to an ``.npz`` (and register it).
- :func:`register` -- record processed recordings in the registry CSV.
- :func:`rebuild_registry` -- rebuild that registry by scanning the store (recovery path).
- :func:`save_burst_data` / :func:`load_burst_data` -- per-recording burst CSVs.
- :func:`update_burst_log` -- per-experiment burst summary CSV.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
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


def save_preprocessed(
    datastore: str | Path,
    well: dict,
    *,
    overwrite: bool = True,
    registry_path: str | Path | None = None,
) -> Path:
    """Write one well's cleaned data to a compressed ``.npz`` under the managed store.

    The file lands at ``<datastore>/<exp_id>/<chip>/well<well>/DIV<DIV>_<plate_date>_<chip>_<exp_id>_well<well>_exp_data.npz``.
    Fields produced by optional cleaning steps (channel map, binning, stimulation removal) are written
    when present and defaulted otherwise, so the pipeline still saves if a user omits a step. The
    provenance dict ``well['preprocessing_params']`` is saved alongside the data.

    The recording is also upserted into the registry CSV (via :func:`register`) so the registry is
    always refreshed whenever an ``.npz`` is written -- keeping its timestamps current without a
    separate manual step.

    :param datastore: Directory under which to write (typically ``config.preprocessed_dir``).
    :type datastore: str or Path
    :param well: A well data dict after running the pipeline.
    :type well: dict
    :param overwrite: If ``False``, avoid clobbering an existing file by adding a numeric suffix.
    :type overwrite: bool
    :param registry_path: Path to the registry CSV. Defaults to ``<datastore>/../registry.csv``,
        matching the managed-store layout (``data_root/preprocessed`` + ``data_root/registry.csv``).
    :type registry_path: str or Path, optional
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
        step_log=np.asarray(well.get("step_log", []), dtype=object),
        phase_spec=np.asarray(well.get("phase_spec"), dtype=object),
    )

    # print(f"Transformed data saved to: {out_path}")

    if registry_path is None:
        registry_path = datastore.parent / "registry.csv"
    register({well_no: well}, registry_path)

    return out_path


def register(data: dict[int, dict], registry_path: str | Path, *, timestamp=None) -> None:
    """Upsert one row per processed recording into the registry CSV.

    Rows are keyed by ``(exp_id, chip, well, div)``; an existing row for the same key is replaced.
    Each row records a ``timestamp`` and the well's ``conditions`` (whatever the well's
    ``experimental_condition`` holds, or blank when absent).

    :param data: Mapping of well number to well data dict.
    :type data: dict[int, dict]
    :param registry_path: Path to the registry CSV (typically ``config.registry_path``).
    :type registry_path: str or Path
    :param timestamp: Value for the rows' ``timestamp`` column. Defaults to now, which is what a live
        write wants; :func:`rebuild_registry` passes each ``.npz``'s modification time instead, so a
        retroactively rebuilt row reflects when the file was actually written.
    """
    registry_path = Path(registry_path)
    os.makedirs(registry_path.parent, exist_ok=True)
    stamp = (pd.Timestamp.now() if timestamp is None else pd.Timestamp(timestamp)).isoformat()

    df = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    # Drop the legacy `status` column if an older registry still carries it, so it disappears on the
    # next write.
    df = df.drop(columns=["status"], errors="ignore")

    for well_no, well in data.items():
        condition = well.get("experimental_condition")
        new_row = {
            "exp_id": well["exp_id"],
            "chip": well["chip"],
            "well": well_no,
            "div": well["DIV"],
            "conditions": "" if condition is None else condition,
            "timestamp": stamp,
        }
        if not df.empty:
            # Compare as strings: an older registry on disk may hold a different dtype for
            # `well`/`div`, which would otherwise leak a duplicate row on re-registration.
            mask = (
                (df["exp_id"].astype(str) == str(well["exp_id"]))
                & (df["chip"].astype(str) == str(well["chip"]))
                & (df["well"].astype(str) == str(well_no))
                & (df["div"].astype(str) == str(well["DIV"]))
            )
            df = df[~mask]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

    # Defensive: collapse any pre-existing duplicates (mixed-dtype rows from older writes).
    df = df.drop_duplicates(subset=["exp_id", "chip", "well", "div"], keep="last")
    df.to_csv(registry_path, index=False)


def rebuild_registry(config, *, registry_path: str | Path | None = None) -> int:
    """Rebuild the registry by scanning every preprocessed ``.npz`` in the managed store.

    :func:`save_preprocessed` keeps the registry current as data is written, so this is a recovery
    path: use it when the registry has been lost or has drifted from what is actually on disk.

    Each recording's identity and conditions are read from *inside* its ``.npz`` rather than inferred
    from the directory names, and each row is upserted through :func:`register`, so a rebuilt registry
    is schema-identical to a live-written one. Rows carry the ``.npz``'s modification time as their
    ``timestamp``. Existing rows are updated in place rather than dropped, so recordings whose ``.npz``
    is no longer on disk survive a rebuild.

    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param registry_path: Override the destination; defaults to ``config.registry_path``.
    :type registry_path: str or Path or None
    :returns: The number of recordings registered.
    :rtype: int
    """
    if registry_path is None:
        registry_path = config.registry_path

    def _value(npz, key):
        """Return a saved field as the plain Python object it went in as.

        ``tolist()`` (unlike ``item()``) handles both the 0-d arrays identity fields are stored in and
        the multi-element ones -- experimental conditions are a ``[left, right]`` pair, and must come
        back out as that list so a rebuilt row's ``conditions`` renders exactly as a live-written one.
        """
        return np.asarray(npz[key]).tolist() if key in npz else None

    n = 0
    for npz_path in sorted(config.preprocessed_dir.glob("*/*/*/DIV*.npz")):
        # np.load is lazy, so reading these few keys never decompresses the spike arrays.
        with np.load(npz_path, allow_pickle=True) as npz:
            well_no = _value(npz, "well")
            register(
                {well_no: {
                    "exp_id": _value(npz, "exp_id"),
                    "chip": _value(npz, "chip"),
                    "DIV": _value(npz, "DIV"),
                    "experimental_condition": _value(npz, "exp_condition"),
                }},
                registry_path,
                timestamp=pd.Timestamp.fromtimestamp(npz_path.stat().st_mtime),
            )
        n += 1

    print(f"Registry rebuilt from {n} recordings -> {registry_path}")
    return n


# --- burst outputs ------------------------------------------------------------------------------


def _burst_csv_path(datastore: Path, recording) -> Path:
    """Return ``<datastore>/<exp>/<chip>/well<well>/DIV<div>_<plate_date>_<chip>_<exp>_well<well>_burst_data.csv``."""
    out_dir = datastore / recording.exp_id / recording.chip / f"well{recording.well}"
    return out_dir / (
        f"DIV{recording.DIV}_{recording.plate_date}_{recording.chip}_"
        f"{recording.exp_id}_well{recording.well}_burst_data.csv"
    )


def _params_json(params):
    """Serialize a params dataclass (or dict) to a JSON string, or ``None`` if ``params`` is ``None``.

    Returning ``None`` (rather than ``"{}"``) lets :func:`update_burst_log` *preserve* an existing
    logged value instead of clobbering it when a later step doesn't carry those params.
    """
    if params is None:
        return None
    if is_dataclass(params):
        params = asdict(params)
    return json.dumps(params)


def save_burst_data(burst_set, datastore, recording, *, overwrite: bool = True) -> Path:
    """Write a recording's bursts to a CSV under the managed store.

    Mirrors the preprocessed-npz layout so path resolution stays symmetric:
    ``<datastore>/<exp_id>/<chip>/well<well>/DIV<DIV>_..._burst_data.csv``.

    :param burst_set: The :class:`~mxtreme.bursting.detection.BurstSet` to write.
    :param datastore: Directory under which to write (typically ``config.burst_data_dir``).
    :param recording: The source :class:`~mxtreme.recording.Recording` (supplies the path fields).
    :param overwrite: If ``False``, avoid clobbering an existing file by adding a numeric suffix.
    :returns: The path written to.
    :rtype: Path
    """
    datastore = Path(datastore)
    out_path = _burst_csv_path(datastore, recording)
    os.makedirs(out_path.parent, exist_ok=True)
    if not overwrite:
        out_path = _unique_path(out_path)

    burst_set.to_dataframe().to_csv(out_path, index=False)
    print(f"Burst data saved to: {out_path}")
    return out_path


def load_burst_data(path: str | Path):
    """Load a burst CSV written by :func:`save_burst_data` into a ``BurstSet``.

    :param path: Path to the burst CSV.
    :returns: A :class:`~mxtreme.bursting.detection.BurstSet`.
    """
    from mxtreme.bursting.detection import BurstSet

    return BurstSet.from_csv(path)


def update_burst_log(datastore, recording, burst_set) -> Path:
    """Upsert one summary row per recording into the per-experiment burst log.

    The log lands at ``<datastore>/<exp_id>_burst_log.csv``; rows are keyed by
    ``(exp_id, chip, well, DIV)``. Recorded per recording: burst counts, detection/feature parameters
    (JSON), and per-step completion timestamps (``detection_completed_at`` / ``features_computed_at``).

    Updates are **field-wise and None-preserving**: for an existing row, only columns whose new value
    is not ``None`` are overwritten. So a detection-only write followed later by a features-only write
    (whose ``BurstSet`` carries no detection timestamp or ``detect_params``) keeps the detection
    timestamp and params intact -- and vice versa.

    :param datastore: Directory under which to write (typically ``config.burst_data_dir``).
    :param recording: The source :class:`~mxtreme.recording.Recording`.
    :param burst_set: The detected/featurized :class:`~mxtreme.bursting.detection.BurstSet`.
    :returns: The path to the burst log.
    :rtype: Path
    """
    datastore = Path(datastore)
    os.makedirs(datastore, exist_ok=True)
    log_path = datastore / f"{recording.exp_id}_burst_log.csv"

    df = burst_set.to_dataframe()
    key = {
        "exp_id": recording.exp_id,
        "chip": recording.chip,
        "well": recording.well,
        "DIV": recording.DIV,
    }
    new_row = {
        **key,
        "plate_date": recording.plate_date,
        "n_bursts": len(burst_set),
        "n_network": int((df["kind"] == "network").sum()) if len(df) else 0,
        "n_mini": int((df["kind"] == "mini").sum()) if len(df) else 0,
        "n_ignored": int(burst_set.n_ignored),
        "detect_params": _params_json(burst_set.detect_params),
        "feature_params": _params_json(burst_set.feature_params),
        "detection_completed_at": burst_set.detected_at,
        "features_computed_at": burst_set.features_computed_at,
    }

    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    rows = log.to_dict("records") if not log.empty else []

    existing = next(
        (i for i, r in enumerate(rows) if all(str(r.get(k)) == str(v) for k, v in key.items())),
        None,
    )
    if existing is not None:
        # Field-wise merge on a plain dict: overwrite only columns with a non-None new value.
        merged = dict(rows[existing])
        merged.update({col: val for col, val in new_row.items() if val is not None})
        rows[existing] = merged
    else:
        rows.append(new_row)

    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path
