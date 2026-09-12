"""Reading and writing preprocessed data -- the *L* (load) of the old ETL pipeline.

All writers take an explicit output directory; none fall back to a lab-specific default. In the example
workflow the directories come from a :class:`mxtreme.config.Config` (e.g. ``config.preprocessed_dir``).

Contents:
- :func:`load_preprocessed` -- read a cleaned ``.npz`` back into a dict.
- :func:`save_preprocessed` -- write one well's cleaned data to an ``.npz`` (and register it).
- :func:`register` -- record processed recordings in the registry CSV.
- :func:`register_scan` -- record an activity or network scan in that same registry.
- :func:`rebuild_registry` -- rebuild that registry by walking the store (recovery path).
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

    from mxtreme import transactions

    transactions.record(
        transactions.transactions_path_for(registry_path),
        "preprocessed.saved",
        batch_id=well.get("batch_id") or None,
        plate_date=plate_date,
        exp_id=exp_id,
        chip=chip,
        well=well_no,
        div=div,
        data={"path": str(out_path), "source": str(well.get("path_to_h5") or "")},
    )

    return out_path


#: Columns identifying one registry row. ``kind`` is part of the key because a scan and the
#: recordings preprocessed from it share an identity; ``batch_id`` is part of it because a chip is
#: re-plated across batches, so ``(chip, well, div)`` alone recurs batch after batch.
REGISTRY_KEY = ["exp_id", "batch_id", "chip", "well", "div", "kind"]

#: What a registry row describes. ``"preprocessed"`` is one well of a cleaned recording;
#: ``"activity_scan"`` and ``"network_scan"`` are one well of the corresponding scan's raw ``.h5``;
#: ``"experiment"`` is one well of an exogenous recording ingested by
#: :func:`mxtreme.store.ingest_recording` (its ``exp_id`` column carries the experiment's name).
REGISTRY_KINDS = ("preprocessed", "activity_scan", "network_scan", "experiment")


def _read_registry(registry_path: Path) -> pd.DataFrame:
    """Read the registry CSV into the current schema, or an empty frame if it does not exist yet.

    Migrations happen on read, so an older registry converges on the current schema at its next
    write rather than needing a separate upgrade step:

    - the legacy ``status`` column is dropped;
    - a missing ``kind`` column is backfilled with ``"preprocessed"``. Registries written before
      scans were indexed hold cleaned recordings and nothing else, so that is what those rows are;
    - missing ``batch_id`` / ``plate_date`` columns (from before the recordings tree) are added,
      blank -- those rows predate batch identity;
    - blank string key fields come back from CSV as NaN and are normalized to ``""`` so the string
      comparisons in :func:`register` treat "no value" consistently.

    :param registry_path: Path to the registry CSV.
    :returns: The registry, in the current schema.
    :rtype: pandas.DataFrame
    """
    if not registry_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(registry_path).drop(columns=["status"], errors="ignore")
    if df.empty:
        return df

    if "kind" not in df.columns:
        position = df.columns.get_loc("div") + 1 if "div" in df.columns else len(df.columns)
        df.insert(position, "kind", "preprocessed")
    after_exp = df.columns.get_loc("exp_id") + 1 if "exp_id" in df.columns else 0
    for i, column in enumerate(("batch_id", "plate_date")):
        if column not in df.columns:
            df.insert(after_exp + i, column, "")
    for column in ("exp_id", "batch_id"):
        df[column] = df[column].fillna("")
    return df


def register(
    data: dict[int, dict],
    registry_path: str | Path,
    *,
    kind: str = "preprocessed",
    timestamp=None,
) -> None:
    """Upsert one row per processed recording into the registry CSV.

    Rows are keyed by :data:`REGISTRY_KEY`; an existing row for the same key is replaced. Each row
    records a ``timestamp`` and the well's ``conditions`` (whatever the well's
    ``experimental_condition`` holds, or blank when absent). Batch identity (``batch_id``,
    ``plate_date``) is recorded when the well dict carries it; rows from before the recordings tree
    simply leave it blank.

    :param data: Mapping of well number to well data dict.
    :type data: dict[int, dict]
    :param registry_path: Path to the registry CSV (typically ``config.registry_path``).
    :type registry_path: str or Path
    :param kind: What these rows describe -- see :data:`REGISTRY_KINDS`. Because ``kind`` is part of
        the key, an activity scan and the recordings later preprocessed from the same well and DIV
        coexist as separate rows instead of overwriting one another.
    :type kind: str
    :param timestamp: Value for the rows' ``timestamp`` column. Defaults to now, which is what a live
        write wants; :func:`rebuild_registry` passes each file's modification time instead, so a
        retroactively rebuilt row reflects when the file was actually written.
    """
    registry_path = Path(registry_path)
    os.makedirs(registry_path.parent, exist_ok=True)
    stamp = (pd.Timestamp.now() if timestamp is None else pd.Timestamp(timestamp)).isoformat()

    df = _read_registry(registry_path)

    for well_no, well in data.items():
        condition = well.get("experimental_condition")
        new_row = {
            "exp_id": well.get("exp_id", ""),
            "batch_id": well.get("batch_id", ""),
            "plate_date": well.get("plate_date", ""),
            "chip": well["chip"],
            "well": well_no,
            "div": well["DIV"],
            "kind": kind,
            "conditions": "" if condition is None else condition,
            "timestamp": stamp,
        }
        if not df.empty:
            # Compare as strings: an older registry on disk may hold a different dtype for
            # `well`/`div`, which would otherwise leak a duplicate row on re-registration.
            mask = (
                (df["exp_id"].astype(str) == str(new_row["exp_id"]))
                & (df["batch_id"].astype(str) == str(new_row["batch_id"]))
                & (df["chip"].astype(str) == str(well["chip"]))
                & (df["well"].astype(str) == str(well_no))
                & (df["div"].astype(str) == str(well["DIV"]))
                & (df["kind"].astype(str) == str(kind))
            )
            df = df[~mask]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

    # Defensive: collapse any pre-existing duplicates (mixed-dtype rows from older writes).
    df = df.drop_duplicates(subset=REGISTRY_KEY, keep="last")
    df.to_csv(registry_path, index=False)


# --- scans ----------------------------------------------------------------------------------------


def scan_kind(h5_path: str | Path) -> str:
    """What a file in the recordings tree is, from its name.

    A thin wrapper over :func:`mxtreme.store.recording_kind`, kept here so registry callers have
    the whole vocabulary (:data:`REGISTRY_KINDS`) in one module.

    :param h5_path: Path to the ``.h5``.
    :returns: ``"activity_scan"``, ``"network_scan"``, or ``"experiment"``.
    :rtype: str
    """
    from mxtreme.store import recording_kind

    return recording_kind(h5_path)


def register_scan(
    params,
    registry_path: str | Path,
    *,
    kind: str = "activity_scan",
    timestamp=None,
    well_files: dict[int, str | Path] | None = None,
) -> None:
    """Upsert one scan row per scanned well into the registry CSV.

    A scan records every well simultaneously into a single ``.h5``, but it is registered per well so
    the rows key the same way every other row does -- and so "what do I have for this culture" is one
    query over one table, whether the answer is a scan or a cleaned recording.

    Only the identity fields land here; they are also written into the ``.h5`` itself by
    :func:`mxtreme.scans.mx_setup.write_metadata`, which is what :func:`rebuild_registry` reads back.

    :param params: The scan's :class:`~mxtreme.scans.activity_scan.ActivityScanParams`. Only
        ``chip``, ``div``, ``wells``, ``conditions`` and the batch identity (``batch_id``,
        ``plate_date``) are read, so any object carrying those works.
    :param registry_path: Path to the registry CSV (typically ``config.registry_path``).
    :type registry_path: str or Path
    :param kind: Which sort of scan these rows describe -- ``"activity_scan"`` or
        ``"network_scan"``, see :data:`REGISTRY_KINDS`.
    :type kind: str
    :param timestamp: Value for the rows' ``timestamp`` column; defaults to now.
    :param well_files: ``{well: path}`` of the file each well landed in, for the transaction log's
        ``<kind>.registered`` record of each well. Optional; the record is written either way.
    """
    conditions = list(params.conditions)
    register(
        {
            well: {
                # A scan has no experiment name -- its identity is the batch. The blank exp_id is
                # what separates scan rows from `experiment` rows in the same tree.
                "exp_id": getattr(params, "exp_id", ""),
                "batch_id": getattr(params, "batch_id", ""),
                "plate_date": getattr(params, "plate_date", ""),
                "chip": params.chip,
                "DIV": params.div,
                # `conditions` is empty or one label per well -- validated by ActivityScanParams.
                "experimental_condition": conditions[i] if i < len(conditions) else None,
            }
            for i, well in enumerate(params.wells)
        },
        registry_path,
        kind=kind,
        timestamp=timestamp,
    )

    from mxtreme import transactions

    log_path = transactions.transactions_path_for(registry_path)
    for well in params.wells:
        path = None if well_files is None else well_files.get(well)
        transactions.record(
            log_path,
            f"{kind}.registered",
            batch_id=getattr(params, "batch_id", "") or None,
            plate_date=getattr(params, "plate_date", None) or None,
            exp_id=getattr(params, "exp_id", ""),
            chip=params.chip,
            well=well,
            div=params.div,
            data={"path": "" if path is None else str(path)},
        )


def rebuild_registry(config, *, registry_path: str | Path | None = None) -> int:
    """Rebuild the registry by walking the managed store: every preprocessed ``.npz`` and every raw
    ``.h5`` in the recordings tree.

    :func:`save_preprocessed`, :func:`register_scan` and :func:`mxtreme.store.ingest_recording` keep
    the registry current as data is written, so this is a recovery path: use it when the registry has
    been lost or has drifted from what is actually on disk.

    Each preprocessed recording's identity and conditions are read from *inside* its ``.npz`` rather
    than inferred from the directory names, and each row is upserted through :func:`register`, so a
    rebuilt registry is schema-identical to a live-written one. Rows carry the source file's
    modification time as their ``timestamp``. Existing rows are updated in place rather than dropped,
    so recordings whose file is no longer on disk survive a rebuild.

    A file in the recordings tree is identified from its *path*: the tree's directory names are
    fixed-format and carry the full identity (batch, plating date, chip, well, DIV) -- see
    :func:`mxtreme.store.parse_recording_path`. Its ``/assay/metadata`` blob, when readable, supplies
    the well's condition label; a file whose path does not follow the layout is counted and reported
    rather than guessed at.

    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param registry_path: Override the destination; defaults to ``config.registry_path``.
    :type registry_path: str or Path or None
    :returns: The number of recordings and raw files registered.
    :rtype: int
    """
    from mxtreme.store import parse_recording_path

    if registry_path is None:
        registry_path = config.registry_path

    def _value(npz, key):
        """Return a saved field as the plain Python object it went in as.

        ``tolist()`` (unlike ``item()``) handles both the 0-d arrays identity fields are stored in and
        the multi-element ones -- experimental conditions are a ``[left, right]`` pair, and must come
        back out as that list so a rebuilt row's ``conditions`` renders exactly as a live-written one.
        """
        return np.asarray(npz[key]).tolist() if key in npz else None

    n_recordings = 0
    for npz_path in sorted(config.preprocessed_dir.glob("*/*/*/DIV*.npz")):
        # np.load is lazy, so reading these few keys never decompresses the spike arrays.
        with np.load(npz_path, allow_pickle=True) as npz:
            well_no = _value(npz, "well")
            register(
                {well_no: {
                    "exp_id": _value(npz, "exp_id"),
                    "plate_date": _value(npz, "plate_date") or "",
                    "chip": _value(npz, "chip"),
                    "DIV": _value(npz, "DIV"),
                    "experimental_condition": _value(npz, "exp_condition"),
                }},
                registry_path,
                timestamp=pd.Timestamp.fromtimestamp(npz_path.stat().st_mtime),
            )
        n_recordings += 1

    n_raw, n_unparseable = 0, 0
    for h5_path in sorted(config.recordings_dir.glob("*/*/*/*/*.h5")):
        location = parse_recording_path(h5_path, config.recordings_dir)
        if location is None:
            n_unparseable += 1
            continue

        register(
            {location.well: {
                "exp_id": location.exp_id,
                "batch_id": location.batch.id,
                "plate_date": location.plate_date,
                "chip": location.chip,
                "DIV": location.div,
                "experimental_condition": _embedded_condition(h5_path, location.well),
            }},
            registry_path,
            kind=location.kind,
            timestamp=pd.Timestamp.fromtimestamp(h5_path.stat().st_mtime),
        )
        n_raw += 1

    print(
        f"Registry rebuilt from {n_recordings} recording(s) and {n_raw} raw file(s) "
        f"-> {registry_path}"
    )
    if n_unparseable:
        print(
            f"  {n_unparseable} file(s) skipped: their paths do not follow the recordings-tree "
            "layout, so their identity could not be recovered."
        )

    from mxtreme import transactions

    # One record for the rebuild, not one per row: the rows describe files that were journaled
    # when they were written; the event here is that the index was reconstructed.
    transactions.record(
        transactions.transactions_path_for(registry_path),
        "registry.rebuilt",
        data={
            "registry": str(registry_path),
            "n_recordings": n_recordings,
            "n_raw": n_raw,
            "n_unparseable": n_unparseable,
        },
    )
    return n_recordings + n_raw


def _embedded_condition(h5_path: Path, well: int):
    """One well's condition label from a raw file's ``/assay/metadata`` blob, or ``None``.

    The blob is enrichment, not identity -- the path already names the recording -- so a missing or
    unreadable blob simply yields no condition.
    """
    embedded = _embedded_metadata(h5_path)
    if embedded is None:
        return None
    well_ids = list(embedded.get("Well IDs") or [])
    conditions = list(embedded.get("Conditions") or [])
    if well in well_ids and well_ids.index(well) < len(conditions):
        return conditions[well_ids.index(well)]
    return None


def _embedded_metadata(h5_path: Path) -> dict | None:
    """Read a scan's ``/assay/metadata`` blob, or ``None`` if it has none that can be parsed.

    Same blob, and the same repr-with-quotes-swapped decoding, as :func:`mxtreme.extract.extract`.

    :param h5_path: Path to the scan ``.h5``.
    :returns: The decoded metadata dict, or ``None``.
    """
    import h5py  # local: io is imported off the rig, where the h5 half is often unused

    try:
        with h5py.File(str(h5_path), "r") as f:
            raw = f["/assay/metadata"][:][0]
        return json.loads(raw.decode("utf-8").strip().replace("'", '"'))
    except (OSError, KeyError, ValueError, IndexError):
        return None


def repair_spike_order(config, *, dry_run: bool = False) -> list[Path]:
    """Re-save any stored ``.npz`` whose ``spike_data`` is not ascending by ``frameno``.

    A one-time migration for recordings preprocessed before :func:`mxtreme.extract.extract` began
    sorting the raw spike table (see :func:`mxtreme.utils.sort_spike_data` for why the order matters).
    Files already in order are left untouched, so re-running this is cheap in writes -- though never in
    reads, since each spike array must be decompressed to check it. Expect minutes on a large store.

    Only ``spike_data`` is reordered; every other saved field is written back verbatim, and the registry
    is deliberately left alone (this changes the order of rows, not the recording's provenance). Each
    file is rewritten via a temporary file in the same directory and then moved into place, so an
    interrupted run cannot leave a truncated recording behind.

    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param dry_run: If ``True``, report the files that need repair without writing anything.
    :type dry_run: bool
    :returns: Paths of the recordings repaired (or, under ``dry_run``, that would be).
    :rtype: list[Path]
    """
    from mxtreme.utils import sort_spike_data  # local: utils pulls matplotlib at import

    repaired: list[Path] = []
    n_scanned = 0
    for npz_path in sorted(config.preprocessed_dir.glob("*/*/*/DIV*.npz")):
        n_scanned += 1
        contents = dict(np.load(npz_path, allow_pickle=True))
        spike_data = contents["spike_data"]
        ordered = sort_spike_data(spike_data)
        if ordered is spike_data:  # already ascending -- sort_spike_data returns the input unchanged
            continue

        repaired.append(npz_path)
        if dry_run:
            continue

        contents["spike_data"] = ordered
        tmp_path = npz_path.with_name(npz_path.name + ".tmp")
        with open(tmp_path, "wb") as fh:  # a file object, so savez does not re-append ".npz"
            np.savez_compressed(fh, **contents)
        os.replace(tmp_path, npz_path)

        from mxtreme import transactions

        transactions.record(
            config,
            "preprocessed.repaired",
            exp_id=_scalar(contents.get("exp_id")),
            plate_date=_scalar(contents.get("plate_date")),
            chip=_scalar(contents.get("chip")),
            well=_scalar(contents.get("well")),
            div=_scalar(contents.get("DIV")),
            data={"path": str(npz_path)},
        )

    verb = "would repair" if dry_run else "repaired"
    print(f"Spike order: scanned {n_scanned} recording(s), {verb} {len(repaired)}")
    return repaired


# --- burst outputs ------------------------------------------------------------------------------


def _scalar(value):
    """A scalar out of an ``.npz`` field, which comes back as a 0-d or 1-element array."""
    if value is None:
        return None
    try:
        return np.asarray(value).reshape(-1)[0].item()
    except (IndexError, ValueError, AttributeError):
        return value


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

    from mxtreme import transactions

    transactions.record(
        transactions.transactions_path_for(datastore),
        "bursts.saved",
        exp_id=recording.exp_id,
        plate_date=getattr(recording, "plate_date", None),
        chip=recording.chip,
        well=recording.well,
        div=recording.DIV,
        data={"path": str(out_path), "n_bursts": len(burst_set)},
    )
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
