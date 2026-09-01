"""Record a network scan on the rig and write it to a single ``.h5`` in the managed store.

A *network scan* is the recording an activity scan exists to set up: one fixed set of electrodes per
well, recorded continuously at full rate. The electrodes come straight from
:func:`mxtreme.scans.electrode_selection.select_electrodes`, which returns exactly the
``{well: [electrode, ...]}`` mapping :class:`NetworkScanParams` takes::

    rec_elecs = electrode_selection.select_electrodes(activity_scan_h5, save_path)
    params = network_scan.NetworkScanParams(recording_electrodes=rec_elecs, chip="M07460", ...)

Structurally this mirrors :mod:`mxtreme.scans.activity_scan`, and for the same reasons:

- **Planning** -- :class:`NetworkScanParams` and :func:`describe` are pure Python and import nothing
  rig-specific, so a front end can build and validate a scan on any machine.
- **Execution** -- :func:`run_network_scan` drives the chip and needs MaxWell's ``maxlab`` API,
  imported on the first call rather than at module import.

The difference is that a network scan is one recording rather than a series of them, so there is no
electrode plan to make: what to record is an input, not something this module decides. The one thing
it does decide is the *rest* of the routing -- electrode selection's thresholds usually leave fewer
than the :data:`~mxtreme.scans.activity_scan.MAX_ROUTED_ELECTRODES` the chip can route, and unused
routing capacity is free data, so each well is topped up to the limit with random electrodes (see
:func:`pad_electrodes`).

Like an activity scan it lands in the package-managed store -- the ``.h5`` in the recordings tree at
``.../well_<w>/DIV_<div>/`` (a multi-well recording is split into per-well files on the way in), one
``network_scan`` row per well in ``registry.csv`` -- and setting :attr:`NetworkScanParams.save_path`
writes it elsewhere, unsplit and unregistered.

Typical use::

    from mxtreme.config import Config
    from mxtreme.scans import electrode_selection, network_scan

    config = Config.from_toml("mxtreme.toml")
    rec_elecs = electrode_selection.select_electrodes(str(scan.h5_path), str(scan.h5_path.parent))
    params = network_scan.NetworkScanParams(
        recording_electrodes=rec_elecs,
        batch="fall2026_batch1_DRG_M1", chip="M07460", plate_date=260810, div=25,
        rec_length_sec=300,
    )
    result = network_scan.run_network_scan(params, config)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from mxtreme import io, store
from mxtreme.scans.activity_scan import (
    MAX_ROUTED_ELECTRODES,
    MAX_WELLS,
    N_TOTAL_ELECTRODES,
    ProgressFn,
    StopFn,
    _connected_device,
    _require_maxlab,
    _saved_file,
    _split_into_store,
)

#: Amplifier channels per well. Every one is saved, via a single recording group -- a channel with no
#: electrode routed to it costs a little disk and nothing else, whereas a missing group silently
#: costs data.
N_CHANNELS = 1024

#: How often, in seconds, a long recording reports its progress and asks ``should_stop``.
_POLL_SEC = 0.5

#: How often, in seconds, a progress line is emitted while recording.
_REPORT_SEC = 30


@dataclass
class NetworkScanParams:
    """Everything needed to plan and run one network scan.

    The wells are not given separately: they are the keys of ``recording_electrodes``, so the scan
    records exactly the wells electrode selection produced a set for.

    :param recording_electrodes: ``{well: [electrode, ...]}``, as returned by
        :func:`mxtreme.scans.electrode_selection.select_electrodes`. Numpy arrays are accepted and
        normalized to plain lists of ``int``, so the selector's output can be passed straight in.
        Unless ``pad_to_max`` is off, each well is topped up to
        :data:`~mxtreme.scans.activity_scan.MAX_ROUTED_ELECTRODES` here, so after construction this
        holds exactly what the scan will record.
    :param batch: The plating batch the culture belongs to -- a :class:`mxtreme.store.Batch` or its
        id string (e.g. ``"fall2026_batch1_DRG_M1"``). Required for any scan saved into the managed
        store; validated by :meth:`validate`.
    :param chip: Chip serial, e.g. ``"M07460"``. Written to the file's metadata and used in its name.
    :param plate_date: Plating date as ``YYMMDD``; validated by :func:`mxtreme.scans.mx_setup.write_metadata`.
    :param div: Days *in vitro* at the time of the scan.
    :param conditions: Optional per-well condition labels, in :attr:`wells` order; must be empty or
        the same length as ``wells``.
    :param description: Free-text description written to the file.
    :param rec_length_sec: Length of the recording, in seconds.
    :param pad_to_max: Fill each well's selection up to the routing limit with random electrodes.
        Set it to ``False`` to record only the electrodes selection chose.
    :param seed: Seed for that random fill, for a reproducible electrode set.
    :param save_path: Directory the ``.h5`` is written into. ``None`` -- the default -- means the
        managed store, resolved against the :class:`~mxtreme.config.Config` passed to
        :func:`run_network_scan`. Set it to write somewhere outside the store instead; the scan is
        then not registered.
    """

    recording_electrodes: dict[int, list[int]]

    # Metadata
    batch: store.Batch | str | None = None
    chip: str = "M07460"
    plate_date: int = 260810
    div: int = 1
    conditions: list[str] = field(default_factory=list)
    description: str = "Network scan of the recording electrodes chosen from an activity scan"

    # Scan
    rec_length_sec: int = 300
    pad_to_max: bool = True
    seed: int | None = None

    # Output -- None means "the managed store"; see `resolved`.
    save_path: str | None = None

    def __post_init__(self) -> None:
        """Normalize the electrode mapping to plain ``{int: [int, ...]}``, and fill it to the limit.

        ``select_electrodes`` hands back numpy arrays of numpy integers. Those route perfectly well,
        but they serialize into the metadata blob as ``np.int64(4213)`` and compare unequally to the
        ints everything else uses, so they are converted once here rather than guarded against
        everywhere downstream.

        The padding happens here too, rather than at record time, so that everything downstream --
        :func:`describe`, :meth:`validate`, a front end previewing the scan -- sees the electrodes
        that will actually be recorded. A well selection left empty is *not* padded: recording 1020
        random electrodes in a well nothing was chosen for would be data from nowhere, and would
        quietly satisfy the empty-well check in :meth:`validate`.
        """
        self.recording_electrodes = {
            int(well): [int(e) for e in electrodes]
            for well, electrodes in self.recording_electrodes.items()
        }

        if self.pad_to_max:
            rng = np.random.default_rng(self.seed)
            self.recording_electrodes = {
                well: pad_electrodes(electrodes, rng=rng) if electrodes else electrodes
                for well, electrodes in self.recording_electrodes.items()
            }

    @property
    def wells(self) -> list[int]:
        """The wells this scan records, ascending: the keys of ``recording_electrodes``."""
        return sorted(self.recording_electrodes)

    @property
    def batch_id(self) -> str:
        """The batch id string, however ``batch`` was given; ``""`` when it is unset."""
        return store.Batch.parse(self.batch).id if self.batch is not None else ""

    @property
    def file_name(self) -> str:
        """Base name of the ``.h5``, without MaxWell's ``.raw.h5`` suffix.

        The store's canonical stem (see :func:`mxtreme.store.recording_stem`) plus the
        ``_network_scan`` tail, so :func:`mxtreme.io.scan_kind` can tell it from an activity scan of
        the same culture -- the two sit in the same directory and carry the same metadata blob. A
        multi-well scan's well token is the joined list (``0-3``) until
        :func:`mxtreme.store.split_by_well` breaks it into per-well names.

        :raises ValueError: If ``batch`` is unset -- the name is built from it.
        """
        if self.batch is None:
            raise ValueError(
                "batch is unset, so this scan cannot be named. Every scan needs the plating batch "
                "it records, e.g. batch='fall2026_batch1_DRG_M1'."
            )
        well_token = "-".join(str(w) for w in self.wells)
        return (
            store.recording_stem(self.batch, self.plate_date, self.chip, well_token, self.div)
            + "_network_scan"
        )

    @property
    def h5_path(self) -> Path:
        """Where MaxWell will write the scan: ``<save_path>/<file_name>.raw.h5``.

        :raises ValueError: If ``save_path`` is unset, since the destination then depends on a
            :class:`~mxtreme.config.Config` this object does not have. Call :meth:`resolved` first.
        """
        if self.save_path is None:
            raise ValueError(
                "save_path is unset, so this scan's destination is not known yet: it comes from the "
                "managed store. Call params.resolved(config) for a copy that knows where it writes."
            )
        return Path(self.save_path) / f"{self.file_name}.raw.h5"

    def resolved(self, config=None) -> NetworkScanParams:
        """Return a copy whose ``save_path`` is filled in, so every path property is answerable.

        An explicit ``save_path`` wins and the object is returned unchanged; otherwise the scan
        lands beside the activity scan it came from, in the recordings tree: straight into
        ``.../well_<w>/DIV_<div>/`` for one well, or into the chip directory (split per well
        afterwards) for several.

        :param config: The :class:`~mxtreme.config.Config` describing the managed store. Only needed
            when ``save_path`` is unset.
        :raises ValueError: If there is neither a ``save_path`` nor a ``config`` to derive one from,
            or if the scan is headed for the store without a ``batch`` to file it under.
        :returns: This object, or a copy with ``save_path`` set.
        :rtype: NetworkScanParams
        """
        if self.save_path is not None:
            return self
        if config is None:
            raise ValueError(
                "A network scan needs somewhere to write, and this one has neither. Pass "
                "config=Config.from_toml('mxtreme.toml') to save it into the managed store, or set "
                "NetworkScanParams.save_path to write outside the store."
            )
        if self.batch is None:
            raise ValueError(
                "A scan saved into the managed store needs the plating batch it records: the "
                "recordings tree is keyed by batch. Set batch (e.g. 'fall2026_batch1_DRG_M1')."
            )
        if len(self.wells) == 1:
            directory = store.recording_dir(
                config, self.batch, self.plate_date, self.chip, self.wells[0], self.div
            )
        else:
            directory = store.chip_dir(config, self.batch, self.plate_date, self.chip)
        return replace(self, save_path=str(directory))

    @property
    def estimated_minutes(self) -> float:
        """Recording time in minutes, excluding routing and offset-compensation overhead."""
        return self.rec_length_sec / 60

    def as_metadata(self) -> dict:
        """Build the metadata dict written into the file.

        Shaped for :func:`mxtreme.scans.mx_setup.write_metadata`, and therefore for
        :func:`mxtreme.extract.extract`, which reads the same blob back out -- so a network scan
        extracts and preprocesses like any other recording. A scan has no experiment name, so
        ``Exp ID`` carries the batch id and the preprocessed tree files the data under its batch.
        """
        return {
            "Exp ID": self.batch_id,
            "Batch ID": self.batch_id,
            "Chip ID": self.chip,
            "Plate date": self.plate_date,
            "DIV": self.div,
            "Well IDs": list(self.wells),
            "Conditions": list(self.conditions),
        }

    def validate(self) -> None:
        """Check the parameters are runnable, so failures surface before the chip is touched.

        :raises ValueError: On any parameter that would fail mid-scan -- a missing or malformed
            batch id, no wells, an out-of-range well or electrode, an empty or duplicated electrode
            set, a well asking for more than
            :data:`~mxtreme.scans.activity_scan.MAX_ROUTED_ELECTRODES` electrodes (which the chip
            cannot route, so some would be silently dropped), a mismatched condition list, or a
            non-positive recording length.
        """
        _ = self.file_name  # raises on a missing batch, and Batch.parse raises on a malformed one
        if not self.recording_electrodes:
            raise ValueError(
                "No recording electrodes. Pass the {well: [electrode, ...]} mapping returned by "
                "electrode_selection.select_electrodes()."
            )
        if not all(0 <= w < MAX_WELLS for w in self.wells):
            raise ValueError(f"Wells must be in 0..{MAX_WELLS - 1}, got {self.wells}.")
        if self.conditions and len(self.conditions) != len(self.wells):
            raise ValueError(
                f"Conditions must be empty or one per well: {len(self.conditions)} conditions "
                f"for {len(self.wells)} wells."
            )
        if self.rec_length_sec <= 0:
            raise ValueError(f"rec_length_sec must be positive, got {self.rec_length_sec}.")

        for well in self.wells:
            electrodes = self.recording_electrodes[well]
            if not electrodes:
                raise ValueError(f"Well {well} has no recording electrodes.")
            if len(electrodes) > MAX_ROUTED_ELECTRODES:
                raise ValueError(
                    f"Well {well} asks for {len(electrodes)} electrodes, more than the "
                    f"{MAX_ROUTED_ELECTRODES} the chip can route at once."
                )
            if len(set(electrodes)) != len(electrodes):
                raise ValueError(f"Well {well} has duplicate electrodes.")
            if not all(0 <= e < N_TOTAL_ELECTRODES for e in electrodes):
                raise ValueError(
                    f"Well {well} has electrodes outside 0..{N_TOTAL_ELECTRODES - 1}."
                )


@dataclass
class NetworkScanResult:
    """What a completed (or early-stopped) network scan produced.

    :param h5_path: The file that was written -- after a multi-well store scan is split per well,
        the first well's file. It exists even if the scan stopped early; the file is always
        finalized.
    :param params: The parameters the scan ran with.
    :param recorded_sec: How long the recording actually ran, which is less than
        ``params.rec_length_sec`` only when it was stopped early.
    :param duration_sec: Wall-clock time from the start of recording, including teardown.
    :param well_files: The file holding each well's data. For a single-well scan every well maps to
        ``h5_path``; for a multi-well store scan these are the split per-well files.
    """

    h5_path: Path
    params: NetworkScanParams
    recorded_sec: float
    duration_sec: float
    well_files: dict[int, Path] | None = None

    @property
    def complete(self) -> bool:
        """Whether the recording ran for its full length."""
        return self.recorded_sec >= self.params.rec_length_sec


def pad_electrodes(
    electrodes, n: int = MAX_ROUTED_ELECTRODES, rng=None
) -> list[int]:
    """Top one well's electrode list up to ``n`` with electrodes drawn at random from the rest.

    Electrode selection thresholds a scan down to the electrodes that were genuinely active, which
    is usually well under the chip's routing limit. Routing capacity left unused records nothing, so
    the remainder is filled at random: the chosen electrodes are still recorded exactly as selected,
    and the spare channels are a free look at the rest of the array.

    :param electrodes: The electrodes selection chose for this well.
    :param n: Electrodes to fill up to. A list already at or over ``n`` is returned unchanged.
    :param rng: A :class:`numpy.random.Generator` to draw with; ``None`` draws fresh randomness.
    :returns: The chosen electrodes, in their original order, followed by the random fill.
    :rtype: list[int]
    """
    chosen = [int(e) for e in electrodes]
    shortfall = n - len(chosen)
    if shortfall <= 0:
        return chosen

    rng = np.random.default_rng() if rng is None else rng
    remaining = np.setdiff1d(np.arange(N_TOTAL_ELECTRODES), chosen)
    fill = rng.choice(remaining, size=shortfall, replace=False)

    return chosen + sorted(int(e) for e in fill)


def describe(params: NetworkScanParams) -> str:
    """Render a human-readable summary of a scan, for confirming it before it runs.

    :param params: The scan to describe. An unresolved ``save_path`` (or a still-unset ``batch``)
        is fine; the destination is then reported rather than raised over.
    :returns: A multi-line summary.
    """
    if params.batch is None:
        destination = "the managed store (batch not set yet -- required before running)"
    else:
        destination = f"the managed store, as {params.file_name}.raw.h5"
    per_well = ", ".join(
        f"well {well}: {len(params.recording_electrodes[well])}" for well in params.wells
    )
    lines = [
        f"Chip           : {params.chip}",
        f"Plate date     : {params.plate_date} | DIV: {params.div}",
        f"Batch          : {params.batch_id or '(unset)'}",
        f"Wells          : {params.wells}",
        f"Electrodes     : {per_well or 'none'}",
        (
            "                 (selection topped up to "
            f"{MAX_ROUTED_ELECTRODES} per well with random electrodes)"
            if params.pad_to_max
            else "                 (as chosen by electrode selection; no random fill)"
        ),
        f"Recording time : {params.estimated_minutes:.1f} min (excluding routing overhead)",
        # Describing a scan is a planning step, and planning happens before a Config (or even the
        # batch) is necessarily in hand -- so an unresolved destination is reported, not raised over.
        f"Saving to      : {params.h5_path if params.save_path is not None and params.batch is not None else destination}",
        "                 (MaxLab appends _1, _2 ... if that name is already taken)",
    ]
    return "\n".join(lines)


def run_network_scan(
    params: NetworkScanParams,
    config=None,
    *,
    on_progress: ProgressFn = print,
    should_stop: StopFn | None = None,
) -> NetworkScanResult:
    """Record a network scan on the rig and save every well into one ``.h5``.

    Routes each well's chosen electrodes, compensates amplifier offsets, then records every well
    simultaneously for ``params.rec_length_sec``.

    The scan goes into the managed store described by ``config``: the ``.h5`` into the recordings
    tree (``.../well_<w>/DIV_<div>/`` for one well; a multi-well scan is recorded whole and then
    split into per-well files there), and one ``network_scan`` row per well in
    ``config.registry_path``. Registration happens after the recording finishes, including when it
    stopped early, so a short scan is still findable. An explicit ``params.save_path`` overrides the
    destination, skips the split, and skips registration -- a row for a file outside the store could
    never be resolved back to it.

    The file is finalized and the arrays closed even if the recording fails partway through, so what
    was recorded up to that point is readable. A failure still propagates; registration is skipped on
    that path.

    :param params: The scan to run. Validated before the chip is touched.
    :param config: The :class:`~mxtreme.config.Config` naming the managed store to save into and
        register with. Optional only when ``params.save_path`` says where to write instead.
    :param on_progress: Called with each progress line. Defaults to :func:`print`; pass a callback
        to route it into a UI, or ``lambda _: None`` to silence it.
    :param should_stop: Polled during the recording; return ``True`` to end it early. The file is
        finalized as usual, so the result is a short recording rather than a failed one.
    :raises ModuleNotFoundError: If ``maxlab`` is not installed (i.e. this is not the rig).
    :raises TypeError: If ``config`` is not a :class:`~mxtreme.config.Config`.
    :raises ValueError: If the parameters are not runnable (see :meth:`NetworkScanParams.validate`),
        or if there is neither a ``config`` nor a ``save_path`` to write to.
    :raises RuntimeError: If no device is connected.
    :returns: A :class:`NetworkScanResult` naming the file and how long it recorded for.
    """
    if config is not None and not hasattr(config, "recordings_dir"):
        raise TypeError(
            f"run_network_scan() expects a Config as its second argument, got "
            f"{type(config).__name__}."
        )

    mx = _require_maxlab()
    from mxtreme.scans import mx_setup  # rig-only import; deferred for the same reason as maxlab

    # Registering only makes sense for a scan that actually lands in the store: a row for a file
    # written anywhere else could never be resolved back to it. Decided before resolution, which is
    # what erases the distinction between a derived save_path and a given one.
    registry_path = config.registry_path if config is not None and params.save_path is None else None

    # Resolve and validate before anything is created, so a scan that cannot run fails now rather
    # than after the chip has been reconfigured.
    params = params.resolved(config)
    params.validate()

    on_progress("=== Network scan ===")
    on_progress(describe(params))

    on_progress(f"Device: {_connected_device(mx)}")

    Path(params.save_path).mkdir(parents=True, exist_ok=True)

    on_progress("Initializing chip...")
    mx.initialize()
    time.sleep(mx.Timing.waitInit)

    # Route first, then open the file: a routing failure then leaves no empty .h5 behind. init_well
    # selects, routes and downloads one well; with no stimulation electrodes it neither connects nor
    # powers up any stimulation unit.
    arrays = {}
    for well in params.wells:
        electrodes = params.recording_electrodes[well]
        on_progress(f"  Well {well}: routing {len(electrodes)} electrodes...")
        array, _ = mx_setup.init_well(well, list(electrodes), stim_elecs=[])
        arrays[well] = array
        on_progress(f"  Well {well}: routed and downloaded")

    on_progress(f"Activating wells {params.wells}")
    mx.activate(params.wells)  # init_well activates one well at a time; record them together

    on_progress("Running offset compensation...")
    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)

    s = mx.Saving()
    s.open_directory(params.save_path)
    s.start_file(params.file_name)  # creates the (empty) h5
    s.group_delete_all()  # clear stale group definitions
    mx_setup.write_metadata(s, params.as_metadata())
    mx_setup.write_exp_description(s, params.description)
    for well in params.wells:
        # Without a group there is nothing for the well to save into.
        s.group_define(well, f"all_channels_{well}", list(range(N_CHANNELS)))
    on_progress(f"Opened file: {params.h5_path}")

    recorded_sec = 0.0
    recording = False
    scan_start = time.time()

    try:
        on_progress(f"Recording {params.rec_length_sec} s...")
        s.start_recording(params.wells)
        recording = True
        recorded_sec = _record(params.rec_length_sec, on_progress, should_stop)
    finally:
        on_progress("\nFinalizing file and closing arrays...")
        # Finalize the h5 even if the recording fails partway through, otherwise what was recorded is
        # lost to an unclosed file.
        if recording:
            s.stop_recording()
            time.sleep(mx.Timing.waitAfterRecording)
        s.stop_file()
        s.group_delete_all()
        for array in arrays.values():
            array.close()
        on_progress(f"File closed. Data saved to: {params.save_path}")

    duration = time.time() - scan_start

    # MaxLab may have renamed the file (see _saved_file), so find what it really wrote before
    # reporting a path.
    h5_path = _saved_file(params, scan_start)

    well_files = {well: h5_path for well in params.wells}
    if registry_path is not None and len(params.wells) > 1:
        on_progress("\nSplitting into one file per well...")
        well_files = _split_into_store(h5_path, params, config, "network_scan", on_progress)
        h5_path = well_files[min(well_files)]

    result = NetworkScanResult(
        h5_path=h5_path,
        params=params,
        recorded_sec=recorded_sec,
        duration_sec=duration,
        well_files=well_files,
    )
    _register_scan(result, registry_path, on_progress)

    on_progress(f"\nNetwork scan complete! ({duration / 60:.1f} min)")

    return result


def _record(seconds: int, on_progress: ProgressFn, should_stop: StopFn | None) -> float:
    """Hold for the length of the recording, reporting progress and watching for a stop request.

    A network scan is a single long recording, so a plain ``time.sleep`` would leave a caller no way
    to end it and no sign of life for minutes at a time. Sleeping in short steps costs nothing --
    the recording runs on the chip, not here.

    :param seconds: How long to record for.
    :param on_progress: Where to report elapsed time.
    :param should_stop: Polled each step; ``True`` returns early.
    :returns: Seconds actually recorded.
    :rtype: float
    """
    start = time.time()
    next_report = _REPORT_SEC

    while True:
        elapsed = time.time() - start
        if elapsed >= seconds:
            return float(seconds)
        if should_stop is not None and should_stop():
            on_progress(f"  Stopping early after {elapsed:.0f} s of {seconds} s.")
            return elapsed
        if elapsed >= next_report:
            on_progress(f"  Recording... {elapsed:.0f}/{seconds} s")
            next_report += _REPORT_SEC
        time.sleep(min(_POLL_SEC, seconds - elapsed))


def _register_scan(
    result: NetworkScanResult, registry_path: Path | None, on_progress: ProgressFn
) -> None:
    """Record the scan in the registry, without letting bookkeeping lose a finished scan.

    The same trade as :func:`mxtreme.scans.activity_scan._register_scan`: the recording is the
    expensive, unrepeatable part, so a registry that cannot be written is reported and swallowed
    rather than raised over data already safely on disk. :func:`mxtreme.io.rebuild_registry` recovers
    the row from the file afterwards.

    :param result: The completed (or early-stopped) scan.
    :param registry_path: The registry to record the scan in, or ``None`` to skip registration --
        which is the case when the caller redirected the scan out of the store.
    :param on_progress: Where to report a failure.
    """
    params = result.params

    if registry_path is None:
        on_progress(
            "Not registered: this scan was written to an explicit save_path, so it is not in a "
            "managed store to be indexed by one."
        )
        return

    try:
        io.register_scan(params, registry_path, kind="network_scan")
        on_progress(f"Registered wells {params.wells} in {registry_path}")
    except (OSError, ValueError) as exc:  # ValueError covers a registry CSV pandas cannot parse
        on_progress(
            f"Warning: could not register the scan ({exc}). The file is fine; "
            "io.rebuild_registry(config) will pick it up."
        )
