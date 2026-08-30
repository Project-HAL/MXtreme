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
electrode plan to make: what to record is an input, not something this module decides.

Like an activity scan it lands in the package-managed store -- the ``.h5`` under
``config.scans_dir/<exp_id>/<chip>/``, one ``network_scan`` row per well in ``registry.csv`` -- and
setting :attr:`NetworkScanParams.save_path` writes it elsewhere, unregistered.

Typical use::

    from mxtreme.config import Config
    from mxtreme.scans import electrode_selection, network_scan

    config = Config.from_toml("mxtreme.toml")
    rec_elecs = electrode_selection.select_electrodes(str(scan.h5_path), str(config.scans_dir))
    params = network_scan.NetworkScanParams(
        recording_electrodes=rec_elecs,
        exp_id="May2025_Wave", chip="M07460", plate_date=260810, div=25,
        rec_length_sec=300,
    )
    result = network_scan.run_network_scan(params, config)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from mxtreme import io
from mxtreme.scans.activity_scan import (
    MAX_ROUTED_ELECTRODES,
    MAX_WELLS,
    N_TOTAL_ELECTRODES,
    ProgressFn,
    StopFn,
    _connected_device,
    _require_maxlab,
    _saved_file,
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
    :param chip: Chip serial, e.g. ``"M07460"``. Written to the file's metadata and used in its name.
    :param plate_date: Plating date as ``YYMMDD``; validated by :func:`mxtreme.scans.mx_setup.write_metadata`.
    :param div: Days *in vitro* at the time of the scan.
    :param exp_id: Task name recorded in the metadata blob, and the directory the scan files under.
    :param conditions: Optional per-well condition labels, in :attr:`wells` order; must be empty or
        the same length as ``wells``.
    :param description: Free-text description written to the file.
    :param rec_length_sec: Length of the recording, in seconds.
    :param save_path: Directory the ``.h5`` is written into. ``None`` -- the default -- means the
        managed store, resolved against the :class:`~mxtreme.config.Config` passed to
        :func:`run_network_scan`. Set it to write somewhere outside the store instead; the scan is
        then not registered.
    """

    recording_electrodes: dict[int, list[int]]

    # Metadata
    chip: str = "M07460"
    plate_date: int = 260810
    div: int = 1
    exp_id: str = "network_scan"
    conditions: list[str] = field(default_factory=list)
    description: str = "Network scan of the recording electrodes chosen from an activity scan"

    # Scan
    rec_length_sec: int = 300

    # Output -- None means "the managed store"; see `resolved`.
    save_path: str | None = None

    def __post_init__(self) -> None:
        """Normalize the electrode mapping to plain ``{int: [int, ...]}``.

        ``select_electrodes`` hands back numpy arrays of numpy integers. Those route perfectly well,
        but they serialize into the metadata blob as ``np.int64(4213)`` and compare unequally to the
        ints everything else uses, so they are converted once here rather than guarded against
        everywhere downstream.
        """
        self.recording_electrodes = {
            int(well): [int(e) for e in electrodes]
            for well, electrodes in self.recording_electrodes.items()
        }

    @property
    def wells(self) -> list[int]:
        """The wells this scan records, ascending: the keys of ``recording_electrodes``."""
        return sorted(self.recording_electrodes)

    @property
    def file_name(self) -> str:
        """Base name of the ``.h5``, without MaxWell's ``.raw.h5`` suffix.

        Built like every other name in the store (``DIV<div>_<plate_date>_<chip>_<exp_id>_...``), and
        ending in ``_network_scan`` so :func:`mxtreme.io.scan_kind` can tell it from an activity scan
        of the same culture -- the two sit in the same tree and carry the same metadata blob.
        """
        return f"DIV{self.div}_{self.plate_date}_{self.chip}_{self.exp_id}_network_scan"

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

        An explicit ``save_path`` wins and the object is returned unchanged; otherwise the scan lands
        beside the activity scan it came from, at ``config.scans_dir/<exp_id>/<chip>/``.

        :param config: The :class:`~mxtreme.config.Config` describing the managed store. Only needed
            when ``save_path`` is unset.
        :raises ValueError: If there is neither a ``save_path`` nor a ``config`` to derive one from.
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
        return replace(self, save_path=str(Path(config.scans_dir) / self.exp_id / self.chip))

    @property
    def estimated_minutes(self) -> float:
        """Recording time in minutes, excluding routing and offset-compensation overhead."""
        return self.rec_length_sec / 60

    def as_metadata(self) -> dict:
        """Build the metadata dict written into the file.

        Shaped for :func:`mxtreme.scans.mx_setup.write_metadata`, and therefore for
        :func:`mxtreme.extract.extract`, which reads the same blob back out -- so a network scan
        extracts and preprocesses like any other recording.
        """
        return {
            "Exp ID": self.exp_id,
            "Chip ID": self.chip,
            "Plate date": self.plate_date,
            "DIV": self.div,
            "Well IDs": list(self.wells),
            "Conditions": list(self.conditions),
        }

    def validate(self) -> None:
        """Check the parameters are runnable, so failures surface before the chip is touched.

        :raises ValueError: On any parameter that would fail mid-scan -- no wells, an out-of-range
            well or electrode, an empty or duplicated electrode set, a well asking for more than
            :data:`~mxtreme.scans.activity_scan.MAX_ROUTED_ELECTRODES` electrodes (which the chip
            cannot route, so some would be silently dropped), a mismatched condition list, or a
            non-positive recording length.
        """
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

    :param h5_path: The file that was written. It exists even if the scan stopped early -- the file
        is always finalized.
    :param params: The parameters the scan ran with.
    :param recorded_sec: How long the recording actually ran, which is less than
        ``params.rec_length_sec`` only when it was stopped early.
    :param duration_sec: Wall-clock time from the start of recording, including teardown.
    """

    h5_path: Path
    params: NetworkScanParams
    recorded_sec: float
    duration_sec: float

    @property
    def complete(self) -> bool:
        """Whether the recording ran for its full length."""
        return self.recorded_sec >= self.params.rec_length_sec


def describe(params: NetworkScanParams) -> str:
    """Render a human-readable summary of a scan, for confirming it before it runs.

    :param params: The scan to describe. An unresolved ``save_path`` is fine; the destination is
        then reported as the managed store rather than as a concrete path.
    :returns: A multi-line summary.
    """
    destination = f"the managed store, as {params.file_name}.raw.h5"
    per_well = ", ".join(
        f"well {well}: {len(params.recording_electrodes[well])}" for well in params.wells
    )
    lines = [
        f"Chip           : {params.chip}",
        f"Plate date     : {params.plate_date} | DIV: {params.div}",
        f"Experiment     : {params.exp_id}",
        f"Wells          : {params.wells}",
        f"Electrodes     : {per_well or 'none'}",
        f"Recording time : {params.estimated_minutes:.1f} min (excluding routing overhead)",
        # Describing a scan is a planning step, and planning happens before a Config is necessarily
        # in hand -- so an unresolved destination is reported, not raised over.
        f"Saving to      : {params.h5_path if params.save_path is not None else destination}",
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

    The scan goes into the managed store described by ``config``: the ``.h5`` under
    ``config.scans_dir/<exp_id>/<chip>/``, and one ``network_scan`` row per well in
    ``config.registry_path``. Registration happens after the recording finishes, including when it
    stopped early, so a short scan is still findable. An explicit ``params.save_path`` overrides the
    destination and skips registration -- a row for a file outside the store could never be resolved
    back to it.

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
    if config is not None and not hasattr(config, "scans_dir"):
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
    result = NetworkScanResult(
        h5_path=_saved_file(params, scan_start),
        params=params,
        recorded_sec=recorded_sec,
        duration_sec=duration,
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
