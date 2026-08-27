"""Run an activity scan on the rig and write it to a single ``.h5``.

An *activity scan* sweeps the whole array to find where a culture is firing. The chip can only route
:data:`MAX_ROUTED_ELECTRODES` electrodes at a time, so the array is covered in a series of short
recordings, each routing a different random subset. Feed the resulting file to
:mod:`mxtreme.scans.electrode_selection` to turn it into a network-scan electrode list.

The module is split in two, deliberately:

- **Planning** -- :class:`ActivityScanParams`, :func:`available_electrodes` and
  :func:`plan_scan_electrodes` are pure Python and import nothing rig-specific, so a front end can
  build, validate and cost out a scan on any machine.
- **Execution** -- :func:`run_activity_scan` drives the chip and needs MaxWell's ``maxlab`` API. It
  imports it on the first call rather than at module import, so the planning half stays usable off
  the rig. This is the one difference from :mod:`mxtreme.scans.mx_setup`, which is rig-only end to
  end and therefore guards at import time.

Typical use::

    from mxtreme.scans import activity_scan

    params = activity_scan.ActivityScanParams(chip="M07460", plate_date=260810, div=1, wells=[0])
    result = activity_scan.run_activity_scan(params)
    # -> result.h5_path feeds electrode_selection.select_electrodes()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from mxtreme import device

#: Total electrodes on the array.
N_TOTAL_ELECTRODES = device.CHIP_WIDTH * device.CHIP_HEIGHT  # 26400

#: The most electrodes the chip can route (and so record) simultaneously, per well.
MAX_ROUTED_ELECTRODES = 1020

#: Wells on a MaxTwo plate; MaxOne only has well 0.
MAX_WELLS = 6

#: What the MaxLab server's ``wellplate_query_version`` reply means.
SYSTEM_TYPES = {0: "MaxOne", 1: "MaxTwo"}

#: Shared opening for every "there is nothing to scan with" failure.
_NO_DEVICE = (
    "No MaxWell device detected. Check that the chip is plugged in and MaxLab Live is running"
)

ProgressFn = Callable[[str], None]


@dataclass
class ActivityScanParams:
    """Everything needed to plan and run one activity scan.

    The defaults are a working scan: ~8 minutes of recording over one well, covering roughly a
    quarter of the array. Only ``chip``, the plate metadata and ``save_path`` really have to be set
    per experiment.

    :param chip: Chip serial, e.g. ``"M07460"``. Written to the file's metadata and used in its name.
    :param plate_date: Plating date as ``YYMMDD``; validated by :func:`mxtreme.scans.mx_setup.write_metadata`.
    :param div: Days *in vitro* at the time of the scan.
    :param exp_id: Task name recorded in the metadata blob.
    :param wells: Wells to scan, each in ``0..5``. All are scanned simultaneously.
    :param conditions: Optional per-well condition labels; must be empty or the same length as
        ``wells``.
    :param description: Free-text description written to the file.
    :param n_scans: Number of recordings in the scan. More scans cover more of the array.
    :param rec_length_sec: Length of each individual recording, in seconds.
    :param electrode_spacing: Spatial subsetting of the electrodes eligible for selection. ``0``
        makes every electrode available, ``1`` every other electrode in both rows and columns, and
        so on -- a coarser grid covers more of the array's *area* per scan.
    :param electrodes_per_scan: Electrodes routed per well per recording; capped at
        :data:`MAX_ROUTED_ELECTRODES`.
    :param save_path: Directory the ``.h5`` is written into.
    """

    # Metadata
    chip: str = "M07460"
    plate_date: int = 260810
    div: int = 1
    exp_id: str = "activity scan"
    wells: list[int] = field(default_factory=lambda: [0])
    conditions: list[str] = field(default_factory=list)
    description: str = "Activity scan to determine recording electrodes for each well"

    # Scan
    n_scans: int = 8
    rec_length_sec: int = 60
    electrode_spacing: int = 1
    electrodes_per_scan: int = MAX_ROUTED_ELECTRODES

    # Output
    save_path: str = "/home/mxwbio/Desktop/HAL/data/kam_activity_scan_testing"

    @property
    def file_name(self) -> str:
        """Base name of the ``.h5``, without MaxWell's ``.raw.h5`` suffix."""
        return f"{self.chip}_{self.plate_date}_ACTIVITY_SCAN"

    @property
    def h5_path(self) -> Path:
        """Where MaxWell will write the scan: ``<save_path>/<file_name>.raw.h5``."""
        return Path(self.save_path) / f"{self.file_name}.raw.h5"

    @property
    def estimated_minutes(self) -> float:
        """Total recording time in minutes, excluding routing and offset-compensation overhead."""
        return self.n_scans * self.rec_length_sec / 60

    @property
    def n_available_electrodes(self) -> int:
        """How many electrodes ``electrode_spacing`` leaves eligible for selection."""
        return len(available_electrodes(self.electrode_spacing))

    @property
    def array_coverage(self) -> float:
        """Fraction of the *eligible* electrodes the scan records, as a multiple of full coverage.

        Above 1.0 the scan revisits electrodes it has already recorded.
        """
        return self.n_scans * self.electrodes_per_scan / self.n_available_electrodes

    def as_metadata(self) -> dict:
        """Build the metadata dict written into the file.

        Shaped for :func:`mxtreme.scans.mx_setup.write_metadata`, and therefore for
        :func:`mxtreme.extract.extract`, which reads the same blob back out.
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

        :raises ValueError: On any parameter that would fail mid-scan -- an empty or out-of-range
            well list, a non-positive scan count or recording length, a routing request above
            :data:`MAX_ROUTED_ELECTRODES`, or a spacing so coarse that fewer electrodes remain
            available than each scan records (which would silently pad every scan with repeats).
        """
        if not self.wells:
            raise ValueError("At least one well must be selected.")
        if not all(0 <= w < MAX_WELLS for w in self.wells):
            raise ValueError(f"Wells must be in 0..{MAX_WELLS - 1}, got {self.wells}.")
        if len(set(self.wells)) != len(self.wells):
            raise ValueError(f"Duplicate wells in {self.wells}.")
        if self.conditions and len(self.conditions) != len(self.wells):
            raise ValueError(
                f"Conditions must be empty or one per well: {len(self.conditions)} conditions "
                f"for {len(self.wells)} wells."
            )
        if self.n_scans < 1:
            raise ValueError(f"n_scans must be at least 1, got {self.n_scans}.")
        if self.rec_length_sec <= 0:
            raise ValueError(f"rec_length_sec must be positive, got {self.rec_length_sec}.")
        if self.electrode_spacing < 0:
            raise ValueError(f"electrode_spacing must be non-negative, got {self.electrode_spacing}.")
        if not 1 <= self.electrodes_per_scan <= MAX_ROUTED_ELECTRODES:
            raise ValueError(
                f"electrodes_per_scan must be in 1..{MAX_ROUTED_ELECTRODES}, "
                f"got {self.electrodes_per_scan}."
            )

        # A scan cannot route more distinct electrodes than the spacing leaves available, so every
        # scan would be padded with repeats of the same electrodes.
        n_available = self.n_available_electrodes
        if n_available < self.electrodes_per_scan:
            raise ValueError(
                f"electrode_spacing={self.electrode_spacing} leaves only {n_available} electrodes "
                f"available, fewer than the {self.electrodes_per_scan} that each scan records. "
                "Lower the spacing."
            )


@dataclass
class ActivityScanResult:
    """What a completed (or partially completed) scan produced.

    :param h5_path: The file that was written. It exists even if the scan stopped early -- the file
        is always finalized.
    :param params: The parameters the scan ran with.
    :param scan_electrodes: ``{well: [electrodes_for_scan_0, electrodes_for_scan_1, ...]}``, holding
        only the scans that actually recorded.
    :param completed_scans: How many recordings finished.
    :param duration_sec: Wall-clock time spent scanning.
    """

    h5_path: Path
    params: ActivityScanParams
    scan_electrodes: dict[int, list[list[int]]]
    completed_scans: int
    duration_sec: float

    @property
    def complete(self) -> bool:
        """Whether every planned recording finished."""
        return self.completed_scans == self.params.n_scans


def available_electrodes(spacing: int = 1) -> list[int]:
    """List the electrodes a given spatial subsetting leaves eligible for selection.

    ``spacing=0`` returns every electrode; ``spacing=1`` every other electrode in both rows and
    columns; and so on. Spreading the eligible set out means a fixed number of routed electrodes
    covers more of the array's area.

    :param spacing: Electrodes skipped between eligible ones, in both axes.
    :returns: Eligible electrode numbers, ascending.
    """
    step = spacing + 1
    return [
        n
        for n in range(N_TOTAL_ELECTRODES)
        if n % step == 0 and (n // device.CHIP_WIDTH) % step == 0
    ]


def plan_scan_electrodes(
    params: ActivityScanParams, seed: int | None = None
) -> dict[int, list[list[int]]]:
    """Decide which electrodes each well records in each scan, before touching the chip.

    Each well gets its own shuffled pool of the available electrodes, sliced one scan at a time, so
    the scans of a well are disjoint by construction until the pool is exhausted. Enough shuffled
    copies of the pool are concatenated to cover ``n_scans * electrodes_per_scan`` draws; once the
    scan asks for more than one full coverage the extra copies are reshuffled, so repeats land in
    different scans rather than in the same one.

    :param params: The scan to plan. Validated first.
    :param seed: Seed for the shuffling, for a reproducible plan. ``None`` draws fresh randomness.
    :returns: ``{well: [electrodes_for_scan_0, ...]}``, one list of ``electrodes_per_scan`` per scan.
    """
    params.validate()

    pool = available_electrodes(params.electrode_spacing)
    rng = np.random.default_rng(seed)
    per_scan = params.electrodes_per_scan

    # Number of full array coverages the scan asks for; +1 copy so the final slice is always filled.
    n_coverages = (params.n_scans * per_scan) // len(pool)

    plan: dict[int, list[list[int]]] = {}
    for well in params.wells:
        shuffled = np.concatenate([rng.permutation(pool) for _ in range(n_coverages + 1)])
        plan[well] = [
            shuffled[i * per_scan : (i + 1) * per_scan].tolist() for i in range(params.n_scans)
        ]

    return plan


def describe(params: ActivityScanParams) -> str:
    """Render a human-readable summary of a scan, for confirming it before it runs.

    :param params: The scan to describe.
    :returns: A multi-line summary.
    """
    lines = [
        f"Chip           : {params.chip}",
        f"Plate date     : {params.plate_date} | DIV: {params.div}",
        f"Experiment     : {params.exp_id}",
        f"Wells          : {params.wells}",
        f"Scans          : {params.n_scans} x {params.rec_length_sec} s",
        (
            f"Electrodes     : {params.electrodes_per_scan} per scan per well, "
            f"drawn from {params.n_available_electrodes} available "
            f"(spacing {params.electrode_spacing}) of {N_TOTAL_ELECTRODES} total"
        ),
        f"Array coverage : {params.array_coverage:.2f}x of the available electrodes",
        f"Recording time : {params.estimated_minutes:.1f} min (excluding routing overhead)",
        f"Saving to      : {params.h5_path}",
        "                 (MaxLab appends _1, _2 ... if that name is already taken)",
    ]
    return "\n".join(lines)


def _require_maxlab():
    """Import ``maxlab``, with an error that says what to do when it is missing.

    Imported lazily so the planning half of this module works off the rig.

    :raises ModuleNotFoundError: When MaxLab Live's Python API is not importable.
    """
    try:
        import maxlab as mx
    except ModuleNotFoundError as exc:  # pragma: no cover -- depends on a rig-only install
        raise ModuleNotFoundError(
            "Running an activity scan requires the 'maxlab' Python API, which ships with MaxWell's "
            "MaxLab Live software and is not available from PyPI. Install MaxLab Live on the rig "
            "machine and put its Python package on PYTHONPATH. Planning a scan and selecting "
            "electrodes from one (mxtreme.scans.electrode_selection) do not need maxlab."
        ) from exc
    return mx


#: Where MaxLab's own Activity Scan assay records the length of each recording, and where
#: :func:`mxtreme.scans.electrode_selection.load_activity_scan` looks for it.
RECORD_TIME_PATH = "assay/inputs/record_time"


def has_record_time(h5_path: str | Path) -> bool:
    """Whether a scan file carries the recording length electrode selection needs.

    :param h5_path: Path to the ``.h5``.
    """
    with h5py.File(str(h5_path), "r") as f:
        return RECORD_TIME_PATH in f


def ensure_record_time(h5_path: str | Path, seconds: int | None = None) -> int:
    """Write ``/assay/inputs/record_time`` into a scan file if it is not already there.

    :func:`mxtreme.scans.electrode_selection.load_activity_scan` reads that dataset to turn spike
    counts into firing rates. MaxLab writes it for scans run through its own Activity Scan assay, but
    not for one driven through the ``Saving`` API the way :func:`run_activity_scan` does -- which
    otherwise leaves the file unreadable by the selection pipeline with::

        KeyError: Unable to open object (object 'record_time' doesn't exist)

    Everything else the pipeline reads (``/wells/wellNNN/recNNNN/...``) is already written by MaxLab,
    so this one dataset is all that stands between a scan and electrode selection.

    :param h5_path: Path to the ``.h5``. Opened for writing only when the dataset is missing.
    :param seconds: The recording length to write. When ``None`` it is derived from the recordings'
        own start/stop timestamps, so an already-recorded file can be repaired without knowing what
        it was run with.
    :raises ValueError: If ``seconds`` is not given and the file has no usable timestamps.
    :returns: The recording length now stored in the file, in seconds.
    """
    h5_path = str(h5_path)

    with h5py.File(h5_path, "r") as f:
        if RECORD_TIME_PATH in f:
            return int(f[RECORD_TIME_PATH][0])
        if seconds is None:
            seconds = _recorded_seconds(f)

    with h5py.File(h5_path, "r+") as f:
        f.require_group("assay/inputs").create_dataset(
            "record_time", data=np.array([int(seconds)], dtype=np.int64)
        )

    return int(seconds)


def _recorded_seconds(f: h5py.File) -> int:
    """Derive one recording's length from the start/stop timestamps MaxLab writes per recording.

    The median is used rather than the first recording's span, so one truncated recording (an
    interrupted scan) does not skew the value every firing rate is divided by.

    :param f: An open scan file.
    :raises ValueError: If no recording carries a usable pair of timestamps.
    :returns: Recording length in whole seconds.
    """
    spans = []
    for well in f.get("wells", {}).values():
        for recording in well.values():
            if "start_time" in recording and "stop_time" in recording:
                # MaxLab stores these as epoch milliseconds.
                spans.append((int(recording["stop_time"][0]) - int(recording["start_time"][0])) / 1000)

    spans = [s for s in spans if s > 0]
    if not spans:
        raise ValueError(
            "Cannot work out the recording length: the file has no usable start/stop timestamps. "
            "Pass the length explicitly."
        )

    return int(round(float(np.median(spans))))


def _saved_file(params: ActivityScanParams, started_at: float) -> Path:
    """Find the file MaxLab actually wrote for this scan.

    ``Saving.start_file`` does not overwrite: asked for a name that already exists, it appends
    ``_1``, ``_2`` and so on. Taking :attr:`ActivityScanParams.h5_path` on faith would therefore hand
    the caller a *previous* scan's file -- and the chain into electrode selection would silently
    analyse the wrong recording.

    :param params: The scan that was just run.
    :param started_at: ``time.time()`` from just before recording began; files older than this are
        left-overs from earlier scans.
    :returns: The newest matching file written during this scan, falling back to the predicted path
        if nothing matches.
    """
    candidates = [
        path
        for path in Path(params.save_path).glob(f"{params.file_name}*.h5")
        # A second of slack: the file is created just before `started_at` is taken.
        if path.stat().st_mtime >= started_at - 1
    ]
    if not candidates:
        return params.h5_path

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _connected_device(mx) -> str:
    """Ask the MaxLab server which system it is attached to, and fail if nothing answers.

    Cheap insurance against starting a scan with nothing plugged in: without this, a missing device
    is only discovered inside ``mx.initialize()``, by which point the ``.h5`` has been created.

    :param mx: The imported ``maxlab`` module.
    :raises RuntimeError: If the server cannot be reached or does not name a system.
    :returns: The device name, e.g. ``"MaxOne"``.
    """
    reply = None
    try:
        with mx.comm.api_context() as api:
            reply = api.send("wellplate_query_version")
    except Exception as exc:
        raise RuntimeError(f"{_NO_DEVICE} ({type(exc).__name__}: {exc})") from exc

    # A failed query does not necessarily raise: maxlab's api_context prints its own complaint
    # ("Connection refused / Please check, whether the server is running") and returns normally,
    # leaving the reply unset. So no reply is the usual "nothing answered" signal, not an exception.
    if reply is None or not str(reply).strip():
        raise RuntimeError(f"{_NO_DEVICE} (the server did not answer; see the message above)")

    try:
        system_type = int(str(reply).strip())
    except ValueError:
        raise RuntimeError(f"{_NO_DEVICE} (unrecognised reply from the server: {reply!r})") from None

    return SYSTEM_TYPES.get(system_type, f"unrecognised system type {system_type}")


def run_activity_scan(
    params: ActivityScanParams,
    seed: int | None = None,
    on_progress: ProgressFn = print,
) -> ActivityScanResult:
    """Run an activity scan on the rig and save every recording into one ``.h5``.

    For each of ``params.n_scans`` rounds: route this round's electrode subset in every well,
    compensate amplifier offsets, then record all wells simultaneously for
    ``params.rec_length_sec``. The electrode subsets come from :func:`plan_scan_electrodes`, so the
    rounds of a well do not overlap until the array has been covered once.

    The file is finalized and the arrays closed even if a round fails partway through -- otherwise
    every completed recording would be lost to an unclosed file. A failure still propagates, but the
    partial file on disk is readable.

    :param params: The scan to run. Validated before the chip is touched.
    :param seed: Seed for electrode-subset shuffling, for a reproducible scan.
    :param on_progress: Called with each progress line. Defaults to :func:`print`; pass a callback
        to route it into a UI, or ``lambda _: None`` to silence it.
    :raises ModuleNotFoundError: If ``maxlab`` is not installed (i.e. this is not the rig).
    :raises ValueError: If the parameters are not runnable; see :meth:`ActivityScanParams.validate`.
    :raises RuntimeError: If no device is connected, or if routing fails for any well.
    :returns: An :class:`ActivityScanResult` naming the file and the electrodes each scan recorded.
    """
    mx = _require_maxlab()
    from mxtreme.scans import mx_setup  # rig-only import; deferred for the same reason as maxlab

    plan = plan_scan_electrodes(params, seed=seed)  # validates params

    on_progress("=== Activity scan ===")
    on_progress(describe(params))

    on_progress(f"Device: {_connected_device(mx)}")

    Path(params.save_path).mkdir(parents=True, exist_ok=True)

    on_progress("Initializing chip...")
    mx.initialize()
    time.sleep(mx.Timing.waitInit)

    on_progress(f"Activating wells {params.wells}")
    mx.activate(params.wells)

    # One array per well, to manage that well's electrode selection and routing
    arrays = {well: mx.Array(f"well_{well}_array") for well in params.wells}

    # Begin the file that holds every scan, for every well
    s = mx.Saving()
    s.open_directory(params.save_path)
    s.start_file(params.file_name)  # creates the (empty) h5
    s.group_delete_all()  # clear stale group definitions
    mx_setup.write_metadata(s, params.as_metadata())
    mx_setup.write_exp_description(s, params.description)
    on_progress(f"Opened file: {params.h5_path}")

    recorded: dict[int, list[list[int]]] = {well: [] for well in params.wells}
    completed = 0
    scan_start = time.time()

    try:
        for i in range(params.n_scans):
            on_progress(f"\n--- Scan {i + 1}/{params.n_scans} ---")

            # 1. Route this round's electrodes in every well
            for well in params.wells:
                scan_elecs = plan[well][i]

                on_progress(f"  Well {well}: routing {len(scan_elecs)} electrodes...")
                array = arrays[well]
                array.reset()  # clear server-side state
                array.clear_selected_electrodes()  # clear the selection list
                array.select_electrodes(scan_elecs)  # "I want these <=1020 electrodes"
                result = array.route()  # server solves electrode->channel wiring
                if str(result).upper() != "OK":  # route() returns 'OK'/'ERROR', it never raises
                    raise RuntimeError(f"Routing failed on scan {i}, well {well}: {result!r}")
                array.download([well])  # push the solution to the chip
                recorded[well].append(scan_elecs)
                on_progress(f"  Well {well}: routed and downloaded")

            time.sleep(mx.Timing.waitAfterDownload)
            on_progress("  Running offset compensation...")
            mx.offset()  # amplifier offset compensation
            time.sleep(mx.Timing.waitInMX2Offset)

            # 2. Record and save data from all wells
            on_progress(f"  Recording {params.rec_length_sec} s...")
            s.start_recording(params.wells)
            time.sleep(params.rec_length_sec)
            s.stop_recording()
            time.sleep(mx.Timing.waitAfterRecording)

            completed += 1
            elapsed = time.time() - scan_start
            on_progress(
                f"  Scan {i + 1}/{params.n_scans} complete ({elapsed / 60:.1f} min elapsed)"
            )

    finally:
        on_progress("\nFinalizing file and closing arrays...")
        # Finalize the h5 even if a scan fails partway through, otherwise every completed recording
        # is lost to an unclosed file
        s.stop_file()
        s.group_delete_all()
        for array in arrays.values():
            array.close()
        on_progress(f"File closed. Data saved to: {params.save_path}")

    duration = time.time() - scan_start

    # MaxLab may have renamed the file (see _saved_file), so find what it really wrote before
    # reporting a path or touching it.
    h5_path = _saved_file(params, scan_start)
    try:
        ensure_record_time(h5_path, params.rec_length_sec)
    except (OSError, ValueError) as exc:
        on_progress(
            f"Warning: could not write the recording length into {h5_path.name} ({exc}). "
            "Electrode selection will need it added before it can read this scan."
        )

    on_progress(f"\nActivity scan complete! ({duration / 60:.1f} min)")

    return ActivityScanResult(
        h5_path=h5_path,
        params=params,
        scan_electrodes=recorded,
        completed_scans=completed,
        duration_sec=duration,
    )
