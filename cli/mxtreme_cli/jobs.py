"""Activity scans running in the background, and the status the rest of the CLI reads off them.

A scan takes minutes to tens of minutes, during which the terminal used to be unusable. Here each
scan runs on its own thread and the menus stay live, so the status line in the banner is the only
place a running scan is visible from elsewhere in the CLI.

Two rules shape everything in this module:

- **A job never prints.** Output from a background thread would land in the middle of whatever menu
  is on screen. Progress is captured into the job instead, and the foreground draws it.
- **A job is only ever read through its lock.** The scan thread writes counters while the main
  thread reads them to draw the banner.

The scan itself is unchanged: this drives
:func:`mxtreme.scans.activity_scan.run_activity_scan` through the hooks it already offers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from mxtreme.scans import activity_scan as scan
from mxtreme.scans.activity_scan import ActivityScanParams

#: Progress lines kept per job. Enough to see what a scan is doing without holding a whole run.
_LOG_LIMIT = 400


class State(Enum):
    """Where a scan has got to."""

    RUNNING = "running"
    FINISHED = "finished"
    STOPPED = "stopped"  # ended early at the user's request; the file is still good
    FAILED = "failed"

    @property
    def done(self) -> bool:
        """Whether the thread has ended, whatever the outcome."""
        return self is not State.RUNNING


@dataclass
class ScanJob:
    """One activity scan running on its own thread.

    :param params: The scan being run. Its ``file_name`` identifies the job to the user.
    :param seed: Seed the scan was started with, kept for the report.
    """

    params: ActivityScanParams
    seed: int | None = None

    state: State = State.RUNNING
    completed: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: scan.ActivityScanResult | None = None
    error: BaseException | None = None

    #: Set once the user has been shown this job's report, so a finished scan stops occupying a
    #: line in the banner but is still listed under "scans this session".
    reported: bool = False

    _log: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_LIMIT))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    # -- read by the foreground ---------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier shown in the banner and in menus."""
        return self.params.file_name

    @property
    def total(self) -> int:
        """How many recordings the scan is meant to make."""
        return self.params.n_scans

    def snapshot(self) -> tuple[State, int, float]:
        """Read state, completed rounds and elapsed seconds together, under one lock.

        Reading them separately could mix a stale count with a fresh state, which shows up as a
        finished scan drawn at 7/8.
        """
        with self._lock:
            end = self.finished_at if self.finished_at is not None else time.time()
            return self.state, self.completed, end - self.started_at

    def progress(self) -> float:
        """Fraction of the scan's recordings that have finished, in ``0..1``."""
        _state, completed, _elapsed = self.snapshot()
        return completed / self.total if self.total else 1.0

    def seconds_left(self) -> float | None:
        """Estimated seconds until the scan finishes, or ``None`` if there is nothing to go on.

        Measured from the rounds already done rather than from the nominal recording length, so the
        estimate absorbs routing and offset-compensation overhead instead of ignoring it.
        """
        state, completed, elapsed = self.snapshot()
        if state.done:
            return None
        if completed == 0:
            # Nothing measured yet, so fall back to the plan's own estimate.
            return max(self.params.estimated_minutes * 60 - elapsed, 0.0)
        return max((elapsed / completed) * (self.total - completed), 0.0)

    def log(self) -> list[str]:
        """A copy of the progress lines captured so far."""
        with self._lock:
            return list(self._log)

    def status_line(self) -> str:
        """One line describing the job, for the banner."""
        state, completed, elapsed = self.snapshot()
        if state is State.RUNNING:
            left = self.seconds_left()
            eta = f"~{_duration(left)} left" if left is not None else "estimating"
            return f"{self.name}  {_bar(self.progress())} {completed}/{self.total}  {eta}"
        outcome = {
            State.FINISHED: "complete",
            State.STOPPED: "stopped early",
            State.FAILED: f"failed ({type(self.error).__name__})",
        }[state]
        return f"{self.name}  {completed}/{self.total} recordings  {outcome} in {_duration(elapsed)}"

    # -- control ------------------------------------------------------------------------------

    def stop(self) -> None:
        """Ask the scan to stop after the recording in progress.

        Deliberately not immediate: ending between rounds lets the ``.h5`` be finalized, so every
        recording made so far stays readable. A stop mid-recording would leave an unclosed file.
        """
        self._stop.set()

    @property
    def stopping(self) -> bool:
        """Whether a stop has been asked for but the scan has not ended yet."""
        return self._stop.is_set() and not self.state.done

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the scan thread ends.

        :returns: ``True`` if it ended, ``False`` if the timeout ran out first.
        """
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- the thread ---------------------------------------------------------------------------

    def _capture(self, message: str) -> None:
        """Progress callback: keep the line, print nothing."""
        with self._lock:
            for line in message.splitlines():
                self._log.append(line)

    def _count(self, completed: int, _total: int) -> None:
        """Round callback: advance the counter the banner draws from."""
        with self._lock:
            self.completed = completed

    def _run(self) -> None:
        """Body of the scan thread. Never raises: a failure is recorded on the job instead."""
        try:
            result = scan.run_activity_scan(
                self.params,
                seed=self.seed,
                on_progress=self._capture,
                on_round=self._count,
                should_stop=self._stop.is_set,
            )
        except BaseException as exc:  # noqa: BLE001 -- nothing above this frame could catch it
            with self._lock:
                self.state = State.FAILED
                self.error = exc
                self.finished_at = time.time()
            return

        with self._lock:
            self.result = result
            self.completed = result.completed_scans
            self.state = State.FINISHED if result.complete else State.STOPPED
            self.finished_at = time.time()


# ------------------------------------------------------------------------------------------------
# The registry -- one per session, since one rig is one chip
# ------------------------------------------------------------------------------------------------

_jobs: list[ScanJob] = []
_registry_lock = threading.Lock()


def start(params: ActivityScanParams, seed: int | None = None) -> ScanJob:
    """Start a scan on a background thread and return the job tracking it.

    The thread is not a daemon: quitting the CLI has to deal with a running scan deliberately
    (see :func:`active`), rather than having the interpreter drop it mid-recording and leave an
    unfinalized file behind.
    """
    job = ScanJob(params=params, seed=seed)
    job._thread = threading.Thread(target=job._run, name=f"scan-{job.name}", daemon=False)
    with _registry_lock:
        _jobs.append(job)
    job._thread.start()
    return job


def all_jobs() -> list[ScanJob]:
    """Every job started this session, oldest first."""
    with _registry_lock:
        return list(_jobs)


def active() -> list[ScanJob]:
    """Jobs whose scan is still running."""
    return [job for job in all_jobs() if job.state is State.RUNNING]


def finished() -> list[ScanJob]:
    """Jobs whose scan has ended, however it ended."""
    return [job for job in all_jobs() if job.state.done]


def status_lines() -> list[str]:
    """Banner lines for every job worth showing: all running ones, plus recent outcomes.

    Finished jobs stay listed so a scan that ended while the user was on another screen is not
    silently missed; they are dropped once the user has been shown the report.
    """
    return [job.status_line() for job in all_jobs() if job.state is State.RUNNING or not job.reported]


def clashing(params: ActivityScanParams) -> list[ScanJob]:
    """Running jobs that would write to the same file as ``params``.

    MaxLab never overwrites: a second scan of the same name becomes ``<name>_1.raw.h5``. The point
    of this check is that the user is told, not that the collision is prevented.
    """
    target = Path(params.save_path).expanduser().resolve(strict=False) / params.file_name
    matches = []
    for job in active():
        other = Path(job.params.save_path).expanduser().resolve(strict=False) / job.params.file_name
        if other == target:
            matches.append(job)
    return matches


# ------------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------------


def _bar(fraction: float, width: int = 12) -> str:
    """Render a fraction as a fixed-width bar, so status lines stay aligned as they update."""
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _duration(seconds: float | None) -> str:
    """Render a duration the way someone waiting for it would say it."""
    if seconds is None:
        return "unknown"
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m"
