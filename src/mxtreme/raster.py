"""A quick-look raster of any raw recording, straight from its spike table.

Everything else that draws spikes over time here works on a *preprocessed* recording: extracted,
cleaned, binned. That is the right input for analysis and the wrong one for the question asked
right after a recording ends -- *did it record what it was meant to?* -- because an activity scan
is never preprocessed at all, and a network scan or experiment only later. This module answers it
from the raw ``.h5`` alone: :func:`load_spike_raster` reads one well's spike table as MaxLab wrote
it, recording by recording, and :meth:`SpikeRaster.binned` reduces it to a count matrix a front end
can paint.

Nothing is filtered. No amplitude or firing-rate threshold, no sign filter, no artifact removal:
the point is to see what the rig measured, so the only spikes left out are those on a channel the
recording's mapping does not route (they have no electrode to be drawn on).

Only the spike table and the settings beside it are read, never the raw traces, so this stays cheap
on a multi-gigabyte recording. It needs no rig library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RasterSegment:
    """One recording of a well: what it routed and how long it ran.

    An activity scan has one per electrode configuration; a network scan or an experiment normally
    has exactly one. Rows of the raster are the recording's routed electrodes in ascending
    electrode order -- which is spatial order, row by row across the array -- so row ``i`` of one
    segment and row ``i`` of the next are different electrodes whenever the routing changed.

    :param name: The recording's group name in the file, e.g. ``"rec0003"``.
    :param started_at_ms: MaxLab's ``start_time`` stamp (ms since the epoch), or ``None`` when the
        file carries none.
    :param duration_sec: How long the recording ran. From MaxLab's start/stop stamps when present,
        otherwise the assay's ``record_time``, otherwise the span of the spikes; never shorter than
        the last spike.
    :param n_spikes: Spikes on routed channels -- the ones the raster holds.
    :param n_unrouted: Spikes the table holds on channels the mapping does not route, left out.
    :param electrodes: Electrode number of each row.
    :param channels: Readout channel of each row.
    """

    name: str
    started_at_ms: int | None
    duration_sec: float
    n_spikes: int
    electrodes: np.ndarray
    channels: np.ndarray
    n_unrouted: int = 0

    @property
    def n_rows(self) -> int:
        return len(self.electrodes)


@dataclass(frozen=True)
class BinnedRaster:
    """Spike counts on a display grid. See :meth:`SpikeRaster.binned`.

    :param counts: ``(n_row_bins, n_time_bins)`` spike counts, row 0 first.
    :param t0: Start of the window on the concatenated timeline, in seconds.
    :param t1: End of the window.
    :param rows_per_bin: How many raster rows fold into one row bin (1 = one electrode per row).
    """

    counts: np.ndarray
    t0: float
    t1: float
    rows_per_bin: int

    @property
    def bin_sec(self) -> float:
        return (self.t1 - self.t0) / self.counts.shape[1]


@dataclass(frozen=True)
class SpikeRaster:
    """Every spike of one well of a raw recording, placed in time and on a row.

    The file's recordings are laid end to end on one timeline, each starting where the previous
    one's duration ended (:attr:`offsets`), whatever gap the rig left between them: for an activity
    scan that reads as the scan's configurations side by side.

    :param segments: The well's recordings, in file order.
    :param segment: Per spike, the index into ``segments``.
    :param time_sec: Per spike, seconds since its own recording began.
    :param row: Per spike, its row within its segment (see :class:`RasterSegment`).
    """

    path: Path
    well: int
    samp_rate: float
    segments: tuple[RasterSegment, ...]
    segment: np.ndarray
    time_sec: np.ndarray
    row: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        """Where each segment starts on the concatenated timeline, in seconds."""
        durations = [s.duration_sec for s in self.segments]
        return np.concatenate([[0.0], np.cumsum(durations)[:-1]]) if durations else np.zeros(0)

    @property
    def duration_sec(self) -> float:
        return float(sum(s.duration_sec for s in self.segments))

    @property
    def n_rows(self) -> int:
        """Rows needed to draw every segment: the most electrodes any one recording routed."""
        return max((s.n_rows for s in self.segments), default=0)

    @property
    def n_spikes(self) -> int:
        return len(self.time_sec)

    def binned(
        self,
        n_time_bins: int = 1200,
        max_rows: int = 512,
        t0: float | None = None,
        t1: float | None = None,
    ) -> BinnedRaster:
        """Count spikes on a ``rows x time`` grid over a window of the concatenated timeline.

        :param n_time_bins: Columns of the grid -- the width it will be drawn at.
        :param max_rows: The most rows the grid may have. With more electrodes than that, whole
            numbers of neighbouring rows are folded together (``rows_per_bin``), so a row bin never
            straddles a fraction of an electrode.
        :param t0: Window start in seconds; the beginning when ``None``.
        :param t1: Window end; the end of the last recording when ``None``.
        :returns: The counts and the window they cover.
        """
        n_time_bins = max(1, int(n_time_bins))
        total = self.duration_sec
        t0 = 0.0 if t0 is None else min(max(float(t0), 0.0), total)
        t1 = total if t1 is None else min(max(float(t1), 0.0), total)
        if t1 <= t0:  # an empty or inverted window (or an empty file): one bin's worth, never zero
            t1 = t0 + max(total / n_time_bins, 1e-3)

        rows_per_bin = max(1, -(-self.n_rows // max(1, int(max_rows))))
        n_row_bins = max(1, -(-self.n_rows // rows_per_bin))
        counts = np.zeros((n_row_bins, n_time_bins), dtype=np.uint32)
        if self.n_spikes:
            t = self.offsets[self.segment] + self.time_sec
            inside = (t >= t0) & (t < t1)
            col = ((t[inside] - t0) * (n_time_bins / (t1 - t0))).astype(np.int64)
            np.clip(col, 0, n_time_bins - 1, out=col)
            np.add.at(counts, (self.row[inside] // rows_per_bin, col), 1)
        return BinnedRaster(counts=counts, t0=t0, t1=t1, rows_per_bin=rows_per_bin)

    def save(self, path: str | Path) -> Path:
        """Write the raster to a compressed ``.npz``, for a front end that re-bins it per zoom and
        should not reopen the recording each time. Read it back with :meth:`load`."""
        import json

        path = Path(path)
        meta = {
            "path": str(self.path),
            "well": self.well,
            "samp_rate": self.samp_rate,
            "segments": [
                {"name": s.name, "started_at_ms": s.started_at_ms, "duration_sec": s.duration_sec, "n_spikes": s.n_spikes,
                 "n_unrouted": s.n_unrouted}
                for s in self.segments
            ],
        }
        arrays = {"segment": self.segment, "time_sec": self.time_sec.astype(np.float32), "row": self.row}
        for i, s in enumerate(self.segments):
            arrays[f"electrodes_{i}"] = s.electrodes
            arrays[f"channels_{i}"] = s.channels
        with path.open("wb") as fh:  # an open handle, so numpy does not append ".npz" to the name
            np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SpikeRaster:
        """Read back what :meth:`save` wrote. Spike times come back at float32 precision -- a
        quarter of a millisecond an hour into a recording, below anything a raster resolves."""
        import json

        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            segments = tuple(
                RasterSegment(
                    name=s["name"],
                    started_at_ms=s["started_at_ms"],
                    duration_sec=float(s["duration_sec"]),
                    n_spikes=int(s["n_spikes"]),
                    electrodes=z[f"electrodes_{i}"],
                    channels=z[f"channels_{i}"],
                    n_unrouted=int(s.get("n_unrouted", 0)),
                )
                for i, s in enumerate(meta["segments"])
            )
            return cls(
                path=Path(meta["path"]),
                well=int(meta["well"]),
                samp_rate=float(meta["samp_rate"]),
                segments=segments,
                segment=z["segment"],
                time_sec=z["time_sec"].astype(np.float64),
                row=z["row"],
            )


def _first(dataset, cast=float):
    """The first value of a one-element dataset MaxLab may have written as a number or as text."""
    value = np.asarray(dataset[()]).ravel()[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace").strip()
    return cast(float(value))


def load_spike_raster(h5_path: str | Path, well: int) -> SpikeRaster:
    """Read one well's spikes from a raw MaxWell ``.h5``, recording by recording.

    Works on anything MaxLab writes -- an activity scan (many recordings per well), a network scan
    or an experiment (one) -- whether MXtreme or Scope recorded it, with or without raw traces.

    Spike frame numbers count from when the rig started, not from the recording, so each recording
    needs a zero: the first raw frame when the file holds traces, otherwise its first spike. A
    spike-only recording is therefore drawn from its first spike, and the silence before it (at
    most the wait for one spike on a routed array) is not shown.

    :param h5_path: The recording.
    :param well: Well number, as in ``/wells/well<NNN>``.
    :raises KeyError: If the file holds no such well.
    :raises ValueError: If the file has no ``/wells`` group, or the well has no recording with a
        spike table and a channel mapping.
    :returns: The well's spikes; see :class:`SpikeRaster`.
    """
    import h5py

    h5_path = Path(h5_path)
    segments: list[RasterSegment] = []
    seg_idx: list[np.ndarray] = []
    times: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    samp_rate = 0.0

    with h5py.File(str(h5_path), "r") as f:
        if "wells" not in f:
            raise ValueError(f"{h5_path.name} has no /wells group: not a MaxWell recording.")
        key = f"well{int(well):03d}"
        if key not in f["wells"]:
            have = sorted(int(k.removeprefix("well")) for k in f["wells"])
            raise KeyError(f"{h5_path.name} holds no well {well} (has {have})")
        record_time = None
        if "assay" in f and "inputs" in f["assay"] and "record_time" in f["assay/inputs"]:
            try:
                record_time = _first(f["assay/inputs/record_time"])
            except (ValueError, IndexError):
                record_time = None

        for name in sorted(f["wells"][key]):
            rec = f["wells"][key][name]
            if "spikes" not in rec or "settings" not in rec or "mapping" not in rec["settings"]:
                continue
            mapping = rec["settings/mapping"][:]
            rate = _first(rec["settings/sampling"])
            samp_rate = samp_rate or rate

            # Rows: the routed electrodes in ascending order. A channel wired to two electrodes
            # keeps the first; an electrode number below zero is MaxLab's "not routed".
            routed = mapping[mapping["electrode"] >= 0]
            _, keep = np.unique(routed["channel"], return_index=True)
            routed = np.sort(routed[keep], order="electrode")
            channels = routed["channel"].astype(np.int64)
            row_of_channel = np.full(int(channels.max()) + 1 if len(channels) else 1, -1, dtype=np.int64)
            row_of_channel[channels] = np.arange(len(channels))

            spikes = rec["spikes"][:]
            chan = spikes["channel"].astype(np.int64) if len(spikes) else np.zeros(0, dtype=np.int64)
            known = (chan >= 0) & (chan < len(row_of_channel))
            row = np.where(known, row_of_channel[np.where(known, chan, 0)], -1)
            frames = spikes["frameno"][row >= 0].astype(np.int64) if len(spikes) else np.zeros(0, dtype=np.int64)
            row = row[row >= 0]

            first_frame = None
            if "groups" in rec:
                for group in rec["groups"].values():
                    if "frame_nos" in group and group["frame_nos"].shape[0]:
                        first_frame = int(group["frame_nos"][0])
                        break
            if first_frame is None:
                first_frame = int(frames.min()) if len(frames) else 0
            t = (frames - first_frame) / rate
            t = np.maximum(t, 0.0)  # a spike stamped before the first raw frame belongs at the start

            started = stamped = None
            if "start_time" in rec:
                started = _first(rec["start_time"], int)
                if "stop_time" in rec:
                    span = (_first(rec["stop_time"], int) - started) / 1000.0
                    stamped = span if span > 0 else None
            duration = stamped if stamped is not None else (record_time or 0.0)
            if len(t):
                duration = max(duration, float(t.max()) + 1.0 / rate)

            order = np.argsort(t, kind="stable")
            seg_idx.append(np.full(len(t), len(segments), dtype=np.uint16))
            times.append(t[order])
            rows.append(row[order].astype(np.int32))
            segments.append(
                RasterSegment(
                    name=name,
                    started_at_ms=started,
                    duration_sec=float(duration),
                    n_spikes=len(t),
                    electrodes=routed["electrode"].astype(np.int64),
                    channels=channels,
                    n_unrouted=int(len(spikes) - len(t)),
                )
            )

    if not segments:
        raise ValueError(f"{h5_path.name} well {well} has no recording with a spike table and a channel mapping.")
    return SpikeRaster(
        path=h5_path,
        well=int(well),
        samp_rate=float(samp_rate),
        segments=tuple(segments),
        segment=np.concatenate(seg_idx),
        time_sec=np.concatenate(times),
        row=np.concatenate(rows),
    )
