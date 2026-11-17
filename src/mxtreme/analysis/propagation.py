"""Burst shape measurements from spike frames and channel-map positions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mxtreme.analysis.intervals import burst_windows


def spike_arrays(rec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A recording's spikes as ``(frames, rows, xy)``: frames ascending, each spike's row in the
    channel map, and the rows' ``(x, y)`` in µm. Spikes on channels the map does not list are
    dropped, as :func:`mxtreme.clean.bin_spikes` drops them."""
    sd = rec.spike_data
    frames = np.asarray(sd["frameno"], dtype=np.int64)
    chan = np.asarray(sd["channel"], dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    frames, chan = frames[order], chan[order]
    cm = np.asarray(rec.channelmap, dtype=float)
    chans = cm[:, 1].astype(np.int64)
    xy = cm[:, 3:5].copy()
    if not chans.size:
        return frames[:0], np.empty(0, dtype=np.int64), xy
    by_channel = np.argsort(chans)
    sorted_channels = chans[by_channel]
    positions = np.searchsorted(sorted_channels, chan).clip(max=chans.size - 1)
    known = sorted_channels[positions] == chan
    return frames[known], by_channel[positions[known]].astype(np.int64), xy


def burst_spans(bursts, *, kinds: Sequence[str] | None = ("network",)) -> tuple[np.ndarray, np.ndarray]:
    """Return burst IDs and half-open frame spans, in onset order."""
    windows = burst_windows(bursts, kinds=kinds)
    starts = windows["start_frame"].to_numpy(dtype=np.int64)
    stops = windows["end_frame"].to_numpy(dtype=np.int64) + 1
    return windows["id"].to_numpy(), np.column_stack((starts, stops))


def frame_step(frames, limit: int = 200_000) -> int:
    """The recording's real time step in frames: the greatest common divisor of the differences
    between its distinct spike frames (1 for a 10 kHz recording, 10 for spikes that sit on a
    1 ms grid at a nominal 10 kHz)."""
    f = np.unique(np.asarray(frames, dtype=np.int64))
    if f.size < 2:
        return 1
    return max(1, int(np.gcd.reduce(np.diff(f)[:limit])))


def _first_spikes(frames, rows, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, first frame)`` of every channel that fires in ``[start, stop)``."""
    f = np.asarray(frames)
    a, b = np.searchsorted(f, [start, stop])
    cc, first = np.unique(np.asarray(rows)[a:b], return_index=True)
    return cc, f[a:b][first]


#: What :func:`burst_shape` reports per burst, beside the envelope.
BURST_SHAPE = (
    "engaged", "heading_deg", "heading_r2", "heading_p", "peak_hz", "rise_ms", "fall_ms", "rise_hz_per_ms", "fall_hz_per_ms",
)


def _rise_bins(rate: np.ndarray, peak: float) -> float:
    """Interpolate the 10–90% rise on a rate trace that ends at the peak."""
    low, high = 0.1 * peak, 0.9 * peak
    below = np.flatnonzero(rate < low)
    if not below.size:
        return np.nan

    last_below = int(below[-1])
    first_high = last_below + 1 + int(np.argmax(rate[last_below + 1:] >= high))
    low_crossing = last_below + (low - rate[last_below]) / (rate[last_below + 1] - rate[last_below])
    high_crossing = first_high - 1 + (high - rate[first_high - 1]) / (rate[first_high] - rate[first_high - 1])
    return high_crossing - low_crossing


def burst_shape(
    frames, rows, xy, samp_rate: float, spans, *, pre_ms: float = 20.0, post_ms: float = 100.0, bin_ms: float = 2.0,
    span_ms: float | None = None, points: int = 48, min_channels: int = 50, n_shuffle: int = 200, seed: int = 1,
) -> dict[str, np.ndarray]:
    """One burst as a shape: how much of the array it took, which way it went, how it rose and fell.

    Aligned with ``spans``. Per burst:

    - ``engaged``: the share of the array's channels (the rows of ``xy``) that fired in it;
    - ``heading_deg``: the direction it travelled -- the direction in which first spike time
      increases on a plane fitted to it against position -- in degrees from +x toward +y, in
      the electrodes' own coordinates, ``[0, 360)``. ``heading_r2`` is the share of first spike
      timing that plane explains, and ``heading_p`` how often a plane does at least as well when
      the times are dealt out again among the channels (``n_shuffle`` times; the smallest value
      it can take is ``1 / (n_shuffle + 1)``). A heading is only worth reading when
      ``heading_p`` is small: a plane always points somewhere. ``nan`` with fewer than
      ``min_channels`` channels;
    - ``peak_hz``: the peak of the population rate inside the burst, in spikes per second per
      channel of the array. The rate is counted in ``bin_ms`` bins from ``pre_ms`` before the
      onset to ``post_ms`` after the offset and smoothed over three bins (1-2-1);
    - ``rise_ms``: from the last time before the peak the rate was under 10% of it to the first
      time after that it reached 90%; ``fall_ms``: from the last time it was at 90% to the first
      time after the peak it was back under 10%. Both read between bins by straight lines, and
      are ``nan`` when the window never shows the rate under 10% on that side;
    - ``rise_hz_per_ms`` / ``fall_hz_per_ms``: the slopes those stand for, 80% of the peak over
      the time taken.

    And, for drawing every burst on one axis, ``envelope``: an ``(n, points)`` array of the
    population rate (same units, not smoothed) from ``pre_ms`` before each burst's onset to
    ``span_ms`` after it (default: the longest burst plus ``post_ms``), with
    ``envelope_ms``, the ``points + 1`` bin edges in ms from the onset. A short burst's row
    carries on past its offset into whatever followed it.
    """
    f = np.asarray(frames, dtype=np.int64)
    ch = np.asarray(rows, dtype=np.int64)
    pos = np.asarray(xy, dtype=float)
    hz = float(samp_rate)
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    n_chan = max(1, len(pos))
    rng = np.random.default_rng(seed)
    out = {k: np.full(len(spans), np.nan) for k in BURST_SHAPE}
    if span_ms is None:
        span_ms = (float((spans[:, 1] - spans[:, 0]).max()) * 1000.0 / hz if len(spans) else 0.0) + post_ms
    edges_ms = np.linspace(-pre_ms, span_ms, int(points) + 1)
    edges_fr = edges_ms * hz / 1000.0
    env_scale = 1.0 / n_chan / (np.diff(edges_ms) / 1000.0)
    envelope = np.zeros((len(spans), int(points)))
    per_bin = bin_ms * hz / 1000.0
    pre, post = round(pre_ms * hz / 1000.0), round(post_ms * hz / 1000.0)

    for k, (on, off) in enumerate(spans):
        envelope[k] = np.diff(np.searchsorted(f, on + edges_fr)) * env_scale

        # the rate, finely, and its rise and fall around the peak inside the burst
        t0 = on - pre
        a, b = np.searchsorted(f, [t0, off + post])
        n_bins = int(np.ceil((off + post - t0) / per_bin))
        rate = np.bincount(((f[a:b] - t0) / per_bin).astype(np.int64), minlength=n_bins)[:n_bins] / n_chan / (bin_ms / 1000.0)
        rate = np.convolve(rate, [0.25, 0.5, 0.25], mode="same")
        lo, hi = int(pre / per_bin), max(int(pre / per_bin) + 1, int(np.ceil((off - t0) / per_bin)))
        peak_index = lo + int(np.argmax(rate[lo:hi]))
        peak = float(rate[peak_index])
        if peak > 0:
            out["peak_hz"][k] = peak
            # The falling edge is the same measurement on the reversed trace.
            rising = rate[:peak_index + 1]
            falling = rate[peak_index:][::-1]
            for name, edge in (("rise", rising), ("fall", falling)):
                elapsed_ms = _rise_bins(edge, peak) * bin_ms
                out[f"{name}_ms"][k] = elapsed_ms
                if elapsed_ms > 0:
                    out[f"{name}_hz_per_ms"][k] = 0.8 * peak / elapsed_ms

        # how much of the array, and which way
        cc, first = _first_spikes(f, ch, on, off)
        out["engaged"][k] = cc.size / n_chan
        if cc.size < max(min_channels, 4):
            continue
        t = (first - first.min()) * 1000.0 / hz
        tc = t - t.mean()
        sst = float(tc @ tc)
        pc = pos[cc] - pos[cc].mean(0)
        if sst <= 0 or np.linalg.matrix_rank(pc) < 2:
            continue
        slope = np.linalg.lstsq(pc, tc, rcond=None)[0]  # ms per µm, along x and y
        q = np.linalg.qr(pc)[0]
        r2 = float(np.sum((q.T @ tc) ** 2) / sst)
        out["heading_deg"][k] = float(np.degrees(np.arctan2(slope[1], slope[0])) % 360.0)
        out["heading_r2"][k] = r2
        if n_shuffle:
            dealt = rng.permuted(np.tile(tc, (int(n_shuffle), 1)), axis=1)
            out["heading_p"][k] = (1 + int(np.sum(np.sum((dealt @ q) ** 2, axis=1) / sst >= r2))) / (n_shuffle + 1)
    out["envelope"] = envelope
    out["envelope_ms"] = edges_ms
    return out
