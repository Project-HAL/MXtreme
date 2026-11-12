"""Does activity propagate across the array, and on what time scale?

Read at the recording's own time resolution (one frame, 0.1 ms at 10 kHz), three ways:

- **spike scale** -- :func:`lag_by_distance`: every pair of spikes on different channels less
  than 20 ms apart, counted per lag and per distance between the two electrodes, each distance
  relative to its own level at lags of 2 to 3 ms. A conduction speed shows as an excess at
  short lags whose lag grows with distance.
- **burst scale, pooled** -- :func:`from_leader` and :func:`leader_profile`: each burst's
  *leader* is where and when its earliest channels fire; every other channel's first spike is
  placed by its distance from the leader and its delay after it. Pooled over bursts, the delay
  by which 10, 25 and 50% of the channels at a distance have fired moves out with distance at
  the speed of the front, and stays flat when the channels' positions are shuffled.
- **burst scale, one burst at a time** -- :func:`per_burst`: a plane fitted to first spike
  time against position (its R2 beside the R2 with the times shuffled), how long the channels
  take to come in, and how fast the population rate rises.
- **one burst as a shape** -- :func:`burst_shape`: how much of the array it engaged, which way
  it travelled (the plane's direction, with how often shuffled positions do as well), how fast
  the population rate rose and fell, and that rate's envelope on a grid every burst shares.

Everything takes the spike table as flat arrays -- ``frames`` (ascending), ``rows`` (each
spike's row in the channel map) and ``xy`` (the rows' positions in µm) -- which
:func:`spike_arrays` builds from a :class:`~mxtreme.recording.Recording`, and bursts as
``(n, 2)`` frame spans ``[start, stop)``, which :func:`burst_spans` builds from a burst table.
Times are frames in, milliseconds out.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Pairs of spikes are counted when the later is within this of the earlier.
MAX_LAG_MS = 20.0
#: Lags are resolved one time step at a time up to here.
FINE_MS = 3.0
#: The lags each distance's counts are divided by the mean of: where an excess has died away.
BASELINE_MS = (2.0, 3.0)
#: Distance between the two electrodes of a pair, 100 µm bins up to 1.5 mm.
D_EDGES_UM = np.arange(0.0, 1600.0, 100.0)

#: Distance from a burst's leader, 250 µm bins up to 3.25 mm.
LEADER_D_EDGES_UM = np.arange(0.0, 3251.0, 250.0)
#: Distance bands the share of channels fired is drawn for.
LEADER_BANDS_UM = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 3250.0))


# --- the inputs ---------------------------------------------------------------------------------


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
    by_chan = np.argsort(chans)
    idx = np.clip(np.searchsorted(chans[by_chan], chan), 0, max(len(chans) - 1, 0))
    known = chans[by_chan][idx] == chan if len(chans) else np.zeros(len(chan), dtype=bool)
    return frames[known], by_chan[idx][known].astype(np.int64), cm[:, 3:5].copy()


def burst_spans(bursts, *, kinds: Sequence[str] | None = ("network",), edges: str = "onset",
                min_ms: float = 0.0, samp_rate: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A burst table's bursts as ``(ids, spans)``: frame spans ``[start, stop)`` in time order.

    The windows are :func:`mxtreme.analysis.intervals.burst_windows`' (ignored bursts left out,
    onset to offset where feature extraction set them). ``min_ms`` drops shorter bursts and
    needs ``samp_rate``.
    """
    from mxtreme.analysis.intervals import burst_windows

    w = burst_windows(bursts, edges=edges, kinds=kinds)
    ids = w["id"].to_numpy()
    spans = np.column_stack([w["start_frame"].to_numpy(dtype=np.int64), w["end_frame"].to_numpy(dtype=np.int64) + 1]) if len(w) else np.zeros((0, 2), dtype=np.int64)
    if min_ms > 0 and len(spans):
        if samp_rate is None:
            raise ValueError("min_ms needs samp_rate")
        keep = (spans[:, 1] - spans[:, 0]) * 1000.0 / float(samp_rate) >= min_ms
        ids, spans = ids[keep], spans[keep]
    return ids, spans


def frame_step(frames, limit: int = 200_000) -> int:
    """The recording's real time step in frames: the greatest common divisor of the differences
    between its distinct spike frames (1 for a 10 kHz recording, 10 for spikes that sit on a
    1 ms grid at a nominal 10 kHz)."""
    f = np.unique(np.asarray(frames, dtype=np.int64))
    if f.size < 2:
        return 1
    return max(1, int(np.gcd.reduce(np.diff(f)[:limit])))


def first_spikes(frames, rows, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, first frame)`` of every channel that fires in ``[start, stop)``."""
    f = np.asarray(frames)
    a, b = np.searchsorted(f, [start, stop])
    cc, first = np.unique(np.asarray(rows)[a:b], return_index=True)
    return cc, f[a:b][first]


# --- spike scale --------------------------------------------------------------------------------


def lag_by_distance(
    frames,
    rows,
    xy,
    samp_rate: float,
    *,
    start: int | None = None,
    stop: int | None = None,
    max_lag_ms: float = MAX_LAG_MS,
    fine_ms: float = FINE_MS,
    baseline_ms: tuple[float, float] = BASELINE_MS,
    d_edges_um=D_EDGES_UM,
    step: int = 2,
    max_refs: int | None = None,
    step_frames: int | None = None,
    chunk: int = 4000,
) -> dict:
    """Pairs of spikes on different channels, the later within ``max_lag_ms`` of the earlier.

    Lags are binned on whole time steps of the recording (``step_frames``, by default
    :func:`frame_step`): bin edges at fractions of a millisecond alias against the frame grid
    and leave empty and doubled columns.

    :param start: First frame a reference spike may sit on (the whole recording by default).
    :param stop: One past the last. A reference's partners may lie past it.
    :param step: Every ``step``-th spike is a reference.
    :param max_refs: Thin the references further to at most this many, for a long recording.
    :returns: ``{"fine": counts, (n distances, n lags) with lag ``k`` in column ``k``;
        "ratio": each distance's row over its own mean at the baseline lags, so 1 is no excess;
        "far_per_frame": the mean count per frame at lags of 5 ms and more, per distance;
        "lag_ms": the columns' lags; "ms_per_bin"; "d_edges_um"; "n_refs"; "n_pairs"}``.
    """
    f = np.asarray(frames, dtype=np.int64)
    ch = np.asarray(rows, dtype=np.int64)
    pos = np.asarray(xy, dtype=float)
    hz = float(samp_rate)
    d_edges = np.asarray(d_edges_um, dtype=float)
    n_d = d_edges.size - 1
    unit = int(step_frames) if step_frames else frame_step(f)
    L = int(round(max_lag_ms * hz / 1000.0))
    lo5 = int(round(5.0 * hz / 1000.0))
    n_fine = max(1, int(round(fine_ms * hz / 1000.0 / unit)))
    a = 0 if start is None else int(np.searchsorted(f, start))
    b = f.size if stop is None else int(np.searchsorted(f, stop))
    ref = np.arange(a, b, max(1, int(step)))
    if max_refs is not None and ref.size > max_refs:
        ref = ref[:: int(np.ceil(ref.size / max_refs))]
    fine = np.zeros(n_d * n_fine)
    far = np.zeros(n_d)
    n_pairs = 0
    hi = np.searchsorted(f, f[ref] + L, side="right") if ref.size else np.zeros(0, dtype=np.int64)
    for s in range(0, ref.size, chunk):
        r, h = ref[s:s + chunk], hi[s:s + chunk]
        n = h - r - 1
        total = int(n.sum())
        if total <= 0:
            continue
        i = np.repeat(r, n)
        j = i + 1 + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
        keep = ch[i] != ch[j]
        i, j = i[keep], j[keep]
        lag = f[j] - f[i]
        d = np.digitize(np.hypot(*(pos[ch[i]] - pos[ch[j]]).T), d_edges) - 1
        ok = (d >= 0) & (d < n_d)
        lag, d = lag[ok], d[ok]
        n_pairs += int(lag.size)
        col = lag // unit
        m = col < n_fine
        fine += np.bincount(d[m] * n_fine + col[m], minlength=n_d * n_fine)
        far += np.bincount(d[lag >= lo5], minlength=n_d)
    fine = fine.reshape(n_d, n_fine)
    ms_per_bin = unit * 1000.0 / hz
    lag_ms = np.arange(n_fine) * ms_per_bin
    base = (lag_ms >= baseline_ms[0]) & (lag_ms < baseline_ms[1])
    if not base.any():
        base[-1] = True
    level = np.maximum(fine[:, base].mean(axis=1), 1e-9)
    return {
        "fine": fine, "ratio": fine / level[:, None], "far_per_frame": far / max(L - lo5 + 1, 1),
        "lag_ms": lag_ms, "ms_per_bin": ms_per_bin, "d_edges_um": d_edges, "n_refs": int(ref.size), "n_pairs": n_pairs,
    }


# --- one burst at a time ------------------------------------------------------------------------

#: The columns :func:`per_burst` returns, one value per burst.
PER_BURST = ("channels", "plane_r2", "plane_r2_shuffled", "speed_m_s", "radial_rho", "t10_50_ms", "t10_90_ms",
             "early_sd_um", "rise_10_90_ms")


def rise_time_ms(frames, samp_rate: float, start: int, stop: int, *, pre_ms: float = 10.0, bin_ms: float = 1.0) -> float:
    """How long the population rate takes to go from 10 to 90% of its peak in one burst.

    Spikes per ``bin_ms`` over every channel, from ``pre_ms`` before the burst's onset to its
    end. The 90% point is the first bin at or above 0.9 of the peak bin; the 10% point is the
    end of the last bin before it that was below 0.1 of the peak. ``nan`` when the burst holds
    no spike.
    """
    f = np.asarray(frames)
    hz = float(samp_rate)
    t0 = start - int(round(pre_ms * hz / 1000.0))
    a, b = np.searchsorted(f, [t0, stop])
    if b <= a:
        return float("nan")
    bins = ((f[a:b] - t0) * 1000.0 / hz / bin_ms).astype(np.int64)
    rate = np.bincount(bins)
    peak = rate.max()
    i90 = int(np.argmax(rate >= 0.9 * peak))
    below = np.flatnonzero(rate[:i90] < 0.1 * peak)
    i10 = int(below[-1]) + 1 if below.size else 0
    return float((i90 - i10) * bin_ms)


def per_burst(frames, rows, xy, samp_rate: float, spans, *, min_channels: int = 50, n_shuffle: int = 20, seed: int = 1) -> dict[str, np.ndarray]:
    """Per burst measures of how its channels come in, aligned with ``spans``.

    For each burst, from each channel's first spike in it (ms from the earliest):

    - ``plane_r2``: the share of first spike timing a plane across the array explains, and
      ``plane_r2_shuffled``: the same with the times dealt out again among the channels (the
      mean of ``n_shuffle`` shuffles) -- what a plane explains when position means nothing;
    - ``speed_m_s``: the speed the plane's slope stands for (meaningful when ``plane_r2`` is);
    - ``radial_rho``: the rank correlation of first spike time with distance from the earliest
      channels (the earliest twentieth, at least five);
    - ``t10_50_ms`` / ``t10_90_ms``: from the time 10% of the channels have fired to 50% / 90%;
    - ``early_sd_um``: how spread out those earliest channels are;
    - ``rise_10_90_ms``: :func:`rise_time_ms`;
    - ``channels``: how many channels fired in the burst.

    A burst with fewer than ``min_channels`` channels has its count and rise time and ``nan``
    elsewhere.
    """
    from scipy.stats import spearmanr

    f = np.asarray(frames, dtype=np.int64)
    ch = np.asarray(rows, dtype=np.int64)
    pos = np.asarray(xy, dtype=float)
    hz = float(samp_rate)
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    out = {k: np.full(len(spans), np.nan) for k in PER_BURST}

    def r2(A, tt):
        sst = np.sum((tt - tt.mean()) ** 2)
        if sst <= 0:
            return float("nan")
        return float(1 - np.sum((tt - A @ np.linalg.lstsq(A, tt, rcond=None)[0]) ** 2) / sst)

    for k, (on, off) in enumerate(spans):
        cc, first = first_spikes(f, ch, on, off)
        out["channels"][k] = cc.size
        out["rise_10_90_ms"][k] = rise_time_ms(f, hz, on, off)
        if cc.size < max(min_channels, 4):
            continue
        t = (first - first.min()) * 1000.0 / hz
        p = pos[cc]
        A = np.c_[np.ones(cc.size), p]
        coef = np.linalg.lstsq(A, t, rcond=None)[0]
        early = np.argsort(t, kind="stable")[:max(5, cc.size // 20)]
        dist = np.hypot(*(p - p[early].mean(0)).T)
        q = np.percentile(t, [10, 50, 90])
        out["plane_r2"][k] = r2(A, t)
        out["plane_r2_shuffled"][k] = float(np.mean([r2(A, rng.permutation(t)) for _ in range(n_shuffle)])) if n_shuffle else np.nan
        out["speed_m_s"][k] = 1.0 / max(np.hypot(coef[1], coef[2]), 1e-12) / 1000.0
        out["radial_rho"][k] = spearmanr(dist, t)[0] if np.ptp(t) > 0 and np.ptp(dist) > 0 else np.nan
        out["t10_50_ms"][k] = q[1] - q[0]
        out["t10_90_ms"][k] = q[2] - q[0]
        out["early_sd_um"][k] = np.hypot(*p[early].std(0))
    return out


#: What :func:`burst_shape` reports per burst, beside the envelope.
BURST_SHAPE = (
    "engaged", "heading_deg", "heading_r2", "heading_p", "peak_hz", "rise_ms", "fall_ms", "rise_hz_per_ms", "fall_hz_per_ms",
)


def _crossing(rate, i: int, j: int, level: float) -> float:
    """Where, between bins ``i`` and ``j`` (neighbours), a straight line through their rates
    meets ``level``, in bins."""
    a, b = float(rate[i]), float(rate[j])
    return float(i) if a == b else i + (level - a) / (b - a) * (j - i)


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
    pre, post = int(round(pre_ms * hz / 1000.0)), int(round(post_ms * hz / 1000.0))

    for k, (on, off) in enumerate(spans):
        envelope[k] = np.diff(np.searchsorted(f, on + edges_fr)) * env_scale

        # the rate, finely, and its rise and fall around the peak inside the burst
        t0 = on - pre
        a, b = np.searchsorted(f, [t0, off + post])
        n_bins = int(np.ceil((off + post - t0) / per_bin))
        rate = np.bincount(((f[a:b] - t0) / per_bin).astype(np.int64), minlength=n_bins)[:n_bins] / n_chan / (bin_ms / 1000.0)
        rate = np.convolve(rate, [0.25, 0.5, 0.25], mode="same")
        lo, hi = int(pre / per_bin), max(int(pre / per_bin) + 1, int(np.ceil((off - t0) / per_bin)))
        if rate[lo:hi].max() > 0:
            ip = lo + int(np.argmax(rate[lo:hi]))
            peak = float(rate[ip])
            out["peak_hz"][k] = peak
            under = np.flatnonzero(rate[:ip + 1] < 0.1 * peak)
            if under.size:
                i10 = int(under[-1])
                i90 = i10 + 1 + int(np.argmax(rate[i10 + 1:ip + 1] >= 0.9 * peak))
                out["rise_ms"][k] = (_crossing(rate, i90 - 1, i90, 0.9 * peak) - _crossing(rate, i10, i10 + 1, 0.1 * peak)) * bin_ms
            under = np.flatnonzero(rate[ip:] < 0.1 * peak)
            if under.size:
                j10 = ip + int(under[0])
                j90 = int(np.flatnonzero(rate[ip:j10] >= 0.9 * peak)[-1]) + ip
                out["fall_ms"][k] = (_crossing(rate, j10 - 1, j10, 0.1 * peak) - _crossing(rate, j90, j90 + 1, 0.9 * peak)) * bin_ms
            for name in ("rise", "fall"):
                ms = out[f"{name}_ms"][k]
                if ms > 0:
                    out[f"{name}_hz_per_ms"][k] = 0.8 * peak / ms

        # how much of the array, and which way
        cc, first = first_spikes(f, ch, on, off)
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


# --- from the leader, pooled over bursts ----------------------------------------------------------


def from_leader(frames, rows, xy, samp_rate: float, spans, *, n_lead: int = 5, pre_ms: float = 10.0,
                min_channels: int = 50, shuffle: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Each channel's distance from its burst's leader and its delay after it, pooled over bursts.

    A burst's leader is where and when its ``n_lead`` earliest channels fire: the median of
    their positions and of their first spike times, first spikes counted from ``pre_ms`` before
    the burst's onset. Returned for every other channel of every burst with at least
    ``min_channels`` channels: its distance from the leader (µm) and the delay of its first
    spike after the leader's (ms).

    ``shuffle`` deals the positions out again among the burst's channels first: what the same
    construction gives when where a channel sits has nothing to do with when it fires.
    """
    f = np.asarray(frames, dtype=np.int64)
    ch = np.asarray(rows, dtype=np.int64)
    pos = np.asarray(xy, dtype=float)
    hz = float(samp_rate)
    pre = int(round(pre_ms * hz / 1000.0))
    rng = np.random.default_rng(seed)
    D, T = [], []
    for on, off in np.asarray(spans, dtype=np.int64).reshape(-1, 2):
        cc, first = first_spikes(f, ch, on - pre, off)
        if cc.size < max(min_channels, n_lead + 1):
            continue
        t = (first - first.min()) * 1000.0 / hz
        p = pos[cc]
        if shuffle:
            p = p[rng.permutation(cc.size)]
        lead = np.argsort(t, kind="stable")[:n_lead]
        rest = np.setdiff1d(np.arange(cc.size), lead)
        D.append(np.hypot(*(p[rest] - np.median(p[lead], axis=0)).T))
        T.append(t[rest] - np.median(t[lead]))
    if not D:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(D), np.concatenate(T)


def leader_profile(
    D,
    T,
    D_shuffled=None,
    T_shuffled=None,
    *,
    d_edges_um=LEADER_D_EDGES_UM,
    quantiles: Sequence[float] = (10, 25, 50),
    min_count: int = 30,
    bands_um=LEADER_BANDS_UM,
    t_grid_ms=None,
    slopes: Sequence[float] = (10, 25),
) -> dict:
    """The recruitment front, from :func:`from_leader`'s pooled distances and delays.

    :returns: ``{"d_mid_mm": the distance bins' centres; "count": channels per bin;
        "q": {quantile: the delay by which that share of the channels at each distance have
        fired, ``nan`` for a bin with ``min_count`` channels or fewer};
        "q25_shuffled": the 25% point of the shuffled pairs (flat when there is no front);
        "speed": {quantile: {"ms_per_mm": slope of that point against distance,
        "m_per_s": its inverse, "at_leader_ms": the point in the nearest bin}};
        "bands": [{"lo_um", "hi_um", "n", "fired": share of the band's channels fired by each
        time of "t_grid_ms"}]}``.
    """
    D = np.asarray(D, dtype=float)
    T = np.asarray(T, dtype=float)
    d_edges = np.asarray(d_edges_um, dtype=float)
    d_mid = (d_edges[:-1] + d_edges[1:]) / 2000.0
    grid = np.arange(0.0, 80.25, 0.25) if t_grid_ms is None else np.asarray(t_grid_ms, dtype=float)
    which = np.digitize(D, d_edges) - 1

    def points(dd_which, tt, q):
        return np.array([np.percentile(tt[dd_which == k], q) if (dd_which == k).sum() > min_count else np.nan for k in range(d_mid.size)])

    q = {float(qq): points(which, T, qq) for qq in quantiles}
    out = {
        "d_mid_mm": d_mid, "count": np.bincount(which[(which >= 0) & (which < d_mid.size)], minlength=d_mid.size),
        "q": q, "q25_shuffled": None, "speed": {}, "bands": [], "t_grid_ms": grid, "n": int(D.size),
    }
    if D_shuffled is not None and T_shuffled is not None:
        ws = np.digitize(np.asarray(D_shuffled, dtype=float), d_edges) - 1
        out["q25_shuffled"] = points(ws, np.asarray(T_shuffled, dtype=float), 25)
    for qq in slopes:
        y = q.get(float(qq))
        if y is None:
            y = points(which, T, qq)
        ok = np.isfinite(y)
        if ok.sum() >= 2:
            slope = float(np.polyfit(d_mid[ok], y[ok], 1)[0])
            out["speed"][float(qq)] = {"ms_per_mm": slope, "m_per_s": (1.0 / slope) if slope > 0 else float("nan"), "at_leader_ms": float(y[ok][0])}
    for lo, hi in bands_um:
        m = (D >= lo) & (D < hi)
        fired = np.searchsorted(np.sort(T[m]), grid, side="right") / m.sum() if m.sum() else np.full(grid.size, np.nan)
        out["bands"].append({"lo_um": float(lo), "hi_um": float(hi), "n": int(m.sum()), "fired": fired})
    return out
