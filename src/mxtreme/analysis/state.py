"""The state space of a recording's dynamics: where the array is, moment by moment, and the
loops its bursts trace through that space.

A *state* is the vector of firing rates over the recording's units -- its channels, or regions
of the array (:mod:`mxtreme.analysis.regions`), which make the same space on another day with
other electrodes routed -- in one short time bin, smoothed with the double-exponential kernel the
network analyses use (:func:`mxtreme.analysis.activity.smoothed_firing_rate`), log-compressed and
standardised per unit. :func:`state_vectors` builds that matrix from the spike table for any
window at any bin size, without the recording's binned matrix, so it works on an hour-long
recording loaded without one. :func:`trajectory` is its PCA, through the same decomposition as
:func:`mxtreme.analysis.network.pca`, so the leading axis is the array-wide burst mode with "up"
meaning more activity.

What that trajectory looks like for a bursting culture is a bouquet: a dense quiet state with one
excursion per burst that leaves it and comes back. :func:`burst_loops` cuts those excursions out,
each resampled onto the same time grid around its onset, so they can be drawn over each other and
compared; :func:`loop_distances` and :func:`loop_clusters` say which of them follow the same path
-- the honest version of "families of bursts".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mxtreme.analysis.activity import _double_exponential
from mxtreme.analysis.network import _effective_rank, _pca

#: Smoothing fast enough to keep the shape of a burst: a 5 ms rise and a 50 ms decay, against
#: the 50 ms / 2 s the connectivity analyses use for slow covariation.
TAU_RISE_SEC = 0.005
TAU_DECAY_SEC = 0.05


def state_vectors(
    rec,
    *,
    start_frame: int,
    end_frame: int,
    bin_sec: float = 0.01,
    groups: Mapping[str, Sequence[int]] | None = None,
    tau_rise_sec: float = TAU_RISE_SEC,
    tau_decay_sec: float = TAU_DECAY_SEC,
    transform: str = "log1p",
    standardize: bool = True,
    max_bins: int | None = None,
) -> dict:
    """The recording's state over a window, one vector per time bin.

    :param rec: The recording (``spike_data`` and ``channelmap`` are read; ``spike_bin`` is not).
    :param start_frame: First frame of the window (inclusive).
    :param end_frame: Last frame of the window (inclusive).
    :param bin_sec: Width of a time bin. Keep it well under a burst's duration (10 ms by
        default) or the structure inside a burst is averaged away.
    :param groups: ``{unit id: channel ids}`` to pool channels into units -- regions, from
        :func:`mxtreme.analysis.regions.membership`; each unit's rate is per electrode. ``None``
        makes every mapped channel a unit of its own. A group with no channel in this recording
        is kept as a unit that is always zero, so two recordings aggregate to the same units.
    :param tau_rise_sec: Rise of the double-exponential smoothing kernel.
    :param tau_decay_sec: Its decay; the time scale a state is an average over.
    :param transform: ``"log1p"`` compresses the rates so a burst does not own every axis;
        ``"none"`` keeps them in Hz.
    :param standardize: Centre and scale each unit to unit variance (a silent unit stays zero),
        so a region of three channels weighs as much as one of thirty.
    :param max_bins: Coarsen ``bin_sec`` as needed so the window has at most this many bins.
    :returns: ``{"time_sec": bin centres on the recording's clock, "X": (n_units, n_bins) float32,
        "units": the unit ids, "bin_sec": the bin actually used, "n_channels": channels per unit}``.
    """
    if transform not in ("log1p", "none"):
        raise ValueError(f"transform must be 'log1p' or 'none', not {transform!r}")
    samp_rate = float(rec.samp_rate)
    start_frame, end_frame = int(start_frame), int(end_frame)
    if end_frame < start_frame:
        raise ValueError("end_frame must not be before start_frame")
    span_sec = (end_frame - start_frame + 1) / samp_rate
    bin_sec = float(bin_sec)
    if max_bins is not None and span_sec / bin_sec > max_bins:
        bin_sec = span_sec / max_bins
    n_bins = max(1, int(np.ceil(span_sec / bin_sec)))

    cm = np.asarray(rec.channelmap)
    chans = cm[:, 1].astype(np.int64)
    if groups is None:
        units = [int(c) for c in chans]
        unit_of = {int(c): i for i, c in enumerate(chans)}
        per_unit = np.ones(len(units), dtype=float)
    else:
        units = list(groups)
        unit_of = {}
        per_unit = np.zeros(len(units), dtype=float)
        known = {int(c) for c in chans}
        for i, uid in enumerate(units):
            for c in groups[uid]:
                if int(c) in known and int(c) not in unit_of:
                    unit_of[int(c)] = i
                    per_unit[i] += 1
    frames = np.asarray(rec.spike_data["frameno"], dtype=np.int64)
    i0, i1 = np.searchsorted(frames, start_frame), np.searchsorted(frames, end_frame + 1)
    spike_chans = np.asarray(rec.spike_data["channel"][i0:i1], dtype=np.int64)
    unit_idx = np.fromiter((unit_of.get(int(c), -1) for c in spike_chans), dtype=np.int64, count=len(spike_chans))
    keep = unit_idx >= 0
    bins = np.minimum(((frames[i0:i1][keep] - start_frame) / samp_rate / bin_sec).astype(np.int64), n_bins - 1)
    counts = np.bincount(unit_idx[keep] * n_bins + bins, minlength=len(units) * n_bins).reshape(len(units), n_bins).astype(np.float32)
    # Spikes per second per electrode, then the kernel, then the compression and the scaling.
    rates = counts / (bin_sec * np.maximum(per_unit, 1.0)[:, None])
    X = _double_exponential(rates, bin_sec, tau_rise_sec, tau_decay_sec, bin_sec) if len(units) else rates
    if transform == "log1p":
        X = np.log1p(np.maximum(X, 0.0))
    if standardize:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        X = np.where(sd > 1e-9, (X - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0).astype(np.float32)
    time_sec = (start_frame / samp_rate) + (np.arange(n_bins) + 0.5) * bin_sec
    return {"time_sec": time_sec, "X": np.asarray(X, dtype=np.float32), "units": units, "bin_sec": bin_sec, "n_channels": per_unit.astype(int).tolist()}


def trajectory(X, n_components: int = 3) -> dict:
    """PCA of a state matrix: the path through its leading components.

    :param X: ``(n_units, n_bins)`` from :func:`state_vectors`.
    :returns: ``{"scores": (n_bins, k) float32, "components": (n_units, k), "var_frac": the whole
        spectrum, "effective_rank": float}`` -- the same decomposition, sign convention and rank
        as :func:`mxtreme.analysis.network.pca`.
    """
    p, components, scores = _pca(np.asarray(X, dtype=np.float32), n_components)
    return {"scores": scores.T.astype(np.float32), "components": components, "var_frac": p, "effective_rank": _effective_rank(p)}


def burst_loops(scores, time_sec, onsets_sec, *, pre_sec: float = 0.05, post_sec: float = 0.5, n_points: int = 50) -> dict:
    """Each burst's excursion through state space, on one time grid around its onset.

    :param scores: ``(n_bins, k)`` from :func:`trajectory`.
    :param time_sec: The bins' times.
    :param onsets_sec: One onset per burst.
    :param pre_sec: How far before the onset the loop starts.
    :param post_sec: How far after the onset it ends; past the typical burst, so the return to
        the quiet state is part of the loop.
    :param n_points: Points the loop is resampled to, by linear interpolation in time.
    :returns: ``{"loops": (n_kept, n_points, k) float32, "t_rel": the grid in seconds from onset,
        "kept": indices into onsets_sec of the bursts whose window lay inside the data}``.
    """
    scores = np.asarray(scores, dtype=np.float32)
    time_sec = np.asarray(time_sec, dtype=float)
    grid = np.linspace(-pre_sec, post_sec, int(n_points))
    loops, kept = [], []
    if scores.ndim != 2 or len(time_sec) < 2:
        return {"loops": np.zeros((0, len(grid), scores.shape[1] if scores.ndim == 2 else 0), dtype=np.float32), "t_rel": grid, "kept": []}
    for i, onset in enumerate(onsets_sec):
        t = float(onset) + grid
        if t[0] < time_sec[0] or t[-1] > time_sec[-1]:
            continue
        loops.append(np.column_stack([np.interp(t, time_sec, scores[:, k]) for k in range(scores.shape[1])]))
        kept.append(i)
    out = np.asarray(loops, dtype=np.float32) if loops else np.zeros((0, len(grid), scores.shape[1]), dtype=np.float32)
    return {"loops": out, "t_rel": grid, "kept": kept}


def loop_distances(loops) -> np.ndarray:
    """Mean point-to-point Euclidean distance between every pair of loops, ``(n, n)``.

    Loops are on one time grid from the onset, so this is a distance between *paths*: two
    bursts that spread the same way from the same place are close, whatever their size.
    """
    L = np.asarray(loops, dtype=np.float64)
    n = L.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    flat = L.reshape(n, -1)
    sq = (flat ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0)
    np.fill_diagonal(d2, 0.0)  # the expansion leaves rounding noise where it should be exactly zero
    return np.sqrt(d2 / L.shape[1])


def loop_clusters(distances, n_clusters: int) -> np.ndarray:
    """Average-linkage clustering of the loops into ``n_clusters`` families, labels from 1."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = np.asarray(distances, dtype=float)
    n = D.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1 or n_clusters <= 1:
        return np.ones(n, dtype=int)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=min(int(n_clusters), n), criterion="maxclust").astype(int)
