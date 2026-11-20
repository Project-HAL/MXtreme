"""Composable preprocessing steps -- the menu a user assembles into a :class:`mxtreme.pipeline.Pipeline`.

Each function takes a single well's data dict (as produced by :func:`mxtreme.extract.extract`), mutates
it in place, and returns it, so steps chain naturally::

    from functools import partial
    from mxtreme import clean
    from mxtreme.pipeline import Pipeline

    pipeline = Pipeline([
        clean.normalize_time,
        clean.dac_to_voltage,
        partial(clean.spike_filter, amp_thresh=2e-5),
        ...
    ])

Parameters live on the step functions themselves (with sensible defaults), not in a central params
object: pass a bare function to accept the defaults, or wrap it with :func:`functools.partial` to
override. Every parameterised step records the values it used in ``well['preprocessing_params']`` so the
saved ``.npz`` carries a record of exactly how it was processed.

Default values mirror the historical ``constants.py`` hyperparameters:
``amp_thresh=2e-5`` V, ``refractory_period=0.002`` s, ``post_stim_period=0.0`` s, ``bin_size=0.01`` s.
"""

from __future__ import annotations

import logging

import numpy as np

from mxtreme.utils import _spike_bins, channel_rows

logger = logging.getLogger(__name__)


def _stamp(well: dict, **params) -> None:
    """Record the parameters a step used into ``well['preprocessing_params']`` (provenance)."""
    well.setdefault("preprocessing_params", {}).update(params)


def _already(well: dict, flag: str, step: str) -> bool:
    """True if ``flag`` is set in ``well['preprocessing_params']`` -- i.e. ``step`` has already been
    applied to this well, so it must not run again."""
    if well.get("preprocessing_params", {}).get(flag):
        logger.info("%s already applied to this well; skipping.", step)
        return True
    return False


def normalize_time(well: dict) -> dict:
    """Offset spike and event frames so the recording starts at frame 0.

    Subtracts the raw recording's first frame (``well['raw_start']``) from every spike and event frame.
    Records ``time_normalized=True`` in ``well['preprocessing_params']``; a well that already carries
    it is left unchanged, so frames are never shifted twice.

    :param well: A well data dict.
    :returns: The same dict, with normalized ``spike_data['frameno']`` and ``eventtime``.
    :rtype: dict
    """
    if _already(well, "time_normalized", "normalize_time"):
        return well
    well["spike_data"]["frameno"] -= well["raw_start"]
    well["eventtime"] = well["eventtime"] - well["raw_start"]
    _stamp(well, time_normalized=True)
    return well


def dac_to_voltage(well: dict) -> dict:
    """Convert spike amplitudes from DAC units to volts by multiplying by the least significant bit.

    Records ``amplitude_units="V"`` in ``well['preprocessing_params']``; a well already in volts is left
    unchanged, so amplitudes are never scaled twice.

    :param well: A well data dict.
    :returns: The same dict, with ``spike_data['amplitude']`` expressed in volts.
    :rtype: dict
    """
    if _already(well, "amplitude_units", "dac_to_voltage"):
        return well
    well["spike_data"]["amplitude"] = well["spike_data"]["amplitude"] * well["lsb"]
    _stamp(well, amplitude_units="V")
    return well


def remove_positive_deflections(well: dict) -> dict:
    """Keep only negative-going spikes (drop positive deflections).

    :param well: A well data dict.
    :returns: The same dict, filtered to negative-amplitude spikes.
    :rtype: dict
    """
    well["spike_data"] = well["spike_data"][well["spike_data"]["amplitude"] < 0]
    return well


def spike_filter(well: dict, amp_thresh: float = 2e-5) -> dict:
    """Drop low-amplitude spikes below an absolute voltage threshold (denoising).

    :param well: A well data dict (amplitudes must already be in volts -- see :func:`dac_to_voltage`).
    :param amp_thresh: Minimum absolute spike amplitude to keep, in volts. Defaults to ``2e-5`` (20 µV).
    :type amp_thresh: float
    :raises ValueError: If the threshold removes every spike.
    :returns: The same dict, filtered by amplitude.
    :rtype: dict
    """
    mask = np.abs(well["spike_data"]["amplitude"]) >= amp_thresh
    well["spike_data"] = well["spike_data"][mask]
    if len(well["spike_data"]) == 0:
        raise ValueError("Amplitude threshold is too high.")
    _stamp(well, amp_thresh=amp_thresh)
    return well


def remove_spurious_spikes(well: dict, refractory_period: float = 0.002) -> dict:
    """Remove spikes that fall within a refractory window of another spike on the same channel.

    For each channel, when two spikes are closer than ``refractory_period`` seconds the smaller-amplitude
    spike is kept (the more negative one is discarded), repeated until no violations remain.

    :param well: A well data dict.
    :param refractory_period: Minimum time between spikes on a channel, in seconds. Defaults to ``0.002``.
    :type refractory_period: float
    :returns: The same dict, with within-refractory spikes removed.
    :rtype: dict
    """
    refractory_frames = refractory_period * well["samp_rate"]
    data = well["spike_data"]
    n = len(data)

    # Sort by (channel, frame) so refractory violations are adjacent within each channel. Work on a
    # boolean "alive" mask over the sorted positions and resolve violations in vectorized passes: this
    # reproduces the original per-channel greedy exactly (remove the weaker/less-negative of each adjacent
    # violating pair, ties drop the earlier spike) but without pandas or a Python-per-spike loop.
    order = np.lexsort((data["frameno"], data["channel"]))
    ch = data["channel"][order]
    fr = data["frameno"][order]
    amp = data["amplitude"][order]

    alive = np.ones(n, dtype=bool)
    while True:
        idx = np.flatnonzero(alive)
        if idx.size < 2:
            break
        a, b = idx[:-1], idx[1:]                        # consecutive alive spikes
        viol = (ch[a] == ch[b]) & ((fr[b] - fr[a]) < refractory_frames)
        if not viol.any():
            break
        a, b = a[viol], b[viol]
        # keep the more-negative (larger-magnitude) spike; on ties (amp[a] not < amp[b]) drop the earlier
        alive[np.where(amp[a] < amp[b], b, a)] = False

    keep = np.zeros(n, dtype=bool)
    keep[order[alive]] = True                           # map back to original ordering
    well["spike_data"] = data[keep]
    _stamp(well, refractory_period=refractory_period)
    return well


def remove_spurious_channels(well: dict) -> dict:
    """Drop spikes recorded on channels absent from the config channel map.

    Data is recorded from all channels regardless of the config, but only channels in ``well['mapping']``
    have a known electrode. Spikes on any other channel are removed.

    :param well: A well data dict.
    :returns: The same dict, filtered to mapped channels.
    :rtype: dict
    """
    if len(well["mapping"]["channel"]) != len(np.unique(well["mapping"]["channel"])):
        logger.warning("Duplicate channels found.")
    mask = np.isin(well["spike_data"]["channel"], well["mapping"]["channel"])
    well["spike_data"] = well["spike_data"][mask]
    return well


def build_channel_map(well: dict) -> dict:
    """Build the ``(num_channels, 5)`` channel map used for binning and plotting.

    Columns are ``(index, channel, electrode, x, y)`` with positions in micrometres. The ``index`` column
    is the row a channel occupies in the binned spike matrix.

    :param well: A well data dict.
    :returns: The same dict, with a ``channelmap`` array added.
    :rtype: dict
    """
    num_chan = len(well["mapping"]["channel"])
    channelmap = np.zeros((num_chan, 5))
    channelmap[:, 0] = np.arange(0, num_chan)
    channelmap[:, 1] = well["mapping"]["channel"]
    channelmap[:, 2] = well["mapping"]["electrode"]
    channelmap[:, 3] = well["mapping"]["x"]
    channelmap[:, 4] = well["mapping"]["y"]
    well["channelmap"] = channelmap
    return well


def _stimulation_bounds(well: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pair stimulation markers, ignoring a stop from before the recording."""
    times = np.asarray(well["eventtime"], dtype=np.int64)
    messages = well["event_messages"]
    starts = times[["start_stimulation" in message for message in messages]]
    stops = times[["end_stimulation" in message for message in messages]]
    if starts.size:
        stops = stops[stops >= starts[0]]
        if starts.size != stops.size or np.any(stops < starts):
            raise ValueError("Unpaired stimulation start and end markers")
    return starts, stops


def _blank_frames(well: dict, frames, **params) -> dict:
    well["stim_frames"] = frames
    if len(frames):
        spikes = well["spike_data"]
        well["spike_data"] = spikes[~np.isin(spikes["frameno"], frames)]
    _stamp(well, **params)
    return well


def remove_stim_frames(well: dict, post_stim_period: float = 0.0) -> dict:
    """Remove spikes that occur during stimulation (plus a post-stimulation artefact window).

    Stimulation intervals are read from the ``start_stimulation`` / ``end_stimulation`` event markers.
    Spikes inside any interval -- extended by ``post_stim_period`` seconds past each end -- are dropped.

    :param well: A well data dict.
    :param post_stim_period: Extra time after each stimulation to discard, in seconds. Defaults to ``0.0``.
    :type post_stim_period: float
    :returns: The same dict, with stimulation-window spikes removed and ``stim_frames`` recorded.
    :rtype: dict
    """
    starts, stops = _stimulation_bounds(well)
    post_stim_frames = post_stim_period * well["samp_rate"]
    intervals = [np.arange(start, stop + post_stim_frames, dtype=np.int64)
                 for start, stop in zip(starts, stops)]
    frames = np.concatenate(intervals) if intervals else []
    return _blank_frames(well, frames, post_stim_period=post_stim_period)


def blank_stim_pulses(well: dict, blank_ms: float = 5.0, min_fraction: float = 0.05, search_ms: float = 10.0) -> dict:
    """Remove the spikes around each stimulation *pulse*, keeping the activity between pulses.

    :func:`remove_stim_frames` drops everything between a ``start_stimulation`` and its
    ``end_stimulation`` marker. That is right when a stimulus is one pulse, but when it is a train
    (three pulses at 20 Hz, 100 ms between the markers) it also drops the culture's response
    between the pulses. Here the pulses are found from their artifact: within each marked window
    (widened by ``search_ms`` either side), any 1 ms bin in which at least ``min_fraction`` of the
    recorded channels fire is a pulse. Spikes within ``blank_ms`` of a pulse, or of the start
    marker itself (a weak pulse may leave no artifact to find), are removed.

    :param well: A well data dict after :func:`normalize_time` and :func:`build_channel_map`.
    :param blank_ms: Half-width of the blanked stretch around each pulse, in ms. Defaults to ``5.0``.
    :param min_fraction: Share of recorded channels firing in one 1 ms bin that marks a pulse.
        Defaults to ``0.05``.
    :param search_ms: How far outside the markers to look for pulses, in ms. Defaults to ``10.0``.
    :returns: The same dict, with pulse spikes removed and the blanked frames in ``stim_frames``.
    :rtype: dict
    """
    sr = float(np.asarray(well["samp_rate"]).ravel()[0])
    frames = np.asarray(well["spike_data"]["frameno"], dtype=np.int64)
    params = {
        "stim_blanking": "pulses",
        "blank_ms": blank_ms,
        "pulse_min_fraction": min_fraction,
        "pulse_search_ms": search_ms,
    }
    starts, stops = _stimulation_bounds(well)
    if not starts.size:
        return _blank_frames(well, [], **params)

    spikes = well["spike_data"]
    channelmap = well["channelmap"]
    rows = channel_rows(spikes["channel"], channelmap)
    order = np.argsort(frames, kind="stable")
    order = order[rows[order] >= 0]
    sorted_frames, sorted_rows = frames[order], rows[order]
    n_channels = max(1, len(channelmap))
    per_ms, half, pad = sr / 1000.0, round(blank_ms * sr / 1000.0), round(search_ms * sr / 1000.0)
    spans = []
    for a, b in zip(starts, stops):
        lo, hi = a - pad, b + pad
        first = np.searchsorted(sorted_frames, lo)
        last = np.searchsorted(sorted_frames, hi, side="right")
        seg = sorted_frames[first:last]
        pulses = [int(a)]
        if seg.size:
            bins = ((seg - lo) // per_ms).astype(np.int64)
            unique = np.unique(bins * n_channels + sorted_rows[first:last])
            counts = np.bincount(unique // n_channels)
            for k in np.nonzero(counts >= min_fraction * n_channels)[0]:
                pulses.append(int(lo + (k + 0.5) * per_ms))
        spans.extend((p - half, p + half) for p in pulses)
    blanked = np.unique(np.concatenate([np.arange(x, y + 1, dtype=np.int64) for x, y in spans]))
    return _blank_frames(well, blanked, **params)


def bin_spikes(well: dict, bin_size: float = 0.01) -> dict:
    """Bin spikes into a ``(num_channels, num_bins)`` binary matrix.

    Also records the recording length in seconds (``rec_t_sec``) from the last spike frame. Requires a
    channel map (run :func:`build_channel_map` first).

    :param well: A well data dict.
    :param bin_size: Bin width in seconds. Defaults to ``0.01``.
    :type bin_size: float
    :returns: The same dict, with ``spike_bin``, ``bin_size`` and ``rec_t_sec`` added.
    :rtype: dict
    """
    bin_win = bin_size * well["samp_rate"]  # bin width in frames
    rec_t_samp = np.max(well["spike_data"]["frameno"])
    which_bin, num_bins = _spike_bins(np.asarray(well["spike_data"]["frameno"]), bin_win)
    channelmap = well["channelmap"]
    spike_bin = np.zeros((len(channelmap), num_bins), dtype=np.uint8)
    rows = channel_rows(well["spike_data"]["channel"], channelmap)
    if np.any(rows < 0):
        raise ValueError("Cannot bin spikes on channels absent from the channel map")
    spike_bin[channelmap[rows, 0].astype(np.intp), which_bin] = 1

    well["spike_bin"] = spike_bin
    well["bin_size"] = bin_size
    well["rec_t_sec"] = rec_t_samp / well["samp_rate"]
    _stamp(well, bin_size=bin_size)
    return well
