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
import pandas as pd

logger = logging.getLogger(__name__)


def _stamp(well: dict, **params) -> None:
    """Record the parameters a step used into ``well['preprocessing_params']`` (provenance)."""
    well.setdefault("preprocessing_params", {}).update(params)


def normalize_time(well: dict) -> dict:
    """Offset spike and event frames so the recording starts at frame 0.

    Subtracts the raw recording's first frame (``well['raw_start']``) from every spike and event frame.

    :param well: A well data dict.
    :returns: The same dict, with normalized ``data['frameno']`` and ``eventtime``.
    :rtype: dict
    """
    well["data"]["frameno"] -= well["raw_start"]
    well["eventtime"] = well["eventtime"] - well["raw_start"]
    return well


def dac_to_voltage(well: dict) -> dict:
    """Convert spike amplitudes from DAC units to volts by multiplying by the least significant bit.

    :param well: A well data dict.
    :returns: The same dict, with ``data['amplitude']`` expressed in volts.
    :rtype: dict
    """
    well["data"]["amplitude"] = well["data"]["amplitude"] * well["lsb"]
    return well


def remove_positive_deflections(well: dict) -> dict:
    """Keep only negative-going spikes (drop positive deflections).

    :param well: A well data dict.
    :returns: The same dict, filtered to negative-amplitude spikes.
    :rtype: dict
    """
    well["data"] = well["data"][well["data"]["amplitude"] < 0]
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
    logger.info("Applying amplitude filter with threshold of %s µV...", amp_thresh * 1e6)
    mask = np.abs(well["data"]["amplitude"]) >= amp_thresh
    well["data"] = well["data"][mask]
    if len(well["data"]) == 0:
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
    data = well["data"]
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
    well["data"] = data[keep]
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
    mask = np.isin(well["data"]["channel"], well["mapping"]["channel"])
    well["data"] = well["data"][mask]
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
    post_stim_frames = post_stim_period * well["samp_rate"]

    logger.info("Removing stimulation frames from spike data...")
    event_df = pd.DataFrame({"eventtime": well["eventtime"], "eventmessage": well["event_messages"]})

    if len(event_df) == 0:
        well["stim_frames"] = []
        _stamp(well, post_stim_period=post_stim_period)
        return well

    starts = event_df[event_df["eventmessage"].apply(lambda d: "start_stimulation" in d)]["eventtime"].to_numpy()
    stops = event_df[event_df["eventmessage"].apply(lambda d: "end_stimulation" in d)]["eventtime"].to_numpy()

    if len(starts) == 0:
        well["stim_frames"] = []
        _stamp(well, post_stim_period=post_stim_period)
        return well

    if stops[0] < starts[0]:  # an end marker precedes the first start
        stops = stops[1:]

    assert len(starts) == len(stops)

    well["stim_frames"] = np.concatenate(
        [np.arange(s, e + post_stim_frames, dtype=np.int64) for s, e in zip(starts, stops)]
    )

    mask = ~np.isin(well["data"]["frameno"], well["stim_frames"])
    well["data"] = well["data"][mask]
    _stamp(well, post_stim_period=post_stim_period)
    return well


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
    rec_t_samp = np.max(well["data"]["frameno"])
    num_bins = int(np.ceil(np.divide(rec_t_samp, bin_win)))

    num_chan = well["channelmap"].shape[0]
    spike_bin = np.zeros((num_chan, num_bins), dtype=np.uint8)  # binary matrix; uint8 keeps it 8x smaller
    which_bin = (well["data"]["frameno"] / bin_win).astype(int)
    which_bin = np.clip(which_bin, 0, num_bins - 1)

    logger.info("Binning Spikes...")
    mapping = dict(zip(well["channelmap"][:, 1], well["channelmap"][:, 0]))
    channel_ids = pd.Series(well["data"]["channel"]).map(mapping).to_numpy(dtype=int)
    assert len(channel_ids) == len(which_bin)
    spike_bin[channel_ids, which_bin] = 1

    well["spike_bin"] = spike_bin
    well["bin_size"] = bin_size
    well["rec_t_sec"] = rec_t_samp / well["samp_rate"]
    _stamp(well, bin_size=bin_size)
    return well
