"""Spiking and bursting activity.

Two layers, for each of spiking and bursting:

**Recording level** -- one :class:`~mxtreme.recording.Recording` in, the data out (one row per channel
or burst, per phase). Each result is also written next to the culture's other analysis outputs when an
``analysis_dir`` is given (``save=True`` by default):

- Spiking: :func:`firing_rate`, :func:`instantaneous_firing_rate`, :func:`smoothed_firing_rate`,
  :func:`isi`
- Bursting: :func:`burst_rate`, :func:`ibi`

**Group level** -- pass what to analyse and a :class:`~mxtreme.config.Config`. The target can be a
:class:`~mxtreme.identity.RecordingID` (one recording), a :class:`~mxtreme.identity.CultureID` (one
culture over DIV), a :class:`~mxtreme.identity.CultureSelector` (a group of cultures, each shown and
averaged), or a ``dict``/``list`` of selectors (groups compared against one another):

- :func:`summarize_spike_activity` / :func:`summarize_burst_activity` -- a tidy table
- :func:`plot_spike_activity_summary` / :func:`plot_burst_activity_summary` -- the 4-panel summary
- :func:`spike_activity_distributions` / :func:`burst_activity_distributions` and their ``plot_``
  counterparts -- the full distributions behind the summary, for one culture

Per-culture summaries are cached as CSVs under ``<analysis_dir>/activity/<batch_id>/<chip>/well<N>/``
and filled in incrementally, so a re-run only computes recordings it has not seen.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from mxtreme import io
from mxtreme.config import Config
from mxtreme.identity import CultureID
from mxtreme.params import ActivityParams, NetworkParams
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.recording import Recording
from mxtreme.analysis._group import SummarySpec, culture_summary, fill_docs, plot_summary, summarize
from mxtreme.analysis._paths import _summary_paths, recording_key, unflatten_recording_arrays
from mxtreme.analysis._plotting import plot_cdf_grid, sort_phases

__all__ = [
    # spiking
    "firing_rate", "instantaneous_firing_rate", "smoothed_firing_rate", "isi",
    "summarize_spike_activity", "plot_spike_activity_summary",
    "spike_activity_distributions", "plot_spike_activity_distributions",
    # bursting
    "burst_rate", "ibi",
    "summarize_burst_activity", "plot_burst_activity_summary",
    "burst_activity_distributions", "plot_burst_activity_distributions",
]


# --- shared helpers -------------------------------------------------------------------------------


def _load_recording(npz):
    """Load a Recording from ``npz``.

    Phases are reconstructed automatically from the spec embedded in the preprocessed data (written at
    preprocessing time from the ``Phases`` metadata key); a recording with no embedded spec has a
    single ``"full"`` phase. See :class:`mxtreme.recording.Recording`.
    """
    return Recording(0, io.load_preprocessed(npz))


def _recording_output(rec, analysis_dir, suffix: str) -> Path:
    """Where a recording-level result is saved: next to the culture's summaries, named like its npz."""
    out_dir = Path(analysis_dir) / "activity" / str(rec.batch_id) / str(rec.chip) / f"well{rec.well}"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = io.recording_file_name(rec.DIV, rec.plate_date, rec.chip, rec.batch_id, rec.well, suffix,
                                  rec.experiment or "")
    return out_dir / name


def _should_save(save: bool, analysis_dir) -> bool:
    return bool(save) and analysis_dir is not None


def _channels(rec) -> np.ndarray:
    """Every mapped channel, sorted -- silent ones included, so a quiet channel counts as 0 Hz."""
    return np.unique(np.asarray(rec.channelmap)[:, 1].astype(np.int64))


def _in_window(spike_data, start: int, stop: int):
    frames = spike_data["frameno"]
    return spike_data[(frames >= start) & (frames <= stop)]


def _per_channel(values_channel, channels, weights=None) -> np.ndarray:
    """Sum ``weights`` (or count rows) per channel, aligned with ``channels``."""
    idx = np.searchsorted(channels, values_channel)
    idx = np.clip(idx, 0, len(channels) - 1)
    known = channels[idx] == values_channel  # spikes on unmapped channels are dropped
    return np.bincount(idx[known], weights=None if weights is None else weights[known],
                       minlength=len(channels))


def _channel_isis_ms(spikes, samp_rate):
    """``(channel, isi_ms)`` for every interval between successive spikes on the same channel."""
    if len(spikes) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    order = np.lexsort((spikes["frameno"], spikes["channel"]))
    channel = np.asarray(spikes["channel"])[order].astype(np.int64)
    frames = np.asarray(spikes["frameno"])[order].astype(np.int64)
    same = channel[1:] == channel[:-1]
    return channel[1:][same], np.diff(frames)[same] / samp_rate * 1000


def _global_isis_ms(spikes, samp_rate):
    """Intervals between consecutive spikes anywhere on the array (all channels merged), in ms."""
    frames = np.sort(np.asarray(spikes["frameno"]).astype(np.int64))
    return np.diff(frames) / samp_rate * 1000


def _phase_window(rec, phase: str | None) -> tuple[str, int, int]:
    """Resolve ``phase`` against ``rec``'s own phases, as inclusive ``(name, start_frame, end_frame)``.

    ``None`` selects the recording's first phase -- ``"full"`` for a recording with no user-supplied
    phases, the first of the sequence otherwise. A name the recording doesn't carry falls back to the
    whole recording, so one report phase can be applied across a batch whose recordings don't all
    share the same phase spec.
    """
    named = {p.name: p for p in rec.phases}

    if phase is None:
        first = sort_phases(list(named))[0]
        return first, named[first].start_frame, named[first].end_frame

    if phase in named:
        return phase, named[phase].start_frame, named[phase].end_frame

    frames = rec.spike_data["frameno"]
    start = int(np.min(frames)) if len(frames) else 0
    stop = int(np.max(frames)) if len(frames) else 0
    print(f"  no {phase!r} phase (found: {', '.join(named)}); using the whole recording")
    return phase, start, stop


def _safe(fn, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(fn(arr)) if arr.size else np.nan


# --- spiking: recording level ---------------------------------------------------------------------


def firing_rate(rec: Recording, analysis_dir=None, *, save: bool = True) -> pd.DataFrame:
    """Per-channel firing rate of one recording, per phase.

    The rate is the channel's spike count in the phase divided by the phase's duration. Every mapped
    channel is listed, so a channel that never fired in a phase has a rate of 0.

    :param rec: The recording.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``); the table is saved
        there as ``..._firing_rate.csv`` when ``save`` is on.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: Columns ``channel, phase, n_spikes, duration_sec, firing_rate_hz``.
    """
    channels = _channels(rec)
    frames = []
    for p in rec.phases:
        spikes = _in_window(rec.spike_data, p.start_frame, p.end_frame)
        counts = _per_channel(np.asarray(spikes["channel"]).astype(np.int64), channels)
        duration = p.n_frames / rec.samp_rate
        frames.append(pd.DataFrame({
            "channel": channels,
            "phase": p.name,
            "n_spikes": counts.astype(np.int64),
            "duration_sec": duration,
            "firing_rate_hz": counts / duration if duration > 0 else np.nan,
        }))
    df = pd.concat(frames, ignore_index=True)

    if _should_save(save, analysis_dir):
        path = _recording_output(rec, analysis_dir, "firing_rate.csv")
        df.to_csv(path, index=False)
        print(f"Saved firing rates to {path}")
    return df


def instantaneous_firing_rate(rec: Recording, analysis_dir=None, *, bin_size_sec: float = 1.0,
                              save: bool = True) -> dict:
    """Per-channel firing rate over time: spike counts in fixed bins, divided by the bin width.

    The default 1 s bin gives a usable rate resolution (1 Hz steps) -- at the recording's own 10 ms
    binning, per-channel counts are almost all 0 or 1 -- and keeps an hour of ~1k channels at
    ~14 MB. Plot the result with :func:`mxtreme.visualizations.plot_timeseries_heatmap`.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as
        ``..._instantaneous_firing_rate_<bin>s.npz`` when ``save`` is on.
    :param bin_size_sec: Bin width in seconds.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: ``{"rate": (n_channels, n_bins) float32 Hz, "time_sec": bin start times,
        "channels": channel ids (rows of rate), "bin_size_sec": float,
        "phases": [(name, start_sec, end_sec), ...]}``.
    """
    if bin_size_sec <= 0:
        raise ValueError("bin_size_sec must be positive.")

    channels = _channels(rec)
    frames_per_bin = bin_size_sec * rec.samp_rate
    spike_frames = np.asarray(rec.spike_data["frameno"]).astype(np.int64)
    total_frames = max(int(np.ceil(rec.rec_t_sec * rec.samp_rate)),
                       int(spike_frames.max()) + 1 if spike_frames.size else 0)
    n_bins = max(1, int(np.ceil(total_frames / frames_per_bin)))

    rows = np.searchsorted(channels, np.asarray(rec.spike_data["channel"]).astype(np.int64))
    rows = np.clip(rows, 0, len(channels) - 1)
    known = channels[rows] == np.asarray(rec.spike_data["channel"])
    cols = np.minimum((spike_frames / frames_per_bin).astype(np.int64), n_bins - 1)

    counts = np.zeros((len(channels), n_bins), dtype=np.float32)
    np.add.at(counts, (rows[known], cols[known]), 1)
    rate = counts / np.float32(bin_size_sec)

    result = {
        "rate": rate,
        "time_sec": np.arange(n_bins) * bin_size_sec,
        "channels": channels,
        "bin_size_sec": float(bin_size_sec),
        "phases": [(p.name, p.start_frame / rec.samp_rate, (p.end_frame + 1) / rec.samp_rate)
                   for p in rec.phases],
    }

    if _should_save(save, analysis_dir):
        path = _recording_output(rec, analysis_dir, f"instantaneous_firing_rate_{bin_size_sec:g}s.npz")
        np.savez_compressed(
            path, rate=rate, time_sec=result["time_sec"], channels=channels,
            bin_size_sec=bin_size_sec,
            phase_names=np.array([p[0] for p in result["phases"]]),
            phase_bounds_sec=np.array([p[1:] for p in result["phases"]], dtype=float).reshape(-1, 2),
        )
        print(f"Saved instantaneous firing rate to {path}")
    return result


def _double_exponential(spike_bin, bin_size: float, tau_rise_sec: float, tau_decay_sec: float,
                        sample_bin_sec: float, chunk_channels: int = 64) -> np.ndarray:
    """Filter each row of ``spike_bin`` with a double exponential and decimate to ``sample_bin_sec``.

    Applied as the difference of two first-order IIR recursions (one per time constant) rather than as
    convolution with a truncated kernel: O(n), causal, and free of kernel-truncation error. Chunked over
    channels so peak memory scales with ``chunk_channels`` rather than the channel count.
    """
    sb = np.asarray(spike_bin)
    if sb.ndim != 2:
        raise ValueError(f"spike_bin must be 2-D (n_channels, n_bins); got shape {sb.shape}")
    tau_r, tau_d = float(tau_rise_sec), float(tau_decay_sec)
    if not 0 < tau_r < tau_d:
        raise ValueError(f"need 0 < tau_rise_sec < tau_decay_sec; got {tau_r} and {tau_d}")

    dt = float(bin_size)
    n_chan, n_bins = sb.shape
    decim = _decimation(dt, sample_bin_sec)
    n_out = -(-n_bins // decim)  # ceil division

    # Pole of each single-exponential recursion, and the scale that makes the difference of the two
    # integrate to one spike's worth of rate.
    a_rise, a_decay = np.exp(-dt / tau_r), np.exp(-dt / tau_d)
    scale = dt / (tau_d - tau_r)

    rates = np.empty((n_chan, n_out), dtype=np.float32)
    step = max(int(chunk_channels), 1)
    for start in range(0, n_chan, step):
        chunk = sb[start:start + step].astype(np.float32)
        filtered = lfilter([1.0], [1.0, -a_decay], chunk, axis=1)
        filtered -= lfilter([1.0], [1.0, -a_rise], chunk, axis=1)
        filtered *= scale
        rates[start:start + step] = filtered[:, ::decim]
    return rates


def _decimation(bin_size: float, sample_bin_sec: float) -> int:
    """Whole-number decimation from a recording's ``bin_size`` to ``sample_bin_sec``."""
    return max(round(float(sample_bin_sec) / float(bin_size)), 1)


def smoothed_firing_rate(rec: Recording, analysis_dir=None, *, tau_rise_sec: float | None = None,
                         tau_decay_sec: float | None = None, sample_bin_sec: float | None = None,
                         save: bool = True) -> dict:
    """Per-channel firing rate smoothed with a double-exponential kernel (Sono et al. 2026, Eq. 5-6).

    The binned spike train is filtered with a kernel that rises with ``tau_rise_sec`` and decays with
    ``tau_decay_sec``, then resampled to ``sample_bin_sec``. This is the signal
    :mod:`mxtreme.analysis.network` correlates (functional connectivity) and decomposes (PCA).
    Plot it with :func:`mxtreme.visualizations.plot_timeseries_heatmap`.

    .. note::
       It is built from ``Recording.spike_bin``, which is binary -- a bin holding five spikes and one
       holding a single spike are both 1 -- so rates inside bursts are clipped.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._smoothed_firing_rate.npz`` when
        ``save`` is on.
    :param tau_rise_sec: Rise time constant. Defaults to :class:`~mxtreme.params.NetworkParams`
        (0.05 s), as do the next two.
    :param tau_decay_sec: Decay time constant (default 2 s).
    :param sample_bin_sec: Output resolution, rounded to a whole multiple of ``rec.bin_size``
        (default 0.25 s).
    :param save: Write the result (needs ``analysis_dir``).
    :returns: ``{"rate": (n_channels, n_samples) float32, "time_sec": sample start times,
        "channels": channel ids (rows of rate, in ``spike_bin`` order), "sample_bin_sec": float,
        "phases": [(name, start_sec, end_sec), ...]}``.
    """
    defaults = NetworkParams()
    tau_r = defaults.tau_rise_sec if tau_rise_sec is None else tau_rise_sec
    tau_d = defaults.tau_decay_sec if tau_decay_sec is None else tau_decay_sec
    sample = defaults.sample_bin_sec if sample_bin_sec is None else sample_bin_sec

    rate = _double_exponential(rec.spike_bin, rec.bin_size, tau_r, tau_d, sample, defaults.chunk_channels)
    step = _decimation(rec.bin_size, sample) * rec.bin_size
    result = {
        "rate": rate,
        "time_sec": np.arange(rate.shape[1]) * step,
        "channels": np.asarray(rec.channelmap)[:, 1].astype(np.int64),
        "sample_bin_sec": float(step),
        "phases": [(p.name, p.start_frame / rec.samp_rate, (p.end_frame + 1) / rec.samp_rate)
                   for p in rec.phases],
    }

    if _should_save(save, analysis_dir):
        path = _recording_output(rec, analysis_dir, "smoothed_firing_rate.npz")
        np.savez_compressed(path, rate=rate, time_sec=result["time_sec"], channels=result["channels"],
                            sample_bin_sec=step, tau_rise_sec=tau_r, tau_decay_sec=tau_d)
        print(f"Saved smoothed firing rate to {path}")
    return result


def isi(rec: Recording, analysis_dir=None, *, level: str = "channel",
        isi_threshold_ms: float | None = None, save: bool = True) -> pd.DataFrame:
    """Inter-spike intervals of one recording, per phase.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._isi_<level>.npz`` when ``save``
        is on (``.npz`` rather than CSV: a recording has millions of intervals).
    :param level: ``"channel"`` -- intervals between successive spikes on the same channel;
        ``"global"`` -- intervals between consecutive spikes anywhere on the array, with every channel
        merged into one spike train (the array-wide ISI).
    :param isi_threshold_ms: Drop intervals at or above this, replicating the MaxLab ISI analysis
        (without it, the long gaps between bursts dominate). Defaults to
        :attr:`ActivityParams.isi_threshold_ms <mxtreme.params.ActivityParams>`; pass ``np.inf``
        to keep every interval.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: Columns ``channel, phase, isi_ms`` for ``"channel"``; ``phase, isi_ms`` for ``"global"``.
    """
    if level not in ("channel", "global"):
        raise ValueError(f"level must be 'channel' or 'global', not {level!r}")
    threshold = ActivityParams().isi_threshold_ms if isi_threshold_ms is None else isi_threshold_ms

    phase_names = [p.name for p in rec.phases]
    channel_parts, isi_parts, phase_parts = [], [], []
    for i, p in enumerate(rec.phases):
        spikes = _in_window(rec.spike_data, p.start_frame, p.end_frame)
        if level == "channel":
            channel, values = _channel_isis_ms(spikes, rec.samp_rate)
        else:
            values = _global_isis_ms(spikes, rec.samp_rate)
            channel = None
        keep = values < threshold
        isi_parts.append(values[keep])
        phase_parts.append(np.full(int(keep.sum()), i, dtype=np.int16))
        if channel is not None:
            channel_parts.append(channel[keep])

    isi_ms = np.concatenate(isi_parts) if isi_parts else np.array([])
    phase_codes = np.concatenate(phase_parts) if phase_parts else np.array([], dtype=np.int16)
    # Categorical: one phase string per interval would cost more memory than the intervals.
    columns = {"phase": pd.Categorical.from_codes(phase_codes, categories=phase_names), "isi_ms": isi_ms}
    if level == "channel":
        columns = {"channel": np.concatenate(channel_parts) if channel_parts else np.array([], dtype=np.int64),
                   **columns}
    df = pd.DataFrame(columns)

    if _should_save(save, analysis_dir):
        path = _recording_output(rec, analysis_dir, f"isi_{level}.npz")
        arrays = {"isi_ms": isi_ms, "phase_code": phase_codes, "phase_names": np.array(phase_names),
                  "isi_threshold_ms": threshold}
        if level == "channel":
            arrays["channel"] = columns["channel"]
        np.savez_compressed(path, **arrays)
        print(f"Saved {level} ISIs to {path}")
    return df


# --- bursting: recording level --------------------------------------------------------------------


def _bursts_by_phase(rec, burst_df) -> dict[str, pd.DataFrame]:
    """Network bursts of ``rec`` split by phase, each sorted by peak.

    Uses the burst table's own ``phase`` column (stamped at detection) when present; otherwise each
    burst is assigned by where its peak falls among ``rec``'s phases.
    """
    bursts = burst_df[burst_df["kind"] == "network"] if "kind" in burst_df.columns else burst_df
    out = {}
    for p in rec.phases:
        if "phase" in bursts.columns:
            in_phase = bursts[bursts["phase"] == p.name]
        else:
            in_phase = bursts[bursts["peak_frame"].between(p.start_frame, p.end_frame)]
        out[p.name] = in_phase.sort_values("peak_frame")
    return out


def burst_rate(rec: Recording, burst_df: pd.DataFrame, analysis_dir=None, *,
               save: bool = True) -> pd.DataFrame:
    """Network-burst rate of one recording, per phase: bursts in the phase / the phase's duration.

    A recording doesn't carry its bursts, so pass them in, e.g.
    ``pd.read_csv(recording_paths.require_burst_stats())`` or ``BurstSet.to_dataframe()``.

    :param rec: The recording.
    :param burst_df: Its bursts (only ``kind == "network"`` rows are counted).
    :param analysis_dir: Analysis output root; saved there as ``..._burst_rate.csv`` when ``save`` is on.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: Columns ``phase, n_bursts, duration_sec, burst_rate_hz`` -- every phase, including
        those with no bursts.
    """
    by_phase = _bursts_by_phase(rec, burst_df)
    rows = []
    for p in rec.phases:
        duration = p.n_frames / rec.samp_rate
        n = len(by_phase[p.name])
        rows.append({"phase": p.name, "n_bursts": n, "duration_sec": duration,
                     "burst_rate_hz": n / duration if duration > 0 else np.nan})
    df = pd.DataFrame(rows, columns=["phase", "n_bursts", "duration_sec", "burst_rate_hz"])

    if _should_save(save, analysis_dir):
        path = _recording_output(rec, analysis_dir, "burst_rate.csv")
        df.to_csv(path, index=False)
        print(f"Saved burst rates to {path}")
    return df


def ibi(rec: Recording, burst_df: pd.DataFrame, analysis_dir=None, *, save: bool = True) -> pd.DataFrame:
    """Inter-burst intervals of one recording, per phase: time between successive network-burst peaks.

    :param rec: The recording.
    :param burst_df: Its bursts (only ``kind == "network"`` rows are used).
    :param analysis_dir: Analysis output root; saved there as ``..._ibi.csv`` when ``save`` is on.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: Columns ``phase, burst_index, ibi_sec``; ``burst_index`` is the position (within the
        phase) of the burst that *ends* the interval.
    """
    frames = []
    for phase, bursts in _bursts_by_phase(rec, burst_df).items():
        peaks = bursts["peak_frame"].to_numpy(dtype=float)
        if peaks.size < 2:
            continue
        frames.append(pd.DataFrame({
            "phase": phase,
            "burst_index": np.arange(1, peaks.size),
            "ibi_sec": np.diff(peaks) / rec.samp_rate,
        }))
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=["phase", "burst_index", "ibi_sec"]))

    if _should_save(save, analysis_dir):
        path = _recording_output(rec, analysis_dir, "ibi.csv")
        df.to_csv(path, index=False)
        print(f"Saved IBIs to {path}")
    return df


# --- per-culture summaries (cached) ---------------------------------------------------------------


def _spike_summary_rows(rp) -> list[dict]:
    """One row per phase of recording ``rp``: across-channel statistics of the spiking metrics."""
    rec = _load_recording(rp.npz)
    fr = firing_rate(rec, save=False)
    isis = isi(rec, level="channel", save=False)
    channels = _channels(rec)

    rows = []
    for p in rec.phases:
        fr_p = fr[fr["phase"] == p.name]
        active = fr_p[fr_p["n_spikes"] > 0]
        fr_vals = active["firing_rate_hz"].to_numpy()

        isi_p = isis[isis["phase"] == p.name]
        isi_vals = isi_p.groupby("channel")["isi_ms"].median().to_numpy()  # per-channel median ISI

        spikes = _in_window(rec.spike_data, p.start_frame, p.end_frame)
        spike_channels = np.asarray(spikes["channel"]).astype(np.int64)
        n = _per_channel(spike_channels, channels)
        amp_sum = _per_channel(spike_channels, channels,
                               weights=np.asarray(spikes["amplitude"], dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            amp_vals = np.where(n > 1, amp_sum / n, np.nan) * 1e6  # V -> µV

        n_total, n_active = len(fr_p), len(active)
        rows.append({
            "div": rp.recording_id.div,
            "experiment": rp.recording_id.experiment or "",
            "phase": p.name,
            "mean_fr_hz": _safe(np.mean, fr_vals),
            "median_fr_hz": _safe(np.median, fr_vals),
            "std_fr_hz": _safe(np.std, fr_vals),
            "mean_isi_msec": _safe(np.mean, isi_vals),
            "median_isi_msec": _safe(np.median, isi_vals),
            "std_isi_msec": _safe(np.std, isi_vals),
            "mean_amp_uv": _safe(np.mean, amp_vals),
            "median_amp_uv": _safe(np.median, amp_vals),
            "std_amp_uv": _safe(np.std, amp_vals),
            "n_active_chan": n_active,
            "n_total_chan": n_total,
            "pct_active_chan": 100 * n_active / n_total if n_total else np.nan,
        })
    return rows


def _burst_summary_rows(rp) -> list[dict]:
    """One row per phase of recording ``rp``: network-burst count, rate, and IBI/size/duration stats."""
    rec = _load_recording(rp.npz)
    bursts = pd.read_csv(rp.require_burst_stats())
    rates = burst_rate(rec, bursts, save=False).set_index("phase")
    ibis = ibi(rec, bursts, save=False)
    by_phase = _bursts_by_phase(rec, bursts)

    rows = []
    for p in rec.phases:
        phase_bursts = by_phase[p.name]
        ibi_vals = ibis.loc[ibis["phase"] == p.name, "ibi_sec"].to_numpy()
        sizes = phase_bursts["size_frac_elec"].to_numpy(dtype=float) * 100
        durations = phase_bursts["duration_frames"].to_numpy(dtype=float) / rec.samp_rate
        rows.append({
            "div": rp.recording_id.div,
            "experiment": rp.recording_id.experiment or "",
            "phase": p.name,
            "n_bursts": int(rates.loc[p.name, "n_bursts"]),
            "burst_rate_hz": rates.loc[p.name, "burst_rate_hz"],
            "mean_ibi_sec": _safe(np.mean, ibi_vals),
            "median_ibi_sec": _safe(np.median, ibi_vals),
            "std_ibi_sec": _safe(np.std, ibi_vals),
            "mean_size_pct": _safe(np.mean, sizes),
            "median_size_pct": _safe(np.median, sizes),
            "std_size_pct": _safe(np.std, sizes),
            "mean_dur_sec": _safe(np.mean, durations),
            "median_dur_sec": _safe(np.median, durations),
            "std_dur_sec": _safe(np.std, durations),
        })
    return rows


def _culture_spike_summary(cpath, analysis_dir: Path, use_existing: bool = True) -> pd.DataFrame:
    """Per-culture spiking summary (one row per recording x phase), cached."""
    return culture_summary(cpath, analysis_dir, "activity", "spike_activity_summary", _spike_summary_rows,
                           use_existing)


def _culture_burst_summary(cpath, analysis_dir: Path, use_existing: bool = True) -> pd.DataFrame:
    """Per-culture bursting summary (one row per recording x phase), cached."""
    return culture_summary(cpath, analysis_dir, "activity", "burst_activity_summary", _burst_summary_rows,
                           use_existing)


# --- group level ----------------------------------------------------------------------------------

# The culture-level panels show a central value ± SD across channels/bursts; the population panels
# show the across-culture mean of the same columns ± SEM/SD.
_SPIKE_SPEC = SummarySpec(
    title="Spiking Activity",
    within="SD across channels",
    metrics=[
        ("mean_fr_hz",      "std_fr_hz",    "Firing rate (Hz)",     "Firing Rate"),
        ("mean_isi_msec",   "std_isi_msec", "Median ISI (ms)",      "Inter-Spike Interval"),
        ("mean_amp_uv",     "std_amp_uv",   "Spike amplitude (µV)", "Spike Amplitude"),
        ("pct_active_chan", None,           "Active channels (%)",  "Active Channels"),
    ],
)
_BURST_SPEC = SummarySpec(
    title="Bursting Activity",
    within="SD across bursts",
    metrics=[
        ("median_ibi_sec",  "std_ibi_sec",  "Median IBI (s)",              "Inter-Burst Interval"),
        ("median_size_pct", "std_size_pct", "Median burst size (% elec.)", "Burst Size"),
        ("median_dur_sec",  "std_dur_sec",  "Median duration (s)",         "Burst Duration"),
        ("burst_rate_hz",   None,           "Burst rate (bursts s⁻¹)",     "Burst Rate"),
    ],
)


def summarize_spike_activity(target, config: Config, *, phase: str | None = None,
                             use_existing: bool = True) -> pd.DataFrame:
    """Table of spiking statistics: one row per recording and phase.

    Each row is one recording's phase, summarised across its channels: firing rate
    (``mean_fr_hz`` / ``median_fr_hz`` / ``std_fr_hz``, over channels that fired), per-channel median
    ISI (``*_isi_msec``), mean spike amplitude (``*_amp_uv``), and active channels
    (``n_active_chan``, ``n_total_chan``, ``pct_active_chan``). Identity columns lead: ``batch_id,
    culture_id, chip, well, div, experiment, phase``, plus ``group`` when several selectors are given.

    {target}
    :param phase: Keep only this phase; ``None`` keeps every phase.
    :param use_existing: Reuse cached per-culture summaries; ``False`` recomputes the target's
        recordings.
    :returns: The summary table.
    """
    return summarize(target, config, _culture_spike_summary, phase=phase, use_existing=use_existing)[0]


def summarize_burst_activity(target, config: Config, *, phase: str | None = None,
                             use_existing: bool = True) -> pd.DataFrame:
    """Table of network-burst statistics: one row per recording and phase.

    Each row is one recording's phase: ``n_bursts``, ``burst_rate_hz``, and mean / median / SD of the
    inter-burst interval (``*_ibi_sec``), burst size (``*_size_pct``, % of electrodes recruited) and
    burst duration (``*_dur_sec``). Identity columns lead as in :func:`summarize_spike_activity`.
    Needs burst detection to have been run on every recording.

    {target}
    :param phase: Keep only this phase; ``None`` keeps every phase.
    :param use_existing: Reuse cached per-culture summaries; ``False`` recomputes the target's
        recordings.
    :returns: The summary table.
    """
    return summarize(target, config, _culture_burst_summary, phase=phase, use_existing=use_existing)[0]


def plot_spike_activity_summary(target, config: Config, *, split_by: str = "phase",
                                phase: str | None = None, error: str = "sem",
                                show_plot: bool = True, save_path=None):
    """Four-panel spiking summary: firing rate, ISI, spike amplitude, active channels, vs DIV.

    {plot_doc}
    """
    return plot_summary(target, config, _culture_spike_summary, _SPIKE_SPEC, split_by=split_by,
                        phase=phase, error=error, show_plot=show_plot, save_path=save_path)


def plot_burst_activity_summary(target, config: Config, *, split_by: str = "phase",
                                phase: str | None = None, error: str = "sem",
                                show_plot: bool = True, save_path=None):
    """Four-panel network-burst summary: IBI, burst size, burst duration, burst rate, vs DIV.

    {plot_doc}
    """
    return plot_summary(target, config, _culture_burst_summary, _BURST_SPEC, split_by=split_by,
                        phase=phase, error=error, show_plot=show_plot, save_path=save_path)


fill_docs(
    summarize_fns=(summarize_spike_activity, summarize_burst_activity),
    plot_fns=((plot_spike_activity_summary, "channels"), (plot_burst_activity_summary, "bursts")),
)


# --- distributions (culture level) ----------------------------------------------------------------

#: Keys of :func:`spike_activity_distributions` / :func:`burst_activity_distributions`.
SPIKE_DISTRIBUTION_KEYS = ("fr_hz", "isi_ms")
BURST_DISTRIBUTION_KEYS = ("ibi_sec", "size_pct")

# (key, x label, panel title, log x) for the CDF panels.
SPIKE_CDF_METRICS = [
    ("fr_hz",  "Firing rate (Hz)", "Firing Rate",          True),
    ("isi_ms", "ISI (ms)",         "Inter-Spike Interval", True),
]
BURST_CDF_METRICS = [
    ("ibi_sec",  "IBI (s)",                   "Inter-Burst Interval", True),
    ("size_pct", "Burst size (% electrodes)", "Burst Size",           False),
]


def _culture_paths(culture, config) -> CulturePaths:
    if isinstance(culture, CulturePaths):
        return culture
    if isinstance(culture, CultureID):
        return resolve_paths(culture, config)
    raise TypeError(f"Distributions are per culture: pass a CultureID, not {type(culture).__name__}.")


def _spike_dists(rp, phase, params: ActivityParams) -> dict:
    rec = _load_recording(rp.npz)
    _name, start, stop = _phase_window(rec, phase)
    spikes = _in_window(rec.spike_data, start, stop)
    counts = _per_channel(np.asarray(spikes["channel"]).astype(np.int64), _channels(rec))
    duration = (stop - start + 1) / rec.samp_rate
    _channel, isi_vals = _channel_isis_ms(spikes, rec.samp_rate)
    return {
        "fr_hz": counts[counts > 0] / duration if duration > 0 else np.array([]),
        "isi_ms": isi_vals[isi_vals < params.isi_threshold_ms],
    }


def _burst_dists(rp, phase, params) -> dict:
    rec = _load_recording(rp.npz)
    bursts = pd.read_csv(rp.require_burst_stats())
    bursts = bursts[bursts["kind"] == "network"]
    if phase is not None and "phase" in bursts.columns:
        bursts = bursts[bursts["phase"] == phase]
    bursts = bursts.sort_values("peak_frame")
    peaks = bursts["peak_frame"].to_numpy(dtype=float) / rec.samp_rate
    return {
        "ibi_sec": np.diff(peaks) if peaks.size > 1 else np.array([]),
        "size_pct": bursts["size_frac_elec"].to_numpy(dtype=float) * 100,
    }


def _distributions(cpath, analysis_dir: Path, name: str, dists_fn, *, phase, use_existing, params):
    """``{(div, experiment): {key: array}}`` for ``cpath``, cached as one ``.npz`` per culture per phase."""
    save_dir, npz_path = _summary_paths(cpath, analysis_dir, "distributions",
                                        f"{phase or 'default'}_{name}", ext=".npz")
    wanted = [(rp.recording_id.div, rp.recording_id.experiment or "") for rp in cpath.recordings]

    if use_existing and npz_path.exists():
        with np.load(npz_path) as cached:
            dists = unflatten_recording_arrays(cached)
        if dists is not None and set(wanted) <= set(dists):
            print(f"Loading existing distributions from {npz_path}")
            return {key: dists[key] for key in sorted(wanted)}

    dists = {key: dists_fn(rp, phase, params) for key, rp in zip(wanted, cpath.recordings)}
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        **{f"{recording_key(*key)}__{metric}": values
           for key, per_rec in dists.items() for metric, values in per_rec.items()},
    )
    print(f"Saved distributions to {npz_path}")
    return dict(sorted(dists.items()))


def _spike_distributions(cpath, analysis_dir, *, phase=None, use_existing=True, params=None):
    return _distributions(cpath, analysis_dir, "spike_distributions", _spike_dists, phase=phase,
                          use_existing=use_existing, params=params or ActivityParams())


def _burst_distributions(cpath, analysis_dir, *, phase=None, use_existing=True):
    return _distributions(cpath, analysis_dir, "burst_distributions", _burst_dists, phase=phase,
                          use_existing=use_existing, params=None)


def spike_activity_distributions(culture, config: Config, *, phase: str | None = None,
                                 use_existing: bool = True, params: ActivityParams | None = None) -> dict:
    """Per-recording spiking distributions for one culture -- the data behind the summary statistics.

    Returns ``{(div, experiment): {key: values}}``, one entry per recording, with keys:

    ========== ===============================================================================
    ``fr_hz``  firing rate of every channel that fired in the phase (spikes / phase duration)
    ``isi_ms`` every within-channel inter-spike interval, pooled across channels (thresholded
               as in :func:`isi`)
    ========== ===============================================================================

    Cached as one ``.npz`` per culture per phase under ``<analysis_dir>/distributions/``.

    :param culture: A :class:`~mxtreme.identity.CultureID`.
    :param config: The :class:`~mxtreme.config.Config` describing the store.
    :param phase: Phase to restrict to; ``None`` selects each recording's first phase.
    :param use_existing: Reuse the cached ``.npz``.
    :param params: Activity settings (the ISI threshold); defaults to
        :class:`~mxtreme.params.ActivityParams`.
    :returns: ``{(div, experiment): {key: numpy array}}``, ordered by DIV then label.
    """
    return _spike_distributions(_culture_paths(culture, config), config.analysis_dir, phase=phase,
                                use_existing=use_existing, params=params)


def burst_activity_distributions(culture, config: Config, *, phase: str | None = None,
                                 use_existing: bool = True) -> dict:
    """Per-recording network-burst distributions for one culture.

    Returns ``{(div, experiment): {key: values}}``, one entry per recording, with keys:

    ============ ===========================================================================
    ``ibi_sec``  interval between successive network-burst peaks
    ``size_pct`` network-burst size, as a percentage of electrodes recruited
    ============ ===========================================================================

    :param culture: A :class:`~mxtreme.identity.CultureID`.
    :param config: The :class:`~mxtreme.config.Config` describing the store.
    :param phase: Phase to restrict to; ``None`` uses every burst of the recording.
    :param use_existing: Reuse the cached ``.npz``.
    :returns: ``{(div, experiment): {key: numpy array}}``, ordered by DIV then label.
    """
    return _burst_distributions(_culture_paths(culture, config), config.analysis_dir, phase=phase,
                                use_existing=use_existing)


def plot_spike_activity_distributions(culture, config: Config, *, phase: str | None = None,
                                      show_plot: bool = True, save_path=None):
    """CDFs of firing rate and ISI for one culture, one line per recording (see
    :func:`spike_activity_distributions`).

    :returns: The matplotlib Figure (``None`` if the culture has no recordings).
    """
    cpath = _culture_paths(culture, config)
    dists = _spike_distributions(cpath, config.analysis_dir, phase=phase)
    if not dists:
        return None
    return plot_cdf_grid(dists, SPIKE_CDF_METRICS, ncols=2, figsize=(11, 4.5),
                         suptitle=f"Spiking distributions — {cpath.culture_id}  (phase: {phase or 'first'})",
                         show_plot=show_plot, save_path=save_path)


def plot_burst_activity_distributions(culture, config: Config, *, phase: str | None = None,
                                      show_plot: bool = True, save_path=None):
    """CDFs of IBI and burst size for one culture, one line per recording (see
    :func:`burst_activity_distributions`).

    :returns: The matplotlib Figure (``None`` if the culture has no recordings).
    """
    cpath = _culture_paths(culture, config)
    dists = _burst_distributions(cpath, config.analysis_dir, phase=phase)
    if not dists:
        return None
    return plot_cdf_grid(dists, BURST_CDF_METRICS, ncols=2, figsize=(11, 4.5),
                         suptitle=f"Bursting distributions — {cpath.culture_id}  (phase: {phase or 'all'})",
                         show_plot=show_plot, save_path=save_path)
