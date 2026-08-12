"""The :class:`Burst` record and its per-burst feature computation.

A :class:`Burst` is a plain dataclass: detection fills its timing/classification fields, and
:meth:`Burst.compute_features` fills the rest from a :class:`Recording <mxtreme.recording.Recording>`.

**Units.** Every temporal field is in **frames** (the native, highest-resolution unit; the array
sampling rate is ``recording.samp_rate``). There are deliberately no ``*_sec`` / ``*_bin`` columns --
convert on demand with :func:`mxtreme.utils.frame_to_sec` / :func:`mxtreme.utils.frame_to_bin`.
Coordinates are in µm; ``peak_amp`` and ``size_frac_elec`` are fractions.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
from scipy.stats import median_abs_deviation

from mxtreme import utils


# Column order for DataFrame / CSV output.
_COLUMNS = [
    "id",
    "peak_frame",
    "peak_amp",
    "phase",
    "kind",
    "group_start_frame",
    "group_stop_frame",
    "n_spikes",
    "has_stim",
    "instantaneous",
    "ignore",
    "onset_frame",
    "offset_frame",
    "duration_frames",
    "size_frac_elec",
    "origin_x",
    "origin_y",
    # "weighted_origin_x",   # disabled: uncomment (with the compute in _compute_origin) to re-enable
    # "weighted_origin_y",
    "peak_x",
    "peak_y",
    "origin_chan_counts",
    "constituent_chan_counts",
]

# Columns whose values are dicts and therefore need parsing when read back from CSV.
_DICT_COLUMNS = ("origin_chan_counts", "constituent_chan_counts")


def _window(frameno, lo, hi) -> tuple[int, int]:
    """Index bounds ``(i0, i1)`` of the ascending ``frameno`` where ``lo <= frameno <= hi``.

    ``spike_data[i0:i1]`` is then the inclusive-bounds window, found by binary search (O(log n))
    instead of a full-array boolean mask. Requires ``frameno`` sorted ascending.
    """
    i0 = int(np.searchsorted(frameno, lo, side="left"))
    i1 = int(np.searchsorted(frameno, hi, side="right"))
    return i0, i1


@dataclass
class Burst:
    """One detected burst.

    Detection sets the first block of fields; :meth:`compute_features` fills the rest (they carry
    sentinel defaults until then, so an un-featurized or ignored burst still has every column).

    :param id: Burst index within its recording.
    :param peak_frame: Frame of the burst's array-wide rate peak.
    :param peak_amp: Peak height as a fraction of channels active in that bin.
    :param kind: ``"network"`` (peak >= burst threshold) or ``"mini"`` (between noise and burst).
    :param group_start_frame: First spike frame of the ISI-N spike group this peak belongs to.
    :param group_stop_frame: Last spike frame of that group.
    :param n_spikes: Number of spikes in the group.
    :param phase: Injected phase label of ``peak_frame`` (``"full"`` by default; ``None`` if outside
        all phases).
    """

    # --- set by detection ---
    id: int
    peak_frame: int
    peak_amp: float
    kind: str
    group_start_frame: int
    group_stop_frame: int
    n_spikes: int
    phase: Optional[str] = None

    # --- set by compute_features ---
    has_stim: bool = False
    instantaneous: bool = False
    ignore: bool = False
    onset_frame: int = -1
    offset_frame: int = -1
    duration_frames: int = -1
    size_frac_elec: float = -1.0
    origin_x: float = -1.0
    origin_y: float = -1.0
    # weighted_origin is kept available but not computed or written by default (see _compute_origin).
    weighted_origin_x: float = -1.0
    weighted_origin_y: float = -1.0
    peak_x: float = -1.0
    peak_y: float = -1.0
    origin_chan_counts: dict = field(default_factory=dict)
    constituent_chan_counts: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ serialization

    def to_row(self) -> dict:
        """Return this burst as an ordered dict suitable for a DataFrame row."""
        return {col: getattr(self, col) for col in _COLUMNS}

    @classmethod
    def from_row(cls, row: dict) -> "Burst":
        """Rebuild a :class:`Burst` from a DataFrame/CSV row (parsing dict-valued columns)."""
        kwargs = {}
        valid = {f.name for f in fields(cls)}
        for key, value in row.items():
            if key not in valid:
                continue
            if key in _DICT_COLUMNS and isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    value = {}
            kwargs[key] = value
        return cls(**kwargs)

    # ------------------------------------------------------------------ features

    def compute_features(self, recording, params, *, asdr: Optional[np.ndarray] = None) -> "Burst":
        """Compute all per-burst features in place, reading arrays off ``recording``.

        Onset/offset, duration, origin, constituent channels, peak location, and stimulation overlap.
        Any failure tags the burst ``ignore=True`` (leaving sentinel feature values) rather than
        raising, so one bad burst never aborts a recording.

        Window slices use binary search on the sorted spike ``frameno`` (:func:`_window`, O(log n) each)
        rather than full-array scans, and each distinct window is sliced once.

        :param recording: Source :class:`Recording <mxtreme.recording.Recording>`.
        :param params: :class:`~mxtreme.params.BurstFeatureParams` (onset/offset thresholds).
        :param asdr: Optional precomputed array-wide rate (``recording.asdr``); computed if omitted.
        :returns: ``self``.
        """
        samp_rate = float(recording.samp_rate)
        bin_size = float(recording.bin_size)
        spike_data = recording.spike_data
        channelmap = recording.channelmap
        stim_frames = recording.stim_frames
        asdr = recording.asdr if asdr is None else asdr

        frames_per_bin = max(1, int(round(bin_size * samp_rate)))
        frameno = spike_data["frameno"]  # sorted ascending (see BurstSet.extract_features guard)

        # Reset the flags this method produces so a recompute (e.g. on a BurstSet reloaded from CSV)
        # starts clean. Detection-set fields (peak_frame, kind, group_*, phase, ...) are left untouched.
        self.ignore = False
        self.instantaneous = False

        try:
            self._compute_onset_offset(asdr, spike_data, frameno, samp_rate, bin_size,
                                       frames_per_bin, params)
            self.duration_frames = int(self.offset_frame - self.onset_frame)
            self._compute_origin(spike_data, frameno, channelmap, frames_per_bin)
            self._compute_body(spike_data, frameno, channelmap)
            self._stim_in_burst(stim_frames)
        except Exception as exc:  # skip but tag for further investigation
            print(f"Burst {self.id} feature computation failed: {exc}")
            self.ignore = True

        return self

    # -- individual feature steps (ported from ISIThreshBurst) ----------------------

    def _compute_onset_offset(self, asdr, spike_data, frameno, samp_rate, bin_size,
                              frames_per_bin, params):
        """Find onset/offset frames from binned + raw spike data around the peak."""
        peak_bin = utils.frame_to_bin(self.peak_frame, samp_rate, bin_size)
        group_start_bin = utils.frame_to_bin(self.group_start_frame, samp_rate, bin_size)
        group_stop_bin = utils.frame_to_bin(self.group_stop_frame, samp_rate, bin_size)

        # Local noise floor: MAD of the rate over the spike group, scaled to a std estimate.
        mad_value = median_abs_deviation(asdr[group_start_bin:group_stop_bin])
        noise_std = mad_value * 1.4826

        # Bins whose rate sits at/under the local noise floor act as anchors around the burst.
        zero_points = np.where(asdr <= noise_std)[0]
        pre = zero_points[zero_points < peak_bin]
        post = zero_points[zero_points > peak_bin]

        if len(pre) == 0:
            raise IndexError("Burst too close to start of recording to compute onset.")
        if len(post) == 0:
            raise IndexError("Burst too close to end of recording to compute offset.")

        onset_zero_frame = utils.bin_to_frame(pre[-1], samp_rate, bin_size)
        offset_zero_frame = utils.bin_to_frame(post[0], samp_rate, bin_size)
        peak_frame_start = utils.bin_to_frame(peak_bin, samp_rate, bin_size)
        peak_frame_end = peak_frame_start - frames_per_bin  # lower bound for the offset slice

        o0, o1 = _window(frameno, onset_zero_frame, peak_frame_start)
        f0, f1 = _window(frameno, peak_frame_end, offset_zero_frame)
        onset_slice = spike_data["frameno"][o0:o1]
        offset_slice = spike_data["frameno"][f0:f1]

        # Thresholds on cumulative spike count: onset_thresh_pct into 0->peak, offset_thresh_pct in
        # from peak->0 (i.e. 1 - pct, working outward from the peak).
        onset_thresh = int(len(onset_slice) * params.onset_thresh_pct)
        offset_thresh = int(len(offset_slice) * (1 - params.offset_thresh_pct))

        if len(onset_slice) == 0:
            # No spikes between the left anchor and the peak -> instantaneous onset.
            self.onset_frame = int(onset_zero_frame)
            self.instantaneous = True
        else:
            self.onset_frame = int(onset_slice[min(onset_thresh, len(onset_slice) - 1)])

        if len(offset_slice) == 0:
            self.offset_frame = int(offset_zero_frame)
        else:
            self.offset_frame = int(offset_slice[min(offset_thresh, len(offset_slice) - 1)])

        # Working values reused by the origin step.
        self._onset_zero_frame = int(onset_zero_frame)
        self._peak_frame_start = int(peak_frame_start)

    def _compute_origin(self, spike_data, frameno, channelmap, frames_per_bin):
        """Origin channel counts + (unweighted) origin location, from a single origin-window slice.

        Origin spikes are those from the left anchor to onset, or within the peak bin if the onset was
        instantaneous.
        """
        if self.instantaneous:
            lo, hi = self._peak_frame_start, self._peak_frame_start + frames_per_bin
        else:
            lo, hi = self._onset_zero_frame, self.onset_frame
        i0, i1 = _window(frameno, lo, hi)
        channels = np.asarray(spike_data["channel"][i0:i1], dtype=int)

        counts = {int(k): int(v) for k, v in Counter(channels).items()}
        self.origin_chan_counts = counts

        chanmap_slice = channelmap[np.isin(channelmap[:, 1], list(counts))]
        self.origin_x = float(np.mean(chanmap_slice[:, 3])) if len(chanmap_slice) else -1.0
        self.origin_y = float(np.mean(chanmap_slice[:, 4])) if len(chanmap_slice) else -1.0

        # weighted_origin -- disabled by default. Uncomment here and the _COLUMNS entries to re-enable:
        # if counts:
        #     weights = np.array([counts[int(ch)] for ch in chanmap_slice[:, 1]], dtype=float)
        #     total = weights.sum()
        #     self.weighted_origin_x = float(np.sum(chanmap_slice[:, 3] * weights) / total)
        #     self.weighted_origin_y = float(np.sum(chanmap_slice[:, 4] * weights) / total)

    def _compute_body(self, spike_data, frameno, channelmap):
        """Constituent channels, burst size, and peak location, from a single onset->offset slice."""
        i0, i1 = _window(frameno, self.onset_frame, self.offset_frame)
        channels = np.asarray(spike_data["channel"][i0:i1], dtype=int)

        self.constituent_chan_counts = {int(k): int(v) for k, v in Counter(channels).items()}
        n_mapped = len(channelmap[:, 1])
        self.size_frac_elec = len(self.constituent_chan_counts) / n_mapped if n_mapped else -1.0

        chanmap_slice = channelmap[np.isin(channelmap[:, 1], list(self.constituent_chan_counts))]
        self.peak_x = float(np.mean(chanmap_slice[:, 3])) if len(chanmap_slice) else -1.0
        self.peak_y = float(np.mean(chanmap_slice[:, 4])) if len(chanmap_slice) else -1.0

    def _stim_in_burst(self, stim_frames):
        if stim_frames is None or len(stim_frames) == 0:
            self.has_stim = False
            return
        stim_frames = np.asarray(stim_frames)
        self.has_stim = bool(
            np.any((stim_frames > self.group_start_frame) & (stim_frames < self.group_stop_frame))
        )
