"""Burst detection and the :class:`BurstSet` container.

:class:`BurstDetector` turns a :class:`Recording <mxtreme.recording.Recording>` into a
:class:`BurstSet` (a list of :class:`~mxtreme.bursting.burst.Burst`). Detection is dispatched by name
so more methods can be added without changing the interface.

Two stages, kept as separate, composable pieces:

- **ISI-N grouping** (:meth:`BurstDetector._isi_n_groups`) -- the burst detector of Bakkum et al. 2013
  (*Parameters for burst detection*, Front. Comput. Neurosci. ``fncom.2013.00193``). ``ISI_N`` is the
  span of ``N`` consecutive spikes; spikes ``S_i .. S_{i+N-1}`` are one burst when that span is below
  an auto threshold taken from the valley of the ``log10(ISI_N)`` distribution. Returns spike
  intervals.
- **Rate thresholding** (:meth:`BurstDetector._rate_threshold_groups`) -- within each group's ASDR
  segment, pick peaks and classify them ``"network"`` / ``"mini"`` by height.

Methods: ``"isi_n"`` runs the grouping stage alone (one burst per group); ``"isi_rate"`` composes both.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import gaussian_kde

from mxtreme import utils
from mxtreme.bursting.burst import Burst, _COLUMNS
from mxtreme.params import BurstDetectParams
from mxtreme.phases import Phases

# Fitting gaussian_kde on every inter-spike interval of a long recording is the dominant detection
# cost; a large seeded subsample locates the same log10(ISI_N) valley far faster and reproducibly.
KDE_MAX_SAMPLES = 100_000


def _rec_label(recording) -> str:
    """Compact ``exp_id/chip/wellN/DIVd`` identity string for progress messages."""
    return f"{recording.exp_id}/{recording.chip}/well{recording.well}/DIV{recording.DIV}"


def _log_burst_set(burst_data_dir, recording, burst_set) -> None:
    """Refresh the per-experiment burst log when ``burst_data_dir`` is given (else do nothing).

    Kept as a small helper so both :meth:`BurstDetector.detect` and :meth:`BurstSet.extract_features`
    write the log the same way. The import is local because :mod:`mxtreme.io` lazily imports this
    module, so importing it at module load would risk a circular import.
    """
    if burst_data_dir is None:
        return
    from mxtreme import io

    log_path = io.update_burst_log(burst_data_dir, recording, burst_set)
    print(f"Burst log updated -> {log_path}")


@dataclass
class BurstSet:
    """A recording's detected bursts plus the parameters that produced them.

    :param bursts: The detected :class:`~mxtreme.bursting.burst.Burst` objects.
    :param method: Detection method name.
    :param detect_params: The :class:`~mxtreme.params.BurstDetectParams` used.
    :param feature_params: The :class:`~mxtreme.params.BurstFeatureParams` used, once features run.
    :param n_ignored: Number of bursts flagged ``ignore`` during feature extraction.
    :param detected_at: ISO timestamp of when detection finished (set by :meth:`BurstDetector.detect`).
    :param features_computed_at: ISO timestamp of when feature extraction finished.
    """

    bursts: list[Burst] = field(default_factory=list)
    method: Optional[str] = None
    detect_params: Optional[BurstDetectParams] = None
    feature_params: Optional[object] = None
    n_ignored: int = 0
    detected_at: Optional[str] = None
    features_computed_at: Optional[str] = None

    def __len__(self) -> int:
        return len(self.bursts)

    def __iter__(self):
        return iter(self.bursts)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all bursts as a DataFrame with the canonical column order.

        Ignored bursts are kept (with sentinel feature values) and flagged by the ``ignore`` column,
        rather than dropped, so nothing is silently lost. Filter on ``df["ignore"] == False`` when you
        want only clean bursts.
        """
        if not self.bursts:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame([b.to_row() for b in self.bursts], columns=_COLUMNS)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "BurstSet":
        return cls(bursts=[Burst.from_row(row) for row in df.to_dict(orient="records")])

    @classmethod
    def from_csv(cls, path) -> "BurstSet":
        return cls.from_dataframe(pd.read_csv(path))

    def extract_features(self, recording, params, *, burst_data_dir=None) -> "BurstSet":
        """Compute per-burst features for every burst in place.

        Precomputes the array-wide rate once, then calls :meth:`Burst.compute_features` on each burst,
        tallying ``n_ignored``.

        :param recording: Source :class:`Recording <mxtreme.recording.Recording>`.
        :param params: :class:`~mxtreme.params.BurstFeatureParams`.
        :param burst_data_dir: If given (typically ``config.burst_data_dir``), the per-experiment burst
            log is refreshed automatically via :func:`mxtreme.io.update_burst_log`, stamping this
            recording's ``features_computed_at`` (independently of detection). Pass ``None`` to skip I/O.
        :returns: ``self``.
        """
        self.feature_params = params
        asdr = recording.asdr

        # Feature windows are sliced by binary search, which requires ascending spike frames. Detection
        # already assumes this; verify once here rather than per burst.
        frameno = recording.spike_data["frameno"]
        if frameno.size > 1 and not np.all(frameno[:-1] <= frameno[1:]):
            raise ValueError(
                "spike_data must be sorted by frameno for burst feature extraction. "
                "Call utils.sort_spike_data on it, or io.repair_spike_order to fix the stored .npz."
            )

        print(f"Extracting features for {len(self.bursts)} burst(s) | {_rec_label(recording)}")

        self.n_ignored = 0
        for burst in self.bursts:
            burst.compute_features(recording, params, asdr=asdr)
            if burst.ignore:
                self.n_ignored += 1
        self.features_computed_at = datetime.now().isoformat(timespec="seconds")

        print(f"Features computed | {self.n_ignored} burst(s) flagged ignore")
        _log_burst_set(burst_data_dir, recording, self)
        return self


class BurstDetector:
    """Detect bursts from a recording's binned + raw spike data.

    :param params: :class:`~mxtreme.params.BurstDetectParams`. Defaults are used if omitted.
    :param method: Detection method name -- ``"isi_rate"`` (grouping + rate peaks) or ``"isi_n"``
        (grouping only, one burst per group).
    """

    def __init__(self, params: Optional[BurstDetectParams] = None, method: str = "isi_rate") -> None:
        self.params = params or BurstDetectParams()
        self.method = method
        # Method dispatch table -- register additional detection methods here in the future.
        self._methods = {
            "isi_rate": self._detect_isi_rate,
            "isi_n": self._detect_isi_n,
        }

    def detect(self, recording, phases: Optional[Phases] = None, *, burst_data_dir=None) -> BurstSet:
        """Detect bursts in ``recording`` and label each by phase.

        :param recording: Source :class:`Recording <mxtreme.recording.Recording>`.
        :param phases: Phase intervals for labelling; defaults to ``recording.phases`` (a single
            ``"full"`` phase unless the user injected event-tag phases).
        :param burst_data_dir: If given (typically ``config.burst_data_dir``), the per-experiment burst
            log is refreshed automatically via :func:`mxtreme.io.update_burst_log`, stamping this
            recording's ``detection_completed_at``. Pass ``None`` to skip all I/O.
        :returns: The detected :class:`BurstSet` (features not yet computed).
        """
        handler = self._methods.get(self.method)
        if handler is None:
            raise ValueError(
                f"Unknown burst-detection method {self.method!r}. "
                f"Available: {sorted(self._methods)}"
            )
        if phases is None:
            phases = recording.phases

        print("=" * 60)
        print(f"Detecting bursts | {_rec_label(recording)} | method={self.method}")
        print("-" * 60)

        bursts = handler(recording, phases)
        bursts.detected_at = datetime.now().isoformat(timespec="seconds")

        kinds = Counter(b.kind for b in bursts)
        breakdown = ", ".join(f"{n} {kind}" for kind, n in kinds.items()) if kinds else "none"
        print(f"Detected {len(bursts)} burst(s): {breakdown}")
        _log_burst_set(burst_data_dir, recording, bursts)
        print("=" * 60)
        return bursts

    # ------------------------------------------------------------------ methods

    def _detect_isi_n(self, recording, phases: Phases) -> BurstSet:
        """ISI-N grouping only: one burst per group, peak at the group's max-ASDR bin."""
        samp_rate = float(recording.samp_rate)
        bin_size = float(recording.bin_size)
        asdr = recording.asdr

        groups = self._isi_n_groups(recording.spike_data["frameno"], samp_rate, bin_size)

        bursts: list[Burst] = []
        for i, (group_start, group_stop, n_spikes) in enumerate(groups, start=1):
            start_bin = utils.frame_to_bin(group_start, samp_rate, bin_size)
            stop_bin = utils.frame_to_bin(group_stop, samp_rate, bin_size)
            segment = asdr[start_bin:stop_bin + 1]
            if len(segment) == 0:
                peak_bin, peak_amp = start_bin, 0.0
            else:
                offset = int(np.argmax(segment))
                peak_bin, peak_amp = start_bin + offset, float(segment[offset])
            peak_frame = utils.bin_to_frame(peak_bin, samp_rate, bin_size)
            bursts.append(
                Burst(
                    id=i, peak_frame=peak_frame, peak_amp=peak_amp, kind="burst",
                    group_start_frame=group_start, group_stop_frame=group_stop, n_spikes=n_spikes,
                    phase=phases.label_for(peak_frame),
                )
            )

        return BurstSet(bursts=bursts, method=self.method, detect_params=self.params)

    def _detect_isi_rate(self, recording, phases: Phases) -> BurstSet:
        """ISI-N grouping followed by rate-threshold peak picking within each group."""
        samp_rate = float(recording.samp_rate)
        bin_size = float(recording.bin_size)

        groups = self._isi_n_groups(recording.spike_data["frameno"], samp_rate, bin_size)
        bursts = self._rate_threshold_groups(recording.asdr, groups, phases, samp_rate, bin_size)

        return BurstSet(bursts=bursts, method=self.method, detect_params=self.params)

    # ------------------------------------------------------------------ stages

    def _isi_n_groups(self, spike_frames, samp_rate, bin_size) -> list[tuple[int, int, int]]:
        """Group spikes into bursts by the ISI-N criterion (Bakkum et al. 2013).

        :returns: ``(start_frame, stop_frame, n_spikes)`` per burst, boundaries at the first/last
            spike of the group (per the paper).
        """
        N = self.params.n
        if len(spike_frames) <= N:
            print("Too few spikes in recording to compute ISI-N.")
            return []

        thresh_frames = self._isi_n_threshold(spike_frames, samp_rate)
        if thresh_frames is None:  # ISI-N distribution not bimodal -> no separable bursts
            return []

        burst_number = self._group_spikes(spike_frames, N, thresh_frames)
        if burst_number.max() < 1:
            return []

        # Each positive group id labels one contiguous run of spikes; recover run extents in one pass
        # (avoid an O(n_bursts x n_spikes) per-group scan on long recordings).
        boundaries = np.nonzero(np.diff(burst_number))[0]
        run_starts = np.concatenate(([0], boundaries + 1))
        run_ends = np.concatenate((boundaries, [len(burst_number) - 1]))
        groups = []
        for s_i, e_i in zip(run_starts, run_ends):
            if burst_number[s_i] > 0:
                groups.append((int(spike_frames[s_i]), int(spike_frames[e_i]), int(e_i - s_i + 1)))
        return groups

    def _isi_n_threshold(self, spike_frames, samp_rate) -> Optional[float]:
        """Auto ISI-N threshold (in frames) from the valley of the ``log10(ISI_N)`` distribution.

        ``ISI_N`` is the span of ``N`` consecutive spikes: ``T[i+(N-1)] - T[i]`` (Bakkum et al. 2013).
        The distribution is bimodal (bursting = low, tonic = high); the threshold is the valley between
        the first two modes.
        """
        N = self.params.n
        isi_n = (spike_frames[N - 1:] - spike_frames[:-(N - 1)]) / samp_rate  # span of N spikes
        isi_n = isi_n[isi_n > 0]
        if len(isi_n) < 2:
            return None

        log_isi = np.log10(isi_n)
        if len(log_isi) > KDE_MAX_SAMPLES:  # subsample (seeded) so the KDE stays fast + reproducible
            sample = np.random.default_rng(0).choice(log_isi, KDE_MAX_SAMPLES, replace=False)
        else:
            sample = log_isi

        kde = gaussian_kde(sample)
        xs = np.linspace(log_isi.min(), log_isi.max(), 400)
        ys = kde(xs)
        peaks, _ = scipy.signal.find_peaks(ys)
        if len(peaks) < 2:
            return None

        p1, p2 = peaks[:2]  # valley between the first two modes ("first minima", per the paper)
        # Alternative -- valley between the two *most prominent* modes (uncomment to A/B test):
        # top2 = peaks[np.argsort(ys[peaks])[-2:]]; p1, p2 = sorted(top2)
        valley = np.argmin(ys[p1:p2]) + p1
        return float(10 ** xs[valley] * samp_rate)

    def _rate_threshold_groups(self, asdr, groups, phases, samp_rate, bin_size) -> list[Burst]:
        """Pick and classify rate peaks within each ISI-N group.

        Peaks at/above ``burst_thresh`` are ``"network"``; peaks between ``noise_thresh`` and
        ``burst_thresh`` are ``"mini"``.
        """
        p = self.params
        bursts: list[Burst] = []
        next_id = 1
        for group_start, group_stop, n_spikes in groups:
            start_bin = utils.frame_to_bin(group_start, samp_rate, bin_size)
            stop_bin = utils.frame_to_bin(group_stop, samp_rate, bin_size)
            segment = asdr[start_bin:stop_bin + 1]

            network_peaks, network_props = scipy.signal.find_peaks(
                segment, height=p.burst_thresh, distance=p.min_dist_bins
            )
            mini_peaks, mini_props = scipy.signal.find_peaks(
                segment, height=(p.noise_thresh, p.burst_thresh), distance=p.min_dist_bins
            )

            for kind, pk, heights in (
                ("network", network_peaks, network_props["peak_heights"]),
                ("mini", mini_peaks, mini_props["peak_heights"]),
            ):
                for offset, height in zip(pk, heights):
                    peak_frame = utils.bin_to_frame(offset + start_bin, samp_rate, bin_size)
                    bursts.append(
                        Burst(
                            id=next_id, peak_frame=peak_frame, peak_amp=float(height), kind=kind,
                            group_start_frame=group_start, group_stop_frame=group_stop,
                            n_spikes=n_spikes, phase=phases.label_for(peak_frame),
                        )
                    )
                    next_id += 1
        return bursts

    @staticmethod
    def _group_spikes(spike_frames, N, isi_thresh_frames):
        """Assign each spike a burst-group number via the ISI-N criterion.

        Port of the ISI-N MATLAB algorithm (Bakkum et al. 2013): a spike is "in burst" if the shortest
        window of ``N`` consecutive spikes covering it spans at most ``isi_thresh_frames``. Runs of
        fewer than ``N`` in-burst spikes are discarded, and consecutive bursts not separated by tonic
        spikes are split.
        """
        n_spikes = len(spike_frames)

        # "In-burst" membership: a spike c qualifies iff some N-spike window covering it is short
        # enough. span[s] = width of the window starting at spike s; a qualifying window (span<=thresh)
        # covers spikes [s, s+N-1]. Marking those coverage intervals with an O(n) difference array is
        # equivalent to -- and far cheaper than -- the old (N x n_spikes) min-over-windows matrix.
        span = spike_frames[N - 1:] - spike_frames[:-(N - 1)]          # len n_spikes-(N-1)
        starts = np.nonzero(span <= isi_thresh_frames)[0]             # qualifying window starts
        diff = np.zeros(n_spikes + 1, dtype=int)
        np.add.at(diff, starts, 1)
        np.add.at(diff, starts + N, -1)
        covered = np.cumsum(diff[:-1]) > 0
        criteria = np.zeros(n_spikes, dtype=bool)
        # Restrict to the same valid column range the matrix version filled.
        criteria[N - 1: n_spikes - (N - 1)] = covered[N - 1: n_spikes - (N - 1)]

        burst_number = np.full(n_spikes, -1, dtype=int)
        in_burst = False
        num = 0
        number = -1
        length = 0

        for i in range(N - 1, n_spikes):
            if not in_burst:
                if criteria[i]:
                    in_burst = True
                    num += 1
                    number = num
                    length = 1
            else:
                if not criteria[i]:
                    in_burst = False
                    if length < N:  # too short -> discard
                        burst_number[burst_number == number] = -1
                        num -= 1
                    number = -1
                elif (spike_frames[i] - spike_frames[i - (N - 1)] > isi_thresh_frames) and (length >= N):
                    # split consecutive bursts not separated by tonic spikes
                    num += 1
                    number = num
                    length = 1
                else:
                    length += 1
            burst_number[i] = number

        return burst_number
