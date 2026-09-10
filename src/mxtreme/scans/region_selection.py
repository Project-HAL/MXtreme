"""Choose where an experiment's stimulation regions go, and check they start out independent.

An experiment that stimulates a few patches of the array and reads the reply in another -- HAL's
associative conditioning, say -- needs patches that are each well populated, far enough apart that a
pulse in one does not land in another, and not already moving together before any training. This
module gets that from the two recordings every culture has anyway:

1. :func:`choose_centers` -- from an activity scan (loaded and thresholded by
   :mod:`mxtreme.scans.electrode_selection`), the densest patches a minimum distance apart.
2. :func:`recording_electrodes` -- the electrodes to route: every active electrode inside each patch,
   the rest of the budget spread over the array so a population-burst detector is not biased toward
   wherever the culture is densest.
3. :func:`baseline_spikes`, :func:`network_bursts`, :func:`region_correlations`,
   :func:`cross_correlograms` and :func:`burst_onset_lags` -- from a short network scan recorded with those electrodes and no
   stimulation, how coupled each pair of patches already is: their correlation *outside* network
   bursts, and how consistently one leads the other *into* them. :func:`coupling` folds the two
   into one matrix.
4. :func:`assign_roles` -- the least-coupled, most equilateral triple whose two conditioned sites
   are matched in baseline coupling to the readout, with roles attached so that CS and NS sit
   at matched distances from US and which is CS is counterbalanced, not chosen by the data.

Everything here is offline analysis: no ``maxlab``. Recording the scans is
:mod:`mxtreme.scans.activity_scan` and :mod:`mxtreme.scans.network_scan`.

Typical use::

    from mxtreme.scans import electrode_selection, region_selection

    data = electrode_selection.get_active_electrodes(electrode_selection.load_activity_scan(scan_h5))
    centres = region_selection.choose_centers(data[well], n=4, min_separation_um=1000)
    electrodes = region_selection.recording_electrodes(data[well], centres, radius_um=150)
    # ... record a network scan on sum(electrodes.values(), []) ...
    frames, elecs, fps = region_selection.baseline_spikes(baseline_h5, well)
    bursts = region_selection.network_bursts(frames, fps)
    names, corr = region_selection.region_correlations(frames, elecs, patches, fps, exclude=bursts)
    _, lag_ms, lead, used = region_selection.burst_onset_lags(frames, elecs, patches, fps, bursts)
    roles = region_selection.assign_roles(centres, region_selection.coupling(corr, lead), density)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import combinations

import numpy as np

from mxtreme import device

PITCH_UM = float(device.ELECTRODE_PITCH_UM) if hasattr(device, "ELECTRODE_PITCH_UM") else 17.5
COLS, ROWS = device.CHIP_WIDTH, device.CHIP_HEIGHT

#: The roles the associative experiment assigns, in the order :func:`assign_roles` fills them.
ROLES = ("US", "CS", "NS")


def scan_visits(well_data):
    """How many of a scan's rounds recorded each electrode, from the merged mapping table.

    :func:`mxtreme.scans.electrode_selection.load_activity_scan` concatenates every round's
    channel map, so an electrode routed in three rounds appears three times.

    :returns: A pandas Series indexed by electrode.
    """
    return well_data["mapping"].groupby("electrode").size()


def active_electrodes(well_data, min_rate_hz: float = 0.1, min_amplitude_uv: float = 20.0) -> dict:
    """Active electrodes from a scan, counting each electrode's own recorded time.

    Same two criteria as :func:`mxtreme.scans.electrode_selection.get_active_electrodes` -- a 90th
    percentile spike amplitude above ``min_amplitude_uv`` and a firing rate above ``min_rate_hz``
    -- with one correction. A scan asking for more than one full coverage of the array records
    some electrodes in several rounds (see
    :func:`mxtreme.scans.activity_scan.plan_scan_electrodes`), and their spikes are then summed
    across rounds while the rate is divided by a single round's length, so those electrodes look
    proportionally more active than they are. Dividing by each electrode's own number of rounds
    removes that. On a single-round recording the two agree exactly.

    :param well_data: One well from ``load_activity_scan``, or :func:`well_data_from_npz`.
    :returns: The same dict with ``active_electrodes`` set.
    """
    spikes = well_data["spike_data"]
    spikes = spikes[spikes["amplitude"] < 0].copy()
    lsb = well_data["lsb"]
    by_amplitude = spikes.groupby("electrode").filter(
        lambda g: abs(np.percentile(g["amplitude"], 90) * lsb * 1e6) > min_amplitude_uv
    )
    visits = scan_visits(well_data)
    seconds = well_data["rec_length_sec"]
    by_rate = by_amplitude.groupby("electrode").filter(
        lambda g: len(g) / (max(1, int(visits.get(g.name, 1))) * seconds) > min_rate_hz
    )
    well_data["active_electrodes"] = by_rate.copy()
    return well_data


def _active_xy(well_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electrode numbers and positions of a well's active electrodes.

    :param well_data: One well's dict from :func:`mxtreme.scans.electrode_selection.get_active_electrodes`.
    """
    active = np.unique(well_data["active_electrodes"]["electrode"].to_numpy())
    mapping = well_data["mapping"]
    rows = mapping[mapping["electrode"].isin(active)].drop_duplicates("electrode")
    return (
        rows["electrode"].to_numpy().astype(int),
        rows["x"].to_numpy().astype(float),
        rows["y"].to_numpy().astype(float),
    )


def density_map(
    well_data, radius_um: float, step_electrodes: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count electrodes within ``radius_um`` of each point of a coarse grid over the array.

    Two counts per point: active electrodes, and electrodes recorded at all. The second is what
    the first has to be judged against -- an activity scan records the whole array, a network scan
    only its thousand chosen electrodes, and "30 active in a patch" means something different in
    each.

    :param well_data: One well's dict, thresholded by ``get_active_electrodes``.
    :param radius_um: Patch radius.
    :param step_electrodes: Grid spacing, in electrodes.
    :returns: ``(centres, active, recorded)``: an ``(N, 2)`` array of grid points in um, and the
        two counts at each.
    """
    _, x, y = _active_xy(well_data)
    mapping = well_data["mapping"].drop_duplicates("electrode")
    rx, ry = mapping["x"].to_numpy(dtype=float), mapping["y"].to_numpy(dtype=float)
    gx = np.arange(0, COLS, step_electrodes) * PITCH_UM
    gy = np.arange(0, ROWS, step_electrodes) * PITCH_UM
    centres = np.array([(cx, cy) for cy in gy for cx in gx])
    active = np.zeros(len(centres), dtype=int)
    recorded = np.zeros(len(centres), dtype=int)
    for i, (cx, cy) in enumerate(centres):
        active[i] = np.count_nonzero((x - cx) ** 2 + (y - cy) ** 2 <= radius_um**2)
        recorded[i] = np.count_nonzero((rx - cx) ** 2 + (ry - cy) ** 2 <= radius_um**2)
    return centres, active, recorded


def rank_patches(
    well_data,
    n: int = 3,
    min_separation_um: float = 1000.0,
    radius_um: float = 150.0,
    min_active: int | None = None,
    margin_um: float = 100.0,
    min_active_fraction: float = 0.25,
    min_active_floor: int = 5,
) -> list[dict]:
    """Every grid point of the density map with what :func:`choose_centers` made of it.

    The choice is greedy: densest first, then the densest that is far enough from everything
    chosen so far. This returns the whole walk, densest to sparsest, so a report can show not just
    the winners but the denser patches that lost and why. Each entry is
    ``{"center_um", "count", "recorded", "status"}`` with status one of ``"chosen"``,
    ``"too close to <k>"`` (a chosen patch's index), ``"too few active"`` or ``"edge"``. Points past
    the point where ``n`` were chosen are marked ``"not needed"``.

    A patch qualifies when at least ``min_active_fraction`` of the electrodes *recorded* inside it
    are active, and at least ``min_active_floor`` are -- or, when ``min_active`` is given, when
    that many are, regardless of how many were recorded. The fraction is the default because it
    means the same thing for a whole-array activity scan and for a thousand-electrode network scan.

    :param well_data: One well's dict, thresholded by ``get_active_electrodes``.
    :param n: Patches wanted.
    :param min_separation_um: Minimum centre-to-centre distance.
    :param radius_um: Patch radius, for counting.
    :param min_active: An absolute count that overrides the fraction rule.
    :param margin_um: Keep centres this far from the array's edge, so a stimulation site fits.
    :param min_active_fraction: Active over recorded, within the patch.
    :param min_active_floor: Fewest active electrodes whatever the fraction.
    """
    centres, counts, recorded = density_map(well_data, radius_um)
    if min_active is not None:
        needed = np.full(len(centres), int(min_active))
    else:
        needed = np.maximum(min_active_floor, np.ceil(min_active_fraction * recorded)).astype(int)
    inside = (
        (centres[:, 0] >= margin_um)
        & (centres[:, 0] <= (COLS - 1) * PITCH_UM - margin_um)
        & (centres[:, 1] >= margin_um)
        & (centres[:, 1] <= (ROWS - 1) * PITCH_UM - margin_um)
    )
    order = np.argsort(-counts, kind="stable")

    chosen: list[tuple[float, float]] = []
    ranked: list[dict] = []
    for i in order:
        cx, cy = float(centres[i][0]), float(centres[i][1])
        entry = {"center_um": (cx, cy), "count": int(counts[i]), "recorded": int(recorded[i])}
        if len(chosen) == n:
            entry["status"] = "not needed"
        elif counts[i] < needed[i]:
            entry["status"] = "too few active"
        elif not inside[i]:
            entry["status"] = "edge"
        else:
            near = [
                k
                for k, (px, py) in enumerate(chosen)
                if (cx - px) ** 2 + (cy - py) ** 2 < min_separation_um**2
            ]
            if near:
                entry["status"] = f"too close to {near[0]}"
            else:
                chosen.append((cx, cy))
                entry["status"] = "chosen"
        ranked.append(entry)
    return ranked


def choose_centers(
    well_data,
    n: int = 3,
    min_separation_um: float = 1000.0,
    radius_um: float = 150.0,
    min_active: int | None = None,
    margin_um: float = 100.0,
) -> list[tuple[float, float]]:
    """The ``n`` densest patches that are at least ``min_separation_um`` apart.

    Greedy: the densest patch first, then the densest patch far enough from every patch chosen so
    far (the full walk is :func:`rank_patches`). Density is the only tie-breaker between
    far-enough candidates, deliberately: whether two patches are already coupled is a question for
    the baseline recording (:func:`region_correlations`), not something to guess from a scan. Ask
    for one or two more than you need, and let :func:`assign_roles` drop the coupled one.

    :raises ValueError: If fewer than ``n`` patches qualify.
    :returns: Centres in um, densest first.
    """
    ranked = rank_patches(well_data, n, min_separation_um, radius_um, min_active, margin_um)
    chosen = [r["center_um"] for r in ranked if r["status"] == "chosen"]
    if len(chosen) < n:
        raise ValueError(
            f"only {len(chosen)} patches with enough active electrodes and >= "
            f"{min_separation_um} um apart; lower the separation, the threshold or the radius"
        )
    return chosen


def well_data_from_npz(path: str) -> dict:
    """A preprocessed ``.npz`` (:func:`mxtreme.io.load_preprocessed` layout) as the per-well dict
    :func:`mxtreme.scans.electrode_selection.get_active_electrodes` takes.

    A network scan or any unstimulated recording of enough of the array will do in place of an
    activity scan for choosing regions; this is how one is fed in. Amplitudes in the ``.npz`` are
    already volts, so ``lsb`` is 1.

    :param path: The ``.npz``.
    """
    import pandas as pd

    d = np.load(path, allow_pickle=True)
    channelmap = d["channelmap"]  # (index, channel, electrode, x, y)
    mapping = pd.DataFrame(
        {
            "channel": channelmap[:, 1].astype(int),
            "electrode": channelmap[:, 2].astype(int),
            "x": channelmap[:, 3],
            "y": channelmap[:, 4],
        }
    )
    spikes = pd.DataFrame(d["spike_data"])
    spikes = spikes.merge(mapping[["channel", "electrode"]], on="channel").sort_values("frameno")
    return {
        "spike_data": spikes,
        "mapping": mapping,
        "samp_rate": float(np.asarray(d["samp_rate"]).ravel()[0]),
        "lsb": 1.0,
        "rec_length_sec": float(np.asarray(d["rec_t_sec"]).ravel()[0]),
    }


def patch_density(well_data, centres: Sequence[tuple[float, float]], radius_um: float) -> list[int]:
    """Active electrodes within ``radius_um`` of each centre."""
    _, x, y = _active_xy(well_data)
    return [int(np.count_nonzero((x - cx) ** 2 + (y - cy) ** 2 <= radius_um**2)) for cx, cy in centres]


def recording_electrodes(
    well_data,
    centres: Sequence[tuple[float, float]],
    radius_um: float,
    budget: int = 1020,
    reserve: int = 32,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Which electrodes to route: every active electrode inside each patch, then the rest of the
    budget spread over the remaining active electrodes.

    The spread is furthest-point-first rather than random, so the "global" set covers the whole
    array evenly; a burst detector counting a fraction of routed channels is then not biased toward
    the densest corner.

    :param well_data: One well's dict, thresholded by ``get_active_electrodes``.
    :param centres: Patch centres in um.
    :param radius_um: Patch radius.
    :param budget: Electrodes the chip can route.
    :param reserve: Held back for the stimulation electrodes the experiment adds itself.
    :param seed: For the starting point of the spread.
    :raises ValueError: If the patches alone exceed the budget.
    :returns: ``{"patch_0": [...], ..., "global": [...]}``.
    """
    rng = np.random.default_rng(seed)
    electrodes, x, y = _active_xy(well_data)
    taken = np.zeros(len(electrodes), dtype=bool)
    out: dict[str, list[int]] = {}

    for k, (cx, cy) in enumerate(centres):
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius_um**2) & ~taken
        out[f"patch_{k}"] = electrodes[mask].tolist()
        taken |= mask

    remaining = budget - reserve - int(taken.sum())
    if remaining < 0:
        raise ValueError(
            f"the patches alone use {int(taken.sum())} electrodes, over the budget of {budget - reserve}"
        )

    left = np.flatnonzero(~taken)
    picked: list[int] = []
    if len(left) and remaining > 0:
        dist = np.full(len(left), np.inf)
        current = int(rng.choice(left))
        for _ in range(min(remaining, len(left))):
            picked.append(current)
            d = (x[left] - x[current]) ** 2 + (y[left] - y[current]) ** 2
            dist = np.minimum(dist, d)
            dist[left == current] = -1
            current = int(left[int(np.argmax(dist))])
    out["global"] = electrodes[picked].tolist()
    return out


def baseline_spikes(h5_path: str, well: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Spike times and electrodes from one continuous recording, for the coupling measures.

    Reads the same datasets :func:`mxtreme.extract.extract` does, without the metadata it insists
    on: a baseline for region selection is a scratch recording, not one headed for the store.

    **Not an activity scan.** A scan covers the array in a series of short recordings, each
    routing a different subset of electrodes, so two electrodes from different rounds were never
    observed at the same time and no correlation between them means anything. This refuses a file
    holding more than one recording rather than silently reading the first round, which would be
    one arbitrary eighth of the array. Record the baseline with the candidate patches routed --
    :func:`mxtreme.scans.network_scan.run_network_scan` does exactly that.

    :param h5_path: A ``.raw.h5`` of a single continuous recording with the candidate electrodes
        routed, or a preprocessed ``.npz`` of one well (then ``well`` is ignored).
    :param well: Well number.
    :raises ValueError: If the file holds more than one recording.
    :returns: ``(frames, electrodes, fps)``: one entry per spike, and the sampling rate.
    """
    if str(h5_path).endswith(".npz"):
        well_data = well_data_from_npz(str(h5_path))
        spikes = well_data["spike_data"]
        return (
            spikes["frameno"].to_numpy().astype(np.int64),
            spikes["electrode"].to_numpy().astype(int),
            float(well_data["samp_rate"]),
        )

    import h5py

    with h5py.File(h5_path, "r") as f:
        recordings = sorted(k for k in f.get("recordings", {}) if k.startswith("rec"))
        if len(recordings) > 1:
            raise ValueError(
                f"{os.path.basename(str(h5_path))} holds {len(recordings)} recordings, so it is a "
                "scan rather than one continuous recording. Each round routes a different subset "
                "of electrodes, so electrodes from different rounds were never recorded at the "
                "same moment and their correlation would be meaningless. Use a scan to choose "
                "candidate patches, then record a baseline with those patches routed and pass "
                "that here."
            )
        rec = f[f"/recordings/rec0000/well{well:03d}"]
        spikes = rec["spikes"][:]
        mapping = rec["settings/mapping"][:]
        fps = float(np.asarray(rec["settings/sampling"][:]).ravel()[0])

    electrode_of = dict(zip(mapping["channel"].astype(int), mapping["electrode"].astype(int)))
    channels = spikes["channel"].astype(int)
    electrodes = np.array([electrode_of.get(int(c), -1) for c in channels])
    keep = electrodes >= 0
    return spikes["frameno"][keep].astype(np.int64), electrodes[keep], fps


def network_bursts(
    spike_frames: np.ndarray,
    fps: float,
    bin_ms: float = 10.0,
    threshold_sd: float = 3.0,
    margin_ms: float = 50.0,
    min_gap_ms: float = 100.0,
) -> list[tuple[int, int]]:
    """Network bursts in a recording, as ``[(start_frame, end_frame)]``.

    The population rate in ``bin_ms`` bins, thresholded at ``threshold_sd`` standard deviations
    above its mean; runs of bins over the threshold are bursts, widened by ``margin_ms`` on each
    side so their edges are covered too, and runs closer than ``min_gap_ms`` are one burst. A
    plain detector, on purpose: it only has to say which stretches of the recording are
    burst-dominated, so that :func:`region_correlations` can leave them out.

    :param spike_frames: Frame of every spike in the recording, any electrode.
    :param fps: Sampling rate.
    """
    if len(spike_frames) == 0:
        return []
    bin_frames = max(1, round(bin_ms / 1000.0 * fps))
    first = int(spike_frames.min())
    counts = np.bincount(((spike_frames - first) // bin_frames).astype(int))
    over = counts > counts.mean() + threshold_sd * counts.std()
    if not over.any():
        return []

    margin = round(margin_ms / bin_ms)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], over.astype(int), [0]])))
    runs = [(int(a) - margin, int(b) - 1 + margin) for a, b in zip(edges[::2], edges[1::2])]

    merged: list[list[int]] = []
    gap = round(min_gap_ms / bin_ms)
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(first + max(0, a) * bin_frames, first + (b + 1) * bin_frames) for a, b in merged]


def _binned(spike_frames, spike_electrodes, regions, bin_frames, first, num_bins):
    series = np.zeros((len(regions), num_bins))
    for i, name in enumerate(regions):
        member = np.isin(spike_electrodes, np.asarray(list(regions[name])))
        bins = (spike_frames[member] - first) // bin_frames
        np.add.at(series[i], bins.astype(int), 1)
    return series


def region_correlations(
    spike_frames: np.ndarray,
    spike_electrodes: np.ndarray,
    regions: dict[str, Sequence[int]],
    fps: float,
    bin_ms: float = 10.0,
    exclude: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pearson correlation between the regions' binned spike counts.

    From a recording with no stimulation. Two regions that already move together are a poor pair
    to associate: any coupling measured after training would be confounded with what was there
    before. A region with no spikes at all correlates as ``nan``.

    Pass the network bursts (:func:`network_bursts`) as ``exclude`` and the bins inside them are
    left out. Without that, in a bursting culture every pair of patches correlates strongly
    because every burst lifts them all at once, and the number says how bursty the culture is
    rather than how specifically two patches are tied.

    :param spike_frames: Frame of each spike.
    :param spike_electrodes: Electrode of each spike.
    :param regions: ``{name: electrodes}``.
    :param fps: Sampling rate.
    :param bin_ms: Bin width.
    :param exclude: ``[(start_frame, end_frame)]`` stretches to leave out.
    :returns: Region names, and the ``n x n`` correlation matrix in that order.
    """
    names = list(regions)
    if len(spike_frames) == 0:
        return names, np.full((len(names), len(names)), np.nan)
    bin_frames = max(1, round(bin_ms / 1000.0 * fps))
    first, last = int(spike_frames.min()), int(spike_frames.max())
    num_bins = (last - first) // bin_frames + 1
    series = _binned(spike_frames, spike_electrodes, regions, bin_frames, first, num_bins)

    keep = np.ones(num_bins, dtype=bool)
    for a, b in exclude or []:
        keep[max(0, (a - first) // bin_frames) : max(0, (b - first) // bin_frames + 1)] = False
    if keep.sum() < 2:
        return names, np.full((len(names), len(names)), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(series[:, keep])
    return names, np.atleast_2d(matrix)


def patch_rates(
    spike_frames: np.ndarray,
    spike_electrodes: np.ndarray,
    regions: dict[str, Sequence[int]],
    fps: float,
    bin_ms: float = 100.0,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Each region's firing rate over the recording, for looking at it.

    :returns: Region names, ``t_sec`` at each bin's left edge, and rates in spikes per second,
        shaped ``(n_regions, n_bins)``.
    """
    names = list(regions)
    if len(spike_frames) == 0:
        return names, np.zeros(0), np.zeros((len(names), 0))
    bin_frames = max(1, round(bin_ms / 1000.0 * fps))
    first, last = int(spike_frames.min()), int(spike_frames.max())
    num_bins = (last - first) // bin_frames + 1
    series = _binned(spike_frames, spike_electrodes, regions, bin_frames, first, num_bins)
    t = np.arange(num_bins) * bin_frames / fps
    return names, t, series / (bin_frames / fps)


def cross_correlograms(
    spike_frames: np.ndarray,
    spike_electrodes: np.ndarray,
    regions: dict[str, Sequence[int]],
    fps: float,
    bin_ms: float = 5.0,
    max_lag_ms: float = 200.0,
    exclude: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Normalized cross-correlation between every pair of regions, by lag.

    ``corr[i, j, k]`` is the correlation between region ``i`` at time ``t`` and region ``j`` at
    ``t + lags_ms[k]``, so a peak at a positive lag means **i leads j**. The value at lag 0 is what
    :func:`region_correlations` reports at that bin width.

    With ``exclude``, the excluded bins are set to the series mean before correlating, so they
    contribute nothing: an approximation (the bins on either side of a gap are still neighbours)
    but enough to show whether a pair's correlation survives leaving the network bursts out.

    :returns: Region names, the lags in ms, and ``corr`` shaped ``(n, n, n_lags)``.
    """
    names = list(regions)
    n = len(names)
    max_lag_bins = max(1, round(max_lag_ms / bin_ms))
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_ms
    if len(spike_frames) == 0:
        return names, lags, np.full((n, n, len(lags)), np.nan)

    bin_frames = max(1, round(bin_ms / 1000.0 * fps))
    first, last = int(spike_frames.min()), int(spike_frames.max())
    num_bins = (last - first) // bin_frames + 1
    series = _binned(spike_frames, spike_electrodes, regions, bin_frames, first, num_bins)

    keep = np.ones(num_bins, dtype=bool)
    for a, b in exclude or []:
        keep[max(0, (a - first) // bin_frames) : max(0, (b - first) // bin_frames + 1)] = False
    if keep.sum() < 2 or num_bins <= max_lag_bins:
        return names, lags, np.full((n, n, len(lags)), np.nan)

    x = series.astype(float)
    x -= x[:, keep].mean(axis=1, keepdims=True)
    x[:, ~keep] = 0.0
    norm = np.sqrt((x**2).sum(axis=1))

    corr = np.full((n, n, len(lags)), np.nan)
    centre = num_bins - 1
    for i in range(n):
        for j in range(n):
            if norm[i] == 0 or norm[j] == 0:
                continue
            full = np.correlate(x[i], x[j], mode="full")
            # full[centre + m] = sum_t x_i[t] x_j[t - m], so lag L = -m.
            window = full[centre - max_lag_bins : centre + max_lag_bins + 1][::-1]
            corr[i, j] = window / (norm[i] * norm[j])
    return names, lags, corr


def correlogram_peak(lags_ms: np.ndarray, corr: np.ndarray, min_z: float = 4.0) -> tuple[float, float, bool]:
    """The peak of one cross-correlogram, and whether it stands above the correlogram's own noise.

    The noise is taken from the outer thirds of the lag range, where a pair with a real short-lag
    interaction should be flat. A peak below ``min_z`` robust standard deviations of that is not
    evidence of anything: in a quiet culture the correlogram wanders at a few percent and its
    argmax lands wherever the wander happens to be highest.

    :returns: ``(lag_ms, height, stands_out)``.
    """
    corr = np.asarray(corr, dtype=float)
    if not np.any(np.isfinite(corr)):
        return 0.0, float("nan"), False
    k = int(np.nanargmax(corr))
    lag, height = float(lags_ms[k]), float(corr[k])
    outer = corr[np.abs(lags_ms) > (2 / 3) * np.nanmax(np.abs(lags_ms))]
    outer = outer[np.isfinite(outer)]
    if len(outer) < 5:
        return lag, height, False
    centre = float(np.median(outer))
    spread = float(np.median(np.abs(outer - centre))) * 1.4826
    if spread <= 0:
        return lag, height, False
    return lag, height, (height - centre) / spread >= min_z


def _onset(times: np.ndarray, fps: float, min_spikes: int = 3, window_ms: float = 20.0) -> float:
    """When a region joins a burst: its first spike that has ``min_spikes - 1`` more within
    ``window_ms``. The burst windows carry a margin, so a lone background spike just before the
    burst must not count as the region's onset. Falls back to the first spike when the region
    fires fewer than ``min_spikes`` times in the burst."""
    if len(times) < min_spikes:
        return float(times[0])
    window = window_ms / 1000.0 * fps
    ahead = times[min_spikes - 1 :] - times[: len(times) - min_spikes + 1]
    starts = np.flatnonzero(ahead <= window)
    return float(times[starts[0]]) if len(starts) else float(times[0])


def burst_onsets(
    spike_frames: np.ndarray,
    spike_electrodes: np.ndarray,
    regions: dict[str, Sequence[int]],
    fps: float,
    bursts: Sequence[tuple[int, int]],
) -> tuple[list[str], np.ndarray]:
    """When each region joined each network burst.

    :returns: Region names, and ``onsets`` shaped ``(n_bursts, n_regions)`` in frames, ``nan``
        where a region did not fire in that burst.
    """
    names = list(regions)
    onsets = np.full((len(bursts), len(names)), np.nan)
    for k, (a, b) in enumerate(bursts):
        inside = (spike_frames >= a) & (spike_frames < b)
        for i, name in enumerate(names):
            member = inside & np.isin(spike_electrodes, np.asarray(list(regions[name])))
            if member.any():
                onsets[k, i] = _onset(np.sort(spike_frames[member]), fps)
    return names, onsets


def burst_onset_lags(
    spike_frames: np.ndarray,
    spike_electrodes: np.ndarray,
    regions: dict[str, Sequence[int]],
    fps: float,
    bursts: Sequence[tuple[int, int]],
    min_bursts: int = 5,
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    """How each region enters the network bursts relative to each other.

    A region's onset in a burst is :func:`burst_onsets`. For every pair, ``lag_ms[i, j]`` is the
    median of onset_i minus onset_j over the bursts both took part in, and ``lead[i, j]`` is the
    fraction of those bursts in which ``i`` fired first. A pair that always enters in the same
    order (``lead`` near 0 or 1) is wired that way already; a pair with ``lead`` near 0.5 is not.
    Pairs seen in fewer than ``min_bursts`` bursts are ``nan``.

    :returns: Region names, ``lag_ms``, ``lead``, and how many bursts were usable at all.
    """
    names, onsets = burst_onsets(spike_frames, spike_electrodes, regions, fps, bursts)
    n = len(names)
    lag_ms = np.full((n, n), np.nan)
    lead = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                lag_ms[i, j], lead[i, j] = 0.0, 0.5
                continue
            both = ~np.isnan(onsets[:, i]) & ~np.isnan(onsets[:, j])
            if both.sum() < min_bursts:
                continue
            d = onsets[both, i] - onsets[both, j]
            lag_ms[i, j] = float(np.median(d)) / fps * 1000.0
            lead[i, j] = float(np.mean(d < 0) + 0.5 * np.mean(d == 0))
    usable = int(np.sum(np.sum(~np.isnan(onsets), axis=1) >= 2))
    return names, lag_ms, lead, usable


def coupling(corr: np.ndarray, lead: np.ndarray | None = None) -> np.ndarray:
    """One number per pair in ``[0, 1]`` for :func:`assign_roles`.

    The larger of the out-of-burst correlation's magnitude and the burst-order consistency
    ``|2 * lead - 1|`` (0 when a pair enters bursts in random order, 1 when always the same). A
    pair ``nan`` on both counts stays ``nan``, which :func:`assign_roles` treats as fully coupled.
    """
    corr = np.abs(np.asarray(corr, dtype=float))
    if lead is None:
        return corr
    order = np.abs(2.0 * np.asarray(lead, dtype=float) - 1.0)
    out = np.where(np.isnan(corr), order, np.where(np.isnan(order), corr, np.maximum(corr, order)))
    np.fill_diagonal(out, 1.0)
    return out


def triangle(centres: Sequence[tuple[float, float]]) -> tuple[float, list[float]]:
    """(shortest side / longest side, the three sides in um) of a triple of centres.

    1 is equilateral; a triple in a line, which would put one region between the other two, is
    near 0. The associative design wants CS and NS at similar distances from US and from each
    other, so that "CS evoked US and NS did not" cannot be explained by NS simply being
    further away.
    """
    (ax, ay), (bx, by), (cx, cy) = centres
    sides = [
        float(np.hypot(ax - bx, ay - by)),
        float(np.hypot(bx - cx, by - cy)),
        float(np.hypot(ax - cx, ay - cy)),
    ]
    return (min(sides) / max(sides) if max(sides) > 0 else 0.0), sides


def assign_roles(
    centres: Sequence[tuple[float, float]],
    matrix: np.ndarray,
    density: Sequence[int],
    roles: Sequence[str] = ROLES,
    min_side_ratio: float = 0.6,
    counterbalance: int = 0,
) -> dict[str, tuple[float, float]]:
    """Pick the triple of candidates that is least coupled, closest to equilateral, and whose two
    conditioned sites are matched in their baseline coupling to the readout; name them.

    Triples whose shortest side is under ``min_side_ratio`` of their longest are set aside (all
    of them, if none passes -- then the least elongated are preferred). Among what remains the
    ranking is: smallest largest pairwise coupling; then the smallest *difference* between the two
    CS-to-US couplings, so that CS and NS start from the same footing; then the more equilateral;
    then the denser.

    Within the triple, US is the vertex whose distances to the other two differ least, so the
    conditioned and control stimuli are equally far from the readout. Which of the other two is
    CS is **not** decided by the data: a pair that already shares a pathway is easier to
    potentiate, so letting coupling choose would bias the outcome either way. The two are ordered
    by their coupling to US and ``counterbalance`` (0 or 1) picks which is CS; alternate it
    across cultures. A ``nan`` coupling counts as fully coupled.

    :param centres: Candidate centres, as from :func:`choose_centers`.
    :param matrix: Their pairwise coupling, as from :func:`coupling` (or a correlation matrix).
    :param density: Active electrodes in each, as from :func:`patch_density`.
    :param roles: Names to hand out; ``len(roles)`` candidates are chosen.
    :param min_side_ratio: Shortest over longest side a triple should have.
    :param counterbalance: 0 or 1; which of the two conditioned sites is CS.
    :raises ValueError: If there are fewer candidates than roles.
    :returns: ``{role: centre}``.
    """
    k = len(roles)
    if len(centres) < k:
        raise ValueError(f"{len(centres)} candidates for {k} roles")
    coupled = np.nan_to_num(np.abs(np.asarray(matrix, dtype=float)), nan=1.0)

    def worst(triple):
        return max(coupled[a, b] for a, b in combinations(triple, 2))

    def shape(triple):
        return triangle([centres[i] for i in triple])[0] if k == 3 else 1.0

    def apex(triple):
        # The vertex whose two distances differ least; density breaks ties.
        def imbalance(i):
            others = [j for j in triple if j != i]
            d = [np.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1]) for j in others]
            return abs(d[0] - d[1]) / max(d)

        return min(triple, key=lambda i: (round(imbalance(i), 2), -density[i]))

    def cs_mismatch(triple):
        if k != 3:
            return 0.0
        us = apex(triple)
        a, b = [j for j in triple if j != us]
        return abs(coupled[us, a] - coupled[us, b])

    triples = list(combinations(range(len(centres)), k))
    well_shaped = [t for t in triples if shape(t) >= min_side_ratio]
    pool = well_shaped or triples
    best = min(
        pool,
        key=lambda t: (
            round(worst(t), 2),
            round(cs_mismatch(t), 2),
            -round(shape(t), 2),
            -sum(density[i] for i in t),
        ),
    )

    us = apex(best) if k == 3 else max(best, key=lambda i: density[i])
    rest = sorted((i for i in best if i != us), key=lambda i: (-coupled[us, i], i))
    if counterbalance % 2:
        rest = rest[::-1]
    return {role: tuple(centres[i]) for role, i in zip(roles, [us, *rest])}
