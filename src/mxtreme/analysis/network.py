"""Network structure of a culture's dynamics: functional connectivity and PCA.

Where :mod:`~mxtreme.analysis.activity` describes electrodes one at a time, this module describes how
they relate to *each other*. That separates a culture whose electrodes fire together as one blob from
one with differentiated sub-populations -- two cultures that can look identical in mean firing rate.

Two separate analyses run on the same signal, the double-exponential smoothed firing rate
(:func:`mxtreme.analysis.activity.smoothed_firing_rate`), following Sono et al. 2026 (PNAS), Fig. 2:

- **Functional connectivity** (their Fig. 2B, C): the Pearson correlation between every pair of
  electrodes -- :func:`connectivity`, :func:`plot_connectivity`, :func:`plot_connectivity_grid`.
  :func:`communities` partitions that matrix into groups of electrodes more correlated with each
  other than with the rest (Louvain modularity by default), and the plots can reorder the matrix by
  them so the blocks are visible.
- **PCA** (their Fig. 2D): the eigendecomposition of the electrode covariance -- :func:`pca`. It
  gives the cumulative variance explained against component count
  (:func:`plot_pc_variance_explained`), the **effective rank** ``exp(H(p))`` of the normalized
  spectrum (their Eq. 17-18) -- a continuous stand-in for "how many components it takes" that tracks
  over DIV better than the integer counts -- and the recording's path through PC space over time
  (:func:`plot_pc_trajectory`, :func:`plot_pc_trajectory_grid`, and :func:`animate_pc_trajectory`,
  which plays it alongside the spike raster). :func:`plot_pc_trajectory_by_label` colours that same
  path by a column of the burst table instead of by time, which is what shows whether different
  kinds of burst trace *different* excursions; :func:`animate_pc_trajectory_by_label` replays it or
  rotates a 3-D view of it. Every trajectory function takes an optional precomputed :func:`pca`
  ``result``, so several figures of one recording can share a single eigendecomposition.

- **GPFA** (Gaussian Process Factor Analysis, Yu et al. 2009): :func:`gpfa` and :func:`plot_gpfa`.
  Both analyses above are instantaneous -- PCA finds a subspace but has no notion of time, so the
  smoothness of its trajectories is whatever the double-exponential filter imposed. GPFA puts the
  smoothing inside the model instead: each latent dimension carries a Gaussian-process prior whose
  **timescale is fitted** along with the factor-analysis loadings, on square-rooted spike counts
  rather than on the filtered rate. The outcome is a few smooth latent signals plus a timescale per
  signal that belongs to the network rather than to our settings. :func:`plot_gpfa` draws them
  against time, optionally coloured by burst kind as :func:`plot_pc_trajectory_by_label` does.

:func:`network_metrics` reduces both correlation and PCA to per-phase scalars, and the group-level
:func:`summarize_network_metrics` / :func:`plot_network_summary` take a ``RecordingID``,
``CultureID``, ``CultureSelector`` or several selectors, as in :mod:`~mxtreme.analysis.activity`.

The paper uses **spontaneous** activity (its Fig. 2 is the pre-training baseline). Nothing here
enforces that: every result is per phase, so reading the ``pre`` phase is the equivalent.

.. note::
   The smoothed rate is built from ``Recording.spike_bin``, which is **binary**. On a real hour-long
   recording that discards ~30% of spikes, inside bursts, which is where the correlation structure
   lives -- so correlations here are a lower bound on burst-driven coupling. :func:`gpfa` is the
   exception: it bins ``Recording.spike_data`` instead, because its observation model wants actual
   counts.
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D

from mxtreme import io
from mxtreme.config import Config
from mxtreme.identity import CultureID
from mxtreme.params import CommunityParams, GPFAParams, NetworkParams
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.recording import Recording
from mxtreme.utils import get_n_colors
from mxtreme.visualizations import _burst_spans_frames, _display_raster, finish_figure
from mxtreme.analysis import _communities, _gpfa
from mxtreme.analysis.activity import _double_exponential, _decimation
from mxtreme.analysis._group import SummarySpec, culture_summary, fill_docs, plot_summary, summarize
from mxtreme.analysis._paths import _summary_paths, recording_key, recording_label, unflatten_recording_arrays
from mxtreme.analysis._plotting import categorical_colors, is_unphased, natural_key, phase_tag, sort_phases

__all__ = [
    "connectivity", "communities", "community_table", "pca", "gpfa", "network_metrics",
    "summarize_network_metrics", "plot_network_summary",
    "plot_connectivity", "plot_pc_trajectory", "animate_pc_trajectory", "plot_gpfa",
    "plot_connectivity_grid", "plot_pc_variance_explained", "plot_pc_trajectory_grid",
    "plot_pc_trajectory_by_label", "animate_pc_trajectory_by_label",
]

# PCs kept in the per-culture cache behind the trajectory grid: enough for any pair of the first three.
_CACHED_PCS = 3


# --- compute layer (pure: arrays in, arrays out) --------------------------------------------------


def _pc_column(threshold: float) -> str:
    """Column name for the component count at a cumulative-variance ``threshold`` (0.9 -> ``n_pc_90``)."""
    return f"n_pc_{round(threshold * 100)}"


def _rates(rec, params: NetworkParams) -> np.ndarray:
    """The recording's smoothed rate matrix (channels x samples) under ``params``."""
    return _double_exponential(rec.spike_bin, rec.bin_size, params.tau_rise_sec, params.tau_decay_sec,
                               params.sample_bin_sec, params.chunk_channels)


def _phase_columns(rec, phase, n_samples: int, params: NetworkParams) -> tuple[int, int]:
    """Column bounds of ``phase``'s window in the decimated rate matrix."""
    frames_per_sample = rec.samp_rate * rec.bin_size * _decimation(rec.bin_size, params.sample_bin_sec)
    start = int(phase.start_frame / frames_per_sample)
    stop = int(phase.end_frame / frames_per_sample)
    return max(start, 0), min(stop, n_samples)


def _resolve_phase(rec, phase: str | None):
    """``rec``'s phase named ``phase`` (``None``: the first in the sequence), or ``None`` to use the
    whole recording when it carries no such phase -- so one phase name applies across a batch whose
    recordings don't all share a phase spec."""
    named = {p.name: p for p in rec.phases}
    if phase is None:
        return named[sort_phases(list(named))[0]]
    if phase in named:
        return named[phase]
    print(f"  no {phase!r} phase (found: {', '.join(named)}); using the whole recording")
    return None


def _sample_sec(rec, params: NetworkParams) -> float:
    """Seconds between columns of the decimated rate matrix."""
    return _decimation(rec.bin_size, params.sample_bin_sec) * rec.bin_size


def _window(rec, phase: str | None, params: NetworkParams) -> tuple[np.ndarray, str, int]:
    """The smoothed rates restricted to ``phase``'s window, the phase's resolved name, and the
    window's first column in the whole recording's rate matrix."""
    rates = _rates(rec, params)
    window = _resolve_phase(rec, phase)
    if window is None:
        return rates, phase or "full", 0
    start, stop = _phase_columns(rec, window, rates.shape[1], params)
    return rates[:, start:stop], window.name, start


def _correlation(rates, min_var: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Pearson correlation between rows of ``rates``, dropping rows that never vary.

    A silent electrode, left in, contributes an entire row and column of NaN (a zero denominator) and
    poisons every statistic taken over the matrix.

    :returns: ``(corr, keep)`` -- the ``(n_kept, n_kept)`` float32 matrix and the kept row indices.
    """
    rates = np.asarray(rates)
    keep = np.flatnonzero(rates.var(axis=1) > min_var).astype(np.int32)
    if keep.size < 2:
        return np.zeros((keep.size, keep.size), dtype=np.float32), keep
    return np.asarray(np.corrcoef(rates[keep]), dtype=np.float32), keep


def _spectrum(rates) -> np.ndarray:
    """Normalized eigenvalue spectrum of the row covariance: ``p_i`` is PC ``i``'s share of variance.

    Covariance-side rather than SVD-side because the row count is the electrode count (at most ~1k on
    this hardware) while the sample count is orders of magnitude larger.
    """
    rates = np.asarray(rates)
    if rates.shape[0] < 1 or rates.shape[1] < 2:
        return np.array([], dtype=float)
    centered = rates - rates.mean(axis=1, keepdims=True)
    cov = (centered @ centered.T) / (centered.shape[1] - 1)
    eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))[::-1]
    eigvals = np.clip(eigvals, 0.0, None)  # tiny negatives are float noise on a PSD matrix
    total = eigvals.sum()
    return eigvals / total if total > 0 else np.zeros(eigvals.size, dtype=float)


def _effective_rank(p) -> float:
    """``exp(Shannon entropy(p))`` of a normalized spectrum (Sono et al. Eq. 17-18): 1 when one
    component carries everything, ``d`` when variance spreads evenly over ``d`` of them."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p)))) if p.size else np.nan


def _pca(rates, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full PCA of the rows of ``rates`` (electrodes x samples): the spectrum plus the leading axes.

    Each component's sign is arbitrary in the decomposition, so it is fixed here: flipped so its
    loadings sum to a positive number. For PC1 -- mostly the array-wide burst mode -- "up" then always
    means "more network activity", and the same culture doesn't mirror itself between DIVs.

    :returns: ``(p, components, scores)`` -- the normalized spectrum as in :func:`_spectrum`, the
        ``(n_electrodes, k)`` loadings and the ``(k, n_samples)`` float32 projection of the centred
        rates onto them, with ``k = min(n_components, n_electrodes)``.
    """
    rates = np.asarray(rates)
    n, t = rates.shape
    if n < 1 or t < 2:
        return np.array([], dtype=float), np.zeros((n, 0)), np.zeros((0, t), dtype=np.float32)
    centered = rates - rates.mean(axis=1, keepdims=True)
    cov = (centered @ centered.T) / (t - 1)
    eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    eigvals, eigvecs = np.clip(eigvals[::-1], 0.0, None), eigvecs[:, ::-1]
    total = eigvals.sum()
    p = eigvals / total if total > 0 else np.zeros(eigvals.size, dtype=float)

    components = eigvecs[:, :min(int(n_components), n)]
    components = components * np.where(components.sum(axis=0) < 0, -1.0, 1.0)
    scores = (components.T.astype(np.float32) @ centered).astype(np.float32)
    return p, components, scores


def _metrics(rates, params: NetworkParams) -> dict:
    """Functional-connectivity, community and PCA scalars for one rate window."""
    corr, keep = _correlation(rates, min_var=params.min_var)
    p = _spectrum(np.asarray(rates)[keep])

    metrics = {
        "n_electrodes_used": int(keep.size),
        "mean_corr": np.nan, "median_corr": np.nan, "std_corr": np.nan,
        "modularity": np.nan, "n_communities": np.nan,
        "pc1_var_frac": float(p[0]) if p.size else np.nan,
        "effective_rank": _effective_rank(p),
    }
    if corr.shape[0] >= 2:
        off_diagonal = corr[np.triu_indices_from(corr, k=1)]
        metrics["mean_corr"] = float(np.nanmean(off_diagonal))
        metrics["median_corr"] = float(np.nanmedian(off_diagonal))
        metrics["std_corr"] = float(np.nanstd(off_diagonal))
        found = _communities.detect(corr, params.communities, with_order=False)
        metrics["modularity"] = found["modularity"]
        metrics["n_communities"] = found["n_communities"]

    cumulative = np.cumsum(p) if p.size else np.array([])
    for threshold in params.var_thresholds:
        # searchsorted finds the first component whose cumulative variance reaches the threshold;
        # +1 turns that index into a count.
        metrics[_pc_column(threshold)] = (int(np.searchsorted(cumulative, threshold) + 1)
                                          if cumulative.size else np.nan)
    return metrics


# --- recording level ------------------------------------------------------------------------------


def _recording_output(rec, analysis_dir, suffix: str) -> Path:
    out_dir = Path(analysis_dir) / "network" / str(rec.batch_id) / str(rec.chip) / f"well{rec.well}"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = io.recording_file_name(rec.DIV, rec.plate_date, rec.chip, rec.batch_id, rec.well, suffix,
                                  rec.experiment or "")
    return out_dir / name


def connectivity(rec: Recording, analysis_dir=None, *, phase: str | None = None,
                 params: NetworkParams | None = None, save: bool = True) -> dict:
    """Functional connectivity of one recording: Pearson correlation between every pair of electrodes.

    Computed on the smoothed firing rate (see :func:`mxtreme.analysis.activity.smoothed_firing_rate`)
    within ``phase``'s window. Electrodes whose rate never varies are dropped first. Electrodes stay in
    raw index order -- it follows the array's physical layout, which is what makes block structure from
    spatially separated sub-populations visible.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._connectivity_<phase>.npz`` when
        ``save`` is on.
    :param phase: Phase to use; ``None`` selects the recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: ``{"corr": (n, n) float32, "channels": channel id of each row/column,
        "elec_index": the rows of ``spike_bin`` kept, "phase": str}``.
    """
    params = params or NetworkParams()
    rates, name, _start = _window(rec, phase, params)
    corr, keep = _correlation(rates, min_var=params.min_var)
    result = {"corr": corr, "channels": np.asarray(rec.channelmap)[keep, 1].astype(np.int64),
              "elec_index": keep, "phase": name}

    if save and analysis_dir is not None:
        path = _recording_output(rec, analysis_dir, f"connectivity_{name}.npz")
        np.savez_compressed(path, corr=corr, channels=result["channels"], elec_index=keep)
        print(f"Saved connectivity to {path}")
    return result


def communities(conn, *, params: CommunityParams | None = None, save_path=None) -> dict:
    """Community structure of a functional-connectivity matrix: groups of electrodes more correlated
    with each other than with the rest of the array.

    The algorithm is ``params.method``. The default, ``"louvain"``, maximises modularity (Blondel et
    al. 2008): it finds the number of communities itself, and the modularity ``Q`` it reaches says
    how strong the structure is -- near 0 when the array fires as one synchronized blob, rising as
    sub-networks separate. Two caveats: a partition can differ slightly between seeds (hence
    ``params.n_runs``), and modularity cannot resolve communities much smaller than the network
    (the *resolution limit*), which ``params.resolution`` above 1 counteracts.

    .. note::
       More algorithms can be added in :mod:`mxtreme.analysis._communities`: write a function
       ``fn(corr, params) -> (labels, modularity)``, register it in ``METHODS``, and select it with
       ``CommunityParams(method="<name>")``. Everything below works the same for any method.

    :param conn: The result of :func:`connectivity` -- or a bare square correlation matrix, whose
        rows are then numbered ``0..n-1`` in place of channel ids.
    :param params: Settings; defaults to :class:`~mxtreme.params.CommunityParams`.
    :param save_path: Write the channel -> community mapping (:func:`community_table`) to this CSV.
    :returns: ``{"labels": community of each row of the matrix (0 = largest community), "channels",
        "elec_index": as in ``conn``, "order": a permutation of the rows grouping them by community,
        largest first, and by similarity within each, "boundaries": where each community's block ends
        in that order, "modularity": float, "n_communities": int, "phase", "method": str}``.
    """
    params = params or CommunityParams()
    result = _find_communities(conn, params)
    phase = result["phase"]
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        community_table(result).assign(
            modularity=result["modularity"], method=params.method, resolution=params.resolution,
            negative=params.negative, phase=phase,
        ).to_csv(save_path, index=False)
        print(f"Saved communities to {save_path}")
    return result


def _find_communities(conn, params: CommunityParams) -> dict:
    """:func:`communities` without the saving -- callable from functions whose own ``communities``
    argument shadows the public name."""
    if isinstance(conn, dict):
        corr = conn["corr"]
        channels, elec_index, phase = conn["channels"], conn["elec_index"], conn.get("phase")
    else:
        corr = np.asarray(conn)
        channels = elec_index = np.arange(corr.shape[0])
        phase = None
    return {**_communities.detect(corr, params), "channels": np.asarray(channels),
            "elec_index": np.asarray(elec_index), "phase": phase, "method": params.method}


def community_table(result: dict) -> pd.DataFrame:
    """The channel -> community mapping from :func:`communities`, one row per electrode.

    :returns: Columns ``channel, elec_index, community``, sorted by community then channel.
    """
    return (pd.DataFrame({"channel": result["channels"], "elec_index": result["elec_index"],
                          "community": result["labels"]})
            .sort_values(["community", "channel"], kind="stable").reset_index(drop=True))


def pca(rec: Recording, analysis_dir=None, *, phase: str | None = None,
        params: NetworkParams | None = None, n_components: int = 10, save: bool = True) -> dict:
    """PCA of one recording's network dynamics: the principal components of the smoothed rates.

    Each electrode is mean-centred and the electrode covariance eigendecomposed, within ``phase``'s
    window. Electrodes whose rate never varies are left out, as in :func:`connectivity`. The result
    covers both views of it: how the variance spreads over components (``var_frac``, ``cumvar``,
    ``effective_rank`` -- see :func:`plot_pc_variance_explained`), and where the recording travels in
    PC space over time (``scores`` -- see :func:`plot_pc_trajectory`).

    Each component's sign is fixed so its loadings sum to a positive number: up along PC1 -- largely
    the array-wide burst mode -- then always means more network activity.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._pca_<phase>.npz`` when ``save``
        is on.
    :param phase: Phase to use; ``None`` selects the recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param n_components: How many leading components to return loadings and scores for. The
        spectrum (``var_frac``) always covers every component.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: ``{"var_frac": each component's share (descending, sums to 1), "cumvar": its cumulative
        sum, "effective_rank": float, "components": (n_electrodes, n_components) loadings,
        "scores": (n_components, n_samples) float32 projection onto each component, "time_sec":
        each sample's time in the recording, "channels", "elec_index": the electrodes the loadings
        refer to, "phase": str}``.
    """
    params = params or NetworkParams()
    rates, name, start = _window(rec, phase, params)
    _corr, keep = _correlation(rates, min_var=params.min_var)
    p, components, scores = _pca(rates[keep], n_components)
    step = _sample_sec(rec, params)
    result = {"var_frac": p, "cumvar": np.cumsum(p), "effective_rank": _effective_rank(p),
              "components": components, "scores": scores,
              "time_sec": (start + np.arange(scores.shape[1])) * step,
              "channels": np.asarray(rec.channelmap)[keep, 1].astype(np.int64), "elec_index": keep,
              "phase": name}

    if save and analysis_dir is not None:
        path = _recording_output(rec, analysis_dir, f"pca_{name}.npz")
        np.savez_compressed(path, **{k: v for k, v in result.items() if k != "phase"})
        print(f"Saved PCA to {path}")
    return result


def network_metrics(rec: Recording, analysis_dir=None, *, params: NetworkParams | None = None,
                    save: bool = True) -> pd.DataFrame:
    """Functional-connectivity and PCA scalars of one recording, per phase.

    The filter runs once over the whole recording and each phase is a slice of the result -- cheaper
    than one pass per phase, and a phase inherits the decay tail of what preceded it rather than
    starting from zero.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._network_metrics.csv`` when
        ``save`` is on.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: One row per phase: ``phase, n_electrodes_used``; functional connectivity -- ``mean_corr``,
        ``median_corr``, ``std_corr`` over electrode pairs, and ``modularity`` / ``n_communities``
        of its community structure (see :func:`communities`; settings in ``params.communities``);
        PCA -- ``pc1_var_frac``, ``effective_rank``, and ``n_pc_<pct>`` (components to reach each of
        ``params.var_thresholds``).
    """
    params = params or NetworkParams()
    rates = _rates(rec, params)
    rows = []
    for p in rec.phases:
        start, stop = _phase_columns(rec, p, rates.shape[1], params)
        rows.append({"phase": p.name, **_metrics(rates[:, start:stop], params)})
    df = pd.DataFrame(rows)

    if save and analysis_dir is not None:
        path = _recording_output(rec, analysis_dir, "network_metrics.csv")
        df.to_csv(path, index=False)
        print(f"Saved network metrics to {path}")
    return df


# --- group level ----------------------------------------------------------------------------------

_SPEC = SummarySpec(
    title="Network Metrics",
    metrics=[
        ("mean_corr",      None, "Mean pairwise r",          "Functional Connectivity"),
        ("effective_rank", None, "exp(entropy of spectrum)", "Effective Rank"),
        ("n_pc_90",        None, "Components",               "PCs for 90% Variance"),
        ("pc1_var_frac",   None, "Variance explained",       "PC1 Dominance"),
    ],
)


def _stamp(params: NetworkParams) -> dict:
    """Settings each cached row records, so changing them recomputes rather than reuses. A cache from
    before the community metrics lacks the ``community_*`` columns, so it is recomputed too."""
    c = params.communities
    return {"tau_rise_sec": params.tau_rise_sec, "tau_decay_sec": params.tau_decay_sec,
            "sample_bin_sec": params.sample_bin_sec, "min_var": params.min_var,
            "community_method": c.method, "community_resolution": c.resolution,
            "community_negative": c.negative, "community_n_runs": c.n_runs, "community_seed": c.seed}


def _culture_network_summary(cpath, analysis_dir: Path, use_existing: bool = True,
                             params: NetworkParams | None = None) -> pd.DataFrame:
    """Per-culture network metrics (one row per recording x phase), cached."""
    params = params or NetworkParams()

    def rows(rp):
        rec = Recording(0, io.load_preprocessed(rp.npz))
        df = network_metrics(rec, params=params, save=False)
        return [{"div": rp.recording_id.div, "experiment": rp.recording_id.experiment or "", **row}
                for row in df.to_dict("records")]

    return culture_summary(cpath, analysis_dir, "network", "network_summary", rows, use_existing,
                           stamp=_stamp(params))


def summarize_network_metrics(target, config: Config, *, phase: str | None = None,
                              use_existing: bool = True, params: NetworkParams | None = None) -> pd.DataFrame:
    """Table of network metrics: one row per recording and phase (columns as in :func:`network_metrics`).

    Identity columns lead: ``batch_id, culture_id, chip, well, div, experiment, phase``, plus ``group``
    when several selectors are given. The filter settings are stamped on each row; cached rows
    computed with different ``params`` are recomputed.

    {target}
    :param phase: Keep only this phase; ``None`` keeps every phase.
    :param use_existing: Reuse cached per-culture summaries; ``False`` recomputes the target's
        recordings.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :returns: The summary table.
    """
    def summary(cpath, analysis_dir, use):
        return _culture_network_summary(cpath, analysis_dir, use, params)
    return summarize(target, config, summary, phase=phase, use_existing=use_existing)[0]


def plot_network_summary(target, config: Config, *, split_by: str = "phase", phase: str | None = None,
                         error: str = "sem", params: NetworkParams | None = None,
                         show_plot: bool = True, save_path=None):
    """Four-panel network summary: mean correlation, effective rank, PCs for 90% variance, PC1
    dominance, vs DIV.

    {plot_doc}
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    """
    def summary(cpath, analysis_dir, use):
        return _culture_network_summary(cpath, analysis_dir, use, params)
    return plot_summary(target, config, summary, _SPEC, split_by=split_by, phase=phase, error=error,
                        show_plot=show_plot, save_path=save_path)


fill_docs(summarize_fns=(summarize_network_metrics,), plot_fns=((plot_network_summary, None),))


# --- culture level: matrices and curves -----------------------------------------------------------


def _culture_paths(culture, config) -> CulturePaths:
    if isinstance(culture, CulturePaths):
        return culture
    if isinstance(culture, CultureID):
        return resolve_paths(culture, config)
    raise TypeError(f"Pass a CultureID, not {type(culture).__name__}.")


def _culture_connectivity(cpath, analysis_dir: Path, *, phase: str | None = None,
                          use_existing: bool = True, params: NetworkParams | None = None) -> dict:
    """``{(div, experiment): {"corr", "cumvar", "var_frac", "elec_index", "scores", "time_sec",
    "phase"}}`` for every recording of ``cpath`` -- ``scores`` holding the first few PCs, enough for
    the trajectory grid.

    Cached as one ``.npz`` per culture per phase: the arrays are small next to the cost of producing
    them (decompressing and filtering every recording's spike matrix). A cache from before the
    trajectory arrays were added is recomputed.
    """
    params = params or NetworkParams()
    save_dir, npz_path = _summary_paths(cpath, analysis_dir, "network",
                                        f"{phase or 'default'}_connectivity", ext=".npz")
    wanted = [(rp.recording_id.div, rp.recording_id.experiment or "") for rp in cpath.recordings]

    if use_existing and npz_path.exists():
        with np.load(npz_path) as cached:
            conn = unflatten_recording_arrays(cached)
        if (conn is not None and set(wanted) <= set(conn)
                and all("scores" in conn[key] for key in wanted)):
            print(f"Loading existing connectivity from {npz_path}")
            return {key: conn[key] for key in sorted(wanted)}

    conn = {}
    for key, rp in zip(wanted, cpath.recordings):
        rec = Recording(0, io.load_preprocessed(rp.npz))
        rates, name, start = _window(rec, phase, params)
        corr, keep = _correlation(rates, min_var=params.min_var)
        p, _components, scores = _pca(rates[keep], _CACHED_PCS)
        conn[key] = {"corr": corr, "cumvar": np.cumsum(p), "var_frac": p, "elec_index": keep,
                     "scores": scores,
                     "time_sec": (start + np.arange(scores.shape[1])) * _sample_sec(rec, params),
                     "phase": np.array(name)}

    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        **{f"{recording_key(*key)}__{name}": values
           for key, per_rec in conn.items() for name, values in per_rec.items()},
    )
    print(f"Saved connectivity to {npz_path}")
    return dict(sorted(conn.items()))


def _culture_phase_tag(conn: dict, phase: str | None) -> str:
    """Title suffix naming the phase a culture plot shows -- nothing when every recording is
    unphased."""
    names = [str(per_rec["phase"]) for per_rec in conn.values() if "phase" in per_rec]
    return phase_tag(phase or "first", is_unphased(names), "  (phase: {phase})")


def _plot_matrix(corr, ax, title=None, vmin=-1.0, vmax=1.0, order=None, boundaries=None):
    """Electrode x electrode correlation heat map on a pinned colour scale, optionally reordered by
    ``order`` with a thin line closing each community block (``boundaries``)."""
    corr = np.asarray(corr)
    if order is not None:
        corr = corr[np.ix_(order, order)]
    image = ax.imshow(corr, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest",
                      aspect="equal")
    # A 1000x1000 matrix is a million vector quads in a PDF and looks identical as a raster.
    image.set_rasterized(True)
    if boundaries is not None:
        for edge in np.asarray(boundaries)[:-1]:
            for line in (ax.axhline, ax.axvline):
                line(edge - 0.5, color="black", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("Electrode" + (" (by community)" if order is not None else ""), fontsize=8)
    ax.set_ylabel("Electrode" + (" (by community)" if order is not None else ""), fontsize=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    return image


def plot_connectivity(conn: dict, *, communities=None, community_params: CommunityParams | None = None,
                      ax=None, title: str | None = None, colorbar: bool = True,
                      show_plot: bool = True, save_path=None, dpi: int = 150):
    """Functional connectivity of one recording: its correlation matrix as a heat map.

    In raw electrode order the matrix follows the array's layout. Ordered by community, electrodes
    of one community sit together -- largest community first, the most similar electrodes adjacent
    within it -- and a thin line closes each block, so the community structure shows as blocks on
    the diagonal.

    :param conn: The result of :func:`connectivity`.
    :param communities: ``None`` keeps raw electrode order; ``True`` runs :func:`communities` with
        ``community_params``; a :func:`communities` result uses that partition.
    :param community_params: Settings when ``communities=True``.
    :param ax: Axes to draw on; a new figure when ``None``.
    :param title: Axes title (default: the phase, and Q when ordered by community).
    :param colorbar: Draw a colour bar.
    :param show_plot: Show the figure (only when this function created it).
    :param save_path: Full path (including filename) to save the figure to.
    :param dpi: Resolution for the saved figure and the rasterized heat map.
    :returns: The matplotlib Figure.
    """
    if communities is True:
        communities = _find_communities(conn, community_params or CommunityParams())
    order = boundaries = None
    if communities:
        if len(communities["labels"]) != np.asarray(conn["corr"]).shape[0]:
            raise ValueError("`communities` was computed on a different matrix than `conn`.")
        order, boundaries = communities["order"], communities["boundaries"]
        title = title or (f"{communities['n_communities']} communities, "
                          f"Q = {communities['modularity']:.2f}")

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    else:
        fig = ax.figure
    image = _plot_matrix(conn["corr"], ax, title=title, order=order, boundaries=boundaries)
    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    if owns_fig:
        return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
    return fig


def plot_connectivity_grid(culture, config: Config, *, phase: str | None = None,
                           params: NetworkParams | None = None, order: str = "index",
                           community_params: CommunityParams | None = None, show_plot: bool = True,
                           save_path=None, ncols: int | None = None, dpi: int = 150):
    """Functional connectivity of one culture: a correlation heat map per recording, on one colour scale.

    :param culture: A :class:`~mxtreme.identity.CultureID`.
    :param config: The :class:`~mxtreme.config.Config` describing the store.
    :param phase: Phase to use; ``None`` selects each recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param order: ``"index"`` keeps raw electrode order, which follows the array's layout;
        ``"communities"`` reorders each panel by that recording's own communities (see
        :func:`plot_connectivity`) and adds the community count and Q to its title.
    :param community_params: Community-detection settings for ``order="communities"``.
    :param show_plot: Show the figure.
    :param save_path: Full path (including filename) to save the figure to.
    :param ncols: Panels per row; defaults to at most 4.
    :param dpi: Resolution for the saved figure and the rasterized heat maps.
    :returns: The matplotlib Figure (``None`` if the culture has no recordings).
    """
    if order not in ("index", "communities"):
        raise ValueError(f"order must be 'index' or 'communities', not {order!r}")
    cpath = _culture_paths(culture, config)
    conn = _culture_connectivity(cpath, config.analysis_dir, phase=phase, params=params)
    keys = sorted(conn)
    if not keys:
        return None

    ncols = ncols or min(4, len(keys))
    nrows = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        title = recording_label(*key)
        found = None
        if order == "communities":
            found = communities(conn[key]["corr"], params=community_params)
            title += f"\n{found['n_communities']} communities, Q = {found['modularity']:.2f}"
        plot_connectivity(conn[key], communities=found, ax=ax, title=title, colorbar=False)
        ax.title.set_fontsize(10 if found else 11)
        ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_ylabel("")
    for ax in axes[len(keys):]:
        ax.set_visible(False)

    # One colour bar for the figure: every panel shares the pinned [-1, 1] scale.
    fig.colorbar(axes[0].images[0], ax=axes[:len(keys)].tolist(), fraction=0.025, pad=0.02,
                 label="Pearson r")
    fig.suptitle(f"Functional Connectivity — {cpath.culture_id}{_culture_phase_tag(conn, phase)}",
                 fontsize=13, fontweight="bold")
    return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)


def plot_pc_variance_explained(culture, config: Config, *, phase: str | None = None,
                               params: NetworkParams | None = None, thresholds=(0.8, 0.9),
                               show_plot: bool = True, save_path=None, cmap_name: str = "viridis"):
    """PCA of one culture: cumulative variance explained vs component count, one line per recording.

    A culture whose curve rises later over development is spreading its dynamics across more
    independent components -- what the effective rank in :func:`network_metrics` reduces to one number.

    :param culture: A :class:`~mxtreme.identity.CultureID`.
    :param config: The :class:`~mxtreme.config.Config` describing the store.
    :param phase: Phase to use; ``None`` selects each recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param thresholds: Cumulative-variance levels to mark with a guide line.
    :param show_plot: Show the figure.
    :param save_path: Full path (including filename) to save the figure to.
    :param cmap_name: Sequential colormap over the recordings in DIV order, so colour reads as
        developmental time.
    :returns: The matplotlib Figure (``None`` if the culture has no recordings).
    """
    cpath = _culture_paths(culture, config)
    conn = _culture_connectivity(cpath, config.analysis_dir, phase=phase, params=params)
    keys = sorted(conn)
    if not keys:
        return None

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = dict(zip(keys, get_n_colors(len(keys), cmap_name=cmap_name)))
    for key in keys:
        cumvar = np.asarray(conn[key].get("cumvar", []), dtype=float)
        if cumvar.size:
            ax.plot(np.arange(1, cumvar.size + 1), cumvar, color=colors[key], linewidth=1.5,
                    label=recording_label(*key))

    for threshold in thresholds:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(1, threshold, f" {threshold:.0%}", fontsize=7, color="gray", va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Number of principal components", fontsize=10)
    ax.set_ylabel("Cumulative variance explained", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, framealpha=0.8, ncol=max(1, len(keys) // 6 + 1), loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"PC Variance Explained — {cpath.culture_id}{_culture_phase_tag(conn, phase)}",
                 fontsize=13, fontweight="bold")
    return finish_figure(fig, save_path=save_path, show_plot=show_plot)


# --- PC trajectories ------------------------------------------------------------------------------

_TIME_UNITS = {"s": 1.0, "min": 60.0}


def _pc_indices(pcs, n_available: int, *, allow_3d: bool = False) -> tuple[int, ...]:
    """0-based rows of ``scores`` for the 1-based component numbers ``pcs``, checked.

    Two components, unless ``allow_3d`` also permits three (for the 3-D trajectory plots).
    """
    allowed = (2, 3) if allow_3d else (2,)
    if len(pcs) not in allowed or min(pcs) < 1:
        wanted = "two or three" if allow_3d else "two"
        raise ValueError(f"pcs must be {wanted} 1-based component numbers, e.g. (1, 2); got {pcs!r}")
    if max(pcs) > n_available:
        raise ValueError(f"PC{max(pcs)} requested but only {n_available} available.")
    return tuple(p - 1 for p in pcs)


def _segments(*coords) -> np.ndarray:
    """Consecutive point pairs of a polyline, the shape :class:`LineCollection` takes.

    Two coordinate arrays give ``(n - 1, 2, 2)`` segments; three give ``(n - 1, 2, 3)``, which is
    what :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection` takes.
    """
    points = np.column_stack(coords).reshape(-1, 1, len(coords))
    return np.concatenate([points[:-1], points[1:]], axis=1)


def _pca_result(rec, result, *, phase, params, n_components):
    """The caller's precomputed :func:`pca` result, or a fresh one.

    Lets a caller run :func:`pca` once and feed the same result to several trajectory figures rather
    than paying for the eigendecomposition per figure.
    """
    if result is None:
        return pca(rec, phase=phase, params=params, n_components=n_components, save=False)
    missing = {"scores", "time_sec", "var_frac", "phase"} - set(result)
    if missing:
        raise ValueError(f"result is missing {sorted(missing)}; pass a dict from network.pca().")
    if phase is not None and result["phase"] != phase:
        raise ValueError(f"result was computed for phase {result['phase']!r}, but phase={phase!r} "
                         "was requested; pass one or the other.")
    return result


def _draw_trajectory(ax, scores, time_sec, var_frac, pcs, cmap, time_unit, norm=None,
                     linewidth=1.0, label_axes=True):
    """Draw the PC-space path as a line coloured by time; returns the :class:`LineCollection`."""
    i, j = _pc_indices(pcs, scores.shape[0])
    t = np.asarray(time_sec) / _TIME_UNITS[time_unit]
    lines = LineCollection(_segments(scores[i], scores[j]), cmap=cmap, linewidth=linewidth,
                           norm=norm or plt.Normalize(t[0], t[-1]))
    lines.set_array(t[:-1])
    # An hour at 50 ms is ~72k segments: as vectors that bloats a PDF for no visible difference.
    lines.set_rasterized(True)
    ax.add_collection(lines)
    ax.autoscale_view()
    if label_axes:
        ax.set_xlabel(f"PC{pcs[0]} ({var_frac[i]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC{pcs[1]} ({var_frac[j]:.1%})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return lines


def plot_pc_trajectory(rec: Recording, *, phase: str | None = None, params: NetworkParams | None = None,
                       pcs=(1, 2), cmap: str = "viridis", time_unit: str = "min",
                       result: dict | None = None, ax=None, colorbar: bool = True,
                       show_plot: bool = True, save_path=None, dpi: int = 150):
    """The recording's path through PC space: one PC against another, coloured by time.

    Each point is the whole array's smoothed firing rate at one moment, projected onto two principal
    components (see :func:`pca`). A network burst shows as an excursion along PC1 -- the array-wide
    mode -- and back; structure in the other components shows which electrodes lead or lag.

    :param rec: The recording.
    :param phase: Phase to use; ``None`` selects the recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param pcs: The two components to plot, 1-based (``(1, 3)`` plots PC1 against PC3).
    :param cmap: Colormap for time.
    :param time_unit: ``"min"`` or ``"s"``, for the colour bar.
    :param result: A precomputed :func:`pca` result to plot, so several figures of the same
        recording can share one eigendecomposition; ``None`` computes it (and ``params`` is then
        unused). Must cover ``max(pcs)`` components, and its phase must match ``phase`` if given.
    :param ax: Axes to draw on; a new figure when ``None``.
    :param colorbar: Draw a time colour bar.
    :param show_plot: Show the figure (only when this function created it).
    :param save_path: Full path (including filename) to save the figure to.
    :param dpi: Resolution for the saved figure and the rasterized line.
    :returns: The matplotlib Figure.
    """
    if time_unit not in _TIME_UNITS:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}, not {time_unit!r}")
    result = _pca_result(rec, result, phase=phase, params=params, n_components=max(pcs))

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    else:
        fig = ax.figure
    lines = _draw_trajectory(ax, result["scores"], result["time_sec"], result["var_frac"], pcs, cmap,
                             time_unit)
    if colorbar:
        fig.colorbar(lines, ax=ax, label=f"Time ({time_unit})")
    if owns_fig:
        unphased = is_unphased([result["phase"]])
        ax.set_title(f"PC trajectory — {recording_label(rec.DIV, rec.experiment)}"
                     f"{phase_tag(result['phase'], unphased, '  (phase: {phase})')}",
                     fontsize=11, fontweight="bold")
        return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
    return fig


def plot_pc_trajectory_grid(culture, config: Config, *, phase: str | None = None,
                            params: NetworkParams | None = None, pcs=(1, 2), cmap: str = "viridis",
                            time_unit: str = "min", show_plot: bool = True, save_path=None,
                            ncols: int | None = None, dpi: int = 150):
    """PC trajectories of one culture: one panel per recording (see :func:`plot_pc_trajectory`).

    Every panel has its own axes -- each recording has its own PC basis, so their coordinates aren't
    comparable -- but the components' signs are fixed the same way throughout (see :func:`pca`), so
    the panels' shapes are. Colour is time within each recording.

    :param culture: A :class:`~mxtreme.identity.CultureID`.
    :param config: The :class:`~mxtreme.config.Config` describing the store.
    :param phase: Phase to use; ``None`` selects each recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param pcs: The two components to plot, 1-based; up to PC3.
    :param cmap: Colormap for time.
    :param time_unit: ``"min"`` or ``"s"``, for the colour bars.
    :param show_plot: Show the figure.
    :param save_path: Full path (including filename) to save the figure to.
    :param ncols: Panels per row; defaults to at most 4.
    :param dpi: Resolution for the saved figure and the rasterized lines.
    :returns: The matplotlib Figure (``None`` if the culture has no recordings).
    """
    if time_unit not in _TIME_UNITS:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}, not {time_unit!r}")
    _pc_indices(pcs, _CACHED_PCS)
    cpath = _culture_paths(culture, config)
    conn = _culture_connectivity(cpath, config.analysis_dir, phase=phase, params=params)
    keys = sorted(conn)
    if not keys:
        return None

    ncols = ncols or min(4, len(keys))
    nrows = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for ax, key in zip(axes, keys):
        per_rec = conn[key]
        scores = np.asarray(per_rec["scores"])
        if scores.shape[0] < max(pcs) or scores.shape[1] < 2:
            ax.text(0.5, 0.5, "not enough active electrodes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
        else:
            lines = _draw_trajectory(ax, scores, per_rec["time_sec"], per_rec["var_frac"], pcs, cmap,
                                     time_unit, linewidth=0.8)
            fig.colorbar(lines, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title(recording_label(*key), fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
    for ax in axes[len(keys):]:
        ax.set_visible(False)

    fig.suptitle(f"PC Trajectories — {cpath.culture_id}{_culture_phase_tag(conn, phase)}"
                 f"\ncolour: time ({time_unit})", fontsize=13, fontweight="bold")
    return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)


# --- PC trajectories coloured by a burst label ----------------------------------------------------

_NO_BURST = -1


def _label_codes(t, burst_df, label_col, labels, span, samp_rate):
    """Which label covers each sample of ``t``.

    :returns: ``(codes, counts, n_fell_back)`` -- an index into ``labels`` per sample (``-1`` where
        no burst covers it), the number of bursts per label, and how many of those bursts fell back
        to ISI-N bounds (see :func:`~mxtreme.visualizations._burst_spans_frames`).
    """
    start_frames, stop_frames, fell_back = _burst_spans_frames(burst_df, span)
    codes = np.full(np.shape(t), _NO_BURST, dtype=int)
    counts = {}
    for code, label in enumerate(labels):
        selected = (burst_df[label_col] == label).to_numpy()
        # Later labels win where two bursts overlap -- the same precedence the legend order shows.
        for on, off in zip(start_frames[selected] / samp_rate, stop_frames[selected] / samp_rate):
            # `t` is strictly increasing (pca builds it as (start + arange(n)) * step), so a binary
            # search beats a boolean mask over every sample, per burst.
            codes[np.searchsorted(t, on, "left"):np.searchsorted(t, off, "right")] = code
        counts[label] = int(selected.sum())
    return codes, counts, int(fell_back.sum())


def _padded(values, frac: float = 0.05):
    """``(lo, hi)`` spanning ``values`` with a margin -- 3-D axes don't autoscale from collections."""
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = (hi - lo) * frac or max(abs(hi), 1.0) * frac  # a flat component still gets a visible axis
    return lo - margin, hi + margin


def _labelled_trajectory_figure(rec, burst_df, label_col, *, phase, params, pcs, span, labels,
                                colors, result, start_sec, stop_sec, base_color, linewidth, elev,
                                azim, legend, ax):
    """Draw the PC path split by burst label; shared by the static and animated versions.

    :returns: ``(fig, ax, result, coords, t, codes, labels, colours, note)`` -- ``note`` is the
        ISI-N-fallback warning for the title, or ``""``.
    """
    if label_col not in burst_df.columns:
        raise ValueError(f"burst_df has no column {label_col!r}; available: {sorted(burst_df.columns)}")
    result = _pca_result(rec, result, phase=phase, params=params, n_components=max(pcs))
    idx = _pc_indices(pcs, result["scores"].shape[0], allow_3d=True)
    t = np.asarray(result["time_sec"])

    keep = np.ones(t.shape, bool)
    if start_sec is not None:
        keep &= t >= start_sec
    if stop_sec is not None:
        keep &= t <= stop_sec
    t = t[keep]
    coords = [result["scores"][k][keep] for k in idx]

    if labels is None:
        # Sorted, not first-seen: colour assignment must not depend on the row order of burst_df.
        labels = sorted(burst_df[label_col].dropna().unique(), key=natural_key)
    else:
        labels = list(labels)
    colours = categorical_colors(labels, colors)
    codes, counts, n_fell_back = _label_codes(t, burst_df, label_col, labels, span, rec.samp_rate)

    is_3d = len(pcs) == 3
    owns_fig = ax is None
    if owns_fig:
        fig = plt.figure(figsize=(7.5, 6.0) if is_3d else (7, 5.5), constrained_layout=True)
        ax = fig.add_subplot(projection="3d") if is_3d else fig.add_subplot()
    else:
        fig = ax.figure
        if is_3d and getattr(ax, "name", "") != "3d":
            raise ValueError("pcs names three components but ax is not a 3-D axes; create it with "
                             "fig.add_subplot(projection='3d').")

    if is_3d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection  # keeps the 2-D path mplot3d-free

    segments = _segments(*coords)
    seg_codes = codes[:-1]
    for code, colour, width, zorder in [(_NO_BURST, base_color, 0.6, 1)] + [
            (c, colours[label], linewidth, 2 + c) for c, label in enumerate(labels)]:
        selected = seg_codes == code
        if not selected.any():
            continue
        if is_3d:
            lines = Line3DCollection(segments[selected], colors=colour, linewidth=width,
                                     zorder=zorder)
            # autolim=True would read col._segments3d and raise on an empty collection; limits are
            # set below from the whole trajectory, not from whichever labels happen to be present.
            ax.add_collection3d(lines, autolim=False)
        else:
            lines = LineCollection(segments[selected], colors=colour, linewidth=width, zorder=zorder)
            ax.add_collection(lines)
        # An hour at 50 ms is ~72k segments: as vectors that bloats a PDF for no visible difference.
        lines.set_rasterized(True)

    ax.set_xlabel(f"PC{pcs[0]} ({result['var_frac'][idx[0]]:.1%})", fontsize=9)
    ax.set_ylabel(f"PC{pcs[1]} ({result['var_frac'][idx[1]]:.1%})", fontsize=9)
    if is_3d:
        ax.set_zlabel(f"PC{pcs[2]} ({result['var_frac'][idx[2]]:.1%})", fontsize=9)
        ax.set_xlim(*_padded(coords[0]))
        ax.set_ylim(*_padded(coords[1]))
        ax.set_zlim(*_padded(coords[2]))
        ax.set_box_aspect((1, 1, 1))  # don't let one PC's larger scale distort the shape
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.autoscale_view()
        ax.spines[["top", "right"]].set_visible(False)

    if legend:
        # The counts expose the commonest mistake: burst_df from a different phase than the PCA
        # paints nothing, which otherwise looks like a plain grey trajectory with no error.
        handles = [Line2D([], [], color=colours[label], lw=2, label=f"{label} (n={counts[label]})")
                   for label in labels]
        handles.append(Line2D([], [], color=base_color, lw=2, label="no burst"))
        ax.legend(handles=handles, fontsize=8, frameon=False)

    note = (f" · {n_fell_back}/{len(burst_df)} bursts use ISI-N bounds (no features)"
            if n_fell_back else "")
    return fig, ax, result, coords, t, codes, labels, colours, note


def plot_pc_trajectory_by_label(rec: Recording, burst_df, label_col: str, *,
                                phase: str | None = None, params: NetworkParams | None = None,
                                pcs=(1, 2), span: str = "isi_n", labels=None, colors=None,
                                result: dict | None = None, start_sec: float | None = None,
                                stop_sec: float | None = None, base_color: str = "0.85",
                                linewidth: float = 1.3, elev: float = 22.0, azim: float = -60.0,
                                legend: bool = True, ax=None, show_plot: bool = True,
                                save_path=None, dpi: int = 150):
    """The recording's path through PC space, coloured by which kind of burst is happening.

    Where :func:`plot_pc_trajectory` colours the path by time, this colours it by a column of the
    burst table: every sample inside a burst takes that burst's label colour, everything else stays
    grey. That answers whether different bursts trace *different* excursions -- whether, say, mini
    bursts stay near the origin while network bursts swing out along PC1, or whether two
    communities leave along separate arms.

    Any column works as the label: ``"kind"``, a community assignment, a cluster id you added
    yourself. Bursts are taken exactly as given -- filter ``burst_df`` before calling.

    Pass three components (``pcs=(1, 2, 3)``) for a 3-D plot, with ``elev`` / ``azim`` setting the
    viewing angle; :func:`animate_pc_trajectory_by_label` with ``mode="spin"`` turns the same
    figure into a rotating animation.

    :param rec: The recording.
    :param burst_df: Burst DataFrame from :meth:`BurstSet.to_dataframe
        <mxtreme.bursting.detection.BurstSet.to_dataframe>`, pre-filtered to the bursts of interest.
    :param label_col: Column of ``burst_df`` to colour by. Rows whose label is missing are ignored.
    :param phase: Phase to use; ``None`` selects the recording's first phase. The bursts are *not*
        filtered by it -- if the legend shows ``n=0`` for every label, ``burst_df`` covers a
        different phase than the PCA.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param pcs: Two components for a 2-D plot, or three for a 3-D one; 1-based.
    :param span: Which burst boundaries to paint -- ``"isi_n"`` for the ISI-N group bounds, or
        ``"features"`` for the extracted onset/offset. Bursts with no extracted onset/offset fall
        back to their ISI-N bounds, and the title then says how many did.
    :param labels: Label values to draw, in order (later ones win where bursts overlap); ``None``
        takes every value in the column, naturally sorted.
    :param colors: Optional ``{label: colour}`` overrides; the rest come from
        :data:`~mxtreme.analysis._plotting.CATEGORICAL_COLORS`.
    :param result: A precomputed :func:`pca` result to plot, so several figures of the same
        recording can share one eigendecomposition; ``None`` computes it. Must cover ``max(pcs)``
        components, and its phase must match ``phase`` if given.
    :param start_sec: Plot only from this time; ``None`` starts at the phase start.
    :param stop_sec: Plot only up to this time; ``None`` runs to the phase end.
    :param base_color: Colour of the path outside any burst.
    :param linewidth: Width of the in-burst lines.
    :param elev: 3-D elevation angle in degrees (ignored in 2-D).
    :param azim: 3-D azimuth angle in degrees (ignored in 2-D).
    :param legend: Draw the per-label legend, with each label's burst count.
    :param ax: Axes to draw on; a new figure when ``None``. Must be a 3-D axes for three ``pcs``.
    :param show_plot: Show the figure (only when this function created it).
    :param save_path: Full path (including filename) to save the figure to.
    :param dpi: Resolution for the saved figure and the rasterized lines.
    :returns: The matplotlib Figure.
    """
    owns_fig = ax is None
    fig, ax, result, _, t, _, _, _, note = _labelled_trajectory_figure(
        rec, burst_df, label_col, phase=phase, params=params, pcs=pcs, span=span, labels=labels,
        colors=colors, result=result, start_sec=start_sec, stop_sec=stop_sec,
        base_color=base_color, linewidth=linewidth, elev=elev, azim=azim, legend=legend, ax=ax)

    if owns_fig:
        unphased = is_unphased([result["phase"]])
        span_text = f"{t[0]:.0f}-{t[-1]:.0f} s" if t.size else "empty"
        ax.set_title(f"PC trajectory by {label_col} — {recording_label(rec.DIV, rec.experiment)}"
                     f"{phase_tag(result['phase'], unphased, '  (phase: {phase})')}"
                     f"  ({span_text}){note}", fontsize=11, fontweight="bold")
        return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
    return fig


def animate_pc_trajectory(rec: Recording, save_path, *, phase: str | None = None,
                          params: NetworkParams | None = None, pcs=(1, 2),
                          start_sec: float | None = None, duration_sec: float = 120.0,
                          step_sec: float = 0.25, raster_window_sec: float = 30.0,
                          display_bin_sec: float = 0.1, order: str = "index",
                          community_params: CommunityParams | None = None, show_asdr: bool = True,
                          fps: int = 20, dpi: int = 90, cmap: str = "viridis",
                          result: dict | None = None, max_frames: int | None = 3000) -> Path:
    """Animate the PC trajectory being drawn next to the spike raster of the same moment.

    Left, the recording's path through PC space (see :func:`plot_pc_trajectory`): the whole phase in
    faint grey for orientation, the part already played coloured by time, and a dot at the current
    moment. Right, the raster for a ``raster_window_sec`` window centred on that moment, with a cursor
    -- and below it, optionally, the array-wide spike count (ASDR). A network burst then reads as a
    column of spikes in the raster and, at the same moment, an excursion of the dot along PC1.

    The clip covers ``duration_sec`` of the recording from ``start_sec``, advancing ``step_sec`` per
    frame, so it has ``duration_sec / step_sec`` frames and plays for that many ``/ fps`` seconds.
    The defaults -- two minutes at 0.25 s per frame -- make 480 frames, a 24 s GIF. On a real
    1,000-electrode recording that is about 10 MB and takes about 1.5 minutes to render (~0.2 s per
    frame); lower ``dpi`` or ``duration_sec`` for a smaller, faster file.

    :param rec: The recording.
    :param save_path: Where to write the animation: ``.gif`` (Pillow, always available) or ``.mp4``
        (needs ffmpeg on the PATH).
    :param phase: Phase whose PC basis is used; ``None`` selects the recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param pcs: The two components to plot, 1-based.
    :param start_sec: Clip start, in seconds of the recording; ``None`` starts at the phase start.
        The clip must lie inside the phase.
    :param duration_sec: Clip length in seconds of recording (cut short at the phase end).
    :param step_sec: Recording time advanced per frame.
    :param raster_window_sec: Width of the raster window.
    :param display_bin_sec: Bin width the raster is summed into for display.
    :param order: Raster row order: ``"index"`` (raw electrode order) or ``"communities"`` (grouped by
        the phase's community structure, see :func:`communities`, with silent electrodes left out).
    :param community_params: Community-detection settings for ``order="communities"``.
    :param show_asdr: Draw the ASDR strip under the raster.
    :param fps: Frames per second of the output.
    :param dpi: Resolution of the output frames.
    :param cmap: Colormap for time along the trajectory.
    :param result: A precomputed :func:`pca` result to animate, so several figures of the same
        recording can share one eigendecomposition; ``None`` computes it. Must cover ``max(pcs)``
        components, and its phase must match ``phase`` if given. ``params`` is still used for
        ``order="communities"``.
    :param max_frames: Refuse to render more frames than this (``None``: no limit) -- a whole hour at
        the default step is 14,400 frames.
    :returns: The path written.
    """
    from matplotlib import animation

    if order not in ("index", "communities"):
        raise ValueError(f"order must be 'index' or 'communities', not {order!r}")
    save_path = Path(save_path)
    suffix = save_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Writing .mp4 needs ffmpeg on the PATH; save as .gif instead.")
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError(f"save_path must end in .gif or .mp4, not {save_path.suffix!r}")

    params = params or NetworkParams()
    result = _pca_result(rec, result, phase=phase, params=params, n_components=max(pcs))
    i, j = _pc_indices(pcs, result["scores"].shape[0])
    x, y, t = result["scores"][i], result["scores"][j], result["time_sec"]
    if t.size < 2:
        raise ValueError("The phase is too short, or has too few active electrodes, to animate.")
    phase_start, phase_end = float(t[0]), float(t[-1])

    clip_start = phase_start if start_sec is None else float(start_sec)
    clip_end = min(clip_start + float(duration_sec), phase_end)
    if not phase_start <= clip_start < phase_end:
        raise ValueError(f"start_sec={clip_start:g} is outside the phase ({phase_start:g}-{phase_end:g} s).")
    frame_times = np.arange(clip_start, clip_end, float(step_sec))
    print(f"Animating {frame_times.size} frames ({clip_start:.1f}-{clip_end:.1f} s) -> {save_path}")
    if max_frames is not None and frame_times.size > max_frames:
        raise ValueError(f"{frame_times.size} frames exceeds max_frames={max_frames}; shorten "
                         "duration_sec, raise step_sec, or pass max_frames=None.")

    # Raster rows, and the bins it covers: the clip plus half a window either side.
    if order == "communities":
        conn = connectivity(rec, phase=phase, params=params, save=False)
        found = communities(conn, params=community_params)
        rows = np.asarray(conn["elec_index"])[found["order"]]
        boundaries = found["boundaries"]
    else:
        rows, boundaries = np.arange(rec.spike_bin.shape[0]), None
    half = raster_window_sec / 2
    first_bin = max(int((clip_start - half) / rec.bin_size), 0)
    last_bin = min(int(np.ceil((clip_end + half) / rec.bin_size)), rec.spike_bin.shape[1])
    counts, factor = _display_raster(rec.spike_bin, rows, rec.bin_size, display_bin_sec, first_bin,
                                     last_bin)
    raster_t0, raster_dt = first_bin * rec.bin_size, factor * rec.bin_size
    raster_t1 = raster_t0 + counts.shape[1] * raster_dt

    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2 if show_asdr else 1, 2, width_ratios=[1, 1.5],
                            height_ratios=[4, 1] if show_asdr else [1])
    ax_pc = fig.add_subplot(grid[:, 0])
    ax_raster = fig.add_subplot(grid[0, 1])
    ax_asdr = fig.add_subplot(grid[1, 1], sharex=ax_raster) if show_asdr else None

    # Trajectory: the whole phase faint, then the clip drawn in as it plays.
    ax_pc.plot(x, y, color="0.85", linewidth=0.6, zorder=1)
    norm = plt.Normalize(clip_start, clip_end)
    drawn = LineCollection([], cmap=cmap, norm=norm, linewidth=1.4, zorder=2)
    ax_pc.add_collection(drawn)
    (dot,) = ax_pc.plot([], [], "o", color="crimson", markersize=6, zorder=3)
    ax_pc.set_xlabel(f"PC{pcs[0]} ({result['var_frac'][i]:.1%})", fontsize=9)
    ax_pc.set_ylabel(f"PC{pcs[1]} ({result['var_frac'][j]:.1%})", fontsize=9)
    ax_pc.spines[["top", "right"]].set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_pc, label="Time (s)",
                 location="bottom", fraction=0.05, pad=0.02)

    # Saturate early: a raster should show a lone spike, not only the densest burst bins.
    vmax = max(float(np.percentile(counts[counts > 0], 75)) if np.any(counts) else 1.0, 1.0)
    ax_raster.imshow(counts, aspect="auto", cmap="Greys", vmin=0, vmax=vmax, interpolation="nearest",
                     extent=(raster_t0, raster_t1, counts.shape[0] - 0.5, -0.5))
    if boundaries is not None:
        for edge in np.asarray(boundaries)[:-1]:
            ax_raster.axhline(edge - 0.5, color="tab:blue", linewidth=0.5, alpha=0.6)
    ax_raster.set_ylabel("Electrode" + (" (by community)" if boundaries is not None else ""), fontsize=9)
    cursors = [ax_raster.axvline(clip_start, color="crimson", linewidth=1.0)]
    if ax_asdr is not None:
        centers = raster_t0 + (np.arange(counts.shape[1]) + 0.5) * raster_dt
        ax_asdr.plot(centers, counts.sum(axis=0) / raster_dt, color="black", linewidth=0.7)
        ax_asdr.set_ylabel("Spikes/s", fontsize=8)
        ax_asdr.set_xlabel("Time (s)", fontsize=9)
        ax_asdr.tick_params(labelsize=7)
        ax_asdr.spines[["top", "right"]].set_visible(False)
        cursors.append(ax_asdr.axvline(clip_start, color="crimson", linewidth=1.0))
        ax_raster.tick_params(labelbottom=False)
    else:
        ax_raster.set_xlabel("Time (s)", fontsize=9)
    ax_raster.tick_params(labelsize=7)
    unphased = is_unphased([result["phase"]])
    title = fig.suptitle("", fontsize=11, fontweight="bold")
    label = (f"{recording_label(rec.DIV, rec.experiment)}"
             f"{phase_tag(result['phase'], unphased, '  (phase: {phase})')}")

    start_idx = int(np.searchsorted(t, clip_start))

    def update(now):
        k = int(np.searchsorted(t, now, side="right"))  # samples up to and including `now`
        if k - start_idx >= 2:
            drawn.set_segments(_segments(x[start_idx:k], y[start_idx:k]))
            drawn.set_array(t[start_idx:k - 1])
        if k:
            dot.set_data([x[k - 1]], [y[k - 1]])
        ax_raster.set_xlim(now - half, now + half)
        for cursor in cursors:
            cursor.set_xdata([now, now])
        title.set_text(f"{label}   t = {now:6.1f} s")
        return [drawn, dot, *cursors, title]

    anim = animation.FuncAnimation(fig, update, frames=frame_times, blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(save_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"Saved animation to {save_path}")
    return save_path


def _animation_writer(save_path, fps: int):
    """The writer for ``save_path``'s suffix, as :func:`animate_pc_trajectory` picks it."""
    from matplotlib import animation

    suffix = Path(save_path).suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Writing .mp4 needs ffmpeg on the PATH; save as .gif instead.")
        return animation.FFMpegWriter(fps=fps)
    raise ValueError(f"save_path must end in .gif or .mp4, not {Path(save_path).suffix!r}")


def _write_animation(anim, fig, save_path, writer, dpi: int):
    """Render ``anim`` to ``save_path``, always closing ``fig``."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(save_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"Saved animation to {save_path}")
    return save_path


def _segment_colours(codes, labels, colours, base_color):
    """RGBA per segment of the polyline, from the per-sample label codes."""
    lookup = [to_rgba(base_color)] + [to_rgba(colours[label]) for label in labels]
    return np.array([lookup[c + 1] for c in codes[:-1]])


def animate_pc_trajectory_by_label(rec: Recording, burst_df, save_path, label_col: str, *,
                                   mode: str = "time", phase: str | None = None,
                                   params: NetworkParams | None = None, pcs=(1, 2),
                                   span: str = "isi_n", labels=None, colors=None,
                                   result: dict | None = None, base_color: str = "0.85",
                                   linewidth: float = 1.3, elev: float = 22.0, azim: float = -60.0,
                                   fps: int = 20, dpi: int = 100, max_frames: int | None = 3000,
                                   start_sec: float | None = None, stop_sec: float | None = None,
                                   duration_sec: float = 120.0, step_sec: float = 0.25,
                                   raster_window_sec: float = 30.0, display_bin_sec: float = 0.1,
                                   show_asdr: bool = True, n_frames: int = 72) -> Path:
    """Animate the burst-labelled PC trajectory, either over time or by rotating a 3-D view.

    Two animations of the figure :func:`plot_pc_trajectory_by_label` draws:

    - ``mode="time"`` replays the recording. Left, the path through PC space drawn in as it
      happens, coloured by which burst is running; right, the raster for a window around that
      moment, tinted by the same labels, over the population rate with the bursts shaded. A burst
      then reads as a column of colour in the raster and, at the same moment, an excursion of the
      dot in PC space.
    - ``mode="spin"`` holds the finished 3-D trajectory still and sweeps the camera a full turn
      around it, which is how you see depth that a single 2-D projection hides. Needs three ``pcs``.

    :param rec: The recording.
    :param burst_df: Burst DataFrame, pre-filtered to the bursts of interest.
    :param save_path: Where to write: ``.gif`` (Pillow, always available) or ``.mp4`` (needs ffmpeg).
    :param label_col: Column of ``burst_df`` to colour by.
    :param mode: ``"time"`` (replay) or ``"spin"`` (rotate a 3-D view).
    :param phase: Phase whose PC basis is used; ``None`` selects the recording's first phase.
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :param pcs: Two or three components, 1-based. ``"spin"`` requires three.
    :param span: ``"isi_n"`` or ``"features"`` burst boundaries, as in
        :func:`plot_pc_trajectory_by_label`.
    :param labels: Label values to draw, in order; ``None`` takes every value in the column.
    :param colors: Optional ``{label: colour}`` overrides.
    :param result: A precomputed :func:`pca` result, so several figures of the same recording can
        share one eigendecomposition; ``None`` computes it.
    :param base_color: Colour of the path outside any burst.
    :param linewidth: Width of the in-burst lines.
    :param elev: 3-D elevation angle in degrees.
    :param azim: 3-D azimuth angle; for ``"spin"``, where the turn starts.
    :param fps: Frames per second of the output.
    :param dpi: Resolution of the output frames.
    :param max_frames: Refuse to render more frames than this (``None``: no limit).
    :param start_sec: ``"time"``: clip start, in seconds of the recording; ``None`` starts at the
        phase start. ``"spin"``: plot only from this time.
    :param stop_sec: ``"time"``: clip end, overriding ``duration_sec`` when earlier. ``"spin"``:
        plot only up to this time.
    :param duration_sec: ``"time"`` only: clip length in seconds of recording.
    :param step_sec: ``"time"`` only: recording time advanced per frame.
    :param raster_window_sec: ``"time"`` only: width of the raster window.
    :param display_bin_sec: ``"time"`` only: bin width the raster is summed into for display.
    :param show_asdr: ``"time"`` only: draw the population-rate strip under the raster.
    :param n_frames: ``"spin"`` only: frames in one full turn (72 is 5 degrees per frame).
    :returns: The path written.
    """
    from matplotlib import animation

    if mode not in ("time", "spin"):
        raise ValueError(f"mode must be 'time' or 'spin', not {mode!r}")
    if mode == "spin" and len(pcs) != 3:
        raise ValueError("mode='spin' rotates a 3-D plot; pcs must name three components, "
                         f"e.g. (1, 2, 3); got {pcs!r}")
    writer = _animation_writer(save_path, fps)
    params = params or NetworkParams()
    is_3d = len(pcs) == 3

    if mode == "spin":
        fig, ax, _result, _coords, _t, _codes, _labels, _colours, note = _labelled_trajectory_figure(
            rec, burst_df, label_col, phase=phase, params=params, pcs=pcs, span=span, labels=labels,
            colors=colors, result=result, start_sec=start_sec, stop_sec=stop_sec,
            base_color=base_color, linewidth=linewidth, elev=elev, azim=azim, legend=True, ax=None)
        ax.set_title(f"PC trajectory by {label_col} — "
                     f"{recording_label(rec.DIV, rec.experiment)}{note}", fontsize=11,
                     fontweight="bold")
        # endpoint=False: one full turn with no repeated frame, so the GIF loops seamlessly.
        azims = azim + np.linspace(0.0, 360.0, int(n_frames), endpoint=False)
        print(f"Animating {azims.size} frames (one full turn) -> {save_path}")
        if max_frames is not None and azims.size > max_frames:
            plt.close(fig)
            raise ValueError(f"{azims.size} frames exceeds max_frames={max_frames}; lower "
                             "n_frames, or pass max_frames=None.")

        def spin(now_azim):
            ax.view_init(elev=elev, azim=now_azim)
            return []

        # Freeze the layout: constrained_layout re-solves every frame, and a rotating 3-D
        # projection changes the bounding box, which makes the output jitter.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        anim = animation.FuncAnimation(fig, spin, frames=azims, blit=False)
        return _write_animation(anim, fig, save_path, writer, dpi)

    # --- mode="time" ---
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2 if show_asdr else 1, 2, width_ratios=[1, 1.5],
                            height_ratios=[4, 1] if show_asdr else [1])
    ax_pc = fig.add_subplot(grid[:, 0], projection="3d") if is_3d else fig.add_subplot(grid[:, 0])
    ax_raster = fig.add_subplot(grid[0, 1])
    ax_asdr = fig.add_subplot(grid[1, 1], sharex=ax_raster) if show_asdr else None

    # The whole phase is the backdrop, so the clip is drawn against where it will go.
    fig, ax_pc, result, coords, t, codes, labels, colours, note = _labelled_trajectory_figure(
        rec, burst_df, label_col, phase=phase, params=params, pcs=pcs, span=span, labels=labels,
        colors=colors, result=result, start_sec=None, stop_sec=None, base_color=base_color,
        linewidth=linewidth, elev=elev, azim=azim, legend=True, ax=ax_pc)
    for collection in ax_pc.collections:
        collection.set_alpha(0.25)
    if t.size < 2:
        plt.close(fig)
        raise ValueError("The phase is too short, or has too few active electrodes, to animate.")

    phase_start, phase_end = float(t[0]), float(t[-1])
    clip_start = phase_start if start_sec is None else float(start_sec)
    clip_end = min(clip_start + float(duration_sec), phase_end)
    if stop_sec is not None:
        clip_end = min(clip_end, float(stop_sec))
    if not phase_start <= clip_start < phase_end:
        raise ValueError(f"start_sec={clip_start:g} is outside the phase "
                         f"({phase_start:g}-{phase_end:g} s).")
    frame_times = np.arange(clip_start, clip_end, float(step_sec))
    print(f"Animating {frame_times.size} frames ({clip_start:.1f}-{clip_end:.1f} s) -> {save_path}")
    if max_frames is not None and frame_times.size > max_frames:
        plt.close(fig)
        raise ValueError(f"{frame_times.size} frames exceeds max_frames={max_frames}; shorten "
                         "duration_sec, raise step_sec, or pass max_frames=None.")

    seg_colours = _segment_colours(codes, labels, colours, base_color)
    if is_3d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        drawn = Line3DCollection([], linewidth=linewidth * 1.4, zorder=10)
        ax_pc.add_collection3d(drawn, autolim=False)
        (dot,) = ax_pc.plot([], [], [], "o", color="crimson", markersize=6, zorder=11)
    else:
        drawn = LineCollection([], linewidth=linewidth * 1.4, zorder=10)
        ax_pc.add_collection(drawn)
        (dot,) = ax_pc.plot([], [], "o", color="crimson", markersize=6, zorder=11)

    half = raster_window_sec / 2
    rows = np.arange(rec.spike_bin.shape[0])
    first_bin = max(int((clip_start - half) / rec.bin_size), 0)
    last_bin = min(int(np.ceil((clip_end + half) / rec.bin_size)), rec.spike_bin.shape[1])
    counts, factor = _display_raster(rec.spike_bin, rows, rec.bin_size, display_bin_sec, first_bin,
                                     last_bin)
    raster_t0, raster_dt = first_bin * rec.bin_size, factor * rec.bin_size
    raster_t1 = raster_t0 + counts.shape[1] * raster_dt
    centers = raster_t0 + (np.arange(counts.shape[1]) + 0.5) * raster_dt

    # Tint each raster column by the label running at that moment: white where the array is quiet,
    # saturating to the label's colour where it fires. Greyscale would hide the labelling entirely.
    col_codes, _, _ = _label_codes(centers, burst_df, label_col, labels, span, rec.samp_rate)
    col_rgb = np.array([to_rgb(colours[labels[c]]) if c >= 0 else to_rgb("black")
                        for c in col_codes])
    vmax = max(float(np.percentile(counts[counts > 0], 75)) if np.any(counts) else 1.0, 1.0)
    intensity = np.clip(counts / vmax, 0, 1)[..., None]
    ax_raster.imshow(1.0 - intensity * (1.0 - col_rgb[None, :, :]), aspect="auto",
                     interpolation="nearest", extent=(raster_t0, raster_t1, counts.shape[0] - 0.5,
                                                      -0.5))
    ax_raster.set_ylabel("Electrode", fontsize=9)
    ax_raster.tick_params(labelsize=7)
    cursors = [ax_raster.axvline(clip_start, color="crimson", linewidth=1.0)]

    if ax_asdr is not None:
        start_frames, stop_frames, _ = _burst_spans_frames(burst_df, span)
        label_of = {label: i for i, label in enumerate(labels)}
        for on, off, label in zip(start_frames / rec.samp_rate, stop_frames / rec.samp_rate,
                                  burst_df[label_col]):
            if label in label_of and off >= raster_t0 and on <= raster_t1:
                ax_asdr.axvspan(on, off, color=colours[label], alpha=0.25, linewidth=0)
        ax_asdr.plot(centers, counts.sum(axis=0) / raster_dt, color="black", linewidth=0.7)
        ax_asdr.set_ylabel("Spikes/s", fontsize=8)
        ax_asdr.set_xlabel("Time (s)", fontsize=9)
        ax_asdr.tick_params(labelsize=7)
        ax_asdr.spines[["top", "right"]].set_visible(False)
        cursors.append(ax_asdr.axvline(clip_start, color="crimson", linewidth=1.0))
        ax_raster.tick_params(labelbottom=False)
    else:
        ax_raster.set_xlabel("Time (s)", fontsize=9)

    unphased = is_unphased([result["phase"]])
    title = fig.suptitle("", fontsize=11, fontweight="bold")
    label_text = (f"{recording_label(rec.DIV, rec.experiment)} by {label_col}"
                  f"{phase_tag(result['phase'], unphased, '  (phase: {phase})')}{note}")
    start_idx = int(np.searchsorted(t, clip_start))

    def replay(now):
        k = int(np.searchsorted(t, now, side="right"))  # samples up to and including `now`
        if k - start_idx >= 2:
            drawn.set_segments(_segments(*[c[start_idx:k] for c in coords]))
            drawn.set_color(seg_colours[start_idx:k - 1])
        if k:
            if is_3d:
                dot.set_data_3d([coords[0][k - 1]], [coords[1][k - 1]], [coords[2][k - 1]])
            else:
                dot.set_data([coords[0][k - 1]], [coords[1][k - 1]])
        ax_raster.set_xlim(now - half, now + half)
        for cursor in cursors:
            cursor.set_xdata([now, now])
        title.set_text(f"{label_text}   t = {now:6.1f} s")
        return [drawn, dot, *cursors, title]

    anim = animation.FuncAnimation(fig, replay, frames=frame_times, blit=False)
    return _write_animation(anim, fig, save_path, writer, dpi)


# --- Gaussian Process Factor Analysis -------------------------------------------------------------

_SOURCES = ("spike_data", "spike_bin")


def _counts(rec, bin_sec: float, source: str) -> np.ndarray:
    """Spike counts per (electrode row, bin) at ``bin_sec``, rows aligned with ``rec.spike_bin``.

    ``"spike_data"`` bins the spike times, so a bin holding five spikes reads five. ``"spike_bin"``
    sums the binary matrix instead, which is what the rest of this module uses -- cheaper, but it
    has already clipped every bin to one spike (see the module note), so the counts it produces are
    a lower bound.

    Spikes at a negative frame number are dropped rather than folded into bin 0. Real recordings
    carry a few (eight out of 4.5 million on the recording this was measured on), from the offset
    between the recording's start and the first frame.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {list(_SOURCES)}, not {source!r}")
    channelmap = np.asarray(rec.channelmap)
    spike_bin = np.asarray(rec.spike_bin)
    n_rows = channelmap.shape[0]
    # Both sources go through the same whole-number decimation, so they return the same shape.
    decim = _decimation(rec.bin_size, bin_sec)
    n_bins = spike_bin.shape[1] // decim

    if source == "spike_bin":
        return spike_bin[:, :n_bins * decim].reshape(n_rows, n_bins, decim).sum(axis=2,
                                                                                dtype=np.float64)

    # channelmap column 1 is the channel id; spike_data stores the channel, so invert that mapping
    # once rather than searching it per spike.
    channels = channelmap[:, 1].astype(np.int64)
    row_of_channel = np.full(channels.max() + 1 if n_rows else 0, -1, dtype=np.int64)
    row_of_channel[channels] = np.arange(n_rows)

    channel = rec.spike_data["channel"].astype(np.int64)
    # Masked rather than clipped: a channel id outside the map would otherwise land on whichever
    # row sits at the clip bound and have its spikes counted as that electrode's.
    known = (channel >= 0) & (channel < row_of_channel.size)
    rows = np.where(known, row_of_channel[np.where(known, channel, 0)], -1)
    bins = (rec.spike_data["frameno"] / (rec.samp_rate * bin_sec)).astype(np.int64)
    keep = (rows >= 0) & (bins >= 0) & (bins < n_bins)
    flat = np.bincount(rows[keep] * n_bins + bins[keep], minlength=n_rows * n_bins)
    return flat.reshape(n_rows, n_bins).astype(np.float64)


def _gpfa_window(rec, phase: str | None, start_sec, stop_sec, bin_sec: float, n_bins: int):
    """Bin bounds of the window to fit, and the resolved phase name.

    The phase's own window first (as :func:`_phase_columns` does, in bins of ``bin_sec`` rather than
    of the rate matrix), then narrowed by ``start_sec`` / ``stop_sec``, which are absolute recording
    seconds as in :func:`plot_pc_trajectory_by_label`.
    """
    window = _resolve_phase(rec, phase)
    name = phase or "full" if window is None else window.name
    lo, hi = 0, n_bins
    if window is not None:
        frames_per_bin = rec.samp_rate * bin_sec
        lo = max(int(window.start_frame / frames_per_bin), 0)
        hi = min(int(window.end_frame / frames_per_bin), n_bins)
    if start_sec is not None:
        lo = max(lo, int(start_sec / bin_sec))
    if stop_sec is not None:
        hi = min(hi, int(np.ceil(stop_sec / bin_sec)))
    return lo, max(hi, lo), name


def _select_channels(counts: np.ndarray, bin_sec: float, params: GPFAParams) -> np.ndarray:
    """Rows of ``counts`` worth fitting: busy enough, and at most ``max_channels`` of them.

    A near-silent electrode adds a row to ``C`` and to ``R`` and no signal, and GPFA's cost is linear
    in the electrode count, so the quiet tail of a 1,000-electrode array is pure overhead.
    """
    totals = counts.sum(axis=1)
    rates = totals / max(counts.shape[1] * bin_sec, 1e-12)
    keep = np.flatnonzero(rates >= params.min_rate_hz)
    if params.max_channels is not None and keep.size > params.max_channels:
        # Busiest first, then back into row order so the loadings stay in array layout.
        keep = np.sort(keep[np.argsort(totals[keep])[::-1][:int(params.max_channels)]])
    return keep.astype(np.int32)


def gpfa(rec: Recording, analysis_dir=None, *, phase: str | None = None,
         start_sec: float | None = None, stop_sec: float | None = None, n_latents: int = 3,
         params: GPFAParams | None = None, save: bool = True) -> dict:
    """Gaussian Process Factor Analysis of one recording: a few smooth latent signals over time.

    Where :func:`pca` finds a subspace and says nothing about time, GPFA (Yu et al. 2009,
    *J Neurophysiol*) puts the smoothing inside the model. Each latent dimension carries a
    Gaussian-process prior with its own timescale, and that timescale is **fitted** alongside the
    loadings, so the smoothness of the trajectories is a property of the network rather than of our
    filter settings. The observation model is factor analysis, ``y = C x + d + noise``, on
    square-rooted spike counts.

    The window is cut into independent segments of ``params.segment_sec`` that share every
    parameter, since spontaneous activity has no trials to use. Any part of the window short of a
    whole segment is dropped, and said so. Latents are extracted per segment inside a padded window
    and stitched (see :class:`~mxtreme.params.GPFAParams`), so the trace has no dip at each seam.

    What comes back in ``latents`` is **orthonormalized**: ``C`` is replaced by the ``U`` of its SVD
    and the latents transformed to match, which leaves the model's predictions identical but orders
    the dimensions by the share of shared variance they carry, as PCA's components are. Plot those.
    ``latents_raw`` is the model's own basis, which ``tau_sec`` refers to -- read it only when the
    timescales are well separated, because latents with near-equal timescales can rotate freely into
    each other.

    .. note::
       On the culture data this was built against, **the fitted timescales saturate**: at bin widths
       of 20, 50, 100 and 250 ms, all three landed within 1% of whichever lower bound they were
       given. The shared signal in a bursting culture is the array-wide burst transient, which is
       sharper than any of those resolutions, so the GP asks for a delta and gets the bound. A
       warning is printed when that happens. Read ``tau_sec`` as a limit, not a measurement, unless
       it comes back clear of the bounds -- the latents and ``shared_var_frac`` are still fine.

    .. note::
       A fit takes about a minute on a full hour-long phase at defaults. Pass ``start_sec`` /
       ``stop_sec`` while exploring, and hand the result to :func:`plot_gpfa` through its ``result``
       argument rather than fitting again per figure.

    :param rec: The recording.
    :param analysis_dir: Analysis output root; saved there as ``..._gpfa_<phase>_p<n>.npz`` (plus the
        window when one is given) when ``save`` is on. The file name does **not** record the rest of
        ``params``, so a cached file from different settings is overwritten rather than kept apart;
        every setting is stored inside it.
    :param phase: Phase to use; ``None`` selects the recording's first phase.
    :param start_sec: Fit only from this time, in absolute recording seconds; ``None`` starts at the
        phase start.
    :param stop_sec: Fit only up to this time; ``None`` runs to the phase end.
    :param n_latents: Number of latent dimensions. Two or three is the usual starting point; it is
        clamped to the number of electrodes kept.
    :param params: Settings; defaults to :class:`~mxtreme.params.GPFAParams`.
    :param save: Write the result (needs ``analysis_dir``).
    :returns: ``{"latents": (p, n_bins) orthonormalized latents -- what you plot, "latents_raw":
        the model's own basis, "time_sec": each bin's time in the recording, "tau_sec": the GP
        timescale of each raw latent, "C", "C_orth": (n_kept, p) loadings, "d", "R": per-electrode
        offset and private variance, "singular_values", "shared_var_frac": each latent's share of
        ``trace(C C')``, "shared_var_total": shared over total variance, "channels", "elec_index":
        the electrodes kept, "log_likelihood": the EM trace, "n_iter", "converged", "bin_sec",
        "segment_sec", "n_segments", "n_latents", "phase", "window_sec"}``.
    """
    params = params or GPFAParams()
    if int(n_latents) < 1:
        raise ValueError(f"n_latents must be at least 1, not {n_latents!r}")
    bin_sec = _decimation(rec.bin_size, params.bin_sec) * rec.bin_size
    if params.learn_tau and params.tau_bounds_sec[0] < bin_sec:
        raise ValueError(f"tau_bounds_sec starts at {params.tau_bounds_sec[0]} s, below the "
                         f"{bin_sec:g} s bin width; a timescale under one bin is indistinguishable "
                         "from noise. Raise the lower bound or lower bin_sec.")

    counts = _counts(rec, bin_sec, params.source)
    lo, hi, name = _gpfa_window(rec, phase, start_sec, stop_sec, bin_sec, counts.shape[1])
    n_bins_per_seg = max(round(params.segment_sec / bin_sec), 1)
    if hi - lo < n_bins_per_seg:
        raise ValueError(f"the window is {(hi - lo) * bin_sec:.3g} s but segment_sec is "
                         f"{params.segment_sec:g} s, so there is not one whole segment to fit. "
                         "Widen the window or lower segment_sec.")

    keep = _select_channels(counts[:, lo:hi], bin_sec, params)
    window = counts[np.ix_(keep, np.arange(lo, hi))]
    if params.sqrt_transform:
        window = np.sqrt(window)
    dropped = (hi - lo) % n_bins_per_seg
    if dropped:
        print(f"  dropping the last {dropped * bin_sec:.3g} s: short of a whole segment")

    model = _gpfa.fit(window, params, n_latents=n_latents, bin_sec=bin_sec,
                      n_bins_per_seg=n_bins_per_seg)
    pad = _gpfa.pad_bins(params, model["tau_sec"], n_bins_per_seg, bin_sec)
    raw = _gpfa.latents(window, model, params, bin_sec=bin_sec, n_bins_per_seg=n_bins_per_seg,
                        pad=pad)
    c_orth, orth, singular, var_frac = _gpfa.orthonormalize(model["C"], raw)
    shared = float(np.sum(model["C"] ** 2))
    total_var = shared + float(np.sum(model["R"]))

    n_kept = raw.shape[1]
    result = {
        "latents": orth.astype(np.float32), "latents_raw": raw.astype(np.float32),
        "time_sec": (lo + np.arange(n_kept)) * bin_sec, "tau_sec": model["tau_sec"],
        "C": model["C"], "C_orth": c_orth, "d": model["d"], "R": model["R"],
        "singular_values": singular, "shared_var_frac": var_frac,
        "shared_var_total": shared / total_var if total_var else np.nan,
        "channels": np.asarray(rec.channelmap)[keep, 1].astype(np.int64), "elec_index": keep,
        "log_likelihood": model["log_likelihood"], "n_iter": model["n_iter"],
        "converged": model["converged"], "bin_sec": bin_sec,
        "segment_sec": n_bins_per_seg * bin_sec, "n_segments": model["n_segments"],
        "n_latents": orth.shape[0], "phase": name,
        "window_sec": np.array([lo * bin_sec, (lo + n_kept) * bin_sec]),
    }

    if save and analysis_dir is not None:
        windowed = "" if start_sec is None and stop_sec is None else \
            f"_{round(result['window_sec'][0])}-{round(result['window_sec'][1])}s"
        path = _recording_output(rec, analysis_dir,
                                 f"gpfa_{name}_p{result['n_latents']}{windowed}.npz")
        stored = {k: v for k, v in result.items() if k != "phase"}
        np.savez_compressed(path, phase=np.array(name), **stored,
                            **{f"params_{f.name}": np.array(getattr(params, f.name))
                               for f in fields(params)})
        print(f"Saved GPFA to {path}")
    return result


def _gpfa_result(rec, result, *, phase, start_sec, stop_sec, n_latents, params):
    """The caller's precomputed :func:`gpfa` result, or a fresh fit.

    Worth more than :func:`_pca_result`'s equivalent: a fit costs minutes, so refitting once per
    figure is not something to do by accident.
    """
    if result is None:
        return gpfa(rec, phase=phase, start_sec=start_sec, stop_sec=stop_sec, n_latents=n_latents,
                    params=params, save=False)
    missing = {"latents", "time_sec", "phase"} - set(result)
    if missing:
        raise ValueError(f"result is missing {sorted(missing)}; pass a dict from network.gpfa().")
    if phase is not None and result["phase"] != phase:
        raise ValueError(f"result was computed for phase {result['phase']!r}, but phase={phase!r} "
                         "was requested; pass one or the other.")
    return result


def plot_gpfa(rec: Recording, *, phase: str | None = None, start_sec: float | None = None,
              stop_sec: float | None = None, n_latents: int = 3, params: GPFAParams | None = None,
              result: dict | None = None, burst_df=None, label_col: str = "kind", labels=None,
              colors=None, span: str = "isi_n", time_unit: str = "min", offset=True,
              base_color: str = "0.85", linewidth: float = 1.0, legend: bool = True, ax=None,
              show_plot: bool = True, save_path=None, dpi: int = 150):
    """The GPFA latents against time, optionally coloured by which kind of burst is happening.

    One trace per latent dimension, stacked, strongest first (see :func:`gpfa` on
    orthonormalization). Given a ``burst_df``, every sample inside a burst takes that burst's label
    colour and everything else stays grey -- the same encoding as
    :func:`plot_pc_trajectory_by_label`, so the two read alike. That is what shows whether network
    bursts are excursions of the leading latent while minis stay near zero, or whether a kind of
    burst has a signature of its own further down.

    Any column of the burst table works as the label: ``"kind"``, a community assignment, a cluster
    id you added. Bursts are taken exactly as given -- filter ``burst_df`` before calling.

    :param rec: The recording.
    :param phase: Phase to use; ``None`` selects the recording's first phase.
    :param start_sec: Start of the window, in absolute recording seconds; ``None`` is the phase
        start. With a precomputed ``result`` this narrows what is drawn, which is how you zoom into
        a long fit without paying for it again.
    :param stop_sec: End of the window; ``None`` is the phase end.
    :param n_latents: Number of latent dimensions to fit (ignored when ``result`` is given).
    :param params: Settings; defaults to :class:`~mxtreme.params.GPFAParams`.
    :param result: A precomputed :func:`gpfa` result to plot, so a fit can serve several figures;
        ``None`` fits, which takes minutes on a long window. Its phase must match ``phase`` if given.
    :param burst_df: Burst DataFrame from :meth:`BurstSet.to_dataframe
        <mxtreme.bursting.detection.BurstSet.to_dataframe>`; ``None`` draws plain traces.
    :param label_col: Column of ``burst_df`` to colour by.
    :param labels: Label values to draw, in order (later ones win where bursts overlap); ``None``
        takes every value in the column, naturally sorted.
    :param colors: Optional ``{label: colour}`` overrides; the rest come from
        :data:`~mxtreme.analysis._plotting.CATEGORICAL_COLORS`.
    :param span: Which burst boundaries to paint -- ``"isi_n"`` for the ISI-N group bounds, or
        ``"features"`` for the extracted onset/offset.
    :param time_unit: ``"min"`` or ``"s"`` for the time axis.
    :param offset: Vertical spacing between traces: ``True`` picks one from their spread, ``False``
        overlays them, a number sets it.
    :param base_color: Colour of a trace outside any burst.
    :param linewidth: Width of the traces.
    :param legend: Draw the per-label legend, with each label's burst count.
    :param ax: Axes to draw on; a new figure when ``None``.
    :param show_plot: Show the figure (only when this function created it).
    :param save_path: Full path (including filename) to save the figure to.
    :param dpi: Resolution for the saved figure and the rasterized traces.
    :returns: The matplotlib Figure.
    """
    if time_unit not in _TIME_UNITS:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}, not {time_unit!r}")
    result = _gpfa_result(rec, result, phase=phase, start_sec=start_sec, stop_sec=stop_sec,
                          n_latents=n_latents, params=params)
    traces = np.asarray(result["latents"], dtype=float)
    t_sec = np.asarray(result["time_sec"], dtype=float)
    # Applied to the samples as well as to the fit, so a window means the same thing whether the fit
    # happened here or was handed in through `result` -- a precomputed result covers its own window,
    # and narrowing it further is how you zoom in without refitting.
    keep = np.ones(t_sec.shape, bool)
    if start_sec is not None:
        keep &= t_sec >= start_sec
    if stop_sec is not None:
        keep &= t_sec <= stop_sec
    traces, t_sec = traces[:, keep], t_sec[keep]
    if t_sec.size < 2:
        raise ValueError(f"the window leaves {t_sec.size} sample(s) of the fit to plot; widen "
                         "start_sec/stop_sec, or check they fall inside the fitted window "
                         f"{np.asarray(result.get('window_sec', [np.nan, np.nan]))}.")
    t = t_sec / _TIME_UNITS[time_unit]
    p = traces.shape[0]

    codes, counts, note = None, {}, ""
    if burst_df is not None:
        if label_col not in burst_df.columns:
            raise ValueError(f"burst_df has no column {label_col!r}; available: "
                             f"{sorted(burst_df.columns)}")
        if labels is None:
            # Sorted, not first-seen: colour assignment must not depend on burst_df's row order.
            labels = sorted(burst_df[label_col].dropna().unique(), key=natural_key)
        else:
            labels = list(labels)
        codes, counts, n_fell_back = _label_codes(t_sec, burst_df, label_col, labels, span,
                                                  rec.samp_rate)
        note = (f" · {n_fell_back}/{len(burst_df)} bursts use ISI-N bounds (no features)"
                if n_fell_back else "")
    colours = categorical_colors(labels or [], colors)

    # A fixed step keeps the traces apart without rescaling any of them, so their amplitudes stay
    # comparable between dimensions -- the leading latent really is the biggest.
    if offset is True:
        spread = float(np.percentile(np.abs(traces), 99)) if traces.size else 1.0
        step = 3.0 * (spread or 1.0)
    else:
        step = 0.0 if offset is False else float(offset)
    centres = -step * np.arange(p)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(10, 1.1 * p + 2.0), constrained_layout=True)
    else:
        fig = ax.figure

    for i in range(p):
        y = traces[i] + centres[i]
        if codes is None:
            ax.plot(t, y, color="#333333", linewidth=linewidth, rasterized=True)
            continue
        segments = _segments(t, y)
        seg_codes = codes[:-1]
        for code, colour, width, zorder in [(_NO_BURST, base_color, linewidth * 0.8, 1)] + [
                (c, colours[label], linewidth, 2 + c) for c, label in enumerate(labels)]:
            selected = seg_codes == code
            if not selected.any():
                continue
            lines = LineCollection(segments[selected], colors=colour, linewidth=width,
                                   zorder=zorder)
            # An hour at 20 ms is 180k segments per trace: as vectors that bloats a PDF.
            lines.set_rasterized(True)
            ax.add_collection(lines)
    if codes is not None:
        ax.set_xlim(t[0], t[-1])
        ax.autoscale_view(scalex=False)

    ax.set_yticks(centres)
    ax.set_yticklabels([f"x{i + 1}" + (f"\n{result['shared_var_frac'][i]:.0%}"
                                       if "shared_var_frac" in result else "")
                        for i in range(p)], fontsize=8)
    ax.set_xlabel(f"Time ({time_unit})", fontsize=9)
    ax.set_ylabel("Latent", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if legend and codes is not None:
        handles = [Line2D([], [], color=colours[label], lw=2, label=f"{label} (n={counts[label]})")
                   for label in labels]
        handles.append(Line2D([], [], color=base_color, lw=2, label="no burst"))
        # The legend sits in a strip of headroom opened above the traces, rather than over the
        # leading latent -- which is exactly where the bursts it is labelling are.
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.18 * (high - low))
        ax.legend(handles=handles, fontsize=8, frameon=False, ncol=len(handles),
                  loc="upper center", bbox_to_anchor=(0.5, 1.0))

    if owns_fig:
        unphased = is_unphased([result["phase"]])
        taus = ", ".join(f"{tau:.3g}" for tau in np.asarray(result.get("tau_sec", [])))
        by = f" by {label_col}" if codes is not None else ""
        ax.set_title(f"GPFA latents{by} — {recording_label(rec.DIV, rec.experiment)}"
                     f"{phase_tag(result['phase'], unphased, '  (phase: {phase})')}"
                     f"  (τ = {taus} s){note}", fontsize=11, fontweight="bold")
        return finish_figure(fig, save_path=save_path, show_plot=show_plot, dpi=dpi)
    return fig
