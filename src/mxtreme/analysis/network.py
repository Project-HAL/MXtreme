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
  which plays it alongside the spike raster).

:func:`network_metrics` reduces both to per-phase scalars, and the group-level
:func:`summarize_network_metrics` / :func:`plot_network_summary` take a ``RecordingID``,
``CultureID``, ``CultureSelector`` or several selectors, as in :mod:`~mxtreme.analysis.activity`.

The paper uses **spontaneous** activity (its Fig. 2 is the pre-training baseline). Nothing here
enforces that: every result is per phase, so reading the ``pre`` phase is the equivalent.

.. note::
   The smoothed rate is built from ``Recording.spike_bin``, which is **binary**. On a real hour-long
   recording that discards ~30% of spikes, inside bursts, which is where the correlation structure
   lives -- so correlations here are a lower bound on burst-driven coupling.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from mxtreme import io
from mxtreme.config import Config
from mxtreme.identity import CultureID
from mxtreme.params import CommunityParams, NetworkParams
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.recording import Recording
from mxtreme.utils import get_n_colors
from mxtreme.visualizations import finish_figure
from mxtreme.analysis import _communities
from mxtreme.analysis.activity import _double_exponential, _decimation
from mxtreme.analysis._group import SummarySpec, culture_summary, fill_docs, plot_summary, summarize
from mxtreme.analysis._paths import _summary_paths, recording_key, recording_label, unflatten_recording_arrays
from mxtreme.analysis._plotting import is_unphased, phase_tag, sort_phases

__all__ = [
    "connectivity", "communities", "community_table", "pca", "network_metrics",
    "summarize_network_metrics", "plot_network_summary",
    "plot_connectivity", "plot_pc_trajectory", "animate_pc_trajectory",
    "plot_connectivity_grid", "plot_pc_variance_explained", "plot_pc_trajectory_grid",
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


def _pc_indices(pcs, n_available: int) -> tuple[int, int]:
    """0-based rows of ``scores`` for the 1-based component numbers ``pcs``, checked."""
    if len(pcs) != 2 or min(pcs) < 1:
        raise ValueError(f"pcs must be two 1-based component numbers, e.g. (1, 2); got {pcs!r}")
    if max(pcs) > n_available:
        raise ValueError(f"PC{max(pcs)} requested but only {n_available} available.")
    return pcs[0] - 1, pcs[1] - 1


def _segments(x, y) -> np.ndarray:
    """Consecutive point pairs of a polyline, the shape :class:`LineCollection` takes."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


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
                       pcs=(1, 2), cmap: str = "viridis", time_unit: str = "min", ax=None,
                       colorbar: bool = True, show_plot: bool = True, save_path=None, dpi: int = 150):
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
    :param ax: Axes to draw on; a new figure when ``None``.
    :param colorbar: Draw a time colour bar.
    :param show_plot: Show the figure (only when this function created it).
    :param save_path: Full path (including filename) to save the figure to.
    :param dpi: Resolution for the saved figure and the rasterized line.
    :returns: The matplotlib Figure.
    """
    if time_unit not in _TIME_UNITS:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}, not {time_unit!r}")
    result = pca(rec, phase=phase, params=params, n_components=max(pcs), save=False)

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


def _display_raster(spike_bin, rows, bin_size: float, display_bin_sec: float, first_bin: int,
                    last_bin: int):
    """Spike counts of ``rows`` summed into ``display_bin_sec`` bins over ``[first_bin, last_bin)``: a
    raster small enough to redraw every frame. Returns ``(counts, bins_per_display_bin)``."""
    factor = max(round(display_bin_sec / bin_size), 1)
    last_bin = first_bin + (last_bin - first_bin) // factor * factor
    # Slice time before rows: the time slice is a view, so only the clip is ever copied.
    block = np.asarray(spike_bin[:, first_bin:last_bin][rows], dtype=np.float32)
    return block.reshape(block.shape[0], -1, factor).sum(axis=2), factor


def animate_pc_trajectory(rec: Recording, save_path, *, phase: str | None = None,
                          params: NetworkParams | None = None, pcs=(1, 2),
                          start_sec: float | None = None, duration_sec: float = 120.0,
                          step_sec: float = 0.25, raster_window_sec: float = 30.0,
                          display_bin_sec: float = 0.1, order: str = "index",
                          community_params: CommunityParams | None = None, show_asdr: bool = True,
                          fps: int = 20, dpi: int = 90, cmap: str = "viridis",
                          max_frames: int | None = 3000) -> Path:
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
    result = pca(rec, phase=phase, params=params, n_components=max(pcs), save=False)
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
