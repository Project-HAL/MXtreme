"""Functional connectivity and dimensionality of a network's spontaneous dynamics.

Where the other topic modules describe electrodes one at a time (:mod:`~mxtreme.analysis.activity`) or
where they sit (:mod:`~mxtreme.analysis.spatial`), this one describes how they relate to *each other*:
how correlated the array is, and how many independent components its activity actually spans. That is
what separates a culture whose electrodes fire together as one blob from one with differentiated
sub-populations -- two cultures that can look identical in mean firing rate.

Both analyses follow Sono et al. 2026 (PNAS), *"Online supervised learning of temporal patterns in
biological neural networks under feedback control"*, Fig. 2:

**Functional connectivity** (their Fig. 2 B, C) is the Pearson correlation ``r_ij`` between every pair
of electrodes, computed on a *firing rate* rather than on the raw spike train. The rate is the spike
train convolved with a double exponential (their Eq. 5-6; rise 0.05 s, decay 2 s), which is what
:func:`firing_rate_matrix` produces. Fig. 2B is one electrode x electrode heatmap per network
(:func:`plot_connectivity_matrix`); Fig. 2C compares the network-averaged mean of all pairwise
``r_ij`` across network types -- here, ``mean_corr`` tracked over DIV.

**Dimensionality** (their Fig. 2D) eigendecomposes the covariance of that same rate matrix and plots
cumulative variance explained against component count (:func:`plot_variance_explained`). Alongside the
component counts this reports the **effective rank** ``exp(H(p))`` of the normalized eigenvalue
spectrum (their Eq. 17-18, used in their Fig. 4G) -- a continuous stand-in for "how many components it
takes", which tracks over DIV far better than the integer counts do.

The paper runs both on **spontaneous** activity specifically (its Fig. 2 is pre-training baseline
characterization). Nothing here enforces that: :func:`network_summary` emits one row per (DIV, phase),
so restricting to spontaneous activity is a matter of reading the ``pre`` phase rather than a
restriction baked into the module.

.. note::
   The rate is built from ``Recording.spike_bin``, which is **binary** -- a bin holding five spikes and
   a bin holding one are both 1. On a real hour-long recording this discards ~30% of spikes, and it
   discards them inside bursts, which is exactly where the correlation structure lives. Correlations
   here are therefore computed on a burst-clipped signal and should be read as a lower bound on
   burst-driven coupling. Rebuilding counts from ``Recording.spike_data`` would remove the bias.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from mxtreme import io
from mxtreme.params import NetworkParams
from mxtreme.recording import Recording
from mxtreme.utils import get_n_colors
from mxtreme.analysis._paths import _summary_paths, load_population_summaries
from mxtreme.analysis._plotting import plot_metric_grid, sort_phases
from mxtreme.analysis._stats import aggregate_by_div_phase


def _load_recording(npz):
    """Load a :class:`~mxtreme.recording.Recording` from ``npz`` (phases resolved from its own spec)."""
    return Recording(0, io.load_preprocessed(npz))


def _decimation_factor(bin_size: float, params: NetworkParams) -> int:
    """Whole-number decimation from the recording's ``bin_size`` to ``params.sample_bin_sec``."""
    return max(round(params.sample_bin_sec / float(bin_size)), 1)


def _pc_column(threshold: float) -> str:
    """Column name for the component count at a cumulative-variance ``threshold`` (0.9 -> ``n_pc_90``)."""
    return f"n_pc_{round(threshold * 100)}"


# --- compute layer (pure: no I/O, no plotting) ----------------------------------------------------


def firing_rate_matrix(spike_bin, bin_size: float, params: NetworkParams | None = None) -> np.ndarray:
    """Double-exponential-filtered firing rate per electrode (Sono et al. Eq. 5-6).

    The filter is applied as the difference of two first-order IIR recursions (one per time constant)
    rather than as convolution with a truncated kernel: it is O(n), causal, and carries no
    kernel-truncation error. The result is decimated to ``params.sample_bin_sec``, which is what keeps
    a full recording's rate matrix in a few hundred MB rather than 1.5 GB.

    Filtering is chunked over electrodes so peak memory scales with ``params.chunk_channels`` rather
    than with the electrode count.

    :param spike_bin: ``(n_channels, n_bins)`` binned spike matrix (``Recording.spike_bin``).
    :param bin_size: Width of one input bin, in seconds (``Recording.bin_size``).
    :param params: Filter settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :returns: ``(n_channels, n_samples)`` float32 rate matrix at ``params.sample_bin_sec`` resolution.
    """
    params = params or NetworkParams()

    sb = np.asarray(spike_bin)
    if sb.ndim != 2:
        raise ValueError(f"spike_bin must be 2-D (n_channels, n_bins); got shape {sb.shape}")

    tau_r, tau_d = float(params.tau_rise_sec), float(params.tau_decay_sec)
    if not 0 < tau_r < tau_d:
        raise ValueError(
            f"need 0 < tau_rise_sec < tau_decay_sec; got {tau_r} and {tau_d}"
        )

    dt = float(bin_size)
    n_chan, n_bins = sb.shape
    decim = _decimation_factor(dt, params)
    n_out = -(-n_bins // decim)  # ceil division

    # Pole of each single-exponential recursion, and the scale that makes the difference of the two
    # integrate to one spike's worth of rate.
    a_rise, a_decay = np.exp(-dt / tau_r), np.exp(-dt / tau_d)
    scale = dt / (tau_d - tau_r)

    rates = np.empty((n_chan, n_out), dtype=np.float32)
    step = max(int(params.chunk_channels), 1)

    for start in range(0, n_chan, step):
        chunk = sb[start:start + step].astype(np.float32)
        filtered = lfilter([1.0], [1.0, -a_decay], chunk, axis=1)
        filtered -= lfilter([1.0], [1.0, -a_rise], chunk, axis=1)
        filtered *= scale
        rates[start:start + step] = filtered[:, ::decim]

    return rates


def connectivity_matrix(rates, min_var: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Pearson correlation between electrodes (Sono et al. Fig. 2B).

    Electrodes whose rate never varies -- a silent electrode, most often -- are dropped first. Left in,
    each would contribute an entire row and column of NaN (a zero denominator) and poison every
    summary statistic taken over the matrix.

    :param rates: ``(n_channels, n_samples)`` rate matrix, e.g. from :func:`firing_rate_matrix`.
    :param min_var: Electrodes with variance at or below this are dropped.
    :returns: ``(corr, keep)`` -- the ``(n_kept, n_kept)`` float32 correlation matrix, and the indices
        of the electrodes it describes (rows of ``rates``), so matrix positions stay traceable back to
        electrodes.
    """
    rates = np.asarray(rates)
    if rates.ndim != 2:
        raise ValueError(f"rates must be 2-D (n_channels, n_samples); got shape {rates.shape}")

    keep = np.flatnonzero(rates.var(axis=1) > min_var).astype(np.int32)
    if keep.size < 2:
        return np.zeros((keep.size, keep.size), dtype=np.float32), keep

    corr = np.corrcoef(rates[keep])
    return np.asarray(corr, dtype=np.float32), keep


def variance_explained(rates) -> np.ndarray:
    """Normalized eigenvalue spectrum of the electrode covariance (Sono et al. Fig. 2D).

    Each electrode is mean-centered, the ``d x d`` covariance is eigendecomposed, and the eigenvalues
    are normalized so ``p_i`` is the fraction of variance explained by principal component ``i``.
    Covariance-side rather than SVD-side because ``d`` is the electrode count (at most ~1024 on this
    hardware) while the sample count is orders of magnitude larger.

    :param rates: ``(n_channels, n_samples)`` rate matrix, e.g. from :func:`firing_rate_matrix`.
    :returns: ``p``, descending, summing to 1. Empty if the input has fewer than two samples.
    """
    rates = np.asarray(rates)
    if rates.ndim != 2:
        raise ValueError(f"rates must be 2-D (n_channels, n_samples); got shape {rates.shape}")
    if rates.shape[0] < 1 or rates.shape[1] < 2:
        return np.array([], dtype=float)

    centered = rates - rates.mean(axis=1, keepdims=True)
    cov = (centered @ centered.T) / (centered.shape[1] - 1)

    eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))[::-1]
    eigvals = np.clip(eigvals, 0.0, None)  # tiny negatives are float noise on a PSD matrix

    total = eigvals.sum()
    if total <= 0:
        return np.zeros(eigvals.size, dtype=float)
    return eigvals / total


def effective_rank(p) -> float:
    """``exp(Shannon entropy(p))`` of a normalized eigenvalue spectrum (Sono et al. Eq. 17-18).

    A continuous count of how many components the dynamics really span: 1 when a single component
    carries everything, ``d`` when variance is spread evenly across ``d`` of them. Preferred over a
    cumulative-variance component count for tracking over DIV, which steps in integers and so moves in
    jumps where this moves smoothly.

    :param p: Normalized eigenvalues, e.g. from :func:`variance_explained`.
    :returns: The effective rank, or NaN for an empty spectrum.
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_network_metrics(rates, params: NetworkParams | None = None) -> dict:
    """Connectivity and dimensionality scalars for one rate matrix.

    :param rates: ``(n_channels, n_samples)`` rate matrix, e.g. from :func:`firing_rate_matrix`.
    :param params: Settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :returns: dict of metrics -- ``n_electrodes_used``, ``mean_corr`` / ``median_corr`` / ``std_corr``
        (over the off-diagonal pairs only), ``pc1_var_frac``, ``effective_rank``, and one
        ``n_pc_<pct>`` per entry in ``params.var_thresholds``.
    """
    params = params or NetworkParams()

    corr, keep = connectivity_matrix(rates, min_var=params.min_var)
    p = variance_explained(np.asarray(rates)[keep])

    metrics = {
        'n_electrodes_used': int(keep.size),
        'mean_corr':         np.nan,
        'median_corr':       np.nan,
        'std_corr':          np.nan,
        'pc1_var_frac':      np.nan,
        'effective_rank':    effective_rank(p),
    }

    if corr.shape[0] >= 2:
        off_diagonal = corr[np.triu_indices_from(corr, k=1)]
        metrics['mean_corr'] = float(np.nanmean(off_diagonal))
        metrics['median_corr'] = float(np.nanmedian(off_diagonal))
        metrics['std_corr'] = float(np.nanstd(off_diagonal))

    if p.size:
        metrics['pc1_var_frac'] = float(p[0])

    cumulative = np.cumsum(p) if p.size else np.array([])
    for threshold in params.var_thresholds:
        # searchsorted finds the first component whose cumulative variance reaches the threshold;
        # +1 turns that index into a count.
        count = int(np.searchsorted(cumulative, threshold) + 1) if cumulative.size else np.nan
        metrics[_pc_column(threshold)] = count

    return metrics


# --- per-culture summary --------------------------------------------------------------------------


def _phase_columns(rec, phase, decim: int, n_samples: int) -> tuple[int, int]:
    """Column bounds of ``phase``'s window in a rate matrix decimated by ``decim``."""
    frames_per_sample = rec.samp_rate * rec.bin_size * decim
    start = int(phase.start_frame / frames_per_sample)
    stop = int(phase.end_frame / frames_per_sample)
    return max(start, 0), min(stop, n_samples)


def network_summary(cpath, analysis_dir: Path, use_existing=True, show_plot=True, save_plot=False,
                    params: NetworkParams | None = None):
    """Connectivity and dimensionality metrics per (DIV, phase) for one culture.

    One row per phase per recording, so an unphased culture yields a single ``"full"`` row per DIV and
    a phased one yields a row per phase. The filter runs **once** per recording over the whole
    timebase and each phase is then a column slice of the result -- cheaper than one pass per phase,
    and it lets a phase inherit the decay tail of whatever preceded it rather than starting from zero.

    :param cpath: The culture's :class:`~mxtreme.paths.CulturePaths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param use_existing: Reuse the cached CSV when one exists. The cache is keyed by culture only, so
        after changing ``params`` you must pass ``False`` to force a recompute.
    :param show_plot: Show the per-culture metric grid.
    :param save_plot: Write that grid next to the CSV.
    :param params: Settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :returns: The summary :class:`~pandas.DataFrame`.
    """
    params = params or NetworkParams()
    cid = cpath.culture_id

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "network", "network_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:
        rows = []

        for div in sorted(cpath.recordings):
            rec = _load_recording(cpath.recordings[div].npz)

            decim = _decimation_factor(rec.bin_size, params)
            rates = firing_rate_matrix(rec.spike_bin, rec.bin_size, params)
            n_samples = rates.shape[1]

            for p in rec.phases:
                start, stop = _phase_columns(rec, p, decim, n_samples)
                window = rates[:, start:stop]

                rows.append({
                    'culture_id': str(cid),
                    'div': div,
                    'phase': p.name,
                    **compute_network_metrics(window, params),
                    # Stamped so a cached CSV says what produced it -- `_summary_paths` keys on the
                    # culture alone, so the filename can't.
                    'tau_rise_sec': params.tau_rise_sec,
                    'tau_decay_sec': params.tau_decay_sec,
                    'sample_bin_sec': params.sample_bin_sec,
                })

            del rates

        summary_df = pd.DataFrame(rows)

        os.makedirs(save_path, exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_network_summary(summary_df, cid, analysis_dir=save_path,
                              show_plot=show_plot, save_plot=save_plot)

    return summary_df


def _plot_network_summary(df: pd.DataFrame, cid, analysis_dir: Path, show_plot=True, save_plot=False):
    """Four-panel grid of connectivity/dimensionality metrics vs DIV (see :func:`network_summary`)."""

    metrics = [
        # (y_col,            err_col, y_label,                      panel_title)
        ('mean_corr',        None,    'Mean pairwise r',            'Functional Connectivity'),
        ('effective_rank',   None,    'exp(entropy of spectrum)',   'Effective Rank'),
        ('n_pc_90',          None,    'Components',                 'PCs for 90% Variance'),
        ('pc1_var_frac',     None,    'Variance explained',         'PC1 Dominance'),
    ]

    save_path = (analysis_dir / f"{cid}_network_summary.png") if save_plot else None

    plot_metric_grid(
        df, metrics,
        suptitle=f'Network Summary — {cid}',
        save_path=save_path,
        show_plot=show_plot,
        figsize=(11, 8),
        dpi=150,
    )


# --- per-culture matrices and spectra ------------------------------------------------------------
#
# The summary above reduces each recording to scalars. These keep the two objects the paper actually
# plots -- the correlation matrix and the variance curve -- which are arrays, not scalars, and so are
# cached as an .npz rather than a CSV.


CONNECTIVITY_KEYS = ('corr', 'cumvar', 'elec_index')


def _resolve_phase(rec, phase: str | None):
    """Resolve ``phase`` against ``rec``'s own phases.

    ``None`` selects the recording's first phase -- ``"full"`` for a recording with no user-supplied
    phases, the first of the sequence otherwise. A name the recording doesn't carry falls back to the
    whole recording, so one phase name can be applied across a batch whose recordings don't all share
    a phase spec.
    """
    named = {p.name: p for p in rec.phases}

    if phase is None:
        return named[sort_phases(list(named))[0]]
    if phase in named:
        return named[phase]

    print(f"  no {phase!r} phase (found: {', '.join(named)}); using the whole recording")
    return None


def culture_connectivity(cpath, analysis_dir: Path, *, phase: str | None = None,
                         use_existing: bool = True, params: NetworkParams | None = None):
    """Per-DIV correlation matrices and variance curves for one culture.

    Returns ``{div: {key: array}}`` with these keys:

    ============== =========================================================================
    ``corr``       ``(n, n)`` electrode x electrode Pearson correlation (Sono et al. Fig. 2B)
    ``cumvar``     cumulative variance explained vs component count (Sono et al. Fig. 2D)
    ``elec_index`` rows of ``Recording.spike_bin`` the ``corr`` axes correspond to
    ============== =========================================================================

    Each is restricted to ``phase``'s window, matching how :func:`network_summary` computes its
    statistics. The result is cached as one ``.npz`` per culture per phase: the arrays are small next
    to the cost of producing them, which means decompressing and filtering every recording's full
    spike matrix.

    :param cpath: The culture's :class:`~mxtreme.paths.CulturePaths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param phase: Phase to restrict to; ``None`` selects each recording's first phase.
    :param use_existing: Reuse the cached ``.npz`` when one exists.
    :param params: Settings; defaults to :class:`~mxtreme.params.NetworkParams`.
    :returns: ``{div: {key: numpy array}}``, keyed by DIV.
    """
    params = params or NetworkParams()

    save_path, npz_path = _summary_paths(
        cpath, analysis_dir, "network", f"{phase or 'default'}_connectivity", ext=".npz"
    )

    if use_existing and npz_path.exists():
        print(f"Loading existing connectivity from {npz_path}")
        with np.load(npz_path) as cached:
            conn: dict[int, dict[str, np.ndarray]] = {}
            for flat_key in cached.files:
                div_str, key = flat_key.split("__", 1)
                conn.setdefault(int(div_str), {})[key] = cached[flat_key]
        return conn

    conn = {}
    for div in sorted(cpath.recordings):
        rec = _load_recording(cpath.recordings[div].npz)

        decim = _decimation_factor(rec.bin_size, params)
        rates = firing_rate_matrix(rec.spike_bin, rec.bin_size, params)

        window = _resolve_phase(rec, phase)
        if window is not None:
            start, stop = _phase_columns(rec, window, decim, rates.shape[1])
            rates = rates[:, start:stop]

        corr, keep = connectivity_matrix(rates, min_var=params.min_var)
        p = variance_explained(rates[keep])

        conn[div] = {
            'corr':       corr,
            'cumvar':     np.cumsum(p),
            'elec_index': keep,
        }

    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(
        npz_path,
        **{f"{div}__{key}": values for div, per_div in conn.items() for key, values in per_div.items()},
    )
    print(f"Saved connectivity to {npz_path}")

    return conn


# --- plots ----------------------------------------------------------------------------------------


def plot_connectivity_matrix(corr, ax=None, title=None, vmin=-1.0, vmax=1.0, colorbar=True):
    """Electrode x electrode correlation heatmap (Sono et al. Fig. 2B).

    Electrodes stay in their raw index order rather than being clustered: that ordering follows the
    array's physical layout, which is what makes block structure from spatially separated
    sub-populations legible in the first place.

    :param corr: ``(n, n)`` correlation matrix, e.g. from :func:`connectivity_matrix`.
    :param ax: Axes to draw on; a new figure is created when omitted.
    :param title: Panel title, or None to omit.
    :param vmin: Lower colour limit. Pinned by default so panels stay comparable.
    :param vmax: Upper colour limit.
    :param colorbar: Whether to attach a colour bar.
    :returns: The ``AxesImage``.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.5, 5))

    image = ax.imshow(np.asarray(corr), cmap='RdBu_r', vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='equal')
    # A 1000x1000 matrix is a million vector quads in a PDF and looks identical as a raster.
    image.set_rasterized(True)

    ax.set_xlabel('Electrode', fontsize=10)
    ax.set_ylabel('Electrode', fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')

    return image


def plot_connectivity_grid(conn, cid=None, suptitle=None, save_path=None, show_plot=True,
                           ncols=None, figsize=None, dpi=150):
    """One correlation heatmap per DIV, on a shared colour scale.

    :param conn: ``{div: {'corr': matrix, ...}}``, as returned by :func:`culture_connectivity`.
    :param cid: Culture id, used to build a default ``suptitle``.
    :param suptitle: Figure title; defaults to one naming ``cid``.
    :param save_path: Full path (including filename) to save to, or None to skip saving.
    :param show_plot: Whether to call ``plt.show()``.
    :param ncols: Panels per row; defaults to at most 4.
    :param figsize: Figure size; derived from the panel count when omitted.
    :param dpi: Resolution for the saved figure and the rasterized heatmaps.
    :returns: The matplotlib :class:`~matplotlib.figure.Figure`, or None if ``conn`` is empty.
    """
    divs = sorted(conn)
    if not divs:
        return None

    ncols = ncols or min(4, len(divs))
    nrows = -(-len(divs) // ncols)
    figsize = figsize or (3.6 * ncols, 3.9 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    for ax, div in zip(axes, divs):
        plot_connectivity_matrix(conn[div]['corr'], ax=ax, title=f"DIV{div}", colorbar=False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        ax.set_ylabel(ax.get_ylabel() if ax is axes[0] else "", fontsize=8)

    for ax in axes[len(divs):]:
        ax.set_visible(False)

    # One colour bar for the figure: every panel already shares the pinned [-1, 1] scale.
    fig.colorbar(axes[0].images[0], ax=axes[:len(divs)].tolist(), fraction=0.025, pad=0.02,
                 label='Pearson r')

    fig.suptitle(suptitle or f'Functional Connectivity — {cid}', fontsize=13, fontweight='bold')

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot → {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_variance_explained(conn, cid=None, suptitle=None, save_path=None, show_plot=True,
                            thresholds=(0.8, 0.9), figsize=(7, 5), dpi=150, cmap_name='viridis'):
    """Cumulative variance explained vs component count, one line per DIV (Sono et al. Fig. 2D).

    A culture whose curve rises later over development is spreading its dynamics across more
    independent components -- the thing the effective rank in :func:`network_summary` summarizes to one
    number.

    :param conn: ``{div: {'cumvar': array, ...}}``, as returned by :func:`culture_connectivity`.
    :param cid: Culture id, used to build a default ``suptitle``.
    :param suptitle: Figure title; defaults to one naming ``cid``.
    :param save_path: Full path (including filename) to save to, or None to skip saving.
    :param show_plot: Whether to call ``plt.show()``.
    :param thresholds: Cumulative-variance levels to mark with a guide line.
    :param figsize: Figure size.
    :param dpi: Resolution for the saved figure.
    :param cmap_name: Sequential colormap mapped over the DIVs, so colour reads as developmental time.
    :returns: The matplotlib :class:`~matplotlib.figure.Figure`, or None if ``conn`` is empty.
    """
    divs = sorted(conn)
    if not divs:
        return None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = dict(zip(divs, get_n_colors(len(divs), cmap_name=cmap_name)))

    for div in divs:
        cumvar = np.asarray(conn[div].get('cumvar', []), dtype=float)
        if cumvar.size == 0:
            continue
        ax.plot(np.arange(1, cumvar.size + 1), cumvar, color=colors[div],
                linewidth=1.5, label=f"DIV{div}")

    for threshold in thresholds:
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.text(1, threshold, f" {threshold:.0%}", fontsize=7, color='gray',
                va='bottom', ha='left')

    ax.set_xscale('log')
    ax.set_xlabel('Number of principal components', fontsize=10)
    ax.set_ylabel('Cumulative variance explained', fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, framealpha=0.8, ncol=max(1, len(divs) // 6 + 1), loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(suptitle or f'Variance Explained — {cid}', fontsize=13, fontweight='bold')

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot → {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_population_network_summary(sel_paths, analysis_dir: Path, phase: str = None,
                                    savename: str = None):
    """Aggregate per-culture network CSVs into a population-level plot.

    Plots mean ± SEM across cultures vs DIV over a faint line per culture, one series per phase.

    :param sel_paths: ``{exp_id: ExperimentPaths}``, as returned by
        :func:`mxtreme.paths.resolve_paths` for a :class:`~mxtreme.identity.CultureSelector`.
    :param analysis_dir: Root summary directory (typically ``config.analysis_dir``); CSVs are read
        from ``<analysis_dir>/network/<exp_id>/<chip>/well<well>/``.
    :param phase: If given, filter to this phase only; ``None`` plots all phases as separate lines.
    :param savename: Filename for the saved figure, written into ``<analysis_dir>/network/``. If
        ``None`` the figure is shown but not saved.
    :returns: The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    pop_df = load_population_summaries(
        sel_paths, data_dir=analysis_dir / "network", suffix='network_summary'
    )

    if phase is not None:
        pop_df = pop_df[pop_df['phase'] == phase]

    n_cultures = pop_df['culture_id'].nunique()
    n_exps = len(sel_paths)
    exp_ids = list(sel_paths.keys())

    value_cols = ['mean_corr', 'effective_rank', 'n_pc_90', 'pc1_var_frac']
    stats = aggregate_by_div_phase(pop_df, value_cols=value_cols)

    metrics = [
        # (y_col,          err_col,                y_label,                    panel_title)
        ('mean_corr',      'sem_mean_corr',        'Mean pairwise r',          'Functional Connectivity'),
        ('effective_rank', 'sem_effective_rank',   'exp(entropy of spectrum)', 'Effective Rank'),
        ('n_pc_90',        'sem_n_pc_90',          'Components',               'PCs for 90% Variance'),
        ('pc1_var_frac',   'sem_pc1_var_frac',     'Variance explained',       'PC1 Dominance'),
    ]

    exp_label = f"{n_exps} experiments pooled ({', '.join(exp_ids)})" if n_exps > 1 else exp_ids[0]
    suptitle = f'Population Network Summary — {exp_label}\nmean ± SEM, n = {n_cultures} cultures'

    save_path = (Path(analysis_dir) / "network" / savename) if savename else None

    return plot_metric_grid(stats, metrics, overlay_df=pop_df, suptitle=suptitle,
                            save_path=save_path, show_plot=True, figsize=(11, 8), dpi=300)
