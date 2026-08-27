import os
import pandas as pd
from pathlib import Path

from mxtreme import io
from mxtreme.recording import Recording
from mxtreme.analysis._paths import _summary_paths, load_population_summaries
from mxtreme.analysis._plotting import plot_metric_grid
from mxtreme.analysis._stats import aggregate_by_div_phase


def _get_stim_info(rec: Recording):
    """Return the recording's stimulation events, with the pulse phase in a ``stim_phase`` column.

    Event messages are dicts (see :attr:`Recording.event_df <mxtreme.recording.Recording.event_df>`);
    a stimulation event is one carrying both a ``start_stimulation`` and a ``phase_us`` key. Messages
    that aren't dicts (older stores hold plain strings) are skipped rather than raising.

    .. note:: The ``start_stimulation`` / ``phase_us`` key names are maxlab closed-loop-stim specific.
    """
    is_stim = rec.event_df['eventmessage'].map(
        lambda d: isinstance(d, dict) and ('phase_us' in d) and ('start_stimulation' in d)
    )
    stim_rows = rec.event_df[is_stim].copy()
    stim_rows['stim_phase'] = stim_rows['eventmessage'].map(
        lambda d: d.get('phase_us', float('nan'))
    ).astype(float)

    return stim_rows


def stim_summary(cpath, analysis_dir: Path, use_existing=True, show_plot=True, save_plot=False):
    """Summarise stimulation delivered to one culture, one row per DIV and phase.

    For each (DIV, phase): ``total_stim_ms`` is the summed pulse phase of the stimulation events that
    fall inside that phase's window (maxlab reports these in **microseconds** as ``phase_us``, so they
    are divided by 1000 to give milliseconds), and ``phase_dur_min`` is the span of the window itself.

    Phases come from the recording (see :mod:`mxtreme.phases`), exactly as in the activity summaries,
    so a recording with no user-supplied phases yields a single ``"full"`` row covering it.

    :param cpath: The culture's :class:`~mxtreme.paths.CulturePaths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param use_existing: Reuse the cached summary CSV when one already exists.
    :param show_plot: Show the per-culture summary figure.
    :param save_plot: Save that figure next to the summary CSV.
    :returns: One row per DIV/phase with columns ``div``, ``phase``, ``total_stim_ms``,
        ``phase_dur_min``, ``culture_id``.
    :rtype: pandas.DataFrame
    """
    cid = cpath.culture_id
    columns = ['div', 'phase', 'total_stim_ms', 'phase_dur_min', 'culture_id']

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "stimulation", "stim_summary")

    # A cache without `phase` predates per-phase attribution: its stim totals cover whole recordings
    # and can't be filtered to a phase. Recompute rather than load it back.
    cached = pd.read_csv(csv_path) if use_existing and csv_path.exists() else None
    if cached is not None and 'phase' not in cached.columns:
        print(f"Ignoring pre-phase summary at {csv_path} (no phase column); recomputing.")
        cached = None

    if cached is not None:
        print(f"Loading existing summary from {csv_path}")
        summary_df = cached
    else:
        rows = []
        for div in cpath.recordings:

            npz = cpath.recordings[div].npz

            rec = Recording(0, io.load_preprocessed(npz))

            stim_df = _get_stim_info(rec)

            for phase in rec.phases:
                in_phase = stim_df['eventtime'].between(
                    phase.start_frame, phase.end_frame, inclusive='left'
                )
                rows.append({
                    'div':            div,
                    'phase':          phase.name,
                    # phase_us (µs) -> ms
                    'total_stim_ms':  stim_df.loc[in_phase, 'stim_phase'].sum() / 1000,
                    'phase_dur_min':  (phase.end_frame - phase.start_frame) / rec.samp_rate / 60,
                    'culture_id':     str(cid),
                })

        summary_df = pd.DataFrame(rows, columns=columns)

        os.makedirs(save_path, exist_ok=True)

        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_stim_summary(summary_df, cid=cpath.culture_id, analysis_dir=save_path, show_plot=show_plot, save_plot=save_plot)

    return summary_df


STIM_METRICS = [
    # (y_col,           err_col, y_label,                       panel_title)
    ('total_stim_ms',   None,    'Total stim time (ms)',        'Stimulation Delivered'),
    ('phase_dur_min',   None,    'Window duration (min)',       'Recorded Window'),
]

# Population variants of STIM_METRICS -- same panels, but with the across-culture SEM.
POP_STIM_METRICS = [(y, f'sem_{y}', label, title) for (y, _e, label, title) in STIM_METRICS]


def _plot_stim_summary(df: pd.DataFrame, cid, analysis_dir, show_plot, save_plot, title_suffix: str = ''):
    """Two-panel figure (stim delivered, window duration) vs DIV, one series per phase."""

    save_path = (Path(analysis_dir) / f"{cid}_stim_summary.png") if save_plot else None
    suffix = f' — {title_suffix}' if title_suffix else ''

    plot_metric_grid(
        df, STIM_METRICS,
        suptitle=f'Stimulation Summary — {cid}{suffix}',
        save_path=save_path,
        show_plot=show_plot,
        figsize=(11, 4.5),
        dpi=150,
    )


def plot_population_stim_summary(sel_paths, analysis_dir: Path, phase: str = None, savename=None):
    """Stimulation vs DIV pooled across cultures: mean ± SEM, with a faint line per culture.

    :param sel_paths: ``{exp_id: ExperimentPaths}`` from :func:`mxtreme.paths.resolve_paths`.
    :param analysis_dir: Analysis output root (typically ``config.analysis_dir``).
    :param phase: If given, restrict to this phase only; ``None`` plots all phases as separate lines.
    :param savename: Filename for the saved figure, written into ``<analysis_dir>/stimulation/``.
    """

    # TODO: update if sel_paths has more than one exp_id, decide whether to combine them or plot them separately

    pop_df = load_population_summaries(sel_paths, data_dir=analysis_dir/"stimulation", suffix='stim_summary')

    if phase is not None:
        pop_df = pop_df[pop_df['phase'] == phase]

    stats = aggregate_by_div_phase(pop_df, value_cols=['total_stim_ms', 'phase_dur_min'])

    n_cultures = pop_df['culture_id'].nunique()
    save_path = (Path(analysis_dir) / "stimulation" / savename) if savename else None

    plot_metric_grid(
        stats, POP_STIM_METRICS,
        overlay_df=pop_df,
        suptitle=f'Population Stimulation\nmean ± SEM, n = {n_cultures} cultures',
        save_path=save_path,
        show_plot=True,
        figsize=(11, 4.5),
        dpi=300,
    )
