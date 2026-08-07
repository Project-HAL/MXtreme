import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from mxtreme.utils import load_data
from mxtreme import constants
from mxtreme.recording import Recording
from mxtreme.analysis._paths import ANALYSIS_DIR, _summary_paths, load_population_summaries
from mxtreme.analysis._plotting import plot_metric_grid
from mxtreme.analysis._stats import aggregate_by_div_phase


def burst_activity_summary(cpath, analysis_dir: Path = ANALYSIS_DIR, use_existing=True, show_plot=True, save_plot=False):

    cid = cpath.culture_id
    rows = []

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "activity", "burst_activity_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:

        for div in cpath.recordings:

            npz = cpath.recordings[div].npz
            rec = Recording(0,exp_data = load_data(npz))
            rec_t_sec = rec.rec_t_sec

            pre_sec = post_sec = 20*60 # 20 minutes fixed
            train_sec = rec_t_sec - pre_sec - post_sec
            phase_time_dict = {'pre':pre_sec,
                            'train':train_sec,
                            'post':post_sec}

            bin_size = rec.bin_size

            burst_stats = cpath.recordings[div].burst_stats
            burst_data = pd.read_csv(burst_stats)

            # Filter burst dataframe
            burst_data = burst_data[burst_data['type']=='HAL_like']

            t_burst_sec = burst_data['t_peak_bin']*bin_size #TODO: replace with t_peak_sec when it gets fixed in the burst CSV

            for phase in burst_data['phase'].dropna().unique():

                phase_bursts = burst_data[burst_data['phase']==phase]

                # Bursting rate
                burst_rate = len(phase_bursts)/phase_time_dict[phase]

                # IBI
                ibis = np.diff(phase_bursts['t_peak_bin'])*bin_size
                mean_ibi = np.mean(ibis) if len(ibis) > 0 else np.nan
                median_ibi = np.median(ibis) if len(ibis) > 0 else np.nan
                std_ibi = np.std(ibis) if len(ibis) > 0 else np.nan


                # burst size
                burst_sizes = phase_bursts['size_pct_elec']
                mean_size   = burst_sizes.mean()
                median_size = burst_sizes.median()
                std_size    = burst_sizes.std()

                # burst duration
                durations = phase_bursts['duration']
                mean_dur    = durations.mean()
                median_dur  = durations.median()
                std_dur     = durations.std()

                rows.append({
                    'culture_id':      cid,
                    'div':             div,
                    'phase':           phase,
                    'n_bursts':        len(phase_bursts),
                    'burst_rate_hz':   burst_rate,
                    'mean_ibi_sec':    mean_ibi,
                    'median_ibi_sec':  median_ibi,
                    'std_ibi_sec':     std_ibi,
                    'mean_size_pct':   mean_size,
                    'median_size_pct': median_size,
                    'std_size_pct':    std_size,
                    'mean_dur_sec':    mean_dur,
                    'median_dur_sec':  median_dur,
                    'std_dur_sec':     std_dur,
                })

        summary_df = pd.DataFrame(rows)

        os.makedirs(save_path, exist_ok=True)

        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_burst_activity_summary(summary_df, cid,
                            analysis_dir=save_path,
                            show_plot=show_plot,
                            save_plot=save_plot)

    return summary_df

def _plot_burst_activity_summary(df: pd.DataFrame, cid: str,
                        analysis_dir: Path,
                        show_plot: bool = True,
                        save_plot: bool = False):
    """
    Four-panel figure (IBI, burst size, duration, burst rate) vs DIV,
    with each phase as a separate labelled series.
    Error bars show ±1 SD where available.
    """

    metrics = [
        # (y_col,            err_col,           y_label,                       panel_title)
        ('median_ibi_sec',   'std_ibi_sec',   'Median IBI (s)',               'Inter-Burst Interval'),
        ('median_size_pct',  'std_size_pct',  'Median burst size (% elec.)',  'Burst Size'),
        ('median_dur_sec',   'std_dur_sec',   'Median duration (s)',          'Burst Duration'),
        ('burst_rate_hz',    None,            'Burst rate (bursts s⁻¹)',      'Burst Rate'),
    ]

    save_path = (analysis_dir / f"{cid}_burst_activity_summary.png") if save_plot else None

    plot_metric_grid(
        df, metrics,
        suptitle=f'Burst Activity Summary — {cid}',
        save_path=save_path,
        show_plot=show_plot,
        figsize=(10, 7),
        dpi=150,
    )

def channel_activity_summary(cpath, analysis_dir: Path = ANALYSIS_DIR, use_existing=True, show_plot=True, save_plot=False):

    cid = cpath.culture_id
    rows = []

    save_path, csv_path = _summary_paths(cpath, analysis_dir, "activity", "channel_activity_summary")

    if use_existing and csv_path.exists():
        print(f"Loading existing summary from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:

        for div in cpath.recordings:

            npz = cpath.recordings[div].npz
            rec = Recording(0,exp_data = load_data(npz))

            for phase in rec.epochs:

                start = rec.epochs[phase][0]
                stop = rec.epochs[phase][1]

                phase_spike_data = rec.spike_data[(rec.spike_data["frameno"]>start)&(rec.spike_data["frameno"]<stop)]

                # --- Per-channel metrics: {channel: value} dicts ---------- #
                channel_firing_rates = firing_rate_chan(phase_spike_data, rec.samp_rate)
                median_channel_isis  = ISI_chan(phase_spike_data, rec.samp_rate)
                mean_spike_amps      = mean_spike_amplitude_chan(phase_spike_data, rec.samp_rate)

                # --- Aggregate over channels -------------------------------- #
                fr_vals  = np.array([v for v in channel_firing_rates.values() if v is not None], dtype=float)
                isi_vals = np.array([v for v in median_channel_isis.values()  if v is not None], dtype=float)
                amp_vals = np.array([v*1e6 for v in mean_spike_amps.values()      if v is not None], dtype=float) # convery to uV for plotting and summary csv

                # Fraction of channels that were active (i.e. returned a
                # firing rate rather than NaN — channels with ≤1 spike)
                n_total   = len(channel_firing_rates)
                n_active  = int(np.sum(~np.isnan(fr_vals)))
                pct_active = 100 * n_active / n_total if n_total > 0 else np.nan

                def _safe(fn, arr):
                    return fn(arr[~np.isnan(arr)]) if np.any(~np.isnan(arr)) else np.nan

                rows.append({
                    'culture_id':      str(cid),
                    'div':             div,
                    'phase':           phase,
                    # Firing rate
                    'mean_fr_hz':      _safe(np.mean,   fr_vals),
                    'median_fr_hz':    _safe(np.median, fr_vals),
                    'std_fr_hz':       _safe(np.std,    fr_vals),
                    # ISI
                    'mean_isi_sec':    _safe(np.mean,   isi_vals),
                    'median_isi_sec':  _safe(np.median, isi_vals),
                    'std_isi_sec':     _safe(np.std,    isi_vals),
                    # Amplitude
                    'mean_amp_uv':     _safe(np.mean,   amp_vals),
                    'median_amp_uv':   _safe(np.median, amp_vals),
                    'std_amp_uv':      _safe(np.std,    amp_vals),
                    # Network-level
                    'n_active_chan':   n_active,
                    'n_total_chan':    n_total,
                    'pct_active_chan': pct_active,
                })

        summary_df = pd.DataFrame(rows)

        os.makedirs(save_path, exist_ok=True)

        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary to {save_path}")

    if show_plot or save_plot:
        _plot_channel_activity_summary(summary_df, cid,
                            analysis_dir=save_path,
                            show_plot=show_plot,
                            save_plot=save_plot)

    return summary_df


def firing_rate_chan(spike_data, samp_rate):

    spike_df = pd.DataFrame(spike_data)
    grouped_by_chan = spike_df.groupby('channel')

    chan_fr = {chan: (len(x)/((np.max(x)-np.min(x))/samp_rate) if len(x) > 1 else np.nan)
           for chan, x in grouped_by_chan['frameno']}

    return chan_fr

def ISI_chan(spike_data, samp_rate):

    spike_df = pd.DataFrame(spike_data)
    grouped_by_chan = spike_df.groupby('channel')

    # filtering out ISIs > 200 ms - replicating MaxLab ISI analysis
    median_chan_isi = {}
    for channel, group_df in grouped_by_chan:
        ISI_ms = np.diff(group_df['frameno'])/samp_rate*1000
        ISI_filtered = ISI_ms[ISI_ms<constants.ISI_threshold]
        if len(ISI_filtered) > 0:
            median_chan_isi[channel]=np.median(ISI_filtered)

    return median_chan_isi


def mean_spike_amplitude_chan(spike_data, samp_rate):

    spike_df = pd.DataFrame(spike_data)
    grouped_by_chan = spike_df.groupby('channel')

    mean_chan_spike_amp = {chan: (np.mean(x) if len(x) > 1 else np.nan)
                    for chan, x in grouped_by_chan['amplitude']} # in V

    return mean_chan_spike_amp

def _plot_channel_activity_summary(
    df: pd.DataFrame,
    cid: str,
    analysis_dir: Path,
    show_plot: bool = True,
    save_plot: bool = False,
):
    """
    Four-panel figure (firing rate, ISI, amplitude, active channels) vs DIV,
    with each phase as a separate labelled series.
    Error bars show ±1 SD (within-culture spread across channels).
    """
    metrics = [
        # (mean_col,        err_col,            y_label,               panel_title)
        ('mean_fr_hz',      'std_fr_hz',      'Firing rate (Hz)',     'Firing Rate'),
        ('mean_isi_sec',    'std_isi_sec',    'Median ISI (s)',       'Inter-Spike Interval'),
        ('mean_amp_uv',     'std_amp_uv',     'Spike amplitude (µV)', 'Spike Amplitude'),
        ('pct_active_chan', None,             'Active channels (%)', 'Active Channels'),
    ]

    save_path = (analysis_dir / f"{cid}_channel_activity_summary.png") if save_plot else None

    plot_metric_grid(
        df, metrics,
        suptitle=f'Channel Activity Summary — {cid}',
        save_path=save_path,
        show_plot=show_plot,
        figsize=(11, 8),
        dpi=150,
    )


def plot_population_burst_summary(
    sel_paths,
    analysis_dir: Path = ANALYSIS_DIR,
    phase: str = 'pre',
    savename: str = None,
):
    """
    Aggregate per-culture burst-activity CSVs and plot population-level
    measures (mean ± SEM across cultures) vs DIV, with one line per phase.

    Parameters
    ----------
    sel_paths   : dict  {exp_id: ExperimentPath}  — same structure used elsewhere
    analysis_dir : Path  — root summary directory; CSVs are expected under
                          <analysis_dir>/activity/<exp_id>/<chip>/well<well>/
    savename    : str | None — filename for the saved figure (saved into
                          <analysis_dir>/activity/).  If None the figure is
                          shown but not saved.

    Notes
    -----
    * If sel_paths contains more than one exp_id the cultures are pooled and
      a note is added to the figure title.  If you later want per-experiment
      panels, split sel_paths before calling this function.
    """

    pop_df = load_population_summaries(
        sel_paths,
        data_dir=analysis_dir / "activity",
        suffix='burst_activity_summary',
    )

    if phase is not None:
        pop_df = pop_df[pop_df['phase']==phase]

    n_cultures = pop_df['culture_id'].nunique()
    n_exps     = len(sel_paths)
    exp_ids    = list(sel_paths.keys())

    stats = aggregate_by_div_phase(
        pop_df, value_cols=['burst_rate_hz', 'median_ibi_sec', 'median_size_pct', 'median_dur_sec']
    )

    metrics = [
        # (y_col,            err_col,                  y_label,                       panel_title)
        ('median_ibi_sec',  'sem_median_ibi_sec',   'Median IBI (s)',               'Inter-Burst Interval'),
        ('median_size_pct', 'sem_median_size_pct',  'Median burst size (% elec.)',  'Burst Size'),
        ('median_dur_sec',  'sem_median_dur_sec',   'Median duration (s)',          'Burst Duration'),
        ('burst_rate_hz',   'sem_burst_rate_hz',    'Burst rate (bursts s⁻¹)',      'Burst Rate'),
    ]

    exp_label = f"{n_exps} experiments pooled ({', '.join(exp_ids)})" if n_exps > 1 else exp_ids[0]
    suptitle = f'Population Burst Activity — {exp_label}\nmean ± SEM, n = {n_cultures} cultures'

    save_path = (Path(analysis_dir) / "activity" / savename) if savename else None

    plot_metric_grid(stats, metrics, suptitle=suptitle, save_path=save_path,
                      show_plot=True, figsize=(11, 8), dpi=300)


def plot_population_channel_activity(sel_paths,
                                    analysis_dir: Path = ANALYSIS_DIR,
                                    phase: str = 'pre',
                                    savename: str = None,
                                    ):
    """
    Aggregate per-culture electrode-activity CSVs and plot population-level
    measures (mean ± SEM across cultures) vs DIV, with one line per phase.

    Parameters
    ----------
    sel_paths   : dict  {exp_id: ExperimentPath}
    analysis_dir : Path  — root summary directory; CSVs are expected under
                          <analysis_dir>/activity/<exp_id>/<chip>/well<well>/
    phase       : str | None — if provided, filter to this phase only;
                          pass None to plot all phases as separate lines.
    savename    : str | None — filename for the saved figure (saved into
                          <analysis_dir>/activity/).  If None the figure is
                          shown but not saved.

    Notes
    -----
    * If sel_paths contains more than one exp_id the cultures are pooled and
      a note is added to the figure title.  If you later want per-experiment
      panels, split sel_paths before calling this function.
    """

    pop_df = load_population_summaries(
        sel_paths,
        data_dir=analysis_dir / "activity",
        suffix='channel_activity_summary',
    )

    if phase is not None:
        pop_df = pop_df[pop_df['phase'] == phase]

    n_cultures = pop_df['culture_id'].nunique()
    n_exps     = len(sel_paths)
    exp_ids    = list(sel_paths.keys())

    stats = aggregate_by_div_phase(
        pop_df, value_cols=['mean_fr_hz', 'mean_isi_sec', 'mean_amp_uv', 'pct_active_chan']
    )

    metrics = [
        # (y_col,            err_col,                 y_label,               panel_title)
        ('mean_fr_hz',      'sem_mean_fr_hz',      'Firing rate (Hz)',     'Firing Rate'),
        ('mean_isi_sec',    'sem_mean_isi_sec',    'Median ISI (s)',       'Inter-Spike Interval'),
        ('mean_amp_uv',     'sem_mean_amp_uv',     'Spike amplitude (µV)', 'Spike Amplitude'),
        ('pct_active_chan', 'sem_pct_active_chan', 'Active channels (%)',  'Active Channels'),
    ]

    exp_label = f"{n_exps} experiments pooled ({', '.join(exp_ids)})" if n_exps > 1 else exp_ids[0]
    suptitle = f'Population Channel Activity — {exp_label}\nmean ± SEM, n = {n_cultures} cultures'

    save_path = (Path(analysis_dir) / "activity" / savename) if savename else None

    plot_metric_grid(stats, metrics, suptitle=suptitle, save_path=save_path,
                      show_plot=True, figsize=(11, 8), dpi=300)
