"""Assemble a multi-section MXtreme PDF report for a culture or group of cultures.

:func:`generate_report` resolves a selection (single :class:`~mxtreme.identity.CultureID` or a
:class:`~mxtreme.identity.CultureSelector` group) against a :class:`~mxtreme.config.Config`, runs the
requested analysis sections, and writes them into one PDF via ``matplotlib``'s :class:`PdfPages`
(no extra dependencies). Each section reuses the per-topic analysis functions and the shared
``ax``-aware visualizations, so the report never re-implements a metric.

Sections:
- ``"overview"``   -- single culture: ASDR + MEA layout per DIV; group: a concise summary table.
- ``"activity"``   -- firing rate / ISI / spike amplitude / active channels (always available).
- ``"bursting"``   -- detection diagnostics + burst stats (IBI, rate, size, duration). Optional.
- ``"stimulation"``-- total stim/train time per DIV. Optional; auto-skips non-stim cultures.
- ``"performance"``-- learning curve from a pluggable objective. Optional.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from mxtreme import io
from mxtreme.identity import CultureID, CultureSelector, RecordingID
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.recording import Recording
from mxtreme import visualizations as viz

from mxtreme.analysis import activity, spatial, stimulation, performance
from mxtreme.analysis._plotting import plot_metric_grid
from mxtreme.analysis._stats import aggregate_by_div_phase

DEFAULT_SECTIONS = ("overview", "activity", "bursting")

# Per-culture metric-grid specs (mirror the topic modules' _plot_* helpers).
_CHANNEL_METRICS = [
    ('mean_fr_hz',      'std_fr_hz',   'Firing rate (Hz)',     'Firing Rate'),
    ('mean_isi_sec',    'std_isi_sec', 'Median ISI (s)',       'Inter-Spike Interval'),
    ('mean_amp_uv',     'std_amp_uv',  'Spike amplitude (µV)', 'Spike Amplitude'),
    ('pct_active_chan', None,          'Active channels (%)',  'Active Channels'),
]
_BURST_METRICS = [
    ('median_ibi_sec',  'std_ibi_sec',  'Median IBI (s)',              'Inter-Burst Interval'),
    ('median_size_pct', 'std_size_pct', 'Median burst size (% elec.)', 'Burst Size'),
    ('median_dur_sec',  'std_dur_sec',  'Median duration (s)',         'Burst Duration'),
    ('burst_rate_hz',   None,           'Burst rate (bursts s⁻¹)',     'Burst Rate'),
]


# --- selection helpers --------------------------------------------------------------------------


def _flatten_cultures(resolved) -> list[CulturePaths]:
    """Return a flat list of ``CulturePaths`` from a ``resolve_paths`` result."""
    if isinstance(resolved, CulturePaths):
        return [resolved]
    # dict[exp_id, ExperimentPaths]
    cultures: list[CulturePaths] = []
    for epath in resolved.values():
        cultures.extend(epath.cultures.values())
    return cultures


def _device_type(chip: str) -> str:
    """Heuristic device label from the chip id prefix (MaxOne chips start 'P', MaxTwo start 'M')."""
    chip = str(chip)
    if chip.startswith("P"):
        return "MaxOne"
    if chip.startswith("M"):
        return "MaxTwo"
    return "unknown"


def _selection_slug(cultures: list[CulturePaths]) -> str:
    if len(cultures) == 1:
        return str(cultures[0].culture_id)
    return f"group_{len(cultures)}cultures"


# --- figure builders ----------------------------------------------------------------------------


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Append a simple text page (title + body lines)."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.82, title, ha="center", va="center", fontsize=22, fontweight="bold")
    fig.text(0.1, 0.70, "\n".join(lines), ha="left", va="top", fontsize=12, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _grid_page(pdf: PdfPages, df: pd.DataFrame, metrics, suptitle: str) -> None:
    """Append a metric-grid figure (metric vs DIV, one series per phase)."""
    if df is None or df.empty:
        return
    fig = plot_metric_grid(df, metrics, suptitle=suptitle, show_plot=False, save_path=None)
    pdf.savefig(fig)
    plt.close(fig)


# --- sections -----------------------------------------------------------------------------------


def _section_overview(pdf, cultures, single, analysis_dir):
    if single:
        cpath = cultures[0]
        cid = cpath.culture_id
        for div in sorted(cpath.recordings):
            rec = Recording(0, io.load_preprocessed(cpath.recordings[div].npz))
            fig, (ax_asdr, ax_mea) = plt.subplots(
                2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1, 2]}
            )
            viz.plot_asdr(rec.spike_bin, ax=ax_asdr, title=f"ASDR — {cid} DIV{div}")
            viz.MEA(ax_mea, rec.channelmap, rec.stim_elecs, title=f"MEA layout — DIV{div}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    else:
        # Concise group overview table.
        rows = []
        for cpath in cultures:
            cid = cpath.culture_id
            divs = sorted(cpath.recordings)
            rows.append([
                cid.exp_id, cid.chip, f"well{cid.well}", _device_type(cid.chip),
                len(divs), ", ".join(str(d) for d in divs),
            ])
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(f"Group overview — {len(cultures)} cultures", fontsize=16, fontweight="bold")
        table = ax.table(
            cellText=rows,
            colLabels=["Experiment", "Chip", "Well", "Device", "# DIVs", "DIVs"],
            loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        pdf.savefig(fig)
        plt.close(fig)


def _section_activity(pdf, cultures, single, sel_paths, analysis_dir):
    # Compute (and cache) each culture's channel-activity summary.
    for cpath in cultures:
        activity.channel_activity_summary(cpath, analysis_dir, show_plot=False, save_plot=False)

    if single:
        df = activity.channel_activity_summary(cultures[0], analysis_dir, show_plot=False)
        _grid_page(pdf, df, _CHANNEL_METRICS, f"Channel Activity — {cultures[0].culture_id}")
    else:
        pop_df = activity.load_population_summaries(
            sel_paths, data_dir=analysis_dir / "activity", suffix="channel_activity_summary"
        )
        stats = aggregate_by_div_phase(
            pop_df, value_cols=["mean_fr_hz", "mean_isi_sec", "mean_amp_uv", "pct_active_chan"]
        )
        metrics = [(y, f"sem_{y}" if y != "pct_active_chan" else "sem_pct_active_chan", lbl, t)
                   for (y, _e, lbl, t) in _CHANNEL_METRICS]
        _grid_page(pdf, stats, metrics, f"Population Channel Activity — {len(cultures)} cultures")


def _section_bursting(pdf, cultures, single, sel_paths, analysis_dir):
    # Detection diagnostics: one page per culture (representative DIV).
    for cpath in cultures:
        divs = sorted(cpath.recordings)
        if not divs:
            continue
        div = divs[0]
        rec = Recording(0, io.load_preprocessed(cpath.recordings[div].npz))
        burst_df = pd.read_csv(cpath.recordings[div].burst_stats)
        fig, (ax_asdr, ax_origin) = plt.subplots(1, 2, figsize=(14, 5))
        viz.plot_bursts_on_asdr(rec, burst_df, ax=ax_asdr,
                                title=f"Bursts on ASDR — {cpath.culture_id} DIV{div}")
        viz.plot_origin_heatmap(rec, burst_df, ax=ax_origin, title="Network-burst origins")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Burst stats.
    for cpath in cultures:
        activity.burst_activity_summary(cpath, analysis_dir, show_plot=False, save_plot=False)

    if single:
        df = activity.burst_activity_summary(cultures[0], analysis_dir, show_plot=False)
        _grid_page(pdf, df, _BURST_METRICS, f"Burst Activity — {cultures[0].culture_id}")
    else:
        pop_df = activity.load_population_summaries(
            sel_paths, data_dir=analysis_dir / "activity", suffix="burst_activity_summary"
        )
        stats = aggregate_by_div_phase(
            pop_df, value_cols=["burst_rate_hz", "median_ibi_sec", "median_size_pct", "median_dur_sec"]
        )
        metrics = [
            ('median_ibi_sec',  'sem_median_ibi_sec',  'Median IBI (s)',              'Inter-Burst Interval'),
            ('median_size_pct', 'sem_median_size_pct', 'Median burst size (% elec.)', 'Burst Size'),
            ('median_dur_sec',  'sem_median_dur_sec',  'Median duration (s)',         'Burst Duration'),
            ('burst_rate_hz',   'sem_burst_rate_hz',   'Burst rate (bursts s⁻¹)',     'Burst Rate'),
        ]
        _grid_page(pdf, stats, metrics, f"Population Burst Activity — {len(cultures)} cultures")


def _section_stimulation(pdf, cultures, analysis_dir):
    for cpath in cultures:
        try:
            df = stimulation.stim_summary(cpath, analysis_dir, show_plot=False, save_plot=False)
        except Exception as exc:  # noqa: BLE001 -- stimulation parsing is experiment-specific
            print(f"Skipping stimulation for {cpath.culture_id}: {exc}")
            continue
        if df.empty or df["total_stim_ms"].fillna(0).sum() == 0:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = range(len(df))
        ax1.bar(x, df["total_stim_ms"], color="steelblue", edgecolor="black")
        ax1.set_xticks(list(x)); ax1.set_xticklabels(df["div"])
        ax1.set_xlabel("DIV"); ax1.set_ylabel("Total stim time (ms)")
        ax2.bar(x, df["total_train_min"], color="darkorange", edgecolor="black")
        ax2.set_xticks(list(x)); ax2.set_xticklabels(df["div"])
        ax2.set_xlabel("DIV"); ax2.set_ylabel("Total train time (min)")
        fig.suptitle(f"Stimulation — {cpath.culture_id}", fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _section_performance(pdf, cultures, analysis_dir, objective_fn):
    kwargs = {} if objective_fn is None else {"objective_fn": objective_fn}    

    for cpath in cultures:
        df = performance.performance_summary(
            cpath, analysis_dir, show_plot=False, save_plot=False, **kwargs
        )
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for phase in df["phase"].unique():
            pdf_phase = df[df["phase"] == phase].sort_values("div")
            ax.plot(pdf_phase["div"], pdf_phase["score"], marker="o", label=phase)
        ax.set_xlabel("DIV"); ax.set_ylabel("Score"); ax.set_ylim(-0.01, 1.01)
        ax.set_title(f"Performance — {cpath.culture_id}"); ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# --- entry point --------------------------------------------------------------------------------


def generate_report(
    target: CultureID | CultureSelector | RecordingID,
    config,
    *,
    sections=DEFAULT_SECTIONS,
    output_path: str | Path | None = None,
    objective_fn=None,
    title: str | None = None,
) -> Path:
    """Generate a multi-section PDF report for ``target`` and return the written path.

    :param target: A :class:`~mxtreme.identity.CultureID` (single culture) or
        :class:`~mxtreme.identity.CultureSelector` (a group, possibly across experiments).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param sections: Which sections to include (see module docstring). Defaults to overview + activity
        + bursting; add ``"stimulation"`` / ``"performance"`` as needed.
    :param output_path: Destination PDF. Defaults to
        ``config.analysis_dir/reports/<slug>_report.pdf``.
    :param objective_fn: Optional objective for the performance section (``callable(burst_df)->float``).
    :param title: Optional report title (defaults to a description of the selection).

    Per-phase activity/burst windows are reconstructed automatically from the phase spec embedded in
    each recording's preprocessed data (written at preprocessing time from the ``Phases`` metadata
    key). A recording with no embedded spec has a single ``"full"`` phase.
    :returns: The path to the written PDF.
    """
    resolved = resolve_paths(target, config)
    if isinstance(resolved, CulturePaths):
        cultures = [resolved]
        sel_paths = resolve_paths(CultureSelector(cultures=[resolved.culture_id]), config)
    else:
        cultures = _flatten_cultures(resolved)
        sel_paths = resolved

    if not cultures:
        raise ValueError("Selection resolved to zero cultures; check the registry and selector.")

    single = len(cultures) == 1
    analysis_dir = config.analysis_dir

    if output_path is None:
        output_path = analysis_dir / "reports" / f"{_selection_slug(cultures)}_report.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = title or (f"MXtreme Report — {cultures[0].culture_id}" if single
                      else f"MXtreme Report — {len(cultures)} cultures")

    with PdfPages(output_path) as pdf:
        # Title page.
        info = [
            f"Generated: {_dt.date.today().isoformat()}",
            f"Cultures:  {len(cultures)}",
            "",
            "Selection:",
        ] + [f"  - {c.culture_id}  (DIVs {', '.join(str(d) for d in sorted(c.recordings))})"
             for c in cultures] + [
            "",
            f"Sections:  {', '.join(sections)}",
        ]
        _text_page(pdf, title, info)

        if "overview" in sections:
            _section_overview(pdf, cultures, single, analysis_dir)
        if "activity" in sections:
            _section_activity(pdf, cultures, single, sel_paths, analysis_dir)
        if "bursting" in sections:
            _section_bursting(pdf, cultures, single, sel_paths, analysis_dir)
        if "stimulation" in sections:
            _section_stimulation(pdf, cultures, analysis_dir)
        if "performance" in sections:
            _section_performance(pdf, cultures, analysis_dir, objective_fn)

    print(f"Report written to: {output_path}")
    return output_path
