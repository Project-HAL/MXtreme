"""Assemble a multi-section MXtreme PDF report for a culture or group of cultures.

:func:`generate_report` resolves a selection (single :class:`~mxtreme.identity.CultureID` or a
:class:`~mxtreme.identity.CultureSelector` group) against a :class:`~mxtreme.config.Config`, runs the
requested analysis sections, and writes them into one PDF via ``matplotlib``'s :class:`PdfPages`
(no extra dependencies). Each section reuses the per-topic analysis functions and the shared
``ax``-aware visualizations, so the report never re-implements a metric.

The report answers two questions in order -- *how is the group developing?* and *what is each culture
doing?*:

1. a title page naming the selection and the phase being plotted;
2. one **group** page per topic (spiking, bursting, and optionally stimulation / performance), each a
   grid of metric-vs-DIV panels showing the across-culture mean +/- SEM over a faint line per culture,
   so an outlier never hides inside the average;
3. two **per-culture** pages: the recording itself over DIV (ASDR and burst origins), then the
   distributions behind the summary statistics (CDFs, one line per DIV).

Sections:
- ``"activity"``    -- spiking activity: firing rate / ISI / spike amplitude / active electrodes.
- ``"bursting"``    -- burst statistics: IBI, size, duration, rate.
- ``"cultures"``    -- the two per-culture pages, for every culture in the selection.
- ``"stimulation"``-- stimulation delivered per DIV. Optional; auto-skips non-stim cultures.
- ``"performance"``-- learning curve from a pluggable objective. Optional.
- ``"overview"``   -- currently disabled (see :func:`_section_overview`).

Phases: exactly one phase is plotted per report. ``phase=None`` selects the first phase present in
the data -- ``"full"`` for recordings with no user-supplied phases, otherwise the first of the
sequence -- and any phase name can be requested explicitly.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from mxtreme import device, io
from mxtreme.identity import CultureID, CultureSelector, RecordingID
from mxtreme.paths import CulturePaths, resolve_paths
from mxtreme.recording import Recording
from mxtreme import visualizations as viz

from mxtreme.analysis import activity, stimulation, performance
from mxtreme.analysis._paths import stamp_identity
from mxtreme.analysis._plotting import plot_cdf_grid, plot_metric_grid, sort_phases
from mxtreme.analysis._stats import aggregate_by_div_phase

DEFAULT_SECTIONS = ("activity", "bursting", "cultures")

# Population metric-grid specs: (mean_col, sem_col, y_label, panel_title). The SEM columns are the
# ones `aggregate_by_div_phase` produces, so these drive both the aggregate series and (via the
# mean_col) the faint per-culture overlay drawn from the un-aggregated frame.
_SPIKING_METRICS = [
    ('mean_fr_hz',      'sem_mean_fr_hz',      'Firing rate (Hz)',     'Firing Rate'),
    ('mean_isi_msec',   'sem_mean_isi_msec',   'Median ISI (ms)',      'Inter-Spike Interval'),
    ('mean_amp_uv',     'sem_mean_amp_uv',     'Spike amplitude (µV)', 'Spike Amplitude'),
    ('pct_active_chan', 'sem_pct_active_chan', 'Active electrodes (%)', 'Active Electrodes'),
]
_BURST_METRICS = [
    ('median_ibi_sec',  'sem_median_ibi_sec',  'Median IBI (s)',              'Inter-Burst Interval'),
    ('median_size_pct', 'sem_median_size_pct', 'Median burst size (% elec.)', 'Burst Size'),
    ('median_dur_sec',  'sem_median_dur_sec',  'Median duration (s)',         'Burst Duration'),
    ('burst_rate_hz',   'sem_burst_rate_hz',   'Burst rate (bursts s⁻¹)',     'Burst Rate'),
]
# Width-to-height ratio of a per-culture ASDR panel. Above ~1 the trace reads as a rectangle rather
# than a tall ribbon; raise it for squatter panels on a shorter page.
ASDR_PANEL_ASPECT = 1.7

# (distribution key, x label, panel title, log x) -- see `activity.culture_distributions`.
_CDF_METRICS = [
    ('fr_hz',    'Firing rate (Hz)',           'Firing Rate',          True),
    ('isi_ms',   'ISI (ms)',                   'Inter-Spike Interval', True),
    ('ibi_sec',  'IBI (s)',                    'Inter-Burst Interval', True),
    ('size_pct', 'Burst size (% electrodes)',  'Burst Size',           False),
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


def _resolve_phase(pop_df: pd.DataFrame, phase: str | None) -> str:
    """Return the phase to plot, validated against what the data actually holds.

    :func:`~mxtreme.analysis._plotting.sort_phases` orders ``full`` ahead of ``pre``/``train``/
    ``post``, so with no request an unphased selection resolves to ``"full"`` and a phased one to the
    first phase of its sequence. A requested phase the selection doesn't have fails here -- once, up
    front -- rather than producing a report of empty panels.
    """
    available = sort_phases(pop_df["phase"].dropna().unique())
    if not available:
        raise ValueError("No phases found in the activity summaries; cannot pick one to plot.")

    if phase is None:
        return available[0]
    if phase not in available:
        raise ValueError(f"No data for phase {phase!r} (available: {', '.join(available)}).")
    return phase


def _pool(pairs) -> pd.DataFrame:
    """Concatenate ``(CulturePaths, summary_df)`` pairs into one frame stamped with each culture.

    The summaries are already in hand by the time a section plots them, so pooling them here avoids
    re-reading the CSVs that were just written -- and avoids failing on a culture whose summary was
    legitimately skipped.
    """
    frames = [stamp_identity(df, cpath.culture_id)
              for cpath, df in pairs if df is not None and not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _for_phase(pop_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Restrict a pooled summary to one phase, warning when that leaves nothing to plot."""
    if pop_df.empty:
        return pop_df
    sub = pop_df[pop_df["phase"] == phase]
    if sub.empty:
        available = ", ".join(sorted(pop_df["phase"].dropna().unique())) or "none"
        print(f"No rows for phase {phase!r} (available: {available}); skipping the section.")
    return sub


# --- figure builders ----------------------------------------------------------------------------


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Append a simple text page (title + body lines)."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.82, title, ha="center", va="center", fontsize=22, fontweight="bold")
    fig.text(0.1, 0.70, "\n".join(lines), ha="left", va="top", fontsize=12, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _population_page(pdf, pop_df, metrics, value_cols, suptitle) -> None:
    """Append one group page: across-culture mean ± SEM over a faint line per culture."""
    if pop_df is None or pop_df.empty:
        return
    stats = aggregate_by_div_phase(pop_df, value_cols=value_cols)
    fig = plot_metric_grid(stats, metrics, overlay_df=pop_df, suptitle=suptitle,
                           show_plot=False, save_path=None)
    pdf.savefig(fig)
    plt.close(fig)


# --- sections -----------------------------------------------------------------------------------


def _section_overview(pdf, cultures, single, analysis_dir):
    """Disabled: the group overview table put too much detail on the report's first page.

    The title page now carries the selection, and the per-DIV ASDR / MEA views a single culture used
    to get here live in the richer per-culture section (:func:`_section_cultures`). Kept -- commented
    out -- because the table itself is worth restoring behind a flag later.
    """
    return

    # if single:
    #     cpath = cultures[0]
    #     cid = cpath.culture_id
    #     for div in sorted(cpath.recordings):
    #         rec = Recording(0, io.load_preprocessed(cpath.recordings[div].npz))
    #         fig, (ax_asdr, ax_mea) = plt.subplots(
    #             2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1, 2]}
    #         )
    #         viz.plot_asdr(rec.spike_bin, ax=ax_asdr, title=f"ASDR — {cid} DIV{div}")
    #         viz.MEA(ax_mea, rec.channelmap, rec.stim_elecs, title=f"MEA layout — DIV{div}")
    #         fig.tight_layout()
    #         pdf.savefig(fig)
    #         plt.close(fig)
    # else:
    #     # Concise group overview table.
    #     rows = []
    #     for cpath in cultures:
    #         cid = cpath.culture_id
    #         divs = sorted(cpath.recordings)
    #         rows.append([
    #             cid.exp_id, cid.chip, f"well{cid.well}", _device_type(cid.chip),
    #             len(divs), ", ".join(str(d) for d in divs),
    #         ])
    #     fig, ax = plt.subplots(figsize=(11, 8.5))
    #     ax.axis("off")
    #     ax.set_title(f"Group overview — {len(cultures)} cultures", fontsize=16, fontweight="bold")
    #     table = ax.table(
    #         cellText=rows,
    #         colLabels=["Experiment", "Chip", "Well", "Device", "# DIVs", "DIVs"],
    #         loc="center", cellLoc="left",
    #     )
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(9)
    #     table.scale(1, 1.5)
    #     pdf.savefig(fig)
    #     plt.close(fig)


def _spiking_summaries(cultures, analysis_dir) -> pd.DataFrame:
    """Pooled per-culture spiking summaries -- also the frame the report's phase is resolved from."""
    return _pool(
        (cpath, activity.channel_activity_summary(cpath, analysis_dir, show_plot=False, save_plot=False))
        for cpath in cultures
    )


def _section_activity(pdf, cultures, pop_df, phase):
    _population_page(
        pdf, _for_phase(pop_df, phase), _SPIKING_METRICS,
        value_cols=["mean_fr_hz", "mean_isi_msec", "mean_amp_uv", "pct_active_chan"],
        suptitle=f"Spiking Activity — {len(cultures)} cultures  |  phase: {phase}",
    )


def _section_bursting(pdf, cultures, analysis_dir, phase):
    pop_df = _pool(
        (cpath, activity.burst_activity_summary(cpath, analysis_dir, show_plot=False, save_plot=False))
        for cpath in cultures
    )
    _population_page(
        pdf, _for_phase(pop_df, phase), _BURST_METRICS,
        value_cols=["burst_rate_hz", "median_ibi_sec", "median_size_pct", "median_dur_sec"],
        suptitle=f"Burst Activity — {len(cultures)} cultures  |  phase: {phase}",
    )


def _section_stimulation(pdf, cultures, analysis_dir, phase):
    pairs = []
    for cpath in cultures:
        try:
            df = stimulation.stim_summary(cpath, analysis_dir, show_plot=False, save_plot=False)
        except Exception as exc:  # noqa: BLE001 -- stimulation parsing is experiment-specific
            print(f"Skipping stimulation for {cpath.culture_id}: {exc}")
            continue
        # A culture that was never stimulated would otherwise drag the population mean toward zero.
        if df.empty or df["total_stim_ms"].fillna(0).sum() == 0:
            continue
        pairs.append((cpath, df))

    if not pairs:
        print("No stimulated cultures in the selection; skipping the stimulation section.")
        return

    _population_page(
        pdf, _for_phase(_pool(pairs), phase), stimulation.POP_STIM_METRICS,
        value_cols=["total_stim_ms", "phase_dur_min"],
        suptitle=f"Stimulation — {len(pairs)} cultures  |  phase: {phase}",
    )


def _section_performance(pdf, cultures, analysis_dir, phase, objective_fn):
    kwargs = {} if objective_fn is None else {"objective_fn": objective_fn}

    pop_df = _pool(
        (cpath, performance.performance_summary(
            cpath, analysis_dir, show_plot=False, save_plot=False, **kwargs))
        for cpath in cultures
    )
    _population_page(
        pdf, _for_phase(pop_df, phase), [('score', 'sem_score', 'Score', 'Performance')],
        value_cols=["score"],
        suptitle=f"Performance — {len(cultures)} cultures  |  phase: {phase}",
    )


# --- per-culture pages ---------------------------------------------------------------------------


def _origin_density_vmax(recordings, phase):
    """A single density ceiling across a culture's DIVs, so the origin maps are comparable.

    :func:`~mxtreme.visualizations.plot_origin_heatmap` scales each plot to its own peak unless
    ``vmax`` is pinned, which makes two panels' colors mean different things. Recomputing the density
    here is cheap (a weighted 2-D histogram per DIV) next to the panels themselves.
    """
    chip_wd_um = viz.device.CHIP_WIDTH * viz.device.ELEC_SIZE
    chip_ht_um = viz.device.CHIP_HEIGHT * viz.device.ELEC_SIZE

    peaks = []
    for rec, burst_df in recordings:
        bursts = burst_df[(burst_df["kind"] == "network") & (burst_df["phase"] == phase)]
        xs, ys, weights = viz._origin_channel_density(rec.channelmap, bursts, chip_ht_um)
        if not len(xs):
            continue
        density, _xe, _ye = viz._smoothed_density(
            xs, ys, weights / max(len(bursts), 1),
            bandwidth_um=viz._median_nn_distance(rec.channelmap),
            chip_wd_um=chip_wd_um, chip_ht_um=chip_ht_um,
        )
        peaks.append(density.max())

    return max(peaks) if peaks else None


def _culture_recording_page(pdf, cpath, phase):
    """Page A: the recording itself over DIV -- ASDR on top, burst-origin map below, one column per DIV."""
    cid = cpath.culture_id
    divs = sorted(cpath.recordings)
    if not divs:
        return

    # One load per DIV, reused by both rows and by the shared vmax pass.
    loaded = []
    for div in divs:
        rec = Recording(0, io.load_preprocessed(cpath.recordings[div].npz))
        loaded.append((rec, pd.read_csv(cpath.recordings[div].burst_stats)))

    vmax = _origin_density_vmax(loaded, phase)

    # The page is sized from its panels rather than the other way round: a fixed letter-landscape page
    # would squeeze eight DIVs into unreadable slivers across, and stretch each panel into a tall
    # ribbon down. Both rows get a height derived from the column width -- the ASDR traces a squat
    # rectangle, the origin maps the array's own 3850 x 2100 µm aspect -- so the page ends up wide and
    # short. PdfPages takes each page at whatever size its figure is.
    page_w = max(11, 2.4 * len(divs))
    col_w = page_w / len(divs)
    asdr_h = col_w / ASDR_PANEL_ASPECT
    origin_h = col_w * device.CHIP_HEIGHT / device.CHIP_WIDTH
    # + room for the suptitle, panel titles and axis labels, which don't scale with the panels.
    fig, axes = plt.subplots(2, len(divs), figsize=(page_w, asdr_h + origin_h + 1.9), squeeze=False,
                             gridspec_kw={"height_ratios": [asdr_h, origin_h]})

    for col, (div, (rec, burst_df)) in enumerate(zip(divs, loaded)):
        ax_asdr, ax_origin = axes[0][col], axes[1][col]

        # Both rows show the selected phase only, like every other page in the report -- and at this
        # panel size a full hour of ASDR next to a 20-minute phase would just be a solid block.
        window = next((p for p in rec.phases if p.name == phase), None)
        zoom = None
        if window is not None:
            bins_per_frame = 1 / (rec.samp_rate * rec.bin_size)
            zoom = (window.start_frame * bins_per_frame, window.end_frame * bins_per_frame)

        viz.plot_bursts_on_asdr(rec, burst_df, ax=ax_asdr, zoom=zoom, title=f"DIV{div}")
        # `plot_bursts_on_asdr` works in bins; minutes are what a reader wants on the axis.
        bins_per_min = 60 / rec.bin_size
        ax_asdr.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _pos, b=bins_per_min: f"{x / b:.0f}")
        )
        ax_asdr.set_xlabel("Time (min)")

        viz.plot_origin_heatmap(rec, burst_df, ax=ax_origin, phase=phase, method="hist",
                                vmax=vmax, colorbar=False, title="",
                                marker_size=25, elec_marker_size=2)
        # The array is 3850 x 2100 µm; without this the map stretches to whatever shape the panel is.
        ax_origin.set_aspect("equal")

        for ax in (ax_asdr, ax_origin):
            ax.set_title(ax.get_title(), fontsize=9)
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)
            ax.set_ylabel(ax.get_ylabel() if col == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            # An hour of ASDR is ~360k points and the array is ~1k electrodes, per panel per DIV.
            # As vectors that runs to tens of MB per report for detail no printer can resolve; as
            # rasters inside the still-vector page it is a fraction of that and looks the same.
            for artist in [*ax.get_lines(), *ax.collections]:
                artist.set_rasterized(True)

    # A shared ASDR y-scale is what lets a reader compare activity between DIVs by eye.
    top = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(0, top)

    fig.suptitle(f"{cid}  |  ASDR and burst origins over DIV  (phase: {phase})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)  # resolution of the rasterized panels above
    plt.close(fig)


def _culture_distribution_page(pdf, cpath, analysis_dir, phase):
    """Page B: the distributions behind the summary statistics, one CDF line per DIV."""
    cid = cpath.culture_id
    dists = activity.culture_distributions(cpath, analysis_dir, phase=phase, use_existing=True)
    if not dists:
        return

    fig = plot_cdf_grid(
        dists, _CDF_METRICS,
        suptitle=f"{cid}  |  distributions over DIV  (phase: {phase})",
        show_plot=False, save_path=None,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _section_cultures(pdf, cultures, analysis_dir, phase):
    for cpath in cultures:
        print(f"Building culture pages for {cpath.culture_id}")
        _culture_recording_page(pdf, cpath, phase)
        _culture_distribution_page(pdf, cpath, analysis_dir, phase)


# --- entry point --------------------------------------------------------------------------------


def generate_report(
    target: CultureID | CultureSelector | RecordingID,
    config,
    *,
    sections=DEFAULT_SECTIONS,
    phase: str | None = None,
    output_path: str | Path | None = None,
    objective_fn=None,
    title: str | None = None,
) -> Path:
    """Generate a multi-section PDF report for ``target`` and return the written path.

    :param target: A :class:`~mxtreme.identity.CultureID` (single culture) or
        :class:`~mxtreme.identity.CultureSelector` (a group, possibly across experiments).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param sections: Which sections to include (see module docstring). Defaults to activity +
        bursting + per-culture pages; add ``"stimulation"`` / ``"performance"`` as needed.
    :param phase: Which phase to plot. ``None`` (the default) uses the first phase present in the
        data: ``"full"`` for recordings with no user-supplied phases, otherwise the first of the
        sequence. Pass a name (e.g. ``"train"``) to report on that window instead.
    :param output_path: Destination PDF. Defaults to
        ``config.analysis_dir/reports/<slug>_report.pdf``.
    :param objective_fn: Optional objective for the performance section (``callable(burst_df)->float``).
    :param title: Optional report title (defaults to a description of the selection).

    Phase windows are reconstructed automatically from the phase spec embedded in each recording's
    preprocessed data (written at preprocessing time from the ``Phases`` metadata key). A recording
    with no embedded spec has a single ``"full"`` phase.
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

    # The phase is resolved once, from the spiking summaries, and every section is then filtered to
    # it -- so a report always describes one window and says which.
    spiking_df = _spiking_summaries(cultures, analysis_dir)
    phase = _resolve_phase(spiking_df, phase)

    with PdfPages(output_path) as pdf:
        # Title page.
        info = [
            f"Generated: {_dt.date.today().isoformat()}",
            f"Phase:     {phase}",
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
            _section_activity(pdf, cultures, spiking_df, phase)
        if "bursting" in sections:
            _section_bursting(pdf, cultures, analysis_dir, phase)
        if "stimulation" in sections:
            _section_stimulation(pdf, cultures, analysis_dir, phase)
        if "performance" in sections:
            _section_performance(pdf, cultures, analysis_dir, phase, objective_fn)
        if "cultures" in sections:
            _section_cultures(pdf, cultures, analysis_dir, phase)

    print(f"Report written to: {output_path}")
    return output_path
