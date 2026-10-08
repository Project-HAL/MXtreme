"""Assemble a multi-section MXtreme PDF report for a culture or group of cultures.

:func:`generate_report` resolves a selection (single :class:`~mxtreme.identity.CultureID` or a
:class:`~mxtreme.identity.CultureSelector` group) against a :class:`~mxtreme.config.Config`, runs the
requested analysis sections, and writes them into one PDF via ``matplotlib``'s :class:`PdfPages`
(no extra dependencies). Each section reuses the per-topic analysis functions and the shared
``ax``-aware visualizations, so the report never re-implements a metric.

The report answers two questions in order -- *how is the group developing?* and *what is each culture
doing?*:

1. a title page naming the phase being plotted, with the selection as a compact table -- one row per
   culture, DIVs as ranges -- that continues onto further pages for a large selection;
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
- ``"overview"``   -- accepted for compatibility; the selection table on the title page replaces it.

Phases: exactly one phase is plotted per report. ``phase=None`` selects the first phase present in
the data -- ``"full"`` for recordings with no user-supplied phases, otherwise the first of the
sequence -- and any phase name can be requested explicitly. When every recording is unphased the
phase is left out of the titles and legends.
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
from mxtreme.analysis._paths import recording_label, stamp_identity
from mxtreme.analysis._plotting import is_unphased, natural_key, phase_tag, plot_cdf_grid, plot_metric_grid, sort_phases
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

# How a per-culture page names its phase (dropped when the data is unphased).
_CULTURE_TAG = "  (phase: {phase})"

# (distribution key, x label, panel title, log x) -- see `activity.spike_activity_distributions` and
# `activity.burst_activity_distributions`.
_CDF_METRICS = activity.SPIKE_CDF_METRICS + activity.BURST_CDF_METRICS


# --- selection helpers --------------------------------------------------------------------------


def _flatten_cultures(resolved) -> list[CulturePaths]:
    """Return a flat list of ``CulturePaths`` from a ``resolve_paths`` result."""
    if isinstance(resolved, CulturePaths):
        return [resolved]
    # dict[batch_id, BatchPaths]
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


def _report_path(analysis_dir: Path, cultures, output_path, name) -> Path:
    """Where the report goes: ``output_path`` as given, else ``<analysis_dir>/reports/`` under
    ``name`` (or the selection's default slug)."""
    if output_path is not None and name is not None:
        raise ValueError("Pass either `name` or `output_path`, not both.")
    if output_path is not None:
        return Path(output_path)
    if name is None:
        return analysis_dir / "reports" / f"{_selection_slug(cultures)}_report.pdf"
    name = str(name)
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"`name` must be a bare file name, not {name!r}; use `output_path` for a "
                         "full path.")
    return analysis_dir / "reports" / (name if name.lower().endswith(".pdf") else f"{name}.pdf")


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


def _div_ranges(divs) -> str:
    """Compact DIV list: consecutive runs collapse to ranges, e.g. ``32–36, 38, 40–43``."""
    divs = sorted({int(d) for d in divs})
    runs = []
    for d in divs:
        if runs and d == runs[-1][1] + 1:
            runs[-1][1] = d
        else:
            runs.append([d, d])
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs)


# Overview table: (header, width as a fraction of the table, wrap width in characters or None).
_OVERVIEW_COLUMNS = [
    ("Batch",       0.20, 22),
    ("Chip",        0.10, None),
    ("Well",        0.06, None),
    ("Device",      0.08, None),
    ("# rec",       0.06, None),
    ("DIVs",        0.22, 26),
    ("Experiments", 0.28, 34),
]
_OVERVIEW_FONTSIZE = 9
_OVERVIEW_LEADING = 1.45  # line height as a multiple of the font size
_PAGE_SIZE = (11, 8.5)
_TABLE_BOTTOM = 0.05      # figure fraction the table stops at
_CONTINUED_TOP = 0.91     # figure fraction a continuation page's table starts at
_FACTS_FONTSIZE = 11


def _first_table_top(n_facts: int) -> float:
    """Figure fraction the first page's table starts at, below the title and ``n_facts`` lines."""
    return 0.86 - (n_facts + 1) * _FACTS_FONTSIZE * 1.3 / 72 / _PAGE_SIZE[1]


def _table_lines(top: float) -> int:
    """Text lines of table body that fit between ``top`` and the bottom margin (header row excluded)."""
    band_pt = (top - _TABLE_BOTTOM) * _PAGE_SIZE[1] * 72
    return int(band_pt // (_OVERVIEW_FONTSIZE * _OVERVIEW_LEADING)) - 2


def _overview_rows(cultures) -> list[list[str]]:
    """One wrapped row per culture, sorted by batch."""
    import textwrap

    rows = []
    for c in sorted(cultures, key=lambda c: (str(c.culture_id.batch_id), str(c.culture_id.chip),
                                             str(c.culture_id.well))):
        cid = c.culture_id
        recordings = list(c.recordings)
        experiments = sorted({r.recording_id.experiment for r in recordings if r.recording_id.experiment},
                             key=natural_key)
        values = [
            str(cid.batch_id),
            str(cid.chip), str(cid.well), _device_type(cid.chip), str(len(recordings)),
            _div_ranges(r.recording_id.div for r in recordings),
            ", ".join(experiments) or "—",
        ]
        rows.append([textwrap.fill(v, width) if width else v
                     for v, (_h, _w, width) in zip(values, _OVERVIEW_COLUMNS)])
    return rows


def _row_lines(row) -> int:
    return max(cell.count("\n") + 1 for cell in row)


def _paginate(rows, first_lines: int, continued_lines: int) -> list[list]:
    """Split ``rows`` into pages by the number of text lines each page can hold. A row costs one line
    per line of its tallest wrapped cell, plus one of padding."""
    pages, page, used, budget = [], [], 0, first_lines
    for row in rows:
        if page and used + _row_lines(_blank_repeat(row, page)) + 1 > budget:
            pages.append(page)
            page, used, budget = [], 0, continued_lines
        used += _row_lines(_blank_repeat(row, page)) + 1
        page.append(row)
    pages.append(page)
    return [[_blank_repeat(row, page[:j]) for j, row in enumerate(page)] for page in pages]


def _blank_repeat(row, previous):
    """``row`` with its batch blanked when the row above it on the page (``previous[-1]``) shares it,
    so a batch is named once per page: on its first row there, even when the page starts mid-batch."""
    return ["", *row[1:]] if previous and previous[-1][0] == row[0] else row


def _draw_overview_table(fig, rows, top: float, bottom: float = _TABLE_BOTTOM) -> None:
    """Draw ``rows`` as a table filling the band of ``fig`` between ``top`` and ``bottom``."""
    ax = fig.add_axes([0.05, bottom, 0.90, top - bottom])
    ax.axis("off")
    if not rows:
        return
    # Row height in axes units: one text line (with leading) per wrapped line, plus padding.
    line_h = (_OVERVIEW_FONTSIZE * _OVERVIEW_LEADING / 72) / (fig.get_figheight() * (top - bottom))
    table = ax.table(cellText=rows, colLabels=[h for h, _w, _wrap in _OVERVIEW_COLUMNS],
                     colWidths=[w for _h, w, _wrap in _OVERVIEW_COLUMNS],
                     loc="upper center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(_OVERVIEW_FONTSIZE)
    for (r, _c), cell in table.get_celld().items():
        n_lines = 1 if r == 0 else _row_lines(rows[r - 1])
        cell.set_height(line_h * (n_lines + 1))
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")


def _overview_figures(title: str, facts: list[str], cultures) -> list:
    """The report's opening pages: title, the key facts, and the selection as a table that continues
    onto further pages rather than running off the edge of the first."""
    first_top = _first_table_top(len(facts))
    pages = _paginate(_overview_rows(cultures), _table_lines(first_top), _table_lines(_CONTINUED_TOP))
    figures = []
    for i, page in enumerate(pages):
        fig = plt.figure(figsize=_PAGE_SIZE)
        if i == 0:
            fig.text(0.5, 0.93, title, ha="center", va="center", fontsize=20, fontweight="bold",
                     wrap=True)
            fig.text(0.06, 0.86, "\n".join(facts), ha="left", va="top", fontsize=_FACTS_FONTSIZE,
                     family="monospace")
            top = first_top
        else:
            fig.text(0.06, 0.95, "Selection (continued)", ha="left", va="center", fontsize=14,
                     fontweight="bold")
            top = _CONTINUED_TOP
        _draw_overview_table(fig, page, top)
        figures.append(fig)
    return figures


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


def _spiking_summaries(cultures, analysis_dir) -> pd.DataFrame:
    """Pooled per-culture spiking summaries -- also the frame the report's phase is resolved from."""
    return _pool(
        (cpath, activity._culture_spike_summary(cpath, analysis_dir))
        for cpath in cultures
    )


def _section_activity(pdf, cultures, pop_df, phase, unphased=False):
    _population_page(
        pdf, _for_phase(pop_df, phase), _SPIKING_METRICS,
        value_cols=["mean_fr_hz", "mean_isi_msec", "mean_amp_uv", "pct_active_chan"],
        suptitle=f"Spiking Activity — {len(cultures)} cultures{phase_tag(phase, unphased)}",
    )


def _section_bursting(pdf, cultures, analysis_dir, phase, unphased=False):
    pop_df = _pool(
        (cpath, activity._culture_burst_summary(cpath, analysis_dir))
        for cpath in cultures
    )
    _population_page(
        pdf, _for_phase(pop_df, phase), _BURST_METRICS,
        value_cols=["burst_rate_hz", "median_ibi_sec", "median_size_pct", "median_dur_sec"],
        suptitle=f"Burst Activity — {len(cultures)} cultures{phase_tag(phase, unphased)}",
    )


def _section_stimulation(pdf, cultures, analysis_dir, phase, unphased=False):
    pairs = []
    for cpath in cultures:
        try:
            df = stimulation._culture_stim_summary(cpath, analysis_dir)
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
        pdf, _for_phase(_pool(pairs), phase),
        [(y, f"sem_{y}", label, title) for (y, _e, label, title) in stimulation._SPEC.metrics],
        value_cols=["total_stim_ms", "phase_dur_min"],
        suptitle=f"Stimulation — {len(pairs)} cultures{phase_tag(phase, unphased)}",
    )


def _section_performance(pdf, cultures, analysis_dir, phase, objective_fn, unphased=False):
    objective_fn = objective_fn or performance.default_direction_objective

    pop_df = _pool(
        (cpath, performance._culture_performance_summary(cpath, analysis_dir, objective_fn=objective_fn))
        for cpath in cultures
    )
    _population_page(
        pdf, _for_phase(pop_df, phase), [('score', 'sem_score', 'Score', 'Performance')],
        value_cols=["score"],
        suptitle=f"Performance — {len(cultures)} cultures{phase_tag(phase, unphased)}",
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


def _culture_recording_page(pdf, cpath, phase, unphased=False):
    """Page A: the recording itself over DIV -- ASDR on top, burst-origin map below, one column per
    recording (several on a DIV with more than one labelled recording)."""
    cid = cpath.culture_id
    recordings = list(cpath.recordings)
    if not recordings:
        return

    # One load per recording, reused by both rows and by the shared vmax pass.
    loaded = []
    for rp in recordings:
        rec = Recording(0, io.load_preprocessed(rp.npz))
        loaded.append((rec, pd.read_csv(rp.require_burst_stats())))

    vmax = _origin_density_vmax(loaded, phase)

    # The page is sized from its panels rather than the other way round: a fixed letter-landscape page
    # would squeeze eight DIVs into unreadable slivers across, and stretch each panel into a tall
    # ribbon down. Both rows get a height derived from the column width -- the ASDR traces a squat
    # rectangle, the origin maps the array's own 3850 x 2100 µm aspect -- so the page ends up wide and
    # short. PdfPages takes each page at whatever size its figure is.
    page_w = max(11, 2.4 * len(recordings))
    col_w = page_w / len(recordings)
    asdr_h = col_w / ASDR_PANEL_ASPECT
    origin_h = col_w * device.CHIP_HEIGHT / device.CHIP_WIDTH
    # + room for the suptitle, panel titles and axis labels, which don't scale with the panels.
    fig, axes = plt.subplots(2, len(recordings), figsize=(page_w, asdr_h + origin_h + 1.9), squeeze=False,
                             gridspec_kw={"height_ratios": [asdr_h, origin_h]})

    for col, (rp, (rec, burst_df)) in enumerate(zip(recordings, loaded)):
        ax_asdr, ax_origin = axes[0][col], axes[1][col]

        # Both rows show the selected phase only, like every other page in the report -- and at this
        # panel size a full hour of ASDR next to a 20-minute phase would just be a solid block.
        window = next((p for p in rec.phases if p.name == phase), None)
        zoom = None
        if window is not None:
            bins_per_frame = 1 / (rec.samp_rate * rec.bin_size)
            zoom = (window.start_frame * bins_per_frame, window.end_frame * bins_per_frame)

        title = recording_label(rp.recording_id.div, rp.recording_id.experiment)
        viz.plot_bursts_on_asdr(rec, burst_df, ax=ax_asdr, zoom=zoom, title=title)
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

    fig.suptitle(f"{cid}  |  ASDR and burst origins over DIV{phase_tag(phase, unphased, _CULTURE_TAG)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)  # resolution of the rasterized panels above
    plt.close(fig)


def _culture_distribution_page(pdf, cpath, analysis_dir, phase, unphased=False):
    """Page B: the distributions behind the summary statistics, one CDF line per recording."""
    cid = cpath.culture_id
    spikes = activity._spike_distributions(cpath, analysis_dir, phase=phase)
    bursts = activity._burst_distributions(cpath, analysis_dir, phase=phase)
    dists = {key: {**spikes.get(key, {}), **bursts.get(key, {})} for key in sorted({*spikes, *bursts})}
    if not dists:
        return

    fig = plot_cdf_grid(
        dists, _CDF_METRICS,
        suptitle=f"{cid}  |  distributions over DIV{phase_tag(phase, unphased, _CULTURE_TAG)}",
        show_plot=False, save_path=None,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _section_cultures(pdf, cultures, analysis_dir, phase, unphased=False):
    for cpath in cultures:
        print(f"Building culture pages for {cpath.culture_id}")
        _culture_recording_page(pdf, cpath, phase, unphased)
        _culture_distribution_page(pdf, cpath, analysis_dir, phase, unphased)


# --- entry point --------------------------------------------------------------------------------


def generate_report(
    target: CultureID | CultureSelector | RecordingID,
    config,
    *,
    sections=DEFAULT_SECTIONS,
    phase: str | None = None,
    output_path: str | Path | None = None,
    name: str | None = None,
    objective_fn=None,
    title: str | None = None,
) -> Path:
    """Generate a multi-section PDF report for ``target`` and return the written path.

    :param target: A :class:`~mxtreme.identity.CultureID` (single culture) or
        :class:`~mxtreme.identity.CultureSelector` (a group, possibly across batches).
    :param config: The :class:`~mxtreme.config.Config` describing the managed store.
    :param sections: Which sections to include (see module docstring). Defaults to activity +
        bursting + per-culture pages; add ``"stimulation"`` / ``"performance"`` as needed.
    :param phase: Which phase to plot. ``None`` (the default) uses the first phase present in the
        data: ``"full"`` for recordings with no user-supplied phases, otherwise the first of the
        sequence. Pass a name (e.g. ``"train"``) to report on that window instead.
    :param output_path: Destination PDF. Defaults to
        ``config.analysis_dir/reports/<slug>_report.pdf``.
    :param name: File name for the report, kept in the default ``config.analysis_dir/reports/``
        directory (``.pdf`` is added if missing). Use ``output_path`` instead to choose the directory
        too; passing both is an error.
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

    output_path = _report_path(analysis_dir, cultures, output_path, name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = title or (f"MXtreme Report — {cultures[0].culture_id}" if single
                      else f"MXtreme Report — {len(cultures)} cultures")

    # The phase is resolved once, from the spiking summaries, and every section is then filtered to
    # it -- so a report always describes one window and says which.
    spiking_df = _spiking_summaries(cultures, analysis_dir)
    phase = _resolve_phase(spiking_df, phase)

    # Recordings with no user-supplied phases are all "full": naming it everywhere says nothing.
    unphased = is_unphased(spiking_df["phase"].dropna().unique())

    with PdfPages(output_path) as pdf:
        facts = [f"Generated: {_dt.date.today().isoformat()}"]
        if not unphased:
            facts.append(f"Phase:     {phase}")
        facts += [f"Cultures:  {len(cultures)}",
                  f"Sections:  {', '.join(s for s in sections if s != 'overview')}"]
        for fig in _overview_figures(title, facts, cultures):
            pdf.savefig(fig)
            plt.close(fig)

        if "activity" in sections:
            _section_activity(pdf, cultures, spiking_df, phase, unphased)
        if "bursting" in sections:
            _section_bursting(pdf, cultures, analysis_dir, phase, unphased)
        if "stimulation" in sections:
            _section_stimulation(pdf, cultures, analysis_dir, phase, unphased)
        if "performance" in sections:
            _section_performance(pdf, cultures, analysis_dir, phase, objective_fn, unphased)
        if "cultures" in sections:
            _section_cultures(pdf, cultures, analysis_dir, phase, unphased)

    print(f"Report written to: {output_path}")

    from mxtreme import transactions

    for c in cultures:
        transactions.record(
            config,
            "report.written",
            batch_id=c.culture_id.batch_id,
            chip=c.culture_id.chip,
            well=c.culture_id.well,
            data={
                "path": str(output_path),
                "divs": c.divs,
                "recordings": [str(r.recording_id) for r in c.recordings],
                "sections": list(sections),
            },
        )
    return output_path
