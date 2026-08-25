# Data flow

MXtreme is organised around one idea: **each stage writes its output to disk, and the next stage
reads it back.** Preprocessing a long recording is expensive and happens once; every later analysis
is cheap and re-runnable. This page traces what moves between the stages and which module owns
each hop.

```
   raw .raw.h5                (your files, read-only, referenced by explicit path)
        │
        │  mxtreme.extract.extract()
        ▼
   dict[well_no, well_dict]   (in memory)
        │
        │  mxtreme.pipeline.Pipeline.run()  — steps from mxtreme.clean
        ▼
   preprocessed/…/*.npz  +  registry.csv      ◄── managed store begins here
        │
        │  mxtreme.io.load_preprocessed() → mxtreme.recording.Recording
        │  mxtreme.bursting.BurstDetector.detect() → .extract_features()
        ▼
   burst_data/…/*_burst_data.csv  +  per-experiment burst log
        │
        │  mxtreme.analysis.{activity,spatial,stimulation,performance}
        ▼
   analysis/<category>/…/*.csv
        │
        │  mxtreme.analysis.generate_report()
        ▼
   analysis/reports/*.pdf
```

Everything from `preprocessed/` down lives inside the **managed store**, a single directory tree the
package owns. Raw `.h5` inputs are deliberately *outside* it — you point at those by explicit path,
and MXtreme never writes near them. See [Configuration](configuration.md).

## Stage 1 — Extract

{func}`mxtreme.extract.extract` opens one raw Maxwell `.h5` and returns a plain
`dict[int, dict]`: well number → that well's spike table plus the metadata and settings needed to
clean, bin, and later analyse it.

Newer Maxwell files carry an embedded `/assay/metadata` blob that is read automatically. Older files
lack it, so the caller supplies a `metadata` dict. A supplied dict is merged **over** the embedded
blob per-key, so you can override individual fields — including an optional `Conditions` list giving
each well's experimental condition — without restating the rest.

Nothing is written at this stage.

## Stage 2 — Clean

{mod}`mxtreme.clean` is a *menu*, not a fixed sequence. Each step takes one well's dict, mutates it
in place, and returns it, so steps chain naturally. Parameters live on the step functions themselves
with sensible defaults rather than in a central params object — pass a bare function to accept the
defaults, or wrap it in {func}`functools.partial` to override.

The available steps, in the order they normally run:

| Step | Effect |
|---|---|
| {func}`~mxtreme.clean.normalize_time` | Re-base frame numbers so the recording starts at 0 |
| {func}`~mxtreme.clean.dac_to_voltage` | Convert DAC units to volts using the file's LSB |
| {func}`~mxtreme.clean.remove_positive_deflections` | Keep only negative-going spikes |
| {func}`~mxtreme.clean.spike_filter` | Drop spikes below an amplitude threshold (default 20 µV) |
| {func}`~mxtreme.clean.remove_spurious_spikes` | Enforce a refractory period (default 2 ms) |
| {func}`~mxtreme.clean.remove_spurious_channels` | Drop channels absent from the channel map |
| {func}`~mxtreme.clean.build_channel_map` | Build the electrode↔channel mapping — **required before binning** |
| {func}`~mxtreme.clean.remove_stim_frames` | Drop spikes inside stimulation artifact windows |
| {func}`~mxtreme.clean.bin_spikes` | Bin into a channels × time matrix (default 10 ms) |

Order matters in places — `build_channel_map` must precede `bin_spikes`, and amplitude filtering
should follow the DAC conversion — but the pipeline is otherwise yours to arrange, and your own step
functions slot in anywhere.

{class}`~mxtreme.pipeline.Pipeline` offers two entry points:

- {meth}`~mxtreme.pipeline.Pipeline.transform` — run the steps, return the data, write nothing.
- {meth}`~mxtreme.pipeline.Pipeline.run` — transform and save **one well at a time**, so at most one
  well's `spike_bin` is resident. This is the memory-safe path for long multi-well recordings, and
  it's what you normally want.

Every parameterised step stamps the values it used into `well['preprocessing_params']`, so each
saved file carries a record of exactly how it was produced.

## Stage 3 — The managed store

{func}`mxtreme.io.save_preprocessed` writes one compressed `.npz` per well at:

```
preprocessed/<exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_exp_data.npz
```

Alongside the arrays (`spike_data`, `spike_bin`, `channelmap`, `eventtime`, …) it stores identity
fields, the `preprocessing_params` provenance dict, the `step_log`, and the `phase_spec`. Fields
produced by optional cleaning steps are defaulted when absent, so the pipeline still saves if you
skip a step.

Each write also **upserts the recording into `registry.csv`**, the store's index of what has been
processed. This happens on every save rather than as a separate manual step, so registry timestamps
stay current automatically. If the registry is ever lost or falls out of sync,
{func}`mxtreme.io.rebuild_registry` reconstructs it by scanning the store.

Reading back is {func}`mxtreme.io.load_preprocessed`, which returns a dict you hand to
{class}`~mxtreme.recording.Recording`.

## Stage 4 — Recording

{class}`~mxtreme.recording.Recording` is a thin, stable data-access view over one well's preprocessed
`.npz`. It holds the arrays and identity, and exposes a few cheap derived views (`asdr`, `event_df`).

It makes **no** experiment-specific assumptions. Detection, feature extraction, saving, and plotting
all deliberately live elsewhere, which is what keeps `Recording` stable while those evolve.

### Phases ride along with the data

A *phase* is a named interval of a recording — `"pre"` / `"train"` / `"post"` in a closed-loop
training experiment, say. MXtreme never assumes phase names, or that phases exist at all.

Phases are anchored to maxlab event tags already present in the recording's event data. You choose
which tags mark boundaries and what to call each interval. There are three ways to supply that, in
increasing order of convenience:

1. **Nothing.** The recording is a single `"full"` phase spanning its whole duration.
2. **At `Recording` construction** — pass `phase_tags=` and `end_tag=`.
3. **At extraction time** — pass a `Phases` key in `metadata`. The spec is written into every `.npz`
   as `phase_spec`, and every later `Recording` resolves it with no arguments at all.

Option 3 is the one to reach for on a real experiment: specify the phase structure once, and
everything downstream — burst labelling, per-phase analysis windows, report sections — picks it up
automatically. A burst's phase is then just {meth}`Phases.label_for <mxtreme.phases.Phases.label_for>`
of its peak frame.

## Stage 5 — Burst detection

{class}`~mxtreme.bursting.BurstDetector` turns a `Recording` into a
{class}`~mxtreme.bursting.detection.BurstSet`. Detection is dispatched by name, so new methods can be
added without changing the interface. Two composable stages ship today:

- **ISI-N grouping** — the detector of Bakkum et al. 2013 (*Parameters for burst detection*, Front.
  Comput. Neurosci.). `ISI_N` is the time span of `N` consecutive spikes; a run of spikes is one
  burst when that span falls below a threshold taken automatically from the valley of the
  `log10(ISI_N)` distribution.
- **Rate thresholding** — within each group's ASDR segment, pick peaks and classify them
  `"network"` or `"mini"` by height.

`method="isi_n"` runs grouping alone (one burst per group); the default `"isi_rate"` composes both.

:::{note}
Fitting a KDE over every inter-spike interval is the dominant cost of detection on a long recording.
A large seeded subsample (`KDE_MAX_SAMPLES = 100_000`) locates the same valley far faster and
reproducibly.
:::

Feature extraction is a separate call, so you can detect once and re-extract features with different
{class}`~mxtreme.params.BurstFeatureParams` without re-detecting.

Passing `burst_data_dir=` to either call writes the results out — per-recording CSVs at
`burst_data/<exp_id>/<chip>/well<N>/DIV<d>_…_burst_data.csv`, plus a per-experiment burst log that
records detection and feature-extraction timestamps independently. Omit it for pure in-memory work.

## Stage 6 — Analysis

Each topic module in {mod}`mxtreme.analysis` follows the same two-level shape:

- A **culture-level** function takes one `CulturePaths`, computes across all its DIVs, and writes a
  tidy per-DIV CSV under `analysis/<category>/<exp_id>/<chip>/well<N>/`.
- A **population-level** function reads those CSVs back and pools them across cultures.

| Module | Culture level | Population level |
|---|---|---|
| {mod}`~mxtreme.analysis.activity` | `burst_activity_summary`, `channel_activity_summary` | `plot_population_burst_summary`, `plot_population_channel_activity` |
| {mod}`~mxtreme.analysis.spatial` | `mea_layout_summary`, `spatial_summary` | `plot_population_spatial_summary` |
| {mod}`~mxtreme.analysis.stimulation` | `stim_summary` | `plot_population_stim_summary` |
| {mod}`~mxtreme.analysis.performance` | `performance_summary` | `plot_population_performance_summary` |

Most culture-level functions take `use_existing=True`, which reuses a previously written CSV instead
of recomputing — the whole reason the intermediate CSVs exist.

### Performance scoring is pluggable

{mod}`~mxtreme.analysis.performance` tracks one user-chosen scalar per recording so it can be plotted
as a learning curve over development. The scalar comes from an `objective_fn(burst_df) -> float`;
studying a different task means supplying your own.

The shipped default scores burst propagation *toward the side the culture was trained to burst from*,
read from each recording's `exp_condition`. Scoring both directions on the same "fraction toward the
target" scale is what makes scores comparable across cultures — pooling a left-trained and a
right-trained culture under one fixed direction would average two opposing conventions and pin the
population mean near 0.5. When the condition is unknown it falls back to a paradigm-free
left-to-right propagation fraction.

## Stage 7 — Reports

{func}`mxtreme.analysis.generate_report` resolves a selection — a single
{class}`~mxtreme.identity.CultureID` or a {class}`~mxtreme.identity.CultureSelector` group — runs the
requested sections, and writes them into one PDF via matplotlib's `PdfPages` (no extra dependencies).

| Section | Contents |
|---|---|
| `overview` | Single culture: ASDR + MEA layout per DIV. Group: a summary table. |
| `activity` | Firing rate, ISI, spike amplitude, active channels. Always available. |
| `bursting` | Detection diagnostics and burst stats (IBI, rate, size, duration). |
| `stimulation` | Total stim/train time per DIV. Auto-skips non-stim cultures. |
| `performance` | Learning curve from the objective function. |

Every section reuses the per-topic analysis functions and the shared `ax`-aware visualizations, so
the report never re-implements a metric — a number in the PDF and the same number from a notebook
come from the same code path.
