# Configuration

MXtreme needs one piece of local information: **where to store its outputs.** 

## The managed store

The *managed store* is a single directory tree the package owns. Every output goes there, and every
later stage reads back from there. The structure is as follows:

```
<data_root>/
├── recordings/                    raw .h5 — scans and ingested experiments, one file per well
│   └── plating_<plate_date>_<batch_id>/
│       └── chip_<M1|M2>_<chip>/
│           └── well_<N>/
│               └── DIV_<d>/
│                   ├── plating_…_chip_<chip>_well_<N>_DIV_<d>_activity_scan.raw.h5
│                   ├── plating_…_chip_<chip>_well_<N>_DIV_<d>_network_scan.raw.h5
│                   └── plating_…_chip_<chip>_well_<N>_DIV_<d>_<exp_id>.raw.h5
├── preprocessed/                  cleaned .npz, one per well per recording
│   └── <exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_exp_data.npz
├── burst_data/                    per-recording burst CSVs + per-experiment burst logs
│   └── <exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_burst_data.csv
├── analysis/                      per-culture summary CSVs, plots, PDF reports
│   ├── <category>/<exp_id>/<chip>/well<N>/<culture_id>_<name>.csv
│   └── reports/<slug>_report.pdf
├── registry.csv                   index of every raw file and recording in the store
└── transactions.jsonl             append-only journal of everything that changed the store
```

The recordings tree is keyed by **plating batch** — one plating event, named at the bench — and every
file in it holds exactly one well, so one culture's whole history sits in one directory. A multi-well
recording (a MaxTwo) is recorded into a single `.h5` and split per well on the way in
({func}`mxtreme.store.split_by_well`).

A scan MXtreme ran itself lands there automatically. A raw `.h5` acquired **outside** MXtreme joins
the same tree through {func}`mxtreme.store.ingest_recording`, which files and registers it under an
`exp_id` you supply — the free-string tail of its file name.

## Batches

A *batch id* names one plating event, following a fixed convention:

```
<semester><year>_batch<n>_<cell_type>_<M1|M2>      e.g.  fall2026_batch1_DRG_M1
```

{class}`mxtreme.store.Batch` generates and validates them — construct one at the start of a batch
(`Batch(semester="fall", year=2026, number=1, cell_type="DRG", system="M1")`), or pass the id string
anywhere a `batch` is accepted and it is parsed and validated on the way in. Every new activity or
network scan headed for the store requires one, as does every ingest.

## `mxtreme.toml`

The root comes from a small TOML file so it never has to be hard-coded in library code:

```toml
# mxtreme.toml
[data]
root = "/abs/path/to/managed/store"
```

`root` may be relative or use `~`; it is expanded when loaded. `examples/mxtreme.toml` in the
repository is a working template.

Load it into a {class}`~mxtreme.config.Config`:

```python
from mxtreme.config import Config

config = Config.from_toml("mxtreme.toml")
```

{meth}`~mxtreme.config.Config.from_toml` raises {exc}`FileNotFoundError` if the file is missing and
{exc}`KeyError` if `[data].root` is absent — it will not fall back to a default.

## What `Config` gives you

`Config` is a frozen dataclass holding one field, `data_root`, plus derived properties for each
subtree:

| Property | Directory |
|---|---|
| {attr}`~mxtreme.config.Config.recordings_dir` | `data_root/recordings` |
| {attr}`~mxtreme.config.Config.preprocessed_dir` | `data_root/preprocessed` |
| {attr}`~mxtreme.config.Config.burst_data_dir` | `data_root/burst_data` |
| {attr}`~mxtreme.config.Config.analysis_dir` | `data_root/analysis` |
| {attr}`~mxtreme.config.Config.registry_path` | `data_root/registry.csv` |
| {attr}`~mxtreme.config.Config.transactions_path` | `data_root/transactions.jsonl` |

These are what you hand to the functions that write.

## Naming a recording

Three small frozen dataclasses in {mod}`mxtreme.identity` name things without touching the
filesystem:

- {class}`~mxtreme.identity.CultureID` — `exp_id`, `chip`, `well`. One culture across all its DIVs.
- {class}`~mxtreme.identity.RecordingID` — a `CultureID` plus a `div`. One recording. Its `.culture`
  property drops back to the culture.
- {class}`~mxtreme.identity.CultureSelector` — a *query*: whole experiments by `exp_ids`, an explicit
  `cultures` list (which takes precedence), and an optional `divs` filter (`None` means all).

```python
from mxtreme.identity import CultureID, RecordingID, CultureSelector

one_rec  = RecordingID("May2025_Wave", "M07459", "0", div=14)
culture  = one_rec.culture
group    = CultureSelector(exp_ids=["May2025_Wave"], divs=[7, 14, 21])
```

## Turning names into files

{mod}`mxtreme.paths` resolves any of those against a `Config`. Two entry points, for the two shapes
callers actually need:

**{func}`~mxtreme.paths.resolve_paths` mirrors the selection's own structure** — what reporting and
per-experiment grouping want:

| Input | Output |
|---|---|
| `RecordingID` | `RecordingPaths` |
| `CultureID` | `CulturePaths` (all available DIVs) |
| `CultureSelector` | `dict[exp_id, ExperimentPaths]` |

**{func}`~mxtreme.paths.resolve_recordings` flattens** any of those into a
{class}`~mxtreme.paths.RecordingSet` — an ordered, iterable list with `.npz` and `.burst_stats`
accessors, for the common "just give me the files to loop over" case:

```python
from mxtreme.paths import resolve_recordings

recs = resolve_recordings(group, config)
for rp in recs:
    ...
print(recs.npz)          # list[Path] of the cleaned .npz files
print(recs.to_frame())   # tidy DataFrame of the selection
```

Because resolution is driven entirely by the `Config`, nothing here is coupled to a particular
machine. Move the store, edit one line of TOML, and every path follows.

## The registry

`registry.csv` indexes what is in the store. It is upserted on **every** `.npz` write by
{func}`mxtreme.io.register` (called from {func}`~mxtreme.io.save_preprocessed`), on every scan by
{func}`mxtreme.io.register_scan`, and on every ingest by {func}`mxtreme.store.ingest_recording`.

Rows are keyed by `(exp_id, batch_id, chip, well, div, kind)`:

| Column | |
|---|---|
| `exp_id` | the experiment's name — blank for scans, which have none |
| `batch_id`, `plate_date` | which plating batch — blank for rows from before the recordings tree |
| `chip`, `well`, `div` | which culture, on which day |
| `kind` | `preprocessed`, `activity_scan`, `network_scan`, or `experiment` (an ingested raw file) |
| `conditions` | the well's experimental condition, when it has one |
| `timestamp` | when the row was written |

`kind` is part of the key, so a scan and the recordings later preprocessed from the same well and DIV
are separate rows rather than one overwriting the other; `batch_id` is part of it because a chip is
re-plated across batches, so `(chip, well, div)` alone recurs batch after batch.

A scan is registered once per well it recorded, so "what do I have for this culture" stays a single
query over a single table. Path resolution ({mod}`mxtreme.paths`) reads only the `preprocessed` rows —
a raw `.h5` has no cleaned `.npz` behind it, so counting those rows would resolve to files that do
not exist. An older registry (no `kind`, or no `batch_id`/`plate_date`) is migrated on read and gains
the columns on its next write.

If the registry is deleted or drifts out of sync with the files on disk,
{func}`mxtreme.io.rebuild_registry` reconstructs it by walking the store — the preprocessed `.npz`
files and the recordings tree — and returns how many it found. A recording is rebuilt from inside its
`.npz`; a raw file's identity is read back from its *path*, since the recordings tree's fixed-format
directory names carry the batch, plating date, chip, well and DIV in full (its `/assay/metadata`
blob, when readable, supplies the condition label). A file whose path does not follow the layout is
reported rather than guessed at.

## The transaction log

`transactions.jsonl` is the store's journal: one JSON record per line, appended and never
rewritten, for everything that changed the store and every decision made about the data
({mod}`mxtreme.transactions`). The registry says *what is there*; the log says *what happened,
when, and by whom*.

MXtreme writes to it itself, from the function that made the change, right after the change is on
disk: a scan registered (`activity_scan.registered`, `network_scan.registered`, one per well), an
outside recording ingested (`recording.ingested`), a well preprocessed (`preprocessed.saved`), its
spike order repaired (`preprocessed.repaired`), bursts detected (`bursts.saved`), a report written
(`report.written`, one per culture in it), the registry rebuilt (`registry.rebuilt`). A front end
records people's decisions in the same file: `culture.mark_dead` / `culture.mark_alive`,
`batch.mark_dead` / `batch.mark_alive`, `chip.set_device`, `note`. The full vocabulary is
{data}`mxtreme.transactions.OPS`.

Every record carries as much identity as the writer had — `batch_id`, `plate_date`, `exp_id`,
`chip`, `well`, `div` — plus `actor` (the OS user, for MXtreme's own entries), a `note` and
op-specific `data` (paths written, counts). One culture's history is
{func}`mxtreme.transactions.for_culture`; the alive/dead state of the decision records is a fold,
{func}`~mxtreme.transactions.batch_states` / {func}`~mxtreme.transactions.culture_states` /
{func}`~mxtreme.transactions.is_dead`, computed on read.

Nothing is edited or deleted; a correction is another line. A store that predates the log gets a
history once from its registry and burst logs with {func}`mxtreme.transactions.backfill`, which
skips what is already journaled and marks what it adds as back-filled.
