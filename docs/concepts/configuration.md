# Configuration

MXtreme needs one piece of local information: **where to store its outputs.** 

## The managed store

The *managed store* is a single directory tree the package owns. Every output goes there, and every
later stage reads back from there. The structure is as follows:

```
<data_root>/
├── scans/                         activity-scan .h5, as written on the rig
│   └── <exp_id>/<chip>/DIV<d>_<plate_date>_<chip>_<exp_id>_activity_scan.raw.h5
├── preprocessed/                  cleaned .npz, one per well per recording
│   └── <exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_exp_data.npz
├── burst_data/                    per-recording burst CSVs + per-experiment burst logs
│   └── <exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_burst_data.csv
├── analysis/                      per-culture summary CSVs, plots, PDF reports
│   ├── <category>/<exp_id>/<chip>/well<N>/<culture_id>_<name>.csv
│   └── reports/<slug>_report.pdf
└── registry.csv                   index of every scan and recording in the store
```

An activity scan has no `well<N>` level: one `.h5` holds every well it recorded.

Raw `.h5` recordings acquired **outside** MXtreme are not resolved through the store — you point at
those by explicit path. A scan MXtreme ran itself is an output, and does live in the store.

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
| {attr}`~mxtreme.config.Config.scans_dir` | `data_root/scans` |
| {attr}`~mxtreme.config.Config.preprocessed_dir` | `data_root/preprocessed` |
| {attr}`~mxtreme.config.Config.burst_data_dir` | `data_root/burst_data` |
| {attr}`~mxtreme.config.Config.analysis_dir` | `data_root/analysis` |
| {attr}`~mxtreme.config.Config.registry_path` | `data_root/registry.csv` |

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
{func}`mxtreme.io.register`, called from {func}`~mxtreme.io.save_preprocessed`, and on every activity
scan by {func}`mxtreme.io.register_scan`.

Rows are keyed by `(exp_id, chip, well, div, kind)`:

| Column | |
|---|---|
| `exp_id`, `chip`, `well`, `div` | which culture, on which day |
| `kind` | `preprocessed` for a cleaned recording, `activity_scan` for a scan |
| `conditions` | the well's experimental condition, when it has one |
| `timestamp` | when the row was written |

`kind` is part of the key, so a scan and the recordings later preprocessed from the same well and DIV
are separate rows rather than one overwriting the other:

```
exp_id,chip,well,div,kind,conditions,timestamp
May2025_Wave,M07459,0,14,activity_scan,,2026-08-27T09:14:02
May2025_Wave,M07459,0,14,preprocessed,"[2, 0]",2026-08-27T11:40:57
```

A scan is registered once per well it recorded, so "what do I have for this culture" stays a single
query over a single table. Path resolution ({mod}`mxtreme.paths`) reads only the `preprocessed` rows —
a scan is a raw `.h5` with no cleaned `.npz` behind it, so counting those rows would resolve to files
that do not exist. A registry written before scans were indexed has no `kind` column; it is read as
all-`preprocessed` and gains the column on its next write.

If the registry is deleted or drifts out of sync with the files on disk,
{func}`mxtreme.io.rebuild_registry` reconstructs it by scanning the store — both the preprocessed
`.npz` files and the scans — and returns how many it found. Both are rebuilt from inside the file
itself: recordings from the `.npz`, scans from the `/assay/metadata` blob in the `.h5` that
{func}`mxtreme.extract.extract` also reads. File names are never parsed back into fields, since an
`exp_id` may contain the same underscores the name separates fields with. A scan with no such blob is
reported rather than guessed at.
