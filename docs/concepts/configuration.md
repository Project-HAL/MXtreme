# Configuration

MXtreme needs one piece of local information: **where to store its outputs.** 

## The managed store

The *managed store* is a single directory tree the package owns. Every output goes there, and every
later stage reads back from there. The structure is as follows:

```
<data_root>/
├── preprocessed/                  cleaned .npz, one per well per recording
│   └── <exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_exp_data.npz
├── burst_data/                    per-recording burst CSVs + per-experiment burst logs
│   └── <exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_burst_data.csv
├── analysis/                      per-culture summary CSVs, plots, PDF reports
│   ├── <category>/<exp_id>/<chip>/well<N>/<culture_id>_<name>.csv
│   └── reports/<slug>_report.pdf
└── registry.csv                   index of everything that has been processed
```

Raw `.h5` inputs are **not** resolved through the store. You point at those by explicit path.

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

`registry.csv` indexes what has been processed. It is upserted on **every** `.npz` write by
{func}`mxtreme.io.register`, called from {func}`~mxtreme.io.save_preprocessed`.

If it is deleted or drifts out of sync with the files on disk,
{func}`mxtreme.io.rebuild_registry` reconstructs it by scanning the store and returns the number of
recordings found.
