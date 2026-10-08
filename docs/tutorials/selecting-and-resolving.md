# Selecting recordings and resolving paths

This tutorial shows how {mod}`mxtreme.identity` and {mod}`mxtreme.paths` work together: you pick a
single recording, a single culture, or a group of cultures, resolve all of their files, and then
reach them dictionary style.

## The structure

{mod}`mxtreme.identity` only names things; it never touches disk. {mod}`mxtreme.paths` takes those
names, looks them up in `registry.csv` (`kind == "preprocessed"` rows only), and returns file paths.
What you get back depends on what you pass in:

| You build | {func}`~mxtreme.paths.resolve_paths` returns |
|---|---|
| {class}`RecordingID(batch_id, chip, well, div) <mxtreme.identity.RecordingID>` | {class}`~mxtreme.paths.RecordingPaths`: one recording |
| {class}`CultureID(batch_id, chip, well) <mxtreme.identity.CultureID>` | {class}`~mxtreme.paths.CulturePaths`: every recording on record |
| {class}`CultureSelector(batch_ids=..., cultures=..., divs=...) <mxtreme.identity.CultureSelector>` | `dict[batch_id, BatchPaths]`: a group |

They nest like this:

```
dict[batch_id -> BatchPaths]
  BatchPaths.batch_id, .burst_summary
  BatchPaths.cultures      dict[CultureID -> CulturePaths]
    CulturePaths.culture_id
    CulturePaths.recordings   RecordingSet (RecordingPaths ordered by DIV, then experiment label)
      RecordingPaths.recording_id, .npz, .burst_stats (None until burst detection)
```

## Single recording

```python
from mxtreme.identity import RecordingID
from mxtreme.paths import resolve_paths

rid = RecordingID("fall2026_batch1_E18", "P006677", "0", div=14)
rec = resolve_paths(rid, config)

rec.npz                    # Path to the .npz
rec.burst_stats            # Path, or None if bursts haven't been detected
rec.require_burst_stats()  # same, but raises a clear error if None
```

## Single culture (all DIVs)

```python
from mxtreme.identity import CultureID

culture = CultureID(batch_id="fall2026_batch1_E18", chip="P006677", well="0")
cp = resolve_paths(culture, config)

cp.divs                      # [7, 14, 21, ...]: the DIVs found in the registry
cp.recording(14).npz         # the one recording on DIV 14 (an int, not "14")

for rec in cp.recordings:    # every recording, ordered by DIV then label
    print(rec.recording_id.div, rec.recording_id.experiment, rec.npz)
```

## A group of cultures

```python
from mxtreme.identity import CultureSelector

sel = CultureSelector(batch_ids=["fall2026_batch1_E18"])                 # whole batch
sel = CultureSelector(batch_ids=["fall2026_batch1_E18"], divs=[7, 14])   # only some DIVs
sel = CultureSelector(cultures=[CultureID("fall2026_batch1_E18", "P006677", "0"),
                                CultureID("fall2026_batch1_E18", "P006678", "1")])  # hand-picked

batch_paths = resolve_paths(sel, config)
bp = batch_paths["fall2026_batch1_E18"]   # outer dict is keyed by batch_id string

for cid, cp in bp.cultures.items():
    for rec in cp.recordings:
        print(cid.chip, cid.well, rec.recording_id.div, rec.npz)
```

**Note:** `bp.cultures` is keyed by `CultureID` objects, not strings.
`bp.cultures["P006677"]` raises `KeyError` so you must look up a culture up with an ID instead:

```python
cp = bp.cultures[CultureID("fall2026_batch1_E18", "P006677", "0")]
cp = bp.culture(CultureID("fall2026_batch1_E18", "P006677", "0"))   # same
```

`CultureID` converts its fields to strings, so `well=0` and `well="0"` give the same key. If you'd
rather use string keys, build them yourself:

```python
by_name = {f"{cid.chip}/well{cid.well}": cp for cid, cp in bp.cultures.items()}
by_name["P006677/well0"].recording(14).npz
```

## Use `resolve_paths_flat` to get a flat list of file names

{func}`~mxtreme.paths.resolve_paths_flat` takes any of the three targets and flattens the result
into an ordered list, which is often easier than walking the nested dicts:

```python
from mxtreme.paths import resolve_paths_flat

recs = resolve_paths_flat(sel, config)   # also accepts a CultureID or RecordingID
recs.npz            # list[Path]
recs.burst_stats    # list[Path | None]
recs.ids            # list[RecordingID]
df = recs.to_frame()   # columns: batch_id, chip, well, div, experiment, npz, burst_stats
```

`to_frame()` is the easiest way to filter by columns:

```python
df[(df.chip == "P006677") & (df.div >= 14)].npz.tolist()
```

## Labeled recordings (`experiment`)

A recording can carry an `experiment` label, such as `NS` or `train`, which also ends its file
name. By default, resolution matches recordings whatever their label, so you don't need to know
the labels to find a culture's files.

To narrow it to one label, specify the experiment with `experiment=...`. `""` means unlabelled recordings only (i.e. no provided experiment label):

```python
resolve_paths(culture, config, experiment="NS")
CultureSelector(batch_ids=["fall2026_batch1_E18"], experiment="NS")
RecordingID("fall2026_batch1_E18", "P004722", "0", 27, experiment="NS")
```

### Several recordings on one DIV

A culture can have more than one recording on the same DIV, as long as each has its own experiment label to distinguish them. When the default `experiment=None` is passed to the `resolve_paths` function all of the recordings are kept in the output. Every summary row carries an `experiment` column, and, in the analysis, population plots group by `(div, experiment, phase)`, so replicates on one DIV stay separate instead of being averaged together.

```python
cp = resolve_paths(culture, config)
cp.divs                          # [12, 27]
cp.experiments                   # ['NS', 'train']
cp.at_div(27).to_frame()         # one row per recording on DIV 27
cp.recording(27, "train")    # one of them, by label
cp.recording(27)                 # ValueError: names the three labels and asks you to pick one
```

A `RecordingID` without a label on such a DIV raises a
`ValueError`. To avoid this, give it `experiment=...`.