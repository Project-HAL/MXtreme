# Quickstart - Scans 

## Activity Scanning

An *activity scan* sweeps the whole array to find where a culture is firing. The chip can only route
1020 electrodes at a time per well, so the array is covered by a series of short recordings, each
routing a different subset. The result is one `.h5` that feeds
{mod}`~mxtreme.scans.electrode_selection` to pick the electrodes for a network scan.

### 1. Describe the scan with ActivityScanParams

```python
from mxtreme.config import Config
from mxtreme.scans import activity_scan

config = Config.from_toml("mxtreme.toml")

params = activity_scan.ActivityScanParams(
    batch="fall2026_batch1_DRG",  # the plating batch -- see Configuration: batches
    chip="M07460",
    plate_date=260810,   # YYMMDD
    div=1,
    wells=[0],           # 0..5; all selected wells are scanned simultaneously
    n_scans=8,           # number of recordings
    rec_length_sec=60,   # length of each recording
    electrode_spacing=1, # 0 = every electrode, 1 = every other in rows and columns
)

print(activity_scan.describe(params))
```

{func}`~mxtreme.scans.activity_scan.describe` prints the run time and how much of the array the scan covers, so you can adjust before committing to the recording. The defaults are a working scan: ~8 minutes over one well, covering roughly a quarter of the array. Use `params.estimated_minutes` and `params.array_coverage` if you want the numbers directly — coverage above `1.0` means the scan revisits electrodes it has already recorded.

{meth}`~mxtreme.scans.activity_scan.ActivityScanParams.validate` runs automatically before the chip is touched, so a missing or malformed batch id, bad wells, an over-long routing request or too coarse a spacing fail immediately rather than mid-scan.

### 2. Run it (requires Maxwell device connection)

```python
result = activity_scan.run_activity_scan(params, config)

print(result.h5_path, result.completed_scans, result.complete)
```

The `.h5` is always finalized — even if a round fails or you stop early — so a partial scan is still readable.

### 3. Where it lands

A scan is an MXtreme output, so it goes into the managed store's recordings tree, keyed by plating
batch:

```
recordings/plating_<plate_date>_<batch_id>/chip_<M1|M2>_<chip>/well_<w>/DIV_<div>/
    plating_<plate_date>_<batch_id>_chip_<chip>_well_<w>_DIV_<div>_activity_scan.raw.h5
```

with one `activity_scan` row per scanned well in `registry.csv`. Each file in the tree holds one
well: a multi-well scan (on a MaxTwo) records into a single `.h5` and is then split into one file
per well ({func}`mxtreme.store.split_by_well`), so every culture's directory holds its own data.
Set `save_path` to write somewhere else instead (a scratch directory on the rig, say); the scan is
then left unsplit and out of the registry, since nothing says which store it belongs to.

```python
params = activity_scan.ActivityScanParams(..., save_path="/tmp/scratch")
```

### 4. Select recording electrodes

Feed the file to {func}`~mxtreme.scans.electrode_selection.select_electrodes`, which filters down to an active subset across all rounds of the activity scan. 

```python
from mxtreme.scans import electrode_selection

rec_elecs = electrode_selection.select_electrodes(str(result.h5_path), save_path="…")
```

It returns `{well: [electrode, ...]}` and writes the summary plots and per-well `recording_electrodes` `.npz` files.

## Network Scanning

A *network scan* is the recording the activity scan exists to set up: one fixed set of electrodes per
well, recorded continuously at full rate. It takes the `{well: [electrode, ...]}` mapping
{func}`~mxtreme.scans.electrode_selection.select_electrodes` returns and records exactly those wells.

Same shape as an activity scan — planning is offline, running needs `maxlab` on the rig, and the
`.h5` lands in the managed store.

```python
from mxtreme.scans import electrode_selection, network_scan

rec_elecs = electrode_selection.select_electrodes(str(result.h5_path), save_path="…")

params = network_scan.NetworkScanParams(
    recording_electrodes=rec_elecs,   # {well: [electrode, ...]}; the wells come from its keys
    batch="fall2026_batch1_DRG",
    chip="M07460",
    plate_date=260810,
    div=25,
    rec_length_sec=300,
)

print(network_scan.describe(params))
```

There is no electrode plan to make — what to record is an input — so the only scan parameter is the
recording length. {meth}`~mxtreme.scans.network_scan.NetworkScanParams.validate` catches a well
asking for more than the 1020 electrodes the chip can route, a duplicated or out-of-range electrode,
and a bad well before the chip is touched.

Electrode selection thresholds each well down to the electrodes that were genuinely active, usually
well under the routing limit. Spare routing capacity records nothing, so each well is topped up to
1020 with random electrodes — the chosen ones are recorded exactly as selected, and the rest is a
free look at the array. Pass `seed=` for a reproducible fill, or `pad_to_max=False` to record only
what selection chose. A well selection left empty is not padded.

Then run it on the rig:

```python
scan = network_scan.run_network_scan(params, config)

print(scan.h5_path, scan.recorded_sec, scan.complete)
```

Each well's electrodes are routed, amplifier offsets are compensated, then every well records
simultaneously. Progress is printed every 30 s; pass `on_progress=` to route it elsewhere and
`should_stop=` for a callback that ends the recording early. The `.h5` is always finalized, so a
scan cut short is still readable.

The file goes into the recordings tree beside the activity scan it came from — the same
`.../well_<w>/DIV_<div>/` directory, as `..._network_scan.raw.h5` — with one `network_scan` row per
well in `registry.csv`. A multi-well scan is split per well on the way in, and `save_path` writes it
elsewhere, unsplit and unregistered — the same rules as an activity scan.

A network scan is an ordinary recording, so it feeds straight into the analysis workflow below:

```python
from mxtreme import extract

data = extract.extract(str(scan.h5_path))
```

## Ingesting an outside recording

A raw `.h5` recorded outside MXtreme (MaxLab's own Record tab, a collaborator's export) joins the
same tree through {func}`mxtreme.store.ingest_recording`. You supply the identity the file cannot —
its batch, chip, DIV — plus a free-form `experiment` name, which becomes the file-name tail and the
registry row's `experiment`. The batch stays the recording's identity; the name only tells it apart
from a scan of the same culture on the same DIV:

```python
from mxtreme.store import ingest_recording

ingest_recording(
    "/path/to/export.raw.h5", config,
    batch="fall2026_batch1_DRG", plate_date=260810,
    chip="M07460", div=21, experiment="stim_trial_3",
)
```

The file is copied into `.../well_<w>/DIV_<div>/` under the canonical name (pass `move=True` to move
it instead), a multi-well file is split into one file per well, and each well gets an `experiment`
row in `registry.csv`.

A scan recorded by MaxLab Live's own assays (Scope's Activity Scan, a network recording made from
its Record tab) is not an experiment: pass `kind="activity_scan"` or `kind="network_scan"` instead
of an `experiment` name, and it is filed under the scan tail and registered as that kind — indistinguishable
in the store from a scan MXtreme ran, so electrode selection and the analysis chain read it as one.
`wells=[...]` keeps only the plated wells of a multi-well plate. Before deciding what a file is,
{func}`mxtreme.store.describe_recording` reads what it says about itself — the chip Scope wrote to
`/wellplate/id`, the wells with data, the recording count per well, when it was recorded, the
plating date typed into Scope, and a guess at the kind — and lists any reason it cannot be
ingested, without raising:

```python
from mxtreme.store import describe_recording

d = describe_recording("/path/to/M07460_260831.h5")
if d.ok:
    ingest_recording(d.path, config, batch="fall2026_batch1_DRG", plate_date=260810,
                     chip=d.chip, div=21, kind=d.kind_guess, wells=[w.well for w in d.wells])
else:
    print(d.problems)
```
