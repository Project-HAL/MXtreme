# Quickstart 

## Scans 

### Activity Scanning

An *activity scan* sweeps the whole array to find where a culture is firing. The chip can only route
1020 electrodes at a time per well, so the array is covered by a series of short recordings, each
routing a different subset. The result is one `.h5` that feeds
{mod}`~mxtreme.scans.electrode_selection` to pick the electrodes for a network scan.

#### 1. Describe the scan with ActivityScanParams

```python
from mxtreme.config import Config
from mxtreme.scans import activity_scan

config = Config.from_toml("mxtreme.toml")

params = activity_scan.ActivityScanParams(
    batch="fall2026_batch1_DRG_M1",  # the plating batch -- see Configuration: batches
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

#### 2. Run it (requires Maxwell device connection)

```python
result = activity_scan.run_activity_scan(params, config)

print(result.h5_path, result.completed_scans, result.complete)
```

The `.h5` is always finalized — even if a round fails or you stop early — so a partial scan is still readable.

#### 3. Where it lands

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

#### 4. Select recording electrodes

Feed the file to {func}`~mxtreme.scans.electrode_selection.select_electrodes`, which filters down to an active subset across all rounds of the activity scan. 

```python
from mxtreme.scans import electrode_selection

rec_elecs = electrode_selection.select_electrodes(str(result.h5_path), save_path="…")
```

It returns `{well: [electrode, ...]}` and writes the summary plots and per-well `recording_electrodes` `.npz` files.

### Network Scanning

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
    batch="fall2026_batch1_DRG_M1",
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

### Ingesting an outside recording

A raw `.h5` recorded outside MXtreme (MaxLab's own Record tab, a collaborator's export) joins the
same tree through {func}`mxtreme.store.ingest_recording`. You supply the identity the file cannot —
its batch, chip, DIV — plus a free-form `exp_id` naming the experiment, which becomes the file-name
tail and the registry row's `exp_id`:

```python
from mxtreme.store import ingest_recording

ingest_recording(
    "/path/to/export.raw.h5", config,
    batch="fall2026_batch1_DRG_M1", plate_date=260810,
    chip="M07460", div=21, exp_id="stim_trial_3",
)
```

The file is copied into `.../well_<w>/DIV_<div>/` under the canonical name (pass `move=True` to move
it instead), a multi-well file is split into one file per well, and each well gets an `experiment`
row in `registry.csv`.

## Analysis 

Raw `.h5` file(s) to a PDF report. Each step writes to the managed store, so you can
stop after any of them and pick up later without recomputing.

Before starting, create an `mxtreme.toml` pointing at a directory you can write to — see
[Configuration](concepts/configuration.md).

### 1. Point at the managed store

```python
from mxtreme.config import Config

config = Config.from_toml("mxtreme.toml")
print("managed store:", config.data_root.resolve())
```

### 2. Extract a raw recording

```python
from mxtreme import extract

data = extract.extract("/path/to/recording.raw.h5")
```

`data` is a plain `dict` keyed by well number. Newer Maxwell files carry an embedded
`/assay/metadata` blob that is read automatically; older files don't, so pass one explicitly:

```python
data = extract.extract(
    "/path/to/recording.raw.h5",
    metadata={"Exp ID": "May2025_Wave", "Chip ID": "M07459",
              "Plate date": 250512, "DIV": 0, "Conditions": [[2, 0]]},
)
```

A user-supplied `metadata` dict is merged over the embedded metadata, so you can override or
fill in individual, optional fields without restating all of them.

See {func}`~mxtreme.extact.extract` for the metadata structure. 

### 3. Build and run a cleaning pipeline

A {class}`~mxtreme.pipeline.Pipeline` is an ordered list of functions. Pass a bare function to
accept its defaults, or wrap it in {func}`functools.partial` to change a parameter:

```python
from functools import partial
from mxtreme import clean
from mxtreme.pipeline import Pipeline

pipeline = Pipeline([
    clean.normalize_time,                                       # frames -> start at 0
    clean.dac_to_voltage,                                       # DAC units -> V
    clean.remove_positive_deflections,                          # keep negative spikes
    partial(clean.spike_filter, amp_thresh=2e-5),               # drop below 20 µV
    partial(clean.remove_spurious_spikes, refractory_period=0.002),
    clean.remove_spurious_channels,                             # drop unmapped channels
    clean.build_channel_map,                                    # required before binning
    partial(clean.remove_stim_frames, post_stim_period=0.05),   # drop stim-window artifacts
    partial(clean.bin_spikes, bin_size=0.01),                   # 10 ms bins
])

paths = pipeline.run(data, datastore=config.preprocessed_dir)
```

{meth}`~mxtreme.pipeline.Pipeline.run` transforms and saves **one well at a time**, so at most one
well's `spike_bin` is in memory at once, important for long multi-well recordings. Wells left with
no spikes are skipped rather than saved. Every saved well is recorded and time-stamped in `registry.csv` automatically.

Each parameterized step stamps the values it used into `well['preprocessing_params']`, so the saved
`.npz` carries a record of exactly how it was processed.

### 4. Load a recording

```python
from mxtreme import io
from mxtreme.recording import Recording

rec = Recording(0, io.load_preprocessed(paths[0]))
print(rec.exp_id, rec.chip, rec.well, rec.DIV)
```

### 5. Detect bursts

```python
from mxtreme import io
from mxtreme.bursting import BurstDetector
from mxtreme.params import BurstDetectParams, BurstFeatureParams

bursts = BurstDetector(BurstDetectParams()).detect(rec, burst_data_dir=config.burst_data_dir)

bursts.extract_features(rec, BurstFeatureParams(), burst_data_dir=config.burst_data_dir)

burst_df = bursts.to_dataframe() # Veiw burst data as Pandas dataframe

csv_path = io.save_burst_data(bursts, config.burst_data_dir, rec) # Save burst data as csv

```

Passing `burst_data_dir=` refreshes the per-experiment burst log automatically. Omit it for pure
in-memory use.

The default `"isi_rate"` method composes two stages: ISI-N grouping (Bakkum et al. 2013) to find
candidate burst intervals, then rate thresholding within each group to classify peaks as `"network"`
or `"mini"`. Use `BurstDetector(params, method="isi_n")` for grouping alone.

### 6. Plot

Every plotting helper takes an optional `ax`, so the same call works standalone or inside a figure
you're composing:

Visualize recording:

```python
import matplotlib.pyplot as plt
from mxtreme import visualizations as viz

fig, (ax_asdr, ax_rast) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
viz.plot_asdr(rec.spike_bin, ax=ax_asdr, title="ASDR & raster")
viz.raster(rec.spike_bin, ax=ax_rast)
plt.tight_layout()
```

Visualize burst detection:

```python
fig, (ax_asdr, ax_mea) = plt.subplots(1, 2, figsize=(16, 5))
viz.plot_bursts_on_asdr(rec, burst_df, ax=ax_asdr, title="ASDR with network bursts") # can pass zoom=(start, stop) as an optional parameter to zoom in on the ASDR plot
viz.plot_origin_heatmap(rec, burst_df, ax=ax_mea, title="Network-burst origins")
plt.tight_layout()
plt.show()
```


### 7. Generate a report over multiple DIVs

```python
from mxtreme.identity import CultureID
from mxtreme.analysis import generate_report

pdf = generate_report(
    CultureID(exp_id=[EXP_ID], chip=[CHIP_ID], well=[WELL_ID]),
    config,
    sections=("overview", "activity", "bursting"),
)
```

Add `"stimulation"` and `"performance"` as needed; the stimulation section auto-skips cultures with
no stimulation. To compare cultures, pass a {class}`~mxtreme.identity.CultureSelector` instead of a
single {class}`~mxtreme.identity.CultureID`:

```python
from mxtreme.identity import CultureSelector

pdf = generate_report(CultureSelector(exp_ids=["May2025_Wave"]), config)
```

## Where to next

- [Data flow](concepts/data-flow.md) — what each stage writes and which module owns it.
- [Configuration](concepts/configuration.md) — the managed store and path resolution.
- [API reference](api/index.md) — every public function and class.
- Runnable versions of the above live in `examples/` in the MXtreme repository.
