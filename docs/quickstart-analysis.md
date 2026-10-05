
# Quickstart - Analysis 

This quickstart guide covers how to go from raw `.h5` files to a PDF report. Each step of the pipeline writes to a managed data store so you can stop after step and pick up later without recomputing.

Before starting, create an `mxtreme.toml` pointing at a directory you can write to. A simple `mxtreme.toml` file looks something like:

```toml
# mxtreme.toml
# Edit `root` to a location you can write to.

[data]
root = "~/mxtreme-data/"
```
See [Configuration](concepts/configuration.md) for more details.

## 1. Configure the managed store for data organization

```python
from mxtreme.config import Config

config = Config.from_toml("mxtreme.toml")
print("managed store:", config.data_root.resolve())
```

## 2. Extract a raw recording

```python
from mxtreme import extract

data = extract.extract("/path/to/recording.raw.h5")
```

`data` is a plain `dict` keyed by well number. Raw .h5 files generated from MXtreme activity and network scans carry an embedded `/assay/metadata` blob that is read automatically. Raw .h5 files generated from the Maxlab Live Scope do not, so one must be passed explicitly:

```python
data = extract.extract(
    "/path/to/recording.raw.h5",
    metadata={"Batch ID": "fall2026_batch1_DRG", "Chip ID": "M01234",
              "Plate date": 250101, "DIV": 0}
    )
```

A user-supplied `metadata` dict is merged over the embedded metadata, so you can override or
fill in individual, optional fields without restating all of them.

See {func}`~mxtreme.extact.extract` for the metadata structure. 

## 3. Build and run a cleaning pipeline

A {class}`~mxtreme.pipeline.Pipeline` is an ordered list of functions. Passing bare functions accepts their defaults. Wrapping the functions in {func}`functools.partial` allows user-defined parameters to be passed:

```python
from functools import partial
from mxtreme import clean
from mxtreme.pipeline import Pipeline

pipeline = Pipeline([
    clean.normalize_time,                                       # frames -> start at 0
    clean.dac_to_voltage,                                       # DAC units -> V
    clean.remove_positive_deflections,                          # keep only negative spikes
    partial(clean.spike_filter, amp_thresh=2e-5),               # drop spikes below 20 µV
    partial(clean.remove_spurious_spikes, refractory_period=0.002), # remove spikes within 2 ms of another spike on the same channel
    clean.remove_spurious_channels,                             # drop unmapped channels
    clean.build_channel_map,                                    # required before binning
    partial(clean.remove_stim_frames, post_stim_period=0.05),   # drop stimulation frames and frames post_stim_period seconds after the stimulation to remove aftifacts due to stimulation 
    partial(clean.bin_spikes, bin_size=0.01),                   # 10 ms bins
])

paths = pipeline.run(data, datastore=config.preprocessed_dir)
```
Executing the pipeline with {meth}`~mxtreme.pipeline.Pipeline.run` transforms the data *AND* saves one well's data at a time (important for multiwell recordings). Wells with no spikes are skipped rather than saved. Every saved well is recorded and time-stamped in `registry.csv` automatically. 

Each parameterized step stamps the values it used into `well['preprocessing_params']`, so the saved
`.npz` carries a record of exactly how it was processed.

```python
paths = pipeline.run(data, datastore=config.preprocessed_dir)
```

If working with the data in a notebook where saving the preprocessed data isn't necessary, it can be transformed without saving by using the {meth}`~mxtreme.pipeline.Pipeline.transform` function. 

```python
data_clean = pipeline.transform(data)
```

See [Data dictionary](concepts/data-dictionary.md) for the structure of the data dictionary: every key, and the layout of the spike table, channel map, and binned spike matrix. 

## 4. Load a recording

Load preprocessed data and create a Recording object to easily access data. See {class}`~mxtreme.recording.Recording` for more information.

```python
from mxtreme import io
from mxtreme.recording import Recording

rec = Recording(0, io.load_preprocessed(paths[0]))
print(rec.batch_id, rec.chip, rec.well, rec.DIV)
```

## 5. Detect bursts

```python
from mxtreme import io
from mxtreme.bursting import BurstDetector
from mxtreme.params import BurstDetectParams, BurstFeatureParams

# Define parameters for burst detection
detection_params = BurstDetectParams()  

# Initialize the burst detector
burst_detector = BurstDetector(params=detection_params, method="isi_rate").detect(rec, burst_data_dir=config.burst_data_dir)

# Detect bursts
bursts = burst_detector.detect(rec, burst_data_dir=config.burst_data_dir)

# Define burst feature extraction parameters
feature_params = BurstFeatureParams()

# Extract burst features
bursts.extract_features(rec, params=feature_params, burst_data_dir=config.burst_data_dir)

# Veiw burst data as Pandas dataframe
burst_df = bursts.to_dataframe() 

# Save burst data to file
csv_path = io.save_burst_data(bursts, config.burst_data_dir, rec) # Save burst data as csv

```

Passing `burst_data_dir=` refreshes the per-batch burst log automatically. Omit it for pure
in-memory use.

The default `"isi_rate"` method composes two stages: ISI-N grouping (Bakkum et al. 2013) to find
candidate burst intervals, then rate thresholding within each group to classify peaks as `"network"`
or `"mini"`. Use `BurstDetector(params, method="isi_n")` for grouping alone.

## 6. Plot

Every plotting helper takes an optional `ax`, so the same function works standalone or inside a figure:

Visualize a recording's activity:

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


## 7. Generate a report over multiple DIVs

```python
from mxtreme.identity import CultureID
from mxtreme.analysis import generate_report


culture = CultureID(batch_id=[BATCH_ID], chip=[CHIP_ID], well=[WELL_ID])

pdf = generate_report(
    culture,
    config,
    sections=("overview", "activity", "bursting"),
)
```

Add `"stimulation"` and `"performance"` as needed; the stimulation section auto-skips cultures with
no stimulation. To compare cultures, pass a {class}`~mxtreme.identity.CultureSelector` instead of a
single {class}`~mxtreme.identity.CultureID`:

```python
from mxtreme.identity import CultureSelector

selection = CultureSelector(batch_ids=["fall2026_batch1_DRG"])

pdf = generate_report(selection, config)
```

# Where to next

- [Data flow](concepts/data-flow.md) — what each stage writes and which module owns it.
- [Configuration](concepts/configuration.md) — the managed store and path resolution.
- [API reference](api/index.md) — every public function and class.
- Runnable versions of the above live in `examples/` in the MXtreme repository.
