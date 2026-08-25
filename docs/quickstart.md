# Quickstart

One pass from a raw Maxwell `.h5` to a PDF report. Each step writes to the managed store, so you can
stop after any of them and pick up later without recomputing.

Before starting, create an `mxtreme.toml` pointing at a directory you can write to — see
[Configuration](concepts/configuration.md).

## 1. Point at the managed store

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

`data` is a plain `dict` keyed by well number. Newer Maxwell files carry an embedded
`/assay/metadata` blob that is read automatically; older files don't, so pass one explicitly:

```python
data = extract.extract(
    "/path/to/recording.raw.h5",
    metadata={"Exp ID": "May2025_Wave", "Chip ID": "M07459",
              "Plate date": 250512, "DIV": 0, "Conditions": [[2, 0]]},
)
```

A caller-supplied `metadata` dict is merged *over* the embedded blob per-key, so you can override or
fill in individual fields without restating all of them.

## 3. Build and run a cleaning pipeline

A {class}`~mxtreme.pipeline.Pipeline` is an ordered list of step functions. Pass a bare function to
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
well's `spike_bin` is in memory at once — this matters on long multi-well recordings. Wells left with
no spikes are skipped rather than saved. Every saved well is recorded in `registry.csv` automatically.

Each parameterised step stamps the values it used into `well['preprocessing_params']`, so the saved
`.npz` carries a record of exactly how it was processed.

## 4. Load a recording

```python
from mxtreme import io
from mxtreme.recording import Recording

rec = Recording(0, io.load_preprocessed(paths[0]))
print(rec.exp_id, rec.chip, rec.well, rec.DIV)
```

With no phase information, `rec` has a single `"full"` phase. For a phased experiment, name the
maxlab event tags that mark the boundaries:

```python
rec = Recording(
    0, io.load_preprocessed(paths[0]),
    phase_tags={"pre": "pre_recording_start",
                "train": "closed_loop_start",
                "post": "post_recording_start"},
    end_tag="end_experiment",
)
```

Better still, supply that spec once at extraction time under the `Phases` metadata key and it rides
along inside every `.npz` — every later `Recording` picks it up with no arguments. See
[Data flow](concepts/data-flow.md#phases-ride-along-with-the-data).

## 5. Detect bursts

```python
from mxtreme.bursting import BurstDetector
from mxtreme.params import BurstDetectParams, BurstFeatureParams

bursts = BurstDetector(BurstDetectParams()).detect(rec, burst_data_dir=config.burst_data_dir)
bursts.extract_features(rec, BurstFeatureParams(), burst_data_dir=config.burst_data_dir)

df = bursts.to_dataframe()
```

Passing `burst_data_dir=` refreshes the per-experiment burst log automatically. Omit it for pure
in-memory use.

The default `"isi_rate"` method composes two stages: ISI-N grouping (Bakkum et al. 2013) to find
candidate burst intervals, then rate thresholding within each group to classify peaks as `"network"`
or `"mini"`. Use `BurstDetector(params, method="isi_n")` for grouping alone.

## 6. Plot

Every plotting helper takes an optional `ax`, so the same call works standalone or inside a figure
you're composing:

```python
import matplotlib.pyplot as plt
from mxtreme import visualizations as viz

fig, (ax_asdr, ax_rast) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
viz.plot_asdr(rec.spike_bin, ax=ax_asdr, title="ASDR & raster")
viz.raster(rec.spike_bin, ax=ax_rast)
plt.tight_layout()
```

## 7. Generate a report

Once several DIVs of a culture are preprocessed, assemble them into one PDF:

```python
from mxtreme.identity import CultureID
from mxtreme.analysis import generate_report

pdf = generate_report(
    CultureID(exp_id="May2025_Wave", chip="M07459", well="0"),
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
- Runnable versions of the above live in `examples/` in the repository.
