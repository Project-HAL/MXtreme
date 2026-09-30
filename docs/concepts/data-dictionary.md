# Data dictionary

Between extraction and analysis, a recording is carried around as a plain Python `dict`, one per
well. This page lists every key in that dict and what it holds, at each of the three places you
meet it:

1. **After extraction**: {func}`mxtreme.extract.extract` returns `dict[int, dict]`, mapping each
   well number to that well's dict.
2. **After cleaning**: each {mod}`mxtreme.clean` step changes a well dict in place and adds keys.
   {meth}`Pipeline.transform <mxtreme.pipeline.Pipeline.transform>` returns the same `dict[int, dict]`.
3. **On disk**: {func}`mxtreme.io.save_preprocessed` writes one well to a `.npz`, and
   {func}`mxtreme.io.load_preprocessed` reads it back as a dict. A few keys are **renamed** or
   **dropped** on the way to disk; see [The saved `.npz`](#the-saved-npz).

You rarely need the `.npz` dict directly. {class}`~mxtreme.recording.Recording` wraps it and exposes
the same data as attributes (see [Recording attributes](#recording-attributes)).

The shapes and dtypes below come from a real one-hour MaxOne recording: 1,016 mapped channels,
20 kHz sampling, with stimulation.

## Units at a glance

| Quantity | Unit |
|---|---|
| Time: `frameno`, `eventtime`, `raw_start`, `stim_frames` | **frames** (samples). Divide by `samp_rate` to get seconds. |
| `samp_rate` | Hz (MaxOne: 20000) |
| `amplitude` | **DAC units** after extraction; **volts** after {func}`~mxtreme.clean.dac_to_voltage` |
| `x`, `y` electrode positions | µm |
| `bin_size`, `rec_t_sec` | seconds |

Frame numbers from extraction are the recorder's absolute frame counter.
{func}`~mxtreme.clean.normalize_time` subtracts `raw_start`, so after cleaning the recording starts
at frame 0.

## After extraction

`extract()` produces these keys for every well:

| Key | Type | Description |
|---|---|---|
| `well` | `int` | Well number (0–5; always 0 on a MaxOne). |
| `exp_id` | `str` | Experiment ID, from metadata `"Exp ID"`. |
| `chip` | `str` | Chip ID, from metadata `"Chip ID"`. |
| `plate_date` | `str` or `int` | Plating date, from metadata `"Plate date"`, e.g. `"111825"`. It is stored exactly as the metadata gives it, so it can be a string or an int. |
| `DIV` | `int` | Days in vitro, from metadata `"DIV"`. |
| `path_to_h5` | `str` | Path of the raw `.h5` this was extracted from. |
| `data` | structured `ndarray`, shape `(n_spikes,)` | The spike table; see [Spike table](#spike-table). Sorted by `frameno`. |
| `samp_rate` | `float64` | Sampling rate in Hz. |
| `mapping` | structured `ndarray`, shape `(n_channels,)` | The recording configuration's channel → electrode map; see [Channel mapping](#channel-mapping-mapping). |
| `lsb` | `ndarray` `float64`, shape `(1,)` | Least significant bit: the volts per DAC unit. |
| `stim_elecs` | `ndarray` `int64`, or `None` | Electrode numbers used for stimulation, from `/assay/stim_elecs`. `None` when the file has none. |
| `raw_start` | `int` | Absolute frame number of the recording's first frame. |
| `event_messages` | `list[dict]` | One parsed JSON message per maxlab event, e.g. `{"pre_recording_start": "0"}` or `{"start_stimulation": ..., "phase_us": ...}`. Values are usually strings. |
| `eventtime` | `ndarray` `int64`, shape `(n_events,)` | Frame number of each event. It lines up with `event_messages` by index. |
| `experimental_condition` | `list`, or `None` | This well's entry from the metadata `"Conditions"` list, e.g. `[2, 0]`. `None` when not supplied. |
| `phase_spec` | `dict`, or `None` | The experiment's phase spec, from metadata `"Phases"`. See [Data flow](data-flow.md#phases-ride-along-with-the-data). |

### Spike table

`data` is a NumPy structured array with one row per detected spike:

| Field | dtype | Description |
|---|---|---|
| `frameno` | `int64` | Frame at which the spike was detected. |
| `channel` | `int32` | Amplifier channel (0–1023) that detected it. Use the channel map to get the electrode. |
| `amplitude` | `float32` | Spike amplitude. It is in DAC units until {func}`~mxtreme.clean.dac_to_voltage`, then in volts. After {func}`~mxtreme.clean.remove_positive_deflections` every amplitude is negative. |

Access a field by name, e.g. `well["data"]["frameno"]`. To get a DataFrame, use
`pd.DataFrame(well["data"])`.

### Channel mapping (`mapping`)

`mapping` is the raw routing table from the recording configuration, one row per routed channel:

| Field | dtype | Description |
|---|---|---|
| `channel` | `int32` | Amplifier channel. |
| `electrode` | `int32` | Electrode number on the array (0–26399 on a MaxOne). |
| `x` | `float64` | Electrode x position, µm. |
| `y` | `float64` | Electrode y position, µm. |

Spikes can occur on channels that are not in `mapping`.
{func}`~mxtreme.clean.remove_spurious_channels` drops them. `mapping` itself is **not saved** to the
`.npz`; {func}`~mxtreme.clean.build_channel_map` turns it into `channelmap`, which is saved.

## After cleaning

The cleaning steps filter rows out of `data` (the dtype stays the same) and change units as
described above. They also add these keys:

| Key | Added by | Type | Description |
|---|---|---|---|
| `channelmap` | {func}`~mxtreme.clean.build_channel_map` | `ndarray` `float64`, shape `(n_channels, 5)` | See [Channel map](#channel-map-channelmap). |
| `stim_frames` | {func}`~mxtreme.clean.remove_stim_frames` | `ndarray` `int64`, or `[]` | Every frame that was blanked for stimulation: each `start_stimulation` → `end_stimulation` interval, plus `post_stim_period` seconds after it. Spikes on these frames were removed. Empty when there is no stimulation. |
| `spike_bin` | {func}`~mxtreme.clean.bin_spikes` | `ndarray` `uint8`, shape `(n_channels, n_bins)` | See [Binned spikes](#binned-spikes-spike-bin). |
| `bin_size` | {func}`~mxtreme.clean.bin_spikes` | `float` | Bin width in seconds (default `0.01`). |
| `rec_t_sec` | {func}`~mxtreme.clean.bin_spikes` | `float64` | Recording length in seconds, measured to the **last spike** (`max(frameno) / samp_rate`). |
| `preprocessing_params` | every parameterized step | `dict` | The parameter values the steps actually used, e.g. `{"amp_thresh": 2e-05, "refractory_period": 0.002, "post_stim_period": 0.05, "bin_size": 0.01}`. |
| `step_log` | {class}`~mxtreme.pipeline.Pipeline` | `list[dict]` | One entry per step, in order: `{"step", "n_before", "n_after", "removed", "seconds"}`. These record spike counts before and after the step, and its run time. |

(channel-map-channelmap)=
### Channel map (`channelmap`)

A plain 2-D `float64` array with one row per mapped channel. Cast the ID columns to `int` before
using them as indices.

| Column | Meaning |
|---|---|
| 0 | **Row index**: which row of `spike_bin` this channel occupies (`0 … n_channels-1`). |
| 1 | Amplifier channel (matches `data["channel"]`). |
| 2 | Electrode number. |
| 3 | x position, µm. |
| 4 | y position, µm. |

For example, to find the electrode and position of `spike_bin` row `i`: `channelmap[i, 2:5]`.

(binned-spikes-spike-bin)=
### Binned spikes (`spike_bin`)

A binary channels × time matrix:

- **Rows** follow the `channelmap` row index (column 0).
- **Column** `j` covers frames `[j·w, (j+1)·w)`, where `w = bin_size × samp_rate` frames. With the
  defaults that is 200 frames, or 10 ms.
- A value is `1` if the channel fired **at least once** in that bin, otherwise `0`. It records
  presence, not a spike count.

The mean over rows, `spike_bin.mean(axis=0)`, is the array-wide spike detection rate. It is
available as `Recording.asdr`.

(the-saved-npz)=
## The saved `.npz`

{func}`~mxtreme.io.save_preprocessed` writes to
`preprocessed/<exp_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<exp_id>_well<N>_exp_data.npz`.
Compared with the in-memory dict:

| In memory | In the `.npz` | Notes |
|---|---|---|
| `data` | **`spike_data`** | Renamed; same structured array. |
| `experimental_condition` | **`exp_condition`** | Renamed. |
| `mapping` | — | Not saved (use `channelmap`). |
| `samp_rate`, `rec_t_sec` | same name | Wrapped as shape-`(1,)` arrays. |
| `well`, `exp_id`, `chip`, `plate_date`, `DIV`, `raw_start`, `path_to_h5`, `bin_size` | same name | 0-d arrays. Use `.item()` to get the Python value. |
| `event_messages`, `step_log` | same name | `object` arrays of dicts. |
| `preprocessing_params`, `phase_spec` | same name | 0-d `object` arrays. Use `.item()` to get the dict (or `None`). |
| `channelmap`, `spike_bin`, `bin_size`, `rec_t_sec`, `stim_frames` | same name | These come from optional steps. If a step didn't run, its key is saved with an empty default: `channelmap` → shape `(0, 5)`, `spike_bin` → shape `(0, 0)`, `bin_size` / `rec_t_sec` → `NaN`, `stim_frames` → empty. |

Object arrays need pickle support, so {func}`~mxtreme.io.load_preprocessed` loads with
`allow_pickle=True`. Only load `.npz` files that you trust.

(recording-attributes)=
## Recording attributes

{class}`~mxtreme.recording.Recording` unwraps the `.npz` dict. Scalars become Python values and
shape-`(1,)` arrays become `float`s:

| Attribute | From `.npz` key |
|---|---|
| `exp_id`, `chip`, `well`, `DIV`, `plate_date`, `raw_start`, `path_to_h5` | same name |
| `exp_condition` | `exp_condition` (`None` if absent) |
| `spike_data`, `channelmap`, `spike_bin`, `stim_elecs`, `eventtime`, `event_messages` | same name |
| `stim_frames` | `stim_frames` (`None` if absent) |
| `samp_rate`, `lsb`, `rec_t_sec`, `bin_size` | same name, as `float` |
| `asdr` | derived: `spike_bin.mean(axis=0)` |
| `event_df` | derived: a DataFrame with columns `eventtime`, `eventmessage` |
| `phases` | derived from `phase_spec` (see {mod}`mxtreme.phases`) |
