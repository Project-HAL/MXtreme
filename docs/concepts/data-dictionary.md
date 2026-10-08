# Data Structures

Between extraction and analysis, a recording is carried around as a plain Python `dict`. 
This page lists every key in that dict and what it holds, at each of the three places you
meet it:

1. **After extraction**: {func}`mxtreme.extract.extract` returns `dict[int, dict]`, mapping each
   well number to that well's dict.
2. **After cleaning**: each {mod}`mxtreme.clean` step changes a well dict in place and adds keys.
   {meth}`Pipeline.transform <mxtreme.pipeline.Pipeline.transform>` returns the same `dict[int, dict]`.
3. **On disk**: {func}`mxtreme.io.save_preprocessed` writes one well to a `.npz` under the **same
   key names**, and {func}`mxtreme.io.load_preprocessed` reads it back as a dict; see
   [The saved `.npz`](#the-saved-npz).

{class}`~mxtreme.recording.Recording` wraps the dictionary stored in `.npz` and exposes
the same data as attributes (see [Recording attributes](#recording-attributes)).

## Units at a glance

| Quantity | Unit |
|---|---|
| Time: `frameno`, `eventtime`, `raw_start`, `stim_frames` | **frames** (samples). Divide by `samp_rate` to get seconds. |
| `samp_rate` | Hz (MaxOne: 20k, MaxTwo: 10k) |
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
| `batch_id` | `str` | Plating batch, from metadata `"Batch ID"`. |
| `experiment` | `str` | An optional label describing the type of recording, from metadata `"Experiment"`; `""` for plain scans. |
| `chip` | `str` | Chip ID, from metadata `"Chip ID"`. |
| `plate_date` | `str` or `int` | Plating date, from metadata `"Plate date"`, e.g. `"111825"`. It is stored exactly as the metadata gives it, so it can be a string or an int. |
| `DIV` | `int` | Days in vitro, from metadata `"DIV"`. |
| `path_to_h5` | `str` | Path of the raw `.h5` this was extracted from. |
| `spike_data` | structured `ndarray`, shape `(n_spikes,)` | The spike table; see [Spike table](#spike-table). Sorted by `frameno`. |
| `samp_rate` | `float64` | Sampling rate in Hz. |
| `mapping` | structured `ndarray`, shape `(n_channels,)` | The recording configuration's channel → electrode map; see [Channel mapping](#channel-mapping-mapping). |
| `lsb` | `ndarray` `float64`, shape `(1,)` | Least significant bit: the volts per DAC unit. |
| `stim_elecs` | `ndarray` `int64`, or `None` | Electrode numbers used for stimulation, from `/assay/stim_elecs`. `None` when the file has none. |
| `raw_start` | `int` | Absolute frame number of the recording's first frame. |
| `event_messages` | `list[dict]` | One parsed JSON message per maxlab event, e.g. `{"pre_recording_start": "0"}` or `{"start_stimulation": ..., "phase_us": ...}`. Values are usually strings. |
| `eventtime` | `ndarray` `int64`, shape `(n_events,)` | Frame number of each event. It lines up with `event_messages` by index. |
| `exp_condition` | `list`, or `None` | This well's entry from the metadata `"Conditions"` list, e.g. `[2, 0]`. `None` when not supplied. |
| `phase_spec` | `dict`, or `None` | The experiment's phase spec, from metadata `"Phases"`. See [Data flow](data-flow.md#phases-ride-along-with-the-data). |

### Spike table

`spike_data` is a NumPy structured array with one row per detected spike:

| Field | dtype | Description |
|---|---|---|
| `frameno` | `int64` | Frame at which the spike was detected. |
| `channel` | `int32` | Amplifier channel (0–1023) that detected it. Use the channel map to get the electrode. |
| `amplitude` | `float32` | Spike amplitude. It is in DAC units until {func}`~mxtreme.clean.dac_to_voltage`, then in volts. After {func}`~mxtreme.clean.remove_positive_deflections` every amplitude is negative. |

Access a field by name, e.g. `well["spike_data"]["frameno"]`. To get a DataFrame, use
`pd.DataFrame(well["spike_data"])`.

### Channel mapping (`mapping`)

`mapping` is the raw routing table from the recording configuration, one row per routed channel:

| Field | dtype | Description |
|---|---|---|
| `channel` | `int32` | Amplifier channel. |
| `electrode` | `int32` | Electrode number on the array (0–26399 on a MaxOne). |
| `x` | `float64` | Electrode x position, µm. |
| `y` | `float64` | Electrode y position, µm. |

Spikes can occur on channels that are not in `mapping`.
{func}`~mxtreme.clean.remove_spurious_channels` drops them. {func}`~mxtreme.clean.build_channel_map`
turns `mapping` into `channelmap`. Both are saved to the `.npz`.

## After cleaning

The cleaning steps filter rows out of `spike_data` (the dtype stays the same) and change units as
described above. They also add these keys:

| Key | Added by | Type | Description |
|---|---|---|---|
| `channelmap` | {func}`~mxtreme.clean.build_channel_map` | `ndarray` `float64`, shape `(n_channels, 5)` | See [Channel map](#channel-map-channelmap). |
| `stim_frames` | {func}`~mxtreme.clean.remove_stim_frames` | `ndarray` `int64`, or `[]` | Every frame that was blanked for stimulation: each `start_stimulation` → `end_stimulation` interval, plus `post_stim_period` seconds after it. Spikes on these frames were removed. Empty when there is no stimulation. |
| `spike_bin` | {func}`~mxtreme.clean.bin_spikes` | `ndarray` `uint8`, shape `(n_channels, n_bins)` | See [Binned spikes](#binned-spikes-spike-bin). |
| `bin_size` | {func}`~mxtreme.clean.bin_spikes` | `float` | Bin width in seconds (default `0.01`). |
| `rec_t_sec` | {func}`~mxtreme.clean.bin_spikes` | `float64` | Recording length in seconds, measured to the **last spike** (`max(frameno) / samp_rate`). |
| `preprocessing_params` | every parameterized step | `dict` | The parameter values the steps actually used, e.g. `{"amp_thresh": 2e-05, "refractory_period": 0.002, "post_stim_period": 0.05, "bin_size": 0.01}`. Also `time_normalized: True` (set by {func}`~mxtreme.clean.normalize_time`) and `amplitude_units: "V"` (set by {func}`~mxtreme.clean.dac_to_voltage`). Those two steps are skipped when their flag is already set, so they never apply twice. |
| `step_log` | {class}`~mxtreme.pipeline.Pipeline` | `list[dict]` | One entry per step, in order: `{"step", "n_before", "n_after", "removed", "seconds"}`. These record spike counts before and after the step, and its run time. Each pipeline pass appends to it, so reprocessed data keeps the history of every pass. |

(channel-map-channelmap)=
### Channel map (`channelmap`)

A plain 2-D `float64` array with one row per mapped channel. Cast the ID columns to `int` before
using them as indices.

| Column | Meaning |
|---|---|
| 0 | **Row index**: which row of `spike_bin` this channel occupies (`0 … n_channels-1`). |
| 1 | Amplifier channel (matches `spike_data["channel"]`). |
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
`preprocessed/<batch_id>/<chip>/well<N>/DIV<d>_<plate_date>_<chip>_<batch_id>_well<N>[_<experiment>]_exp_data.npz`.
A file saved before `batch_id` was the identity carries an `exp_id` key instead; `load_preprocessed`
returns it as `batch_id`.
Every key is saved under its in-memory name. A `.npz` can only hold arrays, so single values are
stored wrapped, and {func}`~mxtreme.io.load_preprocessed` unwraps them again. **A loaded file has the
same form as a cleaned well dict**, so you can pass it to {class}`~mxtreme.recording.Recording` or
straight back to a {mod}`mxtreme.clean` step, e.g. `clean.bin_spikes(io.load_preprocessed(path), bin_size=0.05)`.

| Key | Stored in the `.npz` as | `load_preprocessed` returns |
|---|---|---|
| `spike_data`, `mapping`, `eventtime`, `lsb`, `stim_elecs` | arrays | the same arrays. Files written before `mapping` was saved lack it. |
| `samp_rate`, `rec_t_sec` | shape-`(1,)` arrays | `float` |
| `well`, `batch_id`, `experiment`, `chip`, `plate_date`, `DIV`, `raw_start`, `path_to_h5`, `bin_size` | 0-d arrays | Python values (`int`, `str`, `float`) |
| `exp_condition` | `int` array, or 0-d `object` array for `None` | `list`, or `None` |
| `event_messages`, `step_log` | `object` arrays of dicts | `list[dict]` |
| `preprocessing_params`, `phase_spec` | 0-d `object` arrays | `dict` (or `None` for `phase_spec`) |

`channelmap`, `spike_bin`, `bin_size`, `rec_t_sec` and `stim_frames` come from optional steps. If a
step didn't run, its key is saved with an empty default: `channelmap` → shape `(0, 5)`, `spike_bin` →
shape `(0, 0)`, `bin_size` / `rec_t_sec` → `NaN`, `stim_frames` → empty.

Object arrays need pickle support, so {func}`~mxtreme.io.load_preprocessed` loads with
`allow_pickle=True`. Only load `.npz` files that you trust.

(recording-attributes)=
## Recording attributes

{class}`~mxtreme.recording.Recording` exposes a loaded `.npz` dict as attributes. It also accepts an
in-memory well dict straight from {meth}`Pipeline.transform <mxtreme.pipeline.Pipeline.transform>`,
since the two have the same form:

| Attribute | From `.npz` key |
|---|---|
| `batch_id`, `experiment`, `chip`, `well`, `DIV`, `plate_date`, `raw_start`, `path_to_h5` | same name |
| `exp_condition` | `exp_condition` (`None` if absent) |
| `spike_data`, `channelmap`, `spike_bin`, `stim_elecs`, `eventtime`, `event_messages` | same name |
| `stim_frames` | `stim_frames` (`None` if absent) |
| `samp_rate`, `lsb`, `rec_t_sec`, `bin_size` | same name, as `float` |
| `asdr` | derived: `spike_bin.mean(axis=0)` |
| `event_df` | derived: a DataFrame with columns `eventtime`, `eventmessage` |
| `phases` | derived from `phase_spec` (see {mod}`mxtreme.phases`) |
