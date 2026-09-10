# Procedure: dry run on saline

Run the whole pipeline on a chip filled with saline before a culture is committed to it. Nothing
fires, so nothing can be learned, but every pulse leaves an artifact on the electrodes it went
through. That answers the question a dry run is actually asking: **did each stimulus fire where and
when it was meant to?**

The parts that need a culture (choosing patches, calibrating amplitude, measuring connectivity) are
skipped or expected to come back empty. The parts that need hardware (routing, stimulation-unit
assignment, sequence length, DAC timing, artifact size) are exactly what is under test.

Everything below runs from the MXtreme repository root. Substitute your own chip serial for `M07459`.

## Before you start

```bash
uv run python -c "import maxlab; print(maxlab.__file__)"
DRY=/home/mxwbio/mxtreme-data/dryruns/M07459_$(date +%y%m%d)
mkdir -p $DRY
```

**Why not the managed store.** A real experiment goes into the store with `--config mxtreme.toml`,
filed under its plating batch, chip, well and DIV. Saline has none of those: it is not a culture,
has no plating date, and registering it would put a fictitious batch and DIV into `registry.csv`
where later queries would find it. So a dry run writes to an explicit `save_path` *beside* the
store rather than inside it, and is not registered. Set `DRY` to wherever you keep scratch data; every command below uses it.

## 1. Name the three sites

There is no activity to choose from, so give the centres directly, and reuse an electrode set some
earlier run already used on this chip rather than inventing one.

```bash
uv run python -m mxtreme.experiments.associative select \
  --params src/mxtreme/experiments/associative/params_dryrun.json \
  --out runs/dryrun --centers "1000,500;2500,500;1750,1600" \
  --electrodes <any earlier .cfg, .raw.h5 or .npz from this chip>
```

`--electrodes` accepts a MaxLab `.cfg`, a raw recording, or a preprocessed `.npz`. The patches
become whatever part of that set falls within `region_radius_um` of each centre, and the rest fills
the routing budget. Without it the command still runs, but routes every electrode near each centre
plus a synthetic lattice, and says so.

Writes `runs/dryrun/params_M07140_well0.json`. **Every command below points at that file.**

## 2. Look at it before touching the chip

```bash
uv run python -m mxtreme.experiments.associative preview --params runs/dryrun/params_M07140_well0.json
```

Check the three sites, each one's driven block and return ring, the schedule, the pulse waveform,
and the pulse counts per region printed in the summary.

## 3. Calibration mode

Each run writes its recording, `_protocol.json` and `_fired.csv` under one name, so give each run
its own `exp_id`. `save_path` keeps saline data out of the managed store.

```bash
uv run python -m mxtreme.experiments.associative run \
  --params runs/dryrun/params_M07140_well0.json --mode calibration \
  --set exp_id=dry_cal --set save_path=$DRY --set chip=M07459 --set div=1
```

About four minutes: six amplitudes times two polarities times ten repeats, per region. This is the
step most likely to fail, and the one to watch:

- that `maxlab` accepts every sequence the protocol builds;
- that each driven block lands on **distinct** stimulation units. If `init_well_regions` raises
  about two electrodes sharing a unit, raise `inner_gap` in `stim_site` and redo step 1;
- that the amplitude ladder loads and fires.

## 4. Connectivity mode

```bash
uv run python -m mxtreme.experiments.associative run \
  --params runs/dryrun/params_M07140_well0.json --mode connectivity \
  --set exp_id=dry_conn --set save_path=$DRY --set chip=M07459 --set div=1
```

Seven minutes of single pulses interleaved across the three sites.

## 5. The full conditioning schedule

```bash
uv run python -m mxtreme.experiments.associative run \
  --params runs/dryrun/params_M07140_well0.json \
  --set exp_id=dry_cond --set save_path=$DRY --set chip=M07459 --set div=1
```

Six minutes, one of every kind of presentation. The first delivery of the pairing sequence
(`stim_cs_us`, both DACs stepping together).

## 6. Two variants for the paths the first three miss

**Per-pulse unit switching.** With a nonzero offset the pair's units are connected and released
around each pulse instead of held for the whole train. The start event should read `units switched`
and the two deflections should be 20 ms apart.

```bash
uv run python -m mxtreme.experiments.associative run \
  --params runs/dryrun/params_M07140_well0.json --set dt_cs_us=20 \
  --set exp_id=dry_offset --set save_path=$DRY --set chip=M07459 --set div=1
```

**A full-length training sequence.** About 2000 commands in one sequence, which is where a
sequence-length limit in `maxlab` would appear. `encode_iti` has to grow with `t_stim` and the
protocol refuses the run if it does not.

```bash
uv run python -m mxtreme.experiments.associative run \
  --params runs/dryrun/params_M07140_well0.json \
  --set t_stim=600 --set encode_iti=660 --set 'encode_pattern=["PAIR"]' --set encode_cycles=1 \
  --set exp_id=dry_long --set save_path=$DRY --set chip=M07459 --set div=1
```

## 7. Read each of the five recordings back

```bash
uv run python -m mxtreme.experiments.associative report \
  $DRY/<stem>.raw.h5 \
  --protocol $DRY/<stem>_protocol.json \
  -o check_<name>.png --csv check_<name>.csv
```

The report opens by noticing the recording is silent, so the response checks are skipped rather
than reporting three dead sites. What to read instead:

| section | a healthy dry run |
|---|---|
| presentations against the schedule | every token fired the expected number of times, `ok` on each row |
| interval median | matches `probe_iti` / `encode_iti` to within a fraction of a second |
| first deflection, driven electrode | positive first (anodic-first), a few hundred µV |
| first deflection, return electrode | the mirror image, same moment |
| the pair | US and CS deflect in the same frame with `dt_cs_us` 0; 20 ms apart in `dry_offset` |
| artifact vs readout window | spikes in 0-2 ms, near-none in 5-50 ms. If the readout window is not empty, the artifact outlasts it and `pulse_window_ms` needs a later start |

## A note on site size

`preview` prints the stimulation site's footprint next to the radius over which responses are
measured. The default spans 88 um of driven electrodes with the return ring at 150 um, matching
`region_radius_um`, so "stimulate region X" and "measure region X" mean the same patch. The chip's
32 stimulation units are what limit this; see the experiment-day procedure for the options that fit.

## What a dry run cannot tell you

Amplitude. Saline has no cells, so artifact size says nothing about what a culture will do. The
amplitude for a real run comes from a calibration run on that culture.

That is a reason to run calibration mode on saline anyway, not a reason to skip it: it builds and
sends a sequence at every amplitude and both polarities, the widest sweep of the stimulation code
any run makes.

## Checklist

- [ ] `maxlab` imports
- [ ] step 1 wrote a parameter file
- [ ] preview looks right
- [ ] calibration ran without a routing error
- [ ] connectivity ran
- [ ] conditioning ran
- [ ] `dt_cs_us=20` variant ran and its events say `units switched`
- [ ] long-sequence variant ran
- [ ] all five reports show every presentation firing on schedule
- [ ] driven and return deflections have opposite signs at the same moment
- [ ] the 5-50 ms readout window is empty on saline
