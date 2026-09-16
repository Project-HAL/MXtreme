# Procedure: dry run on saline

The same path as the [experiment day](associative-experiment-day.md), on a chip filled with saline.
Nothing fires, so nothing is learned, but every pulse leaves an artifact on the electrodes it went
through, so the recordings answer what a dry run is asking: **did each stimulus fire where and
when it was meant to?** The steps that need a culture (choosing regions, calibrating amplitude,
measuring connectivity) run and come back empty; the steps that need hardware (routing,
stimulation-unit assignment, sequence length, DAC timing, artifact shape) are under test.

## How it differs from the real day

| | experiment day | dry run |
|---|---|---|
| where it writes | the managed store, registered | `<store root>/dryruns/<chip>_<date>/`, beside the store, not registered |
| batch id | the culture's | `fall2026_batch1_saline_M1` (or `_M2`), which says what it is |
| scans | braintrix-cli, into the store | braintrix-cli, with its **Save directory** set to the dry-run directory |
| baseline | network scan of the culture's active set | network scan of some real culture's active set, recorded on the saline chip |
| regions | chosen from the baseline | placed by hand with `--centers` |
| raw traces | none (spikes only) | the three regions' electrodes, for the artifact check |

Saline has no plating batch or DIV, so registering it would put a fictitious culture into
`registry.csv`. The store's own walks only look inside `recordings/`, so a `dryruns/` directory
beside it is never mistaken for data.

## 0. Set up, on the rig

```bash
cd <path to>/braintrix-cli
source .venv/bin/activate
PKG=../MXtreme/src/mxtreme/experiments/associative
grep root ~/.config/mxtreme/mxtreme.toml      # or the file braintrix-cli's header names
```

```bash
BATCH=fall2026_batch1_saline_M1     # M1 on a MaxOne, M2 on a MaxTwo: run checks it against the device
CHIP=M07459
PLATE=$(date +%y%m%d)               # saline has no plating date; today's keeps every file's name consistent
STORE=<root from the grep above>
DRY=$STORE/dryruns/${CHIP}_$(date +%y%m%d)
mkdir -p $DRY
```

## 1. Activity scan, in braintrix-cli

```bash
python run_cli.py
```

Skip loading a batch. **Record on device → Activity scan**, pick **Kam Scan**, enter the saline
identity (batch `$BATCH`, chip `$CHIP`, plate date `$PLATE`, DIV 0, well 0), and set **Save
directory** to the `$DRY` path. This checks the scan path end to end. On saline it finds no active
electrodes, which is expected.

## 2. Baseline, in braintrix-cli

braintrix-cli's network scan selects electrodes from an activity scan, and the saline scan has
none, so it would stop with "no network scan to run". Point it at a real culture's activity scan
instead, so it records that culture's active set on the saline chip:

**Record on device → Network scan**, choose to point at an `.h5`, give it any recent real
`..._activity_scan.raw.h5`, then in the parameter editor set the identity back to the saline one
and **Save directory** to `$DRY`. Recording length `300` s, or `60` to save time.

If that is awkward, skip this step and pass `--electrodes <that activity scan or any .cfg>` to
`select` in step 3 instead of `--baseline`; everything after works the same.

```bash
ls $DRY
```

## 3. Place the three regions by hand

```bash
python -m mxtreme.experiments.associative select \
  --params $PKG/params_dryrun.json --out $DRY \
  --baseline $DRY/*_network_scan.raw.h5 \
  --centers "1000,500;2500,500;1750,1600" \
  --set batch=$BATCH --set chip=$CHIP --set plate_date=$PLATE --set div=0 --set save_path=$DRY
P=$(ls $DRY/*_params.json)
```

A silent baseline has nothing to rank or correlate, so `select` says so, skips coupling, says it
has no per-electrode activity to place the driven blocks on (so they sit at the centres as given),
and takes the routing from the electrodes the baseline recorded. `save_path` is stored in `$P`, so
every run below writes to `$DRY` and is not registered. `params_dryrun.json` also carries
`amplitudes_source: "saline"`: on a culture a connectivity or conditioning run refuses to start
until calibration's numbers have been copied in, and saline has nothing to calibrate. `params_dryrun.json` keeps raw traces on
the three regions (`raw_traces: "regions"`), which the artifact check in step 7 needs.

## 4. Look before touching the chip

```bash
python -m mxtreme.experiments.associative preview --params $P
```

It should say the recording goes to `$DRY`, "not registered", raw traces on the three regions,
and draw the sites (each driven block centred in its return ring), the schedule and the waveform.

## 5. The three runs

```bash
python -m mxtreme.experiments.associative run --params $P --mode calibration --set exp_id=dry_cal --work $DRY
python -m mxtreme.experiments.associative run --params $P --mode connectivity --set exp_id=dry_conn --work $DRY
python -m mxtreme.experiments.associative run --params $P --set exp_id=dry_cond --work $DRY
```

About 6, 7 and 6 minutes. Then the same conditioning schedule the way the real day runs it: one
command, one recording per phase — a baseline, one per encode cycle, a retrieval — with the rig
set up once and the files closed one after another, and the session read as one at the end:

```bash
python -m mxtreme.experiments.associative run --params $P --set exp_id=dry_split --phase session --work $DRY
python -m mxtreme.experiments.associative report $DRY/*_dry_split_*.raw.h5
```

About 7 minutes for the four files. While it runs, from a second terminal, read the baseline file
as soon as it exists — the mid-session read the real day uses:

```bash
python -m mxtreme.experiments.associative report $DRY/*_dry_split_baseline.raw.h5
```

It should end with "no retrieval block yet". The session `report` afterwards should list the four
recordings in order, count the same presentations as `dry_cond` did, and lay them out on one
timeline with the seconds of file handling between them.

Calibration is the step most likely to fail, so watch it:

- `run` prints the connected device and refuses if it does not match the batch's `M1`/`M2`;
- `maxlab` has to accept every sequence;
- each driven block has to land on **distinct** stimulation units. If `init_well_regions` raises
  about two electrodes sharing a unit, raise `inner_gap` in `$P` and re-run;
- `run` reports whether the protocol went into the recording. If it says it did not, the copy in
  `$DRY` is what `report` needs (`--protocol`).

## 6. Two variants for the paths the first three miss

```bash
python -m mxtreme.experiments.associative run --params $P --set exp_id=dry_offset --set dt_cs_us=20 --work $DRY
python -m mxtreme.experiments.associative run --params $P --set exp_id=dry_long --work $DRY \
  --set t_stim=600 --set encode_iti=660 --set 'encode_pattern=["PAIR"]' --set encode_cycles=1
python -m mxtreme.experiments.associative run --params $P --mode calibration --set exp_id=dry_cal_grid --work $DRY \
  --set 'stim_site={"shape":"grid","size":3,"gap":2}'
python -m mxtreme.experiments.associative compare $DRY/*_dry_cal.raw.h5 $DRY/*_dry_cal_grid.raw.h5
```

`dry_cal_grid` is the other stimulation pattern the real day may calibrate (a 3x3 grid, 27
stimulation units, three DACs and no return ring), so routing and sequence building for it get
exercised too; `compare` on two silent recordings should say there is nothing to compare, which
is that path working.

`dry_offset` exercises per-pulse unit switching: its start events should read `units switched`,
and the two regions' deflections should be 20 ms apart. `dry_long` builds a sequence of about 2000
commands, where a sequence-length limit would show: 198 pulse events over ten minutes, more than
twice the real run's 89-event trains, at the same amplitude as any other. It tests length, not
dose, and bounds what the real run will ask of the sequencer.

## 7. Read each recording back

For each of `dry_cal`, `dry_conn`, `dry_cond`, `dry_offset`, `dry_long`, `dry_cal_grid`, and the
four `dry_split_*` files together:

```bash
python -m mxtreme.experiments.associative report $DRY/*_dry_cal.raw.h5 -o $DRY/dry_cal.png --csv $DRY/dry_cal.csv
python -m mxtreme.experiments.associative report $DRY/*_dry_split_*.raw.h5 -o $DRY/dry_split.png --csv $DRY/dry_split.csv
```

No `--protocol`: `report` reads it from inside the recording, which is itself a check that
embedding worked. It notices the recording is silent and skips the response verdicts. Read
instead:

| section | a healthy dry run |
|---|---|
| presentations against the schedule | every token fired the expected number of times, `ok` on each row |
| interval median | matches `probe_iti` / `encode_iti` to within a fraction of a second |
| first deflection, driven electrode | positive first (anodic-first), a few hundred µV |
| first deflection, return electrode | the mirror image, same moment |
| the pair | US and CS deflect in the same frame in `dry_cond`; 20 ms apart in `dry_offset` |
| artifact vs readout window | spikes in 0-2 ms, near-none in 5-50 ms. If not, the artifact outlasts the window and `pulse_window_ms` needs a later start |

The raw traces need MaxWell's HDF5 compression filter, which MaxLab installs on the rig; elsewhere
the artifact section says it cannot read them.

## What a dry run cannot tell you

Amplitude. Saline has no cells, so artifact size says nothing about what a culture will do; that
comes from calibration on the culture. Calibration on saline is still worth running, since it
builds and sends a sequence at every amplitude and both polarities, the widest sweep of the
stimulation code any run makes.

## Checklist

- [ ] braintrix-cli activity scan wrote into `$DRY`, not the store
- [ ] braintrix-cli network scan wrote into `$DRY` (or step 3 used `--electrodes`)
- [ ] `select` wrote `$P` and said the baseline is silent
- [ ] `preview` says `$DRY`, not registered, raw traces on the regions, rings centred
- [ ] calibration printed the right device and ran without a routing error
- [ ] connectivity and conditioning ran
- [ ] `--phase session` wrote four `dry_split_*` recordings from one command; the baseline one reported while the rest ran; one `report` over all four read them as one session, in order
- [ ] `dry_offset` events say `units switched`; `dry_long` ran
- [ ] `dry_cal_grid` routed 27 stimulation electrodes and ran; `compare` said nothing to compare
- [ ] every `report` ran with no `--protocol`
- [ ] every presentation fired on schedule
- [ ] driven and return deflections have opposite signs at the same moment
- [ ] the 5-50 ms readout window is empty
- [ ] nothing new under `recordings/` or in `registry.csv`
