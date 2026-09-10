# Procedure: associative conditioning on a culture

The day's plan for one culture, from an activity scan to the readout. Roughly six hours end to end,
of which the conditioning run is four. Do the [dry run on saline](associative-dry-run.md) first;
this document assumes the hardware path is already known to work.

Run everything from the MXtreme repository root. Substitute your chip serial, batch and DIV
throughout. `RUN=/home/mxwbio/assoc/<chip>_<date>` is used below as the run directory.

## Where the output goes

Real experiments belong in the **managed store**, and `--config mxtreme.toml` puts them there
automatically: the recording lands in
`<data_root>/recordings/plating_<date>_<batch>/chip_<M1|M2>_<chip>/well_<N>/DIV_<d>/` and a row is
added to `registry.csv`. That is the convention every other MXtreme output follows, and it is what
makes a culture's whole history sit in one directory. Do not pass `save_path` for a real run.

## The shape of the day

| step | what | how long | what it decides |
|---|---|---|---|
| 1 | activity scan | ~8 min | where the culture is firing |
| 2 | choose candidates | seconds | four to six patches to consider |
| 3 | baseline recording | 5 min | how coupled those patches already are |
| 4 | roles | seconds | which patch is US, CS, NS |
| 5 | calibration | ~20 min | each region's amplitude |
| 6 | connectivity check | ~10 min | are the sites independent, and connected at all |
| 7 | conditioning | 220 min | the experiment |
| 8 | readout | minutes | did it learn |
| 9 | follow-up, later days | ~45 min each | did it last |

Steps 2 and 4 are free, and steps 5 and 6 are gates: each can send you back to step 2 with a clear
reason. Budget for that rather than being surprised by it.

## 0. Set up

```bash
mkdir -p $RUN
uv run python -c "import maxlab; print(maxlab.__file__)"
```

Copy the shipped defaults so the day's parameters are a file you can edit and keep:

```bash
cp src/mxtreme/experiments/associative/params_default.json $RUN/params.json
```

Edit `batch`, `chip`, `plate_date`, `div` and `well` in it. Leave `regions`, `rec_electrodes` and
`amplitudes_mv` alone; steps 2 to 5 fill them in.

## 1-4. Scan, baseline, candidates, roles

One command does all four. It runs the activity scan, picks candidate patches, records the
baseline with those patches routed, measures coupling and assigns roles.

```bash
uv run python -m mxtreme.experiments.associative select \
  --params $RUN/params.json --out $RUN \
  --scan-with mxtreme.toml --record-baseline mxtreme.toml --candidates 6
```

To reuse a scan you already have, swap `--scan-with mxtreme.toml` for
`--activity-scan <file>.raw.h5`. To reuse a baseline, swap `--record-baseline` for
`--baseline <file>.raw.h5`. The baseline must be **one continuous recording** with the candidates
routed; a multi-round scan is refused, because electrodes from different rounds were never observed
at the same moment and their correlation would mean nothing.

**Read before continuing.** Two figures land in `$RUN`:

- `regions_<chip>_well<N>.png`: the scan with every candidate patch, why each one lost, the chosen
  triangle with its side lengths, and the coupling matrix.
- `coupling_<chip>_well<N>.png`: the evidence behind the coupling numbers, five panels. Patch rates
  with bursts shaded; correlation with all bins next to correlation with bursts masked; the
  burst-order matrix; a cross-correlogram per pair; the burst-initiation raster; coupling against
  distance.

What you are checking:

| look at | good | bad |
|---|---|---|
| correlation, bursts masked | below ~0.1 for the three chosen | any chosen pair above ~0.3 |
| burst order | near 50% for the chosen pairs | one patch first in most bursts (it drives the others) |
| triangle | ratio above 0.6, US-CS and US-NS within ~15% | one region between the other two |
| coupling vs distance | no clear trend | rising with distance (the measure is reading geometry) |

If the selection refuses (`only N patches qualify`), it writes a rejection figure showing why.
Lower `--separation`, or `region_radius_um` in the parameter file, or name patches by hand with
`--centers "x,y;x,y;x,y"`.

Step 1-4 writes `$RUN/params_<chip>_well<N>.json`. **Every command below points at that file.**

## 5. Calibration

```bash
uv run python -m mxtreme.experiments.associative run \
  --params $RUN/params_<chip>_well<N>.json --mode calibration \
  --set exp_id=cal --config mxtreme.toml
```

Six amplitudes times two polarities times ten repeats per region, shuffled. Then:

```bash
uv run python -m mxtreme.experiments.associative report \
  <the cal .h5> --protocol <the cal _protocol.json> --csv $RUN/cal.csv
```

The report prints a table of region by amplitude by polarity with spikes per pulse and burst rate,
then a recommended `amplitudes_mv`. The rule it applies is the largest amplitude that evokes at
least 0.5 spikes per pulse locally while starting a network burst on at most 20% of pulses.

Copy the recommended `amplitudes_mv` into `$RUN/params_<chip>_well<N>.json`. If it reports `STOP`
instead, the message says which of the two failure modes it is: nothing responds even at the top of
the ladder (extend it, or the site has too few neurons and you go back to step 2), or everything
that responds also bursts (shorter phases, a sparser site, or back to step 2).

**From here the regions, the routed config and the amplitudes are frozen for this culture.**
`_protocol.json` beside every recording records exactly what was used.

## 6. Connectivity check

```bash
uv run python -m mxtreme.experiments.associative run \
  --params $RUN/params_<chip>_well<N>.json --mode connectivity \
  --set exp_id=conn --config mxtreme.toml
```

Five minutes of single pulses at the calibrated amplitudes, interleaved across the three sites.

```bash
uv run python -m mxtreme.experiments.associative report \
  <the conn .h5> --protocol <the conn _protocol.json> --csv $RUN/conn.csv
```

This prints the cross-talk matrix: evoked spikes per pulse in each region when each region is
stimulated, then the same as a fraction of the stimulated region's own response. Four verdicts:

| verdict | meaning | what to do |
|---|---|---|
| a site evokes almost nothing locally | amplitude or site is wrong | back to step 5, then step 2 |
| a site's pulses start bursts >20% | it is driving the whole culture | lower its amplitude |
| a site drives another >30% of its own response | the two are not independent | back to step 2 with a larger separation, or accept a high floor |
| no response either way between CS and US | there may be no path to strengthen | back to step 2; a null result here would say nothing |

This is the last cheap moment to change your mind. Everything after it costs four hours.

## 7. Conditioning

```bash
uv run python -m mxtreme.experiments.associative preview \
  --params $RUN/params_<chip>_well<N>.json -o $RUN/plan.png
```

Confirm the schedule and the pulse budget printed in the summary, then:

```bash
uv run python -m mxtreme.experiments.associative run \
  --params $RUN/params_<chip>_well<N>.json --set exp_id=cond --config mxtreme.toml
```

220 minutes with the shipped defaults: 10 min quiet, baseline probes, three cycles of pairing,
pairing, NS with rests between, a decay period, then three retrieval blocks 15 minutes apart, then
10 min quiet. Around 1400 pulses each to US and CS.

The terminal prints each block as it starts and each presentation as it fires, with its offset from
plan. `_fired.csv` records the same thing. MaxLab Live shows the culture live; nothing in this
package needs watching.

To shorten the day, cut `encode_cycles` or a retrieval block, not `t_stim`:

```bash
  --set encode_cycles=2 --set num_retrievals=2
```

## 8. Readout

```bash
uv run python -m mxtreme.experiments.associative report \
  <the cond .h5> --protocol <the cond _protocol.json> \
  -o $RUN/readout.png --csv $RUN/readout.csv
```

The figure's top panel is the experiment: spikes in US in the pulse-locked window after each
presentation, coloured by what was presented. **Learning is CS-alone probes rising in US from
baseline to retrieval while NS-alone probes do not, with US-alone probes flat.** The report also
recomputes the coupling matrix from the `pre` and `post` blocks, using the same measure that chose
the regions, so you can see how they changed.

`readout.csv` has one row per presentation with every region's count in every window. That is the
file to check the conclusion against independently: the effect should be visible as a difference of
group means in a spreadsheet, without any of this code.

## 9. Follow-up on later days

Same parameter file, same regions and amplitudes, no encoding:

```bash
uv run python -m mxtreme.experiments.associative run \
  --params $RUN/params_<chip>_well<N>.json --config mxtreme.toml \
  --set exp_id=day2 --set div=<today's DIV> \
  --set 'encode_pattern=[]' --set pre_min=5 --set post_min=5 --set num_retrievals=3
```

An empty `encode_pattern` gives probes without pairing, which is the long-horizon question: has the
CS-to-US response grown, held, or gone?

Every probe is also a little extinction, so probing frequency is itself a variable. If you have two
cultures, probing one daily and one every third day measures how much of the decay is the probing.

## How large is a "region"?

Three numbers should be read together, and `preview` prints all three:

```
site: 4 driven electrodes spanning 88 um, return ring at 150 um;
      response measured over 150 um; 24 of 32 stimulation units
```

The **driven span** is how wide the stimulation actually is, the **ring radius** is where the
inverted return electrodes sit, and **`region_radius_um`** is the radius over which that region's
response is counted. The default puts the drive across 88 um with the ring at exactly the measured
radius, so the stimulation site fills the region it is named after rather than being a point inside
it.

The binding constraint is stimulation units: the chip has 32, every stimulation electrode needs
one, and three regions share them. That rules out some otherwise sensible sites:

| site | units/region | total | driven span | fits? |
|---|---|---|---|---|
| focal 2x2, `inner_gap` 0 | 8 | 24 | 17.5 um | yes, but single-neuron scale |
| focal 2x2, `inner_gap` 4 (default) | 8 | 24 | 87.5 um | yes |
| focal 2x2, `inner_gap` 7, ring 9 | 8 | 24 | 140 um | yes, ring then sits outside the measured region |
| focal 3x3, ring 4 | 13 | 39 | 105 um | **no**, refused |
| grid 3x3, gap 2 (no ring) | 9 | 27 | 105 um | yes, but no spatial charge balance |
| grid 3x3, gap 4 (no ring) | 9 | 27 | 175 um | yes, same caveat |

So widen a focal site with `inner_gap`, never with `inner`. If you would rather have nine driven
points than an inverted return ring, use a grid site and accept that the field is no longer
confined; `validate` refuses anything that cannot route, before the chip is touched.

## Safety notes

- **Amplitude is in mV per phase; peak to peak is twice that.** The shipped ladder is 20 to 120 mV
  per phase, which is 40 to 240 peak to peak, the range Ronchi et al. 2019 characterised on these
  arrays. `max_amplitude_mv` refuses anything above 150 per phase unless you raise it deliberately.
- **Pulses are biphasic and charge balanced** in both time (equal and opposite phases) and space
  (a focal site's return ring carries the inverted pulse). Net charge into the electrode is
  nominally zero, which is what protects it from electrolysis.
- **The duty cycle is very low.** The 220-minute run delivers about 1400 pulses to each region, and
  at 100 µs per phase that is under a third of a second of driven electrode across the whole day.
  The preview prints this figure for whatever parameters you are about to run.
- **Widening the site does not change the dose per electrode.** Spreading four driven
  electrodes over 88 um instead of 17.5 um delivers the same charge at each one; it covers more
  tissue, it does not push any electrode harder.
- **A long sequence is not a large dose.** The 2000-command sequence in the dry run is 600 pulses
  spread over ten minutes at the same amplitude as any other. What it tests is whether `maxlab`
  accepts a sequence that long, not whether the electrodes can take it.

## Files each step leaves behind

| file | what |
|---|---|
| `params_<chip>_well<N>.json` | the frozen configuration for this culture |
| `regions_<chip>_well<N>.json` / `.png` | the region choice and everything behind it |
| `coupling_<chip>_well<N>.png` | the baseline coupling evidence |
| `<stem>_protocol.json` | exactly what a run was going to do; `report` reads it |
| `<stem>_fired.csv` | when each presentation actually fired, against plan |
| `<stem>.raw.h5` | the recording, with an event at the start and end of every presentation |
| `readout.csv` | per-presentation counts, for checking the conclusion by hand |
