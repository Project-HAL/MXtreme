# Procedure: associative conditioning on a culture

The day's plan for one culture. Scans come from `braintrix-cli`, everything after them from
`mxtreme.experiments.associative`. Do the [dry run on saline](associative-dry-run.md) first; this
document assumes the hardware path is already known to work.

## The shape of the day

| step | what | tool | how long | writes into the store |
|---|---|---|---|---|
| 1 | activity scan | braintrix-cli | ~8 min | `<stem>_activity_scan.raw.h5` |
| 2 | baseline: network scan of the active set | braintrix-cli | 5 min | `<stem>_network_scan.raw.h5` |
| 3 | place the three regions | `select`, offline | seconds | nothing |
| 4 | calibration | `run --mode calibration` | 22 min | `<stem>_assoc_cal.raw.h5` |
| 5 | connectivity check | `run --mode connectivity` | 9 min | `<stem>_assoc_conn.raw.h5` |
| 6 | conditioning: one command, six recordings | `run --phase session` | 180 min | `<stem>_assoc_baseline`, `_encode_1..4`, `_retrieval` `.raw.h5` |
| 7 | read each recording back, and the session as one | `report`, offline | minutes | nothing |

Every recording is spikes only and small (the whole session is 0.1-0.35 GB, see below); the
`.raw.h5` extension is MaxWell's for every file it writes, whatever the file holds.

`<stem>` is the store's `plating_<date>_<batch>_chip_<chip>_well_<N>_DIV_<d>`. Steps 4 and 5 are
gates: each can send you back to step 3 with a stated reason. Budget for that. Step 6 is one
command but six recordings, each closed before the next opens, so each can be read as soon as it
is done: the baseline says whether the floor is low before 100 minutes of encoding are spent, and
each cycle's checkpoint says whether the curve is moving.

## Where everything goes

**Into the store: recordings, and only recordings.** Every file above lands in the culture's own
directory, `<store>/recordings/plating_<date>_<batch>/chip_<M1|M2>_<chip>/well_<N>/DIV_<d>/`,
beside the scans it came from, and each is registered. The experiment's recordings are
self-describing: the protocol (the parameters, which electrodes are US, CS and NS, every stimulus
and block) is written inside each one at `/assay/associative_protocol`, the routing is MaxLab's own
channel map, and every presentation is an event with its frame. Nothing else sits beside them.
(braintrix-cli's network scan also writes its own `electrode_selection/` figures beside the
activity scan; that is its convention, not this experiment's.)

**Everything else goes in a work directory outside the store**: the parameter file `select`
writes, its figures, a copy of each run's protocol, each run's planned-against-actual fire times,
and every report. All of it can be regenerated from the recordings.

**The experiment's recordings keep spikes only.** The readout is spike counts, so raw voltage adds
nothing to it, and a four-hour run is then a few hundred megabytes. The scans are braintrix-cli's
and keep raw traces as they always do.

| file | size, MaxTwo | MaxOne |
|---|---|---|
| activity scan (8 min, raw) | ~2 GB | ~4 GB |
| network scan (5 min, raw) | ~1.2 GB | ~2.4 GB |
| calibration + connectivity (spikes only) | ~0.03 GB together | same |
| conditioning session, 180 min over six recordings (spikes only) | **0.09 to 0.35 GB** | same |

A spike costs 16 bytes on disk (frame, channel, amplitude), and the two cultures measured here fired
0.5 and 0.9 spikes per routed channel per second, so 1024 routed channels come to 0.5-1.9 MB a
minute — the range is the culture's own liveliness, not an uncertainty in the format. Sample rate
does not enter into it, so a MaxOne costs the same as a MaxTwo. `preview` prints the estimate for
whatever you are about to run.

The raw figures are measured too (3.8 kB per channel-second of compressed raw on a MaxTwo; a
MaxOne samples twice as fast for twice the size). `raw_traces` in the parameter file changes what
the experiment keeps: `"regions"` adds raw traces for the three regions' electrodes (at 3.8 kB per
channel-second, 100 such channels over 180 min is ~4 GB), `"all"` every channel (~42 GB on a
MaxTwo, ~83 GB on a MaxOne — do not, unless you have decided to). `preview` prints the number of
channels and the total either way.

## What you are checking at each step

Every step prints its decision and the numbers behind it; every recording has a `report`; nothing
past step 3 needs the terminal scrollback, since the parameter file and the recordings carry it.
What to look at, and what would stop you:

| after | look at | go on if | stop and do instead |
|---|---|---|---|
| scans | `ls $CULTURE`; braintrix-cli's electrode-selection figure | hundreds of active electrodes, spread over the array | a quiet or one-sided array: another culture |
| `select` | its terminal output and `*_regions.png`, `*_coupling.png` | three regions ≥ 1000 µm apart, coupling < 0.3, 3-4 of 4 driven electrodes active each | `--centers`, or `--separation 800` |
| calibration `report` | the table and the verdicts | an `ok` per region with an amplitude | `STOP`: back to `select` for that region |
| `compare` (optional) | the verdicts | a pattern named per region | ties: keep the default |
| connectivity `report` | the cross-talk matrix and the verdicts | `ok`, or warnings you have read and accept | `STOP`: amplitude or site is wrong; a CS-US path warning: closer regions |
| `preview` | the timeline and the size line | 180 min, spikes only, < 0.5 GB, regions drawn | anything surprising: fix `$P` first |
| baseline `report`, at 16 min, from a second terminal | the conditioning verdict | no high-floor warning, no burst warning, "no retrieval block yet" | a high floor: Ctrl-C the run, reselect with wider separation; bursting: Ctrl-C, lower the amplitudes, start again |
| encode `report`s (all files so far), whenever | the checkpoint table | a curve with one more point per cycle; bursting under 20% | bursting: Ctrl-C, lower the amplitudes, `--phase encode_<next>,retrieval` |
| session `report` (all files) | the conditioning verdict, the table, the figure | a verdict either way: association, excitability, or null with a next step | — it is data either way |

Why each number is what it is: [the stimulus rationale](associative-stimulus-rationale.md).

## Before the day

- The rig's `../MXtreme` checkout needs the `boire/associative-learning` branch. braintrix-cli
  installs MXtreme in editable mode, so whatever is checked out there is what runs.
- The dry run has passed.

## 0. Set up, on the rig

Everything runs from the environment braintrix-cli runs from on the rig (`conda activate jkts`
at the time of writing), which already has MXtreme and `maxlab`; nothing here installs anything.
The dry run's step 0 says how to check it sees the right MXtreme checkout and branch.

```bash
cd <path to>/braintrix-cli
conda activate jkts
PKG=../MXtreme/src/mxtreme/experiments/associative
CFG=~/.config/mxtreme/mxtreme.toml     # the store braintrix-cli's header names; $MXTREME_CONFIG wins if set
grep root $CFG                         # the store's root, for STORE below
```

The culture's identity, used by every command:

```bash
BATCH=fall2026_batch1_iPSC_M1     # its last token must match the device: M1 MaxOne, M2 MaxTwo
CHIP=M07459
PLATE=260401                      # plating date, YYMMDD
DIV=27
WELL=0
STORE=<root from the grep above>
CULTURE=$STORE/recordings/plating_${PLATE}_${BATCH}/chip_${BATCH##*_}_${CHIP}/well_${WELL}/DIV_${DIV}
WORK=~/assoc-work/${CHIP}_DIV${DIV}
mkdir -p $WORK
```

## 1-2. Activity scan and baseline, in braintrix-cli

```bash
python run_cli.py
```

Load the plating batch at startup, then **Record on device → Activity + network scan**. Pick
**Kam Scan**, confirm the identity matches the variables above, and set the network scan's
**Recording length** to `300` s. One run records both and files both into the store.

When it finishes:

```bash
ls $CULTURE
```

should show `..._activity_scan.raw.h5` and `..._network_scan.raw.h5`. Quit braintrix-cli.

## 3. Place the three regions

```bash
python -m mxtreme.experiments.associative select \
  --params $PKG/params_default.json --out $WORK \
  --baseline $CULTURE/*_network_scan.raw.h5 \
  --activity-scan $CULTURE/*_activity_scan.raw.h5 \
  --candidates 6 \
  --set batch=$BATCH --set chip=$CHIP --set plate_date=$PLATE --set div=$DIV --set well=$WELL
P=$(ls $WORK/*_assoc_params.json)
```

What it does, from the baseline: ranks 150 µm patches by active electrodes, keeps the densest ones
at least 1000 µm apart, measures their coupling (correlation with network bursts left out, and
burst-onset order), and picks the least-coupled, near-equilateral triple, with US at the vertex
equidistant from the other two and CS versus NS set by the seed. Then, per region, it moves the
driven block by up to 70 µm so that its four electrodes sit on electrodes the activity scan
recorded spikes from, and prints what it found:

```
stimulation sites: each driven block moved by up to 70 um so its electrodes sit on ones the activity scan recorded spikes from:
  US   4 of 4 driven electrodes active (moved 25 um): e6736 4.7 Hz 60 uV, e6741 1.2 Hz 35 uV, ...
  CS   3 of 4 driven electrodes active (moved 0 um): ...
  NS   2 of 4 driven electrodes active (moved 52 um): e23402 0.4 Hz 24 uV, e23407 silent, ...
```

A region being dense does not make its four particular electrodes live, and a driven electrode
over glass stimulates nothing. **Three or four of four is what you want; two is workable; zero
gets a warning and means calibration will most likely find nothing there** — give that region
another centre with `--centers` rather than spend 22 minutes confirming it. The activity scan is
the right source because it covers every electrode; the baseline covers only its thousand.

It then routes every electrode the baseline recorded, plus every electrode the activity scan found
active inside the three chosen regions: standard electrode selection keeps electrodes 100 µm
apart, which leaves only a handful in a region, and the readout is counted on these.

If a scan was repeated that day the globs match two files; name the one you mean instead.

`$P` is the culture's parameter file. **Every later command points at it**, and it carries the
culture's identity, so none of them repeats it. Read the two figures `select` wrote into `$WORK`
before continuing:

| look at | good | bad |
|---|---|---|
| correlation, bursts left out | below ~0.1 for the three chosen | any chosen pair above ~0.3 |
| burst order | near 50% for the chosen pairs | one patch first in most bursts (it drives the others) |
| triangle | ratio above 0.6, US-CS and US-NS within ~15% | one region between the other two |
| coupling vs distance | no clear trend | rising with distance |
| driven electrodes | 3-4 of 4 active per region | 0 of 4 anywhere |

If it refuses (`only N patches qualify`), it writes a rejection figure saying why. Lower
`--separation`, or place the regions by hand with `--centers "x,y;x,y;x,y"`.

## 4. Calibration

```bash
python -m mxtreme.experiments.associative run --params $P --config $CFG \
  --mode calibration --set exp_id=assoc_cal --set pre_min=2 --set post_min=2 --work $WORK
python -m mxtreme.experiments.associative report $CULTURE/*_assoc_cal.raw.h5 \
  -o $WORK/assoc_cal.png --csv $WORK/assoc_cal.csv
```

`run` first checks that the connected device matches the batch's `M1`/`M2`, and refuses if not:
that token decides which `chip_M1_...` or `chip_M2_...` directory the recording is filed under.

Six amplitudes (20 to 120 mV per phase) times two polarities times ten repeats per region,
shuffled, with two minutes of quiet either side. The report prints, per region and amplitude,
`local` (spikes per pulse in the stimulated region), `remote` (in the other two: how far the pulse
reaches), `spread` (remote over local) and the burst rate, then a recommended `amplitudes_mv`: the
largest amplitude that evokes at least 0.5 spikes per pulse locally while starting a network burst
on at most 20% of pulses. A `STOP` says which way it failed: nothing responds even at 120 mV (the
site has too few neurons; back to step 3), or everything that responds also bursts (back to
step 3).

**Copy the two lines the report prints** — `amplitudes_mv` and `amplitudes_source` — into `$P`.
Without the second, every later `run` refuses to start.

### 4b. Which pattern to drive through: focal 2x2 or grid 3x3

The default drives a 2x2 block with a return ring, which confines the field. A 3x3 grid with no
ring drives nine electrodes, reaches more tissue, and spreads further. Which is the better trade
on this culture is an empirical question, and the same calibration answers it: run it once more
with the other pattern on the same sites, then set the two side by side.

```bash
python -m mxtreme.experiments.associative run --params $P --config $CFG \
  --mode calibration --set exp_id=assoc_cal_grid --set pre_min=2 --set post_min=2 --work $WORK \
  --set 'stim_site={"shape":"grid","size":3,"gap":2}'
python -m mxtreme.experiments.associative compare \
  $CULTURE/*_assoc_cal.raw.h5 $CULTURE/*_assoc_cal_grid.raw.h5 -o $WORK/assoc_cal_compare.png
```

`compare` prints each run's table, then per region: the threshold amplitude for each pattern, the
local response at the largest amplitude both reached, and the spread there, and a verdict — the
pattern with the lower threshold, or the same threshold and more local response, or the same of
both and less spread; flagged instead of decided when the harder-driving pattern also spreads more
than a third further. Another 22 minutes; skip it on a culture you are short of time with, and
the default stands on the argument in the rationale (a confined field is the cheapest way to three
independent sites).

**Copy the winning pattern's `stim_site`, its `amplitudes_mv` and its `amplitudes_source` into
`$P`.** The sites' centres
were placed for the 2x2's electrodes (step 3), so a 3x3 at the same centre has five electrodes
that were not checked for activity; `compare` measures what they evoke regardless.

From here the regions, pattern and amplitudes are frozen for this culture.

## 5. Connectivity check

```bash
python -m mxtreme.experiments.associative run --params $P --config $CFG \
  --mode connectivity --set exp_id=assoc_conn --set pre_min=2 --set post_min=2 --work $WORK
python -m mxtreme.experiments.associative report $CULTURE/*_assoc_conn.raw.h5 \
  -o $WORK/assoc_conn.png --csv $WORK/assoc_conn.csv
```

Single pulses at the calibrated amplitudes, interleaved across the three sites. The report prints
how much stimulating each site alone drives the other two, and a verdict:

| verdict | meaning | do |
|---|---|---|
| a site evokes almost nothing locally | amplitude or site is wrong | back to step 4, then 3 |
| a site's pulses start bursts >20% of the time | it drives the whole culture | lower its amplitude |
| a site drives another >30% of its own response | the two are not independent | back to step 3, wider separation |
| no response either way between CS and US | there may be no path to strengthen | back to step 3 |

This is the last cheap moment to change your mind.

## 6. Conditioning: one command, six recordings

```bash
python -m mxtreme.experiments.associative preview --params $P --config $CFG
```

prints the whole session's schedule, that it keeps spikes only, and the pulse count each region
will receive, then draws the plan. Then:

```bash
python -m mxtreme.experiments.associative run --params $P --config $CFG --phase session --work $WORK
```

sets the rig up once and records the session as six files, one after another with nobody at the
keyboard between them: `..._assoc_baseline`, `..._assoc_encode_1` to `_4`, `..._assoc_retrieval`.
Each has its own protocol inside it and is closed before the next opens, so every file is small
and stands on its own; the gaps between them are a few seconds of file handling. `run` refuses to
start while `amplitudes_source` is still `default`, which is its way of asking whether step 4
happened. Three hours; the terminal prints each block as it starts and each presentation as it
fires with its offset from plan, and MaxLab Live shows the culture live.

**Reading it as it goes.** Every file closes when its phase ends, so from a second terminal:

```bash
python -m mxtreme.experiments.associative report $CULTURE/*_assoc_baseline.raw.h5
python -m mxtreme.experiments.associative report $CULTURE/*_assoc_baseline.raw.h5 $CULTURE/*_assoc_encode_*.raw.h5
```

The first, at 16 minutes, is the floor check with 60 pulses per role: **CS alone should evoke
little in US** (under 30% of what US alone does) and no more than 20% of probes should start a
network burst; "no retrieval block yet" is the expected last line. The second, any time after,
prints the checkpoint table with one row per finished cycle: the curve of the association forming,
as far as it has got. If the baseline shows a high floor, stop the run (Ctrl-C closes the current
file properly and records no further phase) and reselect with a wider separation before spending
100 minutes encoding on top of it.

**One phase at a time**, if you would rather decide between them: `--phase baseline`, then
`--phase encode_1` and so on, then `--phase retrieval`; or `--phase encode` for the four cycles
together. The files and the schedule are the same either way — every cycle is built whatever the
phase, so the seeded shuffles agree — and the gaps you take are in the files' clocks, so the
session's timeline stays true. `--phase` left off records the whole schedule as one file, the
1-minute leads dropped.

`preview` prints the blocks with what each one delivers:

| phase | block | length | what happens |
|---|---|---|---|
| baseline | pre | 5 min | quiet; the coupling matrix before |
| baseline | baseline | 11 min | 9 probes: each role alone for 60 s (20 pulses), 75 s apart, shuffled; 60 pulses per role |
| encode_1..4 | lead | 1 min | quiet |
| encode_1..4 | encode | 25, 25, 25, 20 min | CS+US, NS, CS+US, NS — 4.5 min trains 5.5 min apart — then (cycles 1-3) 2 min rest and a checkpoint. 712 paired pulses over the four |
| retrieval | decay | 5 min | quiet |
| retrieval | retrieval_1..3 | 11 min each, 10 min apart | the baseline block again, identically |
| retrieval | post | 5 min | quiet; the coupling matrix after |

A presentation is a *train*, not a pulse: a training presentation is 89 pulses over 4.5 minutes, a
probe 20 over one. The terminal prints each block as it starts and each presentation as it fires
with its offset from plan. MaxLab Live shows the culture live. 180 minutes of recording; add
whatever you spend between them.

Probes are kept small against the paired pulses (60 per role per block against 712 paired) because
every probe is also a little extinction. Retrieval probes all three roles, not just CS, so that a
general change in excitability cannot be read as learning.

### Checkpoints: the association as it forms

Between encoding cycles, after the rest, CS and NS are each probed alone once — a short probe,
`encode_probe_sec` 30 s, 10 pulses. Three checkpoints, two probes each: **60 probe pulses across
the whole encode block against 712 paired ones**, for 3 minutes of the run.

A checkpoint is an unpaired CS presentation in the middle of acquisition, which is what extinction
is, so it is kept to the fewest pulses that will carry a measurement. A probe block's probes stay
at 60 s and 20 pulses: those sit outside encoding, where there is nothing to extinguish yet or
left to protect.

That turns three end-point measurements into a curve. `report` groups every probe-alone
presentation into the checkpoint it belongs to and prints it in time order, **per pulse**, so the
10-pulse checkpoints and the 20-pulse probe blocks share one axis:

```
association, by checkpoint: spikes in US per pulse, stimulating one region alone
  checkpoint       min    US alone    CS alone    NS alone  probes
  baseline           0        0.95        0.11        0.12       9
  encode 1          35           -        0.15        0.11       2
  encode 2          61           -        0.22        0.10       2
  encode 3          88           -        0.26        0.12       2
  retrieval_1      116        0.97        0.28        0.11       9
```

Twenty pulses is a noisy estimate — one checkpoint moving means nothing. The curve is worth
reading only as a trend across all its points, with the baseline and retrieval blocks (60 pulses
per role) as its anchors. The CSV carries a `checkpoint` column, so the same grouping plots
directly.

Under the table, `report` (given every recording of the session so far) prints a **conditioning
verdict**: the checks a person would make from it, made every time — CS alone already driving US at baseline (a high floor), pulses starting
network bursts, US's own excitability drifting between baseline and retrieval, and the result:
CS alone at the retrievals against baseline at two standard errors on the pooled pulses, with NS
alone the same way. *CS up, NS flat* is the association; *both up* is excitability; *neither* is
the null, and the line says what to try next. [The rationale](associative-stimulus-rationale.md)
lists where each sign shows and what to do.

`--set 'encode_probe_roles=[]'` turns checkpoints off; `--set encode_probe_reps=2` doubles each of
them (120 pulses, +3 min) if the first culture's curve is too ragged to read;
`--set 'encode_probe_roles=["CS","NS","US"]'` adds the excitability control at each one.

### The encoding structure

The rate is the part not to raise. A pulse evokes a network response lasting a few hundred
milliseconds, and the response depresses if the next pulse comes too soon, so sustained pairing in
vitro sits at 0.2-1 Hz for tens of minutes: 0.3-1 Hz until the response appears (Shahaf & Marom
2001), 0.2-0.33 Hz in 10-15 min blocks with rests, over hours (le Feber). The default is 0.33 Hz.
What is worth choosing is the NS dose and the length:

| | default | more pairing |
|---|---|---|
| encode pattern | `["PAIR","NS","PAIR","NS"]`, 4.5 min trains, 4 cycles | `["PAIR","PAIR","NS"]`, 10 min trains, 3 cycles |
| paired pulses | 712 | 1188 |
| pulses to CS : NS | 1 : 1 | 2 : 1 |
| total run | 180 min | 220 min |

```bash
# more pairing, at the cost of the control and 40 minutes
--set t_stim=600 --set encode_iti=660 --set 'encode_pattern=["PAIR","PAIR","NS"]' \
--set encode_cycles=3 --set encode_cycle_rest=300 --set pre_min=10 --set post_min=10
```

The default gives NS exactly as much stimulation as CS, never together with US. That is what rules
out "any region driven for an hour grows a path to US", and it is worth more than the extra paired
pulses: an ambiguous positive is worth less than a clean negative. NS trains stay at least 60 s
from any pairing, far outside any plasticity window.

The 180 minutes also came out of the quiet stretches, not the stimulation: `pre`, `post` and
`decay` are 5 minutes rather than 10, the probes sit 75 s apart rather than 90, and the retrievals
are 10 minutes apart rather than 15. The probe count per block is unchanged, so baseline and
retrieval are still 60 pulses per role.

To shorten it, cut cycles or retrievals rather than presentation length:
`--set encode_cycles=2 --set num_retrievals=2`.

## 7. Reading the session back

```bash
python -m mxtreme.experiments.associative report $CULTURE/*_assoc_*.raw.h5 \
  -o $WORK/assoc_session.png --csv $WORK/assoc_session.csv
```

Every recording of the session, in any order; `report` sorts them by their clocks. The calibration
and connectivity recordings match that glob too and are harmless in it (each is read on its own
terms), but name the six conditioning files if you want the output short.

The figure's top panel is the experiment: spikes in US in the pulse-locked window after each
presentation, by what was presented. **Learning is CS-alone probes rising in US from baseline to
retrieval while NS-alone probes do not, with US-alone probes flat.** The report also recomputes
the coupling matrix from the quiet periods before and after, using the measure that chose the
regions.

`assoc_session.csv` has one row per presentation with every region's count in every window: the file
to check the conclusion against independently, as a difference of group means in a spreadsheet.

## Later days

Same `$P`, no encoding, a new DIV (so a new directory in the store):

```bash
python -m mxtreme.experiments.associative run --params $P --config $CFG \
  --set exp_id=assoc_retest --set div=<today's DIV> --work $WORK \
  --set 'encode_pattern=[]' --set num_retrievals=3
```

An empty `encode_pattern` gives probes without pairing: has the CS-to-US response grown, held, or
gone? Every probe is also a little extinction, so with two cultures, probing one daily and one every
third day measures how much of any decay is the probing.

## How large is a "region"?

`preview` prints the stimulation site's footprint next to the radius over which responses are
counted:

```
site: 4 driven electrodes spanning 122 um, return ring at 210 um;
      response measured over 150 um; 18 of 32 stimulation units
```

The driven block spans 122 µm; the two return electrodes sit at opposite corners around it,
centred on it, outside the 150 µm counting radius, so a region's count is its driven block's
response. The binding constraint is stimulation units: 32 on the chip, one per stimulation
electrode — and, measured on a MaxOne, each *area* of the chip exposes only about seven of them,
so a site of eight electrodes (a 2x2 with a four-corner ring) cannot be connected while one of six
can. That is why the default ring is two corners.

| site | units total | per site | driven span | fits? |
|---|---|---|---|---|
| focal 2x2, `inner_gap` 6, two returns (default) | 18 | 6 | 122.5 µm | yes |
| focal 2x2, `inner_gap` 4, four returns | 24 | 8 | 87.5 µm | routes, but one area rarely has 8 free units |
| focal 2x2, `inner_gap` 7, `return_radius` 9, two returns | 18 | 6 | 140 µm | yes |
| focal 3x3, two returns | 33 | 11 | 122.5 µm | **no**, refused |
| grid 3x3, gap 2 (no ring) | 27 | 9 | 105 µm | routes, but 9 per area is over the ~7 available |

Widen a focal site with `inner_gap`, never `inner`.

## Safety notes

- **Amplitude is in mV per phase; peak to peak is twice that.** The ladder is 20 to 120 mV per
  phase, 40 to 240 peak to peak, the range Ronchi et al. 2019 characterised on these arrays.
  `max_amplitude_mv` refuses anything above 150 per phase unless raised deliberately.
- **Pulses are charge balanced** in time (equal and opposite phases) and in space (the return ring,
  centred on the driven block, carries the inverted pulse).
- **The duty cycle is very low.** The conditioning run delivers about 1400 pulses to each of US and
  CS; at 100 µs per phase that is under a third of a second of driven electrode across the day.
  `preview` prints this for whatever you are about to run.
- **Widening a site does not raise the dose per electrode**, and **a long sequence is not a large
  dose**: both spread the same pulse over more tissue or more time.
