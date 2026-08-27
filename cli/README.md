# MXtreme CLI

A menu-driven front end for this repository. Everything scientific lives in `mxtreme`; this folder
only collects parameters, calls the package, and prints what happened.

## Running

```bash
python cli/run_cli.py          # from the repo root, in the project environment
```

or, from inside `cli/`:

```bash
python -m mxtreme_cli
```

The launcher works whether or not `mxtreme` is installed: it prefers the installed package and falls
back to `src/` on the path. Nothing outside `cli/` has to be configured, and no existing module is
modified.

### Getting around

Lists are driven with the arrow keys: **↑/↓** moves the highlighted row, **Enter** chooses it, and
**q** or **Esc** goes back. Typing a row's number jumps to it, and in a list short enough that every
row has a single digit, the number *is* the choice — no Enter needed. Ctrl-C backs out of the
current step rather than killing the session.

The parameter editor works the same way. Its first row is "Continue with these values" and the
cursor starts there, so the fast path is unchanged: open the screen, press Enter, run with the
defaults. Choosing a parameter prompts for a new value and returns you to the same row.

Where the terminal cannot deliver single keypresses — piped input, output to a log, Windows — every
list falls back to the older numbered form, where you type a number and press Enter. That fallback
is what keeps the CLI scriptable, so both paths are maintained rather than one being a legacy
leftover.

Set `MXTREME_CLI_TRACEBACK=1` to see a traceback when a screen raises something unexpected.

## Menu

1. **Activity scan** — pick a scan protocol, review its parameters, run it on the rig. Requires
   MaxWell's `maxlab` API, so this screen only works on the rig machine; the other screens do not.
2. **Electrode selection** — read an activity scan `.h5`, filter to active electrodes, and choose a
   spatially spread set of recording electrodes for a network scan. Offline; runs anywhere.
3. **Data analysis** — stub. Will front `mxtreme.extract` / `clean` / `bursting` / `analysis` for
   arbitrary `.h5` recordings.

After a scan finishes, the CLI offers to run electrode selection on the file it just produced.

### Scans run in the background

Starting an activity scan hands you straight back to the main menu. The scan runs on its own thread,
and every screen's header carries its progress — recordings done, a bar, and an estimate of the time
left worked out from the rounds already recorded rather than from the nominal recording length, so
routing and offset-compensation overhead are included.

Choosing **Activity scan** while one is running warns you first and offers to start another anyway,
to stop the running one, or to go back. Starting another is allowed but rarely what you want: one
rig drives one chip, and two scans sharing it will interleave their routing and recording.

Nothing is ever overwritten. MaxLab refuses to reuse a filename that already exists and appends
`_1`, `_2`, … instead, so an earlier scan's `.h5` survives a new scan of the same name — you end up
with both. The screen says which case you are in before anything is written.

Quitting with a scan still running is refused. The recording in progress has to end before the `.h5`
can be closed, so you are offered either to wait it out or to stop after the current recording; both
finalize the file, and everything recorded so far stays readable.

### Scan protocols

The Activity Scan screen lists protocols from `SCANS` in `mxtreme_cli/screens/activity_scan.py`. Today there
is one, **Kam Scan**: the array is covered by a series of short recordings, each routing a different
random subset of the available electrodes, drawn without replacement so a well's recordings do not
overlap until the array has been covered once.

To add another protocol, append a `ScanProtocol` to `SCANS` with a factory that returns its default
`ActivityScanParams`. A protocol needing parameters the shared list does not cover gets its own field
list; nothing else in the screen changes.

Every parameter is shown with its default before anything runs, so the common case is: pick the scan,
read the summary, press Enter.

## Outputs

An activity scan writes `<save dir>/<CHIP>_<PLATE DATE>_ACTIVITY_SCAN.raw.h5`, plus
`..._scan_electrodes.npz` recording which electrodes each round routed (the `.h5` does not preserve
this per-round, and it is the only record of what was *attempted*).

Electrode selection writes, into the output directory:

| File | Contents |
| --- | --- |
| `<prefix>_AS_well<N>.png` | Firing rate, amplitude, active electrodes, ISI for the whole scan |
| `<prefix>_network_well<N>.png` | The selected electrodes over the array layout |
| `<prefix>_network_well<N>.npz` | The electrode list, under key `recording_electrodes` |

## Layout

| File | Role |
| --- | --- |
| `run_cli.py` | Launcher; sets up `sys.path` and calls `app.main()` |
| `mxtreme_cli/app.py` | Top-level menu and error handling |
| `mxtreme_cli/screens/activity_scan.py` | Activity Scan screen and the protocol list |
| `mxtreme_cli/screens/electrode_selection.py` | Electrode Selection screen |
| `mxtreme_cli/screens/data_analysis.py` | Data Analysis stub |
| `mxtreme_cli/prompts.py` | Menus, typed prompts, the parameter editor |
| `mxtreme_cli/keys.py` | Single-keypress input for the arrow-driven lists |
| `mxtreme_cli/jobs.py` | Background scan threads and the status the banner reads |
| `mxtreme_cli/ui.py` | Headings, status lines, tables, colour |

One module per menu entry lives under `mxtreme_cli/screens/`; `app.py` imports each lazily, so a
missing dependency takes out only the screen that needs it. Adding a screen is a module there plus a
row in `MAIN_MENU`.

The scan itself lives in the package, not here: `mxtreme.scans.activity_scan`. Its planning half
(`ActivityScanParams`, `plan_scan_electrodes`) is pure Python, which is what lets this CLI show and
validate scan parameters on a machine that has no `maxlab`.
