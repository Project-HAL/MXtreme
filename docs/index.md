# MXtreme

Analysis toolkit for Maxwell Biosystems MaxOne/MaxTwo HD-MEA recordings.

MXtreme takes a raw Maxwell `.raw.h5` recording and carries it through to per-culture metrics and a
PDF report, in four stages you can enter or leave at any point:

| Stage | What happens | Modules |
|---|---|---|
| **Extract** | Read the raw `.h5`; pull each well's spike table plus metadata | {mod}`mxtreme.extract` |
| **Clean** | Filter, de-noise, and bin spikes through a user-assembled pipeline | {mod}`mxtreme.clean`, {mod}`mxtreme.pipeline` |
| **Detect** | Find network and mini bursts; extract per-burst features | {mod}`mxtreme.bursting` |
| **Analyse** | Per-DIV summaries, population comparisons, a multi-section PDF | {mod}`mxtreme.analysis` |

Each stage writes its output to a **managed data store** and the next stage reads it back, so a long
preprocessing run happens once and every later analysis is cheap. The store's location is the one
thing you have to configure — see [Configuration](concepts/configuration.md).

## Design commitments

The package is deliberately **experiment-agnostic**. It assumes nothing about your phase names, your
stimulation paradigm, or your directory layout:

- **Phases are injected.** A recording with no phase information is a single `"full"` phase. If your
  experiment has `pre`/`train`/`post` structure, you say which maxlab event tags mark the
  boundaries — see {mod}`mxtreme.phases`.
- **Pipeline steps are plain functions.** Drop them, reorder them, wrap them in
  {func}`functools.partial` to change a parameter, or write your own. See {mod}`mxtreme.clean`.
- **Scoring is pluggable.** {mod}`mxtreme.analysis.performance` takes an
  `objective_fn(burst_df) -> float`; the shipped default is one option, not a requirement.
- **No hard-coded paths.** Every writer takes an explicit output directory, resolved from a
  {class}`~mxtreme.config.Config` you supply.

`import mxtreme` is kept fast and light — only the pure-Python identity types are re-exported at the
top level. Anything that pulls in matplotlib, seaborn, or h5py lives behind an explicit submodule
import.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/data-flow
concepts/configuration
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
contributing
```
