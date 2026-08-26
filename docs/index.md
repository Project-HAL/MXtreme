# MXtreme

Experimental and analysis toolkit for Maxwell Biosystems MaxOne/MaxTwo HD-MEA recordings including:
- Customizable activity and network scans
- Track basic metrics over DIVs (Day In Vitro) and generate reports 

MXtreme analysis occurs in four stages starting from the raw `.h5` file(s):

| Stage | Purpose | MXtreme Module(s) |
|---|---|---|
| **Extract** | Read the raw `.h5`; pull each well's spike data and metadata | {mod}`mxtreme.extract` |
| **Clean** | Filter, de-noise, and bin spikes through a user-assembled pipeline | {mod}`mxtreme.clean`, {mod}`mxtreme.pipeline` |
| **Burst Detection** | Find network bursts and extract per-burst features | {mod}`mxtreme.bursting` |
| **Analyze** | Per-DIV summaries, population comparisons, a multi-section PDF | {mod}`mxtreme.analysis` |

Each stage writes its output to a user-defined **managed data store** — see [Configuration](concepts/configuration.md).


```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/configuration
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
contributing
```
