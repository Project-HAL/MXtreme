# API reference

Generated from the docstrings in `src/mxtreme/`. The pages are grouped by pipeline stage rather than
listed alphabetically, so they read in the order data actually moves through the package.

```{toctree}
:maxdepth: 2

core
preprocessing
bursting
analysis
plotting
scans
```

## Not documented here

- **`mxtreme.constants`** — legacy module-level hyperparameters and lab-specific filesystem paths.
  Superseded by {mod}`mxtreme.config` for locations and {mod}`mxtreme.params` for detection knobs.
  Deliberately excluded from the published site.
- **`mxtreme.utils`, `mxtreme.device`** — small internal helpers with no stable public contract.
- **`mxtreme.analysis._paths`, `._plotting`, `._stats`** — private, by the underscore convention.
