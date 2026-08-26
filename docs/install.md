# Installation

## Requirements

- **Python 3.12+**
- For development: **[uv](https://docs.astral.sh/uv/)**, which manages the environment and a
  committed lock file so every contributor works against an identical dependency stack.

End users installing the published package need only pip.

## Users

```bash
pip install mxtreme
```

```python
import mxtreme
print(mxtreme.__version__)
```

matplotlib and seaborn are **required**, not optional — they are imported at module level by the
recording, bursting, visualization, and analysis modules, so an install without them cannot import
the package.

## Developers

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell afterward so `uv` is on your `PATH`.

### 2. Clone and sync

```bash
git clone git@github.com:Project-HAL/MXtreme.git
cd MXtreme
uv sync
```

`uv sync` creates a project-local `.venv/`, installs MXtreme **editable** (so `import mxtreme`
resolves to your live `src/mxtreme/`), installs the `dev` dependency group, and resolves `uv.lock`.

The `.venv/` is git-ignored; `uv.lock` **is** committed.

### 3. Run through uv

There is no environment to activate — prefix commands with `uv run`:

```bash
uv run pytest
uv run ruff check .
```

:::{warning}
Don't run a bare `python` or `pip`. That hits your system interpreter, not the project environment.
:::

### 4. Notebooks

Register the kernel once, then work as usual:

```bash
uv run python -m ipykernel install --user --name mxtreme --display-name "MXtreme (uv)"
```

When iterating on the library from inside a notebook, enable autoreload so edits under `src/` are
picked up without a kernel restart:

```python
%load_ext autoreload
%autoreload 2
```

## Rig machines (`mxtreme.scans`)

{mod}`mxtreme.scans.mx_setup` drives the array — wells, sequences, stimulation units — through
MaxWell's `maxlab` Python API.

:::{warning}
`maxlab` **cannot be installed with pip.** It is proprietary and is not published to PyPI or any
other index. It ships with MaxWell's MaxLab Live software: install that on the rig machine and make
sure its Python package is on your `PYTHONPATH`.
:::

Everything else in `mxtreme.scans` runs anywhere, with nothing beyond the core dependencies:

| Module | Needs `maxlab`? | What it does |
|---|---|---|
| {mod}`~mxtreme.scans.electrode_selection` | No | Pick recording electrodes from an activity scan |
| {mod}`~mxtreme.scans.mx_config` | No | Read and write MaxWell `.cfg` electrode files |
| {mod}`~mxtreme.scans.mx_setup` | **Yes** | Configure and run experiments on the array |

So choosing electrodes for a network scan is ordinary laptop work — a plain `pip install mxtreme` is
enough. Importing `mx_setup` without `maxlab` raises an error saying exactly this.

## Building these docs locally

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Then serve them:

```bash
uv run python -m http.server -d docs/_build/html 8000
```

## A note on version floors

Runtime dependencies are declared as lower bounds (`pandas>=2.0`, `numpy>=1.24`) so MXtreme installs
alongside whatever a consumer already has. A fresh resolve therefore pulls the newest compatible
majors, which can differ from what earlier work was tested against — pandas 3.x in particular carries
real breaking changes. The committed `uv.lock` pins exact versions for development and CI so this
doesn't drift silently.

When bumping a floor, run the checks against the new stack before committing the bump:

```bash
uv sync --upgrade-package pandas
uv run pytest
```
