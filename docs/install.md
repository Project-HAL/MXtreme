# Installation

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)**, which manages the environment and a committed lock file so
  everyone works against an identical dependency stack.

MXtreme is not yet published to PyPI, so cloning the repository is the only supported install — see
[Developers](#developers) below.

matplotlib and seaborn are **required**, not optional — they are imported at module level by the
recording, bursting, visualization, and analysis modules, so an install without them cannot import
the package.

% TODO(pypi): restore this section once mxtreme is published to PyPI.
% Release plan and background: mxtreme-claude/claude_configs/mxt_management_and_use.md
%
% ## Users
%
% ```bash
% pip install mxtreme
% ```
%
% ```python
% import mxtreme
% print(mxtreme.__version__)
% ```

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

## Maxlab Live Dependency (`mxtreme.scans` only)

{mod}`mxtreme.scans.mx_setup` drives the array — wells, sequences, stimulation units — through
MaxWell's `maxlab` Python API for activity and network scans.

:::{warning}
`maxlab` **cannot be installed with pip.** It is proprietary and is not published to PyPI or any
other index. It ships with MaxWell's MaxLab Live software: install that on the rig machine and make
sure its Python package is on your `PYTHONPATH`.
:::

Everything else in `mxtreme.scans` runs anywhere, with nothing beyond the core dependencies:


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
