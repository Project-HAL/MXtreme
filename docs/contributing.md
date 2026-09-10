# Contributing

Environment setup lives in [Installation](install.md#developers). This page covers working on the
documentation, and the checks to run before opening a pull request.

## Where documentation content lives

Two places, edited differently:

| To change… | Edit | Example |
|---|---|---|
| Prose, guides, page structure | Markdown under `docs/` | `docs/quickstart.md` |
| Anything under **API reference** | The **docstring in `src/`** | `Recording`'s docstring in `src/mxtreme/recording.py` |

The API pages (`docs/api/*.rst`) are thin lists of `automodule` directives and contain no prose of
their own. To fix wording on an API page, edit the docstring in the source file. You only touch the
`.rst` to **add or remove a module** from the site.

## The local loop

Build:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Serve, in a second terminal:

```bash
uv run python -m http.server -d docs/_build/html 8000
```

Then open <http://localhost:8000>. The server reads from disk, so after a rebuild just refresh the
browser — no restart needed.

Rebuilds are incremental (Sphinx caches parsed documents in `docs/_build/.doctrees`), so after the
first one they take a second or two. Sphinx registers the Python modules as build dependencies, so
editing a **docstring** correctly invalidates the API page that documents it — no cache-clearing
step to remember. If a page ever does look stale, force a full rebuild:

```bash
rm -rf docs/_build
```

### Live reload

`sphinx-autobuild` collapses the loop into one watching, auto-reloading process. It is not part of
the `docs` group by default; add it if you want it:

```bash
uv add --group docs sphinx-autobuild
```

```bash
uv run sphinx-autobuild --watch src docs docs/_build/html
```

`--watch src` is the important flag — without it, docstring edits don't trigger a rebuild.

## Before you push

```bash
uv run sphinx-build -b html -W --keep-going docs docs/_build/html
```

`-W` promotes warnings to errors, which is exactly what CI runs. This catches malformed docstring
RST and cross-references that point at nothing — a `:class:` typo becomes a silent dead link
otherwise. A non-zero exit means fix it before pushing.

## Docstring conventions

Docstrings are **native reStructuredText**, not Google or NumPy style. Napoleon is deliberately not
enabled, so NumPy-style `Parameters\n----------` headings will not render correctly.

Use the field-list form:

```python
def detect(self, recording, phases=None, *, burst_data_dir=None):
    """Detect bursts in ``recording`` and label each by phase.

    :param recording: Source :class:`Recording <mxtreme.recording.Recording>`.
    :param phases: Phase intervals for labelling; defaults to ``recording.phases``.
    :returns: The detected :class:`BurstSet` (features not yet computed).
    """
```

Useful roles, all of which become real links:

| Role | Points at |
|---|---|
| `` :class:`~mxtreme.config.Config` `` | A class (`~` shows only the last component) |
| `` :func:`mxtreme.io.register` `` | A function |
| `` :meth:`Phases.label_for` `` | A method |
| `` :mod:`mxtreme.clean` `` | A module |

Double backticks are literal text; use them for values, filenames, and parameter names. Indent
continuation lines of a `:param:` by four spaces. Leave a blank line before and after a `::` literal
block.

Markdown pages use MyST, so the same cross-references are written `` {class}`~mxtreme.config.Config` ``.

## Adding a module to the API reference

Add an `automodule` block to the appropriate file in `docs/api/`, grouped by pipeline stage:

```rst
Recording
---------

.. automodule:: mxtreme.recording
   :members:
```

Some modules are excluded on purpose — see the "Not documented here" section of the API reference
index. Notably, lab-specific paths and legacy constants are kept off the published site.

## Continuous integration

`.github/workflows/docs.yml` has two jobs.

**`build`** runs on every push to `main`, every pull request, and on demand. It installs from the
lock file with `uv sync --locked` (a stale `uv.lock` fails the build rather than silently
re-resolving), runs the strict Sphinx build, and uploads the rendered site as a workflow artifact —
so you can download and preview the built HTML from any PR's run summary.

**`deploy`** publishes to GitHub Pages. It runs only on `main`, so it is *skipped* — not failed — on
pull requests.

### The published site

The site is live at <https://project-hal.github.io/MXtreme/>. Every merge to `main` rebuilds and
redeploys it automatically — there is nothing to publish by hand and no version to bump.

The only piece of setup that lives outside the repository is **Settings → Pages → Source** =
"GitHub Actions". If `deploy` ever fails with a Pages error, check that first.

## Code changes

1. **Branch** off `main`.
2. **Sync** after pulling, so dependencies match the lock file: `uv sync`.
3. **Make changes** in `src/mxtreme/`, with tests alongside in `testing/`.
4. **Run the checks**:

   ```bash
   uv run pytest
   ```

   ```bash
   uv run ruff check .
   ```

5. **Open a pull request** describing the change and the reasoning behind it.

Dependencies are managed through uv so `pyproject.toml` and `uv.lock` stay in sync — `uv add
<package>`, `uv add --group dev <package>`, `uv add --group docs <package>`. Commit both files
together. See [Installation](install.md#a-note-on-version-floors) for the version-floor policy.
