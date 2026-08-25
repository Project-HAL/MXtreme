"""Sphinx configuration for the MXtreme documentation site.

The package's docstrings are native reStructuredText (``:param:``, ``:class:``, ``:meth:`` roles),
so autodoc renders them directly and Napoleon is deliberately absent. MyST-Parser is enabled so the
narrative pages can be Markdown while the docstrings stay RST.

No ``sys.path`` manipulation is needed: ``uv sync`` installs MXtreme editable, so autodoc imports
``mxtreme`` the same way any consumer would.
"""

import mxtreme

project = "MXtreme"
author = "Project-HAL"
copyright = "2026, Project-HAL"
release = mxtreme.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",  # writes .nojekyll so Pages serves _static/ and _sources/
    "myst_parser",
]

exclude_patterns = ["_build"]

# -- autodoc ---------------------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"

# The frozen dataclasses in params.py/identity.py/config.py carry their field documentation in the
# class docstring; without this, autodoc also emits the generated __init__ signature docstring.
autoclass_content = "class"

# -- intersphinx -----------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MyST ------------------------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- HTML ------------------------------------------------------------------------------------

html_theme = "furo"
html_title = f"MXtreme {release}"
# Referenced out of the repo's existing img/ rather than duplicating the 1 MB PNG under docs/.
html_logo = "../img/MXtreme_logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/Project-HAL/MXtreme/",
    "source_branch": "main",
    "source_directory": "docs/",
}
