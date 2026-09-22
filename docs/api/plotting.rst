Plotting
========

Every plotting helper takes an optional ``ax``, so the same function serves an interactive notebook
and a report section without re-implementing the figure. :mod:`mxtreme.analysis.report` builds its
PDF entirely out of these.

.. automodule:: mxtreme.visualizations
   :members:

Quick-look raster
-----------------

The one view here that needs no preprocessing: spike counts over time read straight from a raw
recording's spike table, for checking a scan or an experiment right after it was recorded.

.. automodule:: mxtreme.raster
   :members:
