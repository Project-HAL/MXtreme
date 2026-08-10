"""User-defined preprocessing pipeline.

A :class:`Pipeline` is just an ordered list of step functions (see :mod:`mxtreme.clean`). Calling
:meth:`Pipeline.run` pushes extracted data through those steps, well by well, and saves each cleaned
well to an ``.npz``::

    from functools import partial
    from mxtreme import extract, clean
    from mxtreme.pipeline import Pipeline

    data = extract.extract(raw_h5)
    pipeline = Pipeline([
        clean.normalize_time,
        clean.dac_to_voltage,
        clean.remove_positive_deflections,
        partial(clean.spike_filter, amp_thresh=2e-5),
        clean.remove_spurious_channels,
        clean.build_channel_map,
        clean.bin_spikes,
    ])
    paths = pipeline.run(data, datastore=config.preprocessed_dir)

Steps are ordinary callables, so users can drop them, reorder them, wrap them with
:func:`functools.partial` to change parameters, or supply their own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from mxtreme import io


class Pipeline:
    """An ordered sequence of preprocessing steps applied per well, then saved.

    :param steps: Step callables, each ``(well: dict) -> dict``, applied in order. Wrap parameterised
        steps with :func:`functools.partial` to override their defaults.
    :type steps: list[Callable[[dict], dict]]
    """

    def __init__(self, steps: list[Callable[[dict], dict]]):
        self.steps = list(steps)

    def transform(self, data: dict[int, dict]) -> dict[int, dict]:
        """Apply every step to every well, in place, and return the data. Performs no I/O.

        Useful for inspecting/plotting the result of a (possibly partial) pipeline without writing files.

        :param data: Mapping of well number to well data dict (from :func:`mxtreme.extract.extract`).
        :type data: dict[int, dict]
        :returns: The same ``data`` mapping, with each well transformed by the steps.
        :rtype: dict[int, dict]
        """
        for well_no in list(data):
            well = data[well_no]
            for step in self.steps:
                well = step(well)
            data[well_no] = well
        return data

    def run(
        self,
        data: dict[int, dict],
        *,
        datastore: str | Path,
        overwrite: bool = True,
    ) -> list[Path]:
        """Transform every well (via :meth:`transform`) and save the cleaned data.

        Wells that end up with no spikes after the steps are skipped (with a warning) rather than saved.

        :param data: Mapping of well number to well data dict (from :func:`mxtreme.extract.extract`).
        :type data: dict[int, dict]
        :param datastore: Directory to save cleaned ``.npz`` files under (e.g. ``config.preprocessed_dir``).
        :type datastore: str or Path
        :param overwrite: Passed through to :func:`mxtreme.io.save_preprocessed`; if ``False``, existing
            files are suffixed instead of overwritten.
        :type overwrite: bool
        :returns: Paths of the ``.npz`` files written (one per saved well).
        :rtype: list[Path]
        """
        self.transform(data)

        saved: list[Path] = []
        for well_no in list(data):
            well = data[well_no]
            if len(well["data"]) == 0:
                print(f"Well {well_no} has 0 spikes after the pipeline. Skipping...")
                continue
            saved.append(io.save_preprocessed(datastore, well, overwrite=overwrite))
        return saved
