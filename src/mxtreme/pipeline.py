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

import logging
from pathlib import Path
from time import perf_counter
from typing import Callable

from mxtreme import io

logger = logging.getLogger(__name__)


def _step_name(step: Callable) -> str:
    """Human-readable name for a step, unwrapping :func:`functools.partial`."""
    return getattr(getattr(step, "func", step), "__name__", repr(step))


class Pipeline:
    """An ordered sequence of preprocessing steps applied per well, then saved.

    :param steps: Step callables, each ``(well: dict) -> dict``, applied in order. Wrap parameterised
        steps with :func:`functools.partial` to override their defaults.
    :type steps: list[Callable[[dict], dict]]
    """

    def __init__(self, steps: list[Callable[[dict], dict]]):
        self.steps = list(steps)

    def _apply_steps(self, well_no: int, well: dict) -> dict:
        """Apply every step to one well, logging each step's spike attrition and timing.

        Records a per-step log (step name, spikes before/after, removed, seconds) at INFO level and stashes
        it in ``well['step_log']``, then logs which step removed the most spikes.

        :param well_no: Well number (for log context).
        :param well: The well data dict to transform (mutated in place).
        :returns: The transformed well.
        :rtype: dict
        """
        step_log = []
        for step in self.steps:
            name = _step_name(step)
            n_before = len(well["data"])
            t = perf_counter()
            well = step(well)
            seconds = perf_counter() - t
            n_after = len(well["data"])
            pct = (n_after - n_before) / n_before * 100 if n_before else 0.0
            step_log.append(
                {"step": name, "n_before": n_before, "n_after": n_after,
                 "removed": n_before - n_after, "seconds": seconds}
            )
            logger.info("well %s | %-28s %s -> %s (%+.1f%%) %.2fs",
                        well_no, name, f"{n_before:,}", f"{n_after:,}", pct, seconds)

        well["step_log"] = step_log
        reducers = [r for r in step_log if r["removed"] > 0]
        if reducers:
            top = max(reducers, key=lambda r: r["removed"])
            logger.info("well %s | biggest reducer: %s removed %s spikes",
                        well_no, top["step"], f"{top['removed']:,}")
        return well

    def transform(self, data: dict[int, dict]) -> dict[int, dict]:
        """Apply every step to every well, in place, and return the data. Performs no I/O.

        Useful for inspecting/plotting the result of a (possibly partial) pipeline without writing files.
        All wells are retained in ``data``.

        :param data: Mapping of well number to well data dict (from :func:`mxtreme.extract.extract`).
        :type data: dict[int, dict]
        :returns: The same ``data`` mapping, with each well transformed by the steps.
        :rtype: dict[int, dict]
        """
        for well_no in list(data):
            data[well_no] = self._apply_steps(well_no, data[well_no])
        return data

    def run(
        self,
        data: dict[int, dict],
        *,
        datastore: str | Path,
        overwrite: bool = True,
        free: bool = True,
    ) -> list[Path]:
        """Transform each well and save the cleaned data, one well at a time.

        Unlike :meth:`transform`, wells are transformed and saved individually so at most one well's
        ``spike_bin`` is held in memory at once. Wells left with no spikes are skipped rather than saved.

        :param data: Mapping of well number to well data dict (from :func:`mxtreme.extract.extract`).
        :type data: dict[int, dict]
        :param datastore: Directory to save cleaned ``.npz`` files under (e.g. ``config.preprocessed_dir``).
        :type datastore: str or Path
        :param overwrite: Passed through to :func:`mxtreme.io.save_preprocessed`; if ``False``, existing
            files are suffixed instead of overwritten.
        :type overwrite: bool
        :param free: If ``True`` (default), drop each well from ``data`` once saved to release its memory
            (notably ``spike_bin``) before processing the next. Set ``False`` to keep transformed wells.
        :type free: bool
        :returns: Paths of the ``.npz`` files written (one per saved well).
        :rtype: list[Path]
        """
        saved: list[Path] = []
        for well_no in list(data):
            well = self._apply_steps(well_no, data[well_no])
            if len(well["data"]) == 0:
                logger.info("Well %s has 0 spikes after the pipeline. Skipping...", well_no)
            else:
                saved.append(io.save_preprocessed(datastore, well, overwrite=overwrite))
            if free:
                del data[well_no]
            else:
                data[well_no] = well
        return saved
