"""A single recording: a thin, experiment-agnostic view over one preprocessed ``.npz``.

``Recording`` holds the arrays and identity of one well's cleaned recording and exposes a few cheap
derived views (``asdr``, ``event_df``). It makes **no** experiment-specific assumptions: phases are an
injected input (see :mod:`mxtreme.phases`), defaulting to a single ``"full"`` phase spanning the
recording, and experimental conditions are carried through opaquely.

Burst detection, feature extraction, saving, and plotting deliberately live elsewhere:

    from mxtreme.bursting import BurstDetector      # detection + features
    from mxtreme import io                          # save/load burst CSVs
    from mxtreme import visualizations as viz       # ax-aware plots

This keeps ``Recording`` a stable data-access object that those modules (and ``mxtreme.analysis``)
consume.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from mxtreme.phases import Phases, TagSpec, phases_from_event_tags


def _item(value):
    """Return a Python scalar from a 0-d or size-1 array (or the value itself)."""
    arr = np.asarray(value)
    return arr.item() if arr.size == 1 else value


def _scalar_float(value) -> float:
    """Return the first element of an array-like as a float (handles ``(1,)`` and scalar arrays)."""
    return float(np.asarray(value).ravel()[0])


class Recording:
    """Data-access wrapper around one preprocessed recording (one well).

    :param id: Caller-assigned recording id (kept for convenience; not interpreted).
    :param exp_data: Preprocessed data dict, as returned by :func:`mxtreme.io.load_preprocessed`.
    :param phases: Optional pre-built :class:`~mxtreme.phases.Phases` labelling parts of the recording.
    :param phase_tags: Optional event-tag spec (an ordered ``name -> tag`` mapping or ``(name, tag)``
        sequence). When given, phases are resolved here from this recording's own ``event_df`` via
        :func:`~mxtreme.phases.phases_from_event_tags`, so callers no longer need to build a
        provisional ``Recording`` just to read its event tags. Mutually exclusive with ``phases``.
    :param end_tag: Optional event tag marking the end of the final phase (used only with
        ``phase_tags``); the last phase runs to this tag's frame if present, else to the last frame.

    When neither ``phases`` nor ``phase_tags`` is supplied, a single ``"full"`` phase spans the whole
    recording.
    """

    def __init__(
        self,
        id,
        exp_data: dict,
        phases: Optional[Phases] = None,
        *,
        phase_tags: Optional[TagSpec] = None,
        end_tag: Optional[str] = None,
    ) -> None:
        self.id = id

        # --- identity ---
        self.exp_id = _item(exp_data["exp_id"])
        self.chip = _item(exp_data["chip"])
        self.well = _item(exp_data["well"])
        self.DIV = _item(exp_data["DIV"])
        self.plate_date = _item(exp_data["plate_date"])
        self.raw_start = _item(exp_data["raw_start"])
        self.path_to_h5 = _item(exp_data["path_to_h5"])
        self.exp_condition = _item(exp_data["exp_condition"]) if "exp_condition" in exp_data else None

        # --- arrays ---
        self.spike_data = exp_data["spike_data"]
        self.channelmap = exp_data["channelmap"]  # columns: index, channel, electrode, x, y
        self.stim_elecs = exp_data["stim_elecs"]
        self.spike_bin = exp_data["spike_bin"]
        self.eventtime = exp_data["eventtime"]
        self.event_messages = exp_data["event_messages"]
        self.stim_frames = np.asarray(exp_data["stim_frames"]) if "stim_frames" in exp_data else None

        self.rec_t_sec = _scalar_float(exp_data["rec_t_sec"])
        self.samp_rate = _scalar_float(exp_data["samp_rate"])
        self.lsb = _scalar_float(exp_data["lsb"])
        self.bin_size = float(_item(exp_data["bin_size"]))

        # --- derived views ---
        self.asdr = self.spike_bin.mean(axis=0)  # array-wide spike detection rate (per bin)
        self.event_df = pd.DataFrame({"eventtime": self.eventtime, "eventmessage": self.event_messages})

        # --- phases ---
        # Resolved here (not before construction) so a tag spec can be turned into Phases using this
        # recording's own event_df. Priority: an explicit ``phases`` object, else a ``phase_tags`` spec
        # resolved against the events, else a single ``"full"`` phase spanning the recording.
        frames = self.spike_data["frameno"]
        start_frame = int(np.min(frames)) if len(frames) else 0
        end_frame = int(np.max(frames)) if len(frames) else 0

        if phases is not None and phase_tags is not None:
            raise ValueError("Pass either `phases` or `phase_tags`, not both.")

        if phases is not None:
            self.phases = phases
        elif phase_tags is not None:
            self.phases = phases_from_event_tags(
                self.event_df, phase_tags, end_frame=end_frame, end_tag=end_tag
            )
        else:
            self.phases = Phases.full(start_frame, end_frame)
