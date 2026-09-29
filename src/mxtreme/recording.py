"""A ``Recording`` object is a view over a single preprocessed ``.npz``.

It holds the arrays and identity of one well's cleaned recording and exposes a few
derived views such as the array-wide spike detection rate array (``asdr``) and the event data (``event_df``). 

Reocording phases are a user input (see :mod:`mxtreme.phases`), defaulting to a single ``"full"`` phase spanning the
entire recording.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from mxtreme.phases import Phases, PhaseSpec, phases_from_spec


def _item(value):
    """Return a Python scalar from a 0-d or size-1 array (or the value itself)."""
    arr = np.asarray(value)
    return arr.item() if arr.size == 1 else value


def _scalar_float(value) -> float:
    """Return the first element of an array-like as a float (handles ``(1,)`` and scalar arrays)."""
    return float(np.asarray(value).ravel()[0])


class Recording:
    """Data-access wrapper around one preprocessed recording (one well).

    :param id: User-assigned recording id.
    :param exp_data: Preprocessed data dict, as returned by :func:`mxtreme.io.load_preprocessed`.
    :param phases: Optional pre-built :class:`~mxtreme.phases.Phases` labelling parts of the recording.
    :param phase_tags: Optional phase spec (an ordered ``name -> definition`` mapping; each phase has a
        ``start`` and optional ``end`` boundary given as minutes, an event key, or a key/value
        matcher -- see :func:`~mxtreme.phases.phases_from_spec`). When given, phases are resolved here
        from this recording's own ``event_df``, so callers no longer need to build a provisional
        ``Recording`` just to read its event tags. Mutually exclusive with ``phases``.

    When neither ``phases`` nor ``phase_tags`` is supplied, phases default to a spec embedded in
    ``exp_data`` under ``"phase_spec"`` (written at preprocessing time from the ``Phases`` metadata key
    and resolved here via :func:`~mxtreme.phases.phases_from_spec`). When no embedded spec is present
    either, a single ``"full"`` phase spans the whole recording. An explicit ``phases`` / ``phase_tags``
    argument overrides the embedded spec.
    """

    def __init__(
        self,
        id,
        exp_data: dict,
        phases: Optional[Phases] = None,
        *,
        phase_tags: Optional[PhaseSpec] = None,
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
        # Resolved here (not before construction) so a spec can be turned into Phases using this
        # recording's own event_df. Priority: an explicit ``phases`` object, else an explicit
        # ``phase_tags`` spec, else a phase spec embedded in the preprocessed data (the default source,
        # written at preprocessing time), else a single ``"full"`` phase spanning the recording.
        frames = self.spike_data["frameno"]
        start_frame = int(np.min(frames)) if len(frames) else 0
        end_frame = int(np.max(frames)) if len(frames) else 0

        # Phase spec embedded in the npz metadata (absent in older stores / synthetic fixtures).
        embedded_spec = _item(exp_data["phase_spec"]) if "phase_spec" in exp_data else None

        if phases is not None and phase_tags is not None:
            raise ValueError("Pass either `phases` or `phase_tags`, not both.")

        spec = phase_tags if phase_tags is not None else embedded_spec
        if phases is not None:
            self.phases = phases
        elif isinstance(spec, Mapping) and spec:
            self.phases = phases_from_spec(
                spec, self.event_df,
                start_frame=start_frame, end_frame=end_frame, samp_rate=self.samp_rate,
            )
        else:
            self.phases = Phases.full(start_frame, end_frame)
