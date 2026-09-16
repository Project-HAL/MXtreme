"""Turning a :class:`~mxtreme.stimulation.timeline.Timeline` into a MaxLab sequence, and routing
the stimulation units it needs.

Everything here talks to the rig, so ``maxlab`` is imported on first use. Build the timeline with
:mod:`mxtreme.stimulation.timeline` (which needs nothing) and hand it in here.

Amplitudes are in mV per phase and converted with the DAC's least significant bit, as MaxLab's
own examples do.
"""

from __future__ import annotations

from collections.abc import Iterable

from mxtreme.stimulation.timeline import (
    STIM_CLOCK_HZ,
    PulseGeometry,
    RegionUnits,
    region_timeline,
)

# Event ids are opaque to the analysis side, which matches events on their text, but they must not
# collide with the ids the scans module hands out (small integers). Start high.
_next_event_id = 100


def _require_maxlab():
    from mxtreme.scans.activity_scan import _require_maxlab

    return _require_maxlab()


def next_event_id() -> int:
    """A fresh ``mx.Event`` id, unique within this process."""
    global _next_event_id
    _next_event_id += 1
    return _next_event_id - 1


def dac_lsb_mv() -> float:
    """The DAC's least significant bit in mV, from the system; 1.0 if it does not answer."""
    mx = _require_maxlab()
    try:
        return float(mx.query_DAC_lsb_mV())
    except (TypeError, ValueError):
        return 1.0


class MaxlabEmitter:
    """The :meth:`Timeline.compile` target that appends to a ``maxlab.Sequence``."""

    def __init__(self, seq, mx) -> None:
        self.seq = seq
        self.mx = mx

    def delay(self, samples: int) -> None:
        self.seq.append(self.mx.DelaySamples(samples))

    def dac(self, dac: int, value: int) -> None:
        self.seq.append(self.mx.DAC(dac, value))

    def connect(self, unit: int, dac: int) -> None:
        self.seq.append(
            self.mx.StimulationUnit(unit).power_up(True).connect(True).set_voltage_mode().dac_source(dac)
        )

    def disconnect(self, unit: int) -> None:
        self.seq.append(self.mx.StimulationUnit(unit).power_up(False).connect(False))


def build_region_sequence(
    token: str,
    well: int,
    regions: Iterable[RegionUnits],
    delays_sec: dict[str, float],
    num_events: int,
    pulse_hz: float,
    phase_us: float,
    amplitude_mv: float | None = None,
    pulses_per_burst: int = 1,
    burst_hz: float = 20.0,
    polarity: str = "anodic-first",
    persistent: bool = True,
):
    """Register a sequence on the system: a pulse train to each region, each at its own delay.

    See :func:`mxtreme.stimulation.timeline.region_timeline` for how the trains and the unit
    switching are laid out. The sequence carries an ``mx.Event`` at its start and end whose text
    is key-value pairs (``start_stimulation <token> events <n> ...``), which is how the analysis
    side finds every presentation in the ``.h5``.

    :param token: The sequence's name on the system, and the event's ``start_stimulation`` value.
    :param well: Well the events are tagged with.
    :returns: ``(sequence, seconds)``: the registered ``mx.Sequence`` and its length.
    """
    mx = _require_maxlab()
    regions = list(regions)
    geometry = PulseGeometry.of(phase_us, pulse_hz, pulses_per_burst, burst_hz)
    timeline, switched = region_timeline(
        regions, delays_sec, num_events, geometry, dac_lsb_mv(), amplitude_mv, polarity
    )

    # Drop any sequence already registered under this name, then make a fresh one.
    seq = mx.Sequence(token, persistent=False)
    del seq
    seq = mx.Sequence(token, persistent=persistent)
    emitter = MaxlabEmitter(seq, mx)

    if not switched:
        for region in regions:
            for unit, dac in region.connections():
                emitter.connect(unit, dac)

    # Read back from the h5 as key-value pairs, so: an even list of words, no spaces in a value.
    description = " ".join(
        f"{r.role} drive_dac{r.drive_dac}"
        + (f":return_dac{r.return_dac}" if r.return_units else "")
        + f":amp_mV{amplitude_mv if amplitude_mv is not None else r.amplitude_mv}"
        + f":delay_s{delays_sec.get(r.role, 0.0)}"
        for r in regions
    )
    seq.append(
        mx.Event(
            well,
            1,
            next_event_id(),
            f"start_stimulation {token} events {num_events} hz {pulse_hz} "
            f"per_burst {pulses_per_burst} burst_hz {burst_hz} phase_us {phase_us} "
            f"polarity {polarity} units {'switched' if switched else 'held'} {description}",
        )
    )
    timeline.compile(emitter)
    seq.append(mx.Event(well, 1, next_event_id(), f"end_stimulation {token}"))

    if not switched:
        for region in regions:
            for unit, _ in region.connections():
                emitter.disconnect(unit)

    return seq, timeline.end_frame() / STIM_CLOCK_HZ


def init_well_regions(well: int, rec_elecs, groups: dict[str, list[int]]):
    """Route recording electrodes plus several named groups of stimulation electrodes.

    Like :func:`mxtreme.scans.mx_setup.init_well`, with the stimulation electrodes grouped so the
    stimulation units come back grouped the same way. The units are left powered *down*: a
    sequence powers the ones it needs on its own DAC for the pulse and down again after, which is
    what keeps a pulse on one DAC from reaching a site on another.

    :param well: Well number.
    :param rec_elecs: Recording electrodes, as a list or a ``.cfg`` path.
    :param groups: ``{name: [electrodes]}``, e.g. ``{"US_drive": [...], "US_return": [...]}``.
    Where two stimulation electrodes would need the same stimulation unit, the later one is
    swapped for a neighbour (see :func:`mxtreme.scans.mx_setup.init_well_stim`), so the
    electrodes actually connected are returned alongside the units and may differ from
    ``groups`` by a pitch or two.

    :raises RuntimeError: If an electrode is in two groups, or no clash-free set can be found.
    :returns: The ``mx.Array``, ``{name: [stimulation units]}`` and ``{name: [electrodes used]}``,
        both in the order of ``groups``.
    """
    from mxtreme.scans import mx_setup

    all_stim = [e for elecs in groups.values() for e in elecs]
    if len(all_stim) != len(set(all_stim)):
        raise RuntimeError("a stimulation electrode is in more than one group")

    array, units, used = mx_setup.init_well_stim(well, rec_elecs, all_stim)

    per_group, per_group_electrodes, i = {}, {}, 0
    for name, elecs in groups.items():
        per_group[name] = list(units[i : i + len(elecs)])
        per_group_electrodes[name] = list(used[i : i + len(elecs)])
        i += len(elecs)
    return array, per_group, per_group_electrodes


def mark(well: int, text: str) -> None:
    """Put an event into the recording now, e.g. ``mark(0, "encode_start 0")``."""
    mx = _require_maxlab()
    mx.send(mx.Event(well, 1, next_event_id(), text))
