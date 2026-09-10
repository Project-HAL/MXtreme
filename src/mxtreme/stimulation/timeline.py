"""Stimulation as a timeline of commands, compiled into a MaxLab sequence.

A MaxLab sequence is a list of commands with delays between them. Writing one directly gets hard
as soon as two things overlap -- two regions pulsing with an offset, say, or stimulation units that
have to be connected just before a pulse and released just after. So stimulation is described here
as *what happens at which frame*, and :meth:`Timeline.compile` works the delays out.

Timing runs on the stimulation clock, :data:`STIM_CLOCK_HZ`: ``DelaySamples`` counts 50 us steps
whatever the device records at (``maxlab.system.DelaySamples``: "independent of the data sampling
rate"), so the recording's frames per second must never be used here.

Nothing in this module needs ``maxlab``. :meth:`Timeline.compile` hands each command to an
*emitter* -- :class:`ListEmitter` for tests and dry runs, :class:`mxtreme.stimulation.sequences.MaxlabEmitter`
on the rig.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: The stimulation clock: 50 us per step on MaxOne and MaxTwo alike.
STIM_CLOCK_HZ = 20000

#: The DAC's mid-scale, meaning no stimulation.
DAC_REST = 512

#: ``DelaySamples`` takes a uint16, so longer delays are chained.
MAX_DELAY = (1 << 16) - 1

#: The two orders a biphasic pulse can go.
POLARITIES = ("anodic-first", "cathodic-first")


def amplitude_bits(amplitude_mv: float, lsb_mv: float) -> int:
    """DAC steps for an amplitude in mV, given the DAC's least significant bit in mV.

    :raises ValueError: If the amplitude is outside what the DAC can swing from mid-scale.
    """
    bits = round(amplitude_mv / lsb_mv)
    if not 0 <= bits <= 511:
        raise ValueError(f"{amplitude_mv} mV is {bits} DAC steps, outside 0..511")
    return bits


@dataclass(frozen=True)
class PulseGeometry:
    """A pulse train's timings in stimulation-clock frames.

    :param phase_frames: One phase of the biphasic pulse.
    :param interval_frames: Between pulse events.
    :param burst_frames: Between the pulses of one event; 0 when an event is a single pulse.
    :param pulses_per_burst: Pulses per event.
    """

    phase_frames: int
    interval_frames: int
    burst_frames: int
    pulses_per_burst: int

    @classmethod
    def of(
        cls, phase_us: float, pulse_hz: float, pulses_per_burst: int = 1, burst_hz: float = 20.0
    ) -> PulseGeometry:
        """From the physical numbers.

        :raises ValueError: If the pulses of a burst would overlap.
        """
        phase = max(1, round(phase_us * STIM_CLOCK_HZ / 1e6))
        interval = round(STIM_CLOCK_HZ / pulse_hz)
        burst = round(STIM_CLOCK_HZ / burst_hz) if pulses_per_burst > 1 else 0
        if pulses_per_burst > 1 and burst < 2 * phase:
            raise ValueError(f"{burst_hz} Hz within a burst does not leave room for {phase_us} us phases")
        return cls(phase, interval, burst, int(pulses_per_burst))

    @property
    def event_span(self) -> int:
        """Frames from an event's first pulse starting to its last ending."""
        return (self.pulses_per_burst - 1) * self.burst_frames + 2 * self.phase_frames


@dataclass(frozen=True)
class Step:
    """One command at one frame. ``order`` sorts commands within a frame: connections first,
    DAC steps, then disconnections."""

    frame: int
    order: int
    index: int
    kind: str
    args: tuple


class Timeline:
    """Commands at absolute frames, in any order of construction."""

    BEFORE, DAC, AFTER = 0, 1, 2

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._seen: set[tuple] = set()

    def command(self, frame: int, kind: str, *args, order: int = DAC) -> None:
        """Place a command. A DAC step identical to one already at that frame is dropped: two
        regions sharing a DAC put the same steps on it, and the sequence needs them once."""
        key = (int(frame), kind, args)
        if kind == "dac" and key in self._seen:
            return
        self._seen.add(key)
        self.steps.append(Step(int(frame), order, len(self.steps), kind, args))

    def step(self, frame: int, dac: int, value: int) -> None:
        self.command(frame, "dac", int(dac), int(value))

    def connect(self, frame: int, unit: int, dac: int) -> None:
        """Power a stimulation unit up on a DAC, before anything else at that frame."""
        self.command(frame, "connect", int(unit), int(dac), order=self.BEFORE)

    def disconnect(self, frame: int, unit: int) -> None:
        """Power a stimulation unit down, after everything else at that frame."""
        self.command(frame, "disconnect", int(unit), order=self.AFTER)

    def pulse(self, frame: int, dac: int, bits: int, phase_frames: int, anodic_first: bool = True) -> None:
        """A biphasic, charge-balanced square pulse: one phase up and one down, or the reverse.

        Anodic-first was the more effective order for voltage stimulation on these arrays
        (Ronchi et al. 2019, Fig 2) and is the default.
        """
        first = bits if anodic_first else -bits
        self.step(frame, dac, DAC_REST + first)
        self.step(frame + phase_frames, dac, DAC_REST - first)
        self.step(frame + 2 * phase_frames, dac, DAC_REST)

    def train(
        self,
        start_frame: int,
        dac: int,
        bits: int,
        geometry: PulseGeometry,
        num_events: int,
        anodic_first: bool = True,
    ) -> None:
        """``num_events`` events ``geometry.interval_frames`` apart, each a burst of
        ``geometry.pulses_per_burst`` pulses."""
        for onset in self.event_frames(start_frame, num_events, geometry.interval_frames):
            for j in range(geometry.pulses_per_burst):
                self.pulse(onset + j * geometry.burst_frames, dac, bits, geometry.phase_frames, anodic_first)

    @staticmethod
    def event_frames(start_frame: int, num_events: int, interval_frames: int) -> list[int]:
        return [start_frame + k * interval_frames for k in range(num_events)]

    def end_frame(self) -> int:
        return max((s.frame for s in self.steps), default=0)

    def ordered(self) -> list[Step]:
        """The steps as they will be emitted: by frame, then order, then construction."""
        return sorted(self.steps, key=lambda s: (s.frame, s.order, s.index))

    def compile(self, emitter) -> None:
        """Hand every command to ``emitter`` in order, with ``emitter.delay(n)`` between frames.

        The emitter needs ``delay``, ``dac``, ``connect`` and ``disconnect`` methods.
        """
        now = 0
        for step in self.ordered():
            gap = step.frame - now
            while gap > 0:
                chunk = min(gap, MAX_DELAY)
                emitter.delay(chunk)
                gap -= chunk
            getattr(emitter, step.kind)(*step.args)
            now = step.frame


class ListEmitter:
    """Collects what :meth:`Timeline.compile` emits, as ``(kind, *args)`` tuples."""

    def __init__(self) -> None:
        self.commands: list[tuple] = []

    def delay(self, samples: int) -> None:
        self.commands.append(("delay", samples))

    def dac(self, dac: int, value: int) -> None:
        self.commands.append(("dac", dac, value))

    def connect(self, unit: int, dac: int) -> None:
        self.commands.append(("connect", unit, dac))

    def disconnect(self, unit: int) -> None:
        self.commands.append(("disconnect", unit))


@dataclass
class RegionUnits:
    """One stimulation site as the sequence builder needs it.

    A grid site has only driven units. A focal site also has return units on a second DAC, which
    receive the pulse inverted so that the field is confined to the site.

    :param role: Name of the region, e.g. ``"US"``.
    :param drive_dac: DAC the driven units are on.
    :param drive_units: Stimulation units under the driven electrodes.
    :param amplitude_mv: Amplitude per phase, mV.
    :param return_dac: DAC the return units are on, if any.
    :param return_units: Stimulation units under the return electrodes.
    """

    role: str
    drive_dac: int
    drive_units: list[int]
    amplitude_mv: float
    return_dac: int | None = None
    return_units: list[int] = field(default_factory=list)

    def connections(self) -> list[tuple[int, int]]:
        """``(unit, dac)`` for every unit of the site."""
        return [(u, self.drive_dac) for u in self.drive_units] + [
            (u, self.return_dac) for u in self.return_units
        ]


def region_timeline(
    regions: Iterable[RegionUnits],
    delays_sec: dict[str, float],
    num_events: int,
    geometry: PulseGeometry,
    lsb_mv: float,
    amplitude_mv: float | None = None,
    polarity: str = "anodic-first",
) -> tuple[Timeline, bool]:
    """A pulse train to each region, each starting at its own delay, as a timeline.

    Units are connected only while they are pulsed, so a DAC shared between regions only ever
    reaches the region meant to fire. When every region has the same timing that is one connection
    at the start and one release at the end (``switched`` is ``False``, and the caller does the
    connecting around the timeline). When two regions with different timing share a DAC -- a
    pairing with a nonzero offset on focal sites -- each region's units are connected just before
    each of its events and released just after (``switched`` is ``True``). That relies on the unit
    switching settling within the offset, so keep such offsets above a millisecond or so.

    :param regions: The sites this sequence stimulates.
    :param delays_sec: ``{role: seconds}`` each region's train starts after the sequence does.
    :param num_events: Pulse events per train.
    :param geometry: The train's timings.
    :param lsb_mv: The DAC's least significant bit, mV.
    :param amplitude_mv: When given, every region uses it (calibration); otherwise its own.
    :param polarity: ``"anodic-first"`` or ``"cathodic-first"`` on the driven electrodes; return
        electrodes get the opposite.
    :returns: The timeline, and whether units are switched per event.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}, not {polarity!r}")
    regions = list(regions)
    anodic_first = polarity == "anodic-first"
    starts = {r.role: round(delays_sec.get(r.role, 0.0) * STIM_CLOCK_HZ) for r in regions}
    dacs_used = [d for r in regions for _, d in r.connections()]
    shared_dac = len(dacs_used) != len(set(dacs_used))
    switched = shared_dac and len(set(starts.values())) > 1

    timeline = Timeline()
    for region in regions:
        bits = amplitude_bits(amplitude_mv if amplitude_mv is not None else region.amplitude_mv, lsb_mv)
        start = starts[region.role]
        timeline.train(start, region.drive_dac, bits, geometry, num_events, anodic_first)
        if region.return_units:
            timeline.train(start, region.return_dac, bits, geometry, num_events, not anodic_first)
        if switched:
            for onset in Timeline.event_frames(start, num_events, geometry.interval_frames):
                for unit, dac in region.connections():
                    timeline.connect(onset, unit, dac)
                    timeline.disconnect(onset + geometry.event_span, unit)
    return timeline, switched


def waveform(
    delays_sec: dict[str, float],
    num_events: int,
    geometry: PulseGeometry,
    amplitudes_mv: dict[str, float],
    polarity: str = "anodic-first",
    with_return: bool = False,
    roles: Sequence[str] | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    """``{trace: (t_sec, mV)}`` of the DAC output for drawing, as a step function.

    One trace per role; with ``with_return``, a second ``"<role> return"`` trace per role with the
    polarity inverted, which is what a focal site's ring receives.
    """
    phase = geometry.phase_frames / STIM_CLOCK_HZ
    first = 1.0 if polarity == "anodic-first" else -1.0
    out = {}
    for role in roles or list(amplitudes_mv):
        amp = amplitudes_mv[role]
        t, v = [0.0], [0.0]
        for onset in Timeline.event_frames(
            round(delays_sec.get(role, 0.0) * STIM_CLOCK_HZ), num_events, geometry.interval_frames
        ):
            for k in range(geometry.pulses_per_burst):
                t0 = (onset + k * geometry.burst_frames) / STIM_CLOCK_HZ
                t += [t0, t0, t0 + phase, t0 + phase, t0 + 2 * phase, t0 + 2 * phase]
                v += [0.0, first * amp, first * amp, -first * amp, -first * amp, 0.0]
        out[role] = (t, v)
        if with_return:
            out[f"{role} return"] = (list(t), [-x for x in v])
    return out
