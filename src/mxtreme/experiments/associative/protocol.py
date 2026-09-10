"""The associative protocol, as data: where the regions are and what fires when.

Everything that decides *what* the run will do is computed here from
:class:`~mxtreme.experiments.associative.params.AssociativeParams`, and nowhere else. :mod:`.run`
executes it, :mod:`.preview` draws it, :mod:`.report` reads a recording against it. Because all
three use the same functions, the picture the preview shows is the run that happens.

Nothing here needs ``maxlab``.

Roles and tokens
----------------
The three regions play US, CS and NS. A presentation is a pulse train to one region, or to CS
and US together with US delayed by ``dt_cs_us``. Each kind of presentation is one MaxLab sequence,
named by token:

    probe_us, probe_cs, probe_ns        one region alone, ``t_probe`` long
    stim_cs_us, stim_ns, ...            training presentations, ``t_stim`` long
    stim_us_a100_ac, ...                  calibration: one region, one amplitude, one polarity

Blocks
------
The run is a list of blocks. In conditioning mode:

    pre          ``pre_min`` of nothing
    baseline     every ``probe_roles`` alone, ``probe_reps`` times, ``probe_iti`` apart
    encode       ``encode_cycles`` cycles of: ``encode_pattern`` presentations ``encode_iti``
                 apart, then ``encode_cycle_rest`` of nothing
    decay        ``decay_min`` of nothing
    retrieval_1  ``retrieval_roles`` alone, ``probe_reps`` times
    rest_1       ``retrieval_interval_min`` of nothing
    ...          up to ``num_retrievals``
    post         ``post_min`` of nothing

The encode block is where the culture is meant to learn, and the in vitro literature says that
takes tens of minutes of intermittent stimulation, not a few presentations: 0.2-0.33 Hz for 10 min
then 5 min off, repeated for hours (le Feber 2010); 0.2 Hz for 15 min, or 100 Hz bursts every 5 s
for 10-15 min, four cycles an hour apart (le Feber 2015); 0.3-1 Hz until the response appears,
within tens of minutes (Shahaf & Marom 2001). The cycle parameters exist so that regime can be
written without touching code.

Calibration mode replaces everything between pre and post with one block per region, stepping
through ``calibration_amplitudes_mv`` and ``calibration_polarities`` with single pulses in a
shuffled order, as Ronchi et al. 2019 did so drift does not read as dose.

Connectivity mode is the gate between the two: one ``check`` block of single pulses at each
region's chosen amplitude, ``check_reps`` each, interleaved. :mod:`.report` turns it into a
cross-talk matrix -- how much stimulating each site alone drives the other two -- and says whether
the sites are independent enough to condition, and whether a path between CS and US exists at all.
Run it after calibration and before committing hours to conditioning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mxtreme.experiments.associative.params import PAIR, ROLES, AssociativeParams
from mxtreme.stimulation.timeline import PulseGeometry, waveform

# The array. Electrode e sits at column e % COLS, row e // COLS; a .cfg's coordinates are
# column * PITCH_UM, row * PITCH_UM.
COLS, ROWS, PITCH_UM = 220, 120, 17.5
NUM_ELECTRODES = COLS * ROWS

POLARITY_TAG = {"anodic-first": "ac", "cathodic-first": "ca"}


# ---------------------------------------------------------------
#                            GEOMETRY
# ---------------------------------------------------------------


def electrode_xy(electrode: int) -> tuple[float, float]:
    """(x, y) in um of an electrode number."""
    return (electrode % COLS) * PITCH_UM, (electrode // COLS) * PITCH_UM


def electrode_at(col: int, row: int) -> int | None:
    """Electrode number at a column and row, or None off the array."""
    if 0 <= col < COLS and 0 <= row < ROWS:
        return row * COLS + col
    return None


def nearest_electrode(x_um: float, y_um: float) -> int:
    col = min(COLS - 1, max(0, round(x_um / PITCH_UM)))
    row = min(ROWS - 1, max(0, round(y_um / PITCH_UM)))
    return electrode_at(col, row)


def stim_grid(center_um, size: int, gap: int) -> list[int]:
    """A size x size grid of electrodes around a point, ``gap`` empty electrodes between
    neighbours, centred on the nearest electrode (an even size extends one step less to the right
    and down than to the left and up).

    :raises ValueError: If any corner falls off the array.
    """
    centre = nearest_electrode(*center_um)
    col0, row0 = centre % COLS, centre // COLS
    step = gap + 1
    offsets = [(i - size // 2) * step for i in range(size)]
    grid = []
    for dy in offsets:
        for dx in offsets:
            electrode = electrode_at(col0 + dx, row0 + dy)
            if electrode is None:
                raise ValueError(f"stim grid at {center_um} um runs off the array")
            grid.append(electrode)
    return grid


def stim_site(center_um, site: dict) -> tuple[list[int], list[int]]:
    """The (drive, return) electrodes of one region's stimulation site.

    ``grid`` (``{"shape": "grid", "size": 3, "gap": 2}``): a size x size grid, all driven with the
    same pulse; no return.

    ``focal`` (``{"shape": "focal", "inner": 2, "inner_gap": 0, "return_radius": 3,
    "return_points": "corners"}``): an inner x inner block of driven electrodes (``inner_gap``
    empty electrodes between them), ringed by return electrodes that get the same pulse inverted.
    Seen from further away than the ring the charges cancel, so the field is confined to the site
    (Ronchi et al. 2019). ``"corners"`` puts one return electrode at each corner of the square
    ``return_radius`` electrodes out; ``"corners+edges"`` adds the four edge midpoints. Adjacent
    electrodes can land on the same stimulation unit, which routing refuses; ``inner_gap`` 1 is the
    first thing to try then.
    """
    shape = site.get("shape", "grid")
    if shape == "grid":
        return stim_grid(center_um, int(site.get("size", 3)), int(site.get("gap", 2))), []
    if shape != "focal":
        raise ValueError(f"stim_site shape must be 'grid' or 'focal', not {shape!r}")

    inner = int(site.get("inner", 2))
    inner_gap = int(site.get("inner_gap", 0))
    radius = int(site.get("return_radius", 3))
    if radius <= (inner // 2) * (inner_gap + 1):
        raise ValueError("stim_site return_radius must reach outside the inner block")
    drive = stim_grid(center_um, inner, inner_gap)
    centre = nearest_electrode(*center_um)
    col0, row0 = centre % COLS, centre // COLS
    points = [(-radius, -radius), (radius, -radius), (-radius, radius), (radius, radius)]
    if site.get("return_points", "corners") == "corners+edges":
        points += [(0, -radius), (0, radius), (-radius, 0), (radius, 0)]
    ring = []
    for dx, dy in points:
        electrode = electrode_at(col0 + dx, row0 + dy)
        if electrode is None:
            raise ValueError(f"return ring at {center_um} um runs off the array")
        ring.append(electrode)
    return drive, ring


#: Stimulation units on the chip. Every stimulation electrode needs one, so this is what limits
#: how large a stimulation site can be: three regions share these.
STIM_UNITS = 32


def site_footprint(site: dict) -> dict:
    """How large a stimulation site is, and what it costs.

    The point of the numbers is to be compared with ``region_radius_um``, the radius over which
    the region's response is *measured*. A site much smaller than that stimulates a point and
    measures a disc, which is a mismatch if the three sites are meant to be regions rather than
    single neurons.

    :returns: ``{"driven_span_um", "ring_radius_um", "driven", "return", "units"}``.
    """
    drive, ring = stim_site((1750.0, 1000.0), site)
    xy = [electrode_xy(e) for e in drive]
    span = max(
        max(p[0] for p in xy) - min(p[0] for p in xy),
        max(p[1] for p in xy) - min(p[1] for p in xy),
    )
    ring_um = separation_um(electrode_xy(ring[0]), (1750.0, 1000.0)) if ring else 0.0
    return {
        "driven_span_um": span,
        "ring_radius_um": ring_um,
        "driven": len(drive),
        "return": len(ring),
        "units": len(drive) + len(ring),
    }


def within(electrodes, center_um, radius_um: float) -> list[int]:
    """Those of ``electrodes`` inside a circle."""
    cx, cy = center_um
    return [
        e
        for e in electrodes
        if (electrode_xy(e)[0] - cx) ** 2 + (electrode_xy(e)[1] - cy) ** 2 <= radius_um**2
    ]


def separation_um(a_um, b_um) -> float:
    return ((a_um[0] - b_um[0]) ** 2 + (a_um[1] - b_um[1]) ** 2) ** 0.5


@dataclass
class RegionSpec:
    """One region: where it is, what stimulates it, what records it."""

    role: str
    center_um: tuple[float, float]
    drive_dac: int
    drive_electrodes: list[int]
    return_dac: int | None = None
    return_electrodes: list[int] = field(default_factory=list)
    rec_electrodes: list[int] = field(default_factory=list)
    amplitude_mv: float = 0.0

    @property
    def stim_electrodes(self) -> list[int]:
        """Every electrode the site stimulates through, driven and return."""
        return self.drive_electrodes + self.return_electrodes

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "center_um": list(self.center_um),
            "drive_dac": self.drive_dac,
            "drive_electrodes": self.drive_electrodes,
            "return_dac": self.return_dac,
            "return_electrodes": self.return_electrodes,
            "stim_electrodes": self.stim_electrodes,
            "rec_electrodes": self.rec_electrodes,
            "amplitude_mv": self.amplitude_mv,
        }


def regions_for(params: AssociativeParams, rec_electrodes=None) -> dict[str, RegionSpec]:
    """The three regions from the parameters.

    ``regions`` gives each role a centre in um; the stimulation site is ``stim_site`` around it;
    recording electrodes are those of ``rec_electrodes`` (default the parameters' own) within
    ``region_radius_um``. A grid site takes its role's DAC from ``region_dacs``; a focal site uses
    ``drive_dac`` and ``return_dac``, the same two for every region.

    :raises ValueError: On a missing role, or centres closer than ``min_region_separation_um``.
    """
    if rec_electrodes is None:
        rec_electrodes = params.rec_electrodes or []
    regions = {}
    for role in ROLES:
        if role not in params.regions:
            raise ValueError(f"regions has no centre for {role}")
        center = tuple(params.regions[role])
        drive, ring = stim_site(center, params.stim_site)
        if ring:
            drive_dac, return_dac = int(params.drive_dac), int(params.return_dac)
        else:
            drive_dac, return_dac = int(params.region_dacs[role]), None
        regions[role] = RegionSpec(
            role=role,
            center_um=center,
            drive_dac=drive_dac,
            drive_electrodes=drive,
            return_dac=return_dac,
            return_electrodes=ring,
            rec_electrodes=within(rec_electrodes, center, params.region_radius_um),
            amplitude_mv=float(params.amplitudes_mv[role]),
        )
    for a in ROLES:
        for b in ROLES:
            if a < b:
                d = separation_um(regions[a].center_um, regions[b].center_um)
                if d < params.min_region_separation_um:
                    raise ValueError(
                        f"regions {a} and {b} are {d:.0f} um apart, "
                        f"min_region_separation_um is {params.min_region_separation_um}"
                    )
    return regions


# ---------------------------------------------------------------
#                            SCHEDULE
# ---------------------------------------------------------------


@dataclass
class Presentation:
    """One sequence firing, ``t_sec`` into its block."""

    t_sec: float
    token: str
    roles: list[str]


@dataclass
class Block:
    label: str
    duration_sec: float
    presentations: list[Presentation] = field(default_factory=list)


@dataclass
class Stimulus:
    """What a token does when fired: the input to the sequence builder and the report."""

    token: str
    roles: list[str]
    delays_sec: dict[str, float]
    num_events: int
    pulse_hz: float
    pulses_per_burst: int
    burst_hz: float
    duration_sec: float
    offsets_sec: list[float]  # event onsets, for the pulse-locked readout
    polarity: str
    amplitude_mv: float | None = None  # overrides the regions' own; calibration only

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def token_for(roles, kind: str = "stim", amplitude_mv=None, polarity=None) -> str:
    """``stim_cs_us``, ``probe_ns``, ``stim_us_a300_ac`` ..."""
    name = f"{kind}_" + "_".join(role.lower() for role in roles)
    if amplitude_mv is not None:
        name += f"_a{round(amplitude_mv)}"
    if polarity is not None:
        name += f"_{POLARITY_TAG[polarity]}"
    return name


def roles_in(token: str) -> list[str]:
    """The roles a token stimulates, from its name."""
    return [part.upper() for part in token.split("_")[1:] if part.upper() in ROLES]


def roles_of(item: str) -> list[str]:
    if item.upper() == PAIR:
        return ["CS", "US"]
    if item.upper() in ROLES:
        return [item.upper()]
    raise ValueError(f"'{item}' is not a role or PAIR")


def geometry_of(params: AssociativeParams) -> PulseGeometry:
    return PulseGeometry.of(params.stim_phase_us, params.pulse_hz, params.pulses_per_burst, params.burst_hz)


def stimulus_for(
    params: AssociativeParams,
    roles,
    kind: str = "stim",
    amplitude_mv=None,
    num_events=None,
    polarity=None,
    tag_polarity=False,
) -> Stimulus:
    """One presentation to ``roles``.

    ``kind`` picks the train length: ``"stim"`` is ``t_stim``, ``"probe"`` is ``t_probe``. The
    pairing puts US ``dt_cs_us`` after CS; a negative ``dt_cs_us`` puts US first (a CS pulse
    inside a US burst, as in Chiappalone et al. 2008). ``polarity`` defaults to the parameters'
    and goes into the token only when ``tag_polarity`` is set.
    """
    polarity = polarity or params.pulse_polarity
    geometry = geometry_of(params)
    if num_events is None:
        if kind == "check":
            num_events = 1  # one pulse, so the response to it is unambiguous
        else:
            length = params.t_probe if kind == "probe" else params.t_stim
            num_events = max(1, round(length * params.pulse_hz))
    delays = {role: 0.0 for role in roles}
    if list(roles) == ["CS", "US"]:
        dt = params.dt_cs_us / 1000.0
        delays["CS"], delays["US"] = max(0.0, -dt), max(0.0, dt)
    train_sec = ((num_events - 1) * geometry.interval_frames + geometry.event_span) / 20000.0
    duration = train_sec + max(delays.values())
    offsets = [k * geometry.interval_frames / 20000.0 for k in range(num_events)]
    return Stimulus(
        token=token_for(roles, kind, amplitude_mv, polarity if tag_polarity else None),
        roles=list(roles),
        delays_sec=delays,
        num_events=num_events,
        pulse_hz=params.pulse_hz,
        pulses_per_burst=params.pulses_per_burst,
        burst_hz=params.burst_hz,
        duration_sec=duration,
        offsets_sec=offsets,
        polarity=polarity,
        amplitude_mv=amplitude_mv,
    )


def stimulus_waveform(stim: Stimulus, params: AssociativeParams, amplitudes_mv: dict, with_return=False):
    """``{trace: (t_sec, mV)}`` for drawing one presentation."""
    amps = {r: (stim.amplitude_mv if stim.amplitude_mv is not None else amplitudes_mv[r]) for r in stim.roles}
    return waveform(
        stim.delays_sec, stim.num_events, geometry_of(params), amps, stim.polarity, with_return, stim.roles
    )


def response_windows(params: AssociativeParams) -> dict[str, tuple[float, float]]:
    """``{label: (start_ms, end_ms)}`` relative to a presentation's start, for the report: given as
    ``response_windows_ms``, or the same length before and during a probe and one second after."""
    if params.response_windows_ms:
        return {label: (float(a), float(b)) for label, (a, b) in params.response_windows_ms.items()}
    t_ms = params.t_probe * 1000.0
    return {"base": (-t_ms, 0.0), "during": (0.0, t_ms), "after": (t_ms, t_ms + 1000.0)}


def _probe_block(label, roles, reps, iti, rng, stimuli, tail_sec) -> Block:
    """Every role in ``roles``, ``reps`` times, shuffled within each rep so no role is always
    first. The block ends ``tail_sec`` after the last presentation starts."""
    block = Block(label=label, duration_sec=0.0)
    t = 0.0
    for _ in range(reps):
        order = list(roles)
        rng.shuffle(order)
        for role in order:
            stim = stimuli[token_for(roles_of(role), "probe")]
            block.presentations.append(Presentation(t, stim.token, stim.roles))
            t += iti
    block.duration_sec = t - iti + tail_sec if block.presentations else 0.0
    return block


def build_schedule(params: AssociativeParams) -> tuple[list[Block], dict[str, Stimulus]]:
    """The blocks of the run and the stimuli they use.

    :raises ValueError: If an inter-trial interval cannot hold its presentation and the response
        windows the report will read around it.
    """
    params.validate()
    rng = random.Random(params.seed)
    stimuli: dict[str, Stimulus] = {}
    blocks = [Block("pre", params.pre_min * 60.0)]
    reach = max(b for _, b in response_windows(params).values()) / 1000.0
    pulse_end = params.pulse_window_ms[1] / 1000.0

    if params.mode == "conditioning":
        for role in ROLES:
            probe = stimulus_for(params, [role], "probe")
            stimuli[probe.token] = probe
        for item in dict.fromkeys(params.encode_pattern):
            stim = stimulus_for(params, roles_of(item), "stim")
            stimuli[stim.token] = stim

        # A presentation must be over, and its windows closed, before the next one starts.
        probe_tail = (
            max(max(s.duration_sec for s in stimuli.values() if s.token.startswith("probe")), reach)
            + pulse_end
        )
        stim_tail = (
            max(
                max([s.duration_sec for s in stimuli.values() if s.token.startswith("stim")], default=0.0),
                reach,
            )
            + pulse_end
        )
        if params.probe_iti < probe_tail:
            raise ValueError(
                f"probe_iti = {params.probe_iti}s is shorter than a probe plus its response windows ({probe_tail:.1f}s)"
            )
        if params.encode_pattern and params.encode_iti < stim_tail:
            raise ValueError(
                f"encode_iti = {params.encode_iti}s is shorter than a training presentation plus its response windows ({stim_tail:.1f}s)"
            )

        blocks.append(
            _probe_block(
                "baseline", params.probe_roles, params.probe_reps, params.probe_iti, rng, stimuli, probe_tail
            )
        )

        if params.encode_pattern:
            encode = Block("encode", 0.0)
            t = 0.0
            for cycle in range(params.encode_cycles):
                for item in params.encode_pattern:
                    stim = stimuli[token_for(roles_of(item), "stim")]
                    encode.presentations.append(Presentation(t, stim.token, stim.roles))
                    t += params.encode_iti
                if cycle < params.encode_cycles - 1:
                    t += params.encode_cycle_rest
            encode.duration_sec = t - params.encode_iti + stim_tail
            blocks.append(encode)

        blocks.append(Block("decay", params.decay_min * 60.0))
        for k in range(1, params.num_retrievals + 1):
            blocks.append(
                _probe_block(
                    f"retrieval_{k}",
                    params.retrieval_roles,
                    params.probe_reps,
                    params.probe_iti,
                    rng,
                    stimuli,
                    probe_tail,
                )
            )
            if k < params.num_retrievals:
                blocks.append(Block(f"rest_{k}", params.retrieval_interval_min * 60.0))

    elif params.mode == "connectivity":
        # One pulse per region at its chosen amplitude, many times, interleaved: how much does
        # stimulating each site alone drive the other two? Single pulses rather than trains so
        # each response belongs to one stimulus, and interleaved so drift cannot look like a
        # difference between regions.
        for role in ROLES:
            check = stimulus_for(params, [role], "check")
            stimuli[check.token] = check
        block = Block("check", 0.0)
        t = 0.0
        for _ in range(params.check_reps):
            order = list(ROLES)
            rng.shuffle(order)
            for role in order:
                block.presentations.append(Presentation(t, token_for([role], "check"), [role]))
                t += params.check_iti
        block.duration_sec = t
        blocks.append(block)

    else:  # calibration
        iti = params.calibration_iti
        polarities = list(params.calibration_polarities)
        for role in ROLES:
            block = Block(f"calibrate_{role.lower()}", 0.0)
            t = 0.0
            for _ in range(params.calibration_reps):
                grid = [(amp, pol) for amp in params.calibration_amplitudes_mv for pol in polarities]
                rng.shuffle(grid)
                for amp, pol in grid:
                    stim = stimulus_for(
                        params,
                        [role],
                        "stim",
                        amplitude_mv=amp,
                        num_events=1,
                        polarity=pol,
                        tag_polarity=len(polarities) > 1,
                    )
                    stimuli.setdefault(stim.token, stim)
                    block.presentations.append(Presentation(t, stim.token, [role]))
                    t += iti
            block.duration_sec = t
            blocks.append(block)

    blocks.append(Block("post", params.post_min * 60.0))
    return blocks, stimuli


def block_starts_sec(blocks: list[Block]) -> list[float]:
    starts, t = [], 0.0
    for block in blocks:
        starts.append(t)
        t += block.duration_sec
    return starts


def total_minutes(blocks: list[Block]) -> float:
    return sum(b.duration_sec for b in blocks) / 60.0


def stimulation_budget(blocks: list[Block], stimuli: dict[str, Stimulus]) -> dict[str, dict]:
    """How much stimulation each region receives over the whole run.

    Worth reading before a run rather than after: the number of pulses, and the total time the
    electrodes spend driven, are what decide both whether the culture gets enough training and
    whether the electrodes are being asked to do too much. ``driven_sec`` counts both phases of
    every pulse, so it is the time the site is not at rest.

    :returns: ``{role: {"pulses", "presentations", "driven_sec"}}``.
    """
    out: dict[str, dict] = {}
    for block in blocks:
        for p in block.presentations:
            stim = stimuli[p.token]
            pulses = stim.num_events * stim.pulses_per_burst
            for role in stim.roles:
                entry = out.setdefault(role, {"pulses": 0, "presentations": 0, "driven_sec": 0.0})
                entry["pulses"] += pulses
                entry["presentations"] += 1
    return out


def summary(blocks: list[Block], stimuli: dict[str, Stimulus], phase_us: float | None = None) -> str:
    """The run in numbers: each block, and how much of the encode block is stimulation."""
    lines = [f"{total_minutes(blocks):.1f} min in total"]
    for block, start in zip(blocks, block_starts_sec(blocks)):
        lines.append(
            f"  {start / 60:7.1f} min  {block.label:<14} {block.duration_sec / 60:6.1f} min  {len(block.presentations)} presentation(s)"
        )
    for block in blocks:
        if block.label != "encode" or not block.presentations:
            continue
        on = sum(stimuli[p.token].duration_sec for p in block.presentations)
        paired = sum(stimuli[p.token].duration_sec for p in block.presentations if len(p.roles) > 1)
        lines.append(
            f"  encode: stimulating {on / 60:.1f} of {block.duration_sec / 60:.1f} min (duty {on / block.duration_sec:.2f}), {paired / 60:.1f} min of it paired"
        )
    budget = stimulation_budget(blocks, stimuli)
    if budget:
        for role, v in budget.items():
            driven = ""
            if phase_us:
                driven = f", {v['pulses'] * 2 * phase_us / 1e6:.2f} s of driven electrode in total"
            lines.append(
                f"  {role} receives {v['pulses']} pulses in {v['presentations']} presentation(s){driven}"
            )
    for stim in stimuli.values():
        burst = (
            f" x {stim.pulses_per_burst} pulses at {stim.burst_hz} Hz" if stim.pulses_per_burst > 1 else ""
        )
        lines.append(
            f"  {stim.token:<18} {stim.num_events} event(s) at {stim.pulse_hz} Hz{burst}, {stim.duration_sec:.1f} s, {stim.polarity}, delays {stim.delays_sec}"
        )
    return "\n".join(lines)


def schedule_record(
    params: AssociativeParams, blocks, stimuli, regions: dict[str, RegionSpec], extra=None
) -> dict:
    """Everything the run decided, for the ``_protocol.json`` written beside the recording and
    read back by :mod:`.report`."""
    from dataclasses import asdict

    return {
        "params": asdict(params),
        "regions": {role: spec.as_dict() for role, spec in regions.items()},
        "stimuli": {token: stim.as_dict() for token, stim in stimuli.items()},
        "blocks": [
            {
                "label": b.label,
                "duration_sec": b.duration_sec,
                "presentations": [p.__dict__ for p in b.presentations],
            }
            for b in blocks
        ],
        "response_windows_ms": {k: list(v) for k, v in response_windows(params).items()},
        **(extra or {}),
    }
