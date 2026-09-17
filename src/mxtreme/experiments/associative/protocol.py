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
    checkpoint_cs, checkpoint_ns        one region alone mid-encode, ``encode_probe_sec`` long
    stim_cs_us, stim_ns, ...            training presentations, ``t_stim`` long
    stim_us_a100_ac, ...                  calibration: one region, one amplitude, one polarity

Blocks
------
The run is a list of blocks. In conditioning mode:

    pre          ``pre_min`` of nothing
    baseline     every ``probe_roles`` alone, ``probe_reps`` times, ``probe_iti`` apart
    encode       ``encode_cycles`` cycles of: ``encode_pattern`` presentations ``encode_iti``
                 apart, then ``encode_cycle_rest`` of nothing, then a checkpoint of
                 ``encode_probe_roles`` probed alone (none after the last cycle, which the
                 retrieval blocks follow)
    decay        ``decay_min`` of nothing
    retrieval_1  ``retrieval_roles`` alone, ``probe_reps`` times
    rest_1       ``retrieval_interval_min`` of nothing
    ...          up to ``num_retrievals``
    post         ``post_min`` of nothing

``phase`` cuts that into separate recordings, each read back before the next: ``baseline`` is
pre and the baseline probes; ``encode_<k>`` is a minute of quiet then cycle k (its trains, rest
and checkpoint); ``retrieval`` is decay, the retrievals and post. Every cycle is built whatever
the phase, so the seeded shuffles agree and the phases together deliver exactly the ``all``
schedule.

The encode block is where the culture is meant to learn, and the in vitro literature says that
takes tens of minutes of intermittent stimulation, not a few presentations: 0.2-0.33 Hz for 10 min
then 5 min off, repeated for hours (le Feber 2010); 0.2 Hz for 15 min, or 100 Hz bursts every 5 s
for 10-15 min, four cycles an hour apart (le Feber 2015); 0.3-1 Hz until the response appears,
within tens of minutes (Shahaf & Marom 2001). The cycle parameters exist so that regime can be
written without touching code.

Calibration mode replaces everything between pre and post with one ``calibrate`` block: every
(region, amplitude, polarity) once per repeat, single pulses, the whole set shuffled together so
the regions interleave and drift does not read as dose (Ronchi et al. 2019 shuffled amplitudes
for the same reason).

The session's baseline block is the gate between the two: each region probed alone at its
chosen amplitude, 60 pulses per role, interleaved. :mod:`.report` turns it into a cross-talk
matrix -- how much stimulating each site alone drives the other two -- and says whether the sites
are independent enough to condition, and whether a path between CS and US exists at all. It is
read 16 minutes into the session, before encoding starts; ``--phase baseline`` alone is the same
measurement.
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


def _offsets(size: int, step: int) -> list[int]:
    """Column (or row) offsets of a ``size``-wide line of electrodes ``step`` apart, as nearly
    centred on 0 as whole electrodes allow.

    An odd size centres exactly. An even size cannot put an electrode on its own centre, so its
    centre falls half a step off; rounding every offset the same way (down) keeps the spacing
    exact and the error to at most half an electrode, rather than laying the whole line out on one
    side of 0.
    """
    import math

    return [math.floor((i - (size - 1) / 2) * step) for i in range(size)]


def stim_grid(center_um, size: int, gap: int) -> list[int]:
    """A size x size grid of electrodes around a point, ``gap`` empty electrodes between
    neighbours, as nearly centred on the nearest electrode as whole electrodes allow (see
    :func:`_offsets`).

    :raises ValueError: If any corner falls off the array.
    """
    centre = nearest_electrode(*center_um)
    col0, row0 = centre % COLS, centre // COLS
    offsets = _offsets(size, gap + 1)
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

    ``focal`` (``{"shape": "focal", "inner": 2, "inner_gap": 4, "return_radius": 6,
    "return_points": "corners"}``): an inner x inner block of driven electrodes (``inner_gap``
    empty electrodes between them), ringed by return electrodes that get the same pulse inverted.
    Seen from further away than the ring the charges cancel, so the field is confined to the site
    (Ronchi et al. 2019).

    The ring is laid out around the driven block itself, a whole number of electrodes beyond its
    edge on every side, so it is symmetric about the block however the block falls on the grid.
    ``return_radius`` is the ring's distance from the site's centre along each axis, in
    electrodes, rounded outward to the nearest symmetric position. ``"diagonal"`` puts a return
    electrode at two opposite corners; ``"corners"`` at all four; ``"corners+edges"`` adds the
    four edge midpoints, which sit half an electrode off centre when the block's span is odd.

    The chip's 32 stimulation units are not free for any electrode: each serves an area of the
    array, and an area exposes only about seven of them (measured on a MaxOne: a site's electrodes
    all drew from the same seven). A 2x2 block with a four-corner ring asks one area for eight,
    which cannot be met; with ``"diagonal"`` it asks for six, which can.
    """
    import math

    shape = site.get("shape", "grid")
    if shape == "grid":
        return stim_grid(center_um, int(site.get("size", 3)), int(site.get("gap", 2))), []
    if shape != "focal":
        raise ValueError(f"stim_site shape must be 'grid' or 'focal', not {shape!r}")

    inner = int(site.get("inner", 2))
    step = int(site.get("inner_gap", 0)) + 1
    radius = float(site.get("return_radius", 3))
    offsets = _offsets(inner, step)
    half_span = (offsets[-1] - offsets[0]) / 2
    if radius <= half_span:
        raise ValueError(
            f"stim_site return_radius {radius} does not reach outside the driven block, whose "
            f"electrodes reach {half_span} from its centre"
        )
    margin = max(1, math.ceil(radius - half_span))

    drive = stim_grid(center_um, inner, step - 1)
    centre = nearest_electrode(*center_um)
    col0, row0 = centre % COLS, centre // COLS
    lo, hi = offsets[0] - margin, offsets[-1] + margin
    kind = site.get("return_points", "corners")
    if kind == "diagonal":
        points = [(lo, lo), (hi, hi)]
    elif kind in ("corners", "corners+edges"):
        points = [(lo, lo), (hi, lo), (lo, hi), (hi, hi)]
        if kind == "corners+edges":
            mid = math.floor((offsets[0] + offsets[-1]) / 2)
            points += [(mid, lo), (mid, hi), (lo, mid), (hi, mid)]
    else:
        raise ValueError(f"return_points must be diagonal, corners or corners+edges, not {kind!r}")
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
    """How large a stimulation site is, what it costs, and whether it is balanced.

    The point of the numbers is to be compared with ``region_radius_um``, the radius over which
    the region's response is *measured*. A site much smaller than that stimulates a point and
    measures a disc, which is a mismatch if the three sites are meant to be regions rather than
    single neurons. ``ring_offset_um`` is how far the ring's centre is from the driven block's,
    which should be 0: an off-centre ring puts a return electrode beside one driven electrode and
    far from the opposite one.

    :returns: ``{"driven_span_um", "ring_radius_um", "ring_offset_um", "driven", "return",
        "units"}``, with the ring radius measured from the driven block's centre to a corner.
    """
    drive, ring = stim_site((1750.0, 1000.0), site)
    xy = [electrode_xy(e) for e in drive]
    span = max(
        max(p[0] for p in xy) - min(p[0] for p in xy),
        max(p[1] for p in xy) - min(p[1] for p in xy),
    )
    block = (sum(p[0] for p in xy) / len(xy), sum(p[1] for p in xy) / len(xy))
    ring_um = offset_um = 0.0
    if ring:
        corners = [electrode_xy(e) for e in ring[: min(4, len(ring))]]
        ring_centre = (sum(p[0] for p in corners) / len(corners), sum(p[1] for p in corners) / len(corners))
        ring_um = separation_um(corners[0], block)
        offset_um = separation_um(ring_centre, block)
    return {
        "driven_span_um": span,
        "ring_radius_um": ring_um,
        "ring_offset_um": offset_um,
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

    ``regions`` gives each role a centre in um; the stimulation site is ``stim_site`` around it,
    unless ``stim_electrodes`` names the driven electrodes for that same site, in which case those
    are used as they are (they are what a previous run actually connected);
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
        own = params.stim_electrodes.get(role) if params.stim_electrodes_site == params.stim_site else None
        if own:
            far = [e for e in own if separation_um(electrode_xy(e), center) > params.region_radius_um]
            if far or len(own) != len(drive):
                raise ValueError(
                    f"stim_electrodes for {role} ({own}) do not fit its centre {center} and stim_site: "
                    "run select again, or delete stim_electrodes from the parameter file"
                )
            drive = [int(e) for e in own]
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

    ``kind`` picks the train length: ``"stim"`` is ``t_stim``, ``"probe"`` is ``t_probe``,
    ``"checkpoint"`` is ``encode_probe_sec``. The
    pairing puts US ``dt_cs_us`` after CS; a negative ``dt_cs_us`` puts US first (a CS pulse
    inside a US burst, as in Chiappalone et al. 2008). ``polarity`` defaults to the parameters'
    and goes into the token only when ``tag_polarity`` is set.
    """
    polarity = polarity or params.pulse_polarity
    geometry = geometry_of(params)
    if num_events is None:
        length = {"probe": params.t_probe, "checkpoint": params.encode_probe_sec}.get(kind, params.t_stim)
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
        if params.encode_pattern and params.encode_cycles > 1:
            for role in dict.fromkeys(params.encode_probe_roles):
                check = stimulus_for(params, [role], "checkpoint")
                stimuli[check.token] = check

        # A presentation must be over, and its windows closed, before the next one starts.
        probe_tail = (
            max(
                max(s.duration_sec for s in stimuli.values() if s.token.startswith(("probe", "checkpoint"))),
                reach,
            )
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
            longest = max(
                (s for s in stimuli.values() if s.token.startswith(("probe", "checkpoint"))),
                key=lambda s: s.duration_sec,
            )
            raise ValueError(
                f"probe_iti = {params.probe_iti}s is shorter than {longest.token} plus its response "
                f"windows ({probe_tail:.1f}s)"
            )
        if params.encode_pattern and params.encode_iti < stim_tail:
            raise ValueError(
                f"encode_iti = {params.encode_iti}s is shorter than a training presentation plus its response windows ({stim_tail:.1f}s)"
            )

        baseline = _probe_block(
            "baseline", params.probe_roles, params.probe_reps, params.probe_iti, rng, stimuli, probe_tail
        )

        # The encode block, cycle by cycle. Every cycle is built whatever the phase, so the seeded
        # shuffles come out the same and a session split over several recordings delivers
        # exactly what one recording would.
        cycles: list[Block] = []
        if params.encode_pattern:
            for cycle in range(params.encode_cycles):
                block = Block("encode", 0.0)
                t = 0.0
                for item in params.encode_pattern:
                    stim = stimuli[token_for(roles_of(item), "stim")]
                    block.presentations.append(Presentation(t, stim.token, stim.roles))
                    t += params.encode_iti
                if cycle == params.encode_cycles - 1:
                    # The last cycle has no checkpoint: decay and the first retrieval follow it.
                    block.duration_sec = t - params.encode_iti + stim_tail
                else:
                    t += params.encode_cycle_rest
                    # A checkpoint: each role alone, unpaired, after the cycle's rest. The
                    # retrieval blocks say where the association ended up; these say how fast it
                    # got there, which is the difference between a result and a curve. They sit
                    # after the rest so the network is settled rather than still ringing from the
                    # last train.
                    for _ in range(params.encode_probe_reps):
                        order = list(params.encode_probe_roles)
                        rng.shuffle(order)
                        for role in order:
                            stim = stimuli[token_for(roles_of(role), "checkpoint")]
                            block.presentations.append(Presentation(t, stim.token, stim.roles))
                            t += params.probe_iti
                    # Ends where the next cycle's first train would start.
                    block.duration_sec = t
                cycles.append(block)

        retrievals: list[Block] = [Block("decay", params.decay_min * 60.0)]
        for k in range(1, params.num_retrievals + 1):
            retrievals.append(
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
                retrievals.append(Block(f"rest_{k}", params.retrieval_interval_min * 60.0))

        if params.phase == "all":
            blocks.append(baseline)
            if cycles:
                encode = Block("encode", 0.0)
                offset = 0.0
                for block in cycles:
                    encode.presentations += [
                        Presentation(offset + p.t_sec, p.token, p.roles) for p in block.presentations
                    ]
                    offset += block.duration_sec
                encode.duration_sec = offset
                blocks.append(encode)
            blocks += retrievals
        elif params.phase == "baseline":
            blocks.append(baseline)
            return blocks, _used(stimuli, blocks)
        elif params.phase == "retrieval":
            blocks = retrievals
        else:
            k = int(params.phase.removeprefix("encode_"))
            blocks = [Block("lead", params.phase_lead_min * 60.0), cycles[k - 1]]
            return blocks, _used(stimuli, blocks)

    else:  # calibration
        # Every (region, amplitude, polarity) once per repeat, the whole set shuffled together, so
        # the regions are interleaved: no region takes a pulse every 3 s for minutes on end, and
        # drift over the run cannot read as a difference between regions or amplitudes.
        iti = params.calibration_iti
        polarities = list(params.calibration_polarities)
        block = Block("calibrate", 0.0)
        t = 0.0
        for _ in range(params.calibration_reps):
            grid = [
                (role, amp, pol)
                for role in ROLES
                for amp in params.calibration_amplitudes_mv
                for pol in polarities
            ]
            rng.shuffle(grid)
            for role, amp, pol in grid:
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
    return blocks, (_used(stimuli, blocks) if params.phase != "all" else stimuli)


def _used(stimuli: dict[str, Stimulus], blocks: list[Block]) -> dict[str, Stimulus]:
    """Only the stimuli these blocks fire, so a phase builds no sequence it will not use."""
    tokens = {p.token for b in blocks for p in b.presentations}
    return {t: s for t, s in stimuli.items() if t in tokens}


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
        # Pulses, not presentations: one training presentation is a ten-minute train of hundreds
        # of pulses, one probe a minute of twenty, and counting presentations makes the block that
        # stimulates most look like the one that stimulates least.
        pulses = sum(
            stimuli[p.token].num_events * stimuli[p.token].pulses_per_burst for p in block.presentations
        )
        on = sum(stimuli[p.token].duration_sec for p in block.presentations)
        detail = (
            f"  {len(block.presentations)} presentation(s), {pulses} pulses, stimulating {on / 60:.1f} min"
            if block.presentations
            else "  quiet"
        )
        lines.append(f"  {start / 60:7.1f} min  {block.label:<14} {block.duration_sec / 60:6.1f} min{detail}")
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
