"""Everything one associative run is decided by, as one object.

Loaded from a JSON file whose keys are these field names (keys starting with ``_`` are comments
and ignored). :meth:`AssociativeParams.validate` refuses a file that does not add up before the rig
is touched; :meth:`~AssociativeParams.resolved` fills in where the recording lands, in the managed
store like a scan does. ``params_default.json`` beside this module is a starting point sized for a
four-hour first session; ``params_dryrun.json`` is the short version for a saline chip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from mxtreme import store

ROLES = ("US", "CS", "NS")
PAIR = "PAIR"
MODES = ("conditioning", "calibration")
RAW_TRACES = ("none", "regions", "all")


@dataclass
class AssociativeParams:
    """Parameters of one associative run.

    Identity (as the scans use it):

    :param batch: The plating batch, e.g. ``"fall2026_batch1_iPSC_M1"``.
    :param chip: Chip serial.
    :param plate_date: ``YYMMDD``.
    :param div: Days *in vitro* on the day of the run.
    :param well: The well. One culture per run.
    :param exp_id: Names the recording's file tail and registry row, e.g. ``"assoc"``.
    :param phase: Which part of a conditioning run this recording is: ``"all"`` in one recording,
        or ``"baseline"``, ``"encode_1"`` ... ``"encode_<encode_cycles>"``, ``"retrieval"`` as
        separate recordings, each read back and judged before the next starts. The phases together
        deliver exactly the ``"all"`` schedule; a phase's name is appended to the file name.
    :param phase_lead_min: Minutes of quiet at the start of an encode phase's recording, so the
        network has settled and the first train has a window before it.
    :param description: Free text written into the file.
    :param save_path: Where to record. ``None`` means the managed store, via the ``Config`` given
        to :meth:`resolved`.

    Regions (filled in by :mod:`.select`, or by hand):

    :param rec_electrodes: Every electrode to record. Must be set to run.
    :param regions: ``{"US": [x, y], "CS": [...], "NS": [...]}`` centres in um.
    :param stim_electrodes: ``{"US": [e, e, e, e], ...}``: the electrodes each region is driven
        through, as ``select`` planned them and as ``run`` last connected them (a driven electrode
        can be swapped for a routed neighbour to get a stimulation unit of its own, and the swap is
        written back here so every later run drives the same electrodes). Honoured only while
        ``stim_electrodes_site`` still equals ``stim_site``; a different site is built from the
        centres again.
    :param routing_cfg: The routing (electrode-to-channel map) the first run solved, saved as a
        MaxLab ``.cfg`` in the work directory and written back here by ``run``. Later runs load
        it instead of routing again: the router is not reproducible, a stimulation unit is a
        function of the channel, so only the same routing gives the same units and the same
        driven electrodes.
    :param amplitudes_mv: Per role, **mV per phase**, from a calibration run. Peak to peak is
        twice this, which is the number stimulation papers usually quote: 80 here is 160 mV
        peak to peak.
    :param amplitude_thresholds_mv: Per region, the lowest amplitude calibration found usable --
        the lowest at which the whole site evoked a countable response over the region's disc.
        Written by ``report --apply``. ``preview`` prints the chosen amplitude as a multiple of
        it, which is the headroom the region is driven with. It is *not* the field at which a
        neuron fires: it is an amplitude, measured on a site, against a population count.
    :param field_model: Path to the JSON ``fieldmap report --save-model`` wrote: how the
        potential falls off through the medium, measured on a saline chip. Optional; with it
        ``preview`` shades the array with the potential the pulses put on it and prints what each
        site puts at the other two.
    :param amplitudes_source: Where ``amplitudes_mv`` came from -- the calibration recording's
        name, as its report prints it. ``"default"`` means nobody calibrated, and a
        conditioning run refuses to start on it.
    :param min_region_separation_um: :mod:`.select` refuses closer centres.
    :param region_radius_um: Recording electrodes within this of a centre belong to the region.
    :param stim_site: ``{"shape": "hex", "gap"}`` (seven electrodes, the default),
        ``{"shape": "grid", "size", "gap"}`` or ``{"shape": "focal", "inner", "inner_gap",
        "return_radius", "return_points"}``; see :func:`.protocol.stim_site`.
    :param region_dacs: Grid sites: which DAC drives each role.
    :param drive_dac: Focal sites: the DAC every site's driven electrodes share.
    :param return_dac: Focal sites: the DAC every site's return ring shares.

    The pulse:

    :param stim_phase_us: One phase of the biphasic pulse.
    :param pulse_polarity: ``"anodic-first"`` or ``"cathodic-first"`` on the driven electrodes.
    :param pulse_hz: Pulse events per second.
    :param pulses_per_burst: Pulses per event; 1 is a single pulse, 11 at ``burst_hz`` 20 is a
        Jimbo-style tetanus.
    :param burst_hz: Rate within an event.
    :param t_stim: Length of a training presentation, seconds.
    :param t_probe: Length of a probe, seconds.
    :param dt_cs_us: US train starts this many ms after CS's in a pairing; negative for US first.

    The schedule (see :mod:`.protocol`):

    :param mode: ``"conditioning"``, ``"calibration"`` (find each region's amplitude), or.
    :param pre_min: Minutes of nothing before the first probe.
    :param probe_roles: Roles probed in the baseline block.
    :param probe_reps: Probes of each role per probe block.
    :param probe_iti: Seconds between probes; must cover ``t_probe`` plus the response windows.
    :param encode_pattern: e.g. ``["PAIR", "PAIR", "NS"]``.
    :param encode_iti: Seconds between training presentations; must cover ``t_stim``.
    :param encode_cycles: Times the pattern runs.
    :param encode_cycle_rest: Seconds of nothing after each cycle.
    :param encode_probe_roles: Roles probed alone after each cycle's rest, to watch the association
        form rather than only see where it ended up. Empty for no checkpoints. Every probe is also
        a little extinction, so this is deliberately fewer roles and fewer repeats than a probe
        block.
    :param encode_probe_reps: Checkpoint probes of each role.
    :param encode_probe_sec: Length of a checkpoint probe, against ``t_probe`` for a probe block's.
        Shorter on purpose: a checkpoint is an unpaired CS presentation in the middle of
        acquisition, which is what extinction is, so it buys its measurement with as few pulses as
        it can. The report counts per pulse, so a short checkpoint is still comparable with a full
        probe block.
    :param decay_min: Minutes of nothing after encoding.
    :param num_retrievals: Retrieval probe blocks.
    :param retrieval_roles: Roles probed in each.
    :param retrieval_interval_min: Minutes of nothing between them.
    :param post_min: Minutes of nothing at the end.
    :param seed: For the shuffles within probe blocks.
    :param calibration_amplitudes_mv: The amplitude ladder, mV per phase. Starts low (10, 20 mV
        per phase) because thresholds on these arrays can sit near the bottom of Ronchi's range.
    :param calibration_polarities: Polarities swept. Both by default: the report picks, per
        region, the polarity that reaches threshold with less voltage, and writes it to
        ``pulse_polarity``. Anodic-first is the literature's answer (Wagenaar 2004, Ronchi 2019),
        but which DAC direction is "anodic" at the electrode on this rig has not been confirmed,
        so the sweep is what decides rather than the label.
    :param calibration_reps: Times each (amplitude, polarity) is presented per region.
    :param calibration_iti: Seconds between calibration pulses.
    :param max_amplitude_mv: Refuse any amplitude above this, in mV per phase. The default of 120
        (240 mV peak to peak) is the top of the range Ronchi et al. 2019 characterised on these
        arrays, and well inside the water window that keeps voltage stimulation from electrolysing
        the electrode. Raising it is a deliberate act: nothing in
        this package knows your electrodes' impedance or history.
    :param crosstalk_warn: Connectivity mode: warn when a region's response to another region's
        stimulus reaches this fraction of that region's own local response.
    :param burst_warn: Connectivity mode: warn when this fraction of pulses is followed by a
        network burst, which means the stimulus is driving the whole culture rather than a site.
    :param raw_traces: Which channels' raw voltage traces the recording keeps. ``"none"`` keeps
        spikes only, which is all the readout uses and costs roughly a megabyte a minute;
        ``"regions"`` adds the traces of the three regions' recording and stimulation electrodes,
        enough to see the stimulation artifact; ``"all"`` keeps every channel, around 4 kB per
        channel-second compressed on a MaxTwo and twice that on a MaxOne -- tens of gigabytes
        for a four-hour run. Spikes are always kept, for every routed channel.
    :param pulse_window_ms: Counted after every pulse for the readout: ``[start, end]``.
    :param response_windows_ms: Optional ``{label: [start, end]}`` around a presentation; by
        default base/during/after sized by ``t_probe``.
    """

    batch: str = ""
    chip: str = ""
    plate_date: int = 0
    div: int = 0
    well: int = 0
    exp_id: str = "assoc"
    phase: str = "all"
    phase_lead_min: float = 1.0
    description: str = "associative conditioning: US+CS paired and NS alone; every region probed alone before, after, and after rests"
    save_path: str | None = None

    rec_electrodes: list[int] | None = None
    regions: dict[str, list[float]] = field(default_factory=dict)
    stim_electrodes: dict[str, list[int]] = field(default_factory=dict)
    stim_electrodes_site: dict = field(default_factory=dict)
    routing_cfg: str | None = None
    amplitudes_mv: dict[str, float] = field(default_factory=lambda: {r: 80.0 for r in ROLES})
    amplitude_thresholds_mv: dict[str, float] = field(default_factory=dict)
    amplitudes_source: str = "default"
    field_model: str | None = None
    min_region_separation_um: float = 1000.0
    region_radius_um: float = 150.0
    stim_site: dict = field(default_factory=lambda: {"shape": "hex", "gap": 5})
    region_dacs: dict[str, int] = field(default_factory=lambda: {"US": 0, "CS": 1, "NS": 2})
    drive_dac: int = 0
    return_dac: int = 1

    stim_phase_us: float = 100.0
    pulse_polarity: str = "anodic-first"
    pulse_hz: float = 0.33
    pulses_per_burst: int = 1
    burst_hz: float = 20.0
    t_stim: float = 600.0
    t_probe: float = 60.0
    dt_cs_us: float = 0.0

    mode: str = "conditioning"
    pre_min: float = 10.0
    probe_roles: list[str] = field(default_factory=lambda: list(ROLES))
    probe_reps: int = 3
    probe_iti: float = 90.0
    encode_pattern: list[str] = field(default_factory=lambda: ["PAIR", "PAIR", "NS"])
    encode_iti: float = 660.0
    encode_cycles: int = 3
    encode_cycle_rest: float = 300.0
    encode_probe_roles: list[str] = field(default_factory=lambda: ["CS", "NS"])
    encode_probe_reps: int = 1
    encode_probe_sec: float = 30.0
    decay_min: float = 10.0
    num_retrievals: int = 3
    retrieval_roles: list[str] = field(default_factory=lambda: list(ROLES))
    retrieval_interval_min: float = 15.0
    post_min: float = 10.0
    seed: int = 1
    calibration_amplitudes_mv: list[float] = field(default_factory=lambda: [10, 20, 30, 45, 60, 80, 100])
    calibration_polarities: list[str] = field(default_factory=lambda: ["anodic-first", "cathodic-first"])
    calibration_reps: int = 6
    calibration_iti: float = 3.0
    max_amplitude_mv: float = 120.0
    crosstalk_warn: float = 0.3
    burst_warn: float = 0.2
    raw_traces: str = "none"
    pulse_window_ms: list[float] = field(default_factory=lambda: [5.0, 50.0])
    response_windows_ms: dict[str, list[float]] | None = None

    # --- files ---

    @classmethod
    def from_json(cls, path: str | Path) -> AssociativeParams:
        """Read a parameter file. Unknown keys are an error, since a misspelt one would otherwise
        silently fall back to the default."""
        with open(path) as f:
            raw = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown parameter(s) {sorted(unknown)}")
        return cls(**raw)

    def with_overrides(self, assignments: list[str] | None) -> AssociativeParams:
        """Apply ``key=value`` strings to a copy, for trying a variant without a new file.

        The value is read as JSON when it parses (``20``, ``0.5``, ``["PAIR"]``, ``true``) and
        taken as a plain string when it does not (``anodic-first``). :meth:`validate` still has
        the last word, so a value of the wrong shape fails before the rig is touched.

        :raises ValueError: On a key that is not a parameter.
        """
        import json as _json

        out = replace(self)
        for item in assignments or []:
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in self.__dataclass_fields__:
                raise ValueError(f"{key!r} is not a parameter; see params_default.json")
            try:
                value = _json.loads(raw)
            except _json.JSONDecodeError:
                value = raw
            setattr(out, key, value)
        return out

    @staticmethod
    def update_file(path: str | Path, updates: dict) -> None:
        """Change a few keys of a parameter file in place, keeping every other key and the ``_``
        notes as they are. Used after calibration (the amplitudes) and after a run (the electrodes
        actually driven)."""
        with open(path) as f:
            current = json.load(f)
        current.update(updates)
        with open(path, "w") as f:
            json.dump(current, f, indent=2)
            f.write("\n")

    def to_json(self, path: str | Path, notes: dict[str, str] | None = None) -> None:
        """Write the parameters, with optional ``_comment`` entries placed before the keys they
        describe."""
        out = {}
        for key, value in asdict(self).items():
            if notes and key in notes:
                out[f"_{key}"] = notes[key]
            out[key] = value
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    # --- identity, as the scans have it ---

    @property
    def wells(self) -> list[int]:
        return [self.well]

    @property
    def conditions(self) -> list:
        return []

    @property
    def batch_id(self) -> str:
        return store.Batch.parse(self.batch).id if self.batch else ""

    @property
    def run_id(self) -> str:
        """``exp_id``, plus the phase when the run is one phase of a conditioning session."""
        return self.exp_id if self.phase == "all" else f"{self.exp_id}_{self.phase}"

    @property
    def file_name(self) -> str:
        """``plating_<date>_<batch>_chip_<chip>_well_<w>_DIV_<div>_<exp_id>[_<phase>]``, the
        store's stem."""
        if not self.batch:
            raise ValueError("batch is unset, so the recording cannot be named")
        return (
            store.recording_stem(self.batch, self.plate_date, self.chip, self.well, self.div)
            + f"_{self.run_id}"
        )

    @property
    def h5_path(self) -> Path:
        if self.save_path is None:
            raise ValueError("save_path is unset; call params.resolved(config) first")
        return Path(self.save_path) / f"{self.file_name}.raw.h5"

    def resolved(self, config=None) -> AssociativeParams:
        """A copy whose ``save_path`` is known: the culture's directory in the managed store."""
        if self.save_path is not None:
            return self
        if config is None:
            raise ValueError(
                "no save_path and no config: pass Config.from_toml('mxtreme.toml') or set save_path"
            )
        directory = store.recording_dir(config, self.batch, self.plate_date, self.chip, self.well, self.div)
        return replace(self, save_path=str(directory))

    def as_metadata(self) -> dict:
        return {
            "Exp ID": self.run_id,
            "Batch ID": self.batch_id,
            "Chip ID": self.chip,
            "Plate date": self.plate_date,
            "DIV": self.div,
            "Well IDs": [self.well],
            "Conditions": [],
        }

    # --- checks ---

    def validate(self, for_run: bool = False) -> None:
        """Refuse what cannot work. With ``for_run``, also require what the rig needs."""
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        if self.raw_traces not in RAW_TRACES:
            raise ValueError(f"raw_traces must be one of {RAW_TRACES}, not {self.raw_traces!r}")
        if self.pulse_polarity not in ("anodic-first", "cathodic-first"):
            raise ValueError(
                f"pulse_polarity must be anodic-first or cathodic-first, not {self.pulse_polarity!r}"
            )
        for item in self.encode_pattern:
            if item.upper() not in ROLES + (PAIR,):
                raise ValueError(f"encode_pattern has {item!r}; use US, CS, NS or PAIR")
        for name, roles in (
            ("probe_roles", self.probe_roles),
            ("retrieval_roles", self.retrieval_roles),
            ("encode_probe_roles", self.encode_probe_roles),
        ):
            for role in roles:
                if role not in ROLES:
                    raise ValueError(f"{name} has {role!r}")
        if self.stim_site.get("shape") == "focal" and self.drive_dac == self.return_dac:
            raise ValueError("drive_dac and return_dac must differ")
        # Every stimulation electrode needs one of the chip's stimulation units, and the three
        # regions share them. A site that does not fit cannot be routed, so it is refused here
        # rather than partway through init_well_regions.
        from mxtreme.experiments.associative.protocol import STIM_UNITS, site_footprint

        footprint = site_footprint(self.stim_site)
        needed = footprint["units"] * len(ROLES)
        if needed > STIM_UNITS:
            raise ValueError(
                f"this stim_site needs {footprint['units']} stimulation units per region, "
                f"{needed} for {len(ROLES)} regions, and the chip has {STIM_UNITS}. A hex site "
                f"costs 7, a grid size^2, a focal site inner^2 + the return points; widen a site "
                f"with gap (or inner_gap) rather than with more electrodes."
            )
        if self.t_probe <= 0 or self.t_stim <= 0 or self.pulse_hz <= 0:
            raise ValueError("t_probe, t_stim and pulse_hz must be positive")
        if self.phase != "all":
            if self.mode != "conditioning":
                raise ValueError(f"phase {self.phase!r} only applies to conditioning, not {self.mode}")
            cycle = self.phase.removeprefix("encode_")
            in_range = cycle.isdigit() and 1 <= int(cycle) <= self.encode_cycles
            if self.phase not in ("baseline", "retrieval") and not in_range:
                raise ValueError(
                    f"phase must be all, baseline, encode_1..encode_{self.encode_cycles} or retrieval, "
                    f"not {self.phase!r}"
                )
        # Amplitude is the one parameter that can damage hardware, so it is checked wherever a
        # parameter file is read rather than only before a run.
        asked = list(self.amplitudes_mv.values()) + list(self.calibration_amplitudes_mv)
        over = sorted({a for a in asked if a > self.max_amplitude_mv})
        if over:
            raise ValueError(
                f"amplitude(s) {over} mV per phase are above max_amplitude_mv "
                f"({self.max_amplitude_mv} mV per phase, {2 * self.max_amplitude_mv} mV peak to "
                f"peak). Lower them, or raise max_amplitude_mv deliberately if you know these "
                f"electrodes tolerate it."
            )
        if for_run:
            _ = self.file_name
            if not self.rec_electrodes:
                raise ValueError("rec_electrodes is unset: run `select` first, or list them")
            missing = [r for r in ROLES if r not in self.regions]
            if missing:
                raise ValueError(f"regions has no centre for {missing}: run `select` first")
            if set(self.amplitudes_mv) != set(ROLES):
                raise ValueError("amplitudes_mv needs US, CS and NS")
            if self.mode != "calibration" and self.amplitudes_source == "default":
                raise ValueError(
                    "amplitudes_mv are uncalibrated (amplitudes_source is 'default'). Run calibration, "
                    "copy the amplitudes_mv line its report prints into the parameter file, and set "
                    "amplitudes_source to the calibration recording's name -- or, for a saline dry "
                    "run, --set amplitudes_source=saline"
                )
