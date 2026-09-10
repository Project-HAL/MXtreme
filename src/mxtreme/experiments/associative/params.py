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
MODES = ("conditioning", "calibration", "connectivity")


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
    :param description: Free text written into the file.
    :param save_path: Where to record. ``None`` means the managed store, via the ``Config`` given
        to :meth:`resolved`.

    Regions (filled in by :mod:`.select`, or by hand):

    :param rec_electrodes: Every electrode to record. Must be set to run.
    :param regions: ``{"US": [x, y], "CS": [...], "NS": [...]}`` centres in um.
    :param amplitudes_mv: Per role, **mV per phase**, from a calibration run. Peak to peak is
        twice this, which is the number stimulation papers usually quote: 80 here is 160 mV
        peak to peak.
    :param min_region_separation_um: :mod:`.select` refuses closer centres.
    :param region_radius_um: Recording electrodes within this of a centre belong to the region.
    :param stim_site: ``{"shape": "grid", "size", "gap"}`` or ``{"shape": "focal", "inner",
        "inner_gap", "return_radius", "return_points"}``; see :func:`.protocol.stim_site`.
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

    :param mode: ``"conditioning"``, ``"calibration"`` (find each region's amplitude), or
        ``"connectivity"`` (check how much stimulating each region alone drives the other two).
    :param pre_min: Minutes of nothing before the first probe.
    :param probe_roles: Roles probed in the baseline block.
    :param probe_reps: Probes of each role per probe block.
    :param probe_iti: Seconds between probes; must cover ``t_probe`` plus the response windows.
    :param encode_pattern: e.g. ``["PAIR", "PAIR", "NS"]``.
    :param encode_iti: Seconds between training presentations; must cover ``t_stim``.
    :param encode_cycles: Times the pattern runs.
    :param encode_cycle_rest: Seconds of nothing after each cycle.
    :param decay_min: Minutes of nothing after encoding.
    :param num_retrievals: Retrieval probe blocks.
    :param retrieval_roles: Roles probed in each.
    :param retrieval_interval_min: Minutes of nothing between them.
    :param post_min: Minutes of nothing at the end.
    :param seed: For the shuffles within probe blocks.
    :param calibration_amplitudes_mv: The amplitude ladder, mV per phase.
    :param calibration_polarities: Polarities swept.
    :param calibration_reps: Times each (amplitude, polarity) is presented per region.
    :param calibration_iti: Seconds between calibration pulses.
    :param max_amplitude_mv: Refuse any amplitude above this, in mV per phase. The default of 150
        (300 mV peak to peak) sits a little above the 40-240 mV peak-to-peak range Ronchi et al.
        2019 characterised on these arrays, and well inside the water window that keeps voltage
        stimulation from electrolysing the electrode. Raising it is a deliberate act: nothing in
        this package knows your electrodes' impedance or history.
    :param check_reps: Connectivity mode: single pulses per region.
    :param check_iti: Connectivity mode: seconds between pulses.
    :param crosstalk_warn: Connectivity mode: warn when a region's response to another region's
        stimulus reaches this fraction of that region's own local response.
    :param burst_warn: Connectivity mode: warn when this fraction of pulses is followed by a
        network burst, which means the stimulus is driving the whole culture rather than a site.
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
    description: str = "associative conditioning: US+CS paired and NS alone; every region probed alone before, after, and after rests"
    save_path: str | None = None

    rec_electrodes: list[int] | None = None
    regions: dict[str, list[float]] = field(default_factory=dict)
    amplitudes_mv: dict[str, float] = field(default_factory=lambda: {r: 80.0 for r in ROLES})
    min_region_separation_um: float = 1000.0
    region_radius_um: float = 150.0
    stim_site: dict = field(
        default_factory=lambda: {
            "shape": "focal",
            "inner": 2,
            "inner_gap": 4,
            "return_radius": 6,
            "return_points": "corners",
        }
    )
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
    decay_min: float = 10.0
    num_retrievals: int = 3
    retrieval_roles: list[str] = field(default_factory=lambda: list(ROLES))
    retrieval_interval_min: float = 15.0
    post_min: float = 10.0
    seed: int = 1
    calibration_amplitudes_mv: list[float] = field(default_factory=lambda: [20, 40, 60, 80, 100, 120])
    calibration_polarities: list[str] = field(default_factory=lambda: ["anodic-first", "cathodic-first"])
    calibration_reps: int = 10
    calibration_iti: float = 3.0
    max_amplitude_mv: float = 150.0
    check_reps: int = 20
    check_iti: float = 5.0
    crosstalk_warn: float = 0.3
    burst_warn: float = 0.2
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
    def file_name(self) -> str:
        """``plating_<date>_<batch>_chip_<chip>_well_<w>_DIV_<div>_<exp_id>``, the store's stem."""
        if not self.batch:
            raise ValueError("batch is unset, so the recording cannot be named")
        return (
            store.recording_stem(self.batch, self.plate_date, self.chip, self.well, self.div)
            + f"_{self.exp_id}"
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
            "Exp ID": self.exp_id,
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
        if self.pulse_polarity not in ("anodic-first", "cathodic-first"):
            raise ValueError(
                f"pulse_polarity must be anodic-first or cathodic-first, not {self.pulse_polarity!r}"
            )
        for item in self.encode_pattern:
            if item.upper() not in ROLES + (PAIR,):
                raise ValueError(f"encode_pattern has {item!r}; use US, CS, NS or PAIR")
        for name, roles in (("probe_roles", self.probe_roles), ("retrieval_roles", self.retrieval_roles)):
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
                f"{needed} for {len(ROLES)} regions, and the chip has {STIM_UNITS}. A focal site "
                f"costs inner^2 + the return points; widen it with inner_gap rather than with "
                f"inner, or use a grid site, which spends nothing on a return ring."
            )
        if self.t_probe <= 0 or self.t_stim <= 0 or self.pulse_hz <= 0:
            raise ValueError("t_probe, t_stim and pulse_hz must be positive")
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
