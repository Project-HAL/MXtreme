"""Draw what a parameter file will do, before running it: the array with the regions and their
sites, the run as a timeline, and the pulse waveform. Needs no rig."""

from __future__ import annotations

from pathlib import Path

from mxtreme.experiments.associative import display, protocol
from mxtreme.experiments.associative.params import AssociativeParams

#: One spike on disk: ``frameno`` (i8), ``channel`` (i4), ``amplitude`` (f4). Measured, not
#: assumed: every MaxWell recording's ``spikes`` table has this dtype.
SPIKE_BYTES = 16

#: Spikes per routed channel per second, low and high. Two cultures measured here came out at 0.48
#: (484 channels, DIV 35) and 0.88 (989 channels, DIV 29); the high end leaves room for a livelier
#: one, and a stimulated culture fires more than a resting one.
SPIKE_HZ_PER_CHANNEL = (0.5, 2.0)

#: Bytes per channel-second of compressed raw, by sample rate. Measured on a MaxTwo (10 kHz);
#: MaxOne samples twice as fast for twice the size.
RAW_BYTES_PER_CHANNEL_SEC = {10000: 3800.0, 20000: 7600.0}


def field_overlay(params, regions, samples: int = 260):
    """The potential over the whole array from all three sites firing at their own amplitudes,
    for the preview's map; ``None`` unless a field model is set.

    All three never fire at once, so this is not a moment of the experiment: it is the three
    sites' fields drawn together so their extents can be compared at a glance.

    :returns: ``(values_uv, extent)`` for ``imshow``, or ``None``.
    """
    if not params.field_model or not regions:
        return None
    import numpy as np

    from mxtreme.experiments import fieldmap

    model = fieldmap.load_model(params.field_model)
    width, height = protocol.COLS * protocol.PITCH_UM, protocol.ROWS * protocol.PITCH_UM
    xs = np.linspace(0, width, samples)
    ys = np.linspace(0, height, max(2, round(samples * height / width)))
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    total = np.zeros(len(points))
    for spec in regions.values():
        total = np.maximum(
            total, fieldmap.site_field_uv(model, spec.drive_electrodes, spec.amplitude_mv, points)
        )
    return total.reshape(gx.shape), (0.0, width, height, 0.0)


def field_text(params, regions) -> list[str]:
    """What the field model says about these regions at these amplitudes, as lines for the
    printed preview: the field each site puts at the other two, and what does and does not
    follow from it."""
    if not regions:
        return []
    if not params.field_model:
        return [
            (
                "  field of a pulse: unknown, since no field model is set. Measure it on a saline chip "
                "(python -m mxtreme.experiments.fieldmap run ..., then report --save-model) and point "
                "field_model at the JSON."
            )
        ]
    from mxtreme.experiments import fieldmap

    model = fieldmap.load_model(params.field_model)
    lines = [
        (
            f"  field of a pulse, from {Path(params.field_model).name}: "
            f"V(r) = {model['uv_per_mv_at_one_pitch']:.1f} uV/mV x A x (r / {model['pitch_um']:.1f} um)^"
            f"-{model['exponent']:.2f}, where A is the amplitude in mV and r the distance from a driven "
            f"electrode in um, summed over a site's electrodes; measured in saline out to "
            f"{model.get('measured_to_um', 0):.0f} um"
        )
    ]
    for role, spec in regions.items():
        own = fieldmap.field_at_um(
            model,
            spec.drive_electrodes,
            spec.amplitude_mv,
            (spec.center_um[0], spec.center_um[1] + params.region_radius_um),
        )
        others = []
        for other, o_spec in regions.items():
            if other == role:
                continue
            at = fieldmap.field_at_um(model, spec.drive_electrodes, spec.amplitude_mv, o_spec.center_um)
            distance = protocol.separation_um(spec.center_um, o_spec.center_um)
            others.append(f"{at:.0f} uV at {other} ({distance:.0f} um, {at / own:.0%} of its own)")
        headroom = params.amplitude_thresholds_mv.get(role)
        lines.append(
            f"    {role:<4} {spec.amplitude_mv:.0f} mV"
            + (f" ({spec.amplitude_mv / headroom:.1f}x its {headroom:.0f} mV threshold)" if headroom else "")
            + f": {own:.0f} uV at its own {params.region_radius_um:.0f} um edge; "
            + ", ".join(others)
        )
    lines += [
        (
            f"    The potential falls slowly (r^-{model['exponent']:.2f}), so the *artifact* reaches the "
            f"whole array. What drives a long straight axon is its second derivative (Rattay 1986), which "
            f"falls as r^-{model['exponent'] + 2:.2f}; at axon terminals and bends it is the first "
            f"derivative instead (Rattay 1999), r^-{model['exponent'] + 1:.2f}. Either way the excitation "
            f"is far more local than the artifact. Whatever the threshold is, a quantity falling as r^-p "
            f"puts the excited radius at A^(1/p), so doubling the amplitude grows it by "
            f"{2 ** (1 / (model['exponent'] + 2)):.2f}x to {2 ** (1 / (model['exponent'] + 1)):.2f}x, not 2x."
        ),
        (
            "    No radius is drawn here on purpose. Converting a field in uV into 'this is how far the "
            "stimulus reaches' needs the field at which tissue fires, which a saline map does not "
            "measure and calibration does not either: calibration finds the lowest amplitude at which a "
            "whole site evokes a countable response over a 150 um disc. Calibration's spread column and "
            "the baseline gate are the measurements of independence."
        ),
    ]
    return lines


def storage_estimate(params: AssociativeParams, minutes: float, channels: int) -> dict:
    """How large the recording will be, in GB.

    Spikes are always stored, for every routed channel; ``raw_traces`` decides how much voltage
    rides along. The spike figure is a range because it is the culture's firing rate, which is not
    known until it fires.

    :param channels: Routed recording channels; the parameters' own ``rec_electrodes`` if unset.
    :returns: ``{"spikes_gb": (low, high), "raw_gb", "total_gb": (low, high), "channels",
        "raw_channels", "sample_hz"}``.
    """
    seconds = minutes * 60.0
    sample_hz = 20000 if str(params.batch).rsplit("_", 1)[-1].upper() == "M1" else 10000
    spikes = tuple(hz * channels * seconds * SPIKE_BYTES / 1e9 for hz in SPIKE_HZ_PER_CHANNEL)

    if params.raw_traces == "all":
        raw_channels = 1024
    elif params.raw_traces == "regions" and params.regions:
        regions = protocol.regions_for(params, params.rec_electrodes)
        raw_channels = len({e for s in regions.values() for e in s.rec_electrodes + s.stim_electrodes})
    else:
        raw_channels = 0
    raw = raw_channels * seconds * RAW_BYTES_PER_CHANNEL_SEC[sample_hz] / 1e9
    return {
        "spikes_gb": spikes,
        "raw_gb": raw,
        "total_gb": (spikes[0] + raw, spikes[1] + raw),
        "channels": channels,
        "raw_channels": raw_channels,
        "sample_hz": sample_hz,
    }


def destination(params: AssociativeParams, config=None, work=None) -> dict[str, str]:
    """Where a run with these parameters would write, without touching the rig.

    The recording is the only file a run puts in the store; the protocol travels inside it. With
    ``work``, the two convenience copies go there.

    :param config: A :class:`~mxtreme.config.Config`, or the path of an ``mxtreme.toml``; not
        needed when ``params.save_path`` is set.
    :returns: ``{"directory", "recording", "protocol", "raw traces", "registered"}``, plus
        ``"work copies"`` when ``work`` is given.
    """
    from mxtreme.config import Config
    from mxtreme.experiments.associative.run import PROTOCOL_KEY

    if config is not None and not hasattr(config, "recordings_dir"):
        config = Config.from_toml(config)
    in_store = params.save_path is None
    resolved = params.resolved(config)
    stem = resolved.file_name
    directory = resolved.save_path
    out = {
        "directory": directory,
        "recording": f"{directory}/{stem}.raw.h5",
        "protocol": f"inside the recording, /assay/{PROTOCOL_KEY}",
        "raw traces": {
            "none": "none (spikes only)",
            "regions": "the three regions' electrodes",
            "all": "every channel",
        }[params.raw_traces],
        "registered": f"{config.registry_path} (kind experiment)"
        if in_store
        else "not registered (save_path set)",
    }
    if work is not None:
        out["work copies"] = f"{work}/{stem}_protocol.json, {work}/{stem}_fired.csv"
    return out


def preview(
    params: AssociativeParams, out_png: str | None = None, cfg_path: str | None = None, show: bool = False
) -> str:
    """Print the schedule summary and draw the figure. Returns the summary text.

    :param cfg_path: A MaxLab ``.cfg`` to draw the routed electrodes from, when the parameters do
        not list ``rec_electrodes`` yet.
    """
    routed = None
    rec = params.rec_electrodes
    if cfg_path:
        routed = display.routed_positions(cfg_path)
        from mxtreme.scans import mx_config

        rec = mx_config.read_config_elecs(cfg_path)
    regions = protocol.regions_for(params, rec) if all(r in params.regions for r in protocol.ROLES) else {}
    blocks, stimuli = protocol.build_schedule(params)

    footprint = protocol.site_footprint(params.stim_site)
    text = [
        f"well {params.well}, mode {params.mode}",
        f"  site: {footprint['driven']} driven electrodes spanning {footprint['driven_span_um']:.0f} um"
        + (f", return ring at {footprint['ring_radius_um']:.0f} um" if footprint["return"] else "")
        + f"; response measured over {params.region_radius_um:.0f} um"
        + f"; {footprint['units'] * 3} of {protocol.STIM_UNITS} stimulation units",
    ]
    for role, spec in regions.items():
        text.append(
            f"  {role:<4} centre {spec.center_um} um  drive dac {spec.drive_dac} {spec.drive_electrodes}"
            + (f"  return dac {spec.return_dac} {spec.return_electrodes}" if spec.return_electrodes else "")
            + f"  {len(spec.rec_electrodes)} recording electrodes  {spec.amplitude_mv} mV"
        )
    text += field_text(params, regions)
    text.append(protocol.summary(blocks, stimuli, params.stim_phase_us))

    channels = len(rec or []) or 1024
    size = storage_estimate(params, protocol.total_minutes(blocks), channels)
    kept = {
        "none": "spikes only",
        "regions": (
            f"spikes, and raw traces for {size['raw_channels']} region channels"
            if size["raw_channels"]
            else "spikes, and raw traces for the region channels (how many, once select has placed them)"
        ),
        "all": "spikes and raw traces for every channel",
    }[params.raw_traces]
    text.append(
        f"  one recording, every block of it, {protocol.total_minutes(blocks):.0f} min: {kept}"
        f" from {channels} routed channels"
        f" -> {size['total_gb'][0]:.2f} to {size['total_gb'][1]:.2f} GB"
        + (f" (raw {size['raw_gb']:.1f} GB of it)" if size["raw_gb"] else "")
    )
    print("\n".join(text))

    import matplotlib

    if out_png and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 13))
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 1.4, 1.2], width_ratios=[3, 2], hspace=0.38, wspace=0.25)
    ax_map = fig.add_subplot(grid[0, :])
    ax_time = fig.add_subplot(grid[1, :])
    ax_train = fig.add_subplot(grid[2, 0])
    ax_pulse = fig.add_subplot(grid[2, 1])
    overlay = field_overlay(params, regions)
    if regions:
        display.draw_map(
            ax_map,
            {r: s.as_dict() for r, s in regions.items()},
            routed,
            params.region_radius_um,
            title=f"well {params.well}: {params.stim_site.get('shape')} sites {params.stim_site}, recording radius {params.region_radius_um} um"
            + (", shading = the pulse's potential (log uV)" if overlay else ""),
            field=overlay,
        )
        display.draw_triangle(ax_map, {r: s.as_dict() for r, s in regions.items()})
    else:
        ax_map.text(0.5, 0.5, "no regions yet: run select", ha="center", transform=ax_map.transAxes)
    display.draw_timeline(
        ax_time,
        blocks,
        dt_cs_us_ms=params.dt_cs_us,
        stimuli=stimuli,
        recording=f"one recording, all {protocol.total_minutes(blocks):.0f} min: {kept}, "
        f"{size['total_gb'][0]:.2f}-{size['total_gb'][1]:.2f} GB",
    )
    ax_time.set_title(
        f"{protocol.total_minutes(blocks):.0f} min; training t_stim {params.t_stim:.0f} s, probes t_probe {params.t_probe:.0f} s, {params.pulse_hz} Hz, dt_cs_us {params.dt_cs_us} ms",
        loc="left",
        fontsize=10,
    )
    shown = next((s for s in stimuli.values() if len(s.roles) > 1), next(iter(stimuli.values())))
    display.draw_waveforms(
        ax_train,
        ax_pulse,
        shown,
        params,
        params.amplitudes_mv,
        with_return=params.stim_site.get("shape") == "focal",
    )
    if out_png:
        fig.savefig(out_png, dpi=110, facecolor="white", bbox_inches="tight")
        print(f"wrote {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return "\n".join(text)
