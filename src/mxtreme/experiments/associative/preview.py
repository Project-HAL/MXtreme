"""Draw what a parameter file will do, before running it: the array with the regions and their
sites, the run as a timeline, and the pulse waveform. Needs no rig."""

from __future__ import annotations

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
    if regions:
        display.draw_map(
            ax_map,
            {r: s.as_dict() for r, s in regions.items()},
            routed,
            params.region_radius_um,
            title=f"well {params.well}: {params.stim_site.get('shape')} sites {params.stim_site}, recording radius {params.region_radius_um} um",
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
