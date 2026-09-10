"""Draw what a parameter file will do, before running it: the array with the regions and their
sites, the run as a timeline, and the pulse waveform. Needs no rig."""

from __future__ import annotations

from mxtreme.experiments.associative import display, protocol
from mxtreme.experiments.associative.params import AssociativeParams


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
    display.draw_timeline(ax_time, blocks, dt_cs_us_ms=params.dt_cs_us)
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
