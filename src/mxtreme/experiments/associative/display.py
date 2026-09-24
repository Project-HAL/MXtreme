"""Drawing the associative protocol: the array map, the presentation timeline, the waveform.
Shared by :mod:`.preview` (before the run), :mod:`.select` (choosing regions) and :mod:`.report`
(after it), so they look the same and differ only where the run did."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from mxtreme.experiments.associative import protocol
from mxtreme.experiments.associative.params import ROLES

ROLE_COLOURS = {"US": "#3b82f6", "CS": "#ef4444", "NS": "#22c55e"}
PAIR_COLOUR = "#7c3aed"
ARRAY_BG = "#0b0b12"
GRID_COLOUR = "#1e1e2b"
ROUTED_COLOUR = "#4a4a5e"


def routed_positions(cfg_path: str) -> np.ndarray:
    """(N, 2) um positions of every electrode in a MaxLab ``.cfg``."""
    from mxtreme.scans import mx_config

    return np.array([(float(e[2]), float(e[3])) for e in mx_config.read_config(cfg_path)])


def draw_map(
    ax,
    regions: dict[str, dict],
    routed_xy: np.ndarray | None = None,
    radius_um: float = 150.0,
    fill: dict[str, float] | None = None,
    title: str = "",
    field: tuple | None = None,
):
    """The 220 x 120 array with the three regions on it: driven electrodes filled, return
    electrodes hollow, recording electrodes faint, a circle per region.

    :param regions: ``{role: RegionSpec.as_dict()}``.
    :param routed_xy: (N, 2) positions of every routed electrode, or None.
    :param fill: ``{role: 0..1}`` to shade each region's circle.
    :param field: ``(values_uv, extent)`` from :func:`mxtreme.experiments.associative.preview.field_overlay`,
        shaded behind everything on a log scale. It is the potential the pulses put on the array,
        not a claim about how far they excite: no contour here is a threshold.
    """
    from matplotlib.patches import Circle

    ax.set_facecolor(ARRAY_BG)
    ax.set_aspect("equal")
    ax.set_xlim(-2 * protocol.PITCH_UM, (protocol.COLS + 1) * protocol.PITCH_UM)
    ax.set_ylim((protocol.ROWS + 1) * protocol.PITCH_UM, -2 * protocol.PITCH_UM)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    if title:
        ax.set_title(title, loc="left", fontsize=10)

    if field is not None:
        values, extent = field
        ax.imshow(
            np.log10(np.maximum(values, 1.0)),
            extent=extent,
            origin="upper",
            cmap="magma",
            alpha=0.55,
            zorder=0,
            interpolation="bilinear",
        )
    gx, gy = np.meshgrid(
        np.arange(protocol.COLS) * protocol.PITCH_UM, np.arange(protocol.ROWS) * protocol.PITCH_UM
    )
    ax.scatter(gx.ravel(), gy.ravel(), marker="s", s=1.5, c=GRID_COLOUR, edgecolors="none", zorder=1)
    if routed_xy is not None and len(routed_xy):
        ax.scatter(
            routed_xy[:, 0], routed_xy[:, 1], marker="s", s=4, c=ROUTED_COLOUR, edgecolors="none", zorder=2
        )

    for role, region in regions.items():
        colour = ROLE_COLOURS.get(role, "#dddddd")
        cx, cy = region["center_um"]
        for key, kw in (
            ("rec_electrodes", {"s": 6, "c": colour, "alpha": 0.5, "edgecolors": "none", "zorder": 3}),
            (
                "drive_electrodes",
                {"s": 22, "c": colour, "edgecolors": "white", "linewidths": 0.4, "zorder": 5},
            ),
            (
                "return_electrodes",
                {"s": 22, "facecolors": "none", "edgecolors": colour, "linewidths": 1.2, "zorder": 5},
            ),
        ):
            elecs = region.get(key) or []
            if elecs:
                xy = np.array([protocol.electrode_xy(e) for e in elecs])
                ax.scatter(xy[:, 0], xy[:, 1], marker="s", **kw)
        alpha = 0.0 if fill is None else 0.6 * float(np.clip(fill.get(role, 0.0), 0.0, 1.0))
        ax.add_patch(
            Circle(
                (cx, cy), radius_um, facecolor=colour, alpha=alpha, edgecolor=colour, linewidth=1.2, zorder=4
            )
        )
        ax.text(
            cx,
            cy - radius_um - 25,
            role,
            color=colour,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=6,
        )


def draw_triangle(ax, regions: dict[str, dict]):
    pts = [regions[r]["center_um"] for r in ROLES]
    for p0, p1 in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[0], pts[2])):
        side = protocol.separation_um(p0, p1)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#cccccc", lw=0.8, ls="--", zorder=3)
        ax.text(
            (p0[0] + p1[0]) / 2,
            (p0[1] + p1[1]) / 2,
            f"{side:.0f} um",
            color="#cccccc",
            fontsize=7,
            ha="center",
            zorder=6,
        )


def draw_timeline(
    ax,
    blocks: Sequence[protocol.Block],
    block_starts: Sequence[float] | None = None,
    now_sec: float | None = None,
    fired: Iterable[tuple[float, str]] | None = None,
    bursts: Iterable[float] | None = None,
    dt_cs_us_ms: float = 0.0,
    stimuli: dict[str, protocol.Stimulus] | dict[str, dict] | None = None,
    recording: str | None = None,
):
    """Rows for US, CS, NS; a bar per presentation; blocks shaded and labelled. ``fired`` are
    ``(t_sec, token)`` presentations that actually happened, drawn over the plan.

    With ``stimuli`` each presentation is drawn as long as it really is, which is the difference
    between a one-minute probe and a ten-minute training train: without it they look alike, and a
    training block of a few long trains reads as less stimulation than a probe block of many short
    ones when it is in fact far more.
    """
    starts = list(block_starts) if block_starts is not None else protocol.block_starts_sec(blocks)
    rows = {role: i for i, role in enumerate(reversed(ROLES))}
    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.set_ylim(-0.9, len(rows) - 0.05)
    ax.set_xlabel("minutes")
    for y in rows.values():
        ax.axhline(y, color="#cccccc", lw=0.6, zorder=1)
    for i, block in enumerate(blocks):
        t0 = starts[i] / 60.0
        t1 = (starts[i] + block.duration_sec) / 60.0 if i + 1 >= len(starts) else starts[i + 1] / 60.0
        if block.presentations:
            ax.axvspan(t0, t1, color="#f3f4f6" if i % 2 else "#e5e7eb", zorder=0)
        ax.text(
            (t0 + t1) / 2,
            len(rows) - (0.45 if i % 2 else 0.24),  # staggered: short blocks crowd their labels
            block.label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        for p in block.presentations:
            t = (starts[i] + p.t_sec) / 60.0
            stim = (stimuli or {}).get(p.token)
            span = (
                stim["duration_sec"] if isinstance(stim, dict) else getattr(stim, "duration_sec", 0.0)
            ) / 60.0
            for role in p.roles:
                offset = dt_cs_us_ms / 60000.0 if (role == "US" and len(p.roles) > 1) else 0.0
                alpha = 0.5 if fired else 1.0
                if span:
                    ax.fill_between(
                        [t + offset, t + offset + span],
                        rows[role] - 0.28,
                        rows[role] + 0.28,
                        color=ROLE_COLOURS[role],
                        lw=0,
                        alpha=0.35 * alpha,
                        zorder=2,
                    )
                ax.plot(
                    [t + offset] * 2,
                    [rows[role] - 0.35, rows[role] + 0.35],
                    color=ROLE_COLOURS[role],
                    lw=1.6,
                    zorder=3,
                    alpha=alpha,
                )
    for t, token in fired or []:
        for role in protocol.roles_in(token):
            ax.plot(
                [t / 60.0] * 2,
                [rows[role] - 0.35, rows[role] + 0.35],
                color=ROLE_COLOURS[role],
                lw=2.2,
                zorder=4,
            )
    for t in bursts or []:
        ax.axvline(t / 60.0, color="#a78bfa", lw=0.5, alpha=0.6, zorder=2)
    if now_sec is not None:
        ax.axvline(now_sec / 60.0, color="black", lw=1.0, zorder=5)
    total = (starts[-1] + blocks[-1].duration_sec) / 60.0 if blocks else 1.0
    ax.set_xlim(0, max(total, (now_sec or 0) / 60.0) * 1.01)
    if recording:
        # The run is one recording from end to end, quiet blocks included: draw the span it
        # covers, so what is kept is not left to be inferred from where the ticks are.
        ax.fill_between([0, total], -0.86, -0.74, color="#111827", lw=0, zorder=3)
        ax.text(
            total / 2,
            -0.71,
            recording,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
        )


def draw_waveforms(ax_train, ax_pulse, stim: protocol.Stimulus, params, amplitudes_mv, with_return=False):
    """One presentation's DAC output: the whole train on the left, one event on the right, a
    trace per role stacked vertically, a focal site's ring trace dashed."""
    waves = protocol.stimulus_waveform(stim, params, amplitudes_mv, with_return)
    amp = max(max(abs(v) for v in vs) for _, vs in waves.values()) or 1.0
    step = 2.4 * amp
    for ax in (ax_train, ax_pulse):
        ax.set_yticks([i * step for i in range(len(waves))], list(waves))
        ax.grid(axis="x", color="#eeeeee")
    for i, (label, (t, v)) in enumerate(waves.items()):
        colour = ROLE_COLOURS[label.split()[0]]
        style = "--" if label.endswith("return") else "-"
        ax_train.plot(t, [x + i * step for x in v], style, color=colour, lw=0.9)
        ax_pulse.plot([x * 1000.0 for x in t], [x + i * step for x in v], style, color=colour, lw=1.4)
    ax_train.set_xlim(-0.02 * stim.duration_sec, stim.duration_sec * 1.02 + 0.001)
    ax_train.set_xlabel("seconds")
    ax_train.set_title(
        f"{stim.token}: {stim.num_events} event(s) at {stim.pulse_hz} Hz, {stim.duration_sec:.0f} s",
        loc="left",
        fontsize=10,
    )
    geometry = protocol.geometry_of(params)
    span = max(stim.delays_sec.values()) + geometry.event_span / 20000.0
    ax_pulse.set_xlim(-0.15 * span * 1000.0 - 0.1, span * 1000.0 * 1.15 + 0.1)
    ax_pulse.set_xlabel("ms")
    amps = ", ".join(
        f"{r} {(stim.amplitude_mv if stim.amplitude_mv is not None else amplitudes_mv[r]):.0f} mV"
        for r in stim.roles
    )
    ax_pulse.set_title(
        f"one event: {stim.pulses_per_burst} {stim.polarity} pulse(s)"
        + (f" at {stim.burst_hz} Hz" if stim.pulses_per_burst > 1 else "")
        + f", {params.stim_phase_us:.0f} us per phase, {amps}"
        + (f", US delayed {params.dt_cs_us:.0f} ms" if len(stim.roles) > 1 else ""),
        loc="left",
        fontsize=10,
    )
