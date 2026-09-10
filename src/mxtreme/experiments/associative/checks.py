"""Turning a calibration or connectivity run into numbers and a verdict.

These are the two gates before hours are spent conditioning a culture:

- **calibration** asks, per region, which amplitude evokes a local response without setting off
  the whole culture (:func:`calibration_table`, :func:`calibration_verdicts`);
- **connectivity** asks how much stimulating each region *alone* drives the other two
  (:func:`crosstalk`, :func:`connectivity_verdicts`).

Both come down to the same measurement: around every pulse, the spikes each region fired just
after it, minus the spikes it fired just before. :func:`pulse_responses` does that and everything
else reads its output, so the arithmetic lives in one place and can be checked without a
recording.

A verdict is advice, not a decision. :class:`Verdict` carries a level -- ``"ok"``, ``"warn"`` or
``"stop"`` -- and a sentence saying what to do about it; :mod:`.report` prints them and hands them
back, and the experimenter decides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: How each level is marked when a verdict is printed.
MARKS = {"ok": "ok  ", "warn": "WARN", "stop": "STOP"}


@dataclass
class Verdict:
    """One thing worth telling the experimenter.

    :param level: ``"ok"``, ``"warn"`` (worth a look, the run can go ahead) or ``"stop"``
        (running as things stand would waste the culture).
    :param message: What was measured and what to do about it.
    """

    level: str
    message: str

    def __str__(self) -> str:
        return f"[{MARKS.get(self.level, '?')}] {self.message}"


def pulse_responses(
    spike_frames: np.ndarray,
    spike_electrodes: np.ndarray,
    regions: dict[str, Sequence[int]],
    fps: float,
    pulses: Sequence[tuple[str, int]],
    window_ms: tuple[float, float] = (5.0, 50.0),
) -> dict[str, dict[str, np.ndarray]]:
    """Spikes each region fired after each pulse, above what it was already firing.

    The window before a pulse is the mirror of the window after it, so the two are the same length
    and the difference is what the pulse added. A negative value means the region happened to be
    quieter after the pulse than before, which is what a region unaffected by the stimulus does
    half the time.

    :param pulses: ``(stimulated_role, frame)`` for every pulse delivered.
    :param window_ms: The response window after each pulse; its mirror is the baseline.
    :returns: ``{stimulated_role: {measured_region: evoked counts, one per pulse}}``.
    """
    a, b = int(window_ms[0] / 1000 * fps), int(window_ms[1] / 1000 * fps)
    members = {name: np.isin(spike_electrodes, np.asarray(list(elecs))) for name, elecs in regions.items()}
    order = np.argsort(spike_frames, kind="stable")
    frames = np.asarray(spike_frames)[order]
    members = {name: mask[order] for name, mask in members.items()}

    out: dict[str, dict[str, list[float]]] = {}
    for role, frame in pulses:
        for name, mask in members.items():
            after = _count(frames, mask, frame + a, frame + b)
            before = _count(frames, mask, frame - b, frame - a)
            out.setdefault(role, {}).setdefault(name, []).append(after - before)
    return {
        role: {name: np.array(v, dtype=float) for name, v in inner.items()} for role, inner in out.items()
    }


def _count(frames: np.ndarray, mask: np.ndarray, lo: int, hi: int) -> int:
    i, j = int(np.searchsorted(frames, lo)), int(np.searchsorted(frames, hi))
    return int(np.count_nonzero(mask[i:j]))


def burst_rate(
    pulses: Sequence[tuple[str, int]],
    bursts: Sequence[tuple[int, int]],
    fps: float,
    within_ms: float = 500.0,
) -> dict[str, float]:
    """Fraction of each region's pulses followed by the start of a network burst.

    A stimulus that reliably starts a burst is driving the whole culture, which both floods the
    readout and couples every region to every other by construction.
    """
    starts = np.array([a for a, _ in bursts], dtype=float)
    reach = within_ms / 1000.0 * fps
    hit: dict[str, list[float]] = {}
    for role, frame in pulses:
        near = bool(np.any((starts >= frame) & (starts <= frame + reach))) if len(starts) else False
        hit.setdefault(role, []).append(float(near))
    return {role: float(np.mean(v)) for role, v in hit.items()}


def crosstalk(responses: dict[str, dict[str, np.ndarray]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """How much stimulating each region drives the others, relative to its own local response.

    ``ratio[i, j]`` is the mean evoked response in region ``j`` when region ``i`` was stimulated,
    divided by the mean evoked response in ``i`` itself. The diagonal is 1 by construction. A
    region whose local response is not positive gives a row of ``nan``: there is nothing to take a
    ratio of, which :func:`connectivity_verdicts` reports separately.

    :returns: Region names, the mean evoked response ``(n, n)``, and the ratio ``(n, n)``.
    """
    names = list(responses)
    for inner in responses.values():
        for name in inner:
            if name not in names:
                names.append(name)
    n = len(names)
    evoked = np.full((n, n), np.nan)
    for i, stim in enumerate(names):
        for j, measured in enumerate(names):
            values = responses.get(stim, {}).get(measured)
            if values is not None and len(values):
                evoked[i, j] = float(np.mean(values))
    ratio = np.full((n, n), np.nan)
    for i in range(n):
        local = evoked[i, i]
        if np.isfinite(local) and local > 0:
            ratio[i] = evoked[i] / local
    return names, evoked, ratio


def connectivity_verdicts(
    names: Sequence[str],
    evoked: np.ndarray,
    ratio: np.ndarray,
    bursts: dict[str, float],
    crosstalk_warn: float = 0.3,
    burst_warn: float = 0.2,
    min_local: float = 0.5,
    min_path: float = 0.02,
    us: str = "US",
    cs: str = "CS",
) -> list[Verdict]:
    """Read a connectivity check: are the three sites independent enough, and is there a path?

    Four things are checked, in the order they would sink a run:

    1. a stimulus that evokes almost nothing locally -- the amplitude or the site is wrong;
    2. a stimulus that starts network bursts -- it is driving the culture, not a site;
    3. a site that drives another above ``crosstalk_warn`` of its own local response -- the two
       are not independent, and any association found later was partly there already;
    4. no measurable response in either direction between CS and US -- there may be no path
       between them to strengthen, so a null result would say nothing.

    The last one is why decorrelation alone is not enough to choose sites on: two patches can be
    uncorrelated because nothing connects them.
    """
    index = {name: i for i, name in enumerate(names)}
    out: list[Verdict] = []

    for name in names:
        i = index[name]
        local = evoked[i, i]
        if not np.isfinite(local) or local < min_local:
            out.append(
                Verdict(
                    "stop",
                    f"stimulating {name} evokes {local:.2f} spikes per pulse in {name} "
                    f"itself, under {min_local}. Raise its amplitude or move the site; "
                    f"nothing measured through it will mean anything.",
                )
            )
        rate = bursts.get(name)
        if rate is not None and rate > burst_warn:
            out.append(
                Verdict(
                    "stop",
                    f"{rate:.0%} of {name} pulses are followed by a network burst "
                    f"(over {burst_warn:.0%}). The stimulus is driving the whole culture, "
                    f"which couples every region to every other. Lower its amplitude.",
                )
            )

    for i, stim in enumerate(names):
        for j, measured in enumerate(names):
            if i == j or not np.isfinite(ratio[i, j]):
                continue
            if ratio[i, j] > crosstalk_warn:
                out.append(
                    Verdict(
                        "warn",
                        f"stimulating {stim} evokes {evoked[i, j]:.2f} spikes per pulse in "
                        f"{measured}, {ratio[i, j]:.0%} of {stim}'s own response. The two sites "
                        f"are not independent; consider re-running selection with a larger "
                        f"separation, or accept that a {stim}-to-{measured} association starts "
                        f"from a high floor.",
                    )
                )

    if us in index and cs in index:
        forward = ratio[index[cs], index[us]]
        backward = ratio[index[us], index[cs]]
        both = [v for v in (forward, backward) if np.isfinite(v)]
        if both and max(both) < min_path:
            out.append(
                Verdict(
                    "warn",
                    f"neither direction between {cs} and {us} shows a measurable response "
                    f"(at most {max(both):.1%} of the local response). The two sites may not "
                    f"be connected at all, in which case conditioning has nothing to "
                    f"strengthen and a null result would say nothing about learning.",
                )
            )

    if not out:
        out.append(
            Verdict(
                "ok",
                "each site responds to its own stimulus, none of them starts network bursts, "
                "cross-talk is below the threshold, and CS and US are measurably connected.",
            )
        )
    return out


def silent_recording(spike_frames: np.ndarray, fps: float, per_second: float = 1.0) -> Verdict | None:
    """A verdict when a recording has almost no spikes in it, or ``None`` when it does.

    On a saline plate nothing fires, so every biological check below would report each site as
    dead -- true, but not what the run was testing. Saying so once, at the top, keeps a dry run
    from reading like a failure: what a saline run tests is that the pulses fired where and when
    they were meant to, which :mod:`.report` checks against the schedule regardless.
    """
    if len(spike_frames) == 0:
        return Verdict(
            "ok",
            "no spikes at all in this recording. On saline that is expected; the "
            "response checks below cannot mean anything, so read the schedule and "
            "artifact sections instead.",
        )
    span = (float(np.max(spike_frames)) - float(np.min(spike_frames))) / fps
    rate = len(spike_frames) / span if span > 0 else float("inf")
    if rate < per_second:
        return Verdict(
            "ok",
            f"only {len(spike_frames)} spikes in {span:.0f} s ({rate:.2f}/s across the "
            f"whole array). On saline that is expected; the response checks below cannot "
            f"mean anything, so read the schedule and artifact sections instead. On a "
            f"culture it means the plate is silent, which is its own problem.",
        )
    return None


def calibration_table(
    responses: dict[str, dict[str, np.ndarray]],
    bursts: dict[str, float],
    tokens: dict[str, tuple[str, float, str]],
) -> list[dict]:
    """One row per (region, amplitude, polarity) tried in a calibration run.

    :param tokens: ``{token: (role, amplitude_mv, polarity)}``, from the schedule.
    :returns: Rows with ``role``, ``amplitude_mv``, ``polarity``, ``local`` (mean evoked spikes per
        pulse in the stimulated region), ``pulses`` and ``burst_rate``, sorted by role then
        amplitude.
    """
    rows = []
    for token, (role, amplitude, polarity) in tokens.items():
        values = responses.get(token, {}).get(role)
        if values is None:
            continue
        rows.append(
            {
                "role": role,
                "amplitude_mv": amplitude,
                "polarity": polarity,
                "local": float(np.mean(values)),
                "pulses": len(values),
                "burst_rate": float(bursts.get(token, 0.0)),
            }
        )
    return sorted(rows, key=lambda r: (r["role"], r["amplitude_mv"], r["polarity"]))


def calibration_verdicts(
    rows: Sequence[dict],
    roles: Sequence[str],
    min_local: float = 0.5,
    burst_warn: float = 0.2,
) -> tuple[dict[str, dict | None], list[Verdict]]:
    """Pick each region's amplitude, and say what went wrong where it did.

    The choice is the largest amplitude that evokes at least ``min_local`` spikes per pulse
    locally while starting a network burst on at most ``burst_warn`` of its pulses -- the same
    rule by hand, made explicit. Where no amplitude satisfies both, the reason is reported instead
    of a number.

    :returns: ``{role: chosen row or None}`` and the verdicts.
    """
    chosen: dict[str, dict | None] = {}
    out: list[Verdict] = []
    for role in roles:
        mine = [r for r in rows if r["role"] == role]
        if not mine:
            chosen[role] = None
            out.append(Verdict("stop", f"{role} was never stimulated in this run."))
            continue
        usable = [r for r in mine if r["local"] >= min_local and r["burst_rate"] <= burst_warn]
        if usable:
            best = max(usable, key=lambda r: (r["amplitude_mv"], r["local"]))
            chosen[role] = best
            out.append(
                Verdict(
                    "ok",
                    f"{role}: {best['amplitude_mv']:.0f} mV {best['polarity']} evokes "
                    f"{best['local']:.1f} spikes per pulse, bursts on {best['burst_rate']:.0%}.",
                )
            )
            continue
        chosen[role] = None
        responsive = [r for r in mine if r["local"] >= min_local]
        if not responsive:
            best = max(mine, key=lambda r: r["local"])
            out.append(
                Verdict(
                    "stop",
                    f"{role}: no amplitude up to {max(r['amplitude_mv'] for r in mine):.0f} mV "
                    f"evoked {min_local} spikes per pulse (best was {best['local']:.2f} at "
                    f"{best['amplitude_mv']:.0f} mV). Extend the ladder, or the site has too "
                    f"few neurons under it and should be reselected.",
                )
            )
        else:
            quietest = min(responsive, key=lambda r: r["burst_rate"])
            out.append(
                Verdict(
                    "stop",
                    f"{role}: every amplitude that evokes a response also starts network "
                    f"bursts (the quietest, {quietest['amplitude_mv']:.0f} mV, bursts on "
                    f"{quietest['burst_rate']:.0%} of pulses). Try shorter phases or a "
                    f"sparser site; conditioning through it would drive the whole culture.",
                )
            )
    return chosen, out
