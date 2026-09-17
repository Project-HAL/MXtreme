"""Turning a calibration run, or the session's baseline block, into numbers and a verdict.

These are the two gates before hours are spent conditioning a culture:

- **calibration** asks, per region, which amplitude evokes a local response without setting off
  the whole culture (:func:`calibration_table`, :func:`calibration_verdicts`);
- **the baseline gate** asks how much stimulating each region *alone* drives the other two
  (:func:`crosstalk`, :func:`independence_verdicts`).

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

#: Spikes per pulse a region has to evoke locally for an amplitude to count as usable.
MIN_LOCAL = 0.5
#: Below this a detection is a noise crossing, not a spike: the same amplitude the scans' "active
#: electrode" definition uses. A plate whose detections are all under it is silent.
MIN_SPIKE_UV = 20.0

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
    ratio of, which :func:`independence_verdicts` reports separately.

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


def independence_verdicts(
    names: Sequence[str],
    evoked: np.ndarray,
    ratio: np.ndarray,
    bursts: dict[str, float],
    crosstalk_warn: float = 0.3,
    burst_warn: float = 0.2,
    min_local: float = MIN_LOCAL,
    min_path: float = 0.02,
    us: str = "US",
    cs: str = "CS",
) -> list[Verdict]:
    """Read the baseline gate: are the three sites independent enough, and is there a path?

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


def outside_artifacts(
    spike_frames: np.ndarray, pulse_frames, fps: float, before_ms: float = 1.0, after_ms: float = 5.0
) -> np.ndarray:
    """The spikes not within the stimulation artifact's window around any pulse.

    The detector fires on the artifact on every channel a pulse reaches, so a calibration holds
    tens of thousands of such "spikes" whatever is on the plate, and they would make a silent one
    look lively to :func:`silent_recording`.
    """
    frames = np.sort(np.asarray(spike_frames, dtype=np.int64))
    if not len(frames) or not len(pulse_frames):
        return frames
    starts = np.sort(np.asarray(pulse_frames, dtype=np.int64)) - int(before_ms / 1000 * fps)
    ends = starts + int((before_ms + after_ms) / 1000 * fps)
    # For each spike, the last pulse window that starts at or before it; the spike is inside an
    # artifact if that window has not ended.
    k = np.searchsorted(starts, frames, side="right") - 1
    inside = (k >= 0) & (frames < ends[np.clip(k, 0, len(ends) - 1)])
    return frames[~inside]


def silent_recording(spike_frames: np.ndarray, fps: float, per_second: float = 1.0) -> Verdict | None:
    """A verdict when a recording has almost no spikes outside the stimulation artifacts, or
    ``None`` when it has plenty: on a silent plate the response checks cannot mean anything, and
    saying so once at the top is clearer than three dead-site verdicts."""
    if len(spike_frames) == 0:
        return Verdict(
            "stop",
            "no spikes outside the stimulation artifacts: the plate is silent, so the response "
            "checks below cannot mean anything. The schedule and artifact sections still can.",
        )
    span = (float(np.max(spike_frames)) - float(np.min(spike_frames))) / fps
    rate = len(spike_frames) / span if span > 0 else float("inf")
    if rate < per_second:
        return Verdict(
            "stop",
            f"only {len(spike_frames)} spikes outside the stimulation artifacts in {span:.0f} s "
            f"({rate:.2f}/s across the whole array): the plate is silent, so the response checks "
            f"below cannot mean anything. The schedule and artifact sections still can.",
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
        pulse in the stimulated region), ``se`` (its standard error over the pulses), ``remote``
        (the mean over the other regions: how far the pulse reaches), ``spread`` (remote over
        local), ``pulses`` and ``burst_rate``, sorted by role then amplitude.
    """
    rows = []
    for token, (role, amplitude, polarity) in tokens.items():
        values = responses.get(token, {}).get(role)
        if values is None:
            continue
        others = [float(np.mean(v)) for name, v in responses[token].items() if name != role and len(v)]
        local = float(np.mean(values))
        remote = float(np.mean(others)) if others else float("nan")
        rows.append(
            {
                "role": role,
                "amplitude_mv": amplitude,
                "polarity": polarity,
                "local": local,
                "se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan"),
                "remote": remote,
                "spread": remote / local if local > 0 and np.isfinite(remote) else float("nan"),
                "pulses": len(values),
                "burst_rate": float(bursts.get(token, 0.0)),
            }
        )
    return sorted(rows, key=lambda r: (r["role"], r["amplitude_mv"], r["polarity"]))


def compare_calibrations(
    tables: dict[str, Sequence[dict]],
    roles: Sequence[str],
    min_local: float = MIN_LOCAL,
    burst_warn: float = 0.2,
) -> tuple[list[dict], list[Verdict]]:
    """Set two or more calibration runs of the same regions side by side -- typically the same
    sites driven through different electrode patterns -- and say which pattern did better.

    Per pattern and region, three numbers decide it, in this order:

    1. **threshold**: the lowest amplitude that evokes ``min_local`` spikes per pulse locally
       without bursting -- lower means the pattern reaches neurons more easily;
    2. **local response at the shared amplitude**: the largest amplitude every pattern has a
       usable row at, so the patterns are compared at equal drive;
    3. **spread** at that amplitude: the remote response as a fraction of the local one -- lower
       means the pattern stays a site rather than a broadcast, which is what the return ring is
       for.

    A pattern wins a region when it has the lower threshold, or the same threshold and the higher
    local response, or the same of both and less spread -- unless it wins on drive while
    spreading more than a third further than the runner-up, which is flagged rather than
    decided. Equal on all three is a draw, and the verdict says so.

    :param tables: ``{pattern label: rows from calibration_table}``.
    :returns: One summary row per (pattern, region), and the verdicts.
    """
    summary: list[dict] = []
    for label, rows in tables.items():
        for role in roles:
            mine = [r for r in rows if r["role"] == role]
            usable = [r for r in mine if r["local"] >= min_local and r["burst_rate"] <= burst_warn]
            threshold = min((r["amplitude_mv"] for r in usable), default=None)
            summary.append(
                {
                    "pattern": label,
                    "role": role,
                    "threshold_mv": threshold,
                    "rows": mine,
                    "usable": usable,
                }
            )

    verdicts: list[Verdict] = []
    labels = list(tables)
    for role in roles:
        entries = {s["pattern"]: s for s in summary if s["role"] == role}
        shared = None
        common = set.intersection(*({r["amplitude_mv"] for r in e["usable"]} for e in entries.values()))
        if common:
            shared = max(common)
        for e in entries.values():
            at = [r for r in e["usable"] if r["amplitude_mv"] == shared] if shared is not None else []
            e["at_shared"] = at[0] if at else None
            e["local_at_shared"] = at[0]["local"] if at else None
            e["spread_at_shared"] = (
                (at[0]["remote"] / at[0]["local"])
                if at and at[0]["local"] > 0 and np.isfinite(at[0]["remote"])
                else None
            )
        with_threshold = [e for e in entries.values() if e["threshold_mv"] is not None]
        if not with_threshold:
            verdicts.append(
                Verdict("stop", f"{role}: no pattern evoked {min_local} spikes per pulse without bursting.")
            )
            continue
        if len(with_threshold) < len(entries):
            dead = [e["pattern"] for e in entries.values() if e["threshold_mv"] is None]
            for e in with_threshold:
                verdicts.append(
                    Verdict(
                        "ok",
                        f"{role}: {e['pattern']} responds (threshold {e['threshold_mv']:.0f} mV); "
                        f"{', '.join(dead)} never did. Use {e['pattern']}.",
                    )
                )
            continue
        ranked = sorted(
            with_threshold,
            key=lambda e: (
                e["threshold_mv"],
                -(e["local_at_shared"] or 0.0),
                e["spread_at_shared"] if e["spread_at_shared"] is not None else float("inf"),
            ),
        )
        if len(ranked) == 1:
            verdicts.append(
                Verdict("ok", f"{role}: {ranked[0]['pattern']} threshold {ranked[0]['threshold_mv']:.0f} mV.")
            )
            continue
        best, runner = ranked[0], ranked[1]
        spread_b, spread_r = best["spread_at_shared"], runner["spread_at_shared"]
        worse_spread = spread_b is not None and spread_r is not None and spread_b > spread_r * 4 / 3
        detail = (
            f"threshold {best['threshold_mv']:.0f} vs {runner['threshold_mv']:.0f} mV"
            + (
                f"; at {shared:.0f} mV local {best['local_at_shared']:.2f} vs {runner['local_at_shared']:.2f} spikes/pulse"
                if shared is not None
                else ""
            )
            + (
                f", spread {spread_b:.0%} vs {spread_r:.0%}"
                if spread_b is not None and spread_r is not None
                else ""
            )
        )
        same_drive = best["threshold_mv"] == runner["threshold_mv"] and (best["local_at_shared"] or 0) == (
            runner["local_at_shared"] or 0
        )
        if same_drive and (spread_b is None or spread_r is None or spread_b == spread_r):
            verdicts.append(
                Verdict("warn", f"{role}: {best['pattern']} and {runner['pattern']} tie ({detail}).")
            )
        elif same_drive:
            verdicts.append(
                Verdict("ok", f"{role}: {best['pattern']}, the same drive with less spread ({detail}).")
            )
        elif worse_spread:
            verdicts.append(
                Verdict(
                    "warn",
                    f"{role}: {best['pattern']} drives the site harder but spreads further "
                    f"({detail}). Prefer {runner['pattern']} unless local response is what is short.",
                )
            )
        else:
            verdicts.append(Verdict("ok", f"{role}: {best['pattern']} ({detail})."))
    if len(labels) < 2:
        verdicts.append(Verdict("warn", "only one calibration given; nothing to compare it with."))
    return summary, verdicts


def calibration_verdicts(
    rows: Sequence[dict],
    roles: Sequence[str],
    min_local: float = MIN_LOCAL,
    burst_warn: float = 0.2,
    crosstalk_warn: float = 0.3,
    min_pulses: int = 3,
) -> tuple[dict[str, dict | None], list[Verdict]]:
    """Pick each region's polarity and amplitude, and say what went wrong where it did.

    Per region, a row is *usable* when, over at least ``min_pulses`` pulses, it evokes at least
    ``min_local`` spikes per pulse locally and that mean is at least twice its standard error (so
    a handful of chance detections cannot pass), while the other regions respond by no more than
    ``crosstalk_warn`` of the local response (the pulse stays a site, not a broadcast) and a
    network burst starts on at most ``burst_warn`` of its pulses. When more than one
    polarity was swept, the polarity with the lower usable threshold wins -- it reaches neurons
    with less voltage -- with anodic-first on a tie (the more effective order in the literature).
    Within that polarity the choice is the largest usable amplitude, since a site that barely
    responds at threshold responds unreliably. Then the three are made comparable: a region
    whose response is more than twice the weakest region's is stepped down to the largest usable
    amplitude within that bound. Where nothing is usable, the reason is reported instead of a
    number.

    :returns: ``{role: chosen row or None}`` (the row carries ``polarity``) and the verdicts.
    """
    chosen: dict[str, dict | None] = {}
    out: list[Verdict] = []
    usable_by_role: dict[str, list[dict]] = {}
    for role in roles:
        mine = [r for r in rows if r["role"] == role]
        if not mine:
            chosen[role] = None
            out.append(Verdict("stop", f"{role} was never stimulated in this run."))
            continue

        def reliable(r):
            se = r.get("se", 0.0)
            return r.get("pulses", min_pulses) >= min_pulses and (not np.isfinite(se) or r["local"] >= 2 * se)

        def focal(r):
            spread = r.get("spread", float("nan"))
            return not np.isfinite(spread) or spread <= crosstalk_warn

        usable = [
            r
            for r in mine
            if r["local"] >= min_local and reliable(r) and focal(r) and r["burst_rate"] <= burst_warn
        ]
        usable_by_role[role] = usable
        if usable:
            thresholds = {}
            for r in usable:
                thresholds[r["polarity"]] = min(
                    thresholds.get(r["polarity"], r["amplitude_mv"]), r["amplitude_mv"]
                )
            polarity = min(thresholds, key=lambda pol: (thresholds[pol], pol != "anodic-first"))
            best = max(
                (r for r in usable if r["polarity"] == polarity),
                key=lambda r: (r["amplitude_mv"], r["local"]),
            )
            chosen[role] = best
            others = {pol: t for pol, t in thresholds.items() if pol != polarity}
            why = (
                f" ({polarity} reaches threshold at {thresholds[polarity]:.0f} mV, "
                + ", ".join(f"{pol} at {t:.0f}" for pol, t in others.items())
                + ")"
                if others
                else ""
            )
            out.append(
                Verdict(
                    "ok",
                    f"{role}: {best['amplitude_mv']:.0f} mV {polarity} evokes "
                    f"{best['local']:.1f} spikes per pulse, bursts on {best['burst_rate']:.0%}{why}.",
                )
            )
            top = max(r["amplitude_mv"] for r in mine)
            if thresholds[polarity] == top:
                # Responds only at the top rung: the site is on few neurons, and the amplitude
                # has nowhere to go if the response fades. Say so now, while reselecting is cheap.
                out.append(
                    Verdict(
                        "warn",
                        f"{role} responds only at the top of the ladder ({top:.0f} mV). The site is "
                        f"on few neurons; a new centre (select --centers) is a better fix than "
                        f"raising max_amplitude_mv.",
                    )
                )
            continue
        chosen[role] = None
        responsive = [r for r in mine if r["local"] >= min_local and reliable(r)]
        if not responsive:
            best = max(mine, key=lambda r: r["local"])
            few = [r for r in mine if r["local"] >= min_local and not reliable(r)]
            if few:
                out.append(
                    Verdict(
                        "stop",
                        f"{role}: {len(few)} amplitude(s) reached {min_local} spikes per pulse but not "
                        f"reliably (fewer than {min_pulses} pulses, or a mean under twice its standard "
                        f"error): chance detections, or too few repeats to tell. More calibration_reps.",
                    )
                )
            else:
                out.append(
                    Verdict(
                        "stop",
                        f"{role}: no amplitude up to {max(r['amplitude_mv'] for r in mine):.0f} mV "
                        f"evoked {min_local} spikes per pulse (best was {best['local']:.2f} at "
                        f"{best['amplitude_mv']:.0f} mV). Extend the ladder, or the site has too "
                        f"few neurons under it and should be reselected.",
                    )
                )
        elif not any(focal(r) for r in responsive):
            tightest = min(responsive, key=lambda r: r.get("spread", float("inf")))
            out.append(
                Verdict(
                    "stop",
                    f"{role}: every amplitude that evokes a response also drives the other regions "
                    f"(the most confined, {tightest['amplitude_mv']:.0f} mV, spreads "
                    f"{tightest.get('spread', float('nan')):.0%} of its local response, over "
                    f"{crosstalk_warn:.0%}). The regions are not independent at any usable drive: "
                    f"wider separation, or return electrodes to confine the field.",
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
    # The three regions are compared with each other, so what should match is the drive each
    # delivers (spikes per pulse at its site), not the voltage: a site on more neurons needs fewer
    # mV for the same effect. A region whose response is more than twice the weakest's is stepped
    # down to the largest usable rung of its polarity within that bound, and flagged if there is
    # none. No region is privileged: US is not driven harder because it is the US.
    picked = {role: r for role, r in chosen.items() if r}
    if len(picked) > 1:
        weakest_role, weakest = min(picked.items(), key=lambda kv: kv[1]["local"])
        bound = 2 * weakest["local"]
        for role, r in picked.items():
            if r["local"] <= bound:
                continue
            within = [
                u
                for u in usable_by_role.get(role, [])
                if u["polarity"] == r["polarity"] and u["local"] <= bound
            ]
            if within:
                step = max(within, key=lambda u: (u["amplitude_mv"], u["local"]))
                chosen[role] = step
                out.append(
                    Verdict(
                        "ok",
                        f"{role} stepped down from {r['amplitude_mv']:.0f} to {step['amplitude_mv']:.0f} mV "
                        f"({step['local']:.1f} spikes per pulse) to be comparable with {weakest_role} "
                        f"({weakest['local']:.1f}): the three regions are compared with each other, so "
                        "none should be driven much harder than another.",
                    )
                )
            else:
                out.append(
                    Verdict(
                        "warn",
                        f"{role} is not matched in drive: it evokes {r['local']:.1f} spikes per pulse "
                        f"against {weakest['local']:.1f} for {weakest_role}, and no lower usable rung of "
                        "its ladder gets within twice that. Read its results knowing it is driven harder.",
                    )
                )
    return chosen, out


def _rate(group: dict, role: str) -> tuple[float, float] | None:
    """Per-pulse rate and its Poisson standard error for one role in one checkpoint, or None."""
    pulses = group.get("pulses", {}).get(role, 0)
    if not pulses:
        return None
    spikes = max(0, int(group.get("spikes", {}).get(role, 0)))
    return spikes / pulses, (max(spikes, 1) ** 0.5) / pulses


def conditioning_verdicts(
    curve: Sequence[dict],
    bursts: dict[str, float] | None = None,
    crosstalk_warn: float = 0.3,
    burst_warn: float = 0.2,
    drift_warn: float = 0.5,
    us: str = "US",
    cs: str = "CS",
    ns: str = "NS",
) -> list[Verdict]:
    """Read a conditioning run's checkpoint curve: was the experiment sound, and what did it show?

    ``curve`` is :func:`~mxtreme.experiments.associative.report.checkpoints`' output: per
    checkpoint, per role probed alone, the spikes in US per pulse with the totals behind it. The
    error bars are Poisson on the pooled pulses, the same arithmetic as the probe design.

    Checked, in the order they would undermine a result:

    1. **high floor** -- at baseline, CS alone already drives US above ``crosstalk_warn`` of US's
       own local response; the sites were not independent and the association starts from a
       floor. Visible after the baseline block, 16 minutes in;
    2. **bursting probes** -- more than ``burst_warn`` of probe or training pulses started a
       network burst: the stimulus is driving the culture, not a site;
    3. **excitability drift** -- US alone at the retrievals differs from baseline by more than
       ``drift_warn``; a change in CS alone then has to be read against it;
    4. **the result** -- CS alone at the retrievals against baseline, at two standard errors, and
       NS alone the same way. CS up and NS flat is the association; both up is excitability;
       neither is a null, and the note says what to try next.

    :param bursts: ``{"probe": fraction, "train": fraction}`` of pulses followed by a burst.
    """
    out: list[Verdict] = []
    if not curve:
        return [Verdict("stop", "no probe-alone presentations found; nothing to read.")]
    baseline = curve[0]
    retrievals = [c for c in curve if str(c["label"]).startswith("retrieval")]
    encode = [c for c in curve if str(c["label"]).startswith("encode")]

    # 1. Independence at baseline.
    base_cs, base_us = _rate(baseline, cs), _rate(baseline, us)
    if base_cs and base_us and base_us[0] > 0 and base_cs[0] / base_us[0] > crosstalk_warn:
        out.append(
            Verdict(
                "warn",
                f"at baseline {cs} alone already evokes {base_cs[0]:.2f} spikes per pulse in {us}, "
                f"{base_cs[0] / base_us[0]:.0%} of {us}'s own response (over {crosstalk_warn:.0%}). The "
                f"sites are not independent, and the association starts from a high floor: any rise "
                f"is measured against it, and a reselection with wider separation is the fix for the "
                f"next culture.",
            )
        )

    # 2. Bursts.
    for kind, label in (("probe", "probe pulses"), ("train", "training pulses")):
        rate = (bursts or {}).get(kind)
        if rate is not None and rate > burst_warn:
            out.append(
                Verdict(
                    "warn",
                    f"{rate:.0%} of {label} were followed by a network burst (over {burst_warn:.0%}); "
                    f"the stimulus is driving the whole culture and the readout window is flooded. "
                    f"Lower the amplitudes next time.",
                )
            )

    if not retrievals:
        out.append(
            Verdict(
                "ok",
                f"no retrieval block yet: {len(encode)} checkpoint(s) after baseline. The curve so far is "
                f"all there is to read.",
            )
        )
        return out

    def pooled(groups, role):
        spikes = sum(int(g.get("spikes", {}).get(role, 0)) for g in groups)
        pulses = sum(int(g.get("pulses", {}).get(role, 0)) for g in groups)
        return (spikes / pulses, (max(spikes, 1) ** 0.5) / pulses) if pulses else None

    # 3. Excitability.
    ret_us = pooled(retrievals, us)
    if base_us and ret_us and base_us[0] > 0:
        drift = (ret_us[0] - base_us[0]) / base_us[0]
        if abs(drift) > drift_warn:
            out.append(
                Verdict(
                    "warn",
                    f"{us} alone changed by {drift:+.0%} from baseline to the retrievals: {us}'s own "
                    f"excitability drifted, so read {cs} alone relative to it, not on its own.",
                )
            )

    # 4. The result, at two standard errors on the pooled pulses.
    def change(role):
        before, after = _rate(baseline, role), pooled(retrievals, role)
        if not before or not after:
            return None
        delta = after[0] - before[0]
        se = (before[1] ** 2 + after[1] ** 2) ** 0.5
        return delta, se, before[0], after[0]

    d_cs, d_ns = change(cs), change(ns)
    if d_cs is None:
        out.append(Verdict("stop", f"{cs} alone was not probed at both baseline and retrieval."))
        return out
    cs_up = d_cs[0] > 2 * d_cs[1]
    ns_up = d_ns is not None and d_ns[0] > 2 * d_ns[1]
    trend = ""
    if encode:
        cs_points = [_rate(c, cs) for c in encode]
        if all(cs_points):
            trend = ", checkpoints " + " -> ".join(f"{p[0]:.2f}" for p in cs_points)
    if cs_up and not ns_up:
        out.append(
            Verdict(
                "ok",
                f"association: {cs} alone in {us} rose from {d_cs[2]:.2f} to {d_cs[3]:.2f} spikes per "
                f"pulse ({d_cs[0] / d_cs[1]:.1f} SE){trend}"
                + (f"; {ns} alone {d_ns[2]:.2f} to {d_ns[3]:.2f}, within noise." if d_ns else "."),
            )
        )
    elif cs_up and ns_up:
        out.append(
            Verdict(
                "warn",
                f"both {cs} alone ({d_cs[2]:.2f} to {d_cs[3]:.2f}) and {ns} alone ({d_ns[2]:.2f} to "
                f"{d_ns[3]:.2f}) rose in {us}: a general change in excitability, not an association. "
                f"The specific part, if any, is the difference between them.",
            )
        )
    else:
        out.append(
            Verdict(
                "warn",
                f"no association: {cs} alone in {us} went {d_cs[2]:.2f} to {d_cs[3]:.2f} spikes per pulse "
                f"({d_cs[0] / d_cs[1]:+.1f} SE){trend}. A flat curve after 712 paired pulses says single "
                f"pulses did not change anything at this age; the tetanic variant "
                f"(pulses_per_burst=11, burst_hz=20) is the next thing to try, with the burst cap watched.",
            )
        )
    return out
