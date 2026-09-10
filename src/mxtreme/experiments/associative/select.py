"""Choosing where the three regions go: from a scan to a parameter file with ``regions`` and
``rec_electrodes`` filled in, and a picture of every decision on the way.

The steps, each of which can take its input from a file so the whole thing can be re-run or tried
offline:

1. an activity scan (``activity_scan``), or any unstimulated preprocessed recording of the well
   (``scan_npz``), or ``scan_with`` to run a scan now, or ``centers`` to name candidates by hand.
   A scan is the right input here and only here: it says where the culture is firing, which is a
   per-electrode question its separate rounds can each answer;
2. candidate patches: the densest ``candidates`` patches at least ``min_region_separation_um``
   apart (:func:`mxtreme.scans.region_selection.rank_patches`);
3. a baseline: ``baseline`` (a recording with the candidates routed and no stimulation) or
   ``record_baseline`` to record one now, at most five minutes. This must be **one continuous
   recording**, not a scan: coupling is a question about two patches at the same moment, and a
   scan's rounds never observed two different electrode subsets together. Passing a multi-round
   file is refused rather than silently answered from its first round;
4. coupling and roles: correlation outside network bursts and burst-order consistency, the
   least-coupled near-equilateral triple whose two conditioned sites are matched in coupling to
   the readout, US at the vertex equidistant from the other two, and CS versus NS decided by
   the seed's parity so it alternates across cultures rather than following the data
   (:func:`mxtreme.scans.region_selection.assign_roles`).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np

from mxtreme.experiments.associative import display, protocol
from mxtreme.experiments.associative.params import ROLES, AssociativeParams

BASELINE_SEC_MAX = 300


def parse_centers(text: str) -> list[tuple[float, float]]:
    """``'x,y;x,y;...'`` in um -> ``[(x, y), ...]``."""
    out = []
    for pair in text.split(";"):
        x, y = pair.split(",")
        out.append((float(x), float(y)))
    return out


def _load_scan(path, well):
    from mxtreme.scans import electrode_selection, region_selection

    data = electrode_selection.load_activity_scan(path)
    if well not in data:
        raise ValueError(f"{path} has no well {well}; it has {sorted(data)}")
    # region_selection's own thresholding rather than electrode_selection's, because a scan
    # records some electrodes in more than one round and their rates have to be divided by their
    # own recorded time; see region_selection.active_electrodes.
    return region_selection.active_electrodes(data[well])


def _run_scan(params: AssociativeParams, toml_path: str):
    from mxtreme.config import Config
    from mxtreme.scans import activity_scan

    scan = activity_scan.ActivityScanParams(
        batch=params.batch,
        chip=params.chip,
        plate_date=params.plate_date,
        div=params.div,
        wells=[params.well],
        description="activity scan for associative region selection",
    )
    result = activity_scan.run_activity_scan(scan, Config.from_toml(toml_path))
    activity_scan.ensure_record_time(result.h5_path)
    return str(result.h5_path)


def _record_baseline(params: AssociativeParams, electrodes, seconds, toml_path: str):
    from mxtreme.config import Config
    from mxtreme.scans import network_scan

    scan = network_scan.NetworkScanParams(
        recording_electrodes={params.well: electrodes},
        batch=params.batch,
        chip=params.chip,
        plate_date=params.plate_date,
        div=params.div,
        rec_length_sec=int(seconds),
        pad_to_max=False,
        description="baseline for associative region selection: candidate patches, no stimulation",
    )
    return str(network_scan.run_network_scan(scan, Config.from_toml(toml_path)).h5_path)


def electrodes_from(path: str, well: int = 0) -> list[int]:
    """The electrodes a previous run recorded, from whatever names them.

    Accepts a MaxLab ``.cfg``, a raw ``.raw.h5``, or an MXtreme-preprocessed ``.npz``. This is how
    a run with no activity scan gets a sensible routing: reuse the electrode set some earlier test
    or scan already used on this chip, rather than inventing one.

    :param path: The ``.cfg``, ``.h5`` or ``.npz``.
    :param well: Well number, for a multi-well ``.h5``.
    """
    text = str(path)
    if text.endswith(".cfg"):
        from mxtreme.scans import mx_config

        return sorted(int(e) for e in mx_config.read_config_elecs(text))
    if text.endswith(".npz"):
        return sorted({int(e) for e in np.load(text, allow_pickle=True)["channelmap"][:, 2]})

    import h5py

    with h5py.File(text, "r") as f:
        mapping = f[f"/recordings/rec0000/well{well:03d}/settings/mapping"][:]
    return sorted({int(e) for e in mapping["electrode"]})


def _lattice(used: set[int], budget: int) -> list[int]:
    """A spread of electrodes over the whole array, thinned until it fits the budget."""
    step = 6
    while True:
        lattice = [
            e
            for e in range(protocol.NUM_ELECTRODES)
            if (e % protocol.COLS) % step == 0 and (e // protocol.COLS) % step == 0 and e not in used
        ]
        if len(lattice) <= budget or step > 40:
            return lattice[: max(0, budget)]
        step += 1


def select(
    params: AssociativeParams,
    out_dir: str,
    *,
    activity_scan=None,
    scan_npz=None,
    scan_with=None,
    centers=None,
    electrodes=None,
    baseline=None,
    record_baseline=None,
    baseline_sec=BASELINE_SEC_MAX,
    candidates=4,
    separation=None,
    min_active=None,
    on_progress=print,
) -> tuple[AssociativeParams, str]:
    """Choose the regions and write ``<out_dir>/params_<chip>_well<w>.json``.

    :returns: The parameters with ``regions``, ``rec_electrodes`` and (if overridden)
        ``min_region_separation_um`` set, and the path they were written to.
    """
    from mxtreme.scans import region_selection

    os.makedirs(out_dir, exist_ok=True)
    radius = params.region_radius_um
    if separation is not None:
        params = replace(params, min_region_separation_um=float(separation))
    separation = params.min_region_separation_um
    seconds = min(baseline_sec, BASELINE_SEC_MAX)
    well = params.well

    # --- 1, 2: candidates ---
    well_data, ranked, scan_path = None, [], None
    if centers:
        centres = parse_centers(centers)
        if len(centres) < 3:
            raise ValueError("centers needs at least three")
    else:
        if scan_npz:
            well_data = region_selection.active_electrodes(region_selection.well_data_from_npz(scan_npz))
            scan_path = scan_npz
        else:
            scan_path = activity_scan or (_run_scan(params, scan_with) if scan_with else None)
            if scan_path is None:
                raise ValueError("give an activity scan, a scan npz, a store to scan into, or centers")
            well_data = _load_scan(scan_path, well)
        active = well_data["active_electrodes"]["electrode"].nunique()
        recorded = well_data["mapping"]["electrode"].nunique()
        on_progress(
            f"scan {os.path.basename(scan_path)}: {recorded} electrodes recorded, {active} active "
            f"(>= 0.1 Hz and 90th-percentile amplitude >= 20 uV)"
        )
        ranked = region_selection.rank_patches(
            well_data, n=candidates, min_separation_um=separation, radius_um=radius, min_active=min_active
        )
        centres = [r["center_um"] for r in ranked if r["status"] == "chosen"]
        rule = (
            f"at least {min_active} active"
            if min_active is not None
            else "at least a quarter of the recorded electrodes active, and at least five"
        )
        on_progress(
            f"patches of radius {radius:.0f} um, densest first, at least {separation:.0f} um apart, {rule}:"
        )
        shown = 0
        for r in ranked:
            if r["status"] == "chosen" or shown < 8:
                on_progress(
                    f"  ({r['center_um'][0]:5.0f}, {r['center_um'][1]:5.0f}) um  {r['count']:4d} active "
                    f"of {r['recorded']:4d} recorded  {r['status']}"
                )
                shown += 1
        if len(centres) < 3:
            png = os.path.join(out_dir, f"regions_{params.chip}_well{well}_rejected.png")
            _draw_rejection(well_data, ranked, radius, png, separation)
            raise ValueError(
                f"only {len(centres)} patch(es) qualify at {separation:.0f} um separation; see {png}. "
                f"Lower the separation or region_radius_um, or give centers."
            )
    on_progress("candidates (um): " + ", ".join(f"({x:.0f}, {y:.0f})" for x, y in centres))

    # Recording electrodes: the patches, then a spread over the rest.
    if well_data is not None:
        sets = region_selection.recording_electrodes(well_data, centres, radius)
        density = region_selection.patch_density(well_data, centres, radius)
    else:
        # No scan, so nothing says which electrodes are worth recording. Reuse the set some
        # earlier run on this chip used when one is named, and only fall back to a synthetic
        # lattice when nothing is.
        if electrodes:
            pool = electrodes_from(electrodes, well)
            on_progress(f"routing from {os.path.basename(str(electrodes))}: {len(pool)} electrodes")
        else:
            pool = list(range(protocol.NUM_ELECTRODES))
            on_progress(
                "no scan and no electrode set given: routing every electrode near each "
                "centre, then a synthetic lattice over the rest. Pass electrodes=<cfg|h5|npz> "
                "to reuse a real routing instead."
            )
        sets = {f"patch_{k}": protocol.within(pool, c, radius) for k, c in enumerate(centres)}
        used = {e for s in sets.values() for e in s}
        budget = 1020 - 32 - len(used)
        rest = [e for e in pool if e not in used]
        sets["global"] = rest[:budget] if electrodes else _lattice(used, budget)
        density = [len(sets[f"patch_{k}"]) for k in range(len(centres))]
    patches = {f"patch_{k}": sets[f"patch_{k}"] for k in range(len(centres))}
    rec_electrodes = sorted({e for s in sets.values() for e in s})
    on_progress(
        f"routing {len(rec_electrodes)} electrodes: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items())
    )

    # --- 3, 4: baseline, coupling, roles ---
    names = list(patches)
    matrix = np.full((len(names), len(names)), np.nan)
    corr_raw = lag_ms = lead = None
    bursts, used = [], 0
    baseline_path = baseline
    if record_baseline:
        baseline_path = _record_baseline(params, rec_electrodes, seconds, record_baseline)
    baseline_data = None
    if baseline_path:
        frames, elecs, fps = region_selection.baseline_spikes(baseline_path, well)
        baseline_data = (frames, elecs, fps)
        # A patch whose electrodes were not recorded in the baseline has no coupling to measure,
        # and everything downstream would quietly be nan. Say so instead.
        seen = {int(e) for e in np.unique(elecs)}
        empty = {name: len(set(map(int, elecs_)) & seen) for name, elecs_ in patches.items()}
        thin = {name: n for name, n in empty.items() if n < 3}
        if thin:
            raise ValueError(
                f"in {os.path.basename(baseline_path)}, patch(es) {thin} have fewer than three "
                "electrodes carrying spikes. The baseline must be recorded with the candidate "
                "patches routed; a recording of a different electrode set cannot judge them."
            )
        bursts = region_selection.network_bursts(frames, fps)
        _, corr_raw = region_selection.region_correlations(frames, elecs, patches, fps)
        names, outside = region_selection.region_correlations(frames, elecs, patches, fps, exclude=bursts)
        _, lag_ms, lead, used = region_selection.burst_onset_lags(frames, elecs, patches, fps, bursts)
        span = (frames.max() - frames.min()) / fps if len(frames) else 0.0
        on_progress(
            f"baseline {os.path.basename(baseline_path)}: {len(frames)} spikes at {fps:.0f} Hz, "
            f"{len(bursts)} network bursts in {span:.0f} s, {used} with two or more patches taking part"
        )

        def show(title, m, fmt="{:7.2f}"):
            on_progress(title)
            for name, row in zip(names, m):
                on_progress(f"  {name:<8}" + "".join(fmt.format(v) for v in row))

        show("correlation of patch spike counts, 10 ms bins, all of the recording:", corr_raw)
        show("the same with the network bursts left out:", outside)
        show(
            "burst order: fraction of shared bursts in which the row patch fired before the column patch:",
            lead,
        )
        show("median lead of row over column at burst onset, ms:", lag_ms, "{:7.1f}")
        matrix = region_selection.coupling(outside, lead)
        show("coupling = max(|correlation outside bursts|, |2 * order - 1|):", matrix)
        coupling_png = os.path.join(out_dir, f"coupling_{params.chip}_well{well}.png")
        _draw_coupling(
            baseline_data, patches, centres, bursts, corr_raw, outside, lead, lag_ms, matrix, coupling_png
        )
        on_progress(f"coupling diagnostics: {coupling_png}")
    else:
        on_progress("no baseline: roles from geometry and density alone")

    counterbalance = params.seed % 2
    roles = region_selection.assign_roles(centres, matrix, density, counterbalance=counterbalance)
    tri_ratio, sides = region_selection.triangle([roles[r] for r in ROLES])
    on_progress(
        "roles: among triples with shortest/longest side >= 0.6, the least coupled, then the one whose "
        "two conditioned sites are best matched in coupling to US; US at the vertex equidistant from "
        f"the other two; which of those is CS is counterbalanced by seed (seed {params.seed} -> "
        f"{counterbalance}):"
    )
    for role, centre in roles.items():
        k = centres.index(centre)
        on_progress(
            f"  {role:<4} patch_{k} at ({centre[0]:.0f}, {centre[1]:.0f}) um, {density[k]} electrodes"
        )
    on_progress(
        f"  triangle US-CS {sides[0]:.0f} um, CS-NS {sides[1]:.0f} um, US-NS {sides[2]:.0f} um "
        f"(shortest/longest {tri_ratio:.2f})"
    )

    # Shared bursts are the one connectivity signal available before any stimulation: two patches
    # that never take part in the same burst may simply not be connected, and decorrelation alone
    # cannot tell that apart from independence. The stimulation check (mode "connectivity") is what
    # settles it, so this is a pointer rather than a refusal.
    if baseline_data is not None and bursts:
        _, onsets = region_selection.burst_onsets(
            baseline_data[0], baseline_data[1], patches, baseline_data[2], bursts
        )
        index = {c: k for k, c in enumerate(centres)}
        for a, b in (("US", "CS"), ("US", "NS"), ("CS", "NS")):
            i, j = index[roles[a]], index[roles[b]]
            shared = int(np.sum(~np.isnan(onsets[:, i]) & ~np.isnan(onsets[:, j])))
            if shared < max(5, 0.2 * len(bursts)):
                on_progress(
                    f"  note: {a} and {b} took part in the same network burst only {shared} of "
                    f"{len(bursts)} times. They may not be well connected; run the connectivity "
                    f"check (mode 'connectivity') before conditioning."
                )

    # --- 5: the parameter file, the record, the picture ---
    chosen = replace(
        params,
        regions={r: [float(c[0]), float(c[1])] for r, c in roles.items()},
        rec_electrodes=rec_electrodes,
    )
    params_path = os.path.join(out_dir, f"params_{params.chip}_well{well}.json")
    chosen.to_json(params_path)
    record = {
        "well": well,
        "scan": scan_path,
        "baseline": baseline_path,
        "candidates_um": centres,
        "density": density,
        "patch_electrodes": patches,
        "coupling": {
            "names": names,
            "matrix": np.nan_to_num(matrix, nan=-9).tolist(),
            "correlation_all": None if corr_raw is None else np.nan_to_num(corr_raw, nan=-9).tolist(),
            "burst_order": None if lead is None else np.nan_to_num(lead, nan=-9).tolist(),
            "burst_lag_ms": None if lag_ms is None else np.nan_to_num(lag_ms, nan=-9).tolist(),
            "bursts": len(bursts),
            "bursts_used": used,
        },
        "roles": {r: list(c) for r, c in roles.items()},
        "triangle_um": sides,
        "triangle_ratio": tri_ratio,
        "params": params_path,
    }
    with open(os.path.join(out_dir, f"regions_{params.chip}_well{well}.json"), "w") as f:
        json.dump(record, f, indent=2)
    png = os.path.join(out_dir, f"regions_{params.chip}_well{well}.png")
    _draw_decision(chosen, well_data, ranked, centres, names, matrix, lead, baseline_path, radius, png)
    on_progress(f"wrote {params_path}\n      {png}")
    return chosen, params_path


# ---------------------------------------------------------------
#                     THE COUPLING DIAGNOSTICS
#
# What the numbers behind the role assignment actually look like. Written
# whenever there is a baseline, because "least coupled" is a judgement worth
# seeing rather than taking on trust.
# ---------------------------------------------------------------


def _draw_coupling(baseline_data, patches, centres, bursts, corr_raw, outside, lead, lag_ms, coupling, png):
    """Five views of the baseline: the regions' rates with the bursts marked, the correlation
    before and after masking those bursts, the cross-correlograms, who enters each burst first,
    and whether coupling is just distance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mxtreme.scans import region_selection

    frames, elecs, fps = baseline_data
    names = list(patches)
    n = len(names)
    colours = plt.cm.tab10(np.linspace(0, 1, 10))[:n]

    fig = plt.figure(figsize=(17, 4.2 * n + 7))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.0, 1.6 * n / 3, 1.0], hspace=0.42, wspace=0.28)

    # 1. Rates over time, with the detected bursts shaded.
    ax = fig.add_subplot(grid[0, :])
    _, t, rates = region_selection.patch_rates(frames, elecs, patches, fps, bin_ms=100.0)
    first = int(frames.min())
    for a, b in bursts:
        ax.axvspan((a - first) / fps, (b - first) / fps, color="#fde68a", alpha=0.55, lw=0)
    for i, name in enumerate(names):
        ax.plot(t, rates[i], lw=0.7, color=colours[i], label=name)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_xlabel("seconds")
    ax.set_ylabel("spikes/s in the patch")
    ax.legend(fontsize=7, ncol=n, loc="upper right")
    ax.set_title(
        f"baseline: each candidate patch's rate, {len(bursts)} network bursts shaded. "
        "Everything rises together in a burst, which is why the correlation is computed with them left out.",
        fontsize=9,
        loc="left",
    )

    # 2. The two correlation matrices and the burst order, side by side.
    for k, (title, m, vmin, vmax, cmap, fmt) in enumerate(
        (
            (
                "correlation, all bins\n(what a bursting culture always shows)",
                corr_raw,
                -1,
                1,
                "RdBu_r",
                "{:.2f}",
            ),
            ("correlation, network bursts left out\n(what counts)", outside, -1, 1, "RdBu_r", "{:.2f}"),
            ("burst order: row fired first in\nthis fraction of shared bursts", lead, 0, 1, "PuOr", "{:.0%}"),
        )
    ):
        ax = fig.add_subplot(grid[1, k])
        if m is None:
            ax.axis("off")
            continue
        im = ax.imshow(np.nan_to_num(m, nan=0.0), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(n), names, rotation=45, fontsize=7)
        ax.set_yticks(range(n), names, fontsize=7)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                text = "" if np.isnan(m[i, j]) else fmt.format(m[i, j])
                if k == 2 and lag_ms is not None and not np.isnan(lag_ms[i, j]):
                    text += f"\n{lag_ms[i, j]:+.0f} ms"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.75)

    # 3. Cross-correlograms, one panel per pair, before and after masking.
    _, lags, corr_all = region_selection.cross_correlograms(
        frames, elecs, patches, fps, bin_ms=5.0, max_lag_ms=250.0
    )
    _, _, corr_out = region_selection.cross_correlograms(
        frames, elecs, patches, fps, bin_ms=5.0, max_lag_ms=250.0, exclude=bursts
    )
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cols = min(3, max(1, len(pairs)))
    rows = int(np.ceil(len(pairs) / cols))
    sub = grid[2, :].subgridspec(rows, cols, hspace=0.55, wspace=0.22)
    finite = [np.nanmax(np.abs(corr_all[i, j])) for i, j in pairs if np.any(np.isfinite(corr_all[i, j]))]
    peak = max(finite) if finite else 1.0
    for k, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(sub[k // cols, k % cols])
        ax.axhline(0, color="#dddddd", lw=0.6)
        ax.axvline(0, color="#dddddd", lw=0.6)
        ax.plot(lags, corr_all[i, j], lw=1.0, color="#9ca3af", label="all bins")
        ax.plot(lags, corr_out[i, j], lw=1.2, color="#111827", label="bursts left out")
        best, height, stands_out = region_selection.correlogram_peak(lags, corr_all[i, j])
        known = np.any(np.isfinite(corr_all[i, j]))
        if stands_out:
            ax.axvline(best, color="#ef4444", lw=0.8, ls="--")
        ax.set_ylim(-0.15 * peak, 1.1 * peak)
        title = f"{names[i]} -> {names[j]}: "
        if not known:
            title += "not measurable"
        elif not stands_out:
            title += "flat, no consistent lead"
        else:
            title += f"peak {height:.2f} at {best:+.0f} ms" + (
                f", {names[i]} leads" if best > 0 else (f", {names[j]} leads" if best < 0 else "")
            )
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("lag (ms)", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("correlation", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")

    # 4. Who joins each burst first.
    ax = fig.add_subplot(grid[3, :2])
    _, onsets = region_selection.burst_onsets(frames, elecs, patches, fps, bursts)
    if len(onsets) and np.any(np.isfinite(onsets)):
        with np.errstate(invalid="ignore"):
            earliest = np.nanmin(onsets, axis=1)
        for i, name in enumerate(names):
            delay = (onsets[:, i] - earliest) / fps * 1000.0
            ax.plot(np.arange(len(delay)), delay, "o", ms=4, color=colours[i], label=name)
        ax.set_xlabel("network burst")
        ax.set_ylabel("ms after the first patch to join")
        ax.legend(fontsize=7, ncol=n)
        firsts = [names[int(i)] for i in np.nanargmin(np.where(np.isnan(onsets), np.inf, onsets), axis=1)]
        tally = ", ".join(f"{name} {firsts.count(name)}" for name in names if firsts.count(name))
        ax.set_title(
            f"burst initiation: first to join is {tally} (of {len(onsets)} bursts)", fontsize=9, loc="left"
        )
        ax.text(
            0.0,
            -0.28,
            "A patch that leads most bursts is driving the others rather than being independent of them.",
            transform=ax.transAxes,
            fontsize=8,
            color="#6b7280",
        )

    # 5. Is coupling just distance?
    ax = fig.add_subplot(grid[3, 2])
    xs, ys, labels = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            xs.append(protocol.separation_um(centres[i], centres[j]))
            ys.append(coupling[i, j])
            labels.append(f"{i}-{j}")
    ax.scatter(xs, ys, c="#111827", s=22)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, color="#6b7280")
    ax.set_xlabel("centre-to-centre distance (um)")
    ax.set_ylabel("coupling")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("coupling against distance", fontsize=9, loc="left")
    ax.text(
        0.0,
        -0.28,
        "A rising trend would mean the measure is only reading distance.",
        transform=ax.transAxes,
        fontsize=8,
        color="#6b7280",
    )

    fig.savefig(png, dpi=105, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
#                            THE PICTURE
# ---------------------------------------------------------------


def _draw_scan(ax, well_data, ranked, radius, fig=None):
    """The scan's active electrodes by firing rate, with every considered patch: green chosen,
    orange denser-but-too-close, grey at the edge."""
    from matplotlib.patches import Circle

    spikes = well_data["spike_data"]
    rate = spikes.groupby("electrode").size() / well_data["rec_length_sec"]
    mapping = well_data["mapping"].drop_duplicates("electrode").set_index("electrode")
    active = set(well_data["active_electrodes"]["electrode"])
    xs, ys = mapping.loc[rate.index, "x"], mapping.loc[rate.index, "y"]
    is_active = np.array([e in active for e in rate.index])
    ax.set_facecolor(display.ARRAY_BG)
    ax.scatter(xs[~is_active], ys[~is_active], s=5, c="#444455", marker="s")
    sc = ax.scatter(
        xs[is_active],
        ys[is_active],
        s=9,
        c=np.clip(rate[is_active], 0, 5),
        cmap="magma",
        vmin=0,
        vmax=5,
        marker="s",
    )
    if fig is not None:
        fig.colorbar(sc, ax=ax, shrink=0.7, label="firing rate (Hz, capped at 5)")
    for r in ranked:
        colour = {"chosen": "#22c55e", "edge": "#888888"}.get(
            r["status"], "#f97316" if r["status"].startswith("too close") else None
        )
        if colour is None:
            continue
        ax.add_patch(
            Circle(
                r["center_um"],
                radius,
                fill=False,
                edgecolor=colour,
                lw=1.6 if r["status"] == "chosen" else 0.5,
                alpha=0.9,
            )
        )
    ax.set_title(
        f"scan: {len(rate)} electrodes with spikes, {len(active)} active; green = chosen, "
        f"orange = qualifies but too close to a chosen patch, grey = edge",
        fontsize=9,
        loc="left",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-30, protocol.COLS * protocol.PITCH_UM + 30)
    ax.set_ylim(protocol.ROWS * protocol.PITCH_UM + 30, -30)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def _draw_ranking(ax, ranked, radius):
    considered = [r for r in ranked if r["status"] != "not needed"][:25]
    colours = [
        "#22c55e"
        if r["status"] == "chosen"
        else "#f97316"
        if r["status"].startswith("too close")
        else "#888888"
        for r in considered
    ]
    ax.barh(range(len(considered)), [r["count"] for r in considered], color=colours)
    for i, r in enumerate(considered):
        ax.plot([r["recorded"]], [i], "|", color="#555555", ms=8)
    ax.set_yticks(
        range(len(considered)),
        [f"({r['center_um'][0]:.0f},{r['center_um'][1]:.0f}) {r['status']}" for r in considered],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel(f"active electrodes within {radius:.0f} um (tick = recorded)")
    ax.set_title("candidate patches, densest first", fontsize=9, loc="left")


def _draw_rejection(well_data, ranked, radius, png, separation):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_scan, ax_rank) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [3, 1]})
    _draw_scan(ax_scan, well_data, ranked, radius, fig)
    _draw_ranking(ax_rank, ranked, radius)
    fig.suptitle(f"no three patches {separation:.0f} um apart", fontsize=10)
    fig.savefig(png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _draw_decision(params, well_data, ranked, centres, names, matrix, lead, baseline_path, radius, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = protocol.regions_for(params)
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.3, wspace=0.15)
    ax_scan, ax_corr = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_map, ax_rank = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    if well_data is not None:
        _draw_scan(ax_scan, well_data, ranked, radius, fig)
        _draw_ranking(ax_rank, ranked, radius)
    else:
        ax_scan.text(0.5, 0.5, "no scan: centres given by hand", ha="center", transform=ax_scan.transAxes)
        ax_rank.axis("off")
    routed = np.array([protocol.electrode_xy(e) for e in params.rec_electrodes])
    specs = {r: s.as_dict() for r, s in regions.items()}
    display.draw_map(
        ax_map,
        specs,
        routed,
        radius,
        title=f"{params.chip} well {params.well}: {len(centres)} candidates, roles assigned; {len(params.rec_electrodes)} electrodes to route",
    )
    display.draw_triangle(ax_map, specs)
    for k, (cx, cy) in enumerate(centres):
        ax_map.text(cx, cy + radius + 60, f"patch_{k}", color="#aaaaaa", ha="center", fontsize=8)
    im = ax_corr.imshow(np.nan_to_num(matrix, nan=1.0), vmin=0, vmax=1, cmap="Reds")
    ax_corr.set_xticks(range(len(names)), names, rotation=45)
    ax_corr.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            cell = "" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"
            if lead is not None and not np.isnan(lead[i, j]):
                cell += f"\n{lead[i, j]:.0%} first"
            ax_corr.text(j, i, cell, ha="center", va="center", fontsize=7)
    ax_corr.set_title(
        "baseline coupling: |correlation outside bursts| or burst-order consistency,\nwhichever is larger; "
        "'x% first' = row fired before column in x% of shared bursts"
        if baseline_path
        else "no baseline",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax_corr, shrink=0.7)
    fig.savefig(png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
