"""Choosing where the three regions go, offline, from a baseline recording.

``select`` touches no hardware and writes nothing into the managed store. It reads recordings the
scans already made and writes a parameter file, and a picture of every decision, into a directory
you name. The parameter file is what every later run points at.

The input is the culture's **baseline**: one continuous recording of its active set, which is
exactly what a network scan is (``braintrix-cli``'s *Network scan*, or
:func:`mxtreme.scans.network_scan.run_network_scan`). From it:

1. candidate patches: the densest ``candidates`` patches at least ``min_region_separation_um``
   apart (:func:`mxtreme.scans.region_selection.rank_patches`), counting the baseline's active
   electrodes;
2. coupling between the candidates: correlation with network bursts left out, and how
   consistently one enters the bursts before another;
3. roles: the least-coupled, near-equilateral triple whose two conditioned sites are matched in
   coupling to the readout, US at the vertex equidistant from the other two, and CS versus NS
   decided by the seed's parity so it alternates across cultures
   (:func:`mxtreme.scans.region_selection.assign_roles`);
4. the stimulation sites: each driven block is moved by up to 70 um so that its electrodes sit on
   electrodes the activity scan (or, failing that, the baseline) recorded spikes from
   (:func:`place_site`). A region is dense, but the seven driven electrodes are particular
   electrodes, and one over glass stimulates nothing;
5. the experiment's routing: every electrode the baseline recorded, with the three chosen regions
   first. Standard electrode selection keeps electrodes at least 100 um apart, which leaves only a
   handful inside a region; given the **activity scan** as well, every electrode it found active
   inside a chosen region is added, so the readout is counted on as many electrodes as the culture
   offers there.

``centers`` places the three regions by hand instead, which is what a saline dry run does: a
silent baseline has nothing to rank or correlate, so it only supplies the routing.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np

from mxtreme.experiments.associative import display, protocol
from mxtreme.experiments.associative.params import ROLES, AssociativeParams


def _stem(params: AssociativeParams) -> str:
    """The file-name stem select writes under: the store's own
    ``plating_<date>_<batch>_chip_<chip>_well_<w>_DIV_<d>`` plus the experiment id, so the
    parameter file and figures sort and read alongside the recordings they belong to. Falls back
    to a short name when there is no batch to build the stem from."""
    if not params.batch:
        return f"{params.exp_id}_{params.chip or 'chip'}_well{params.well}"
    from mxtreme import store

    return (
        store.recording_stem(params.batch, params.plate_date, params.chip, params.well, params.div)
        + f"_{params.exp_id}"
    )


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


def electrodes_from(path: str, well: int = 0) -> list[int]:
    """The electrodes a previous run recorded, from whatever names them.

    Accepts a MaxLab ``.cfg``, a raw ``.raw.h5``, or an MXtreme-preprocessed ``.npz``. This is how
    a run with no activity scan gets a sensible routing: reuse an electrode set some earlier test
    or scan already used, rather than inventing one.

    Only the electrode *numbers* are read. They are positions on the 220 x 120 array, laid out
    the same on every MaxOne and MaxTwo, so a set taken from another chip names the same places on
    this one; the chip the file came from is never consulted. What can differ between chips is
    which electrodes route, and routing is solved afresh on the real chip when a run starts.

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


def electrode_activity(well_data) -> dict[int, tuple[float, float]]:
    """``{electrode: (rate_hz, amplitude_uv)}`` for every active electrode of a scan or baseline,
    each rate over the electrode's own recorded time (see
    :func:`mxtreme.scans.region_selection.active_electrodes`)."""
    from mxtreme.scans import region_selection

    active = well_data.get("active_electrodes")
    if active is None or not len(active):
        return {}
    visits = region_selection.scan_visits(well_data)
    seconds = float(well_data["rec_length_sec"])
    lsb = float(well_data["lsb"])
    out = {}
    for electrode, group in active.groupby("electrode"):
        rate = len(group) / (max(1, int(visits.get(electrode, 1))) * seconds)
        amplitude = abs(float(np.percentile(group["amplitude"], 90)) * lsb * 1e6)
        out[int(electrode)] = (float(rate), amplitude)
    return out


def place_site(centre_um, site: dict, activity: dict, max_shift_um: float = 70.0) -> dict:
    """Move a stimulation site, by at most ``max_shift_um``, so its driven electrodes sit on
    electrodes that have recorded spikes.

    A region is chosen for how many active electrodes it holds, but the driven block is a few
    electrodes at fixed spacing around the region's centre, and nothing so far asked whether those
    particular electrodes have a neuron under them. A driven electrode with no recorded spikes may
    be over glass, and a site of four such electrodes can deliver every pulse of the day to nothing.
    Calibration would find that out, at the cost of a reselection; this asks first.

    Every shift of up to ``max_shift_um`` along each axis is tried, in whole electrodes. The score
    is how many driven electrodes are active, then the smallest shift, then the summed firing
    rate: a site that already sits on activity does not move, and one that has to move goes no
    further than it must, since the region it sits in was chosen where it is for a reason. The
    recording region moves with the site, which is what keeps the ring centred on what is measured.

    :param activity: From :func:`electrode_activity`; empty means nothing is known and the site
        stays where it is.
    :returns: ``{"center_um", "shift_um", "driven": {electrode: (rate, amplitude) or None},
        "active", "of"}``.
    """
    steps = int(max_shift_um // protocol.PITCH_UM)
    best = None
    for dy in range(-steps, steps + 1):
        for dx in range(-steps, steps + 1):
            candidate = (centre_um[0] + dx * protocol.PITCH_UM, centre_um[1] + dy * protocol.PITCH_UM)
            try:
                drive, _ring = protocol.stim_site(candidate, site)
            except ValueError:
                continue
            hits = {e: activity.get(e) for e in drive}
            score = (
                sum(1 for v in hits.values() if v),
                -(dx * dx + dy * dy),
                sum(v[0] for v in hits.values() if v),
            )
            if best is None or score > best[0]:
                best = (score, candidate, hits, (dx, dy))
    _score, centre, hits, (dx, dy) = best
    return {
        "center_um": (float(centre[0]), float(centre[1])),
        "shift_um": float(((dx * protocol.PITCH_UM) ** 2 + (dy * protocol.PITCH_UM) ** 2) ** 0.5),
        "driven": hits,
        "active": sum(1 for v in hits.values() if v),
        "of": len(hits),
    }


#: Electrodes a run asks the router for, recording and stimulation together. MaxLab's own limit is
#: 1020, and it says routing "will not converge well" near it; 1000 leaves it room.
ROUTING_BUDGET = 1000


def _experiment_routing(pool, roles, radius, scan_data, budget, stim_electrodes=()):
    """Recording electrodes for the experiment: the chosen regions' electrodes first -- the
    baseline's, plus the activity scan's active ones when given -- then the rest of the baseline's
    set, cut to the routing budget. The stimulation electrodes are routed on their own account
    and left out of the recording set, so they never count twice.

    :returns: ``(electrodes, {role: electrodes inside it})``.
    """
    stim = set(stim_electrodes)
    region_sets = {}
    for role, centre in roles.items():
        inside = set(protocol.within(pool, centre, radius))
        if scan_data is not None:
            active = scan_data["active_electrodes"]["electrode"].unique().tolist()
            inside |= set(protocol.within(active, centre, radius))
        region_sets[role] = sorted(inside - stim)
    first = list(dict.fromkeys(e for role in ROLES for e in region_sets[role]))
    rest = [e for e in pool if e not in set(first) and e not in stim]
    excess = len(first) + len(rest) - budget
    if excess > 0:
        # Thin the rest evenly along the array rather than cutting its tail: electrode numbers run
        # row by row, so a tail cut would leave the bottom rows of the chip unrecorded.
        drop = set(np.linspace(0, len(rest) - 1, num=min(excess, len(rest)), dtype=int).tolist())
        rest = [e for k, e in enumerate(rest) if k not in drop]
    return sorted(first + rest), region_sets


def select(
    params: AssociativeParams,
    out_dir: str,
    *,
    baseline=None,
    activity_scan=None,
    centers=None,
    electrodes=None,
    candidates=4,
    separation=None,
    min_active=None,
    on_progress=print,
) -> tuple[AssociativeParams, str]:
    """Choose the regions and write ``<out_dir>/<stem>_params.json``, where the stem is the store's
    ``plating_..._DIV_<d>`` name plus ``exp_id``.

    :param out_dir: Where to write. Required, and meant to be outside the managed store: nothing
        written here is experimental data, and all of it can be regenerated.
    :param baseline: The culture's baseline recording (``.raw.h5`` or ``.npz``): candidates,
        coupling and the routing come from it.
    :param activity_scan: Optional ``.raw.h5`` activity scan: its active electrodes inside the
        chosen regions are added to the routing.
    :param centers: ``'x,y;x,y;...'`` in um: place the regions by hand instead of from the baseline.
    :param electrodes: A ``.cfg``/``.h5``/``.npz`` to take the routing from when there is no
        baseline.
    :returns: The parameters with ``regions`` and ``rec_electrodes`` set, and the file path.
    """
    from mxtreme.scans import region_selection

    if not out_dir:
        raise ValueError("select needs an output directory, outside the managed store")
    os.makedirs(out_dir, exist_ok=True)
    radius = params.region_radius_um
    if separation is not None:
        params = replace(params, min_region_separation_um=float(separation))
    separation = params.min_region_separation_um
    well = params.well

    # --- the baseline ---
    base, silent = None, True
    if baseline:
        base = region_selection.load_recording(baseline, well)
        spikes = base["spike_data"]
        rate = len(spikes) / max(base["rec_length_sec"], 1e-9)
        silent = rate < 1.0
        on_progress(
            f"baseline {os.path.basename(str(baseline))}: {base['mapping']['electrode'].nunique()} electrodes "
            f"recorded, {base['active_electrodes']['electrode'].nunique()} active (>= 0.1 Hz, 90th-percentile "
            f"amplitude >= 20 uV), {len(spikes)} spikes in {base['rec_length_sec']:.0f} s"
            + ("; silent, so it supplies the routing only" if silent else "")
        )

    # --- 1: candidates ---
    ranked = []
    if centers:
        centres = parse_centers(centers)
        if len(centres) < 3:
            raise ValueError("centers needs at least three")
    else:
        if base is None:
            raise ValueError(
                "give a baseline recording to choose from, or centers to place the regions by hand"
            )
        if silent:
            raise ValueError(
                "the baseline is silent, so there is nothing to choose patches from; give centers"
            )
        ranked = region_selection.rank_patches(
            base, n=candidates, min_separation_um=separation, radius_um=radius, min_active=min_active
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
            png = os.path.join(out_dir, f"{_stem(params)}_regions_rejected.png")
            _draw_rejection(base, ranked, radius, png, separation)
            raise ValueError(
                f"only {len(centres)} patch(es) qualify at {separation:.0f} um separation; see {png}. "
                f"Lower the separation or region_radius_um, or give centers."
            )
    on_progress("candidates (um): " + ", ".join(f"({x:.0f}, {y:.0f})" for x, y in centres))

    # --- the pool the routing is drawn from ---
    if base is not None:
        pool = sorted(base["mapping"]["electrode"].astype(int).unique().tolist())
    elif electrodes:
        pool = electrodes_from(electrodes, well)
        on_progress(f"routing from {os.path.basename(str(electrodes))}: {len(pool)} electrodes")
    else:
        pool = None
        on_progress(
            "no baseline and no electrode set: routing every electrode near each centre, then a "
            "synthetic lattice. Pass a baseline, or electrodes=<cfg|h5|npz>, to use a real routing."
        )
    patches = {
        f"patch_{k}": protocol.within(pool if pool is not None else range(protocol.NUM_ELECTRODES), c, radius)
        for k, c in enumerate(centres)
    }
    density = (
        region_selection.patch_density(base, centres, radius)
        if base is not None and not silent
        else [len(v) for v in patches.values()]
    )

    # --- 2: coupling ---
    names = list(patches)
    matrix = np.full((len(names), len(names)), np.nan)
    corr_raw = lag_ms = lead = None
    bursts, used, baseline_data = [], 0, None
    if base is not None and not silent:
        spikes = base["spike_data"]
        frames = spikes["frameno"].to_numpy().astype(np.int64)
        elecs = spikes["electrode"].to_numpy().astype(int)
        fps = float(base["samp_rate"])
        baseline_data = (frames, elecs, fps)
        seen = {int(e) for e in np.unique(elecs)}
        thin = {name: len(set(map(int, v)) & seen) for name, v in patches.items()}
        thin = {name: n for name, n in thin.items() if n < 3}
        if thin:
            raise ValueError(
                f"in {os.path.basename(str(baseline))}, patch(es) {thin} have fewer than three electrodes "
                "carrying spikes, so their coupling cannot be measured. Move them, or widen region_radius_um."
            )
        bursts = region_selection.network_bursts(frames, fps)
        _, corr_raw = region_selection.region_correlations(frames, elecs, patches, fps)
        names, outside = region_selection.region_correlations(frames, elecs, patches, fps, exclude=bursts)
        _, lag_ms, lead, used = region_selection.burst_onset_lags(frames, elecs, patches, fps, bursts)
        on_progress(f"{len(bursts)} network bursts, {used} with two or more patches taking part")

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
        coupling_png = os.path.join(out_dir, f"{_stem(params)}_coupling.png")
        _draw_coupling(
            baseline_data, patches, centres, bursts, corr_raw, outside, lead, lag_ms, matrix, coupling_png
        )
        on_progress(f"coupling diagnostics: {coupling_png}")
    else:
        on_progress("no coupling to measure: roles from geometry alone")

    # --- 3: roles ---
    counterbalance = params.seed % 2
    roles = region_selection.assign_roles(centres, matrix, density, counterbalance=counterbalance)
    tri_ratio, sides = region_selection.triangle([roles[r] for r in ROLES])
    on_progress(
        "roles: among triples with shortest/longest side >= 0.6, the least coupled, then the one whose "
        "two conditioned sites are best matched in coupling to US; US at the vertex equidistant from "
        f"the other two; which of those is CS is counterbalanced by seed (seed {params.seed} -> {counterbalance}):"
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
    if baseline_data is not None and bursts:
        _, onsets = region_selection.burst_onsets(*baseline_data[:2], patches, baseline_data[2], bursts)
        index = {c: k for k, c in enumerate(centres)}
        for a, b in (("US", "CS"), ("US", "NS"), ("CS", "NS")):
            i, j = index[roles[a]], index[roles[b]]
            shared = int(np.sum(~np.isnan(onsets[:, i]) & ~np.isnan(onsets[:, j])))
            if shared < max(5, 0.2 * len(bursts)):
                on_progress(
                    f"  note: {a} and {b} took part in the same network burst only {shared} of {len(bursts)} "
                    f"times. They may not be well connected; the baseline gate will say."
                )

    # --- 4: the stimulation sites, on electrodes that have shown spikes ---
    scan_data = _load_scan(activity_scan, well) if activity_scan else None
    activity = electrode_activity(scan_data) if scan_data is not None else {}
    if not activity and base is not None and not silent:
        activity = electrode_activity(base)
    placements = {}
    if activity:
        on_progress(
            f"stimulation sites: each driven block moved by up to 70 um so its electrodes sit on ones "
            f"the {'activity scan' if scan_data is not None else 'baseline'} recorded spikes from:"
        )
        for role in ROLES:
            placed = place_site(roles[role], params.stim_site, activity)
            placements[role] = placed
            roles[role] = placed["center_um"]
            cells = ", ".join(
                f"e{e} {v[0]:.1f} Hz {v[1]:.0f} uV" if v else f"e{e} silent"
                for e, v in placed["driven"].items()
            )
            on_progress(
                f"  {role:<4} {placed['active']} of {placed['of']} driven electrodes active"
                f" (moved {placed['shift_um']:.0f} um): {cells}"
            )
            if placed["active"] == 0:
                on_progress(
                    f"  WARNING: no driven electrode of {role} sits on an electrode with recorded spikes. "
                    f"Its pulses may reach nothing; expect calibration to say so. Consider --centers."
                )
        # Moving the sites can only shorten the sides; the separation rule still has to hold.
        protocol.regions_for(replace(params, regions={r: list(c) for r, c in roles.items()}), [])
    else:
        on_progress(
            "stimulation sites: no per-electrode activity to place the driven blocks on (silent baseline, "
            "no activity scan), so they sit at the region centres as given."
        )

    # --- 5: the experiment's routing ---
    sites = protocol.regions_for(replace(params, regions={r: list(c) for r, c in roles.items()}), [])
    stim_electrodes = [e for spec in sites.values() for e in spec.stim_electrodes]
    stim_units = len(stim_electrodes)
    budget = ROUTING_BUDGET - stim_units
    if pool is None:
        every = list(range(protocol.NUM_ELECTRODES))
        near = sorted({e for c in roles.values() for e in protocol.within(every, c, radius)})
        pool = near + _lattice(set(near), budget - len(near))
    rec_electrodes, region_sets = _experiment_routing(pool, roles, radius, scan_data, budget, stim_electrodes)
    if len(pool) > budget:
        on_progress(
            f"experiment routing: the baseline's {len(pool)} electrodes exceed the budget of {budget} "
            f"({ROUTING_BUDGET} less {stim_units} stimulation), so {len(pool) - budget} outside the regions "
            "are dropped, spread evenly over the array"
        )
    on_progress(
        f"experiment routing: {len(rec_electrodes)} electrodes plus {stim_units} stimulation; in the regions "
        + ", ".join(f"{r} {len(v)}" for r, v in region_sets.items())
        + (
            f" (the activity scan added {sum(len(v) for v in region_sets.values()) - sum(len(protocol.within(pool, roles[r], radius)) for r in ROLES)})"
            if scan_data is not None
            else ""
        )
    )

    # --- the parameter file, the record, the picture ---
    chosen = replace(
        params,
        regions={r: [float(c[0]), float(c[1])] for r, c in roles.items()},
        stim_electrodes={r: list(spec.drive_electrodes) for r, spec in sites.items()},
        stim_electrodes_site=dict(params.stim_site),
        rec_electrodes=rec_electrodes,
    )
    params_path = os.path.join(out_dir, f"{_stem(params)}_params.json")
    chosen.to_json(params_path)
    record = {
        "well": well,
        "baseline": str(baseline) if baseline else None,
        "activity_scan": str(activity_scan) if activity_scan else None,
        "candidates_um": centres,
        "density": density,
        "patch_electrodes": patches,
        "region_electrodes": region_sets,
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
        "sites": {
            r: {
                "shift_um": p["shift_um"],
                "active_driven": p["active"],
                "driven": {str(e): (list(v) if v else None) for e, v in p["driven"].items()},
            }
            for r, p in placements.items()
        },
        "triangle_um": sides,
        "triangle_ratio": tri_ratio,
        "params": params_path,
    }
    with open(os.path.join(out_dir, f"{_stem(params)}_regions.json"), "w") as f:
        json.dump(record, f, indent=2)
    png = os.path.join(out_dir, f"{_stem(params)}_regions.png")
    _draw_decision(
        chosen,
        base if not silent else None,
        ranked,
        centres,
        names,
        matrix,
        lead,
        baseline if not silent else None,
        radius,
        png,
    )
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

    fig = plt.figure(figsize=(17, 4.2 * n + 7.5))
    grid = fig.add_gridspec(
        4, 3, height_ratios=[1.1, 1.0, 1.6 * n / 3, 1.0], hspace=0.42, wspace=0.28, top=0.93
    )
    fig.text(
        0.01,
        0.985,
        "How to read this figure. The question is which three candidate patches are the most independent of "
        "each other at baseline, so that a CS-to-US response that grows later is new. Row 1: the candidates' "
        "rates over time; network bursts (shaded) sweep every patch at once and would make any pair look "
        "coupled, so they are masked. Row 2: the coupling numbers; the middle matrix is the one that "
        "counts, and the right one says whether one patch consistently enters bursts ahead of another "
        "(near 50% = no consistent order). Row 3: the same pairs as time-lagged correlograms; a flat line is "
        "independence, a peak off zero is one patch leading the other by that many ms. Row 4, left: which "
        "patch enters each burst first; right: whether the coupling numbers just track how far apart the "
        "patches are.",
        fontsize=8.5,
        color="#374151",
        wrap=True,
        va="top",
    )

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
            (
                "correlation, network bursts left out\n(what the selection uses; want near 0)",
                outside,
                -1,
                1,
                "RdBu_r",
                "{:.2f}",
            ),
            (
                "burst order: row entered the burst before column\nin this fraction of shared bursts (want near 50%)",
                lead,
                0,
                1,
                "PuOr",
                "{:.0%}",
            ),
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
            f"which patch enters each network burst first: {tally} (of {len(onsets)} bursts)",
            fontsize=9,
            loc="left",
        )
        ax.text(
            0.0,
            -0.2,
            "Each dot is one burst: how many ms after the first patch that patch joined in.\nA patch sitting at "
            "0 for most bursts is where the bursts start, so it drives the others rather than being independent "
            "of them.",
            transform=ax.transAxes,
            fontsize=8,
            color="#6b7280",
            va="top",
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
    ax.set_title("coupling against distance, every candidate pair", fontsize=9, loc="left")
    ax.text(
        0.0,
        -0.2,
        "Coupling normally falls with distance. If it does here, the least-coupled\ntriple is mostly the "
        "widest one and separation is doing the work; if not,\ncoupling is telling you something separation "
        "cannot.",
        transform=ax.transAxes,
        fontsize=8,
        color="#6b7280",
        va="top",
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
    # A few electrodes fire at tens of Hz; a linear scale up to the maximum would leave every
    # ordinary electrode black, so the colour saturates at the 95th percentile of the active ones.
    cap = max(1.0, float(np.percentile(rate[is_active], 95))) if is_active.any() else 1.0
    sc = ax.scatter(
        xs[is_active],
        ys[is_active],
        s=9,
        c=np.clip(rate[is_active], 0, cap),
        cmap="magma",
        vmin=0,
        vmax=cap,
        marker="s",
    )
    if fig is not None:
        fig.colorbar(sc, ax=ax, shrink=0.7, label=f"firing rate (Hz), saturating at {cap:.1f} (95th pct)")
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


def _draw_ranking(ax, ranked, radius, centres=(), roles=None):
    """Every candidate, densest first: the chosen ones (with the role each got) always shown,
    then the densest of the rest up to 25 bars. Labelled by patch number and centre in um, the
    same names the map uses."""
    chosen = [r for r in ranked if r["status"] == "chosen"]
    others = [r for r in ranked if r["status"] not in ("chosen", "not needed")]
    considered = sorted(chosen + others[: max(0, 25 - len(chosen))], key=lambda r: -r["count"])
    index = {tuple(c): k for k, c in enumerate(centres)}
    role_of = {}
    for role, centre in (roles or {}).items():
        near = min(centres, key=lambda c: protocol.separation_um(c, centre), default=None)
        if near is not None:
            role_of[tuple(near)] = role

    def label(r):
        c = tuple(r["center_um"])
        name = f"patch_{index[c]} " if c in index else ""
        status = r["status"]
        if status == "chosen" and c in role_of:
            status = f"chosen -> {role_of[c]}"
        return f"{name}({c[0]:.0f},{c[1]:.0f}) {status}"

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
    ax.set_yticks(range(len(considered)), [label(r) for r in considered], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"active electrodes within {radius:.0f} um (tick = recorded)")
    ax.set_title(
        "candidate patches, densest first (green = the candidates; the roles went to the least-coupled triple)",
        fontsize=9,
        loc="left",
    )


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
        _draw_ranking(ax_rank, ranked, radius, [tuple(c) for c in centres], params.regions)
    else:
        ax_scan.text(0.5, 0.5, "centres given by hand", ha="center", transform=ax_scan.transAxes)
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
