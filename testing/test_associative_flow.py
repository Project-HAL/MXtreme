"""The offline halves of the associative flow, end to end on synthetic MaxWell-shaped files:
select choosing from a baseline, and report reading a run back from the recording alone."""

import json

import h5py
import numpy as np
import pytest

from mxtreme.experiments.associative import protocol
from mxtreme.experiments.associative.params import AssociativeParams

FPS = 20000.0
SPIKE = [("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")]
MAP = [("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")]


def _write(path, recordings, well=0, assay=None):
    """recordings: list of dicts with 'spikes' [(frame, electrode)], 'electrodes', optional
    'events' [(frame, dict)] and 'raw' (channels) for a raw group."""
    with h5py.File(path, "w") as f:
        for k, r in enumerate(recordings):
            g = f.create_group(f"recordings/rec{k:04d}/well{well:03d}")
            channel_of = {e: i for i, e in enumerate(r["electrodes"])}
            g.create_dataset(
                "spikes", data=np.array([(fr, channel_of[e], -60.0) for fr, e in r["spikes"]], dtype=SPIKE)
            )
            g.create_dataset(
                "settings/mapping",
                data=np.array([(c, e, *protocol.electrode_xy(e)) for e, c in channel_of.items()], dtype=MAP),
            )
            g.create_dataset("settings/sampling", data=np.array([FPS]))
            g.create_dataset("settings/lsb", data=np.array([1e-6]))
            g.create_dataset("start_time", data=np.array([0]))
            g.create_dataset("stop_time", data=np.array([int(r.get("seconds", 60) * 1000)]))
            ev = np.array(
                [(fr, 1, 1, json.dumps(m).encode()) for fr, m in r.get("events", [])],
                dtype=[
                    ("frameno", "<i8"),
                    ("eventtype", "<u4"),
                    ("eventid", "<u4"),
                    ("eventmessage", h5py.string_dtype()),
                ],
            )
            g.create_dataset("events", data=ev)
            if r.get("raw"):
                n = int(r.get("seconds", 60) * FPS)
                grp = g.create_group("groups/raw_regions")
                grp.create_dataset("channels", data=np.array(r["raw"]))
                grp.create_dataset("frame_nos", data=np.arange(n, dtype=np.int64) + r.get("first", 0))
                grp.create_dataset("raw", data=np.zeros((len(r["raw"]), n), dtype=np.uint16))
        for key, value in (assay or {}).items():
            f.create_dataset(f"assay/{key}", data=np.array([value.encode()]))


CENTRES = [(600.0, 1000.0), (2400.0, 600.0), (1500.0, 1900.0)]


def _baseline(tmp_path, seconds=60, silent=False, spread=40):
    rng = np.random.default_rng(3)
    patches = [protocol.within(range(protocol.NUM_ELECTRODES), c, 120)[::9][:14] for c in CENTRES]
    everywhere = list(range(0, protocol.NUM_ELECTRODES, 677))[:spread]
    electrodes = sorted({e for p in patches for e in p} | set(everywhere))
    spikes = []
    if not silent:
        for e in electrodes:
            n = rng.poisson(2.0 * seconds)
            spikes += [(int(t), e) for t in rng.integers(0, int(seconds * FPS), n)]
        for k in range(int(seconds / 3)):  # network bursts every 3 s, every patch taking part
            start = int(k * 3 * FPS)
            for e in electrodes:
                spikes += [(start + int(rng.integers(0, 2000)) + j * 60, e) for j in range(8)]
    spikes.sort()
    path = tmp_path / "baseline.raw.h5"
    _write(path, [{"spikes": spikes, "electrodes": electrodes, "seconds": seconds}])
    return path, patches, electrodes


def _params(**kw):
    base = {
        "batch": "fall2026_batch1_iPSC_M1",
        "chip": "M07459",
        "plate_date": 260401,
        "div": 27,
        "well": 0,
        "min_region_separation_um": 1000.0,
    }
    base.update(kw)
    return AssociativeParams(**base)


def test_select_chooses_from_the_baseline_and_writes_outside_the_store(tmp_path):
    from mxtreme.experiments.associative.select import select

    baseline, _patches, electrodes = _baseline(tmp_path)
    out = tmp_path / "work"
    chosen, path = select(
        _params(), str(out), baseline=str(baseline), candidates=3, on_progress=lambda _: None
    )
    assert str(path).startswith(str(out))
    assert path.endswith("_DIV_27_assoc_params.json")
    # The three regions land on the three patches the baseline was built with.
    for centre in chosen.regions.values():
        assert min(protocol.separation_um(centre, c) for c in CENTRES) < 150
    # Routing: every baseline electrode fits here, so all of them are kept.
    assert set(chosen.rec_electrodes) == set(electrodes)
    written = sorted(p.name for p in out.iterdir())
    assert any(n.endswith("_coupling.png") for n in written)
    assert any(n.endswith("_regions.json") for n in written)


def test_select_adds_the_scans_active_electrodes_inside_the_chosen_regions(tmp_path):
    from mxtreme.experiments.associative.select import select

    baseline, _patches, electrodes = _baseline(tmp_path)
    # An activity scan that also found activity on electrodes the baseline did not record, near
    # the first patch.
    extra = [
        e for e in protocol.within(range(protocol.NUM_ELECTRODES), CENTRES[0], 120) if e not in electrodes
    ][:10]
    rng = np.random.default_rng(1)
    scan_spikes = sorted((int(t), e) for e in extra + electrodes for t in rng.integers(0, 60 * FPS, 30))
    scan = tmp_path / "scan.raw.h5"
    _write(scan, [{"spikes": scan_spikes, "electrodes": extra + electrodes, "seconds": 60}])
    with h5py.File(scan, "a") as f:  # what load_activity_scan reads its length from
        f.create_dataset("assay/inputs/record_time", data=np.array([60]))
        f.copy("recordings/rec0000/well000", f.require_group("wells"), name="well000")
        f.move("wells/well000", "wells/well000_tmp")
        f.create_group("wells/well000")
        f.move("wells/well000_tmp", "wells/well000/rec0000")

    chosen, _ = select(
        _params(),
        str(tmp_path / "w"),
        baseline=str(baseline),
        activity_scan=str(scan),
        candidates=3,
        on_progress=lambda _: None,
    )
    assert set(extra) <= set(chosen.rec_electrodes)


def test_select_places_regions_by_hand_on_a_silent_baseline(tmp_path):
    from mxtreme.experiments.associative.select import select

    baseline, _, electrodes = _baseline(tmp_path, silent=True)
    with pytest.raises(ValueError, match="silent"):
        select(_params(), str(tmp_path / "w"), baseline=str(baseline), on_progress=lambda _: None)
    chosen, _ = select(
        _params(),
        str(tmp_path / "w"),
        baseline=str(baseline),
        centers=";".join(f"{x},{y}" for x, y in CENTRES),
        on_progress=lambda _: None,
    )
    assert set(chosen.rec_electrodes) == set(electrodes)  # the silent baseline supplied the routing


def test_select_leaves_room_for_the_stimulation_electrodes(tmp_path):
    from mxtreme.experiments.associative.select import _experiment_routing

    pool = list(range(0, protocol.NUM_ELECTRODES, 20))[:1020]
    roles = {"US": CENTRES[0], "CS": CENTRES[1], "NS": CENTRES[2]}
    rec, regions = _experiment_routing(pool, roles, 150.0, None, 1020 - 24)
    assert len(rec) == 1020 - 24
    assert all(set(v) <= set(rec) for v in regions.values())  # the regions survive the cut


def test_raw_channels_follow_raw_traces():
    from mxtreme.experiments.associative.run import raw_channels

    params = _params(
        regions={"US": list(CENTRES[0]), "CS": list(CENTRES[1]), "NS": list(CENTRES[2])},
        rec_electrodes=list(range(0, protocol.NUM_ELECTRODES, 7)),
    )
    regions = protocol.regions_for(params)

    class Routed:
        def get_channels_for_electrodes(self, electrodes):
            return [e % 1024 for e in electrodes]

    assert raw_channels(params, regions, Routed()) == []
    assert raw_channels(params.with_overrides(["raw_traces=all"]), regions, Routed()) == list(range(1024))
    some = raw_channels(params.with_overrides(["raw_traces=regions"]), regions, Routed())
    wanted = {e % 1024 for s in regions.values() for e in s.rec_electrodes + s.stim_electrodes}
    assert set(some) == wanted


def _record(params, blocks, stimuli):
    regions = protocol.regions_for(params)
    return protocol.schedule_record(params, blocks, stimuli, regions, {"units": {}}), regions


def test_report_reads_a_spikes_only_multi_recording_file_by_itself(tmp_path, capsys):
    from mxtreme.experiments.associative.report import load_protocol, report
    from mxtreme.experiments.associative.run import PROTOCOL_KEY

    params = _params(
        regions={"US": list(CENTRES[0]), "CS": list(CENTRES[1]), "NS": list(CENTRES[2])},
        rec_electrodes=list(range(0, protocol.NUM_ELECTRODES, 11)),
        mode="connectivity",
        check_reps=2,
        check_iti=2,
        pre_min=0.05,
        post_min=0.05,
    )
    blocks, stimuli = protocol.build_schedule(params)
    record, regions = _record(params, blocks, stimuli)
    electrodes = sorted(
        {e for s in regions.values() for e in s.rec_electrodes} | set(params.rec_electrodes[:50])
    )

    # Two recordings in one file, each with its own presentations; no raw groups at all.
    rng = np.random.default_rng(2)
    recs = []
    frame = 0
    for half in (0, 1):
        spikes, events = [], []
        for p in blocks[1].presentations[half * 3 : (half + 1) * 3]:
            frame += int(2 * FPS)
            events.append((frame, {"start_stimulation": f"{p.token}_0"}))
            us = regions[p.roles[0]].rec_electrodes
            spikes += [(frame + int(0.010 * FPS), e) for e in us[:4]]  # a local response
        spikes += [(int(t), int(e)) for e, t in zip(rng.choice(electrodes, 200), rng.integers(0, frame, 200))]
        recs.append({"spikes": sorted(spikes), "electrodes": electrodes, "events": events, "seconds": 30})
    path = tmp_path / "run.raw.h5"
    _write(path, recs, assay={PROTOCOL_KEY: json.dumps(record, separators=(",", ":"))})

    assert load_protocol(str(path))["params"]["mode"] == "connectivity"
    rows = report(str(path))  # no --protocol, no outputs
    text = capsys.readouterr().out
    assert "2 recording(s)" in text
    assert len(rows) == 6
    assert "kept spikes only" in text
    assert not list(tmp_path.glob("*_report.png")) and not list(tmp_path.glob("*.csv"))


def test_report_says_where_the_protocol_went_when_the_recording_has_none(tmp_path):
    from mxtreme.experiments.associative.report import load_protocol

    path = tmp_path / "old.raw.h5"
    _write(path, [{"spikes": [(1, 5)], "electrodes": [5]}])
    with pytest.raises(KeyError, match="--protocol"):
        load_protocol(str(path))


def test_storage_estimate_scales_with_channels_and_counts_raw_only_when_asked():
    from mxtreme.experiments.associative.preview import SPIKE_BYTES, storage_estimate

    spikes_only = AssociativeParams(batch="b_M2", chip="M1", plate_date=1, div=1, raw_traces="none")
    one = storage_estimate(spikes_only, minutes=10, channels=100)
    assert one["raw_gb"] == 0.0
    # 0.5-2 spikes per channel-second, 16 bytes each: the range is the culture, not the format.
    assert one["spikes_gb"] == pytest.approx(
        (0.5 * 100 * 600 * SPIKE_BYTES / 1e9, 2.0 * 100 * 600 * SPIKE_BYTES / 1e9)
    )
    assert storage_estimate(spikes_only, 20, 200)["spikes_gb"][0] == pytest.approx(4 * one["spikes_gb"][0])

    everything = AssociativeParams(batch="b_M2", chip="M1", plate_date=1, div=1, raw_traces="all")
    maxone = AssociativeParams(batch="b_M1", chip="M1", plate_date=1, div=1, raw_traces="all")
    assert storage_estimate(everything, 10, 100)["raw_channels"] == 1024
    assert storage_estimate(maxone, 10, 100)["raw_gb"] == pytest.approx(
        2 * storage_estimate(everything, 10, 100)["raw_gb"]
    )


def test_checkpoints_group_probes_and_split_the_encode_block_at_each_training_run():
    from mxtreme.experiments.associative.report import checkpoints

    def row(i, block, token, us, frame):
        return {"index": i, "block": block, "checkpoint": "", "token": token, "US_pulse": us, "frame": frame}

    rows = [
        row(0, "baseline", "probe_cs", 40, 0),
        row(1, "baseline", "probe_cs", 80, 60_000),
        row(2, "encode", "stim_cs_us", 99, 120_000),
        row(3, "encode", "checkpoint_cs", 45, 180_000),
        row(4, "encode", "stim_cs_us", 99, 240_000),
        row(5, "encode", "checkpoint_cs", 60, 300_000),
        row(6, "retrieval_1", "probe_cs", 160, 360_000),
    ]
    # A probe is 20 pulses and a mid-encode checkpoint 10, so the curve has to count per pulse for
    # the two to sit on one axis.
    stimuli = {
        "probe_cs": {"num_events": 20, "pulses_per_burst": 1},
        "checkpoint_cs": {"num_events": 10, "pulses_per_burst": 1},
        "stim_cs_us": {"num_events": 89, "pulses_per_burst": 1},
    }
    curve = checkpoints(rows, FPS, stimuli)
    assert [c["label"] for c in curve] == ["baseline", "encode 1", "encode 2", "retrieval_1"]
    assert [c["per_pulse"]["CS"] for c in curve] == [3.0, 4.5, 6.0, 8.0]
    assert curve[0]["n"] == 2 and curve[1]["n"] == 1
    assert curve[-1]["minutes"] == pytest.approx(360_000 / FPS / 60)
    # Training presentations stay out of the curve, and every row says which checkpoint it is in.
    assert rows[2]["checkpoint"] == "" and rows[3]["checkpoint"] == "encode 1"


def test_place_site_moves_the_driven_block_onto_electrodes_with_spikes():
    from mxtreme.experiments.associative.select import place_site

    site = {"shape": "focal", "inner": 2, "inner_gap": 4, "return_radius": 6, "return_points": "corners"}
    centre = (1750.0, 1000.0)
    drive, _ = protocol.stim_site(centre, site)
    # Activity two electrodes to the right of every driven electrode, and nowhere else.
    activity = {e + 2: (1.5, 40.0) for e in drive}
    placed = place_site(centre, site, activity)
    assert placed["active"] == 4 and placed["of"] == 4
    assert placed["shift_um"] == pytest.approx(2 * protocol.PITCH_UM)
    assert set(placed["driven"]) == set(activity)

    # Nothing known: the site stays put and says every electrode is silent.
    still = place_site(centre, site, {})
    assert still["center_um"] == centre and still["shift_um"] == 0.0 and still["active"] == 0

    # Already on activity: no move, even if a move would find as much.
    here = place_site(centre, site, {e: (1.0, 30.0) for e in drive} | activity)
    assert here["shift_um"] == 0.0 and here["active"] == 4


def test_calibration_table_reports_how_far_a_pulse_reaches():
    from mxtreme.experiments.associative.checks import calibration_table, compare_calibrations

    def responses(local, remote):
        return {
            "stim_us_a80": {
                "US": np.array([local] * 4),
                "CS": np.array([remote] * 4),
                "NS": np.array([remote] * 4),
            },
            "stim_us_a40": {
                "US": np.array([local / 5] * 4),
                "CS": np.array([0.0] * 4),
                "NS": np.array([0.0] * 4),
            },
        }

    tokens = {"stim_us_a80": ("US", 80.0, "anodic-first"), "stim_us_a40": ("US", 40.0, "anodic-first")}
    focal = calibration_table(responses(2.0, 0.2), {}, tokens)
    grid = calibration_table(responses(2.0, 1.0), {}, tokens)
    assert focal[1]["remote"] == pytest.approx(0.2) and grid[1]["remote"] == pytest.approx(1.0)

    # Same threshold and local response; the grid spreads five times as far, so focal wins.
    _summary, verdicts = compare_calibrations({"grid": grid, "focal": focal}, ["US"])
    assert verdicts[0].level == "ok" and verdicts[0].message.startswith("US: focal")

    # A pattern that reaches threshold at a lower amplitude wins outright.
    eager = calibration_table(responses(2.0, 0.2), {}, tokens)
    eager[0]["local"] = 0.9  # 40 mV already evokes a response
    _summary, verdicts = compare_calibrations({"focal": focal, "eager": eager}, ["US"])
    assert verdicts[0].level == "ok" and verdicts[0].message.startswith("US: eager")


def test_compare_names_the_pattern_that_drives_the_site_without_spreading(tmp_path, capsys):
    from mxtreme.experiments.associative.report import compare
    from mxtreme.experiments.associative.run import PROTOCOL_KEY

    def calibration(site, remote_spikes):
        params = _params(
            regions={"US": list(CENTRES[0]), "CS": list(CENTRES[1]), "NS": list(CENTRES[2])},
            rec_electrodes=list(range(0, protocol.NUM_ELECTRODES, 11)),
            mode="calibration",
            stim_site=site,
            calibration_amplitudes_mv=[40, 80],
            calibration_polarities=["anodic-first"],
            calibration_reps=3,
            calibration_iti=1.0,
            pre_min=0.05,
            post_min=0.05,
            burst_warn=1.0,  # random background can look like a burst; this test is about spread
        )
        blocks, stimuli = protocol.build_schedule(params)
        record, regions = _record(params, blocks, stimuli)
        electrodes = sorted({e for s in regions.values() for e in s.rec_electrodes})
        rng = np.random.default_rng(5)
        spikes, events = [], []
        frame = int(FPS)
        for block in blocks:
            for p in block.presentations:
                frame += int(FPS)
                events.append((frame, {"start_stimulation": f"{p.token}_0"}))
                if stimuli[p.token].amplitude_mv >= 80:
                    role = p.roles[0]
                    spikes += [(frame + int(0.010 * FPS), e) for e in regions[role].rec_electrodes[:3]]
                    for other in protocol.ROLES:
                        if other != role:
                            spikes += [
                                (frame + int(0.012 * FPS), e)
                                for e in regions[other].rec_electrodes[:remote_spikes]
                            ]
        # Enough background that a three-spike evoked response is not itself a "network burst".
        spikes += [
            (int(t), int(e)) for e, t in zip(rng.choice(electrodes, 3000), rng.integers(0, frame, 3000))
        ]
        path = tmp_path / f"{site['shape']}.raw.h5"
        _write(
            path,
            [
                {
                    "spikes": sorted(spikes),
                    "electrodes": electrodes,
                    "events": events,
                    "seconds": frame / FPS + 1,
                }
            ],
            assay={PROTOCOL_KEY: json.dumps(record, separators=(",", ":"))},
        )
        return str(path)

    focal = calibration({"shape": "focal", "inner": 2, "inner_gap": 4, "return_radius": 6}, remote_spikes=0)
    grid = calibration({"shape": "grid", "size": 3, "gap": 2}, remote_spikes=2)
    verdicts = compare([grid, focal], str(tmp_path / "compare.png"))
    text = capsys.readouterr().out
    assert "=== patterns compared ===" in text
    assert "focal 2x2 gap 4" in text and "grid 3x3 gap 2" in text
    # Same threshold and local response; the grid reaches the other regions, the focal does not.
    assert all(v.level == "ok" and "focal" in v.message.split("(")[0] for v in verdicts)
    assert (tmp_path / "compare.png").exists()


def test_calibration_warns_when_a_site_responds_only_at_the_top_of_the_ladder():
    from mxtreme.experiments.associative.checks import calibration_verdicts

    def row(role, mv, local):
        return {
            "role": role,
            "amplitude_mv": mv,
            "polarity": "anodic-first",
            "local": local,
            "remote": 0.0,
            "pulses": 10,
            "burst_rate": 0.0,
        }

    rows = [
        row("US", 40, 0.1),
        row("US", 80, 0.2),
        row("US", 120, 1.0),
        row("CS", 40, 0.1),
        row("CS", 80, 0.9),
        row("CS", 120, 1.5),
    ]
    chosen, verdicts = calibration_verdicts(rows, ["US", "CS"])
    assert chosen["US"]["amplitude_mv"] == 120 and chosen["CS"]["amplitude_mv"] == 120
    messages = [v.message for v in verdicts if v.level == "warn"]
    assert len(messages) == 1 and messages[0].startswith("US responds only at the top of the ladder")


def _curve(base_cs, ret_cs, base_ns=0.1, ret_ns=0.1, base_us=1.0, ret_us=1.0, pulses=60):
    """A checkpoint curve with baseline and one retrieval, rates given per pulse."""

    def group(label, cs, ns, us, n):
        return {
            "label": label,
            "minutes": 0.0,
            "n": 9,
            "per_pulse": {"CS": cs, "NS": ns, "US": us},
            "spikes": {"CS": round(cs * n), "NS": round(ns * n), "US": round(us * n)},
            "pulses": {"CS": n, "NS": n, "US": n},
        }

    return [
        group("baseline", base_cs, base_ns, base_us, pulses),
        group("encode 1", (base_cs + ret_cs) / 2, base_ns, 0, 20),
        group("retrieval_1", ret_cs, ret_ns, ret_us, pulses),
    ]


def test_conditioning_verdicts_read_the_curve():
    from mxtreme.experiments.associative.checks import conditioning_verdicts

    levels = lambda verdicts: [(v.level, v.message.split(":")[0]) for v in verdicts]

    # CS rises well beyond two standard errors, NS does not: the association.
    assert levels(conditioning_verdicts(_curve(0.10, 0.60))) == [("ok", "association")]
    # Nothing moves: the null, with the tetanic variant named.
    flat = conditioning_verdicts(_curve(0.10, 0.12))
    assert levels(flat) == [("warn", "no association")] and "pulses_per_burst" in flat[0].message
    # Both rise: excitability, not association.
    assert levels(conditioning_verdicts(_curve(0.10, 0.60, base_ns=0.1, ret_ns=0.6))) == [
        ("warn", "both CS alone (0.10 to 0.60) and NS alone (0.10 to 0.60) rose in US")
    ]
    # A high floor at baseline is flagged before the result, and the result is still read.
    floor = conditioning_verdicts(_curve(0.50, 0.90, base_us=1.0))
    assert floor[0].level == "warn" and floor[0].message.startswith("at baseline CS alone already")
    assert floor[-1].level == "ok"
    # Bursting pulses and US drift are their own warnings.
    noisy = conditioning_verdicts(_curve(0.10, 0.60, ret_us=2.0), {"probe": 0.05, "train": 0.4})
    kinds = [v.message.split(" ")[0] for v in noisy]
    assert any("training pulses" in v.message for v in noisy) and any(
        "excitability drifted" in v.message for v in noisy
    )
    assert kinds[-1] == "association:"
    # No retrieval yet: says so, and reads what there is.
    early = conditioning_verdicts(_curve(0.10, 0.60)[:2])
    assert early[-1].level == "ok" and "no retrieval block yet" in early[-1].message


def test_calibration_warns_when_cs_and_ns_are_not_matched_in_drive():
    from mxtreme.experiments.associative.checks import calibration_verdicts

    def row(role, mv, local):
        return {
            "role": role,
            "amplitude_mv": mv,
            "polarity": "anodic-first",
            "local": local,
            "remote": 0.0,
            "pulses": 10,
            "burst_rate": 0.0,
        }

    rows = [
        row("CS", 40, 0.6),
        row("CS", 80, 1.2),
        row("CS", 120, 3.0),
        row("NS", 40, 0.2),
        row("NS", 80, 0.9),
        row("NS", 120, 1.1),
    ]
    chosen, verdicts = calibration_verdicts(rows, ["CS", "NS"])
    assert chosen["CS"]["local"] == 3.0 and chosen["NS"]["local"] == 1.1
    match = [v for v in verdicts if "not matched in drive" in v.message]
    assert len(match) == 1 and match[0].level == "warn"
    assert "80 mV brings CS to 1.2" in match[0].message


def _session(tmp_path, phases, learned_from="retrieval", gap_sec=120.0):
    """One synthetic recording per phase, each with its own start clock, gap_sec apart."""
    from mxtreme.experiments.associative.run import PROTOCOL_KEY

    base = {
        "regions": {"US": list(CENTRES[0]), "CS": list(CENTRES[1]), "NS": list(CENTRES[2])},
        "rec_electrodes": list(range(0, protocol.NUM_ELECTRODES, 11)),
        "t_stim": 6,
        "t_probe": 6,
        "encode_probe_sec": 3,
        "pulse_hz": 1.0,
        "probe_iti": 8,
        "encode_iti": 8,
        "encode_cycles": 2,
        "encode_cycle_rest": 1,
        "encode_pattern": ["PAIR", "NS"],
        "num_retrievals": 1,
        "probe_reps": 2,
        "pre_min": 0.05,
        "post_min": 0.05,
        "decay_min": 0.05,
        "retrieval_interval_min": 0.05,
        "phase_lead_min": 0.05,
        "amplitudes_source": "test",
        "burst_warn": 1.0,
    }
    paths, clock_ms, order = [], 1_000_000, list(phases)
    for phase in order:
        params = _params(**base, phase=phase)
        blocks, stimuli = protocol.build_schedule(params)
        record, regions = _record(params, blocks, stimuli)
        electrodes = sorted({e for s in regions.values() for e in s.rec_electrodes})
        rng = np.random.default_rng(len(paths))
        spikes, events = [], []
        frame0 = int(FPS)
        learned = order.index(phase) >= order.index(learned_from)
        for block, st in zip(blocks, protocol.block_starts_sec(blocks)):
            events.append((frame0 + int(st * FPS), {f"{block.label}_start": 1}))
            for p in block.presentations:
                frame = frame0 + int((st + p.t_sec) * FPS)
                events.append((frame, {"start_stimulation": f"{p.token}_0"}))
                for off in stimuli[p.token].offsets_sec:
                    f = frame + int(off * FPS)
                    if "US" in p.roles:
                        spikes += [(f + int(0.010 * FPS), e) for e in regions["US"].rec_electrodes[:3]]
                    if p.roles == ["CS"] and learned:
                        spikes += [(f + int(0.015 * FPS), e) for e in regions["US"].rec_electrodes[:2]]
        end = frame0 + int(sum(b.duration_sec for b in blocks) * FPS)
        spikes += [(int(t), int(e)) for e, t in zip(rng.choice(electrodes, 3000), rng.integers(0, end, 3000))]
        path = tmp_path / f"{phase}.raw.h5"
        _write(
            path,
            [
                {
                    "spikes": sorted(spikes),
                    "electrodes": electrodes,
                    "events": events,
                    "seconds": end / FPS + 1,
                }
            ],
            assay={PROTOCOL_KEY: json.dumps(record, separators=(",", ":"))},
        )
        with h5py.File(path, "a") as f:
            f["recordings/rec0000/well000/start_time"][...] = clock_ms
        clock_ms += int((end / FPS + gap_sec) * 1000)
        paths.append(str(path))
    return paths


def test_report_reads_a_split_session_as_one_curve(tmp_path, capsys):
    from mxtreme.experiments.associative.report import report

    paths = _session(tmp_path, ["baseline", "encode_1", "encode_2", "retrieval"])
    # Given out of order: the recordings' clocks put them right.
    rows = report(
        [paths[2], paths[0], paths[3], paths[1]],
        out_csv=str(tmp_path / "s.csv"),
        out_png=str(tmp_path / "s.png"),
    )
    text = capsys.readouterr().out
    assert "session: 4 recordings" in text
    assert [r["file"] for r in rows] == ["baseline"] * 6 + ["encode_1"] * 4 + ["encode_2"] * 2 + [
        "retrieval"
    ] * 6
    assert [r["index"] for r in rows] == list(range(18))
    # Session time carries the two-minute gaps between recordings.
    assert rows[6]["sec"] - rows[5]["sec"] > 120
    curve_labels = [line.split()[0] for line in text.split("association, by checkpoint")[1].splitlines()[2:5]]
    assert curve_labels == ["baseline", "encode", "retrieval_1"]
    assert "[ok  ] association" in text
    assert (tmp_path / "s.csv").exists() and (tmp_path / "s.png").exists()

    # One recording alone still reads, and says the session is not over.
    report(paths[0])
    assert "no retrieval block yet" in capsys.readouterr().out
