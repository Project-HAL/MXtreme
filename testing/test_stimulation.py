"""The stimulation subpackage: timelines compile as designed, without maxlab."""

import sys

import pytest

from mxtreme.stimulation import timeline as tl


def test_timeline_module_imports_without_maxlab():
    import mxtreme.stimulation
    from mxtreme.stimulation import timeline

    assert mxtreme.stimulation.timeline is timeline
    assert "maxlab" not in sys.modules


def test_pulse_geometry_on_the_stimulation_clock():
    g = tl.PulseGeometry.of(phase_us=100, pulse_hz=0.2)
    assert g.phase_frames == 2  # 100 us at 50 us per step, whatever the device records at
    assert g.interval_frames == 100000
    assert g.burst_frames == 0 and g.event_span == 4
    burst = tl.PulseGeometry.of(100, 0.2, pulses_per_burst=11, burst_hz=20)
    assert burst.burst_frames == 1000 and burst.event_span == 10 * 1000 + 4
    with pytest.raises(ValueError):
        tl.PulseGeometry.of(phase_us=30000, pulse_hz=1, pulses_per_burst=2, burst_hz=20)


def test_amplitude_bits():
    assert tl.amplitude_bits(200, 2.94) == 68
    with pytest.raises(ValueError):
        tl.amplitude_bits(2000, 2.94)


def test_pulse_is_biphasic_and_polarity_flips():
    t = tl.Timeline()
    t.pulse(10, 0, 50, 2, anodic_first=True)
    steps = [(s.frame, s.args) for s in t.ordered()]
    assert steps == [(10, (0, 562)), (12, (0, 462)), (14, (0, 512))]
    t = tl.Timeline()
    t.pulse(10, 0, 50, 2, anodic_first=False)
    assert [s.args[1] for s in t.ordered()] == [462, 562, 512]


def test_compile_chains_long_delays_and_orders_within_a_frame():
    t = tl.Timeline()
    t.disconnect(70000, 3)
    t.step(70000, 0, 512)
    t.connect(70000, 3, 0)
    out = tl.ListEmitter()
    t.compile(out)
    assert out.commands[:2] == [("delay", 65535), ("delay", 70000 - 65535)]
    assert [c[0] for c in out.commands[2:]] == ["connect", "dac", "disconnect"]


def test_shared_dac_steps_are_emitted_once():
    a = tl.RegionUnits("CS", 0, [1], 200.0, 1, [2])
    b = tl.RegionUnits("US", 0, [3], 200.0, 1, [4])
    g = tl.PulseGeometry.of(100, 1.0)
    timeline, switched = tl.region_timeline([a, b], {"CS": 0.0, "US": 0.0}, 2, g, 2.94)
    assert not switched
    dac_steps = [s for s in timeline.ordered() if s.kind == "dac"]
    assert len(dac_steps) == 2 * 3 * 2  # two DACs, three steps per pulse, two events
    # Drive goes up first, return goes down first.
    first = {s.args[0]: s.args[1] for s in dac_steps if s.frame == 0}
    assert first == {0: 512 + 68, 1: 512 - 68}


def test_offset_pairing_on_shared_dacs_switches_units_per_event():
    a = tl.RegionUnits("CS", 0, [1], 200.0, 1, [2])
    b = tl.RegionUnits("US", 0, [3], 200.0, 1, [4])
    g = tl.PulseGeometry.of(100, 1.0)
    timeline, switched = tl.region_timeline([a, b], {"CS": 0.0, "US": 0.02}, 2, g, 2.94)
    assert switched
    out = tl.ListEmitter()
    timeline.compile(out)
    kinds = [c[0] for c in out.commands]
    # CS's units connect, pulse, release; then 20 ms - a pulse later US's do the same.
    assert kinds[:2] == ["connect", "connect"]
    first_disconnect = kinds.index("disconnect")
    assert kinds[first_disconnect + 2] == "delay"
    assert out.commands[first_disconnect + 2] == ("delay", 400 - 4)
    assert kinds.count("connect") == 2 * 2 * 2 and kinds.count("disconnect") == 2 * 2 * 2


def test_different_dacs_per_region_never_switch():
    a = tl.RegionUnits("CS", 1, [1], 200.0)
    b = tl.RegionUnits("US", 0, [3], 200.0)
    g = tl.PulseGeometry.of(100, 1.0)
    _, switched = tl.region_timeline([a, b], {"CS": 0.0, "US": 0.5}, 3, g, 2.94)
    assert not switched


def test_waveform_matches_timeline():
    g = tl.PulseGeometry.of(100, 1.0)
    waves = tl.waveform({"US": 0.0}, 2, g, {"US": 200.0}, with_return=True)
    t, v = waves["US"]
    assert v[2] == 200.0 and v[4] == -200.0  # up then down
    assert waves["US return"][1][2] == -200.0
    assert t[7] == pytest.approx(1.0)  # second event a second later
