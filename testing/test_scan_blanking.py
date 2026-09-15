"""The scan runners drive the MaxLab server the way MaxLab Live's own Activity Scan assay does.

Run against a fake ``maxlab`` that records every command it is asked for, since the real one needs
the rig. What is pinned down here is the order on the wire around the two steps that slam every
amplifier -- ``system_initialize`` and ``mea_array_download`` -- and the waits between them:

- the stream is blanked (``stream_blanking_on <s>``) before the step, the runner sleeps ``s``, and
  the blank is lifted (``stream_blanking_off``) -- the Scope assay's sequence;
- on a multi-well chip every well's download happens inside one blank;
- after ``offset()`` (which sleeps the system-specific settle time itself) the wait is
  ``Timing.waitAfterOffset``, never the MaxTwo settle time a second time.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import time
import types

import pytest

MAXONE, MAXTWO = 0, 1


def _fake_maxlab(system_type: int, calls: list) -> types.ModuleType:
    """A ``maxlab`` whose every server-facing call appends to ``calls``."""

    class Timing:
        waitInit = 2
        waitAfterDownload = 5
        waitAfterOffset = 5
        waitInMX1Offset = 5
        waitInMX2Offset = 15
        waitAfterBankSwitch = 2
        waitAfterRecording = 2

    class Array:
        def __init__(self, token="online", persistent=False):
            self.token, self.persistent = token, persistent
            calls.append(("mea_array_new", token))

        def close(self):
            if not self.persistent:
                calls.append(("mea_array_delete", self.token))

        def reset(self):
            return "OK"

        def clear_selected_electrodes(self):
            return "OK"

        def select_electrodes(self, electrodes, weight=1):
            return "OK"

        def select_stimulation_electrodes(self, electrodes):
            return "OK"

        def route(self):
            calls.append(("route", self.token))
            return "OK"

        def download(self, wells=None):
            calls.append(("download", self.token, tuple(wells or [0])))

    class Saving:
        def __getattr__(self, name):  # every Saving method is one server command
            def method(*args, **_kwargs):
                calls.append((f"saving_{name}",) + args)
                return "ok"

            return method

    class _Api:
        def send(self, msg):
            return str(system_type) if msg == "wellplate_query_version" else "ok"

    comm = types.ModuleType("maxlab.comm")

    @contextlib.contextmanager
    def api_context(*_args, **_kwargs):
        yield _Api()

    comm.api_context = api_context

    mx = types.ModuleType("maxlab")
    mx.__spec__ = importlib.machinery.ModuleSpec("maxlab", None)
    mx.Timing = Timing
    mx.Array = Array
    mx.Saving = Saving
    mx.comm = comm
    mx.initialize = lambda wells=None: calls.append(("initialize", wells))
    mx.activate = lambda wells: calls.append(("activate", tuple(wells)))
    mx.offset = lambda: calls.append(("offset",))
    mx.send_raw = lambda msg: calls.append(("send_raw", msg)) or "ok"
    return mx


@pytest.fixture(params=[MAXONE, MAXTWO], ids=["MaxOne", "MaxTwo"])
def rig(request, monkeypatch):
    """Install the fake ``maxlab`` for one system type; yields ``(system_type, calls)``.

    ``mx_setup`` binds ``maxlab`` at import, so it is dropped from ``sys.modules`` on both sides of
    the test and re-imports against the fake.
    """
    import mxtreme.scans

    calls: list = []
    fake = _fake_maxlab(request.param, calls)
    monkeypatch.setitem(sys.modules, "maxlab", fake)
    monkeypatch.setitem(sys.modules, "maxlab.comm", fake.comm)
    # `from mxtreme.scans import mx_setup` also finds the module cached on the package object.
    monkeypatch.delitem(sys.modules, "mxtreme.scans.mx_setup", raising=False)
    monkeypatch.delattr(mxtreme.scans, "mx_setup", raising=False)
    monkeypatch.setattr(time, "sleep", lambda s: calls.append(("sleep", s)))
    yield request.param, calls
    sys.modules.pop("mxtreme.scans.mx_setup", None)
    if hasattr(mxtreme.scans, "mx_setup"):
        delattr(mxtreme.scans, "mx_setup")


def _wells(system_type):
    return [0] if system_type == MAXONE else [0, 1]


def _blanked(calls, kind):
    """Each ``(blank seconds, calls inside the blank)`` window that contains a ``kind`` call."""
    windows, inside = [], None
    for call in calls:
        if call[0] == "send_raw" and call[1].startswith("stream_blanking_on "):
            inside = (float(call[1].split()[1]), [])
        elif call == ("send_raw", "stream_blanking_off"):
            if any(c[0] == kind for c in inside[1]):
                windows.append(inside)
            inside = None
        elif inside is not None:
            inside[1].append(call)
    assert inside is None, "a blank was never lifted"
    return windows


def _check_initialize(calls):
    [(seconds, inside)] = _blanked(calls, "initialize")
    assert seconds == 2
    assert inside == [("initialize", None), ("sleep", 2)]


def test_activity_scan_blanks_initialize_and_every_download(rig, tmp_path):
    from mxtreme.scans.activity_scan import ActivityScanParams, run_activity_scan

    system_type, calls = rig
    wells = _wells(system_type)
    params = ActivityScanParams(
        batch="fall2026_batch1_E18_M1", wells=wells, n_scans=2, rec_length_sec=1,
        save_path=str(tmp_path),
    )
    run_activity_scan(params, on_progress=lambda _: None)

    _check_initialize(calls)

    windows = _blanked(calls, "download")
    assert len(windows) == 2  # one per round
    for seconds, inside in windows:
        assert seconds == 5
        # every well's wiring goes down inside the one blank, then the settling sleep, then off
        assert inside == [("download", f"well_{w}_array", (w,)) for w in wells] + [("sleep", 5)]

    # after offset: the Scope assay's wait, not the MaxTwo settle time on top of offset()'s own
    after_offset = [calls[i + 1] for i, c in enumerate(calls) if c == ("offset",)]
    assert after_offset == [("sleep", 5)] * 2
    assert ("sleep", 15) not in calls

    # the recording only starts once the blank is off and the offset has settled
    order = [c for c in calls if c[0] in ("download", "send_raw", "offset", "saving_start_recording")]
    first_round = order[: order.index(("saving_start_recording", wells)) + 1]
    assert first_round[-3:] == [
        ("send_raw", "stream_blanking_off"), ("offset",), ("saving_start_recording", wells)
    ]


def test_network_scan_blanks_initialize_and_each_well_download(rig, tmp_path):
    from mxtreme.scans.network_scan import NetworkScanParams, run_network_scan

    system_type, calls = rig
    wells = _wells(system_type)
    params = NetworkScanParams(
        recording_electrodes={w: list(range(w * 100, w * 100 + 50)) for w in wells},
        batch="fall2026_batch1_E18_M1", rec_length_sec=1, pad_to_max=False,
        save_path=str(tmp_path),
    )
    run_network_scan(params, on_progress=lambda _: None)

    _check_initialize(calls)

    windows = _blanked(calls, "download")
    assert [w for _, w in windows] == [
        [("download", f"stimulation{w}", (w,)), ("sleep", 5)] for w in wells
    ]
    assert all(seconds == 5 for seconds, _ in windows)

    after_offset = [calls[i + 1] for i, c in enumerate(calls) if c == ("offset",)]
    assert after_offset == [("sleep", 5)]
    assert ("sleep", 15) not in calls
