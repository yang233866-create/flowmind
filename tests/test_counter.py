"""Directional counter: crossing sense, dedup, and profile binning.

`mask = crossed_in | crossed_out` counts the traffic leaving the intersection as
well as the traffic arriving, which roughly doubles a two-way approach's measured
flow. The sense is now an explicit part of the ROI config.
"""
from __future__ import annotations

import numpy as np
import pytest

from vision.counter import DirectionalCounter

LINES = {"count_lines": {d: [[0, 100], [200, 100]] for d in
                         ("north", "south", "east", "west")}}


class _Det:
    """Minimal stand-in for sv.Detections (only what update() touches)."""

    def __init__(self, tracker_ids, class_ids=None):
        self.tracker_id = np.asarray(tracker_ids)
        self.class_id = None if class_ids is None else np.asarray(class_ids)

    def __len__(self) -> int:
        return len(self.tracker_id)


class _Zone:
    """Replays a scripted (crossed_in, crossed_out) per trigger() call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def trigger(self, detections):
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        cin, cout = self.script[i]
        return np.asarray(cin, dtype=bool), np.asarray(cout, dtype=bool)


def _counter(sense=None, script=((( True,), (True,)),)):
    cfg = dict(LINES)
    if sense is not None:
        cfg["count_sense"] = sense
    c = DirectionalCounter(cfg, fps=10.0)
    for d in list(c.zones):
        c.zones[d] = _Zone(script)
    return c


def test_default_sense_is_both_and_is_reported():
    c = _counter()
    assert c.get_sense() == {d: "both" for d in c.get_observed()}


def test_inbound_only_sense_ignores_outbound_crossings():
    """One vehicle in, one out: 'both' counts 2, 'in' counts the arrival only."""
    script = [((True, False), (False, True))]
    both = _counter(sense="both", script=script)
    both.update(_Det([1, 2], [2, 2]), 0)
    assert both.get_counts()["north"] == 2

    inbound = _counter(sense="in", script=script)
    inbound.update(_Det([1, 2], [2, 2]), 0)
    assert inbound.get_counts()["north"] == 1

    outbound = _counter(sense="out", script=script)
    outbound.update(_Det([1, 2], [2, 2]), 0)
    assert outbound.get_counts()["north"] == 1


def test_sense_can_be_set_per_direction():
    c = _counter(sense={"north": "in", "south": "out"})
    s = c.get_sense()
    assert s["north"] == "in" and s["south"] == "out"
    assert s["east"] == "both"  # unspecified falls back to the default


def test_unknown_sense_is_rejected_loudly():
    with pytest.raises(ValueError, match="count_sense"):
        DirectionalCounter({**LINES, "count_sense": "inbound"}, fps=10.0)


def test_a_tracker_is_counted_once_per_direction():
    c = _counter(script=[((True,), (False,))])
    for frame in range(5):
        c.update(_Det([7], [2]), frame)
    assert c.get_counts()["north"] == 1
    assert c.get_class_counts()["north"] == {"car": 1}


def test_missing_lines_stay_unobserved():
    c = DirectionalCounter({"count_lines": {"north": [[0, 0], [10, 10]],
                                            "south": [[0, 0]]}}, fps=10.0)
    assert c.get_observed() == ["north"]
    assert set(c.get_sense()) == {"north"}


def test_bin_flow_profile_reports_the_short_final_span():
    c = _counter(script=[((True,), (False,))])
    c.update(_Det([1], [2]), 0)     # t = 0.0 s
    c.update(_Det([2], [5]), 155)   # t = 15.5 s -> final partial bin
    counts, spans = c.bin_flow_profile(bin_sec=5.0, total_duration_sec=16.5)
    assert len(spans) == 4 and spans[:3] == [5.0, 5.0, 5.0]
    assert spans[-1] == pytest.approx(1.5)
    assert counts["north"] == [1, 0, 0, 1]
    assert sum(counts["north"]) == c.get_counts()["north"]


def test_crossings_after_the_declared_duration_land_in_the_last_bin():
    c = _counter(script=[((True,), (False,))])
    c.update(_Det([1], [2]), 300)  # t = 30 s, past total_duration_sec
    counts, _ = c.bin_flow_profile(bin_sec=5.0, total_duration_sec=10.0)
    assert counts["north"] == [0, 1]
