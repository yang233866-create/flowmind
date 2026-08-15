"""Metrics contract: the 7 keys, and 1 Hz sampling for both pipelines."""
from __future__ import annotations

import types

import pytest

from simulation.metrics import MetricsCollector, parse_tripinfo

CONTRACT_KEYS = {"avg_waiting_s", "avg_travel_time_s", "throughput_veh",
                 "avg_queue_veh", "max_queue_veh", "avg_speed_mps", "teleports"}

TRIPINFO = """<tripinfos>
  <tripinfo id="v0" duration="100.0" routeLength="500.0" waitingTime="20.0"/>
  <tripinfo id="v1" duration="50.0" routeLength="500.0" waitingTime="10.0"/>
</tripinfos>
"""


def _fake_traci(queue_per_edge: int = 3, teleports_per_step: int = 0):
    edge = types.SimpleNamespace(
        getLastStepHaltingNumber=lambda e: queue_per_edge,
        getWaitingTime=lambda e: 1.5,
    )
    return types.SimpleNamespace(
        edge=edge,
        vehicle=types.SimpleNamespace(getIDCount=lambda: 7),
        trafficlight=types.SimpleNamespace(getPhase=lambda tls: 2),
        simulation=types.SimpleNamespace(
            getStartingTeleportNumber=lambda: teleports_per_step),
    )


def test_parse_tripinfo_returns_exactly_the_contract_keys(tmp_path):
    p = tmp_path / "tripinfo.xml"
    p.write_text(TRIPINFO, encoding="utf-8")
    m = parse_tripinfo(p)
    assert set(m) == CONTRACT_KEYS
    assert m["throughput_veh"] == 2.0
    assert m["avg_waiting_s"] == pytest.approx(15.0)
    assert m["avg_travel_time_s"] == pytest.approx(75.0)
    assert m["avg_speed_mps"] == pytest.approx((500 / 100 + 500 / 50) / 2)


def test_empty_tripinfo_does_not_divide_by_zero(tmp_path):
    p = tmp_path / "tripinfo.xml"
    p.write_text("<tripinfos>\n</tripinfos>\n", encoding="utf-8")
    m = parse_tripinfo(p)
    assert m["throughput_veh"] == 0.0
    assert m["avg_waiting_s"] == 0.0


def test_queue_metrics_come_from_the_timeseries(tmp_path):
    p = tmp_path / "tripinfo.xml"
    p.write_text(TRIPINFO, encoding="utf-8")
    c = MetricsCollector(_fake_traci(queue_per_edge=3), {"north": "N_in", "south": "S_in",
                                                         "east": "E_in", "west": "W_in"}, "TL")
    for t in range(1, 11):
        c.step(t)
    m = c.parse_tripinfo(p)
    assert m["avg_queue_veh"] == pytest.approx(12.0)  # 4 approaches x 3
    assert m["max_queue_veh"] == 12.0


def test_meta_construction_form_matches_explicit_form():
    meta = {"tls_id": "TL", "approaches": {a: {"in_edge": f"{a[0].upper()}_in"}
                                           for a in ("north", "south", "east", "west")}}
    tr = _fake_traci()
    from_meta = MetricsCollector(tr, meta)
    explicit = MetricsCollector(tr, {a: f"{a[0].upper()}_in" for a in
                                     ("north", "south", "east", "west")}, "TL")
    assert from_meta.in_edges == explicit.in_edges
    assert from_meta.tls_id == explicit.tls_id == "TL"


def test_one_row_per_sample_and_teleports_accumulate():
    """RL evaluation used to sample every 5 s, undercounting teleports 5x."""
    c = MetricsCollector(_fake_traci(teleports_per_step=1), {"north": "N_in", "south": "S_in",
                                                             "east": "E_in", "west": "W_in"}, "TL")
    for t in range(1, 1801):
        c.step(t)
    df = c.to_dataframe()
    assert len(df) == 1800
    assert df["t"].iloc[0] == 1 and df["t"].iloc[-1] == 1800
    assert c.teleports == 1800


def test_timeseries_columns_are_stable():
    c = MetricsCollector(_fake_traci(), {"north": "N_in", "south": "S_in",
                                         "east": "E_in", "west": "W_in"}, "TL")
    c.step(1)
    assert list(c.to_dataframe().columns) == [
        "t", "queue_north", "queue_south", "queue_east", "queue_west",
        "queue_total", "waiting_total", "running_veh", "phase"]
