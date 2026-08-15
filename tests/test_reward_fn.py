"""Custom reward-function invariants.

The default diff-waiting-time reward only reacts to stopped vehicles, so an agent
can learn to keep traffic trickling slowly (low queue + low waiting) while total
travel time worsens. `speed-weighted-queue` must instead penalize slow clearing.
"""
from __future__ import annotations

import pytest

from simulation.sumo_home import ensure_sumo_home

ensure_sumo_home()

from optimization.rl_agents import register_reward_fn
from sumo_rl.environment.traffic_signal import TrafficSignal

register_reward_fn()


def _make_ts(**overrides):
    """A minimal stand-in for TrafficSignal carrying only what the reward reads.

    The reward now reads absolute speed via _get_veh_list() + sumo.vehicle.getSpeed,
    so the fake mirrors that interface rather than the normalized get_average_speed.
    """
    speed = overrides.get("speed", 13.89)  # m/s absolute
    queue = overrides.get("queue", 10)
    speed = [speed] if speed > 0 else []

    class _FakeSumo:
        class vehicle:
            @staticmethod
            def getSpeed(v):
                return v

    ts = type("FakeTS", (), {
        "_get_veh_list": lambda self: speed,
        "sumo": _FakeSumo(),
        "get_total_queued": lambda self: queue,
    })()
    return ts


def _fn():
    return TrafficSignal.reward_fns["speed-weighted-queue"]


def test_speed_weighted_queue_is_registered():
    assert "speed-weighted-queue" in TrafficSignal.reward_fns


def test_reward_is_negative_for_nonzero_queue():
    r = _fn()(_make_ts(queue=10, speed=13.89))
    assert r < 0


def test_empty_queue_gives_zero_reward():
    r = _fn()(_make_ts(queue=0, speed=13.89))
    assert r == 0.0


def test_slower_speed_magnifies_queue_penalty():
    # Same queue, slower clearing throughput -> larger magnitude penalty.
    r_fast = _fn()(_make_ts(queue=10, speed=13.89))
    r_slow = _fn()(_make_ts(queue=10, speed=3.0))
    assert r_slow < r_fast  # more negative when crawling


def test_stopped_queue_is_bounded_not_infinite():
    # speed near zero would divide by zero; the floor keeps the penalty finite.
    r = _fn()(_make_ts(queue=10, speed=0.1))
    assert r > -1e6 and r < 0
