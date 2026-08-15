"""Ranking guard: a strategy mean is only meaningful on a balanced panel.

The Home page and the Arena page both used to average `avg_waiting_s` over every
row of arena_summary.csv, so a strategy that had only been run on the easy
scenarios won by construction.
"""
from __future__ import annotations

import pandas as pd

from app.common import comparable_panel

STRATEGIES = ("fixed", "actuated", "dqn", "ppo")


def _arena(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["scenario", "strategy", "seed",
                                       "avg_waiting_s", "base_state_sha1"])


def _full_panel(scenarios=("normal", "event_surge"), seeds=(0, 1), sha="aaa"):
    rows = []
    for sc in scenarios:
        for st in STRATEGIES:
            for sd in seeds:
                wait = 40.0 if sc == "normal" else 90.0
                rows.append((sc, st, sd, wait, sha))
    return _arena(rows)


def test_balanced_panel_is_returned_untouched():
    df = _full_panel()
    out, notes = comparable_panel(df)
    assert len(out) == len(df)
    assert notes == []


def test_cells_missing_a_strategy_are_dropped():
    df = _full_panel()
    # dqn never ran the hard scenario -- exactly the case that faked a winner
    df = df[~((df["scenario"] == "event_surge") & (df["strategy"] == "dqn"))]
    out, notes = comparable_panel(df)
    assert set(out["scenario"]) == {"normal"}
    assert len(notes) == 1 and "剔除" in notes[0]
    means = out.groupby("strategy")["avg_waiting_s"].mean()
    assert means.nunique() == 1  # nobody wins on the easy scenario alone


def test_unbalanced_ranking_would_have_picked_the_wrong_winner():
    df = _full_panel()
    df = df[~((df["scenario"] == "event_surge") & (df["strategy"] == "dqn"))]
    naive = df.groupby("strategy")["avg_waiting_s"].mean().idxmin()
    assert naive == "dqn"  # the bug
    out, _ = comparable_panel(df)
    fair = out.groupby("strategy")["avg_waiting_s"].mean()
    assert fair.nunique() == 1 and set(fair.index) == set(STRATEGIES)


def test_mixed_base_states_keep_only_the_dominant_one():
    good = _full_panel(sha="new")
    stale = _full_panel(scenarios=("normal",), seeds=(0,), sha="old")
    out, notes = comparable_panel(pd.concat([good, stale], ignore_index=True))
    assert set(out["base_state_sha1"]) == {"new"}
    assert any("TrafficState" in n for n in notes)


def test_no_common_cell_yields_an_empty_frame():
    df = _arena([("normal", "fixed", 0, 40.0, "aaa"),
                 ("event_surge", "dqn", 0, 10.0, "aaa")])
    out, notes = comparable_panel(df)
    assert out.empty
    assert notes


def test_missing_columns_are_tolerated():
    df = pd.DataFrame({"strategy": ["fixed", "dqn"], "avg_waiting_s": [40.0, 30.0]})
    out, notes = comparable_panel(df)
    assert len(out) == 2 and notes == []
    empty = pd.DataFrame()
    assert comparable_panel(empty)[0].empty
