# RL Training Reward Diagnostic

**First written**: 2026-08-14 (conclusion: "not a bug")
**Corrected**: 2026-08-14, later the same day — **the original conclusion was wrong**
**Status**: ✅ root cause found and fixed; both checkpoints retrained

---

## Symptom

Every `rollout/ep_rew_mean` point in the 2026-08-13 DQN and PPO runs is **exactly**
`0.0` (69 and 98 TensorBoard points respectively, see `train_dqn_full.log`,
`train_ppo_full.log`).

## The first (wrong) conclusion

The first pass explained this as diff-waiting-time telescoping: sumo-rl's reward is
`r_t = -(W_t - W_{t-1})`, so the episode return collapses to `W_start - W_end`, and
"by episode end most vehicles have exited, so the return is ≈ 0". It recommended **no
code change** and only re-plotted `train/loss` instead of the reward.

That reasoning stops one question short: telescoping explains a *small* return, not an
*exactly zero* one, on 167 consecutive points. `W_end == W_start == 0` means the
intersection was empty at both ends of every episode — that is a demand bug, not a
reward property.

## Actual root cause

The training route file supplied demand for roughly the **first 25 seconds** of each
1800 s episode. Two independent defects in `simulation/route_generator._flow_bins`,
both triggered by a vision-derived TrafficState:

1. **Profile bins were held past the observed window.** `demo_001.json` is a 21.52 s
   clip with `profile_bins_sec=5`, i.e. 5 bins per approach. The old loop ran
   `while t < duration` and held the *last* bin value for the rest of the episode. The
   last bin of both observed approaches is `0.0` vph (north `[…, 720, 0, 0]`, south
   `[…, 720, 1440, 0]`), so north/south demand was frozen at zero from t=25 s to
   t=1800 s. Fallback should have been the aggregate `flow_vph` (north 1003.7,
   south 1171.0), not whatever value the video happened to end on.
2. **Flows were emitted grouped by approach, so `begin` was not monotonic.** Order was
   north(0,5,10,15,20) → south(0,5,10,15,20) → east(0) → west(0). SUMO requires route
   input sorted by departure time; it *warns* and then **silently discards** every
   out-of-order element. With `sumo_warnings=False` in `make_env` there was no visible
   symptom. Effectively only north's first five bins survived: south, east and west
   were dropped entirely.

Net effect: an agent trained for 100k steps on an almost-empty intersection.
`W_t = 0` at every step ⇒ `r_t = 0` ⇒ `ep_rew_mean = 0.0` exactly. The same route
defect measured on the arena side gave 40 vehicles of throughput instead of ~1450.

## Fix

- `simulation/route_generator.py`: profile bins are clamped to the state's own
  `duration_sec`, the remainder of the episode gets one aggregate `flow_vph` bin, all
  flows are collected and **emitted sorted by `begin`**, and `--duration` lets the
  consumer's horizon override the state's.
- `simulation/route_generator.check_sorted()` / `route_horizon()`: raise instead of
  losing demand silently. Called from `optimization.rl_agents.make_env` (every
  episode) and from `train_common.train` (which also refuses to start when the route
  file's horizon is shorter than `--episode-sec`).
- `optimization/rl_agents.make_env`: `sumo_warnings` back to `True` by default;
  training passes `False` explicitly only after the route file has been checked.

Training routes are now 84 flows, sorted, horizon 1800 s, all four approaches present
(north 36 / south 30 / east 9 / west 9 flow elements).

## Evidence after the fix

`train_dqn_fixed.log` (100k steps, same hyperparameters, same seed):

| timesteps | 1.4k | 11.5k | 21.6k | 31.7k | 41.8k | 51.8k | 61.9k |
|---|---|---|---|---|---|---|---|
| `ep_rew_mean` | -21.5 | -18.1 | -16.1 | -14.4 | -12.3 | -10.9 | -10.4 |

The reward is now non-zero and improving monotonically — the agent is reducing the
waiting time still standing at episode end.

## What remains true from the original analysis

Telescoping is real: the return is `W_start - W_end`, so it measures *residual*
waiting at the horizon, not total waiting over the episode. Read `ep_rew_mean` as
"how much queue is left at t=1800", and keep using tripinfo metrics
(`simulation.metrics.MetricsCollector`) for the actual strategy comparison — those are
computed identically for baselines and RL, so they are the comparable numbers.

`scripts/plot_training_curves.py` now plots reward **and** loss, and only the newest
SB3 run directory per algorithm (`DQN_1`, `DQN_2`, … would otherwise be spliced
together at the same step numbers).

## Invalidated by this bug

- `models/dqn_cross_basic.zip`, `models/ppo_cross_basic.zip` (2026-08-13) — retrained.
- Every run under `data/results/experiments/` from before the fix, plus
  `data/results/arena_summary.csv` and the arena figures — regenerated. Runs now carry
  a `run_meta.json` input fingerprint so a stale directory can no longer be reused
  (see `experiments/scenario_runner.run_fingerprint`).

## Reproduce

```bash
# a route file that would be truncated by SUMO now fails loudly
.venv\Scripts\python.exe -c "from simulation.route_generator import check_sorted; check_sorted('data/simulations/train_cross_basic/routes.rou.xml')"

# demand horizon of a route file (must be >= --episode-sec)
.venv\Scripts\python.exe -c "from simulation.route_generator import route_horizon; print(route_horizon('data/simulations/train_cross_basic/routes.rou.xml'))"

# reward trajectory of the latest run
grep -o "ep_rew_mean *| *[-0-9.]*" train_dqn_fixed.log | tail -5

.venv\Scripts\python.exe -m pytest tests/test_route_generator.py -q
```

---

**Conclusion**: `ep_rew_mean = 0.0` was a demand bug, not a property of
diff-waiting-time. An exactly-zero metric deserves more suspicion than a noisy one.
