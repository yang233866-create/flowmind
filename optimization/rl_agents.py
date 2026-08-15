"""RL environment construction and evaluation for FlowMind.

Training and evaluation both go through sumo-rl's SumoEnvironment so the
agent always sees the observation space it was trained on. Metrics for
evaluation episodes come from the same tripinfo/MetricsCollector pipeline
as the fixed/actuated baselines (see docs/CONTRACTS.md).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from simulation.route_generator import check_sorted
from simulation.sumo_home import ensure_sumo_home

ensure_sumo_home()  # must run before importing sumo_rl

TEMPLATES_ROOT = Path("simulation/templates")
MODELS_ROOT = Path("models")

# sumo-rl agent timing (contract: delta_time=5, yellow_time=3, min_green=5)
ENV_KWARGS = dict(delta_time=5, yellow_time=3, min_green=5, max_green=60)

# Default reward used for training/evaluation when none is requested. Matches
# sumo-rl's default ("diff-waiting-time"); see DEFAULT_REWARD_FN below.
DEFAULT_REWARD_FN = "diff-waiting-time"


def register_reward_fn():
    """Register FlowMind custom reward functions against sumo-rl's registry.

    Called once per process before any SuoEnvironment is built (training or
    evaluation). Rewards registered here can be selected by name via
    `make_env(..., reward_fn="<name>")`.

    speed-weighted-queue: penalizes queue length MORE harshly when vehicles are
    crawling. The default diff-waiting-time reward only reacts to vehicles that
    are stopped, so the agent learns to keep traffic trickling slowly (low queue
    + low waiting) while total travel time actually worsens. Weighting the queue
    penalty by target/speed forces the agent to clear the queue at speed.
    """
    from sumo_rl.environment.traffic_signal import TrafficSignal

    SPEED_TARGET = 13.89  # m/s = cross_basic straight-through speed limit (50 km/h)
    V_FLOOR = 2.0         # m/s; caps the multiplier so crawling can't blow up /0

    def speed_weighted_queue(ts) -> float:
        # Penalize queue length MORE harshly when the vehicles that ARE present
        # are crawling. The default diff-waiting-time reward only reacts to
        # stopped vehicles, so an agent learns to trickle traffic (low waiting)
        # while total travel time worsens. Weight the queue penalty by the ratio
        # target/avg_speed so clearing slowly is penalized, clearing at speed
        # is rewarded with a smaller penalty.
        #
        # NOTE: ts.get_average_speed() is *normalized* (speed / per-veh limit)
        # and returns 1.0 when empty, so it can't distinguish crawling — compute
        # absolute speed over the controlled lanes directly instead.
        SPEED_TARGET = 13.89  # m/s = cross_basic straight-through limit (50 km/h)
        V_FLOOR = 2.0         # m/s; cap the multiplier so a stopped queue is bounded

        vehs = ts._get_veh_list()
        if vehs:
            speeds = [ts.sumo.vehicle.getSpeed(v) for v in vehs]
            avg = sum(speeds) / len(speeds)
        else:
            avg = SPEED_TARGET  # nothing queued -> no crawling to penalize
        speed = max(avg, V_FLOOR)
        queue = int(ts.get_total_queued())
        return -queue * (SPEED_TARGET / speed)

    speed_weighted_queue.__name__ = "speed-weighted-queue"  # CLI/docs name
    if "speed-weighted-queue" not in TrafficSignal.reward_fns:
        TrafficSignal.register_reward_fn(speed_weighted_queue)


def template_paths(template: str) -> tuple[Path, dict]:
    tdir = TEMPLATES_ROOT / template
    meta = json.loads((tdir / "meta.json").read_text(encoding="utf-8"))
    return tdir / "net.net.xml", meta


def model_path(strategy: str, template: str, reward_fn: str | None = None) -> Path:
    if reward_fn:
        slug = reward_fn.replace("-", "_")
        return MODELS_ROOT / f"{strategy}_{template}__{slug}.zip"
    return MODELS_ROOT / f"{strategy}_{template}.zip"


def make_env(
    template: str,
    route_file: str | Path,
    duration_sec: int = 1800,
    seed: int = 0,
    out_csv_name: str | None = None,
    tripinfo_out: str | Path | None = None,
    use_libsumo: bool = False,
    sumo_warnings: bool = True,
    reward_fn: str | None = None,
):
    """Create a single-agent sumo-rl environment on a FlowMind template.

    `sumo_warnings` defaults to True on purpose: with warnings off, SUMO silently
    discards an unsorted route file and the episode runs on a fraction of the
    intended demand with no visible symptom. Only turn it off for long training
    runs, and only once the route file has passed `check_sorted`.
    """
    if use_libsumo:
        os.environ["LIBSUMO_AS_TRACI"] = "1"
    else:
        os.environ.pop("LIBSUMO_AS_TRACI", None)

    # sumo_rl reads LIBSUMO at import time -> import after env var is set
    import importlib

    import sumo_rl.environment.env as sumo_rl_env

    if sumo_rl_env.LIBSUMO != use_libsumo:
        importlib.reload(sumo_rl_env)

    check_sorted(route_file)  # fail loudly instead of losing demand silently

    # Resolve a named reward to its callable before handing it to sumo-rl. sumo-rl
    # accepts str|callable|dict, but the str form does a registry lookup inside
    # TrafficSignal.__init__ that can miss custom rewards registered in-process
    # (especially under libsumo). Passing the callable directly is robust.
    reward = reward_fn if reward_fn is not None else DEFAULT_REWARD_FN
    if isinstance(reward, str):
        register_reward_fn()  # import-time no-op if already present
        from sumo_rl.environment.traffic_signal import TrafficSignal

        if reward not in TrafficSignal.reward_fns:
            raise KeyError(
                f"Unknown reward '{reward}'. Available: "
                f"{sorted(TrafficSignal.reward_fns)}"
            )
        reward = TrafficSignal.reward_fns[reward]

    net_file, _ = template_paths(template)
    additional = "--no-step-log --duration-log.disable"
    if tripinfo_out is not None:
        additional += f" --tripinfo-output {tripinfo_out}"

    return sumo_rl_env.SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        single_agent=True,
        num_seconds=duration_sec,
        sumo_seed=seed,
        out_csv_name=out_csv_name,
        time_to_teleport=300,
        additional_sumo_cmd=additional,
        sumo_warnings=sumo_warnings,
        reward_fn=reward,
        **ENV_KWARGS,
    )


def load_model(strategy: str, template: str, reward_fn: str | None = None):
    """Load a trained SB3 model; raises with a clear message if missing."""
    from stable_baselines3 import DQN, PPO

    path = model_path(strategy, template, reward_fn=reward_fn)
    if not path.exists():
        which = f" (reward {reward_fn})" if reward_fn else ""
        raise RuntimeError(
            f"No trained {strategy.upper()} checkpoint{which} for template '{template}' "
            f"(expected {path}). Train first: python -m optimization.train_{strategy} "
            f"--template {template} --route <routes.rou.xml>"
            + (f" --reward-fn {reward_fn}" if reward_fn else "")
        )
    cls = {"dqn": DQN, "ppo": PPO}[strategy]
    return cls.load(str(path), device="cpu")  # inference is tiny; avoid GPU init cost


def run_rl_episode(
    template: str,
    route_file: str | Path,
    strategy: str,
    seed: int,
    out_dir: Path,
    duration_sec: int = 1800,
    lane_closures: list[dict] | None = None,
    gui: bool = False,
    reward_fn: str | None = None,
):
    """Evaluate a trained agent for one episode; returns (metrics, timeseries_df).

    Timeseries/metrics are collected exactly like the baselines: per-second
    halting counts on approach in-edges + tripinfo aggregation.

    `reward_fn` selects which checkpoint to load (the one trained under that
    reward). It does NOT change how metrics are computed -- evaluation always
    uses tripinfo, independent of the reward the agent was trained to optimize.
    """
    import pandas as pd

    from simulation.metrics import MetricsCollector

    _, meta = template_paths(template)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo = out_dir / "tripinfo.xml"

    model = load_model(strategy, template, reward_fn=reward_fn)
    env = make_env(
        template, route_file, duration_sec=duration_sec, seed=seed,
        tripinfo_out=tripinfo, use_libsumo=False,  # traci for evaluation stability
    )
    if gui:
        env.use_gui = True

    obs, _ = env.reset()
    sumo = env.sumo  # traci-compatible handle
    if lane_closures:
        for closure in lane_closures:
            in_edge = meta["approaches"][closure["approach"]]["in_edge"]
            n_total = meta["approaches"][closure["approach"]]["n_lanes"]
            for li in range(min(int(closure["n_lanes"]), n_total)):
                sumo.lane.setAllowed(f"{in_edge}_{li}", ["authority"])

    collector = MetricsCollector(sumo, meta)
    # Baselines sample queues/teleports once per simulated second. env.step()
    # advances delta_time (5 s) at a time, so sampling per agent step would give
    # a 5x coarser series -- max_queue_veh biased low and teleports undercounted
    # (getStartingTeleportNumber only reports the *current* step). Hook the
    # per-second step instead so both pipelines produce comparable timeseries.
    hooked = False
    if hasattr(env, "_sumo_step"):
        _orig_sumo_step = env._sumo_step

        def _sumo_step_with_metrics() -> None:
            _orig_sumo_step()
            collector.step(int(env.sim_step))

        env._sumo_step = _sumo_step_with_metrics
        hooked = True

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        if not hooked:  # fallback: one sample per agent step
            collector.step(int(env.sim_step))
    env.close()

    ts = collector.to_dataframe()
    metrics = collector.parse_tripinfo(tripinfo)  # includes teleport count
    ts.to_csv(out_dir / "timeseries.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics, ts
