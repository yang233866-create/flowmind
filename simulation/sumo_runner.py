"""Run one SUMO episode with a given signal-control strategy.

CLI:
    python -m simulation.sumo_runner --template cross_basic --route <rou.xml> \
        --strategy fixed --seed 0 --out <dir> --duration 600
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from simulation.metrics import MetricsCollector
from simulation.route_generator import check_sorted, load_meta
from simulation.sumo_home import ensure_sumo_home, sumo_binary, use_libsumo

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STRATEGIES = ("fixed", "actuated", "dqn", "ppo")


@dataclass
class RunResult:
    metrics: dict
    timeseries: pd.DataFrame
    out_dir: Path


def _write_sumocfg(out_dir: Path, net: Path, route_file: Path, tripinfo: Path,
                   additionals: list[Path], seed: int, duration_sec: int) -> Path:
    add = ""
    if additionals:
        joined = ",".join(str(p) for p in additionals)
        add = f'        <additional-files value="{joined}"/>\n'
    cfg = (
        '<configuration>\n'
        '    <input>\n'
        f'        <net-file value="{net}"/>\n'
        f'        <route-files value="{route_file}"/>\n'
        f'{add}'
        '    </input>\n'
        '    <time>\n'
        '        <begin value="0"/>\n'
        f'        <end value="{duration_sec}"/>\n'
        '    </time>\n'
        '    <processing>\n'
        '        <time-to-teleport value="300"/>\n'
        '    </processing>\n'
        '    <random_number>\n'
        f'        <seed value="{seed}"/>\n'
        '    </random_number>\n'
        '    <output>\n'
        f'        <tripinfo-output value="{tripinfo}"/>\n'
        '    </output>\n'
        '    <report>\n'
        '        <no-step-log value="true"/>\n'
        '    </report>\n'
        '</configuration>\n'
    )
    cfg_path = out_dir / "episode.sumocfg"
    cfg_path.write_text(cfg, encoding="utf-8")
    return cfg_path


def _apply_lane_closures(traci_mod, meta: dict, lane_closures: list[dict]) -> list[str]:
    closed = []
    for closure in lane_closures or []:
        in_edge = meta["approaches"][closure["approach"]]["in_edge"]
        for i in range(int(closure["n_lanes"])):
            lane_id = f"{in_edge}_{i}"
            traci_mod.lane.setAllowed(lane_id, ["authority"])
            closed.append(lane_id)
    return closed


def run_episode(template: str, route_file: str | Path, strategy: str, seed: int,
                out_dir: str | Path, gui: bool = False, duration_sec: int = 1800,
                lane_closures: list[dict] | None = None) -> RunResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got '{strategy}'")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_file = Path(route_file).resolve()
    # SUMO drops out-of-order route elements with a warning only; refuse to
    # produce metrics that would silently describe a fraction of the demand.
    check_sorted(route_file)

    if strategy in ("dqn", "ppo"):
        try:
            from optimization.rl_agents import run_rl_episode
        except ImportError as exc:
            raise RuntimeError(
                f"strategy '{strategy}' requires optimization.rl_agents.run_rl_episode "
                f"(model models/{strategy}_{template}.zip must be trained first): {exc}"
            ) from exc
        metrics, timeseries = run_rl_episode(
            template=template, route_file=route_file, strategy=strategy,
            seed=seed, out_dir=out_dir, duration_sec=duration_sec,
            lane_closures=lane_closures, gui=gui)
        (out_dir / "config.json").write_text(json.dumps({
            "template": template, "route_file": str(route_file),
            "strategy": strategy, "seed": seed, "duration_sec": duration_sec,
            "lane_closures": lane_closures or [], "gui": gui,
        }, indent=2), encoding="utf-8")
        return RunResult(metrics=metrics, timeseries=timeseries, out_dir=out_dir)

    ensure_sumo_home()
    if use_libsumo():
        import libsumo as traci_mod
    else:
        import traci as traci_mod

    tpl_dir = TEMPLATES_DIR / template
    meta = load_meta(template)
    tls_id = meta["tls_id"]
    net = (tpl_dir / "net.net.xml").resolve()
    tripinfo = (out_dir / "tripinfo.xml").resolve()

    additionals = []
    if strategy == "actuated":
        additionals.append((tpl_dir / "tls_act.add.xml").resolve())

    cfg = _write_sumocfg(out_dir, net, route_file, tripinfo,
                         additionals, seed, duration_sec)

    in_edges = {a: meta["approaches"][a]["in_edge"]
                for a in ("north", "south", "east", "west")}

    traci_mod.start([sumo_binary(gui=gui), "-c", str(cfg)])
    try:
        program_id = meta["programs"]["static" if strategy == "fixed" else "actuated"]
        traci_mod.trafficlight.setProgram(tls_id, program_id)
        closed_lanes = _apply_lane_closures(traci_mod, meta, lane_closures)

        collector = MetricsCollector(traci_mod, in_edges, tls_id)
        for t in range(duration_sec):
            traci_mod.simulationStep()
            collector.step(t + 1)
    finally:
        traci_mod.close()

    metrics = collector.parse_tripinfo(tripinfo)
    timeseries = collector.to_dataframe()

    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    timeseries.to_csv(out_dir / "timeseries.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({
        "template": template,
        "route_file": str(route_file),
        "strategy": strategy,
        "seed": seed,
        "duration_sec": duration_sec,
        "lane_closures": lane_closures or [],
        "closed_lanes": closed_lanes,
        "gui": gui,
    }, indent=2), encoding="utf-8")

    return RunResult(metrics=metrics, timeseries=timeseries, out_dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one SUMO episode")
    ap.add_argument("--template", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--strategy", default="fixed", choices=STRATEGIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=int, default=1800)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    result = run_episode(args.template, args.route, args.strategy, args.seed,
                         args.out, gui=args.gui, duration_sec=args.duration)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
