"""app 模块的仿真运行入口（薄封装）。

CONTRACTS.md 只定义了 simulation.sumo_runner.run_episode 的 Python API，
没有定义其命令行参数；app 页面通过 subprocess 调用本脚本来触发单次仿真，
保证界面进程不 import 仿真代码，同时可流式读取输出。

用法：
    python -m app.run_episode_cli --template cross_basic --route <rou.xml>
        --strategy fixed --seed 0 --out <dir> --duration-sec 1800 [--scenario name]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Run a single SUMO episode via simulation.sumo_runner")
    p.add_argument("--template", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--strategy", required=True, choices=["fixed", "actuated", "dqn", "ppo"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--duration-sec", type=int, default=1800)
    p.add_argument("--scenario", default=None, help="scenario name recorded in config.json")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[app] run_episode template={a.template} strategy={a.strategy} "
          f"seed={a.seed} duration={a.duration_sec}s", flush=True)
    print(f"[app] route={a.route}", flush=True)
    print(f"[app] out={out}", flush=True)

    from simulation.sumo_runner import run_episode  # noqa: PLC0415 子进程内才 import

    result = run_episode(
        template=a.template,
        route_file=Path(a.route),
        strategy=a.strategy,
        seed=a.seed,
        out_dir=out,
        gui=False,
        duration_sec=a.duration_sec,
    )

    metrics = dict(result.metrics)

    # 按结果目录契约补齐产物（sumo_runner 已写则不覆盖）
    mpath = out / "metrics_summary.json"
    if not mpath.exists():
        mpath.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    tpath = out / "timeseries.csv"
    ts = getattr(result, "timeseries", None)
    if not tpath.exists() and ts is not None:
        ts.to_csv(tpath, index=False)
    cpath = out / "config.json"
    if not cpath.exists():
        cpath.write_text(json.dumps({
            "scenario": a.scenario or out.name,
            "strategy": a.strategy,
            "seed": a.seed,
            "template": a.template,
            "route_file": str(a.route),
            "duration_sec": a.duration_sec,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[app] metrics: " + json.dumps(metrics, ensure_ascii=False), flush=True)
    print("[app] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
