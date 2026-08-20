from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import matplotlib as mpl

# Final delivery contract is explicit at the reproducible entry point.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    }
)
FORMAL_EXPORTS = ("figure.svg", "figure.pdf", "figure.png")
raster_dpi = 600
fig_width_mm = 183

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visualization.data_loader import load_visualization_data
from scripts.visualization.design_system import FIGURE_STEMS, write_manifest
from scripts.visualization.fig01_vision_to_twin import build as build_01
from scripts.visualization.fig02_strategy_tradeoffs import build as build_02
from scripts.visualization.fig03_scenario_robustness import build as build_03
from scripts.visualization.fig04_queue_dynamics import build as build_04
from scripts.visualization.fig05_training_evidence import build as build_05
from scripts.visualization.fig06_decision_map import build as build_06
from scripts.visualization.fig07_regret_landscape import build as build_07
from scripts.visualization.fig08_paired_transitions import build as build_08
from scripts.visualization.fig09_operating_state_density import build as build_09
from scripts.visualization.fig10_scenario_timeline_atlas import build as build_10
from scripts.visualization.fig11_perception_composition_flow import build as build_11
from scripts.visualization.statistics import paired_effects, per_run_queue_summary


BUILDERS = (
    build_01, build_02, build_03, build_04, build_05, build_06,
    build_07, build_08, build_09, build_10, build_11,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 FlowMind 高级科研可视化证据链")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    figure_dir = root / "outputs/figures"
    source_dir = root / "outputs/source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("读取并验证真实项目数据……", flush=True)
    data = load_visualization_data(root)
    print(f"  [OK] {len(data.arena)} 次正式实验，{len(data.timeseries)} 条秒级时序，DQN/PPO 100k 训练日志", flush=True)

    data.arena.to_csv(source_dir / "arena_summary_validated.csv", index=False)
    for index, (stem, builder) in enumerate(zip(FIGURE_STEMS, BUILDERS), start=1):
        print(f"[{index}/{len(BUILDERS)}] 生成 {stem}", flush=True)
        builder(data, figure_dir, source_dir)

    effects = paired_effects(data.arena)
    wait = effects.query("strategy == 'actuated' and metric == 'avg_waiting_s'").iloc[0]
    queue = per_run_queue_summary(data.timeseries, data.arena).groupby("strategy")["share_above_40_pct"].mean()
    manifest = write_manifest(figure_dir)

    elapsed = time.perf_counter() - started
    print("", flush=True)
    print("正式证据链生成完成", flush=True)
    print(f"  Actuated 平均等待：减少 {wait.benefit_mean:.1f} 秒（95% CI {wait.ci_low:.1f}–{wait.ci_high:.1f}）", flush=True)
    print(f"  严重拥堵时间占比：Fixed {queue['fixed']:.1f}% → DQN {queue['dqn']:.1f}% / PPO {queue['ppo']:.1f}%", flush=True)
    print(f"  图表目录：{figure_dir}", flush=True)
    print(f"  数据目录：{source_dir}", flush=True)
    print(f"  清单：{manifest}", flush=True)
    print(f"  用时：{elapsed:.1f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())