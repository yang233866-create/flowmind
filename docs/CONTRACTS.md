# FlowMind 模块间契约（CONTRACTS）

所有模块必须严格遵守本文件。修改契约需先更新本文件再改代码。

## 目录所有权

| 目录 | 内容 | 所有者 |
|---|---|---|
| `vision/` | 视频分析 → TrafficState | Vision 模块 |
| `simulation/` | SUMO 模板、路由生成、仿真运行、指标 | Sim 模块 |
| `optimization/` | Fixed/Actuated/DQN/PPO 控制器与训练 | Opt 模块 |
| `experiments/` | 场景定义、批量实验、对比 | Experiments 模块 |
| `app/` | Streamlit 界面 | App 模块 |
| `scripts/` | 公共工具（sci_style 等） | 共享（只读，勿改 sci_style.py） |
| `data/`, `models/`, `figures/` | 运行产物 | 各模块按下述布局写入 |

## 运行环境

- Python: `.venv/Scripts/python.exe`（3.12，uv 管理）
- 所有 CLI 从仓库根目录运行：`.venv/Scripts/python.exe -m vision.analyze ...`
- 每个包目录必须有 `__init__.py`
- SUMO 通过 pip 包 `eclipse-sumo` 提供，二进制在 `site-packages/sumo/bin`。
  统一调用 `simulation.sumo_home.ensure_sumo_home()`（返回 SUMO_HOME 路径，
  设置 `os.environ["SUMO_HOME"]` 并把 bin 加入 PATH）。若环境变量
  `FLOWMIND_LIBSUMO=1` 且 libsumo 可用，用 libsumo 替代 traci。

## TrafficState JSON（schema 1.1）— 系统唯一跨模块数据接口

写入 `data/traffic_states/<scenario_id>.json`：

```json
{
  "schema_version": "1.1",
  "scenario_id": "demo_001",
  "source": {"video": "data/videos/demo.mp4", "fps": 25.0, "frames": 7500,
              "duration_sec": 300.0, "analyzed_at": "2026-08-13T12:00:00"},
  "duration_sec": 300.0,
  "approaches": {
    "north": {"flow_vph": 820.0, "queue_est": 11.0,
               "vehicle_mix": {"car": 0.82, "bus": 0.05, "truck": 0.10, "motorcycle": 0.03},
               "observed": true},
    "south": {"...": "同上"}, "east": {}, "west": {}
  },
  "turning_ratio": {
    "north": {"left": 0.15, "straight": 0.70, "right": 0.15},
    "south": {}, "east": {}, "west": {}
  },
  "profile_bins_sec": 300,
  "flow_profile": {"north": [800, 900], "south": [], "east": [], "west": []}
}
```

规则：
- 四个方向键固定为 `north/south/east/west`（方向指"车流来自哪个方位的进口道"）。
- 视频未覆盖的方向：`observed: false`，`flow_vph` 填对侧观测值或 400.0 默认值。
- `queue_est`、`flow_profile` 可为 `null`。`turning_ratio` 缺省 0.15/0.70/0.15。
- `vehicle_mix` 四类之和为 1；未识别类别并入 car。

## 场景规格 Scenario Spec JSON

```json
{
  "name": "morning_peak",
  "base_state": "data/traffic_states/demo_001.json",
  "flow_multipliers": {"north": 1.6, "south": 1.6, "east": 1.0, "west": 1.0},
  "lane_closures": [{"approach": "east", "n_lanes": 1}],
  "duration_sec": 1800
}
```

## SUMO 模板（simulation/templates/<name>/）

三个模板：`cross_basic`（十字，每进口 2 车道）、`cross_leftturn`（十字，
专用左转道 + 保护左转相位）、`arterial_minor`（主干 3 车道 × 次干 1 车道）。

每个模板目录包含：
- 平面定义 XML（`.nod.xml/.edg.xml/.con.xml/.tll.xml`）与 `build.py`（调 netconvert）
- 预构建的 `net.net.xml`（提交入库，运行时不重建）
- `meta.json`：

```json
{
  "name": "cross_basic",
  "tls_id": "TL",
  "approaches": {
    "north": {"in_edge": "N_in", "out_edge": "N_out", "n_lanes": 2},
    "south": {}, "east": {}, "west": {}
  },
  "movements": {
    "north": {"left": "E_out", "straight": "S_out", "right": "W_out"},
    "south": {}, "east": {}, "west": {}
  },
  "programs": {"static": "0", "actuated": "act"}
}
```

信号程序：net 内置静态程序 programID="0"；同一 tls 附加 actuated 程序
programID="act"（additional 文件 `tls_act.add.xml`）。相位时长默认：
绿 min 5 / max 60，黄 3。

## 仿真与指标（simulation/）

```python
from simulation.sumo_runner import run_episode
result = run_episode(template="cross_basic", route_file=..., strategy="fixed",
                     seed=0, out_dir=Path(...), gui=False,
                     duration_sec=1800, lane_closures=[...])
# -> RunResult(dataclass):
#    metrics: dict  (见下), timeseries: pandas.DataFrame, out_dir: Path
```

`strategy` ∈ `{"fixed", "actuated", "dqn", "ppo"}`。fixed/actuated 由
sumo_runner 直接跑；dqn/ppo 由 sumo_runner 调用
`optimization.rl_agents.RLPhaseController`（加载 `models/<strategy>_<template>.zip`）。

metrics 键（全部 float）：
- `avg_waiting_s`：tripinfo waitingTime 均值
- `avg_travel_time_s`：tripinfo duration 均值
- `throughput_veh`：完成行程车辆数
- `avg_queue_veh`：每秒对四进口道 halting 数(速度<0.1m/s)求和后取时间均值
- `max_queue_veh`：上述总排队的最大值
- `avg_speed_mps`：tripinfo routeLength/duration 的均值
- `teleports`：瞬移次数（拥堵指示）

timeseries.DataFrame 列：`t, queue_north, queue_south, queue_east, queue_west,
queue_total, waiting_total, running_veh, phase`（每仿真秒一行）。

## 结果目录布局

```
data/results/experiments/<exp_id>/     # exp_id = <scenario>__<strategy>__s<seed>
  config.json          # 场景+策略+seed+sim 参数
  metrics_summary.json # metrics dict
  run_meta.json        # 输入指纹（见下），跑成功后才写
  timeseries.csv       # 每仿真秒一行（基线与 RL 一致）
  tripinfo.xml
data/results/arena_summary.csv         # 所有实验汇总（一行一实验）
data/results/tb/                       # TensorBoard 日志
models/<strategy>_<template>.zip       # SB3 checkpoint
figures/                               # 论文图，PNG，DPI>=300
```

**`run_meta.json`（实验身份，`experiments.scenario_runner.run_fingerprint`）**：
`base_state` + `base_state_sha1`、`flow_multipliers`、`lane_closures`、`duration_sec`、
`template`、`strategy`、`seed`、`model_sha1`（RL 策略才有）、`route_sha1`，以及它们的
哈希 `key`。目录名只编码 `(scenario, strategy, seed)`，**不足以标识一次运行**：缓存命中
必须比对 `key`，不一致就重跑。

`arena_summary.csv` 除指标列外必须携带 provenance 列
`base_state, base_state_sha1, model_sha1, run_key`；缺 `run_meta.json` 的运行默认不进表
（`strategy_compare --lax` 才保留），不同 `base_state_sha1` 的运行不可同表比较。

**路由文件约定**：`<flow>`/`<vehicle>` 必须按 `begin`/`depart` 升序排列，且需求要覆盖到
episode 结束。SUMO 对乱序元素只警告一次然后静默丢弃，对提前断供无任何提示 ——
两者都会让运行"成功"但结果无效。生产方是 `simulation.route_generator`，
消费方（`make_env` / `train_common`）用 `check_sorted()` 与 `route_horizon()` 显式校验。

## RL 约定（optimization/）

- 环境：sumo-rl `SumoEnvironment(single_agent=True)`，官方默认 observation
  与 reward（diff-waiting-time），`delta_time=5, yellow_time=3, min_green=5`。
- 训练 CLI：`python -m optimization.train_dqn --template cross_basic
  --route <rou.xml> --timesteps 100000`；PPO 同理。checkpoint 存
  `models/dqn_cross_basic.zip`；TensorBoard 到 `data/results/tb/`。
- 评估时 RL 用与训练一致的 obs 计算（通过 sumo-rl env 跑评估回合），
  但指标采集必须与 fixed/actuated 相同（MetricsCollector + tripinfo），
  且**采样频率也必须相同**：`env.step()` 一次推进 `delta_time=5` 秒，所以要挂
  `env._sumo_step` 每仿真秒采一次，否则 `max_queue_veh` 偏低、teleports 漏计。
- 训练前必须校验路由文件：有序（`check_sorted`）且 `route_horizon >= --episode-sec`。
  在空路口上训练时 diff-waiting-time 奖励恒为 0，日志看不出任何异常。

## 图表

- 一律 `from scripts.sci_style import apply_style, NPG, STRATEGY_COLORS, save_fig`
- PNG，DPI≥300，Times New Roman，英文标签（论文用）
- 只出 PNG，不出 PDF

## CLI 入口清单

- `python -m vision.analyze --video ... --roi-config ... --state-out ... [--annotated-video ...] [--figures-dir figures] [--max-frames N]`
- `python -m simulation.route_generator --state <json> --template cross_basic --out <dir> [--scenario <spec.json>] --seed 0`
- `python -m experiments.scenario_runner --scenario <spec.json|name> --strategy fixed --seed 0`
- `python -m experiments.strategy_compare --template cross_basic --scenarios all --strategies fixed,actuated,dqn,ppo --seeds 3`
- `python -m optimization.train_dqn / train_ppo ...`
- `streamlit run app/Home.py`
