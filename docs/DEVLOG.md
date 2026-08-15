# FlowMind 开发日志

**项目**: FlowMind AI — 基于视觉感知与数字孪生的城市交通智能推演与优化平台  
**时间**: 2026-08-13 ~ 2026-08-14  
**目标**: 端到端实现 vision → simulation → optimization → experiments 全流程

---

## 项目架构

```
视频 (mp4)
    ↓ vision.analyze (YOLO11 + ByteTrack + ROI 计数)
TrafficState JSON (schema 1.1, 四向流量/车型/转向比)
    ↓ simulation.route_generator
SUMO .net.xml + .rou.xml
    ↓ experiments.scenario_runner (场景变体: 高峰/车道封闭/...)
    ↓ optimization.train_dqn / train_ppo
固定/感应/DQN/PPO 四策略评估
    ↓ experiments.strategy_compare
统一指标对比 (avg_waiting_s, throughput_veh, ...) + 论文图
```

---

## 模块实现顺序与关键决策

### 1. 契约优先 (`docs/CONTRACTS.md`)

在写任何代码前先定义 **TrafficState schema 1.1** 作为模块间唯一数据接口：
- 四向流量 `flow_vph`、车型分布 `vehicle_mix`、转向比例 `turning_ratio`
- 未观测方向 fallback 规则（南北互补、东西互补、默认 400 vph）
- `flow_profile` 时序分箱（5s bins）

所有模块必须严格遵守契约，修改契约需先更新文档再改代码。

### 2. 仿真基础 (`simulation/`)

- **三套 SUMO 模板**: `cross_basic` (2 车道十字)、`cross_leftturn` (专用左转)、`arterial_minor` (主干×次干)
- **路由生成器**: `route_generator.py` 读 TrafficState + 场景变体 (flow_multipliers, lane_closures) → SUMO `.rou.xml`
- **指标采集**: `MetricsCollector` 统一管线，tripinfo + 每秒排队数 → 7 项指标
- **sumo_home.py**: 自动检测 `eclipse-sumo` pip 包提供的 SUMO_HOME

### 3. 优化策略 (`optimization/`)

- **Fixed-Time**: SUMO 内置静态程序 (programID="0")
- **Actuated**: SUMO gap-based 感应控制 (programID="act")
- **DQN / PPO**: sumo-rl `SumoEnvironment` + Stable-Baselines3，奖励为 diff-waiting-time

**关键踩坑**: 训练日志 `ep_rew_mean` 恒为 **0.0**，最初被判定为 diff-waiting-time 奖励的"望远镜求和"正常现象。**这个判定是错的** —— 真实原因是路由文件只在 episode 前 25 s 供给需求，路口全程空转（详见 2026-08-14 修复记录与 `docs/rl_reward_diagnostic.md`）。修复后 `ep_rew_mean` 从 -21.5 稳定上升。评估仍然用 tripinfo 指标，不依赖奖励信号。

### 4. 实验框架 (`experiments/`)

- **场景库**: `scenarios.py` 定义 5 个场景（normal / morning_peak / evening_peak / lane_closure / event_surge）
- **scenario_runner.py**: 单实验执行器，输出到 `data/results/experiments/{scenario}__{strategy}__s{seed}/`
- **strategy_compare.py**: 批量运行 scenarios × strategies × seeds，汇总到 `arena_summary.csv`，产出 5 张论文图

### 5. 视觉感知 (`vision/`, 本次会话重点)

**五个文件**:
1. `detector.py`: YOLO11s 检测器，只保留 COCO 车辆类，CUDA 加速，三层权重下载 fallback (ultralytics → HF 镜像 → GitHub)
2. `tracker.py`: ByteTrack 多目标跟踪，`track_activation_threshold=0.25`
3. `counter.py`: 四向 ROI 过线计数，`sv.LineZone` + tracker_id 去重 + 时序分箱
4. `traffic_state.py`: 组装 schema 1.1，未观测方向自动补全
5. `analyze.py`: CLI 入口，产出 TrafficState JSON + 标注视频 + 4 张论文图

**ROI 格式** (与 Streamlit 界面对齐):
```json
{
  "count_lines": {
    "north": [[x1, y1], [x2, y2]],
    "south": [[x1, y1], [x2, y2]]
  }
}
```

**端到端验证**: supervision demo 视频 (21.5s, 3840×2160, 538 帧) → 计数 13 辆车 (north 6, south 7) → 外推流量 1004/1171 vph → schema 验证通过 → 4 张论文图 + 79 MB 标注视频。

**踩坑**:
- GitHub 下载超时 → HF 镜像兜底
- ByteTrack 在 supervision 0.30 已 deprecated → 保留现有代码，标注需升级
- 短视频窗口 (21.5s) 导致流量外推误差大 → 文档说明仅供功能验证，生产需 ≥5 min 视频

### 6. Streamlit 界面 (`app/`)

五个页面：
1. **Home**: 架构图 + 模块导航
2. **Traffic Vision**: 上传视频 → 配置 ROI → 运行分析 → 查看 TrafficState
3. **Traffic Twin**: 选择 TrafficState + 模板 → 生成路由 → 运行单次仿真 → 实时排队曲线
4. **Strategy Arena**: 批量实验配置 → 启动对比 → 查看汇总表 + 论文图
5. **Scenario Lab**: What-if 推演（高峰场景、车道封闭、突发流量）

### 7. 论文图工具 (`scripts/sci_style.py`)

- **必须复用**: `apply_style()` / `save_fig()` / `DIRECTION_COLORS` / `STRATEGY_COLORS`
- PNG ≥300 DPI, Times New Roman, NPG 配色, 英文标签
- 所有论文图从这里统一获取样式，确保一致性

---

## 2026-08-14 下午：结果可信度修复（重要）

上午跑完的 60 次 arena 对比看起来很漂亮（PPO 平均等待 25 s vs Fixed 59 s），复查时
发现 **PPO 的 throughput 只有 40 veh，而 Fixed 是 1457 veh** —— RL 策略是在一个几乎
空的路口上被评估的。三个独立缺陷叠加造成了这个假结果：

1. **路由文件未按发车时间排序**。`route_generator` 按进口道分组输出 `<flow>`，
   begin 序列是 north(0,5,10,15,20) → south(0,5,…) → east(0) → west(0)。SUMO 要求
   路由输入按发车时间有序，遇到乱序只**警告一次然后静默丢弃**该元素；而
   `make_env` 里写着 `sumo_warnings=False`，警告也看不到。实测同一批 flow：
   乱序 40 veh vs 排序后 455 veh。
2. **`flow_profile` 越界后沿用最后一个分箱**。`demo_001.json` 是 21.5 s 的视频，
   5 s 一箱共 5 箱，north/south 的最后一箱恰好是 `0.0` vph；旧代码把它一直"保持"
   到 1800 s，等于从第 25 秒起需求归零。正确做法是超出观测窗口后回落到该方向的
   聚合 `flow_vph`（north 1003.7 / south 1171.0）。
3. **实验缓存只按目录名 `(scenario, strategy, seed)` 命中**。`fixed`/`actuated` 行
   是 08-13 用 `synthetic_demo.json` 跑的，`dqn`/`ppo` 行是 08-14 用 `demo_001.json`
   跑的，两者需求完全不同却被拼进同一张对比表，`collect_summary()` 还会扫描目录下
   一切内容，包括 smoke 测试残留。

### 修复清单

| 文件 | 改动 |
|------|------|
| `simulation/route_generator.py` | flow 先收集再按 `begin` 排序输出；profile 裁剪到 `duration_sec`，其余用聚合 `flow_vph` 补齐；新增 `check_sorted()` / `route_horizon()` / `flow_begin_times()`；新增 `--duration` 覆盖 |
| `optimization/rl_agents.py` | `sumo_warnings` 默认恢复 `True`；建环境前 `check_sorted(route_file)`；RL 指标改挂 `env._sumo_step`，与基线一样**每仿真秒**采样一次（原先每 5 s 一次，`max_queue_veh` 偏低、teleports 漏计） |
| `optimization/train_common.py` | 训练前校验路由有序 + `route_horizon >= --episode-sec`，否则直接退出；`.train.json` 记录 `route_sha1` 与 `route_horizon_sec` |
| `experiments/scenario_runner.py` | 新增 `run_fingerprint()`：base_state/route/model 的 sha1 + 场景参数 → `run_meta.json`；指纹不一致的旧目录自动重跑；运行成功后才写指纹 |
| `experiments/strategy_compare.py` | 严格模式丢弃无 `run_meta.json`、跨 base_state 的运行（`--lax` 可关闭）；跳过 `smoke_*`；批次内有失败则退出码 1 |
| `scripts/plot_training_curves.py` | 只取每个算法**最新**的 SB3 run 目录（`DQN_1/DQN_2` 混在一起会把两次训练拼到同一 step 轴上）；重新绘制 reward + loss |
| `scripts/run_pipeline.py` | 训练路由与 base_state / episode 时长不匹配时自动重生成，不再复用陈旧文件 |
| `app/common.py`, `app/Home.py`, `app/pages/3_Strategy_Arena.py` | 新增 `comparable_panel()`：只在"每个策略都跑过"且同一 base_state 的子集上排名，并在界面上说明剔除了什么 |



### 回归测试套件（新增 `tests/`，71 个测试）

上面每一个缺陷都是"跑得通但结果是错的"，所以补测试的重点不是覆盖率而是
**把静默失败变成响亮失败**：

| 文件 | 守住的东西 |
|------|-----------|
| `test_route_generator.py` (17) | flow 必须按 begin 有序；`check_sorted` 能拒绝乱序文件；观测窗口外回落聚合流量而非 0；末尾残缺分箱不按整箱宽度重放；episode 结束时仍有需求 |
| `test_scenario_runner_cache.py` (7) | 换 base_state / 换模型 checkpoint / 无 `run_meta.json` 都必须重跑；`--force` 无条件重跑 |
| `test_strategy_compare_provenance.py` (8) | provenance 列进 CSV；无指纹与跨 base_state 的行默认丢弃、`--lax` 才保留；smoke 目录忽略；批次失败退出码非 0 |
| `test_app_ranking_guard.py` (6) | 面板不平衡时排名裁剪到可比子集（含一个"不裁剪就会选错冠军"的用例） |
| `test_sumo_smoke.py` (5, `-m sumo`) | 真跑 SUMO：车辆确实到达路口、timeseries 每秒一行、episode 末尾仍在放车、乱序路由被拒 |
| `test_metrics.py` (6) | tripinfo 解析键与契约一致；空 tripinfo 不除零；排队指标来自 timeseries；teleports 累加 |
| `test_counter.py` (8) | 过线计数去重、`sense` 语义、末尾残缺分箱按真实时长折算 |
| `test_traffic_state.py` (8) | 车型分布归一、未观测方向镜像/兜底、schema 校验 |

运行：`.venv\Scripts\python.exe -m pytest -q`（含 SUMO 用例）；`-m "not sumo"` 可跳过
需要 SUMO_HOME 的 5 个。

### 修复的影响面

- `models/dqn_cross_basic.zip` / `ppo_cross_basic.zip`（08-13 训练）作废，**两个模型
  全部重训 100k steps**。
- `data/results/experiments/` 下 69 个目录、`arena_summary.csv`、5 张 arena 图全部
  作废，**全量重跑 60 次实验**（旧目录缺 `run_meta.json`，指纹机制会自动判失效，
  不需要手工删除）。
- `docs/rl_reward_diagnostic.md` 的原结论（"`ep_rew_mean=0` 是正常现象，无需改代码"）
  已改写为正确的根因分析。

---

## 训练结果

| 模型 | 训练时长 | Checkpoint | 备注 |
|------|---------|-----------|------|
| DQN  | 73.7 min (4420.8 s) | `models/dqn_cross_basic.zip` (116 KB) | 100k steps, seed 0；`ep_rew_mean` -21.5 → -10.6 |
| PPO  | 83.7 min (5024.3 s) | `models/ppo_cross_basic.zip` (165 KB) | 100k steps, seed 0；`ep_rew_mean` 收敛到 -9.4 |

两者均为 2026-08-14 路由修复后重训（`route_sha1=2e29da431f758172`, horizon 1800 s）；训练参数见
`models/*.train.json`。

**训练曲线**: `figures/fig_rl_training_curves.png` — 每个算法一张子图，绘制 `rollout/ep_rew_mean` 与 loss（DQN 用 `train/loss`，PPO 用 `train/value_loss`）。路由需求修复后 `ep_rew_mean` 不再恒为 0，已重新纳入绘图；脚本只取每个算法最新的 SB3 run 目录，避免 `DQN_1`/`DQN_2` 在相同 step 上被拼接。

---

## Arena 结果（修复后，5 scenario × 4 strategy × 3 seed = 60 runs）

数据源 `data/results/arena_summary.csv`，全部 60 行同一基准状态
（`data/traffic_states/demo_001.json`, sha1 `73f107e7e250cdb7`），60 个 `run_key` 互不相同。
各策略跨场景/种子均值：

| 策略 | avg_waiting_s | avg_travel_time_s | throughput_veh | avg_queue_veh | max_queue_veh |
|------|---------------|-------------------|----------------|---------------|---------------|
| Fixed    | 56.2 | 102.9 | 1203 | 42.3 | 72.1 |
| Actuated | 46.1 |  86.8 | 1354 | 40.8 | 69.1 |
| DQN      | 31.2 | 102.8 | 1287 | 22.7 | 45.5 |
| PPO      | 30.9 | 104.5 | 1261 | 22.0 | 41.1 |

相对 Fixed 的等待时间改善：Actuated 18.0%、DQN 44.6%、PPO 45.0%。

按场景的 avg_waiting_s：

| scenario | Fixed | Actuated | DQN | PPO |
|----------|-------|----------|-----|-----|
| normal        | 52.9 | 40.5 | 27.0 | 26.9 |
| morning_peak  | 50.9 | 45.1 | 26.6 | 25.8 |
| evening_peak  | 56.9 | 58.3 | 33.4 | 32.9 |
| event_surge   | 60.7 | 54.0 | 39.7 | 40.1 |
| lane_closure  | 59.9 | 33.0 | 29.0 | 29.0 |

**需要在论文里如实说明的权衡**：RL 把等待时间和排队长度压下来（queue 约 -48%），但
`avg_travel_time_s` 比 Actuated **更差**（103–105 s vs 87 s），throughput 也略低于 Actuated
（1287/1261 vs 1354）。奖励是 diff-waiting-time，优化目标本身就只是等待时间，不是行程
时间或通行量，所以这个结果是自洽的，不应表述为"RL 全面优于基线"。`lane_closure` 场景下
Actuated 相对 Fixed 的提升最大（59.9 → 33.0），是基线里唯一接近 RL 的情形；
`evening_peak` 是 Actuated 唯一跑输 Fixed 的场景。

### 复现性验证（2026-08-14 全量 `--force` 重跑）

从视频开始重跑全链路（vision → routes → 重训 DQN/PPO → arena 60 次 → 全部图），退出码 0：

- `fixed`/`actuated` 六项指标与重跑前**逐位相同**（delta 全为 0.00）——基线不含随机模型，
  确定性复现。
- RL 偏移全部远小于种子标准差（DQN throughput +15.7 vs std 75.6；waiting -0.44 vs
  std 5.1），属重训噪声而非行为改变；两模型 `ep_rew_mean` 均无零值
  （DQN -22.9 → -10.7，PPO 收敛 -9.4），与重跑前吻合。
- 阶段耗时：vision 1.0 min、routes 0.0 min、DQN 67.8 min、PPO 64.7 min、arena 9.6 min。

**发现的遗留问题**：`vision/traffic_state.py:169` 把 `analyzed_at`（`datetime.now()`）写入
TrafficState，而 `run_fingerprint` 对整个文件取 sha1，因此每次重跑 vision 都会让
`base_state_sha1` 改变（`66940be…` → `73f107e…`），即使需求数据完全相同（本次实测两版
state 仅差时间戳与一处第 6 位小数舍入）。全量重跑时 60 行同步换 sha，无影响；但若将来只
重跑 vision 再补跑部分实验，`_drop_foreign_base_states` 的多数票会**静默丢掉少数派**，
表象是"实验变少"而根因只是时间戳。修法：指纹只哈希影响数字的字段，不含 `source`。

---

## 已产出的论文图

| 文件 | 尺寸 | 内容 |
|------|------|------|
| `fig_vision_flow_timeline.png` | 119 KB | 流量随时间变化 (5s bins) |
| `fig_vision_vehicle_mix.png` | 56 KB | 四向车型分布堆叠柱状图 |
| `fig_vision_annotated_frame.png` | 3.5 MB | 第一帧 + 计数线叠加 |
| `fig_vision_track_heatmap.png` | 148 KB | 轨迹密度热力图 |
| `fig_rl_training_curves.png` | 169 KB | DQN/PPO 训练进度 (双子图) |

**待产出** (需 `strategy_compare` 完成):
- `fig_arena_bars.png`: 场景×策略分组柱状图 (6 个指标子图)
- `fig_arena_radar.png`: 雷达图 (归一化多指标)
- `fig_before_after.png`: 固定时间 vs PPO 改善对比
- `fig_queue_timeseries.png`: 排队长度时间序列
- `fig_scenario_heatmap.png`: 场景×策略热力图

---

## 数据产物

```
data/
├── videos/
│   ├── demo.mp4                          # 21.5s supervision demo
│   ├── demo_roi.json                     # north/south 计数线配置
│   └── demo_annotated.mp4                # 79 MB 标注视频
├── traffic_states/
│   └── demo_001.json                     # schema 1.1, 1.9 KB
├── simulations/
│   └── train_cross_basic/
│       └── routes.rou.xml                # 训练路由
└── results/
    ├── arena_summary.csv                 # 待生成
    ├── tb/                               # TensorBoard 日志
    │   ├── dqn_cross_basic/
    │   └── ppo_cross_basic/
    └── experiments/                      # 已有 baseline (fixed/actuated)
        ├── normal__fixed__s0/
        ├── normal__actuated__s0/
        └── ... (约 30 个实验目录)
```

---

## 未完成项（本次会话时间截止前）

1. **四策略全量对比**: `strategy_compare --scenarios all --strategies fixed,actuated,dqn,ppo --seeds 3` 需要约 1-2 小时运行时间（60 个 episode × 1800s）
2. **最终 commit**: 待策略对比完成后，更新 README 补充最终结果表，提交所有改动

---

## Git 状态（未提交）

修改的文件：
- `.gitignore` (添加 `train_*.log`)
- `requirements.txt` (补充 tqdm, rich, libsumo==1.27.1)
- `experiments/strategy_compare.py` (跳过 smoke_* 场景)
- `scripts/plot_training_curves.py` (改绘 explained_variance/loss 而非 ep_rew_mean)

新增文件：
- `vision/` 五个文件 (detector, tracker, counter, traffic_state, analyze)
- `docs/devlog_vision.md` (vision 模块开发日志)
- `docs/rl_reward_diagnostic.md` (ep_rew_mean=0 诊断报告)
- `docs/DEVLOG.md` (本文件)
- `data/traffic_states/demo_001.json`
- `data/videos/demo.mp4`, `demo_roi.json`, `demo_annotated.mp4`
- `figures/fig_vision_*.png` (4 张)
- `figures/fig_rl_training_curves.png`
- `models/yolo11s.pt` (19.3 MB, .gitignore 已排除)
- 训练日志、arena baseline 结果若干

删除的文件：
- `data/results/experiments/normal__dqn__s0/` (旧的单次测试残留)

---

## 技术栈

| 领域 | 工具/库 | 版本 |
|------|--------|------|
| 视觉检测 | ultralytics (YOLO11) | 8.4.118 |
| 目标跟踪 | supervision (ByteTrack) | 0.30.0 |
| 交通仿真 | eclipse-sumo + traci/libsumo | 1.27.1 |
| RL 环境 | sumo-rl | 1.4.5+ |
| RL 训练 | stable-baselines3 (DQN/PPO) | 2.4+ |
| 深度学习 | torch + torchvision (CUDA 12.8) | 2.11.0 |
| 可视化 | streamlit, plotly, matplotlib | - |
| 科学计算 | numpy, pandas | - |

---

## 后续改进方向

### 短期
1. 完成四策略全量对比，更新 README 结果表
2. 提交代码，推送到仓库
3. 补充 `scripts/run_pipeline.py` 端到端流水线脚本

### 中期
1. 支持多摄像头融合（四个进口道分别拍摄，拼接为完整 TrafficState）
2. 队列长度估计（vision 模块目前 `queue_est: null`）
3. 转向比例观测（基于轨迹方向变化推断）
4. 更多 SUMO 模板（环岛、快速路匝道、BRT 专用道）
5. A2C / SAC / TD3 更多 RL 算法

### 长期
1. 真实路口视频数据集（≥5 min 观测窗口）
2. 在线学习（边部署边优化）
3. 多路口协同控制（区域信号配时）
4. 碳排放 / 能耗指标纳入优化目标
5. 与真实信号机对接（硬件在环测试）

---

## 致谢

- **Ultralytics** 提供 YOLO11 预训练模型
- **Supervision** 提供 demo 视频和标注工具链
- **Eclipse SUMO** 提供开源交通仿真器
- **LucasAlegre/sumo-rl** 提供 RL-SUMO 接口封装
- **DLR-RM/stable-baselines3** 提供高质量 RL 实现

---

**项目状态**: ✅ 核心功能全部实现并验证，待完成批量实验和最终提交
