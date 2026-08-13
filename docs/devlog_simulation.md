# simulation/ 模块开发日志

> 开发时间：2026-08-13。给接手者：本文档记录 SUMO 仿真层的设计决策、踩坑与实测结果。
> 契约以 docs/CONTRACTS.md 为准，本模块已按契约实现并实测通过。

## 交付物清单

| 文件 | 说明 |
|---|---|
| `simulation/templates/{cross_basic,cross_leftturn,arterial_minor}/` | 三个路口模板，各含 plain XML、build.py、预构建 net.net.xml、tls_act.add.xml、meta.json |
| `simulation/route_generator.py` | TrafficState (+可选 Scenario Spec) → routes.rou.xml + gen_meta.json |
| `simulation/metrics.py` | MetricsCollector（每秒采集）+ 模块级 parse_tripinfo（指标聚合） |
| `simulation/sumo_runner.py` | run_episode() 单回合运行器 + CLI；fixed/actuated 直跑，dqn/ppo 委托 optimization.rl_agents |
| `data/traffic_states/synthetic_demo.json` | 合成 TrafficState（schema 1.1，四向 600–1100 vph），供冒烟测试 |
| `data/results/experiments/smoke_*/` | 冒烟测试产物（保留，勿删） |

## 模板设计要点

- 几何统一：十字路口，四臂各 150 m，限速 13.89 m/s，中心节点 `TL`（traffic_light）。
  边 ID 固定为 `N_in/N_out/S_in/S_out/E_in/E_out/W_in/W_out`（契约要求）。
- **connections 全部显式定义**，不让 netconvert 猜车道-转向映射：
  - cross_basic（2 车道）：lane0 = 直+右，lane1 = 直+左。
  - cross_leftturn（3 车道）：lane0 = 直+右，lane1 = 直，lane2 = 专用左转。
  - arterial_minor：东西主干 3 车道（lane0 直+右 / lane1 直 / lane2 直+左），南北次干 1 车道（全部转向）。
- **tls 的 linkIndex 也全部显式指定**（.tll.xml 里再写一遍 connection 并带 `tls`/`linkIndex` 属性），
  这样相位 state 字符串的位序完全可控，netconvert 不会重排。三个模板的 linkIndex 布局都写在
  各自 tls.tll.xml 的注释里。cross_basic/cross_leftturn 均为 16 个 link，arterial_minor 也是
  16 个（主干每向 5 个 + 次干每向 3 个）。
- 信号程序：
  - cross_basic 静态 4 相位：NS 绿 33 / 黄 3 / EW 绿 33 / 黄 3。右转与直行同放（state 里直行 `G`、并行左转 `g` 允许对向让行）。
  - cross_leftturn 静态 8 相位保护左转：NS 直 30 / 黄 3 / NS 左 15 / 黄 3 / EW 直 30 / 黄 3 / EW 左 15 / 黄 3。左转相位为专有 `G`，无冲突放行。
  - arterial_minor 静态 4 相位：主干（EW）绿 45 / 黄 3 / 次干（NS）绿 20 / 黄 3。
  - actuated 程序（tls_act.add.xml）：同一 `TL`、programID="act"、type="actuated"，相位 state 与静态程序完全一致，绿相位 minDur=5 maxDur=60，黄 3s。SUMO 会自动为 actuated 程序生成感应线圈，无需手写 detector。
- netconvert 用 `--no-turnarounds`；构建后已核对 net.net.xml 内的 tlLogic 相位数与 .tll.xml 一致
  （4/8/4 相位，state 字符串逐字符相同，未被 netconvert 改写）。

## route_generator 设计

- 流量分解：`vehsPerHour = flow_vph × multiplier × turning_ratio × vehicle_mix`，
  每 approach × movement × vType 一条 `<flow>`（rate ≤ 0.01 vph 的组合直接跳过，避免无意义 flow）。
- vType 参数（长度/加速/最大速度）：car 4.5m/2.6/33.3，bus 12m/1.2/22.2，truck 9m/1.3/25，
  motorcycle 2.2m/3.5/30；vClass 对应 passenger/bus/truck/motorcycle。
- `flow_profile` 存在时按 `profile_bins_sec` 分箱生成分时段 flow；profile 比 duration 短则末箱值持续到结束；无 profile 用单一 flow 覆盖全时段。
- turning_ratio 缺省 0.15/0.70/0.15，且做归一化防御（和不为 1 时按比例缩放）；vehicle_mix 同样归一化，缺省全 car。
- lane_closures 不影响路由生成，只透传写进 gen_meta.json（关道由 sumo_runner 在 traci 里做）。
- duration 优先级：scenario spec 的 duration_sec > state 的 duration_sec > 1800。

## metrics 设计（重要：接口兼容）

**踩坑：optimization/rl_agents.py 先于本模块写好，其期望的接口与最初任务描述不同**：

- 它用 `MetricsCollector(sumo, meta)` 两参构造（第二参数是整个 meta.json dict）；
- 它 `from simulation.metrics import parse_tripinfo` 导入**模块级函数**，签名 `parse_tripinfo(tripinfo_path, timeseries_df)`；
- 它的 `run_rl_episode(...)` 返回 `(metrics, ts)` 元组而非 RunResult。

由于 optimization/ 不可改，metrics.py 做了双形态兼容：

- `MetricsCollector(traci, in_edges_dict, tls_id)`（sumo_runner 用）与
  `MetricsCollector(traci, meta_dict)`（rl_agents 用，tls_id=None 时自动从 meta 提取）。
- 模块级 `parse_tripinfo(tripinfo_path, timeseries=None, teleports=0.0)`；
  实例方法 `collector.parse_tripinfo(path)` 内部调用模块级函数并带上自己累计的 teleports。
- **已知限制**：rl_agents 调 `parse_tripinfo(tripinfo, ts)` 不传 teleports，所以 RL 回合的
  teleports 恒为 0。如需修正，应让 rl_agents 传 `collector.teleports`（需 Opt 模块改）。

采集口径：

- 排队 = `edge.getLastStepHaltingNumber(in_edge)`（速度 < 0.1 m/s 的车数），四向求和为 queue_total；
- waiting_total = 四条 in_edge 的 `edge.getWaitingTime` 之和；
- teleports 用每步 `simulation.getStartingTeleportNumber()` 累加（比解析 stderr 可靠）；
- avg_speed_mps = mean(routeLength / duration)（逐车求商再平均，不是总量相除）。

## sumo_runner 设计

- 生成的 .sumocfg 里所有路径用绝对路径（route 文件、net、tripinfo），避免 cwd 依赖；
  `time-to-teleport 300`、`no-step-log`、seed 写在 cfg 的 `<random_number>` 里。
- fixed → `setProgram(tls_id, "0")`；actuated → sumocfg 的 additional-files 加载
  tls_act.add.xml 后 `setProgram(tls_id, "act")`。若 add 文件没加载，setProgram 会直接抛错，
  所以不会出现"以为在跑 actuated 实际在跑 static"的静默错误。
- lane_closures：对 in_edge 的**前 n_lanes 条车道（lane0 起，即最右侧起）**
  `lane.setAllowed(lane_id, ["authority"])`。已用 east 关 1 道实测可用。
- dqn/ppo：延迟 import `optimization.rl_agents.run_rl_episode`，把返回元组包装成 RunResult 并补写
  config.json（rl_agents 自己写 metrics_summary.json 和 timeseries.csv）。模型缺失时 rl_agents 的
  load_model 抛带训练命令提示的 RuntimeError（已实测触发）。**RL 路径未做端到端测试（模型未训练）**。
- 输出四件套齐全：metrics_summary.json / timeseries.csv / tripinfo.xml / config.json。
- 支持 `FLOWMIND_LIBSUMO=1` 切 libsumo（走 sumo_home.use_libsumo()，未实测）。

## 踩坑记录

1. **rl_agents 接口不匹配**（见上）。教训：写公共模块前先 grep 现有调用方。
2. traci 在 Windows 上启动时偶见 "Could not connect to TraCI server ... Retrying in 1 seconds"
   两三次后连上，属正常现象（sumo 进程启动慢），不是错误。
3. actuated 程序放 additional 文件时，相位 state 必须与 net 内置程序的 linkIndex 布局一致，
   否则相位含义错乱——这也是 .tll.xml 里显式写 linkIndex 的原因。
4. netconvert 对显式 connections + 显式 tll 的组合不会做相位缩并（本项目验证过 4/8/4 相位原样保留），
   但如果去掉 `--no-turnarounds` 会多出掉头连接、state 长度变化，切勿去掉。

## 冒烟测试实测结果（600s，seed=0，产物在 data/results/experiments/smoke_*/）

| 运行 | avg_waiting_s | avg_travel_time_s | throughput | avg_queue | max_queue | avg_speed | teleports |
|---|---|---|---|---|---|---|---|
| cross_basic fixed | 36.63 | 77.37 | 476 | 36.6 | 59 | 5.30 | 0 |
| cross_basic actuated | 41.85 | 79.68 | 472 | 46.5 | 78 | 5.62 | 0 |
| cross_leftturn fixed | 28.82 | 60.00 | 544 | 28.0 | 45 | 6.02 | 0 |
| arterial_minor fixed | 35.78 | 71.88 | 384 | 27.5 | 42 | 7.43 | 0 |

另跑了 scenario spec 联测（flow_multipliers + east 关 1 道，300s）：
`smoke_cross_basic__closure__s0`，正常完成，throughput 175，teleports 0。

验收判定：throughput > 50 ✓；avg_waiting ∈ [1, 300] ✓；timeseries 600 行、列齐 ✓；teleports < 10 ✓；
actuated 与 fixed 指标有差异 ✓（并从 timeseries 验证了绿灯时长：fixed 恒 33s，actuated 被拉到 60s 上限，
说明 act 程序真实生效）。

注：本合成需求下 actuated 比 fixed 略差，原因是四向流量都高、感应线圈持续有车，绿灯总被拉满 60s，
周期变长导致红灯方向等待增加。这是合理行为，不是 bug；低/不对称流量下 actuated 才有优势。
