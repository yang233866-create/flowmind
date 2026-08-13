# FlowMind AI 开发日志

> 面向接手者：读完本文件 + `docs/CONTRACTS.md` 即可继续开发。
> 环境：Windows 11，Python 3.12（uv 管理，`.venv/`），RTX 5050 8GB。

## 2026-08-12 ~ 08-13 | 立项与环境

**方案来源**：`D:\edge download\FlowMind_AI_项目整体实施方案.docx`（全文提取存于
`C:\Users\y\flowmind_doc.txt`）。核心链路：视频感知 → TrafficState →
SUMO 数字孪生 → Fixed/Actuated/DQN/PPO 多策略对比 → What-if 推演。

**对原方案的补充/修改决策**：

1. **TrafficState 升级到 schema 1.1**：在原方案基础上增加
   `flow_profile`（5 分钟分箱流量，支持时变需求）、逐方向
   `turning_ratio`、逐方向 `vehicle_mix`、`observed` 标记（视频未覆盖
   方向用默认值填充并标记）。原方案的固定 70/15/15 作为缺省值保留。
2. **指标采集统一化**：所有策略（含 RL）通过同一个 MetricsCollector +
   tripinfo 输出采集指标，避免"RL 用 env 内部指标、基线用 tripinfo"导致
   的不可比。这是方案未明确、但论文可信度的关键。
3. **libsumo 加速**：训练时可用 `FLOWMIND_LIBSUMO=1` 走 libsumo（比 traci
   socket 快 5-10 倍），评估默认 traci 保证稳定。
4. **技术栈确认**（调研结论）：ultralytics YOLO11 + supervision(ByteTrack)
   + eclipse-sumo(pip 版，免手动装 SUMO) + sumo-rl + stable-baselines3 +
   streamlit。全部 pip 可装，无需系统级安装器。
5. **论文图表**：统一 `scripts/sci_style.py`（Times New Roman、DPI≥300、
   NPG 配色、只出 PNG）。规划的图：架构图、检测/跟踪标注帧、四向流量
   时序、车型构成、轨迹热力图、SUMO 路网图、RL 训练曲线、策略对比柱状
   图（带置信区间）、雷达图、场景×策略热力图、排队时序、Before/After。

**环境搭建记录**：

- `uv` 装在系统 Python 3.14 下（`python -m uv ...`），项目 venv 用 3.12
  （3.14 太新，torch/SB3 生态不支持）。
- torch 装 CUDA 12.8 版（RTX 5050 是 Blackwell 架构 sm_120，**必须**
  cu128 及以上，cu121 会报 no kernel image）。
- 安装命令见 `requirements.txt` 头部注释。

**踩坑记录（重要）**：

- 2026-08-12 下午 Claude Code 执行环境瘫痪（所有工具 spawn 失败），排查
  半天：重装、杀毒加白均无效，最终**换新会话恢复**——旧会话状态损坏。
  接手者若遇到同类问题（工具全部无响应），直接开新会话，不要修旧会话。
- 后台安装 torch 2.6GB 被会话重启打断过一次，重跑即可（uv 有缓存）。

## 2026-08-13 | 架构落地

- 写定 `docs/CONTRACTS.md`（模块间接口契约，改接口必须先改它）。
- 项目骨架 + git 初始化完成。
- 并行开发启动：Vision / Sim+Opt / App 三个模块并行编写，
  Experiments 模块（场景定义、批量实验、对比图）由主线完成。

### 待办主线

- [ ] 依赖装完 → 验证 torch CUDA + SUMO + sumo-rl
- [ ] 演示视频获取（supervision assets 或公开数据）
- [ ] 视频 → TrafficState.json 跑通
- [ ] TrafficState → SUMO 路由 → fixed/actuated 基线跑通
- [ ] DQN 训练（cross_basic，100k steps）→ PPO
- [ ] Strategy Arena 批量实验（场景×策略×3 seeds）
- [ ] 全部论文图输出到 figures/
- [ ] Streamlit 界面联调
- [ ] README 完善 + 最终提交
