# App 模块开发日志（Streamlit 界面）

日期：2026-08-13 · 负责：App 模块 · 依据：`docs/CONTRACTS.md`

## 一、文件结构

```
app/
  __init__.py
  common.py                 # 公共工具：路径、读取、子进程流式运行、Plotly 样式
  run_episode_cli.py        # 单次仿真的 CLI 薄封装（见"设计决策 4"）
  Home.py                   # 🚦 Dashboard 总览
  pages/
    1_Traffic_Vision.py     # 🎥 视频感知
    2_Digital_Twin.py       # 🛣️ 数字孪生
    3_Strategy_Arena.py     # 🚥 策略竞技场
    4_Scenario_Lab.py       # 🧪 What-if 场景实验室
```

启动：`.venv/Scripts/python.exe -m streamlit run app/Home.py`

## 二、各页面功能

### Home（Dashboard）
- Hero 横幅（NPG 渐变）+ 五节点数据流架构卡片（视频感知 → TrafficState → SUMO 孪生 → 策略竞技场 → What-if），带各页快捷链接。
- 四张指标卡：TrafficState 数 / 已完成实验数 / 最佳策略 / 相对 Fixed-Time 的等待时间改善百分比。
- **排名口径（2026-08-14 修）**：「最佳策略」不再直接对 `arena_summary.csv` 全表求均值，而是先经 `common.comparable_panel()` 裁剪到"同一 base TrafficState + 每个策略都跑过的 (场景, seed) 组合"子集；被剔除的部分以 caption 说明。否则只跑了简单场景的策略会凭构造夺冠，跨 TrafficState 的运行也会被混进同一个平均值。
- 最近实验表（arena_summary.csv 尾部 10 行，倒序，策略名中文化）。
- `figures/` PNG 画廊（按修改时间倒序，3 列网格）。

### 1_Traffic_Vision
- 视频上传（存 `data/videos/`）或选择已有视频；读第一帧（cv2，带 `st.cache_data` 缓存，mtime 参与缓存键）。
- ROI/计数线 JSON 编辑器：优先预填 `<video>_roi.json` → `demo_roi.json` → 按画面尺寸生成的四向示例模板；启发式解析任意嵌套结构中的线段（`[[x1,y1],[x2,y2]]` 或 `[x1,y1,x2,y2]`），PIL 在第一帧上叠画彩线 + 端点 + 标签实时预览，线色按方向名匹配 `DIRECTION_COLORS`。
- 「开始分析」→ `python -m vision.analyze --video ... --roi-config ... --state-out ... --annotated-video ... --figures-dir figures [--max-frames N]`，`st.status` 内逐行流式输出。ROI 文本先落盘为 `<video>_roi.json` 再传给 CLI。
- 结果区：四向 `flow_vph` 指标卡（区分 observed 状态）、车型构成堆叠柱状图（plotly）、转向比表、`flow_profile` 时变折线、原始 JSON expander、标注视频 `st.video`、与本场景相关的 figures。

### 2_Digital_Twin
- 选 TrafficState（下拉，附四向流量摘要）+ 模板（cross_basic/cross_leftturn/arterial_minor），展示 `meta.json` 的进口道车道数、边、tls_id、信号程序表；meta 缺失时降级为提示。
- 「生成 SUMO 场景」→ `python -m simulation.route_generator --state ... --template ... --out data/results/twin/<state>__<template> --seed N`；解析生成的 `*.rou.xml` 展示摘要（flow 条数、总 vph、vType 数、显式 vehicle 数）。
- 「运行仿真」（策略 × 时长 × 种子）→ `python -m app.run_episode_cli ...`（见设计决策 4），产物按契约写入 `data/results/experiments/<exp_id>/`。
- 结果区：7 指标卡（两行 ×4）+ timeseries.csv 排队时序折线（四向 + 总排队黑线）；当前组合无结果时提供历史运行下拉。

### 3_Strategy_Arena
- 场景来源三级兜底：`experiments.scenarios` 注册表（try/except import，尝试 SCENARIOS/REGISTRY 等常见命名）→ `data/results/scenario_specs/*.json` → 扫描 experiments 目录解析 `<scenario>__<strategy>__s<seed>`。
- 「一键对比」→ `python -m experiments.strategy_compare --template ... --scenarios <sel|all> --strategies ... --seeds N`。
- 结果区（读 arena_summary.csv，支持按场景过滤）：
  - 排名卡（🥇🥈🥉，按平均等待时间，非 fixed 策略附 "vs Fixed" 改善 delta）；排名同样走 `comparable_panel()` 的可比子集，卡片下方注明剔除原因（面板不平衡 / 混入其它 base TrafficState）；
  - 策略×指标透视表（每列最优值绿底加粗，方向感知：throughput/speed 越大越好）；
  - 每指标一个 tab 的柱状图，误差棒 = seed 间 std，柱色 = 契约策略固定色；
  - Before/After 表：各策略相对 fixed 的逐指标改善百分比（绿正红负）。

### 4_Scenario_Lab
- 四向流量倍率滑块（0.2–3.0，步长 0.1）+ 五个预设按钮（Normal / Morning Peak / Evening Peak / Event Surge / Lane Closure）联动滑块与封闭配置 + 车道封闭（方向 + 1–3 条）+ 时长。
- Scenario Spec JSON 实时预览（严格按契约字段）+ baseline vs what-if 流量对比预览柱状图；保存到 `data/results/scenario_specs/<name>.json`。
- 「运行推演」→ 先落盘 spec，再 `python -m experiments.scenario_runner --scenario <spec.json> --strategy ... --seed N`。
- 结果对比：自动查找 `<name>__<strategy>__s*` 最新结果，与同策略 baseline（场景名 normal/baseline/whatif_normal）对比——指标卡带 delta、归一化"相对 baseline 变化 %"柱状图（改善绿/恶化红，方向感知）、总排队时序对比折线（baseline 虚线灰、what-if 实线红）。baseline 缺失时降级为只展示本次结果并提示先跑 Normal 预设。

## 三、设计决策

1. **契约优先、不 import 其他模块**：所有数据交互只走 CONTRACTS.md 定义的文件格式与 CLI；唯一例外 `experiments.scenarios` 按要求 try/except 兜底，失败则扫描产物目录。
2. **空状态全覆盖**：每页对"上游产物不存在"都有 `st.info` 引导 + 页面链接（`st.page_link` 包了兜底，单页测试环境不炸），全部页面在**空数据仓库**下可正常渲染（已测）。
3. **子进程流式日志**：`common.stream_command` 用 `Popen(stdout=PIPE, stderr=STDOUT, encoding='utf-8', errors='replace')` 逐行读取，写入 `st.status` 内的 `st.empty()` 占位 code 块（保留尾部 300 行）；设置 `PYTHONUNBUFFERED=1` + `PYTHONIOENCODING=utf-8` 防 Windows GBK 编码崩溃。命令本身用 `st.code` 展示，便于复现。
4. **`app/run_episode_cli.py`**：契约只给了 `simulation.sumo_runner.run_episode` 的 Python API，没有仿真单跑的 CLI。为了保持"界面进程不 import 仿真代码 + 流式日志"，加了这个属于 app 模块的薄封装：子进程内 import sumo_runner，跑完按结果目录契约补写 `metrics_summary.json` / `timeseries.csv` / `config.json`（已存在则不覆盖）。
5. **配色**：Plotly 统一走 `common.style_fig`（plotly_white + NPG colorway + Times New Roman 字族 + 顶部水平图例）；策略色/方向色与 `scripts/sci_style.py` 完全一致（策略 fixed=#3C5488, actuated=#4DBBD5, dqn=#E64B35, ppo=#00A087）。#3C5488/#4DBBD5 饱和度与浅底对比度偏低，因此柱状图一律带直接数值标签（textposition="outside"），不依赖纯颜色辨识。
6. **指标方向感知**：`throughput_veh`、`avg_speed_mps` 越大越好，其余越小越好；改善百分比、表格高亮、红绿着色都据此统一计算（`common.improvement_pct`），避免"通过量下降也显示为改善"。
7. **Streamlit 1.61 API**：用 `width="stretch"` 替代已弃用的 `use_container_width`；指标卡用 `border=True`。
8. **Scenario Lab 联动**：预设按钮渲染在滑块之前，点击时直接写滑块的 `session_state` key（该轮 widget 尚未实例化，合法），实现一键联动且用户仍可再手动微调。

## 四、测试结果（全部实际运行）

环境：streamlit 1.61.1 / plotly 6.9.0 / pandas 3.0.5 / Python 3.12。

1. **py_compile**：7 个文件（common、run_episode_cli、Home、4 个 pages）全部 OK。
2. **AppTest 冒烟（streamlit.testing.v1）**，两轮：
   - 空数据仓库（无任何 data 产物）：5 页全部 PASS，无异常；
   - 构造 fixture（TrafficState、arena_summary.csv 8 行、3 个实验目录含 metrics+timeseries、小视频、figure PNG）：5 页全部 PASS；测后 fixture 已清理。
3. **Headless 健康检查**：`streamlit run app/Home.py --server.headless true --server.port 8511` 后台启动；
   - `GET /_stcore/health` → 200 `ok`；
   - `GET /`、`/Traffic_Vision`、`/Digital_Twin`、`/Strategy_Arena`、`/Scenario_Lab` → 全部 200，无 500；
   - 测试结束后已 taskkill 进程，端口释放。

未覆盖：按钮触发的 CLI 真实执行（vision/simulation/experiments 模块并行开发中，尚无实现），页面侧已按契约拼好命令并做了失败态展示（退出码非 0 → st.status error）。

## 五、遗留 / 待联调

- `experiments.scenarios` 的注册表变量名是猜测集（SCENARIOS/REGISTRY/list_scenarios()），Experiments 模块落地后按实际命名对齐即可（`common.known_scenarios`一处）。
- `vision.analyze` 标注视频若用 mp4v 编码，浏览器 `st.video` 可能无法解码（页面已附本地路径提示）；建议 vision 模块输出 H.264（如 `avc1`）。
- `strategy_compare` 的 `--template` 参数与场景名语义、arena_summary.csv 是否含 `template`/`exp_id` 列，联调时确认（页面对列缺失已兜底）。
