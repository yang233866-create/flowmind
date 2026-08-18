# FlowMind Streamlit UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 FlowMind 改造成默认展示科研成果、同时保留完整实验操作的双标签 Streamlit 应用。

**Architecture:** 新建独立的成果展示组件模块，集中管理正式图目录、图表映射、结论页眉、KPI 卡片和数据口径。`app/common.py` 继续负责业务数据与运行工具，但统一视觉令牌；首页与六个页面只组合公共组件和现有业务逻辑，避免复制格式化规则。

**Tech Stack:** Python 3.11、Streamlit、pandas、Plotly、pytest、现有 PNG/SVG/PDF 科研图资产。

---

## 文件结构

- Create: `app/presentation.py` — 成果视图组件、正式图映射和可追溯信息。
- Create: `tests/test_presentation_system.py` — 视觉令牌、图文件映射、格式化和页面契约测试。
- Modify: `app/common.py` — 统一颜色、正式图目录、全局 CSS、Plotly 样式和侧栏。
- Modify: `app/Home.py` — 完整技术证据链首页。
- Modify: `app/pages/1_Traffic_Vision.py` — 感知成果视图与原工作台。
- Modify: `app/pages/2_Digital_Twin.py` — 孪生成果视图与原工作台。
- Modify: `app/pages/3_Strategy_Arena.py` — 策略证据视图与原工作台。
- Modify: `app/pages/4_Scenario_Lab.py` — 场景推演成果视图与原工作台。
- Modify: `app/pages/5_Scenario_Analysis.py` — 场景鲁棒性成果视图与分析工作台。
- Modify: `app/pages/6_Performance_Dive.py` — 稳定性成果视图与性能工作台，移除虚构因果瀑布分解。

### Task 1: 公共成果展示组件

**Files:**
- Create: `tests/test_presentation_system.py`
- Create: `app/presentation.py`
- Modify: `app/common.py`

- [ ] **Step 1: 写正式图和视觉令牌失败测试**

```python
from app.common import FIGURES_DIR, STRATEGY_COLORS
from app.presentation import figure_asset, format_scope


def test_strategy_palette_matches_publication_figures():
    assert STRATEGY_COLORS == {
        "fixed": "#8B98A8", "actuated": "#1E9E8F",
        "dqn": "#3569D4", "ppo": "#E46F51",
    }


def test_formal_figure_resolves_from_outputs():
    path = figure_asset("strategy_tradeoffs", "png")
    assert FIGURES_DIR.name == "figures"
    assert FIGURES_DIR.parent.name == "outputs"
    assert path.name == "02_strategy_tradeoffs.png"
    assert path.exists()


def test_scope_formatter_is_explicit():
    assert format_scope(runs=60, scenarios=5, strategies=4, seeds=3) == (
        "60 次运行 · 5 类场景 · 4 种策略 · 3 个随机种子"
    )
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py -q`
Expected: FAIL，原因是 `app.presentation` 尚不存在或颜色/目录仍为旧值。

- [ ] **Step 3: 实现展示模块和统一令牌**

`app/presentation.py` 提供以下稳定接口：

```python
FIGURE_CATALOG = {
    "vision_to_twin": "01_vision_to_twin",
    "strategy_tradeoffs": "02_strategy_tradeoffs",
    "scenario_robustness": "03_scenario_robustness",
    "queue_dynamics": "04_queue_dynamics",
    "training_evidence": "05_training_evidence",
    "decision_map": "06_decision_map",
    "regret_landscape": "07_regret_landscape",
    "paired_transitions": "08_paired_transitions",
    "operating_state_density": "09_operating_state_density",
    "scenario_timeline_atlas": "10_scenario_timeline_atlas",
    "perception_composition_flow": "11_perception_composition_flow",
}


def figure_asset(key: str, extension: str = "png") -> Path:
    stem = FIGURE_CATALOG[key]
    return FIGURES_DIR / f"{stem}.{extension}"


def format_scope(*, runs: int, scenarios: int, strategies: int, seeds: int) -> str:
    return f"{runs} 次运行 · {scenarios} 类场景 · {strategies} 种策略 · {seeds} 个随机种子"
```

同时实现 `render_page_intro`、`render_kpi_row`、`render_formal_figure`、`render_figure_pair`、`render_data_note` 和 `result_workbench_tabs`。正式图缺失时显示预期文件名和生成命令提示，下载按钮只在 SVG/PDF 存在时显示。

将 `app/common.py` 的 `FIGURES_DIR` 改为 `ROOT / "outputs" / "figures"`，策略色改为规范色，替换旧渐变 CSS，并将 Plotly 字体、边距、网格和图例调轻。

- [ ] **Step 4: 运行测试并确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py -q`
Expected: PASS。

- [ ] **Step 5: 提交公共组件**

```bash
git add app/common.py app/presentation.py tests/test_presentation_system.py
git commit -m "feat: add Streamlit evidence presentation system"
```

### Task 2: 重构首页证据链

**Files:**
- Modify: `app/Home.py`
- Modify: `tests/test_presentation_system.py`

- [ ] **Step 1: 写首页契约失败测试**

```python
from pathlib import Path


def test_home_uses_complete_evidence_chain_and_all_pages():
    source = Path("app/Home.py").read_text(encoding="utf-8")
    assert "视频感知 → 状态提取 → 数字孪生 → 策略决策 → 场景验证 → 性能结论" in source
    assert 'render_formal_figure("vision_to_twin"' in source
    for page in range(1, 7):
        assert f"pages/{page}_" in source
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py::test_home_uses_complete_evidence_chain_and_all_pages -q`
Expected: FAIL，旧首页缺少完整证据链和两个后续页面入口。

- [ ] **Step 3: 实现首页**

首页只使用真实 `arena_summary.csv`、实验目录和正式图计算/展示：

- 结论式 hero，不使用 emoji 和大面积彩虹渐变；
- KPI 显示运行数、场景数、策略数和可追溯的代表性改善；
- 主图使用 `vision_to_twin`；
- 辅助证据使用 `strategy_tradeoffs` 与 `scenario_robustness`；
- 六个页面入口全部出现；
- 最近实验移入折叠区，旧 `figures/` 画廊改为 `outputs/figures/` 精选证据。

- [ ] **Step 4: 运行首页契约和导入烟雾测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py tests/test_p0_pages.py -q`
Expected: PASS。

- [ ] **Step 5: 提交首页**

```bash
git add app/Home.py tests/test_presentation_system.py
git commit -m "feat: rebuild Streamlit home as evidence chain"
```

### Task 3: 为六个页面增加默认成果视图

**Files:**
- Modify: `app/pages/1_Traffic_Vision.py`
- Modify: `app/pages/2_Digital_Twin.py`
- Modify: `app/pages/3_Strategy_Arena.py`
- Modify: `app/pages/4_Scenario_Lab.py`
- Modify: `app/pages/5_Scenario_Analysis.py`
- Modify: `app/pages/6_Performance_Dive.py`
- Modify: `tests/test_presentation_system.py`

- [ ] **Step 1: 写双标签和正式图映射失败测试**

```python
from pathlib import Path

PAGE_FIGURES = {
    "1_Traffic_Vision.py": ("vision_to_twin", "perception_composition_flow"),
    "2_Digital_Twin.py": ("queue_dynamics", "operating_state_density"),
    "3_Strategy_Arena.py": ("strategy_tradeoffs", "decision_map"),
    "4_Scenario_Lab.py": ("scenario_timeline_atlas", "scenario_robustness"),
    "5_Scenario_Analysis.py": ("scenario_robustness", "regret_landscape"),
    "6_Performance_Dive.py": ("queue_dynamics", "operating_state_density"),
}


def test_pages_default_to_results_and_map_real_figures():
    for filename, keys in PAGE_FIGURES.items():
        source = (Path("app/pages") / filename).read_text(encoding="utf-8")
        assert 'result_workbench_tabs()' in source
        assert source.index("成果视图") < source.index("实验工作台")
        for key in keys:
            assert key in source
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py::test_pages_default_to_results_and_map_real_figures -q`
Expected: FAIL，旧页面尚无统一双标签。

- [ ] **Step 3: 逐页封装原工作台并增加成果视图**

每页将现有交互逻辑移入 `render_workbench()`，不改变其中的数据、参数、命令或统计公式；页面顶层调用：

```python
results_tab, workbench_tab = result_workbench_tabs()
with results_tab:
    render_results()
with workbench_tab:
    render_workbench()
```

成果视图使用设计规范第 4 节的结论、主图与辅助图。KPI 只从该页已有真实数据计算；无法计算时以数据说明替代，不补零。

- [ ] **Step 4: 运行页面契约和现有业务测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py tests/test_p0_pages.py tests/test_scenario_lab_matching.py tests/test_app_ranking_guard.py -q`
Expected: PASS。

- [ ] **Step 5: 提交六页改造**

```bash
git add app/pages tests/test_presentation_system.py
git commit -m "feat: add result-first views to Streamlit pages"
```

### Task 4: 修正性能页的证据语义

**Files:**
- Modify: `app/pages/6_Performance_Dive.py`
- Modify: `tests/test_presentation_system.py`

- [ ] **Step 1: 写禁止虚构因果分解的失败测试**

```python
from pathlib import Path


def test_performance_page_has_no_synthetic_causal_waterfall():
    source = Path("app/pages/6_Performance_Dive.py").read_text(encoding="utf-8")
    assert "* 0.3" not in source
    assert "性能瀑布分解" not in source
    assert "假设目标" not in source
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py::test_performance_page_has_no_synthetic_causal_waterfall -q`
Expected: FAIL，旧页面包含人为 30% 权重和假设目标。

- [ ] **Step 3: 用真实的目标/分布诊断替代**

删除人为 `* 0.3` 的瀑布贡献。工作台保留策略选择，但改为真实均值、标准差、相对 Fixed 改善和可选用户目标线；成果页使用队列动态、状态密度与训练证据。任何目标必须明确标注为“用户设定目标”，不得表达为实验结论。

- [ ] **Step 4: 运行测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py tests/test_visualization_data.py -q`
Expected: PASS。

- [ ] **Step 5: 提交语义修正**

```bash
git add app/pages/6_Performance_Dive.py tests/test_presentation_system.py
git commit -m "fix: keep performance evidence causally honest"
```

### Task 5: 全量验证和浏览器验收

**Files:**
- Modify only if verification exposes a defect in files already in scope.

- [ ] **Step 1: 运行 Streamlit 相关测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_presentation_system.py tests/test_p0_pages.py tests/test_scenario_lab_matching.py tests/test_app_ranking_guard.py -q`
Expected: PASS。

- [ ] **Step 2: 运行科研图数据与导出测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_visualization_data.py tests/test_visualization_exports.py tests/test_paper_figure_suite_contract.py tests/test_training_figure_paper_contract.py -q`
Expected: PASS。

- [ ] **Step 3: 启动 Streamlit**

Run: `.venv\Scripts\python.exe -m streamlit run app/Home.py --server.headless true --server.port 8501`
Expected: 服务启动，首页和六个页面可访问，无 Python traceback。

- [ ] **Step 4: 人工检查**

检查首页、策略竞技和性能深潜三个代表页面：标题无截断；KPI 的数字、单位与口径分层；正式图自适应容器；成果视图默认在前；工作台功能仍可见；缺失数据有明确回退。

- [ ] **Step 5: 汇总变更**

Run: `git status --short`
Expected: 只列出本轮计划内改动和用户原有未提交改动；不修改实验数据。