"""FlowMind AI · Dashboard 总览页。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.common import (
    ARENA_CSV, FIGURES_DIR, METRIC_LABELS, STRATEGY_LABELS,
    arena_metric_cols, comparable_panel, empty_state, improvement_pct,
    list_experiments, load_arena, load_states, page_setup, safe_page_link,
)

page_setup("Dashboard", "🚦")

# ---------------------------------------------------------------- Hero + 架构

st.markdown(
    """
<div class="fm-hero">
  <h1>🚦 FlowMind AI — 城市路口智能信控数字孪生平台</h1>
  <p>从一段路口视频出发：AI 感知交通流 → 构建 SUMO 数字孪生 → 对比四种信号控制策略 → What-if 场景推演，全流程纯软件闭环。</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="fm-flow">
  <div class="fm-node"><div class="fm-node-icon">🎥</div><div class="fm-node-title">视频感知</div><div class="fm-node-sub">目标检测 + 计数线过车统计</div></div>
  <div class="fm-arrow">➜</div>
  <div class="fm-node"><div class="fm-node-icon">📊</div><div class="fm-node-title">TrafficState</div><div class="fm-node-sub">四向流量 / 车型 / 转向比（schema 1.1）</div></div>
  <div class="fm-arrow">➜</div>
  <div class="fm-node"><div class="fm-node-icon">🛣️</div><div class="fm-node-title">SUMO 数字孪生</div><div class="fm-node-sub">路口模板 + 路由生成 + 微观仿真</div></div>
  <div class="fm-arrow">➜</div>
  <div class="fm-node"><div class="fm-node-icon">🚥</div><div class="fm-node-title">策略竞技场</div><div class="fm-node-sub">Fixed / Actuated / DQN / PPO</div></div>
  <div class="fm-arrow">➜</div>
  <div class="fm-node"><div class="fm-node-icon">🧪</div><div class="fm-node-title">What-if 推演</div><div class="fm-node-sub">高峰 / 事件涌入 / 车道封闭</div></div>
</div>
""",
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    safe_page_link("pages/1_Traffic_Vision.py", "视频感知 Traffic Vision", icon="🎥")
with nav2:
    safe_page_link("pages/2_Digital_Twin.py", "数字孪生 Digital Twin", icon="🛣️")
with nav3:
    safe_page_link("pages/3_Strategy_Arena.py", "策略竞技场 Strategy Arena", icon="🚥")
with nav4:
    safe_page_link("pages/4_Scenario_Lab.py", "场景实验室 Scenario Lab", icon="🧪")

st.divider()

# ---------------------------------------------------------------- 关键指标卡片

states = load_states()
experiments = list_experiments()
arena = load_arena()

best_name = None
best_wait = None
impr_vs_fixed = None
rank_notes: list[str] = []
if arena is not None and "strategy" in arena.columns and "avg_waiting_s" in arena.columns:
    # 只在"每个策略都跑过"的场景/种子子集上排名，否则跳过难场景的策略会白赢
    panel, rank_notes = comparable_panel(arena)
    mean_wait = panel.groupby("strategy")["avg_waiting_s"].mean().dropna() \
        if not panel.empty else pd.Series(dtype=float)
    if len(mean_wait) > 1:
        best_name = mean_wait.idxmin()
        best_wait = float(mean_wait.min())
        if "fixed" in mean_wait.index:
            impr_vs_fixed = improvement_pct(best_wait, float(mean_wait["fixed"]), "avg_waiting_s")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 TrafficState 数", len(states) or "—",
          help="data/traffic_states/ 中符合 schema 1.1 的状态文件数", border=True)
c2.metric("🧪 已完成实验数", len(experiments) or "—",
          help="data/results/experiments/ 中的实验目录数", border=True)
c3.metric("🏆 最佳策略",
          STRATEGY_LABELS.get(best_name, best_name) if best_name else "—",
          delta=f"平均等待 {best_wait:.1f} s" if best_wait is not None else None,
          delta_color="off",
          help="在所有策略都跑过的场景/种子上，平均等待时间最低者", border=True)
c4.metric("⚡ 相对 Fixed-Time 改善",
          f"{impr_vs_fixed:+.1f}%" if impr_vs_fixed is not None else "—",
          delta="等待时间下降" if impr_vs_fixed and impr_vs_fixed > 0 else None,
          help="最佳策略平均等待时间相对 Fixed-Time 的改善", border=True)

for note in rank_notes:
    st.caption(f"⚠️ 排名口径：{note}")

# ---------------------------------------------------------------- 最近实验

st.subheader("🕘 最近实验")
if arena is None:
    empty_state(
        "尚无实验汇总（data/results/arena_summary.csv 不存在）。"
        "请先在「策略竞技场」运行一键对比，或在「数字孪生」页运行单次仿真。",
        "pages/3_Strategy_Arena.py", "前往策略竞技场",
    )
else:
    metric_cols = arena_metric_cols(arena)
    id_cols = [c for c in ("scenario", "strategy", "seed", "template", "exp_id") if c in arena.columns]
    show = arena.tail(10).iloc[::-1][id_cols + metric_cols].copy()
    if "strategy" in show.columns:
        show["strategy"] = show["strategy"].map(lambda s: STRATEGY_LABELS.get(s, s))
    rename = {"scenario": "场景", "strategy": "策略", "seed": "种子", "template": "模板",
              "exp_id": "实验 ID", **METRIC_LABELS}
    st.dataframe(
        show.rename(columns=rename),
        hide_index=True, width="stretch",
        column_config={METRIC_LABELS[k]: st.column_config.NumberColumn(format="%.2f")
                       for k in metric_cols if k not in ("throughput_veh", "teleports")},
    )
    st.caption(f"数据源：`{ARENA_CSV.relative_to(_ROOT)}` · 共 {len(arena)} 条实验记录，展示最近 10 条")

# ---------------------------------------------------------------- 图表画廊

st.subheader("🖼️ 论文图表画廊")
pngs = sorted(FIGURES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) \
    if FIGURES_DIR.exists() else []
if not pngs:
    empty_state("figures/ 目录暂无 PNG 图表。运行视频分析或策略对比后，论文图会自动出现在这里。")
else:
    n_cols = 3
    for i in range(0, len(pngs), n_cols):
        cols = st.columns(n_cols)
        for col, p in zip(cols, pngs[i:i + n_cols]):
            with col:
                st.image(str(p), caption=p.name, width="stretch")
    st.caption(f"共 {len(pngs)} 张 · 来自 `figures/`（PNG, DPI≥300, NPG 配色）")
