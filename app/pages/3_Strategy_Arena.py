"""🚥 Strategy Arena — 策略竞技场：多策略 × 多场景 × 多种子一键对比。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import (
    METRIC_HIGHER_BETTER, METRIC_INT, METRIC_LABELS, PYTHON, STRATEGIES,
    STRATEGY_COLORS, STRATEGY_LABELS, TEMPLATE_LABELS, TEMPLATES,
    arena_metric_cols, empty_state, fmt_metric, improvement_pct,
    known_scenarios, load_arena, page_setup, run_cli, style_fig,
)

page_setup("Strategy Arena 策略竞技场", "🚥")
st.title("🚥 Strategy Arena · 策略竞技场")
st.caption("Fixed-Time / Actuated / DQN / PPO 同场竞技：多种子重复实验，读取 arena_summary.csv 汇总对比")

# ================================================================ 1. 配置并运行

st.subheader("1️⃣ 一键对比")

scenarios = known_scenarios()
c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c1:
    template = st.selectbox("路口模板", TEMPLATES, format_func=lambda t: TEMPLATE_LABELS[t])
with c2:
    if scenarios:
        sel_scenarios = st.multiselect("场景（留空 = all）", scenarios, default=[])
    else:
        sel_scenarios = []
        st.caption("未发现已注册场景（experiments.scenarios 不可用且无历史产物），将使用 `all`。")
with c3:
    sel_strategies = st.multiselect("策略", STRATEGIES, default=STRATEGIES,
                                    format_func=lambda s: STRATEGY_LABELS[s])
with c4:
    seeds = st.number_input("种子数", 1, 10, 3)

if st.button("🏁 一键对比", type="primary", disabled=not sel_strategies):
    cmd = [PYTHON, "-m", "experiments.strategy_compare",
           "--template", template,
           "--scenarios", ",".join(sel_scenarios) if sel_scenarios else "all",
           "--strategies", ",".join(sel_strategies),
           "--seeds", str(int(seeds))]
    run_cli(cmd, "策略对比实验")

# ================================================================ 2. 结果

st.subheader("2️⃣ 对比结果（arena_summary.csv）")

arena = load_arena()
if arena is None or "strategy" not in arena.columns:
    empty_state("尚无实验汇总数据（data/results/arena_summary.csv 缺失或为空）。"
                "点击上方「一键对比」运行实验；DQN/PPO 需要先训练好 models/ 下的 checkpoint。")
    st.stop()

metric_cols = arena_metric_cols(arena)
if not metric_cols:
    st.warning("arena_summary.csv 中未找到契约定义的 7 个指标列。")
    st.dataframe(arena, width="stretch")
    st.stop()

# 场景过滤
flt = arena
if "scenario" in arena.columns:
    opts = ["全部场景"] + sorted(arena["scenario"].dropna().unique().tolist())
    pick = st.selectbox("筛选场景", opts)
    if pick != "全部场景":
        flt = arena[arena["scenario"] == pick]

present = [s for s in STRATEGIES if s in flt["strategy"].unique()]
if not present:
    st.warning("筛选结果为空。")
    st.stop()

grp = flt.groupby("strategy")[metric_cols]
mean_df = grp.mean().reindex(present)
std_df = grp.std().reindex(present)
n_runs = flt.groupby("strategy").size().reindex(present)

# ---- 排名卡片（按平均等待时间）
if "avg_waiting_s" in mean_df.columns:
    order = mean_df["avg_waiting_s"].sort_values().index.tolist()
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    st.markdown("**策略排名（按平均等待时间）**")
    cols = st.columns(len(order))
    fixed_wait = mean_df.loc["fixed", "avg_waiting_s"] if "fixed" in mean_df.index else None
    for col, (rank, s) in zip(cols, enumerate(order)):
        wait = mean_df.loc[s, "avg_waiting_s"]
        delta = None
        if fixed_wait is not None and s != "fixed":
            pct = improvement_pct(wait, fixed_wait, "avg_waiting_s")
            if pct is not None:
                delta = f"{pct:+.1f}% vs Fixed"
        col.metric(f"{medals[rank]} {STRATEGY_LABELS.get(s, s)}",
                   f"{wait:.2f} s",
                   delta=delta or ("基准 Baseline" if s == "fixed" else None),
                   delta_color="normal" if delta else "off",
                   help=f"{int(n_runs[s])} 次运行的均值", border=True)

# ---- 透视表（条件着色：每个指标最优加粗标绿）
st.markdown("**策略 × 指标透视表**（均值，绿色 = 该指标最优）")


def _highlight(col: pd.Series):
    if col.name not in METRIC_LABELS.values():
        return [""] * len(col)
    key = {v: k for k, v in METRIC_LABELS.items()}[col.name]
    best = col.max() if key in METRIC_HIGHER_BETTER else col.min()
    return ["background-color:#00A08726;font-weight:700" if v == best else ""
            for v in col]


pivot = mean_df.rename(columns=METRIC_LABELS)
pivot.index = [STRATEGY_LABELS.get(s, s) for s in pivot.index]
pivot.index.name = "策略"
fmt = {METRIC_LABELS[k]: ("{:,.0f}" if k in METRIC_INT else "{:,.2f}") for k in metric_cols}
st.dataframe(pivot.style.apply(_highlight, axis=0).format(fmt), width="stretch")
st.caption(f"每格为 {'×'.join(str(int(n)) for n in sorted(set(n_runs)))} 次运行（种子）的均值 · "
           f"共 {len(flt)} 条实验记录")

# ---- 分组柱状图（每指标一个 tab，误差棒 = seed 间 std）
st.markdown("**指标对比（误差棒 = 种子间标准差）**")
tabs = st.tabs([METRIC_LABELS[k] for k in metric_cols])
for tab, key in zip(tabs, metric_cols):
    with tab:
        fig = go.Figure()
        fig.add_bar(
            x=[STRATEGY_LABELS.get(s, s) for s in present],
            y=mean_df[key].tolist(),
            error_y=dict(type="data", array=std_df[key].fillna(0).tolist(),
                         color="#555555", thickness=1.4, width=6),
            marker=dict(color=[STRATEGY_COLORS.get(s, "#999") for s in present],
                        line=dict(color="#ffffff", width=1)),
            text=[fmt_metric(key, v) for v in mean_df[key]],
            textposition="outside",
            showlegend=False,
        )
        better = "↑ 越大越好" if key in METRIC_HIGHER_BETTER else "↓ 越小越好"
        fig.update_layout(yaxis_title=METRIC_LABELS[key])
        st.plotly_chart(style_fig(fig, height=380, title=f"{METRIC_LABELS[key]}（{better}）"),
                        width="stretch")

# ---- Before / After：相对 Fixed 的改善
st.markdown("**Before / After · 相对 Fixed-Time 的改善（正数 = 更好）**")
if "fixed" not in mean_df.index:
    st.info("筛选结果中没有 fixed 策略，无法计算改善百分比。", icon="ℹ️")
else:
    rows = []
    for s in present:
        if s == "fixed":
            continue
        row = {"策略": STRATEGY_LABELS.get(s, s)}
        for k in metric_cols:
            row[METRIC_LABELS[k]] = improvement_pct(mean_df.loc[s, k], mean_df.loc["fixed", k], k)
        rows.append(row)
    if rows:
        imp = pd.DataFrame(rows).set_index("策略")

        def _color(v):
            if pd.isna(v):
                return ""
            return "color:#00A087;font-weight:600" if v > 0 else "color:#E64B35;font-weight:600"

        st.dataframe(imp.style.map(_color).format("{:+.1f}%", na_rep="—"), width="stretch")
        st.caption("正数 = 该指标相对 Fixed-Time 改善（等待/排队等取下降幅度，通过量/车速取上升幅度）")

with st.expander("📄 原始 arena_summary.csv"):
    st.dataframe(arena, hide_index=True, width="stretch")
