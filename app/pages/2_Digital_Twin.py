"""🛣️ Digital Twin — 数字孪生：TrafficState → SUMO 场景生成 → 单次仿真 + 指标。"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import (
    DIRECTION_COLORS, DIRECTION_LABELS, DIRECTIONS, EXPERIMENTS_DIR,
    METRIC_LABELS, PYTHON, STRATEGIES, STRATEGY_LABELS, TEMPLATE_LABELS,
    TEMPLATES, TWIN_OUT_DIR, empty_state, fmt_metric, load_json,
    load_states, load_templates_meta, load_timeseries, page_setup,
    run_cli, style_fig,
)

page_setup("Digital Twin 数字孪生", "🛣️")
st.title("🛣️ Digital Twin · SUMO 数字孪生")
st.caption("选择 TrafficState 与路口模板 → 生成 SUMO 路由 → 运行仿真 → 查看 7 项指标与排队时序")


def summarize_routes(rou_path: Path) -> dict:
    root = ET.parse(rou_path).getroot()
    flows = root.findall(".//flow")
    total_vph = 0.0
    for f in flows:
        v = f.get("vehsPerHour") or f.get("vehsperhour")
        if v:
            total_vph += float(v)
        elif f.get("period"):
            p = float(f.get("period"))
            if p > 0:
                total_vph += 3600.0 / p
    return {
        "flows": len(flows),
        "total_vph": total_vph,
        "vtypes": len(root.findall(".//vType")),
        "vehicles": len(root.findall(".//vehicle")),
    }


# ================================================================ 1. 输入选择

st.subheader("1️⃣ 选择 TrafficState 与路口模板")

states = load_states()
if not states:
    empty_state("data/traffic_states/ 下暂无 TrafficState。请先在「视频感知」页分析一段视频。",
                "pages/1_Traffic_Vision.py", "前往视频感知")
    st.stop()

sel_col, tpl_col = st.columns(2)
with sel_col:
    state_name = st.selectbox("TrafficState", list(states), format_func=lambda s: f"{s}.json")
    state = states[state_name]
    flows = {d: (state.get("approaches", {}).get(d, {}) or {}).get("flow_vph") for d in DIRECTIONS}
    st.caption(" · ".join(
        f"{DIRECTION_LABELS[d].split()[0]} {v:,.0f} vph" if isinstance(v, (int, float)) else
        f"{DIRECTION_LABELS[d].split()[0]} —"
        for d, v in flows.items()))
with tpl_col:
    template = st.selectbox("路口模板", TEMPLATES, format_func=lambda t: TEMPLATE_LABELS[t])
    meta = load_templates_meta().get(template)
    if meta is None:
        st.caption("⚠️ 模板 meta.json 未找到（Sim 模块尚未提交），仍可尝试生成。")
    else:
        rows = [{"进口道": DIRECTION_LABELS.get(d, d),
                 "车道数": (info or {}).get("n_lanes"),
                 "入边": (info or {}).get("in_edge"),
                 "出边": (info or {}).get("out_edge")}
                for d, info in (meta.get("approaches") or {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        programs = meta.get("programs") or {}
        st.caption(f"tls_id=`{meta.get('tls_id', '?')}` · 信号程序: "
                   + ", ".join(f"{k}→`{v}`" for k, v in programs.items()))

# ================================================================ 2. 生成场景

st.subheader("2️⃣ 生成 SUMO 场景（路由）")

scen_dir = TWIN_OUT_DIR / f"{state_name}__{template}"
gen_col, seed_col = st.columns([3, 1])
gen_seed = seed_col.number_input("路由种子", 0, 9999, 0)
with gen_col:
    if st.button("⚙️ 生成 SUMO 场景", type="primary"):
        cmd = [PYTHON, "-m", "simulation.route_generator",
               "--state", str(Path("data/traffic_states") / f"{state_name}.json"),
               "--template", template,
               "--out", str(scen_dir),
               "--seed", str(int(gen_seed))]
        run_cli(cmd, "路由生成")

rou_files = sorted(scen_dir.glob("*.rou.xml")) if scen_dir.exists() else []
if not rou_files:
    empty_state(f"尚未生成路由文件（`{scen_dir.relative_to(_ROOT)}/*.rou.xml`）。点击上方按钮生成。")
    route_file = None
else:
    route_file = rou_files[0]
    try:
        s = summarize_routes(route_file)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("flow 条数", s["flows"], border=True)
        m2.metric("总需求 (vph)", f"{s['total_vph']:,.0f}", border=True)
        m3.metric("车型 vType 数", s["vtypes"], border=True)
        m4.metric("显式 vehicle 数", s["vehicles"], border=True)
        st.caption(f"路由文件：`{route_file.relative_to(_ROOT)}`")
    except ET.ParseError as e:
        st.error(f"routes 文件解析失败：{e}")

# ================================================================ 3. 运行仿真

st.subheader("3️⃣ 运行仿真")

r1, r2, r3 = st.columns([2, 1, 1])
strategy = r1.selectbox("信号控制策略", STRATEGIES, format_func=lambda s: STRATEGY_LABELS[s])
duration = r2.select_slider("仿真时长 (s)", options=[300, 600, 900, 1200, 1800, 2400, 3600],
                            value=1800)
sim_seed = r3.number_input("仿真种子", 0, 9999, 0)

exp_id = f"{state_name}-{template}__{strategy}__s{int(sim_seed)}"
exp_dir = EXPERIMENTS_DIR / exp_id

if st.button("▶️ 运行仿真", type="primary", disabled=route_file is None):
    cmd = [PYTHON, "-m", "app.run_episode_cli",
           "--template", template,
           "--route", str(route_file),
           "--strategy", strategy,
           "--seed", str(int(sim_seed)),
           "--out", str(exp_dir),
           "--duration-sec", str(int(duration)),
           "--scenario", f"{state_name}-{template}"]
    run_cli(cmd, f"SUMO 仿真（{STRATEGY_LABELS[strategy]}）")

# ================================================================ 4. 结果

st.subheader("4️⃣ 仿真结果")

metrics = load_json(exp_dir / "metrics_summary.json")
if metrics is None:
    candidates = [d for d in (EXPERIMENTS_DIR.glob(f"{state_name}-{template}__*") if
                              EXPERIMENTS_DIR.exists() else [])
                  if (d / "metrics_summary.json").exists()]
    if candidates:
        pick = st.selectbox("查看历史运行", candidates, format_func=lambda p: p.name)
        exp_dir = pick
        metrics = load_json(exp_dir / "metrics_summary.json")

if metrics is None:
    empty_state("暂无本组合的仿真结果。生成路由后点击「运行仿真」。")
else:
    st.caption(f"实验目录：`{exp_dir.relative_to(_ROOT)}`")
    keys = [k for k in METRIC_LABELS if k in metrics]
    row1, row2 = keys[:4], keys[4:]
    for chunk in (row1, row2):
        cols = st.columns(4)
        for col, k in zip(cols, chunk):
            col.metric(METRIC_LABELS[k], fmt_metric(k, metrics.get(k)), border=True)

    ts = load_timeseries(exp_dir)
    if ts is None:
        st.caption("timeseries.csv 缺失，无法绘制排队时序。")
    else:
        fig = go.Figure()
        for d in DIRECTIONS:
            col_name = f"queue_{d}"
            if col_name in ts.columns:
                fig.add_scatter(x=ts["t"], y=ts[col_name], mode="lines",
                                name=DIRECTION_LABELS[d],
                                line=dict(color=DIRECTION_COLORS[d], width=1.6))
        if "queue_total" in ts.columns:
            fig.add_scatter(x=ts["t"], y=ts["queue_total"], mode="lines",
                            name="总排队 Total", line=dict(color="#222222", width=2.6))
        fig.update_layout(xaxis_title="仿真时间 (s)", yaxis_title="排队车辆数 (veh)")
        st.plotly_chart(style_fig(fig, height=400, title="排队长度时序（四进口道 + 总计）",
                                  unified_hover=True), width="stretch")
        with st.expander("📄 timeseries.csv 数据预览"):
            st.dataframe(ts.head(200), hide_index=True, width="stretch")
