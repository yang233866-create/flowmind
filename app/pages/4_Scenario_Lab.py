"""🧪 Scenario Lab — What-if 推演：流量倍率 / 车道封闭 → Scenario Spec → 运行与对比。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import streamlit as st

from app.common import (
    DIRECTION_LABELS, DIRECTIONS, EXPERIMENTS_DIR, METRIC_HIGHER_BETTER,
    METRIC_LABELS, NPG, PYTHON, SPECS_DIR, STRATEGIES, STRATEGY_LABELS,
    empty_state, fmt_metric, improvement_pct, list_state_files, load_json,
    load_timeseries, page_setup, run_cli, style_fig,
)

page_setup("Scenario Lab 场景实验室", "🧪")
st.title("🧪 Scenario Lab · What-if 场景推演")
st.caption("调整四向流量倍率、封闭车道 → 生成 Scenario Spec → 运行仿真 → 与 baseline(normal) 对比")

PRESETS = {
    "Normal 平峰": {"mult": {"north": 1.0, "south": 1.0, "east": 1.0, "west": 1.0}, "icon": "🟢"},
    "Morning Peak 早高峰": {"mult": {"north": 1.6, "south": 1.6, "east": 1.0, "west": 1.0}, "icon": "🌅"},
    "Evening Peak 晚高峰": {"mult": {"north": 1.0, "south": 1.0, "east": 1.6, "west": 1.6}, "icon": "🌆"},
    "Event Surge 事件涌入": {"mult": {"north": 2.2, "south": 1.3, "east": 1.3, "west": 2.2}, "icon": "🎆"},
    "Lane Closure 车道封闭": {"mult": {"north": 1.0, "south": 1.0, "east": 1.0, "west": 1.0},
                              "closure": {"approach": "east", "n_lanes": 1}, "icon": "🚧"},
}

# ================================================================ 1. 基础状态

st.subheader("1️⃣ 基础 TrafficState")
state_files = list_state_files()
if not state_files:
    empty_state("data/traffic_states/ 下暂无 TrafficState，What-if 推演需要一个基础状态。"
                "请先在「视频感知」页生成。", "pages/1_Traffic_Vision.py", "前往视频感知")
    st.stop()
base_path = st.selectbox("base_state", state_files, format_func=lambda p: p.name)

# ================================================================ 2. 预设 + 倍率

st.subheader("2️⃣ 场景参数")

# 预设按钮在滑块之前渲染：点击时本轮尚未实例化下方 widget，
# 直接写它们的 session_state key 即可实现联动。
pcols = st.columns(len(PRESETS))
for col, (pname, cfg) in zip(pcols, PRESETS.items()):
    if col.button(f"{cfg['icon']} {pname.split()[0]}", width="stretch", help=pname):
        for d in DIRECTIONS:
            st.session_state[f"mult_{d}"] = float(cfg["mult"][d])
        closure = cfg.get("closure")
        st.session_state["closure_dir"] = closure["approach"] if closure else "无"
        st.session_state["closure_n"] = closure["n_lanes"] if closure else 1
        st.session_state["lab_name"] = "whatif_" + re.sub(
            r"[^a-z0-9]+", "_", pname.split()[0].lower()).strip("_")

s1, s2 = st.columns([3, 2])
with s1:
    st.markdown("**四向流量倍率**")
    mcols = st.columns(4)
    mult = {}
    for col, d in zip(mcols, DIRECTIONS):
        st.session_state.setdefault(f"mult_{d}", 1.0)
        mult[d] = col.slider(DIRECTION_LABELS[d], 0.2, 3.0, step=0.1, key=f"mult_{d}")
with s2:
    st.markdown("**车道封闭**")
    cc1, cc2 = st.columns(2)
    st.session_state.setdefault("closure_dir", "无")
    st.session_state.setdefault("closure_n", 1)
    closure_dir = cc1.selectbox(
        "封闭方向", ["无"] + DIRECTIONS, key="closure_dir",
        format_func=lambda d: "不封闭" if d == "无" else DIRECTION_LABELS[d])
    closure_n = cc2.number_input("封闭车道数", 1, 3, key="closure_n",
                                 disabled=closure_dir == "无")

    duration = st.select_slider("仿真时长 (s)", options=[600, 900, 1200, 1800, 2400, 3600],
                                value=1800)

# ================================================================ 3. Spec 预览与保存

st.subheader("3️⃣ Scenario Spec")

st.session_state.setdefault("lab_name", "whatif_custom")
name = st.text_input("场景名称", key="lab_name")
safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "whatif_custom"

spec = {
    "name": safe_name,
    "base_state": str(base_path.relative_to(_ROOT)).replace("\\", "/"),
    "flow_multipliers": {d: round(mult[d], 2) for d in DIRECTIONS},
    "lane_closures": ([] if closure_dir == "无"
                      else [{"approach": closure_dir, "n_lanes": int(closure_n)}]),
    "duration_sec": int(duration),
}

pv1, pv2 = st.columns([2, 3])
with pv1:
    st.json(spec)
with pv2:
    base_state = load_json(base_path) or {}
    base_flows = {d: (base_state.get("approaches", {}).get(d, {}) or {}).get("flow_vph") or 0.0
                  for d in DIRECTIONS}
    fig = go.Figure()
    fig.add_bar(x=[DIRECTION_LABELS[d] for d in DIRECTIONS],
                y=[base_flows[d] for d in DIRECTIONS],
                name="Baseline 流量", marker=dict(color="#B0B7C3"))
    fig.add_bar(x=[DIRECTION_LABELS[d] for d in DIRECTIONS],
                y=[base_flows[d] * mult[d] for d in DIRECTIONS],
                name="What-if 流量", marker=dict(color=NPG[0]),
                text=[f"×{mult[d]:.1f}" for d in DIRECTIONS], textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="流量 (vph)")
    st.plotly_chart(style_fig(fig, height=320, title="流量倍率效果预览"), width="stretch")

spec_path = SPECS_DIR / f"{safe_name}.json"
sv1, sv2 = st.columns([1, 3])
if sv1.button("💾 保存 Spec", width="stretch"):
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    sv2.success(f"已保存到 `{spec_path.relative_to(_ROOT)}`")
elif spec_path.exists():
    sv2.caption(f"`{spec_path.relative_to(_ROOT)}` 已存在，点击保存将覆盖。")

# ================================================================ 4. 运行

st.subheader("4️⃣ 运行 What-if 仿真")

r1, r2 = st.columns([2, 1])
strategy = r1.selectbox("信号控制策略", STRATEGIES, format_func=lambda s: STRATEGY_LABELS[s])
run_seed = r2.number_input("种子", 0, 9999, 0)

if st.button("🚀 运行推演", type="primary"):
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd = [PYTHON, "-m", "experiments.scenario_runner",
           "--scenario", str(spec_path),
           "--strategy", strategy,
           "--seed", str(int(run_seed))]
    run_cli(cmd, f"What-if 推演（{safe_name} · {STRATEGY_LABELS[strategy]}）")

# ================================================================ 5. 与 baseline 对比

st.subheader("5️⃣ What-if vs Baseline 对比")


def find_exp(scenario: str, strat: str) -> Path | None:
    """找 <scenario>__<strategy>__s* 中 metrics 存在的最新目录。"""
    if not EXPERIMENTS_DIR.exists():
        return None
    cands = [d for d in EXPERIMENTS_DIR.glob(f"{scenario}__{strat}__s*")
             if (d / "metrics_summary.json").exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


whatif_dir = find_exp(safe_name, strategy)

baseline_names = ["normal", "baseline", "whatif_normal"]
baseline_dir = None
for bn in baseline_names:
    baseline_dir = find_exp(bn, strategy)
    if baseline_dir:
        break

if whatif_dir is None:
    empty_state(f"暂无本场景（`{safe_name}` × {STRATEGY_LABELS[strategy]}）的实验结果。"
                "点击上方「运行推演」。")
else:
    wm = load_json(whatif_dir / "metrics_summary.json") or {}
    bm = load_json(baseline_dir / "metrics_summary.json") if baseline_dir else None

    if bm is None:
        st.info(f"未找到 baseline（{'/'.join(baseline_names)} × 同策略）结果，只展示本次 what-if。"
                "可先用「Normal 平峰」预设跑一次作为基准。", icon="ℹ️")
        cols = st.columns(4)
        keys = [k for k in METRIC_LABELS if k in wm]
        for i, k in enumerate(keys):
            cols[i % 4].metric(METRIC_LABELS[k], fmt_metric(k, wm.get(k)), border=True)
    else:
        st.caption(f"What-if：`{whatif_dir.name}` · Baseline：`{baseline_dir.name}`")
        keys = [k for k in METRIC_LABELS if k in wm and k in bm]

        cols = st.columns(4)
        for i, k in enumerate(keys[:4]):
            pct = improvement_pct(wm[k], bm[k], k)
            cols[i].metric(
                METRIC_LABELS[k], fmt_metric(k, wm[k]),
                delta=f"{pct:+.1f}% vs baseline" if pct is not None else None,
                delta_color="normal", border=True,
                help=f"baseline: {fmt_metric(k, bm[k])}")

        fig = go.Figure()
        # 数值跨度大（等待秒 vs 通过千辆），用归一化百分比对比更可读
        ratio = []
        for k in keys:
            base = bm[k]
            ratio.append((wm[k] / base * 100.0 - 100.0) if base else 0.0)
        colors = ["#E64B35" if ((r > 0) != (k in METRIC_HIGHER_BETTER)) and abs(r) > 1e-9
                  else "#00A087"
                  for r, k in zip(ratio, keys)]
        fig.add_bar(x=[METRIC_LABELS[k] for k in keys], y=ratio,
                    marker=dict(color=colors),
                    text=[f"{r:+.1f}%" for r in ratio], textposition="outside")
        fig.add_hline(y=0, line_color="#666666", line_width=1)
        fig.update_layout(yaxis_title="相对 baseline 变化 (%)")
        st.plotly_chart(style_fig(fig, height=380,
                                  title="What-if 相对 Baseline 的指标变化（绿=改善，红=恶化）"),
                        width="stretch")

    # 排队时序对比
    wts = load_timeseries(whatif_dir)
    bts = load_timeseries(baseline_dir) if baseline_dir else None
    if wts is not None and "queue_total" in wts.columns:
        fig = go.Figure()
        if bts is not None and "queue_total" in bts.columns:
            fig.add_scatter(x=bts["t"], y=bts["queue_total"], mode="lines",
                            name="Baseline (normal)",
                            line=dict(color="#8491B4", width=2, dash="dash"))
        fig.add_scatter(x=wts["t"], y=wts["queue_total"], mode="lines",
                        name=f"What-if ({safe_name})",
                        line=dict(color=NPG[0], width=2.4))
        fig.update_layout(xaxis_title="仿真时间 (s)", yaxis_title="总排队车辆数 (veh)")
        st.plotly_chart(style_fig(fig, height=380, title="总排队时序对比", unified_hover=True),
                        width="stretch")
