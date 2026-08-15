"""FlowMind app 公共工具：路径常量、数据读取、子进程流式运行、Plotly 样式。

所有页面只依赖 docs/CONTRACTS.md 定义的文件格式，不 import 其他模块的代码
（experiments.scenarios 例外，用 try/except 兜底）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- 路径常量

ROOT = Path(__file__).resolve().parents[1]

_venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_py) if _venv_py.exists() else sys.executable

DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
STATES_DIR = DATA_DIR / "traffic_states"
RESULTS_DIR = DATA_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
SPECS_DIR = RESULTS_DIR / "scenario_specs"
VISION_OUT_DIR = RESULTS_DIR / "vision"
TWIN_OUT_DIR = RESULTS_DIR / "twin"
ARENA_CSV = RESULTS_DIR / "arena_summary.csv"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
TEMPLATES_DIR = ROOT / "simulation" / "templates"


def ensure_dirs() -> None:
    for d in (VIDEOS_DIR, STATES_DIR, EXPERIMENTS_DIR, SPECS_DIR,
              VISION_OUT_DIR, TWIN_OUT_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 配色与文案（与 scripts/sci_style.py 保持一致）

NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]

STRATEGY_COLORS = {
    "fixed": "#3C5488",
    "actuated": "#4DBBD5",
    "dqn": "#E64B35",
    "ppo": "#00A087",
}
STRATEGY_LABELS = {
    "fixed": "Fixed-Time 定时",
    "actuated": "Actuated 感应",
    "dqn": "DQN",
    "ppo": "PPO",
}
STRATEGIES = list(STRATEGY_COLORS)

DIRECTION_COLORS = {
    "north": "#3C5488",
    "south": "#4DBBD5",
    "east": "#E64B35",
    "west": "#00A087",
}
DIRECTIONS = ["north", "south", "east", "west"]
DIRECTION_LABELS = {
    "north": "北 North",
    "south": "南 South",
    "east": "东 East",
    "west": "西 West",
}

TEMPLATES = ["cross_basic", "cross_leftturn", "arterial_minor"]
TEMPLATE_LABELS = {
    "cross_basic": "cross_basic · 十字路口（每进口 2 车道）",
    "cross_leftturn": "cross_leftturn · 十字路口（专用左转道 + 保护左转相位）",
    "arterial_minor": "arterial_minor · 主干 3 车道 × 次干 1 车道",
}

METRIC_LABELS = {
    "avg_waiting_s": "平均等待时间 (s)",
    "avg_travel_time_s": "平均行程时间 (s)",
    "throughput_veh": "通过车辆数 (veh)",
    "avg_queue_veh": "平均排队 (veh)",
    "max_queue_veh": "最大排队 (veh)",
    "avg_speed_mps": "平均车速 (m/s)",
    "teleports": "瞬移次数",
}
# 数值越大越好的指标；其余越小越好
METRIC_HIGHER_BETTER = {"throughput_veh", "avg_speed_mps"}
METRIC_INT = {"throughput_veh", "teleports"}

VEHICLE_LABELS = {"car": "小汽车", "bus": "公交", "truck": "货车", "motorcycle": "摩托车"}


def fmt_metric(key: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if key in METRIC_INT:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def improvement_pct(value: float, baseline: float, key: str) -> float | None:
    """相对 baseline 的改善百分比，正数=更好。"""
    try:
        if baseline is None or value is None or pd.isna(baseline) or pd.isna(value) or baseline == 0:
            return None
    except TypeError:
        return None
    change = (value - baseline) / abs(baseline) * 100.0
    return change if key in METRIC_HIGHER_BETTER else -change


# ---------------------------------------------------------------- 页面骨架

_CSS = """
<style>
.fm-hero {
  background: linear-gradient(120deg, #3C5488 0%, #4DBBD5 55%, #00A087 100%);
  border-radius: 14px; padding: 26px 32px; color: #fff; margin-bottom: 6px;
}
.fm-hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; color: #fff; }
.fm-hero p { margin: 0; opacity: .92; font-size: 1.0rem; }
.fm-flow { display: flex; align-items: stretch; gap: 8px; margin: 12px 0 4px 0; flex-wrap: wrap; }
.fm-node {
  flex: 1 1 140px; background: var(--secondary-background-color, #f5f6fa);
  border: 1px solid rgba(60,84,136,.18); border-radius: 12px;
  padding: 12px 10px; text-align: center; min-width: 130px;
}
.fm-node-icon { font-size: 1.5rem; }
.fm-node-title { font-weight: 700; margin-top: 4px; }
.fm-node-sub { font-size: .78rem; opacity: .7; margin-top: 2px; }
.fm-arrow { align-self: center; font-size: 1.2rem; color: #3C5488; opacity: .6; }
div[data-testid="stMetric"] { border-radius: 12px; }
</style>
"""


def page_setup(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} · FlowMind AI", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    ensure_dirs()
    with st.sidebar:
        st.markdown("## 🚦 FlowMind AI")
        st.caption(
            "视频感知 → TrafficState → SUMO 数字孪生 → 多策略信号控制对比 → What-if 推演。"
            "纯软件交通信控优化平台。"
        )
        st.divider()


def safe_page_link(page: str, label: str, icon: str | None = None) -> None:
    """st.page_link 在单页运行/测试环境下会抛异常，做兜底。"""
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.caption(f"{icon or ''} {label}")


def empty_state(message: str, link: str | None = None, link_label: str = "") -> None:
    st.info(message, icon="💡")
    if link:
        safe_page_link(link, label=link_label or link, icon="👉")


# ---------------------------------------------------------------- 数据读取

def cv2_available() -> bool:
    """OpenCV 是否可用。云展示版不装 opencv，视频页据此优雅降级。"""
    from importlib.util import find_spec

    return find_spec("cv2") is not None


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def list_state_files() -> list[Path]:
    if not STATES_DIR.exists():
        return []
    return sorted(STATES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_states() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in list_state_files():
        data = load_json(p)
        if isinstance(data, dict) and "approaches" in data:
            out[p.stem] = data
    return out


def load_arena() -> pd.DataFrame | None:
    if not ARENA_CSV.exists():
        return None
    try:
        df = pd.read_csv(ARENA_CSV)
    except Exception:
        return None
    return df if len(df) else None


def arena_metric_cols(df: pd.DataFrame) -> list[str]:
    return [k for k in METRIC_LABELS if k in df.columns]


def comparable_panel(df: pd.DataFrame, group: str = "strategy",
                     cells: tuple[str, ...] = ("scenario", "seed"),
                     ) -> tuple[pd.DataFrame, list[str]]:
    """把 arena 表裁剪成"可以互相排名"的子集，并说明剔除了什么。

    两件事会让"按平均等待时间排名"变得没有意义：
    1) 面板不平衡 —— 只跑了简单场景的策略会凭构造获胜；
    2) 来源混杂 —— 不同 TrafficState 生成的运行本来就不可比。
    返回 (可用子集, 提示信息列表)。
    """
    notes: list[str] = []
    if df is None or df.empty or group not in df.columns:
        return df, notes

    if "base_state_sha1" in df.columns and df["base_state_sha1"].notna().any():
        shas = sorted(df["base_state_sha1"].dropna().unique())
        if len(shas) > 1:
            keep = df["base_state_sha1"].value_counts().idxmax()
            notes.append(f"发现 {len(shas)} 个不同的基准 TrafficState，"
                         f"仅保留 `{str(keep)[:8]}`（其余不可比）")
            df = df[df["base_state_sha1"] == keep]

    key = [c for c in cells if c in df.columns]
    if key and not df.empty:
        n_groups = df[group].nunique(dropna=True)
        per_cell = df.groupby(key)[group].nunique()
        full = {t if isinstance(t, tuple) else (t,)
                for t in per_cell[per_cell == n_groups].index}
        if len(full) < len(per_cell):
            notes.append(f"{len(per_cell) - len(full)} 个 {'/'.join(key)} 组合"
                         f"缺少部分策略，已从排名中剔除（保证同条件对比）")
            if full:
                mask = df[key].apply(lambda r: tuple(r) in full, axis=1)
                df = df[mask]
            else:
                df = df.iloc[0:0]
    return df, notes


_EXP_ID_RE = re.compile(r"^(?P<scenario>.+)__(?P<strategy>[A-Za-z0-9]+)__s(?P<seed>\d+)$")


def parse_exp_id(name: str) -> dict | None:
    m = _EXP_ID_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    return d


def list_experiments() -> list[dict]:
    """扫描 data/results/experiments/，返回 [{exp_id, scenario, strategy, seed, dir, metrics}]。"""
    if not EXPERIMENTS_DIR.exists():
        return []
    rows = []
    for d in sorted(EXPERIMENTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        info = parse_exp_id(d.name) or {"scenario": d.name, "strategy": "?", "seed": -1}
        info["exp_id"] = d.name
        info["dir"] = d
        info["metrics"] = load_json(d / "metrics_summary.json")
        rows.append(info)
    return rows


def load_timeseries(exp_dir: Path) -> pd.DataFrame | None:
    p = Path(exp_dir) / "timeseries.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    return df if len(df) else None


def load_templates_meta() -> dict[str, dict | None]:
    return {name: load_json(TEMPLATES_DIR / name / "meta.json") for name in TEMPLATES}


def known_scenarios() -> list[str]:
    """场景名：优先 experiments.scenarios 注册表，失败则扫描已有产物。"""
    names: list[str] = []
    try:
        import importlib

        mod = importlib.import_module("experiments.scenarios")
        for attr in ("SCENARIOS", "SCENARIO_REGISTRY", "REGISTRY", "ALL_SCENARIOS", "scenarios"):
            obj = getattr(mod, attr, None)
            if isinstance(obj, dict):
                names.extend(str(k) for k in obj)
                break
            if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, str) for x in obj):
                names.extend(obj)
                break
        else:
            fn = getattr(mod, "list_scenarios", None)
            if callable(fn):
                names.extend(str(x) for x in fn())
    except Exception:
        pass
    if SPECS_DIR.exists():
        names.extend(p.stem for p in sorted(SPECS_DIR.glob("*.json")))
    for row in list_experiments():
        names.append(row["scenario"])
    seen: set[str] = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


# ---------------------------------------------------------------- 子进程运行（流式输出）

def stream_command(cmd: list[str], log_placeholder, max_lines: int = 300) -> tuple[int, list[str]]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
    except OSError as e:
        log_placeholder.code(f"启动失败: {e}", language="text")
        return -1, [str(e)]
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_placeholder.code("\n".join(lines[-max_lines:]) or "…", language="text")
    proc.wait()
    return proc.returncode, lines


def run_cli(cmd: list[str], title: str) -> bool:
    """展示命令 + st.status 流式日志，返回是否成功。"""
    st.code(subprocess.list2cmdline(cmd), language="bash")
    with st.status(f"{title}…", expanded=True) as status:
        placeholder = st.empty()
        code, _ = stream_command(cmd, placeholder)
        if code == 0:
            status.update(label=f"✅ {title} 完成", state="complete", expanded=False)
        else:
            status.update(label=f"❌ {title} 失败（退出码 {code}）", state="error", expanded=True)
    return code == 0


# ---------------------------------------------------------------- Plotly 样式

def style_fig(fig, height: int = 380, title: str | None = None, unified_hover: bool = False):
    fig.update_layout(
        template="plotly_white",
        colorway=NPG,
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=52 if title else 30, b=10),
        font=dict(family="Times New Roman, Georgia, serif", size=13, color="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified" if unified_hover else "closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(60,84,136,.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------- 视频分析结果展示

def render_vision_state(state: dict, scenario_id: str, video_path: Path | None = None):
    """渲染一份 TrafficState 的分析结果（流量卡片 / 车型构成 / 转向表 / 时序 / 视频）。

    纯展示函数：不依赖 OpenCV/YOLO，云端展示版与本地完整版共用。
    video_path 仅用于 st.video 播放标注视频，为 None 时跳过该块。
    """
    import plotly.graph_objects as go

    approaches = state.get("approaches", {})
    src = state.get("source", {})

    # 顶部说明（隐藏本地源视频的绝对路径，云端不存在）
    _src = {k: v for k, v in src.items() if k != "video"}
    cap = f"scenario_id=`{state.get('scenario_id', scenario_id)}` · schema {state.get('schema_version', '?')}"
    if _src:
        cap += " · 时长 {:.2f} s".format(state.get("duration_sec", 0.0)) if state.get("duration_sec") else ""
        cap += " · 分析时间 {}".format(_src.get("analyzed_at", "?"))
    st.caption(cap)

    mcols = st.columns(4)
    for col, d in zip(mcols, DIRECTIONS):
        a = approaches.get(d, {}) or {}
        flow = a.get("flow_vph")
        observed = a.get("observed", False)
        col.metric(
            f"{'🟢' if observed else '⚪'} {DIRECTION_LABELS[d]}",
            f"{flow:,.0f} vph" if isinstance(flow, (int, float)) else "—",
            delta=None if observed else "未观测（默认/对侧值）",
            delta_color="off",
            help=f"排队估计: {a.get('queue_est', '—')} veh", border=True,
        )

    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        fig = go.Figure()
        for i, vt in enumerate(list(VEHICLE_LABELS)):
            fig.add_bar(
                x=[DIRECTION_LABELS[d] for d in DIRECTIONS],
                y=[(approaches.get(d, {}).get("vehicle_mix") or {}).get(vt, 0) * 100
                   for d in DIRECTIONS],
                name=f"{VEHICLE_LABELS[vt]} {vt}",
                marker=dict(color=NPG[i % len(NPG)], line=dict(color="#ffffff", width=1)),
            )
        fig.update_layout(barmode="stack", yaxis_title="构成占比 (%)")
        st.plotly_chart(style_fig(fig, height=360, title="各进口道车型构成"), width="stretch")
    with table_col:
        rows = []
        for d in DIRECTIONS:
            a = approaches.get(d, {}) or {}
            tr = (state.get("turning_ratio") or {}).get(d, {}) or {}
            rows.append({
                "进口道": DIRECTION_LABELS[d],
                "流量 (vph)": a.get("flow_vph"),
                "排队估计 (veh)": a.get("queue_est"),
                "左转": tr.get("left"), "直行": tr.get("straight"), "右转": tr.get("right"),
                "观测": "✔" if a.get("observed") else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("转向比缺省 0.15 / 0.70 / 0.15（契约默认值）")

    profile = state.get("flow_profile") or {}
    bins_sec = state.get("profile_bins_sec") or 300
    if any(profile.get(d) for d in DIRECTIONS):
        fig = go.Figure()
        for d in DIRECTIONS:
            ys = profile.get(d) or []
            if ys:
                fig.add_scatter(
                    x=[i * bins_sec / 60 for i in range(len(ys))], y=ys,
                    mode="lines+markers", name=DIRECTION_LABELS[d],
                    line=dict(color=DIRECTION_COLORS[d], width=2), marker=dict(size=8),
                )
        fig.update_layout(xaxis_title=f"时间 (min, 每 {bins_sec}s 一档)", yaxis_title="流量 (vph)")
        st.plotly_chart(style_fig(fig, height=340, title="流量随时间变化", unified_hover=True),
                        width="stretch")

    with st.expander("📄 原始 TrafficState JSON"):
        st.json(state)

    if video_path is not None and Path(video_path).exists():
        st.markdown("**🎬 标注视频**")
        st.video(str(video_path))

    vision_figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    vision_figs = [p for p in vision_figs
                   if scenario_id.lower() in p.stem.lower() or "vision" in p.stem.lower()]
    if vision_figs:
        st.markdown("**📈 视频分析图表**")
        cols = st.columns(min(3, len(vision_figs)))
        for col, p in zip(cols * (len(vision_figs) // 3 + 1), vision_figs):
            with col:
                st.image(str(p), caption=p.name, width="stretch")
