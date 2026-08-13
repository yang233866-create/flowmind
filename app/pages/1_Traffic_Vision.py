"""🎥 Traffic Vision — 视频感知：上传视频、配置计数线、运行分析、查看 TrafficState。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import (
    DIRECTION_COLORS, DIRECTION_LABELS, DIRECTIONS, FIGURES_DIR, NPG, PYTHON,
    STATES_DIR, VEHICLE_LABELS, VIDEOS_DIR, VISION_OUT_DIR,
    empty_state, load_json, page_setup, run_cli, style_fig,
)

page_setup("Traffic Vision 视频感知", "🎥")
st.title("🎥 Traffic Vision · 视频感知")
st.caption("上传路口视频 → 配置 ROI / 计数线 → 运行 `vision.analyze` → 生成 TrafficState（schema 1.1）")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def list_videos() -> list[Path]:
    if not VIDEOS_DIR.exists():
        return []
    return sorted(
        (p for p in VIDEOS_DIR.iterdir()
         if p.suffix.lower() in VIDEO_EXTS and "_annotated" not in p.stem),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )


@st.cache_data(show_spinner=False)
def first_frame(path: str, mtime: float):
    """返回视频第一帧（RGB ndarray），失败返回 None。mtime 参与缓存键。"""
    import cv2

    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_lines(obj, prefix: str = "") -> list[tuple[str, list]]:
    """从任意结构的 ROI JSON 里启发式提取线段 [(name, [[x1,y1],[x2,y2]])]。"""
    found: list[tuple[str, list]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += extract_lines(v, str(k) if not prefix else f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        if (len(obj) == 2 and all(
                isinstance(p, (list, tuple)) and len(p) == 2
                and all(isinstance(c, (int, float)) for c in p) for p in obj)):
            return [(prefix, [list(obj[0]), list(obj[1])])]
        if len(obj) == 4 and all(isinstance(c, (int, float)) for c in obj):
            return [(prefix, [[obj[0], obj[1]], [obj[2], obj[3]]])]
        for i, v in enumerate(obj):
            found += extract_lines(v, f"{prefix}[{i}]")
    return found


def draw_lines_on_frame(frame, lines: list[tuple[str, list]]):
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, (name, ((x1, y1), (x2, y2))) in enumerate(lines):
        color = NPG[i % len(NPG)]
        for d in DIRECTIONS:
            if d in name.lower():
                color = DIRECTION_COLORS[d]
                break
        draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
        r = 6
        for (x, y) in ((x1, y1), (x2, y2)):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        draw.text((min(x1, x2) + 6, min(y1, y2) - 16), name or f"line{i}", fill=color)
    return img


# ================================================================ 1. 视频来源

st.subheader("1️⃣ 选择或上传视频")
videos = list_videos()
src_col, up_col = st.columns([3, 2])
with up_col:
    uploaded = st.file_uploader("上传新视频（保存到 data/videos/）", type=[e[1:] for e in VIDEO_EXTS])
    if uploaded is not None:
        dest = VIDEOS_DIR / uploaded.name
        if not dest.exists() or dest.stat().st_size != uploaded.size:
            dest.write_bytes(uploaded.getbuffer())
            st.success(f"已保存到 {dest.relative_to(_ROOT)}")
            videos = list_videos()
with src_col:
    if not videos:
        empty_state("data/videos/ 下暂无视频。请先在右侧上传一段路口视频（俯拍/高点固定机位效果最佳）。")
        st.stop()
    default_idx = 0
    if uploaded is not None:
        for i, p in enumerate(videos):
            if p.name == uploaded.name:
                default_idx = i
                break
    video_path = st.selectbox("已有视频", videos, index=default_idx, format_func=lambda p: p.name)

frame = first_frame(str(video_path), video_path.stat().st_mtime)

# ================================================================ 2. ROI / 计数线

st.subheader("2️⃣ ROI / 计数线配置")

demo_roi = VIDEOS_DIR / "demo_roi.json"
video_roi = VIDEOS_DIR / f"{video_path.stem}_roi.json"
if video_roi.exists():
    roi_default = video_roi.read_text(encoding="utf-8")
elif demo_roi.exists():
    roi_default = demo_roi.read_text(encoding="utf-8")
else:
    h, w = (frame.shape[:2] if frame is not None else (720, 1280))
    roi_default = json.dumps({
        "count_lines": {
            "north": [[int(w * 0.35), int(h * 0.25)], [int(w * 0.65), int(h * 0.25)]],
            "south": [[int(w * 0.35), int(h * 0.75)], [int(w * 0.65), int(h * 0.75)]],
            "east": [[int(w * 0.75), int(h * 0.35)], [int(w * 0.75), int(h * 0.65)]],
            "west": [[int(w * 0.25), int(h * 0.35)], [int(w * 0.25), int(h * 0.65)]],
        }
    }, indent=2)
    st.caption("⚠️ 未找到 data/videos/demo_roi.json，以下为按画面比例生成的示例模板，请按实际路口调整。")

edit_col, prev_col = st.columns([2, 3])
with edit_col:
    roi_text = st.text_area("计数线 JSON（每方向一条线段 [[x1,y1],[x2,y2]]，方向=车流来自的进口道）",
                            roi_default, height=320)
    roi_obj, roi_err = None, None
    try:
        roi_obj = json.loads(roi_text)
    except json.JSONDecodeError as e:
        roi_err = str(e)
    if roi_err:
        st.error(f"JSON 解析失败：{roi_err}")
    else:
        st.success(f"JSON 有效 · 识别到 {len(extract_lines(roi_obj))} 条线段")
with prev_col:
    if frame is None:
        st.warning("无法读取视频第一帧（编码不受支持？），预览不可用，但仍可运行分析。")
    else:
        lines = extract_lines(roi_obj) if roi_obj else []
        img = draw_lines_on_frame(frame, lines) if lines else frame
        st.image(img, caption=f"{video_path.name} 第一帧 + 计数线预览", width="stretch")

# ================================================================ 3. 运行分析

st.subheader("3️⃣ 运行视频分析")

p1, p2, p3 = st.columns([2, 1, 1])
scenario_id = p1.text_input("场景 ID（TrafficState 文件名）", value=video_path.stem)
limit_frames = p2.toggle("限制帧数（快速试跑）", value=False)
max_frames = p3.number_input("最大帧数", 100, 100000, 1500, step=100, disabled=not limit_frames)

state_out = STATES_DIR / f"{scenario_id}.json"
annotated_out = VISION_OUT_DIR / f"{scenario_id}_annotated.mp4"

if st.button("🚀 开始分析", type="primary", disabled=roi_obj is None or not scenario_id):
    video_roi.write_text(roi_text, encoding="utf-8")
    cmd = [PYTHON, "-m", "vision.analyze",
           "--video", str(video_path),
           "--roi-config", str(video_roi),
           "--state-out", str(state_out),
           "--annotated-video", str(annotated_out),
           "--figures-dir", "figures"]
    if limit_frames:
        cmd += ["--max-frames", str(int(max_frames))]
    if run_cli(cmd, "视频分析"):
        st.balloons()

# ================================================================ 4. 结果展示

st.subheader("4️⃣ 分析结果 · TrafficState")

state = load_json(state_out)
if state is None:
    empty_state(f"尚未生成 `{state_out.relative_to(_ROOT)}`。配置计数线后点击「开始分析」。")
else:
    src = state.get("source", {})
    st.caption(
        f"scenario_id=`{state.get('scenario_id', '?')}` · schema {state.get('schema_version', '?')} · "
        f"时长 {state.get('duration_sec', '?')} s · 分析时间 {src.get('analyzed_at', '?')}"
    )
    approaches = state.get("approaches", {})

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
        veh_types = list(VEHICLE_LABELS)
        for i, vt in enumerate(veh_types):
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

    if annotated_out.exists():
        st.markdown("**🎬 标注视频**")
        st.video(str(annotated_out))
        st.caption("若浏览器无法播放（编码为 mp4v），可在本地播放器打开："
                   f"`{annotated_out.relative_to(_ROOT)}`")

    vision_figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    vision_figs = [p for p in vision_figs
                   if scenario_id.lower() in p.stem.lower() or "vision" in p.stem.lower()]
    if vision_figs:
        st.markdown("**📈 视频分析图表**")
        cols = st.columns(min(3, len(vision_figs)))
        for col, p in zip(cols * (len(vision_figs) // 3 + 1), vision_figs):
            with col:
                st.image(str(p), caption=p.name, width="stretch")
