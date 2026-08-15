"""🎥 Traffic Vision — 视频感知：上传视频、配置计数线、运行分析、查看 TrafficState。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.common import (
    DIRECTION_COLORS, DIRECTIONS, NPG, PYTHON,
    STATES_DIR, VIDEOS_DIR, VISION_OUT_DIR,
    cv2_available, empty_state, load_json, page_setup, render_vision_state,
    run_cli,
)

page_setup("Traffic Vision 视频感知", "🎥")
st.title("🎥 Traffic Vision · 视频感知")
st.caption("上传路口视频 → 配置 ROI / 计数线 → 运行 `vision.analyze` → 生成 TrafficState（schema 1.1）")

# 云展示版不装 OpenCV/YOLO：不渲染上传/分析控件，展示本地预生成的 demo 分析结果。
if not cv2_available():
    st.info(
        "🚧 此为静态展示版，未安装视频分析依赖（OpenCV / YOLO）。"
        "以下展示的是本地已预生成的 `demo` TrafficState 分析结果；"
        "上传视频 + 运行分析需在本地完整环境体验。",
    )
    demo_state = load_json(STATES_DIR / "demo.json")
    if demo_state is not None:
        st.subheader("📼 预生成分析结果 · demo")
        render_vision_state(demo_state, "demo")
    else:
        st.warning("未找到预生成的 demo TrafficState。")
    st.stop()

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
    render_vision_state(state, scenario_id, video_path=annotated_out if annotated_out.exists() else None)
