import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.visualization import (
    fig01_vision_to_twin,
    fig02_strategy_tradeoffs,
    fig03_scenario_robustness,
    fig04_queue_dynamics,
    fig06_decision_map,
    fig07_regret_landscape,
    fig08_paired_transitions,
    fig09_operating_state_density,
    fig10_scenario_timeline_atlas,
    fig11_perception_composition_flow,
)
from scripts.visualization import design_system


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
VIS_DIR = ROOT / "scripts" / "visualization"

EXPECTED_TITLES = {
    "01_vision_to_twin": "视觉感知与方向交通流输入构建",
    "02_strategy_tradeoffs": "不同控制策略的多指标性能比较",
    "03_scenario_robustness": "五类场景下相对 Fixed 的性能变化率",
    "04_queue_dynamics": "不同控制策略的队列动态特征",
    "05_training_evidence": "DQN 与 PPO 训练过程",
    "06_decision_map": "不同场景与指标下的最优策略分布",
    "07_regret_landscape": "不同场景下各策略的标准化后悔值",
    "08_paired_transitions": "不同策略相对 Fixed 的配对实验结果",
    "09_operating_state_density": "总排队—总等待运行状态密度",
    "10_scenario_timeline_atlas": "五类场景下的队列时序变化",
    "11_perception_composition_flow": "方向交通需求、车辆组成与数据来源",
}

MODULES = [
    fig01_vision_to_twin,
    fig02_strategy_tradeoffs,
    fig03_scenario_robustness,
    fig04_queue_dynamics,
    fig06_decision_map,
    fig07_regret_landscape,
    fig08_paired_transitions,
    fig09_operating_state_density,
    fig10_scenario_timeline_atlas,
    fig11_perception_composition_flow,
]

FORBIDDEN_REPORT_TEXT = (
    "FLOWMIND",
    "证据链",
    "读图结论",
    "证据解读",
    "机制解释",
    "决策结论",
    "动态结论",
    "机制结论",
    "数据诚实边界",
)


def test_shared_publication_design_tokens_are_exact():
    assert design_system.BACKGROUND == "#FFFFFF"
    assert design_system.SURFACE == "#FFFFFF"
    assert design_system.STRATEGY_COLORS == {
        "fixed": "#8B98A8",
        "actuated": "#1E9E8F",
        "dqn": "#3569D4",
        "ppo": "#E46F51",
    }
    assert design_system.OBSERVED_COLOR == "#0EA5A4"
    assert design_system.INFERRED_COLOR == "#E59A36"
    assert design_system.MAIN_TITLE_SIZE in range(16, 19)
    assert design_system.PANEL_TITLE_SIZE >= 11
    assert design_system.AXIS_LABEL_SIZE >= 10
    assert design_system.TICK_LABEL_SIZE >= 9
    assert design_system.LEGEND_SIZE >= 9


def test_every_figure_module_declares_its_neutral_title():
    expected = {k: v for k, v in EXPECTED_TITLES.items() if k != "05_training_evidence"}
    actual = {module.__name__.split(".")[-1].replace("fig", "", 1): module.FIGURE_TITLE for module in MODULES}
    assert actual == expected


def test_heatmap_and_provenance_comparison_contracts_are_public():
    assert fig03_scenario_robustness.HEATMAP_BENEFIT_LIMIT == 70
    assert fig01_vision_to_twin.OBSERVED_LINESTYLE == "-"
    assert fig01_vision_to_twin.INFERRED_LINESTYLE != "-"
    assert fig11_perception_composition_flow.OBSERVED_LINESTYLE == fig01_vision_to_twin.OBSERVED_LINESTYLE
    assert fig11_perception_composition_flow.INFERRED_LINESTYLE == fig01_vision_to_twin.INFERRED_LINESTYLE


def test_figure01_crop_same_frame_returns_an_independent_metadata_slice():
    image = np.arange(6 * 8 * 3).reshape(6, 8, 3)
    metadata = {"zoom": {"crop_xyxy": [2, 1, 7, 5]}}

    crop = fig01_vision_to_twin.crop_same_frame(image, metadata)

    np.testing.assert_array_equal(crop, image[1:5, 2:7])
    assert not np.shares_memory(crop, image)


def test_figure01_crop_same_frame_returns_none_without_zoom():
    image = np.arange(6 * 8 * 3).reshape(6, 8, 3)

    assert fig01_vision_to_twin.crop_same_frame(image, {"zoom": None}) is None


def test_figure01_draw_frame_evidence_renders_exact_same_frame_inset():
    image = np.arange(6 * 8 * 3).reshape(6, 8, 3)
    metadata = {
        "frame": {"timestamp_sec": 1.24},
        "no_detections": False,
        "zoom": {"track_id": 17, "crop_xyxy": [2, 1, 7, 5]},
    }
    figure, axis = plt.subplots()
    try:
        inset = fig01_vision_to_twin.draw_frame_evidence(axis, image, metadata)

        assert inset is not None
        assert len(axis.child_axes) == 1
        assert axis.child_axes[0] is inset
        np.testing.assert_array_equal(
            np.asarray(inset.images[0].get_array()),
            image[1:5, 2:7],
        )
        assert any(text.get_text() == "视频检测与跟踪示例" for text in axis.texts)
        assert any(
            text.get_text() == "自动代表帧 · t = 1.24 s"
            for text in axis.texts
        )
    finally:
        plt.close(figure)


def test_figure01_draw_frame_evidence_degrades_without_detections():
    image = np.arange(6 * 8 * 3).reshape(6, 8, 3)
    metadata = {
        "frame": {"timestamp_sec": 0.0},
        "no_detections": True,
        "zoom": None,
    }
    figure, axis = plt.subplots()
    try:
        inset = fig01_vision_to_twin.draw_frame_evidence(axis, image, metadata)

        assert inset is None
        assert len(axis.child_axes) == 0
        assert any(text.get_text() == "计数线布设示例" for text in axis.texts)
    finally:
        plt.close(figure)


def test_figure01_build_writes_exact_frame_provenance(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame.png"
    plt.imsave(frame_path, np.zeros((6, 8, 3), dtype=np.uint8))
    metadata = {
        "selector": {
            "name": "active-tracks-trace-points",
            "version": "active-tracks-trace-points-v1",
        },
        "frame": {"timestamp_sec": 0.0},
        "zoom": None,
        "no_detections": True,
        "note": "同帧证据",
    }
    approaches = {
        "north": {"observed": True, "flow_vph": 1200.0},
        "south": {"observed": True, "flow_vph": 1100.0},
        "east": {"observed": False, "flow_vph": 900.0},
        "west": {"observed": False, "flow_vph": 800.0},
    }
    data = SimpleNamespace(
        annotated_frame=frame_path,
        annotated_frame_meta=metadata,
        traffic_state={
            "approaches": approaches,
            "profile_bins_sec": 5,
            "flow_profile": {
                "north": [1200, 1300, 1250, 1400, 1350],
                "south": [1100, 1050, 1200, 1150, 1250],
                "east": [],
                "west": [],
            },
        },
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    def fake_export(figure, output_dir, stem):
        assert stem == "01_vision_to_twin"
        plt.close(figure)
        return {}

    monkeypatch.setattr(fig01_vision_to_twin, "export_figure", fake_export)

    fig01_vision_to_twin.build(data, tmp_path / "output", source_dir)

    assert (source_dir / "01_vision_frame_provenance.json").read_text(
        encoding="utf-8"
    ) == json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"


def test_rendered_svgs_use_neutral_titles_without_report_copy():
    for stem, title in EXPECTED_TITLES.items():
        text = (FIGURE_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        assert title in text, stem
        assert not any(value in text for value in FORBIDDEN_REPORT_TEXT), stem


def test_rendered_pngs_have_white_corners():
    for stem in EXPECTED_TITLES:
        with Image.open(FIGURE_DIR / f"{stem}.png") as image:
            rgb = image.convert("RGB")
            corners = (
                rgb.getpixel((0, 0)),
                rgb.getpixel((rgb.width - 1, 0)),
                rgb.getpixel((0, rgb.height - 1)),
                rgb.getpixel((rgb.width - 1, rgb.height - 1)),
            )
            assert all(min(pixel) >= 250 for pixel in corners), stem


def test_figure06_uses_count_bars_and_does_not_repeat_cell_text():
    source = (VIS_DIR / "fig06_decision_map.py").read_text(encoding="utf-8")
    assert "barh(" in source
    assert "ax_matrix.text(" not in source
    assert "imshow(" not in source


def test_figure06_svg_has_no_repeated_cell_strategy_names():
    svg = (FIGURE_DIR / "06_decision_map.svg").read_text(encoding="utf-8")
    assert svg.count(">Actuated</text>") == 0
    assert svg.count(">DQN</text>") <= 1
    assert svg.count(">PPO</text>") <= 1
