from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import pytest

from scripts.visualization import (
    fig10_scenario_timeline_atlas,
    fig11_perception_composition_flow,
)
from scripts.visualization.data_loader import load_visualization_data


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_visualization_data(ROOT)


def _capture_figure(module, bundle, tmp_path, monkeypatch):
    captured = {}

    def fake_export(figure, output_dir, stem):
        figure.canvas.draw()
        captured["figure"] = figure
        return {
            suffix: str(tmp_path / f"{stem}.{suffix}")
            for suffix in ("svg", "png", "pdf")
        }

    monkeypatch.setattr(module, "export_figure", fake_export)
    output_dir = tmp_path / "figures"
    source_dir = tmp_path / "source_data"
    output_dir.mkdir()
    source_dir.mkdir()

    paths = module.build(bundle, output_dir, source_dir)

    assert set(paths) == {"svg", "png", "pdf"}
    return captured["figure"]


def _intersection_area(first, second):
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _separation(first, second):
    horizontal = max(first.x0 - second.x1, second.x0 - first.x1, 0.0)
    vertical = max(first.y0 - second.y1, second.y0 - first.y1, 0.0)
    return (horizontal**2 + vertical**2) ** 0.5


def test_figure10_unit_label_has_zero_overlap_and_four_point_clearance(
    bundle, tmp_path, monkeypatch
):
    figure = _capture_figure(
        fig10_scenario_timeline_atlas, bundle, tmp_path, monkeypatch
    )
    try:
        renderer = figure.canvas.get_renderer()
        unit_labels = [text for text in figure.texts if text.get_text() == "队列（辆）"]
        assert len(unit_labels) == 1
        unit_bbox = unit_labels[0].get_window_extent(renderer=renderer)

        scenario_texts = [text for axis in figure.axes for text in axis.texts]
        assert len(scenario_texts) == 10
        for scenario_text in scenario_texts:
            scenario_bbox = scenario_text.get_window_extent(renderer=renderer)
            assert _intersection_area(unit_bbox, scenario_bbox) == 0

        clearance = figure.dpi * 4 / 72
        nearest_separation = min(
            _separation(unit_bbox, figure.axes[0].get_window_extent(renderer=renderer)),
            _separation(unit_bbox, figure.legends[0].get_window_extent(renderer=renderer)),
        )
        assert nearest_separation >= clearance

        figure_bbox = figure.bbox
        corners = (
            (unit_bbox.x0, unit_bbox.y0),
            (unit_bbox.x0, unit_bbox.y1),
            (unit_bbox.x1, unit_bbox.y0),
            (unit_bbox.x1, unit_bbox.y1),
        )
        assert all(
            figure_bbox.x0 <= x <= figure_bbox.x1
            and figure_bbox.y0 <= y <= figure_bbox.y1
            for x, y in corners
        )
    finally:
        plt.close(figure)


def _by_gid(axis, gid):
    matches = [artist for artist in axis.get_children() if artist.get_gid() == gid]
    assert len(matches) == 1
    return matches[0]


def test_figure11_option_b_node_geometry_is_exact(bundle, tmp_path, monkeypatch):
    figure = _capture_figure(
        fig11_perception_composition_flow, bundle, tmp_path, monkeypatch
    )
    try:
        axis = figure.axes[0]
        renderer = figure.canvas.get_renderer()
        expected = {
            "direction-north": (0.005, 0.25, 0.10),
            "direction-south": (0.005, 0.25, 0.10),
            "direction-east": (0.005, 0.25, 0.10),
            "direction-west": (0.005, 0.25, 0.10),
            "class-car": (0.76, 0.22, 0.10),
            "class-bus": (0.76, 0.22, 0.10),
            "class-truck": (0.76, 0.22, 0.10),
            "class-motorcycle": (0.76, 0.22, 0.10),
        }
        clearance = figure.dpi * 2 / 72
        rectangles = {}
        names = {}
        values = {}

        for key, (expected_x, expected_width, expected_height) in expected.items():
            rectangle = _by_gid(axis, f"{key}-box")
            name = _by_gid(axis, f"{key}-name")
            value = _by_gid(axis, f"{key}-value")
            rectangles[key] = rectangle
            names[key] = name
            values[key] = value

            assert isinstance(rectangle, Rectangle)
            assert rectangle.get_x() == pytest.approx(expected_x)
            assert rectangle.get_width() == pytest.approx(expected_width)
            assert rectangle.get_height() == pytest.approx(expected_height)
            assert name.get_position()[0] == pytest.approx(
                rectangle.get_x() + rectangle.get_width() / 2
            )
            assert value.get_position()[1] == pytest.approx(
                rectangle.get_y() - 0.018
            )
            assert value.get_fontsize() == pytest.approx(8)
            assert value.get_verticalalignment() == "top"

            rectangle_bbox = rectangle.get_window_extent(renderer=renderer)
            name_bbox = name.get_window_extent(renderer=renderer)
            value_bbox = value.get_window_extent(renderer=renderer)
            axis_bbox = axis.get_window_extent(renderer=renderer)

            assert name_bbox.x0 - rectangle_bbox.x0 >= clearance
            assert rectangle_bbox.x1 - name_bbox.x1 >= clearance
            assert name_bbox.y0 - rectangle_bbox.y0 >= clearance
            assert rectangle_bbox.y1 - name_bbox.y1 >= clearance
            assert rectangle_bbox.y0 - value_bbox.y1 >= clearance
            assert _intersection_area(rectangle_bbox, value_bbox) == 0
            assert axis_bbox.x0 <= value_bbox.x0 <= value_bbox.x1 <= axis_bbox.x1
            assert axis_bbox.y0 <= value_bbox.y0 <= value_bbox.y1 <= axis_bbox.y1

        for key, value in values.items():
            value_bbox = value.get_window_extent(renderer=renderer)
            for rectangle in rectangles.values():
                assert _intersection_area(
                    value_bbox, rectangle.get_window_extent(renderer=renderer)
                ) == 0
            for name in names.values():
                assert _intersection_area(
                    value_bbox, name.get_window_extent(renderer=renderer)
                ) == 0
            for other_key, other_value in values.items():
                if other_key != key:
                    assert _intersection_area(
                        value_bbox, other_value.get_window_extent(renderer=renderer)
                    ) == 0

        flow_paths = [
            patch for patch in axis.patches if isinstance(patch, PathPatch)
        ]
        assert flow_paths
        start_y_values = {0.82, 0.61, 0.36, 0.15}
        end_y_values = {0.80, 0.58, 0.36, 0.14}
        for flow_path in flow_paths:
            vertices = flow_path.get_path().vertices
            start_x, start_y = vertices[0]
            end_x, end_y = vertices[-1]
            assert 0.005 <= start_x <= 0.255
            assert 0.76 <= end_x <= 0.98
            assert any(start_y == pytest.approx(value) for value in start_y_values)
            assert any(end_y == pytest.approx(value) for value in end_y_values)
    finally:
        plt.close(figure)
