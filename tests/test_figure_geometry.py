from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import pytest

from scripts.visualization import fig10_scenario_timeline_atlas
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
