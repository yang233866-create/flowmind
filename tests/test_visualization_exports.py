from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from scripts.visualization.design_system import FIGURE_STEMS


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
EXPECTED_STEMS = (
    "01_vision_to_twin",
    "02_strategy_tradeoffs",
    "03_scenario_robustness",
    "04_queue_dynamics",
    "05_training_evidence",
    "06_decision_map",
    "07_regret_landscape",
    "08_paired_transitions",
    "09_operating_state_density",
    "10_scenario_timeline_atlas",
    "11_perception_composition_flow",
)


def test_formal_figure_contract_contains_eleven_distinct_stems():
    assert FIGURE_STEMS == EXPECTED_STEMS
    assert len(set(FIGURE_STEMS)) == 11


def test_all_formal_figures_have_three_export_formats():
    missing = []
    for stem in FIGURE_STEMS:
        for suffix in (".png", ".svg", ".pdf"):
            path = FIGURE_DIR / f"{stem}{suffix}"
            if not path.exists() or path.stat().st_size < 10_000:
                missing.append(str(path))
    assert not missing, "Missing or implausibly small outputs:\n" + "\n".join(missing)


def test_png_outputs_are_large_and_300_dpi():
    for stem in FIGURE_STEMS:
        with Image.open(FIGURE_DIR / f"{stem}.png") as image:
            assert image.width >= 3000, stem
            assert image.height >= 1800, stem
            dpi = image.info.get("dpi")
            assert dpi is not None, stem
            assert min(dpi) >= 295, (stem, dpi)


def test_svg_outputs_keep_editable_text():
    for stem in FIGURE_STEMS:
        path = FIGURE_DIR / f"{stem}.svg"
        root = ET.parse(path).getroot()
        texts = root.findall(".//{http://www.w3.org/2000/svg}text")
        assert len(texts) >= 8, f"{stem} text was outlined or missing"


def test_revised_pngs_have_white_outer_border_without_clipped_artists():
    for stem in (
        "01_vision_to_twin",
        "10_scenario_timeline_atlas",
        "11_perception_composition_flow",
    ):
        with Image.open(FIGURE_DIR / f"{stem}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        border = np.concatenate(
            (
                pixels[:2, :, :].reshape(-1, 3),
                pixels[-2:, :, :].reshape(-1, 3),
                pixels[:, :2, :].reshape(-1, 3),
                pixels[:, -2:, :].reshape(-1, 3),
            ),
            axis=0,
        )
        assert int(border.min()) >= 250, stem
