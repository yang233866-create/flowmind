"""Generate SUMO route files (.rou.xml) from a TrafficState JSON + template meta.

CLI:
    python -m simulation.route_generator --state <json> --template cross_basic \
        --out <dir> [--scenario <spec.json>] --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

APPROACHES = ("north", "south", "east", "west")
MOVEMENTS = ("left", "straight", "right")

DEFAULT_TURNING = {"left": 0.15, "straight": 0.70, "right": 0.15}
DEFAULT_MIX = {"car": 1.0, "bus": 0.0, "truck": 0.0, "motorcycle": 0.0}
DEFAULT_FLOW_VPH = 400.0

# length[m], accel[m/s^2], decel[m/s^2], maxSpeed[m/s], sigma
VTYPES = {
    "car": dict(length=4.5, accel=2.6, decel=4.5, maxSpeed=33.3, sigma=0.5),
    "bus": dict(length=12.0, accel=1.2, decel=4.0, maxSpeed=22.2, sigma=0.5),
    "truck": dict(length=9.0, accel=1.3, decel=4.0, maxSpeed=25.0, sigma=0.5),
    "motorcycle": dict(length=2.2, accel=3.5, decel=6.0, maxSpeed=30.0, sigma=0.5),
}
VCLASS = {"car": "passenger", "bus": "bus", "truck": "truck",
          "motorcycle": "motorcycle"}


def load_meta(template: str) -> dict:
    meta_path = TEMPLATES_DIR / template / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"unknown template '{template}': {meta_path} missing")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _turning(state: dict, approach: str) -> dict:
    ratios = (state.get("turning_ratio") or {}).get(approach) or {}
    result = {}
    for mv in MOVEMENTS:
        val = ratios.get(mv)
        result[mv] = DEFAULT_TURNING[mv] if val is None else float(val)
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    return result


def _vehicle_mix(app_state: dict) -> dict:
    mix = app_state.get("vehicle_mix") or DEFAULT_MIX
    out = {k: float(mix.get(k, 0.0)) for k in VTYPES}
    total = sum(out.values())
    if total <= 0:
        return dict(DEFAULT_MIX)
    return {k: v / total for k, v in out.items()}


def _flow_bins(state: dict, approach: str, duration: float) -> list[tuple[float, float, float]]:
    """Return [(begin, end, vph), ...] for one approach."""
    app = (state.get("approaches") or {}).get(approach) or {}
    base_vph = float(app.get("flow_vph") or DEFAULT_FLOW_VPH)
    profile = (state.get("flow_profile") or {}).get(approach)
    bin_sec = float(state.get("profile_bins_sec") or 300)
    if not profile:
        return [(0.0, duration, base_vph)]
    bins = []
    t = 0.0
    i = 0
    while t < duration:
        end = min(t + bin_sec, duration)
        # if the profile is shorter than duration, hold the last bin value
        vph = float(profile[min(i, len(profile) - 1)])
        bins.append((t, end, vph))
        t = end
        i += 1
    return bins


def generate_routes(state_path: str | Path, template: str, out_dir: str | Path,
                    scenario: str | Path | dict | None = None, seed: int = 0) -> Path:
    """Build <out_dir>/routes.rou.xml + gen_meta.json. Returns route file path."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    meta = load_meta(template)
    spec = {}
    if isinstance(scenario, dict):
        spec = scenario
    elif scenario is not None:
        spec = json.loads(Path(scenario).read_text(encoding="utf-8"))

    multipliers = spec.get("flow_multipliers") or {}
    lane_closures = spec.get("lane_closures") or []
    duration = float(spec.get("duration_sec")
                     or state.get("duration_sec") or 1800.0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ['<routes>']
    for vt, p in VTYPES.items():
        lines.append(
            f'    <vType id="{vt}" vClass="{VCLASS[vt]}" length="{p["length"]}" '
            f'accel="{p["accel"]}" decel="{p["decel"]}" maxSpeed="{p["maxSpeed"]}" '
            f'sigma="{p["sigma"]}"/>')

    for approach in APPROACHES:
        app_state = (state.get("approaches") or {}).get(approach) or {}
        in_edge = meta["approaches"][approach]["in_edge"]
        mult = float(multipliers.get(approach, 1.0))
        turning = _turning(state, approach)
        mix = _vehicle_mix(app_state)
        for begin, end, vph in _flow_bins(state, approach, duration):
            for mv in MOVEMENTS:
                to_edge = meta["movements"][approach][mv]
                for vt, share in mix.items():
                    rate = vph * mult * turning[mv] * share
                    if rate <= 0.01:
                        continue
                    fid = f"f_{approach}_{mv}_{vt}_{int(begin)}"
                    lines.append(
                        f'    <flow id={quoteattr(fid)} type="{vt}" '
                        f'from="{in_edge}" to="{to_edge}" '
                        f'begin="{begin:.0f}" end="{end:.0f}" '
                        f'vehsPerHour="{rate:.3f}" '
                        f'departLane="best" departSpeed="max"/>')
    lines.append('</routes>')

    route_path = out_dir / "routes.rou.xml"
    route_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gen_meta = {
        "state": str(state_path),
        "scenario_id": state.get("scenario_id"),
        "template": template,
        "scenario_spec": str(scenario) if scenario else None,
        "flow_multipliers": multipliers,
        "lane_closures": lane_closures,
        "duration_sec": duration,
        "seed": seed,
    }
    (out_dir / "gen_meta.json").write_text(
        json.dumps(gen_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return route_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SUMO routes from TrafficState")
    ap.add_argument("--state", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = generate_routes(args.state, args.template, args.out,
                           scenario=args.scenario, seed=args.seed)
    print(f"routes written: {path}")


if __name__ == "__main__":
    main()
