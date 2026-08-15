"""Generate SUMO route files (.rou.xml) from a TrafficState JSON + template meta.

CLI:
    python -m simulation.route_generator --state <json> --template cross_basic \
        --out <dir> [--scenario <spec.json>] --seed 0
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
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


def flow_begin_times(route_path: str | Path) -> list[float]:
    """Departure times of every <flow>/<vehicle> in document order."""
    root = ET.parse(str(route_path)).getroot()
    times = []
    for el in root:
        val = el.get("begin") or el.get("depart")
        if val is not None:
            times.append(float(val))
    return times


def route_horizon(route_path: str | Path) -> float:
    """Last second at which the route file still injects demand."""
    root = ET.parse(str(route_path)).getroot()
    ends = []
    for el in root:
        val = el.get("end") or el.get("begin") or el.get("depart")
        if val is not None:
            ends.append(float(val))
    return max(ends) if ends else 0.0


def check_sorted(route_path: str | Path) -> None:
    """Raise if the route file would be silently truncated by SUMO.

    SUMO only warns ("Route file should be sorted by departure time, ignoring
    '<id>'!") and then drops every out-of-order element, so an unsorted file
    looks like a successful run on a fraction of the intended demand.
    """
    times = flow_begin_times(route_path)
    for i, (a, b) in enumerate(zip(times, times[1:])):
        if b < a:
            raise ValueError(
                f"{route_path} is not sorted by departure time "
                f"(element {i + 1} begins at {b} after {a}); SUMO would silently "
                "drop it. Regenerate with simulation.route_generator."
            )


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
    """Return [(begin, end, vph), ...] for one approach, covering [0, duration).

    `flow_profile` only describes the *observed* window (for a 21.5 s clip with
    profile_bins_sec=5 that is 21.5 s of a 1800 s episode). Beyond it we fall back
    to the approach's aggregate `flow_vph` in a single bin. Holding the last
    profile bin instead would freeze whatever value the video happened to end on
    -- typically 0.0 -- and collapse the whole episode's demand to zero.

    The profile is clamped to the state's own `duration_sec`: the final bin is
    usually partial (21.5 s / 5 s leaves 1.5 s), and its vph was computed over
    that short span, so replaying it for a full `profile_bins_sec` would inject
    several times the vehicles the camera actually saw.
    """
    app = (state.get("approaches") or {}).get(approach) or {}
    base_vph = float(app.get("flow_vph") or DEFAULT_FLOW_VPH)
    profile = (state.get("flow_profile") or {}).get(approach)
    # A profile with no usable bin width has no time axis; assuming a default
    # would stretch e.g. a 25 s observation over 1500 s, so drop to the
    # aggregate rate instead of inventing one.
    raw_bin = state.get("profile_bins_sec")
    bin_sec = float(raw_bin) if raw_bin else 0.0
    if not profile or bin_sec <= 0:
        return [(0.0, duration, base_vph)]

    observed = float(state.get("duration_sec") or 0.0)
    horizon = min(duration, observed) if observed > 0 else duration

    bins: list[tuple[float, float, float]] = []
    for i, vph in enumerate(profile):
        begin = i * bin_sec
        if begin >= horizon:
            break
        bins.append((begin, min(begin + bin_sec, horizon), float(vph)))

    covered = bins[-1][1] if bins else 0.0
    if covered < duration:
        bins.append((covered, duration, base_vph))
    return bins


def generate_routes(state_path: str | Path, template: str, out_dir: str | Path,
                    scenario: str | Path | dict | None = None, seed: int = 0,
                    duration_sec: float | None = None) -> Path:
    """Build <out_dir>/routes.rou.xml + gen_meta.json. Returns route file path.

    `duration_sec` overrides both the scenario spec and the TrafficState; pass it
    when the consumer's horizon is authoritative (e.g. RL training episodes),
    otherwise the flows can stop long before the episode does.
    """
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    meta = load_meta(template)
    spec = {}
    if isinstance(scenario, dict):
        spec = scenario
    elif scenario is not None:
        spec = json.loads(Path(scenario).read_text(encoding="utf-8"))

    multipliers = spec.get("flow_multipliers") or {}
    lane_closures = spec.get("lane_closures") or []
    duration = float(duration_sec or spec.get("duration_sec")
                     or state.get("duration_sec") or 1800.0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ['<routes>']
    for vt, p in VTYPES.items():
        lines.append(
            f'    <vType id="{vt}" vClass="{VCLASS[vt]}" length="{p["length"]}" '
            f'accel="{p["accel"]}" decel="{p["decel"]}" maxSpeed="{p["maxSpeed"]}" '
            f'sigma="{p["sigma"]}"/>')

    # (begin, flow_id, xml) -- collected first, emitted sorted by begin below
    flows: list[tuple[float, str, str]] = []
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
                    flows.append((begin, fid,
                        f'    <flow id={quoteattr(fid)} type="{vt}" '
                        f'from="{in_edge}" to="{to_edge}" '
                        f'begin="{begin:.0f}" end="{end:.0f}" '
                        f'vehsPerHour="{rate:.3f}" '
                        f'departLane="best" departSpeed="max"/>'))

    # SUMO requires route input sorted by departure time and *silently ignores*
    # every out-of-order element (warning only, easy to miss with
    # --no-warnings). Emitting per approach would interleave begin times as
    # 0,5,10,.. per approach and drop ~90% of the demand, so sort here.
    flows.sort(key=lambda f: (f[0], f[1]))
    lines.extend(xml for _, _, xml in flows)
    lines.append('</routes>')

    route_path = out_dir / "routes.rou.xml"
    route_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    check_sorted(route_path)

    gen_meta = {
        "state": str(state_path),
        "scenario_id": state.get("scenario_id"),
        "template": template,
        "scenario_spec": str(scenario) if scenario else None,
        "flow_multipliers": multipliers,
        "lane_closures": lane_closures,
        "duration_sec": duration,
        "n_flows": len(flows),
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
    ap.add_argument("--duration", type=float, default=None,
                    help="episode horizon in seconds; overrides the scenario spec "
                         "and the TrafficState's own duration_sec")
    args = ap.parse_args()
    path = generate_routes(args.state, args.template, args.out,
                           scenario=args.scenario, seed=args.seed,
                           duration_sec=args.duration)
    print(f"routes written: {path}")


if __name__ == "__main__":
    main()
