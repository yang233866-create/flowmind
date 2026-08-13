"""Per-second metrics collection during a SUMO run + tripinfo aggregation."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

APPROACHES = ("north", "south", "east", "west")


def parse_tripinfo(tripinfo_path: str | Path,
                   timeseries: pd.DataFrame | None = None,
                   teleports: float = 0.0) -> dict:
    """Aggregate tripinfo.xml (+ optional timeseries) into the contract metrics dict.

    avg/max_queue_veh come from the timeseries `queue_total` column;
    teleports must be counted by the caller (traci) and passed in.
    """
    waiting, durations, speeds = [], [], []
    root = ET.parse(str(tripinfo_path)).getroot()
    for ti in root.iter("tripinfo"):
        dur = float(ti.get("duration", 0.0))
        length = float(ti.get("routeLength", 0.0))
        waiting.append(float(ti.get("waitingTime", 0.0)))
        durations.append(dur)
        if dur > 0:
            speeds.append(length / dur)

    has_ts = timeseries is not None and len(timeseries) > 0
    n = len(durations)
    return {
        "avg_waiting_s": float(sum(waiting) / n) if n else 0.0,
        "avg_travel_time_s": float(sum(durations) / n) if n else 0.0,
        "throughput_veh": float(n),
        "avg_queue_veh": float(timeseries["queue_total"].mean()) if has_ts else 0.0,
        "max_queue_veh": float(timeseries["queue_total"].max()) if has_ts else 0.0,
        "avg_speed_mps": float(sum(speeds) / len(speeds)) if speeds else 0.0,
        "teleports": float(teleports),
    }


class MetricsCollector:
    """Collects per-second queue/waiting stats via a traci-like handle.

    Two construction forms are supported:
      MetricsCollector(traci, {"north": "N_in", ...}, "TL")   # in_edges + tls_id
      MetricsCollector(traci, meta_dict)                       # template meta.json
    """

    def __init__(self, traci_mod, in_edges: dict, tls_id: str | None = None):
        self.traci = traci_mod
        if tls_id is None:
            meta = in_edges
            self.in_edges = {a: meta["approaches"][a]["in_edge"] for a in APPROACHES}
            self.tls_id = meta["tls_id"]
        else:
            self.in_edges = dict(in_edges)
            self.tls_id = tls_id
        self.rows: list[dict] = []
        self.teleports = 0

    def step(self, t: float) -> None:
        tr = self.traci
        row = {"t": t}
        total = 0
        waiting = 0.0
        for approach in APPROACHES:
            edge = self.in_edges[approach]
            q = tr.edge.getLastStepHaltingNumber(edge)
            row[f"queue_{approach}"] = q
            total += q
            waiting += tr.edge.getWaitingTime(edge)
        row["queue_total"] = total
        row["waiting_total"] = waiting
        row["running_veh"] = tr.vehicle.getIDCount()
        row["phase"] = tr.trafficlight.getPhase(self.tls_id)
        self.rows.append(row)
        self.teleports += tr.simulation.getStartingTeleportNumber()

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["t", "queue_north", "queue_south", "queue_east", "queue_west",
                "queue_total", "waiting_total", "running_veh", "phase"]
        return pd.DataFrame(self.rows, columns=cols)

    def parse_tripinfo(self, tripinfo_path: str | Path) -> dict:
        return parse_tripinfo(tripinfo_path, self.to_dataframe(), self.teleports)
