"""Scenario compiler: registry + routines + planted relations -> seeded event schedule.

Events are dicts-as-dataclasses on a 1-D time axis in seconds. The compiler is the
only place where randomness for the load schedule is drawn, so the same seed always
gives the same schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .registry import Registry


@dataclass(frozen=True)
class LoadEvent:
    appliance: str
    start_s: float
    end_s: float
    level_w: float
    origin: str = "routine"          # routine | trigger:<source> | anomaly:<kind>

    def overlaps(self, other: "LoadEvent") -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s


@dataclass(frozen=True)
class PlantedRelation:
    """Ground-truth relation instance, kept for evaluation only."""
    type: str
    source: str
    target: str
    delay_s: float | None = None


def _routine_events(reg: Registry, rng: np.random.Generator, duration_s: float, start_hour: float):
    events: list[LoadEvent] = []
    for app in reg.appliances:
        cfg = reg.routines.get(app.id)
        if not cfg:
            continue
        weights = np.asarray(cfg["hours"], dtype=float)
        lo, hi = cfg["duration_s"]
        n_steps = int(duration_s // 60)
        for step in range(n_steps):
            t = step * 60.0
            hour = int(((start_hour * 3600 + t) // 3600) % 24)
            w = weights[hour] / max(weights.mean(), 1e-9)
            p = cfg["rate_per_hour"] / 60.0 * w          # per-minute start probability
            if rng.random() < p:
                dur = float(rng.uniform(lo, hi))
                level = float(app.on_w * rng.uniform(0.95, 1.0))
                events.append(LoadEvent(app.id, t, min(t + dur, duration_s), level))
    return events


def _drop_overlapping_same_device(events: list[LoadEvent]) -> list[LoadEvent]:
    kept: list[LoadEvent] = []
    for ev in sorted(events, key=lambda e: e.start_s):
        if any(k.appliance == ev.appliance and k.overlaps(ev) for k in kept):
            continue
        kept.append(ev)
    return kept


def _apply_exclusions(events: list[LoadEvent], reg: Registry) -> list[LoadEvent]:
    pairs = [(r.params["a"], r.params["b"]) for r in reg.relations if r.type == "exclusion"]
    kept: list[LoadEvent] = []
    for ev in sorted(events, key=lambda e: e.start_s):
        clash = False
        for a, b in pairs:
            other = b if ev.appliance == a else a if ev.appliance == b else None
            if other and any(k.appliance == other and k.overlaps(ev) for k in kept):
                clash = True
                break
        if not clash:
            kept.append(ev)
    return kept


def _apply_safety(events: list[LoadEvent], reg: Registry, duration_s: float) -> list[LoadEvent]:
    """Drop (earliest-first kept) any event that would push a zone or the board over its cap."""
    n = int(np.ceil(duration_s / reg.sample_period_s))
    zone_load = {z.id: np.zeros(n) for z in reg.zones}
    total = np.zeros(n)
    kept: list[LoadEvent] = []
    for ev in sorted(events, key=lambda e: e.start_s):
        z = reg.appliance_zone(ev.appliance)
        a, b = int(ev.start_s / reg.sample_period_s), int(ev.end_s / reg.sample_period_s)
        if (zone_load[z.id][a:b].max(initial=0) + ev.level_w > reg.safety["max_zone_w"]
                or total[a:b].max(initial=0) + ev.level_w > reg.safety["max_total_w"]):
            continue
        zone_load[z.id][a:b] += ev.level_w
        total[a:b] += ev.level_w
        kept.append(ev)
    return kept


def compile_schedule(
    reg: Registry,
    seed: int,
    duration_s: float,
    start_hour: float = 0.0,
    plant_relations: bool = True,
):
    """Return (events, planted_relations). With plant_relations=False this is the null scenario."""
    rng = np.random.default_rng(seed)
    events = _routine_events(reg, rng, duration_s, start_hour)
    planted: list[PlantedRelation] = []

    if plant_relations:
        extra: list[LoadEvent] = []
        for rel in reg.relations:
            if rel.type != "trigger":
                continue
            p = rel.params
            target = reg.appliance(p["target"])
            lo, hi = reg.routines[p["target"]]["duration_s"]
            for ev in [e for e in events if e.appliance == p["source"]]:
                if rng.random() < p["prob"]:
                    start = ev.start_s + p["delay_s"] + rng.uniform(-p["jitter_s"], p["jitter_s"])
                    start = max(start, 0.0)
                    if start >= duration_s:
                        continue
                    dur = float(rng.uniform(lo, hi))
                    extra.append(LoadEvent(target.id, start, min(start + dur, duration_s),
                                           float(target.on_w * rng.uniform(0.95, 1.0)),
                                           origin=f"trigger:{p['source']}"))
            planted.append(PlantedRelation("trigger", p["source"], p["target"], p["delay_s"]))
        events = events + extra
        planted += [PlantedRelation("exclusion", r.params["a"], r.params["b"])
                    for r in reg.relations if r.type == "exclusion"]

    events = _drop_overlapping_same_device(events)
    if plant_relations:
        events = _apply_exclusions(events, reg)
    events = _apply_safety(events, reg, duration_s)
    return sorted(events, key=lambda e: e.start_s), planted


def events_to_records(events: list[LoadEvent]) -> list[dict]:
    return [asdict(e) for e in events]
