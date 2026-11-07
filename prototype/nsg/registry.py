"""Device registry: topology and device profiles as data, loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Appliance:
    id: str
    levels_w: tuple[float, ...]
    pf: float
    noise_w: float

    @property
    def on_w(self) -> float:
        return max(self.levels_w)


@dataclass(frozen=True)
class Zone:
    id: str
    meter: str
    rated_w: float
    own_resistance_ohm: float
    appliances: tuple[Appliance, ...]


@dataclass(frozen=True)
class Relation:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Registry:
    sample_period_s: float
    nominal_v: float
    nominal_hz: float
    shared_resistance_ohm: float
    zones: tuple[Zone, ...]
    meter: dict[str, float]
    safety: dict[str, float]
    routines: dict[str, dict[str, Any]]
    relations: tuple[Relation, ...]

    def appliance_zone(self, appliance_id: str) -> Zone:
        for z in self.zones:
            if any(a.id == appliance_id for a in z.appliances):
                return z
        raise KeyError(appliance_id)

    def appliance(self, appliance_id: str) -> Appliance:
        for z in self.zones:
            for a in z.appliances:
                if a.id == appliance_id:
                    return a
        raise KeyError(appliance_id)

    @property
    def appliances(self) -> list[Appliance]:
        return [a for z in self.zones for a in z.appliances]

    @property
    def meters(self) -> list[str]:
        return [z.meter for z in self.zones]


def load_registry(path: str | Path) -> Registry:
    raw = yaml.safe_load(Path(path).read_text())
    zones = tuple(
        Zone(
            id=z["id"],
            meter=z["meter"],
            rated_w=float(z["rated_w"]),
            own_resistance_ohm=float(z.get("own_resistance_ohm", 0.0)),
            appliances=tuple(
                Appliance(
                    id=a["id"],
                    levels_w=tuple(float(x) for x in a["levels_w"]),
                    pf=float(a["pf"]),
                    noise_w=float(a.get("noise_w", 0.0)),
                )
                for a in z["appliances"]
            ),
        )
        for z in raw["zones"]
    )
    relations = tuple(
        Relation(type=r["type"], params={k: v for k, v in r.items() if k != "type"})
        for r in raw.get("relations", [])
    )
    reg = Registry(
        sample_period_s=float(raw["sample_period_s"]),
        nominal_v=float(raw["supply"]["nominal_v"]),
        nominal_hz=float(raw["supply"]["nominal_hz"]),
        shared_resistance_ohm=float(raw["supply"]["shared_resistance_ohm"]),
        zones=zones,
        meter=dict(raw["meter"]),
        safety={k: float(v) for k, v in raw["safety"].items()},
        routines=dict(raw.get("routines", {})),
        relations=relations,
    )
    _validate(reg)
    return reg


def _validate(reg: Registry) -> None:
    ids = [a.id for a in reg.appliances]
    if len(ids) != len(set(ids)):
        raise ValueError("appliance ids must be unique")
    if len(reg.meters) != len(set(reg.meters)):
        raise ValueError("meter ids must be unique")
    known = set(ids)
    for r in reg.relations:
        for key in ("source", "target", "a", "b"):
            if key in r.params and r.params[key] not in known:
                raise ValueError(f"relation references unknown appliance: {r.params[key]}")
    for z in reg.zones:
        if sum(a.on_w for a in z.appliances) > z.rated_w:
            raise ValueError(f"zone {z.id}: appliances can exceed rated_w")
        if reg.safety["max_zone_w"] > z.rated_w:
            raise ValueError(f"zone {z.id}: max_zone_w above rated_w")
