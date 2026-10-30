"""Shared, method-independent experiment configurations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    track: str
    sensor: str
    traction: bool
    description: str
    racing_speed: float = 24.
    backup_duration: float = 3.5
    epsilon: float = .01
    delta: float = .05
    samples: int = 1000
    candidate_limit: int = 64


SCENARIOS={
    "barcelona":Scenario("barcelona","Barcelona · reference","barcelona","uniform",False,
        "The original circuit with bounded acceleration observations."),
    "traction":Scenario("traction","Scenario 1 · 1% tolerance","technical","quantized_yaw",True,
        "Gentle backup driving gives little yaw excitation. Informative MPPI excites yaw dynamics to identify grip; every candidate plus fallback must pass the combined tire-force and state checks.",55.,10.),
    "traction_relaxed":Scenario("traction_relaxed","Scenario 2 · 5% tolerance","technical","quantized_yaw",True,
        "The same technical circuit and controllers, with a higher allowed candidate failure probability.",55.,10.,.05),
}


def get_scenario(name):
    if name not in SCENARIOS: raise ValueError(f"Unknown scenario: {name}")
    return SCENARIOS[name]
