from dataclasses import dataclass, fields
from pathlib import Path
import yaml
import numpy as np
CONFIG_DIR=Path(__file__).parent/'configs'

@dataclass(frozen=True)
class ControlConfig:
    alpha: float = 2.0
    sliding_lambda: float = 2.0
    disturbance_bound: float = 0.08
    eta: float = 0.1
    drone_radius: float = 0.1
    velocity_limit: float = 2.5
    input_limits: tuple = (15.,15.,25.)
    gravity: tuple = (0.,0.,-9.81)
    control_points: int = 8
    subdivisions: int = 3
    min_horizon: float = 2.0
    max_horizon: float = 45.0
    candidate_increment: float = 2.0
    mission_input_weight: float = 0.02
    mission_position_weight: float = 0.3
    information_epsilon: float = 1e-6
    informative_gamma: float = 8.0
    weighted_gamma: float = 2.0
    weighted_information_discount: float = 0.0
    score_discount: float = 0.25
    minimum_information_advantage: float = 0.0
    exploration_budget_fraction: float = 0.1
    simulation_step: float = 0.02
    smid_window: float = 0.2
    measurement_noise: float = 0.0
    nominal_altitude: float | None = None
    excitation_seed_amplitudes: tuple = (.7,1.3)
    homotopy_seed_offset: float = 2.5
    goal_tolerance: float = 0.22
    goal_velocity_tolerance: float = 0.4
    max_replans: int = 30
    solver_tolerance: float = 1e-7
    solver_iterations: int = 150
    solver_hessian_approximation: str = 'exact'
    informative_solver_iterations: int | None = None
    informative_hessian_approximation: str = 'exact'
    solver_acceptable_tolerance: float = 0.1
    smooth_max_width: float = 0.001
    smooth_abs_width: float = 0.0001
    initial_interval: tuple = (0.,0.5)
    def __post_init__(self):
        if self.alpha<=0 or self.sliding_lambda<=0: raise ValueError("positive fixed bandwidth required")
        if self.disturbance_bound<0 or self.eta<=0: raise ValueError("invalid uncertainty margin")
        if self.minimum_information_advantage<0:raise ValueError('information advantage threshold must be nonnegative')
        if self.weighted_information_discount<0:raise ValueError('information discount must be nonnegative')
        if self.informative_hessian_approximation not in ('exact','limited-memory'):raise ValueError('invalid Hessian approximation')
    @property
    def initial_boundary_layer(self): return np.full(3,(self.disturbance_bound+self.eta)/self.alpha)

@dataclass(frozen=True)
class Scenario:
    name: str
    start: tuple
    goal: tuple
    lower: tuple
    upper: tuple
    obstacles: tuple
    description: str

def load_config(scenario='informative_demo'):
    common=yaml.safe_load((CONFIG_DIR/'common.yaml').read_text())
    s=yaml.safe_load((CONFIG_DIR/f'{scenario}.yaml').read_text())
    common.update(s.pop('overrides',{}))
    known={f.name for f in fields(ControlConfig)}
    if set(common)-known: raise ValueError(f"Unknown settings: {set(common)-known}")
    return ControlConfig(**common),Scenario(**{**s,'obstacles':tuple(s['obstacles'])})
