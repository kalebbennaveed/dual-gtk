from dataclasses import dataclass, field
from enum import Enum
import numpy as np

@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    def __post_init__(self):
        if not 0 <= self.lower <= self.upper:
            raise ValueError("Empty or negative drag interval")
    @property
    def width(self): return self.upper-self.lower
    @property
    def midpoint(self): return (self.lower+self.upper)/2
    def contains(self, value, tol=1e-9): return self.lower-tol <= value <= self.upper+tol

@dataclass(frozen=True)
class DisturbanceSet:
    bound: float

class CandidateStatus(str, Enum):
    GENERATED="generated"
    UNSAFE="unsafe"
    SAFE="safe"
    BUDGET_INFEASIBLE="budget_infeasible"
    FEASIBLE="feasible"
    COMMITTED="committed"

@dataclass
class RobustPlan:
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    timestamps: np.ndarray
    boundary_layer: np.ndarray
    tube_profile: np.ndarray
    cost: float
    horizon: float
    curve: object
    tube: object
    estimate: float
    solver_status: str
    residual: float
    planning_time: float
    information: float
    certificate: dict = field(default_factory=dict)
    def state(self,t): return self.curve.state(t)
    def control(self,t,gravity):
        v=self.state(t)[3:]
        return self.curve.acceleration(t)+self.estimate*np.linalg.norm(v)*v-gravity

@dataclass
class Candidate:
    horizon: float
    status: CandidateStatus = CandidateStatus.GENERATED
    plan: RobustPlan | None = None
    safe: bool = False
    reconnects: bool = False
    budget_feasible: bool = False
    backup_cost: float = 0.0
    candidate_cost: float = 0.0
    exploration_cost: float = 0.0
    predicted_reduction: float = 0.0
    backup_predicted_reduction: float = 0.0
    information_advantage: float = 0.0
    score: float = 0.0
    selected: bool = False
    reason: str = ""
