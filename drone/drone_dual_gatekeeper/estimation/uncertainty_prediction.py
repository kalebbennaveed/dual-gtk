"""Dual-Gatekeeper V-B (15)-(17). Returns a WIDTH bound, not fictitious measured endpoints.
The midpoint-centered interval is explicitly a visualization surrogate, never used
for planning/SMID; the posterior's location is unknowable before measurements.
"""
from dataclasses import dataclass
import numpy as np
from ..types import Interval
from ..dynamics.quadrotor_drag import drag_regressor

@dataclass(frozen=True)
class UncertaintyPrediction:
    predicted_interval: Interval
    predicted_width: float
    fractional_reduction: float
    absolute_reduction: float
    regressor_matrix: np.ndarray
    information_eigenvalue: float
    condition_number: float
    interval_is_surrogate: bool = True

def predict_uncertainty_reduction(candidate,current_interval,disturbance_bound):
    A=drag_regressor(candidate.nominal_states[:,3:])
    largest=float(np.max(abs(A)))
    width=current_interval.width if largest<1e-12 else min(current_interval.width,4*disturbance_bound.bound/largest)
    reduction=current_interval.width-width;mid=current_interval.midpoint
    eigen=float(np.sum(A*A));cond=1. if eigen>1e-12 else float('inf')
    return UncertaintyPrediction(Interval(mid-width/2,mid+width/2),width,
          reduction/current_interval.width if current_interval.width else 0.,reduction,A,eigen,cond)
