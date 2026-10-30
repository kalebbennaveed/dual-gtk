"""Set membership for a fixed scalar friction coefficient; no ground-truth access."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FrictionSet:
    lower: float = .45
    upper: float = 1.05

    @property
    def center(self): return .5*(self.lower+self.upper)
    @property
    def radius(self): return .5*(self.upper-self.lower)
    @property
    def width(self): return self.upper-self.lower

    def update(self, residual, regressor, noise_bound):
        lo,hi=self.lower,self.upper
        for y,g,w in zip(residual,regressor,noise_bound):
            if abs(g)<1e-10:
                if abs(y)>w+1e-9: raise ValueError("Measurement inconsistent with the declared noise bound")
                continue
            a,b=(y-w)/g,(y+w)/g
            lo,hi=max(lo,min(a,b)),min(hi,max(a,b))
        if lo>hi+1e-10: raise ValueError("Empty friction set: model or sensor bounds are inconsistent")
        self.lower,self.upper=lo,max(lo,hi)


SENSOR_BOUND=np.array([.6,.6,.18])


def predicted_width(width, phis, sensor="uniform"):
    # Expected interval width for a centred, bounded-error synthetic measurement.
    # Used only for scoring; safety always uses the current (unshrunk) support.
    sensitivity=(np.max(np.abs(phis[:,5])/.18,initial=0.) if sensor=="quantized_yaw"
                 else np.max(np.abs(phis[:,3:6])/SENSOR_BOUND,initial=0.))
    return min(width,2./max(sensitivity,1e-12))


def excitation_score(phis,sensor="uniform"):
    """Log noise-scaled regressor energy per step, a persistent design reward.

    This is an excitation surrogate, not predicted interval contraction or
    Bayesian information gain. Unlike the proposed shrinkage score, it does
    not vanish once the current parameter interval is already narrow.
    """
    scaled=phis[:,5:6]/.18 if sensor=="quantized_yaw" else phis[:,3:6]/SENSOR_BOUND
    return float(np.log1p(np.mean(np.sum(scaled*scaled,axis=1))))


def predicted_quantized_width(belief,states,phis,dt=.05):
    """Point-model prediction through the actual quantizer and strip update.

    Predicting only 2*noise/peak_regressor misses the intersection of multiple
    quantized observations. Use the current interval center, never plant truth.
    This remains a planning estimate; safety uses the unshrunk current set.
    """
    posterior=FrictionSet(belief.lower,belief.upper)
    rate=np.diff(states[:,5])/dt
    for r,g in zip(rate,phis[:,5]):
        residual=.36*np.rint(r/.36)-r+belief.center*g
        posterior.update([residual],[g],[.18+1e-12])
    return posterior.width


def observation(rate,phi,f0,rng,sensor="uniform"):
    if sensor=="quantized_yaw":
        # A coarse yaw-acceleration identification channel. Quantization error
        # is bounded and correlated; repeated quiet readings do not generate
        # artificial information as independent uniform-noise samples would.
        quantum=.36
        measured=quantum*np.rint(rate[5]/quantum)
        return np.array([measured-f0[5]]),phi[5:6],np.array([quantum/2+1e-12])
    if sensor!="uniform": raise ValueError("Unknown identification sensor")
    return rate[3:6]+rng.uniform(-SENSOR_BOUND,SENSOR_BOUND)-f0[3:6],phi[3:6],SENSOR_BOUND
