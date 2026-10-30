"""Hidden truth and bounded disturbances live here, outside all planning interfaces.
True RK4 dynamics use smaller steps than the spline optimizer's tube cells.
Integrated regression is accumulated at the same RK4 stages as dynamics.
"""
from dataclasses import dataclass
import numpy as np
from ..dynamics.quadrotor_drag import derivative,drag_regressor,rk4
from ..control.ancillary_controller import ancillary_control
from ..environment.collision import obstacle_clearance

@dataclass(frozen=True)
class SimulationTruth:
    true_drag_coefficient: float
    seed: int
    disturbance_scale: float = 1.0
    measurement_noise: float = 0.0
    disturbance_profile: str = 'seeded'

class TrueDroneSimulator:
    def __init__(self,truth,config,scenario):
        self.truth=truth;self.config=config;self.scenario=scenario
        rng=np.random.default_rng(truth.seed)
        self.disturbances=rng.uniform(-1,1,(30000,3))*config.disturbance_bound*truth.disturbance_scale
        if truth.disturbance_profile == 'constant_positive':
            self.disturbances[:]=config.disturbance_bound*truth.disturbance_scale
        elif truth.disturbance_profile != 'seeded':
            raise ValueError('Unknown disturbance profile')
        self.sensor_rng=np.random.default_rng(truth.seed+9823)
        self.sensor_noise_values=self.sensor_rng.uniform(-1,1,(30000,3))
    def sensor_noise(self,t):
        # The same bounded noise realization on the same physical clock,
        # regardless of a method's number or duration of replanning epochs.
        coordinate=max(0.,t/self.config.smid_window)
        index=min(int(coordinate),len(self.sensor_noise_values)-2)
        fraction=min(coordinate-index,1.)
        scale=self.truth.measurement_noise+self.config.measurement_noise
        return scale*((1-fraction)*self.sensor_noise_values[index]+fraction*self.sensor_noise_values[index+1])
    def disturbance(self,t): return self.disturbances[min(int((t+1e-9)/0.1),len(self.disturbances)-1)]
    def execute(self,plan,interval,state,start_time,duration,stop_at_goal=False):
        c=self.config;s=self.scenario;g=np.array(c.gravity)
        assert interval.contains(self.truth.true_drag_coefficient),'Truth excluded before execution'
        rows=[];observations=[];y=np.r_[state,np.zeros(6)];elapsed=0.;window_start=0.;v_start=state[3:].copy()
        noise=self.truth.measurement_noise+c.measurement_noise
        v_start+=self.sensor_noise(start_time)
        def applied(t,x):
            phi,_=plan.tube.at(t)
            anc=ancillary_control(x,plan.state(t),phi,plan.tube.forcing_at(t),interval,c)
            return plan.control(t,g)+anc,anc
        def ode(t,z):
            u,_=applied(t,z[:6]);return np.r_[derivative(z[:6],u,self.truth.true_drag_coefficient,g,self.disturbance(start_time+t)),
                                              drag_regressor(z[3:6]),g+u]
        while True:
            x=y[:6];nominal=plan.state(elapsed);u,anc=applied(elapsed,x)
            utilization=plan.tube.utilization(x-nominal,elapsed)
            clearance=min([obstacle_clearance(x[:3],o,c.drone_radius) for o in s.obstacles] or [float('inf')])
            violations=[]
            if utilization>1+2e-5:violations.append('tube containment')
            if clearance< -1e-6:violations.append('collision')
            if np.any(np.abs(u)>np.array(c.input_limits)+1e-6):violations.append('input authority')
            if np.any(x[:3]<np.array(s.lower)-1e-6) or np.any(x[:3]>np.array(s.upper)+1e-6):violations.append('position bounds')
            if np.any(abs(x[3:])>c.velocity_limit+1e-6):violations.append('velocity bounds')
            phi,rho=plan.tube.at(elapsed)
            rows.append({'time':start_time+elapsed,'state':x.tolist(),'nominal':nominal.tolist(),'control':u.tolist(),
                         'ancillary':anc.tolist(),'phi':phi.tolist(),'rho':rho.tolist(),'tube_utilization':float(utilization),
                         'clearance':float(clearance),'interval':[interval.lower,interval.upper],
                         'rate_cost':float(c.mission_input_weight*np.dot(u,u)+c.mission_position_weight*np.sum((x[:3]-s.goal)**2))})
            if violations:
                raise AssertionError({'violations':violations,'sample':rows[-1],'certificate':plan.certificate})
            reached=stop_at_goal and np.linalg.norm(x[:3]-s.goal)<=c.goal_tolerance and np.linalg.norm(x[3:])<=c.goal_velocity_tolerance
            if reached or elapsed>=duration-1e-10:
                if elapsed-window_start>1e-9:
                    observed_v=y[3:6]+self.sensor_noise(start_time+elapsed)
                    observations.append({'duration':elapsed-window_start,'regressor':y[6:9].tolist(),
                                         'output':(observed_v-v_start-y[9:12]).tolist(),
                                         'integration_allowance':1e-8,'measurement_allowance':2*noise})
                break
            h=min(c.simulation_step,duration-elapsed,c.smid_window-(elapsed-window_start))
            y=rk4(ode,elapsed,y,h);elapsed+=h
            if elapsed-window_start>=c.smid_window-1e-9 or elapsed>=duration-1e-9:
                observed_v=y[3:6]+self.sensor_noise(start_time+elapsed)
                observations.append({'duration':elapsed-window_start,'regressor':y[6:9].tolist(),
                                     'output':(observed_v-v_start-y[9:12]).tolist(),
                                     'integration_allowance':1e-8,'measurement_allowance':2*noise})
                y[6:]=0;window_start=elapsed;v_start=observed_v.copy()
        return y[:6],rows,observations
