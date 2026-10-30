"""Six-state acceleration dynamics with scalar quadratic drag (paper VI-A)."""
import numpy as np

def drag_regressor(velocity):
    v=np.asarray(velocity)
    return -np.linalg.norm(v,axis=-1,keepdims=True)*v

def derivative(state,control,drag_coefficient,gravity,disturbance=None):
    x=np.asarray(state)
    d=np.zeros(3) if disturbance is None else np.asarray(disturbance)
    return np.r_[x[3:], drag_coefficient*drag_regressor(x[3:])+gravity+control+d]

def rk4(fun,t,state,step):
    k1=fun(t,state); k2=fun(t+step/2,state+step*k1/2)
    k3=fun(t+step/2,state+step*k2/2); k4=fun(t+step,state+step*k3)
    return state+step*(k1+2*k2+2*k3+k4)/6
