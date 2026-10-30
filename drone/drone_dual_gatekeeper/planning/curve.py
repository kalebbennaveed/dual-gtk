"""Cubic B-spline direct transcription; inverse dynamics are exact in continuous time.
Bernstein hulls give bounds over whole cells, including between sample points.
"""
from functools import lru_cache
from math import comb
import numpy as np
from scipy.interpolate import BSpline

@lru_cache(maxsize=32)
def basis(control_points,subdivisions):
    degree=3; spans=control_points-degree
    knots=np.r_[np.zeros(degree),np.linspace(0,1,spans+1),np.ones(degree)]
    b=BSpline(knots,np.eye(control_points),degree)
    edges=np.linspace(0,1,spans*subdivisions+1)
    hulls=[]
    for left,right in zip(edges[:-1],edges[1:]):
        matrices=[]
        for derivative in range(3):
            n=degree-derivative
            z=np.linspace(0,1,n+1)
            bern=np.array([[comb(n,j)*s**j*(1-s)**(n-j) for j in range(n+1)] for s in z])
            matrices.append(np.linalg.solve(bern,b(left+(right-left)*z,nu=derivative)))
        hulls.append(tuple(matrices))
    return b,edges,tuple(hulls)

class Curve:
    def __init__(self,coefficients,horizon,subdivisions):
        self.coefficients=np.asarray(coefficients,dtype=float)
        self.horizon=float(horizon)
        self.basis,self.edges,self.hulls=basis(len(coefficients),subdivisions)
    def value(self,t,derivative=0):
        s=np.minimum(np.maximum(np.asarray(t)/self.horizon,0),1)
        return self.basis(s,nu=derivative)@self.coefficients/self.horizon**derivative
    def state(self,t): return np.concatenate([self.value(t),self.value(t,1)],axis=-1)
    def acceleration(self,t): return self.value(t,2)
    def bernstein_coefficients(self,index,derivative=0):
        return self.hulls[index][derivative]@self.coefficients/self.horizon**derivative
    def cell_bounds(self,index,derivative=0):
        c=self.bernstein_coefficients(index,derivative)
        return c.min(axis=0),c.max(axis=0)
