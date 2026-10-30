"""Expression graph for the plant and generated sampled-rollout kernel.

The generated fast model is regression-tested against this graph; both use
the same dynamics, feedback controller, and actuator clipping.
"""
import math
import numpy as np
from numba import njit


class Expr:
    nodes = []

    def __init__(self, op, a=-1, b=-1, value=0.):
        self.i = len(self.nodes)
        self.nodes.append((op, a, b, value))

    @staticmethod
    def cast(x):
        return x if isinstance(x, Expr) else Expr(0, value=float(x))

    def __add__(self, x): return Expr(2, self.i, self.cast(x).i)
    __radd__ = __add__
    def __neg__(self): return self * -1
    def __sub__(self, x): return self + (-self.cast(x))
    def __rsub__(self, x): return self.cast(x) + (-self)
    def __mul__(self, x): return Expr(3, self.i, self.cast(x).i)
    __rmul__ = __mul__
    def __truediv__(self, x): return self * Expr(4, self.cast(x).i)
    def __rtruediv__(self, x): return self.cast(x) / self
    def sin(self): return Expr(5, self.i)
    def cos(self): return Expr(6, self.i)
    def atan(self): return Expr(7, self.i)
    def clip(self, lo, hi):
        return Expr(9, Expr(8, self.i, value=lo).i, value=hi)
    def curvature(self): return Expr(10, self.i)


@njit(cache=True)
def lookup(x, k, ds):
    z = (x/ds) % (len(k)-1)
    i = int(z)
    return k[i] + (z-i)*(k[i+1]-k[i])


@njit(cache=True)
def evaluate(inputs, graph, outputs, k, ds):
    v = np.empty(len(graph))
    for i in range(len(graph)):
        op, a, b = int(graph[i,0]), int(graph[i,1]), int(graph[i,2])
        c = graph[i,3]
        if op == 0: v[i] = c
        elif op == 1: v[i] = inputs[a]
        elif op == 2: v[i] = v[a]+v[b]
        elif op == 3: v[i] = v[a]*v[b]
        elif op == 4: v[i] = 1/v[a]
        elif op == 5: v[i] = math.sin(v[a])
        elif op == 6: v[i] = math.cos(v[a])
        elif op == 7: v[i] = math.atan(v[a])
        elif op == 8: v[i] = max(v[a],c)
        elif op == 9: v[i] = min(v[a],c)
        elif op == 10: v[i] = lookup(v[a],k,ds)
    return v[outputs]
