"""Barcelona geometry, locally packaged from uncertain-racecar-gym."""
from importlib.resources import files
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline


class Track:
    def __init__(self, name="barcelona"):
        self.name=name
        if name=="technical":
            # A compact circuit with a fast straight, linked bends and a tight
            # reverse curve. Resample the smooth spline by physical arc length.
            points=1.5*np.array([[0,0],[170,0],[350,0],[480,5],[540,40],
                [540,115],[480,150],[380,150],[300,105],[240,100],[180,155],
                [80,155],[0,135],[-50,80],[-45,30],[0,0]],dtype=float)
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            curve=CubicSpline(arc,points,bc_type="periodic")
            dense=curve(np.linspace(0,arc[-1],16001))
            length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
            raw=np.column_stack([np.interp(np.linspace(0,length[-1],4097)[:-1],length,dense[:,j]) for j in (0,1)])
        elif name=="barcelona":
            path = files(__package__).joinpath("assets/barcelona.csv")
            raw = np.loadtxt(path, delimiter=",", skiprows=1)[:, 1:3]
            raw = gaussian_filter1d(raw, 3.0, axis=0, mode="wrap")
        else:
            raise ValueError(f"Unknown track: {name}")
        raw = np.vstack([raw, raw[0]])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        spline = CubicSpline(arc, raw, bc_type="periodic")
        self.length = float(arc[-1])
        self.s = np.linspace(0, self.length, 4097)
        self.xy = spline(self.s)
        d, dd = spline(self.s, 1), spline(self.s, 2)
        self.heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.curvature = (d[:, 0]*dd[:, 1]-d[:, 1]*dd[:, 0]) / np.linalg.norm(d, axis=1)**3
        self.curvature[-1] = self.curvature[0]
        self.width = 16.0
        self.ds = self.length / (len(self.s)-1)

    def pose(self, state):
        state = np.asarray(state)
        arc = state[..., 0] % self.length
        heading = np.interp(arc, self.s, self.heading)
        x = np.interp(arc, self.s, self.xy[:, 0]) - state[..., 1]*np.sin(heading)
        y = np.interp(arc, self.s, self.xy[:, 1]) + state[..., 1]*np.cos(heading)
        return np.stack([x, y, heading+state[..., 2]], axis=-1)

    def kappa(self, s):
        return np.interp(np.asarray(s) % self.length, self.s, self.curvature)
