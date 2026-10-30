"""Conservative full-cell collision certificates from Bernstein position boxes.
Spheres/cylinders are enlarged by a circumscribed projection of the position
error box plus physical drone radius. Box obstacles use exact expanded boxes.
"""
import numpy as np

def obstacle_clearance(position,obstacle,radius=0.):
    p=np.asarray(position);kind=obstacle['kind']
    if kind=='sphere': return np.linalg.norm(p-np.array(obstacle['center']))-obstacle['radius']-radius
    if kind=='cylinder':
        c=np.array(obstacle['center']); q=np.array([np.linalg.norm(p[:2]-c[:2])-obstacle['radius'],abs(p[2]-c[2])-obstacle['height']/2])
        return np.linalg.norm(np.maximum(q,0))+min(max(q),0)-radius
    lo=np.array(obstacle['lower']);hi=np.array(obstacle['upper']);q=np.abs(p-(lo+hi)/2)-(hi-lo)/2
    return np.linalg.norm(np.maximum(q,0))+min(max(q),0)-radius

def cell_clearance(lo,hi,rho,obstacle,physical,position_coefficients=None):
    if obstacle['kind']=='box':
        ol=np.array(obstacle['lower'])-rho-physical;oh=np.array(obstacle['upper'])+rho+physical
        return max(np.max(ol-hi),np.max(lo-oh))
    center=np.array(obstacle['center']);dims=2 if obstacle['kind']=='cylinder' else 3
    if dims==2 and (hi[2]+rho[2]+physical < center[2]-obstacle['height']/2 or
                    lo[2]-rho[2]-physical > center[2]+obstacle['height']/2):
        return max(center[2]-obstacle['height']/2-hi[2]-rho[2]-physical,
                   lo[2]-rho[2]-physical-center[2]-obstacle['height']/2)
    d=np.maximum(np.maximum(lo[:dims]-center[:dims],center[:dims]-hi[:dims]),0)
    clearance=np.linalg.norm(d)-obstacle['radius']-physical-np.linalg.norm(rho[:dims])
    if position_coefficients is not None:
        # A separating plane for the complete Bernstein convex hull preserves
        # correlation between coordinates, unlike its axis-aligned box.
        points=np.asarray(position_coefficients)[:,:dims]
        direction=points.mean(axis=0)-center[:dims]
        norm=np.linalg.norm(direction)
        if norm>1e-12:
            normal=direction/norm
            plane=np.min((points-center[:dims])@normal)-obstacle['radius']-physical-np.abs(normal)@rho[:dims]
            clearance=max(clearance,float(plane))
    return clearance

def certify_plan(curve,tube,interval,config,scenario):
    worst=0.;clearance=float('inf');active=[]
    for i in range(len(curve.edges)-1):
        lo,hi=curve.cell_bounds(i);vl,vh=curve.cell_bounds(i,1);al,ah=curve.cell_bounds(i,2)
        phi,rho,ev=tube.cell_envelope(i)
        va=np.maximum(abs(vl),abs(vh))+ev
        gain=np.maximum(0,delta_radius(interval)*np.linalg.norm(va)*va-tube.drag_forcing[i]+config.alpha*phi)
        # Direct upper bound on Eq.29 applied input (B=I). Each coordinate is bounded.
        input_bound=np.maximum(abs(al-np.array(config.gravity)),abs(ah-np.array(config.gravity)))+interval.midpoint*np.linalg.norm(va)*va+config.sliding_lambda*ev+gain
        violations=np.r_[np.array(scenario.lower)+rho-lo,hi+rho-np.array(scenario.upper),
                         va-config.velocity_limit,input_bound-np.array(config.input_limits)]
        worst=max(worst,float(np.max(violations)))
        for o in scenario.obstacles:
            c=cell_clearance(lo,hi,rho,o,config.drone_radius,curve.bernstein_coefficients(i));clearance=min(clearance,c);worst=max(worst,-c)
        if np.max(violations)>-0.02: active.append(i)
    return {'safe':worst<=3e-6,'worst_residual':worst,'minimum_tube_clearance':clearance,'active_cells':active}

def delta_radius(interval): return interval.width/2
