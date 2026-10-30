"""Independent geometry and excitation audit for the retained drone scenario."""
import argparse
from copy import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from drone_dual_gatekeeper.config import ControlConfig, Scenario
from drone_dual_gatekeeper.control.tube_propagation import build_tube
from drone_dual_gatekeeper.environment.collision import certify_plan, obstacle_clearance
from drone_dual_gatekeeper.planning.curve import Curve
from drone_dual_gatekeeper.planning.dual_gatekeeper import reconnection, mission_cost
from drone_dual_gatekeeper.simulation.logging import save_result
from scripts.validate_results import reconstruct_plan


def sweep_clearance(points, scenario, radius):
    """Distance for every straight interpolated display segment, not just nodes."""
    minimum = float('inf')
    for start,end in zip(points[:-1],points[1:]):
        direction=end-start
        for obstacle in scenario.obstacles:
            if obstacle['kind']=='cylinder' and (
                min(start[2],end[2])>=obstacle['center'][2]-obstacle['height']/2 and
                max(start[2],end[2])<=obstacle['center'][2]+obstacle['height']/2):
                d=direction[:2];relative=start[:2]-obstacle['center'][:2]
                t=np.clip(-relative@d/max(d@d,1e-30),0.,1.)
                clearance=np.linalg.norm(relative+t*d)-obstacle['radius']-radius
            else:
                distance=lambda t: obstacle_clearance(start+t*direction,obstacle,radius)
                fit=minimize_scalar(distance,bounds=(0.,1.),method='bounded')
                clearance=min(distance(0),distance(1),fit.fun)
            minimum=min(minimum,float(clearance))
    return minimum


def reflected(plan, interval, config):
    result=copy(plan)
    result.curve=Curve(plan.curve.coefficients[:,[1,0,2]],plan.horizon,config.subdivisions)
    result.tube=build_tube(result.curve,interval,config)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('results/focused_demo'))
    parser.add_argument('--scenario',default='informative_demo')
    args=parser.parse_args();out=args.source
    rows=[]
    for method in ['proposed','passive','weighted','robust']:
        record=json.loads((out/f'{args.scenario}_{method}.json').read_text())
        config=ControlConfig(**record['config']);scenario=Scenario(**record['scenario'])
        actual=np.array([s['state'][:3] for s in record['samples']])
        nominal=np.array([s['nominal'][:3] for s in record['samples']])
        actual_margin=sweep_clearance(actual,scenario,config.drone_radius)
        nominal_margin=sweep_clearance(nominal,scenario,config.drone_radius)
        assert actual_margin>0 and nominal_margin>0,'Displayed swept safety sphere intersects obstacle'
        rows.append({'method':method,'physical_radius':config.drone_radius,
                     'minimum_actual_swept_clearance':actual_margin,
                     'minimum_nominal_swept_clearance':nominal_margin,
                     'nominal_z_range':[float(nominal[:,2].min()),float(nominal[:,2].max())],
                     'actual_z_range':[float(actual[:,2].min()),float(actual[:,2].max())],
                     'nominal_altitude_constraint':config.nominal_altitude})
        delta=np.array(scenario.goal[:2])-np.array(scenario.start[:2])
        fraction=np.clip((actual[:,:2]-scenario.start[:2])@delta/(delta@delta),0,1)
        straight=np.array(scenario.start[:2])+fraction[:,None]*delta
        rows[-1]['maximum_xy_line_deviation']=float(np.max(np.linalg.norm(actual[:,:2]-straight,axis=1)))
        rows[-1]['actual_path_length']=float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum())
        if scenario.name=='spatial_learning_course':
            low=actual[(actual[:,0]>=7.5)&(actual[:,0]<=8.5)]
            high=actual[(actual[:,0]>=12.5)&(actual[:,0]<=13.5)]
            assert len(low) and len(high)
            assert low[:,2].min()>2.4+config.drone_radius-1e-6
            assert high[:,2].max()<1.8-config.drone_radius+1e-6
            assert rows[-1]['maximum_xy_line_deviation']>.5
            rows[-1]['altitude_over_low_wall']=[float(low[:,2].min()),float(low[:,2].max())]
            rows[-1]['altitude_under_overhead_wall']=[float(high[:,2].min()),float(high[:,2].max())]
        if scenario.name=='slalom_course':
            passages=[]
            for obstacle in scenario.obstacles:
                crossing=actual[(actual[:,0]>=obstacle['lower'][0])&(actual[:,0]<=obstacle['upper'][0])]
                assert len(crossing),'Missing barrier crossing'
                if obstacle['lower'][1]==scenario.lower[1]:
                    assert crossing[:,1].min()>obstacle['upper'][1]+config.drone_radius-1e-6
                else:
                    assert crossing[:,1].max()<obstacle['lower'][1]-config.drone_radius+1e-6
                passages.append([float(crossing[:,1].min()),float(crossing[:,1].max())])
            rows[-1]['lateral_barrier_crossings']=passages
        if method=='proposed':proposed=record
    if args.scenario!='informative_demo':
        save_result({'scenario':args.scenario,'methods':rows},out/'geometry_audit.json')
        print(json.dumps(rows,indent=2))
        return
    config=ControlConfig(**proposed['config']);scenario=Scenario(**proposed['scenario'])
    epoch=next(e for e in proposed['epochs'] if e['informative_committed'])
    from drone_dual_gatekeeper.types import Interval
    interval=Interval(*epoch['interval_before'])
    active=reconstruct_plan(epoch['committed'],config);backup=reconstruct_plan(epoch['backup'],config)
    mirror=reflected(active,interval,config);mirror_backup=reflected(backup,interval,config)
    cert=certify_plan(mirror.curve,mirror.tube,interval,config,scenario)
    assert cert['safe'] and reconnection(mirror,mirror_backup,epoch['duration'])
    times=np.linspace(0,active.horizon,10001)
    information=lambda p: float(np.trapezoid(np.sum(p.state(times)[:,3:]**2,axis=1)**2,times))
    active_info=information(active);mirror_info=information(mirror);backup_info=information(backup)
    mirror_cost=mission_cost(mirror,config,np.array(scenario.goal))
    active_cost=mission_cost(active,config,np.array(scenario.goal))
    assert abs(mirror_info-active_info)<1e-8 and abs(mirror_cost-active_cost)<1e-8
    selected=next(c for c in epoch['candidates'] if c['selected'])
    report={'methods':rows,'renderer':{'collision_radius':config.drone_radius,
                  'old_render_scale':2.5,'old_mesh_circumsphere_approx':.379,
                  'fixed_mesh_radius':.995*config.drone_radius,
                  'fix':'All transformed mesh vertices fitted inside collision sphere; no enlarged glyphs.'},
            'excitation':{'selected_horizon':epoch['duration'],'solver_status':epoch['committed']['solver_status'],
                  'constraint_residual':epoch['committed']['residual'],
                  'selected_score':selected['score'],'safe_mirrored_plan':cert['safe'],
                  'reconnects_to_mirrored_backup':reconnection(mirror,mirror_backup,epoch['duration']),
                  'reconnects_to_original_backup':reconnection(mirror,backup,epoch['duration']),
                  'active_cost':active_cost,'mirror_cost':mirror_cost,
                  'active_gramian':active_info,'mirror_gramian':mirror_info,
                  'backup_prefix_gramian':backup_info,
                  'interpretation':'Scalar-drag information proxy is integral ||v||^4. The side of the excursion is not uniquely required; mirrored x/y plans are equally safe and have identical objectives.'}}
    save_result(report,out/'geometry_and_excitation_audit.json')
    print((out/'geometry_and_excitation_audit.json').read_text())


if __name__=='__main__':main()
