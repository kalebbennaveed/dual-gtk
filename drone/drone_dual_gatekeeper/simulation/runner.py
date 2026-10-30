import time
from dataclasses import asdict
import numpy as np
from ..types import Interval,DisturbanceSet
from ..methods.registry import METHODS
from ..planning.optimizer import TrajectoryOptimizer,PlanningFailure
from ..planning.dual_gatekeeper import generate_and_select,mission_cost,reconnection
from ..estimation.smid import update_interval
from ..estimation.uncertainty_prediction import predict_uncertainty_reduction
from .simulator import TrueDroneSimulator
from .logging import plan_record,candidate_record,revision

def shifted_backup_guess(backup,duration,config):
    """Fit the previous backup's remaining geometry as a solver guess only.

    A fresh nonlinear solve and full certificate remain mandatory. No old or
    interpolated path is executed as a fallback after a failed solve.
    """
    from copy import copy
    from ..planning.curve import Curve,basis
    remaining=backup.horizon-duration
    if remaining<1e-6:return None
    horizon=max(remaining,config.min_horizon)
    grid=np.linspace(0,1,101)
    points=backup.curve.value(np.minimum(duration+grid*horizon,backup.horizon))
    b,_,_=basis(config.control_points,config.subdivisions)
    coefficients=np.linalg.lstsq(b(grid),points,rcond=None)[0]
    guess=copy(backup);guess.curve=Curve(coefficients,horizon,config.subdivisions);guess.horizon=horizon
    return guess

def run_method(config,scenario,method_name,truth,quiet=False):
    method=METHODS[method_name];interval=Interval(*config.initial_interval)
    state=np.array(scenario.start,dtype=float);optimizer=TrajectoryOptimizer(config,scenario)
    optimizer.verbose=not quiet
    simulator=TrueDroneSimulator(truth,config,scenario)
    current_time=0.;spent=0.;budget=None;samples=[];epochs=[];weighted_failures=[];observations_all=[]
    backup_guess=None
    for k in range(config.max_replans):
        planning_start=time.perf_counter();before=interval
        backup=optimizer.solve(state,interval,warm=backup_guess)
        if budget is None:budget=config.exploration_budget_fraction*mission_cost(backup,config,np.array(scenario.goal))
        selected_plan=backup;duration=min(config.candidate_increment,backup.horizon);rows=[];selected=None
        if method.strategy=='weighted':
            try:
                selected_plan=optimizer.solve(state,interval,gamma=config.weighted_gamma,warm=backup,
                    information_discount=config.weighted_information_discount)
                duration=min(config.candidate_increment,selected_plan.horizon)
            except PlanningFailure as error:
                weighted_failures.append({'epoch':k,'diagnostics':error.diagnostics})
                if not quiet:print('Weighted optimization failed; certified backup selected:',error.diagnostics,flush=True)
        elif method.strategy=='gatekeeper':
            selected,rows=generate_and_select(optimizer,backup,state,interval,spent,budget)
            if selected:
                selected_plan=selected.plan;duration=selected.horizon;spent+=selected.exploration_cost
                assert reconnection(selected_plan,backup,duration)
                assert spent<=budget+1e-8
        # Use only the EXECUTED prefix for the before/after prediction diagnostic.
        from copy import copy
        prefix=copy(selected_plan);prefix.nominal_states=selected_plan.state(np.linspace(0,duration,101))
        predicted=predict_uncertainty_reduction(prefix,interval,DisturbanceSet(config.disturbance_bound))
        plan_time=time.perf_counter()-planning_start
        if not quiet:
            print(f'{method.label}: epoch={k} t={current_time:.2f} Theta=[{interval.lower:.6f},{interval.upper:.6f}] duration={duration:.2f} budget={spent:.3f}/{budget:.3f}',flush=True)
            if rows:
                print('Candidate Horizon Tube Reconnect Budget BackupCost InfoCost Excess PredReduction Score Selected Reason',flush=True)
                for i,r in enumerate(rows):print(i,f'{r.horizon:.2f}',r.safe,r.reconnects,r.budget_feasible,f'{r.backup_cost:.3f}',f'{r.candidate_cost:.3f}',f'{r.exploration_cost:.3f}',f'{r.predicted_reduction:.5f}',f'{r.score:.5f}',r.selected,r.reason,flush=True)
        planned_duration=duration
        state,new_samples,observations=simulator.execute(selected_plan,interval,state,current_time,duration,stop_at_goal=True)
        duration=new_samples[-1]['time']-current_time
        backup_guess=shifted_backup_guess(backup,duration,config)
        observations_all.extend(observations)
        if method.learning:interval=update_interval(interval,observations,config.disturbance_bound)
        assert interval.contains(truth.true_drag_coefficient),'SMID excluded truth; investigate data consistency'
        assert before.lower<=interval.lower<=interval.upper<=before.upper
        epoch={'index':k,'time':current_time,'duration':duration,'planned_duration':planned_duration,'interval_before':[before.lower,before.upper],
               'interval_after':[interval.lower,interval.upper],'backup':plan_record(backup,before),
               'committed':plan_record(selected_plan,before),'candidates':[candidate_record(r,before) for r in rows],
               'informative_committed':selected is not None,'passive_fallback':method.strategy=='gatekeeper' and selected is None,'exploration_spent':spent,'budget':budget,
               'predicted_reduction':predicted.absolute_reduction,'realized_reduction':before.width-interval.width,
               'planning_time':plan_time}
        epochs.append(epoch);samples.extend(new_samples if not samples else new_samples[1:]);current_time+=duration
        if np.linalg.norm(state[:3]-scenario.goal)<=config.goal_tolerance and np.linalg.norm(state[3:])<=config.goal_velocity_tolerance:break
    costs=np.array([x['rate_cost'] for x in samples]);ts=np.array([x['time'] for x in samples])
    accumulated=np.r_[0,np.cumsum((costs[1:]+costs[:-1])/2*np.diff(ts))]
    for row,cost in zip(samples,accumulated):row['mission_cost']=float(cost)
    success=np.linalg.norm(state[:3]-scenario.goal)<=config.goal_tolerance and np.linalg.norm(state[3:])<=config.goal_velocity_tolerance
    candidates=[r for e in epochs for r in e['candidates']]
    metrics={'method':method_name,'success':bool(success),'violations':0,'mission_cost':float(accumulated[-1]),
             'completion_time':current_time,'final_interval_width':interval.width,
             'uncertainty_reduction':1-interval.width/Interval(*config.initial_interval).width,
             'minimum_clearance':min(x['clearance'] for x in samples),
             'maximum_tube_utilization':max(x['tube_utilization'] for x in samples),
             'mean_planning_time':float(np.mean([e['planning_time'] for e in epochs])),
             'replans':len(epochs),'informative_candidates':len(candidates),
             'rejected_safety':sum(r['reason']=='solver_or_tube_infeasible' or r['reason']=='tube_safety' for r in candidates),
             'rejected_reconnection':sum(r['reason']=='reconnection' for r in candidates),
             'rejected_budget':sum(r['reason']=='budget' for r in candidates),
             'committed':sum(e['informative_committed'] for e in epochs),
             'exploration_spent':spent if method_name=='proposed' else 0.,'exploration_budget':budget,
             'control_saturation_count':sum(bool(np.any(np.abs(v['control'])>=np.array(config.input_limits)-1e-6)) for v in samples),'worst_solver_residual':max(e['committed']['residual'] for e in epochs),
             'final_distance_to_goal':float(np.linalg.norm(state[:3]-scenario.goal)),
             'weighted_fallbacks':len(weighted_failures),'solver_nonconvergences':sum(not e['committed']['certificate'].get('solver_converged',True) for e in epochs)}
    return {'config':asdict(config),'scenario':asdict(scenario),'truth':asdict(truth),'git_revision':revision(),
            'method':method_name,'metrics':metrics,'samples':samples,'epochs':epochs,
            'observations':observations_all,'weighted_failures':weighted_failures}
