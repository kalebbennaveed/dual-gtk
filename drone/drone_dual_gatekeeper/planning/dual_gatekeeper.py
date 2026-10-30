"""Algorithm 1; separate safety/reconnection, excess-cost budget and score."""
import numpy as np
from ..types import Candidate,CandidateStatus,DisturbanceSet
from ..estimation.uncertainty_prediction import predict_uncertainty_reduction
from .optimizer import PlanningFailure

def mission_cost(plan,config,goal,horizon=None):
    end=plan.horizon if horizon is None else min(horizon,plan.horizon)
    t=np.linspace(0,end,max(17,int(end/.02)+1));x=plan.state(t)
    u=plan.curve.acceleration(t)+plan.estimate*np.linalg.norm(x[:,3:],axis=1,keepdims=True)*x[:,3:]-np.array(config.gravity)
    return float(np.trapezoid(config.mission_input_weight*np.sum(u*u,axis=1)+config.mission_position_weight*np.sum((x[:,:3]-goal)**2,axis=1),t))

def reconnection(candidate,backup,horizon,tolerance=3e-6):
    state_error=float(np.max(abs(candidate.state(horizon)-backup.state(horizon))))
    pi,ri=candidate.tube.at(horizon);pb,rb=backup.tube.at(horizon)
    return state_error<=tolerance and np.all(pi<=pb+tolerance) and np.all(ri<=rb+tolerance)

def budget_feasible(spent,excess,budget): return spent+excess<=budget+1e-8

def candidate_horizons(backup_horizon,increment):
    return [min(i*increment,backup_horizon) for i in range(1,int(np.ceil(backup_horizon/increment))+1)]

def generate_and_select(optimizer,backup,state,interval,spent,budget):
    c=optimizer.config;s=optimizer.scenario;rows=[]
    # A certified nominal velocity cannot exceed the component speed limits.
    # This bounds the scalar quadratic regressor and the best possible width
    # prediction. Reject impossible learning advantages before a costly solve.
    best_width=min(interval.width,4*c.disturbance_bound/(np.sqrt(3)*c.velocity_limit**2))
    if c.minimum_information_advantage>0 and interval.width-best_width<c.minimum_information_advantage:
        return None,[Candidate(T,reason='insufficient_information_advantage')
                     for T in candidate_horizons(backup.horizon,c.candidate_increment)]
    for T in candidate_horizons(backup.horizon,c.candidate_increment):
        current_best=max([r.score for r in rows if r.status==CandidateStatus.FEASIBLE] or [0.])
        # Width reduction is at most the current width. Once the discounted
        # upper bound cannot beat a solved candidate, longer horizons cannot
        # change the selection. Preserve the earliest maximum on ties.
        if current_best>0 and np.exp(-c.score_discount*T)*interval.width<=current_best:
            break
        if getattr(optimizer,'verbose',False):print(f'Solving informative horizon {T:.2f} s',flush=True)
        row=Candidate(T);row.backup_cost=mission_cost(backup,c,np.array(s.goal),T)
        try:
            row.plan=optimizer.solve(state,interval,gamma=c.informative_gamma,horizon=T,endpoint=backup.state(T),terminal_tube=backup.tube.at(T),warm=backup)
        except PlanningFailure as error:
            row.status=CandidateStatus.UNSAFE;row.reason='solver_or_tube_infeasible';row.diagnostics=error.diagnostics
            # Retain finite rejected iterates for audit/visualization, never execution.
            attempt=next((v for v in reversed(error.diagnostics.get('attempts',[])) if 'coefficients' in v),None)
            if attempt and np.all(np.isfinite(attempt['coefficients'])):
                from .curve import Curve
                from ..control.tube_propagation import build_tube
                from ..environment.collision import certify_plan
                from ..types import RobustPlan
                curve=Curve(attempt['coefficients'],T,c.subdivisions);tube=build_tube(curve,interval,c)
                cert=certify_plan(curve,tube,interval,c,s);ts=np.linspace(0,T,max(21,int(T/.1)+1));xs=curve.state(ts)
                us=curve.acceleration(ts)+interval.midpoint*np.linalg.norm(xs[:,3:],axis=1,keepdims=True)*xs[:,3:]-np.array(c.gravity)
                pr=[tube.at(t) for t in ts]
                row.plan=RobustPlan(xs,us,ts,np.array([v[0] for v in pr]),np.array([np.r_[v[1],v[0]+c.sliding_lambda*v[1]] for v in pr]),0.,T,curve,tube,interval.midpoint,attempt['status'],attempt['residual'],0.,0.,cert)
                row.safe=cert['safe'] and np.max(abs(row.plan.state(0)-state))<=3e-6
                row.reconnects=reconnection(row.plan,backup,T)
                row.candidate_cost=mission_cost(row.plan,c,np.array(s.goal))
                row.reason='tube_safety' if not row.safe else ('reconnection' if not row.reconnects else 'solver_residual')
            rows.append(row);continue
        row.safe=row.plan.certificate['safe'];row.reconnects=reconnection(row.plan,backup,T)
        if not row.safe or not row.reconnects:
            row.status=CandidateStatus.UNSAFE;row.reason='tube_safety' if not row.safe else 'reconnection';rows.append(row);continue
        row.status=CandidateStatus.SAFE;row.candidate_cost=mission_cost(row.plan,c,np.array(s.goal))
        row.exploration_cost=max(0,row.candidate_cost-row.backup_cost)
        row.budget_feasible=budget_feasible(spent,row.exploration_cost,budget)
        if not row.budget_feasible:
            row.status=CandidateStatus.BUDGET_INFEASIBLE;row.reason='budget';rows.append(row);continue
        prediction=predict_uncertainty_reduction(row.plan,interval,DisturbanceSet(c.disturbance_bound))
        row.predicted_reduction=prediction.absolute_reduction;row.score=np.exp(-c.score_discount*T)*row.predicted_reduction
        from copy import copy
        baseline=copy(backup)
        baseline.nominal_states=backup.state(row.plan.timestamps)
        row.backup_predicted_reduction=predict_uncertainty_reduction(baseline,interval,DisturbanceSet(c.disturbance_bound)).absolute_reduction
        row.information_advantage=row.predicted_reduction-row.backup_predicted_reduction
        # Optional experiment-level deadband against learning already available
        # from passive execution. Zero preserves the paper's absolute score gate.
        if c.minimum_information_advantage>0 and row.information_advantage<c.minimum_information_advantage:
            row.reason='insufficient_information_advantage';rows.append(row);continue
        row.status=CandidateStatus.FEASIBLE;rows.append(row)
    feasible=[r for r in rows if r.status==CandidateStatus.FEASIBLE and r.score>1e-10]
    selected=max(feasible,key=lambda r:r.score) if feasible else None
    if selected:
        selected.status=CandidateStatus.COMMITTED;selected.selected=True
    return selected,rows
