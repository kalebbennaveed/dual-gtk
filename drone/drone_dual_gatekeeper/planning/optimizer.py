"""Nonlinear B-spline direct optimal-control transcription with IPOPT.
Only position spline coefficients and (backup/weighted) duration are decisions.
Dynamics are satisfied by inverse feedforward; Phi/rho are exact, eliminated
LTI recursions driven by Bernstein cell bounds, not independent decisions.
"""
import time
from dataclasses import replace
import casadi as ca
import numpy as np
from .curve import basis,Curve
from ..types import RobustPlan
from ..control.tube_propagation import propagate,build_tube
from ..environment.collision import certify_plan

class PlanningFailure(RuntimeError):
    def __init__(self,message,diagnostics):
        super().__init__(message+': '+str(diagnostics));self.diagnostics=diagnostics

class TrajectoryOptimizer:
    def __init__(self,config,scenario):
        self.config=config;self.scenario=scenario;self.cache={};self.failures=[]
    def _build(self,horizon,reconnect,information_discount=0.):
        c=self.config;s=self.scenario;n=c.control_points
        b,edges,hulls=basis(n,c.subdivisions)
        cp=ca.MX.sym('position_coefficients',n,3)
        variable_time=horizon is None
        duration=ca.MX.sym('duration') if variable_time else float(horizon)
        z=ca.vertcat(ca.vec(cp),duration) if variable_time else ca.vec(cp)
        pars=ca.MX.sym('planning_parameters',21)
        x0=pars[:6];target=pars[6:12];theta=pars[12:14];gamma=pars[14]
        terminal_phi=pars[15:18];terminal_rho=pars[18:21]
        estimate=(theta[0]+theta[1])/2;delta=(theta[1]-theta[0])/2
        def maximum(a,b): return (a+b+ca.sqrt((a-b)**2+c.smooth_max_width**2))/2
        def absolute(a): return ca.sqrt(a*a+c.smooth_abs_width**2)
        def maxall(a):
            result=a[0]
            for j in range(1,a.numel()):result=maximum(result,a[j])
            return result
        equal=[];ineq=[]
        equal.extend([cp[0,:].T-x0[:3],cp[-1,:].T-target[:3],
                      (ca.DM(b(0,nu=1)).T@cp).T/duration-x0[3:],
                      (ca.DM(b(1,nu=1)).T@cp).T/duration-target[3:]])
        if c.nominal_altitude is not None:
            # A planar nominal mission still tracks the full 6D uncertain plant.
            # The four endpoint coefficients already enforce measured endpoint
            # states; only unconstrained interior altitude coefficients are fixed.
            equal.append(cp[2:-2,2]-c.nominal_altitude)
        phi=ca.DM(c.initial_boundary_layer);rho=ca.DM.zeros(3)
        mission=0;gramian=0
        for i,(p_mat,v_mat,a_mat) in enumerate(hulls):
            h=(edges[i+1]-edges[i])*duration
            pc=ca.DM(p_mat)@cp;vc=ca.DM(v_mat)@cp/duration;ac=ca.DM(a_mat)@cp/duration**2
            def bounds(mat):
                return ca.vertcat(*[-maxall(-mat[:,j]) for j in range(3)]),ca.vertcat(*[maxall(mat[:,j]) for j in range(3)])
            lo,hi=bounds(pc);vl,vh=bounds(vc);al,ah=bounds(ac)
            vabs=maximum(absolute(vl),absolute(vh))
            drag=delta*ca.sqrt(ca.sumsqr(vabs)+1e-16)*vabs
            phi1,rho1=propagate(phi,rho,drag+c.disturbance_bound+c.eta,h,c.alpha,c.sliding_lambda,ca.exp)
            pmax=maximum(phi,phi1)
            rmax=maximum(rho,rho*ca.exp(-c.sliding_lambda*h)+pmax*(1-ca.exp(-c.sliding_lambda*h))/c.sliding_lambda)
            ev=pmax+c.sliding_lambda*rmax;va=vabs+ev
            gain=maximum(0,delta*ca.sqrt(ca.sumsqr(va)+1e-16)*va-drag+c.alpha*pmax)
            abound=maximum(absolute(al-ca.DM(c.gravity)),absolute(ah-ca.DM(c.gravity)))
            ubound=abound+estimate*ca.sqrt(ca.sumsqr(va)+1e-16)*va+c.sliding_lambda*ev+gain
            ineq.extend([lo-rmax-ca.DM(s.lower),ca.DM(s.upper)-hi-rmax,
                         c.velocity_limit-va,ca.DM(c.input_limits)-ubound])
            for obs in s.obstacles:
                if obs['kind']=='box':
                    ol=ca.DM(obs['lower'])-rmax-c.drone_radius;oh=ca.DM(obs['upper'])+rmax+c.drone_radius
                    ineq.append(maxall(ca.vertcat(ol-hi,lo-oh)))
                else:
                    dims=2 if obs['kind']=='cylinder' else 3
                    center=ca.DM(obs['center'])
                    points=pc[:,:dims]
                    direction=(ca.sum1(points)/points.size1()).T-center[:dims]
                    normal=direction/ca.sqrt(ca.sumsqr(direction)+1e-16)
                    projection=(points-ca.repmat(center[:dims].T,points.size1(),1))@normal
                    # Conservative lower support bound for the entire convex
                    # hull plus the propagated position-error box.
                    horizontal=-maxall(-projection)-obs['radius']-c.drone_radius-ca.dot(absolute(normal),rmax[:dims])
                    if dims==2:
                        vertical=maximum(center[2]-obs['height']/2-hi[2]-rmax[2]-c.drone_radius,
                                           lo[2]-rmax[2]-c.drone_radius-center[2]-obs['height']/2)
                        ineq.append(maximum(horizontal,vertical)-c.smooth_max_width/2)
                    else: ineq.append(horizontal)
            # Simpson quadrature of mission and information, no tube-size penalty.
            for q,w in [(edges[i],1/6),((edges[i]+edges[i+1])/2,4/6),(edges[i+1],1/6)]:
                r=(ca.DM(b(q)).T@cp).T;v=(ca.DM(b(q,nu=1)).T@cp).T/duration
                a=(ca.DM(b(q,nu=2)).T@cp).T/duration**2
                u=a+estimate*ca.sqrt(ca.sumsqr(v)+1e-16)*v-ca.DM(c.gravity)
                mission+=h*w*(c.mission_input_weight*ca.sumsqr(u)+c.mission_position_weight*ca.sumsqr(r-ca.DM(s.goal)))
                # Receding-horizon weighting values excitation near the current
                # decision instead of deferring all information to the goal.
                # The zero-discount setting preserves the original objective.
                discount=ca.exp(-information_discount*q*duration) if information_discount else 1.
                gramian+=h*w*discount*ca.sumsqr(v)**2
            phi,rho=phi1,rho1
        if reconnect:
            ineq.extend([terminal_phi-phi,terminal_rho-rho])
        else:
            ineq.extend([c.goal_tolerance/np.sqrt(3)-rho,c.goal_velocity_tolerance/np.sqrt(3)-(phi+c.sliding_lambda*rho)])
        eq=ca.vertcat(*equal);iq=ca.vertcat(*ineq);g=ca.vertcat(eq,iq)
        options={'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes',
                 'ipopt.max_iter':c.solver_iterations,'ipopt.tol':c.solver_tolerance,
                 'ipopt.acceptable_tol':c.solver_acceptable_tolerance,'ipopt.acceptable_constr_viol_tol':1e-6,'ipopt.acceptable_dual_inf_tol':1.0,'ipopt.acceptable_obj_change_tol':1e-6,'ipopt.acceptable_iter':3}
        if not variable_time:
            options['ipopt.hessian_approximation']=c.informative_hessian_approximation
            if c.informative_solver_iterations is not None:
                options['ipopt.max_iter']=c.informative_solver_iterations
        else:
            options['ipopt.hessian_approximation']=c.solver_hessian_approximation
        solver=ca.nlpsol('trajectory','ipopt',{'x':z,'p':pars,'f':mission-gamma*ca.log(gramian+c.information_epsilon),'g':g},options)
        evaluate=ca.Function('diagnostics',[z,pars],[g,mission,gramian])
        nz=int(z.numel());lb=np.full(nz,-np.inf);ub=np.full(nz,np.inf)
        if variable_time: lb[-1]=c.min_horizon;ub[-1]=c.max_horizon
        return solver,evaluate,int(eq.numel()),int(g.numel()),lb,ub,variable_time
    def solve(self,state,interval,*,gamma=0.,horizon=None,endpoint=None,terminal_tube=None,warm=None,information_discount=0.):
        started=time.perf_counter();c=self.config;s=self.scenario
        if information_discount<0:raise ValueError('information discount must be nonnegative')
        reconnect=terminal_tube is not None;key=(horizon,reconnect,information_discount)
        if key not in self.cache:self.cache[key]=self._build(horizon,reconnect,information_discount)
        solver,ev,neq,ng,lb,ub,variable_time=self.cache[key]
        endpoint=np.r_[s.goal,np.zeros(3)] if endpoint is None else np.asarray(endpoint)
        terminal_phi,terminal_rho=(np.ones(3)*1e6,np.ones(3)*1e6) if terminal_tube is None else terminal_tube
        pars=np.r_[state,endpoint,interval.lower,interval.upper,gamma,terminal_phi,terminal_rho]
        duration=float(horizon) if horizon is not None else min(c.max_horizon,max(6,np.linalg.norm(endpoint[:3]-state[:3])/1.2+4))
        if horizon is None and warm is not None:duration=float(np.clip(warm.horizon,c.min_horizon,c.max_horizon))
        n=c.control_points;bas,_,_=basis(n,c.subdivisions)
        grid=np.linspace(0,1,n)
        if warm is not None:
            positions=warm.curve.value(grid*(horizon if horizon is not None else warm.horizon))
            initial=np.linalg.lstsq(bas(grid),positions,rcond=None)[0]
        else: initial=(1-grid[:,None])*state[:3]+grid[:,None]*endpoint[:3]
        initial[0]=state[:3];initial[-1]=endpoint[:3]
        knotspan=1/(n-3)
        initial[1]=state[:3]+state[3:]*duration*knotspan/3
        initial[-2]=endpoint[:3]-endpoint[3:]*duration*knotspan/3
        seeds=[initial]
        if horizon is None:
            # Homotopy seeds only. Every final path is optimized, no waypoint fallback.
            for sign in [-1,1]:
                seed=initial.copy();seed[2:-2,1]+=sign*c.homotopy_seed_offset*np.sin(np.pi*grid[2:-2]);seeds.append(seed)
            if gamma>0 and information_discount>0:
                heading=endpoint[:3]-state[:3]
                lateral=np.array([-heading[1],heading[0],0.])
                lateral/=max(np.linalg.norm(lateral),1e-12)
                for sign in (-1,1):
                    seed=initial.copy()
                    seed[2:-2]+=sign*c.homotopy_seed_offset*np.sin(2*np.pi*grid[2:-2,None])*lateral
                    seeds.append(seed)
        elif gamma>0:
            # Informative local minima can require a velocity sign reversal.
            # These are optimization guesses, never executable fallback paths.
            # Endpoint coefficients stay fixed; every result must pass the
            # identical dynamics, tube, authority and reconnection constraints.
            heading=endpoint[:3]-state[:3]
            heading=heading/max(np.linalg.norm(heading),1e-12)
            for amplitude in c.excitation_seed_amplitudes:
                seed=initial.copy()
                seed[2]=state[:3]-amplitude*heading
                seed[3]=state[:3]+amplitude*heading
                seeds.append(seed)
        attempts=[];best=None
        for seed in seeds:
            z0=seed.flatten(order='F');z0=np.r_[z0,duration] if variable_time else z0
            sol=solver(x0=z0,p=pars,lbx=lb,ubx=ub,lbg=np.r_[np.zeros(neq),np.zeros(ng-neq)],
                       ubg=np.r_[np.zeros(neq),np.full(ng-neq,np.inf)])
            z=np.array(sol['x']).ravel();gg,j,info=ev(z,pars);gg=np.array(gg).ravel()
            residual=max(float(np.max(np.abs(gg[:neq]))),float(np.max(np.maximum(-gg[neq:],0))))
            status=solver.stats()['return_status']
            attempt={'status':status,'residual':residual,'iterations':solver.stats()['iter_count'],'cost':float(j),'duration':float(z[-1]) if variable_time else horizon};attempts.append(attempt)
            histories=solver.stats().get('iterations',{})
            attempt['dual_residual']=float(histories.get('inf_du',[float('nan')])[-1])
            attempt['coefficients']=z[:3*n].reshape(n,3,order='F').tolist()
            attempt['converged']=status in ['Solve_Succeeded','Solved_To_Acceptable_Level']
            # A certified feasible iterate is a valid robust policy even if IPOPT
            # reaches its iteration cap. Preserve the raw status; do not claim
            # stationarity or global optimality. All methods use this same rule.
            if status not in ['Solve_Succeeded','Solved_To_Acceptable_Level','Maximum_Iterations_Exceeded','Search_Direction_Becomes_Too_Small'] or not np.all(np.isfinite(gg)) or residual>3e-6: continue
            cp=z[:3*n].reshape(n,3,order='F');T=z[-1] if variable_time else horizon
            curve=Curve(cp,T,c.subdivisions);tube=build_tube(curve,interval,c)
            cert=certify_plan(curve,tube,interval,c,s)
            cert['solver_converged']=attempt['converged'];cert['dual_residual']=attempt['dual_residual']
            if not cert['safe']:attempt['certificate']=cert;continue
            ts=np.linspace(0,T,max(2,int(np.ceil(T/0.1))+1));xs=curve.state(ts)
            us=curve.acceleration(ts)+interval.midpoint*np.linalg.norm(xs[:,3:],axis=1,keepdims=True)*xs[:,3:]-np.array(c.gravity)
            phir=[tube.at(t) for t in ts]
            plan=RobustPlan(xs,us,ts,np.array([x[0] for x in phir]),np.array([np.r_[x[1],x[0]+c.sliding_lambda*x[1]] for x in phir]),
                            float(j),float(T),curve,tube,interval.midpoint,status,residual,time.perf_counter()-started,float(info),cert)
            objective=float(j)-gamma*np.log(float(info)+c.information_epsilon)
            if best is None or objective<best[0]:best=(objective,plan)
            # Compare weighted local minima: a straight warm start alone can
            # postpone excitation to a terminal loop. Seed paths are never flown.
            if not (variable_time and gamma>0 and information_discount>0):
                break  # Other solves retain their existing feasible-iterate rule.
        if best is not None:best[1].certificate['solver_attempts']=attempts
        if best is None:
            self.failures.extend(attempts)
            if variable_time and c.solver_hessian_approximation!='limited-memory':
                # A numerical failure is not a safety certificate. Retry the
                # same constrained problem with the spatial-course solver;
                # every returned trajectory still passes the full tube audit.
                retry=TrajectoryOptimizer(replace(c,solver_hessian_approximation='limited-memory',
                                                   solver_iterations=max(1000,c.solver_iterations)),s)
                result=retry.solve(state,interval,gamma=gamma,horizon=horizon,endpoint=endpoint,
                                   terminal_tube=terminal_tube,warm=warm,information_discount=information_discount)
                result.certificate['solver_retry']={'hessian':'limited-memory','maximum_iterations':retry.config.solver_iterations,
                                                    'initial_attempts':attempts}
                result.planning_time=time.perf_counter()-started
                return result
            raise PlanningFailure('No certified nonlinear solution',{'attempts':attempts,'horizon':horizon,'interval':[interval.lower,interval.upper]})
        best[1].planning_time=time.perf_counter()-started
        return best[1]
