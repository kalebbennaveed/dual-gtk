"""Paper-ready tables for executed failures, separated from rollout predictions."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import shutil
import zipfile
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from .experiments import load_rows,load_record,atomic_json,METHODS
from .paper_report import NAMES,COLORS,style,save_figure,table_files,write_csv,fmt
from .track import Track
from .model import Model,commands
from .legacy_gatekeeper import Gatekeeper,Policy
from .belief import FrictionSet
from .safety import binomial_upper,sampled_rollouts
from .stress_diagnostics import rollout_outcomes


def interval(k,n):
    return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.)


def aggregate(rows):
    result=[]
    for group in ('launch','lap'):
        for scenario,eps in (('traction',.01),('traction_relaxed',.05)):
            for method in METHODS:
                rs=[r for r in rows if r['case']['group']==group and r['case']['scenario']==scenario and r['case']['method']==method]
                if not rs:continue
                ms=[r['metrics'] for r in rs if r['status']=='ok'];n=len(rs)
                tire=sum((m.get('violation') or {}).get('kind')=='tire_force' for m in ms)
                state=sum((m.get('violation') or {}).get('kind')=='state' for m in ms)
                unavailable=sum(m['certificate_unavailable'] for m in ms);errors=n-len(ms)
                completed=[m for m in ms if m['lap_complete']]
                lower,upper=interval(tire+state,n)
                result.append(dict(cohort=group,epsilon=eps,method=method,trials=n,tire_violations=tire,
                    state_violations=state,violations=tire+state,violation_rate=(tire+state)/n,
                    ci95_lower=lower,ci95_upper=upper,certificate_unavailable=unavailable,audit_errors=errors,
                    unsafe_or_unresolved_upper95=binomial_upper(tire+state+unavailable+errors,n,.05),
                    smid_exclusions=sum(m['containment_violations']>0 for m in ms),
                    completed_laps=len(completed),
                    horizon_reached=sum(m['termination']=='timeout' and abs(m['time']-rs[0]['case']['seconds'])<1e-8 for m in ms),
                    mean_completed_lap=float(np.mean([m['time'] for m in completed])) if completed else None,
                    mean_completed_cost=float(np.mean([m['cost'] for m in completed])) if completed else None,
                    minimum_tire_margin=min((m['minimum_tire_margin'] for m in ms if m['minimum_tire_margin'] is not None),default=None)))
    return result


def diagnostic(destination):
    """Illustrative seed fixed during discovery; independent holdout afterwards."""
    tr=Track('technical');model=Model(tr);b=FrictionSet(.76,1.05)
    x=np.array([0.,0.,0.,8.,0.,8*tr.kappa(0.),np.arctan(3.05*tr.kappa(0.))])
    rows=[];profiles=[]
    mus=np.random.default_rng(20261193).uniform(.76,1.05,100000)
    grid=np.linspace(.76,1.05,4001)
    for scenario,eps in (('traction',.01),('traction_relaxed',.05)):
        g=Gatekeeper(model,scenario=scenario,seed=7);g.replan(x,b,0.)
        a=g.events[-1]['selected'];c=a['certificate'];p=g.committed
        refs=np.array([p.action_reference(i*.05) for i in range(round(c['horizon']/.05))])
        times,kinds=rollout_outcomes(x,mus,refs,tr.curvature,tr.ds)
        n=len(mus);fail=int((kinds>0).sum());lo,hi=interval(fail,n)
        rows.append(dict(epsilon=eps,policy=p.kind,speed=p.speed,amplitude=p.amplitude,duration=p.duration,
            certificate_failures=c['failures'],certificate_samples=c['samples'],certificate_upper=c['failure_upper'],
            certificate_seed=c['sample_seed'],holdout_seed=20261193,holdout_samples=n,holdout_failures=fail,
            holdout_physical=int(((kinds==1)|(kinds==2)).sum()),holdout_terminal=int((kinds==3).sum()),
            holdout_nonfinite=int((kinds==4).sum()),holdout_rate=fail/n,ci95_lower=lo,ci95_upper=hi))
        t,kind=rollout_outcomes(x,grid,refs,tr.curvature,tr.ds)
        for mu,age,reason in zip(grid,t,kind):profiles.append(dict(epsilon=eps,true_mu=mu,failure_time=float(age) if np.isfinite(age) else None,failure_kind=int(reason)))
    write_csv(destination/'tables'/'policy_holdout.csv',rows)
    write_csv(destination/'tables'/'policy_friction_profile.csv',profiles)
    return rows,profiles


def build_report(directory):
    directory=Path(directory).resolve();manifest,rows=load_rows(directory)
    if manifest.get('protocol',2)>=3:raise ValueError('Use the MPPI report for schema-3 records; this renderer is historical.')
    out=directory/'report';out.mkdir(exist_ok=True)
    figs=out/'figures';tables=out/'tables';figs.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    summary=aggregate(rows);write_csv(tables/'safety_statistics.csv',summary)
    flat=[];failures=[];rechecks=[];track=Track('technical')
    for r in rows:
        flat.append({**r['case'],'status':r['status'],**r.get('metrics',{})})
        if r.get('metrics',{}).get('constraint_violated'):
            record=load_record(directory,r);m=r['metrics'];v=m['violation']
            e=next(e for e in reversed(record['events']) if e['status']=='committed' and e['time']<=v['time'])
            c=e['selected']['certificate'];belief=record['frames'][-1]['mu']
            assert c['accepted'] and c['failure_upper']<=c['epsilon']
            assert c['support'][0]<=r['case']['true_mu']<=c['support'][1]
            failures.append(dict(id=r['case']['id'],cohort=r['case']['group'],method=r['case']['method'],epsilon=m['epsilon'],
                true_mu=r['case']['true_mu'],seed=r['case']['seed'],time=v['time'],kind=v['kind'],margin=v.get('margin'),
                commitment_time=e['time'],policy=e['policy']['kind'],failure_mode=record['frames'][-1]['mode'],certificate_upper=c['failure_upper'],
                sample_failures=c['failures'],samples=c['samples'],certified_lower=c['support'][0],certified_upper=c['support'][1],
                true_fraction_in_certified_interval=(r['case']['true_mu']-c['support'][0])/(c['support'][1]-c['support'][0]),
                smid_lower=belief[0],smid_upper=belief[1],audit_passed=r['audit']['passed']))
            # Reproduce the original random certificate, then use independent
            # samples for diagnosis. Neither set changes the executed trial.
            x=np.array(record['frames'][round(e['time']/record['dt'])]['state'])
            policy=Policy(**e['policy'])
            refs=np.array([commands(j*.05,policy.duration,policy.speed,policy.amplitude,8.) for j in range(round(c['horizon']/.05))])
            mus=np.random.default_rng(c['sample_seed']).uniform(*c['support'],c['samples'])
            replay,*_=sampled_rollouts(x,mus,refs,track.curvature,track.ds,np.max(np.abs(track.curvature)),True,len(mus))
            assert replay==c['failures'],'Original accepted sample count does not reproduce'
            age,reason=rollout_outcomes(x,np.array([r['case']['true_mu']]),refs,track.curvature,track.ds)
            assert reason[0] in (1,2) and abs(age[0]-(r['metrics']['time']-e['time']))<1e-8
            seed=int(np.random.SeedSequence([20261211,len(rechecks)]).generate_state(1)[0]);n=10000
            mus=np.random.default_rng(seed).uniform(*c['support'],n)
            _,kind=rollout_outcomes(x,mus,refs,track.curvature,track.ds)
            k=int((kind>0).sum());lo,hi=interval(k,n)
            rechecks.append(dict(id=r['case']['id'],certificate_seed=c['sample_seed'],certificate_failures=c['failures'],
                replay_failures=int(replay),certificate_upper=c['failure_upper'],actual_failure_age=float(age[0]),
                actual_failure_kind=int(reason[0]),holdout_seed=seed,holdout_samples=n,holdout_failures=k,
                holdout_physical=int(((kind==1)|(kind==2)).sum()),holdout_terminal=int((kind==3).sum()),
                holdout_rate=k/n,ci95_lower=lo,ci95_upper=hi))
    write_csv(tables/'all_episodes.csv',flat);write_csv(tables/'executed_failures.csv',failures)
    write_csv(tables/'failed_policy_rechecks.csv',rechecks)
    headers=['Tolerance','Method','Trials','Tire violations','State violations','Observed failure','95% interval','Unresolved / errors']
    def table(group):
        return table_files(tables,group+'_safety',headers,
            [[f"{100*r['epsilon']:g}%",NAMES[r['method']],r['trials'],r['tire_violations'],r['state_violations'],
              f"{100*r['violation_rate']:.2f}%",f"[{100*r['ci95_lower']:.2f}, {100*r['ci95_upper']:.2f}]%",
              f"{r['certificate_unavailable']} / {r['audit_errors']}"] for r in summary if r['cohort']==group])
    launch_html=table('launch');lap_html=table('lap')
    perf_html=table_files(tables,'lap_performance',['Tolerance','Method','Completed / trials','Mean completed lap [s]','Mean completed cost'],
        [[f"{100*r['epsilon']:g}%",NAMES[r['method']],f"{r['completed_laps']} / {r['trials']}",fmt(r['mean_completed_lap']),fmt(r['mean_completed_cost'])] for r in summary if r['cohort']=='lap'])
    failure_html=table_files(tables,'failure_examples',['Cohort','Tolerance','Method','True μ','Failure time [s]','Accepted upper','Failed certificate samples'],
        [[r['cohort'],f"{100*r['epsilon']:g}%",NAMES[r['method']],fmt(r['true_mu'],6),fmt(r['time']),f"{100*r['certificate_upper']:.3f}%",f"{r['sample_failures']} / {r['samples']}"] for r in failures[:12]])
    holdout,profiles=diagnostic(out)
    holdout_html=table_files(tables,'policy_validation',['Tolerance','Initial policy','Online upper','Holdout failures','Holdout rate','95% interval'],
        [[f"{100*r['epsilon']:g}%",r['policy'],f"{100*r['certificate_upper']:.3f}%",f"{r['holdout_failures']} / {r['holdout_samples']}",f"{100*r['holdout_rate']:.3f}%",f"[{100*r['ci95_lower']:.3f}, {100*r['ci95_upper']:.3f}]%"] for r in holdout])
    style()
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.0),gridspec_kw={'width_ratios':[1,2]},layout='constrained')
    for ax,group in zip(axs,('launch','lap')):
        rs=[r for r in summary if r['cohort']==group]
        labels=[]
        for j,r in enumerate(rs):
            y=100*r['violation_rate'];lo=100*r['ci95_lower'];hi=100*r['ci95_upper']
            ax.errorbar(j,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=3,color=COLORS[r['method']])
            labels.append(f"{NAMES[r['method']].replace('Weighted information','Weighted')}\n{100*r['epsilon']:g}%")
        ax.set(xticks=range(len(rs)),xticklabels=labels,ylabel='Executed constraint violations [%]',title='8-second launch trials' if group=='launch' else 'Independent full-lap trials')
        ax.tick_params(axis='x',labelrotation=45 if group=='lap' else 0);ax.set_ylim(bottom=-.25)
    save_figure(fig,figs,'01_executed_failure_rates')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for ax,group in zip(axs,('launch','lap')):
        for eps,color,marker in ((.01,'#167f69','o'),(.05,'#b17932','x')):
            rs=[r for r in rows if r['case']['group']==group and r['case']['method']=='proposed' and r.get('metrics',{}).get('epsilon')==eps]
            xs=[r['case']['true_mu'] for r in rs];ys=[int(r['metrics']['constraint_violated']) for r in rs]
            ax.scatter(xs,ys,s=12,alpha=.55,marker=marker,color=color,label=f'{100*eps:g}% tolerance')
        ax.set(xlabel='True tire coefficient μ',yticks=[0,1],yticklabels=['No violation','Tire constraint'],title='Launch cohort' if group=='launch' else 'Full-lap cohort',ylim=(-.15,1.3))
    axs[0].legend(frameon=False);save_figure(fig,figs,'02_failures_vs_true_friction')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for r,color in zip(holdout,('#167f69','#b17932')):
        eps=r['epsilon'];rs=[z for z in profiles if z['epsilon']==eps]
        axs[0].plot([z['true_mu'] for z in rs],[z['failure_kind']>0 for z in rs],color=color,label=f'{100*eps:g}% tolerance')
    axs[0].set(xlabel='True tire coefficient μ',ylabel='Complete committed policy fails',yticks=[0,1],xlim=(.76,.79));axs[0].legend(frameon=False)
    for i,r in enumerate(holdout):
        y=100*r['holdout_rate'];axs[1].errorbar(i,y,yerr=[[y-100*r['ci95_lower']],[100*r['ci95_upper']-y]],fmt='o',color='#167f69',capsize=4)
        axs[1].scatter(i,100*r['certificate_upper'],marker='D',color='#b17932')
        axs[1].scatter(i,100*r['epsilon'],marker='_',s=300,color='#333')
    axs[1].set(xticks=[0,1],xticklabels=['1% setting','5% setting'],ylabel='Failure probability / bound [%]')
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[Line2D([],[],marker='o',color='#167f69',ls='',label='100,000 fresh rollouts'),Line2D([],[],marker='D',color='#b17932',ls='',label='Online upper bound'),Line2D([],[],marker='_',color='#333',ls='',label='Allowed tolerance')],frameon=False,fontsize=8)
    save_figure(fig,figs,'03_accepted_nonzero_risk')
    # A complete empirical time-to-violation curve, including censored safe runs.
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for ax,group,horizon in zip(axs,('launch','lap'),(8.,360.)):
        t=np.linspace(0,horizon,721)
        for eps,color in ((.01,'#167f69'),(.05,'#b17932')):
            rs=[r for r in rows if r['case']['group']==group and r['case']['method']=='proposed' and r.get('metrics',{}).get('epsilon')==eps]
            if not rs:continue
            failed=[r['metrics']['time'] for r in rs if r['metrics']['constraint_violated']]
            y=[100*sum(f<=age for f in failed)/len(rs) for age in t]
            ax.step(t,y,where='post',color=color,label=f'{100*eps:g}% tolerance')
        ax.set(xlabel='Episode time [s]',ylabel='Trials with an observed violation [%]',title=group.capitalize()+' cohort');ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False);save_figure(fig,figs,'04_time_to_violation')
    # Save one matched failed/safe execution from the confirmatory launch cohort.
    chosen=next((r for r in rows if r['case']['group']=='launch' and r.get('metrics',{}).get('constraint_violated')),None)
    example_info=None
    if chosen:
        peer=next(r for r in rows if r['case']['group']=='launch' and r['case']['trial']==chosen['case']['trial'] and r['case']['scenario']=='traction')
        records=[load_record(directory,r) for r in (peer,chosen)]
        example_info=dict(trial=chosen['case']['trial'],true_mu=chosen['case']['true_mu'],selection='First failed launch trial in prespecified manifest order',ids=[peer['case']['id'],chosen['case']['id']])
        fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
        for record,color in zip(records,('#167f69','#b17932')):
            fs=record['frames'];t=[f['t'] for f in fs];eps=record['scenario']['epsilon'];label=f'{100*eps:g}% tolerance'
            axs[0].plot(t,[f['state'][3] for f in fs],color=color,marker='o' if len(fs)==1 else None,label=label)
            axs[1].plot(t,[f.get('traction_margin',np.nan) for f in fs],color=color,marker='o' if len(fs)==1 else None,label=label)
            axs[2].plot(t,[f['mu'][1]-f['mu'][0] for f in fs],color=color,marker='o' if len(fs)==1 else None,label=label)
        axs[1].axhline(0,color='#555',ls='--');axs[0].legend(frameon=False)
        for ax,label in zip(axs,('Speed [m/s]','Squared tire-force slack','SMID interval width')):ax.set(xlabel='Time [s]',ylabel=label)
        fig.suptitle(f"Matched executed trial · true μ = {example_info['true_mu']:.6f}",fontsize=11)
        save_figure(fig,figs,'05_matched_failure_trace')
        for record,name in zip(records,('matched-1pct.json','matched-5pct.json')):atomic_json(out/name,record)
    later=next((r for r in rows if r['case']['group']=='launch' and r.get('metrics',{}).get('constraint_violated') and r['metrics']['time']>0),None)
    later_info=None
    if later:
        peer=next(r for r in rows if r['case']['group']=='launch' and r['case']['trial']==later['case']['trial'] and r['case']['scenario']=='traction')
        paired=[load_record(directory,r) for r in (peer,later)]
        failtime=later['metrics']['time'];truth=later['case']['true_mu']
        later_info=dict(trial=later['case']['trial'],true_mu=truth,time=failtime,selection='First positive-time launch failure in manifest order; descriptive example')
        fig,axs=plt.subplots(2,2,figsize=(10.5,6.5),layout='constrained')
        for record,color in zip(paired,('#167f69','#b17932')):
            fs=record['frames'];t=np.array([f['t'] for f in fs]);eps=record['scenario']['epsilon'];label=f'{100*eps:g}% tolerance'
            axs[0,0].plot(t,[f['state'][3] for f in fs],color=color,label=label)
            axs[0,1].plot(t,[f.get('traction_margin',np.nan) for f in fs],color=color,label=label)
            lo=np.array([f['mu'][0] for f in fs])-truth;hi=np.array([f['mu'][1] for f in fs])-truth
            axs[1,0].fill_between(t,lo,hi,color=color,alpha=.2)
            axs[1,0].plot(t,lo,color=color,lw=1);axs[1,0].plot(t,hi,color=color,lw=1)
            events=[e for e in record['events'] if e['status']=='committed']
            axs[1,1].scatter([e['time'] for e in events],[100*e['selected']['certificate']['failure_upper'] for e in events],color=color,s=16,label=label)
            axs[1,1].axhline(100*eps,color=color,ls=':',lw=1)
        axs[0,0].legend(frameon=False);axs[0,1].axhline(0,color='#555',ls='--')
        axs[0,1].set_xlim(max(0,failtime-1),8.);axs[0,1].set_yscale('symlog',linthresh=1e-3);axs[0,1].set_ylim(-.005,.8)
        axs[0,1].scatter(failtime,later['metrics']['violation']['margin'],marker='x',s=70,color='#b74640',zorder=4)
        axs[1,0].axhline(0,color='#333',ls='--');axs[1,0].set_yscale('symlog',linthresh=1e-5)
        for ax,label in zip(axs.flat,('Speed [m/s]','Tire-force slack (zoom)','SMID bounds relative to true μ','Accepted upper bound [%]')):ax.set(xlabel='Time [s]',ylabel=label)
        fig.suptitle(f'Violation after driving · true μ = {truth:.6f} · failure at {failtime:.2f} s',fontsize=11)
        save_figure(fig,figs,'06_failure_after_driving')
        for record,name in zip(paired,('later-1pct.json','later-5pct.json')):atomic_json(out/name,record)
    lap_failure=next((r for r in rows if r['case']['group']=='lap' and r['case']['method']=='proposed' and r.get('metrics',{}).get('constraint_violated') and r['metrics']['time']>0),None)
    lap_example=None
    if lap_failure:
        peer=next(r for r in rows if r['case']['group']=='lap' and r['case']['method']=='proposed' and r['case']['trial']==lap_failure['case']['trial'] and r['case']['scenario']=='traction')
        paired=[load_record(directory,r) for r in (peer,lap_failure)]
        age=lap_failure['metrics']['time'];truth=lap_failure['case']['true_mu'];horizon=age+1.
        lap_example=dict(trial=lap_failure['case']['trial'],true_mu=truth,time=age,selection='First positive-time proposed full-lap failure in manifest order; descriptive example')
        fig,axs=plt.subplots(2,2,figsize=(10.5,6.5),layout='constrained')
        for record,color in zip(paired,('#167f69','#b17932')):
            fs=[f for f in record['frames'] if f['t']<=horizon];t=np.array([f['t'] for f in fs]);eps=record['scenario']['epsilon'];label=f'{100*eps:g}% tolerance'
            axs[0,0].plot(t,[f['state'][0] for f in fs],color=color,label=label)
            axs[0,1].plot(t,[f.get('traction_margin',np.nan) for f in fs],color=color)
            lo=np.array([f['mu'][0] for f in fs])-truth;hi=np.array([f['mu'][1] for f in fs])-truth
            axs[1,0].fill_between(t,lo,hi,color=color,alpha=.2)
            axs[1,0].plot(t,lo,color=color,lw=1);axs[1,0].plot(t,hi,color=color,lw=1)
            events=[e for e in record['events'] if e['status']=='committed' and e['time']<=horizon]
            axs[1,1].scatter([e['time'] for e in events],[100*e['selected']['certificate']['failure_upper'] for e in events],color=color,s=10)
            axs[1,1].axhline(100*eps,color=color,ls=':',lw=1)
        axs[0,0].legend(frameon=False);axs[0,1].axhline(0,color='#555',ls='--');axs[0,1].set_xlim(age-1.,age+.5);axs[0,1].set_yscale('symlog',linthresh=1e-3);axs[0,1].set_ylim(-.005,.8)
        axs[0,1].scatter(age,lap_failure['metrics']['violation']['margin'],marker='x',s=70,color='#b74640',zorder=4)
        axs[1,0].axhline(0,color='#333',ls='--');axs[1,0].set_yscale('symlog',linthresh=1e-5)
        for ax,label in zip(axs.flat,('Circuit progress [m]','Tire-force slack (zoom)','SMID bounds relative to true μ','Accepted upper bound [%]')):ax.set(xlabel='Time [s]',ylabel=label)
        fig.suptitle(f'Full-lap trial · true μ = {truth:.6f} · 5% failure at {age:.2f} s',fontsize=11)
        save_figure(fig,figs,'07_full_lap_failure')
        for record,name in zip(paired,('lap-example-1pct.json','lap-example-5pct.json')):atomic_json(out/name,record)
    atomic_json(out/'analysis.json',dict(summary=summary,illustration=example_info,
        later_failure=later_info,
        lap_failure=lap_example,
        original_certificates_reproduced=len(rechecks),
        report_sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('stress_report.py','stress_diagnostics.py')},
        total=len(rows),violations=len(failures),audit_errors=sum(r['status']!='ok' for r in rows)))
    shutil.copy2(directory/'manifest.json',out/'manifest.json')
    shutil.copytree(directory/'discovery',out/'discovery',dirs_exist_ok=True)
    shutil.copytree(directory/'sources',out/'sources',dirs_exist_ok=True)
    for name in ('stress_report.py','stress_diagnostics.py'):shutil.copy2(Path(__file__).parent/name,out/'sources'/name)
    (out/'README.md').write_text('''# Accepted-risk launch stress study

The initial tire interval is [0.76, 1.05], compared with [0.45, 1.05] in the reference demos. The track, dynamics, sensor, controllers, candidate library, N=1000 samples, delta=0.05 per decision, and 1% / 5% risk tolerances are unchanged. The two reference demos and their 940 recorded episodes remain separate.

An exploratory search screened 120 starting positions and then six prior lower bounds. That search selected this setting. The manifest was frozen before fresh IID confirmatory trials: 1,000 paired eight-second proposed-controller launches per tolerance, plus 100 independently drawn paired full-lap trials per method and tolerance. Raw failed runs are retained; there is no retry until success, no friction outside the declared bounds, and no changed plant or safety constraints. Discovery records are included separately. Actual sample sizes are recorded in the manifest and tables if CLI counts are changed.

Each true coefficient is sampled uniformly from the declared interval and held fixed for the episode. Certification samples use independent random streams. Failure percentages use all planned trials in the corresponding cohort/method/tolerance, not only completed laps. Two-sided 95% Clopper-Pearson intervals describe the observed episode rates. Candidate epsilon is not a whole-lap bound; no pooling of the launch and lap cohorts is used for inference.

Tire-force failure means the committed control requests a combined longitudinal/lateral force outside the modeled friction circle at the true coefficient. The simulator records that violation and stops without inventing a saturation or crash trajectory. Track collisions, other state violations, terminal-set rollout failures, unavailable certificates, and audit errors are distinct outcomes. A simulation failure at t=0 is an infeasible first committed input, not a completed lap with zero time or cost.

The fixed-seed policy holdout illustration was selected during discovery. It uses 100,000 fresh uniform coefficients per setting to explain acceptance of nonzero risk. These open-policy rollouts are not counted as independent executed episodes. The matched closed-loop example is the first failed launch in manifest order, not a selected worst outcome.

The sampled DRS specialization has beta=0. Conditional friction is assumed uniform on the current SMID interval; the slow centerline backup set is assumed invariant with alpha=0. These modeling assumptions and finite-sample confidence remain explicit. This study demonstrates observed violations under probabilistic acceptance, not proof that violations must occur in every probabilistic controller.
''')
    captions=[('01_executed_failure_rates','Executed failure rates','Dots show observed episode rates; whiskers are two-sided 95% exact binomial intervals. The launch and full-lap cohorts use separate independent friction draws.'),
        ('02_failures_vs_true_friction','Where violations occur','Every point is a proposed-controller episode. The true coefficient remains within the declared support; no out-of-support friction is used.'),
        ('03_accepted_nonzero_risk','Why a safe certificate can permit failures','The fixed-seed initial commitments are checked across friction values and against 100,000 fresh samples. A 5% certificate can correctly permit a nonzero failure probability.'),
        ('04_time_to_violation','When executed constraints fail','Cumulative observed failure fractions retain every trial in the denominator. A launch can fail on its first committed tire-force request.'),
        ('05_matched_failure_trace','A matched failed and safe execution','The same true coefficient and seed under both tolerances. A single orange point indicates a violation on the first selected input, after which execution stops.'),
        ('06_failure_after_driving','A failure after driving','First positive-time launch failure in manifest order. The red cross marks an actual tire-force constraint violation. The true coefficient remains within the SMID interval; dashed lines in the certificate panel show the two decision tolerances.'),
        ('07_full_lap_failure','Failure during the full-lap experiment','First positive-time proposed failure in the independent full-lap cohort. The displayed time window ends one second after the failed run; full records are retained. Its matched 1% run and the 5% run share true friction and seed. The red cross marks the actual tire-force violation.')]
    return package_local_report(out, captions)


def package_local_report(out, captions=()):
    """Package existing results locally without rerunning any simulations."""
    out=Path(out)
    analysis=json.loads((out/'analysis.json').read_text())
    lines=['# Safety stress results', '',
        f"{analysis['total']:,} executed episodes; {analysis['violations']} tire-force constraint violations.", '',
        'Initial μ ∼ Uniform[0.76, 1.05]. Launch and full-lap cohorts are independent. '
        'Failures are infeasible committed tire-force requests, not simulated collisions. '
        'The online 1% / 5% tolerance applies per certification decision, not per lap.', '',
        '| Cohort | Tolerance | Method | Trials | Violations | Observed rate | 95% interval |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in analysis['summary']:
        lines.append(f"| {row['cohort']} | {100*row['epsilon']:g}% | {NAMES[row['method']]} | "
            f"{row['trials']} | {row['violations']} | {100*row['violation_rate']:.2f}% | "
            f"[{100*row['ci95_lower']:.2f}, {100*row['ci95_upper']:.2f}]% |")
    lines += ['', '[Protocol](README.md) · [Every episode](tables/all_episodes.csv) · '
        '[Every failure](tables/executed_failures.csv) · '
        '[Certificate rechecks](tables/failed_policy_rechecks.csv)', '',
        '## Figures', '', 'Each figure is available as PDF, SVG and PNG.', '']
    caption_map={stem:(title,caption) for stem,title,caption in captions}
    for path in sorted((out/'figures').glob('*.pdf')):
        title,caption=caption_map.get(path.stem,(path.stem.replace('_',' '),''))
        lines.append(f'- [{title}](figures/{path.name})'+(f' — {caption}' if caption else ''))
    lines += ['', '## Tables and records', '',
        'The `tables/` directory contains CSV data and LaTeX tables. The archive includes '
        'figures, tables, representative trajectories, the frozen manifest, and source snapshots. '
        'All raw episode records and individual audits remain in `../records/` and `../runs/`; '
        'they are not duplicated in this compact archive.', '',
        'These results are local files and are not included in the website demo.', '']
    (out/'SUMMARY.md').write_text('\n'.join(lines))
    (out/'index.html').unlink(missing_ok=True)
    (out/'paper-results.zip').unlink(missing_ok=True)
    archive=out/'stress-results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix not in ('.zip','.html'):
                z.write(path,path.relative_to(out))
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('results/launch-stress'))
    parser.add_argument('--package-only',action='store_true',help='Package existing figures and tables without recomputing diagnostics')
    args=parser.parse_args()
    print(package_local_report(args.input/'report') if args.package_only else build_report(args.input))
