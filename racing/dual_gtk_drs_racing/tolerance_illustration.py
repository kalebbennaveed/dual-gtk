"""Declared near-feasibility illustration, excluded from Monte Carlo inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from .validation import audit_record
from .paper_report import METHODS,NAMES,COLORS,table_files,save_figure,write_csv,fmt
from .track import Track
from .model import commands
from .safety import sampled_rollouts,binomial_upper


def build_illustration(source,destination):
    source=Path(source);destination=Path(destination);records={};provenance=[];rows=[]
    for eps,folder in ((.01,'boundary-risk-01'),(.05,'boundary-risk-05')):
        for method in METHODS:
            path=source/folder/(method+'.json')
            if not path.exists():return ''
            r=json.loads(path.read_text())
            if r.get('schema',2)>=3:raise ValueError('This diagnostic is for historical schema-2 policies; use check_mppi_certificates.py for MPPI records.')
            audit_record(r);records[eps,method]=r
            provenance.append(dict(epsilon=eps,method=method,path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),implementation=r['implementation']))
            s=r['summary'];checks=[c for e in r['events'] for c in e.get('certification_checks',[]) if c['accepted']]
            rows.append(dict(epsilon=eps,method=method,true_mu=s['true_mu'],lap_time=s['time'] if s['lap_complete'] else None,cost=s['cost'] if s['lap_complete'] else None,completed=s['lap_complete'],outcome=s['termination'],minimum_clearance=s['minimum_clearance'],final_width=s['final_mu'][1]-s['final_mu'][0],peak_speed=max(f['state'][3] for f in r['frames']),maximum_accepted_upper=max((c['failure_upper'] for c in checks),default=0)))
    tables=destination/'tables';figs=destination/'figures'
    write_csv(tables/'tolerance_illustration.csv',rows)
    (destination/'illustration_manifest.json').write_text(json.dumps(dict(selection='Exploratory scan near the initial acceleration feasibility threshold identified mu=0.77. This is a descriptive matched illustration, not independent confirmatory Monte Carlo evidence.',records=provenance),indent=2))
    tab=table_files(tables,'tolerance_illustration_summary',['Tolerance','Method','Lap [s]','Cost','Peak speed [m/s]','Final width','Largest accepted upper'],
        [[f"{100*r['epsilon']:g}%",NAMES[r['method']],fmt(r['lap_time']),fmt(r['cost']),fmt(r['peak_speed']),fmt(r['final_width'],6),fmt(100*r['maximum_accepted_upper'],3)+'%'] for r in rows])
    fig,axs=plt.subplots(2,2,figsize=(11,6.8),layout='constrained')
    for eps,color,ls in ((.01,'#167f69','-'),(.05,'#ab7230','--')):
        r=records[eps,'proposed'];fs=r['frames'];t=[f['t'] for f in fs];label=f'Proposed · {100*eps:g}%'
        axs[0,0].plot(t,[f['state'][0] for f in fs],color=color,ls=ls,label=label)
        axs[0,1].plot(t,[f['state'][3] for f in fs],color=color,ls=ls,label=label)
        axs[1,0].plot(t,[f['mu'][1]-f['mu'][0] for f in fs],color=color,ls=ls,label=label)
        events=[e for e in r['events'] if e['status']=='committed']
        axs[1,1].scatter([e['time'] for e in events],[100*e['selected']['certificate']['failure_upper'] for e in events],color=color,s=10,alpha=.65,label=label)
        axs[1,1].axhline(100*eps,color=color,ls=':',lw=1)
    for ax,label in zip(axs.flat,('Distance [m]','Speed [m/s]','SMID interval width','Committed policy upper bound [%]')):ax.set(xlabel='Time [s]',ylabel=label)
    axs[0,0].legend(frameon=False);axs[1,0].set_yscale('log');fig.suptitle('Matched tolerance illustration · true μ = 0.77',fontsize=11)
    save_figure(fig,figs,'03_tolerance_boundary_case')
    # Fixed diagnostic selection: the 12 largest-bound commitments per tolerance.
    # Validation samples are fresh and never influence the controller.
    validation=[];tr=Track('technical');n=10000
    for eps in (.01,.05):
        r=records[eps,'proposed']
        events=sorted([e for e in r['events'] if e['status']=='committed'],key=lambda e:-e['selected']['certificate']['failure_upper'])[:12]
        for i,e in enumerate(events):
            candidate=e['selected'];c=candidate['certificate'];x=np.array(r['frames'][int(round(e['time']/r['dt']))]['state'])
            seed=int(np.random.SeedSequence([92017,int(100*eps),i]).generate_state(1)[0])
            mus=np.random.default_rng(seed).uniform(*c['support'],n)
            refs=np.array([commands(j*.05,candidate['duration'],candidate['speed'],candidate['amplitude'],8.) for j in range(int(round(c['horizon']/.05)))])
            k,evaluated,minimum,steps=sampled_rollouts(x,mus,refs,tr.curvature,tr.ds,max(abs(tr.curvature)),True,n)
            validation.append(dict(epsilon=eps,time=e['time'],certificate_upper=c['failure_upper'],certificate_failures=c['failures'],certificate_samples=c['samples'],holdout_failures=int(k),holdout_samples=n,holdout_rate=k/n,holdout_upper95=binomial_upper(k,n,.05),holdout_seed=seed))
    write_csv(tables/'certificate_holdout.csv',validation)
    followups=[]
    # Preserve and investigate every diagnostic exceedance; never resample the
    # original certificate or replace its seed to make the result pass.
    for j,row in enumerate(r for r in validation if r['holdout_rate']>r['certificate_upper']):
        r=records[row['epsilon'],'proposed'];event=next(e for e in r['events'] if e['time']==row['time'] and e['status']=='committed')
        candidate=event['selected'];c=candidate['certificate'];x=np.array(r['frames'][int(round(event['time']/r['dt']))]['state'])
        refs=np.array([commands(i*.05,candidate['duration'],candidate['speed'],candidate['amplitude'],8.) for i in range(int(round(c['horizon']/.05)))])
        count=200000;seed=876354+j
        mus=np.random.default_rng(seed).uniform(*c['support'],count)
        k,_,_,_=sampled_rollouts(x,mus,refs,tr.curvature,tr.ds,max(abs(tr.curvature)),True,count)
        from scipy.stats import beta as beta_dist
        followups.append(dict(epsilon=row['epsilon'],time=row['time'],certificate_upper=c['failure_upper'],failures=int(k),samples=count,rate=k/count,lower95=float(beta_dist.ppf(.05,k,count-k+1)) if k else 0.,upper95=binomial_upper(k,count,.05),seed=seed))
    write_csv(tables/'certificate_followup.csv',followups)
    diagnostic_note=''
    if followups:
        worst=max(followups,key=lambda r:r['rate']-r['certificate_upper'])
        diagnostic_note=f"<p><strong>Observed certification error:</strong> at t = {worst['time']:.2f} s, the online upper bound was {100*worst['certificate_upper']:.3f}%. A follow-up with {worst['samples']:,} fresh rollouts estimated {100*worst['rate']:.3f}% failures (one-sided 95% lower bound {100*worst['lower95']:.3f}%). This exceeds the stated {100*worst['epsilon']:g}% tolerance. The original sample count and seed reproduce exactly and were not replaced. Finite-sample confidence permits certification errors; these observations must not be described as deterministic guarantees. The executed illustration lap completed without a constraint violation.</p><p><a href='tables/certificate_followup.csv'>Follow-up counts and seeds</a></p>"

    fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained')
    for eps,color in ((.01,'#167f69'),(.05,'#ab7230')):
        rs=[r for r in validation if r['epsilon']==eps]
        ax.scatter([100*r['certificate_upper'] for r in rs],[100*r['holdout_rate'] for r in rs],color=color,s=24,label=f'{100*eps:g}% tolerance')
    if followups:ax.scatter([100*r['certificate_upper'] for r in followups],[100*r['rate'] for r in followups],marker='*',s=100,color='#b74640',label='200,000-sample follow-up')
    ax.plot([0,5.7],[0,5.7],ls='--',color='#555',lw=1)
    ax.set(xlabel='Online corrected upper bound [%]',ylabel='Fresh 10,000-sample failure rate [%]',xlim=(0,5.7),ylim=(-.1,5.7));ax.legend(frameon=False)
    save_figure(fig,figs,'04_certificate_holdout')
    return f'''<section><h2>Near-feasibility illustration · μ = 0.77</h2><p>This additional matched case was chosen after a short exploratory scan near the acceleration-feasibility threshold. It is excluded from Monte Carlo confidence calculations. Higher tolerance changes acceptance decisions here; it does not guarantee a shorter lap.</p>{tab}<p><a href="tables/tolerance_illustration_summary.tex">LaTeX</a> · <a href="tables/tolerance_illustration.csv">CSV</a> · <a href="../boundary-1/index.html">1% replay</a> · <a href="../boundary-2/index.html">5% replay</a></p></section><article><h2>How tolerance changes the proposed controller</h2><img src="figures/03_tolerance_boundary_case.png" alt="Progress speed uncertainty and certificate bounds at both tolerances"><p>The plant, true friction, seed and candidate library are matched. Dashed horizontal lines mark the two online risk tolerances. The uncertainty plot uses actual SMID measurements.</p><p><a href="figures/03_tolerance_boundary_case.pdf">PDF</a> · <a href="figures/03_tolerance_boundary_case.svg">SVG</a></p></article><article><h2>Independent rollout check of accepted certificates</h2><img src="figures/04_certificate_holdout.png" alt="Independent validation failure rates against online binomial bounds"><p>The 12 largest-bound proposed commitments per tolerance are reevaluated using 10,000 fresh IID coefficients each, fixed over each rollout. This is a conditional-distribution diagnostic, not a whole-lap guarantee or a population-wide calibration test. Validation samples never affect planning.</p>{diagnostic_note}<p><a href="figures/04_certificate_holdout.pdf">PDF</a> · <a href="tables/certificate_holdout.csv">Counts, bounds and seeds</a></p></article>'''
