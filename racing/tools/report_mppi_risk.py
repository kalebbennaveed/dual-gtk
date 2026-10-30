"""Audit exploratory risk probes and explain the accepted low-friction failure."""
from pathlib import Path
import csv,gzip,hashlib,json,shutil
import numpy as np
from scipy.stats import beta
from dual_gtk_drs_racing.experiments import implementation_signature
from dual_gtk_drs_racing.control_safety import control_rollouts
from dual_gtk_drs_racing.track import Track


def main():
    root=Path('results/mppi-risk-report');root.mkdir(exist_ok=True)
    signature=implementation_signature();all_rows=[];groups=[]
    for source in ('mppi-risk-probe','mppi-risk-refinement'):
        folder=Path('results')/source;manifest=json.loads((folder/'manifest.json').read_text())
        rows=json.loads((folder/'all_outcomes.json').read_text());expected={c['id']:c for c in manifest['cases']}
        assert manifest['implementation']==signature and len(rows)==len(expected)
        assert {r['case']['id'] for r in rows}==set(expected)
        for r in rows:
            assert r['implementation']==signature and r['case']==expected[r['case']['id']]
            if 'summary' in r:
                path=folder/'records'/f"{r['case']['id']}.json.gz"
                assert hashlib.sha256(path.read_bytes()).hexdigest()==r['record_sha256']
                record=json.load(gzip.open(path,'rt'))
                assert record['scenario']['epsilon']==int(r['case']['scenario'].split('_')[1])/100
                assert record['summary']==r['summary'] and r.get('audit',{}).get('passed',False)
            r['source']=source
        all_rows.extend(rows)
        shutil.copyfile(folder/'manifest.json',root/f'{source}-manifest.json')
    for prefix,label in (('paired','Original distribution, 10%'),('boundary','Targeted boundary grid'),('near-boundary','Post-discovery nearby friction')):
        for eps in (1,5,10):
            rows=[r for r in all_rows if r['case']['id'].startswith(prefix+'-') and r['case']['scenario']==f'probe_{eps}']
            if not rows:continue
            groups.append(dict(group=label,epsilon=eps/100,runs=len(rows),violations=sum(r.get('summary',{}).get('constraint_violated',False) for r in rows),
                unresolved=sum('error' in r or 'audit_error' in r or r.get('summary',{}).get('certificate_unavailable',False) for r in rows)))
    fields=['source','id','epsilon','true_mu','start','seed','termination','time','violation_kind','minimum_tire_margin','audit_passed','error','audit_error']
    with (root/'all_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
        for r in all_rows:
            s=r.get('summary',{});row={k:r['case'][k] for k in fields if k in r['case']}
            row.update({k:s[k] for k in fields if k in s})
            row.update(source=r['source'],epsilon=int(r['case']['scenario'].split('_')[1])/100,violation_kind=(s.get('violation') or {}).get('kind',''),
                minimum_tire_margin=r.get('minimum_tire_margin'),audit_passed=r.get('audit',{}).get('passed',False),error=r.get('error',''),audit_error=r.get('audit_error',''))
            w.writerow(row)
    track=Track('technical');examples=[];records=[];holdout_rows=[]
    for eps in (1,5,10):
        record=json.load(gzip.open(Path('results/mppi-risk-probe/records')/f'boundary-0-00-{eps:02d}.json.gz','rt'));records.append(record)
        event=next(e for e in record['events'] if abs(e['time']-.3)<1e-9)
        a=event['selected'];cert=a['certificate'];x=np.array(next(f['state'] for f in record['frames'] if abs(f['t']-.3)<1e-9))
        controls=np.asarray(a['controls']);refs=np.asarray(a['references']);tail=round(cert['horizon']/.05)-len(controls)
        args=(controls,refs,tail,track.curvature,track.ds,max(abs(track.curvature)),True,10000)
        mus=np.random.default_rng(cert['sample_seed']).uniform(*cert['support'],1000)
        original=control_rollouts(x,mus,*args)
        assert original[0]==cert['failures'] and original[1]==1000
        holdout=np.random.default_rng(2026101302).uniform(*cert['support'],10000)
        observed=control_rollouts(x,holdout,*args)
        failures=np.array([control_rollouts(x,np.array([mu]),*args)[0] for mu in holdout],dtype=int)
        assert failures.sum()==observed[0]
        holdout_rows.extend(dict(epsilon=eps/100,draw=i,true_mu=float(mu),candidate_failed=int(failed)) for i,(mu,failed) in enumerate(zip(holdout,failures)))
        count=int(failures.sum())
        examples.append(dict(epsilon=eps/100,decision_time=.3,true_mu=.45,seed=20100,support=cert['support'],kind=a['kind'],duration=a['duration'],
            certificate_horizon=cert['horizon'],sample_failures=cert['failures'],samples=1000,failure_upper=cert['failure_upper'],
            holdout_failures=observed[0],holdout_samples=10000,holdout_seed=2026101302,termination=record['summary']['termination'],
            holdout_ci95=[float(beta.ppf(.025,count,10001-count)) if count else 0.,float(beta.isf(.025,count+1,10000-count)) if count<10000 else 1.],
            violation=record['summary']['violation']))
    with (root/'all_holdout_draws.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['epsilon','draw','true_mu','candidate_failed'],lineterminator='\n');w.writeheader();w.writerows(holdout_rows)
    matched=[]
    events=[next(e for e in r['events'] if abs(e['time']-.3)<1e-9) for r in records]
    for duration in (.3,1.2):
        attempts=[next(a for a in e['informative_attempts'] if abs(a['duration']-duration)<1e-9) for e in events]
        assert attempts[0]['controls']==attempts[1]['controls']==attempts[2]['controls']
        assert attempts[0]['references']==attempts[1]['references']==attempts[2]['references']
        assert len({a['certificate']['sample_seed'] for a in attempts})==1
        matched.append(dict(duration=duration,identical_controls_and_references=True,
            full_batch_failures=attempts[-1]['certificate']['failures'],full_batch_upper=attempts[-1]['certificate']['failure_upper'],
            acceptance=[dict(epsilon=eps/100,accepted=a['certificate']['accepted'],evaluated=a['certificate']['evaluated'],failures=a['certificate']['failures']) for eps,a in zip((1,5,10),attempts)]))
    summary=dict(implementation=signature,controller_logic_changed=False,groups=groups,examples=examples,matched_informative_candidates=matched,
        verified_records=sum('summary' in r for r in all_rows),audits_passed=sum(r.get('audit',{}).get('passed',False) for r in all_rows),
        scope='Exploratory boundary/refinement cases are not IID failure-rate estimates. Independent holdouts evaluate the fixed accepted prefix plus fallback, including terminal recovery; closed-loop executions can replan. A tire-force violation is not a simulated collision.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'mppi-risk-probe'})
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);colors=['#16846b','#467eb7','#b9503b']
    for color,eps,record,e in zip(colors,(1,5,10),records,examples):
        frames=[f for f in record['frames'] if f['t']<=1.25 and 'traction_margin' in f]
        axes[0].plot([f['t'] for f in frames],[f['traction_margin'] for f in frames],label=f'{eps}% tolerance',color=color,linestyle='--' if eps==5 else '-',linewidth=2)
        if record['summary']['constraint_violated']:axes[0].scatter([.5],[record['summary']['violation']['margin']],color=color,s=50,marker='x',zorder=5)
    axes[0].axhline(0,color='#444',linewidth=1);axes[0].set(xlabel='Executed time (s)',ylabel='Minimum normalized tire-force margin',title='Same seed and true friction μ = 0.45');axes[0].legend(frameon=False)
    x=np.arange(3);axes[1].bar(x,[100*e['failure_upper'] for e in examples],color=colors,width=.55,label='Certificate upper bound')
    axes[1].scatter(x,[1,5,10],marker='_',s=900,c='#222',label='Acceptance tolerance')
    axes[1].set(xticks=x,xticklabels=['1%','5%','10%'],xlabel='Candidate tolerance',ylabel='Candidate failure probability (%)',title='Accepted decision at 0.30 seconds');axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    save(fig,figures/'01_accepted_failure')
    fig,ax=plt.subplots(figsize=(9,3),constrained_layout=True)
    for j,eps in enumerate((1,5,10)):
        rows=[r for r in all_rows if (r['case']['id'].startswith('near-boundary-') or r['case']['id']==f'boundary-0-00-{eps:02d}') and r['case']['scenario']==f'probe_{eps}']
        for r in rows:
            failure=r.get('summary',{}).get('constraint_violated',False)
            ax.scatter(r['case']['true_mu'],j,color='#b9503b' if failure else '#16846b',marker='x' if failure else 'o',s=65)
    ax.scatter([],[],c='#16846b',label='Eight seconds completed',s=50);ax.scatter([],[],c='#b9503b',marker='x',label='Tire-force violation',s=50)
    ax.set(yticks=[0,1,2],yticklabels=['1% tolerance','5% tolerance','10% tolerance'],xlabel='True friction μ (targeted values, same seed)',title='Nearby friction values: failure is not confined to the endpoint',ylim=(-.6,2.8));ax.legend(frameon=False,loc='upper right');ax.grid(alpha=.15)
    save(fig,figures/'02_nearby_friction')
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for i,(e,color) in enumerate(zip(examples,colors)):
        rate=100*e['holdout_failures']/e['holdout_samples'];lo,hi=np.array(e['holdout_ci95'])*100
        ax.bar(i,rate,.5,color=color)
        ax.errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],fmt='none',ecolor='#222',capsize=6)
        ax.annotate(f'{rate:.2f}%',(i,hi),xytext=(0,7),textcoords='offset points',ha='center')
    ax.scatter(np.arange(3),[1,5,10],marker='_',s=1200,c='#222',label='Candidate tolerance')
    ax.set(xticks=[0,1,2],xticklabels=['1%: nominal 1.2 s','5%: informative 0.3 s','10%: informative 1.2 s'],ylabel='Held-out candidate failure rate (%)',ylim=(-.1,11),title='10,000 independent friction rollouts per accepted plan')
    ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15)
    fig.text(.5,-.025,'Fixed prefix + fallback + terminal recovery; error bars: pointwise 95% exact intervals.',ha='center',fontsize=9)
    save(fig,figures/'03_independent_candidate_risk')
    print(json.dumps(summary,indent=2))


def save(fig,path):
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=170,bbox_inches='tight')
    svg=path.with_suffix('.svg');svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__=='__main__':main()
