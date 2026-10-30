"""Verify and plot the completed current-MPPI launch study without changing its data."""
from pathlib import Path
import argparse,csv,hashlib,json,platform
import numpy as np
from dual_gtk_drs_racing.experiments import implementation_signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('results/mppi-stress'));a=p.parse_args();root=a.input
    protocol=json.loads((root/'manifest.json').read_text());current=implementation_signature()
    assert protocol['implementation']==current,'Numerical implementation changed'
    progress=json.loads((root/'progress.json').read_text());assert progress['complete'],'Study still running'
    rows=json.loads((root/'all_outcomes.json').read_text());expected={c['id']:c for c in protocol['cases']}
    assert len(rows)==len(expected) and {r['case']['id'] for r in rows}==set(expected)
    for r in rows:
        assert r['case']==expected[r['case']['id']] and r['implementation']==current
        if 'summary' in r:
            record=root/'records'/f"{r['case']['id']}.json.gz"
            assert hashlib.sha256(record.read_bytes()).hexdigest()==r['record_sha256']
            s=r['summary'];assert s['true_mu']==r['case']['true_mu'] and s['initial_mu']==protocol['initial_mu']
            assert s['method']=='proposed' and 0<=s['time']<=protocol['seconds']
            if s['termination']=='timeout':assert s['time']==protocol['seconds']
            assert r.get('audit',{}).get('passed',False) or 'audit_error' in r
    with (root/'all_episodes.csv').open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;episodes=list(reader)
    assert len(episodes)==len(expected)
    failures=[r for r in episodes if r['constraint_violated']=='True' or r['certificate_unavailable']=='True' or r['error'] or r['audit_error']]
    with (root/'failures.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(failures)
    verification=dict(implementation=current,planned=len(expected),executed=len(rows),unique_trials=protocol['trials_per_tolerance'],
        record_hashes_verified=sum('summary' in r for r in rows),replay_audits_passed=sum(r.get('audit',{}).get('passed',False) for r in rows),
        simulation_errors=sum('error' in r for r in rows),audit_errors=sum('audit_error' in r for r in rows),
        safety_sha256=hashlib.sha256(Path('dual_gtk_drs_racing/safety.py').read_bytes()).hexdigest(),python=platform.python_version(),numpy=np.__version__)
    assert verification['safety_sha256']=='b4806091be209d5a60344ce1014de0e5d5845ab27778d0a1ba1ae061efd98bdd'
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'current-mppi-stress'})
    figures=root/'figures';figures.mkdir(exist_ok=True)
    stats=json.loads((root/'aggregate.json').read_text());colors=['#16846b','#467eb7']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,r in enumerate(stats):
        rate=100*r['violation_rate'];lo=100*r['ci95_lower'];hi=100*r['ci95_upper']
        axes[0].errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],fmt='o',capsize=7,color=colors[i],markersize=7)
        axes[0].annotate(f"{r['violations']} / {r['executed']:,}",(i,hi),xytext=(0,9),textcoords='offset points',ha='center')
        axes[1].bar(i-.2,r['safe_horizon_completed'],.35,color=colors[i],label='Safe horizon' if i==0 else None)
        axes[1].bar(i+.2,r['violations']+r['unresolved'],.35,color='#bf6550',label='Violation / unresolved' if i==0 else None)
    axes[0].set(xlim=(-.5,1.5),ylim=(-.025,max(.5,max(100*r['ci95_upper'] for r in stats)*1.3)),ylabel='Executed violation rate (%)',title='Launch outcomes with 95% exact intervals')
    axes[1].set(ylabel='Number of eight-second launches',title='All planned trials retained');axes[1].legend(frameon=False)
    for ax in axes:ax.set_xticks([0,1],['1% candidate tolerance','5% candidate tolerance']);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Current MPPI · proposed γ = 2 · 1,000 paired launches per tolerance')
    save(fig,figures/'01_launch_safety')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,scenario in enumerate(('traction','traction_relaxed')):
        group=[r for r in rows if r['case']['scenario']==scenario and 'summary' in r]
        mus=[r['case']['true_mu'] for r in group];widths=[max(1e-10,r['summary']['final_mu'][1]-r['summary']['final_mu'][0]) for r in group]
        axes[0].scatter(mus,widths,s=9,alpha=.4,color=colors[i],label=f'{1 if i==0 else 5}% tolerance')
        axes[1].scatter(mus,[r['summary']['budget_charged'] for r in group],s=9,alpha=.4,color=colors[i])
    axes[0].set(yscale='log',xlabel='True friction μ',ylabel='Final measured SMID interval width',title='Identification after eight seconds');axes[0].legend(frameon=False)
    axes[1].set(xlabel='True friction μ',ylabel='Exploration budget charged',title='Reserved exploration budget')
    for ax in axes:ax.grid(alpha=.15)
    save(fig,figures/'02_learning_and_budget')
    print(json.dumps(verification,indent=2))


def save(fig,path):
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=170)
    svg=path.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()).rstrip()+'\n')
    import matplotlib.pyplot as plt
    plt.close(fig)

if __name__=='__main__':main()
