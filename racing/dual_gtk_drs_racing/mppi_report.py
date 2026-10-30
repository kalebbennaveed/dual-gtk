"""Generate traceable MPPI comparison tables and plots from executed records."""
from pathlib import Path
import csv,html,json,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'proposed':'#168365','passive':'#377bbb','weighted':'#b98024','robust':'#995777'}

def label(key,record):
    method=record['summary']['method']
    return {'proposed':'Proposed','passive':'Passive','weighted':'Weighted','robust':'No learning'}[method]+(f" (γ={record['config']['information_weight']:g})" if method=='weighted' else '')

def csv_file(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def table(rows,columns):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(title)}</th>' for _,title in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(row.get(key,"—")))}</td>' for key,_ in columns)+'</tr>' for row in rows)+'</tbody></table></div>'

def save(fig,directory,name):
    fig.savefig(directory/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(directory/f'{name}.svg',bbox_inches='tight')
    svg=directory/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(directory/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def build(demo_source='output/mppi',validation_source='results/mppi-validation',destination='results/mppi-report'):
    source=Path(demo_source);validation=Path(validation_source);out=Path(destination)
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text());all_rows=[];cards=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for suffix in ('01','05'):
        folder=source/f'risk-{suffix}'
        order=('proposed','passive','robust','weighted_zero','weighted_low','weighted','weighted_high')
        runs={name:json.loads((folder/f'{name}.json').read_text()) for name in order if (folder/f'{name}.json').exists()}
        if not runs:raise ValueError(f'No replays in {folder}')
        if any(r['implementation']!=manifest['implementation'] for r in runs.values()):raise ValueError('Mixed source versions')
        if runs['proposed']['config']['information_weight']!=2.:raise ValueError('Proposed must use its fixed gamma=2 informative objective')
        rows=[]
        for key,r in runs.items():
            s=r['summary'];rows.append(dict(risk=suffix,variant=key,method=label(key,r),gamma=r['config']['information_weight'],
                completed=s['lap_complete'],termination=s['termination'],time=s['time'],cost=round(s['cost'],3),
                final_width=s['final_mu'][1]-s['final_mu'][0],informative_seconds=s['informative_seconds'],
                exploration_charge=round(s['budget_charged'],5),constraint_violation=s['constraint_violated'],
                minimum_clearance=s['minimum_clearance'],planner_seconds=sum(d['solve_seconds'] for e in r['events'] for d in e.get('planner_diagnostics',{}).values())))
        all_rows+=rows
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for key in ('proposed','passive','weighted','robust'):
            r=runs[key];fs=r['frames'];t=np.array([f['t'] for f in fs]);x=np.array([f['state'] for f in fs]);mu=np.array([f['mu'] for f in fs]);c=COLORS[key]
            style=dict(color=c,label=label(key,r),lw=1.6)
            axes[0,0].plot(t,x[:,0]-x[0,0],**style)
            axes[0,1].plot(t,x[:,3],**style)
            axes[0,2].semilogy(t,np.maximum(1e-10,mu[:,1]-mu[:,0]),**style)
            axes[1,0].plot(t,[f['cost'] for f in fs],**style)
            axes[1,1].plot(t,x[:,1],**style)
            axes[1,2].plot(t,[f['budget'] for f in fs],**style)
        for a,title,y in zip(axes.flat,['Progress','Speed','Observed SMID contraction','Mission cost','Lateral motion','Exploration expenditure'],['m','m/s','Interval width','Cost','m','Reserved cost']):
            a.set(title=title,ylabel=y,xlabel='Time (s)');a.grid(alpha=.18)
        axes[0,0].legend(fontsize=8);save(fig,figures,f'risk-{suffix}-comparison')
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        subset=sorted([r for r in rows if runs[r['variant']]['summary']['method']=='weighted'],key=lambda r:r['gamma'])
        proposed=next(r for r in rows if r['variant']=='proposed')
        for ax,metric,title in zip(axes,['time','cost','informative_seconds'],['Lap time (s)','Mission cost','Informative driving (s)']):
            ax.plot([r['gamma'] for r in subset],[r[metric] if r['completed'] else np.nan for r in subset],'-o',color=COLORS['weighted'],label='Weighted baseline')
            if proposed['completed']:ax.axhline(proposed[metric],ls='--',color=COLORS['proposed'],label='Proposed (fixed)')
            ax.set(xlabel='Weighted baseline γ',ylabel=title);ax.grid(alpha=.2)
        axes[0].legend();save(fig,figures,f'risk-{suffix}-weights')
        cards.append(f'<section><h2>{int(suffix)}% candidate tolerance · μ = 0.85</h2>'+table(rows,[('method','Method'),('termination','Outcome'),('time','Time (s)'),('cost','Mission cost'),('informative_seconds','Informative (s)'),('exploration_charge','Budget charge')])+f'<img src="figures/risk-{suffix}-comparison.png" alt="Executed comparisons"><img src="figures/risk-{suffix}-weights.png" alt="Weight sensitivity"><p><a href="figures/risk-{suffix}-comparison.svg">Comparison SVG</a> · <a href="figures/risk-{suffix}-weights.svg">Weight sensitivity SVG</a></p></section>')
    csv_file(out/'demonstrations.csv',all_rows)
    (out/'demonstrations.json').write_text(json.dumps(all_rows,indent=2))
    tex=[r'\begin{tabular}{llrrr}',r'\toprule',r'Tolerance & Method & Lap (s) & Mission cost & Exploration charge \\',r'\midrule']
    for row in all_rows:
        name=row['method'].split(' (')[0]
        if row['variant'].startswith('weighted'):name+=r' ($\gamma='+f"{row['gamma']:g}"+'$)'
        lap=f"{row['time']:.2f}" if row['completed'] else '--'
        tex.append(f"{int(row['risk'])}"+r'\% & '+name+f" & {lap} & {row['cost']:.3f} & {row['exploration_charge']:.4f} "+r'\\')
    (out/'demonstrations.tex').write_text('\n'.join(tex+[r'\bottomrule',r'\end{tabular}'])+'\n')
    validation_html='';paired=[];safety=[]
    validation_note='Paired validation has not finished. This preview contains no completed validation-study claim.'
    if (validation/'aggregate.json').exists():
        vm=json.loads((validation/'manifest.json').read_text())
        if vm['implementation']!=manifest['implementation']:raise ValueError('Validation uses different numerical sources')
        outcomes=json.loads((validation/'outcomes.json').read_text());safety=json.loads((validation/'aggregate.json').read_text())
        if {r['case']['id'] for r in outcomes}!={c['id'] for c in vm['cases']}:
            raise ValueError('Validation outcomes do not cover every planned case')
        trial_count=len({c['trial'] for c in vm['cases']});episode_count=len(vm['cases'])
        validation_note=f'Paired validation uses {trial_count} prespecified IID friction draws, shared across four methods and two tolerances. The weighted baseline uses gamma=0.4 in this study. All {episode_count} outcomes are in all_validation_episodes.csv; failed or incomplete episodes are not counted as completed laps. Paired time/cost differences use completed pairs with their denominator shown. A positive difference favors proposed. Small samples cannot establish the online tolerance as a whole-lap bound.'
        flat=[]
        for r in outcomes:
            row={**r['case'],**r.get('summary',{}),'audit_passed':r.get('audit',{}).get('passed',False),'error':r.get('error','')};flat.append(row)
        csv_file(out/'all_validation_episodes.csv',flat);csv_file(out/'validation_safety.csv',safety)
        for scenario,suffix in (('traction','01'),('traction_relaxed','05')):
            fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
            for method in COLORS:
                group=[r for r in outcomes if r['case']['scenario']==scenario and r['case']['method']==method and r.get('summary',{}).get('lap_complete')]
                for ax,metric,title in zip(axes,('time','cost'),('Completed lap time (s)','Mission cost')):
                    ax.scatter([r['case']['true_mu'] for r in group],[r['summary'][metric] for r in group],color=COLORS[method],label={'robust':'No learning','weighted':'Weighted (γ=0.4)'}.get(method,method.title()),s=28)
                    ax.set(xlabel='Episode friction μ',ylabel=title);ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);save(fig,figures,f'risk-{suffix}-validation')
        for scenario in ('traction','traction_relaxed'):
            for comparator in ('passive','weighted','robust'):
                deltas=[];cost_deltas=[]
                trials=sorted({r['case']['trial'] for r in outcomes})
                for trial in trials:
                    pair={r['case']['method']:r.get('summary',{}) for r in outcomes if r['case']['trial']==trial and r['case']['scenario']==scenario}
                    a=pair.get('proposed',{});b=pair.get(comparator,{})
                    if a.get('lap_complete') and b.get('lap_complete'):
                        deltas.append(b['time']-a['time']);cost_deltas.append(b['cost']-a['cost'])
                paired.append(dict(scenario=scenario,comparator=comparator,complete_pairs=len(deltas),planned_pairs=len(trials),
                    proposed_time_wins=sum(d>1e-9 for d in deltas),ties=sum(abs(d)<=1e-9 for d in deltas),losses=sum(d< -1e-9 for d in deltas),
                    mean_time_saved=float(np.mean(deltas)) if deltas else None,mean_cost_saved=float(np.mean(cost_deltas)) if deltas else None))
        csv_file(out/'paired_comparisons.csv',paired)
        safety_display=[dict(r,scenario='1%' if r['scenario']=='traction' else '5%',method={'robust':'No learning'}.get(r['method'],r['method'].title()),unsafe_or_unresolved_upper95=f"{100*r['unsafe_or_unresolved_upper95']:.2f}%") for r in safety]
        paired_display=[dict(r,scenario='1%' if r['scenario']=='traction' else '5%',comparator={'robust':'No learning'}.get(r['comparator'],r['comparator'].title()),mean_time_saved=round(r['mean_time_saved'],3) if r['mean_time_saved'] is not None else None) for r in paired]
        validation_html=f'<section><h2>Paired validation trials</h2><p>{trial_count} IID friction draws, shared across methods and both tolerances: {episode_count} full-lap attempts. Weighted uses γ = 0.4 in this study; the separate reference sweep varies its weight. Completion denominators and every failure are retained. These small-sample episode bounds do not validate a 1% or 5% whole-lap guarantee.</p>'+table(safety_display,[('scenario','Tolerance'),('method','Method'),('trials','Trials'),('completed','Laps'),('violations','Violations'),('unresolved','Unresolved'),('unsafe_or_unresolved_upper95','95% upper bound')])+table(paired_display,[('scenario','Tolerance'),('comparator','Comparator'),('complete_pairs','Complete pairs'),('proposed_time_wins','Wins'),('ties','Ties'),('losses','Losses'),('mean_time_saved','Mean seconds saved')])+'</section>'
        shutil.copyfile(validation/'manifest.json',out/'validation_manifest.json')
        validation_html+='<section><h2>Every completed validation lap</h2><p>Each point is one friction draw. The two tolerances use the same draws; incomplete outcomes remain in the table above.</p>'+''.join(f'<h3>{int(suffix)}% candidate tolerance</h3><img src="figures/risk-{suffix}-validation.png" alt="Completed validation laps by friction">' for suffix in ('01','05'))+'</section>'
    if (out/'certificate_holdouts.json').exists():
        holdouts=json.loads((out/'certificate_holdouts.json').read_text())
        validation_html+='<section><h2>Independent certificate checks</h2><p>The first and highest-speed accepted commitment for proposed, passive and weighted are replayed with their original certification samples, then checked with 10,000 fresh friction draws each. These are conditional-policy diagnostics.</p>'+table(holdouts,[('risk','Tolerance (%)'),('method','Method'),('selection','Selection'),('original_failures','Original failures'),('original_samples','Original N'),('holdout_failures','Fresh failures'),('holdout_samples','Fresh N')])+'<p><a href="certificate_holdouts.json">Counts and seeds</a></p></section>'
    # Store actual tested numerical sources; rebuilding the report must not
    # relabel old records with a newer checkout.
    from .experiments import implementation_signature,NUMERICAL_FILES
    if implementation_signature()!=manifest['implementation']:raise ValueError('Checkout differs from replay sources')
    package=Path(__file__).parent;snap=out/'sources';snap.mkdir(exist_ok=True)
    for name in NUMERICAL_FILES:
        target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/name,target)
    for name in ('validation.py','experiments.py','mppi_report.py'):
        shutil.copyfile(package/name,snap/name)
    for name in ('run_mppi_demo.py','run_mppi_validation.py','check_mppi_certificates.py','run_mppi_stress.py','report_mppi_stress.py','probe_mppi_risk.py','refine_mppi_risk.py','report_mppi_risk.py'):
        target=out/'reproduce'/'tools'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(package.parent/'tools'/name,target)
    shutil.copyfile(source/'manifest.json',out/'manifest.json')
    shutil.copyfile(package.parent/'docs'/'method.md',out/'method.md')
    for name in ('pyproject.toml','uv.lock','README.md'):
        shutil.copyfile(package.parent/name,out/'reproduce'/name)
    shutil.copytree(package,out/'reproduce'/'dual_gtk_drs_racing',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(package.parent/'tests',out/'reproduce'/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(package.parent/'docs',out/'reproduce'/'docs',dirs_exist_ok=True)
    text='''# MPPI racing results

The current implementation optimizes physical control sequences locally. The old finite policy library is not used in these records.

The reference demonstrations use true friction 0.85 and seed 7. Both candidate tolerances (1% and 5%) include one proposed method with a fixed gamma=2 informative generation objective; weighted-baseline gamma 0, 0.1, 0.4, 2; passive and no learning. These are paired demonstrations and sensitivity analyses, not independent safety trials.

VALIDATION_NOTE

The original DRS margins, terminal recovery set, binomial upper bound, sample count and confidence allocation are retained. beta=0, alpha=0; uniform conditional friction and invariant fallback are assumptions. Explicit-input policy rollouts are shared with execution and independently audited.

CSV tables, SVG/PNG figures, exact manifests and numerical source snapshots accompany this report. Full records with every state, measurement, optimized control, rejected candidate and certificate remain in output/mppi and results/mppi-validation/records. The reproduce directory contains the source package, locked dependencies, tests and experiment scripts. From that directory, run the README commands to regenerate the experiments. Proposed uses two MPPI optimizations per decision; each baseline uses one. Reported solve times are simulation measurements, not real-time guarantees.
'''
    (out/'README.md').write_text(text.replace('VALIDATION_NOTE',validation_note))
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPPI racing results</title><style>body{font:15px/1.6 system-ui;background:#f3f6f3;color:#20372d;margin:0}main{max-width:1200px;padding:28px;margin:auto}section{background:white;border:1px solid #d9e3db;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:34px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}td,th{padding:8px;text-align:right;border-bottom:1px solid #e2e8e3}td:first-child,th:first-child{text-align:left}a{color:#087153}</style><main><p>MPPI / SAMPLED DRS</p><h1>Executed comparisons and baseline weight sensitivity</h1><p>Proposed is a single method with a fixed strongly informative objective (γ = 2). Only the weighted baseline has a weight sweep.</p><p><a href="../">Interactive replays</a> · <a href="../results/">Current MPPI stress study</a> · <a href="demonstrations.csv">Demonstration CSV</a> · <a href="all_validation_episodes.csv">Every validation episode</a> · <a href="README.md">Scope and reproducibility</a> · <a href="method.md">Method</a></p>'''+''.join(cards)+validation_html+f'<footer>Numerical source SHA256: {manifest["implementation"]}</footer></main></html>')
    page=out/'index.html'
    page.write_text(page.read_text().replace('<a href="method.md">Method</a>','<a href="method.md">Method</a> · <a href="demonstrations.tex">LaTeX table</a> · <a href="mppi-results.zip">Download figures, tables and sources</a>'))
    if not safety:page.write_text(page.read_text().replace('<a href="all_validation_episodes.csv">Every validation episode</a> · ',''))
    with zipfile.ZipFile(out/'mppi-results.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix!='.zip':bundle.write(path,path.relative_to(out))
    return out/'index.html'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',default='output/mppi');p.add_argument('--validation',default='results/mppi-validation');p.add_argument('--output',default='results/mppi-report');a=p.parse_args()
    print(build(a.source,a.validation,a.output))
