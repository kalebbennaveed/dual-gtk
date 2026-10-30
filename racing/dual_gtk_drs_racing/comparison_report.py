"""Matched comparison of the two risk tolerances; no pooled safety claim."""
from pathlib import Path
import json
import hashlib
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from .experiments import load_rows,atomic_json
from .paper_report import (METHODS,NAMES,COLORS,style,save_figure,table_files,
                           fmt,aggregate,write_csv,bootstrap_mean_ci)


def build_comparison(first,second,destination,illustrations="output/sampled"):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    figs=destination/'figures';figs.mkdir(exist_ok=True)
    tables=destination/'tables';tables.mkdir(exist_ok=True)
    sets=[]
    for folder,source in (('risk-01',first),('risk-05',second)):
        manifest,rows=load_rows(source)
        if not (Path(source)/'index.html').exists():raise ValueError('Build each scenario report first')
        shutil.copytree(source,destination/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('records','runs'))
        sets.append((manifest,rows,aggregate(rows)))
    a,b=sets
    assert a[0]['implementation']==b[0]['implementation'],'Different numerical implementations'
    assert [(r['case']['id'],r['case']['true_mu'],r['case']['seed']) for r in a[1]]==[(r['case']['id'],r['case']['true_mu'],r['case']['seed']) for r in b[1]],'Unpaired scenario designs'
    style();summary=[];certificate=[]
    for manifest,rows,(stats,safety,paired) in sets:
        eps=manifest['scenario']['epsilon'];lookup={(r['method'],r['metric']):r for r in stats}
        for s in safety:
            m=s['method']
            summary.append(dict(scenario=manifest['scenario']['name'],epsilon=eps,method=m,trials=s['trials'],completed=s['laps_completed'],violations=s['constraint_violations'],unavailable=s['certificate_unavailable'],audit_errors=s['errors'],smid_exclusions=s['containment_failure_episodes'],unsafe_or_unresolved_upper95=s['unsafe_or_unresolved_upper95'],mean_lap_time=lookup[m,'time']['mean'],mean_cost=lookup[m,'cost']['mean'],minimum_clearance=s['minimum_clearance'],minimum_tire_margin=s['minimum_tire_margin']))
            certificate.append([f'{100*eps:g}%',NAMES[m],fmt(100*(lookup[m,'certificate_upper_max']['maximum'] or 0),3)+'%',fmt(lookup[m,'certification_rollouts']['mean'],0),fmt(lookup[m,'accepted_sample_failures']['mean'],2)])
    write_csv(tables/'scenario_statistics.csv',summary)
    performance=table_files(tables,'scenario_performance',['Tolerance','Method','Completed / trials','Mean lap [s]','Mean cost','Minimum clearance [m]'],
        [[f"{100*r['epsilon']:g}%",NAMES[r['method']],f"{r['completed']} / {r['trials']}",fmt(r['mean_lap_time']),fmt(r['mean_cost']),fmt(r['minimum_clearance'],6)] for r in summary])
    safety_html=table_files(tables,'scenario_safety',['Tolerance','Method','Constraint violations','Certificate unavailable','Audit errors','SMID exclusions','95% unsafe/unresolved upper'],
        [[f"{100*r['epsilon']:g}%",NAMES[r['method']],r['violations'],r['unavailable'],r['audit_errors'],r['smid_exclusions'],fmt(100*r['unsafe_or_unresolved_upper95'])+'%'] for r in summary])
    certificate_html=table_files(tables,'certification_summary',['Tolerance','Method','Largest accepted upper','Mean rollout count / episode','Mean failed samples in accepted checks'],certificate)
    first_index={r['case']['id']:r for r in a[1] if r['case']['group']=='monte_carlo'}
    second_index={r['case']['id']:r for r in b[1] if r['case']['group']=='monte_carlo'}
    pair_rows=[];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for method in METHODS:
        pairs=[(r,second_index[id]) for id,r in first_index.items() if r['case']['method']==method and r['status']==second_index[id]['status']=='ok' and r['metrics']['lap_complete'] and second_index[id]['metrics']['lap_complete']]
        for ax,metric in zip(axes,('time','cost')):
            differences=[y['metrics'][metric]-x['metrics'][metric] for x,y in pairs]
            lo,hi=bootstrap_mean_ci(differences)
            pair_rows.append(dict(method=method,metric=metric,pairs=len(pairs),mean_5pct_minus_1pct=float(np.mean(differences)) if differences else None,ci95_low=lo,ci95_high=hi,identical=sum(abs(v)<1e-8 for v in differences),lower_at_5pct=sum(v< -1e-8 for v in differences),higher_at_5pct=sum(v>1e-8 for v in differences)))
            ax.scatter([x['case']['true_mu'] for x,y in pairs],differences,s=16,color=COLORS[method],alpha=.65,label=NAMES[method])
            ax.axhline(0,color='#444',ls='--',lw=1)
            ax.set(xlabel='True tire coefficient μ',ylabel='5% − 1% lap time [s]' if metric=='time' else '5% − 1% mission cost')
    axes[0].legend(frameon=False,fontsize=8)
    save_figure(fig,figs,'01_tolerance_effect')
    write_csv(tables/'tolerance_paired_statistics.csv',pair_rows)
    paired_html=table_files(tables,'tolerance_effect',['Method','Metric','Completed pairs','Mean 5% − 1%','95% bootstrap interval','Lower / same / higher'],
        [[NAMES[r['method']],r['metric'],r['pairs'],fmt(r['mean_5pct_minus_1pct'],3),f"[{fmt(r['ci95_low'],3)}, {fmt(r['ci95_high'],3)}]",f"{r['lower_at_5pct']} / {r['identical']} / {r['higher_at_5pct']}"] for r in pair_rows])
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for j,(manifest,rows,(stats,safety,paired)) in enumerate(sets):
        for i,method in enumerate(METHODS):
            rs=[r for r in rows if r['case']['group']=='monte_carlo' and r['case']['method']==method and r['status']=='ok']
            for ax,metric in zip(axes,('time','cost','final_width')):
                values=[r['metrics'][metric] for r in rs if metric=='final_width' or r['metrics']['lap_complete']]
                if values:
                    ax.boxplot([values],positions=[i+(j-.5)*.32],widths=.26,patch_artist=True,showfliers=False,boxprops=dict(facecolor=COLORS[method],alpha=.35 if j==0 else .8),medianprops=dict(color='#222'))
    for ax,label in zip(axes,('Lap time [s]','Mission cost','Final SMID width')):ax.set(xticks=range(4),xticklabels=['Proposed','Passive','Weighted','No learning'],ylabel=label)
    axes[2].set_yscale('log');fig.suptitle('Light / left: 1% tolerance · Dark / right: 5% tolerance',fontsize=10)
    save_figure(fig,figs,'02_tolerance_distributions')
    total=sum(len(s[1]) for s in sets);trials=a[0]['trials']
    manifest=dict(implementation=a[0]['implementation'],analysis_sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('comparison_report.py','tolerance_illustration.py','paper_report.py')},episodes=total,trials_per_method_per_scenario=trials,scenarios=[s[0]['scenario'] for s in sets],source_manifests=['risk-01/manifest.json','risk-05/manifest.json'],scope='Online epsilon/delta are per decision. Offline episode confidence intervals are separate. No distribution-mismatch protection at beta=0.')
    atomic_json(destination/'manifest.json',manifest)
    (destination/'README.md').write_text(f'''# Sampled DRS racing results\n\nTwo matched scenarios: epsilon=0.01 and epsilon=0.05, N=1000 samples per candidate, delta=0.05 per decision, beta=0, alpha=0 (assumed invariant backup). {total} executed episodes; {trials} independent paired Monte Carlo trials per method per scenario. The physical track, plant, sensor and candidate objectives are identical.\n\nThe 5% case permits a higher candidate failure probability. It need not change a trajectory when both thresholds accept the same candidates. Identical outcomes and all failures are retained. No result is forced to show an improvement. The primary demonstration remains informative exploration versus passive and no-learning baselines.\n\nThe conditional friction law is assumed uniform on the SMID interval. This is the nominal probabilistic special case of DRS. Its per-decision safety/confidence parameters are not whole-lap probability claims. Completed-lap statistics and empirical failure bounds are separate.\n\nEach scenario directory contains its full plots, CSV and LaTeX tables, manifest, and protocol. Raw records remain in the original results directories. The ZIP contains shareable reports, not the large raw trajectories. Old interval-verifier results are not mixed into this study.\n''')
    from .tolerance_illustration import build_illustration
    illustration_html=build_illustration(illustrations,destination)
    css='body{margin:0;background:#f4f6f5;color:#23372f;font:16px/1.6 system-ui}main{max-width:1200px;margin:auto;padding:30px 22px}h1{font-size:36px;line-height:1.2}h2{font-size:22px}section,article{background:white;border:1px solid #d9e3dc;border-radius:12px;padding:24px;margin:22px 0}a{color:#137258}img{width:100%}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}.cards a{display:block;text-decoration:none;background:white;border:1px solid #d9e3dc;padding:22px;border-radius:12px}.cards strong{display:block;font-size:24px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}td,th{padding:10px;text-align:right;border-bottom:1px solid #e2e9e4}td:first-child,th:first-child{text-align:left}@media(max-width:650px){.cards{grid-template-columns:1fr}}'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dual Gatekeeper · Sampled DRS results</title><style>{css}</style><main><p>AUTONOMOUS RACING / SAMPLED DRS</p><h1>Dual Gatekeeper: paper results</h1><p>{total} closed-loop episodes · {trials} paired Monte Carlo trials per method and tolerance · 1,000 rollouts per candidate</p><p><a href="../index.html">Interactive racing demo</a> · <a href="paper-results.zip">Download all figures and tables</a> · <a href="README.md">Protocol</a></p><div class="cards"><a href="risk-01/index.html"><strong>Scenario 1 · 1% tolerance</strong>Technical circuit, slow backup, informative grip discovery.<br>Full figures, tables and individual outcomes →</a><a href="risk-05/index.html"><strong>Scenario 2 · 5% tolerance</strong>The same experiment with a higher allowed candidate failure probability.<br>Full figures, tables and individual outcomes →</a></div><p>β = 0: nominal probabilistic certification. Conditional friction is assumed uniform on the current SMID interval. The slow centerline backup set is assumed invariant (α = 0). Online ε and δ apply per decision.</p><section><h2>Task performance</h2><p>Means over completed laps. Completion counts remain visible; failed partial laps are not presented as fast finishes.</p>{performance}<p><a href="tables/scenario_performance.tex">LaTeX</a> · <a href="tables/scenario_statistics.csv">Numerical data</a></p></section><section><h2>Observed safety outcomes</h2>{safety_html}<p>One-sided 95% episode bounds count unsafe or unresolved outcomes under the declared test distribution. They are separate from the online candidate tolerance.</p><p><a href="tables/scenario_safety.tex">LaTeX</a></p></section><section><h2>Effect of the higher tolerance</h2>{paired_html}<p>Negative differences favor the 5% setting. Both tolerances can produce identical trajectories when no decision lies between their thresholds. Intervals use paired-episode bootstrap resampling.</p><p><a href="tables/tolerance_effect.tex">LaTeX</a></p></section><article><h2>Paired tolerance differences</h2><img src="figures/01_tolerance_effect.png" alt="Paired lap time and cost differences versus true friction"><p><a href="figures/01_tolerance_effect.pdf">PDF</a> · <a href="figures/01_tolerance_effect.svg">SVG</a></p></article><article><h2>Outcome distributions</h2><img src="figures/02_tolerance_distributions.png" alt="Lap time cost and uncertainty distributions for both risk tolerances"><p><a href="figures/02_tolerance_distributions.pdf">PDF</a> · <a href="figures/02_tolerance_distributions.svg">SVG</a></p></article><section><h2>Online certificate diagnostics</h2>{certificate_html}<p>Failed sampled rollouts are prediction outcomes, not executed-car failures. Every acceptance uses the complete fixed-size batch and the corrected binomial bound.</p><p><a href="tables/certification_summary.tex">LaTeX</a></p></section>{illustration_html}</main></html>'''
    (destination/'index.html').write_text(page)
    with zipfile.ZipFile(destination/'paper-results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.rglob('*')):
            if path.is_file() and path.suffix!='.zip':z.write(path,path.relative_to(destination))
    return destination/'index.html'
