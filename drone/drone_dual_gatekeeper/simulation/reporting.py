import csv
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).parents[2]/".mpl-cache"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..methods.registry import METHODS
from .logging import save_result

def summarize(results,directory,failures=None):
    failures=[] if failures is None else failures
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scenario in sorted({r['scenario']['name'] for r in results}|{f['scenario'] for f in failures}):
        group=[r for r in results if r['scenario']['name']==scenario]
        base=np.mean([r['metrics']['mission_cost'] for r in group if r['method']=='robust' and r['metrics']['success']])
        for method in METHODS:
            runs=[r for r in group if r['method']==method];ms=[r['metrics'] for r in runs]
            failed=[f for f in failures if f['scenario']==scenario and f['method']==method]
            if not ms and not failed:continue
            row={'scenario':scenario,'method':method,'trials':len(ms)+len(failed),'failed_runs':len(failed),'success_rate':sum(m['success'] for m in ms)/(len(ms)+len(failed)),'violations':sum(m['violations'] for m in ms)+sum('AssertionError' in f['error'] for f in failed)}
            for key in ['mission_cost','completion_time','final_interval_width','uncertainty_reduction','minimum_clearance','maximum_tube_utilization','mean_planning_time','replans','informative_candidates','rejected_safety','rejected_reconnection','rejected_budget','committed','exploration_spent','control_saturation_count','worst_solver_residual','final_distance_to_goal','weighted_fallbacks','solver_nonconvergences']:
                row[key]=float(np.mean([m.get(key) if m.get(key) is not None else float('nan') for m in ms])) if ms else float('nan')
            row['cost_vs_robust']=row['mission_cost']/base if base>0 else float('nan');rows.append(row)
    if rows:
        with (out/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    headers=['Method','Success','Violations','Cost','Cost/robust','Time','Final width','Reduction','Clearance','Tube utilization','Planning s']
    lines=[]
    for scenario in sorted({r['scenario'] for r in rows}):
        lines.extend([f'## {scenario}','', '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)])
        for row in [r for r in rows if r['scenario']==scenario]:
            vals=[METHODS[row['method']].label,f"{row['success_rate']:.1%}",str(row['violations']),f"{row['mission_cost']:.3f}",f"{row['cost_vs_robust']:.3f}",f"{row['completion_time']:.2f}",f"{row['final_interval_width']:.6f}",f"{row['uncertainty_reduction']:.1%}",f"{row['minimum_clearance']:.3f}",f"{row['maximum_tube_utilization']:.3f}",f"{row['mean_planning_time']:.3f}"]
            lines.append('| '+' | '.join(vals)+' |')
        lines.append('')
    (out/'comparison.md').write_text('\n'.join(lines))
    save_result({'rows':rows},out/'comparison.json')
    return rows

def plots(results,directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    for scenario in sorted({r['scenario']['name'] for r in results}):
        group=[r for r in results if r['scenario']['name']==scenario]
        # deterministic demonstration, or first common trial for compact profiles.
        selected=[]
        for method in METHODS:
            matches=[r for r in group if r['method']==method]
            if matches:selected.append(matches[0])
        fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        titles=['Drag interval width','Actual accumulated mission cost','Distance to goal','Position tube max half-width','Actual obstacle clearance','Planning time per epoch']
        for r in selected:
            samples=r['samples'];ts=np.array([v['time'] for v in samples]);label=METHODS[r['method']].label
            values=[np.array([v['interval'][1]-v['interval'][0] for v in samples]),
                    np.array([v['mission_cost'] for v in samples]),
                    np.linalg.norm(np.array([v['state'][:3] for v in samples])-np.array(r['scenario']['goal']),axis=1),
                    np.max(np.array([v['rho'] for v in samples]),axis=1),
                    np.array([v['clearance'] if v['clearance'] is not None else np.nan for v in samples])]
            for ax,y in zip(axes.flat,values):ax.plot(ts,y,label=label)
            axes.flat[5].plot([e['time'] for e in r['epochs']],[e['planning_time'] for e in r['epochs']],marker='.',label=label)
        for ax,title in zip(axes.flat,titles):ax.set_title(title);ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8);fig.savefig(out/f'{scenario}_profiles.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        errors=[]
        for r in group:
            if r['method']=='robust':continue
            for e in r['epochs']:
                axes[0].scatter(e['predicted_reduction'],e['realized_reduction'],s=12,label=r['method'])
                errors.append(e['predicted_reduction']-e['realized_reduction'])
        axes[0].plot([0,.5],[0,.5],'k--');axes[0].set_xlabel('Predicted width reduction');axes[0].set_ylabel('Realized width reduction')
        for r in selected:
            if r['method']!='proposed':continue
            for e in r['epochs']:
                for candidate in e['candidates']:
                    axes[1].scatter(e['time'],candidate['score'],s=12+candidate['horizon']*3,c='green' if candidate['selected'] else 'gray')
            from collections import Counter
            reasons=Counter(v['reason'] or v['status'] for e in r['epochs'] for v in e['candidates'])
            axes[2].bar(reasons.keys(),reasons.values());axes[2].tick_params(axis='x',labelrotation=30)
        axes[1].set_title('Candidate scores (size = horizon)');axes[1].set_xlabel('Epoch time (s)');axes[2].set_title('Candidate decisions')
        fig.savefig(out/f'{scenario}_decisions.png',dpi=150);plt.close(fig)
        save_result({'scenario':scenario,'prediction_bias':float(np.mean(errors)) if errors else None,
                     'prediction_mean_absolute_error':float(np.mean(abs(np.array(errors)))) if errors else None,
                     'segments':len(errors)},out/f'{scenario}_prediction_diagnostics.json')
