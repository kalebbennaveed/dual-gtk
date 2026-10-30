"""Export actual simulation records with local JavaScript dependencies."""
from importlib.resources import files
from pathlib import Path
import json
import hashlib
import shutil
from .track import Track


def build(source:Path,destination:Path,results:Path|None=None):
    runs={}
    order=("proposed","passive","robust","weighted_zero","weighted_low","weighted","weighted_high","weighted_tuned","weighted_fast")
    for name in order:
        path=source/f"{name}.json"
        if path.exists(): runs[name]=json.loads(path.read_text())
    if not runs: raise ValueError("No simulation records found; run drs-racing simulate first")
    # A weight comparison intentionally changes only the scalarization weight.
    # Keep the common dynamics, sensor, initial support and random seed paired.
    signatures={(r.get("implementation"),json.dumps({k:v for k,v in r.get("config",{}).items()
        if k!="information_weight"},sort_keys=True),r["seed"]) for r in runs.values()}
    if len(signatures)>1: raise ValueError("Comparison records have different configurations or implementation versions; rerun all methods together")
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copytree(files(__package__).joinpath("web"),destination,dirs_exist_ok=True)
    shutil.copyfile(files(__package__).joinpath("assets/racecar.glb"),destination/"racecar.glb")
    shutil.copyfile(files(__package__).joinpath("assets/LICENSE_Kenney.txt"),destination/"LICENSE_Kenney.txt")
    experiment=next(iter(runs.values())).get("scenario",{})
    track=Track(experiment.get("track","barcelona"))
    # Full observation strips and audit batches remain in the source records.
    # Export every state at its original timestamp, with just the replay fields.
    display_runs={}
    for name,record in runs.items():
        display=dict(record)
        display['frames']=[{k:v for k,v in frame.items() if k not in ('observation','control')} for frame in record['frames']]
        display['events']=[]
        for event in record['events']:
            item={k:v for k,v in event.items() if k not in ('certification_checks','previous','pairs')}
            if 'policy' in item:item['policy']={k:v for k,v in item['policy'].items() if k not in ('controls','references')}
            if 'selected' in item:item['selected']={k:v for k,v in item['selected'].items() if k not in ('controls','references')}
            for kind in ('nominal_attempts','informative_attempts','weighted_attempts'):
                if kind in item:item[kind]=[{k:v for k,v in a.items() if k not in ('controls','references')} for a in item[kind]]
            display['events'].append(item)
        display_runs[name]=display
    # Display precision only; full scientific records retain every float.
    def compact(value):
        if isinstance(value,float):return float(f"{value:.8g}")
        if isinstance(value,list):return [compact(v) for v in value]
        if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
        return value
    data=dict(runs=compact(display_runs),scenario=experiment,track=dict(xy=track.xy[::2].tolist(),heading=track.heading[::2].tolist(),width=track.width,length=track.length))
    (destination/"data.js").write_text("window.RACING_DATA="+json.dumps(data,allow_nan=False,separators=(",",":"))+";\n")
    index=destination/'index.html'; page=index.read_text()
    for asset in ('data.js','app.js','style.css'):
        version=hashlib.sha256((destination/asset).read_bytes()).hexdigest()[:12]
        page=page.replace(f'"{asset}"',f'"{asset}?v={version}"')
    index.write_text(page)
    if results is not None:
        if not (results/"index.html").exists():
            raise ValueError("Generate the paper report before exporting it")
        report=destination/"paper"
        shutil.copytree(results,report,dirs_exist_ok=True,ignore=shutil.ignore_patterns("runs","records","stress"))
        readme=report/"README.md"
        if readme.exists():
            readme.write_text(readme.read_text().replace("../../docs/method.md","method.md"))
            method=Path(__file__).parent.parent/"docs"/"method.md"
            if method.exists():shutil.copyfile(method,report/"method.md")
        index=destination/"index.html"
        index.write_text(index.read_text().replace('<nav id="tabs"', '<p><a href="paper/" style="color:#137258;font-weight:600">Paper results · figures and tables ↗</a></p><nav id="tabs"'))
    return destination.resolve()/"index.html"


def build_suite(first,second,destination,results=None):
    """Two risk scenarios, each with the reference and optional boundary case."""
    destination=Path(destination).resolve()
    entries=[('reference',.01,Path(first),destination),('reference',.05,Path(second),destination/'scenario-2')]
    base=Path(first).parent
    for epsilon,name,target in ((.01,'boundary-risk-01','boundary-1'),(.05,'boundary-risk-05','boundary-2')):
        if (base/name/'proposed.json').exists():entries.append(('boundary',epsilon,base/name,destination/target))
    import os
    for case,epsilon,source,target in entries:
        build(source,target)
        nav='<div class="scenario-nav" aria-label="Risk tolerance">'
        for c,e,s,out in entries:
            if c!=case:continue
            href=Path(os.path.relpath(out,target)).as_posix()+'/'
            nav+=f'<a class="scenario-card {"active" if e==epsilon else ""}" href="{href}"><span>SCENARIO {1 if e==.01 else 2}</span><strong>{100*e:g}% failure tolerance</strong><small>Same circuit · 1,000 rollouts per candidate</small></a>'
        nav+='</div><div class="case-nav">Friction case: '
        for c,e,s,out in entries:
            if e!=epsilon:continue
            href=Path(os.path.relpath(out,target)).as_posix()+'/'
            label='Reference · μ = 0.85' if c=='reference' else 'Tolerance stress · μ = 0.77'
            nav+=f'<a class="{"active" if c==case else ""}" href="{href}">{label}</a> '
        nav+='</div>'
        index=target/'index.html'
        title=f'Dual Gatekeeper Racing · Scenario {1 if epsilon==.01 else 2}'+(' · Friction stress' if case=='boundary' else '')
        page=index.read_text().replace('<title>Dual Gatekeeper Racing</title>',f'<title>{title}</title>')
        index.write_text(page.replace('<section id="experiment"',nav+'<section id="experiment"'))
    if results is not None and (Path(results)/'index.html').exists():
        report=destination/'paper'
        if report.exists():
            archive=destination.parent/'archive';archive.mkdir(exist_ok=True)
            previous=archive/'previous-paper-report'
            if previous.exists():shutil.rmtree(previous)
            report.rename(previous)
        shutil.copytree(results,report,ignore=shutil.ignore_patterns('runs','records','stress'))
    from .results_web import build_results_portal
    if results is not None:
        build_results_portal(Path(results).resolve().parent,destination)
    return destination/'index.html'
