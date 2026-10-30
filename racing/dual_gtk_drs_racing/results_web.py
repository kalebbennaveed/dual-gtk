"""Publish results produced by the current MPPI implementation only."""
from pathlib import Path
import csv, html, json, shutil, zipfile
from .experiments import implementation_signature

STYLE='''body{font:16px/1.6 system-ui;margin:0;background:#f4f6f5;color:#20372d}main{max-width:1160px;margin:auto;padding:28px}section{background:white;border:1px solid #d9e3db;border-radius:12px;padding:24px;margin:22px 0}h1{font-size:34px;line-height:1.2}h2{font-size:24px}a{color:#087153}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #e2e8e3;text-align:left}img,video{max-width:100%;border-radius:8px}.notice{background:#fff1d9;border-left:4px solid #bb8631;padding:14px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}.cards section{margin:0}.count{font-size:28px;font-weight:650}small{color:#52685c}code{font-size:13px;overflow-wrap:anywhere}'''

def page(title, body):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style><main>{body}</main></html>'

def build_results_portal(results:Path,destination:Path):
    out=destination/'results'
    current=implementation_signature()
    manifest=json.loads((results/'mppi-report/manifest.json').read_text())
    if manifest['implementation']!=current:raise ValueError('Reference results do not match the current MPPI implementation')
    stress_manifest=results/'mppi-stress/manifest.json'
    if stress_manifest.exists() and json.loads(stress_manifest.read_text())['implementation']!=current:
        raise ValueError('Stress records do not match the current MPPI implementation')
    risk_summary=results/'mppi-risk-report/summary.json'
    if risk_summary.exists() and json.loads(risk_summary.read_text())['implementation']!=current:
        raise ValueError('Risk probes do not match the current MPPI implementation')
    if out.exists():shutil.rmtree(out)
    out.mkdir()
    safety=list(csv.DictReader((results/'mppi-report/validation_safety.csv').open()))
    trials=sum(int(r['trials']) for r in safety);completed=sum(int(r['completed']) for r in safety)
    violations=sum(int(r['violations']) for r in safety)
    body=f'<p><a href="../">Interactive comparison</a> · <a href="../scenario-2/">5% tolerance</a></p><h1>MPPI racing results</h1><p>All experiments on this page use the current implementation. Proposed uses a fixed informative objective (γ = 2).</p>'
    body+='<section><h2>What the reported outcomes mean</h2><p><strong>Executed violation:</strong> the simulated car violates a state or tire-force constraint during closed-loop driving. The simulator stops at that violation; it does not simulate the resulting collision or loss of control.</p><p><strong>Candidate rollout failure:</strong> a fixed candidate prefix followed by fallback fails a state constraint, tire-force constraint, or terminal recovery condition for one sampled friction value. This is the event counted in certification and independent candidate tests.</p><p><strong>Candidate tolerance ε:</strong> the maximum accepted upper confidence bound on candidate failure probability. The 1%, 5% and 10% settings are thresholds for individual candidate checks, not target violation rates for driving episodes.</p></section>'
    stress=results/'mppi-stress'
    if (stress/'manifest.json').exists():
        protocol=json.loads((stress/'manifest.json').read_text())
        if protocol['implementation']!=current:raise ValueError('Stress records do not match the current MPPI implementation')
        progress=json.loads((stress/'progress.json').read_text()) if (stress/'progress.json').exists() else dict(completed=0,planned=len(protocol['cases']),complete=False)
        stats=json.loads((stress/'aggregate.json').read_text()) if (stress/'aggregate.json').exists() else []
        target=out/'stress';target.mkdir()
        body+=f'<section><h2>Current MPPI: {protocol["trials_per_tolerance"]:,}-trial launch stress study</h2><p>{progress["completed"]:,} / {progress["planned"]:,} runs finished'+(' · complete' if progress['complete'] else ' · running')+'.</p><p>Eight-second closed-loop launches; 1,000 paired trials at each of the 1% and 5% candidate tolerances. True friction is IID Uniform[0.76, 1.05], fixed within each episode. The controller, optimizer settings and DRS checks are unchanged. Every failure and unresolved outcome is retained.</p>'
        details='<p><a href="../">MPPI results</a> · <a href="../../">Interactive replay</a></p><h1>Current MPPI launch stress study</h1><p>Each trial executes eight seconds of closed-loop control. These are launch trials, not full laps or the 1,000 friction rollouts used inside each candidate certificate.</p>'
        if stats:
            table='<div class="scroll"><table><tr><th>Tolerance</th><th>Planned</th><th>Finished</th><th>Safe horizon</th><th>Violations</th><th>Unresolved</th><th>Violation rate</th><th>95% interval</th></tr>'
            for r in stats:
                epsilon=1 if r['scenario']=='traction' else 5
                rate='—' if r['violation_rate'] is None else f'{100*r["violation_rate"]:.2f}%'
                table+=f'<tr><td>{epsilon}%</td><td>{r["planned"]:,}</td><td>{r["executed"]:,}</td><td>{r["safe_horizon_completed"]:,}</td><td>{r["violations"]}</td><td>{r["unresolved"]}</td><td>{rate}</td><td>[{100*r["ci95_lower"]:.2f}, {100*r["ci95_upper"]:.2f}]%</td></tr>'
            table+='</table></div>'
            summary='<div class="scroll"><table><tr><th>Tolerance</th><th>Runs</th><th>Safe</th><th>Violations</th><th>Unresolved</th></tr>'
            for r in stats:
                summary+=f'<tr><td>{1 if r["scenario"]=="traction" else 5}%</td><td>{r["executed"]:,}</td><td>{r["safe_horizon_completed"]:,}</td><td>{r["violations"]}</td><td>{r["unresolved"]}</td></tr>'
            body+=summary+'</table></div>';details+=table
        if not progress['complete']:
            body+='<p>Study in progress. Partial outcomes are not the final safety result.</p>'
        else:
            body+='<p>These counts measure executed violations over eight seconds with friction in [0.76, 1.05]. Zero observed violations does not establish zero risk. The low-friction failure experiments below use the wider initial interval [0.45, 1.05], so they test different operating conditions.</p>'
        body+='<p><a href="stress/">Protocol, safety breakdown, figures and individual outcomes →</a></p></section>'
        details+='<p>Safe horizon means the full eight seconds completed without a modeled state/tire-force violation, unavailable certificate, or simulation/audit error. Tire-force infeasibility is not a simulated crash. The intervals are two-sided 95% Clopper–Pearson intervals. The candidate risk tolerance applies per certification decision, not per episode. It is an acceptance ceiling, not a target violation rate: zero observed violations is possible and does not establish zero risk. These empirical bounds describe this eight-second launch distribution, not whole-lap safety.</p>'
        if stats:
            details+='<h2>Unresolved outcomes and conservative bounds</h2><div class="scroll"><table><tr><th>Tolerance</th><th>Certificate unavailable</th><th>Simulation errors</th><th>Audit errors</th><th>95% unsafe-or-unresolved upper</th></tr>'
            for r in stats:
                details+=f'<tr><td>{1 if r["scenario"]=="traction" else 5}%</td><td>{r["certificate_unavailable"]}</td><td>{r["simulation_errors"]}</td><td>{r["audit_errors"]}</td><td>{100*r["unsafe_or_unresolved_upper95"]:.3f}%</td></tr>'
            details+='</table></div>'
        for name in ('manifest.json','aggregate.json','progress.json','all_episodes.csv','failures.csv','verification.json'):
            if (stress/name).exists():
                shutil.copyfile(stress/name,target/name)
                details+=f'<p><a href="{name}">{name}</a></p>'
        if (stress/'figures').exists():
            shutil.copytree(stress/'figures',target/'figures')
            for f in sorted((target/'figures').glob('*.png')):
                details+=f'<figure><img src="figures/{f.name}" alt="{f.stem.replace("_"," ")}"></figure>'
        details+=f'<p>Numerical source SHA256: <code>{current}</code></p><p>Frozen protocol and seeds are in the manifest. Full compressed records and per-run audits are retained in the local implementation checkout’s results/mppi-stress directory; the public download contains every episode’s outcome.</p>'
        (target/'index.html').write_text(page('Current MPPI stress study',details))
        if progress['complete']:
            with zipfile.ZipFile(target/'stress-results.zip','w',zipfile.ZIP_DEFLATED) as z:
                for f in sorted(target.rglob('*')):
                    if f.is_file() and f.suffix!='.zip':z.write(f,f.relative_to(target))
            (target/'index.html').write_text(page('Current MPPI stress study',details+'<p><a href="stress-results.zip">Download figures, counts, every episode and protocol</a></p>'))
    if risk_summary.exists():
        build_risk_page(results/'mppi-risk-report',out/'risk-probe',current)
        body+='<section><h2>How the probabilistic acceptance rule permits a violation</h2><p><strong>Matched example:</strong> true friction μ = 0.45, initial uncertainty [0.45, 1.05], seed 20100. At 0.30 s, the three tolerance settings generate identical informative controls and reference trajectories. A 0.3 s informative prefix has a 4.081% upper failure bound: 1% rejects it, while 5% and 10% accept it. A 1.2 s prefix has a 9.447% bound and passes only at 10%.</p><p>The 1% controller selects a nominal prefix and completes eight seconds without a violation. The 5% and 10% controllers select informative prefixes and violate the tire-force constraint at 0.50 s. The realized friction lies in the unsafe subset permitted by their probabilistic certificates.</p><p><strong>Separate candidate test:</strong> 10,000 fresh friction draws per selected policy produce 0, 230 and 530 failed prefix-plus-fallback rollouts. The corresponding 0%, 2.30% and 5.30% are observed candidate failure rates, not whole-episode crash rates.</p><p>Across 214 exploratory executions, 18 contain tire-force violations. All 214 replay audits reproduce the recorded outcomes correctly, including those 18 violations. MPPI, SMID, selection and DRS logic are unchanged. Targeted case counts do not estimate general driving failure probabilities.</p><p><a href="risk-probe/">Full explanation, matched comparisons, confidence intervals and data →</a></p></section>'
    body+=f'<section><h2>Full-lap validation and comparisons</h2><p>{completed} / {trials} validation laps completed; {violations} observed constraint violations. Eight paired friction draws, four methods and both tolerances. The reference demonstrations include all weighted-baseline settings.</p><p><a href="../paper/">Lap times, costs, safety intervals, comparisons and certificate holdouts →</a> · <a href="../paper/all_validation_episodes.csv">Every full-lap validation episode</a></p></section>'
    body+='<section><h2>Informative trajectory implementation</h2><p><code>gatekeeper.py:Gatekeeper.replan</code> calls <code>self.informative.optimize</code> with fixed γ = 2. <code>mppi.py:objective</code> accumulates the friction-regressor Gramian and subtracts <code>γ log(1 + I)</code>; <code>MPPI.optimize</code> returns optimized controls and predicted states. <code>belief.py:predicted_quantized_width</code> predicts SMID contraction. The gatekeeper applies paired safety, exploration-budget and discounted-contraction selection.</p></section>'
    media=results.parent/'docs/media'
    if (media/'comparison.mp4').exists():
        body+='<section><h2>Comparison film</h2><p>Actual demo renderer and recorded trajectories; follow, overhead and circuit views at 3× simulation speed.</p><video controls playsinline preload="metadata" poster="../media/comparison-follow.jpg" src="../media/comparison.mp4"></video></section>'
        (destination/'media').mkdir(exist_ok=True)
        for f in media.glob('comparison*'):shutil.copyfile(f,destination/'media'/f.name)
    (out/'index.html').write_text(page('MPPI racing results',body))
    return out/'index.html'


def build_risk_page(source,target,current):
    shutil.copytree(source,target)
    report=json.loads((source/'summary.json').read_text())
    body='<p><a href="../">All MPPI results</a> · <a href="../stress/">Completed 2,000-run study</a></p><h1>How probabilistic acceptance permits an executed violation</h1><p>In matched low-friction tests, increasing the allowed candidate risk changes which informative prefix is accepted. Some accepted prefixes then violate the modeled tire-force constraint during execution. The MPPI optimizer, informative γ = 2, SMID update, selection rule and DRS test are unchanged.</p>'
    body+='<section><h2>1. What 1%, 5% and 10% mean</h2><p>At each decision, a candidate consists of a planned prefix followed by the fallback controller. Certification draws 1,000 friction values from the assumed uniform distribution over the current SMID interval. Each friction value stays constant throughout its rollout.</p><p>A <strong>candidate rollout failure</strong> means a state or tire-force constraint is violated, or the terminal recovery condition is not met. From the failure count, the test computes a binomial upper confidence bound U on this event’s probability. It accepts the candidate only when <strong>U ≤ ε</strong>.</p><p>Here ε is 1%, 5% or 10%. The confidence allocation is δ/64 = 0.05/64 per candidate: a one-sided 99.921875% bound for each completed check, with the confidence budget allocated across at most 64 checks per decision. This confidence level is distinct from the allowed failure probability ε.</p><p><strong>ε applies to a candidate plus fallback.</strong> It is not a target failure frequency or a whole-episode safety guarantee. A candidate can pass while some friction realizations still violate its constraints.</p></section>'
    body+='<section><h2>2. What happened during actual driving</h2><p>The three runs use the same starting state at 0 m, seed 20100, and constant true friction μ = 0.45. The initial uncertainty is [0.45, 1.05]. At t = 0.30 s, executed measurements have narrowed it to [0.45, 0.578775]. The tolerance settings then produce these decisions:</p>'
    body+='<div class="scroll"><table><tr><th>ε</th><th>Selected prefix</th><th>Certification failures</th><th>Upper bound U</th><th>Executed outcome</th></tr>'
    for r in report['examples']:
        execution='No violation through 8 s' if r['violation'] is None else f'Tire-force violation at {r["violation"]["time"]:.2f} s'
        kind='Nominal' if r['kind']=='mission' else 'Informative'
        body+=f'<tr><td>{100*r["epsilon"]:.0f}%</td><td>{kind}, {r["duration"]:.1f} s</td><td>{r["sample_failures"]} / 1,000</td><td>{100*r["failure_upper"]:.3f}%</td><td>{execution}</td></tr>'
    body+='</table></div><p>At 5% and 10%, the accepted informative action requires combined longitudinal/lateral tire forces beyond the modeled capacity at μ = 0.45. At t = 0.50 s, the normalized tire-force margin is −0.002639. The simulator stops at this constraint violation; it does not simulate a collision. The 1% run completes its eight-second test without an executed violation.</p><figure><img src="figures/01_accepted_failure.png" alt="Executed tire-force margin becomes negative at 0.50 s for 5% and 10%; the 1% run stays above zero"><figcaption>Left: measured simulation outcome, with zero marking the tire-force constraint boundary. The 5% and 10% traces overlap through the violation. Right: the certificate upper bounds that permitted the selected prefixes.</figcaption></figure></section>'
    body+='<section><h2>3. Isolating the effect of the tolerance</h2><p>At t = 0.30 s, the generated informative controls, reference states and certification sample seeds are identical across the three runs. Only the risk threshold changes which prefixes pass:</p><div class="scroll"><table><tr><th>Same informative candidate</th><th>Full-batch upper bound</th><th>1%</th><th>5%</th><th>10%</th></tr>'
    for r in report['matched_informative_candidates']:
        body+=f'<tr><td>{r["duration"]:.1f} s prefix</td><td>{100*r["full_batch_upper"]:.3f}%</td>'+''.join(f'<td>{"Accept" if c["accepted"] else "Reject"}</td>' for c in r['acceptance'])+'</tr>'
    body+='</table></div><p>Rejection can stop sampling early. The displayed full-batch bounds come from the matching completed 1,000-sample checks at the higher tolerances; partial rejection batches are not treated as completed certificates.</p><p>This comparison shows the acceptance mechanism directly: the 5% setting admits the short informative prefix that 1% rejects, and the 10% setting additionally admits the longer prefix that 5% rejects. The selected prefixes still pass the existing pairing, exploration-budget and information-gain selection rules.</p></section>'
    body+='<section><h2>4. What the 2.30% and 5.30% estimates measure</h2><p>This is a separate experiment. Each selected policy is frozen at the recorded t = 0.30 s state. We draw 10,000 fresh friction values from Uniform[0.45, 0.578775] and execute that fixed prefix followed by fallback, checking constraints and terminal recovery. There is no subsequent replanning in these tests.</p><p>The draws are independent of the certification samples and paired across the three policies. Each reported percentage is the fraction of these candidate tests that fail any certification condition. It is not an observed whole-episode crash rate.</p><div class="scroll"><table><tr><th>Policy selected at ε</th><th>Prefix + fallback horizon</th><th>Failed tests</th><th>Observed candidate failure rate</th><th>95% interval</th></tr>'
    for r in report['examples']:
        lo,hi=r['holdout_ci95'];rate=100*r['holdout_failures']/r['holdout_samples']
        body+=f'<tr><td>{100*r["epsilon"]:.0f}%</td><td>{r["certificate_horizon"]:.1f} s</td><td>{r["holdout_failures"]} / {r["holdout_samples"]:,}</td><td>{rate:.2f}%</td><td>[{100*lo:.4f}%, {100*hi:.4f}%]</td></tr>'
    body+='</table></div><p>These are pointwise two-sided 95% Clopper–Pearson intervals for the independent test results. They differ from the confidence allocation used to accept candidates. In particular, 0/10,000 is zero observed failures, not proof of zero failure probability.</p><figure><img src="figures/03_independent_candidate_risk.png" alt="Independent fixed-policy tests yield observed candidate failure rates of 0%, 2.30% and 5.30%"><figcaption>The horizontal marks are candidate tolerances. Bars and error bars describe independent prefix-plus-fallback tests, including terminal recovery. <a href="all_holdout_draws.csv">Every friction draw and outcome</a>.</figcaption></figure><p>The observed candidate failure rates remain below their allowed tolerances. Nevertheless, the actual μ = 0.45 realization violates the tire-force constraint for the two informative policies. This is the residual risk that the probabilistic acceptance rule permits.</p></section>'
    body+='<section><h2>5. Nearby friction values and the earlier zero-violation study</h2><p>We then tested ten nearby friction values, using the same seed and initial interval: 0.4501, 0.4505, 0.451, 0.4515, 0.452, 0.4525, 0.453, 0.4535, 0.454 and 0.455. Executed violation counts were <strong>0/10 at 1%, 6/10 at 5%, and 10/10 at 10%</strong>. The 5% failures occur through μ = 0.4525; 10% also fails at the four higher tested values.</p><figure><img src="figures/02_nearby_friction.png" alt="Targeted nearby-friction executions: zero violations at 1%, six at 5%, ten at 10%, plus the original endpoint example"><figcaption>The figure also includes the original μ = 0.45 example. These are deliberately selected sensitivity tests, so their failure fractions do not estimate general driving failure probabilities.</figcaption></figure><p>The earlier 2,000 launches sampled true friction from <strong>[0.76, 1.05]</strong>, with the same initial interval. They observed zero violations over eight seconds. The new failures occur near <strong>μ = 0.45</strong>, with initial uncertainty <strong>[0.45, 1.05]</strong>. The two studies test different operating conditions. An additional 64 launches at 10% using the original [0.76, 1.05] distribution also completed without violations.</p></section>'
    body+='<section><h2>6. Complete counts and what the audits verify</h2><p>The boundary grid deliberately emphasizes low friction; the nearby-value refinement was chosen after discovering the failure. The table preserves all exploratory outcomes, but its targeted-case fractions are not population risk estimates. “Unresolved” counts an unavailable certificate, simulation error or audit error.</p><div class="scroll"><table><tr><th>Experiment</th><th>ε</th><th>Executed runs</th><th>Executed violations</th><th>Unresolved</th></tr>'
    for r in report['groups']:
        body+=f'<tr><td>{html.escape(r["group"])}</td><td>{100*r["epsilon"]:.0f}%</td><td>{r["runs"]}</td><td>{r["violations"]}</td><td>{r["unresolved"]}</td></tr>'
    violations=sum(r['violations'] for r in report['groups'])
    body+=f'</table></div><p><strong>{report["verified_records"]} executions include {violations} tire-force violations.</strong> All {report["audits_passed"]} replay audits passed: the recorded controls, measurements, decisions and outcomes were reproduced correctly, including the violating executions. An audit pass is a consistency check, not a claim that the execution was safe. The original certification sample counts also matched when regenerated for the three illustrated decisions.</p><p><a href="all_episodes.csv">Every exploratory execution</a> · <a href="summary.json">Certificate and independent-test details</a> · <a href="mppi-risk-probe-manifest.json">Initial probe protocol</a> · <a href="mppi-risk-refinement-manifest.json">Refinement protocol</a></p></section>'
    body+='<section><h2>Supported conclusion</h2><p>These matched experiments demonstrate that a higher permitted candidate risk can admit informative prefixes with larger unsafe subsets of the current friction uncertainty interval, and that particular friction realizations can produce executed tire-force violations. They do not establish a universal relationship between ε and whole-episode failure probability.</p></section>'
    body+=f'<p>Numerical source SHA256: <code>{current}</code>. Reproduce with <code>tools/probe_mppi_risk.py</code>, <code>tools/refine_mppi_risk.py</code>, then <code>tools/report_mppi_risk.py</code> in the implementation repository.</p>'
    (target/'index.html').write_text(page('Probabilistic acceptance and executed violations',body))
    with zipfile.ZipFile(target/'risk-probe-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(target.rglob('*')):
            if f.is_file() and f.suffix!='.zip':z.write(f,f.relative_to(target))
    (target/'index.html').write_text(page('Probabilistic acceptance and executed violations',body+'<p><a href="risk-probe-results.zip">Download figures, all outcomes and protocols</a></p>'))
