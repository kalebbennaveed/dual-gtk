"""Publication figures and tables from complete, auditable experiment records."""
from pathlib import Path
import csv
import html
import hashlib
import json
import zipfile
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .experiments import METHODS, load_rows, load_record, atomic_json

NAMES = dict(proposed="Proposed", passive="Passive", weighted="Weighted information", robust="No learning")
COLORS = dict(proposed="#168268", passive="#387cba", weighted="#bb8124", robust="#ab6687")
STYLES = dict(proposed="-", passive="--", weighted="-.", robust=":")


def failure_upper(k, n, confidence=.95):
    """One-sided Clopper–Pearson bound, including the zero-failure case."""
    if not 0 <= k <= n or n < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid binomial experiment")
    return 1. if k == n else float(beta.ppf(confidence, k + 1, n - k))


def bootstrap_mean_ci(values, seed=421, samples=5000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [.025, .975]))


def summary_stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(n=0, mean=None, sd=None, median=None, p05=None, p95=None, minimum=None, maximum=None)
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values)>1 else None,
                median=float(np.median(values)), p05=float(np.quantile(values,.05)), p95=float(np.quantile(values,.95)),
                minimum=float(values.min()), maximum=float(values.max()))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def tex_escape(value):
    text = str(value)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(a, b)
    return text.replace("−", "-").replace("—", "--").replace("±", r"$\pm$").replace("μ", r"$\mu$")


def table_files(directory, name, headers, data):
    """Booktabs fragments; long grids paginate using longtable."""
    environment='longtable' if len(data)>25 else 'tabular'
    tex = [r"\begin{" + environment + "}{l" + "r"*(len(headers)-1) + "}", r"\toprule",
           " & ".join(map(tex_escape, headers)) + r" \\", r"\midrule"]
    if environment=='longtable':
        tex += [r'\endfirsthead',r'\toprule'," & ".join(map(tex_escape,headers))+r' \\',r'\midrule',r'\endhead']
    tex += [" & ".join(map(tex_escape, row)) + r" \\" for row in data]
    tex += [r"\bottomrule", r"\end{" + environment + "}"]
    (directory / (name + ".tex")).write_text("\n".join(tex) + "\n")
    write_csv(directory / (name + ".csv"), [dict(zip(headers, row)) for row in data])
    return "<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in data) + "</tbody></table></div>"


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":10,
        "axes.spines.top":False, "axes.spines.right":False, "axes.grid":True,
        "grid.alpha":.16, "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none"})


def legend(fig, truth=False):
    handles = [Line2D([],[],color=COLORS[m],linestyle=STYLES[m],label=NAMES[m]) for m in METHODS]
    if truth:
        handles.append(Line2D([],[],color="#202020",linestyle="--",label="True μ"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)


def save_figure(fig, directory, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{name}.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def line(ax, x, y, method, **kwargs):
    return ax.plot(x,y,color=COLORS[method],linestyle=STYLES[method],linewidth=1.5,**kwargs)


def aggregate(rows):
    mc = sorted([r for r in rows if r["case"]["group"] == "monte_carlo"],
                key=lambda r:(r['case']['trial'],r['case']['method']))
    statistics, safety, paired = [], [], []
    for method in METHODS:
        rs = [r for r in mc if r["case"]["method"] == method]
        ok = [r["metrics"] for r in rs if r["status"] == "ok"]
        completed = [m for m in ok if m["lap_complete"]]
        errors = len(rs) - len(ok)
        # An error is unresolved, so the conservative upper bound counts it as
        # a failure; no successful-run-only denominator is used for safety.
        excluded = sum(m["containment_violations"]>0 for m in ok)
        violations=sum(m.get('constraint_violated',False) for m in ok)
        unavailable=sum(m.get('certificate_unavailable',False) for m in ok)
        safety.append(dict(method=method, trials=len(rs), audited=len(ok), errors=errors,
            laps_completed=len(completed), constraint_violations=violations, certificate_unavailable=unavailable, containment_failure_episodes=excluded,
            audited_frames=sum(m["frames"] for m in ok),
            unsafe_or_unresolved_upper95=failure_upper(errors+violations+unavailable, len(rs)),
            containment_or_unresolved_upper95=failure_upper(errors+excluded, len(rs)),
            minimum_clearance=min((m["minimum_clearance"] for m in ok),default=None),
            minimum_tire_margin=min((m["minimum_tire_margin"] for m in ok),default=None),
            minimum_containment_margin=min((m["containment_margin"] for m in ok),default=None),
            contraction_episodes=sum(m["final_width"]<.6-1e-8 for m in ok)))
        for metric in ("time", "cost", "peak_speed", "mean_speed", "final_width", "width_integral", "informative_seconds",
                       "mission_seconds", "backup_seconds", "time_to_90pct_contraction", "time_to_99pct_contraction",
                       "first_mission_time", "first_fifty_time", "minimum_clearance", "minimum_tire_margin",
                       "budget_charged", "wall_seconds", "accepted_plans", "retained_plans", "candidate_rejections",
                       "certificate_upper_max", "accepted_sample_failures", "certification_rollouts"):
            # A partial trajectory's cost/time is not called a lap result.
            eligible = completed if metric in ("time", "cost") else ok
            statistics.append(dict(method=method,metric=metric,**summary_stats([m.get(metric) for m in eligible])))
    keyed = {(r["case"]["trial"],r["case"]["method"]):r for r in mc}
    for other in METHODS[1:]:
        for subgroup, predicate in (("all_mu",lambda mu:True),("mu_at_least_0.75",lambda mu:mu>=.75),("mu_below_0.75",lambda mu:mu<.75)):
            pairs = [(r,keyed[(i,other)]) for (i,m),r in keyed.items() if m=="proposed" and predicate(r["case"]["true_mu"])]
            complete = [(a["metrics"],b["metrics"]) for a,b in pairs if a["status"]==b["status"]=="ok" and a["metrics"]["lap_complete"] and b["metrics"]["lap_complete"]]
            for metric in ("time", "cost"):
                diffs = [b[metric]-a[metric] for a,b in complete]
                relative = [100*(b[metric]-a[metric])/b[metric] for a,b in complete]
                lo,hi = bootstrap_mean_ci(diffs)
                paired.append(dict(comparator=other,subgroup=subgroup,metric=metric,planned_pairs=len(pairs),complete_pairs=len(complete),
                    mean_improvement=float(np.mean(diffs)) if diffs else None,ci95_low=lo,ci95_high=hi,
                    mean_percent_improvement=float(np.mean(relative)) if relative else None,
                    proposed_wins=sum(d>1e-8 for d in diffs),ties=sum(abs(d)<=1e-8 for d in diffs),comparator_wins=sum(d< -1e-8 for d in diffs)))
    return statistics, safety, paired


def figures(directory, rows):
    out = directory / "figures"
    out.mkdir(exist_ok=True)
    ok = [r for r in rows if r["status"] == "ok"]
    mc = [r for r in ok if r["case"]["group"] == "monte_carlo"]
    grid = [r for r in ok if r["case"]["group"] == "mu_grid"]
    captions = []
    def save(fig,name,caption):
        save_figure(fig,out,name)
        captions.append(dict(name=name,caption=caption))
    if grid:
        fig,axs=plt.subplots(2,3,figsize=(12,6.8),layout="constrained")
        measures=[("time","Lap time [s]"),("cost","Lap mission cost"),("peak_speed","Peak speed [m/s]"),
                  ("final_width","Final SMID width"),("informative_seconds","Informative driving [s]"),("width_integral","Integrated SMID width [s]")]
        for method in METHODS:
            rs=sorted([r for r in grid if r["case"]["method"]==method],key=lambda r:r["case"]["true_mu"])
            for ax,(metric,label) in zip(axs.flat,measures):
                eligible=[r for r in rs if r["metrics"]["lap_complete"] or metric not in ("time","cost")]
                line(ax,[r["case"]["true_mu"] for r in eligible],[r["metrics"][metric] for r in eligible],method,marker="o",markersize=3)
                ax.set(xlabel="True tire coefficient μ",ylabel=label)
        axs[1,0].set_yscale("log");axs[1,2].set_yscale('log');legend(fig)
        save(fig,"01_friction_sweep","Deterministic friction sweep, including both support endpoints. Lap time and lap cost include completed laps only; any missing runs remain listed in the tables. No statistical confidence interval is attached to this grid.")

    fig,axs=plt.subplots(2,2,figsize=(10,6.5),layout="constrained")
    for ax,(metric,label) in zip(axs.flat,[("time","Lap time [s]"),("cost","Lap mission cost"),("final_width","Final SMID width"),("informative_seconds","Informative driving [s]")]):
        for j,method in enumerate(METHODS):
            vals=[r["metrics"][metric] for r in mc if r["case"]["method"]==method and (r["metrics"]["lap_complete"] or metric not in ("time","cost"))]
            if vals:
                ax.boxplot([vals],positions=[j],widths=.45,showfliers=False,patch_artist=True,
                    boxprops=dict(facecolor=COLORS[method],alpha=.25),medianprops=dict(color=COLORS[method],linewidth=2))
                jitter=np.random.default_rng(5).uniform(-.14,.14,len(vals))
                ax.scatter(j+jitter,vals,s=6,color=COLORS[method],alpha=.35)
        ax.set(xticks=range(4),xticklabels=["Proposed","Passive","Weighted","No learning"],ylabel=label)
        if metric=="final_width":ax.set_yscale("log")
    save(fig,"02_monte_carlo_distributions","Independent μ ∼ Uniform[0.45, 1.05], paired across methods. Points are episodes; boxes show quartiles and medians, whiskers extend to 1.5 IQR. These are outcome distributions, not confidence bands.")

    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
    for method in METHODS:
        all_method=[r for r in rows if r['case']['group']=='monte_carlo' and r['case']['method']==method]
        rs=[r for r in all_method if r['status']=='ok']
        horizon=max(r['case']['seconds'] for r in all_method)
        for ax,metric in zip(axs,('time','time_to_99pct_contraction')):
            events=sorted(r['metrics'][metric] for r in rs if r['metrics'][metric] is not None and (metric!='time' or r['metrics']['lap_complete']))
            xs=np.r_[0.,events,horizon];ys=np.r_[0.,np.arange(1,len(events)+1)/len(all_method),len(events)/len(all_method)]
            line(ax,xs,ys,method,drawstyle='steps-post')
            ax.set(xlabel='Time [s]',ylabel='Fraction of all trials',ylim=(-.02,1.04))
        axs[0].set_title('Lap completed');axs[1].set_title('SMID width ≤ 1% of prior width')
    legend(fig);save(fig,'12_completion_and_identification','Empirical event fractions use every planned Monte Carlo trial in the denominator. Episodes that never cross a threshold, fail, or do not finish remain in the denominator; curves can plateau below one. Events are counted once per episode, not once per simulation frame.')

    selected={}
    for mu in (.45,.85,1.05):
        for method in METHODS:
            match=next((r for r in grid if r["case"]["method"]==method and abs(r["case"]["true_mu"]-mu)<1e-8),None)
            if match:selected[(mu,method)]=load_record(directory,match)
    if selected:
        for centered in (False,True):
            fig,axs=plt.subplots(3,4,figsize=(12,7),layout="constrained")
            for i,mu in enumerate((.45,.85,1.05)):
                for j,method in enumerate(METHODS):
                    ax=axs[i,j];r=selected.get((mu,method))
                    if r:
                        fs=r["frames"];t=np.array([f["t"] for f in fs]);bounds=np.array([f["mu"] for f in fs])
                        y=bounds-mu if centered else bounds
                        ax.fill_between(t,y[:,0],y[:,1],alpha=.18,color=COLORS[method])
                        for c in (0,1):line(ax,t,y[:,c],method)
                        ax.axhline(0 if centered else mu,color="#202020",linestyle="--",linewidth=1)
                        ax.set(xlim=(0,r["summary"]["time"] if centered else 10),title=rf"{NAMES[method]} · $\mu_{{\rm true}} = {mu:g}$")
                    if centered:ax.set_yscale("symlog",linthresh=1e-4)
                    else:ax.set_ylim(.425,1.075)
                    if j==0:ax.set_ylabel("Bound − true μ" if centered else "SMID interval")
                    if i==2:ax.set_xlabel("Time [s]")
            fig.suptitle('Black dashed: zero error (true parameter)' if centered else 'Black dashed: true tire coefficient',fontsize=10)
            save(fig,"04_smid_error_bounds" if centered else "03_smid_bounds_truth",
                "SMID bounds relative to truth over complete episodes. Dashed zero must stay inside every band. The symmetric-log axis resolves small intervals; it is not a probabilistic confidence band." if centered else
                "First 10 s of set-membership identification (SMID). Each panel includes the actual lower/upper bounds and a dashed true μ. All methods use the same bounded-error sensor; no learning retains the prior.")

    fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout="constrained")
    for method in METHODS:
        rs=[r for r in mc if r["case"]["method"]==method]
        for ax,(metric,label) in zip(axs,[("minimum_clearance","Minimum footprint clearance [m]"),("minimum_tire_margin","Minimum tire-force margin"),("containment_margin","Minimum SMID containment margin")]):
            ax.scatter([r["case"]["true_mu"] for r in rs],[r["metrics"][metric] for r in rs],s=12,color=COLORS[method],alpha=.6,marker={"proposed":"o","passive":"s","weighted":"^","robust":"x"}[method])
            ax.axhline(0,color="#303030",linestyle="--",linewidth=1)
            ax.set(xlabel="True tire coefficient μ",ylabel=label)
    axs[2].set_yscale("symlog",linthresh=1e-8);legend(fig)
    save(fig,"05_safety_and_containment","One point per audited Monte Carlo episode. Tire margin is the dimensionless squared friction-circle slack. Containment margin is min over time of min(μtrue − lower, upper − μtrue); negative values would indicate exclusion. Unresolved runs are reported separately, never treated as safe points.")

    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    index={(r["case"]["trial"],r["case"]["method"]):r for r in mc}
    for other in METHODS[1:]:
        pairs=[(a,index[(i,other)]) for (i,m),a in index.items() if m=="proposed" and (i,other) in index and a["metrics"]["lap_complete"] and index[(i,other)]["metrics"]["lap_complete"]]
        for ax,metric in zip(axs,("time","cost")):
            ax.scatter([a["case"]["true_mu"] for a,b in pairs],[b["metrics"][metric]-a["metrics"][metric] for a,b in pairs],color=COLORS[other],s=14,label=NAMES[other],alpha=.65)
            ax.axhline(0,color="#333",linestyle="--",linewidth=1)
            ax.set(xlabel="True tire coefficient μ",ylabel="Comparator − proposed [s]" if metric=="time" else "Comparator − proposed cost")
    axs[0].legend(frameon=False,fontsize=8)
    save(fig,"06_paired_improvement","Paired differences at identical true μ. Positive values favor proposed; negative values favor the comparator. This exposes cases where learning cannot make the racing request feasible. Bootstrap intervals for the mean paired differences are in the tables.")

    gain_rows=[]
    for row in mc:
        if row['case']['method']!='proposed':continue
        record=load_record(directory,row)
        widths=np.array([f['mu'][1]-f['mu'][0] for f in record['frames']])
        for event in record['events']:
            candidate=event.get('selected',{})
            if candidate.get('kind')!='informative':continue
            i=int(round(event['time']/record['dt']))
            j=int(round((event['time']+candidate['duration'])/record['dt']))
            if j>=len(widths):continue
            gain_rows.append(dict(trial=row['case']['trial'],true_mu=row['case']['true_mu'],time=event['time'],
                duration=candidate['duration'],predicted_contraction=candidate['gain'],
                realized_contraction=float(widths[i]-widths[j]),reserved_charge=candidate['charge']))
    if gain_rows:
        write_csv(directory/'tables'/'informative_prediction.csv',gain_rows)
        fig,ax=plt.subplots(figsize=(5.7,4.5),layout='constrained')
        points=ax.scatter([r['predicted_contraction'] for r in gain_rows],[r['realized_contraction'] for r in gain_rows],
            c=[r['true_mu'] for r in gain_rows],cmap='viridis',s=14,alpha=.65)
        ax.plot([0,.6],[0,.6],color='#303030',linestyle='--',linewidth=1)
        ax.set(xlabel='Predicted SMID contraction (midpoint model)',ylabel='Realized SMID contraction',xlim=(-.01,.61),ylim=(-.01,.61))
        fig.colorbar(points,ax=ax,label='True tire coefficient μ')
        save(fig,'13_information_prediction','Predicted versus realized total interval contraction over fully executed proposed informative prefixes in the Monte Carlo batch. The diagonal denotes equality. These correlated commitments are diagnostic points, not independent statistical trials. This compares total contraction, not the unobserved counterfactual extra gain over a paired mission trajectory.')

    weights=[r for r in ok if r["case"]["group"]=="weight_sweep"]
    if weights:
        fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
        for color,mu in zip(("#426caa","#bd7b38","#34947d"),(.65,.85,1.05)):
            rs=sorted([r for r in weights if abs(r["case"]["true_mu"]-mu)<1e-8],key=lambda r:r["case"]["information_weight"])
            for ax,(metric,label) in zip(axs,[("time","Lap time [s]"),("final_width","Final SMID width"),("informative_seconds","Informative driving [s]")]):
                rs_valid=[r for r in rs if metric!="time" or r["metrics"]["lap_complete"]]
                ax.plot([r["case"]["information_weight"] for r in rs_valid],[r["metrics"][metric] for r in rs_valid],"o-",color=color,label=f"μ = {mu:g}",markersize=4)
                ax.set(xlabel="Fixed excitation weight λ",ylabel=label)
        axs[1].set_yscale("log");axs[0].legend(frameon=False)
        save(fig,"07_weight_sensitivity","Weighted-comparator sensitivity at three fixed friction values, including λ = 0. This checks the effect of persistent excitation without claiming proposed dominates every possible information objective.")

    if (.85,"proposed") in selected:
        sensor_rows=[]
        fig,axs=plt.subplots(2,2,figsize=(10,6),layout="constrained")
        for ax,method in zip(axs.flat,METHODS):
            r=selected.get((.85,method))
            if not r:continue
            fs=r['frames'];t=np.array([f['t'] for f in fs]);x=np.array([f['state'] for f in fs])
            rates=np.diff(x[:,5])/r['dt'];measured=.36*np.rint(rates/.36)
            width=np.array([f['mu'][1]-f['mu'][0] for f in fs])
            strip_slack=[]
            for f in fs[:-1]:
                o=f['observation'];y=np.array(o['residual']);g=np.array(o['regressor']);b=np.array(o['bound'])
                strip_slack.append(float(np.min(b-np.maximum(np.abs(y-.45*g),np.abs(y-1.05*g)))))
            sensor_rows.append(dict(method=method,true_mu=.85,observations=len(rates),
                nonzero_quantized_observations=int(np.count_nonzero(measured)),
                contracting_updates=int(np.sum(np.diff(width)<-1e-10)),
                peak_abs_yaw_acceleration=float(np.abs(rates).max()),
                prior_strip_minimum_slack=float(min(strip_slack)),final_width=float(width[-1])))
            ax.axhspan(-.18,.18,color='#a9b9ad',alpha=.2,label='Zero-output quantization bin')
            line(ax,t[:-1],rates,method)
            ax.step(t[:-1],measured,where='post',color='#303630',alpha=.5,linewidth=.7)
            ax.set(xlabel='Time [s]',ylabel='Yaw acceleration [rad/s²]',title=NAMES[method])
        axs[0,0].legend(frameon=False,fontsize=7)
        write_csv(directory/'tables'/'sensor_diagnostics.csv',sensor_rows)
        table_files(directory/'tables','sensor_diagnostics_summary',
            ['Method','Samples','Nonzero readings','Contracting updates','Peak yaw accel.','Min prior-strip slack'],
            [[NAMES[r['method']],r['observations'],r['nonzero_quantized_observations'],r['contracting_updates'],fmt(r['peak_abs_yaw_acceleration'],4),fmt(r['prior_strip_minimum_slack'],5)] for r in sensor_rows])
        save(fig,'11_sensor_excitation','Executed yaw acceleration and quantized sensor output at μ = 0.85. Shading is the ±0.18 rad/s² zero-output bin. A nonnegative prior-strip slack in sensor_diagnostics.csv means every observation remains compatible with the entire initial interval. The passive estimator still processes every sample.')

        fig,axs=plt.subplots(2,2,figsize=(11,6.5),layout="constrained")
        components=("time_cost","speed_cost","lateral_cost","heading_cost","force_cost","steering_cost")
        component_colors=("#467ca3","#aac3d1","#6a947b","#a7bf86","#d3a258","#b28091")
        for j,method in enumerate(METHODS):
            r=selected.get((.85,method))
            if not r:continue
            fs=r["frames"];t=np.array([f["t"] for f in fs]);w=np.array([f["mu"][1]-f["mu"][0] for f in fs])
            line(axs[0,0],t,[f["state"][3] for f in fs],method)
            line(axs[0,1],t,w,method)
            for mode,color in (("mission","#168268"),("informative","#d29c35"),("backup","#acb4b8")):
                segments=[(f["t"],r["dt"]) for f in fs[:-1] if f["mode"]==mode]
                axs[1,0].broken_barh(segments,(j-.3,.6),facecolors=color)
            row=next(v for v in grid if v["case"]["method"]==method and abs(v["case"]["true_mu"]-.85)<1e-8)
            bottom=0
            for metric,color in zip(components,component_colors):
                value=row["metrics"][metric]
                axs[1,1].bar(j,value,bottom=bottom,color=color,label=metric.replace("_cost","") if j==0 else None)
                bottom+=value
        axs[0,0].set(xlabel="Time [s]",ylabel="Executed speed [m/s]")
        axs[0,1].set(xlabel="Time [s]",ylabel="SMID interval width",yscale="log")
        axs[1,0].set(xlabel="Time [s]",yticks=range(4),yticklabels=["Proposed","Passive","Weighted","No learning"])
        axs[1,0].set_title("Green: mission · gold: informative · gray: backup",fontsize=8)
        axs[1,1].set(xticks=range(4),xticklabels=["Proposed","Passive","Weighted","No learning"],ylabel="Integrated mission cost")
        axs[1,1].legend(frameon=False,fontsize=7,ncol=2);legend(fig)
        save(fig,"08_mechanism_and_cost","Mechanism at μ = 0.85: executed speed, identification, committed-policy mode, and all six terms of the common mission cost. The plotted cost is executed cost, not the separately reserved exploration budget.")

        fig,axs=plt.subplots(1,2,figsize=(11,3.8),layout="constrained")
        for method in METHODS:
            r=selected.get((.85,method))
            if not r:continue
            fs=r["frames"];t=np.array([f["t"] for f in fs]);line(axs[0],t,[f["budget"] for f in fs],method)
            events=[e for e in r["events"] if e["status"] in ("committed","retained")]
            accepted=np.cumsum([e["status"]=="committed" for e in events])
            line(axs[1],[e["time"] for e in events],accepted,method)
        axs[0].set(xlabel="Time [s]",ylabel="Reserved exploration cost",title="Proposed budget limit = 8 (not reached)")
        axs[1].set(xlabel="Time [s]",ylabel="Cumulative accepted commitments")
        legend(fig);save(fig,"09_budget_and_commitments","Exploration charges and accepted commitments at μ = 0.85. Proposed's budget is not active in this case; its stopping behavior cannot be attributed to exhausting the budget. Weighted has no budget, and its zero charge is not zero physical cost.")

    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    for j,method in enumerate(METHODS):
        rs=[r for r in mc if r["case"]["method"]==method]
        times=[r["metrics"]["wall_seconds"] for r in rs]
        axs[0].scatter(np.full(len(times),j),times,s=9,color=COLORS[method],alpha=.5)
        axs[1].scatter([r["case"]["true_mu"] for r in rs],[r["metrics"]["retained_plans"] for r in rs],s=12,color=COLORS[method],alpha=.6)
    axs[0].set(xticks=range(4),xticklabels=["Proposed","Passive","Weighted","No learning"],ylabel="Simulation wall time [s]")
    axs[1].set(xlabel="True tire coefficient μ",ylabel="Replans retaining the old commitment")
    legend(fig);save(fig,"10_runtime_and_retention","Warm-cache simulation wall time under concurrent workers, excluding initial certification and post-run audit. These are throughput observations, not isolated real-time planner latency measurements. Retention counts include failed replacement searches.")
    cert_rows=[]
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    epsilon=rows[0].get('metrics',{}).get('epsilon',.01)
    for method in METHODS:
        rs=[r for r in mc if r['case']['method']==method]
        for r in rs:
            m=r['metrics']
            cert_rows.append(dict(method=method,trial=r['case']['trial'],true_mu=r['case']['true_mu'],
                epsilon=m['epsilon'],maximum_accepted_upper=m['certificate_upper_max'],
                accepted_sample_failures=m['accepted_sample_failures'],evaluated_rollouts=m['certification_rollouts']))
        vals=[r for r in rs if r['metrics']['certificate_upper_max'] is not None]
        axs[0].scatter([r['case']['true_mu'] for r in vals],[100*r['metrics']['certificate_upper_max'] for r in vals],s=12,color=COLORS[method],alpha=.65)
        axs[1].scatter([r['case']['true_mu'] for r in rs],[r['metrics']['accepted_sample_failures'] for r in rs],s=12,color=COLORS[method],alpha=.65)
    axs[0].axhline(100*epsilon,color='#333',linestyle='--')
    axs[0].set(xlabel='True tire coefficient μ',ylabel='Largest accepted candidate upper bound [%]')
    axs[1].set(xlabel='True tire coefficient μ',ylabel='Failed sampled rollouts in accepted candidates')
    legend(fig);save(fig,'14_candidate_certificates','Online certification diagnostics. Every accepted candidate completed N=1000 samples and its corrected binomial upper bound was <= the dashed epsilon threshold. Sampled failures are prediction outcomes, not executed-car violations. Points are episodes; certificate maxima are diagnostic, not independent trials.')
    write_csv(directory/'tables'/'candidate_certificates.csv',cert_rows)
    return captions


def build_report(directory):
    directory=Path(directory).resolve()
    manifest,rows=load_rows(directory)
    specification=Path(__file__).parent.parent/'docs'/'method.md'
    if specification.exists():
        (directory/'method.md').write_text(specification.read_text())
    style()
    tables=directory/"tables";tables.mkdir(exist_ok=True)
    stats,safety,paired=aggregate(rows)
    atomic_json(directory/"statistics.json",dict(summary=stats,safety=safety,paired=paired))
    atomic_json(directory/'analysis.json',dict(
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('experiments.py','paper_report.py','validation.py')},
        confidence=.95,binomial_bound='one-sided Clopper-Pearson, per method',
        bootstrap_resamples=5000,bootstrap_seed=421,bootstrap_unit='paired episode',
        containment_tolerance=1e-9,contraction_threshold=.6-1e-8,
        lap_metrics='completed laps only; completion counts reported separately',
        errors='retained and conservatively treated as unresolved failures in upper bounds'))
    write_csv(tables/"summary_statistics.csv",stats)
    write_csv(tables/"safety_statistics.csv",safety)
    write_csv(tables/"paired_statistics.csv",paired)
    flat=[]
    for row in rows:
        metrics=row.get("metrics",{})
        flat.append(dict(**row["case"],status=row["status"],error=row.get("error",""),
            **{k:v for k,v in metrics.items() if k not in row["case"] and not isinstance(v,(list,dict))}))
    write_csv(tables/"all_episodes.csv",flat)
    for group in ("mu_grid","weight_sweep"):
        write_csv(tables/(group+".csv"),[r for r in flat if r["group"]==group])
    lookup={(r["method"],r["metric"]):r for r in stats}
    performance=[]
    for method in METHODS:
        cells=[NAMES[method]]
        for metric in ("time","cost","peak_speed","final_width","informative_seconds"):
            r=lookup[(method,metric)];digits=5 if metric=="final_width" else 2
            cells.append(f"{fmt(r['mean'],digits)} ± {fmt(r['sd'],digits)}")
        performance.append(cells)
    table_html=table_files(tables,"performance",["Method","Lap time [s]","Mission cost","Peak speed [m/s]","Final μ width","Informative time [s]"],performance)
    for name in ('mu_grid','weight_sweep'):
        entries=[r for r in flat if r['group']==name]
        if entries:
            table_files(tables,name+'_summary',['True μ','Method','Weight','Lap time [s]','Cost','Final width','Lap complete'],
                [[fmt(r['true_mu'],3),NAMES[r['method']],fmt(r['information_weight'],1),fmt(r.get('time')) if r.get('lap_complete') else '—',fmt(r.get('cost')) if r.get('lap_complete') else '—',fmt(r.get('final_width'),6),r.get('lap_complete',False)] for r in entries])
    safety_html=table_files(tables,"safety",["Method","Trials","Completed laps","Constraint violations","Certificate unavailable","Audit errors","SMID exclusions","95% unsafe/unresolved upper"],
        [[NAMES[r['method']],r['trials'],r['laps_completed'],r['constraint_violations'],r['certificate_unavailable'],r['errors'],r['containment_failure_episodes'],fmt(100*r['unsafe_or_unresolved_upper95'])+"%"] for r in safety])
    margins_html=table_files(tables,"safety_margins",["Method","Minimum clearance [m]","Minimum tire margin","Minimum SMID containment"],
        [[NAMES[r['method']],fmt(r['minimum_clearance'],6),fmt(r['minimum_tire_margin'],7),fmt(r['minimum_containment_margin'],9)] for r in safety])
    paired_html=table_files(tables,"paired_improvement",["Comparator","Pairs","Time saved [s]","95% bootstrap interval","Wins / ties / losses"],
        [[NAMES[r['comparator']],r['complete_pairs'],fmt(r['mean_improvement']),f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]",f"{r['proposed_wins']} / {r['ties']} / {r['comparator_wins']}"] for r in paired if r['subgroup']=='all_mu' and r['metric']=='time'])
    table_files(tables,'paired_subgroups',['Comparator','Friction group','Metric','Pairs','Mean improvement','95% interval'],
        [[NAMES[r['comparator']],r['subgroup'],r['metric'],r['complete_pairs'],fmt(r['mean_improvement']),f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"] for r in paired])
    captions=sorted(figures(directory,rows),key=lambda c:c['name'])
    errors=[r for r in rows if r["status"]!="ok"]
    atomic_json(directory/"failures.json",[r for r in rows if r["status"]!="ok" or r.get("metrics",{}).get("constraint_violated") or r.get("metrics",{}).get("certificate_unavailable")])
    completed=sum(r.get("metrics",{}).get("lap_complete",False) for r in rows)
    passive=next(r for r in safety if r['method']=='passive')
    total_frames=sum(r.get("metrics",{}).get("frames",0) for r in rows)
    epsilon=manifest['scenario']['epsilon']
    report_text=f"""# Racing experiment results — {100*epsilon:g}% tolerance

{len(rows)} executed episodes; {completed} completed laps; {len(errors)} audit/runtime errors. {total_frames:,} recorded states were independently replayed. Actual constraint violations and unavailable certificates are separate outcomes in the safety table.

## Protocol

- {manifest['trials']} IID true friction draws from Uniform[0.45,1.05], seed {manifest['seed']}, paired across methods and risk tolerances. Each coefficient stays fixed for the whole episode.
- All methods use the same track, state, sensor, bicycle dynamics, actuator limits and sampled safety test. The racing target is 55 m/s, the centerline backup target 8 m/s.
- Horizon: {manifest['seconds']:g} s or one lap; integration step 0.05 s. Grid and weight sweeps are separate from Monte Carlo inference.
- Quantized yaw acceleration has resolution 0.36 rad/s² and bounded error 0.18 rad/s². Passive runs SMID normally; it contracted its interval in {passive['contraction_episodes']}/{passive['trials']} Monte Carlo episodes. No learning retains the prior.

## Sampled safety certificate

DRS-gatekeeper with beta=0: the nominal probabilistic special case, with no distribution-mismatch protection. The conditional friction law is explicitly assumed uniform over the current SMID interval. Uniformity is a modeling assumption, not a consequence of SMID containment.

Each candidate uses N=1000 IID friction samples, fixed within each rollout, and includes its backup continuation and terminal recovery check. The backup controller is assumed to keep the slow centerline terminal set robustly invariant (alpha=0). Epsilon={epsilon:g}; delta=0.05 per decision. A Bonferroni allocation delta/64 covers at most 64 independently sampled candidate checks, including informative and backup options. Accepted candidates complete all 1000 rollouts and satisfy the exact binomial upper bound <= epsilon. A partial batch can only reject, using the fixed-N failure threshold. No sampled pass is reported as all-parameter certification.

Fresh certification batches are separate from candidate generation. Sample seeds, evaluated counts, failures and upper bounds are saved. The 95% confidence and epsilon limit are per certification decision, not a whole-lap probability guarantee. Repeated replanning is assessed by independent closed-loop experiments. The plant is a discrete model with fixed friction and exact controller states; no physical-vehicle or continuous-time guarantee is asserted.

## Statistical interpretation

One-sided 95% Clopper–Pearson bounds use independent episodes per method, never frames or pooled paired controllers. Observed constraint violations, unavailable certificates and unresolved errors count in the conservative unsafe-or-unresolved endpoint. Zero failures in 100 episodes gives 2.9513%; this empirical bound is distinct from epsilon. It cannot validate a 1% whole-lap failure claim with only 100 episodes.

Lap time and lap cost include completed laps only; completion counts accompany them. Partial failed runs retain their distance, time, cost and failure state in the records. Paired mean differences use 5000 bootstrap resamples of whole paired trials; friction subgroups are descriptive. SMID intervals are bounded-error consistency sets, not confidence intervals.

## Comparators and scope

{('This schema-3 experiment optimizes physical force and steering-rate sequences using local MPPI. Proposed uses a fixed gamma=2 informative objective with paired certification, predicted SMID contraction and an exploration budget. Only the weighted baseline varies its information coefficient. All observed losses and failures remain in the results.' if manifest.get('protocol',2)>=3 else 'These historical schema-2 results use the finite policy library. They are not evidence for the current MPPI implementation.')}

## Files

Figures: vector PDF, SVG and PNG. Tables: CSV and booktabs LaTeX fragments. `all_episodes.csv` includes all outcomes. `records/*.json.gz` stores full states, observations and certification decisions; `runs/*.json` stores metrics and audits. `manifest.json` fixes the cases and numerical source hash. `paper-results.zip` contains the shareable figures, tables and protocol; raw records remain local.

Runtime is simulation throughput under {manifest['environment']['workers']} workers, excluding initial certification and post-run audit, not an isolated planner deadline.

## Reproduce

```bash
uv sync --extra dev
uv run --locked drs-racing experiments --scenario {manifest['scenario']['name']} --trials {manifest['trials']} --seed {manifest['seed']} --seconds {manifest['seconds']:g} --workers 2 --output {directory.name}
uv run --locked drs-racing report --input {directory.name}
```

Reference: [DRS gatekeeper, Algorithm 1](https://arxiv.org/html/2605.12974v3). The only change to the multiple-candidate confidence allocation is conservative Bonferroni in place of the paper's independence-based Sidak correction.
"""
    (directory/"README.md").write_text(report_text)
    cards="".join(f"<article><h2>{c['name'].replace('_',' ').title()}</h2><img loading='lazy' src='figures/{c['name']}.png' alt='{html.escape(c['caption'],quote=True)}'><p>{html.escape(c['caption'])}</p><p><a href='figures/{c['name']}.pdf'>PDF</a> · <a href='figures/{c['name']}.svg'>SVG</a> · <a href='figures/{c['name']}.png'>PNG</a></p></article>" for c in captions)
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dual Gatekeeper — Paper results</title><style>
body{{margin:0;background:#f4f6f5;color:#23372f;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 22px}}h1{{font-size:36px;line-height:1.2;margin:5px 0 16px}}h2{{font-size:21px}}.eyebrow{{text-transform:uppercase;letter-spacing:.15em;font-size:12px}}article,section{{background:white;border:1px solid #d9e3dc;border-radius:12px;padding:24px;margin:22px 0}}img{{width:100%;height:auto}}a{{color:#137258}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}}td,th{{padding:10px;text-align:right;border-bottom:1px solid #e2e9e4}}td:first-child,th:first-child{{text-align:left}}.note{{color:#55695e}}code{{background:#eef2ef;padding:2px 5px}}.badge{{display:inline-block;padding:5px 12px;background:#e3efe8;border-radius:6px;margin:3px}}
</style><main><p class="eyebrow">Autonomous racing / reproducible experiments</p><h1>Scenario {1 if epsilon==.01 else 2}: {100*epsilon:g}% tolerance</h1><p><a href="../index.html">Both scenarios</a> · N = 1,000 per candidate · β = 0 · α = 0 assumed · δ = 0.05 per decision</p><p><span class="badge">{manifest['trials']} independent paired trials</span> <span class="badge">{len(rows)} closed-loop episodes</span> <span class="badge">{completed} completed laps</span> <span class="badge">{len(errors)} errors</span></p><p>True μ is drawn once per episode and kept fixed. All methods share the same dynamics, sensor, track and safety verifier. The deterministic sensor is not randomized between identical runs.</p><p><a href="README.md">Protocol and interpretation</a> · <a href="manifest.json">Manifest</a> · <a href="tables/all_episodes.csv">Every episode (CSV)</a> · <a href="failures.json">Errors</a></p>
<p><a href="paper-results.zip">Download figures, tables and protocol (ZIP)</a></p><section><h2>Monte Carlo performance</h2><p>Mean ± sample standard deviation. Completed-lap metrics; audit and completion counts below.</p>{table_html}<p><a href="tables/performance.tex">LaTeX</a> · <a href="tables/summary_statistics.csv">All statistics</a></p></section>
<section><h2>Observed safety, completion and SMID containment</h2>{safety_html}<p class="note">One-sided 95% exact binomial bounds use episodes per method. Errors are conservatively counted as unresolved failures. They apply to Uniform[0.45,1.05], not arbitrary distributions. These empirical episode bounds are separate from the online candidate tolerance. The candidate check uses 1,000 sampled rollouts, β = 0, and a Bonferroni confidence correction; it does not verify the entire friction interval.</p><p>Passive reduced its interval in {passive['contraction_episodes']}/{passive['trials']} trials. Its estimator is active; this is measured behavior under the shared quantized sensor.</p><p><a href="tables/safety.tex">LaTeX</a> · <a href="tables/safety_statistics.csv">Safety data</a></p></section>
<section><h2>Observed safety margins</h2>{margins_html}<p><a href="tables/safety_margins.tex">LaTeX</a></p></section>
<section><h2>Paired task-performance differences</h2>{paired_html}<p class="note">Positive time saved favors proposed. Confidence intervals resample whole paired trials (5,000 bootstrap draws). All outcomes, including losses, are retained.</p><p><a href="tables/paired_improvement.tex">LaTeX</a> · <a href="tables/paired_statistics.csv">Time, cost and subgroup results</a></p></section>
{cards}<footer>Numerical implementation: <code>{manifest['implementation'][:16]}</code>. Full scope and assumptions: README. No continuous-time or physical-vehicle guarantee is asserted.</footer></main></html>"""
    (directory/"index.html").write_text(page)
    atomic_json(directory/"figures.json",captions)
    with zipfile.ZipFile(directory/'paper-results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            relative=path.relative_to(directory)
            if path.is_file() and relative.parts[0] not in ('runs','records') and path.suffix!='.zip':
                archive.write(path,str(relative))
    return directory/"index.html"
