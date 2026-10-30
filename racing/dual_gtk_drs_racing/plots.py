"""Scientific plots from recorded plant states; no reconstructed trajectories."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(source,output):
    names={"proposed":"Proposed","passive":"Passive","weighted":"Weighted information","robust":"No learning"}
    colors={"proposed":"#16936b","passive":"#518ad0","weighted":"#ca9640","robust":"#ad768a"}
    records={k:json.loads((Path(source)/f"{k}.json").read_text()) for k in names if (Path(source)/f"{k}.json").exists()}
    if not records: raise ValueError("No simulation records found")
    if len({r["implementation"] for r in records.values()})!=1:
        raise ValueError("Records were produced by different implementations")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for k,r in records.items():
        f=r["frames"];t=np.array([a["t"] for a in f]);x=np.array([a["state"] for a in f]);mu=np.array([a["mu"] for a in f])
        style=dict(color=colors[k],label=names[k],linewidth=1.8,linestyle="--" if k=="weighted" else "-")
        ax[0,0].plot(t,mu[:,0],**style)
        ax[0,0].plot(t,mu[:,1],**{**style,'label':None})
        ax[0,0].fill_between(t,mu[:,0],mu[:,1],color=colors[k],alpha=.08)
        ax[0,1].plot(t,np.maximum(mu[:,1]-mu[:,0],1e-10),**style)
        ax[0,2].plot(t,x[:,0]-x[0,0],**style)
        ax[1,0].plot(t,x[:,3],**style)
        ax[1,1].plot(t,[a["cost"] for a in f],**style)
        ax[1,2].plot(t,[a["clearance"] for a in f],**style)
    for truth in sorted({r['summary']['true_mu'] for r in records.values()}):
        ax[0,0].axhline(truth,color='#222222',linestyle='--',label=f'True μ = {truth:g}')
    ax[0,0].set(ylabel="Tire coefficient μ",title="SMID bounds and truth")
    ax[0,1].set(yscale="log",ylabel="Friction interval width",title="Parameter learning")
    ax[0,2].set(ylabel="Distance along track [m]",title="Mission progress")
    ax[1,0].set(ylabel="Longitudinal speed [m/s]",title="Executed speed")
    ax[1,1].set(ylabel="Integrated mission cost",title="Executed cost")
    ax[1,2].set(ylabel="Footprint clearance [m]",title="Track clearance")
    for a in ax.flat:a.set_xlabel("Time [s]");a.grid(alpha=.15)
    ax[0,0].legend(loc="upper right",frameon=False,fontsize=8)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)
    return output.resolve()
