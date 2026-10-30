import json
import subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np

def plan_record(plan,interval):
    return {'coefficients':plan.curve.coefficients.tolist(),'horizon':plan.horizon,'interval':[interval.lower,interval.upper],
            'solver_status':plan.solver_status,'residual':plan.residual,'planning_time':plan.planning_time,'cost':plan.cost,
            'certificate':plan.certificate,'tube_edges':plan.tube.edges.tolist(),'phi':plan.tube.boundary_layer.tolist(),
            'rho':plan.tube.position_radius.tolist(),'drag_forcing':plan.tube.drag_forcing.tolist()}

def candidate_record(row,interval):
    record={k:v for k,v in vars(row).items() if k not in ['plan','diagnostics']}
    record['status']=row.status.value
    if row.plan is not None:record['plan']=plan_record(row.plan,interval)
    if hasattr(row,'diagnostics'):record['diagnostics']=row.diagnostics
    return record

def revision():
    return subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=Path(__file__).parents[2]).stdout.strip()

def save_result(result,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    # Strict JSON: infinity represents absent obstacles/conditioning and is encoded as null.
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        if isinstance(value,(np.floating,float)) and not np.isfinite(value):return None
        if isinstance(value,np.generic):return value.item()
        return value
    p.write_text(json.dumps(clean(result),indent=2,allow_nan=False))
