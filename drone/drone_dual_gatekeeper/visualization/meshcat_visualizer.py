"""MeshCat conventions from fastpathplanning/examples/uav* and RoboSim Visualization.
Swept convex-hull geometry comes from the same propagated tube used by the planner.
Offline static_html retains the animation and can be opened without Python.
"""
from pathlib import Path
import numpy as np
import json
import meshcat
from meshcat import geometry as geo
from meshcat.animation import Animation
from meshcat.transformations import translation_matrix,rotation_matrix
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).parents[2]/".mpl-cache"))
from matplotlib.textpath import TextPath
from scipy.spatial import ConvexHull
from itertools import product
from ..methods.registry import METHODS
from ..config import ControlConfig
from ..planning.curve import Curve
from ..control.tube_propagation import DynamicTube

COLORS={'backup':0x2277dd,'unsafe':0xdd3333,'budget_infeasible':0xff9900,'feasible':0x22bbbb,'committed':0x22aa44,'generated':0x888888,'actual':0xbb22cc}

def material(color,opacity=1): return geo.MeshLambertMaterial(color=color,opacity=opacity,transparent=opacity<1)

def box(vis,path,lower,upper,color,opacity=1):
    lo=np.array(lower);hi=np.array(upper);vis[path].set_object(geo.Box(hi-lo),material(color,opacity))
    vis[path].set_transform(translation_matrix((lo+hi)/2))

def line(vis,path,points,color):
    vis[path].set_object(geo.Line(geo.PointsGeometry(np.asarray(points).T),geo.LineBasicMaterial(color=color,linewidth=3)))

def text(vis,path,label,position,size=.45):
    polygons=TextPath((0,0),label,size=size).to_polygons()
    vertices=[]
    for points in polygons:
        for a,b in zip(points[:-1],points[1:]):vertices.extend([np.r_[a,0]+position,np.r_[b,0]+position])
    vis[path].set_object(geo.LineSegments(geo.PointsGeometry(np.array(vertices).T),geo.LineBasicMaterial(color=0x111111,linewidth=2)))

def environment(vis,scenario,offset,label):
    lo=np.array(scenario['lower']);hi=np.array(scenario['upper']);off=np.array(offset)
    box(vis,'ground',np.r_[lo[:2],-.04]+off,np.r_[hi[:2],-.02]+off,0xdddddd)
    for i,o in enumerate(scenario['obstacles']):
        node=vis[f'obstacles/{i}']
        if o['kind']=='box':box(vis,f'obstacles/{i}',np.array(o['lower'])+off,np.array(o['upper'])+off,0x888888)
        elif o['kind']=='sphere':
            node.set_object(geo.Sphere(o['radius']),material(0x888888));node.set_transform(translation_matrix(np.array(o['center'])+off))
        else:
            node.set_object(geo.Cylinder(o['height'],o['radius']),material(0x888888))
            node.set_transform(translation_matrix(np.array(o['center'])+off)@rotation_matrix(np.pi/2,[1,0,0]))
    for name,point,color in [('start',scenario['start'][:3],0x22aa44),('goal',scenario['goal'],0xffbb22)]:
        vis[name].set_object(geo.Sphere(.16),material(color));vis[name].set_transform(translation_matrix(np.array(point)+off))
    text(vis,'label',label,np.array([lo[0],hi[1]+.2,0])+off)
    box(vis,'drone/body',[-.045,-.025,-.012],[.045,.025,.012],0xcc3333)
    for i,(x,y) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):
        vis[f'drone/rotor{i}'].set_object(geo.Sphere(.02),material(0x333333))
        vis[f'drone/rotor{i}'].set_transform(translation_matrix([x*.04,y*.04,.015]))

def reconstruct(record,config):
    curve=Curve(record['coefficients'],record['horizon'],config.subdivisions)
    tube=DynamicTube(np.array(record['tube_edges']),np.array(record['drag_forcing']),np.array(record['phi']),np.array(record['rho']),config.alpha,config.sliding_lambda,config.disturbance_bound,config.eta)
    return curve,tube

def show_plan(vis,path,record,config,offset,color,tubes):
    curve,tube=reconstruct(record,config);ts=np.linspace(0,curve.horizon,100)
    line(vis,path+'/centerline',curve.value(ts)+offset,color)
    if tubes:
        # The convex hull plus the propagated error box covers a full spline
        # cell without losing correlation between its position coordinates.
        for i in range(len(curve.edges)-1):
            _,rho,_=tube.cell_envelope(i)
            corners=np.array(list(product([-1.,1.],repeat=3)))*rho
            points=(curve.bernstein_coefficients(i)[:,None,:]+corners[None,:,:]).reshape(-1,3)+offset
            hull=ConvexHull(points);faces=hull.simplices.copy()
            for j,face in enumerate(faces):
                a,b,c=points[face]
                if np.dot(np.cross(b-a,c-a),hull.equations[j,:3])<0:faces[j]=face[[0,2,1]]
            vis[f'{path}/tube/{i}'].set_object(geo.TriangularMeshGeometry(points,faces),material(color,.1))

def visualize(results,output,*,comparison=False,overlay=False,tubes=True,candidates=True):
    vis=meshcat.Visualizer();vis['/Background'].set_property('visible',False)
    animation=Animation(default_framerate=20);end=max(r['samples'][-1]['time'] for r in results)
    extent=np.array(results[0]['scenario']['upper'])-np.array(results[0]['scenario']['lower'])
    offsets=[np.array([i%2*(extent[0]+3),i//2*(extent[1]+4),0]) if comparison and not overlay else np.zeros(3) for i in range(len(results))]
    for panel,(result,off) in enumerate(zip(results,offsets)):
        group=vis[f'panel_{panel}'];c=ControlConfig(**result['config']);scenario=result['scenario']
        if not overlay or panel==0:environment(group,scenario,off,METHODS[result['method']].label)
        else:environment(group,{**scenario,'obstacles':[]},off,METHODS[result['method']].label)
        samples=result['samples'];actual=np.array([s['state'][:3] for s in samples]);nominal=np.array([s['nominal'][:3] for s in samples]);ts=np.array([s['time'] for s in samples])
        line(group,'actual',actual+off,[0xbb22cc,0x111111,0xcc8800,0x22aa44][panel%4]);line(group,'nominal',nominal+off,0x2255bb)
        for epoch in result['epochs']:
            path=f"epoch_{epoch['index']}";start=epoch['time'];finish=start+epoch['duration']
            show_plan(group,path+'/backup',epoch['backup'],c,off,COLORS['backup'],tubes)
            show_plan(group,path+'/committed',epoch['committed'],c,off,0x22aa44,tubes)
            comparison_record=result.get('learning_comparison')
            if comparison_record and abs(comparison_record['time']-start)<1e-6:
                show_plan(group,path+'/prior_uncertainty_backup',comparison_record['prior_backup'],c,off,0x777777,False)
            if candidates:
                for j,can in enumerate(epoch['candidates']):
                    if 'plan' in can:show_plan(group,f'{path}/candidate_{j}',can['plan'],c,off,COLORS.get(can['status'],0x888888),False)
            group[path].set_property('visible',False)
            with animation.at_frame(group,int(round(start*20))) as frame:frame[path].set_property('visible','boolean',True)
            with animation.at_frame(group,int(round(finish*20))) as frame:frame[path].set_property('visible','boolean',False)
            print(f"Replay {result['method']} t={start:.2f} epoch={epoch['index']} Theta={epoch['interval_before']} budget={epoch['exploration_spent']:.3f}/{epoch['budget']:.3f}",flush=True)
        for tick,t in enumerate(np.arange(0,end+.05,.05)):
            point=np.array([np.interp(t,ts,actual[:,j]) for j in range(3)])
            with animation.at_frame(group,tick) as frame:frame['drone'].set_transform(translation_matrix(point+off))
    # Center the default orbit frame and fit all translated comparison panels.
    lower=np.array(results[0]['scenario']['lower'])+np.min(offsets,axis=0)
    upper=np.array(results[0]['scenario']['upper'])+np.max(offsets,axis=0)
    center=(lower+upper)/2;center[2]=0
    span=max(upper[0]-lower[0],upper[1]-lower[1],4.)
    vis['/Cameras/default'].set_transform(translation_matrix(center))
    vis['/Cameras/default/rotated'].set_object(geo.PerspectiveCamera(fov=75,far=2000))
    vis['/Cameras/default/rotated/<object>'].set_transform(translation_matrix([span*.4,span*.75,span*.4]))
    # Open paused at the first epoch so the complete plan is available to inspect.
    for panel,result in enumerate(results):
        if result['epochs']:vis[f'panel_{panel}/epoch_0'].set_property('visible',True)
    vis.set_animation(animation,play=False,repetitions=1)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    html=vis.static_html()
    focus_index=next((i for i,r in enumerate(results) if r['method']=='proposed'),0)
    focus=results[focus_index];early_end=focus['epochs'][0]['duration']
    early=np.array([v['state'][:3] for v in focus['samples'] if v['time']<=early_end+1e-8])
    early_center=(early.min(axis=0)+early.max(axis=0))/2+offsets[focus_index];early_center[2]=0
    early_span=max(float(np.ptp(early[:,:2],axis=0).max())+3.,5.)
    auto_focus=len(results)==1 and focus['epochs'][0]['informative_committed']
    legend='<div style="position:fixed;bottom:12px;left:12px;max-width:430px;z-index:20;background:#ffffffe8;padding:8px;font:12px sans-serif">Constant bandwidth dynamic tube MPC<br>Blue: backup; green: committed; red: rejected; orange: budget rejected<br>Tube hulls: nominal sweep + propagated position error.<br>Safety also includes the physical drone radius.<br>Gray: prior-uncertainty backup at the learning replan.</div>'
    timeline=[{'label':METHODS[r['method']].label,
               'epochs':[{'time':e['time'],'index':e['index'],'interval':e['interval_before'],'mode':('informative' if e['informative_committed'] else ('weighted' if r['method']=='weighted' else 'backup'))} for e in r['epochs']],
               'samples':[[v['time'],v['mission_cost']] for v in r['samples']],
               'final_interval':r['epochs'][-1]['interval_after'],
               'learning':({k:r['learning_comparison'][k] for k in ['time','prior_cost','learned_cost','prior_horizon','learned_horizon']} if 'learning_comparison' in r else None)} for r in results]
    legend=legend.replace('</div>', '''<br><button onclick="viewer.animator.play()">Play</button> <button onclick="viewer.animator.pause()">Pause</button> <button onclick="viewer.animator.reset()">Reset</button><button onclick="replayCameraMode='early';replayCameraPhase=null">Early motion</button> <button onclick="replayCameraMode='full';replayCameraPhase=null">Full mission</button><div id="run-status"></div></div>''')
    status_script="""<script>
const replayRows=TIMELINE;
setInterval(()=>{
  if(typeof viewer==='undefined')return;
  const t=viewer.animator.time;
  document.getElementById('run-status').textContent=replayRows.map(r=>{
    let e=r.epochs[0];for(const row of r.epochs)if(row.time<=t+1e-7)e=row;
    let cost=0;for(const row of r.samples)if(row[0]<=t+1e-7)cost=row[1];
    const interval=t>=r.samples[r.samples.length-1][0]?r.final_interval:e.interval;
    const current=`${r.label}: t=${t.toFixed(1)} s, epoch ${e.index} (${e.mode}), Cd=[${interval.map(x=>x.toFixed(4)).join(', ')}], actual cost=${cost.toFixed(2)}`;
    if(r.learning && t>=r.learning.time){const l=r.learning;return current+` — Same-state backup at ${l.time}s: planned cost ${l.prior_cost.toFixed(1)} → ${l.learned_cost.toFixed(1)}, horizon ${l.prior_horizon.toFixed(1)} → ${l.learned_horizon.toFixed(1)}s`;}
    return current;
  }).join('\\n');
  document.getElementById('run-status').style.whiteSpace='pre-line';
},150);
</script>""".replace('TIMELINE',json.dumps(timeline))
    # MeshCat's static export fetches saved commands concurrently. Wait until
    # they settle before framing the camera, so camera object replacement cannot
    # discard an earlier transform on page load.
    html=html.replace('var viewer = new MeshCat.Viewer(document.getElementById("meshcat-pane"));',
                      'var viewer = new MeshCat.Viewer(document.getElementById("meshcat-pane")); const replayLoads=[];')
    html=html.replace('fetch("data:', 'replayLoads.push(fetch("data:')
    html=html.replace('.then(buffer => viewer.handle_command_bytearray(new Uint8Array(buffer)));',
                      '.then(buffer => viewer.handle_command_bytearray(new Uint8Array(buffer))));')
    camera_script="""<script>
let replayCameraMode=AUTO?'auto':'full', replayCameraPhase=null, replayCameraReady=false;
const replayCameraViews={full:{center:CENTER,position:POSITION},early:{center:EARLYCENTER,position:EARLYPOSITION}};
function frameReplayCamera(){
 if(!replayCameraReady)return;
 const phase=replayCameraMode==='auto'?(viewer.animator.time<EARLYEND?'early':'full'):replayCameraMode;
 if(phase===replayCameraPhase)return;
 const v=replayCameraViews[phase];
 viewer.set_transform(['Cameras','default'],v.center);
 const position=v.position.slice();const rect=viewer.dom_element.getBoundingClientRect();
 const fit=Math.max(1,1.1/(rect.width/Math.max(rect.height,1)));
 for(const i of [12,13,14])position[i]*=fit;
 viewer.set_transform(['Cameras','default','rotated','<object>'],position);
 viewer.controls.target.set(0,0,0);viewer.controls.update();viewer.set_dirty();replayCameraPhase=phase;
}
Promise.all(replayLoads).then(()=>{replayCameraReady=true;frameReplayCamera();});
setInterval(frameReplayCamera,150);
window.addEventListener('resize',()=>{replayCameraPhase=null;frameReplayCamera();});
</script>"""
    for key,value in {'EARLYCENTER':translation_matrix(early_center).flatten(order='F').tolist(),
                      'EARLYPOSITION':translation_matrix([early_span*.4,early_span*.75,early_span*.4]).flatten(order='F').tolist(),
                      'CENTER':translation_matrix(center).flatten(order='F').tolist(),
                      'POSITION':translation_matrix([span*.4,span*.75,span*.4]).flatten(order='F').tolist(),
                      'EARLYEND':early_end,'AUTO':auto_focus}.items():camera_script=camera_script.replace(key,json.dumps(value))
    html=html.replace('<body>','<body>'+legend,1).replace('</body>',camera_script+status_script+'</body>');output.write_text(html)
    print('MeshCat:',vis.url(), 'Offline replay:',output.resolve(),flush=True)
    return vis
