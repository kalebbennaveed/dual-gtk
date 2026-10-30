"""Inline isometric previews of the actual scenario obstacle geometry."""
import html
import math
import posixpath
from pathlib import Path

import yaml

SCENARIO_ROUTES = {'informative_demo': '', 'spatial_learning_course': 'scenario-2', 'slalom_course': 'scenario-3'}


def scenario_links(current='informative_demo'):
    return {name: posixpath.relpath(route or '.', SCENARIO_ROUTES[current] or '.')+'/'
            for name,route in SCENARIO_ROUTES.items()}


def preview_svg(config, label):
    lower, upper = config['lower'], config['upper']
    def project(p):
        return p[0] - .65*p[1], -.26*p[0] - .38*p[1] - 1.9*p[2]
    corners = [project([x,y,z]) for x in (lower[0],upper[0])
               for y in (lower[1],upper[1]) for z in (0,upper[2])]
    xmin, xmax = min(p[0] for p in corners), max(p[0] for p in corners)
    ymin, ymax = min(p[1] for p in corners), max(p[1] for p in corners)
    scale = min(340/(xmax-xmin), 120/(ymax-ymin))
    def point(p):
        x,y = project(p)
        return (200+(x-(xmin+xmax)/2)*scale, 78+(y-(ymin+ymax)/2)*scale)
    def points(ps):
        return ' '.join(f'{x:.2f},{y:.2f}' for x,y in map(point,ps))
    def poly(ps, fill, stroke='#8093ad'):
        return f'<polygon points="{points(ps)}" fill="{fill}" stroke="{stroke}" stroke-width=".8"/>'
    ground = [[lower[0],lower[1],0],[upper[0],lower[1],0],
              [upper[0],upper[1],0],[lower[0],upper[1],0]]
    body = [poly(ground,'#edf3f8','#ccd8e5')]
    for i in range(1,8):
        x = lower[0]+(upper[0]-lower[0])*i/8
        body.append(f'<polyline points="{points([[x,lower[1],0],[x,upper[1],0]])}" fill="none" stroke="#d7e1eb" stroke-width=".6"/>')
    for o in sorted(config['obstacles'], key=lambda o: -o.get('center',o.get('lower'))[1]):
        if o['kind']=='box':
            a,b=o['lower'],o['upper']
            body.extend([poly([[a[0],a[1],a[2]],[b[0],a[1],a[2]],[b[0],a[1],b[2]],[a[0],a[1],b[2]]],'#9fb4cc'),
                         poly([[b[0],a[1],a[2]],[b[0],b[1],a[2]],[b[0],b[1],b[2]],[b[0],a[1],b[2]]],'#bacbdd'),
                         poly([[a[0],a[1],b[2]],[b[0],a[1],b[2]],[b[0],b[1],b[2]],[a[0],b[1],b[2]]],'#d4dfeb')])
        elif o['kind']=='cylinder':
            c,r,h=o['center'],o['radius'],o['height']
            rings=[[[c[0]+r*math.cos(t*math.tau/40),c[1]+r*math.sin(t*math.tau/40),z] for t in range(40)] for z in (c[2]-h/2,c[2]+h/2)]
            # Convex side silhouette plus the top disk.
            from scipy.spatial import ConvexHull
            ps=rings[0]+rings[1]
            hull=ConvexHull([point(p) for p in ps])
            body.append(poly([ps[i] for i in hull.vertices],'#a9bfd4'))
            body.append(poly(rings[1],'#cbd9e7'))
    for pos,text,color in ((config['start'][:3],'Start','#326bd1'),(config['goal'],'Goal','#159c75')):
        x,y=point(pos)
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" stroke="white" stroke-width="1.5"/><text x="{x:.2f}" y="{y-8:.2f}" text-anchor="middle" fill="{color}" font-size="10" font-weight="600">{text}</text>')
    return f'<svg viewBox="0 0 400 156" role="img" aria-label="{html.escape(label)}">'+''.join(body)+'</svg>'


def scenario_previews(current='informative_demo'):
    root=Path(__file__).resolve().parents[1]/'drone_dual_gatekeeper'/'configs'
    links=scenario_links(current)
    cards=[]
    for name,title,description in (
        ('informative_demo','Scenario 1','Narrow passage · two columns, same start and goal height'),
        ('spatial_learning_course','Scenario 2','Spatial course · rise over a wall, descend under a wall, turn around columns'),
        ('slalom_course','Scenario 3','Slalom course · weave around three alternating barriers to a higher goal')):
        config=yaml.safe_load((root/f'{name}.yaml').read_text())
        cards.append(f'<a class="scenario-preview" data-scenario="{name}" href="{links[name]}">{preview_svg(config,description)}<div><strong>{title}</strong><span>{description}</span></div></a>')
    return ''.join(cards)
