from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
from .track import Track


def render_video(source,output,seconds):
    from .rendering import PyBulletMirrorRenderer
    result=json.loads(Path(source).read_text());track=Track(result.get("scenario",{}).get("track","barcelona"))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    renderer=PyBulletMirrorRenderer(track,"rgb_array_follow",960,540)
    try:
        with imageio.get_writer(output,fps=20,macro_block_size=2) as writer:
            for f in result["frames"]:
                if f["t"]>seconds: break
                state=dict(x=f["pose"][0],y=f["pose"][1],yaw=f["pose"][2],vx=f["state"][3],
                           steering_angle=f["state"][6],wheel_rotation=f["state"][0]/.265,
                           t=f["t"],mode=f["mode"],mu=f["mu"])
                writer.append_data(renderer.render(state))
    finally: renderer.close()
    print(output.resolve())
