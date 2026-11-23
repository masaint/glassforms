"""Stills of the materials round (r17): our versions of the MagicaCSG reference situations, and sheets.

    python tools/mats_shots.py <exe> <out dir> [tag] [only,names] [w h frames]

Each scene is a material on a ground plane in a soft studio (dark room, blue-grey floor, a large soft
key from the front left, an overhead box, a rim strip), as in the references. Stills go to
<out>/<tag>/<name>.png. Runs through D:\\glassforms\\native\\tools\\gpu_run.py.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_run.py')

# a soft product studio like the references': charcoal room, blue-grey ground lit by the lights
STUDIO = [
    'rig=140,38,34,30,2.6,2.55,2.45,0.8:0,78,60,50,1.3,1.32,1.38,0.8:-65,18,14,40,1.1,1.12,1.2,0.6:-150,12,30,20,0.25,0.26,0.3,0.8',
    'ambient=0.02,0.022,0.026',
    'backdrop=1', 'backWrap=180', 'backHigh=90', 'backSpread=30',
    'backTopR=0.035', 'backTopG=0.036', 'backTopB=0.04',
    'backMidR=0.05', 'backMidG=0.052', 'backMidB=0.058',
    'backLowR=0.03', 'backLowG=0.032', 'backLowB=0.04',
    'floor=1', 'floorMatch=0', 'floorR=0.36', 'floorG=0.42', 'floorB=0.6', 'floorFade=7',
    'cast=2', 'elev=22', 'tone=1', 'exposure=1', 'envRoom=1',
]
# the references' own light, as near as the rig says it: a grey sky all round and one small soft sun from
# the front left (MagicaCSG's renderer lights by a sky and a sun), a dark room behind, a blue-grey ground
SKY = [
    'rig=135,42,7,7,40,39,37,0.6:0,80,80,60,0.5,0.52,0.56,0.9',
    'ambient=0.16,0.17,0.19',
    'backdrop=1', 'backWrap=180', 'backHigh=90', 'backSpread=30', 'backShow=0',
    'backTopR=0.0', 'backTopG=0.0', 'backTopB=0.0',
    'backMidR=0.0', 'backMidG=0.0', 'backMidB=0.0',
    'backLowR=0.0', 'backLowG=0.0', 'backLowB=0.0',
    'floor=1', 'floorMatch=0', 'floorR=0.30', 'floorG=0.35', 'floorB=0.5', 'floorFade=7',
    'cast=2', 'elev=22', 'tone=1', 'exposure=1', 'envRoom=1',
]
SCENES_SKY = [
    ('sky_iron_gear', ['scene=models', 'preset=gear', 'material=iron'] + SKY),
    ('sky_orange', ['scene=models', 'preset=pawn', 'material=stoneware'] + ['bodyR=0.80', 'bodyG=0.33', 'bodyB=0.045'] + SKY),
    ('sky_satin_blue', ['scene=models', 'preset=teapot', 'material=satin', 'absorb=1.6,0.7,0.12'] + SKY),
    ('sky_copper_mirror', ['scene=knots', 'preset=trefoil', 'material=copper', 'floorGloss=0.5', 'floorMirror=1'] + SKY),
]
ORANGE =['bodyR=0.80', 'bodyG=0.33', 'bodyB=0.045']
SCENES = [
    ('metal_gear', ['scene=models', 'preset=gear', 'material=iron'] + STUDIO),
    ('metal_chrome', ['scene=models', 'preset=pawn', 'material=chrome'] + STUDIO),
    ('orange_glazed', ['scene=models', 'preset=pawn', 'material=stoneware'] + ORANGE + STUDIO),
    ('orange_lacquer', ['scene=models', 'preset=teapot', 'material=lacquer'] + ORANGE + STUDIO),
    ('frost_blue', ['scene=models', 'preset=teapot', 'material=frosted', 'absorb=1.6,0.7,0.12'] + STUDIO),
    ('satin_blue', ['scene=models', 'preset=teapot', 'material=satin', 'absorb=1.6,0.7,0.12'] + STUDIO),
    ('copper_knot', ['scene=knots', 'preset=trefoil', 'material=copper'] + STUDIO),
    ('copper_brushed', ['scene=knots', 'preset=trefoil', 'material=copper', 'brush=1'] + STUDIO),
    ('sgold', ['scene=solids', 'preset=superellipsoid', 'material=satingold'] + STUDIO),
]


def main(argv):
    exe, out = argv[1], argv[2]
    tag = argv[3] if len(argv) > 3 else 'now'
    only = argv[4].split(',') if len(argv) > 4 and argv[4] not in ('', 'all') else None
    w, h, frames = (argv[5], argv[6], argv[7]) if len(argv) > 7 else ('960', '720', '96')
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    extra = [a for a in os.environ.get('MATS_EXTRA', '').split(' ') if a]
    for name, args in SCENES + SCENES_SKY:
        if only and name not in only:
            continue
        path = os.path.join(d, name + '.png')
        cmd = [sys.executable, GPU_RUN, '--name', 'mats', '--', exe, '--still', path,
               f'w={w}', f'h={h}', f'frames={frames}'] + args + extra
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print(name, 'FAILED', (r.stderr or r.stdout)[-800:], flush=True)
            continue
        print(name, 'ok', path, flush=True)


if __name__ == '__main__':
    main(sys.argv)
