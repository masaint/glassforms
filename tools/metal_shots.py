"""Stills of the metal scenes (r4 metal), through tools/gpu_run.py, and a sheet.

    python tools/metal_shots.py <exe> <out dir> [w h frames] [only]

sheet.png in the out dir: the stills in a row. env GLASSFORMS_* passes through.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAMP = ['rig=', 'ambient=0,0,0', 'backdrop=0', 'floor=1', 'elev=25', 'cast=2',
        'lamp1=panel,1.4,1.8,0.6,0.6,0.4,1,1,1,25', 'lamp1.rx=90']
GARDEN = ['hdri=hdri/suburban_garden_4k.exr', 'hdriShow=1']
SPH = ['scene=solids', 'preset=superellipsoid']
SCENES = [
    ('gold_knot', ['scene=knots', 'preset=trefoil', 'material=gold', 'studio=metal']),
    ('gold_teapot', ['scene=models', 'preset=teapot', 'material=gold', 'studio=metal']),
    ('lamp_twist', ['scene=forms', 'preset=twistedring', 'material=gold'] + LAMP),
    ('steel_studio', SPH + ['material=brushed', 'studio=metal']),
    ('steel_garden', SPH + ['material=brushed'] + GARDEN),
    ('sgold_studio', SPH + ['material=satingold', 'studio=metal']),
    ('sgold_garden', SPH + ['material=satingold'] + GARDEN),
    ('steel_teapot', ['scene=models', 'preset=teapot', 'material=brushed', 'studio=metal']),
]


def main(argv):
    exe, out = argv[1], argv[2]
    w, h, frames = (argv[3], argv[4], argv[5]) if len(argv) > 5 else ('720', '720', '64')
    only = argv[6].split(',') if len(argv) > 6 else None
    os.makedirs(out, exist_ok=True)
    paths = []
    for name, args in SCENES:
        if only and name not in only:
            continue
        path = os.path.join(out, name + '.png')
        cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'metal', '--', exe, '--still', path,
               f'w={w}', f'h={h}', f'frames={frames}'] + args
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print(name, 'FAILED', (r.stderr or r.stdout)[-600:])
            continue
        print(name, 'ok', path, flush=True)
        paths.append(path)
    if not only and paths:
        from PIL import Image
        ims = [Image.open(p).convert('RGB') for p in paths]
        sheet = Image.new('RGB', (sum(i.width for i in ims), max(i.height for i in ims)))
        x = 0
        for i in ims:
            sheet.paste(i, (x, 0))
            x += i.width
        sheet.save(os.path.join(out, 'sheet.png'))


if __name__ == '__main__':
    main(sys.argv)
