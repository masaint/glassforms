"""Stills of the scenes the surface pass is judged on, and a sheet of them side by side.

    python tools/surface_shots.py <exe> <out dir> [w h frames] [only]

Each still goes through tools/gpu_run.py. The sheet (sheet.png in the out dir) puts the five in a row.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAMP = ['rig=', 'ambient=0,0,0', 'backdrop=0', 'floor=1', 'elev=25', 'cast=2',
        'lamp1=panel,1.4,1.8,0.6,0.6,0.4,1,1,1,25', 'lamp1.rx=90']
SCENES = [
    ('clay_teapot', ['scene=models', 'preset=teapot', 'clay=1']),
    ('gold_knot', ['scene=knots', 'preset=trefoil', 'material=gold', 'studio=metal']),
    ('milk_teapot', ['scene=models', 'preset=teapot', 'material=milk']),
    ('lamp_twist', ['scene=forms', 'preset=twistedring', 'material=gold'] + LAMP),
    ('lamp_clay', ['scene=forms', 'preset=twistedring', 'clay=1'] + LAMP),
    ('clay_key', ['scene=models', 'preset=teapot', 'clay=1', 'rig=100,30,5,5,8,8,8,0.5:-60,20,20,30,1,1,1.1,0.9', 'ambient=0.005,0.005,0.006']),
    ('milk_key', ['scene=models', 'preset=teapot', 'material=milk', 'rig=100,30,5,5,8,8,8,0.5:-60,20,20,30,1,1,1.1,0.9', 'ambient=0.005,0.005,0.006']),
    ('glass_thomas', ['scene=attractors', 'preset=thomas']),
]


def main(argv):
    exe, out = argv[1], argv[2]
    w, h, frames = (argv[3], argv[4], argv[5]) if len(argv) > 5 else ('960', '720', '64')
    only = argv[6] if len(argv) > 6 else None
    os.makedirs(out, exist_ok=True)
    paths = []
    for name, args in SCENES:
        if only and only not in name:
            continue
        path = os.path.join(out, name + '.png')
        cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'surface', '--', exe, '--still', path,
               f'w={w}', f'h={h}', f'frames={frames}'] + args
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(name, 'FAILED', (r.stderr or r.stdout)[-600:])
            continue
        print(name, 'ok', path, flush=True)
        paths.append(path)
    if not only and paths:
        # (a sheet of the stills in a row; with a second out dir given as SHEET_BEFORE, that one's stills above)
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
