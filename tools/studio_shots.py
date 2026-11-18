r"""Stills of the surface studio, for judging by eye, and a contact sheet of them.

    python tools/studio_shots.py [out dir] [w h frames] [only shapes,..] [only stacks,..]

Every still goes through the shared GPU queue (tools/gpu_run.py), with this copy's
program. The sheet is <out dir>/studio_sheet.png: a row a shape, a column a stack (the first column is
the shape with no studio).
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_run.py')
SHAPES = [
    ('sphere', ['scene=solids', 'preset=superellipsoid']),
    ('thomas', ['scene=attractors', 'preset=thomas']),
    ('teapot', ['scene=models', 'preset=teapot']),
    ('gem', ['scene=gems', 'preset=cushion']),
    ('goldpot', ['scene=models', 'preset=teapot', 'material=gold', 'studio=metal']),
]
import json

def L(kind, **k):
    return dict(kind=kind, **k)

# Test stacks for judging, written here only (the program ships no presets).
STACK_DEFS = {
    'none': None,
    'cells': {'layers': [L('cells', scale=0.14, depth=0.1, a=0.75, b=0.9, mode='carve')], 'round': 0.012, 'crumbs': 0.06},
    'foam': {'layers': [L('foam', scale=0.3, depth=0.03, a=0.08, b=0.9, mode='both')], 'round': 0.006, 'crumbs': 0.08},
    'marble': {'layers': [L('warp', scale=0.6, depth=0.08, a=3), L('marble', scale=0.25, depth=0.014, a=1.2, b=4, mode='raise')]},
    'hammered': {'layers': [L('hammered', scale=0.09, depth=0.012, a=1.2, b=0.85, mode='carve')]},
    'gyroid': {'layers': [L('gyroid', scale=0.35, depth=0.035, a=0.3, mode='both')], 'round': 0.006, 'crumbs': 0.08},
    'eroded': {'layers': [L('warp', scale=0.5, depth=0.1, a=3), L('noise', scale=0.35, depth=0.035, a=4, b=0.5, mode='raise'),
                          L('cells', scale=0.16, depth=0.09, a=0.75, b=0.9, mode='carve', mask='above')], 'round': 0.008, 'crumbs': 0.06},
    'growth': {'layers': [L('growth', scale=0.1, depth=0.03, a=3, b=0.6, mode='raise'), L('pores', scale=0.05, depth=0.02, a=0.45, b=0.6, mode='carve')],
               'round': 0.004, 'crumbs': 0.04},
}
STACKS = list(STACK_DEFS)


def shot(out, w, h, frames, shape, stack):
    name, args = shape
    path = os.path.join(out, f'{name}_{stack}.png')
    d = STACK_DEFS[stack]
    extra = [] if d is None else ['sculpt=' + json.dumps(d, separators=(',', ':'))]
    cmd = [sys.executable, QUEUE, '--name', 'studio', '--', EXE, '--still', path, f'w={w}', f'h={h}', f'frames={frames}'] + args + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(path)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{')), '')
    ms = ''
    if '"volumeMs"' in line:
        ms = line.split('"volumeMs":')[1].split(',')[0]
    print(name, stack, 'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-400:], ms, flush=True)
    return path if ok else None


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(NATIVE, 'out', 'r5')
    w, h, frames = (argv[2], argv[3], argv[4]) if len(argv) > 4 else ('1920', '1080', '64')
    shapes = [s for s in SHAPES if len(argv) <= 5 or argv[5] == 'all' or s[0] in argv[5].split(',')]
    stacks = [s for s in STACKS if len(argv) <= 6 or s in argv[6].split(',')]
    os.makedirs(out, exist_ok=True)
    jobs = [(sh, st) for sh in shapes for st in stacks]
    with ThreadPoolExecutor(3) as ex:
        paths = list(ex.map(lambda j: shot(out, w, h, frames, *j), jobs))
    from PIL import Image, ImageDraw
    tw, th = 480, 270
    sheet = Image.new('RGB', (tw * len(stacks), (th + 18) * len(shapes)), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    for k, ((sh, st), p) in enumerate(zip(jobs, paths)):
        r, c = shapes.index(sh), stacks.index(st)
        if p:
            sheet.paste(Image.open(p).convert('RGB').resize((tw, th), Image.LANCZOS), (c * tw, r * (th + 18) + 18))
        d.text((c * tw + 6, r * (th + 18) + 3), f'{sh[0]} / {st}', fill=(200, 205, 215))
    sheet.save(os.path.join(out, 'studio_sheet.png'))


if __name__ == '__main__':
    main(sys.argv)
