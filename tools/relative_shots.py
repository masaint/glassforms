r"""The same studio stacks with sizes as lengths (absolute) and as shares of the shape's size (relative),
on a fat solid, a thin attractor, a tree and the teapot, and a contact sheet of them.

    python tools/relative_shots.py [out dir] [w h frames]

Every still goes through the shared GPU queue (D:\glassforms\native\tools\gpu_run.py), with this copy's
program. The relative stacks are the absolute ones with every size divided by SIZE_REF (0.6, core
sculpt.rs), so that on a shape of that size the two are the same. Each still's line says the shape's size
the program measured (studioSize).
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
SIZE_REF = 0.6
SHAPES = [
    ('superellipsoid', ['scene=solids', 'preset=superellipsoid', 'clay=1']),
    ('thomas', ['scene=attractors', 'preset=thomas', 'clay=1']),
    ('tree', ['scene=trees', 'preset=tree', 'clay=1']),
    ('teapot', ['scene=models', 'preset=teapot', 'clay=1']),
]
STACKS = {
    'cells': {'layers': [dict(kind='cells', scale=0.14, depth=0.1, a=0.75, b=0.9, mode='carve')], 'round': 0.012, 'crumbs': 0.06},
    'foam': {'layers': [dict(kind='foam', scale=0.3, depth=0.03, a=0.08, b=0.9, mode='both')], 'round': 0.006, 'crumbs': 0.08},
    'hammered': {'layers': [dict(kind='hammered', scale=0.09, depth=0.012, a=1.2, b=0.85, mode='carve')]},
}


def relative(stack):
    s = json.loads(json.dumps(stack))
    for l in s['layers']:
        for k in ('scale', 'depth', 'soft'):
            if k in l:
                l[k] /= SIZE_REF
        l.setdefault('soft', 0.01 / SIZE_REF)
        l['relative'] = True
    for k in ('round', 'wall', 'crumbs'):
        if k in s:
            s[k] /= SIZE_REF
    s['relative'] = True
    return s


def shot(out, w, h, frames, shape, stack, how):
    name, args = shape
    path = os.path.join(out, f'{name}_{stack}_{how}.png')
    d = STACKS[stack] if how == 'absolute' else relative(STACKS[stack])
    cmd = [sys.executable, QUEUE, '--name', 'relative', '--', EXE, '--still', path, f'w={w}', f'h={h}', f'frames={frames}'] + args + ['sculpt=' + json.dumps(d, separators=(',', ':'))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(path)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{')), '{}')
    size = json.loads(line).get('studioSize', 0) if ok else 0
    print(f'{name:16} {stack:9} {how:9}', 'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-300:], f'size {size}', flush=True)
    return path if ok else None


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(NATIVE, 'out', 'relative')
    w, h, frames = (argv[2], argv[3], argv[4]) if len(argv) > 4 else ('960', '540', '32')
    os.makedirs(out, exist_ok=True)
    cols = [(s, how) for s in STACKS for how in ('absolute', 'relative')]
    jobs = [(sh, s, how) for sh in SHAPES for (s, how) in cols]
    with ThreadPoolExecutor(3) as ex:
        paths = list(ex.map(lambda j: shot(out, w, h, frames, *j), jobs))
    from PIL import Image, ImageDraw
    tw, th = 400, 225
    sheet = Image.new('RGB', (tw * len(cols), (th + 18) * len(SHAPES)), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    for (sh, s, how), p in zip(jobs, paths):
        r, c = SHAPES.index(sh), cols.index((s, how))
        if p:
            sheet.paste(Image.open(p).convert('RGB').resize((tw, th), Image.LANCZOS), (c * tw, r * (th + 18) + 18))
        d.text((c * tw + 6, r * (th + 18) + 3), f'{sh[0]} / {s} / {how}', fill=(200, 205, 215))
    sheet.save(os.path.join(out, 'relative_sheet.png'))


if __name__ == '__main__':
    main(sys.argv)
