r"""Lattice forms built from primitives with strokes and pattern modifiers, beside the reference pictures they
answer, for judging by eye.

    python tools/pattern_shots.py [out dir] [w h frames] [only names,..]

Every still goes through the shared GPU queue (tools/gpu_run.py). The sheet is
<out dir>/pattern_sheet.png: each build in clay and in glass, with its reference at the left. The builds are
written here only (the program ships no presets).
"""
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = os.path.join(HERE, 'gpu_run.py')
REFS = os.path.join(os.path.dirname(NATIVE), 'docs', 'refs', 'magicacsg')


def J(v):
    return json.dumps(v, separators=(',', ':'))


def part(family, preset, values=None, pos=(0, 0, 0), rot=(0, 0, 0), size=1.0, op='union', smooth=0.0, strokes=None):
    p = dict(family=family, preset=preset, values=values or {}, place=dict(pos=list(pos), rot=list(rot), size=size), op=op, smooth=smooth)
    if strokes:
        s = dict(shell=0, hole=0, round=0, cone=0, star=0, points=5, mirror=[False, False, False])
        s.update(strokes)
        p['strokes'] = s
    return p


FRAME = dict(width=2.0, height=2.0, rim=0.17, depth=0.3, corner=0.0)
# A cone of lattice cells: the motif is a square frame (the cell's border: neighbours share it, so the cells
# join into one lattice), a diamond frame inside it and a small ring at its middle; wrapped over a cone, fewer
# cells to a ring as it narrows, so every cell keeps its size. A collar of a starred ring at the top.
CONE = (['scene=studio', 'preset=rectframe', 'trace=csg', 'rx=25'] + [f'p.{k}={v}' for k, v in FRAME.items()] + [
    'csgJson=' + J([
        part('studio', 'rectframe', FRAME, rot=(0, 0, 45), size=0.6),
        part('studio', 'washer', dict(outer=1.0, hole=0.55, height=0.3), rot=(90, 0, 0), size=0.2),
    ]),
    'mods=' + J([dict(kind='wrap', mode=1, a=dict(v=[1.15, 1.7, 0.42]), size=0.17)]),
])
# A dome of three-armed stars: the motif is a ring with three bars meeting at a small holed hub inside it, standing
# on the form on its own y; wrapped over a sphere from the top down, fewer to a ring toward the pole.
DOME = (['scene=studio', 'preset=washer', 'trace=csg', 'p.outer=1.2', 'p.hole=1.0', 'p.height=0.3', 'rx=35',
         'csgJson=' + J([
             part('studio', 'bar', dict(length=2.2, height=0.2, depth=0.3), pos=(0.55 * math.cos(math.radians(a)), 0, -0.55 * math.sin(math.radians(a))), rot=(0, a, 0), size=0.44)
             for a in (90, 210, 330)
         ] + [part('studio', 'cylinder', dict(radius=0.5, height=0.3), size=0.3, strokes=dict(hole=0.25))]),
         'mods=' + J([dict(kind='wrap', mode=2, a=dict(v=[1.2, 0.95, 0.0]), size=0.21, flip=True)]),
         ])
# The polar repeat with each slice mirrored, and grid rows offset: a ring of ribs, and a staggered wall.
RING = ['scene=studio', 'preset=wedge', 'trace=csg', 'rx=35',
        'strokes=' + J(dict(shell=0, hole=0, round=0, cone=0.5, star=0, points=5, mirror=[False, False, False])),
        'mods=' + J([dict(kind='repeat', mode=1, a=dict(v=[8, 1, 1]), detail=1.6, flip=True)])]
WALL = ['scene=studio', 'preset=block', 'trace=csg', 'p.width=0.9', 'p.height=0.45', 'p.depth=0.3',
        'mods=' + J([dict(kind='repeat', a=dict(v=[4, 5, 1]), b=dict(v=[0.95, 0.48, 1.0]), stagger=0.5)])]
LATHE = ['scene=studio', 'preset=bar', 'trace=csg', 'p.length=1.4', 'p.height=0.16', 'p.depth=0.16',
         'csgJson=' + J([part('studio', 'sphere', {}, pos=(0.5, 0.35, 0), size=0.25), part('studio', 'sphere', {}, pos=(0.5, -0.35, 0), size=0.25)]),
         'mods=' + J([dict(kind='revolve', size=0.35, tilt=90, turn=90)])]

CASES = {
    'cone': ('cone of lattice cells (wrap: cone)', CONE, 'pattern_31.png'),
    'dome': ('dome of three-armed stars (wrap: sphere)', DOME, 'pattern_33.png'),
    'cone_volume': ('the same cone traced from its volume (parts traced: volume)', CONE + ['csgCoarse=1'], 'pattern_31.png'),
    'ring': ('ring of coned wedges, slices mirrored (repeat: ring)', RING, None),
    'wall': ('staggered wall (repeat: grid, rows offset)', WALL, None),
    'lathe': ('lathe: a bar and two balls revolved', LATHE, None),
}


def shot(out, w, h, frames, name, clay):
    title, args, _ = CASES[name]
    path = os.path.join(out, f'{name}{"_clay" if clay else ""}.png')
    cmd = [sys.executable, QUEUE, '--name', 'patterns', '--', EXE, '--still', path, f'w={w}', f'h={h}', f'frames={frames}'] + args
    if clay:
        cmd.append('clay=1')
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(path)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{')), '{}')
    try:
        info = json.loads(line)
    except ValueError:
        info = {}
    print(name, 'clay' if clay else 'glass', 'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-600:], info.get('mods', ''), info.get('volumeMs', ''), flush=True)
    return (name, clay, path if ok else None, info.get('mods', ''))


def font(size):
    for f in ['consola.ttf', 'arial.ttf']:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile(path, tw, th, label, note=''):
    if path:
        im = Image.open(path).convert('RGB')
        im.thumbnail((tw, th), Image.LANCZOS)
        bg = Image.new('RGB', (tw, th), (12, 12, 14))
        bg.paste(im, ((tw - im.size[0]) // 2, (th - im.size[1]) // 2))
        im = bg
    else:
        im = Image.new('RGB', (tw, th), (40, 40, 44))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, tw, 22], fill=(0, 0, 0))
    d.text((6, 3), label, fill=(230, 230, 230), font=font(15))
    if note:
        d.rectangle([0, th - 20, tw, th], fill=(0, 0, 0))
        d.text((6, th - 18), note, fill=(170, 170, 170), font=font(13))
    return im


def main(argv):
    out = argv[0] if argv else os.path.join(NATIVE, 'shots', 'patterns')
    w, h, frames = (int(argv[1]), int(argv[2]), int(argv[3])) if len(argv) >= 4 else (1920, 1080, 96)
    only = set(argv[4].split(',')) if len(argv) >= 5 else None
    os.makedirs(out, exist_ok=True)
    jobs = [(n, c) for n in CASES if only is None or n in only for c in (True, False)]
    with ThreadPoolExecutor(3) as ex:
        done = list(ex.map(lambda j: shot(out, w, h, frames, j[0], j[1]), jobs))
    got = {(n, c): (p, m) for n, c, p, m in done}
    tw, th = 640, 360
    rows = [n for n in CASES if (n, True) in got]
    sheet = Image.new('RGB', (3 * tw + 12, len(rows) * (th + 6)), (24, 24, 24))
    for r, n in enumerate(rows):
        title, _, ref = CASES[n]
        cells = [(f'reference: {ref}' if ref else '', os.path.join(REFS, ref) if ref else None, ''),
                 (f'{title} · clay', got[(n, True)][0], got[(n, True)][1]),
                 (f'{title} · glass', got.get((n, False), (None, ''))[0], got.get((n, False), (None, ''))[1])]
        for c, (label, p, m) in enumerate(cells):
            sheet.paste(tile(p, tw, th, label, m), (c * (tw + 6), r * (th + 6)))
    path = os.path.join(out, 'pattern_sheet.png')
    sheet.save(path)
    print('sheet', path)


if __name__ == '__main__':
    main(sys.argv[1:])
