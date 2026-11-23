r"""Stills of the modifiers (gf_core::mods), for judging by eye, and a contact sheet of them.

    python tools/mods_shots.py [out dir] [w h frames] [only names,..]

Each case is drawn in glass and in clay through the shared GPU queue (D:\glassforms\native\tools\gpu_run.py)
with this copy's program. Pairs: linked against per-axis numbers; the surface lab as a modifier against the
surface studio of the main program (the same stack, `sculpt=`), with the mean difference between them;
exact against baked. The sheet is <out dir>/mods_sheet.png. The stacks are written here only (the program
ships no presets).
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
MAIN_EXE = r'D:\glassforms\native\target\release\glassforms.exe'
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'


def M(kind, **k):
    d = dict(kind=kind)
    for key, v in k.items():
        d[key] = {'v': v[0], 'link': v[1]} if key in ('a', 'b') else v
    return d


CSG = ['trace=csg']
CELLS = {'layers': [{'kind': 'cells', 'scale': 0.14, 'depth': 0.1, 'a': 0.75, 'b': 0.9, 'mode': 'carve'}], 'round': 0.012, 'crumbs': 0.06}

# name: (title, program args, stack)
CASES = {
    'twist': ('twisted block (exact)', ['scene=studio', 'preset=block', 'p.width=0.75', 'p.height=1.9', 'p.depth=0.75'] + CSG, [M('twist', a=([0, 180, 0], False))]),
    'swirl': ('swirled superellipsoid (exact)', ['scene=solids', 'preset=superellipsoid'] + CSG, [M('swirl', a=([0, 200, 0], False), size=0.75)]),
    'bend': ('bent capsule (exact)', ['scene=studio', 'preset=capsule'] + CSG, [M('bend', a=([0, 140, 0], False))]),
    'taper': ('tapered cylinder (exact)', ['scene=studio', 'preset=cylinder'] + CSG, [M('taper', a=([0, -0.6, 0], False))]),
    'gems': ('pear gems mirrored on x, repeated 3 x 2 (exact)', ['scene=gems', 'preset=pear', 'frames=96'] + CSG,
             [M('mirror', a=([1, 0, 0], False)), M('repeat', a=([3, 1, 2], False), b=([3.0, 2.2, 3.8], False))]),
    'shell': ('shelled sphere (exact)', ['scene=studio', 'preset=sphere'] + CSG, [M('shell', size=0.06, mode=1)]),
    'noise': ('noise on Thomas (baked)', ['scene=attractors', 'preset=thomas'], [M('noise', a=([0.025, 0.025, 0.025], True), size=0.22, detail=3)]),
    'lab': ('surface lab on the teapot (baked)', ['scene=models', 'preset=teapot'], None),
    # pairs
    'twist_linked': ('twist linked: 150 on x, y, z', ['scene=studio', 'preset=block', 'p.width=0.75', 'p.height=1.9', 'p.depth=0.75'] + CSG, [M('twist', a=([150, 150, 150], True))]),
    'twist_axis': ('twist per axis: y 150 only', ['scene=studio', 'preset=block', 'p.width=0.75', 'p.height=1.9', 'p.depth=0.75'] + CSG, [M('twist', a=([0, 150, 0], False))]),
    'ripple_linked': ('ripple linked', ['scene=solids', 'preset=superellipsoid'] + CSG, [M('ripple', a=([0.05, 0.05, 0.05], True), size=0.5)]),
    'ripple_axis': ('ripple per axis: x by y only', ['scene=solids', 'preset=superellipsoid'] + CSG, [M('ripple', a=([0.05, 0, 0], False), size=0.5)]),
    'twist_baked': ('twisted block (baked: a volume object)', ['scene=studio', 'preset=block', 'p.width=0.75', 'p.height=1.9', 'p.depth=0.75'], [M('twist', a=([0, 180, 0], False))]),
    'gems_baked': ('pear gems, copies touching (baked)', ['scene=gems', 'preset=pear', 'frames=96'] + CSG,
                   [M('mirror', a=([1, 0, 0], False)), M('repeat', a=([3, 1, 2], False), b=([1.7, 2.2, 2.4], False))]),
}
GRID = ['twist', 'swirl', 'bend', 'taper', 'gems', 'shell', 'noise', 'lab']
PAIRS = [('twist_linked', 'twist_axis'), ('ripple_linked', 'ripple_axis'), ('twist', 'twist_baked'), ('gems', 'gems_baked')]


def shot(out, w, h, frames, name, clay, exe=EXE, tag=''):
    title, args, stack = CASES[name]
    path = os.path.join(out, f'{name}{tag}{"_clay" if clay else ""}.png')
    extra = []
    if stack is not None:
        extra.append('mods=' + json.dumps(stack, separators=(',', ':')))
    if name == 'lab':
        extra.append('sculpt=' + json.dumps(CELLS, separators=(',', ':')))
    fr = next((a.split('=')[1] for a in args if a.startswith('frames=')), str(frames))
    args = [a for a in args if not a.startswith('frames=')]
    cmd = [sys.executable, QUEUE, '--name', 'mods', '--', exe, '--still', path, f'w={w}', f'h={h}', f'frames={fr}'] + args + extra
    if clay:
        cmd.append('clay=1')
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(path)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{')), '{}')
    try:
        info = json.loads(line)
    except ValueError:
        info = {}
    print(name, tag, 'clay' if clay else 'glass', 'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-600:], info.get('mods', ''), info.get('volumeMs', ''), flush=True)
    return (name, tag, clay, path if ok else None, info.get('mods', ''))


def font(size):
    for f in ['consola.ttf', 'arial.ttf']:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile(path, tw, th, label, note=''):
    im = Image.open(path).convert('RGB').resize((tw, th), Image.LANCZOS) if path else Image.new('RGB', (tw, th), (60, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, tw, 22], fill=(0, 0, 0))
    d.text((6, 3), label, fill=(230, 230, 230), font=font(15))
    if note:
        d.rectangle([0, th - 20, tw, th], fill=(0, 0, 0))
        d.text((6, th - 18), note, fill=(170, 170, 170), font=font(13))
    return im


def main(argv):
    out = argv[0] if argv else os.path.join(NATIVE, 'shots', 'mods')
    w, h, frames = (int(argv[1]), int(argv[2]), int(argv[3])) if len(argv) >= 4 else (1920, 1080, 64)
    only = set(argv[4].split(',')) if len(argv) >= 5 else None
    os.makedirs(out, exist_ok=True)
    jobs = []
    names = [n for n in CASES if only is None or n in only]
    for n in names:
        for clay in ([False, True] if n in GRID else [False]):
            jobs.append((n, clay, EXE, ''))
    if 'lab' in names:
        # the old surface studio, drawn by the main program from the same stack
        jobs.append(('lab', False, MAIN_EXE, '_main'))
        jobs.append(('lab', True, MAIN_EXE, '_main'))
    with ThreadPoolExecutor(3) as ex:
        done = list(ex.map(lambda j: shot(out, w, h, frames, j[0], j[1], j[2], j[3]), jobs))
    got = {(n, t, c): (p, m) for n, t, c, p, m in done}
    # the lab against the main program's studio
    labdiff = {}
    for clay in [False, True]:
        a, b = got.get(('lab', '', clay), (None, ''))[0], got.get(('lab', '_main', clay), (None, ''))[0]
        if a and b:
            x = np.asarray(Image.open(a).convert('RGB')).astype(np.float32)
            y = np.asarray(Image.open(b).convert('RGB')).astype(np.float32)
            labdiff[clay] = float(np.abs(x - y).mean())
            print('lab vs main studio', 'clay' if clay else 'glass', 'mean difference %.3f / 255' % labdiff[clay], flush=True)
    if only is not None:
        return
    tw, th = 640, 360
    rows = [(n, [(f'{CASES[n][0]} · glass', got.get((n, '', False), (None, ''))), (f'{CASES[n][0]} · clay', got.get((n, '', True), (None, '')))]) for n in GRID]
    rows.append(('lab_main', [('the main program\'s surface studio · glass (diff %.2f)' % labdiff.get(False, -1), got.get(('lab', '_main', False), (None, ''))),
                              ('the main program\'s surface studio · clay (diff %.2f)' % labdiff.get(True, -1), got.get(('lab', '_main', True), (None, '')))]))
    for a, b in PAIRS:
        rows.append((a, [(CASES[a][0], got.get((a, '', False), (None, ''))), (CASES[b][0], got.get((b, '', False), (None, '')))]))
    sheet = Image.new('RGB', (2 * tw + 6, len(rows) * (th + 6)), (24, 24, 24))
    for r, (_, cells) in enumerate(rows):
        for c, (label, (p, m)) in enumerate(cells):
            sheet.paste(tile(p, tw, th, label, m), (c * (tw + 6), r * (th + 6)))
    path = os.path.join(out, 'mods_sheet.png')
    sheet.save(path)
    print('sheet', path)


if __name__ == '__main__':
    main(sys.argv[1:])
