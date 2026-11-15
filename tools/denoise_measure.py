"""How grainy each noisy material or mode is, in grey levels against the same view settled for many frames.

usage: python tools/denoise_measure.py [--exe path] [--out dir] [--size WxH] [--ref N] [--frames 1,8,32,128]
                                        [--turn N] [--extra "name=value ..."] [case ...]

For each case (all when none is named) it draws, through the GPU queue, one job per picture:
  ref        the view settled over --ref frames, from other rays (seed=), so it shares no samples with the rest
  still-N    the same view after N frames (1, 8, 32, 128 by default)
  turn       the last frame of a view turned for --turn frames (what a moving view shows)
  turnref    that last pose settled over --ref frames
and prints, for each picture against its reference, in grey levels of 255 on the picture as it is shown:
  mean       the mean absolute difference over every channel of every pixel
  blurred    the same after both are blurred over 5 x 5 pixels (what is left is not grain but a difference
             of level or shape: the part a denoiser cannot be blamed for, or a blur it has caused)
  spots      the share of pixels more than 24 levels apart in any channel (speckle and fireflies), percent
The numbers go to <out>/measure.json as well. --extra is laid on every command line (e.g. "denoise=1"),
--refextra on the references only (e.g. "denoise=0"); --reuse DIR takes references already made there.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)

# The noisy settings, each on a shape it is meant for. (Names are those of the program's command line.)
CASES = {
    'plain':      ['scene=surfaces', 'preset=gyroid'],
    'truemilk':   ['scene=surfaces', 'preset=gyroid', 'material=truemilk'],
    'truejade':   ['scene=solids', 'preset=superellipsoid', 'material=truejade'],
    'roughref':   ['scene=surfaces', 'preset=gyroid', 'material=frosted', 'roughRef=1'],
    'satingold':  ['scene=surfaces', 'preset=gyroid', 'material=satingold'],
    'frosted':    ['scene=surfaces', 'preset=gyroid', 'material=frosted'],
    'frost06':    ['scene=surfaces', 'preset=gyroid', 'rough=0.6'],
    'jade':       ['scene=surfaces', 'preset=gyroid', 'material=jade'],
    'gem':        ['scene=gems', 'preset=brilliant'],
    'lens':       ['dof=0', 'aperture=0.25'],
    'lensafter':  ['aperture=0.25'],
    'thomas':     [],
    'gem4':       ['scene=gems', 'preset=emerald'],
    'milkthomas': ['material=truemilk'],
    'jadegem':    ['scene=gems', 'preset=oval', 'material=truejade'],
    'caustics':   ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'cast=2',
                   'studio=lavender', 'floor=1', 'elev=20'],
    'aventurine': ['scene=solids', 'preset=superellipsoid', 'material=aventurine'],
    'film':       ['scene=forms', 'preset=superpebble', 'material=soap'],
    'opal':       ['scene=solids', 'preset=superellipsoid', 'material=whiteopal'],
    'bodies':     ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=gold',
                   'material=frosted', 'cast=2', 'studio=lavender', 'floor=1', 'elev=20'],
    'bodiesjade': ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=truejade'],
}


def opt(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


EXE = opt('--exe', os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe'))
OUT = opt('--out', os.path.join(NATIVE, 'out', 'denoise', 'measure'))
W, H = (int(v) for v in opt('--size', '800x450').split('x'))
REF = int(opt('--ref', '1024'))
FRAMES = [int(v) for v in opt('--frames', '1,8,32,128').split(',')]
TURN = int(opt('--turn', '40'))
EXTRA = opt('--extra', '').split()
# laid on the two reference runs only, after --extra (e.g. "denoise=0": the truth without the denoiser)
REFX = opt('--refextra', '').split()
# --reuse DIR: take the reference pictures from that folder when they are there (made by an earlier run)
REUSE = opt('--reuse', '')
NAMES = sys.argv[1:] or list(CASES)
os.makedirs(OUT, exist_ok=True)


def read_png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def blur(a, n=2):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


def compare(a, b):
    d = np.abs(a - b)
    return {'mean': round(float(d.mean()), 3), 'blurred': round(float(np.abs(blur(a) - blur(b)).mean()), 3),
            'spots': round(float((d.max(axis=2) > 24).mean() * 100), 3)}


def run(png, args, ref=False):
    if ref and REUSE and os.path.exists(png):
        return 0
    cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'denoise', '--', EXE, '--still', png,
           'w=%d' % W, 'h=%d' % H] + args + EXTRA + (REFX if ref else [])
    r = subprocess.run(cmd, cwd=NATIVE, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    if r.returncode != 0 or not os.path.exists(png):
        raise SystemExit('failed: %s\n%s' % (' '.join(cmd), (r.stderr or r.stdout)[-800:]))
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]).get('msPerFrame', 0)
    except (ValueError, IndexError):
        return 0


results = {}
jpath = os.path.join(OUT, 'measure.json')
if os.path.exists(jpath):
    results = json.load(open(jpath))
for name in NAMES:
    args = CASES[name]
    p = lambda tag: os.path.join(OUT, '%s-%s.png' % (name, tag))
    if REUSE:
        for tag in ('ref', 'turnref'):
            if not os.path.exists(p(tag)) and os.path.exists(os.path.join(REUSE, '%s-%s.png' % (name, tag))):
                shutil.copyfile(os.path.join(REUSE, '%s-%s.png' % (name, tag)), p(tag))
    run(p('ref'), args + ['frames=%d' % REF, 'seed=3'], True)
    ref = read_png(p('ref'))
    row = {}
    for n in FRAMES:
        ms = run(p('still%d' % n), args + ['frames=%d' % n])
        row['still%d' % n] = compare(read_png(p('still%d' % n)), ref)
    run(p('turn'), args + ['turn=%d' % TURN])
    run(p('turnref'), args + ['turn=%d' % TURN, 'frames=%d' % REF], True)
    row['turn'] = compare(read_png(p('turn')), read_png(p('turnref')))
    results[name] = row
    print('%-11s ' % name + '  '.join('%s %5.2f/%4.2f/%5.2f%%' % (k, v['mean'], v['blurred'], v['spots']) for k, v in row.items()), flush=True)
    json.dump(results, open(jpath, 'w'), indent=1)
