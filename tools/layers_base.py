"""One layer changes nothing: the volume and the still of six shapes, from two builds of the program, compared.

usage: python tools/layers_base.py <exe before> <exe now> [outdir]

For each shape: the fingerprint of the volume (every bit of it) and a still, from each program; the
fingerprints must be equal and the stills identical pixel for pixel. One line per check, a summary, exit 1
on failure."""
import json, os, subprocess, sys
import numpy as np
from PIL import Image

SHAPES = [
    ['scene=attractors', 'preset=thomas'],
    ['scene=attractors', 'preset=lorenz', 'anim=3.5'],
    ['scene=knots', 'preset=trefoil', 'melt=0'],
    ['scene=knots', 'preset=borromean'],
    ['scene=surfaces', 'preset=gyroid'],
    ['scene=algebraic', 'preset=tanglecube', 'puff=0.01'],
]
# the newer families, where the program has them
NEWER = [
    ['scene=gems', 'preset=brilliant'],
    ['scene=solids', 'preset=icosahedron'],
]

def run(exe, *args):
    r = subprocess.run([exe] + list(args), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])

def main():
    before, now = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join('out', 'base')
    os.makedirs(out, exist_ok=True)
    bad = 0
    have = subprocess.run([before, '--list'], capture_output=True, text=True).stdout
    shapes = SHAPES + [s for s in NEWER if '"%s"' % s[0].split('=')[1] in have]
    for s in shapes:
        name = '_'.join(a.split('=')[1] for a in s)
        # the volume: a still shape as it is first built; an animating one as a frame of video builds it
        vol = [a for a in s if not a.startswith('anim=')] + (['detail=hq'] if any(a.startswith('anim=') for a in s) else [])
        a = run(before, '--vol-hash', 'w=64', 'h=64', *vol)
        b = run(now, '--vol-hash', 'w=64', 'h=64', *vol)
        ok = a['hash'] == b['hash'] and a['volume'] == b['volume']
        bad += not ok
        print('%s volume %s: %s%s' % ('ok  ' if ok else 'FAIL', name, b['hash'], '' if ok else ' against %s' % a['hash']), flush=True)
        pa, pb = os.path.join(out, name + '_before.png'), os.path.join(out, name + '_now.png')
        run(before, '--still', pa, 'w=960', 'h=540', 'frames=64', *s)
        run(now, '--still', pb, 'w=960', 'h=540', 'frames=64', *s)
        x, y = np.asarray(Image.open(pa)).astype(np.int32), np.asarray(Image.open(pb)).astype(np.int32)
        d = int(np.abs(x - y).max()) if x.shape == y.shape else 999
        ok = d == 0
        bad += not ok
        print('%s still  %s: largest difference %d grey levels' % ('ok  ' if ok else 'FAIL', name, d), flush=True)
    print('%d checks, %d failed' % (2 * len(shapes), bad))
    sys.exit(1 if bad else 0)

main()
