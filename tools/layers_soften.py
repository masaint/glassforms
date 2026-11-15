"""How much reading a moved layer between voxels softens it, against the shape built where it stands.

usage: python tools/layers_soften.py        (from the folder that holds target/)

Two measures, each as the mean difference between two stills in grey levels (0 to 255):
  turned    one layer turned about x by 37 degrees while the whole object is turned back by as much: the
            same view of the same shape, once read at its own voxels and once read between them
  moved     a tube moved and turned, read between voxels, against the same tube built in that place
For scale: the same shape built at the next detail down (what losing a third of the detail costs), and
at two voxels a side fewer (the least change of detail there is)."""
import json, os, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'soften')
os.makedirs(OUT, exist_ok=True)

def still(name, *args):
    path = os.path.join(OUT, name + '.png')
    r = subprocess.run([EXE, '--still', path, 'w=960', 'h=540', 'frames=64'] + list(args), capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return path

def diff(a, b):
    x = np.asarray(Image.open(a).convert('RGB')).astype(np.float32)
    y = np.asarray(Image.open(b).convert('RGB')).astype(np.float32)
    return float(np.abs(x - y).mean())

def glass(a):
    # the share of the picture that is not background, to read the differences against
    x = np.asarray(Image.open(a).convert('RGB')).astype(np.float32)
    return float((x.max(axis=2) > 12).mean())

for fam, preset in (('knots', 'trefoil'), ('attractors', 'thomas'), ('surfaces', 'gyroid'), ('algebraic', 'tanglecube')):
    for res in (768, 256):
        s = ['scene=' + fam, 'preset=' + preset, 'res=%d' % res, 'ry=30']
        tag = '%s_%d' % (preset, res)
        plain = still(tag + '_plain', *s, 'rx=20')
        turned = still(tag + '_turned', *s, 'rx=-17', 'L1.rx=37')
        lower = still(tag + '_lower', *[a if not a.startswith('res=') else 'res=%d' % (res * 2 // 3 // 8 * 8) for a in s], 'rx=20')
        near = still(tag + '_near', *[a if not a.startswith('res=') else 'res=%d' % (res - 8) for a in s], 'rx=20')
        line = '%-10s %4d: turned %.3f' % (preset, res, diff(plain, turned))
        if fam in ('knots', 'attractors'):
            a = still(tag + '_moved', *s, 'rx=20', 'L1.x=0.11', 'L1.y=-0.07', 'L1.ry=25')
            b = still(tag + '_inplace', *s, 'rx=20', 'L1.x=0.11', 'L1.y=-0.07', 'L1.ry=25', 'inPlace=1')
            line += ', moved %.3f' % diff(a, b)
        line += '   (for scale: %d voxels instead, %.3f; %d instead, %.3f; glass covers %.0f%% of the picture)' % (res * 2 // 3 // 8 * 8, diff(plain, lower), res - 8, diff(plain, near), 100 * glass(plain))
        print(line, flush=True)
