"""Each stroke (gf_core::csg::Strokes) on a csg cylinder, in clay, and a sheet of them.

    python tools/strokes_shots.py [out dir]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = os.path.join(HERE, 'gpu_run.py')
BASE = ['scene=studio', 'preset=cylinder', 'trace=csg', 'rx=30', 'clay=1', 'p.round=0.0', 'p.bevel=0.0']
CASES = [
    ('none', {}),
    ('shell 0.08 (cut open by a hole)', dict(shell=0.08, hole=0.35)),
    ('hole 0.3', dict(hole=0.3)),
    ('round 0.15', dict(round=0.15)),
    ('cone 0.6', dict(cone=0.6)),
    ('star 0.5, 5 points', dict(star=0.5, points=5)),
    ('star 0.35, 12 points, cone 0.4', dict(star=0.35, points=12, cone=0.4)),
    ('mirror x of a coned, starred one', dict(star=0.4, points=3, cone=0.5, mirror=[True, False, False])),
]


def shot(out, i, label, s):
    path = os.path.join(out, f'stroke_{i}.png')
    full = dict(shell=0, hole=0, round=0, cone=0, star=0, points=5, mirror=[False, False, False])
    full.update(s)
    cmd = [sys.executable, QUEUE, '--name', 'strokes', '--', EXE, '--still', path, 'w=640', 'h=400', 'frames=24'] + BASE + ['strokes=' + json.dumps(full, separators=(',', ':'))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(label, 'ok' if r.returncode == 0 else 'FAILED ' + (r.stderr or r.stdout)[-300:], flush=True)
    return path


def main(argv):
    out = argv[0] if argv else os.path.join(NATIVE, 'shots', 'strokes')
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        paths = list(ex.map(lambda c: shot(out, c[0], c[1][0], c[1][1]), enumerate(CASES)))
    sheet = Image.new('RGB', (4 * 646, 2 * 406), (24, 24, 24))
    for i, (p, (label, _)) in enumerate(zip(paths, CASES)):
        im = Image.open(p).convert('RGB') if os.path.exists(p) else Image.new('RGB', (640, 400))
        ImageDraw.Draw(im).text((8, 6), label, fill=(230, 230, 230))
        sheet.paste(im, ((i % 4) * 646, (i // 4) * 406))
    sheet.save(os.path.join(out, 'strokes_sheet.png'))


if __name__ == '__main__':
    main(sys.argv[1:])
