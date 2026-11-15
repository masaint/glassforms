"""Try the denoiser with other numbers (GLASSFORMS_DENOISE_TUNE) against references already made by
denoise_measure.py, quickly: one frame, eight frames, and the turning view.

usage: python tools/denoise_try.py <refdir> <case> "<tune>" ["<tune>" ...]

Prints for each tuning: grey levels (mean / blurred / spots %) and brightness against the reference.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
REFDIR, CASE = sys.argv[1], sys.argv[2]
import importlib.util
spec = importlib.util.spec_from_file_location('m', os.path.join(HERE, 'denoise_measure.py'))
CASES = {}
src = open(os.path.join(HERE, 'denoise_measure.py'), encoding='utf-8').read()
exec(src[src.index('CASES = {'):src.index('\n}\n') + 3], CASES)
ARGS = CASES['CASES'][CASE]
OUT = os.path.join(NATIVE, 'out', 'denoise', 'try')
os.makedirs(OUT, exist_ok=True)


def png(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)


def blur(a, n=2):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


for tune in sys.argv[3:]:
    env = dict(os.environ, GLASSFORMS_DENOISE_TUNE=tune)
    row = []
    for tag, extra, ref in (('1', ['frames=1'], 'ref'), ('8', ['frames=8'], 'ref'), ('turn', ['turn=40'], 'turnref')):
        out = os.path.join(OUT, '%s-%s.png' % (CASE, tag))
        cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'denoise', '--', EXE, '--still', out, 'w=800', 'h=450'] + ARGS + extra
        r = subprocess.run(cmd, cwd=NATIVE, env=env, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr[-600:])
        a, b = png(out), png(os.path.join(REFDIR, '%s-%s.png' % (CASE, ref)))
        d = np.abs(a - b)
        row.append('%s %5.2f/%4.2f/%5.2f%% x%.3f' % (tag, d.mean(), np.abs(blur(a) - blur(b)).mean(), (d.max(axis=2) > 24).mean() * 100, a.mean() / b.mean()))
    print('%-40s %s' % (tune, '   '.join(row)), flush=True)
