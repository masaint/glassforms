"""What the frost tools share: running the program, reading its pictures, and the measures.

Pictures are compared on the traced picture before bloom and tone mapping (the program's --hdr), in grey
levels (0 to 255) after a plain tone map, so that a number means what it would on screen: the same
measure tools/dof_check.py uses.
"""
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')

THOMAS = 'scene=attractors preset=thomas dist=4.4'
GYROID = 'scene=surfaces preset=gyroid dist=5.2'
CUBE = 'scene=algebraic preset=tanglecube dist=5.4'
GEM = 'scene=gems preset=brilliant'
TREFOIL = 'scene=knots preset=trefoil'


def run(args, exe=None):
    """Run the program with these arguments (a list, or one string split at spaces); returns what it printed."""
    if isinstance(args, str):
        args = args.split()
    r = subprocess.run([exe or EXE] + list(args), capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError('%s\n%s' % (' '.join(args), (r.stderr or r.stdout).strip()[-800:]))
    return r.stdout


def load(path):
    """A --hdr file: (h, w, 4) floats, alpha = distance to the shape (0 = none)."""
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    return np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4).astype(np.float64)


def still(base, settings, w=640, h=360, frames=1, exe=None, extra=()):
    """Render a still to <base>.png and <base>.hdr; returns the traced picture."""
    os.makedirs(os.path.dirname(base), exist_ok=True)
    args = ['--still', base + '.png', '--hdr', base + '.hdr', 'w=%d' % w, 'h=%d' % h, 'bloom=0']
    if frames is not None:
        args.append('frames=%d' % frames)
    run(args + settings.split() + list(extra), exe)
    return load(base + '.hdr')


def bench(settings, w=3840, h=2160, n=40, exe=None):
    """Milliseconds for a turning frame."""
    out = run(['--bench', str(n), 'w=%d' % w, 'h=%d' % h] + settings.split(), exe)
    return json.loads(out.strip().splitlines()[-1])['ms']


def tone(rgb):
    x = np.clip((rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14), 0, 1)
    return np.where(x < 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055) * 255.0


def on(a):
    return a[..., 3] > 0


def grey(a, b, mask=None):
    """Mean difference of two pictures in grey levels after the tone map (over `mask`, or everywhere)."""
    d = np.abs(tone(a[..., :3]) - tone(b[..., :3])).mean(axis=2)
    return float(d[mask].mean() if mask is not None else d.mean())


def level(a, mask=None):
    """Mean light over the pixels that show the shape."""
    m = on(a) if mask is None else mask
    return float(a[..., :3][m].mean()) if m.any() else 0.0


def box(a, n=1):
    """A box blur over 2n+1 pixels each way (edges repeated)."""
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


def inner(a, n=2):
    """Pixels on the shape whose neighbours for n pixels around are on it too (away from the outline)."""
    m = on(a).astype(np.float64)
    return box(m, n) > 0.999


def grain(a, mask=None):
    """Grain: how far each pixel stands from the average of its 3 x 3 neighbourhood, in grey levels after
    the tone map, over the inside of the shape. A smooth picture scores near 0 whatever it shows; fine
    detail and noise both raise it, so it is read against the settled picture's own score."""
    t = tone(a[..., :3]).mean(axis=2)
    d = np.abs(t - box(t, 1))
    m = inner(a) if mask is None else mask
    return float(d[m].mean()) if m.any() else 0.0


def apart(a, b):
    """Mean difference after both are blurred over 5 pixels (which leaves grain out), as a share of their
    level: the materials check's measure."""
    m = on(a) & on(b)
    x, y = a[..., :3], b[..., :3]
    lv = max(0.5 * (x[m].mean() + y[m].mean()), 1e-9)
    return float(np.abs(box(x, 2) - box(y, 2))[m].mean() / lv)
