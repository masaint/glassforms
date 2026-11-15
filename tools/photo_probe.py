"""Measurements behind the photographic look of milky and opaque bodies.

usage: python tools/photo_probe.py <exe> <out folder> [test ...]

Tests (all by default):
  furnace   even white light from everywhere on a milky body that absorbs nothing: every pixel of the body
            should be as bright as the light (1). The estimate and the exact random walk, on a ball, thick
            Thomas and thin Thomas.
  top       one wide light straight overhead and nothing else, on a ball: how the brightness falls from the
            top of the ball to the bottom, for milk glass, the walk, and clay; printed band by band down the
            ball (a dense white body lit from above is close to a matte one: bright on top, dark below).
"""
import os
import subprocess
import sys

import numpy as np

EXE = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
TESTS = sys.argv[3:] or ['furnace', 'top']
os.makedirs(OUT, exist_ok=True)
BALL = 'scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2'
THOMAS = 'scene=attractors preset=thomas dist=4.4'


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    return np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4)


def shot(name, settings, frames=16, size=320):
    hdr = os.path.join(OUT, name + '.hdr')
    png = os.path.join(OUT, name + '.png')
    r = subprocess.run([EXE, '--still', png, '--hdr', hdr, 'w=%d' % size, 'h=%d' % size, 'frames=%d' % frames] + settings.split(), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(name + '\n' + r.stderr[-3000:])
    return load(hdr)


if 'furnace' in TESTS:
    FUR = 'rig= ambient=1,1,1 sss=1 bodyR=1 bodyG=1 bodyB=1 disp=0 bloom=0 absorb=0,0,0 fit=0 sssDepth=0.03'
    for sname, shape in (('ball', BALL), ('thomas_thick', THOMAS + ' thick=0.16'), ('thomas_thin', THOMAS + ' thick=0.05'), ('thomas', THOMAS)):
        row = []
        for mode, extra, fr in (('estimate', '', 8), ('walk', 'sssWalk=1', 256)):
            a = shot('furnace_%s_%s' % (sname, mode), '%s %s %s' % (shape, FUR, extra), fr)
            on = a[..., 3] > 0
            row.append(a[..., :3][on].mean())
        print('furnace %-13s estimate %.3f   walk %.3f   ratio %.3f' % (sname, row[0], row[1], row[0] / row[1]), flush=True)

if 'top' in TESTS:
    TOP = 'rig=0,90,40,40,2,2,2,0.6 ambient=0,0,0 bloom=0 fit=0'
    rows = {}
    for mode, extra, fr in (('milk', 'material=milk', 8), ('walk', 'material=milk sssWalk=1', 256), ('clay', 'clay=1', 8), ('jade', 'material=jade', 8), ('truejade', 'material=jade sssWalk=1', 256)):
        a = shot('top_' + mode, '%s %s %s' % (BALL, TOP, extra), fr)
        on = a[..., 3] > 0
        ys = np.nonzero(on.any(axis=1))[0]
        bands = np.array_split(np.arange(ys[0], ys[-1] + 1), 8)
        lum = a[..., :3].mean(axis=2)
        rows[mode] = [lum[b][on[b]].mean() for b in bands]
        print('top %-9s' % mode, ' '.join('%.3f' % v for v in rows[mode]), flush=True)
