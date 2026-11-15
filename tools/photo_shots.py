"""Stills of the opaque and milky materials under the program's best lighting, for setting beside
photographs of the real materials (and before / after a change to how they are drawn).

usage: python tools/photo_shots.py <exe> <out folder> [only=<part of a case name>] [size=512] [frames=64]

Each case is written as <out folder>/<case>.png. The cases: polished and satin metals on a ball and the
teapot, and the milky bodies on the teapot and on Thomas, each under the garden picture (shown behind),
the white studio and the lavender studio.
"""
import os
import subprocess
import sys

args = [a for a in sys.argv[1:] if '=' not in a]
opts = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
EXE = os.path.abspath(args[0])
OUT = os.path.abspath(args[1])
ONLY = opts.get('only', '')
SIZE = int(opts.get('size', '512'))
FRAMES = int(opts.get('frames', '64'))
os.makedirs(OUT, exist_ok=True)

SHAPES = {
    'ball': 'scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2',
    'teapot': 'scene=models preset=teapot',
    'thomas': 'scene=attractors preset=thomas dist=4.3',
}
LIGHTS = {
    'garden': 'hdri=' + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hdri', 'suburban_garden_4k.exr').replace(os.sep, '/') + ' hdriShow=1',
    'white': 'studio=white',
    'lavender': 'studio=lavender',
}
CASES = []
for m in ('chrome', 'gold', 'brushed', 'satingold'):
    for s in ('ball', 'teapot'):
        for l in ('garden', 'white'):
            CASES.append(('%s_%s_%s' % (m, s, l), '%s %s material=%s' % (SHAPES[s], LIGHTS[l], m), FRAMES))
for m in ('jade', 'wax', 'porcelain', 'milk'):
    for s in ('teapot', 'thomas'):
        for l in ('garden', 'white', 'lavender'):
            CASES.append(('%s_%s_%s' % (m, s, l), '%s %s material=%s' % (SHAPES[s], LIGHTS[l], m), FRAMES))
for m in ('truejade', 'truemilk'):
    for l in ('garden', 'white'):
        CASES.append(('%s_teapot_%s' % (m, l), '%s %s material=%s' % (SHAPES['teapot'], LIGHTS[l], m), 256))

for name, settings, frames in CASES:
    if ONLY and ONLY not in name:
        continue
    path = os.path.join(OUT, name + '.png')
    cmd = [EXE, '--still', path, 'w=%d' % SIZE, 'h=%d' % SIZE, 'frames=%d' % frames] + settings.split() + [a for a in sys.argv[1:] if '=' in a and a.split('=')[0] not in ('only', 'size', 'frames')]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(name, 'ok' if r.returncode == 0 else 'FAILED\n' + r.stderr[-2000:], flush=True)
