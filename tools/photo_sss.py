"""Milky bodies lit from behind and from the front, for seeing whether they read as light scattering
inside them (a glow from within, soft terminators, bright thin parts).

usage: python tools/photo_sss.py <exe> <out folder> [only=<part of a name>] [name=value ...]
Writes <case>.png: milk, wax, jade, porcelain and the random walk of milk and jade, on Thomas and the
teapot, back-lit (one soft light behind, a little above) and front-lit (one soft light beside the camera,
up and to the side, a dim fill), each on black.
"""
import os
import subprocess
import sys

exe = os.path.abspath(sys.argv[1])
out = sys.argv[2]
extra = [a for a in sys.argv[3:] if not a.startswith('only=')]
only = next((a[5:] for a in sys.argv[3:] if a.startswith('only=')), '')
os.makedirs(out, exist_ok=True)
SHAPES = {'thomas': 'scene=attractors preset=thomas dist=4.3', 'teapot': 'scene=models preset=teapot'}
LIGHT = {
    'back': 'rig=0,15,14,14,6,6,6,0.6 ambient=0.01,0.01,0.012',
    'front': 'rig=140,35,16,16,3,3,3,0.6:-60,10,30,30,0.25,0.25,0.28,0.8 ambient=0.01,0.01,0.012',
}
for m, fr in (('milk', 16), ('wax', 16), ('jade', 16), ('porcelain', 16), ('truemilk', 256), ('truejade', 256)):
    for s in SHAPES:
        for l in LIGHT:
            name = '%s_%s_%s' % (m, s, l)
            if only and only not in name:
                continue
            path = os.path.join(out, name + '.png')
            r = subprocess.run([exe, '--still', path, 'w=480', 'h=480', 'frames=%d' % fr, 'bloom=0.05', 'material=' + m] + SHAPES[s].split() + LIGHT[l].split() + extra, capture_output=True, text=True)
            print(name, 'ok' if r.returncode == 0 else r.stderr[-800:], flush=True)
