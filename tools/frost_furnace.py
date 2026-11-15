"""Uniform light from everywhere on glass that absorbs nothing: every pixel of the shape should show
exactly that light, whatever the roughness. Prints the mean, least and most over the shape."""
import os, sys
import numpy as np
import frost_lib as fl
shape = {'gyroid': fl.GYROID, 'thomas': fl.THOMAS, 'cube': fl.CUBE, 'gem': fl.GEM}[sys.argv[1] if len(sys.argv) > 1 else 'gyroid']
extra = ' '.join(sys.argv[2:])
OUT = os.path.join(fl.ROOT, 'out', 'frost_furnace')
for r in (0.0, 0.15, 0.3, 0.6, 1.0):
    for mode, frames in (('live', 4), ('random', 64)):
        if r == 0.0 and mode == 'random':
            continue
        s = '%s rig= ambient=1,1,1 density=0 %s rough=%g%s' % (shape, extra, r, ' roughRef=1' if mode == 'random' else '')
        a = fl.still(os.path.join(OUT, 'f_%s_%03d' % (mode, round(r * 100))), s, w=320, h=180, frames=frames)
        m = fl.inner(a, 2)
        v = a[..., :3].mean(axis=2)[m]
        print('rough %.2f %-6s mean %.4f  least %.3f  1%% %.3f  99%% %.3f  most %.3f' % (r, mode, v.mean(), v.min(), np.percentile(v, 1), np.percentile(v, 99), v.max()), flush=True)
