"""One furnace still (even white light from everywhere, a milky body that absorbs nothing) and its level.
usage: python tools/photo_fur.py <exe> <out.hdr> [name=value ...]   (the shape and anything else as settings)
Prints the mean over the body, the mean over the inner half of the body (by distance from the outline in
pixels: the outline's grazing reflections left out), and the 5th and 50th percentiles."""
import subprocess
import sys

import numpy as np
from scipy import ndimage

import os
exe, hdr = os.path.abspath(sys.argv[1]), sys.argv[2]
FUR = 'rig= ambient=1,1,1 sss=1 bodyR=1 bodyG=1 bodyB=1 disp=0 bloom=0 absorb=0,0,0 fit=0 sssDepth=0.03 w=320 h=320 frames=8'.split()
r = subprocess.run([exe, '--still', hdr[:-4] + '.png', '--hdr', hdr] + FUR + sys.argv[3:], capture_output=True, text=True)
if r.returncode:
    sys.exit(r.stderr[-3000:])
raw = np.fromfile(hdr, dtype=np.uint8)
w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
a = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4)
on = a[..., 3] > 0
lum = a[..., :3].mean(axis=2)
inner = ndimage.distance_transform_edt(on) > 2
print('mean %.3f  inner %.3f  p5 %.3f  p50 %.3f' % (lum[on].mean(), lum[inner].mean(), np.percentile(lum[on], 5), np.percentile(lum[on], 50)))
