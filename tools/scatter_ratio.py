"""How two traced pictures of a ball compare ring by ring from the middle of the picture outwards.
usage: python tools/scatter_ratio.py a.hdr b.hdr"""
import sys
import numpy as np
from scatter_cmp import load, blur
a, b = load(sys.argv[1]), load(sys.argv[2])
m = (a[..., 3] > 0) & (b[..., 3] > 0)
h, w = m.shape
ys, xs = np.nonzero(m)
cy, cx = ys.mean(), xs.mean()
R = np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max())
yy, xx = np.mgrid[0:h, 0:w]
r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
ga, gb = a[..., 1], b[..., 1]
for lo in np.arange(0, 1.0, 0.1):
    k = m & (r >= lo) & (r < lo + 0.1)
    if k.any():
        print('ring %.1f..%.1f: %.4f against %.4f  x %.3f' % (lo, lo + 0.1, ga[k].mean(), gb[k].mean(), ga[k].mean() / gb[k].mean()))
