"""Numbers of a light volume written with --glow-out: how much light the body holds, by depth below its surface.
usage: python tools/scatter_vol.py file.glow"""
import sys
import numpy as np
raw = np.fromfile(sys.argv[1], dtype=np.uint8)
res = int(np.frombuffer(raw[:4].tobytes(), dtype=np.uint32)[0])
a = np.frombuffer(raw[4:].tobytes(), dtype=np.float32).reshape(res, res, res, 4)
h = 3.0 / res
d = a[..., 3] / h
print('cells a side %d; cells inside %d (%.1f%%)' % (res, int((d < 0).sum()), 100 * (d < 0).mean()))
for lo, hi in ((0.5, 3), (0, 0.5), (-0.5, 0), (-1.5, -0.5), (-3, -1.5), (-6, -3), (-12, -6), (-99, -12)):
    m = (d >= lo) & (d < hi)
    if m.any():
        print('  distance %6.1f .. %5.1f cells: %8d cells, light %s' % (lo, hi, int(m.sum()), np.round(a[..., :3][m].mean(axis=0), 3)))
