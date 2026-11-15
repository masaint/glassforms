"""Numbers of a traced picture written with --hdr: its level over the shape, its brightest value, and
whether anything in it is not a number.   usage: python tools/scatter_stat.py file.hdr [file.hdr ...]"""
import sys
import numpy as np
for path in sys.argv[1:]:
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    a = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4)
    on = a[..., 3] > 0
    c = a[..., :3]
    print('%s: %dx%d on shape %.3f  mean rgb %s  max %.2f  not finite %d  above 20: %d' % (path, w, h, on.mean(), np.round(c[on].mean(axis=0), 4), np.nanmax(c), int((~np.isfinite(c)).sum()), int((c > 20).sum())))
