"""Compare traced pictures written with --hdr: level over the shape by colour, and how far apart two are
(mean difference after blurring both over 5 pixels, as a share of their level: tools/material_check.py's measure).
usage: python tools/scatter_cmp.py a.hdr b.hdr"""
import sys
import numpy as np


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    return np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4).astype(np.float64)


def blur(a, n=2):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


def compare(a, b):
    m = (a[..., 3] > 0) & (b[..., 3] > 0)
    x, y = a[..., :3], b[..., :3]
    la, lb = x[m].mean(axis=0), y[m].mean(axis=0)
    lv = max(0.5 * (x[m].mean() + y[m].mean()), 1e-9)
    apart = float(np.abs(blur(x) - blur(y))[m].mean() / lv)
    return la, lb, apart


if __name__ == '__main__':
    a, b = load(sys.argv[1]), load(sys.argv[2])
    la, lb, apart = compare(a, b)
    print('level %s against %s: x %s; apart %.3f' % (np.round(la, 4), np.round(lb, 4), np.round(la / lb, 3), apart))
