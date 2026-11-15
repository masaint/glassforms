"""Numbers about the traced picture before bloom and tone mapping (what `--still x.png --hdr x.hdr` writes:
width, height, then RGBA 32-bit floats, top row first; alpha = distance to the surface, 0 where there is none).

usage: python tools/hdr_stats.py a.hdr [b.hdr]

One file: its mean light (red, green, blue), the share of pixels on the shape, the brightest value.
Two files: also how far apart they are: the mean absolute difference over the shape's pixels as a share
of the mean light there, and the same after both are blurred a little (which leaves out grain).
"""
import sys
import numpy as np


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    return np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4)


def blur(a, n=2):
    # a box blur, n pixels either side, done as running sums
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


def stats(a):
    on = a[..., 3] > 0
    rgb = a[..., :3]
    return {'mean': rgb.mean(axis=(0, 1)).tolist(), 'on': float(on.mean()), 'max': float(rgb.max()),
            'meanOn': rgb[on].mean(axis=0).tolist() if on.any() else [0, 0, 0]}


def compare(a, b):
    on = (a[..., 3] > 0) & (b[..., 3] > 0)
    x, y = a[..., :3], b[..., :3]
    level = max(float(0.5 * (x[on].mean() + y[on].mean())), 1e-9)
    return {'diff': float(np.abs(x - y)[on].mean() / level), 'diffBlurred': float(np.abs(blur(x) - blur(y))[on].mean() / level),
            'brightness': float(x[on].mean() / max(float(y[on].mean()), 1e-9))}


if __name__ == '__main__':
    a = load(sys.argv[1])
    print(stats(a))
    if len(sys.argv) > 2:
        b = load(sys.argv[2])
        print(stats(b))
        print(compare(a, b))
