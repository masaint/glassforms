"""Grain of the live view through a change of motion, frame by frame (reads what `--motion-seq` wrote).

    python tools/motion_grain.py <dir> [<dir> ...]

For every frame that has a settled reference (ref_NNN.png): the mean absolute difference from it in grey
levels of 255 (as docs/denoise.md measures grain), the same after a 5 x 5 box blur (what is left is
not grain but error that does not average away: smear, lag, a wrong level), and the brightness as a
share of the reference's. Frames are numbered from the first moving one; frames.jsonl says which were
moving.
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def read(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def blur(a, n=2):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(n + 1, n) if i == axis else (0, 0) for i in range(a.ndim)], mode='edge'), axis=axis)
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[axis] = slice(2 * n + 1, None)
        lo[axis] = slice(0, -(2 * n + 1))
        a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * n + 1)
    return a


def main():
    for d in sys.argv[1:]:
        frames = [json.loads(l) for l in open(os.path.join(d, 'frames.jsonl'))]
        print(d)
        print('  frame  moving  accum detail   grain  blurred  level')
        for f in frames:
            i = f['i']
            ref = os.path.join(d, f'ref_{i:03d}.png')
            if not os.path.exists(ref):
                continue
            a, b = read(os.path.join(d, f'live_{i:03d}.png')), read(ref)
            g = np.abs(a - b).mean()
            gb = np.abs(blur(a) - blur(b)).mean()
            lv = a.mean() / max(b.mean(), 1e-9)
            print(f"  {i:5d}  {'yes' if f['moving'] else 'no ':3s}  {f['accum']:6d} {f['detail']:6s} {g:7.2f}  {gb:7.2f}  {lv:5.3f}")


if __name__ == '__main__':
    main()
