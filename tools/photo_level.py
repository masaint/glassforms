"""The level of a traced picture over the shape, for several settings in turn, against the first.
usage: python tools/photo_level.py <exe> <out folder> "<settings>" "<settings>" ...
Each still is written with --hdr; prints its mean over the shape (luminance) and the ratio to the first."""
import os
import subprocess
import sys

import numpy as np

exe, out = os.path.abspath(sys.argv[1]), sys.argv[2]
os.makedirs(out, exist_ok=True)
first = None
for i, st in enumerate(sys.argv[3:]):
    hdr = os.path.join(out, 'lv%02d.hdr' % i)
    r = subprocess.run([exe, '--still', hdr[:-4] + '.png', '--hdr', hdr, 'w=320', 'h=320', 'frames=16', 'bloom=0'] + st.split(), capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    raw = np.fromfile(hdr, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    a = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(int(h), int(w), 4)
    on = a[..., 3] > 0
    v = a[..., :3][on].mean(axis=0)
    lum = float(v @ [0.2126, 0.7152, 0.0722])
    first = first or lum
    print('%-70s %.4f  x %.3f   rgb %s' % (st[:70], lum, lum / first, np.round(v, 3)), flush=True)
