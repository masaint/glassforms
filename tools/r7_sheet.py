r"""The refining sheet: for each scene, crops (at 2x) of its noisiest region after 0.25, 1 and 3 s of refining,
before (adapt=0) above after. The region is where the 0.25 s picture before differs most from the 3 s one
after (the noise still to go). Reads <dir>/<scene>_t<time>_a<0|1>.png from tools/r7_times.py.

    python tools/r7_sheet.py <dir> <out.png> [scenes,..]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

d, out = sys.argv[1], sys.argv[2]
scenes = sys.argv[3].split(',') if len(sys.argv) > 3 else ['gem', 'pot', 'thomas', 'milky']
times = ['0.25', '1', '3']
CW, CH = 240, 160          # crop, shown at 2x
LAB = 18


def load(s, t, a):
    return Image.open(os.path.join(d, f'{s}_t{t}_a{a}.png')).convert('RGB')


rows = []
for s in scenes:
    noisy = np.asarray(load(s, '0.25', '0')).astype(np.float32)
    clean = np.asarray(load(s, '3', '1')).astype(np.float32)
    diff = np.abs(noisy - clean).mean(axis=2)
    # the window of the crop's size with the most difference (summed-area table, every 8 pixels)
    sat = diff.cumsum(0).cumsum(1)
    best, at = -1.0, (0, 0)
    for y in range(0, diff.shape[0] - CH, 8):
        for x in range(0, diff.shape[1] - CW, 8):
            v = sat[y + CH - 1, x + CW - 1] - (sat[y - 1, x + CW - 1] if y else 0) - (sat[y + CH - 1, x - 1] if x else 0) + (sat[y - 1, x - 1] if x and y else 0)
            if v > best:
                best, at = v, (x, y)
    box = (at[0], at[1], at[0] + CW, at[1] + CH)
    for a in ('0', '1'):
        rows.append((s, a, box))

sheet = Image.new('RGB', (len(times) * CW * 2, len(rows) * (CH * 2 + LAB)), (10, 10, 12))
dr = ImageDraw.Draw(sheet)
for r, (s, a, box) in enumerate(rows):
    for c, t in enumerate(times):
        im = load(s, t, a).crop(box).resize((CW * 2, CH * 2), Image.NEAREST)
        y = r * (CH * 2 + LAB)
        sheet.paste(im, (c * CW * 2, y + LAB))
        dr.text((c * CW * 2 + 6, y + 3), f'{s} {"after (adaptive)" if a == "1" else "before"}, {t} s refining at 60 Hz  [crop {box[0]},{box[1]}]', fill=(200, 205, 215))
sheet.save(out)
print(out, sheet.size)
