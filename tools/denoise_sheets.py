"""Side-by-side pictures of the denoiser, from what denoise_measure.py drew (out/denoise/off, out/denoise/on)
and the settled pictures (<case>-settled.png in out/denoise/on, drawn with the same rays as the reference).

usage: python tools/denoise_sheets.py [case ...]

Writes out/denoise/<case>.png: top row, a still view without the denoiser (1 frame, 8 frames, settled)
then with it (the same); bottom row, a turning view (the last of 40 frames) without and with it, and
that view settled. Also prints how far the settled picture with the denoiser is from the one without
(the same rays: what is left is what the filter changed).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'out', 'denoise')
CASES = sys.argv[1:] or ['truemilk', 'truejade', 'roughref', 'milkthomas', 'jadegem']
TILE = 440


def tile(path, label):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    im = im.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize((TILE, TILE), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, TILE, 18), fill=(0, 0, 0))
    d.text((6, 3), label, fill=(235, 235, 235))
    return im


for c in CASES:
    off = lambda t: os.path.join(OUT, 'off', '%s-%s.png' % (c, t))
    on = lambda t: os.path.join(OUT, 'on', '%s-%s.png' % (c, t))
    top = [tile(off('still1'), 'without: 1 frame'), tile(off('still8'), 'without: 8 frames'), tile(off('ref'), 'without: settled (1024)'),
           tile(on('still1'), 'denoised: 1 frame'), tile(on('still8'), 'denoised: 8 frames'), tile(on('settled'), 'denoised: settled (1024)')]
    bottom = [tile(off('turn'), 'turning, without'), tile(on('turn'), 'turning, denoised'), tile(off('turnref'), 'that view settled')]
    sheet = Image.new('RGB', (TILE * 6 + 10, TILE * 2 + 10), (40, 40, 40))
    for i, t in enumerate(top):
        sheet.paste(t, (i * TILE + (10 if i >= 3 else 0), 0))
    for i, t in enumerate(bottom):
        sheet.paste(t, (i * TILE, TILE + 10))
    sheet.save(os.path.join(OUT, '%s.png' % c))
    a = np.asarray(Image.open(on('settled')).convert('RGB'), float)
    b = np.asarray(Image.open(off('ref')).convert('RGB'), float)
    print('%-11s settled, denoised against without (same rays): %.2f grey levels, brightness x%.4f' % (c, np.abs(a - b).mean(), a.mean() / b.mean()))
