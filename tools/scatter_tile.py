"""Lay pictures side by side on one sheet, each with its file name under it, for looking at.

usage: python tools/scatter_tile.py out.png cols [height=N] picture.png [picture.png ...]

Every picture is scaled to the same height (the first one's, or N); a name may be given as label=path.
"""
import os
import sys

from PIL import Image, ImageDraw


def tile(out, cols, items, height=None):
    ims = []
    for it in items:
        label, path = it.split('=', 1) if '=' in it and not os.path.exists(it) else (os.path.splitext(os.path.basename(it))[0], it)
        ims.append((label, Image.open(path).convert('RGB')))
    h = height or ims[0][1].height
    ims = [(l, im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)) for l, im in ims]
    w = max(im.width for _, im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w, rows * (h + 16)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(ims):
        x, y = (i % cols) * w, (i // cols) * (h + 16)
        sheet.paste(im, (x + (w - im.width) // 2, y))
        d.text((x + 4, y + h + 2), label, fill=(230, 230, 230))
    sheet.save(out)
    print('wrote', out, sheet.size)


if __name__ == '__main__':
    a = sys.argv[1:]
    height = None
    rest = []
    for x in a[2:]:
        if x.startswith('height='):
            height = int(x[7:])
        else:
            rest.append(x)
    tile(a[0], int(a[1]), rest, height)
