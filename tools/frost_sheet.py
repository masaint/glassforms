"""Lay pictures side by side with a caption under each.

usage: python tools/frost_sheet.py out.png <columns> "caption=path" "caption=path" ... [--height N]

Every picture is scaled to the same height (default: the first one's) and centred in a cell as wide as
the widest. A path that does not exist leaves an empty cell with its caption.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont


def sheet(out, cols, items, height=None, pad=4, bar=18):
    pics = []
    for cap, path in items:
        im = Image.open(path).convert('RGB') if os.path.exists(path) else None
        pics.append((cap, im))
    h = height or next(im.height for _, im in pics if im is not None)
    cells = []
    for cap, im in pics:
        if im is not None and im.height != h:
            im = im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)
        cells.append((cap, im))
    w = max(im.width for _, im in cells if im is not None)
    rows = (len(cells) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * w + (cols - 1) * pad, rows * (h + bar) + (rows - 1) * pad), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('segoeui.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    for i, (cap, im) in enumerate(cells):
        x, y = (i % cols) * (w + pad), (i // cols) * (h + bar + pad)
        if im is not None:
            canvas.paste(im, (x + (w - im.width) // 2, y))
        d.text((x + 4, y + h + 1), cap, fill=(235, 235, 235), font=font)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    canvas.save(out)
    return canvas.size


if __name__ == '__main__':
    argv = sys.argv[1:]
    height = None
    if '--height' in argv:
        i = argv.index('--height')
        height = int(argv[i + 1])
        del argv[i:i + 2]
    out, cols = argv[0], int(argv[1])
    items = [a.split('=', 1) for a in argv[2:]]
    print(sheet(out, cols, items, height))
