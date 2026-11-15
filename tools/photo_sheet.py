"""Pictures side by side, each scaled to one height, with a label under each, for comparing renders with
photographs.   usage: python tools/photo_sheet.py out.png [height=360] [cols=4] <picture>[::label] ...

A picture that is not there leaves an empty place. Sheets that contain a photograph from docs/refs stay
under out/ (they are not ours to publish).
"""
import os
import sys

from PIL import Image, ImageDraw

out = sys.argv[1]
opts = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a and '::' not in a and not os.path.exists(a))
items = [a for a in sys.argv[2:] if a not in ['%s=%s' % kv for kv in opts.items()]]
H = int(opts.get('height', '360'))
COLS = int(opts.get('cols', '4'))
tiles = []
for it in items:
    path, _, label = it.partition('::')
    label = label or os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(path):
        im = Image.open(path).convert('RGB')
        im = im.resize((max(1, round(im.width * H / im.height)), H), Image.LANCZOS)
    else:
        im = Image.new('RGB', (H, H), (40, 0, 0))
    tiles.append((im, label))
rows = [tiles[i:i + COLS] for i in range(0, len(tiles), COLS)]
W = max(sum(t[0].width for t in r) + 4 * (len(r) - 1) for r in rows)
sheet = Image.new('RGB', (W, len(rows) * (H + 18)), (16, 16, 16))
d = ImageDraw.Draw(sheet)
y = 0
for r in rows:
    x = 0
    for im, label in r:
        sheet.paste(im, (x, y))
        d.text((x + 4, y + H + 3), label, fill=(230, 230, 230))
        x += im.width + 4
    y += H + 18
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
sheet.save(out)
print('wrote', out, sheet.size)
