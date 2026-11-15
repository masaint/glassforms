"""Several pictures side by side on one sheet, each with its name under it: for looking at many
materials at once.

usage: python tools/material_sheet.py out.png <columns> a.png b.png ...   (a name may be given as name=file)
"""
import os
import sys
from PIL import Image, ImageDraw

out, cols = sys.argv[1], int(sys.argv[2])
items = []
for a in sys.argv[3:]:
    name, _, path = a.rpartition('=')
    items.append((name or os.path.splitext(os.path.basename(path))[0], path))
ims = [Image.open(p).convert('RGB') for _, p in items]
w, h = ims[0].size
rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * (h + 18)), (0, 0, 0))
d = ImageDraw.Draw(sheet)
for i, im in enumerate(ims):
    x, y = (i % cols) * w, (i // cols) * (h + 18)
    sheet.paste(im.resize((w, h)), (x, y))
    d.text((x + 6, y + h + 3), items[i][0], fill=(230, 230, 230))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
sheet.save(out)
print('wrote', out, sheet.size)
