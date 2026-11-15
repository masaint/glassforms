"""Several pictures side by side on one sheet, each with its name, for looking at many at once.

usage: python tools/layers_sheet.py out.png [--cols N] [--crop x0,y0,x1,y1] picture.png ..."""
import os, sys
from PIL import Image, ImageDraw

args = sys.argv[1:]
out = args.pop(0)
cols, crop = 2, None
while args and args[0].startswith('--'):
    flag = args.pop(0)
    if flag == '--cols':
        cols = int(args.pop(0))
    elif flag == '--crop':
        crop = tuple(int(v) for v in args.pop(0).split(','))
tiles = []
for path in args:
    im = Image.open(path).convert('RGB')
    if crop:
        im = im.crop(crop)
    ImageDraw.Draw(im).text((8, 6), os.path.splitext(os.path.basename(path))[0], fill=(255, 255, 255))
    tiles.append(im)
w, h = tiles[0].size
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * h))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
# no wider than 2000 pixels, so that it can be looked at whole
if sheet.width > 2000:
    sheet = sheet.resize((2000, sheet.height * 2000 // sheet.width), Image.LANCZOS)
sheet.save(out)
print(out, sheet.size)
