"""Before/after sheets for the materials round, next to the reference.

    python tools/mats_sheet.py <out png> <reference png> <title> <label=path> [<label=path> ...]

The reference (a crop of it, if given as path@x0,y0,x1,y1) and each still are scaled to one height and
laid in a row, each with its label under it.
"""
import sys
from PIL import Image, ImageDraw, ImageFont

H = 540


def load(spec):
    path, _, box = spec.partition('@')
    im = Image.open(path).convert('RGB')
    if box:
        im = im.crop(tuple(int(v) for v in box.split(',')))
    return im.resize((max(1, round(im.width * H / im.height)), H), Image.LANCZOS)


def main(argv):
    out, ref, title = argv[1], argv[2], argv[3]
    items = [('reference (MagicaCSG / photo)', load(ref))]
    for a in argv[4:]:
        label, _, path = a.partition('=')
        items.append((label, load(path)))
    try:
        font = ImageFont.truetype('arial.ttf', 22)
        big = ImageFont.truetype('arial.ttf', 28)
    except OSError:
        font = big = ImageFont.load_default()
    gap, top, foot = 12, 50, 40
    w = sum(i.width for _, i in items) + gap * (len(items) + 1)
    sheet = Image.new('RGB', (w, H + top + foot), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 10), title, fill=(235, 235, 235), font=big)
    x = gap
    for label, im in items:
        sheet.paste(im, (x, top))
        d.text((x + 4, top + H + 8), label, fill=(200, 200, 200), font=font)
        x += im.width + gap
    sheet.save(out)
    print(out)


if __name__ == '__main__':
    main(sys.argv)
