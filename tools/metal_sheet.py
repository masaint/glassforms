"""Before/after sheet of the r4 metal stills beside the reference photos.

    python tools/metal_sheet.py <before dir> <after dir> <out png>
Rows: a photo crop (where there is one), before, after, and a 2x crop of before/after.
"""
import os
import sys
from PIL import Image, ImageDraw

S = 360
SBS = 'D:/glassforms/native/out/photo/sbs/metal_after.png'
# (crops of metal_after.png's left column: chrome, gold ring, satin steel, satin gold)
PH = {'chrome': (0, 0, 298, 300), 'gold': (0, 318, 406, 618), 'steel': (0, 636, 384, 936), 'sgold': (0, 954, 400, 1254)}
ROWS = [
    ('gold_knot', 'gold', (240, 240, 480, 480)),
    ('gold_teapot', 'gold', (400, 280, 640, 520)),
    ('lamp_twist', None, (250, 400, 490, 640)),
    ('steel_studio', 'steel', (150, 150, 390, 390)),
    ('steel_garden', 'steel', (150, 150, 390, 390)),
    ('steel_teapot', 'steel', (250, 280, 490, 520)),
    ('sgold_studio', 'sgold', (150, 150, 390, 390)),
    ('sgold_garden', 'sgold', (150, 150, 390, 390)),
]


def main(bdir, adir, out):
    sbs = Image.open(SBS).convert('RGB')
    sheet = Image.new('RGB', (S * 5, (S + 18) * len(ROWS)), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    for r, (name, ph, box) in enumerate(ROWS):
        y = r * (S + 18)
        b = Image.open(os.path.join(bdir, name + '.png')).convert('RGB')
        a = Image.open(os.path.join(adir, name + '.png')).convert('RGB')
        cells = [sbs.crop(PH[ph]).resize((S, S)) if ph else None, b.resize((S, S)), a.resize((S, S)),
                 b.crop(box).resize((S, S)), a.crop(box).resize((S, S))]
        labels = ['photo-' + ph if ph else '', name + ' before', 'after', 'before 1.5x', 'after 1.5x']
        for i, c in enumerate(cells):
            if c:
                sheet.paste(c, (i * S, y))
            d.text((i * S + 4, y + S + 3), labels[i], fill=(220, 220, 220))
    sheet.save(out)


if __name__ == '__main__':
    main(*sys.argv[1:4])
