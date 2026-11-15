"""Pictures of the bodies that scatter light inside (jade, wax, milk glass and their kin), for looking at.

usage: python tools/scatter_sheets.py [exe] [out folder] [which ...]
       (defaults: target/release/glassforms.exe, out/materials/scatter; which: sheets pairs walk others)

Writes into the out folder:
  jade.png, wax.png, milk.png   each material on thin tubes (the Thomas attractor), on a solid (the
                                tanglecube) and on a cut stone, lit from the front and lit from behind
  others.png                    alabaster, porcelain, celadon, pearl, opalescent glass, rose quartz, honey,
                                cloudy ice, front and back
  pair_*.png                    a photograph of the real material beside a picture from this program lit in
                                a like way (the photographs are in docs/refs, sources in scatter-sources.txt;
                                these sheets hold other people's photographs: they are for comparing, not
                                for publishing)
  walk.png                      the clean way (one frame) beside light followed as it wanders (settled)

Run it through the GPU queue: python tools/gpu_run.py --name scatter -- python tools/scatter_sheets.py
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw

args = [a for a in sys.argv[1:]]
EXE = os.path.abspath(args[0] if args and args[0].endswith('.exe') else os.path.join('target', 'release', 'glassforms.exe'))
rest = [a for a in args if not a.endswith('.exe')]
OUT = os.path.abspath(rest[0] if rest and rest[0] not in ('sheets', 'pairs', 'walk', 'others') else os.path.join('out', 'materials', 'scatter'))
WHICH = [a for a in rest if a in ('sheets', 'pairs', 'walk', 'others')] or ['sheets', 'pairs', 'walk', 'others']
TMP = os.path.join(OUT, 'single')
os.makedirs(TMP, exist_ok=True)
REFS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'refs'))

THOMAS = 'scene=attractors preset=thomas dist=4.3'
CUBE = 'scene=algebraic preset=tanglecube dist=5.4'
GEM = 'scene=gems preset=brilliant'
RING = 'scene=solids preset=ring dist=5.0'
SHELL = 'scene=forms preset=seashell dist=5.0'
BALLS = 'scene=forms preset=metaballs dist=5.0'
PEBBLE = 'scene=forms preset=superpebble dist=5.0'
# a soft lamp high on the left in front, a weaker one low on the right, a little light from all round
FRONT = 'rig=215,40,30,30,6,6,6:120,10,25,35,1.2,1.2,1.2 ambient=0.05,0.05,0.05'
# a broad soft light straight behind the shape, and a little light from all round
BACK = 'rig=0,8,45,45,5,5,5 ambient=0.04,0.04,0.04'
n = [0]


def shot(settings, w=420, h=420, frames=32, name=None):
    n[0] += 1
    path = os.path.join(TMP, (name or '%03d' % n[0]) + '.png')
    r = subprocess.run([EXE, '--still', path, 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames] + settings.split(), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('%s\n%s' % (settings, (r.stderr or r.stdout)[-800:]))
    return path


def sheet(name, cols, items, size=420):
    """items: (label, settings, frames)."""
    rows = (len(items) + cols - 1) // cols
    im = Image.new('RGB', (cols * size, rows * (size + 16)), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, (label, settings, frames) in enumerate(items):
        x, y = (i % cols) * size, (i // cols) * (size + 16)
        im.paste(Image.open(shot(settings, size, size, frames)).convert('RGB'), (x, y))
        d.text((x + 4, y + size + 2), label, fill=(230, 230, 230))
    im.save(os.path.join(OUT, name))
    print('wrote', os.path.join(OUT, name), im.size, flush=True)


def pair(name, photo, crop, label, settings, frames=48, height=640):
    """A photograph (cropped to `crop` = left, top, right, bottom as shares of it, or None) beside ours."""
    ph = Image.open(os.path.join(REFS, photo)).convert('RGB')
    if crop:
        ph = ph.crop((int(crop[0] * ph.width), int(crop[1] * ph.height), int(crop[2] * ph.width), int(crop[3] * ph.height)))
    ph = ph.resize((max(1, round(ph.width * height / ph.height)), height), Image.LANCZOS)
    ours = Image.open(shot(settings, height, height, frames, name='pair_' + name)).convert('RGB')
    im = Image.new('RGB', (ph.width + ours.width, height + 16), (0, 0, 0))
    im.paste(ph, (0, 0))
    im.paste(ours, (ph.width, 0))
    d = ImageDraw.Draw(im)
    d.text((4, height + 2), 'photograph: ' + photo, fill=(230, 230, 230))
    d.text((ph.width + 4, height + 2), 'this program: ' + label, fill=(230, 230, 230))
    im.save(os.path.join(OUT, 'pair_' + name + '.png'))
    print('wrote', os.path.join(OUT, 'pair_' + name + '.png'), im.size, flush=True)


if 'sheets' in WHICH:
    for m in ('jade', 'wax', 'milk'):
        items = []
        for lit, rig in (('lit from the front', FRONT), ('lit from behind', BACK)):
            for what, where in (('thin tubes', THOMAS), ('a solid', CUBE), ('a cut stone', GEM)):
                items.append(('%s, %s, %s' % (m, what, lit), '%s material=%s %s' % (where, m, rig), 32))
        sheet(m + '.png', 3, items)

if 'others' in WHICH:
    items = []
    for m in ('alabaster', 'porcelain', 'celadon', 'pearl', 'opal', 'rosequartz', 'honey', 'cloudyice'):
        items.append((m + ', lit from the front', '%s material=%s %s' % (CUBE, m, FRONT), 32))
        items.append((m + ', lit from behind', '%s material=%s %s' % (CUBE, m, BACK), 32))
    sheet('others.png', 4, items, size=360)

if 'pairs' in WHICH:
    # jade with the light behind it: a thin-walled bowl on a light box; ours, a thin-walled shell and a ring
    pair('jade_backlit', 'scatter-jade-bowl-backlit-1.jpg', (0.08, 0.1, 0.92, 0.9), 'jade, a hollow shell, a broad light behind',
         '%s material=jade rig=0,0,50,50,6,6,6 ambient=0.03,0.03,0.03 rx=20' % SHELL)
    pair('jade_backlit_ring', 'scatter-jade-slab-backlit-2.jpg', None, 'jade, a ring, a broad light behind', '%s material=jade %s' % (RING, BACK))
    # jade lit from the front: a carved urn under soft studio light; ours, the tanglecube under a soft lamp
    pair('jade_frontlit', 'scatter-jade-urn-frontlit-grey.jpg', (0.08, 0.15, 0.92, 0.7), 'jade, a solid, a soft lamp in front',
         '%s material=jade rig=215,40,35,35,5,5,5:120,10,25,35,1.5,1.5,1.5:0,80,40,40,1.5,1.5,1.5 ambient=0.08,0.08,0.08' % CUBE)
    pair('jade_frontlit_bowl', 'scatter-jade-bowl-frontlit-1.jpg', (0.05, 0.2, 0.95, 0.85), 'jade, a hollow shell, soft light from the front and above',
         '%s material=jade rig=200,45,40,40,5,5,5:0,85,45,45,3,3,3:120,10,25,35,1,1,1 ambient=0.08,0.08,0.08' % SHELL)
    # wax against a window: daylight from behind; ours, a lump of wax with a broad light behind
    pair('wax_backlit', 'scatter-wax-candle-backlit-window.jpg', (0.1, 0.1, 0.9, 0.75), 'wax, a lump, a broad light behind and a little from the front',
         '%s material=wax rig=0,10,60,50,5,5,5:180,30,40,40,0.5,0.5,0.5 ambient=0.05,0.05,0.05' % BALLS)
    pair('wax_frontlit', 'scatter-wax-candle-unlit-frontlit.jpg', (0.1, 0.1, 0.9, 0.9), 'wax, a pebble, soft daylight from the front and upper right',
         '%s material=wax rig=150,40,40,40,5,5,5:240,10,30,40,1.2,1.2,1.2 ambient=0.1,0.1,0.1' % PEBBLE)
    # milk glass on black, soft light from the front and above; ours, a shell
    pair('milk_frontlit', 'scatter-milkglass-compote-frontlit-black.jpg', (0.15, 0.02, 0.85, 0.98), 'milk glass, a hollow shell, soft light from the front and upper left',
         '%s material=milk rig=215,45,40,40,5,5,5:130,15,30,40,1.5,1.5,1.5 ambient=0.06,0.06,0.06' % SHELL)
    pair('milk_backlit', 'scatter-porcelain-plate-held-to-sky-backlit.jpg', None, 'milk glass, a hollow shell, a broad light behind', '%s material=milk %s' % (SHELL, BACK))

if 'walk' in WHICH:
    items = []
    for m in ('jade', 'milk'):
        for what, where in (('a solid', CUBE), ('thin tubes', THOMAS)):
            for lit, rig in (('front', FRONT), ('behind', BACK)):
                items.append(('%s, %s, lit from %s: clean, one frame' % (m, what, lit), '%s material=%s %s' % (where, m, rig), 1))
                items.append(('the same, light followed as it wanders, 600 frames', '%s material=%s sssWalk=1 %s' % (where, m, rig), 600))
    sheet('walk.png', 4, items, size=360)
