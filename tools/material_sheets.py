"""The materials' contact sheets: every named material on two shapes, and each part of the material on
its own, for looking at.

usage: python tools/material_sheets.py [exe] [out folder] [only=<part of a sheet's name>]
       (default: target/release/glassforms.exe, out/materials, every sheet)

Writes into the out folder:
  named_thomas.png, named_solid.png, named_gem.png   every named material, on the Thomas attractor, on the
                                      tanglecube and on a brilliant-cut stone
  bubble.png                          the soap bubble beside what the reference pictures show: a film draining
                                      from even to black at the top, flowing, and films over opaque bodies
  roughness.png                       frost at four values on glass, on a mirror, and under a clear coat
  film.png                            a thin film at six thicknesses, then uneven, drained, on a metal
  metals.png                          every metal, polished and satin
  milky.png                           jade, wax and milk glass: estimated beside true scattering, thin and solid
  inside.png                          glaze over two bodies, bubbles from few to many and their shapes, glow, the costly modes
  bodies.png                          scenes of several bodies, each of its own material: one body inside another,
                                      bodies pushed half into each other, a film-coated shell with a gold cavity
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

ONLY = next((a[5:] for a in sys.argv[1:] if a.startswith('only=')), '')
args = [a for a in sys.argv[1:] if not a.startswith('only=')]
EXE = os.path.abspath(args[0] if args else os.path.join('target', 'release', 'glassforms.exe'))
OUT = os.path.abspath(args[1] if len(args) > 1 else os.path.join('out', 'materials'))
TMP = os.path.join(OUT, 'single')
os.makedirs(TMP, exist_ok=True)

THOMAS = 'scene=attractors preset=thomas dist=4.3'
GYROID = 'scene=surfaces preset=gyroid dist=5.2'
CUBE = 'scene=algebraic preset=tanglecube dist=5.4'
GEM = 'scene=gems preset=brilliant'
BALL = 'scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2'
n = [0]


def shot(settings, size=300, frames=64):
    n[0] += 1
    path = os.path.join(TMP, '%03d.png' % n[0])
    r = subprocess.run([EXE, '--still', path, 'w=%d' % size, 'h=%d' % size, 'frames=%d' % frames] + settings.split(), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('%s\n%s' % (settings, r.stderr))
    return path


def sheet(name, cols, items, size=300):
    """items: (label, settings, frames) or None for an empty place."""
    if ONLY not in name:
        return
    rows = (len(items) + cols - 1) // cols
    im = Image.new('RGB', (cols * size, rows * (size + 16)), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, it in enumerate(items):
        if it is None:
            continue
        label, settings, frames = it
        x, y = (i % cols) * size, (i // cols) * (size + 16)
        im.paste(Image.open(shot(settings, size, frames)).convert('RGB'), (x, y))
        d.text((x + 4, y + size + 2), label, fill=(230, 230, 230))
    im.save(os.path.join(OUT, name))
    print('wrote', os.path.join(OUT, name), im.size, flush=True)


mats = json.loads(subprocess.run([EXE, '--list-materials'], capture_output=True, text=True, check=True).stdout)
# (the materials that follow light as it wanders are given more frames to settle)
for name, where in (('named_thomas.png', THOMAS), ('named_solid.png', CUBE), ('named_gem.png', GEM)):
    sheet(name, 10, [(m['id'], '%s material=%s' % (where, m['id']), 256 if m['id'].startswith('true') else 48) for m in mats], size=240)

sheet('roughness.png', 5, [(('glass, ' if i == 0 else '') + 'frost %.2f' % r, '%s rough=%.2f' % (GYROID, r), 160) for i, r in enumerate((0.0, 0.15, 0.3, 0.6, 1.0))]
      + [(('mirror, ' if i == 0 else '') + 'frost %.2f' % r, '%s material=mirror rough=%.2f' % (GYROID, r), 96) for i, r in enumerate((0.0, 0.15, 0.3, 0.6, 1.0))]
      + [('gold, frost 0.5', GYROID + ' material=gold rough=0.5', 96), ('the same under a clear coat', GYROID + ' material=gold rough=0.5 coat=1', 96),
         ('frosted glass on Thomas', THOMAS + ' material=frosted', 160), ('satin glass on Thomas', THOMAS + ' material=satin', 160),
         ('frost 0.6 by sampling alone, 1024 frames', GYROID + ' rough=0.6 roughRef=1', 1024)])

film = 'film=%d filmIor=1.8 absorb=6,6,6'
sheet('film.png', 6, [('dark glass, film %d nm' % d, CUBE + ' ' + film % d, 48) for d in (0, 100, 180, 260, 360, 520)]
      + [('360 nm, uneven', CUBE + ' ' + film % 360 + ' filmVar=0.5', 48), ('uneven, finer pattern', CUBE + ' ' + film % 360 + ' filmVar=0.5 filmScale=3', 48),
         ('uneven, flow at 0.5', CUBE + ' ' + film % 360 + ' filmVar=0.5 filmFlow=0.5', 48), ('drained (thin at the top)', CUBE + ' ' + film % 360 + ' filmDrain=0.8', 48),
         ('soap bubble', GYROID + ' material=soap', 64), ('soap bubble on Thomas', THOMAS + ' material=soap', 64),
         ('oil on glass', CUBE + ' material=oil', 48), ('dichroic glass', CUBE + ' material=dichroic', 48), ('carnival glass', CUBE + ' material=carnival', 48),
         ('anodised titanium, 60 nm oxide', GYROID + ' material=titanium film=60 filmIor=2.4', 48), ('the same, 95 nm', GYROID + ' material=titanium film=95 filmIor=2.4', 48),
         ('the same, 140 nm, uneven', GYROID + ' material=titanium film=140 filmIor=2.4 filmVar=0.4', 48)])

sheet('bubble.png', 6, [('soap film, drain %.2f' % d, '%s material=soap filmDrain=%.2f' % (BALL, d), 64) for d in (0.0, 0.2, 0.4, 0.7, 1.0)]
      + [('drain 1, film 250 nm', BALL + ' material=soap filmDrain=1 film=250', 64)]
      + [('flow %.2f' % f, '%s material=soap filmFlow=%.2f' % (BALL, f), 64) for f in (0.0, 0.33, 0.66)]
      + [('no sky (lamps only)', BALL + ' material=soap filmSky=0', 64), ('soap bubble on Thomas', THOMAS + ' material=soap', 64), ('soap bubble on the gyroid', GYROID + ' material=soap', 64),
         ('oil slick', BALL + ' material=oilslick', 64), ('lustre glaze', BALL + ' material=lustre', 64), ('beetle shell', BALL + ' material=beetle', 64),
         ('heat-coloured steel', BALL + ' material=heatsteel', 64), ('bismuth', BALL + ' material=bismuth', 64), ('oil on chrome', BALL + ' material=oilchrome', 64),
         ('film on clay', BALL + ' clay=1 film=320 filmIor=1.9 filmVar=0.4', 32), ('film on jade', BALL + ' material=jade film=300 filmIor=1.9 filmVar=0.4', 64),
         ('film on a cut stone', GEM + ' material=oil', 64), ('anodised titanium', BALL + ' material=anodised', 64), ('pearl', BALL + ' material=pearl', 64), ('dichroic glass', BALL + ' material=dichroic', 64)])

metals = ['mirror', 'silver', 'aluminium', 'gold', 'copper', 'chrome', 'nickel', 'platinum', 'titanium', 'iron', 'tungsten']
sheet('metals.png', 11, [(m, '%s material=%s' % (GYROID, m), 64) for m in metals] + [(m + ', frost 0.45', '%s material=%s rough=0.45' % (GYROID, m), 96) for m in metals], size=220)

milky = []
for where, tag in ((CUBE, 'solid'), (THOMAS, 'tubes')):
    for m in ('jade', 'wax', 'milk'):
        milky.append(('%s, estimated, %s' % (m, tag), '%s material=%s' % (where, m), 64))
        milky.append(('%s, true scattering, %s' % (m, tag), '%s material=%s sssWalk=1' % (where, m), 512))
milky += [('milkiness 0.1', CUBE + ' sss=0.1 fit=1', 64), ('milkiness 0.3', CUBE + ' sss=0.3 fit=1', 64), ('milkiness 1', CUBE + ' sss=1 fit=1', 64),
          ('light travels 0.05', CUBE + ' sss=1 sssDepth=0.05 fit=1 bodyR=0.9 bodyG=0.6 bodyB=0.4', 64), ('light travels 0.2', CUBE + ' sss=1 sssDepth=0.2 fit=1 bodyR=0.9 bodyG=0.6 bodyB=0.4', 64),
          ('light travels 0.6', CUBE + ' sss=1 sssDepth=0.6 fit=1 bodyR=0.9 bodyG=0.6 bodyB=0.4', 64)]
sheet('milky.png', 6, milky)

sheet('inside.png', 6, [
    ('milky body, frost 0.5', CUBE + ' material=milk rough=0.5', 64), ('the same, glazed (porcelain)', CUBE + ' material=porcelain', 64),
    ('gold, frost 0.5', CUBE + ' material=gold rough=0.5', 96), ('the same, glazed', CUBE + ' material=gold rough=0.5 coat=1', 96),
    ('black lacquer', CUBE + ' material=lacquer', 64), ('pearl', CUBE + ' material=pearl', 64),
    ('bubbles 0.03', CUBE + ' bubbles=0.03 bubbleSize=0.045', 64), ('bubbles 0.2', CUBE + ' bubbles=0.2 bubbleSize=0.045', 64), ('bubbles 0.9', CUBE + ' bubbles=0.9 bubbleSize=0.045', 64),
    ('uneven (evenness 0)', CUBE + ' bubbles=0.3 bubbleSize=0.05 bubbleEven=0', 64), ('needles (shape 1)', CUBE + ' bubbles=0.3 bubbleSize=0.012 bubbleShape=1', 64),
    ('discs (shape -0.8)', CUBE + ' bubbles=0.2 bubbleSize=0.03 bubbleShape=-0.8', 64),
    ('glow 3, even', THOMAS + ' glow=3 glowCore=0', 48), ('glow 3, core 0.8', THOMAS + ' glow=3 glowCore=0.8', 48), ('neon tube, more bloom', THOMAS + ' material=neon bloom=0.9', 48),
    ('calcite (double refraction)', CUBE + ' material=calcite', 64), ('graded index 0.35', CUBE + ' material=gradient', 64), ('plain glass, for comparison', CUBE, 64),
    ('thin wall (hollow glass)', THOMAS + ' hollow=1', 64), ('thin wall, tinted', CUBE + ' hollow=1 density=30', 64), ('thin wall with a glowing gas', THOMAS + ' hollow=1 glow=3', 64),
    ('clay', CUBE + ' clay=1', 32), ('clay under a clear coat', CUBE + ' clay=1 coat=1', 32), ('frosted glass under the garden picture', GYROID + ' rough=0.5 hdri=hdri/suburban_garden_4k.exr', 96),
])

# several bodies, each of its own material (the second body's is given as B2.<setting>)
KNOT = 'scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.op=apart'
BALL2 = BALL + ' layer=solids:superellipsoid L2.p.p=2 L2.p.q=2'
# a ball with a hollow cut into it, and in the hollow a thin gold lining (a second ball with its own hollow)
SHELL = (BALL + ' layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 L2.size=0.5 L2.x=-0.25 L2.y=0.2 L2.z=0.55 J12.op=2cuts1'
         ' layer=solids:superellipsoid L3.p.p=2 L3.p.q=2 L3.size=0.54 L3.x=-0.25 L3.y=0.2 L3.z=0.55'
         ' layer=solids:superellipsoid L4.p.p=2 L4.p.q=2 L4.size=0.47 L4.x=-0.28 L4.y=0.224 L4.z=0.616 J34.op=4cuts3'
         ' J13.op=apart J14.op=apart J23.op=apart J24.op=apart')
sheet('bodies.png', 4, [
    ('the scene as one material', KNOT, 64), ('glass knot, gold heart', KNOT + ' B2.material=gold', 64),
    ('glass knot, satin gold heart', KNOT + ' B2.material=satingold', 64), ('glass knot, jade heart', KNOT + ' B2.material=jade', 64),
    ('gold knot, glass heart', KNOT + ' material=gold B2.material=crown', 64), ('glass knot, pearl heart', KNOT + ' B2.material=pearl', 64),
    ('glass ball, a ruby stone inside', BALL + ' layer=gems:brilliant L2.size=0.5 J12.op=apart B2.material=ruby', 64),
    ('soap film, gold inside', BALL + ' material=soap layer=solids:dodecahedron L2.size=0.5 J12.op=apart B2.material=gold', 64),
    ('glass ball, a gold cube pushed half in', BALL + ' layer=solids:cube L2.size=0.5 L2.x=0.9 J12.op=apart B2.material=gold', 64),
    ('crown glass and diamond, interpenetrating', BALL2 + ' material=crown L1.size=0.75 L1.x=-0.4 L2.size=0.75 L2.x=0.4 J12.op=apart B2.material=diamond', 64),
    ('jade, a copper stone poking out', CUBE + ' material=jade layer=gems:brilliant L2.size=0.6 L2.y=0.55 J12.op=apart B2.material=copper', 64),
    ('three bodies: glass, glass joined to it, gold', 'scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.radius=0.15 layer=algebraic:tanglecube L3.size=0.5 L3.x=0.5 L3.y=0.15 J13.op=apart J23.op=apart B3.material=gold', 64),
    ('film on chrome, a gold cavity', SHELL + ' material=chrome film=330 filmIor=1.5 filmSky=0.6 B3.material=gold', 64),
    ('glass, a gold cavity', SHELL + ' B3.material=gold', 64),
    ('frosted glass ball, a neon heart inside', BALL + ' rough=0.35 layer=algebraic:heart L2.size=0.55 J12.op=apart B2.material=neon', 64),
    ('jade ball, an anodised stone poking out', BALL + ' material=jade layer=gems:brilliant L2.size=0.7 L2.y=0.7 J12.op=apart B2.material=anodised', 64),
], size=360)
