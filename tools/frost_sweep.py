"""A sweep of rough surfaces over the cases that have to work, one frame each, as one sheet to look at:
thin tubes, glass seen through glass, a cut stone, metals, a shape in mid-animation, a moving view, an
environment picture behind the shape, focus, layers, and roughness over other materials.

usage: python tools/frost_sweep.py [tag]
"""
import os
import sys

import frost_lib as fl
from frost_sheet import sheet

tag = sys.argv[1] if len(sys.argv) > 1 else 'sweep'
OUT = os.path.join(fl.ROOT, 'out', 'frost_sweep', tag)
GARDEN = 'hdri=hdri/suburban_garden_4k.exr'
CASES = [
    ('Thomas, frosted', fl.THOMAS + ' material=frosted', []),
    ('Thomas, satin', fl.THOMAS + ' material=satin', []),
    ('gyroid, frosted', fl.GYROID + ' material=frosted', []),
    ('cut stone, frosted', fl.GEM + ' material=frosted', []),
    ('cut stone, satin', fl.GEM + ' material=satin', []),
    ('cut stone, satin gold', fl.GEM + ' material=satingold', []),
    ('Thomas, satin gold', fl.THOMAS + ' material=satingold', []),
    ('gyroid, satin steel', fl.GYROID + ' material=brushed', []),
    ('gyroid, mirror frost 1', fl.GYROID + ' material=mirror rough=1', []),
    ('trefoil 3.5 s into its animation, frosted', fl.TREFOIL + ' material=frosted anim=3.5', []),
    ('gyroid frosted, turning (the 30th frame)', fl.GYROID + ' material=frosted', ['turn=30']),
    ('Thomas satin, turning (the 30th frame)', fl.THOMAS + ' material=satin', ['turn=30']),
    ('frosted under the garden picture, shown behind', fl.GYROID + ' material=frosted hdriShow=1 ' + GARDEN, []),
    ('satin gold under the garden picture', fl.CUBE + ' material=satingold ' + GARDEN, []),
    ('frosted with focus (aperture 0.2)', fl.THOMAS + ' material=frosted aperture=0.2 fnear=-0.4 ffar=-0.1', []),
    ('two layers joined, frosted', 'scene=knots preset=trefoil layer=algebraic:heart L2.size=0.7 J12.radius=0.15 material=frosted', []),
    ('jade (milky, frost 0.1)', fl.CUBE + ' material=jade', []),
    ('wax (milky, frost 0.32)', fl.CUBE + ' material=wax', []),
    ('pearl (coat, film, frost)', fl.GYROID + ' material=pearl', []),
    ('porcelain (frost under a coat)', fl.CUBE + ' material=porcelain', []),
    ('frost 0.5 with a thin film', fl.CUBE + ' rough=0.5 film=300 filmVar=0.4', []),
    ('frost 0.4, no dispersion', fl.GYROID + ' rough=0.4 disp=0', []),
    ('frosted thin wall', fl.GYROID + ' hollow=1 rough=0.5', []),
    ('anodised titanium (film on rough metal)', fl.CUBE + ' material=anodised', []),
]
items = []
for i, (name, s, extra) in enumerate(CASES):
    base = os.path.join(OUT, '%02d' % i)
    try:
        frames = None if extra else 1
        a = fl.still(base, s.replace('bloom=0', ''), w=640, h=360, frames=frames, extra=extra)
        import numpy as np
        ok = np.isfinite(a).all()
        print('%-48s level %.3f  grain %5.2f  brightest %6.1f%s' % (name, fl.level(a), fl.grain(a), a[..., :3].max(), '' if ok else '   NOT FINITE'), flush=True)
    except RuntimeError as e:
        print('%-48s FAILED: %s' % (name, str(e)[-300:]), flush=True)
    items.append((name, base + '.png'))
out = os.path.join(fl.ROOT, 'out', 'frost_sweep', tag + '.png')
sheet(out, 4, items, height=270)
print(out)
