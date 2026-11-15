"""A quick look while working: the live way at several roughnesses (one frame each) over the random
way settled, on one shape, as one sheet.

usage: python tools/frost_look.py <tag> [shape] [material settings] [--ref] [--frames N]

shape: gyroid (default), thomas, cube, gem, trefoil. The reference row is made once per shape and
material (out/frost_look/ref_*) unless --ref asks for it again. Prints the measures beside it.
"""
import os
import sys

import frost_lib as fl
from frost_sheet import sheet

argv = sys.argv[1:]
REDO = '--ref' in argv
FRAMES = 2048
if '--frames' in argv:
    i = argv.index('--frames')
    FRAMES = int(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if not a.startswith('--')]
tag = args[0]
shape = args[1] if len(args) > 1 else 'gyroid'
extra = ' '.join(args[2:])
where = {'gyroid': fl.GYROID, 'thomas': fl.THOMAS, 'cube': fl.CUBE, 'gem': fl.GEM, 'trefoil': fl.TREFOIL}[shape]
key = shape + ('_' + extra.replace(' ', '_').replace('=', '') if extra else '')
OUT = os.path.join(fl.ROOT, 'out', 'frost_look')
items, lower = [], []
for r in (0.15, 0.3, 0.45, 0.6, 0.8, 1.0):
    s = "%s %s rough=%g" % (where, extra, r)
    sref = s + " nb=12"
    ref_base = os.path.join(OUT, 'ref_%s_%03d' % (key, round(r * 100)))
    if REDO or not os.path.exists(ref_base + '.hdr'):
        ref = fl.still(ref_base, sref + " roughRef=1", frames=FRAMES)
    else:
        ref = fl.load(ref_base + '.hdr')
    base = os.path.join(OUT, '%s_%s_%03d' % (tag, key, round(r * 100)))
    one = fl.still(base, s, frames=1)
    m = fl.on(ref) & fl.on(one)
    blurred = fl.grey(fl.box(one, 2), fl.box(ref, 2), m)
    print('%s rough %.2f: one frame from the random way settled %5.2f grey levels (%5.2f after both are blurred over 5 px); grain %5.2f (random settled %5.2f); level x%.3f'
          % (key, r, fl.grey(one, ref, m), blurred, fl.grain(one), fl.grain(ref), fl.level(one) / fl.level(ref)), flush=True)
    items.append(('%s %g, one frame' % (tag, r), base + '.png'))
    lower.append(('random, %d frames' % FRAMES, ref_base + '.png'))
out = os.path.join(OUT, '%s_%s.png' % (tag, key))
sheet(out, 6, items + lower, height=300)
print(out)
