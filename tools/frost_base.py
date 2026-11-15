"""Measure a build's rough surfaces against roughness by sampling alone, settled.

usage: python tools/frost_base.py <tag> [exe] [--ref-exe exe] [--quick]

For glass and a mirror on the gyroid and glass on Thomas, at each roughness: one frame and a settled
picture as the build draws them, against the random version (roughRef=1) left to settle; grain; level;
two frames of a turning view one after the other. Pictures go to out/frost_base/<tag>/ and the
reference pictures to out/frost_base/ref/ (made once, reused by later runs).
"""
import json
import os
import sys

import numpy as np

import frost_lib as fl

argv = sys.argv[1:]
QUICK = '--quick' in argv
REF_EXE = None
if '--ref-exe' in argv:
    i = argv.index('--ref-exe')
    REF_EXE = os.path.abspath(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if not a.startswith('--')]
TAG = args[0]
EXE = os.path.abspath(args[1]) if len(args) > 1 else fl.EXE
OUT = os.path.join(fl.ROOT, 'out', 'frost_base', TAG)
REF = os.path.join(fl.ROOT, 'out', 'frost_base', 'ref')
REF_FRAMES = 4096

CASES = [
    ('gyroid_glass', fl.GYROID),
    ('gyroid_mirror', fl.GYROID + ' material=mirror'),
    ('thomas_glass', fl.THOMAS),
]
if not QUICK:
    CASES += [('gyroid_gold', fl.GYROID + ' material=gold'), ('gem_glass', fl.GEM), ('cube_glass', fl.CUBE)]
ROUGH = [0.15, 0.3, 0.6, 1.0]

rows = []
for name, where in CASES:
    for r in ROUGH:
        s = '%s rough=%g' % (where, r)
        tag = '%s_%03d' % (name, round(r * 100))
        ref_base = os.path.join(REF, tag)
        if os.path.exists(ref_base + '.hdr'):
            ref = fl.load(ref_base + '.hdr')
        else:
            ref = fl.still(ref_base, s + ' roughRef=1', frames=REF_FRAMES, exe=REF_EXE or EXE)
        ref2_base = ref_base + '_b'
        if os.path.exists(ref2_base + '.hdr'):
            ref2 = fl.load(ref2_base + '.hdr')
        else:
            ref2 = fl.still(ref2_base, s + ' roughRef=1', frames=REF_FRAMES // 4, exe=REF_EXE or EXE, extra=['seed=3'])
        one = fl.still(os.path.join(OUT, tag + '_one'), s, frames=1, exe=EXE)
        settled = fl.still(os.path.join(OUT, tag + '_settled'), s, frames=256, exe=EXE)
        t30 = fl.still(os.path.join(OUT, tag + '_t30'), s, frames=None, exe=EXE, extra=['turn=30'])
        t31 = fl.still(os.path.join(OUT, tag + '_t31'), s, frames=None, exe=EXE, extra=['turn=31'])
        t31s = fl.still(os.path.join(OUT, tag + '_t31s'), s, frames=256, exe=EXE, extra=['turn=31'])
        m = fl.on(ref) & fl.on(one)
        row = {
            'case': name, 'rough': r,
            'ref_floor': fl.grey(ref2, ref, m),
            'one_ref': fl.grey(one, ref, m), 'settled_ref': fl.grey(settled, ref, m),
            'one_settled': fl.grey(one, settled, m),
            'grain_one': fl.grain(one), 'grain_settled': fl.grain(settled), 'grain_ref': fl.grain(ref),
            'level_one': fl.level(one) / fl.level(ref), 'level_settled': fl.level(settled) / fl.level(ref),
            'turn_step': fl.grey(t30, t31), 'turn_settled': fl.grey(t31, t31s),
            'level_turn': fl.level(t31) / fl.level(t31s),
        }
        rows.append(row)
        print('%-14s rough %.2f | one-ref %5.2f  settled-ref %5.2f  (ref floor %4.2f) | grain one %5.2f settled %5.2f ref %5.2f | level one x%.3f settled x%.3f | turning: step %5.2f, from settled %5.2f, level x%.3f'
              % (name, r, row['one_ref'], row['settled_ref'], row['ref_floor'], row['grain_one'], row['grain_settled'], row['grain_ref'], row['level_one'], row['level_settled'], row['turn_step'], row['turn_settled'], row['level_turn']), flush=True)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'numbers.json'), 'w') as f:
    json.dump(rows, f, indent=1)
