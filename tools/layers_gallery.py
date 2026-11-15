"""Pictures of layers at work, into out/layers: every operation on a tube with an equation shape, hard and
smooth; a metaball-like blend; a morph at several amounts; three layers with their junction filled and
thinned; the three seam profiles; a solid cut by a tube.

usage: python tools/layers_gallery.py [name ...]        (from the folder that holds target/; no names = all)"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'layers')
os.makedirs(OUT, exist_ok=True)

TUBE = ['scene=knots', 'preset=trefoil', 'thick=0.8']
SOLID = ['layer=algebraic:tanglecube', 'L2.size=0.62']
HEART = ['layer=algebraic:heart', 'L2.size=0.75']
THREE = ['scene=knots', 'preset=trefoil', 'thick=0.6', 'layer=algebraic:heart', 'L2.size=0.6', 'L2.x=-0.25', 'layer=algebraic:tanglecube', 'L3.size=0.45', 'L3.x=0.35', 'L3.y=0.1',
         'J12.radius=0.1', 'J13.radius=0.1', 'J23.radius=0.1', 'dist=4.6']

PICTURES = {
    # each operation, hard and smooth
    'join_hard': TUBE + SOLID + ['J12.radius=0'],
    'join_smooth': TUBE + SOLID + ['J12.radius=0.12'],
    'join_metaball': TUBE + SOLID + ['J12.radius=0.45'],
    'cut_solid_by_tube_hard': TUBE + SOLID + ['J12.op=1cuts2', 'L1.show=0', 'J12.radius=0', 'L2.size=0.9'],
    'cut_solid_by_tube_smooth': TUBE + SOLID + ['J12.op=1cuts2', 'L1.show=0', 'J12.radius=0.05', 'L2.size=0.9'],
    'cut_solid_by_tube_with_clearance': TUBE + SOLID + ['J12.op=1cuts2', 'J12.radius=0.03', 'J12.gap=0.05', 'L2.size=0.9'],
    'cut_tube_by_solid_hard': TUBE + SOLID + ['J12.op=2cuts1', 'L2.show=0', 'J12.radius=0'],
    'cut_tube_by_solid_smooth': TUBE + SOLID + ['J12.op=2cuts1', 'L2.show=0', 'J12.radius=0.04'],
    'overlap_hard': TUBE + SOLID + ['J12.op=overlap', 'J12.radius=0', 'L2.size=0.9', 'thick=1.6'],
    'overlap_smooth': TUBE + SOLID + ['J12.op=overlap', 'J12.radius=0.06', 'L2.size=0.9', 'thick=1.6'],
    'apart': TUBE + SOLID + ['J12.op=apart'],
    # a morph from the tube into the solid
    'morph_00': TUBE + HEART + ['J12.op=morph', 'J12.amount=0'],
    'morph_25': TUBE + HEART + ['J12.op=morph', 'J12.amount=0.25'],
    'morph_50': TUBE + HEART + ['J12.op=morph', 'J12.amount=0.5'],
    'morph_75': TUBE + HEART + ['J12.op=morph', 'J12.amount=0.75'],
    'morph_100': TUBE + HEART + ['J12.op=morph', 'J12.amount=1'],
    # the seam's profile
    'profile_smooth': ['scene=knots', 'preset=trefoil', 'thick=0.7'] + HEART + ['J12.radius=0.2', 'J12.profile=smooth', 'dist=4.2'],
    'profile_round': ['scene=knots', 'preset=trefoil', 'thick=0.7'] + HEART + ['J12.radius=0.2', 'J12.profile=round', 'dist=4.2'],
    'profile_bevel': ['scene=knots', 'preset=trefoil', 'thick=0.7'] + HEART + ['J12.radius=0.2', 'J12.profile=bevel', 'dist=4.2'],
    # three layers: the junction as it comes, filled, thinned
    'three_junction_plain': THREE,
    'three_junction_filled': THREE + ['J123.fill=1', 'J123.reach=0.3'],
    'three_junction_thinned': THREE + ['J123.fill=-1', 'J123.reach=0.3'],
    'three_hard': [a for a in THREE if 'radius' not in a] + ['J12.radius=0', 'J13.radius=0', 'J23.radius=0'],
    # thin tubes melting into a solid, and four layers
    'attractor_in_solid': ['scene=attractors', 'preset=thomas', 'layer=add', 'J12.radius=0.2'],
    'four_layers': ['scene=attractors', 'preset=thomas', 'layer=add', 'layer=knots:trefoil', 'L3.size=0.8', 'L3.thick=0.7', 'layer=surfaces:schwarzp', 'L4.size=0.45', 'L4.x=-0.55',
                    'J13.op=3cuts1', 'J13.gap=0.02', 'J13.radius=0.02', 'J24.radius=0.2', 'J14.radius=0.15', 'J34.radius=0.1'],
    # cut stones as layers: one cut by a tube, one joined to a knot, one set in a knot as a body of its own,
    # and what the window's `+` gives on a tube
    'gem_cut_by_tube': ['scene=gems', 'preset=brilliant', 'layer=knots:trefoil', 'J12.op=2cuts1', 'L2.show=0', 'J12.radius=0.02', 'L2.thick=0.5', 'L2.size=0.9'],
    'gem_joined_to_knot': ['scene=knots', 'preset=trefoil', 'layer=gems:brilliant', 'L2.size=0.55', 'J12.radius=0.1'],
    'gem_in_knot_own_body': ['scene=knots', 'preset=trefoil', 'layer=gems:brilliant', 'L2.size=0.55', 'J12.op=apart'],
    'added_layer_default': ['scene=attractors', 'preset=thomas', 'layer=add'],
    # in-motion detail (256 voxels a side): what is seen while a seam animates
    'join_smooth_in_motion_detail': TUBE + SOLID + ['J12.radius=0.12', 'res=256', 'L2.res=256'],
}

names = sys.argv[1:] or list(PICTURES)
for name in names:
    path = os.path.join(OUT, name + '.png')
    r = subprocess.run([EXE, '--still', path, 'w=1280', 'h=720', 'frames=96'] + PICTURES[name], capture_output=True, text=True, cwd=ROOT)
    print(name, 'ok' if r.returncode == 0 else (r.stderr or r.stdout).strip()[:300], flush=True)
