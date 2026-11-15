"""The pictures of shadows and caustics in out/cast (docs/cast.md refers to them by name).

    python tools/cast_gallery.py [name ...]        all of them, or only those named

The pictures named `own-...` show bodies each of its own material, drawn as what they are (glass or gold
over porcelain or jade). Those named `caustic-...` show the light alone: every body is drawn as clay, and
GLASSFORMS_CAST_KINDS says how each is to cast (see cast.rs). Those named `floor-...` show one body over a
plain floor that stands in for the room's (GLASSFORMS_CAST_FLOOR). The lamp in most of them is one small
bright light, because that is what makes a shadow or a caustic one can see: a family's own lights are
large soft boxes all round the shape.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'cast')
REFS = os.path.join(os.path.dirname(ROOT), 'docs', 'refs', 'cast')

LAMP = 'rig=140,40,5,5,400,380,360,0.5:250,10,25,40,1.2,1.4,2.0,0.7 ambient=0.05,0.06,0.08'
SLAB = 'scene=solids preset=superellipsoid p.p=8 p.q=8 p.tall=0.2'
BALL = 'layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 J12.op=apart'
SUN = 'ambient=0.22,0.24,0.28 exposure=0.5 bloom=0.15 ior=1.5 absorb=0,0,0'
TOP = 'L1.y=-0.5 rx=90 ry=0 rz=0 dist=4.2 exposure=0.4 bloom=0.1 ambient=0.08,0.09,0.1'

# one small bright lamp added to a family's own lights (the seventh light of the older families)
KEY = 'light7=150,55,2,2,700,680,650,0.5'

# name: (arguments, how the bodies cast or None, [(suffix, more arguments), ...], the stand-in floor or nothing)
PICTURES = {
    'floor-ball': ('scene=solids preset=superellipsoid p.p=2 p.q=2 L1.size=0.7 castSoft=0.3 rig=150,50,2,2,900,880,850,0.5:300,20,20,30,0.6,0.7,0.9,0.7 '
                   'ambient=0.05,0.06,0.08 dist=6.5 w=1280 h=800', None, [('on', 'cast=1'), ('shadow-only', 'cast=1 castCaustic=0'), ('off', 'cast=0')], '-1.2,40,4'),
    'floor-thomas': (f'scene=attractors preset=thomas {KEY} castSoft=0.3 dist=7.5 w=1280 h=800', None, [('on', 'cast=1'), ('lights-turned', 'cast=1 lyaw=120')], '-1.35,38,4.5'),
    'floor-trefoil': (f'scene=knots preset=trefoil {KEY} castSoft=0.3 dist=7.5 w=1280 h=800', None, [('on', 'cast=1')], '-1.3,38,4.5'),
    'floor-gyroid': (f'scene=surfaces preset=gyroid {KEY} castSoft=0.3 dist=7.5 w=1280 h=800', None, [('on', 'cast=1')], '-1.3,38,4.5'),
    'floor-brilliant': ('scene=gems preset=brilliant castSoft=0.5 dist=7 w=1280 h=800', None, [('on', 'cast=1')], '-1.0,40,4'),
    'floor-tumbler': ('scene=solids preset=superellipsoid p.p=2 p.q=8 p.tall=1 L1.size=0.5 castSoft=0.3 rig=123,35,1.5,1.5,1200,1150,1100,0.5:300,20,20,30,0.5,0.6,0.8,0.7 '
                      'ambient=0.05,0.06,0.08 ior=1.5 disp=0.04 absorb=0,0,0 rx=55 ry=0 rz=0 dist=6.5 w=1280 h=800', None, [('on', 'cast=1')], '-0.45,55,4'),
    'floor-gold-ring': ('scene=solids preset=ring p.fat=0.12 p.section=8 p.flat=2.2 L1.size=0.8 material=gold castSoft=0.3 rig=123,35,1.5,1.5,1200,1150,1000,0.5:300,20,20,30,0.5,0.6,0.8,0.7 '
                        'ambient=0.05,0.06,0.08 rx=60 ry=0 rz=0 dist=6.5 w=1280 h=800', None, [('on', 'cast=1')], '-0.25,60,4'),
    'floor-jade': (f'scene=knots preset=trefoil material=jade {KEY} castSoft=0.3 dist=7.5 w=1280 h=800', None, [('on', 'cast=1')], '-1.3,38,4.5'),
    'own-glass-ball': (f'{SLAB} p.tall=0.3 material=porcelain {BALL} L2.y=0.36 L2.size=0.3 B2.material=crown B2.disp=0.12 L1.y=-0.5 rx=70 ry=0 rz=0 dist=4.4 '
                       'rig=200,25,3,3,300,300,300,0.5:320,30,25,35,1.5,1.7,2.0,0.7 ambient=0.08,0.09,0.1 castSoft=0.3 w=1280 h=900', None,
                       [('off', 'cast=0'), ('shadow-only', 'cast=1 castCaustics=0'), ('on', 'cast=1'), ('lights-turned', 'cast=1 lyaw=70')], None),
    'own-gold-on-jade': (f'{SLAB} p.tall=0.3 material=jade layer=solids:icosahedron J12.op=apart L2.x=0.25 L2.y=0.42 L2.size=0.3 B2.material=gold L1.y=-0.5 rx=70 ry=0 rz=0 dist=4.4 '
                         'rig=200,25,3,3,300,300,300,0.5:320,30,25,35,1.5,1.7,2.0,0.7 ambient=0.08,0.09,0.1 castSoft=0.3 w=1280 h=900', None,
                         [('off', 'cast=0'), ('on', 'cast=1'), ('lights-turned', 'cast=1 lyaw=70')], None),
    'own-gem': ('scene=gems preset=brilliant L1.y=0.02 L1.size=0.4 L1.rx=180 layer=solids:superellipsoid L2.p.p=8 L2.p.q=8 L2.p.tall=0.2 J12.op=apart L2.y=-0.62 B2.material=porcelain '
                'rx=62 ry=0 rz=0 dist=4.2 castSoft=0.5 w=1280 h=900', None, [('off', 'cast=0'), ('on', 'cast=1')], None),
    'own-gold-ring': (f'{SLAB} L1.y=-0.45 material=porcelain layer=solids:ring L2.p.fat=0.12 L2.p.section=8 L2.p.flat=2.2 J12.op=apart L2.size=0.55 L2.y=-0.14 B2.material=gold rx=72 ry=0 rz=0 '
                      'rig=123,9,1.5,1.5,1200,1150,1000,0.5 ambient=0.22,0.24,0.28 exposure=0.5 bloom=0.15 dist=4.6 fov=28 castSoft=0.3 w=1280 h=900', None, [('on', 'cast=1'), ('off', 'cast=0')], None),
    'own-tumbler': (f'{SLAB} L1.y=-0.45 material=porcelain layer=solids:superellipsoid L2.p.p=2 L2.p.q=8 L2.p.tall=1 J12.op=apart L2.size=0.3 L2.x=0.35 L2.y=-0.01 B2.material=crown B2.disp=0.04 rx=70 ry=0 rz=0 '
                    'rig=123,11,1.5,1.5,1200,1150,1100,0.5 ambient=0.22,0.24,0.28 exposure=0.5 bloom=0.15 dist=5 fov=28 castSoft=0.3 w=1280 h=900', None, [('on', 'cast=1'), ('off', 'cast=0')], None),
    'shadow-thomas': (f'scene=attractors preset=thomas clay=1 {LAMP}', None, [('off', 'cast=0'), ('on', 'cast=2 castSoft=1')], None),
    'shadow-two-bodies': ('scene=solids preset=superellipsoid layer=solids:icosahedron J12.op=apart L2.x=0.86 L2.y=0.5 L2.z=0.44 L2.size=0.28 L1.size=0.7 clay=1 rx=0 ry=0 rz=0 '
                          'rig=117,27,5,5,400,380,360,0.5:250,10,25,40,1.2,1.4,2.0,0.7 ambient=0.05,0.06,0.08', None, [('off', 'cast=0'), ('on', 'cast=2 castSoft=1')], None),
    'shadow-jade': (f'scene=attractors preset=thomas material=jade {LAMP}', None, [('off', 'cast=0'), ('on', 'cast=1 castSoft=1')], None),
    'shadow-soft': (f'scene=attractors preset=thomas clay=1 {LAMP} cast=2', None, [('0', 'castSoft=0'), ('035', 'castSoft=0.35'), ('1', 'castSoft=1')], None),
    'caustic-ball': (f'{SLAB} p.tall=0.3 {BALL} L2.y=0.36 L2.size=0.3 clay=1 {TOP} rig=200,25,3,3,300,300,300,0.5 disp=0.15 castSoft=0.3 w=1280 h=960', 'matte,clear',
                     [('off', 'cast=0'), ('shadow-only', 'cast=2 castCaustics=0'), ('on', 'cast=2'), ('lights-turned-60', 'cast=2 lyaw=60'), ('lights-turned-140', 'cast=2 lyaw=140')], None),
    'caustic-cylinder': (f'{SLAB} L1.y=-0.45 layer=solids:superellipsoid L2.p.p=2 L2.p.q=8 L2.p.tall=1 J12.op=apart L2.size=0.3 L2.x=0.35 L2.y=-0.01 clay=1 rx=70 ry=0 rz=0 '
                         f'rig=123,11,1.5,1.5,1200,1150,1100,0.5 {SUN} dist=5 fov=28 disp=0.04 castSoft=0.3 w=1280 h=900', 'matte,clear', [('on', 'cast=2'), ('off', 'cast=0')], None),
    'caustic-lens': (f'{SLAB} L1.y=-0.55 {BALL} L2.p.tall=0.3 L2.size=0.25 L2.rz=55 L2.x=-0.72 L2.y=0.13 clay=1 rx=90 ry=0 rz=0 dist=4.4 exposure=0.4 bloom=0.2 '
                     'rig=235,0,3,3,300,300,300,0.5 ambient=0.08,0.09,0.1 disp=0.12 ior=1.5 absorb=0,0,0 castSoft=0.2 w=1280 h=960', 'matte,clear', [('on', 'cast=2')], None),
    'caustic-knot': (f'{SLAB} layer=knots:trefoil J12.op=apart L1.y=-0.55 L1.x=0.25 L2.y=-0.12 L2.x=-0.45 L2.size=0.42 L2.rx=90 clay=1 rx=90 ry=0 rz=0 dist=4.6 exposure=0.4 bloom=0.1 '
                     'rig=215,10,3,3,300,300,300,0.5 ambient=0.08,0.09,0.1 disp=0.1 castSoft=0.3 w=1280 h=960', 'matte,clear', [('on', 'cast=2')], None),
    'caustic-gem': ('scene=gems preset=brilliant L1.y=0.02 L1.size=0.4 L1.rx=180 layer=solids:superellipsoid L2.p.p=8 L2.p.q=8 L2.p.tall=0.2 J12.op=apart L2.y=-0.62 clay=1 '
                    'rx=62 ry=0 rz=0 dist=4.2 exposure=0.45 bloom=0.2 castSoft=0.5 w=1280 h=900', 'clear,matte', [('on', 'cast=2'), ('off', 'cast=0')], None),
    'caustic-ring': (f'{SLAB} L1.y=-0.45 layer=solids:ring L2.p.fat=0.12 L2.p.section=8 L2.p.flat=2.2 J12.op=apart L2.size=0.55 L2.y=-0.14 clay=1 rx=72 ry=0 rz=0 '
                     f'rig=123,9,1.5,1.5,1200,1150,1000,0.5 {SUN} dist=4.6 fov=28 castSoft=0.3 w=1280 h=900', 'matte,metal', [('on', 'cast=2'), ('off', 'cast=0')], None),
    'caustic-metal': (f'{SLAB} p.tall=0.3 {BALL} L2.y=0.3 L2.x=-0.3 L2.size=0.28 clay=1 {TOP} rig=235,10,3,3,300,300,300,0.5 castSoft=0.3 w=1280 h=960', 'matte,metal', [('on', 'cast=2')], None),
}


def render(name, args, kinds, more, suffix, floor):
    out = os.path.join(OUT, f'{name}-{suffix}.png')
    env = dict(os.environ)
    env.pop('GLASSFORMS_CAST_KINDS', None)
    env.pop('GLASSFORMS_CAST_FLOOR', None)
    if kinds:
        env['GLASSFORMS_CAST_KINDS'] = kinds
    if floor:
        env['GLASSFORMS_CAST_FLOOR'] = floor
    size = [] if ' w=' in args else ['w=1280', 'h=720']
    r = subprocess.run([EXE, '--still', out, *args.split(), *more.split(), 'spin=0', 'frames=32', *size], capture_output=True, text=True, cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(f'{name}-{suffix}: {r.stderr.strip()[-800:]}')
    print(out, flush=True)
    return out


def beside(out, paths, height=560):
    """Pictures put side by side at one height, for looking at together."""
    from PIL import Image
    ims = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        ims.append(im.resize((round(im.width * height / im.height), height), Image.LANCZOS))
    sheet = Image.new('RGB', (sum(i.width for i in ims) + 8 * (len(ims) - 1), height), (0, 0, 0))
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width + 8
    sheet.save(out)
    print(out, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    wanted = [a for a in sys.argv[1:] if not a.startswith('-')] or list(PICTURES)
    made = {}
    for name in wanted:
        args, kinds, list_, floor = PICTURES[name]
        for suffix, more in list_:
            made[f'{name}-{suffix}'] = render(name, args, kinds, more, suffix, floor)
    # a real glass of water in the sun beside the glass cylinder (on a floor, and on a matte slab)
    if 'floor-tumbler-on' in made and os.path.exists(os.path.join(REFS, 'caustic-water-glass.jpg')):
        beside(os.path.join(OUT, 'compare-water-glass.png'), [os.path.join(REFS, 'caustic-water-glass.jpg'), made['floor-tumbler-on']])
    if 'caustic-cylinder-on' in made and os.path.exists(os.path.join(REFS, 'caustic-water-glass.jpg')):
        beside(os.path.join(OUT, 'compare-water-glass-slab.png'), [os.path.join(REFS, 'caustic-water-glass.jpg'), made['caustic-cylinder-on']])
    # a real gold ring in the sun beside the gold ring (on a floor, and drawn as clay on a matte slab)
    if 'floor-gold-ring-on' in made and os.path.exists(os.path.join(REFS, 'caustic-gold-ring.jpg')):
        beside(os.path.join(OUT, 'compare-gold-ring.png'), [os.path.join(REFS, 'caustic-gold-ring.jpg'), made['floor-gold-ring-on']])
    if 'caustic-ring-on' in made and os.path.exists(os.path.join(REFS, 'caustic-gold-ring.jpg')):
        beside(os.path.join(OUT, 'compare-gold-ring-slab.png'), [os.path.join(REFS, 'caustic-gold-ring.jpg'), made['caustic-ring-on']])
    # the same two photographs beside bodies of their own materials: glass and gold over porcelain
    if 'own-tumbler-on' in made and os.path.exists(os.path.join(REFS, 'caustic-water-glass.jpg')):
        beside(os.path.join(OUT, 'compare-water-glass-own.png'), [os.path.join(REFS, 'caustic-water-glass.jpg'), made['own-tumbler-on']])
    if 'own-gold-ring-on' in made and os.path.exists(os.path.join(REFS, 'caustic-gold-ring.jpg')):
        beside(os.path.join(OUT, 'compare-gold-ring-own.png'), [os.path.join(REFS, 'caustic-gold-ring.jpg'), made['own-gold-ring-on']])
    if 'own-glass-ball-on' in made:
        beside(os.path.join(OUT, 'own-glass-ball-steps.png'), [made['own-glass-ball-off'], made['own-glass-ball-shadow-only'], made['own-glass-ball-on']], 480)
    if 'own-gold-on-jade-on' in made:
        beside(os.path.join(OUT, 'own-gold-on-jade-pair.png'), [made['own-gold-on-jade-off'], made['own-gold-on-jade-on']], 480)
    if 'own-gem-on' in made:
        beside(os.path.join(OUT, 'own-gem-pair.png'), [made['own-gem-off'], made['own-gem-on']], 480)
    if 'floor-ball-on' in made:
        beside(os.path.join(OUT, 'floor-ball-steps.png'), [made['floor-ball-off'], made['floor-ball-shadow-only'], made['floor-ball-on']], 400)
    for name in ('shadow-thomas', 'shadow-two-bodies', 'shadow-jade'):
        if f'{name}-on' in made:
            beside(os.path.join(OUT, f'{name}-pair.png'), [made[f'{name}-off'], made[f'{name}-on']], 480)
    if 'caustic-ball-on' in made:
        beside(os.path.join(OUT, 'caustic-ball-steps.png'), [made['caustic-ball-off'], made['caustic-ball-shadow-only'], made['caustic-ball-on']], 480)


main()
