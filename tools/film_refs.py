"""The thin film beside the three reference pictures, and on the other kinds of body.

usage: python tools/film_refs.py [exe] [--only bubble|draining|shell|bodies|slider]

Writes into out/materials/film/ (from the folder that holds target/release/glassforms.exe):
  ref_bubble.png      the soap bubble reference | this program
  ref_draining.png    the draining film reference (six moments) | this program at six places of `filmDrain`
  ref_shell.png       the coated shell reference | this program
  bodies.png          the film on a solid dark body, on a cut stone, on metals, on the Thomas attractor and the gyroid
  slider.png          `filmDrain` from 0 to 1 in twelve steps, and `filmFlow` through one turn
The single pictures are kept beside them in single/. The set-ups are the settings below, nothing more:
every picture is one still of the program. The environment pictures the set-ups use are made by
tools/film_pictures.py (run here if they are missing).

What each set-up is, and why:
  bubble    a ball as a film around air, an even 490 nm, lit by the sample garden picture with its
            ground in shade and its sky whitened, shown against black, the picture not softened. The
            reference's bubble hung in front of something dark with a white sky all round; its ground
            and trees are dark shapes. (Its colours are those of one thickness seen at every angle:
            teal straight on, then blue, magenta, yellow, and green at the rim.)
  draining  a ball cut by a plane into a dome, a film around air 1500 nm thick when fresh, in a studio
            (soft boxes, black elsewhere), `filmDrain` at 0.12, 0.28, 0.46, 0.7, 0.8, 0.95. The room is
            pitched 38 degrees, which leans the film's "up" towards the camera: in the reference the
            thin spot sits three quarters of the way up the dome, facing the camera, not on its top.
  shell     a ball hollowed by a smaller ball and opened by a third, nickel under a film of index 2,
            435 nm, in a room papered in checks with a window, the picture shown behind it.
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw

args = [a for a in sys.argv[1:] if not a.startswith('--')]
ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
if ONLY in args:
    args.remove(ONLY)
EXE = os.path.abspath(args[0] if args else os.path.join('target', 'release', 'glassforms.exe'))
OUT = os.path.abspath(os.path.join('out', 'materials', 'film'))
ONE = os.path.join(OUT, 'single')
ENV = os.path.join(OUT, 'env')
REFS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'refs'))
os.makedirs(ONE, exist_ok=True)

if not all(os.path.exists(os.path.join(ENV, n)) for n in ('garden_shade.hdr', 'studio.hdr', 'room.hdr')):
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'film_pictures.py'), ENV], check=True)


def pic(name):
    return 'hdri=' + os.path.join(ENV, name).replace('\\', '/')


BALL = 'scene=solids preset=superellipsoid p.p=2 p.q=2 rx=0 ry=0 rz=0 spin=0'
BUBBLE = BALL + ' dist=7.0 material=soap film=490 filmDrain=0 filmVar=0.05 filmScale=1.5 ' + pic('garden_shade.hdr') + ' hdriBlur=0 hdriGain=4 exposure=0.8 bloom=0.15 lyaw=110'
# a ball cut by a cube's face into a dome whose foot is at the camera's height
DOME = ('scene=solids preset=superellipsoid p.p=2 p.q=2 L1.y=-0.26 layer=solids:cube L2.y=-1.39 L2.size=2 J12.op=2cuts1 J12.radius=0 L2.show=0 select=1 '
        'rx=0 ry=0 rz=0 spin=0 fov=12 dist=5.73 shift=0,0.667 material=soap film=1500 filmVar=0.04 filmScale=1.2 '
        + pic('studio.hdr') + ' hdriBlur=0 hdriGain=3 exposure=1.25 bloom=0.1 lpitch=38')
DRAINS = (0.12, 0.28, 0.46, 0.70, 0.80, 0.95)
SHELL = ('scene=solids preset=superellipsoid p.p=2 p.q=2 layer=solids:superellipsoid L2.p.p=2 L2.p.q=2 L2.size=0.9 J12.op=2cuts1 J12.radius=0 L2.show=0 '
         'layer=solids:superellipsoid L3.p.p=2 L3.p.q=2 L3.size=0.62 L3.x=0.72 L3.y=0.5 L3.z=0.78 J13.op=3cuts1 J13.radius=0.02 L3.show=0 J23.op=apart select=1 '
         'rx=0 ry=0 rz=0 spin=0 dist=6.0 material=nickel film=435 filmIor=2.0 filmSky=0 nb=8 ' + pic('room.hdr') + ' hdriShow=1 hdriBlur=0 hdriGain=1.6 lpitch=25 bloom=0.1')
THOMAS = 'scene=attractors preset=thomas dist=4.3'
GYROID = 'scene=surfaces preset=gyroid dist=5.2'
CUBE = 'scene=algebraic preset=tanglecube dist=5.4'
GEM = 'scene=gems preset=brilliant'


def shot(name, settings, w, h, frames=64):
    path = os.path.join(ONE, name + '.png')
    r = subprocess.run([EXE, '--still', path, 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames] + settings.split(), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('%s\n%s' % (settings, r.stderr[-800:]))
    return Image.open(path).convert('RGB')


def beside(name, ref, ours, labels=('reference', 'this program')):
    """The reference and ours at the same height, side by side, a caption under each."""
    h = max(ref.height, ours.height)
    a = ref.resize((round(ref.width * h / ref.height), h), Image.LANCZOS)
    b = ours.resize((round(ours.width * h / ours.height), h), Image.LANCZOS)
    im = Image.new('RGB', (a.width + b.width + 12, h + 18), (0, 0, 0))
    im.paste(a, (0, 0))
    im.paste(b, (a.width + 12, 0))
    d = ImageDraw.Draw(im)
    d.text((4, h + 3), labels[0], fill=(210, 210, 210))
    d.text((a.width + 16, h + 3), labels[1], fill=(210, 210, 210))
    im.save(os.path.join(OUT, name))
    print('wrote', os.path.join(OUT, name), im.size, flush=True)


def sheet(name, cols, items, size, frames=64):
    rows = (len(items) + cols - 1) // cols
    im = Image.new('RGB', (cols * size[0], rows * (size[1] + 16)), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, (label, settings) in enumerate(items):
        x, y = (i % cols) * size[0], (i // cols) * (size[1] + 16)
        im.paste(shot('%s_%02d' % (name[:-4], i), settings, size[0], size[1], frames), (x, y))
        d.text((x + 4, y + size[1] + 2), label, fill=(230, 230, 230))
    im.save(os.path.join(OUT, name))
    print('wrote', os.path.join(OUT, name), im.size, flush=True)


def want(what):
    return ONLY in (None, what)


if want('bubble'):
    ref = Image.open(os.path.join(REFS, 'thinfilm-soap-bubble.png')).convert('RGB')
    beside('ref_bubble.png', ref, shot('bubble', BUBBLE, 2 * ref.width, 2 * ref.height, 128))

if want('draining'):
    ref = Image.open(os.path.join(REFS, 'thinfilm-draining-over-time.png')).convert('RGB').crop((352, 92, 730, 1082))
    frames = [shot('draining_%d' % i, '%s filmDrain=%.2f' % (DOME, x), 760, 330, 96) for i, x in enumerate(DRAINS)]
    col = Image.new('RGB', (760, 330 * len(frames)), (0, 0, 0))
    for i, f in enumerate(frames):
        col.paste(f, (0, 330 * i))
    beside('ref_draining.png', ref, col, ('reference: a film draining, top to bottom', 'this program: filmDrain = ' + ', '.join('%.2f' % x for x in DRAINS)))

if want('shell'):
    ref = Image.open(os.path.join(REFS, 'thinfilm-shell-with-gold-interior.png')).convert('RGB')
    beside('ref_shell.png', ref, shot('shell', SHELL, 2 * ref.width, 2 * ref.height, 128))

if want('bodies'):
    sheet('bodies.png', 4, [
        ('oil on a solid dark body', CUBE + ' material=oil'),
        ('the same under the garden picture', CUBE + ' material=oil hdri=hdri/suburban_garden_4k.exr exposure=1.6'),
        ('an oil slick on a black body', CUBE + ' material=oilslick'),
        ('beetle shell', CUBE + ' material=beetle'),
        ('a coated cut stone (a hard film on topaz)', GEM + ' ior=1.62 disp=0.014 film=330 filmIor=2.3 filmVar=0.08 filmSky=1.5'),
        ('oil on a dark cut stone', GEM + ' material=oil'),
        ('anodised titanium', GYROID + ' material=anodised'),
        ('bismuth', CUBE + ' material=bismuth'),
        ('heat-coloured steel', CUBE + ' material=heatsteel'),
        ('oil on chrome', GYROID + ' material=oilchrome'),
        ('a soap film on the Thomas attractor', THOMAS + ' material=soap'),
        ('a soap film on the gyroid', GYROID + ' material=soap'),
        ('Thomas under the garden picture', THOMAS + ' material=soap hdri=hdri/suburban_garden_4k.exr hdriGain=4 exposure=1.2'),
        ('the gyroid under the garden picture', GYROID + ' material=soap hdri=hdri/suburban_garden_4k.exr hdriGain=4 exposure=1.2'),
        ('a soap bubble, as it comes', BALL + ' dist=5.6 material=soap'),
        ('carnival glass', CUBE + ' material=carnival'),
    ], (400, 400))

if want('slider'):
    b = BALL + ' dist=5.6 material=soap'
    sheet('slider.png', 6, [('filmDrain %.2f' % (i / 11), '%s filmDrain=%.3f' % (b, i / 11)) for i in range(12)]
          + [('filmFlow %.2f' % (i / 6), '%s filmFlow=%.3f' % (b, i / 6)) for i in range(6)], (300, 300), 48)
