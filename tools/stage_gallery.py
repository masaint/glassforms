"""Pictures of the studio (backdrop, floor, camera height, tone curves), as sheets to look at.

usage: python tools/stage_gallery.py [name ...]        (from D:\\glassforms\\native; no names = every sheet)

Writes out\\stage\\<name>.png. Each sheet is a few stills side by side with what each is written under it.
Sheets: light (plain glass, a gem and a frosted body on a light backdrop, without and with the floor),
horizon, through (the floor seen through glass), turning (a shape turning on its turntable), focus (the
band in focus laid near, middle and far across a floor), tone (the four curves on one picture), grade,
studios, height (the camera raised), reach (how far round and how high a backdrop goes), floors (matte to
glossy to mirroring)."""
import os, subprocess, sys
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'stage')
TMP = os.path.join(OUT, 'parts')
os.makedirs(TMP, exist_ok=True)
W, H = 640, 360


def still(name, *args, frames=96, size=(W, H)):
    path = os.path.join(TMP, name + '.png')
    r = subprocess.run([EXE, '--still', path, 'w=%d' % size[0], 'h=%d' % size[1], 'frames=%d' % frames, *[str(a) for a in args]], capture_output=True, text=True, cwd=NATIVE, timeout=900)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return Image.open(path).convert('RGB')


def sheet(name, cells, columns):
    """cells: (picture, what is written under it)."""
    w, h = cells[0][0].size
    rows = (len(cells) + columns - 1) // columns
    out = Image.new('RGB', (columns * w, rows * (h + 18)), (24, 24, 28))
    draw = ImageDraw.Draw(out)
    for i, (im, text) in enumerate(cells):
        x, y = (i % columns) * w, (i // columns) * (h + 18)
        out.paste(im, (x, y))
        draw.text((x + 6, y + h + 3), text, fill=(225, 225, 230))
    out.save(os.path.join(OUT, name + '.png'))
    print(name + '.png', flush=True)


THOMAS = ['scene=attractors', 'preset=thomas']
GEM = ['scene=gems', 'preset=brilliant']
CUBE = ['scene=solids', 'preset=cube', 'rx=0', 'ry=30', 'rz=0']
RING = ['scene=solids', 'preset=ring', 'rx=0', 'ry=0', 'rz=0']
KNOT = ['scene=knots', 'preset=trefoil']


def light():
    shapes = [('plain glass', THOMAS, []), ('a gem', GEM, ['rx=-35', 'ry=20', 'rz=0']), ('a frosted body', RING, ['rough=0.3'])]
    cells = []
    for floor in (0, 1):
        for label, shape, extra in shapes:
            args = [*shape, *extra, 'studio=lavender', 'floor=%d' % floor, 'elev=%d' % (22 if floor else 0)]
            cells.append((still('light-%d-%s' % (floor, label.replace(' ', '-')), *args), '%s, %s' % (label, 'on the floor, camera raised 22' if floor else 'backdrop only')))
    sheet('light', cells, 3)


def horizon():
    cells = [(still('horizon-%d' % e, *CUBE, 'studio=lavender', 'elev=%d' % e, 'dist=9', *extra), text) for e, extra, text in [
        (4, [], 'camera raised 4: floor and backdrop meet with no line'),
        (12, [], 'camera raised 12'),
        (4, ['floorMatch=0', 'floorGain=0.5'], 'a floor lit by the lights (floorMatch=0), darker than the backdrop: it melts into it'),
        (4, ['floorMatch=0', 'floorGain=0.5', 'floorFade=6'], 'the same, the melt nearer (floorFade=6)')]]
    sheet('horizon', cells, 2)


def through():
    base = [*CUBE, 'studio=ice', 'elev=28', 'floorR=1', 'floorG=0.55', 'floorB=0.5']
    cells = [
        (still('through-clear', *base), 'a pink floor seen beside and through a clear cube'),
        (still('through-air', *base, 'ior=1', 'disp=0', 'absorb=0,0,0'), 'glass that bends nothing (index 1): the floor through it is the floor beside it'),
        (still('through-frost', *base, 'rough=0.3'), 'through frosted glass'),
        (still('through-mirror', *base, 'metal=1'), 'in a mirror'),
    ]
    sheet('through', cells, 2)


def turning():
    args = [*KNOT, 'studio=dusk', 'elev=20', 'spinMode=1', 'rx=70', 'rz=0', 'smooth=0.9']
    cells = [(still('turning-%d' % n, *args, 'drift=spin:3', frames=n), 'after %d frames of turning, 3 degrees a frame: one frame as the window shows it' % n) for n in (10, 20, 30, 40)]
    sheet('turning', cells, 2)


def focus():
    base = [*CUBE, 'studio=dusk', 'elev=16', 'floorGloss=1', 'dist=7.5']
    cells = [(still('focus-none', *base), 'no lens')]
    for label, near, far in [('near: on the floor in front of the cube', -1.9, -1.5), ('middle: on the cube', -0.3, 0.3), ('far: on the floor behind it', 1.6, 2.2)]:
        cells.append((still('focus-%s' % label.split(':')[0], *base, 'aperture=0.3', 'fnear=%g' % near, 'ffar=%g' % far), 'aperture 0.3, the band ' + label))
    sheet('focus', cells, 2)


def tone():
    names = ['0 filmic (as ever)', '1 neutral', '2 soft', '3 none']
    base = [*THOMAS, 'studio=lavender', 'elev=18', 'exposure=1.25']
    cells = [(still('tone-%d' % t, *base, 'tone=%d' % t), 'tone=' + names[t]) for t in range(4)]
    sheet('tone', cells, 2)


def grade():
    base = [*THOMAS, 'studio=lavender', 'elev=18']
    rows = [([], 'as the studio sets it'), (['contrast=1.4'], 'contrast=1.4'), (['saturation=1.6'], 'saturation=1.6'), (['saturation=0'], 'saturation=0'),
            (['wbTemp=0.6'], 'wbTemp=0.6 (warmer)'), (['wbTemp=-0.6'], 'wbTemp=-0.6 (cooler)'), (['wbTint=0.6'], 'wbTint=0.6 (magenta)'), (['wbTint=-0.6'], 'wbTint=-0.6 (green)')]
    sheet('grade', [(still('grade-%d' % i, *base, *extra), text) for i, (extra, text) in enumerate(rows)], 4)


def studios():
    cells = []
    for s in ('lavender', 'ice', 'white', 'dusk'):
        for label, shape in (('Thomas', THOMAS), ('a cube', CUBE)):
            cells.append((still('studio-%s-%s' % (s, label[-4:].strip()), *shape, 'studio=' + s, 'elev=18', 'dist=7.5'), 'studio=%s, %s' % (s, label)))
    sheet('studios', cells, 4)


def height():
    cells = [(still('height-%d' % e, *RING, 'studio=lavender', 'elev=%d' % e, 'dist=7.5'), 'elev=%d' % e) for e in (0, 15, 35, 60, 85)]
    cells.append((still('height-roll', *RING, 'studio=lavender', 'elev=35', 'roll=20', 'dist=7.5'), 'elev=35 roll=20'))
    sheet('height', cells, 3)


def reach():
    base = [*THOMAS, 'studio=lavender', 'elev=10']
    rows = [(['backWrap=180', 'backHigh=90'], 'a backdrop all round and overhead: little for the glass to go dark against'),
            (['backWrap=105', 'backHigh=50'], 'backWrap=105 backHigh=50 (the studio): dark behind the camera and overhead'),
            (['backWrap=60', 'backHigh=25'], 'backWrap=60 backHigh=25: a narrow sweep'),
            (['backWrap=180', 'backHigh=90', 'backGlow=1.2', 'backGlowSize=22', 'backGlowEl=6'], 'a glow behind the shape (backGlow=1.2, 22 degrees wide)')]
    sheet('reach', [(still('reach-%d' % i, *base, *extra), text) for i, (extra, text) in enumerate(rows)], 2)


def floors():
    base = [*KNOT, 'studio=dusk', 'elev=14', 'rx=70', 'dist=7.5']
    rows = [(['floorGloss=0', 'floorMirror=0'], 'matte (floorGloss=0)'), (['floorGloss=0.5', 'floorMirror=0'], 'floorGloss=0.5: it mirrors the lights'),
            (['floorGloss=0.5', 'floorMirror=1'], 'floorMirror=1: and the shape'), (['floorGloss=1', 'floorMirror=1'], 'floorGloss=1')]
    sheet('floors', [(still('floors-%d' % i, *base, *extra), text) for i, (extra, text) in enumerate(rows)], 2)


ALL = {'light': light, 'horizon': horizon, 'through': through, 'turning': turning, 'focus': focus, 'tone': tone, 'grade': grade, 'studios': studios, 'height': height, 'reach': reach, 'floors': floors}
for name in (sys.argv[1:] or ALL):
    ALL[name]()
