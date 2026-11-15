"""The camera's pictures: rendered with the program, put on labelled sheets in out/camera/.

usage: python tools/camera_pictures.py [name ...]        (no names: all of them)

  focal     one subject through 24, 50 and 200 mm, the camera moved so that it stays the same size
  fstop     an f-stop ladder with the subject as a 20 mm jewel and as a 2 m sculpture
  modes     the clean blur beside the traced lens at the same camera settings
  lenses    every kind of lens, and straight lines bowed either way
  character colour fringing, fall-off, a leaning plane of focus, a shifted lens
  highlights the shapes of out-of-focus highlights, in both ways of focusing
  closeup   a long lens, the frame slid to a part of the shape, shallow focus
  panorama  a 360 x 180 panorama, and a piece of it read back as an ordinary lens's picture
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'camera')
GARDEN = 'hdri=hdri/suburban_garden_4k.exr'
os.makedirs(os.path.join(OUT, 'parts'), exist_ok=True)


def font(size):
    for name in ('segoeui.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def still(name, args, w=800, h=450, frames=48):
    path = os.path.join(OUT, 'parts', name + '.png')
    r = subprocess.run([EXE, '--still', path, 'w=%d' % w, 'h=%d' % h, 'frames=%d' % frames] + args, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError('%s\n%s' % (' '.join(args), r.stderr.strip()))
    return Image.open(path).convert('RGB')


def camera(args, w=800, h=450):
    r = subprocess.run([EXE, '--camera', 'w=%d' % w, 'h=%d' % h] + args, capture_output=True, text=True, cwd=ROOT)
    return json.loads(r.stdout.strip().splitlines()[-1])


def length(mm):
    if mm is None:
        return 'the horizon'
    return '%.2f mm' % mm if mm < 10 else '%.1f mm' % mm if mm < 100 else '%.1f cm' % (mm / 10) if mm < 1000 else '%.2f m' % (mm / 1000)


def sheet(name, title, rows, cols):
    """rows: lists of (picture, caption) ; a caption may have two lines"""
    w, h = rows[0][0][0].size
    pad, cap, head = 10, 44, 46
    out = Image.new('RGB', (pad + cols * (w + pad), head + len(rows) * (h + cap + pad)), (16, 16, 18))
    d = ImageDraw.Draw(out)
    d.text((pad, 10), title, fill=(235, 235, 235), font=font(22))
    for j, row in enumerate(rows):
        for i, (im, text) in enumerate(row):
            x, y = pad + i * (w + pad), head + j * (h + cap + pad)
            out.paste(im, (x, y))
            for k, line in enumerate(text.split('\n')):
                d.text((x + 2, y + h + 3 + k * 19), line, fill=(215, 215, 215) if k == 0 else (150, 150, 155), font=font(15))
    path = os.path.join(OUT, name + '.png')
    out.save(path)
    print(path)


def focal():
    base = ['scene=solids', 'preset=frame', GARDEN, 'hdriShow=1', 'hdriBlur=0.3']
    row = []
    for mm in (24, 50, 200):
        args = base + ['dollyZoom=%d' % mm]
        c = camera(args)
        row.append((still('focal-%d' % mm, args), '%d mm on full frame · %.0f° across · camera %s away\nsubject the same size: what changes is the perspective, and how much garden is behind it' % (mm, c['fieldH'], length(c['distanceMm']))))
    sheet('focal-lengths', 'One subject at 24, 50 and 200 mm, framed the same size (dollyZoom)', [row], 3)


def fstop():
    rows = []
    for subject, what in ((20, 'a 20 mm jewel'), (2000, 'a 2 m sculpture')):
        row = []
        for n in (2.8, 8, 22):
            args = ['scene=attractors', 'preset=thomas', 'dollyZoom=100', 'subject=%d' % subject, 'fstop=%g' % n, 'focus=-0.35', 'band=0']
            c = camera(args)
            row.append((still('fstop-%d-%g' % (subject, n), args), 'as %s · 100 mm · f/%g · camera %s away · %.2g× life size\nsharp from %s to %s' % (what, n, length(c['distanceMm']), c['magnification'], length(c['sharpFromMm']), length(c['sharpToMm']))))
        rows.append(row)
    sheet('fstops', 'The same lens, framing and f-stops on a small subject and on a large one (clean blur, one plane of focus)', rows, 3)


def modes():
    rows = []
    for shape, focus, lens, what in ((['scene=attractors', 'preset=thomas'], -0.35, [], '85 mm f/4, a 60 mm subject, seven blades'),
                                     (['scene=knots'], 0.2, [], '85 mm f/4, a 60 mm subject, seven blades'),
                                     (['scene=attractors', 'preset=thomas'], 0.0, ['tiltY=50'], 'the same with the plane of focus swung 50°')):
        row = []
        for mode, name, frames in ((1, 'clean blur, one frame', 1), (1, 'clean blur, settled', 64), (0, 'traced lens, 400 frames', 400)):
            args = shape + ['dollyZoom=85', 'fstop=4', 'subject=60', 'focus=%g' % focus, 'band=0', 'dof=%d' % mode, 'blades=7', 'lensN=4'] + lens
            row.append((still('modes-%s-%d-%d-%d' % (shape[-1].split('=')[1], mode, frames, len(lens)), args, frames=frames), '%s\n%s' % (name, what)))
        rows.append(row)
    sheet('focus-modes', 'The two ways of focusing, from the same camera settings', rows, 3)


def lenses():
    base = ['scene=knots', GARDEN, 'hdriShow=1', 'hdriBlur=0.3']
    kinds = [
        ('perspective', [], 'perspective · 52 mm'),
        ('barrel', ['distort=0.3'], 'barrel distortion 0.3'),
        ('pincushion', ['distort=-0.3'], 'pincushion distortion -0.3'),
        ('orthographic', ['proj=orthographic'], 'orthographic (the backdrop is one direction: one colour)'),
        ('fisheye', ['proj=fisheye', 'focal=8', 'dist=3'], 'fisheye, equidistant · 8 mm: 172° top to bottom'),
        ('equisolid', ['proj=equisolid', 'focal=15', 'dist=3'], 'fisheye, equisolid · 15 mm'),
        ('stereographic', ['proj=stereographic', 'focal=12', 'dist=3'], 'fisheye, stereographic · 12 mm'),
        ('cylindrical', ['proj=cylindrical', 'focal=14', 'dist=3'], 'cylindrical · 14 mm'),
        ('panorama', ['proj=panorama', 'dist=3'], 'panorama (the picture is cut from 360° × 180°: a 2:1 one is all of it)'),
    ]
    cells = [(still('lens-' + k, base + a), text) for k, a, text in kinds]
    sheet('lenses', 'Every kind of lens: the same subject and garden', [cells[0:3], cells[3:6], cells[6:9]], 3)


def character():
    base = ['scene=attractors', 'preset=thomas']
    garden = ['scene=knots', GARDEN, 'hdriShow=1', 'hdriBlur=0.3', 'vignette=0']
    kinds = [
        ('plain', garden + ['dollyZoom=24'], '24 mm, no character'),
        ('falloff', garden + ['dollyZoom=24', 'falloff=1'], "fall-off 1: the lens's own darkening of the corners (cos⁴ for this lens)"),
        ('vignette', garden + ['dollyZoom=24', 'vignette=1.2'], 'for comparison: the old added vignette at 1.2'),
        ('fringe0', garden + ['dollyZoom=24', 'fringe=0'], 'the right-hand edge of the picture, enlarged four times: no fringing'),
        ('fringe1', garden + ['dollyZoom=24', 'fringe=1'], 'the same with colour fringing 1: edges split into red and blue'),
        ('distort', garden + ['dollyZoom=24', 'distort=0.2', 'fringe=0.6', 'falloff=0.6'], 'a cheap wide lens: barrel 0.2, fringing 0.6, fall-off 0.6'),
        ('level', base + ['fstop=2.8', 'band=0', 'focus=0'], 'f/2.8, the plane of focus level'),
        ('tilt', base + ['fstop=2.8', 'band=0', 'focus=0', 'tiltY=55'], 'the plane of focus swung 55° (tiltY): sharp along a slant'),
        ('shift', ['scene=solids', 'preset=frame', GARDEN, 'hdriShow=1', 'dollyZoom=24', 'shiftY=0.6'], 'a shift lens: the picture slid up by 0.6 (shiftY), verticals stay parallel'),
    ]
    cells = []
    for k, a, text in kinds:
        im = still('char-' + k, a)
        if k.startswith('fringe'):
            w, h = im.size
            im = im.crop((w - w // 4, h // 4, w, h // 2)).resize((w, h), Image.NEAREST)
        cells.append((im, text))
    sheet('character', 'Lens character and movements', [cells[0:3], cells[3:6], cells[6:9]], 3)


def highlights():
    # a cut stone in front of the plane of focus: every sparkle is a picture of the lens opening
    gem = ['scene=gems', 'preset=brilliant', 'dist=3.6', 'fstop=4', 'subject=100', 'focusDist=220', 'band=0', 'exposure=0.7']
    kinds = [
        ('round', [], 'a round opening, evenly lit'),
        ('blades', ['blades=6', 'bokehRot=15'], 'six blades'),
        ('rim', ['bokeh=1'], 'a bright rim (bokeh 1)'),
        ('core', ['bokeh=-1'], 'a soft core (bokeh -1)'),
        ('squeeze', ['squeeze=2'], 'anamorphic: squeeze 2'),
        ('eye', ['catseye=1'], "cat's-eye 1: cut towards the corners, which are darker for it"),
    ]
    cells = [(still('high-' + k, gem + a, frames=64), text + ' · clean blur') for k, a, text in kinds]
    rows = [cells[:3], cells[3:]]
    # glass tubes with beads, behind the plane of focus, in both ways of focusing: on tubes a highlight
    # lies close to the surface that shows it, and the two agree
    tubes = ['scene=attractors', 'preset=thomas', 'p.b=0.19', 'beads=0.5', 'dist=4.4', 'fstop=2.8', 'subject=100', 'focus=-1.2', 'band=0']
    for mode, label, frames in ((1, 'clean blur', 64), (0, 'traced lens, 400 frames', 400)):
        rows.append([(still('high-tube-%s-%d' % (k, mode), tubes + a + ['dof=%d' % mode, 'lensN=4'], frames=frames), '%s · %s' % (text, label))
                     for k, a, text in (('rim', ['bokeh=1'], 'glass tubes: a bright rim'), ('squeeze', ['squeeze=2'], 'anamorphic: squeeze 2'), ('eye', ['catseye=1', 'blades=6'], "six blades and cat's-eye 1"))])
    sheet('highlights', 'The shapes of out-of-focus highlights: a cut stone through the clean blur; glass tubes through both ways of focusing', rows, 3)


def closeup():
    # a long lens from where the camera stands, the picture slid to a part of the shape, a jewel's size,
    # the lens opened and focused by a click on the glass in the middle of the picture
    base = ['scene=attractors', 'preset=thomas', 'focal=200', 'subject=30', 'band=0', 'shiftX=0.9', 'shiftY=0.4', 'pick=0.5,0.5']
    c = camera(base + ['fstop=4'])
    row = [
        (still('close-f4', base + ['fstop=4']), 'glass, 200 mm f/4 as a 30 mm jewel: %.1f× life size\nthe picture slid across the lens (shiftX 0.9, shiftY 0.4), focus by pick=0.5,0.5' % c['magnification']),
        (still('close-f11', base + ['fstop=11']), 'the same at f/11'),
        (still('close-frost', base + ['fstop=4', 'rough=0.6', GARDEN, 'hdriShow=1'], frames=96), 'frosted glass at f/4, the garden behind (blurred as infinitely far)'),
    ]
    sheet('closeup', 'A close-up: a long lens, the frame slid to a part of the shape, shallow focus', [row], 3)


def panorama():
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    argv, sys.argv = sys.argv, sys.argv[:1]
    import camera_check as cc
    sys.argv = argv
    scene = ['scene=knots', 'dist=3', GARDEN, 'hdriShow=1', 'hdriBlur=0.3', 'bloom=0', 'vignette=0', 'grain=0']
    pano = still('pano', scene + ['proj=panorama'], w=4096, h=2048, frames=32)
    w0, h0, fov0 = 800, 450, 60.0
    ref = still('pano-ref', scene + ['fov=%g' % fov0], w=w0, h=h0, frames=48)
    t0 = np.tan(np.radians(fov0) / 2)
    ys, xs = np.mgrid[0:h0, 0:w0]
    qx, qy = (2 * (xs + 0.5) - w0) / h0, (2 * (h0 - (ys + 0.5)) - h0) / h0
    d = np.stack([qx * t0, qy * t0, -np.ones_like(qx)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    fx, fy = cc.to_film('panorama', np.pi / 2, d)
    back = cc.resample(np.asarray(pano).astype(np.float64), fx, fy)
    diff = float(np.abs(back - np.asarray(ref)).mean())
    back = Image.fromarray(np.clip(np.nan_to_num(back), 0, 255).astype(np.uint8))
    half = pano.resize((w0, w0 // 2))
    wide = Image.new('RGB', (w0, h0), (16, 16, 18))
    wide.paste(half, (0, (h0 - w0 // 2) // 2))
    sheet('panorama', 'A 360° × 180° panorama, and its middle read back as a 60° lens would see it', [[
        (wide, 'proj=panorama, 4096 × 2048 (2:1 = the whole sphere), camera 3 units from the knot'),
        (back, 'the panorama read back through a 60° lens (in Python, by the lens\'s law)\n%.2f grey levels from the picture on the right' % diff),
        (ref, 'the program\'s own picture through a 60° lens from the same place'),
    ]], 3)


if __name__ == '__main__':
    todo = sys.argv[1:] or ['focal', 'fstop', 'modes', 'lenses', 'character', 'highlights', 'closeup', 'panorama']
    for name in todo:
        globals()[name]()
