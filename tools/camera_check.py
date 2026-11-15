"""Checks of the camera: the photographer's settings, the two ways of focusing against the thin-lens formula,
every kind of lens, the lens's character, and that none of it changes what was drawn before.

usage: python tools/camera_check.py [--old <glassforms.exe built before the camera went in>] [--quick] [--only <part>]

Run from anywhere: the program is taken from target/release beside this file's folder. Prints one line per
check starting `ok` or `FAIL`, then a summary; exit status 1 if anything failed. What each bound is and why
is said beside it; docs/camera.md has the model and the measurements. With --old, the pictures that must
not have changed are held to that program bit for bit (without it those checks are left out and said so).
--quick leaves out the timing. --only runs the parts whose name contains the word given (nothing changed,
numbers, blur, the two ways of focusing, any lens = plain, lenses, steady, clicks, character, cat's-eye,
materials, video, cost).

How the blur is measured. A picture in which everything is the same way out of focus is the sharp picture
with every point spread over a disc; spreading a point over a disc of radius c adds c^2 / 4 to the picture's
spread (its brightness-weighted variance) along each axis. So the growth of a picture's variance from the
sharp one to the blurred one is the mean of c^2 / 2 over its light, whatever the picture shows, and that
mean can be worked out beforehand from the sharp picture's own stored distances and the thin-lens formula
in millimetres. The subject is a convex pebble drawn as clay, so that nothing hides or shows through
anything and every pixel is at the distance stored for it.
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'camera_check')
GARDEN = os.path.join(ROOT, 'hdri', 'suburban_garden_4k.exr')
if not os.path.exists(GARDEN):
    GARDEN = os.path.join(os.path.dirname(ROOT), 'native', 'hdri', 'suburban_garden_4k.exr')
BACKDROP = ['hdri=' + GARDEN.replace('\\', '/'), 'hdriShow=1']
OLD = sys.argv[sys.argv.index('--old') + 1] if '--old' in sys.argv else None
if OLD and not os.path.isabs(OLD):
    # (given from the program's folder, or from wherever the script was started)
    OLD = next((p for p in (os.path.join(ROOT, OLD), os.path.abspath(OLD)) if os.path.exists(p)), os.path.join(ROOT, OLD))
ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else ''
QUICK = '--quick' in sys.argv
LENS = {'GLASSFORMS_LENS': '1'}
os.makedirs(OUT, exist_ok=True)

results = []


def check(ok, text):
    results.append(bool(ok))
    print(('ok   ' if ok else 'FAIL ') + text, flush=True)


def guarded(name, f):
    try:
        f()
    except Exception as e:                      # a check that cannot even run has failed
        check(False, '%s: %s' % (name, str(e).strip().splitlines()[-1][:300] if str(e).strip() else repr(e)))


def run(args, exe=EXE, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([exe] + args, capture_output=True, text=True, cwd=ROOT, env=e)
    if r.returncode != 0:
        raise RuntimeError('%s %s\n%s' % (os.path.basename(exe), ' '.join(args), r.stderr.strip()))
    return r.stdout


def hdr(name, args, exe=EXE, env=None):
    """Render a still; returns its picture before bloom and tone map: (h, w, 4) floats, a = distance to the glass."""
    path = os.path.join(OUT, name)
    run(['--still', path + '.png', '--hdr', path + '.hdr'] + args, exe, env)
    raw = np.fromfile(path + '.hdr', dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    return np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(h, w, 4).astype(np.float64)


def png(name, args, exe=EXE, env=None, extra=()):
    """Render a still; returns what is on screen: (h, w, 3) in grey levels 0..255."""
    path = os.path.join(OUT, name + '.png')
    run(['--still', path] + list(extra) + args, exe, env)
    a = np.asarray(Image.open(path))
    if a.dtype == np.uint16:
        return a[..., :3].astype(np.float64) / 257.0
    return a[..., :3].astype(np.float64)


def same_file(a, b):
    with open(os.path.join(OUT, a), 'rb') as f, open(os.path.join(OUT, b), 'rb') as g:
        return f.read() == g.read()


def camera(args):
    return json.loads(run(['--camera'] + args).strip().splitlines()[-1])


def tone(rgb):
    x = np.clip((rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14), 0, 1)
    return np.where(x < 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055) * 255.0


def untone(levels):
    """Light before the tone map from grey levels after it (exposure 1, no bloom): the curve read backwards."""
    x = np.linspace(0, 12, 240001)
    return np.interp(levels, tone(x), x)


def grey(a, b):
    return float(np.abs(tone(a[..., :3]) - tone(b[..., :3])).mean())


def lum(p):
    return p[..., :3] @ np.array([0.2126, 0.7152, 0.0722])


SMALL = ['w=480', 'h=270']
THOMAS = ['scene=attractors', 'preset=thomas']
PEBBLE = ['scene=forms', 'preset=superpebble', 'clay=1']


# ---------------------------------------------------------------- 1. nothing that was drawn before has changed
def unchanged():
    if not OLD:
        print('     (no --old program given: the bit-for-bit checks against the program before the camera are left out)')
        return
    cases = [
        ('default view, attractors', THOMAS),
        ('default view, surfaces', ['scene=surfaces']),
        ('default view, gems', ['scene=gems', 'nb=2']),
        ('old names, clean blur', THOMAS + ['fov=33', 'dist=5.5', 'aperture=0.12', 'fnear=-0.4', 'ffar=-0.1', 'shift=0.1,-0.05', 'roll=20']),
        ('old names, traced lens', ['scene=knots', 'fov=40', 'aperture=0.2', 'dof=0', 'blades=6', 'bokehRot=20', 'fnear=0.2', 'ffar=0.5']),
        ('a turning view', THOMAS + ['turn=10']),
        ('clay with a backdrop', ['scene=knots', 'clay=1'] + BACKDROP),
        ('a metal', THOMAS + ['material=gold']),
        ('frosted glass with a lens opening', ['scene=knots', 'rough=0.6', 'aperture=0.1']),
        ('two bodies of different materials', ['scene=knots', 'layer=attractors:thomas', 'J12.op=apart', 'B2.material=gold']),
    ]
    for i, (name, args) in enumerate(cases):
        args = SMALL + ['frames=6'] + args
        hdr('same-new-%d' % i, args)
        hdr('same-old-%d' % i, args, exe=OLD)
        check(same_file('same-new-%d.hdr' % i, 'same-old-%d.hdr' % i) and same_file('same-new-%d.png' % i, 'same-old-%d.png' % i),
              'bit for bit the program before the camera: %s' % name)
    # a setup saved by the old program opens as it was drawn
    setup = os.path.join(OUT, 'old-setup.json')
    args = SMALL + ['frames=6', 'scene=knots', 'fov=31', 'aperture=0.1', 'fnear=-0.2', 'ffar=0.3', 'exposure=1.2']
    run(['--still', os.path.join(OUT, 'setup-old.png'), '--hdr', os.path.join(OUT, 'setup-old.hdr'), '--save-setup', setup] + args, exe=OLD)
    run(['--still', os.path.join(OUT, 'setup-new.png'), '--hdr', os.path.join(OUT, 'setup-new.hdr'), '--setup', setup] + SMALL + ['frames=6'])
    check(same_file('setup-new.hdr', 'setup-old.hdr') and same_file('setup-new.png', 'setup-old.png'), 'a setup saved before the camera had its new names draws the same picture, bit for bit')


# ---------------------------------------------------------------- 2. the photographer's numbers
def numbers():
    c = camera(['focal=50'])
    check(abs(c['fieldV'] - 26.991) < 0.01 and abs(c['stored']['fov'] - 26.991) < 0.01, 'focal=50 on a 24 mm sensor is %.2f degrees top to bottom (2 atan(12 / 50) = 26.99)' % c['fieldV'])
    c = camera(['focal=50', 'sensor=15.7'])
    check(abs(c['fieldV'] - 17.845) < 0.01, 'the same lens on a 15.7 mm sensor: %.2f degrees (17.85)' % c['fieldV'])
    c = camera(['scene=attractors'])
    check(abs(c['focal'] - 51.98) < 0.01 and c['fstop'] is None and not c['lensVariant'], "the attractors' own view (fov 26) reads as a %.1f mm lens with no opening, drawn by the plain shaders" % c['focal'])
    c = camera(['focal=100', 'fstop=4', 'subject=50', 'focusDist=300', 'band=0'])
    want = 100 * 100 / (4 * 300) * 30 / 330
    k = 0.03 * 4 / 1e4
    check(abs(c['focusMm'] - 300) < 0.01 and abs(c['sharpFromMm'] - 1 / (1 / 300 + k)) < 0.01 and abs(c['sharpToMm'] - 1 / (1 / 300 - k)) < 0.01,
          '100 mm f/4 focused at 300 mm: sharp from %.2f to %.2f mm (thin lens: %.2f to %.2f); a point 30 mm behind is a %.3f mm disc on the sensor'
          % (c['sharpFromMm'], c['sharpToMm'], 1 / (1 / 300 + k), 1 / (1 / 300 - k), want))
    a, b = camera(['fov=40', 'focal=85']), camera(['focal=85', 'fov=40'])
    check(abs(a['focal'] - 85) < 1e-9 and abs(b['stored']['fov'] - 40) < 1e-12 and b['stored']['focal'] == 0, 'of a lens said both ways the later one counts (fov then focal: %.0f mm; focal then fov: %.0f degrees)' % (a['focal'], b['stored']['fov']))
    c = camera(['fstop=0.3', 'focal=2'])
    check(len(c['notes']) >= 2 and c['fstop'] is not None and abs(c['fstop'] - 0.3) < 1e-9, 'f/0.3 on a 2 mm lens is drawn, and said to be no real lens: %d notes' % len(c['notes']))
    c = camera(['shutterAngle=180'])
    check(abs(c['shutter'] - 0.5) < 1e-12, 'a shutter angle of 180 degrees is half the frame open')
    # said as a focal length or as the field of view it comes to: the same picture
    fov = 2 * np.degrees(np.arctan(12 / 35))
    a = hdr('focal-35', SMALL + ['frames=6', 'focal=35'] + THOMAS)
    b = hdr('fov-35', SMALL + ['frames=6', 'fov=%.15g' % fov] + THOMAS)
    check(grey(a, b) < 0.02, 'focal=35 draws what fov=%.3f draws (%.4f grey levels apart)' % (fov, grey(a, b)))
    # the lens changed and the camera moved: the subject keeps its size, the perspective does not
    wide = hdr('dolly-24', SMALL + ['frames=4', 'dollyZoom=24', 'scene=solids', 'preset=cube'])
    tele = hdr('dolly-200', SMALL + ['frames=4', 'dollyZoom=200', 'scene=solids', 'preset=cube'])
    hw = np.ptp(np.nonzero((wide[..., 3] > 0).any(axis=1))[0])
    ht = np.ptp(np.nonzero((tele[..., 3] > 0).any(axis=1))[0])
    cw, ct = camera(['dollyZoom=24', 'scene=solids']), camera(['dollyZoom=200', 'scene=solids'])
    check(0.75 < hw / ht < 1.45 and grey(wide, tele) > 1.0 and abs(ct['distanceMm'] / cw['distanceMm'] - 200 / 24) < 1e-6,
          'dollyZoom 24 and 200 mm: the cube is %d and %d pixels high, the camera %.1f and %.1f cm away, and the pictures differ (perspective)' % (hw, ht, cw['distanceMm'] / 10, ct['distanceMm'] / 10))


# ---------------------------------------------------------------- 3. the blur against the thin-lens formula
def spread(p):
    """total light, and the brightness-weighted variance of a picture along x and along y, in pixels^2"""
    l = lum(p)
    h, w = l.shape
    ys, xs = np.mgrid[0:h, 0:w]
    m = l.sum()
    cx, cy = (l * xs).sum() / m, (l * ys).sum() / m
    return m, (l * (xs - cx) ** 2).sum() / m, (l * (ys - cy) ** 2).sum() / m


def predicted(sharp, c, squeeze=1.0, light=0.25):
    """What the thin-lens formula says the variance grows by along x and along y: the mean over the sharp
    picture's light of c^2 / 4 (an evenly lit disc's spread along one axis). Every pixel's disc from its stored distance, in millimetres:
    diameter on the sensor = F^2 / (N s) * |z - s| / z."""
    h, w = sharp.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    tan_half = c['sensor'] / (2 * c['focal'])
    qx = (2 * (xs + 0.5) - w) / h * tan_half
    qy = (2 * (h - (ys + 0.5)) - h) / h * tan_half
    t = sharp[..., 3]
    z = t / np.sqrt(qx * qx + qy * qy + 1) * c['unitMm']                # depth along the axis, millimetres
    s = c['focusMm']
    on_sensor = c['focal'] ** 2 / (c['fstop'] * s) * np.abs(z - s) / np.maximum(z, 1e-9)
    radius = on_sensor / c['sensor'] * h / 2                              # pixels
    l = lum(sharp) * (t > 0)
    mean_c2 = (l * radius ** 2).sum() / l.sum()
    return light * mean_c2 / squeeze ** 2, light * mean_c2, float(np.sqrt(mean_c2))


def blur_cases():
    cases = [
        ('85 mm f/5.6, a 100 mm subject', ['focal=85', 'dist=16', 'fstop=5.6', 'subject=100']),
        ('the same on a 400 mm subject', ['focal=85', 'dist=16', 'fstop=5.6', 'subject=400']),
        ('50 mm f/2, a 100 mm subject', ['focal=50', 'dist=10', 'fstop=2', 'subject=100']),
        ('200 mm f/11, a 30 mm subject', ['focal=200', 'dist=38', 'fstop=11', 'subject=30']),
    ]
    for i, (name, lens) in enumerate(cases):
        # focused in front of the whole pebble, on one plane: everything is behind it
        args = SMALL + PEBBLE + lens + ['focusDist=%g' % (0.62 * camera(PEBBLE + lens)['distanceMm']), 'band=0']
        c = camera(args)
        sharp = hdr('blur-%d-sharp' % i, args + ['aperture=0', 'frames=48'])
        m0, vx0, vy0 = spread(sharp)
        px, py, rms = predicted(sharp, c)
        for mode, label, frames in ((1, 'clean blur', 48), (0, 'traced lens', 160)):
            p = hdr('blur-%d-mode%d' % (i, mode), args + ['dof=%d' % mode, 'frames=%d' % frames, 'lensN=4'])
            m, vx, vy = spread(p)
            got = np.sqrt(max(vx - vx0 + vy - vy0, 0))
            want = np.sqrt(px + py)
            # 6% and a third of a pixel: the picture is a few hundred pixels across, a blur under half a
            # pixel is drawn as none, and discs of a pixel or two are made of very few pixels. The traced
            # lens gets 8%: it also shows a little of the pebble's far side that the sharp picture hides.
            c_got, slack = got * np.sqrt(2), (0.06 if mode else 0.08) * rms + 0.3
            check(abs(c_got - rms) < slack and abs(m / m0 - 1) < 0.03,
                  '%s, %s: discs of %.2f pixels (r.m.s. radius), the thin-lens formula says %.2f; light kept %.1f%%' % (name, label, c_got, rms, 100 * m / m0))


def modes_agree():
    # The two ways of focusing against each other, picture for picture, through each kind of lens: the
    # clean blur takes every pixel's radius from the camera, the traced lens starts its rays from the lens
    # opening and aims them at the surface in focus. Clay, so that every pixel is at its stored distance.
    kinds = [('the plain lens', []), ('a leaning plane of focus', ['tiltY=40', 'tiltX=-15']), ('barrel distortion', ['distort=0.3']), ('an orthographic lens', ['proj=orthographic']),
             ('a fisheye', ['proj=equisolid', 'focal=52']), ('a cylindrical lens', ['proj=cylindrical', 'focal=52']), ('a panorama', ['proj=panorama', 'dist=2.6', 'fstop=1.4'])]
    for name, lens in kinds:
        tag = name.split()[-1]
        args = SMALL + THOMAS + ['clay=1', 'fstop=4', 'focus=0.1', 'band=0'] + lens
        a = hdr('modes-clean-' + tag, args + ['dof=1', 'frames=32'])
        b = hdr('modes-traced-' + tag, args + ['dof=0', 'frames=160', 'lensN=4'])
        none = hdr('modes-sharp-' + tag, args + ['aperture=0', 'frames=32'])
        # (docs/dof.md measured 0.13 to 0.18 for the plain lens at a larger opening and size)
        check(grey(a, b) < 0.5 and grey(a, none) > 3 * grey(a, b) and (none[..., 3] > 0).mean() > 0.01, 'clean blur and traced lens through %s at f/4: %.2f grey levels apart (no lens opening at all: %.2f)' % (name, grey(a, b), grey(a, none)))


# ---------------------------------------------------------------- 4. the shaders for any lens, given the plain lens
def same_through_any_lens():
    cases = [
        ('glass', THOMAS + ['frames=32']),
        ('traced lens with blades', THOMAS + ['aperture=0.2', 'dof=0', 'blades=6', 'frames=64']),
        ('clean blur over a backdrop', ['scene=knots', 'aperture=0.2', 'frames=32'] + BACKDROP),
        ('a turning view with the focus aid', THOMAS + ['turn=12', 'aperture=0.1', 'planes=1']),
        ('clay', ['scene=surfaces', 'clay=1', 'frames=16']),
    ]
    for i, (name, args) in enumerate(cases):
        a = hdr('plain-%d' % i, SMALL + args)
        b = hdr('lens-%d' % i, SMALL + args, env=LENS)
        ap, bp = png('plain-png-%d' % i, SMALL + args), png('lens-png-%d' % i, SMALL + args, env=LENS)
        # the two texts of each shader do the same arithmetic in a different order here and there, and
        # glass turns a last digit into a different sparkle: a hundredth of a grey level on average
        check(grey(a, b) < 0.05 and np.abs(ap - bp).mean() < 0.06 and np.array_equal(a[..., 3] > 0, b[..., 3] > 0),
              'the shaders built for any lens draw the plain lens\'s picture: %s (%.4f grey levels apart; on screen %.4f)' % (name, grey(a, b), np.abs(ap - bp).mean()))


# ---------------------------------------------------------------- 5. every kind of lens
def to_film(proj, tan_half, d):
    """Where a direction from the lens shows on the film (across, up; half-heights), by the kind of lens."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    if proj == 'perspective':
        return x / (-z * tan_half), y / (-z * tan_half)
    if proj in ('fisheye', 'equisolid', 'stereographic'):
        r = np.hypot(x, y)
        th = np.arctan2(r, -z)
        rho = {'fisheye': th, 'equisolid': 2 * np.sin(th / 2), 'stereographic': 2 * np.tan(th / 2)}[proj]
        k = rho / (np.maximum(r, 1e-12) * tan_half)
        return x * k, y * k
    if proj == 'panorama':
        return np.arctan2(x, -z) / (np.pi / 2), np.arctan2(y, np.hypot(x, z)) / (np.pi / 2)
    return np.arctan2(x, -z) / tan_half, y / (np.hypot(x, z) * tan_half)


def bow(fx, fy, distort):
    """A film point of a lens with straight lines to where a lens that bows them puts it."""
    if distort == 0:
        return fx, fy
    a = np.sqrt(3 * abs(distort) / 3.25)
    r = np.maximum(np.hypot(fx, fy), 1e-12)
    k = np.arctan(a * r) / (a * r) if distort > 0 else np.tan(np.minimum(a * r, 1.5)) / (a * r)
    return fx * k, fy * k


def resample(src, fx, fy):
    """The picture `src` read at film points (across, up; half-heights), between its pixels; nan outside it."""
    h, w = src.shape[:2]
    x = (fx * h + w) / 2 - 0.5
    y = h - (fy * h + h) / 2 - 0.5
    ok = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0, y0 = np.clip(np.floor(x).astype(int), 0, w - 2), np.clip(np.floor(y).astype(int), 0, h - 2)
    tx, ty = (x - x0)[..., None], (y - y0)[..., None]
    out = (src[y0, x0] * (1 - tx) + src[y0, x0 + 1] * tx) * (1 - ty) + (src[y0 + 1, x0] * (1 - tx) + src[y0 + 1, x0 + 1] * tx) * ty
    out[~ok] = np.nan
    return out


def reference(scene, tag):
    """The picture of a scene through an ordinary 50 degree lens, and the direction of every pixel's ray."""
    w0, h0, fov0 = 384, 216, 50.0
    t0 = np.tan(np.radians(fov0) / 2)
    ref = png(tag, ['w=%d' % w0, 'h=%d' % h0, 'fov=%g' % fov0] + scene)
    ys, xs = np.mgrid[0:h0, 0:w0]
    qx, qy = (2 * (xs + 0.5) - w0) / h0, (2 * (h0 - (ys + 0.5)) - h0) / h0
    d = np.stack([qx * t0, qy * t0, -np.ones_like(qx)], axis=-1)
    return ref, d / np.linalg.norm(d, axis=-1, keepdims=True)


def read_back(tag, proj, args, distort, w, h, scene, ref, d):
    """A lens's picture of the scene read back at the places its own law puts the reference's rays: how far
    that is from the reference, how far the same picture read by a neighbouring law is, and how much of
    the reference it covers."""
    src = png(tag, ['w=%d' % w, 'h=%d' % h] + args + scene)
    c = camera(['w=%d' % w, 'h=%d' % h] + args + scene)
    tan_half = c['sensor'] / (2 * c['focal'])
    fx, fy = to_film(proj, tan_half, d)
    fx, fy = bow(fx, fy, distort)
    got = resample(src, fx, fy)
    ok = ~np.isnan(got[..., 0])
    diff = float(np.abs(got[ok] - ref[ok]).mean())
    # the same picture read back by a neighbouring law (a little more or less bowed, or another
    # fisheye's): that the check tells them apart is what makes passing it mean something
    if distort != 0:
        wx, wy = bow(*to_film(proj, tan_half, d), distort * 0.6)
    else:
        wx, wy = to_film({'fisheye': 'equisolid', 'equisolid': 'stereographic', 'stereographic': 'fisheye', 'cylindrical': 'perspective', 'panorama': 'cylindrical'}[proj], tan_half if proj != 'panorama' else np.pi / 2, d)
    wrong = resample(src, wx, wy)
    both = ok & ~np.isnan(wrong[..., 0])
    return diff, float(np.abs(wrong[both] - ref[both]).mean()), float(ok.mean())


def lenses():
    # One camera position, one subject, the garden behind. The picture through an ordinary 50 degree lens
    # is the reference; every other lens's picture, read back at the places its own law says each of the
    # reference's rays lands, must be that picture. (Reading between pixels softens a little: 2 grey levels
    # of a busy garden. Read where a neighbouring law says, the pictures are 6 to 19 apart.)
    scene = ['scene=knots', 'dist=4', 'frames=24', 'hdriBlur=0.3', 'bloom=0', 'vignette=0', 'grain=0'] + BACKDROP
    ref, d = reference(scene, 'lens-ref')
    kinds = [
        ('barrel 0.3', 'perspective', ['fov=64', 'distort=0.3'], 0.3, 768, 432),
        ('pincushion -0.3', 'perspective', ['fov=72', 'distort=-0.3'], -0.3, 768, 432),
        ('fisheye', 'fisheye', ['proj=fisheye', 'focal=14'], 0.0, 768, 432),
        ('equisolid', 'equisolid', ['proj=equisolid', 'focal=14'], 0.0, 768, 432),
        ('stereographic', 'stereographic', ['proj=stereographic', 'focal=14'], 0.0, 768, 432),
        ('cylindrical', 'cylindrical', ['proj=cylindrical', 'focal=14'], 0.0, 768, 432),
        ('panorama', 'panorama', ['proj=panorama'], 0.0, 1536, 768),
        ('fisheye with barrel', 'fisheye', ['proj=fisheye', 'focal=16', 'distort=0.2'], 0.2, 768, 432),
    ]
    for name, proj, args, distort, w, h in kinds:
        diff, other, cover = read_back('lens-' + name.replace(' ', '-'), proj, args, distort, w, h, scene, ref, d)
        check(cover > 0.95 and diff < 3.5 and other > 2 * diff,
              '%s: read back through its own law it is the perspective picture to %.2f grey levels (through a neighbouring law: %.1f)' % (name, diff, other))
    # an orthographic lens is the limit of a long lens far away: the same outline as through 3000 mm from 380 units off
    shape = ['scene=attractors', 'preset=thomas', 'frames=4', 'clay=1'] + SMALL
    o = hdr('ortho', shape + ['proj=orthographic'])
    far = hdr('ortho-far', shape + ['dollyZoom=3000'])
    near = hdr('ortho-near', shape)
    a, b, n = o[..., 3] > 0, far[..., 3] > 0, near[..., 3] > 0
    same = (a & b).sum() / (a | b).sum()
    c = camera(shape + ['proj=orthographic'])
    check(same > 0.985 and (a & n).sum() / (a | n).sum() < same - 0.02 and c['viewMm'] is not None,
          'orthographic: the outline is that of a 3000 mm lens from far off (%.1f%% of it shared; with the 52 mm lens: %.1f%%); the view is %.1f mm high' % (100 * same, 100 * (a & n).sum() / (a | n).sum(), c['viewMm']))


def steady():
    # A turning view through each lens, as the window shows it: no glass may be left unfollowed (the debug
    # view paints it red), and the moving picture must stay as close to the settled picture of the same
    # view as it does through the plain lens (a wrong place in the old picture shows as smear).
    kinds = [('plain', []), ('barrel', ['distort=0.3']), ('pincushion', ['distort=-0.3']), ('fisheye', ['proj=fisheye', 'focal=30']), ('equisolid', ['proj=equisolid', 'focal=30']),
             ('stereographic', ['proj=stereographic', 'focal=30']), ('cylindrical', ['proj=cylindrical', 'focal=40']), ('orthographic', ['proj=orthographic']), ('panorama', ['proj=panorama', 'dist=2.6'])]
    base = None
    for name, args in kinds:
        # (clay: a matte surface has no sparkle of its own to drown a smear in)
        args = SMALL + THOMAS + ['clay=1'] + args
        dbg = png('steady-dbg-' + name, args + ['turn=24', 'debug=1'])
        red = float(((dbg[..., 0] > 200) & (dbg[..., 1] < 60) & (dbg[..., 2] < 60)).mean())
        blue = float(((dbg[..., 2] > 200) & (dbg[..., 0] < 60)).mean())
        moving = hdr('steady-moving-' + name, args + ['turn=24'])
        settled = hdr('steady-settled-' + name, args + ['turn=24', 'frames=32'])
        glass = settled[..., 3] > 0
        off = float(np.abs(tone(moving[..., :3]) - tone(settled[..., :3]))[glass].mean())
        if base is None:
            base = off
        check(red < 0.0005 and blue > 0.002 and off < 1.5 * base + 0.3 and glass.mean() > 0.01,
              'turning through the %s lens: %.3f%% of the picture unfollowed (outlines: %.1f%%); the moving picture is %.2f grey levels from the settled one on the shape (plain: %.2f)' % (name, 100 * red, 100 * blue, off, base))


def clicks():
    # pick=x,y focuses as a click does. The glass under the point must then be in focus, and most of the
    # rest of the shape must not be.
    kinds = [('plain', []), ('barrel', ['distort=0.3']), ('fisheye', ['proj=fisheye', 'focal=40']), ('equisolid', ['proj=equisolid', 'focal=40']), ('stereographic', ['proj=stereographic', 'focal=40']),
             ('cylindrical', ['proj=cylindrical', 'focal=45']), ('orthographic', ['proj=orthographic']), ('panorama', ['proj=panorama', 'dist=2.2']), ('leaning plane', ['tiltX=35', 'tiltY=-20'])]
    # (GLASSFORMS_FOCUS_SHOW=radius has the focus stage draw every pixel's blur radius instead of the
    # picture: green where it is under half a pixel, red in front of the plane in focus, blue behind it)
    show = {'GLASSFORMS_FOCUS_SHOW': 'radius'}
    for name, args in kinds:
        args = SMALL + THOMAS + ['frames=8', 'band=0', 'fstop=2.8'] + args
        sharp = hdr('click-sharp-' + name, args + ['aperture=0'])
        glass = sharp[..., 3] > 0
        ys, xs = np.nonzero(glass)
        h, w = sharp.shape[:2]
        # glass well inside one piece of the shape (the stored distance hardly changes for two pixels
        # every way), nearest a point left of the picture's middle
        t = np.where(glass, sharp[..., 3], np.nan)
        win = np.lib.stride_tricks.sliding_window_view(t, (5, 5))
        calm = np.zeros_like(glass)
        calm[2:-2, 2:-2] = (win.max(axis=(2, 3)) - win.min(axis=(2, 3))) < 0.03          # (nan where any of them is not glass: false)
        ys, xs = np.nonzero(calm)
        i = np.argmin((xs - w * 0.42) ** 2 + (ys - h * 0.5) ** 2)
        x, y = int(xs[i]), int(ys[i])
        radius = hdr('click-' + name, args + ['pick=%.5f,%.5f' % ((x + 0.5) / w, (y + 0.5) / h)], env=show)
        here = radius[y, x]
        sharp_share = float((np.abs(radius[..., 1][glass] - 0.2) < 0.01).mean())
        check(abs(here[1] - 0.2) < 0.01 and 0.005 < sharp_share < 0.6,
              'a click through the %s lens: the glass clicked is in focus (blur under half a pixel there), and %.0f%% of all the glass is' % (name, 100 * sharp_share))
    # a leaning plane of focus cuts the shape where the camera says it does: every piece of glass is in
    # front of the plane or behind it as its stored distance, the camera's place and the plane's slope make it
    slope = np.tan(np.radians(50))
    args = SMALL + THOMAS + ['frames=8', 'band=0', 'focus=0', 'fstop=4', 'tiltY=50']
    c = camera(args)
    sharp = hdr('tilt-sharp', args + ['aperture=0'])
    radius = hdr('tilt-radius', args, env=show)
    h, w = sharp.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    t = c['sensor'] / (2 * c['focal'])
    qx, qy = (2 * (xs + 0.5) - w) / h * t, (2 * (h - (ys + 0.5)) - h) / h * t
    z = sharp[..., 3] / np.sqrt(qx * qx + qy * qy + 1)
    depth = z - c['stored']['dist'] - slope * qx * z                       # behind the leaning plane if above 0
    sure = (sharp[..., 3] > 0) & (np.abs(depth) > 0.08)
    behind, front = radius[..., 2] > 0, radius[..., 0] > 0
    agree = float(((depth > 0) == behind)[sure].mean())
    level = float((((z - c['stored']['dist']) > 0) == behind)[sure].mean())
    check(agree > 0.985 and level < 0.9 and front[sure].any() and behind[sure].any(),
          'the plane of focus swung 50 degrees: %.1f%% of the glass is on the side of it the camera says (taking the plane for level: %.0f%%)' % (100 * agree, 100 * level))


# ---------------------------------------------------------------- 6. the lens's character
def character():
    flat = ['bloom=0', 'vignette=0', 'grain=0']
    # fall-off: the corner against the middle is the lens's law, on screen and in the 16-bit still
    for name, args, law in (('a 24 mm lens', ['dollyZoom=24'], lambda cos: cos ** 4), ('a stereographic fisheye', ['proj=stereographic', 'focal=14'], lambda cos: (0.5 + 0.5 * cos) ** 2)):
        scene = ['w=384', 'h=216', 'scene=knots', 'frames=8', 'exposure=1'] + flat + BACKDROP + args
        tag = name.split()[-1]
        for deep in (False, True):
            a = png('falloff0-%s-%d' % (tag, deep), scene, extra=['--16'] if deep else [])
            b = png('falloff1-%s-%d' % (tag, deep), scene + ['falloff=1'], extra=['--16'] if deep else [])
            la, lb = untone(a).mean(axis=2), untone(b).mean(axis=2)
            c = camera(scene)
            h, w = la.shape
            # a patch near the corner and one at the middle; the angle off the axis at the corner patch
            cy, cx = 12, 18
            t = c['sensor'] / (2 * c['focal'])
            fx, fy = (2 * (cx + 0.5) - w) / h, (2 * (h - (cy + 0.5)) - h) / h
            rho = np.hypot(fx, fy) * t
            cos = np.cos(2 * np.arctan(rho / 2)) if 'fisheye' in name else 1 / np.sqrt(1 + rho * rho)
            got = lb[cy - 6:cy + 7, cx - 6:cx + 7].mean() / la[cy - 6:cy + 7, cx - 6:cx + 7].mean()
            mid = lb[h // 2 - 6:h // 2 + 7, w // 2 - 6:w // 2 + 7].mean() / la[h // 2 - 6:h // 2 + 7, w // 2 - 6:w // 2 + 7].mean()
            check(abs(got / law(cos) - 1) < (0.04 if deep else 0.08) and abs(mid - 1) < 0.02,
                  'fall-off of %s, %s: the corner keeps %.3f of its light (its law: %.3f), the middle %.3f' % (name, 'the 16-bit still' if deep else 'on screen', got, law(cos), mid))
    # colour fringing: red and blue part company towards the corners, nothing moves at the middle
    scene = SMALL + ['scene=knots', 'frames=16'] + flat + BACKDROP
    a, b = png('fringe0', scene), png('fringe1', scene + ['fringe=1'])
    deep = png('fringe1-deep', scene + ['fringe=1'], extra=['--16'])
    h, w = a.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.hypot(xs - w / 2, ys - h / 2) / (h / 2)
    d = np.abs(a - b)
    edge, mid = d[r > 1.4].mean(axis=0), d[r < 0.15].mean(axis=0)
    check(edge[0] > 2 * edge[1] and edge[2] > 2 * edge[1] and edge[0] > 1.0 and mid.max() < 0.25 * edge[0] and np.abs(deep - b).mean() < 1.0,
          'colour fringing: towards the corners red moves %.2f, green %.2f, blue %.2f grey levels; at the middle %.2f; the 16-bit still agrees to %.2f' % (edge[0], edge[1], edge[2], mid.max(), np.abs(deep - b).mean()))
    # the shape of blurred points, in both ways of focusing: an anamorphic lens's are narrower than high
    # by its squeeze; a bright rim spreads the light further out (mean square radius 3/4 of the edge's
    # against 1/2 for an even disc), a bright core less far (1/3)
    args = SMALL + PEBBLE + ['focal=85', 'dist=16', 'fstop=5.6', 'subject=100', 'band=0']
    args = args + ['focusDist=%g' % (0.62 * camera(args)['distanceMm'])]
    c = camera(args)
    sharp = hdr('shape-sharp', args + ['aperture=0', 'frames=48'])
    _, vx0, vy0 = spread(sharp)
    for mode, label, frames in ((1, 'clean blur', 48), (0, 'traced lens', 200)):
        more = ['dof=%d' % mode, 'frames=%d' % frames, 'lensN=4']
        _, vx, vy = spread(hdr('shape-even-%d' % mode, args + more))
        even = vx - vx0 + vy - vy0
        _, ax, ay = spread(hdr('shape-squeeze-%d' % mode, args + more + ['squeeze=2']))
        ratio = np.sqrt((ay - vy0) / (ax - vx0))
        _, rx, ry = spread(hdr('shape-rim-%d' % mode, args + more + ['bokeh=1']))
        _, kx, ky = spread(hdr('shape-core-%d' % mode, args + more + ['bokeh=-1']))
        rim, core = (rx - vx0 + ry - vy0) / even, (kx - vx0 + ky - vy0) / even
        check(abs(ratio / 2 - 1) < 0.1 and abs(rim / 1.5 - 1) < 0.1 and abs(core / (2 / 3) - 1) < 0.1,
              '%s: squeeze 2 gives points %.2f times higher than wide; a bright rim spreads light %.2f times as far (1.5), a bright core %.2f (0.67)' % (label, ratio, rim, core))
    # a leaning plane of focus: with the plane swung, one side of a flat-on view is sharp and the other is not


def cats_eye():
    # A small pebble towards the corner of the picture, out of focus. With cat's-eye its blur is cut from
    # both sides along the line from the lens's axis: narrower that way than across it, by what two unit
    # discs 2c apart share (c = the pebble's place on the film x catseye / 2 / 1.803). The same in both
    # ways of focusing, and no light lost before the final pass.
    args = SMALL + PEBBLE + ['L1.x=1.35', 'L1.y=0.7', 'L1.size=0.3', 'dist=5', 'focal=50', 'fstop=1.4', 'subject=100', 'band=0', 'focusDist=120']
    sharp = hdr('eye-sharp', args + ['aperture=0', 'frames=16'])

    def along(p):
        """total light; the picture's spread along the line from its middle through the pebble, and across it; where the pebble is"""
        l = lum(p)
        h, w = l.shape
        ys, xs = np.mgrid[0:h, 0:w]
        m = l.sum()
        cx, cy = (l * xs).sum() / m, (l * ys).sum() / m
        n = np.hypot(cx - w / 2, cy - h / 2)
        ux, uy = (cx - w / 2) / n, (cy - h / 2) / n
        a, b = (xs - cx) * ux + (ys - cy) * uy, -(xs - cx) * uy + (ys - cy) * ux
        return m, (l * a * a).sum() / m, (l * b * b).sum() / m, ((2 * (cx + 0.5) - w) / h, (h - 2 * (cy + 0.5)) / h)

    m0, r0, t0, film = along(sharp)
    c = np.hypot(*film) * 0.5 / np.sqrt(3.25)
    # the cut opening's spread along the cut and across it, against the whole disc's (1/4 each way)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (400000, 2))
    pts = pts[(np.hypot(pts[:, 0] - c, pts[:, 1]) < 1) & (np.hypot(pts[:, 0] + c, pts[:, 1]) < 1)]
    want_r, want_t = (pts[:, 0] ** 2).mean() / 0.25, (pts[:, 1] ** 2).mean() / 0.25
    for mode, label, frames in ((1, 'clean blur', 16), (0, 'traced lens', 240)):
        more = ['dof=%d' % mode, 'frames=%d' % frames, 'lensN=4']
        _, r, t, _ = along(hdr('eye-round-%d' % mode, args + more))
        m, re, te, _ = along(hdr('eye-cut-%d' % mode, args + more + ['catseye=1']))
        got_r, got_t = (re - r0) / (r - r0), (te - t0) / (t - t0)
        check(abs(got_r / want_r - 1) < 0.12 and abs(got_t / want_t - 1) < 0.12 and abs(m / m0 - 1) < 0.03,
              "cat's-eye, %s: a blurred point %.2f of the way to the corner keeps %.2f of its spread along the cut and %.2f across it (the cut opening: %.2f and %.2f); light kept %.1f%%"
              % (label, np.hypot(*film) / np.sqrt(3.25), got_r, got_t, want_r, want_t, 100 * m / m0))
    # what the cut costs the corners in brightness is taken off at the end, on screen and in the 16-bit still alike
    scene = ['w=384', 'h=216', 'scene=knots', 'frames=8', 'exposure=1', 'bloom=0', 'vignette=0', 'grain=0', 'fstop=4'] + BACKDROP
    for deep in (False, True):
        a = untone(png('eye-light0-%d' % deep, scene, extra=['--16'] if deep else [])).mean(axis=2)
        b = untone(png('eye-light1-%d' % deep, scene + ['catseye=1'], extra=['--16'] if deep else [])).mean(axis=2)
        h, w = a.shape
        cy, cx = 14, 20
        x = np.hypot((2 * (cx + 0.5) - w) / h, (h - 2 * (cy + 0.5)) / h) * 0.5 / np.sqrt(3.25)
        law = (2 * np.arccos(x) - 2 * x * np.sqrt(1 - x * x)) / np.pi
        got = b[cy - 6:cy + 7, cx - 6:cx + 7].mean() / a[cy - 6:cy + 7, cx - 6:cx + 7].mean()
        mid = b[h // 2 - 4:h // 2 + 5, w // 2 - 4:w // 2 + 5].mean() / a[h // 2 - 4:h // 2 + 5, w // 2 - 4:w // 2 + 5].mean()
        check(abs(got / law - 1) < 0.08 and abs(mid - 1) < 0.03, "cat's-eye, %s: the corner keeps %.3f of its light (the share of the opening left: %.3f), the middle %.3f" % ('the 16-bit still' if deep else 'on screen', got, law, mid))


def materials():
    # Other materials than plain glass are drawn by parts put into the same tracer, so they go through
    # every lens as glass does. A metal and a frosted body, each three ways.
    gold, frost = ['material=gold'], ['rough=0.6']
    # 1. the tracer built for any lens, given the plain lens, draws the plain tracer's picture, frame for
    #    frame: for the frosted body that means its random numbers are drawn for the same pixels
    for name, mat in (('gold', gold), ('frosted glass', frost)):
        args = SMALL + THOMAS + mat + ['frames=1']
        a, b = hdr('mat-plain-' + name.split()[0], args), hdr('mat-lens-' + name.split()[0], args, env=LENS)
        other = hdr('mat-other-' + name.split()[0], args + ['seed=3'])
        check(grey(a, b) < 0.25 * grey(a, other) + 0.05 and np.array_equal(a[..., 3] > 0, b[..., 3] > 0),
              "%s, one frame: the tracer for any lens draws the plain one's picture (%.3f grey levels apart; another frame of the same view: %.2f)" % (name, grey(a, b), grey(a, other)))
    # 2. the frosted body through lenses whose rays do not say which pixel they came through (every ray
    #    of an orthographic lens has the same direction): what one frame is off from the settled picture
    #    must still be different in every pixel, as through the plain lens, not the same everywhere
    def grain(args):
        one, settled = hdr('mat-one', args + ['frames=1']), hdr('mat-settled', args + ['frames=96'])
        r = lum(one) - lum(settled)
        m = (settled[..., 3] > 0)
        both = m[:, 1:] & m[:, :-1]
        x, y = r[:, 1:][both], r[:, :-1][both]
        return float(np.corrcoef(x, y)[0, 1]), float(np.abs(tone(one[..., :3]) - tone(settled[..., :3]))[m].mean())
    plain = grain(SMALL + THOMAS + frost)
    for name, lens in (('orthographic', ['proj=orthographic']), ('fisheye', ['proj=equisolid', 'focal=40']), ('panorama', ['proj=panorama', 'dist=2.6'])):
        c, off = grain(SMALL + THOMAS + frost + lens)
        check(abs(c) < abs(plain[0]) + 0.15 and off < 2.5 * plain[1] + 1, 'frosted glass through the %s lens: one frame\'s grain is %.2f alike from one pixel to the next (plain lens: %.2f), %.1f grey levels from settled (plain: %.1f)' % (name, c, plain[0], off, plain[1]))
    # 3. a gold knot through a fisheye and a panorama, read back through each lens's own law: the
    #    perspective picture (a mirror shows the garden from wherever the ray goes: any wrong ray shows)
    scene = ['scene=knots', 'dist=4', 'frames=24', 'hdriBlur=0.2', 'bloom=0', 'vignette=0', 'grain=0'] + gold + BACKDROP
    ref, d = reference(scene, 'mat-ref')
    for name, proj, args, w, h in (('fisheye', 'equisolid', ['proj=equisolid', 'focal=14'], 768, 432), ('panorama', 'panorama', ['proj=panorama'], 1536, 768)):
        diff, other, cover = read_back('mat-' + name, proj, args, 0.0, w, h, scene, ref, d)
        check(cover > 0.95 and diff < 3.5 and other > 2 * diff, 'a gold knot through the %s lens, read back through its law: the perspective picture to %.2f grey levels (through a neighbouring law: %.1f)' % (name, diff, other))
    # ... and two bodies of different materials (a gold attractor beside a glass knot), which are drawn by
    # the tracer that follows a ray from body to body
    scene = ['scene=knots', 'layer=attractors:thomas', 'J12.op=apart', 'B2.material=gold', 'L2.size=0.7', 'L2.x=0.5', 'dist=4', 'frames=24', 'hdriBlur=0.2', 'bloom=0', 'vignette=0', 'grain=0'] + BACKDROP
    ref, d = reference(scene, 'bodies-ref')
    for name, proj, args, w, h in (('stereographic fisheye', 'stereographic', ['proj=stereographic', 'focal=14'], 768, 432), ('cylindrical', 'cylindrical', ['proj=cylindrical', 'focal=14'], 768, 432)):
        diff, other, cover = read_back('bodies-' + proj, proj, args, 0.0, w, h, scene, ref, d)
        check(cover > 0.95 and diff < 3.5 and other > 2 * diff, 'a gold body beside a glass one through the %s lens, read back through its law: the perspective picture to %.2f grey levels (through a neighbouring law: %.1f)' % (name, diff, other))
    # 4. frosted glass with the lens opened, both ways of focusing, through an orthographic lens
    args = SMALL + THOMAS + frost + ['proj=orthographic', 'fstop=4', 'focus=0.1', 'band=0']
    a = hdr('mat-clean', args + ['dof=1', 'frames=96'])
    b = hdr('mat-traced', args + ['dof=0', 'frames=240', 'lensN=4'])
    none = hdr('mat-sharp', args + ['aperture=0', 'frames=96'])
    check(grey(a, b) < 1.2 and grey(a, none) > 1.5 * grey(a, b), 'frosted glass through an orthographic lens at f/4: clean blur and traced lens %.2f grey levels apart (no lens opening: %.2f)' % (grey(a, b), grey(a, none)))


# ---------------------------------------------------------------- 7. video, and what it costs
def video():
    out = os.path.join(OUT, 'pull.mp4')
    r = json.loads(run(['--video', out, '--quiet', 'w=320', 'h=180', 'fps=12', 'samples=2', 'frames=4', 'spin=0', 'fstop=4', 'track=P.focal:35:85:1', 'shutterAngle=180'] + THOMAS).strip().splitlines()[-1])
    plan = ' '.join(r.get('plan', []))
    check(os.path.exists(out) and os.path.getsize(out) > 1000 and 'focal' in plan, 'a focal-length pull renders as a loop: %s' % (plan or 'nothing moves'))
    # what the window goes through when another lens is put on: frames go on being drawn through the plain
    # lens while the tracer for any lens is built on another thread, then the picture changes over
    r = json.loads(run(['--load-test', 'lens'] + SMALL + THOMAS).strip().splitlines()[-1])
    check(r['readyAfterMs'] > 0 and r['frameMsAfter'] > 0 and r['whileReading']['longestMs'] < 250,
          'putting a fisheye on while frames are drawn: ready after %.0f ms, %d frames drawn meanwhile (the longest %.1f ms), the frame that changed over %.1f ms'
          % (r['readyAfterMs'], r['whileReading']['frames'], r['whileReading']['longestMs'], r['switchFrameMs']))


def cost():
    if QUICK:
        print('     (--quick: the timing is left out)')
        return
    def bench(args, env=None):
        return json.loads(run(['--bench', '60', 'w=1920', 'h=1080'] + args, env=env).strip().splitlines()[-1])['ms']
    for name, args in (('glass', THOMAS), ('glass, clean blur', THOMAS + ['aperture=0.12']), ('glass, traced lens', THOMAS + ['aperture=0.12', 'dof=0'])):
        # Other programs share the GPU and can only make a go slower: the two are timed in turn, five
        # times over, and the quickest go of each counts. Within 12% and a tenth of a millisecond
        # (docs/camera.md has the figures measured on a quiet GPU at 3840 x 2160).
        plain, lens = [], []
        for _ in range(5):
            plain.append(bench(args))
            lens.append(bench(args, LENS))
        if min(plain) < 0.7 * sorted(plain)[2]:
            # the same work took very different times from one go to the next: something else has the GPU
            print('     cost at 1920 x 1080, %s: not judged, the GPU is busy (plain goes from %.2f to %.2f ms)' % (name, min(plain), max(plain)))
            continue
        plain, lens = min(plain), min(lens)
        check(lens < 1.12 * plain + 0.1, 'cost at 1920 x 1080, %s: %.2f ms through the shaders for any lens, %.2f ms plain (x%.2f)' % (name, lens, plain, lens / plain))


if __name__ == '__main__':
    for name, f in (('nothing changed', unchanged), ('numbers', numbers), ('blur', blur_cases), ('the two ways of focusing', modes_agree), ('any lens = plain', same_through_any_lens), ('lenses', lenses),
                    ('steady', steady), ('clicks', clicks), ('character', character), ("cat's-eye", cats_eye), ('materials', materials), ('video', video), ('cost', cost)):
        if ONLY in name:
            guarded(name, f)
    print('%d of %d checks passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
