"""Checks of focus done to the finished picture (dof=1) against the traced lens (dof=0) and against no lens.

usage: python tools/dof_check.py [--old <glassforms.exe built before the focus stage existed>] [--keep]

Run from anywhere: the program is taken from target/release beside this file's folder. Prints one line per
check starting `ok` or `FAIL`, then a summary; exit status 1 if anything failed. Pictures are compared before
bloom and tone map (the program's --hdr), in grey levels after a plain tone map so that the numbers mean
what they would on screen. What each bound is and why is told beside it; docs/dof.md has the measurements.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'dof_check')
SMALL = ['w=640', 'h=360']
MID = ['w=1280', 'h=720']
THOMAS = ['scene=attractors', 'preset=thomas']
# the two hard settings: nearly everything behind the band, nearly everything in front of it
FAR = ['aperture=0.3', 'fnear=-1.0', 'ffar=-0.6']
NEAR = ['aperture=0.3', 'fnear=0.5', 'ffar=0.9']
DIST, FOV = 6.6, 26.0                    # the attractors' camera

results = []


def check(ok, text):
    results.append(bool(ok))
    print(('ok   ' if ok else 'FAIL ') + text, flush=True)


def run(exe, args):
    r = subprocess.run([exe] + args, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError('%s %s\n%s' % (exe, ' '.join(args), r.stderr.strip()))
    return r.stdout


def still(name, args, exe=EXE):
    """Render a still; returns (rgb, distance) as float arrays, top row first."""
    path = os.path.join(OUT, name)
    run(exe, ['--still', path + '.png', '--hdr', path + '.hdr'] + args)
    raw = np.fromfile(path + '.hdr', dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    a = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(h, w, 4)
    return a[..., :3].astype(np.float64), a[..., 3].astype(np.float64)


def tone(rgb):
    x = np.clip((rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14), 0, 1)
    return np.where(x < 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055) * 255.0


def grey(a, b):
    """mean difference of two pictures, in grey levels (0-255) after the tone map"""
    return float(np.abs(tone(a) - tone(b)).mean())


def depth_from_centre(t):
    """depth of the glass from the object's centre (what fnear and ffar are measured in); nan where none"""
    h, w = t.shape
    tf = np.tan(np.radians(FOV) / 2)
    ys, xs = np.mgrid[0:h, 0:w]
    qx = (2 * (xs + 0.5) - w) / h
    qy = (2 * (h - (ys + 0.5)) - h) / h
    e = t / np.sqrt((qx * tf) ** 2 + (qy * tf) ** 2 + 1) - DIST
    return np.where(t > 0, e, np.nan)


def edges(rgb):
    """how much fine detail a picture has, pixel by pixel: the size of its brightness's slope"""
    y = tone(rgb).mean(axis=2)
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = y[:, 2:] - y[:, :-2]
    gy[1:-1, :] = y[2:, :] - y[:-2, :]
    return np.hypot(gx, gy)


def main():
    os.makedirs(OUT, exist_ok=True)
    old = None
    if '--old' in sys.argv:
        old = os.path.abspath(sys.argv[sys.argv.index('--old') + 1])

    # 1. Mode 0 is the program as it was, and with no lens the new mode is too.
    for scene in (THOMAS, ['scene=knots', 'preset=trefoil'], ['scene=surfaces', 'preset=gyroid', 'anim=3.5']):
        args = scene + SMALL + ['frames=12', 'aperture=0']
        a, _ = still('nolens_0', args + ['dof=0'])
        b, _ = still('nolens_1', args + ['dof=1'])
        check(np.array_equal(a, b), 'with no lens the two modes draw the same picture, bit for bit (%s)' % scene[1])
    if old:
        cases = [
            THOMAS + ['aperture=0'], THOMAS + ['aperture=0.12', 'fnear=-0.4', 'ffar=-0.1'],
            ['scene=knots', 'preset=trefoil', 'aperture=0'], ['scene=knots', 'preset=trefoil', 'aperture=0.12', 'fnear=-0.3', 'ffar=0.1', 'blades=6', 'bokehRot=20', 'lensN=2'],
            ['scene=surfaces', 'preset=gyroid', 'anim=3.5', 'aperture=0'], ['scene=surfaces', 'preset=gyroid', 'anim=3.5', 'aperture=0.12', 'fnear=-0.2', 'ffar=0.2'],
        ]
        for c in cases:
            args = c + SMALL + ['frames=24']
            a, da = still('old', args, exe=old)
            b, db = still('new0', args + ['dof=0'])
            check(np.array_equal(a, b) and np.array_equal(da, db), 'dof=0 draws what the program drew before, bit for bit (%s)' % ' '.join(c[1:]))
            if 'aperture=0' in c:
                b, db = still('new1', args + ['dof=1'])
                check(np.array_equal(a, b) and np.array_equal(da, db), 'dof=1 with no lens draws what the program drew before, bit for bit (%s)' % c[1])

    # 2. The first frame of a view (one traced frame, nothing accumulated) against the traced lens converged.
    # Measured at 1280 x 720: 2.7 against 7.6 grey levels behind the band, 1.5 against 9.5 in front of it.
    # Bound: at least twice as close.
    for name, focus in (('behind', FAR), ('in front of', NEAR)):
        ref, _ = still('ref', THOMAS + MID + focus + ['frames=256', 'lensN=8', 'dof=0'])
        one0, _ = still('one0', THOMAS + MID + focus + ['frames=1', 'dof=0'])
        one1, _ = still('one1', THOMAS + MID + focus + ['frames=1', 'dof=1'])
        e0, e1 = grey(one0, ref), grey(one1, ref)
        check(e1 * 2.0 <= e0, 'one frame, most of the shape %s the band: %.2f grey levels from the converged traced lens, the traced lens\'s own single frame %.2f (bound: half)' % (name, e1, e0))
        # and settled, the new mode stays close to that reference (what is left is mostly what is seen
        # through the glass, which the traced lens blurs by its own depth: measured 1.0 to 1.5)
        conv, _ = still('conv', THOMAS + MID + focus + ['frames=64', 'dof=1'])
        e = grey(conv, ref)
        check(e <= 2.5, 'settled, most of the shape %s the band: %.2f grey levels from the converged traced lens (bound 2.5)' % (name, e))

    # 3. A turning view (thirty frames round, the antialiasing running): closer to the converged reference of
    # the same view than the traced lens is, and steadier from one frame to the next than no lens at all.
    # Measured: 2.1 against 2.7 from the reference; frame to frame 0.5 against 1.3 with no lens and 2.2 traced.
    ref, _ = still('tref', THOMAS + SMALL + NEAR + ['turn=30', 'frames=256', 'lensN=8', 'dof=0'])
    seq = {}
    for mode, args in (('new', NEAR + ['dof=1']), ('traced', NEAR + ['dof=0']), ('none', ['aperture=0'])):
        seq[mode] = [still('turn_%s_%d' % (mode, n), THOMAS + SMALL + args + ['turn=%d' % n])[0] for n in (30, 31)]
    e1, e0 = grey(seq['new'][0], ref), grey(seq['traced'][0], ref)
    check(e1 <= e0, 'a turning view: %.2f grey levels from the converged traced lens, the traced lens itself %.2f' % (e1, e0))
    d1, d0, dn = (grey(*seq[m]) for m in ('new', 'traced', 'none'))
    check(d1 <= dn * 1.05, 'two frames of a turning view, one after the other, differ by %.2f grey levels; with no lens %.2f, with the traced lens %.2f (bound: no more than with no lens, give or take 5%%)' % (d1, dn, d0))

    # 4. Inside the band the picture is the picture with no lens.
    # a) the band takes in the whole shape: nothing is blurred, so nothing may change at all
    plain, t = still('plain', THOMAS + SMALL + ['frames=12', 'aperture=0'])
    whole, _ = still('whole', THOMAS + SMALL + ['frames=12', 'aperture=0.2', 'fnear=-1.6', 'ffar=1.6', 'dof=1'])
    check(np.array_equal(plain, whole), 'the band around the whole shape: the picture is the no-lens picture, bit for bit')
    # b) the band takes in the front half: there the glass is untouched, however blurred the rest is
    front, _ = still('front', THOMAS + SMALL + ['frames=12', 'aperture=0.3', 'fnear=-1.6', 'ffar=-0.3', 'dof=1'])
    e = depth_from_centre(t)
    inside = e < -0.32
    behind = e > -0.2
    same = np.all(front == plain, axis=2)
    check(inside.sum() > 2000 and same[inside].all(), 'the band around the front half: all %d pixels of glass inside it are as with no lens, bit for bit' % inside.sum())
    check(same[behind].mean() < 0.2, 'the same picture: %.0f%% of the glass behind the band has changed' % (100 * (1 - same[behind].mean())))

    # 5. No light is made or lost beyond what glass hides. A blur spreads each pixel's light over the texels
    # of its disc and divides by their number, so by itself it keeps the sum exactly. What changes the sum:
    # blur that falls behind sharp glass is hidden (lost), and what sharp or near glass hides is made up
    # (gained). Measured: -1.7% to +0.5% over ten pictures. Bound: 3%.
    for name, focus in (('behind', FAR), ('in front of', NEAR)):
        blur, _ = still('sum', THOMAS + SMALL + focus + ['frames=12', 'dof=1'])
        ratio = blur.sum() / plain.sum()
        check(abs(ratio - 1) <= 0.03, 'the light in the picture, most of the shape %s the band: %+.2f%% against no lens (bound 3%%)' % (name, 100 * (ratio - 1)))

    # 6. More lens, more blur; further from the band, more blur.
    g0 = edges(plain)
    glass = t > 0
    detail = []
    for a in (0.05, 0.1, 0.2, 0.3):
        pic, _ = still('ap', THOMAS + SMALL + ['frames=12', 'aperture=%g' % a, 'fnear=-1.6', 'ffar=-0.6', 'dof=1'])
        detail.append(edges(pic)[glass].sum() / g0[glass].sum())
    check(all(detail[i + 1] < detail[i] for i in range(3)) and detail[0] < 1.0, 'fine detail left as the lens opens (0.05, 0.1, 0.2, 0.3): %s of the no-lens picture\'s' % ', '.join('%.0f%%' % (100 * d) for d in detail))
    pic, _ = still('far', THOMAS + SMALL + ['frames=12', 'aperture=0.3', 'fnear=-1.6', 'ffar=-0.6', 'dof=1'])
    g1 = edges(pic)
    bins = [(-1.6, -0.7), (-0.5, -0.1), (-0.1, 0.3), (0.3, 0.7), (0.7, 1.6)]
    left = []
    for lo, hi in bins:
        m = glass & (e > lo) & (e <= hi)
        left.append(g1[m].sum() / max(g0[m].sum(), 1e-9))
    check(left[0] > 0.97 and all(left[i + 1] < left[i] for i in range(len(left) - 1)),
          'fine detail left by depth (inside the band, then four steps further behind it): %s' % ', '.join('%.0f%%' % (100 * d) for d in left))

    # 7. The shape of the lens opening shows, and is the traced lens's. A triangle turned a third of a turn
    # is the same triangle; turned a sixth, it is upside down.
    tri = ['aperture=0.3', 'fnear=-1.2', 'ffar=-0.8', 'beads=0.5', 'thick=0.5']
    pics = {}
    for name, extra in (('round', []), ('tri0', ['blades=3']), ('tri60', ['blades=3', 'bokehRot=60']), ('tri120', ['blades=3', 'bokehRot=120'])):
        pics[name] = still('shape_' + name, THOMAS + SMALL + tri + extra + ['frames=12', 'dof=1'])[0]
    a, b, c = grey(pics['round'], pics['tri0']), grey(pics['tri0'], pics['tri60']), grey(pics['tri0'], pics['tri120'])
    check(a > 0.1 and b > 0.1 and c < 0.02, 'three blades against a round opening: %.2f grey levels apart; turned 60 degrees: %.2f; turned 120 degrees: %.3f (the same triangle)' % (a, b, c))
    ref, _ = still('shape_ref', THOMAS + SMALL + tri + ['blades=3', 'frames=256', 'lensN=8', 'dof=0'])
    a, b = grey(pics['tri0'], ref), grey(pics['tri60'], ref)
    check(a < b, 'three blades against the traced lens with three blades: %.2f grey levels; with the triangle upside down %.2f' % (a, b))

    # 8. What the stage costs a turning frame at 3840 x 2160 (the quickest of several goes, lens against no
    # lens in turn). Measured: 0.6 ms at an aperture of 0.12, 1.25 ms at 0.3. Bounds: 1.5 and 2.5 ms.
    for focus, bound in ((['aperture=0.12', 'fnear=-0.4', 'ffar=-0.1'], 1.5), (NEAR, 2.5)):
        r = json.loads(run(EXE, ['--bench-focus', '30', 'rounds=6', 'w=3840', 'h=2160'] + THOMAS + focus + ['dof=1']))
        check(r['focusMs'] <= bound, 'cost at 3840 x 2160, %s: %.2f ms on top of %.2f ms (bound %.1f ms)' % (focus[0].replace('=', ' '), r['focusMs'], r['noLensMs'], bound))

    if '--keep' not in sys.argv:
        for f in os.listdir(OUT):
            os.remove(os.path.join(OUT, f))
        os.rmdir(OUT)
    bad = results.count(False)
    print('dof_check: %d checks, %s' % (len(results), 'all passed' if bad == 0 else '%d FAILED' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as err:                # a crash is a failure too, and must say so in the same form
        print('FAIL dof_check could not run: %s' % err)
        print('dof_check: could not run')
        sys.exit(1)
