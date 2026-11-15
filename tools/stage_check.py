"""The studio of the native Glass Forms: backdrop, floor, the camera raised above it, tone curve and grade.

usage: python tools/stage_check.py [--old <program>] [--keep] [--quick]        (from D:\\glassforms\\native)

Runs the program with no window (target\\release\\glassforms.exe, or the one named by GLASSFORMS_EXE) and
looks at what it draws. One line per check, beginning "ok" or "FAIL"; a count at the end; exit status 1 if
anything failed. Pictures go to out\\stage_check.

  --old <program>   the program as built before the studio went in: with no studio, stills and moving
                    pictures must be the ones it draws, byte for byte
  --quick           leave out the timings

What is checked, and by what measure (pictures are read before bloom and tone mapping unless said):
  nothing moved    without a studio the program draws what it drew (with --old); a studio that is switched
                   on but black and floorless leaves every pixel as it was
  backdrop         where a ray misses the shape, the picture is the backdrop's own arithmetic (worked out
                   here again, independently) to 0.002; through glass that bends nothing (index 1) the same
                   backdrop is seen in the same direction, to 0.03 for a backdrop with a glow 6 degrees wide
                   and edges in the picture (it is read from the lights' map there: 256 x 128 texels, 1.4
                   degrees each), to 0.002 where it is smooth; the same with the camera raised; backShow=0 shows black and lights
                   the glass all the same
  camera height    elev tilts shape and lights together: the picture is the one made by turning the shape
                   and pitching the lights by hand
  floor            the distance stored with every floor pixel is the distance to a level plane (to 0.2%,
                   which is what a 16-bit float holds), whatever way the shape is turned; the floor touches
                   the lowest point of the shape (the lowest glass in the picture stands within 0.01 of
                   it); turning the shape about the vertical leaves the floor's height (to 0.003) and its
                   pixels (exactly) alone
  in motion        a still drawn while the shape turns on its turntable: the floor changes from one frame
                   to the next by less than 0.05 grey levels (the glass itself: over 1)
  one frame        the first frame's floor and backdrop are the settled picture's, to 0.3 grey levels
  through glass    glass that bends nothing shows the floor behind it as the floor is, to 0.03
  focus            floor inside the band in focus is, bit for bit, the floor with no lens; floor in front
                   of the band and behind it is blurred (the shape's image in a glossy floor loses its
                   detail there); a blurred floor does not darken towards the picture's edges
  tone and grade   each curve gives another picture; neutral shows a pastel as it is (within 1.5 grey
                   levels) where the filmic curve moves it by more than 5; contrast, saturation and white
                   balance do what they say; the 16-bit still is toned as the screen is (within 1 level)
  studios          every studio draws, bright or dark as it says; a setup saved with one draws the same
  clay             is lit by the backdrop
  cost             a turning 1920 x 1080 view with each thing switched on and without it, in turn within
                   one run, the quickest go of each: a backdrop (limit 1.15 times), a floor (1.2), a floor
                   that follows a tumbling shape (2), a glossy floor (1.3), the shape mirrored in it (2),
                   tone curve and grade (1.1)"""
import json, math, os, struct, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'stage_check')
OLD = os.path.abspath(sys.argv[sys.argv.index('--old') + 1]) if '--old' in sys.argv else None
QUICK = '--quick' in sys.argv
os.makedirs(OUT, exist_ok=True)
results = []


def say(name, ok, detail=''):
    results.append(bool(ok))
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)


def out(name):
    return os.path.join(OUT, name)


def run(*args, exe=None, env=None):
    """Run the program; returns the last line of JSON it printed."""
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([exe or EXE] + [str(a) for a in args], capture_output=True, text=True, cwd=NATIVE, timeout=900, env=e)
    lines = [l for l in r.stdout.splitlines() if l.startswith('{')]
    if r.returncode != 0 and not lines:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return json.loads(lines[-1]) if lines else {}


def still(name, *args, size=(640, 360), frames=24, exe=None, env=None):
    """A still as the screen shows it: rows x columns x 3, 0 to 255; and what the program reported."""
    path = out(name + '.png')
    info = run('--still', path, 'w=%d' % size[0], 'h=%d' % size[1], 'frames=%d' % frames, *args, exe=exe, env=env)
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float64), info


def traced(name, *args, size=(480, 270), frames=1, env=None):
    """The picture before bloom and tone mapping (after focus, if there is a lens): rows x columns x 4
    floats (red, green, blue, and the distance stored with the pixel); and what the program reported.
    One frame with no antialiasing unless told otherwise, so that every pixel is one ray."""
    path = out(name + '.f32')
    extra = ['smooth=0'] if frames == 1 else []
    info = run('--still', out(name + '.png'), '--hdr', path, 'w=%d' % size[0], 'h=%d' % size[1], 'frames=%d' % frames, *extra, *args, env=env)
    raw = open(path, 'rb').read()
    w, h = struct.unpack('<II', raw[:8])
    return np.frombuffer(raw, dtype='<f4', offset=8).reshape(h, w, 4).astype(np.float64), info


def guarded(name, f):
    try:
        f()
    except Exception as e:
        say(name, False, '%s: %s' % (type(e).__name__, str(e)[:400]))


# ---- the studio's arithmetic, worked out here again ------------------------------------------------------

def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rays(w, h, fov=26.0, dist=6.6, elev=0.0):
    """Every pixel's ray through its middle: where it starts (the camera) and its direction, both in the
    room; rows from the top of the picture."""
    tan = math.tan(math.radians(fov) / 2)
    x = (2 * (np.arange(w) + 0.5) - w) / h
    y = (2 * (h - np.arange(h) - 0.5) - h) / h
    d = np.stack(np.broadcast_arrays(x[None, :] * tan, y[:, None] * tan, -np.ones((h, w))), axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    to_room = rot_x(elev).T
    return to_room @ np.array([0.0, 0.0, dist]), d @ to_room.T


def backdrop(d, top, mid, low, spread, glow=(0, 0, 0), glow_az=0.0, glow_el=0.0, glow_size=30.0, gain=1.0, wrap=180.0, high=90.0):
    """The backdrop in directions d of the room (.., 3)."""
    top, mid, low, glow = (np.array(c, dtype=float) * gain for c in (top, mid, low, glow))
    u = np.clip(np.arcsin(np.clip(d[..., 1], -1, 1)) / math.radians(spread), -1, 1)
    a = np.abs(u)
    w = (a * a * (3 - 2 * a))[..., None]
    col = mid + (np.where(u[..., None] > 0, top, low) - mid) * w
    if wrap < 180:
        # whole up to `wrap` degrees round from straight behind the shape, gone 30 degrees further
        whole, gone = math.cos(math.radians(wrap)), math.cos(math.radians(min(wrap + 30.0, 180.0)))
        s = np.clip((-d[..., 2] - gone) / (whole - gone), 0, 1)
        col = col * (s * s * (3 - 2 * s))[..., None]
    if high < 90:
        # whole up to `high` degrees above the horizon, gone 30 degrees further up
        under, over = math.sin(math.radians(high)), math.sin(math.radians(min(high + 30.0, 90.0)))
        s = np.clip((d[..., 1] - under) / (over - under), 0, 1)
        col = col * (1 - s * s * (3 - 2 * s))[..., None]
    az, el = math.radians(glow_az), math.radians(glow_el)
    g = np.array([math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)])
    return col + glow * np.exp((d @ g - 1.0) / math.radians(glow_size) ** 2)[..., None]


def grown(mask, n=1):
    """A mask with everything within n pixels of it added."""
    o = mask.copy()
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            o |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return o


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255


# a backdrop with strong, unequal colours and a glow off to one side, so that a mistake in any part shows
BACK = dict(top=(0.2, 0.5, 0.9), mid=(0.9, 0.8, 0.3), low=(0.6, 0.2, 0.5), spread=14.0, glow=(0.5, 0.9, 0.2), glow_az=-9.0, glow_el=4.0, glow_size=6.0, wrap=8.0, high=3.0)
BACK_ARGS = ['backdrop=1', 'backSpread=14', 'backTopR=0.2', 'backTopG=0.5', 'backTopB=0.9', 'backMidR=0.9', 'backMidG=0.8', 'backMidB=0.3',
             'backLowR=0.6', 'backLowG=0.2', 'backLowB=0.5', 'backGlow=1', 'backGlowR=0.5', 'backGlowG=0.9', 'backGlowB=0.2',
             'backGlowAz=-9', 'backGlowEl=4', 'backGlowSize=6', 'backWrap=8', 'backHigh=3']
# no lights at all, and glass that neither bends nor reflects nor tints: it shows what is behind it
DARK = ['rig=', 'ambient=0,0,0']
AIR = ['ior=1', 'disp=0', 'absorb=0,0,0']
THOMAS = ['scene=attractors', 'preset=thomas']
CUBE = ['scene=solids', 'preset=cube']
RING = ['scene=solids', 'preset=ring']
VIEW = dict(fov=26.0, dist=6.6)
CAM = ['fov=26', 'dist=6.6']


# ------------------------------------------------------------------ nothing moved
def nothing_moved():
    cases = {
        'thomas': THOMAS,
        'gyroid': ['scene=surfaces', 'preset=gyroid'],
        'lens': THOMAS + ['aperture=0.12', 'fnear=-0.4', 'ffar=-0.1'],
        'traced lens': THOMAS + ['dof=0', 'aperture=0.12', 'lensN=4'],
        'clay': THOMAS + ['clay=1'],
        'moving': THOMAS + ['turn=10'],
        'frosted': THOMAS + ['rough=0.3'],
    }
    if OLD:
        bad = []
        for name, args in cases.items():
            a, _ = still('old-' + name.replace(' ', '-'), *args, frames=8, exe=OLD)
            b, _ = still('new-' + name.replace(' ', '-'), *args, frames=8)
            if a.shape != b.shape or np.abs(a - b).max() > 0:
                bad.append(name)
        say('nothing moved: with no studio, %d pictures are the old program\'s, byte for byte' % len(cases), not bad, 'differ: ' + ', '.join(bad) if bad else '')
    else:
        say('nothing moved: not compared with the program as it was (give --old <program>)', True)
    # a studio that is on, but black and without a floor: another tracer is built, and must draw the same
    a, _ = traced('plain', *THOMAS, frames=4)
    b, _ = traced('black-studio', *THOMAS, 'backdrop=1', 'backGain=0', frames=4)
    worst = np.abs(a - b).max()
    say('nothing moved: a black backdrop leaves every pixel as it was', worst == 0, 'largest difference %.2g' % worst)


# ------------------------------------------------------------------ the backdrop
def the_backdrop():
    for elev in (0.0, 30.0):
        tag = 'elev%d' % elev
        # (32 frames: a pixel's rays land all over it, and their average is the colour at its middle)
        # (a cube: through a shape that folds over itself the far part's outline shows inside the near part)
        px, _ = traced('backdrop-' + tag, *CUBE, *CAM, *DARK, *AIR, *BACK_ARGS, 'elev=%g' % elev, frames=32)
        h, w = px.shape[:2]
        _, d = rays(w, h, elev=elev, **VIEW)
        want = backdrop(d, **BACK)
        glass = px[..., 3] > 0
        # (pixels beside the glass, where a ray grazed it, carry some of the glass's sheen: left out)
        behind = ~grown(glass, 2)
        err = np.abs(px[..., :3] - want).max(axis=-1)
        say('backdrop (%s): behind the shape it is the backdrop\'s own arithmetic' % tag, glass.sum() > 2000 and err[behind].max() < 0.002,
            'largest difference %.5f over %d pixels' % (err[behind].max(), behind.sum()))
        # Glass of index 1. The map the glass reads has 256 x 128 texels (1.4 degrees each), for a backdrop
        # that here changes fully within 14 degrees and has a glow 6 degrees wide. At its very outline
        # even such glass mirrors (light arriving along the surface is all turned back), so the largest
        # difference is taken leaving out the worst pixel in a hundred.
        inner = glass & ~grown(~glass)
        most = np.percentile(err[inner], 99)
        say('backdrop (%s): glass that bends nothing shows the same backdrop in the same direction' % tag, most < 0.03 and np.median(err[inner]) < 0.004,
            'largest difference %.4f (but for the outline), median %.5f, over %d pixels of glass' % (most, np.median(err[inner]), inner.sum()))
    shown, _ = traced('backdrop-shown', *THOMAS, *CAM, *DARK, *BACK_ARGS)
    hidden, _ = traced('backdrop-hidden', *THOMAS, *CAM, *DARK, *BACK_ARGS, 'backShow=0')
    glass = shown[..., 3] > 0
    behind = ~grown(glass, 2)
    say('backdrop: backShow=0 shows black behind the shape and lights the glass all the same',
        hidden[behind][:, :3].max() == 0 and np.abs(shown[glass] - hidden[glass]).max() == 0 and shown[glass][:, :3].mean() > 0.05,
        'behind %.3g; glass differs by %.3g' % (hidden[behind][:, :3].max(), np.abs(shown[glass] - hidden[glass]).max()))


# ------------------------------------------------------------------ the camera raised
def camera_height():
    a, _ = still('elev-by-setting', *THOMAS, 'elev=20', 'rx=10', 'ry=0', 'rz=0', frames=32)
    b, _ = still('elev-by-hand', *THOMAS, 'rx=30', 'ry=0', 'rz=0', 'lpitch=20', frames=32)
    c, _ = still('elev-none', *THOMAS, 'rx=10', 'ry=0', 'rz=0', frames=32)
    same, moved = np.abs(a - b).mean(), np.abs(a - c).mean()
    say('camera height: elev tilts shape and lights together, as turning both by hand does', same < 0.05 and moved > 1.0,
        '%.4f grey levels from the picture made by hand; %.2f from the picture with no tilt' % (same, moved))


# ------------------------------------------------------------------ the floor
def floor_mask(px, eye, d, height, tol=0.002, fov=26.0):
    """Which pixels show the floor: those whose stored distance is the distance at which a ray through
    the pixel meets the plane y = height. A pixel's one ray goes through some point of the pixel, not its
    middle, and towards the horizon the distance changes a great deal within a pixel, so the stored
    distance is held between the distances at the pixel's upper and lower edges. Also returns the
    distance along the ray through the middle."""
    h = px.shape[0]
    half = 0.6 * 2 * math.tan(math.radians(fov) / 2) / h          # a little over half a pixel, as a change of slope
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = (height - eye[1]) / d[..., 1]
        near = (height - eye[1]) / (d[..., 1] - half)
        far = np.where(d[..., 1] + half < 0, (height - eye[1]) / (d[..., 1] + half), np.inf)
    t = px[..., 3]
    return (d[..., 1] - half < 0) & (t > 0) & (t > near * (1 - tol) - 1e-3) & (t < far * (1 + tol) + 1e-3), tf


FLOOR = ['floor=1', 'backdrop=1', 'floorFade=40']


def the_floor():
    heights, pictures = [], []
    for k, (elev, turn) in enumerate([(0.0, 0), (25.0, 0), (25.0, 40), (25.0, 80)]):
        # (the shape's angles are applied x, then y, then z: with no turn about z, y is the room's vertical)
        px, info = traced('floor-%d' % k, *CUBE, *CAM, *FLOOR, 'elev=%g' % elev, 'rx=25', 'ry=%d' % turn, 'rz=0')
        height = info['stage']['floor']
        h, w = px.shape[:2]
        eye, d = rays(w, h, elev=elev, **VIEW)
        mask, tf = floor_mask(px, eye, d, height)
        below = (tf > 0) & (px[..., 3] > 0)                  # everything with a distance where the ray meets the plane
        glass = (px[..., 3] > 0) & ~mask
        p = eye + d * px[..., 3:4]
        if k == 0:
            # The camera level with the shape: the lowest glass in the picture is the shape's lowest point.
            # (Only where the view is steep enough for a pixel's middle to stand for its ray.)
            lowest = p[..., 1][glass & ((d[..., 1] < -0.1) | (d[..., 1] > 0))].min()
            say('floor: it touches the lowest point of the shape', abs(lowest - height) < 0.01, 'floor at %.4f, lowest glass at %.4f' % (height, lowest))
        heights.append(height)
        pictures.append((px, mask, glass))
        # every pixel below the horizon that is not glass is floor: its distance is the plane's
        # Every pixel below the horizon is floor or glass in front of it. The floor's points (each pixel's
        # middle carried out to its stored distance; a pixel's ray does not go through its middle, but
        # that averages out over thousands of pixels) are fitted with a plane: it must be level, and at
        # the floor's height.
        steep = mask & (d[..., 1] < -0.12) & ~grown(glass, 2)
        q = p[steep]
        (ax, az, c), *_ = np.linalg.lstsq(np.stack([q[:, 0], q[:, 2], np.ones(len(q))], axis=1), q[:, 1], rcond=None)
        # (astray: a pixel within the floor's reach that shows nothing, or something beyond the floor)
        stray = below & ~mask & (tf < 25) & ((px[..., 3] == 0) | (px[..., 3] > 1.05 * tf))
        say('floor: level (camera raised %g, shape turned %d): stored distances lie on one level plane' % (elev, turn),
            mask.sum() > 20000 and stray.sum() == 0 and abs(ax) < 0.002 and abs(az) < 0.002 and abs(c - height) < 0.004,
            '%d floor pixels, %d astray; the plane through them slopes %.5f and %.5f and stands at %.4f (the floor: %.4f)' % (mask.sum(), stray.sum(), ax, az, c, height))
    spread = max(heights[1:]) - min(heights[1:])
    # (floor pixels in all three, clear of the glass: where glass touches the floor the two cannot be told apart)
    both = pictures[1][1] & pictures[2][1] & pictures[3][1] & ~grown(pictures[1][2] | pictures[2][2] | pictures[3][2], 3)
    change = max(np.abs(pictures[1][0][both] - pictures[k][0][both]).max() for k in (2, 3))
    say('floor: the shape turned about the vertical leaves the floor where it was', spread < 0.003 and change == 0,
        'heights within %.5f; %d floor pixels common to all three, largest change %.3g' % (spread, both.sum(), change))
    # with the height set by hand the floor is where it is put
    px, info = traced('floor-by-hand', *CUBE, *CAM, *FLOOR, 'floorAuto=0', 'floorY=-0.8', 'elev=25')
    say('floor: floorAuto=0 puts it at floorY', abs(info['stage']['floor'] + 0.8) < 1e-6, 'at %.4f' % info['stage']['floor'])


def in_motion():
    # the floor while the shape turns on its turntable: the still after 24 frames of turning and after 25
    args = [*CUBE, *CAM, *FLOOR, 'elev=25', 'spinMode=1', 'rx=25', 'rz=0', 'floorGloss=0.5', 'grain=0']
    a, ia = traced('motion-24', *args, 'drift=spin:1', frames=24)
    b, ib = traced('motion-25', *args, 'drift=spin:1', frames=25)
    h, w = a.shape[:2]
    eye, d = rays(w, h, elev=25.0, **VIEW)
    ma, _ = floor_mask(a, eye, d, ia['stage']['floor'])
    mb, _ = floor_mask(b, eye, d, ib['stage']['floor'])
    floor, glass = ma & mb & ~grown(~ma & (a[..., 3] > 0), 2), (a[..., 3] > 0) & ~ma & (b[..., 3] > 0) & ~mb
    tone = lambda x: srgb(x / (1 + x))
    df = np.abs(tone(a[..., :3]) - tone(b[..., :3])).mean(axis=-1)
    say('in motion: the floor stands still while the shape turns on it', floor.sum() > 20000 and df[floor].mean() < 0.05 and df[glass].mean() > 0.2 and ia['stage']['measures'] == 1,
        'from one frame to the next the floor changes by %.4f grey levels (the glass: %.2f); its height was measured %d time(s) in %d frames' % (df[floor].mean(), df[glass].mean(), ia['stage']['measures'], 24 * 2))


def one_frame():
    args = [*THOMAS, *CAM, 'studio=lavender', 'elev=20', 'floorGloss=0.3']
    a, ia = traced('first-frame', *args)
    b, _ = traced('settled', *args, frames=128)
    h, w = a.shape[:2]
    eye, d = rays(w, h, elev=20.0, **VIEW)
    floor, _ = floor_mask(a, eye, d, ia['stage']['floor'])
    room = floor | (a[..., 3] == 0)
    # (away from the outline of the glass, where one frame has one ray a pixel and the settled picture many)
    near = np.zeros_like(room)
    g = ~room
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= np.roll(np.roll(g, dy, 0), dx, 1)
    room &= ~near
    tone = lambda x: srgb(np.clip(x, 0, 1))
    df = np.abs(tone(a[..., :3]) - tone(b[..., :3])).mean(axis=-1)[room]
    say('one frame: floor and backdrop are there in the first frame as in the settled picture', df.mean() < 0.3 and df.max() < 2.0, 'mean %.3f grey levels, largest %.2f, over %d pixels' % (df.mean(), df.max(), room.sum()))


def through_glass():
    # a floor and a backdrop of one even colour, no lights: glass that bends nothing must vanish
    even = ['backdrop=1', 'floor=1', 'floorR=1', 'floorG=1', 'floorB=1', 'backSpread=30'] + ['back%s%s=0.5' % (p, c) for p in ('Top', 'Mid', 'Low') for c in 'RGB']
    px, info = traced('through-glass', *THOMAS, *CAM, *DARK, *AIR, *even, 'elev=25')
    glass_err = np.abs(px[..., :3] - 0.5).max()
    say('through glass: glass that bends nothing vanishes against a floor and backdrop of one colour', glass_err < 0.03, 'largest difference %.4f (the floor\'s own colour came to %s)' % (glass_err, info['stage']['floorRgb']))
    # and the floor the glass sees is the floor the camera sees: a dark floor under a bright backdrop
    dark = ['backdrop=1', 'floor=1', 'floorGain=0.3', 'floorFade=60'] + ['back%s%s=0.6' % (p, c) for p in ('Top', 'Mid', 'Low') for c in 'RGB']
    px, info = traced('through-glass-dark', *THOMAS, *CAM, *DARK, *AIR, *dark, 'elev=40')
    h, w = px.shape[:2]
    eye, d = rays(w, h, elev=40.0, **VIEW)
    floor, tf = floor_mask(px, eye, d, info['stage']['floor'])
    glass = (px[..., 3] > 0) & ~floor & (tf > 0) & (tf < 12)
    own = np.array(info['stage']['floorRgb'])
    seen = px[..., :3][glass]
    say('through glass: the floor seen through the glass is the floor beside it', glass.sum() > 2000 and np.abs(seen - own).mean() < 0.03 and np.abs(px[..., :3][floor & (tf < 12)] - own).max() < 0.02,
        'the floor is %.3f; through the glass %.3f on average (%d pixels)' % (own.mean(), seen.mean(), glass.sum()))


# ------------------------------------------------------------------ focus
def focus():
    # a glossy floor that mirrors the shape gives the floor detail to lose; the band lies across the floor
    args = [*CUBE, *CAM, 'studio=dusk', 'elev=18', 'rx=20', 'rz=10', 'floorGloss=1', 'fnear=-0.9', 'ffar=-0.5']
    sharp, info = traced('focus-none', *args, frames=48, size=(640, 360))
    lens, _ = traced('focus-lens', *args, 'aperture=0.3', frames=48, size=(640, 360))
    h, w = sharp.shape[:2]
    eye, d = rays(w, h, elev=18.0, **VIEW)
    floor, tf = floor_mask(sharp, eye, d, info['stage']['floor'])
    # depth from the middle of the shape along the line of sight, as the band's planes are given
    _, dc = rays(w, h, elev=0.0, **VIEW)
    depth = sharp[..., 3] * -dc[..., 2] - VIEW['dist']
    inside = floor & (depth > -0.85) & (depth < -0.55)
    # (pixels well inside the band and away from blurred neighbours: a blur spreads onto what is next to it)
    core = inside.copy()
    for dy in range(-6, 7, 3):
        for dx in range(-6, 7, 3):
            core &= np.roll(np.roll(inside, dy, 0), dx, 1)
    same = np.abs(sharp[..., :3] - lens[..., :3])[core].max() if core.any() else 9
    say('focus: the floor inside the band is the floor with no lens, bit for bit', core.sum() > 300 and same == 0, '%d pixels, largest difference %.3g' % (core.sum(), same))
    # detail = how much a pixel differs from the mean of its four neighbours
    def detail(p):
        lum = p[..., :3].mean(axis=-1)
        return np.abs(lum - (np.roll(lum, 1, 0) + np.roll(lum, -1, 0) + np.roll(lum, 1, 1) + np.roll(lum, -1, 1)) / 4)
    behind = floor & (depth > 0.3)
    busy = behind & (detail(sharp) > np.percentile(detail(sharp)[behind], 90))
    lost = detail(lens)[busy].mean() / detail(sharp)[busy].mean()
    say('focus: the floor behind the band is blurred', busy.sum() > 100 and lost < 0.6, 'the detail of the shape\'s image in the floor falls to %.2f of what it was' % lost)
    # a blurred floor keeps its brightness right up to the picture's edges
    args = [*CUBE, *CAM, 'studio=white', 'elev=30', 'fnear=-0.1', 'ffar=0.1']
    plain, _ = traced('edges-none', *args, frames=8)
    blur, _ = traced('edges-lens', *args, 'aperture=0.3', frames=8)
    edge = np.zeros(plain.shape[:2], bool)
    edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = True
    ratio = blur[..., :3][edge].mean() / plain[..., :3][edge].mean()
    say('focus: a blurred floor does not darken towards the edges of the picture', abs(ratio - 1) < 0.02, 'the outermost three pixels are %.4f times as bright as with no lens' % ratio)


# ------------------------------------------------------------------ tone and grade
def tone_and_grade():
    base = [*THOMAS, *CAM, 'studio=lavender', 'elev=15', 'bloom=0', 'vignette=0', 'grain=0', 'exposure=0.6', 'saturation=1']
    px, _ = traced('tone-traced', *base, frames=16)
    pics = {}
    # (bright enough that every curve has something to press: below 0.6 of white three of them agree)
    bright = [still('tone-bright-%d' % tone, *base, 'exposure=2.5', 'tone=%d' % tone, frames=16, size=(480, 270))[0] for tone in range(4)]
    for tone in range(4):
        pics[tone], _ = still('tone-%d' % tone, *base, 'tone=%d' % tone, frames=16, size=(480, 270))
    differ = min(np.abs(bright[a] - bright[b]).mean() for a in range(4) for b in range(a))
    say('tone: each of the four curves gives another picture', differ > 0.3, 'the two most alike are %.2f grey levels apart' % differ)
    # pastel pixels: everything below 0.8, well away from black
    lin = px[..., :3] * 0.6
    pastel = (lin.max(axis=-1) < 0.78) & (lin.min(axis=-1) > 0.3)
    as_is = srgb(lin)
    held = np.abs(pics[1] - as_is)[pastel]
    moved = np.abs(pics[0] - as_is)[pastel]
    say('tone: neutral shows a pastel as it is; the filmic curve does not', pastel.sum() > 5000 and held.mean() < 1.5 and moved.mean() > 5,
        'neutral %.2f grey levels from the colour itself, filmic %.2f (%d pixels)' % (held.mean(), moved.mean(), pastel.sum()))
    sat = lambda p: (p.max(axis=-1) - p.min(axis=-1)).mean()
    grey, _ = still('grade-grey', *base, 'tone=1', 'saturation=0', frames=16, size=(480, 270))
    rich, _ = still('grade-rich', *base, 'tone=1', 'saturation=1.5', frames=16, size=(480, 270))
    say('grade: saturation 0 is grey, 1.5 is stronger', sat(grey) < 1.0 and sat(rich) > 1.2 * sat(pics[1]), 'colour spread %.2f at 0, %.2f at 1, %.2f at 1.5' % (sat(grey), sat(pics[1]), sat(rich)))
    hard, _ = still('grade-contrast', *base, 'tone=1', 'contrast=1.6', frames=16, size=(480, 270))
    say('grade: contrast draws light and dark apart', hard.std() > 1.15 * pics[1].std(), 'spread of levels %.1f against %.1f' % (hard.std(), pics[1].std()))
    warm, _ = still('grade-warm', *base, 'tone=1', 'wbTemp=0.6', frames=16, size=(480, 270))
    green, _ = still('grade-green', *base, 'tone=1', 'wbTint=-0.6', frames=16, size=(480, 270))
    m = lambda p: p.reshape(-1, 3).mean(axis=0)
    say('grade: warmth adds red and takes blue; a negative tint adds green', m(warm)[0] > m(pics[1])[0] + 2 and m(warm)[2] < m(pics[1])[2] - 2 and m(green)[1] - m(green)[0] > m(pics[1])[1] - m(pics[1])[0] + 2,
        'red, green, blue: %s as it is, %s warm, %s green' % tuple(np.round(m(p)).astype(int).tolist() for p in (pics[1], warm, green)))
    # the 16-bit still goes through the same curve and grade, worked out on the CPU
    worst = 0.0
    for tone, extra in [(1, []), (2, ['contrast=1.3']), (3, ['saturation=0.6', 'wbTemp=0.3']), (0, ['contrast=0.8'])]:
        name = 'deep-%d' % tone
        a, _ = still(name + '-8', *base, 'tone=%d' % tone, *extra, frames=16, size=(480, 270))
        run('--still', out(name + '-16.png'), '--16', 'w=480', 'h=270', 'frames=16', *base, 'tone=%d' % tone, *extra)
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', out(name + '-16.png'), '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], capture_output=True).stdout
        b = np.frombuffer(raw, dtype='<u2').astype(np.float64).reshape(a.shape) / 257.0
        # (the 8-bit still carries half a level of dither)
        worst = max(worst, np.abs(a - b).mean())
    say('tone: the 16-bit still is toned and graded as the screen is', worst < 0.6, 'at most %.2f grey levels apart on average' % worst)


# ------------------------------------------------------------------ studios, setups, clay
def studios():
    want = {'lavender': (150, 256), 'ice': (150, 256), 'white': (170, 256), 'dusk': (3, 150)}
    for name, (lo, hi) in want.items():
        px, info = still('studio-' + name, *THOMAS, 'studio=' + name, 'elev=20', frames=16)
        level = px.mean()
        say('studio %s: draws, with a backdrop and a floor' % name, lo < level < hi and info['stage']['floor'] is not None and info['stage']['shown'], 'mean level %.0f' % level)
    a, _ = still('setup-saved', *CUBE, 'studio=lavender', 'elev=25', 'floorGloss=0.4', 'tone=2', '--save-setup', out('studio-setup.json'), frames=16)
    b, _ = still('setup-loaded', '--setup', out('studio-setup.json'), frames=16)
    say('setups: a setup saved with a studio draws the same still', np.abs(a - b).max() <= 1, 'largest difference %d' % np.abs(a - b).max())
    lit, _ = still('clay-studio', *THOMAS, 'clay=1', 'studio=white', frames=8)
    unlit, _ = still('clay-plain', *THOMAS, 'clay=1', frames=8)
    say('clay: stands on the floor in front of the backdrop', lit.mean() > unlit.mean() + 60 and lit.std() > 5, 'mean level %.0f with the studio, %.0f without' % (lit.mean(), unlit.mean()))


# ------------------------------------------------------------------ cost
def cost():
    # each thing switched on and off in turn within one run, the quickest go of each: other work on the
    # GPU can only make a go slower, and what it does to the one it does to the other
    floor = 'backdrop:1,floor:1,floorAuto:0,floorY:-1.1'
    rows = [('a backdrop', [], 'backdrop:1', 1.15), ('a floor (height by hand)', ['elev=20'], floor, 1.2),
            ('a floor that follows a tumbling shape', ['elev=20'], 'backdrop:1,floor:1', 2.0),
            ('a glossy floor', ['elev=20'], floor + ',floorGloss:0.6', 1.3),
            ('the shape mirrored in it', ['elev=20'], floor + ',floorGloss:0.6,floorMirror:1', 2.0),
            ('tone curve and grade', [], 'tone:1,contrast:1.2,saturation:1.1,wbTemp:0.2', 1.1)]
    for name, same, with_, limit in rows:
        r = run('--bench-with', '30', 'rounds=5', 'w=1920', 'h=1080', *THOMAS, *same, 'with=' + with_)
        say('cost: %s' % name, r['withMs'] < limit * r['withoutMs'] + 0.1, '%.2f ms a turning 1920 x 1080 frame, %.2f without (%.2f times; limit %.2f)' % (r['withMs'], r['withoutMs'], r['times'], limit))


guarded('nothing moved', nothing_moved)
guarded('the backdrop', the_backdrop)
guarded('camera height', camera_height)
guarded('the floor', the_floor)
guarded('in motion', in_motion)
guarded('one frame', one_frame)
guarded('through glass', through_glass)
guarded('focus', focus)
guarded('tone and grade', tone_and_grade)
guarded('studios', studios)
if not QUICK:
    guarded('cost', cost)

print('%d of %d checks passed' % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
