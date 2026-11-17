r"""Settings taken to their hard limits (core/src/limits.rs) and past them, to see that nothing breaks; and the
sliders' keys and typed values in a scripted window.

    python tools/limits_probe.py [out dir] [stills|window|all]

Every run goes through the shared GPU queue (D:\glassforms\native\tools\gpu_run.py) with this copy's program.
A still is a failure if the program ends badly, writes crash.log, or draws nothing; a still that comes out
black is listed for a look (some settings are meant to give black). The window part drives the sliders
with the keys held (alt, alt+shift, ctrl+alt+shift), types values into them, and checks what the
settings became.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
CRASH = os.path.join(os.path.dirname(EXE), 'crash.log')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'

MILKY = ['sss=1', 'bodyR=0.9', 'bodyG=0.9', 'bodyB=0.9']
CAST = ['cast=2', 'floor=1']
# (name, settings): each at a hard limit, or past one (the program holds it there)
CASES = [
    ('ior lo', ['ior=1']), ('ior hi', ['ior=4']), ('ior past', ['ior=1e9']), ('ior below', ['ior=-5']),
    ('disp hi', ['disp=1']), ('nb hi', ['nb=32']), ('nb 0', ['nb=0']), ('ns hi', ['ns=32']), ('nw hi', ['nw=32', 'disp=0.3']),
    ('exposure 0', ['exposure=0']), ('exposure hi', ['exposure=1000']), ('scale hi', ['scale=2']), ('scale lo', ['scale=0.1']),
    ('scale past', ['scale=50']), ('smooth hi', ['smooth=0.99']), ('fov lo', ['fov=0.1']), ('fov hi', ['fov=175']),
    ('fov past', ['fov=400']), ('dist lo', ['dist=0.05']), ('dist hi', ['dist=10000']), ('dist below', ['dist=-3']),
    ('aperture hi', ['aperture=10000', 'dof=0', 'lensN=64']), ('aperture hi blur', ['aperture=10000', 'dof=1']),
    ('blades 1', ['aperture=0.1', 'blades=1']), ('blades 2', ['aperture=0.1', 'blades=2']), ('blades 16', ['aperture=0.1', 'blades=16']),
    ('distort hi', ['distort=0.9']), ('distort lo', ['distort=-0.9']), ('squeeze lo', ['aperture=0.1', 'squeeze=0.1']),
    ('squeeze hi', ['aperture=0.1', 'squeeze=10']), ('tilt hi', ['aperture=0.1', 'tiltX=85', 'tiltY=-85']),
    ('focal lo', ['focal=0.5']), ('focal 0', ['focal=0']), ('sensor lo', ['sensor=1']), ('fstop lo', ['focal=50', 'fstop=0.01']),
    ('subject lo', ['subject=0.01', 'fstop=1.4']), ('proj 0..6', ['proj=6']), ('proj past', ['proj=99']),
    ('fringe hi', ['fringe=4']), ('vignette hi', ['vignette=4']), ('bloom hi', ['bloom=10']), ('grain hi', ['grain=1']),
    ('speed lo', ['speed=0.01', 'anim=3']), ('spin hi', ['spin=50', 'anim=2']),
    ('cast detail hi', CAST + ['castDetail=3', 'castCaustics=100']), ('cast soft hi', CAST + ['castSoft=4']),
    ('density hi', ['density=10000']), ('rough hi', ['rough=1']), ('film hi', ['film=10000']), ('film scale lo', ['film=400', 'filmScale=0.01']),
    ('film scale hi', ['film=400', 'filmScale=1000']), ('coat ior hi', ['coat=1', 'coatIor=4']),
    ('sss depth lo', MILKY + ['sssDepth=0.001']), ('sss depth hi', MILKY + ['sssDepth=100']),
    ('sss forward', MILKY + ['sssForward=0.95']), ('sss back', MILKY + ['sssForward=-0.95']),
    ('sss detail hi', MILKY + ['sssDetail=256']), ('sss detail past', MILKY + ['sssDetail=100000']),
    ('sss walk', MILKY + ['sssWalk=1', 'sssDepth=0.001']),
    ('bubbles most', ['bubbles=1', 'bubbleSize=0.004']), ('glow hi', ['glow=10000']),
    ('grin hi', ['grin=2', 'grinDepth=0.0005']), ('grin lo', ['grin=-2', 'grinDepth=0.0005']), ('biref hi', ['biref=1']),
    ('grad bands', ['grad=1', 'gradMode=2', 'gradBands=500', 'gradSoft=0']),
    ('cracks small', ['cracks=1', 'crackSize=0.01']), ('fractures', ['fractures=1', 'cracks=1', 'crackSize=0.01', 'crackCurve=2']),
    ('play small', ['play=4', 'playSize=0.005']), ('silk', ['silk=4', 'silkSoft=0', 'silkWave=10']), ('piezo', ['piezo=100', 'cracks=1']),
    ('backdrop spread lo', ['backdrop=1', 'backSpread=0.1', 'backGlow=1000', 'backGlowSize=0.1']),
    ('floor fade lo', ['floor=1', 'floorFade=0.1', 'floorGain=1000']), ('elev hi', ['floor=1', 'elev=90']), ('elev lo', ['floor=1', 'elev=-90']),
    ('grade hi', ['contrast=10', 'saturation=10', 'wbTemp=4', 'wbTint=-4', 'tone=3']), ('light gain hi', ['lightGain=1000']),
    ('metal last', ['metal=11']), ('metal past', ['metal=99']), ('mat past', ['mat=9999']),
    ('thick hi', ['thick=10']), ('thick lo', ['thick=0.05']), ('len hi', ['len=10']), ('melt hi', ['melt=0.1']), ('beads hi', ['beads=0.95', 'anim=1']),
    ('thomas b past', ['p.b=-10']), ('knot p q past', ['scene=knots', 'preset=torus', 'p.p=1000', 'p.q=1000']),
    ('puff hi', ['scene=solids', 'preset=superellipsoid', 'puff=0.3']), ('puff lo', ['scene=solids', 'preset=superellipsoid', 'puff=-0.3']),
    ('tree twigs past', ['scene=trees', 'p.twigs=1e9', 'p.leaves=9']),
    ('studio tiny scale', ['scene=solids', 'preset=superellipsoid',
                           'sculpt={"layers":[{"kind":"cells","scale":0.000001,"depth":10,"a":5,"b":5,"mode":"both"}],"round":10,"crumbs":10}']),
    ('setup values past', ['ior=0', 'nb=1e9', 'ns=1e9', 'scale=1e9', 'dist=0', 'fov=0', 'smooth=5', 'sssDetail=1e9']),
]


def still(out, name, args):
    path = os.path.join(out, 'still_' + name.replace(' ', '_').replace('.', '') + '.png')
    if os.path.exists(path):
        os.replace(path, path + '.old')
    cmd = [sys.executable, QUEUE, '--name', 'limits', '--', EXE, '--still', path, 'w=480', 'h=270', 'frames=8'] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        code, said = r.returncode, (r.stderr or r.stdout)[-300:]
    except subprocess.TimeoutExpired:
        return (name, 'HUNG', '', args)
    if code != 0 or not os.path.exists(path):
        return (name, f'FAILED ({code})', said.strip(), args)
    from PIL import Image, ImageStat
    m = sum(ImageStat.Stat(Image.open(path).convert('RGB')).mean) / 3
    return (name, 'ok' if m > 1.0 else 'black', f'mean {m:.1f}', args)


def window(out, script, name, frames=5):
    dump = os.path.join(out, f'{name}.json')
    shot = os.path.join(out, f'{name}.png')
    cmd = [sys.executable, QUEUE, '--name', 'limits', '--', EXE, 'w=1280', 'h=720', 'spin=0', '--ui-frames', str(frames), '--ui-dump', dump,
           '--ui-out', out, '--ui-timeout', '120', '--ui-shot', shot, '--ui-do=' + script]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[-400:]
    with open(dump, encoding='utf-8') as f:
        return json.load(f), ''


def windows(out):
    """The keys and typing, on the index (a setting) and on Thomas' b (a shape's own number)."""
    ior = [133.34375, 193.84375, 197.34375, 205.84375]
    b = [662.65625, 375.84375, 752.65625, 387.84375]
    cx, cy = (ior[0] + ior[2]) / 2, (ior[1] + ior[3]) / 2
    value = (ior[2] + 6 + 22, cy)
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    checks = [
        ('alt past the top', f'apdown:{cx},{cy};pmove:{cx + 200},{cy};pmove:{cx + 400},{cy};pup;wait:3', lambda s: s['settings']['ior'] == 4.0),
        ('alt past the bottom', f'apdown:{cx},{cy};pmove:{cx - 200},{cy};pmove:{cx - 400},{cy};pup;wait:3', lambda s: s['settings']['ior'] == 1.0),
        ('plain drag stays in range', f'pdown:{cx},{cy};pmove:{cx + 400},{cy};pup;wait:3', lambda s: s['settings']['ior'] == 2.7),
        ('alt+shift a tenth', f'aspdown:{cx},{cy};pmove:{cx + 20},{cy};pmove:{cx + 40},{cy};pup;wait:3',
         lambda s: 1.5 < s['settings']['ior'] < 1.7),
        ('ctrl+alt+shift a hundredth, between steps', f'caspdown:{cx},{cy};pmove:{cx + 10},{cy};pmove:{cx + 20},{cy};pup;wait:3',
         lambda s: 1.5 < s['settings']['ior'] < 1.53 and round(s['settings']['ior'] * 100) != s['settings']['ior'] * 100),
        ('typed', f'pdown:{value[0]},{value[1]};pup;wait:2;type:3.25;key:enter;wait:3', lambda s: s['settings']['ior'] == 3.25),
        ('typed past the hard limit', f'pdown:{value[0]},{value[1]};pup;wait:2;type:99;key:enter;wait:3', lambda s: s['settings']['ior'] == 4.0),
        ('typed and left with esc', f'pdown:{value[0]},{value[1]};pup;wait:2;type:2.2;key:escape;wait:3', lambda s: s['settings']['ior'] == 1.5),
        ('typed nonsense', f'pdown:{value[0]},{value[1]};pup;wait:2;type:abc;key:enter;wait:3', lambda s: s['settings']['ior'] == 1.5),
        ('shape number alt past', f'apdown:{bx},{by};pmove:{bx + 300},{by};pmove:{bx + 600},{by};pup;wait:5', None),
    ]
    rows = []
    for name, script, ok in checks:
        s, err = window(out, script, 'win_' + name.replace(' ', '_').replace(',', '').replace('+', ''))
        if s is None:
            rows.append((name, 'FAILED', err))
            continue
        if ok is None:
            knob = next((k for k in s.get('knobs', []) if isinstance(k, dict) and k.get('id') == 'b'), None)
            rows.append((name, 'see', json.dumps(knob)[:200] if knob else 'b: ' + str(s.get('preset'))))
            continue
        rows.append((name, 'ok' if ok(s) else 'WRONG', f"ior {s['settings']['ior']}"))
    return rows


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(NATIVE, 'out', 'limits')
    what = argv[2] if len(argv) > 2 else 'all'
    os.makedirs(out, exist_ok=True)
    crashed_before = os.path.getsize(CRASH) if os.path.exists(CRASH) else 0
    if what in ('stills', 'all'):
        with ThreadPoolExecutor(3) as ex:
            for name, how, said, args in ex.map(lambda c: still(out, *c), CASES):
                print(f'{name:28} {how:12} {said}   {" ".join(args)[:90]}', flush=True)
    if what in ('window', 'all'):
        for name, how, said in windows(out):
            print(f'window: {name:40} {how:8} {said}', flush=True)
    crashed = os.path.getsize(CRASH) if os.path.exists(CRASH) else 0
    print('crash.log', 'grew: see ' + CRASH if crashed > crashed_before else 'unchanged')


if __name__ == '__main__':
    main(sys.argv)
