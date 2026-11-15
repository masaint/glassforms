"""A setup saved by either version of Glass Forms opens in the other, and draws the same picture there.

usage: python tools/setup_cross.py [--keep] [--limit=0.3]        (from D:\\glassforms\\native)

  native -> web   The native program draws a curve (an attractor, partway through its animation, with settings
                  off their defaults), and saves the still and the setup. The web page, opened on another
                  family, loads that file and draws the same number of frames.
  web -> native   The web page, showing an equation (an algebraic surface), is changed by script (settings, the
                  shape's sliders, its orientation, a moment of its animation), draws a still and saves its
                  setup. The native program draws that setup.

In each direction two checks: the setup the receiving version then saves says what the file said, field for
field; and the two stills agree as closely as stills of the two versions do (compare.py: a mean difference under
--limit grey levels of 255).

Prints one line per check. Exit status 1 if any failed. The pictures and setups are kept in native/out/cross
when something failed, or with --keep."""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
WEB = os.path.dirname(NATIVE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
if not os.path.exists(EXE):
    EXE = os.path.join(NATIVE, 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'cross')
SIZE, FRAMES = ('640', '360'), 48
limit = '0.3'
for a in sys.argv[1:]:
    if a.startswith('--limit='):
        limit = a.split('=', 1)[1]
os.makedirs(OUT, exist_ok=True)
out = lambda name: os.path.join(OUT, name)
failed = []

def say(name, ok, detail=''):
    if not ok:
        failed.append(name)
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)

def native(*args):
    r = subprocess.run([EXE] + list(args), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('the native program failed: ' + (r.stderr or r.stdout).strip()[:300])

def web(query, given, png):
    """Open the page for a still, hand tools/setup_still.js `given`, save its picture; returns the setup it reports."""
    with open(out('input.json'), 'w', encoding='utf-8') as f:
        json.dump(given, f)
    if os.path.exists(png):
        os.remove(png)
    r = subprocess.run(['node', os.path.join(WEB, 'tools', 'drive.mjs'), query, '--script', os.path.join(WEB, 'tools', 'setup_still.js'),
                        '--input', out('input.json'), '--save', png, '--timeout', '180000'], capture_output=True, text=True, encoding='utf-8', cwd=WEB)
    setups = [l for l in r.stdout.splitlines() if l.startswith('{"version"')]
    if r.returncode != 0 or not setups or not os.path.exists(png):
        raise RuntimeError('the web page failed: ' + (r.stdout + r.stderr).strip()[-300:])
    return json.loads(setups[-1])

def differences(a, b, path=''):
    """Where b does not say what a says: every field of a must be in b with the same value."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in a for d in (differences(a[k], b[k], path + '.' + k if path else k) if k in b else ['%s is missing' % (path + '.' + k if path else k)])]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i in range(len(a)) for d in differences(a[i], b[i], '%s[%d]' % (path, i))]
    return [] if a == b and isinstance(a, bool) == isinstance(b, bool) else ['%s: %r against %r' % (path, a, b)]

def count(x):
    return sum(count(v) for v in (x.values() if isinstance(x, dict) else x)) if isinstance(x, (dict, list)) else 1

def same_setup(name, sent, back):
    d = differences(sent, back)
    say(name, not d, '%d values, all the same' % count(sent) if not d else '%d of %d values differ: %s' % (len(d), count(sent), '; '.join(d[:4])))

def same_picture(name, a, b):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'compare.py'), a, b, '--limit=' + limit], capture_output=True, text=True)
    say(name, r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else '')

def guarded(name, f):
    try:
        f()
    except Exception as e:
        say(name, False, str(e)[:400])

size = ['w=' + SIZE[0], 'h=' + SIZE[1], 'frames=%d' % FRAMES]
still = 'w=%s&h=%s&frames=1' % SIZE

# ---------------------------------------------------------------- native -> web: a curve
def native_to_web():
    shape = 'attractors:halvorsen'
    native('--still', out('curve-native.png'), '--save-setup', out('curve-native.json'), 'scene=attractors', 'preset=halvorsen', *size,
           'ior=1.62', 'disp=0.12', 'rx=20', 'ry=-35', 'thick=1.3', 'dist=6', 'lights=1', 'lyaw=30', 'bloom=0.6', 'anim=3.5')
    with open(out('curve-native.json'), encoding='utf-8') as f:
        sent = json.load(f)
    # the page starts on another family, so that loading has to change family as well
    back = web('scene=knots&' + still, {'setup': sent, 'frames': FRAMES}, out('curve-web.png'))
    same_setup('native -> web: the page has what the file says (%s)' % shape, sent, back)
    same_picture('native -> web: the same picture (%s, 3.5 s into its animation)' % shape, out('curve-native.png'), out('curve-web.png'))
guarded('native -> web', native_to_web)

# ---------------------------------------------------------------- web -> native: an equation
def web_to_native():
    shape = 'algebraic:chmutov4'
    change = {'frames': FRAMES, 'clock': 2.2, 'turn': [25, 40, 10], 'shape': {'@puff': 0.008},
              'set': {'ior': 1.6, 'disp': 0.12, 'dist': 6, 'fov': 30, 'roll': 15, 'lights': 1, 'lyaw': 40, 'lpitch': 20, 'bloom': 0.6}}
    sent = web('scene=algebraic&preset=chmutov4&' + still, change, out('equation-web.png'))
    with open(out('equation-web.json'), 'w', encoding='utf-8') as f:
        json.dump(sent, f, indent=2)
    moved = sent['settings']['ior'] == 1.6 and sent['clock'] == 2.2 and sent['shape']['puff'] == 0.008 and sent['values']['cheb'] != 1.2
    if not moved:
        raise RuntimeError('the page did not take the changes: %s' % json.dumps(sent)[:300])
    native('--still', out('equation-native.png'), '--setup', out('equation-web.json'), '--save-setup', out('equation-native.json'), *size)
    with open(out('equation-native.json'), encoding='utf-8') as f:
        back = json.load(f)
    same_setup('web -> native: the program has what the file says (%s)' % shape, sent, back)
    same_picture('web -> native: the same picture (%s, 2.2 s into its animation)' % shape, out('equation-web.png'), out('equation-native.png'))
guarded('web -> native', web_to_native)

if not failed and '--keep' not in sys.argv:
    shutil.rmtree(OUT, ignore_errors=True)
print('%d failed' % len(failed) if failed else 'setups cross both ways')
sys.exit(1 if failed else 0)
