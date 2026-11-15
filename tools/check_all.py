"""Every check of the native Glass Forms, in one go. Run it when nothing else is using the GPU.

usage: python tools/check_all.py [--quick]        (from D:\\glassforms\\native; --quick leaves out the slow parts)

  1. the Rust tests (maths against the web version's numbers, shaders valid, state, setups)
  2. pictures against the web version: every preset, then depth of field, mid-animation, other settings
  3. the two ways of building a shape (compute shader, and slice by slice) give the same volume
  4. video: a rendered loop decodes and loops; all three formats carry the right colour tags
  5. a still made from a saved setup matches the still it was saved from; the 16-bit still is 16-bit; a setup
     saved by the web version draws the same picture natively, and the other way round
  6. with no dispersion, the one-path shortcut draws what three paths draw
  7. the tracer's shaders really are built without the run-time checks (it is much slower with them)
  8. the window, if its own check script is there

Prints one line per check and a count at the end; exit status 1 if anything failed."""
import json, os, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
WEB = os.path.dirname(NATIVE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'check')
QUICK = '--quick' in sys.argv
os.makedirs(OUT, exist_ok=True)
results = []

def say(name, ok, detail=''):
    results.append(ok)
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)

def run(args, env=None, cwd=NATIVE, timeout=3600):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(args, capture_output=True, text=True, cwd=cwd, env=e, timeout=timeout)

def exe(*args, env=None):
    r = run([EXE] + list(args), env=env)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return r.stdout

def last_json(text):
    return json.loads([l for l in text.splitlines() if l.startswith('{')][-1])

def diff(a, b):
    x = np.asarray(Image.open(a).convert('RGB')).astype(np.float32)
    y = np.asarray(Image.open(b).convert('RGB')).astype(np.float32)
    return float(np.abs(x - y).mean()) if x.shape == y.shape else 999.0

def guarded(name, f):
    try:
        f()
    except Exception as e:
        say(name, False, str(e)[:400])

# ---------------------------------------------------------------- 1. the Rust tests
r = run(['cargo', 'build', '--release', '-p', 'glassforms'])
say('the program builds', r.returncode == 0, r.stderr.strip()[-300:] if r.returncode else '')
r = run(['cargo', 'test', '--release'])
passed = sum(int(l.split('ok. ')[1].split(' passed')[0]) for l in r.stdout.splitlines() if l.startswith('test result: ok.'))
say('Rust tests', r.returncode == 0, '%d passed' % passed if r.returncode == 0 else (r.stdout + r.stderr)[-600:])

# ---------------------------------------------------------------- 2. pictures against the web version
def parity(name, args):
    r = run([sys.executable, os.path.join(HERE, 'parity.py')] + args)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]
    say(name, r.returncode == 0, last)
parity('pictures against the web: ' + ('one shape of each family' if QUICK else 'every preset'), [] if QUICK else ['--all'])
parity('pictures against the web: depth of field', ['attractors:thomas', 'surfaces:gyroid', '--frames=96', '--set', 'aperture=0.12 lensN=4 blades=6 bokehRot=20 fnear=-0.4 ffar=-0.1 dof=0'])      # dof=0: the lens traced as the web traces it
parity('pictures against the web: 3.5 s into the animation', ['attractors:thomas', 'knots:lissajous', 'surfaces:gyroidnet', 'algebraic:tanglecube', '--set', 'anim=3.5'])
parity('pictures against the web: eased animation at speed 1.7', ['attractors:halvorsen', 'knots:torus', '--set', 'anim=5.2 speed=1.7 ease=8 easeAt=1'])
parity('pictures against the web: the expo and elastic easings', ['attractors:thomas10', 'knots:lissajous', '--set', 'anim=4.3 ease=6 easeAt=0', ], )
parity('pictures against the web: elastic easing at the finish', ['surfaces:gyroidnet', 'algebraic:chmutov4', '--set', 'anim=6.1 ease=9 easeAt=1'])
parity('pictures against the web: other settings', ['knots:trefoil', 'algebraic:barth', '--set', 'roll=25 lights=1 lyaw=40 lpitch=20 ior=1.7 disp=0.2 nb=5 ns=2 nw=6 rx=30 ry=-50 rz=10 dist=5 fov=40 bloom=0.8 exposure=1.4'])

# ---------------------------------------------------------------- 3. two ways of building a shape
def volumes():
    for s in (['scene=attractors', 'preset=thomas', 'res=384'], ['scene=knots', 'preset=trefoil', 'melt=0', 'res=256'], ['scene=attractors', 'preset=lorenz', 'detail=anim']):
        a = last_json(exe('--vol-hash', 'w=64', 'h=64', *s))
        b = last_json(exe('--vol-hash', 'w=64', 'h=64', *s, env={'GLASSFORMS_NO_COMPUTE': '1'}))
        say('volume computed = volume drawn: ' + ' '.join(s), a['hash'] == b['hash'], 'bit for bit, %d voxels a side' % a['volume'] if a['hash'] == b['hash'] else '%s against %s' % (a['hash'], b['hash']))
    # an equation's long arithmetic rounds differently in the two kinds of shader: the pictures must agree
    s = ['scene=surfaces', 'preset=gyroid', 'w=640', 'h=360', 'frames=32']
    exe('--still', os.path.join(OUT, 'f_computed.png'), *s)
    exe('--still', os.path.join(OUT, 'f_drawn.png'), *s, env={'GLASSFORMS_NO_COMPUTE': '1'})
    d = diff(os.path.join(OUT, 'f_computed.png'), os.path.join(OUT, 'f_drawn.png'))
    say('an equation computed and drawn give the same picture', d < 0.1, 'mean difference %.3f grey levels' % d)
guarded('volumes', volumes)

# ---------------------------------------------------------------- 4. video
def video():
    mp4 = os.path.join(OUT, 'loop.mp4')
    info = last_json(exe('--video', mp4, '--quiet', 'scene=attractors', 'preset=thomas', 'w=640', 'h=360', 'fps=30', 'samples=4', 'loop=4', 'spin=0.3'))
    r = run([sys.executable, os.path.join(WEB, 'tools', 'loop_check.py'), mp4])
    say('a rendered loop decodes and loops', 'decodes cleanly' in r.stdout and 'loops cleanly' in r.stdout and info['frames'] == 120, r.stdout.strip().splitlines()[-2] if r.stdout.strip() else r.stderr[-200:])
    want = {'h264': ('h264', 'yuv420p', 'iec61966-2-1'), 'h265': ('hevc', 'yuv420p10le', 'iec61966-2-1'), 'prores': ('prores', 'yuv422p10le', 'bt709')}
    for codec, (name, pix, trc) in want.items():
        f = os.path.join(OUT, 'c_%s.%s' % (codec, 'mov' if codec == 'prores' else 'mp4'))
        exe('--video', f, '--quiet', 'scene=knots', 'preset=trefoil', 'w=640', 'h=360', 'fps=30', 'samples=2', 'loop=2', 'spin=0.5', 'frames=6', 'codec=' + codec, 'shutter=0.5')
        p = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,pix_fmt,color_range,color_space,color_transfer,color_primaries,nb_frames,has_b_frames', '-of', 'json', f])
        st = json.loads(p.stdout)['streams'][0]
        ok = (st['codec_name'], st['pix_fmt'], st.get('color_transfer')) == (name, pix, trc) and st.get('color_space') == 'bt709' and st.get('color_primaries') == 'bt709' and st.get('color_range') == 'tv' and st['nb_frames'] == '6' and st['has_b_frames'] == 0
        say('video format ' + codec, ok, '%(codec_name)s %(pix_fmt)s, %(nb_frames)s frames' % st + ', matrix %s, transfer %s, range %s' % (st.get('color_space'), st.get('color_transfer'), st.get('color_range')))
guarded('video', video)

# ---------------------------------------------------------------- 5. setups and deep stills
def setups():
    a, b, js = (os.path.join(OUT, n) for n in ('s_a.png', 's_b.png', 'setup.json'))
    exe('--still', a, '--save-setup', js, 'scene=surfaces', 'preset=schwarzp', 'w=640', 'h=360', 'frames=48', 'aperture=0.05', 'ior=1.6', 'rx=40', 'p.level=0.4', 'puff=0.01')
    exe('--still', b, '--setup', js, 'w=640', 'h=360', 'frames=48')
    d = diff(a, b)
    say('a still from a saved setup matches the original', d < 0.01, 'mean difference %.4f grey levels' % d)
    deep = os.path.join(OUT, 's_16.png')
    exe('--still', deep, '--16', '--setup', js, 'w=640', 'h=360', 'frames=48')
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', deep, '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], capture_output=True).stdout
    v = np.frombuffer(raw, dtype='<u2').astype(np.float64)
    levels = len(np.unique(v))
    x = np.asarray(Image.open(a).convert('RGB')).astype(np.float64).ravel()
    d = float(np.abs(x - v / 257.0).mean())
    say('the 16-bit still', levels > 5000 and d < 0.6, '%d distinct levels; %.2f grey levels from the 8-bit one (which carries half a level of dither)' % (levels, d))
guarded('setups', setups)

# ---------------------------------------------------------------- 6. the zero-dispersion shortcut
def shortcut():
    a, b = os.path.join(OUT, 'z_one.png'), os.path.join(OUT, 'z_three.png')
    s = ['scene=attractors', 'preset=thomas', 'w=640', 'h=360', 'frames=64']
    exe('--still', a, *s, 'disp=0')
    exe('--still', b, *s, 'disp=0.0000001')         # too little to see, but the three paths are traced
    d = diff(a, b)
    say('no dispersion: one path draws what three do', d < 0.02, 'mean difference %.4f grey levels' % d)
guarded('shortcut', shortcut)

# ---------------------------------------------------------------- 7. the run-time checks are really out
def checks():
    s = ['--bench', '80', 'w=1920', 'h=1080', 'scene=surfaces', 'preset=gyroid']
    best = 0.0
    for _ in range(3):                               # a busy GPU can spoil one comparison: take the best of three
        out = last_json(exe(*s))['ms']
        back = last_json(exe(*s, env={'GLASSFORMS_CHECKED_SHADERS': '1'}))['ms']
        best = max(best, back / out)
    say("shaders are built without naga's loop counters", best > 1.15, 'a frame takes %.2f times as long with them put back' % best)
guarded('checks', checks)

# ---------------------------------------------------------------- the checks that later parts bring with them
for script, name in (('gens_check.py', 'the newer generators'), ('light_check.py', 'lights, environment pictures and clay'), ('dof_check.py', 'depth of field'), ('material_check.py', 'materials'), ('layers_check.py', 'layers and smooth booleans')):
    path = os.path.join(HERE, script)
    if os.path.exists(path):
        r = run([sys.executable, path])
        tail = [l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
        say('%s (%s)' % (name, script), r.returncode == 0, tail[-1][:300] if tail else '')

# ---------------------------------------------------------------- a setup saved by either version opens in the other
cross = os.path.join(HERE, 'setup_cross.py')
if os.path.exists(cross):
    r = run([sys.executable, cross])
    tail = [l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
    say('setups cross between the web and native versions', r.returncode == 0, ' | '.join(tail[-3:])[:300] if tail else '')

# ---------------------------------------------------------------- 8. the window
ui = [f for f in os.listdir(HERE) if f.startswith('ui_check')]
if ui:
    r = run([sys.executable, os.path.join(HERE, ui[0])] if ui[0].endswith('.py') else [os.path.join(HERE, ui[0])])
    tail = [l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
    say('the window (%s)' % ui[0], r.returncode == 0, tail[-1][:300] if tail else '')
else:
    print('     (no window check script yet)')

bad = results.count(False)
print('\n%d checks, %d failed' % (len(results), bad))
sys.exit(1 if bad else 0)
