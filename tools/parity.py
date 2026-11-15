"""Does the native renderer draw what the web version draws?

Renders the same still with both (the web page in headless Chrome through tools/drive.mjs, the native
program with --still), with the same settings, and compares the two pictures.

usage: parity.py [family:preset ...] [--all] [--one-each] [--size=640x360] [--frames=48] [--limit=1.5]
                 [--set "name=value name=value"] [--keep]

  --all        every preset of every family      --one-each   the first preset of each family (the default)
  --set        extra settings given to both (the names are the same on both sides)
  --limit      mean difference allowed, in grey levels (0-255)
  --keep       keep the pictures of shapes that pass too (failures are always kept), in native/out/parity

Exit status 1 if any shape differs by more than the limit."""
import json, os, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
WEB = os.path.dirname(NATIVE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'parity')

opts = {'size': '640x360', 'frames': '48', 'limit': '1.5', 'set': ''}
flags, shapes = set(), []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a.startswith('--') and '=' in a:
        k, v = a[2:].split('=', 1); opts[k] = v
    elif a == '--set':
        i += 1; opts['set'] = argv[i]
    elif a.startswith('--'):
        flags.add(a[2:])
    else:
        shapes.append(a)
    i += 1
w, h = opts['size'].split('x')
extra = [s for s in opts['set'].split() if s]
os.makedirs(OUT, exist_ok=True)

listing = json.loads(subprocess.run([EXE, '--list'], capture_output=True, text=True, check=True).stdout)
# only the families the web version has can be compared with it
WEB_FAMILIES = ('attractors', 'knots', 'surfaces', 'algebraic')
listing = [f for f in listing if f['id'] in WEB_FAMILIES]
if 'all' in flags:
    shapes = ['%s:%s' % (f['id'], p) for f in listing for p in f['presets']]
elif not shapes:
    shapes = ['%s:%s' % (f['id'], f['presets'][0]) for f in listing]

worst, failed = 0.0, []
for s in shapes:
    fam, preset = s.split(':')
    tag = '%s-%s' % (fam, preset)
    web_png, nat_png = os.path.join(OUT, tag + '-web.png'), os.path.join(OUT, tag + '-native.png')
    q = '&'.join(['scene=' + fam, 'preset=' + preset, 'w=' + w, 'h=' + h, 'frames=' + opts['frames']] + extra)
    r = subprocess.run(['node', os.path.join(WEB, 'tools', 'drive.mjs'), q, web_png, '--timeout', '180000'], capture_output=True, text=True, cwd=WEB)
    if r.returncode != 0 or not os.path.exists(web_png):
        print('%-28s WEB FAILED: %s' % (s, (r.stdout + r.stderr).strip()[:300])); failed.append(s); continue
    r = subprocess.run([EXE, '--still', nat_png, 'scene=' + fam, 'preset=' + preset, 'w=' + w, 'h=' + h, 'frames=' + opts['frames']] + extra, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(nat_png):
        print('%-28s NATIVE FAILED: %s' % (s, (r.stdout + r.stderr).strip()[:300])); failed.append(s); continue
    a = np.asarray(Image.open(web_png).convert('RGB')).astype(np.float32)
    b = np.asarray(Image.open(nat_png).convert('RGB')).astype(np.float32)
    if a.shape != b.shape:
        print('%-28s SIZES DIFFER %s %s' % (s, a.shape, b.shape)); failed.append(s); continue
    d = np.abs(a - b)
    px = d.max(axis=2)
    mean = float(d.mean())
    worst = max(worst, mean)
    ok = mean <= float(opts['limit'])
    print('%-28s %s mean %.3f  largest %3.0f  off by >8: %.2f%%  >32: %.3f%%  brightness %.2f / %.2f'
          % (s, 'ok  ' if ok else 'FAIL', mean, d.max(), 100 * (px > 8).mean(), 100 * (px > 32).mean(), a.mean(), b.mean()))
    if not ok:
        failed.append(s)
        Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(os.path.join(OUT, tag + '-diff.png'))
    elif 'keep' not in flags:
        os.remove(web_png); os.remove(nat_png)
print('%d shapes, worst mean difference %.3f grey levels, %d failed%s' % (len(shapes), worst, len(failed), (': ' + ', '.join(failed)) if failed else ''))
sys.exit(1 if failed else 0)
