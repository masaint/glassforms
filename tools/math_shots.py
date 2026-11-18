"""Stills and contact sheets for the mathematical families (fractals, topology, orbitals, polyhedra).

usage: python tools/math_shots.py <family> [--preset=id,id] [--modes=glass,clay,csg,csgclay] [--set "k=v k=v"]
                                  [--tag=name] [--w=1280] [--h=720] [--jobs=3] [--exe=path] [--list]

Every still goes through main's GPU queue (D:/glassforms/native/tools/gpu_run.py). Stills land in
<repo>/../shots/<family>/<tag>/<preset>_<mode>.png, a contact sheet of them all in
<repo>/../shots/<family>/sheet_<tag>.png, and one line per still is printed: the preset, the mode, the
volume's build time, how much of the picture is glass, and a warning when the picture is empty, blown or
holds NaN-black blocks.

  glass    the family's own look
  clay     matte clay (shows the surface itself: holes, speckle, false sheets)
  csg      traced from the formula (trace=csg)
  csgclay  the formula, in clay
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
ROOT = os.path.dirname(NATIVE)
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
MODES = {'glass': [], 'clay': ['clay=1'], 'csg': ['trace=csg'], 'csgclay': ['trace=csg', 'clay=1']}


def opt(argv, name, default=None):
    for a in argv:
        if a.startswith(f'--{name}='):
            return a.split('=', 1)[1]
    if f'--{name}' in argv:
        i = argv.index(f'--{name}')
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def presets(exe, family):
    out = subprocess.run([sys.executable, QUEUE, '--name', 'gensA', '--', exe, '--list'], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith('['))
    return next(f['presets'] for f in json.loads(line) if f['id'] == family)


def still(exe, family, preset, mode, extra, path, w, h):
    cmd = [sys.executable, QUEUE, '--name', 'gensA', '--', exe, '--still', path, f'w={w}', f'h={h}',
           f'scene={family}', f'preset={preset}'] + MODES[mode] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    info = {}
    for line in r.stdout.splitlines():
        if line.startswith('{'):
            try:
                info = json.loads(line)
            except ValueError:
                pass
    if r.returncode != 0 or not os.path.exists(path):
        return {'error': (r.stderr or r.stdout)[-600:]}
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    lum = a.mean(axis=2)
    glass = lum > 6
    info['cover'] = float(glass.mean())
    info['white'] = float((a.min(axis=2) >= 250).mean())
    return info


def main(argv):
    if not argv or argv[0].startswith('--'):
        print(__doc__)
        return 1
    family = argv[0]
    exe = opt(argv, 'exe', os.path.join(NATIVE, 'target', 'release', 'glassforms.exe'))
    if '--list' in argv:
        print(' '.join(presets(exe, family)))
        return 0
    only = opt(argv, 'preset')
    ids = only.split(',') if only else presets(exe, family)
    modes = opt(argv, 'modes', 'glass,clay').split(',')
    extra = (opt(argv, 'set', '') or '').split()
    tag = opt(argv, 'tag', 'look')
    w, h = int(opt(argv, 'w', '1280')), int(opt(argv, 'h', '720'))
    jobs = int(opt(argv, 'jobs', '3'))
    folder = os.path.join(ROOT, 'shots', family, tag)
    os.makedirs(folder, exist_ok=True)
    work = [(p, m) for p in ids for m in modes]

    def one(pm):
        p, m = pm
        path = os.path.join(folder, f'{p}_{m}.png')
        return p, m, path, still(exe, family, p, m, extra, path, w, h)

    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(one, work))
    bad = 0
    for p, m, path, info in results:
        if 'error' in info:
            print(f'{p:16s} {m:8s} FAILED {info["error"]}')
            bad += 1
            continue
        warn = []
        if info['cover'] < 0.02: warn.append('almost empty')
        if info['cover'] > 0.92: warn.append('fills the frame')
        if info['white'] > 0.2: warn.append('blown')
        print(f'{p:16s} {m:8s} build {info.get("volumeMs", 0):7.1f} ms  setup {info.get("setupMs", 0):6.0f} ms  glass {info["cover"] * 100:5.1f}%  {" ".join(warn)}')
    # the contact sheet: one row per preset, one column per mode
    cw, ch = 480, 270
    sheet = Image.new('RGB', (cw * len(modes), (ch + 18) * len(ids)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for p, m, path, info in results:
        if 'error' in info:
            continue
        r, c = ids.index(p), modes.index(m)
        im = Image.open(path).convert('RGB').resize((cw, ch), Image.LANCZOS)
        sheet.paste(im, (c * cw, r * (ch + 18) + 18))
        d.text((c * cw + 6, r * (ch + 18) + 3), f'{family}/{p}  {m}  {extra and " ".join(extra) or ""}'[:80], fill=(220, 220, 220))
    out = os.path.join(ROOT, 'shots', family, f'sheet_{tag}.png')
    sheet.save(out)
    print('sheet:', out)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
