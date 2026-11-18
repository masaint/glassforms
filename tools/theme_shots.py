"""Screenshots of the frosted panels in the dark and the light look, over three backgrounds, at two sizes;
and the same for a program from before (the "before" of the before/after sheet).

    python tools/gpu_run.py --name theme -- python tools/theme_shots.py [before=<exe>] [only=<word>]

Writes out/theme/<size>_<look>_<background>.png (and before_<size>_<background>.png), then the sheet
out/theme/sheet.png. Needs Pillow.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(ROOT, 'out', 'theme')
HDRI = os.path.join(ROOT, 'hdri', 'suburban_garden_4k.exr')
arg = lambda k: next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith(k + '=')), None)
BEFORE = arg('before')
ONLY = arg('only')

SIZES = [('4k', 3840, 2160, 2), ('1440', 2560, 1440, 1.5)]
BACKS = [('studio', []), ('white', ['studio=white']), ('garden', ['hdri=' + HDRI, 'hdriShow=1'])]
# the light room and the materials lab open, so that the floating windows are seen over the panels and the picture
WINDOWS = 'room:open;lab:open'


def shot(exe, name, pairs, size, script):
    path = os.path.join(OUT, name + '.png')
    if ONLY and ONLY not in name:
        return path
    _, w, h, scale = size
    # (a store of its own, new each time: a run writes where its windows were left)
    store = os.path.join(OUT, 'stores', '%s_%d.json' % (name, time.time() * 1000))
    os.makedirs(os.path.dirname(store), exist_ok=True)
    cmd = [exe, 'w=%d' % w, 'h=%d' % h, 'spin=0', *pairs, '--ui-scale', str(scale), '--ui-frames', '60', '--ui-store', store,
           '--ui-timeout', '240', '--ui-shot', path]
    if script:
        cmd += ['--ui-do', script]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
    print(name, 'ok' if r.returncode == 0 and os.path.exists(path) else 'FAILED ' + (r.stderr or r.stdout)[-400:], flush=True)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for size in SIZES:
        for back, pairs in BACKS:
            row = []
            if BEFORE:
                row.append(('before', shot(BEFORE, 'before_%s_%s' % (size[0], back), pairs, size, WINDOWS)))
            for look in ['dark', 'light']:
                row.append((look, shot(EXE, '%s_%s_%s' % (size[0], look, back), pairs, size, 'ui:theme=%s;%s' % (look, WINDOWS))))
            rows.append(('%s %s' % (size[0], back), row))
    # the settings window, the frost turned down and off, and the help, at 2560 x 1440 over the garden
    size, pairs = SIZES[1], BACKS[2][1]
    extra = [('settings, light', 'ui:theme=light;ui:settings=open'), ('subtle, dark', 'ui:frost=subtle;ui:settings=open'),
             ('frost off, dark', 'ui:frost=off;' + WINDOWS), ('help, light', 'ui:theme=light;help:open')]
    rows.append(('extra', [(label, shot(EXE, 'extra_' + label.replace(', ', '_').replace(' ', '-'), pairs, size, script)) for label, script in extra]))
    sheet(rows)


def sheet(rows):
    from PIL import Image, ImageDraw
    tw = 960
    cols = max(len(r) for _, r in rows)
    th = tw * 9 // 16
    img = Image.new('RGB', (cols * (tw + 8) + 8, len(rows) * (th + 30) + 8), (24, 24, 28))
    d = ImageDraw.Draw(img)
    for i, (label, row) in enumerate(rows):
        for j, (look, path) in enumerate(row):
            x, y = 8 + j * (tw + 8), 8 + i * (th + 30)
            d.text((x, y), '%s · %s' % (label, look), fill=(220, 220, 225))
            if os.path.exists(path):
                img.paste(Image.open(path).convert('RGB').resize((tw, th), Image.LANCZOS), (x, y + 18))
    out = os.path.join(OUT, 'sheet.png')
    img.save(out)
    print('sheet:', out)


if __name__ == '__main__':
    main()
