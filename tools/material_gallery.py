"""Every named material on one shape, as contact sheets: for looking at them all.

usage: python tools/material_gallery.py <exe> <out folder> <scene> <preset> [w=480] [h=270] [frames=32] [extra name=value ...]

Writes <out folder>/<scene>_<preset>_<group>.png, one sheet per group of materials.
"""
import json
import os
import subprocess
import sys

exe, out, scene, preset = sys.argv[1:5]
exe = os.path.abspath(exe)
opts = dict(a.split('=', 1) for a in sys.argv[5:])
w, h, frames = opts.pop('w', '480'), opts.pop('h', '270'), opts.pop('frames', '32')
opts_reuse = opts.pop('reuse', '0') == '1'      # reuse=1: keep pictures that are already there
extra = ['%s=%s' % kv for kv in opts.items()]
os.makedirs(os.path.join(out, 'single'), exist_ok=True)
mats = json.loads(subprocess.run([exe, '--list-materials'], capture_output=True, text=True, check=True).stdout)
groups = {}
for m in mats:
    path = os.path.join(out, 'single', '%s_%s_%s.png' % (scene, preset, m['id']))
    if opts_reuse and os.path.exists(path):
        groups.setdefault(m['group'], []).append('%s=%s' % (m['id'], path))
        continue
    r = subprocess.run([exe, '--still', path, 'scene=' + scene, 'preset=' + preset, 'w=' + w, 'h=' + h, 'frames=' + frames, 'material=' + m['id']] + extra, capture_output=True, text=True)
    if r.returncode != 0:
        print('FAILED', m['id'], r.stderr.strip()[:300])
        continue
    groups.setdefault(m['group'], []).append('%s=%s' % (m['id'], path))
here = os.path.dirname(os.path.abspath(__file__))
for g, items in groups.items():
    cols = 4 if len(items) > 4 else len(items)
    subprocess.run([sys.executable, os.path.join(here, 'material_sheet.py'), os.path.join(out, '%s_%s_%s.png' % (scene, preset, g.replace(' ', '_'))), str(cols)] + items, check=True)
