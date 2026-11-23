"""Write the lab's scenes as app argument files: native/tools/scenes_parts/<name>.args and scenes.json.

    python -I native/tools/scenes_from_lab.py

Reads `lab appargs <scene>` (CPU only, the lab's own scene generators and scene files; the lab's source is only read).
<name>.args       the scene as the app's own volume path is given it (the lab's app_args(false): a one-object tree is traced
                  as the app's volume, measured with csgCoarse=1; layered scenes carry their L<n>.trace=csg lines). One
                  key=value per line, in the order the lab gives them; a scene whose state is a saved setup, or whose arguments are too long for a command line (boaty512: in_setup), has <name>.setup.json beside it (passed as --setup).
<name>.csg.args   the same scene for the app's exact tracer (trace=csg) where that differs.
scenes.json       per scene: form, drag (what parts_gate.py drags), group, parts, the setup file and the app's cap if any.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'scenes_parts')
LAB = os.environ.get('LAB_EXE', r'D:\glassforms-lab\target\release\lab.exe')
LAB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(LAB)))
SCENES = ['boaty512', 'obj8', 'wide8', 'owner_volume', 'objtree8', 'lib_chamfer', 'lib_array', 'lib_array_grid', 'lib_bend', 'lib_twist', 'tree1']


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for name in SCENES:
        r = subprocess.run([LAB, 'appargs', name], capture_output=True, text=True, cwd=LAB_DIR)
        line = next((l for l in r.stdout.splitlines() if l.startswith('{')), None)
        if r.returncode != 0 or not line:
            print(f'{name}: FAILED {r.stderr.strip()[-200:]}')
            continue
        d = json.loads(line)
        for suffix, key in (('', 'args_volume'), ('.csg', 'args_exact')):
            if suffix and d['args_exact'] == d['args_volume']:
                continue
            lines = [f'{k}={v}' for k, v in d[key]]
            with open(os.path.join(OUT, f'{name}{suffix}.args'), 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(lines) + '\n')
        if d['setup']:   # the saved setup travels with the scene: parts_gate.py passes --setup <file> before the arguments
            shutil.copy(d['setup'], os.path.join(OUT, f'{name}.setup.json'))
            d['setup'] = f'{name}.setup.json'
        d['in_setup'] = False
        if not d['setup'] and sum(len(k) + len(v) + 1 for k, v in d['args_volume']) > 12000:
            # too long for a command line (Windows allows 32767 characters): the state travels as a saved setup
            subprocess.run([LAB, 'appsetup', name, os.path.join(OUT, f'{name}.setup.json')], check=True, capture_output=True, cwd=LAB_DIR)
            d['setup'], d['in_setup'] = f'{name}.setup.json', True
        meta[name] = {k: d[k] for k in ('about', 'form', 'setup', 'in_setup', 'drag', 'group', 'app_cap', 'n_parts')}
        print(f"{name}: {len(d['args_volume'])} args, form {d['form']}, drag {d['drag']}, setup {d['setup']}, cap {d['app_cap']}")
    with open(os.path.join(OUT, 'scenes.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    sys.exit(main())
