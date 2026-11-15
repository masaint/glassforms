"""Run the picture checks one after another and write each one's summary line to out/after_integration.txt.
Opens no windows (tools/ui_check.py is left out: run it on its own)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
OUT = os.path.join(NATIVE, 'out', 'after_integration.txt')
RUNS = [
    ['parity.py'],
    ['light_check.py'],
    ['dof_check.py', '--old', 'out/dof/before-focus.exe'],
    ['layers_check.py'],
    ['gens_check.py'],
    ['material_check.py', '--quick', '--old', 'out/materials/before-materials.exe'],
    ['models_check.py'],
    ['cast_check.py'],
    ['camera_check.py'],
    ['film_check.py', '--quick'],
    ['stage_check.py'],
]
with open(OUT, 'w', encoding='utf-8') as log:
    for run in RUNS:
        if not os.path.exists(os.path.join(HERE, run[0])):
            log.write('%-20s not present\n' % run[0])
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'lead', '--', sys.executable, os.path.join(HERE, run[0])] + run[1:],
                           cwd=NATIVE, capture_output=True, text=True, encoding='utf-8', errors='replace')
        lines = (r.stdout or '').splitlines()
        fails = [l for l in lines if l.startswith('FAIL')]
        last = lines[-1] if lines else (r.stderr or '').strip().splitlines()[-1:] or ['(no output)']
        log.write('%-20s exit %d  %4.0f s  %s\n' % (run[0], r.returncode, time.time() - t0, last if isinstance(last, str) else last[0]))
        for f in fails:
            log.write('    %s\n' % f[:300])
        if r.returncode not in (0, 1):
            log.write('    stderr: %s\n' % (r.stderr or '')[-600:].replace('\n', ' | '))
        log.flush()
    log.write('done\n')
