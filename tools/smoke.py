"""Does it run? One small still for each kind of drawing, through the GPU queue; prints ok or what went wrong.
Pictures go to out/smoke/. Opens no window."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
# (GLASSFORMS_EXE: another build of the program, e.g. one made in a target folder of its own)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OUT = os.path.join(NATIVE, 'out', 'smoke')
os.makedirs(OUT, exist_ok=True)
SMALL = ['w=640', 'h=360', 'frames=8']
CASES = [
    ('default', []),
    ('clay', ['clay=1']),
    ('gem', ['scene=gems', 'preset=brilliant']),
    ('studio-shape', ['scene=studio']),
    ('model-teapot', ['scene=models', 'preset=teapot']),
    ('material-gold', ['material=gold']),
    ('material-soap', ['scene=forms', 'preset=superpebble', 'material=soap']),
    ('material-frosted', ['scene=surfaces', 'preset=gyroid', 'material=frosted']),
    ('material-satingold', ['material=satingold']),
    ('material-jade', ['scene=surfaces', 'preset=gyroid', 'material=jade']),
    ('two-bodies', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=gold']),
    ('cast', ['scene=knots', 'preset=trefoil', 'clay=1', 'cast=2']),
    ('camera-fisheye', ['proj=fisheye']),
    ('camera-panorama', ['proj=panorama']),
    ('camera-85mm-f2', ['focal=85', 'fstop=2']),
    ('studio-backdrop-floor', ['studio=lavender', 'floor=1', 'elev=20']),
    ('bodies-film', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.film=450', 'B2.filmVar=0.5']),
    ('bodies-soap', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=soap']),
    ('bodies-frost', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=frosted']),
    ('bodies-jade', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=jade']),
    ('bodies-jade-cast', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=jade', 'cast=2']),
    ('jade-cast-switch', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'material=jade', 'cast=1', 'castShadow2=0']),
    ('frost-cast', ['scene=surfaces', 'preset=gyroid', 'material=jade', 'rough=0.5', 'cast=2']),
    ('bodies-frost-cast', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'material=frosted', 'B2.material=jade', 'cast=2']),
    ('floor-cast-glass', ['studio=lavender', 'floor=1', 'elev=20', 'cast=2']),
    ('floor-cast-clay', ['studio=lavender', 'floor=1', 'elev=20', 'cast=2', 'clay=1']),
    ('floor-cast-studio', ['scene=studio', 'studio=lavender', 'floor=1', 'elev=20', 'cast=2']),
    ('everything', ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart', 'B2.material=gold',
                    'material=frosted', 'cast=2', 'studio=lavender', 'floor=1', 'elev=20', 'focal=85', 'fstop=4']),
    ('inner-crackedquartz', ['scene=solids', 'material=crackedquartz']),
    ('inner-crackle-grow', ['scene=solids', 'material=crackle', 'crackGrow=0.5']),
    ('inner-whiteopal', ['scene=solids', 'preset=superellipsoid', 'material=whiteopal']),
    ('inner-blackopal', ['scene=solids', 'preset=superellipsoid', 'material=blackopal']),
    ('inner-aventurine', ['scene=solids', 'preset=superellipsoid', 'material=aventurine']),
    ('inner-tigerseye', ['scene=solids', 'preset=superellipsoid', 'material=tigerseye']),
    ('inner-agate', ['scene=solids', 'preset=superellipsoid', 'material=agate']),
    ('inner-gradient-tint', ['scene=studio', 'material=twotone']),
    ('inner-piezo', ['scene=solids', 'material=piezoquartz']),
    ('denoise-truemilk', ['scene=surfaces', 'preset=gyroid', 'material=truemilk']),
    ('denoise-truejade', ['material=truejade']),
    ('denoise-roughref', ['scene=surfaces', 'preset=gyroid', 'material=frosted', 'roughRef=1']),
    ('denoise-off', ['scene=surfaces', 'preset=gyroid', 'material=truemilk', 'denoise=0']),
    ('denoise-studio-cast', ['material=truejade', 'studio=lavender', 'floor=1', 'elev=20', 'cast=2']),
    ('denoise-fisheye-turning', ['material=truemilk', 'proj=fisheye', 'turn=12']),
    ('denoise-focus', ['material=truemilk', 'focal=85', 'fstop=2']),
    ('denoise-picture', ['material=truejade', 'hdri=hdri/suburban_garden_4k.exr', 'hdriShow=1']),
]
bad = 0
# (names given on the command line: only those cases)
ONLY = sys.argv[1:]
for name, args in CASES:
    if ONLY and name not in ONLY:
        continue
    png = os.path.join(OUT, name + '.png')
    cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'smoke', '--', EXE, '--still', png] + SMALL + args
    try:
        r = subprocess.run(cmd, cwd=NATIVE, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
        ok = r.returncode == 0 and os.path.exists(png)
        msg = ''
        if ok:
            try:
                rep = json.loads(r.stdout.strip().splitlines()[-1])
                msg = '%.2f ms a frame' % rep.get('msPerFrame', 0)
            except (ValueError, IndexError):
                msg = '(no report)'
        else:
            msg = 'exit %d: %s' % (r.returncode, (r.stderr or r.stdout or '').strip().replace('\n', ' | ')[-400:])
    except subprocess.TimeoutExpired:
        ok, msg = False, 'timed out after 180 s'
    bad += not ok
    print('%-4s %-22s %s' % ('ok' if ok else 'FAIL', name, msg), flush=True)
print('%d cases, %d failed' % (len(CASES), bad))
