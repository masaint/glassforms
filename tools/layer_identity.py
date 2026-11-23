"""Pixel identity of scenes of one to four layers between two builds: unify_repeat.py (baseline A,
candidate, baseline B, fixed-seed stills, byte comparison) on layer scenes instead of the perf gate's.

    python tools/gpu_run.py --name layer-identity -- python -I tools/layer_identity.py \
        --unify D:/glassforms-unify/native/tools/pbr --baseline A.exe --candidate B.exe [--scenes a,b]

Every other option is unify_repeat.py's.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = {
    'one_thomas': ['scene=attractors', 'preset=thomas'],
    'two_apart_bodies': ('scene=links preset=hopflink material=acrylic layer=studio:sphere L2.x=0.7 '
                         'L2.size=0.33 J12.op=apart B2.material=chrome').split(),
    'three_joined': ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6',
                     'layer=knots:trefoil', 'L3.size=0.8', 'L3.rx=40'],
    'four_joined': ['scene=attractors', 'preset=thomas', 'layer=surfaces:schwarzp', 'L2.size=0.6',
                    'layer=knots:trefoil', 'L3.size=0.8', 'L3.rx=40', 'layer=algebraic:tanglecube',
                    'L4.size=0.5', 'L4.x=0.5'],
    'four_assembly': (
        'material=satin rough=0.15 coat=1 coatIor=1.5 absorb=5,2.2,0.35 ior=1.7 '
        'scene=solids preset=superellipsoid p=2 q=2 tall=1.0 deep=1.0 '
        'layer=solids:superellipsoid L2.p=2 L2.q=2 L2.tall=1.0 L2.deep=1.0 L2.size=0.9 J12.op=2cuts1 '
        'J12.radius=0 L2.show=0 layer=knots:trefoil L3.size=0.62 L3.thick=1.0 '
        'J13.op=apart J23.op=apart B3.material=iron layer=solids:ring L4.size=0.5 L4.y=-0.25 '
        'L4.rx=70 J14.op=apart J24.op=apart J34.op=apart B4.material=lacquer studio=metal elev=15').split(),
}


def main():
    argv = sys.argv[1:]
    unify = Path(argv[argv.index('--unify') + 1])
    del argv[argv.index('--unify'):argv.index('--unify') + 2]
    sys.path.insert(0, str(unify))
    import unify_repeat as u
    u.NATIVE = HERE.parent
    u.gate_scenes = lambda: {k: list(v) for k, v in SCENES.items()}
    return u.main(argv)


if __name__ == '__main__':
    sys.exit(main())
