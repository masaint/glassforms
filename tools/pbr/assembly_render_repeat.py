"""Baseline-A/candidate/baseline-B fixtures for common assembly dielectric refraction.

Queue the outer invocation through tools/gpu_run.py. The shared harness retains raw PNG/HDR/session
captures and produces one comparison sheet. These fixed-frame images are not timing qualification.
"""
from dielectric_render_repeat import SCENES, main


SCENES.update({
    "walk-assembly": (
        "scene=knots preset=trefoil material=wax sssWalk=1 "
        "layer=algebraic:heart L2.size=0.7 J12.op=apart B2.material=diamond "
        "rx=25 ry=-30 rz=8 scale=1 spin=0 grain=0"
    ).split(),
    "sampled-assembly": (
        "scene=knots preset=trefoil material=crown rough=0.08 roughRef=1 "
        "layer=algebraic:heart L2.size=0.7 J12.op=apart B2.material=diamond "
        "rx=25 ry=-30 rz=8 scale=1 spin=0 grain=0"
    ).split(),
})


if __name__ == "__main__":
    main()
