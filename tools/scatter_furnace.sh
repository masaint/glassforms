#!/bin/sh
# Even light from everywhere on a body that absorbs nothing: every pixel of the body should come out as
# bright as the light (1). Prints the mean and the brightest pixel, for four shapes at three densities.
# usage: sh tools/scatter_furnace.sh [more settings]
E=./target/release/glassforms.exe
FUR="rig= ambient=1,1,1 sss=1 bodyR=1 bodyG=1 bodyB=1 disp=0 bloom=0 absorb=0,0,0"
for s in "scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2" "scene=algebraic preset=tanglecube dist=5.4" "scene=attractors preset=thomas dist=4.4" "scene=gems preset=brilliant"; do
  for dep in 0.1 0.02 0.005; do
    $E --still out/scatter/dbg_fur.png --hdr out/scatter/dbg.hdr w=320 h=240 frames=2 $s $FUR sssDepth=$dep fit=0 "$@" 2>&1 | grep -i "panic\|wgsl" | head -5
    python tools/scatter_stat.py out/scatter/dbg.hdr | cut -c41-110
  done
done
