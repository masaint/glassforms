#!/bin/sh
# Render one material on a few shapes, lit from the front and from behind, onto one sheet, for tuning.
# usage: sh tools/scatter_try.sh name "settings..." [size] [shapes: ring cube thomas gem shell ball]
E=./target/release/glassforms.exe
NAME=$1
SET=$2
SIZE=${3:-400}
SHAPES=${4:-"ring cube thomas gem"}
FRONT="rig=215,40,30,30,6,6,6:120,10,25,35,1.2,1.2,1.2 ambient=0.05,0.05,0.05"
BACK="rig=0,8,45,45,5,5,5 ambient=0.04,0.04,0.04"
mkdir -p out/scatter/try
LIST=""
for s in $SHAPES; do
  case $s in
    ring) SH="scene=solids preset=ring dist=5.0" ;;
    cube) SH="scene=algebraic preset=tanglecube dist=5.4" ;;
    thomas) SH="scene=attractors preset=thomas dist=4.4" ;;
    gem) SH="scene=gems preset=brilliant" ;;
    shell) SH="scene=forms preset=seashell dist=5.0" ;;
    ball) SH="scene=solids preset=superellipsoid p.p=2 p.q=2 dist=5.2" ;;
    pebble) SH="scene=forms preset=superpebble dist=5.0" ;;
    balls) SH="scene=forms preset=metaballs dist=5.0" ;;
  esac
  $E --still out/scatter/try/${NAME}_${s}_front.png w=$SIZE h=$SIZE frames=24 $SH $SET $FRONT > /dev/null
  $E --still out/scatter/try/${NAME}_${s}_back.png w=$SIZE h=$SIZE frames=24 $SH $SET $BACK > /dev/null
  LIST="$LIST out/scatter/try/${NAME}_${s}_front.png out/scatter/try/${NAME}_${s}_back.png"
done
python tools/scatter_tile.py out/scatter/try/${NAME}.png 4 height=$SIZE $LIST
