#!/usr/bin/env bash
# AT1 re-extraction at --res 1024 (finer mesh) + full re-colour chain.
# Non-training: extract -> refine -> bake Gaussian colour -> prune floaters ->
# image-sampling colour bake.  Power already capped (150 W + CPU turbo off).
set -euo pipefail

PY=/home/cxh/anaconda3/envs/APEX/bin/python
SRC=/home/cxh/APEX/MOUD/AT1
MODEL=/home/cxh/APEX/apex-mesh/outputs/moud/AT1
SCRIPTS=/home/cxh/APEX/apex-mesh/src/scripts
cd /home/cxh/APEX/apex-mesh/src

run() { echo; echo "### $(date '+%F %T')  $*"; "$@"; }

# free a little space before the 8x-larger intermediates land
rm -f "$MODEL/mesh/reference_mesh.obj" "$MODEL/mesh/refined_mesh.obj"

run "$PY" extract_implicit.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 65000 --res 1024
rm -f "$MODEL/mesh/reference_mesh.obj"        # ASCII intermediate, not needed

run "$PY" refine_mesh.py --source_path "$SRC" --model_path "$MODEL" --orient_thresh -1.1
rm -f "$MODEL/mesh/refined_mesh.obj" "$MODEL/mesh/reference_mesh.ply"

run "$PY" bake_vertex_color.py --source_path "$SRC" --model_path "$MODEL" --load_iteration 65000
rm -f "$MODEL/mesh/refined_mesh.ply"          # keep refined_mesh.npz only

run "$PY" "$SCRIPTS/prune_floaters.py" AT1
run "$PY" "$SCRIPTS/bake_image_color.py" AT1

echo; echo "### $(date '+%F %T')  AT1 1024 CHAIN DONE"
du -sh "$MODEL/mesh"
