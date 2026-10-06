#!/usr/bin/env bash
# MOUD batch pipeline: reconstruction -> medium -> projective -> flatten ->
# extract -> refine -> mesh reprojection -> eval, per scene.
# Usage: run_moud_batch.sh [AT2 AT3 AT4 AT5]   (defaults to AT2..AT5)
# Power must already be capped (150 W + CPU turbo off).
# NOTE: refine uses --orient_thresh -1.1 (MOUD meshes are self-overlapping; the
# orientation prune would fragment them).
set -euo pipefail

PY=/home/cxh/anaconda3/envs/APEX/bin/python
SRC_ROOT=/home/cxh/APEX/MOUD
MODEL_ROOT=/home/cxh/APEX/apex-mesh/outputs/moud
cd /home/cxh/APEX/apex-mesh/src

SCENES=("${@:-AT2 AT3 AT4 AT5}")

run() { echo; echo "### $(date '+%F %T')  $*"; "$@"; }

for S in "${SCENES[@]}"; do
  SRC="$SRC_ROOT/$S"
  MODEL="$MODEL_ROOT/$S"
  echo; echo "################ $(date '+%F %T')  SCENE $S ################"
  mkdir -p "$MODEL"

  run "$PY" train.py -s "$SRC" -m "$MODEL" --eval --sh_degree 3 --iterations 30000
  run "$PY" train_medium.py --source_path "$SRC" --model_path "$MODEL" \
      --load_iteration 30000 --iterations 5000
  run "$PY" train_projective.py --source_path "$SRC" --model_path "$MODEL" \
      --load_iteration 30000 --iterations 5000 --save_iteration 60000
  run "$PY" train_flatten.py --source_path "$SRC" --model_path "$MODEL" \
      --load_iteration 60000 --save_iteration 65000 --iterations 3000 \
      --lambda_flatten 2.0 --lambda_normal 0.05
  run "$PY" extract_implicit.py --source_path "$SRC" --model_path "$MODEL" \
      --load_iteration 65000 --res 512
  run "$PY" refine_mesh.py --source_path "$SRC" --model_path "$MODEL" --orient_thresh -1.1
  run "$PY" train_mesh.py --source_path "$SRC" --model_path "$MODEL" \
      --load_iteration 65000 --save_iteration 70000 --tag full --iterations 3000
  run "$PY" evaluate_geometry.py --source_path "$SRC" --model_path "$MODEL" --load_iteration 65000
  run "$PY" evaluate_geometry.py --source_path "$SRC" --model_path "$MODEL" \
      --load_iteration 65000 --save_iteration 70000 --stability --tag full

  echo "### $(date '+%F %T')  SCENE $S DONE"
done

echo; echo "### $(date '+%F %T')  MOUD BATCH DONE: ${SCENES[*]}"
