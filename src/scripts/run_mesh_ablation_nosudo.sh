#!/usr/bin/env bash
# Mesh-reprojection ablation driver (no sudo guard: power already capped at 150 W + turbo off).
set -uo pipefail

PY=/home/cxh/anaconda3/envs/APEX/bin/python
SRC=/home/cxh/APEX/apex-mesh/datasets/seathru_undist/Panama
MODEL=/home/cxh/APEX/apex-mesh/outputs/baseline/Panama
cd /home/cxh/APEX/apex-mesh/src

train() {
  local tag=$1; shift
  local save_iter=$1; shift
  "$PY" train_mesh.py --source_path "$SRC" --model_path "$MODEL" --sh_degree 3 \
    --iterations 3000 --save_iteration "$save_iter" --tag "$tag" "$@"
}

eval_geom() {
  "$PY" evaluate_geometry.py --source_path "$SRC" --model_path "$MODEL" --sh_degree 3 \
    --stability --tag "$1" --save_iteration "$2"
}

train full 70000        && eval_geom full 70000
train no_mesh 71000 --lambda_mesh 0.0   && eval_geom no_mesh 71000
train no_norm 72000 --lambda_norm 0.0   && eval_geom no_norm 72000
train no_ms 73000 --lambda_ms 0.0       && eval_geom no_ms 73000
train no_reg 74000 --lambda_reg 0.0     && eval_geom no_reg 74000
train mesh_verts 75000 --train_mesh_verts && eval_geom mesh_verts 75000
train phase1_only 76000 --iterations 1500 --phase1_iters 1500 --phase2_iters 1500 \
  && eval_geom phase1_only 76000

echo "=== ABLATION DONE ==="
