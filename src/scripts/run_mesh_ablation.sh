#!/usr/bin/env bash
# Mesh-reprojection ablation: run the joint mesh-reprojection refinement
# for each config and report held-out geometry + forward-view stability.
#
# Run from the src/ directory. GPU work: the A6000 must be power-limited first
# (150 W + CPU turbo off) or the PSU trips and the box power-cycles — see
# [[a6000-power-limit]]. Sequential on purpose (single card at 150 W).
#
#   bash scripts/run_mesh_ablation.sh

set -euo pipefail

PY=/home/cxh/anaconda3/envs/APEX/bin/python
SRC=/home/cxh/APEX/apex-mesh/datasets/seathru_undist/Panama
MODEL=/home/cxh/APEX/apex-mesh/outputs/baseline/Panama
ITERS=3000

# --- power limit guard (MUST succeed before any GPU work) -------------------
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 150
sudo bash -c 'echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo'
echo "power limit: $(nvidia-smi --query-gpu=power.limit,power.draw --format=csv,noheader)"

cd "$(dirname "$0")/.."   # -> src/

train() {  # tag save_iteration [extra train_mesh.py args...]
  local tag=$1; shift
  local save_iter=$1; shift
  "$PY" train_mesh.py \
    --source_path "$SRC" --model_path "$MODEL" --sh_degree 3 \
    --iterations "$ITERS" --save_iteration "$save_iter" --tag "$tag" \
    "$@"
}

eval_geom() {  # tag save_iteration
  "$PY" evaluate_geometry.py \
    --source_path "$SRC" --model_path "$MODEL" --sh_degree 3 \
    --stability --tag "$1" --save_iteration "$2"
}

# --- ablation variants (each: distinct --tag and --save_iteration) ----------
train full 70000
eval_geom full 70000

train no_mesh 71000 --lambda_mesh 0.0
eval_geom no_mesh 71000

train no_norm 72000 --lambda_norm 0.0
eval_geom no_norm 72000

train no_ms 73000 --lambda_ms 0.0
eval_geom no_ms 73000

train no_reg 74000 --lambda_reg 0.0
eval_geom no_reg 74000

train mesh_verts 75000 --train_mesh_verts
eval_geom mesh_verts 75000

# phase1-only: run only the geometry->mesh stage (never release the medium).
train phase1_only 76000 --iterations 1500 --phase1_iters 1500 --phase2_iters 1500
eval_geom phase1_only 76000

echo
echo "Ablation done. Per-variant reports: $MODEL/mesh/s4_<tag>_geometry_eval.json"
echo "  heldout_reproj_mean_rel_err / forward_view_stability.mean / normal_consistency.mean_cos"
