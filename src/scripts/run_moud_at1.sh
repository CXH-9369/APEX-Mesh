#!/usr/bin/env bash
# MOUD AT1 full pipeline: reconstruction -> medium -> projective -> flatten ->
# extract -> refine -> mesh reprojection -> evaluation.
# Power already capped (150 W + CPU turbo off) from the prior ablation; no sudo needed.
set -euo pipefail

PY=/home/cxh/anaconda3/envs/APEX/bin/python
SRC=/home/cxh/APEX/MOUD/AT1
MODEL=/home/cxh/APEX/apex-mesh/outputs/moud/AT1
cd /home/cxh/APEX/apex-mesh/src

mkdir -p "$MODEL"

run() { echo; echo "### $(date '+%F %T')  $*"; "$@"; }

# Half-Gaussian splatting reconstruction (plain parse_args -> pass everything explicitly).
run "$PY" train.py -s "$SRC" -m "$MODEL" --eval --sh_degree 3 --iterations 30000

# Underwater attenuation medium layer (sentinel -> inherits cfg_args eval/sh_degree/resolution).
run "$PY" train_medium.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 30000 --iterations 5000

# Projective geometry anchoring (saves to iteration 60000).
run "$PY" train_projective.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 30000 --iterations 5000 --save_iteration 60000

# Surface flattening (freeze positions/colour, thin out the Gaussian scales so
# the density field collapses to a clean sheet; saves 65000).
run "$PY" train_flatten.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 60000 --save_iteration 65000 --iterations 3000 \
    --lambda_flatten 2.0 --lambda_normal 0.05

# Offline mesh extraction — implicit MLS surface from the flattened Gaussian
# centres + oriented normals.  The density MT cannot sample the ~0.1mm-thin
# discs, so this zero-set of f(x)=sum w_i n_i.(x-mu_i)/sum w_i is used instead.
run "$PY" extract_implicit.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 65000 --res 512

# Offline refinement: edge-weighted Laplacian + small-component cleanup.
# MOUD meshes are heavily self-overlapping; the orientation prune in cleanup
# would tear the surface into ~11k components. Disable it (keep every face).
run "$PY" refine_mesh.py --source_path "$SRC" --model_path "$MODEL" --orient_thresh -1.1

# Differentiable mesh-reprojection joint refinement (full, tag "full").
run "$PY" train_mesh.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 65000 --save_iteration 70000 --tag full --iterations 3000

# Quantitative geometry: offline refined mesh.
run "$PY" evaluate_geometry.py --source_path "$SRC" --model_path "$MODEL" --load_iteration 65000

# Quantitative geometry: mesh-reprojection checkpoint (forward-view stability + normal consistency).
run "$PY" evaluate_geometry.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 65000 --save_iteration 70000 --stability --tag full

echo; echo "### $(date '+%F %T')  MOUD AT1 PIPELINE DONE"
