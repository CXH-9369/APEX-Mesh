#!/usr/bin/env bash
# APEX-Mesh environment build script
# 1) Remove the precompiled CUDA extension artifacts committed in the upstream repo (not reused)
# 2) Activate the APEX env, rebuild the rasterizer and simple-knn, and install pip deps
set -eo pipefail
# Note: do NOT use `set -u`; it conflicts with conda's MKL activation script (MKL_INTERFACE_LAYER unset)

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
cd "$SRC_DIR"

echo "[1/3] removing committed precompiled artifacts..."
rm -rf submodules/diff-gaussian-rasterization/build \
       submodules/diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info \
       submodules/simple-knn/build \
       submodules/simple-knn/simple_knn.egg-info
find . -name '*.so' -delete

echo "[2/3] activating APEX env..."
source /home/cxh/anaconda3/bin/activate APEX

# A6000 = compute capability 8.6; pin it so nvcc does not build only the default old arch
export TORCH_CUDA_ARCH_LIST="8.6"
# system nvcc is 11.8; compatible with the torch cu116 runtime (this machine's existing combo)
echo "   nvcc: $(which nvcc) ($(nvcc --version | tail -1))"

echo "[3/3] build & install extensions + pip deps..."
pip install submodules/diff-gaussian-rasterization \
            submodules/simple-knn \
            lpips==0.1.4 \
            scipy==1.7.3 \
            matplotlib==3.5.3 \
            opencv-python-headless==5.0.0.93

echo "== done. verifying imports =="
python -c "import diff_gaussian_rasterization, simple_knn; print('OK: extensions importable')"
