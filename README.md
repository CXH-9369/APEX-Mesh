# APEX-Mesh

**Attenuation-aware Projective Edge-preserving eXtraction for Underwater 3D Reconstruction & Rendering**

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.7-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-11.6-76B900?logo=nvidia&logoColor=white" alt="CUDA"></a>
  <a href="#license-and-acknowledgements"><img src="https://img.shields.io/badge/License-Gaussian--Splatting-blue.svg" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/CXH-9369/APEX-Mesh/main/assets/pipeline.svg" width="85%" alt="APEX-Mesh pipeline">
</p>

## Contents

- [Abstract](#abstract)
- [Key contributions](#key-contributions)
- [Method](#method)
- [Installation](#installation)
- [Running](#running)
- [Visualization](#visualization)
- [Repository structure](#repository-structure)
- [BibTeX](#bibtex)
- [License and Acknowledgements](#license-and-acknowledgements)

## Abstract

Multi-view reconstruction of underwater scenes is challenging: light attenuates and
backscatters as it travels through the water column, so standard radiance-field
pipelines — which assume clean-air image formation — produce soft, rounded geometry
and scale-drifted reconstructions. APEX-Mesh overcomes both limitations with a
physically-grounded underwater imaging model, a surface-collapsed Gaussian field, and
an *anisotropic*, edge-preserving implicit surface extraction. The result is a sharp,
watertight, colored mesh that reproduces the fine seabed edges and corners that
isotropic kernels systematically blur.

## Key contributions

- **Physically-grounded underwater imaging.** A differentiable water-column model
  (attenuation + backscatter) is integrated into field optimisation, so reconstruction
  is consistent with the true underwater image formation rather than a clean-air
  assumption.
- **Metric projective anchoring.** Anchored inverse-depth and exact optical-path
  constraints pin the geometry to a physically meaningful scale that is consistent
  across views.
- **Oriented surface collapse.** The half-Gaussian field is flattened into thin,
  oriented surface sheets whose thinnest axis coincides with the surface normal, so the
  representation concentrates exactly on the surface.
- **Edge-preserving surface extraction.** An anisotropic (APSS-style) kernel keeps the
  implicit MLS field razor-thin along the normal but wide tangentially, and a watertight
  marching-tetrahedra iso-surfacing reproduces sharp creases instead of rounding them.
- **Joint mesh–field refinement.** An edge-weighted Laplacian smooths flat regions while
  preserving boundary edges, and a differentiable reprojection stage re-anchors the
  shared mesh to the field.

## Method

APEX-Mesh reconstructs a watertight, colored surface mesh in five coordinated stages.

**Underwater image formation.** A differentiable model `(I, depth) → (Î, t, B)` maps
the raw scene colour to the observed image through a medium transmittance `t` and a
backscatter term `B`, and is baked into the rendering used for optimisation.

**Projective anchoring.** Per-camera projective constraints give the field a physically
meaningful scale and a consistent cross-view coordinate frame.

**Surface flattening.** Each half-Gaussian is collapsed into a thin, surface-aligned
disc: its centre `μᵢ` lies on the surface and its thinnest axis `nᵢ` (up to sign) is the
surface normal.

**Anisotropic implicit surface.** Given the flattened primitives `(μᵢ, nᵢ, αᵢ)`, APEX-Mesh
builds the implicit MLS field

```
        Σᵢ wᵢ(x) · nᵢ · (x − μᵢ)
f(x) = ─────────────────────────
              Σᵢ wᵢ(x)
```

whose zero set is the surface. The kernel `wᵢ` is anisotropic:

```
wᵢ(x) = αᵢ · exp( −‖d⊥‖² / (2σ²)  −  (d·nᵢ)² / (2σ_n²) )
```

with `d = x − μᵢ` and `d⊥ = d − (d·nᵢ) nᵢ`. The tangential radius `σ` is wide, while the
normal radius `σ_n` is razor-thin — so the field follows the discs' normals and preserves
sharp creases that an isotropic kernel (`σ_n = σ`) rounds off. The zero set is extracted
with a watertight marching-tetrahedra decomposition (6 tetrahedra per cell) and welded
via exact vertex deduplication.

**Edge-preserving refinement.** An edge-weighted Laplacian smooths flat regions while
preserving high-gradient boundary edges (and never smoothing across open boundaries),
followed by a differentiable mesh-reprojection loss that re-anchors the shared mesh to
the field.

## Installation

```bash
git clone --recursive https://github.com/CXH-9369/APEX-Mesh.git
cd APEX-Mesh

conda env create -f environment.yml
conda activate APEX

pip install ./src/submodules/diff-gaussian-rasterization
pip install ./src/submodules/simple-knn
```

Requires **PyTorch 1.12.1 + CUDA 11.6 + Python 3.7**. `scripts/setup_env.sh` provides a
one-shot build script.

## Running

The full reconstruction pipeline is a single script:

```bash
bash src/scripts/run_moud_at1.sh
```

Or run each stage explicitly (`SRC` = dataset, `MODEL` = output directory):

```bash
SRC=/path/to/MOUD/AT1
MODEL=/path/to/outputs/AT1

# learn the half-Gaussian field
python train.py -s "$SRC" -m "$MODEL" --eval --sh_degree 3 --iterations 30000

# refine under the underwater image-formation model
python train_medium.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 30000 --iterations 5000

# apply the projective geometry gauge
python train_projective.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 30000 --iterations 5000 --save_iteration 60000

# flatten the field onto the surface
python train_flatten.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 60000 --save_iteration 65000 --iterations 3000 \
    --lambda_flatten 2.0 --lambda_normal 0.05

# extract the anisotropic implicit surface (marching tetrahedra)
#   coarse : --res 512
#   fine   : --res 1920 --sigma 0.06 --w_thresh 0.2
python extract_implicit.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 65000 --res 1920 --sigma 0.06 --w_thresh 0.2

# edge-preserving mesh refinement
#   --orient_thresh -1.1 disables the orientation prune on heavily
#   self-overlapping underwater meshes.
python refine_mesh.py --source_path "$SRC" --model_path "$MODEL" --orient_thresh -1.1

# joint mesh–field refinement
python train_mesh.py --source_path "$SRC" --model_path "$MODEL" \
    --load_iteration 65000 --save_iteration 70000 --tag full --iterations 3000

# geometric evaluation
python evaluate_geometry.py --source_path "$SRC" --model_path "$MODEL" --load_iteration 65000

# colour chain (vertex colour → floater pruning → image-sample colour)
python bake_vertex_color.py --source_path "$SRC" --model_path "$MODEL" --load_iteration 65000
python src/scripts/prune_floaters.py AT1
python src/scripts/bake_image_color.py AT1
```

> **Extraction recipe.** Always pass `--sigma 0.06 --w_thresh 0.2` when extracting at
> fine resolution (`--res ≥ 1920`); the defaults fragment the implicit field and open
> holes in the mesh.

## Visualization

```bash
python src/scripts/export_viewer.py AT1       # export the mesh for the web viewer
python -m http.server 8000 -d src/scripts     # then open http://localhost:8000/viewer.html
```

## Repository structure

```
APEX-Mesh/
├── src/
│   ├── train.py                    # half-Gaussian field optimisation
│   ├── train_medium.py             # underwater image-formation refinement
│   ├── train_projective.py         # projective geometry anchoring
│   ├── train_flatten.py            # surface flattening
│   ├── extract_implicit.py         # anisotropic implicit surface extraction
│   ├── refine_mesh.py              # edge-preserving mesh refinement
│   ├── train_mesh.py               # joint mesh–field refinement
│   ├── evaluate_geometry.py        # geometric evaluation
│   ├── bake_vertex_color.py        # colour baking
│   ├── mesh/                       # extraction / refinement / io / evaluation
│   ├── medium/                     # underwater image-formation model
│   ├── geometry/                   # mesh losses
│   ├── gaussian_renderer/          # CUDA rasterizer bindings
│   ├── submodules/                 # diff-gaussian-rasterization + simple-knn
│   ├── test_*.py                   # unit tests
│   └── scripts/                    # run scripts, viewer, colour chain
├── scripts/                        # environment setup + baselines
└── environment.yml
```

## BibTeX

If you use APEX-Mesh, please cite:

```bibtex
@article{apexmesh,
  title   = {APEX-Mesh: Attenuation-aware Projective Edge-preserving eXtraction for Underwater 3D Reconstruction and Rendering},
  author  = {},   % TODO: fill in authors
  journal = {},   % TODO: fill in venue
  year    = {}    % TODO: fill in year
}
```

The codebase builds on the following works, which should also be cited:

```bibtex
@article{li20243d,
  title   = {3D-HGS: 3D Half-Gaussian Splatting},
  author  = {Li, Haolin and Liu, Jinyang and Sznaier, Mario and Camps, Octavia},
  journal = {arXiv preprint arXiv:2406.02720},
  year    = {2024}
}

@article{kerbl20233dgs,
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal = {ACM Transactions on Graphics},
  volume  = {42},
  number  = {4},
  year    = {2023}
}
```

## License and Acknowledgements

APEX-Mesh is released for research use. If you use it in a publication, please cite
our paper (see [BibTeX](#bibtex) above).

The codebase is derived from [3D-HGS](https://github.com/lihaolin88/3D-Half-Gaussian-Splatting)
and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and
is distributed under the **Gaussian-Splatting License** (non-commercial research use).
See [LICENSE.md](LICENSE.md).
