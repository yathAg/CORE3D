#!/usr/bin/env bash
#
# CORE_PVT — rebuild `core_test` conda env on scratch, targeting torch 2.9.0,
# with vllm installed into the SAME env as the training stack. Single-env
# configuration. No PTv3. No dataset prep (see CHECKLIST.md §4/§5 for that).
#
# Usage
# -----
#   bash scripts/rebuild_core_test.sh              # run the whole chain
#   bash scripts/rebuild_core_test.sh step3        # run one step only
#
# Iron rule
# ---------
#   Every `pip install` below is either `--no-deps` or `--no-build-isolation`,
#   or both. This is NOT stylistic: `torchsparse/setup.py:52-59` has an
#   unpinned `"torch"` in install_requires that, without --no-deps, resolves
#   to the latest PyPI release and silently replaces your pinned torch.
#   `check_torch` runs after every install that could touch the pin and
#   aborts loudly on drift — you fix the cause, re-run from the failed
#   step, then continue.
#
# Not in scope (do NOT uncomment without thinking)
# ------------------------------------------------
#   * PTv3 backbone (`pip install addict timm`)
#   * Dataset prep — ScanNet200, ScanNet++ (see docs/DATA.md)

set -euo pipefail

# ---------- constants ----------------------------------------------------

# User-maintained source trees (installed editable in place). Defaults
# derive from this script's location so a fresh clone Just Works; override
# any of these by exporting the env var before invoking.
_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CORE_PVT_ROOT=${CORE_PVT_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}
FORK_ROOT=${FORK_ROOT:-$(cd "$CORE_PVT_ROOT/.." && pwd)/oneformer3d-fork}

# Third-party source clones and the conda env. Defaults sit next to
# CORE_PVT so cluster users get scratch-resident installs without baking
# any specific username into the script.
BUILD_ROOT=${BUILD_ROOT:-$(cd "$CORE_PVT_ROOT/.." && pwd)/core_build}
ENV_PREFIX=${ENV_PREFIX:-$(cd "$CORE_PVT_ROOT/.." && pwd)/.conda/envs/core_test}

TORCH_PIN="2.9.0"
export TORCH_CUDA_ARCH_LIST="9.0"          # H100 only
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # needed by ezsp at runtime

# ---------- helpers ------------------------------------------------------

log() { printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n!!! FATAL: %s\n' "$*" >&2; exit 1; }

check_torch() {
    local actual
    actual=$(pip show torch 2>/dev/null | awk '/^Version:/ {print $2}')
    if [[ "$actual" != "$TORCH_PIN" ]]; then
        printf '\n!!! torch drifted: expected %s, got "%s"\n' "$TORCH_PIN" "$actual" >&2
        pip show torch >&2 || true
        die "torch pin broken — stop and investigate which pip install did this"
    fi
    printf '    [OK] torch still pinned to %s\n' "$TORCH_PIN"
}

require_env() {
    [[ "${CONDA_PREFIX:-}" == "$ENV_PREFIX" ]] \
        || die "wrong env active: CONDA_PREFIX=${CONDA_PREFIX:-<unset>}, expected $ENV_PREFIX (run: conda activate $ENV_PREFIX)"
}

# Make `conda activate` available inside this non-login shell.
# Prefer the user's existing conda; fall back to the standard miniconda path.
_conda_init() {
    if [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        die "conda not on PATH — source your shell init or load a conda module first"
    fi
}

# ---------- step functions -----------------------------------------------

step0_wipe() {
    log "Step 0: remove broken core_test env (if present) and prep scratch dirs"
    _conda_init
    conda deactivate 2>/dev/null || true

    # Old scratch env (name form)
    if conda env list | awk '{print $1}' | grep -qx core_test; then
        log "Removing old scratch core_test env"
        conda env remove -n core_test -y
    fi

    # Any stale env at $ENV_PREFIX (prefix form) — in case of a partial previous run
    if [[ -d "$ENV_PREFIX" ]]; then
        log "Removing existing $ENV_PREFIX"
        conda env remove --prefix "$ENV_PREFIX" -y || rm -rf "$ENV_PREFIX"
    fi

    mkdir -p "$BUILD_ROOT" "$(dirname "$ENV_PREFIX")"
    echo "BUILD_ROOT    = $BUILD_ROOT"
    echo "ENV_PREFIX    = $ENV_PREFIX"
    echo "CORE_PVT_ROOT = $CORE_PVT_ROOT"
    echo "FORK_ROOT     = $FORK_ROOT"
}

step1_env_and_torch() {
    log "Step 1: conda env + torch $TORCH_PIN (PyPI default index)"
    _conda_init
    conda create --prefix "$ENV_PREFIX" python=3.10 -y
    conda activate "$ENV_PREFIX"
    require_env

    # Keep pip stable; newer pip versions have occasionally shifted
    # --no-deps semantics in ways that bit this project.
    python -m pip install --upgrade "pip<25" wheel

    # Install torch WITH its deps (no --no-deps here). Pip can't upgrade
    # torch beyond 2.9.0 because we pin it exactly, and torch's own
    # dependency list pins exact nvidia-cuda-* versions it was built
    # against — letting pip handle this is the only way to get the right
    # versions (manually listing them breaks when they change between
    # torch releases). The --no-deps discipline applies to EVERYTHING
    # AFTER torch, not to torch itself.
    pip install torch==$TORCH_PIN

    python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available:', torch.cuda.is_available())"
    check_torch
}

step2_pyg() {
    log "Step 2: PyG stack (torch-scatter, torch-cluster, torch-geometric)"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    # Try the torch-2.9 + cu128 wheel index first; if it 404s, fall back
    # to source build.
    pip install --no-deps torch-scatter torch-cluster \
        -f https://data.pyg.org/whl/torch-2.9.0+cu128.html || {
        log "PyG wheels not found for torch 2.9.0+cu128 — falling back to source build"
        pip install --no-deps --no-build-isolation torch-scatter torch-cluster
    }
    pip install --no-deps torch-geometric==2.3.0

    python -c "import torch_scatter, torch_cluster, torch_geometric; print('PyG OK')"
    check_torch
}

step3_build_deps() {
    log "Step 3: build-from-source CUDA deps"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    pip install --no-deps "setuptools<70" wheel ninja

    # OpenBLAS headers for MinkowskiEngine
    conda install -c conda-forge openblas -y
    export CPATH=$CONDA_PREFIX/include:${CPATH:-}
    export LIBRARY_PATH=$CONDA_PREFIX/lib:${LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

    # -------- 3a. MinkowskiEngine (patched) --------
    log "Step 3a: MinkowskiEngine"
    if [[ ! -d "$BUILD_ROOT/MinkowskiEngine" ]]; then
        git clone https://github.com/NVIDIA/MinkowskiEngine.git "$BUILD_ROOT/MinkowskiEngine"
        (cd "$BUILD_ROOT/MinkowskiEngine" \
            && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
            && git apply "$CORE_PVT_ROOT/thirdparty_patches/minkowskiengine_cuda12.patch")
    fi
    (cd "$BUILD_ROOT/MinkowskiEngine" \
        && python setup.py install --blas=openblas --force_cuda)
    python -c "import MinkowskiEngine as ME; print('MinkE', ME.__version__, 'CUDA:', ME.is_cuda_available())"
    check_torch

    # -------- 3b. cumm + spconv (wheels) --------
    log "Step 3b: cumm-cu126 + spconv-cu126 (wheels)"
    pip install --no-deps cumm-cu126==0.7.11
    pip install --no-deps spconv-cu126==2.3.8
    python -c "import spconv; print('spconv OK')"
    check_torch

    # -------- 3c. segmentator (C++ lib + numba Python stub dep) --------
    log "Step 3c: segmentator"
    if [[ ! -d "$BUILD_ROOT/segmentator" ]]; then
        git clone https://github.com/Karbo123/segmentator.git "$BUILD_ROOT/segmentator"
    fi
    (cd "$BUILD_ROOT/segmentator/csrc" \
        && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
        && sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/' CMakeLists.txt \
        && mkdir -p build && cd build \
        && cmake .. \
            -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
            -DPYTHON_INCLUDE_DIR="$(python -c "import sysconfig; print(sysconfig.get_path('include'))")" \
            -DPYTHON_LIBRARY="$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")" \
            -DCMAKE_INSTALL_PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
        && make -j"$(nproc)" \
        && make install)

    # segmentator's Python stub imports numba at module load time —
    # INSTALL.md was missing this for a long time.
    pip install --no-deps numba

    python -c "import segmentator; print('segmentator OK')"
    check_torch

    # -------- 3d. torchsparse (CRITICAL: --no-deps, commit 385f5ce) --------
    log "Step 3d: torchsparse (commit 385f5ce, --no-deps --no-build-isolation)"
    conda install -c conda-forge sparsehash -y
    if [[ ! -d "$BUILD_ROOT/torchsparse" ]]; then
        git clone https://github.com/mit-han-lab/torchsparse.git "$BUILD_ROOT/torchsparse"
    fi
    (cd "$BUILD_ROOT/torchsparse" \
        && git fetch --all \
        && git checkout 385f5ce \
        && pip install --no-build-isolation --no-deps .)
    python -c "import torchsparse; print('torchsparse', torchsparse.__version__)"
    check_torch

    # -------- 3e. FRNN --------
    log "Step 3e: FRNN"
    if [[ ! -d "$BUILD_ROOT/FRNN" ]]; then
        git clone https://github.com/lxxue/FRNN.git "$BUILD_ROOT/FRNN"
        (cd "$BUILD_ROOT/FRNN" && git submodule update --init --recursive)
    fi
    (cd "$BUILD_ROOT/FRNN" && pip install --no-build-isolation --no-deps .)
    python -c "import frnn; print('frnn OK')"
    check_torch
}

step4_openmmlab() {
    log "Step 4: OpenMMLab (mmengine, mmcv from source, mmdet, mmdet3d)"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    pip install --no-deps mmengine==0.10.7

    # mmcv 2.2.0 from source — no torch 2.9 wheel exists upstream
    if [[ ! -d "$BUILD_ROOT/mmcv" ]]; then
        git clone https://github.com/open-mmlab/mmcv.git "$BUILD_ROOT/mmcv"
    fi
    (cd "$BUILD_ROOT/mmcv" \
        && git fetch --tags \
        && git checkout v2.2.0 \
        && MMCV_WITH_OPS=1 pip install --no-build-isolation --no-deps -e .)
    python -c "import mmcv; print('mmcv', mmcv.__version__)"

    pip install --no-deps mmdet==3.3.0
    pip install --no-deps mmdet3d==1.4.0
    python -c "import mmdet3d; print('mmdet3d OK')"
    check_torch
}

step4b_fork() {
    log "Step 4b: OneFormer3D fork — editable install from existing scratch path"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    # PTv3 deps (addict, timm) intentionally SKIPPED — not in scope.
    # The fork lives at $FORK_ROOT on scratch (user-maintained source tree
    # with local mods + git history). Do NOT clone a fresh copy.
    [[ -d "$FORK_ROOT/oneformer3d" ]] \
        || die "Fork not found at $FORK_ROOT. If it was moved, export FORK_ROOT=... before running."
    pip install --no-deps -e "$FORK_ROOT"
    python -c "import oneformer3d; print('fork OK')"
    check_torch
}

step5_core_base() {
    log "Step 5: CORE_PVT base — editable install from existing scratch path"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    # Install CORE_PVT from its existing scratch location (user-maintained).
    # Pure-Python editable install: registers graph/, llm/, ezsp/ on sys.path.
    [[ -d "$CORE_PVT_ROOT/graph" ]] \
        || die "CORE_PVT not found at $CORE_PVT_ROOT. Export CORE_PVT_ROOT=... if it moved."
    pip install --no-deps -e "$CORE_PVT_ROOT"

    # Base runtime deps declared in pyproject.toml (but --no-deps skipped them)
    pip install --no-deps numpy scipy plyfile tqdm h5py

    python -c "import graph.build_graph, llm.run, ezsp.generate_superpoints; print('CORE OK')"
    check_torch
}

step5b_vllm() {
    log "Step 5b: vllm 0.13.0 + transformers + peft + pycocoevalcap (same env)"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    # vllm 0.13.0 matches the user's existing `vllm` conda env (torch 2.9.0 +
    # vllm 0.13.0, proven working). --no-deps keeps torch 2.9.0 pinned.
    pip install --no-deps vllm==0.13.0

    # vllm's Python-only runtime deps that CORE's llm/ code actually needs.
    # Each --no-deps so pip never re-touches torch.
    pip install --no-deps \
        transformers peft pycocoevalcap \
        tokenizers safetensors sentencepiece protobuf \
        accelerate datasets huggingface-hub

    python -c "import vllm; print('vllm', vllm.__version__)"
    python -c "import transformers; print('transformers', transformers.__version__)"
    check_torch
}

step6_smoke_test() {
    log "Step 6: import smoke test (CHECKLIST.md §2 equivalent)"
    _conda_init
    conda activate "$ENV_PREFIX"
    require_env

    python - <<'PY'
import importlib, sys
mods = [
    # framework + CUDA extensions
    "torch", "MinkowskiEngine", "spconv", "torchsparse", "frnn",
    "segmentator", "mmcv", "mmdet3d",
    # PyG
    "torch_scatter", "torch_cluster", "torch_geometric",
    # CORE_PVT packages
    "graph.build_graph", "graph.builders.gt", "graph.builders.oneformer",
    "llm.run", "llm.core", "llm.benchmarks", "llm.datasets", "llm.prompts",
    "llm.eval.eval_scanrefer_iou", "llm.eval.eval_reason3d_iou",
    "llm.eval.eval_surprise3d_iou", "llm.eval.eval_multi3drefer_f1",
    "ezsp.generate_superpoints",
    # vllm + HF
    "vllm", "transformers", "peft",
    # fork
    "oneformer3d",
]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  [OK]  {m}")
    except Exception as e:
        print(f"  [ERR] {m}: {type(e).__name__}: {e}")
        failed.append(m)
if failed:
    print(f"\n{len(failed)} import(s) failed")
    sys.exit(1)
print("\nAll imports passed.")
PY

    python -m graph.build_graph         --help >/dev/null && echo "graph CLI OK"
    python -m llm.run                   --help >/dev/null && echo "llm CLI OK"
    python -m ezsp.generate_superpoints --help >/dev/null && echo "ezsp CLI OK"
    check_torch

    log "ALL DONE — core_test env on scratch is ready."
    echo
    echo "Activate it with:"
    echo "    conda activate $ENV_PREFIX"
}

# ---------- dispatch -----------------------------------------------------

case "${1:-all}" in
    step0|wipe)          step0_wipe ;;
    step1|env)           step1_env_and_torch ;;
    step2|pyg)           step2_pyg ;;
    step3|build)         step3_build_deps ;;
    step4|mm|openmmlab)  step4_openmmlab ;;
    step4b|fork)         step4b_fork ;;
    step5|core)          step5_core_base ;;
    step5b|vllm)         step5b_vllm ;;
    step6|smoke|verify)  step6_smoke_test ;;
    all)
        step0_wipe
        step1_env_and_torch
        step2_pyg
        step3_build_deps
        step4_openmmlab
        step4b_fork
        step5_core_base
        step5b_vllm
        step6_smoke_test
        ;;
    *) die "usage: $0 [step0|step1|step2|step3|step4|step4b|step5|step5b|step6|all]" ;;
esac
