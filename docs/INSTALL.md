# Installation

Single conda env. Python 3.10, PyTorch 2.9.0 (CUDA 12.8), system CUDA 12.6
for nvcc, vLLM 0.13.0 in the same env.

## Paths

```bash
export CORE_PVT_ROOT=/path/to/CORE_PVT
export FORK_ROOT=/path/to/oneformer3d-fork
export BUILD_ROOT=/path/to/core_build
mkdir -p "$BUILD_ROOT"
```

## Step 1: Conda env + PyTorch

```bash
conda create -n core_test python=3.10 -y
conda activate core_test
python -m pip install --upgrade "pip<25" wheel

pip install torch==2.9.0

export TORCH_CUDA_ARCH_LIST="9.0"   # adjust per GPU
```

## Step 2: PyTorch Geometric

```bash
pip install torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
pip install torch-geometric==2.3.0
```

## Step 3: CUDA extensions (build from source)

```bash
pip install "setuptools<70" wheel ninja
```

### MinkowskiEngine

```bash
conda install -c conda-forge openblas -y
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

git clone https://github.com/NVIDIA/MinkowskiEngine.git $BUILD_ROOT/MinkowskiEngine
cd $BUILD_ROOT/MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
git apply $CORE_PVT_ROOT/thirdparty_patches/minkowskiengine_cuda12.patch
python setup.py install --blas=openblas --force_cuda
```

### spconv + cumm

```bash
pip install cumm-cu126==0.7.11
pip install --no-deps spconv-cu126==2.3.8
```

### segmentator

```bash
git clone https://github.com/Karbo123/segmentator.git $BUILD_ROOT/segmentator
cd $BUILD_ROOT/segmentator/csrc
git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696
sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/' CMakeLists.txt
mkdir build && cd build
cmake .. \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DPYTHON_INCLUDE_DIR=`python -c "import sysconfig; print(sysconfig.get_path('include'))"` \
    -DPYTHON_LIBRARY=`python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"` \
    -DCMAKE_INSTALL_PREFIX=`python -c 'import sysconfig; print(sysconfig.get_path("purelib"))'`
make -j$(nproc)
make install
pip install numba
```

### torchsparse

```bash
conda install -c conda-forge sparsehash -y
git clone https://github.com/mit-han-lab/torchsparse.git $BUILD_ROOT/torchsparse
cd $BUILD_ROOT/torchsparse
git checkout 385f5ce
pip install --no-build-isolation --no-deps .
pip install rootpath "backports.cached_property"
```

### FRNN + prefix_sum

```bash
git clone https://github.com/lxxue/FRNN.git $BUILD_ROOT/FRNN
cd $BUILD_ROOT/FRNN && git submodule update --init --recursive
pip install --no-build-isolation --no-deps .

cd $BUILD_ROOT/FRNN/external/prefix_sum
pip install --no-build-isolation --no-deps .
```

## Step 4: OpenMMLab

```bash
pip install mmengine==0.10.7
```

### mmcv (from source)

```bash
git clone https://github.com/open-mmlab/mmcv.git $BUILD_ROOT/mmcv
cd $BUILD_ROOT/mmcv
git checkout v2.2.0
MMCV_WITH_OPS=1 pip install --no-build-isolation --no-deps -e .
```

### mmdet + mmdet3d

```bash
pip install mmdet==3.3.0
pip install mmdet3d==1.4.0
```

### Patch mmcv upper-bound check

```bash
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'  # patched/" \
    "$(python -c 'import mmdet; import os; print(os.path.dirname(mmdet.__file__))')/__init__.py"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'  # patched/" \
    "$(python -c 'import mmdet3d; import os; print(os.path.dirname(mmdet3d.__file__))')/__init__.py"
```

## Step 5: OneFormer3D fork (editable)

```bash
pip install --no-deps -e "$FORK_ROOT"
pip install timm addict
```

## Step 6: CORE_PVT (editable)

```bash
pip install --no-deps -e "$CORE_PVT_ROOT"
pip install numpy scipy plyfile tqdm h5py
pip install colorhash pgeof torch-ransac3d gitpython torchmetrics==0.11.4
```

## Step 7: vLLM

```bash
pip install vllm==0.13.0
pip install transformers peft pycocoevalcap
```

## Runtime env

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

## Checkpoints

Three checkpoints are needed for the full pipeline. Place them under
`checkpoints/` at the repo root.

### OneFormer3D — SpConv backbone (default)
`checkpoints/oneformer3d_scannet200.pth`, `checkpoints/oneformer3d_scannetpp.pth`

Train via the OneFormer3D fork, or use the authors' released weights.
See the fork at https://github.com/yathAg/oneformer3d for configs and
training commands.

### EZ-SP partition checkpoint
`checkpoints/ezsp_scannet200_partition.ckpt`

This checkpoint drives the partition CNN used by
`ezsp.generate_superpoints`. Upstream SPT weights work directly:
download from https://github.com/drprojects/superpoint_transformer and
symlink into place.

To train from scratch on your own ScanNet200 data:

```bash
python -m ezsp.train_partition --raw_root data/scannet200
# See `python -m ezsp.train_partition --help` for all knobs
# (epochs, lr, cache/ckpt/runs dirs, axis_align, etc.).
```

The resulting best checkpoint lands under
`ezsp/checkpoints/scannet200_partition/` by default; copy or symlink it
to `checkpoints/ezsp_scannet200_partition.ckpt` before running
`scripts/run_ezsp.sh`.
