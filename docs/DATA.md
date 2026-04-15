# Data Preparation

Everything required before OneFormer3D inference: raw dataset downloads,
instance-data extraction, MMDet3D info files, EZ-SP superpoints, and LLM
benchmark JSONs.

Target layout (all paths relative to the repo root — directories are
gitignored):

```
CORE_PVT/
├── data/
│   ├── scannet200/             scans/, scans_test/, meta_data/ (in repo),
│   │                            batch_load_scannet_data.py et al.
│   ├── scannetpp/              data/, splits/, metadata/      (metadata in repo)
│   ├── Scanrefer/              ScanRefer_filtered_*.json
│   ├── reason3d/               reason3d_{train,val}.json
│   └── surprise-3d/            surprise-3d-{train,val}.json
└── checkpoints/                oneformer3d_scannet200.pth, oneformer3d_scannetpp.pth,
                                ezsp_scannet200_partition.ckpt
```

## 1. Download raw datasets

All five datasets are user-provided (terms-of-use agreements and account
registrations required upstream). Place them at the paths listed below —
symlink large datasets into the repo rather than copying.

### ScanNet v2
Request access at http://www.scan-net.org/.

```
data/scannet200/
├── scans/
│   └── scene####_##/
│       ├── scene####_##_vh_clean_2.ply
│       ├── scene####_##.aggregation.json
│       ├── scene####_##_vh_clean_2.0.010000.segs.json
│       └── scene####_##.txt
├── scans_test/
│   └── scene####_##/ ...
├── meta_data/        (already in repo: scannetv2_{train,val,test}.txt, scannetv2-labels.combined.tsv)
├── batch_load_scannet_data.py, load_scannet_data.py, scannet_utils.py   (in repo)
```

### ScanNet++
Request access at https://kaldir.vc.in.tum.de/scannetpp/.

```
data/scannetpp/
├── data/
│   └── <scene_id>/scans/{mesh_aligned_0.05.ply, segments_anno.json, segments.json}
├── splits/           nvs_sem_{train,val,test}.txt
└── metadata/         (already in repo: semantic_benchmark/{map_benchmark.csv, top100.txt, top100_instance.txt, top200_semantic.txt, top200_instance.txt})
```

### ScanRefer
Download from https://github.com/daveredrum/ScanRefer.

```
data/Scanrefer/
├── ScanRefer_filtered.json
├── ScanRefer_filtered_{train,val}.json
├── ScanRefer_filtered_{train,val}.txt
└── scanrefer_unique_multiple_lookup.json
```

### Reason3D
Download from https://github.com/reason3d/reason3d.

```
data/reason3d/
├── reason3d_{train,val}.json
└── read_{train,val}.json
```

### Surprise3D
Download from https://github.com/SilvanoBuehler/Surprise3D.

```
data/surprise-3d/
├── surprise-3d-{train,val}.json
└── metadata.json
```

## 2. Install checkpoints

Place the three provided checkpoints under `checkpoints/`:

```
checkpoints/
├── oneformer3d_scannet200.pth           # OneFormer3D trained on ScanNet200
├── oneformer3d_scannetpp.pth            # OneFormer3D trained on ScanNet++
└── ezsp_scannet200_partition.ckpt       # EZ-SP partition network
```

## 3. Preprocess ScanNet v2

> **Dependency:** `batch_load_scannet_data.py` calls
> `segmentator.segment_mesh(...)` from the
> [Karbo123/segmentator](https://github.com/Karbo123/segmentator) C++
> library. Install it per [docs/INSTALL.md](INSTALL.md) before running.

```bash
# 3a — raw export (writes data/scannet200/scannet_instance_data/)
cd data/scannet200
python batch_load_scannet_data.py --scannet200
cd ../..

# 3b — MMDet3D info files
python tools/create_data.py scannet200 \
    --root-path data/scannet200 \
    --out-dir   data/scannet200 \
    --extra-tag scannet200
```

**Output** (`data/scannet200/`):
- `points/{scene}.bin`, `instance_mask/{scene}.bin`, `semantic_mask/{scene}.bin`,
  `super_points/{scene}.bin`
- `scannet200_oneformer3d_infos_{train,val,test}.pkl`

## 4. Preprocess ScanNet++

The fork's `oneformer3d_1xb4_scannetpp_spconv_sdpa_ext.py` config (the
one that matches `checkpoints/oneformer3d_scannetpp.pth`) trains over
**200 semantic / 194 instance** classes, so the raw export and info pkls
must use the `top200_*` allowlists. Using `top100_instance.txt` here
produces a label space that collides with the 200-class output head and
gives near-random IoU at inference time.

```bash
# 4a — raw export (200-class allowlists)
python data/scannet200/batch_load_scannet_data.py --scannetpp \
    --train_scannet_dir     data/scannetpp/data \
    --test_scannet_dir      data/scannetpp/data \
    --train_scan_names_file data/scannetpp/splits/nvs_sem_train.txt \
    --test_scan_names_file  data/scannetpp/splits/nvs_sem_test.txt \
    --label_map_file        data/scannetpp/metadata/semantic_benchmark/map_benchmark.csv \
    --semantic_allowlist    data/scannetpp/metadata/semantic_benchmark/top200_semantic.txt \
    --instance_allowlist    data/scannetpp/metadata/semantic_benchmark/top200_instance.txt \
    --output_folder         data/scannetpp/scannetpp_instance_data

# 4b — MMDet3D info files
python tools/create_data.py scannetpp \
    --root-path       data/scannetpp \
    --out-dir         data/scannetpp \
    --extra-tag       scannetpp \
    --label-allowlist data/scannetpp/metadata/semantic_benchmark/top200_instance.txt
```

## 5. Generate EZ-SP superpoints (ScanNet200)

Writes EZ-SP superpoints to `data/scannet200/super_points_ezsp/`.
Segmentator superpoints at `data/scannet200/super_points/` are left
untouched; the OneFormer3D fork selects which directory to read at
inference time. ScanNet++ uses the segmentator superpoints only.

```bash
bash scripts/run_ezsp.sh data/scannet200 checkpoints/ezsp_scannet200_partition.ckpt
```

## 6. Expected directory tree

```
CORE_PVT/
├── data/
│   ├── scannet200/                   # raw + step 3 (segmentator) + step 5 (EZ-SP)
│   │   ├── scans/, scans_test/       # step 1 (symlinks to raw downloads)
│   │   ├── meta_data/                # in repo (splits + labels tsv)
│   │   ├── batch_load_scannet_data.py, load_scannet_data.py, scannet_utils.py
│   │   ├── scannet_instance_data/    # step 3a intermediate .npy
│   │   ├── points/, instance_mask/, semantic_mask/
│   │   ├── super_points/             # segmentator (default — pkl references this)
│   │   ├── super_points_ezsp/        # EZ-SP (parallel; swapped in by fork config)
│   │   └── scannet200_oneformer3d_infos_{train,val,test}.pkl
│   ├── scannetpp/
│   │   ├── data/, splits/            # step 1
│   │   ├── metadata/                 # in repo
│   │   ├── scannetpp_instance_data/  # step 4a
│   │   ├── points/, instance_mask/, semantic_mask/, super_points/  # step 4b
│   │   └── scannetpp_oneformer3d_infos_{train,val,test}.pkl
│   ├── Scanrefer/                    # step 1
│   ├── reason3d/                     # step 1
│   └── surprise-3d/                  # step 1
├── checkpoints/                      # step 2
│   ├── oneformer3d_scannet200.pth
│   ├── oneformer3d_scannetpp.pth
│   └── ezsp_scannet200_partition.ckpt
└── ezsp/output/super_points_ezsp/    # step 5 staging
```
