# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
from os import path as osp

# Allow bare imports of sibling modules (indoor_converter, update_infos_to_v2)
# when invoked as `python tools/create_data.py` from repo root.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from indoor_converter import create_indoor_info_file
from update_infos_to_v2 import update_pkl_infos


def scannet_data_prep(root_path, info_prefix, out_dir, workers,
                     scannet200=False, scannetpp=False,
                     label_allowlist_file=None,
                     instance_dir=None):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers,
        label_allowlist_file=label_allowlist_file if scannetpp else None,
        instance_dir=instance_dir)
    info_train_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_test.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_test_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--label-allowlist', type=str, default=None,
    help='Optional allowlist file for scannetpp (one label per line).')
parser.add_argument(
    '--semantic-allowlist', type=str, default=None,
    help='Optional semantic allowlist file for scannetpp.')
parser.add_argument(
    '--instance-allowlist', type=str, default=None,
    help='Optional instance allowlist file for scannetpp.')
parser.add_argument(
    '--instance-dir', type=str, default=None,
    help='Optional instance data directory (relative to --root-path or absolute).')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset in ('scannet', 'scannet200', 'scannetpp'):
        # Choose allowlist default for scannetpp if not provided.
        # For ScanNet++, this allowlist defines instance/detection classes.
        allowlist = args.instance_allowlist or args.label_allowlist or args.semantic_allowlist
        if args.dataset == 'scannetpp' and allowlist is None:
            allowlist = osp.join(args.root_path, 'metadata', 'semantic_benchmark', 'top100.txt')
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers,
            scannet200=(args.dataset == 'scannet200'),
            scannetpp=(args.dataset == 'scannetpp'),
            label_allowlist_file=allowlist,
            instance_dir=args.instance_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
