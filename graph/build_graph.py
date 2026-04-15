#!/usr/bin/env python3
"""Build scene graphs from GT masks or OneFormer3D predictions."""

import argparse

from graph.builders.gt import run_gt
from graph.builders.oneformer import run_oneformer
from graph.datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build graphs from GT masks or OneFormer3D predictions.')

    parser.add_argument('--source', choices=['gt', 'oneformer'], default='gt',
                        help='Graph source: gt or oneformer')
    parser.add_argument('--dataset', choices=['s200', 'spp'], default='s200',
                        help='Dataset key (s200 or spp)')

    parser.add_argument('--data-root', default=None,
                        help='Root directory containing points/, instance_mask/, semantic_mask/, infos')
    parser.add_argument('--out-root', default=None,
                        help='Output root directory for graphs')
    parser.add_argument('--results-dir', default=None,
                        help='Prediction .pth directory (oneformer only)')
    parser.add_argument('--label-allowlist', default=None,
                        help='Allowlist text file for spp semantics (top200_instance.txt)')

    parser.add_argument('--knn', type=int, default=5,
                        help='If >0, build k-NN graph (neighbors by centroid distance); else fully connected')
    parser.add_argument('--keep-background', action='store_true',
                        help='Keep instance id 0 (background); default drops it (gt only)')
    parser.add_argument('--enable-axis-allign', dest='enable_axis_align', type=bool, default=True,
                        help='If true (default), apply axis_align_matrix to XYZ before centroids/directions')
    parser.add_argument('--direction-plane', default='xz', choices=['xy', 'xz', 'yz'],
                        help='Axis pair used for left/right/up/down quantization (default xz)')
    parser.add_argument('--use-cardinals', type=bool, default=True,
                        help='If true, use directional labels; if false, label all edges as near')
    parser.add_argument('--enable-observer-frame', type=bool, default=True,
                        help='If true, project deltas into observer frame before labeling (adds front/back).')
    parser.add_argument('--observer-yaw-pitch-roll', type=float, nargs=3,
                        metavar=('YAW', 'PITCH', 'ROLL'),
                        default=(0.0, 0.0, 0.0),
                        help='Observer orientation in degrees (ZYX order).')
    parser.add_argument('--profile-latency', action='store_true', default=False,
                        help='If set, record average per-scene build time and sub-step breakdown')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset, args.data_root, args.label_allowlist)
    defaults = dataset.defaults

    if args.data_root is None:
        args.data_root = defaults['data_root']
    if args.label_allowlist is None and defaults.get('label_allowlist'):
        args.label_allowlist = defaults['label_allowlist']

    if args.source == 'gt':
        if args.out_root is None:
            args.out_root = defaults['out_root_gt']
        if dataset.keep_background_default:
            args.keep_background = True
        run_gt(args, dataset)
        return

    if args.source == 'oneformer':
        if dataset.map_oneformer_labels is None:
            raise NotImplementedError(
                f'OneFormer graphs are not implemented for dataset={dataset.key}')
        if args.out_root is None:
            args.out_root = defaults.get('out_root_oneformer')
        if args.results_dir is None:
            args.results_dir = defaults.get('results_dir_oneformer')
        if args.out_root is None or args.results_dir is None:
            raise ValueError('Missing --out-root or --results-dir for oneformer source')
        run_oneformer(args, dataset)
        return

    raise ValueError(f'Unknown source: {args.source}')


if __name__ == '__main__':
    main()
