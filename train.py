import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
from PIL import Image

@TRANSFORMS.register_module()
class CheckLabelRange(BaseTransform):
    def __init__(self):
        pass

    def transform(self, data_results):
        # check segmentation path
        seg_map_path = data_results.get('seg_map_path', None)
        print(f"Segmentation map full path: {seg_map_path}")
        print(data_results.keys())

        # Print details about the label_map key
        if 'label_map' in data_results:
            labels = data_results['label_map']
            print("Type of label_map:", type(labels))
            print("Content of label_map:", labels)
            if isinstance(labels, np.ndarray):
                print("Min label:", np.min(labels))
                print("Max label:", np.max(labels))
            else:
                print("Label data is not a numpy array.")
        else:
            raise KeyError("Ground truth labels not found in the results.")
        
        return data_results


@TRANSFORMS.register_module()
class CustomLoadAnnotations(BaseTransform):
    def __init__(self, reduce_zero_label=False):
        self.reduce_zero_label = reduce_zero_label

    def transform(self, annotation_results):
        # 获取标注图像路径
        seg_map_path = annotation_results.get('seg_map_path', None)
        if seg_map_path is not None:
            # 替换为 _labelIds.png 文件
            seg_map_path = seg_map_path.replace('_labelTrainIds.png', '_labelIds.png')
            annotation_results['seg_map_path'] = seg_map_path

            # 加载标注图像
            seg_map = np.array(Image.open(seg_map_path), dtype=np.uint8)
            print(f"gt_seg_map added to results: {annotation_results['seg_map_path']}")
            
            # 将标注数据添加到 results 中
            annotation_results['gt_seg_map'] = seg_map

            if self.reduce_zero_label:
                # 如果需要将标签0排除，进行标签映射
                annotation_results['gt_seg_map'][annotation_results['gt_seg_map'] == 0] = 255
            
            annotation_results['label_map'] = annotation_results['gt_seg_map']

        else:
            raise ValueError(f"Segmentation map path is missing in results: {seg_map_path}")

        return annotation_results



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume
    cfg.train_dataloader['dataset']['pipeline'].insert(1, dict(type='CheckLabelRange'))

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
    # python3 /home/wayne/Desktop/PR_final/train.py /home/wayne/Desktop/PR_final/segformer/config.py --work-dir /home/wayne/Desktop/PR_final/segformer/work_dirs
