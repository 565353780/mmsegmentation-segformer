#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

import torch.distributed as dist

class MMSegmentationTrainer:
    def __init__(self):
        self.reset()

        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.local_rank)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = "192.168.1.15"
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = "5678"

        dist.init_process_group('gloo', init_method='env://', rank=0, world_size=1)

    def reset(self):
        self.config = None
        self.checkpoint = None
        self.work_dir = None
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0
        self.local_rank = 0
        self.gpu_ids = range(1)
        self.distributed = False
        self.launcher = False
        self.cfg = None
        self.seed = None
        self.meta = None
        self.deterministic = False
        self.timestamp = None
        self.logger = None
        self.env_info = None
        self.datasets = None
        self.no_validate = False

    def resetTimer(self):
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def startTimer(self):
        self.time_start = time.time()

    def endTimer(self, save_time=True):
        time_end = time.time()

        if not save_time:
            return

        if self.time_start is None:
            print("startTimer must run first!")
            return

        if time_end > self.time_start:
            self.total_time_sum += time_end - self.time_start
            self.detected_num += 1
        else:
            print("Time end must > time start!")

    def getAverageTime(self):
        if self.detected_num == 0:
            return -1

        return 1.0 * self.total_time_sum / self.detected_num

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def setConfig(self, config):
        self.config = config
        self.cfg = Config.fromfile(self.config)

        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        return

    def setCheckPoint(self, checkpoint):
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.cfg.resume_from = self.checkpoint
        return

    def setWorkDir(self, work_dir):
        if work_dir is not None:
            self.work_dir = work_dir
        else:
            self.work_dir = osp.join('./work_dirs',
                                     osp.splitext(osp.basename(self.config))[0])

        self.cfg.work_dir = self.work_dir
        return

    def setSeed(self, seed):
        self.seed = seed
        if self.seed is not None:
            logger.info(f'Set random seed to {self.seed}, deterministic: '
                        f'{self.deterministic}')
            set_random_seed(self.seed, deterministic=self.deterministic)

        self.cfg.seed = self.seed
        return

    def setEnv(self):
        self.cfg.gpu_ids = self.gpu_ids
        mmcv.mkdir_or_exist(osp.abspath(self.work_dir))
        self.cfg.dump(osp.join(self.work_dir, osp.basename(self.config)))
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.work_dir, f'{self.timestamp}.log')
        self.logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

        env_info_dict = collect_env()
        self.env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.logger.info('Environment info:\n' + dash_line + self.env_info + '\n' +
                    dash_line)
        return

    def setMeta(self):
        self.meta = dict()
        self.meta['env_info'] = self.env_info
        self.meta['seed'] = self.seed
        self.meta['exp_name'] = osp.basename(self.config)
        return

    def initEnv(self, work_dir=None, seed=None):
        self.setWorkDir(work_dir)
        self.setSeed(seed)
        self.setEnv()
        self.setMeta()

        self.logger.info(f'Distributed training: {self.distributed}')
        self.logger.info(f'Config:\n{self.cfg.pretty_text}')

    def loadModel(self, config, checkpoint):
        self.setConfig(config)
        self.setCheckPoint(checkpoint)

        self.initEnv()

        self.model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg'))

        self.logger.info(self.model)
        return

    def loadDatasets(self):
        self.datasets = [build_dataset(self.cfg.data.train)]

        if len(self.cfg.workflow) == 2:
            val_dataset = copy.deepcopy(self.cfg.data.val)
            val_dataset.pipeline = self.cfg.data.train.pipeline
            self.datasets.append(build_dataset(val_dataset))

        if self.cfg.checkpoint_config is not None:
            self.cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=self.cfg.pretty_text,
                CLASSES=self.datasets[0].CLASSES,
                PALETTE=self.datasets[0].PALETTE)

        self.model.CLASSES = self.datasets[0].CLASSES
        return

    def train(self):
        train_segmentor(
            self.model,
            self.datasets,
            self.cfg,
            distributed=self.distributed,
            validate=(not self.no_validate),
            timestamp=self.timestamp,
            meta=self.meta)
        return

if __name__ == "__main__":
    config = "../SegFormer/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py"
    checkpoint = None

    mm_segmentation_trainer = MMSegmentationTrainer()

    mm_segmentation_trainer.loadModel(config, checkpoint)
    mm_segmentation_trainer.loadDatasets()

    # for spending less GPU memory
    mm_segmentation_trainer.cfg.data['samples_per_gpu'] = 1

    mm_segmentation_trainer.train()

