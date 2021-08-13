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

class MMSegmentationTrainer:
    def __init__(self):
        self.reset()

        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.local_rank)

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
        self.launcher = False
        self.cfg = None
        self.seed = None
        self.deterministic = None

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

    def initEnv(self, work_dir=None, seed=None):
        if work_dir is not None:
            self.work_dir = work_dir
        else:
            self.work_dir = osp.join('./work_dirs',
                                     osp.splitext(osp.basename(self.config))[0])

        if seed is not None:
            self.seed = seed
            logger.info(f'Set random seed to {self.seed}, deterministic: '
                        f'{self.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
 


        self.cfg.work_dir = self.work_dir
 
        mmcv.mkdir_or_exist(osp.abspath(self.work_dir))
        self.cfg.dump(osp.join(self.work_dir, osp.basename(self.config)))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

        meta = dict()

        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        logger.info(f'Distributed training: {self.distributed}')
        logger.info(f'Config:\n{self.cfg.pretty_text}')

    def loadModel(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint

        cfg = Config.fromfile(self.config)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        pass


