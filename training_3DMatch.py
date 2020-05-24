import os
import time
import shutil
from datasets.ThreeDMatch import ThreeDMatchDataset
from utils.config import Config
from trainer_desc import Trainer
from models.KPFCNN_desc import KPFCNN
from datasets.dataloader import get_dataloader
from utils.loss import BatchHardLoss
from torch import optim
from torch import nn
import torch


class ThreeDMatchConfig(Config):
    # dataset
    dataset = '3DMatch'
    first_features_dim = 32
    safe_radius = 0.1
    first_subsampling_dl = 0.03
    in_features_dim = 1
    data_train_dir = "/ssd2/xuyang/3DMatch/"
    data_test_dir = "/ssd2/xuyang/3DMatch/"
    train_batch_size = 1
    test_batch_size = 1

    # model
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb', # 'resnetb_deformable',
                    'resnetb_strided', # 'resnetb_deformable_strided',
                    'resnetb', # 'resnetb_deformable',
                    'resnetb_strided', # 'resnetb_deformable_strided',
                    'resnetb', # 'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']
    dropout = 0.5
    resume = None
    use_batch_norm = True
    batch_norm_momentum = 0.02
    # https://github.com/pytorch/examples/issues/289 pytorch bn momentum 0.02 == tensorflow bn momentum 0.98

    # kernel point convolution
    KP_influence = 'linear'
    KP_extent = 1.0
    convolution_mode = 'sum'

    # training
    max_epoch = 200
    learning_rate = 1e-1
    momentum = 0.98
    exp_gamma = 0.1 ** (1 / 80)
    exp_interval = 1


class Args(object):
    def __init__(self, config):
        is_test = False
        if is_test:
            self.experiment_id = "KPConvNet" + time.strftime('%m%d%H%M') + 'Test'
        else:
            self.experiment_id = "KPConvNet" + time.strftime('%m%d%H%M')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = True
        self.config = config

        # snapshot
        self.snapshot_interval = 5
        snapshot_root = f'snapshot/{config.dataset}_{self.experiment_id}'
        tensorboard_root = f'tensorboard/{config.dataset}_{self.experiment_id}'
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('.', 'training_3DMatch.py'), os.path.join(snapshot_root, 'train.py'))
        shutil.copy2(os.path.join('.', 'trainer_desc.py'), os.path.join(snapshot_root, 'trainer.py'))
        shutil.copy2(os.path.join('datasets', 'ThreeDMatch.py'), os.path.join(snapshot_root, 'dataset.py'))
        shutil.copy2(os.path.join('datasets', 'dataloader.py'), os.path.join(snapshot_root, 'dataloader.py'))
        shutil.copy2(os.path.join('utils', 'loss.py'), os.path.join(snapshot_root, 'loss.py'))
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        # dataset & dataloader
        self.train_set = ThreeDMatchDataset(root=config.data_train_dir,
                                         split='train',
                                         num_node=64,
                                         config=config,
                                         )
        self.test_set = ThreeDMatchDataset(root=config.data_test_dir,
                                        split='val',
                                        num_node=64,
                                        config=config,
                                        )
        self.train_loader = get_dataloader(dataset=self.train_set,
                                           batch_size=config.train_batch_size,
                                           shuffle=True,
                                           num_workers=config.train_batch_size,
                                           )
        self.test_loader = get_dataloader(dataset=self.test_set,
                                          batch_size=config.test_batch_size,
                                          shuffle=True,
                                          num_workers=config.test_batch_size,
                                          )
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Test set size:", self.test_loader.dataset.__len__())

        # model
        self.model = KPFCNN(config)
        self.resume = config.resume
        # optimizer 
        self.start_epoch = 0
        self.epoch = config.max_epoch
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.exp_gamma)
        self.scheduler_interval = config.exp_interval

        # evaluate
        self.evaluate_interval = 1
        self.evaluate_metric = BatchHardLoss(margin=1.0, metric='euclidean', safe_radius=config.safe_radius)

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    config = ThreeDMatchConfig()
    args = Args(config)
    trainer = Trainer(args)
    trainer.train()
