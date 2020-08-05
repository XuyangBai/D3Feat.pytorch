import os
import time
import shutil
import json 
from config import get_config
from easydict import EasyDict as edict
from datasets.ThreeDMatch import ThreeDMatchDataset, ThreeDMatchTestset
from trainer import Trainer
from models.architectures import KPFCNN
# from models.D3Feat import KPFCNN
from datasets.dataloader import get_dataloader
from utils.loss import ContrastiveLoss, CircleLoss, DetLoss
from torch import optim
from torch import nn
import torch


if __name__ == '__main__':
    config = get_config()
    dconfig = vars(config)
    for k in dconfig:
        print(f"    {k}: {dconfig[k]}")
    config = edict(dconfig)
    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    shutil.copy2(os.path.join('.', 'training_3DMatch.py'), os.path.join(config.snapshot_dir, 'train.py'))
    shutil.copy2(os.path.join('.', 'trainer.py'), os.path.join(config.snapshot_dir, 'trainer.py'))
    shutil.copy2(os.path.join('models', 'architectures.py'), os.path.join(config.snapshot_dir, 'model.py'))  # for the model setting.
    shutil.copy2(os.path.join('models', 'blocks.py'), os.path.join(config.snapshot_dir, 'conv.py'))  # for the conv implementation.
    shutil.copy2(os.path.join('utils', 'loss.py'), os.path.join(config.snapshot_dir, 'loss.py'))
    shutil.copy2(os.path.join('datasets', 'ThreeDMatch.py'), os.path.join(config.snapshot_dir, 'dataset.py'))
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    # create model 
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-1):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('last_unary')
    print("Network Architecture:\n", "".join([layer+'\n' for layer in config.architecture]))

    config.model = KPFCNN(config)
    
    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            # momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    train_set = ThreeDMatchDataset(root=config.root,
                                        split='train',
                                        downsample=config.downsample,
                                        num_node=config.num_node,
                                        augment_noise=config.augment_noise,
                                        augment_axis=config.augment_axis, 
                                        augment_rotation=config.augment_rotation,
                                        augment_translation=config.augment_translation,
                                        config=config,
                                        )
    val_set = ThreeDMatchDataset(root=config.root,
                                    split='val',
                                    num_node=64,
                                    downsample=config.downsample,
                                    augment_noise=config.augment_noise,
                                    augment_axis=config.augment_axis, 
                                    augment_rotation=config.augment_rotation,
                                    augment_translation=config.augment_translation,
                                    config=config,
                                    )
    test_set = ThreeDMatchTestset(root=config.root,
                                    downsample=config.downsample,
                                    config=config,
                                    last_scene=True,
                                    )
    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader,_ = get_dataloader(dataset=val_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        neighborhood_limits=neighborhood_limits
                                        )
    config.test_loader,_ = get_dataloader(dataset=test_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=config.num_workers,
                                        neighborhood_limits=neighborhood_limits
                                        )
    
    # create evaluation
    if config.desc_loss == 'contrastive':
        desc_loss = ContrastiveLoss(
            pos_margin=config.pos_margin,
            neg_margin=config.neg_margin,
            metric='euclidean', 
            safe_radius=config.safe_radius
            )
    else:
        desc_loss = CircleLoss(
            m=config.m,
            log_scale=config.log_scale,
            safe_radius=config.safe_radius
        ) 
    
    config.evaluation_metric = {
        'desc_loss': desc_loss,
        'det_loss': DetLoss(metric='euclidean'),
    }
    config.metric_weight = {
        'desc_loss': config.desc_loss_weight,
        'det_loss': config.det_loss_weight,
    }
    
    trainer = Trainer(config)
    trainer.train()
