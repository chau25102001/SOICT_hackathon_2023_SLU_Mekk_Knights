import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from utils.utils import *
import os
import termcolor
from copy import deepcopy
import math


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'


class BaseTrainer:
    def __init__(self, config):
        self.train_loader = self.get_train_loader(config)
        self.val_loader = self.get_val_loader(config)
        self.config = config
        self.model = self.get_model(config)
        self.model = nn.DataParallel(self.model).to(self.config.device)
        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        self.update_freq = self.config.total_batch_size // self.config.train_batch_size
        self.best_score = -1 if self.config.best_max else 1e8
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                 T_max=len(
                                                                     self.train_loader) * self.config.epochs // self.update_freq,
                                                                 eta_min=1e-7)
        self.current_epoch = 0
        self.ema = ModelEMA(self.model)  # use EMA during training
        count = 0
        for p in self.model.parameters():
            if p.requires_grad:
                count += p.numel()
        print(termcolor.colored(
            f"Training {config.model_card} for {config.epochs} epochs, batch size {config.train_batch_size}, total batch size {config.total_batch_size}, number of params: {count}",
            'blue'))

    def get_model(self, config) -> nn.Module:
        pass

    def get_train_loader(self, config) -> data.DataLoader:
        pass

    def get_val_loader(self, config) -> data.DataLoader:
        pass

    def _train_epoch(self, epoch):
        pass

    def _val_epoch(self, epoch) -> dict:
        pass

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.snapshot_dir, 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema.ema.load_state_dict(checkpoint['ema'])
        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        print("----> load checkpoint")

    def save_checkpoint(self, dir='checkpoint_last.pt'):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.model.module.state_dict(),
                      # 'optimizer': self.optimizer.state_dict(),
                      'ema': deepcopy(self.ema.ema).state_dict(),
                      'epoch': self.current_epoch,
                      'best_score': self.best_score}
        torch.save(checkpoint, checkpoint_path)

    def train(self, resume=False):
        if resume:
            self.load_checkpoint()

        for epoch in range(self.current_epoch, self.config.epochs):
            self._train_epoch(epoch)
            if len(self.val_loader) > 0:
                result = self._val_epoch(epoch)
                if (result['metric'] > self.best_score and self.config.best_max) or (
                        result['metric'] < self.best_score and not self.config.best_max):
                    self.best_score = result['metric']
                    self.save_checkpoint('checkpoint_best.pt')
                    print("---> save new best")
                self.save_checkpoint("checkpoint_last.pt")
            self.save_checkpoint(f"checkpoint_{epoch}.pt")
