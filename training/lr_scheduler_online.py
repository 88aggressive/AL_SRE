# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import sys
import os

sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")
sys.path.insert(0, "AL_SRE/training")
from .optim import *

## Wrapper ✿
# 三种学习率设置  cyclic warmR reduceP

class LRSchedulerWrapper():

    def __init__(self, optimizer, params: dict = {}):
        # Suggested weight_decay: 1e-4 for l2 regularization (sgd, adam) and
        #                         1e-1 for decouped weight decay (sgdw, adamw, radam, ralamb, adamod etc.)
        default_params = {
            "name": "warmR",

            "cyclic.max_lr": 1e-3,
            "cyclic.base_lr": 1e-8,
            "cyclic.step_size_up": 2e4,
            "cyclic.step_size_down": None,
            "cyclic.mode": 'triangular2',
            "cyclic.gamma": 1.0,
            "cyclic.scale_fn": None,
            "cyclic.scale_mode": 'cycle',
            "cyclic.cycle_momentum": False,
            "cyclic.base_momentum": 0.8,
            "cyclic.max_momentum": 0.9,

            "warmR.T_max": 2000,
            "warmR.T_mult": 1,
            "warmR.factor": 1.0,
            "warmR.eta_min": 4e-8,
            "warmR.log_decay": False,
            "warmR.lr_decay_step": 1,

            "reduceP.metric": 'valid_acc',
            "reduceP.check_interval": 0,
            "reduceP.factor": 0.5,
            "reduceP.patience": 10,
            "reduceP.threshold": 0.0001,
            "reduceP.cooldown": 0,
            "reduceP.min_lr": 0.
        }

        used_params = utils.assign_params_dict(default_params, params, force_check=False, support_unknow=True)
        split_params = utils.split_params(used_params)

        if isinstance(optimizer, Lookahead):
            base_optimizer = optimizer.optimizer
        else:
            base_optimizer = optimizer

        self.name = split_params["public"]["name"]
        if self.name == "cyclic":
            base_lr = split_params["cyclic"].pop("base_lr")
            max_lr = split_params["cyclic"].pop("max_lr")
            step_up = float(split_params["cyclic"]["step_size_up"])
            step_down = float(split_params["cyclic"]["step_size_down"]) if split_params["cyclic"][
                                                                               "step_size_down"] is not None else step_up
            self.step_total = int(step_up + step_down)
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(base_optimizer, base_lr, max_lr,
                                                                  **split_params["cyclic"])
        elif self.name == "warmR":
            self.T_0 = split_params["warmR"].pop("T_max")
            self.T_mult = split_params["warmR"]["T_mult"]
            self.lr_decay_step = split_params["warmR"].pop("lr_decay_step")
            self.lr_scheduler = CosineAnnealingWarmRestarts(base_optimizer, self.T_0, **split_params["warmR"])
        elif self.name == "reduceP":
            self.check_interval = split_params["reduceP"].pop("check_interval")
            self.metric = split_params["reduceP"].pop("metric")
            self.min_lr = split_params["reduceP"]["min_lr"]
            if self.metric == "valid_acc":
                mode = "max"
            elif self.metric == "valid_loss":
                mode = "min"
            else:
                raise ValueError("Do not support {} metric for ReduceLROnPlateau strategy.".format(self.metric))
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, mode=mode,
                                                                     **split_params["reduceP"])
            self.init = False

        else:
            raise ValueError("Do not support {0} lr_scheduler now.".format(self.name))

    def is_reduce_point(self, training_point):
        if self.name == "reduceP":
            # It will check the point with a global num_iter value.
            return (self.check_interval > 0 and training_point[2] % self.check_interval == 0) or \
                   (self.check_interval <= 0 and training_point[1] == 1)
        else:
            return False

    # check the save point on the end of cycle
    def is_cycle_point(self, training_point):
        if self.name == "warmR":
            epoch = training_point[0] - 1 if self.lr_decay_step == 0 else training_point[2]
            if self.T_mult == 1:
                if epoch % self.T_0 == 0 and epoch > 0:
                    return False if (self.lr_decay_step == 0 and training_point[1] > 1) else True

            else:
                if math.log(max(0.05, (epoch / self.T_0 * (self.T_mult - 1) + 1)), self.T_mult) % 1 == 0 and epoch > 0:
                    return False if (self.lr_decay_step == 0 and training_point[1] > 1) else True
        if self.name == "cyclic":
            return True if training_point[2] % self.step_total == 0 and training_point[2] > 0 else False

        return False

    def step(self, training_point=None, valid_metric=None):
        if self.name == "warmR":
            if self.lr_decay_step > 0 and training_point[2] % self.lr_decay_step == 0:
                # It will check the point at the start of every epoch (not a global decay-strategy).
                self.lr_scheduler.step(training_point[2])
            elif self.lr_decay_step == 0:
                self.lr_scheduler.step(training_point[0] - 1)
        elif self.name == "cyclic":
            self.lr_scheduler.step()
        elif self.name == "reduceP":
            # Sample a point in which the metrics of valid are computed and adjust learning rate at this point.
            if self.is_reduce_point(training_point):
                metric = valid_metric[0] if self.metric == "valid_loss" else valid_metric[1]
                self.lr_scheduler.step(metric)


## Learn rate scheduler ✿
class CosineAnnealingWarmRestarts(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Base lr decay has been added. [Snowdar 2019-08-29]
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, factor=1.0, log_decay=False, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult <= 0:  # or not isinstance(T_mult, int):
            raise ValueError("Expected T_mult > 0, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.factor = factor
        self.this_factor = 1
        self.T_cur = last_epoch
        self.log_decay = log_decay
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.log_decay:
            eta_min = np.log10(self.eta_min)
            return [10 ** (eta_min + (np.log10(base_lr * self.this_factor) - eta_min) *
                           (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr * self.this_factor - self.eta_min) *
                    (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.this_factor = self.factor ** (epoch // self.T_0)
                else:
                    n = int(math.log(max(0.05, (epoch / self.T_0 * (self.T_mult - 1) + 1)), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    self.this_factor = self.factor ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
