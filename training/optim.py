# -*- coding:utf-8 -*-

import logging
import sys
import types
import math
import itertools as it
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")
sys.path.insert(0, "AL_SRE/training")

import utils

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


## Wrapper ✿
def get_optimizer(model, params: dict = {}):
    # Suggested weight_decay: 1e-4 for l2 regularization (sgd, adam) and
    #                         1e-1 for decouped weight decay (sgdw, adamw, radam, ralamb, adamod etc.)
    default_params = {
        "name": "adamW",
        "learn_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.999,
        "weight_decay": 1e-4,
        "nesterov": False,
        "lookahead.k": 5,
        "lookahead.alpha": 0.,
        "gc": False
    }

    used_params = utils.assign_params_dict(default_params, params)

    # Base params
    name = used_params["name"]
    learn_rate = used_params["learn_rate"]
    beta1 = used_params["beta1"]
    beta2 = used_params["beta2"]
    beta3 = used_params["beta3"]
    weight_decay = used_params["weight_decay"]
    gc = used_params["gc"]
    nesterov = used_params['nesterov']
    extra_params = {}

    # Select optimizer   sgd  adam  adamW
    if name == "sgd":
        base_optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=beta1, weight_decay=weight_decay,
                                   nesterov=nesterov)
    elif name == "adam":
        base_optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "adamW":
        base_optimizer = AdamW(model.parameters(), lr=learn_rate, betas=(beta1, beta2), weight_decay=weight_decay,
                               **extra_params)
    else:
        raise ValueError("Do not support {0} optimizer now.".format(name))

    # Using alpha to decide whether to use lookahead
    if used_params["lookahead.alpha"] > 0:
        logger.info("Use lookahead optimizer with alpha={} and k={}".format(used_params["lookahead.alpha"],
                                                                            used_params["lookahead.k"]))
        optimizer = Lookahead(base_optimizer, k=used_params["lookahead.k"], alpha=used_params["lookahead.alpha"])
    else:
        optimizer = base_optimizer

    return optimizer


## Optim-wrapper ✿
class Lookahead(Optimizer):
    '''
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py

    '''

    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        '''
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        '''
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss


## Optimizer ✿
class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, gc=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gc = gc
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)  # , memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # , memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)  # , memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if self.gc:
                    # For linear layer Y=WX+b, the tensor shape of weight is (outplanes, inplanes),
                    # but for CNN layer(1d and 2d etc.), the tensor shape of weight is (outplanes, inplanes, [cnn-core]).
                    # And here the gc is used in both linear and CNN layer.
                    # It is not influenced by weight decay for weight decay directly changes the p.data rather than p.grad.
                    # But when using gc in adam, the order question should be considered for L2 regularization changes
                    # the p.grad.
                    if len(list(grad.size())) >= 2:
                        grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


