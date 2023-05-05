# -*- coding:utf-8 -*-

import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, 'AL_SRE/support')
import utils
sys.path.insert(0, 'AL_SRE/training')
from deep_cluster_ori import DCN
#############################################

## Loss ✿
class MarginSoftmaxLoss(torch.nn.Module):
    """Margin softmax loss.
    There are AM, AAM.
    # Copyright xmuspeech (Author: Snowdar 2019-05-29)

    """
    def __init__(self,num_targets,
             m=0.2, s=30., t=1.,
             method="am",
             reduction='mean', eps=1.0e-10,):
        super(MarginSoftmaxLoss, self).__init__()
        self.s = s # scale factor with feature normalization
        self.m = m # margin
        self.t = t # temperature
        self.method = method # am | aam

        self.lambda_factor = 0

        self.eps = eps

        p_target = [0.9, 0.95, 0.99]
        suggested_s = [ (num_targets-1)/num_targets*np.log((num_targets-1)*x/(1-x)) for x in p_target ]

        if self.s < suggested_s[0]:
            print("Warning : using small scalar s={s} could result in bad convergence. \
            There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
            s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)


    def forward(self, inputs, targets):
        """
        @inputs: a 2-dimensional tensor (a batch), including [batch_size, num_class]
        @targets: a 1-dimensional tensor (a batch), including [batch_size]
        """
        assert len(inputs.shape) == 2
        cosine_theta = inputs

        ## Margin Penalty
        cosine_theta_target = cosine_theta.gather(1, targets.unsqueeze(1))

        if self.method == "am":
            penalty_cosine_theta = cosine_theta_target - self.m
        elif self.method == "aam":
            penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
        else:
            raise ValueError("Do not support this {0} margin w.r.t [ am | aam | sm1 | sm2 | sm3 ]".format(self.method))

        penalty_cosine_theta = 1 / (1 + self.lambda_factor) * penalty_cosine_theta + \
                               self.lambda_factor / (1 + self.lambda_factor) * cosine_theta_target

        outputs = self.s * cosine_theta.scatter(1, targets.unsqueeze(1), penalty_cosine_theta)


        return self.loss_function(outputs/self.t, targets)

class PPL_CE(torch.nn.Module):
    def __init__(self,ppl_lam=0.001,deep_cluster_params={},
             reduction='mean', eps=1.0e-10,):
        super(PPL_CE, self).__init__()
        default_deep_cluster_params= {"in_dim": 80,  #
                              "lr": 1e-4,
                              "wd": 5e-4,
                              "pre_epoch": 10,  # train deepcluster
                              "lamda": 1.0,
                              "beta": 1.0,
                              "hidden_dims": [500, 1024],
                              "latent_dim": 512,
                              "n_clusters": 30,
                              }

        self.ppl_lam = ppl_lam
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)
        # loss_ppl = DCN(**deep_cluster_params)
        dcn_params = utils.assign_params_dict(default_deep_cluster_params, deep_cluster_params)
        self.deep_cluster = DCN(**dcn_params)

    def forward(self, posterior,t4, targets):
    # def forward(self, inputs, targets):
    #     posterior, t4 = inputs
        # assert len(t4.shape) == 3
        # assert inputs.shape[2] == 1
        L_ppl = self.deep_cluster.fit(t4)
        L_ce = self.loss_ce(posterior, targets)
        loss = L_ce + L_ppl * self.ppl_lam
        # print("L_ppl,L_ce:",L_ppl,L_ce)

        return loss

class PPL_CE2(torch.nn.Module):
    def __init__(self,ppl_lam=0.001,deep_cluster_params={},
             reduction='mean', eps=1.0e-10,):
        super(PPL_CE2, self).__init__()
        default_deep_cluster_params= {"in_dim": 80,  #
                              "n_clusters": 30,
                              }

        self.ppl_lam = ppl_lam
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)
        # loss_ppl = DCN(**deep_cluster_params)
        dcn_params = utils.assign_params_dict(default_deep_cluster_params, deep_cluster_params)
        self.deep_cluster = DCN_ppl(**dcn_params)

    def forward(self, posterior,t4, targets):
    # def forward(self, inputs, targets):
    #     posterior, t4 = inputs
        # assert len(t4.shape) == 3
        # assert inputs.shape[2] == 1
        L_ppl = self.deep_cluster.fit(t4)
        L_ce = self.loss_ce(posterior, targets)
        loss = L_ce + L_ppl * self.ppl_lam
        # print("L_ppl,L_ce:",L_ppl,L_ce)

        return loss