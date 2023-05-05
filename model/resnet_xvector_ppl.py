#!/usr/bin/env Python
# coding=utf-8
# -*- coding:utf-8 -*-

import os
import sys
import torch
import torch.nn.functional as F
import math
sys.path.insert(0, 'AL_SRE/model')

from nnet import *
sys.path.insert(0, 'AL_SRE/training')
from deep_cluster_ori import DCN

class ResNetXvector(torch.nn.Module):
    """ A resnet x-vector framework """
    def __init__(self,inputs_dim, num_targets,
             resnet_params={},deep_cluster_params={},
             fc1=False, fc1_params={}, fc2_params={},
             pooling_params={},extracted_embedding="far"):
        super(ResNetXvector, self).__init__()

        ## Params.
        default_resnet_params = {
            "head_conv":True, "head_conv_params":{"kernel_size":3, "stride":1, "padding":1},
            "head_maxpool":False, "head_maxpool_params":{"kernel_size":3, "stride":1, "padding":1},
            "block":"BasicBlock",
            "layers":[3, 4, 6, 3],
            "planes":[32, 64, 128, 256], # a.k.a channels.
            "use_se": False,
            "se_ratio": 4,
            "convXd":2,
            "norm_layer_params":{"momentum":0.5, "affine":True},
            "full_pre_activation":True,
            "zero_init_residual":False
            }

        default_pooling_params = {
            "num_head":1,
            "hidden_size":64,
            "share":True,
            "affine_layers":1,
            "context":[0],
            "stddev":True,
            "temperature":False, 
            "fixed":True
        }
        
        default_fc_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False, 
            "bn":True, 
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
            }

        default_deep_cluster_params= {"in_dim": 80,  #
                              "lr": 1e-4,
                              "wd": 5e-4,
                              "pre_epoch": 10,  # train deepcluster
                              "lamda": 1.0,
                              "beta": 1.0,
                              "hidden_dims": [500, 500, 2000],
                              "latent_dim": 512,
                              "n_clusters": 30,
                              }

        dcn_params = utils.assign_params_dict(default_deep_cluster_params, deep_cluster_params)
        resnet_params = utils.assign_params_dict(default_resnet_params, resnet_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)

        ## Var.
        self.extracted_embedding = extracted_embedding # only near here.
        self.convXd = resnet_params["convXd"]
        
        ## Nnet.

        # [batch, 1, feats-dim, frames] for 2d and  [batch, feats-dim, frames] for 1d.
        # Should keep the channel/plane is always in 1-dim of tensor (index-0 based).
        inplanes = 1 if self.convXd == 2 else inputs_dim

        self.resnet = ResNet(inplanes, **resnet_params)

        # It is just equal to Ceil function.
        resnet_output_dim = (inputs_dim + self.resnet.get_downsample_multiple() - 1) // self.resnet.get_downsample_multiple() \
                            * self.resnet.get_output_planes() if self.convXd == 2 else self.resnet.get_output_planes()

        # Pooling
        stddev = pooling_params.pop("stddev")
        self.stats = StatisticsPooling(resnet_output_dim, stddev=stddev)

        self.fc1 = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), resnet_params["planes"][3], **fc1_params) if fc1 else None

        if fc1:
            fc2_in_dim = resnet_params["planes"][3]
        else:
            fc2_in_dim = self.stats.get_output_dim()

        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, resnet_params["planes"][3], **fc2_params)

        self.embd_dim=resnet_params["planes"][3]

        self.out_linear = torch.nn.Linear(resnet_params["planes"][3],num_targets)
        dcn_params['in_dim'] = resnet_output_dim
        # self.deep_cluster = DCN(**dcn_params)
        # self.ppl_loss = None

    def forward(self, x):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        # print("x:",x.shape)
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        t4 = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x
        # t4 = x
        x = self.stats(t4)
        x = self.auto(self.fc1, x)
        x = self.fc2(x)
        x = x.squeeze(2)
        x = self.out_linear(x)
        return x,t4

    def forward_loss(self, x):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        # print("x0:",x)
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x
        # t4 = x
        self.ppl_loss = self.deep_cluster.fit(x)
        x = self.stats(x)
        # print("x3:", x)
        x = self.auto(self.fc1, x)
        # print("x4:", x)
        x = self.fc2(x)
        # print("x5:", x)
        x = x.squeeze(2)
        # print("x6:", x)
        x = self.out_linear(x)
        # print("x7:", x)
        return x

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, x):
        """
        x: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) if self.convXd == 2 else x
        # print("s:",x.shape) #[1,2560,219]
        x = self.stats(x)

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))
        # xvector = xvector.squeeze(2)
        return xvector

    def get_loss(self):
        return self.ppl_loss


    def embedding_dim(self) -> int:
        """ Export interface for c++ call, return embedding dim of the model
        """
        return self.embd_dim

    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not
        """
        return layer(x) if layer is not None else x

    def step(self, epoch, this_iter, epoch_batchs):
        pass

    def backward_step(self, epoch, this_iter, epoch_batchs):
        pass

