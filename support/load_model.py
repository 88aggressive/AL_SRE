#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.optim as optim
import os
import sys
import logging
import yaml

from . import utils
sys.path.insert(0, 'AL_SRE/training')
# from deep_cluster_ori import DCN
from loss import *
# Logger
# patch_logging_stream(logging.INFO)
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ##tdnn
# def get_model(model_dir,egs_conf):
#     logger.info("Load model.")
#     # The dict [info] contains feat_dim and num_targets
#     with open(egs_conf, 'r') as fin:
#         egs_params = yaml.load(fin, Loader=yaml.FullLoader)
#
#     model_type = egs_params["model_type"]
#     assert model_type =='resnet'
#
#     model_blueprint = egs_params["model_blueprint"]
#     logger.info("Create model from model blueprint.")
#     model_blueprint = utils.create_model_dir(model_dir, model_blueprint, stage=-1)
#
#     model_py = utils.create_model_from_py(model_blueprint)
#     ##resnet
#     model_params = {
#         "extracted_embedding": "near",
#         "resnet_params": {
#             "head_conv": True, "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
#             "head_maxpool": False, "head_maxpool_params": {"kernel_size": 3, "stride": 2, "padding": 1},
#             "block": "BasicBlock",
#             "layers": [3, 4, 6, 3],
#             "planes": [32, 64, 128, 256],
#             "convXd": 2,
#             "norm_layer_params": {"momentum": 0.5, "affine": True},
#             "full_pre_activation": False,
#             "zero_init_residual": False},
#
#         "pooling_params": {"num_head": 1500,
#                            "share": True,
#                            "affine_layers": 1,
#                            "hidden_size": 64,
#                            "context": [0],
#                            "stddev": True,
#                            "temperature": True,
#                            "fixed": True
#                            },
#         "fc1": False,
#         "fc1_params": {
#             "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
#             "bn-relu": False,
#             "bn": True,
#             "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
#
#         "fc2_params": {
#             "nonlinearity": '', "nonlinearity_params": {"inplace": True},
#             "bn-relu": False,
#             "bn": True,
#             "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
#     }
#     model = model_py.ResNetXvector(egs_params["feat_dim"], egs_params["num_targets"], **model_params)
#     loss_func = torch.nn.CrossEntropyLoss()
#
#     return model,loss_func  #,optimizer

##tdnn
def get_model(model_dir,egs_conf):
    logger.info("Load model.")
    # The dict [info] contains feat_dim and num_targets
    with open(egs_conf, 'r') as fin:
        egs_params = yaml.load(fin, Loader=yaml.FullLoader)

    model_type = egs_params["model_type"]
    assert model_type in ['tdnn', 'ecapa', 'resnet']

    model_blueprint = egs_params["model_blueprint"]
    logger.info("Create model from model blueprint.")
    model_blueprint = utils.create_model_dir(model_dir, model_blueprint, stage=-1)

    model_py = utils.create_model_from_py(model_blueprint)

    if model_type == "tdnn":
        # tdnn
        model_params = {
            "extracted_embedding": "near",
            "tdnn_layer_params": {"nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
                                  "bn-relu": False,
                                  "bn": True,
                                  "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
            "channels": [80, 512, 512, 512, 512, 512, 512],
            # "num_nodes": 1500,
            # "pooling": "statistics",  # statistics, lde, attentive, multi-head, multi-resolution
            "pooling_params": {"num_nodes": 1500,
                               "num_head": 16,
                               "share": True,
                               "affine_layers": 1,
                               "hidden_size": 64,
                               "context": [0],
                               "temperature": False,
                               "fixed": True
                               },
            "tdnn6": False,
            "tdnn7_params": {"nonlinearity": "default", "bn": True},

        }
        #TDNN Xvector
        model = model_py.Xvector(egs_params["feat_dim"], egs_params["num_targets"], **model_params)
    elif model_type == "ecapa":
        model_params = {
            "extracted_embedding": "near",
            "ecapa_params": {"channels": 1024,
                             "embd_dim": 192,
                             "mfa_conv": 1536,
                             "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}},

            "pooling_params": {
                "hidden_size": 128,
                "time_attention": True,
                "stddev": True,
                "num_head": 4,
                "share": True,
                "affine_layers": 1,
                "context": [0],
                "temperature": True,
                "fixed": True,
            },

            "fc1": False,
            "fc1_params": {
                "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
                "bn-relu": False,
                "bn": True,
                "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

            "fc2_params": {
                "nonlinearity": '', "nonlinearity_params": {"inplace": True},
                "bn-relu": False,
                "bn": True,
                "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

        }
        model = model_py.ECAPA_TDNN(egs_params["feat_dim"], egs_params["num_targets"], **model_params)

    else:
        ##resnet
        model_params = {
            "extracted_embedding": "near",
            "resnet_params": {
                "head_conv": True, "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
                "head_maxpool": False, "head_maxpool_params": {"kernel_size": 3, "stride": 2, "padding": 1},
                "block": "BasicBlock",
                "layers": [3, 4, 6, 3],
                "planes": [32, 64, 128, 256],
                "convXd": 2,
                "norm_layer_params": {"momentum": 0.5, "affine": True},
                "full_pre_activation": False,
                "zero_init_residual": False},

            "pooling_params": {"num_head": 1500,
                               "share": True,
                               "affine_layers": 1,
                               "hidden_size": 64,
                               "context": [0],
                               "stddev": True,
                               "temperature": True,
                               "fixed": True
                               },
            "fc1": False,
            "fc1_params": {
                "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
                "bn-relu": False,
                "bn": True,
                "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

            "fc2_params": {
                "nonlinearity": '', "nonlinearity_params": {"inplace": True},
                "bn-relu": False,
                "bn": True,
                "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
        }
        model = model_py.ResNetXvector(egs_params["feat_dim"], egs_params["num_targets"], **model_params)

    loss_func = torch.nn.CrossEntropyLoss()

    return model,loss_func  #,optimizer



def get_model_ppl(model_dir,egs_conf):
    #只有resnet
    logger.info("Load model.")
    # The dict [info] contains feat_dim and num_targets
    with open(egs_conf, 'r') as fin:
        egs_params = yaml.load(fin, Loader=yaml.FullLoader)

    model_type = egs_params["model_type"]
    assert model_type == 'resnet' ,print("only support resnet")

    model_blueprint = egs_params["model_blueprint"]
    logger.info("Create model from model blueprint.")
    model_blueprint = utils.create_model_dir(model_dir, model_blueprint, stage=-1)

    model_py = utils.create_model_from_py(model_blueprint)

    ##resnet
    model_params = {
        "extracted_embedding": "near",
        "resnet_params": {
            "head_conv": True, "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
            "head_maxpool": False, "head_maxpool_params": {"kernel_size": 3, "stride": 2, "padding": 1},
            "block": "BasicBlock",
            "layers": [3, 4, 6, 3],
            "planes": [32, 64, 128, 256],
            "convXd": 2,
            "norm_layer_params": {"momentum": 0.5, "affine": True},
            "full_pre_activation": False,
            "zero_init_residual": False},

        "pooling_params": {"num_head": 1500,
                           "share": True,
                           "affine_layers": 1,
                           "hidden_size": 64,
                           "context": [0],
                           "stddev": True,
                           "temperature": True,
                           "fixed": True
                           },
        "fc1": False,
        "fc1_params": {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

        "fc2_params": {
            "nonlinearity": '', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
    }
    model = model_py.ResNetXvector(egs_params["feat_dim"], egs_params["num_targets"], **model_params)

    ppl_params = {
        "ppl_lam": 0.001, #0.001
        "deep_cluster_params": {
            "in_dim": 2560,  #resnet2560,80
            "lr": 1e-4,
            "wd": 5e-4,
            "pre_epoch": 10,  # train deepcluster
            "lamda": 1.0,
            "beta": 1.0,
            "hidden_dims": [2000,1024], #500,1024
            "latent_dim": 256, #512 80
            "n_clusters": 30,
        }
    }

    loss_func = PPL_CE(**ppl_params)


    return model,loss_func  #,optimizer

def get_model_ppl2(model_dir,egs_conf):
    #只有resnet
    logger.info("Load model.")
    # The dict [info] contains feat_dim and num_targets
    with open(egs_conf, 'r') as fin:
        egs_params = yaml.load(fin, Loader=yaml.FullLoader)

    model_type = egs_params["model_type"]
    assert model_type == 'resnet' ,print("only support resnet")

    model_blueprint = egs_params["model_blueprint"]
    logger.info("Create model from model blueprint.")
    model_blueprint = utils.create_model_dir(model_dir, model_blueprint, stage=-1)

    model_py = utils.create_model_from_py(model_blueprint)

    ##resnet
    model_params = {
        "extracted_embedding": "near",
        "resnet_params": {
            "head_conv": True, "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
            "head_maxpool": False, "head_maxpool_params": {"kernel_size": 3, "stride": 2, "padding": 1},
            "block": "BasicBlock",
            "layers": [3, 4, 6, 3],
            "planes": [32, 64, 128, 256],
            "convXd": 2,
            "norm_layer_params": {"momentum": 0.5, "affine": True},
            "full_pre_activation": False,
            "zero_init_residual": False},

        "pooling_params": {"num_head": 1500,
                           "share": True,
                           "affine_layers": 1,
                           "hidden_size": 64,
                           "context": [0],
                           "stddev": True,
                           "temperature": True,
                           "fixed": True
                           },
        "fc1": False,
        "fc1_params": {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

        "fc2_params": {
            "nonlinearity": '', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
    }
    model = AL_SRE.model.resnet_xvector_ppl2.ResNetXvector(egs_params["feat_dim"], egs_params["num_targets"], **model_params)

    ppl_params = {
        "ppl_lam": 0.1,
        "deep_cluster_params": {
            "in_dim": 80,  #resnet
            "n_clusters": 30,
        }
    }

    loss_func = PPL_CE2(**ppl_params)


    return model,loss_func  #,optimizer