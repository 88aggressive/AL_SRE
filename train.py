# -*- coding:utf-8 -*-

import sys, os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, os.getcwd())
os.getcwd()

from support.load_model import *
# import support.egs_online_csv_vad as egs
import dataio.egs as egs

# import training.trainer_online_nolr as trainer
import training.trainer_online_tqdm as trainer
from training.loss import *
import training.optim as optim
import training.lr_scheduler_online as learn_rate_scheduler


# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
        description="""Train xvector framework with pytorch.""",
        formatter_class=argparse.RawTextHelpFormatter,
        conflict_handler='resolve')

parser.add_argument("--train_data", type=str, default="exp/egs/test",
                    help="save file.")

parser.add_argument("--save_dir", type=str, default="exp/test",
                    help="save file.")

parser.add_argument("--epochs", type=int, default=30,
                    help="get model.")

parser.add_argument("--start_epoch", type=int, default=0,
                    help="start_epoch.")

parser.add_argument("--conf_file", type=str, default="AL_SRE/config/conf_ecapa.yaml",
                    help="conf file")

parser.add_argument("--gpu_id", type=str, default="",
                    help="set use gpu-id.")


args = parser.parse_args()

egs_conf=args.conf_file

##--------------------------------------------------##

##--------------------------------------------------##
## Training options
optimizer_params = {
    "name":"sgd", # sgd | adam | adamW
    "learn_rate":0.01,  ##adamW=0.001,sgd=0.01
    "beta1":0.9,
    "beta2":0.999,
    "beta3":0.999,
    "weight_decay":3e-4,  # Should be large for decouped weight decay (adamW) and small for L2 regularization (sgd, adam).
    "lookahead.k":5,
    "lookahead.alpha":0.,  # 0 means not using lookahead and if used, suggest to set it as 0.5.
    "gc":False # If true, use gradient centralization.
}


lr_scheduler_params = {
    "name":"reduceP", # warmR or reduceP or cyclic

    "warmR.lr_decay_step":0, # 0 means decay after every epoch and 1 means every iter.
    "warmR.T_max":3,
    "warmR.T_mult":2,
    "warmR.factor":1.0,  # The max_lr_decay_factor.
    "warmR.eta_min":1e-6,
    "warmR.log_decay":False,

    "reduceP.metric":'valid_loss',
    "reduceP.check_interval":4000, # 0 means check metric after every epoch and 1 means every iter.
    "reduceP.factor":0.1,  # scale of lr in every times.
    "reduceP.patience":2,
    "reduceP.threshold":0.0001,
    "reduceP.cooldown":0,
    "reduceP.min_lr":1e-8,

    "cyclic.max_lr": 1e-3,
    "cyclic.base_lr": 1e-8,
    "cyclic.step_size_up": 24000,
    "cyclic.mode": 'triangular2',
}

##--------------------------------------------------##
## Main params
train_data = args.train_data
gpu_id = args.gpu_id
epochs = args.epochs
save_dir = args.save_dir
valid_iter = 100
suffix = "params"

start_epoch = args.start_epoch

##--------------------------------------------------##
##
#### Set seed
utils.set_all_seed(1024)

#### Train model
print("Get model_blueprint from model directory.")

print("Load egs to bunch.")
# The dict [info] contains feat_dim and num_targets
with open(egs_conf,'r') as fin:
    egs_params = yaml.load(fin, Loader=yaml.FullLoader)
data = egs.BaseBunch.get_bunch_from_egsdir(train_data, egs_params)

print("Create model from model blueprint.")
model,loss_func = get_model(save_dir,egs_conf)

print("Define optimizer and lr_scheduler.")
optimizer = optim.get_optimizer(model, optimizer_params)
lr_scheduler = learn_rate_scheduler.LRSchedulerWrapper(optimizer, lr_scheduler_params)

print("Init a simple trainer.")
## Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
package = ({"data":data, "model":model,"loss_func":loss_func, "optimizer":optimizer,"lr_scheduler":lr_scheduler},
        {"model_dir":save_dir, "epochs":epochs, "gpu_id":gpu_id,"compute_accuracy":True,"start_epoch":start_epoch,
        "suffix":suffix,"report_interval_iters":valid_iter,"use_tensorboard":True})

trainer = trainer.SimpleTrainer(package)

trainer.run()