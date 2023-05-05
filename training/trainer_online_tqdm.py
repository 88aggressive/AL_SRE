# -*- coding:utf-8 -*-

import os
import sys
import re
import logging
import copy
import math
import shutil
import time
import traceback
import progressbar
import pandas as pd
import numpy as np
import yaml
import torch
from tqdm import tqdm

sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")
sys.path.insert(0, "AL_SRE/training")

import utils

from .reporter import Reporter_new as Reporter
from .lr_scheduler_online import LRSchedulerWrapper
# torch.multiprocessing.set_start_method('spawn')
# Logger
logger = logging.getLogger('libs')
logger.addHandler(logging.NullHandler())



class SimpleTrainer():
    def __init__(self, package):
        default_elements = {"data": None, "model": None, "loss_func":None,
                            "optimizer": None,"lr_scheduler":None}
        default_params = {"model_dir": "", "epochs": 10,"gpu_id": "","suffix": "params",
                          "compute_accuracy":True,"start_epoch":0,
                          "report_interval_iters":100,"use_tensorboard":True}

        elements, params = package
        self.elements = utils.assign_params_dict(default_elements, elements)
        self.params = utils.assign_params_dict(default_params, params, support_unknow=True)

        assert self.elements["data"] is not None
        assert self.elements["model"] is not None
        assert self.elements["optimizer"] is not None

        assert self.params["model_dir"] != ""
        # assert self.params["model_blueprint"] != ""

        self.elements["model_forward"] = self.elements["model"]

        gpu_id = self.params["gpu_id"]
        self.epochs = self.params["epochs"]
        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

        # self.loss_func = self.elements["loss_func"]

        self.start_write_log = False
        self.record_value = []
        self.valid_loss = 10.0
        self.valid_acc = 0.0
        self.record_file = "{0}/log/{1}".format(self.params["model_dir"], "train.csv")

        # (epoch, iter in epoch, global step)
        self.training_point = copy.deepcopy([self.params["start_epoch"], 0, 0])
        self.cycle_point = 0  # for cycle training.
        self.batch_size = 0
        self.num_sample = 1
        self.print_sample = False
        self.num_batch = 0

        self.trans_dict = []

    def init_training(self):
        model = self.elements["model"]
        start_epoch = self.params["start_epoch"]
        model_dir = self.params["model_dir"]
        # model_blueprint = self.params["model_blueprint"]
        suffix = self.params["suffix"]

        # Recover checkpoint | Tansform learning | Initialize parametes
        if start_epoch > 0:
            # This train_stage is equal to number of completed epoch
            logger.info("Recover training from {0} epoch.".format(start_epoch))
            model.load_state_dict(torch.load('{0}/{1}.{2}'.format(model_dir, start_epoch, suffix),
                                             map_location="cpu"))

            info_log_path = '{0}/{1}/{2}.{3}'.format(
                model_dir, "checkpoint_info", start_epoch, "yaml")
            if os.path.exists(info_log_path):

                with open(info_log_path, 'r') as fin:
                    info = yaml.load(fin, Loader=yaml.FullLoader)
                self.training_point[2] = info['step']
                self.start_write_log = True
        elif os.path.exists(self.record_file):
            # Do backup to avoid clearing the loss log when re-running a same launcher.
            bk_file = "{0}.bk.{1}".format(self.record_file,
                                          time.strftime('%Y_%m_%d.%H_%M_%S', time.localtime(time.time())))
            shutil.move(self.record_file, bk_file)

        torch.backends.cudnn.benchmark = True
        model = model.to(self.device)

    def save_model(self, mod="epoch",train_lr=None,valid_loss=None):
        assert mod in ["epoch", "iter", "cycle"]
        if mod == "epoch":
            model_name = self.training_point[0]
        elif mod == "iter":
            model_name = "{}.{}".format(self.training_point[0], self.training_point[1])
        else:
            model_name = "{}_cycle".format(self.cycle_point)
        model_path = '{0}/{1}.{2}'.format(self.params["model_dir"], model_name, self.params["suffix"])

        info_log = {
            'train_lr': train_lr if train_lr else "see train.csv",
            "next_lr": self.elements["optimizer"].state_dict()['param_groups'][0]['lr'],
            'epoch': self.training_point[0],
            'iter in epoch': self.training_point[1],
            'step': self.training_point[2],
            'valid_loss':valid_loss if valid_loss else "see train.csv"
        }

        info_log_path = '{0}/{1}/{2}.{3}'.format(self.params["model_dir"], "checkpoint_info", model_name, "yaml")
        logger.info("Save model to {0}. \n epoch/iter: {1}/{2}.  cur_step: {3}".format(model_path, self.training_point[0],
                                                                                       self.training_point[1], self.training_point[2]))
        torch.save(self.elements["model"].state_dict(), model_path)
        # torch.save(info, info_path)
        with open(info_log_path, 'w') as fout:
            data = yaml.dump(info_log)
            fout.write(data)


    def train_one_batch(self, batch):
        """A normal training core without fetching data from iterator.
        """
        model = self.elements["model"]
        optimizer = self.elements["optimizer"]
        loss_func = self.elements["loss_func"]

        model.train()

        inputs, targets = batch
        # print("in,out:",inputs.shape,type(targets))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        optimizer.zero_grad()

        posterior = model(inputs)

        loss = loss_func(posterior, targets)

        loss.backward()

        loss = loss.item()

        accuracy = self.get_accuracy(posterior.detach(), targets.detach())

        optimizer.step()

        self.training_point[2] += 1 # update step
        if self.training_point[1] % 100==0:
            self.step_lr(loss, accuracy, optimizer, self.elements["lr_scheduler"])
        self.num_batch += 1
        return loss, accuracy

    def compute_validation(self, data_loader):
        """A normal evaluation core.
        """
        model = self.elements["model"]
        loss_func = self.elements["loss_func"]

        model.eval()

        loss = 0.
        accuracy = 0.
        num_samples = 0
        with torch.no_grad():
            for idx,this_data in enumerate(data_loader):
                inputs, targets = this_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                num_utts = targets.size(0)
                if num_utts == 0:
                    continue
                posterior = model(inputs)
                loss += loss_func(posterior, targets).item() * len(targets)
                # loss = loss.item()

                accuracy += self.get_accuracy(posterior.detach(), targets.detach())* len(targets)
                num_samples += len(targets)

            avg_loss = loss / num_samples
            avg_accuracy = accuracy / num_samples

        model.train()
        self.valid_loss = avg_loss
        self.valid_acc = avg_accuracy
        return avg_loss, avg_accuracy

    def step_lr(self,train_loss,train_acc,base_optimizer,lr_scheduler):

        valid_dataloader=self.elements["data"].valid_loader

        lr_scheduler_params = {
            "training_point": self.training_point}
        valid_loss = None
        valid_computed = False
        if lr_scheduler.name == "reduceP" and lr_scheduler.is_reduce_point(self.training_point):
            assert valid_dataloader is not None
            valid_loss, valid_acc = self.compute_validation(valid_dataloader)
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
            valid_computed = True
        if valid_computed or (valid_dataloader is not None ):
            if not valid_computed:
                valid_loss, valid_acc = self.compute_validation(valid_dataloader)

                valid_computed = False
            # real_snapshot is set for tensorboard to avoid workspace problem
            real_snapshot = {"train_loss": train_loss, "valid_loss": valid_loss,
                            "train_acc": train_acc*100, "valid_acc": valid_acc*100}
            snapshot = {"train_loss": "{0:.6f}".format(train_loss), "valid_loss": "{0:.6f}".format(valid_loss),
                        "train_acc": "{0:.2f}".format(train_acc*100), "valid_acc": "{0:.2f}".format(valid_acc*100),
                        "num_sample":self.num_sample,"real": real_snapshot}
            # For ReduceLROnPlateau.
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
        else:
            real_snapshot = {
                "train_loss": train_loss, "train_acc": train_acc*100}
            snapshot = {"train_loss": "{0:.6f}".format(train_loss), "valid_loss": "",
                        "train_acc": "{0:.2f}".format(train_acc*100), "valid_acc": "",
                        "num_sample":self.num_sample,"real": real_snapshot}
        training_point = (self.training_point[0],self.training_point[1],self.training_point[2])
        self.train_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
        # self.reporter.update(snapshot,training_point,self.train_lr)
        print("\ntrain_loss={0:.4f},valid_loss={1:.4f},train_acc={2:.4f},valid_acc={3:.4f}".format(train_loss,
                                                                                                   valid_loss,
                                                                                                   train_acc * 100,
                                                                                                   valid_acc * 100))
        if lr_scheduler is not None:
            # It is not convenient to wrap lr_scheduler (doing).
            if isinstance(lr_scheduler, LRSchedulerWrapper):
                lr_scheduler.step(**lr_scheduler_params)
                current_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
                if lr_scheduler.name == "reduceP":
                    if current_lr < self.last_lr:
                        self.last_lr = current_lr
                        self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)
                    elif current_lr <= lr_scheduler.min_lr and lr_scheduler.is_reduce_point(self.training_point):
                        self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)

                if lr_scheduler.is_cycle_point(self.training_point):
                    self.cycle_point+=1
                    self.save_model(mod="cycle",train_lr=self.train_lr,valid_loss=valid_loss)
            else:
                # For some pytorch lr_schedulers, but it is not available for all.
                lr_scheduler.step(self.training_point[0])
        info_dict = {"train_loss": "{0:.6f}".format(train_loss),
                     "valid_loss": "{0:.6f}".format(valid_loss),
                     "train_acc": "{0:.2f}".format(train_acc * 100),
                     "valid_acc": "{0:.2f}".format(valid_acc * 100),
                     }
        self.record_value.append(info_dict)
        dataframe = pd.DataFrame(self.record_value)
        dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
        self.record_value.clear()

    def get_accuracy(self,posterior, targets):
        prediction = torch.argmax(posterior, dim=1)
        num_correct = (targets == prediction).sum()
        accuracy = num_correct.item() / len(targets)
        accuracy = accuracy

        return accuracy


    def run(self):
        """Main function to start a training process.
        """
        try:
            self.init_training()
            # self.reporter = Reporter(self)
            start_epoch = self.params["start_epoch"]
            epochs = self.params["epochs"]
            data = self.elements["data"]

            # See init_training.
            self.train_lr = self.elements["optimizer"].state_dict()['param_groups'][0]['lr']
            self.last_lr =  self.elements["optimizer"].state_dict()['param_groups'][0]['lr']

            logger.info("Training will run for {0} epochs.".format(epochs))

            for this_epoch in range(start_epoch,epochs):
                self.training_point[0] += 1

                progress = tqdm(data.train_loader, desc="train", unit=" step")
                for batch in progress:
                    progress.set_description("epochs {0}/{1}".format(this_epoch+1, epochs))
                    if self.training_point[1] == 0:
                        self.num_sample = len(data.train_loader)

                        self.num_sample = len(data.train_loader)
                    self.training_point[1] += 1
                    loss, acc = self.train_one_batch(batch)

                    train_acc = acc * 100

                    train_loss = loss

                    progress.set_postfix(
                        loss=f"{train_loss:.4f}",
                        acc=f"{train_acc:.4f}",
                        step=self.training_point[1],
                    )
                    progress.update()
                    info_dict = {"train_loss": "{0:.6f}".format(train_loss),
                                "valid_loss": "{0:.6f}".format(self.valid_loss),
                                "train_acc": "{0:.2f}".format(train_acc),
                                "valid_acc": "{0:.2f}".format(self.valid_acc * 100),
                                }
                    self.record_value.append(info_dict)
                    dataframe = pd.DataFrame(self.record_value)
                    if self.start_write_log:
                        dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
                    else:
                        dataframe.to_csv(self.record_file, header=True, index=False)
                        self.start_write_log = True
                    self.record_value.clear()
                progress.close()


                self.save_model()
                self.training_point[1] = 0

            final_model_name = "{}_cycle".format(self.cycle_point) if self.cycle_point else epochs
            final_model_path = os.path.join(self.params["model_dir"], 'final.params')
            if os.path.exists(final_model_path) or os.path.islink(final_model_path):
                os.remove(final_model_path)

            os.symlink('{0}/{1}.{2}'.format(self.params["model_dir"], final_model_name, self.params["suffix"]),
                       final_model_path)

        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):traceback.print_exc()
            sys.exit(1)

class SimpleTrainer_ppl():
    def __init__(self, package):
        default_elements = {"data": None, "model": None, "loss_func":None,
                            "optimizer": None,"lr_scheduler":None}
        default_params = {"model_dir": "", "epochs": 10,"gpu_id": "","suffix": "params",
                          "compute_accuracy":True,"start_epoch":0,
                          "report_interval_iters":100,"use_tensorboard":True}

        elements, params = package
        self.elements = utils.assign_params_dict(default_elements, elements)
        self.params = utils.assign_params_dict(default_params, params, support_unknow=True)

        assert self.elements["data"] is not None
        assert self.elements["model"] is not None
        assert self.elements["optimizer"] is not None

        assert self.params["model_dir"] != ""
        # assert self.params["model_blueprint"] != ""

        self.elements["model_forward"] = self.elements["model"]

        gpu_id = self.params["gpu_id"]
        self.epochs = self.params["epochs"]
        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

        self.loss_func = self.elements["loss_func"]

        self.start_write_log = False
        self.record_value = []
        self.valid_loss = 10.0
        self.valid_acc = 0.0

        self.record_file = "{0}/log/{1}".format(self.params["model_dir"], "train.csv")

        # (epoch, iter in epoch, global step)
        self.training_point = copy.deepcopy([self.params["start_epoch"], 0, 0])
        self.cycle_point = 0  # for cycle training.
        self.batch_size = 0
        self.num_sample = 1
        self.print_sample = False
        self.num_batch = 0

        self.trans_dict = []

    def init_training(self):
        model = self.elements["model"]
        start_epoch = self.params["start_epoch"]
        model_dir = self.params["model_dir"]
        # model_blueprint = self.params["model_blueprint"]
        suffix = self.params["suffix"]

        # Recover checkpoint | Tansform learning | Initialize parametes
        if start_epoch > 0:
            # This train_stage is equal to number of completed epoch
            print("Recover training from {0} epoch.".format(start_epoch))
            model.load_state_dict(torch.load('{0}/{1}.{2}'.format(model_dir, start_epoch, suffix),
                                             map_location="cpu"))


            info_log_path = '{0}/{1}/{2}.{3}'.format(
                model_dir, "checkpoint_info", start_epoch, "yaml")
            if os.path.exists(info_log_path):
                # info = torch.load(info_log_path)
                # self.elements["optimizer"].load_state_dict(info['optimizer'])
                # for state in self.elements["optimizer"].values():
                #     for k, v in state.items():
                #         if isinstance(v, torch.Tensor):
                #             state[k] = v.cuda()
                with open(info_log_path, 'r') as fin:
                    info = yaml.load(fin, Loader=yaml.FullLoader)
                self.training_point[2] = info['step']
            self.start_write_log = True
        elif os.path.exists(self.record_file):
            # Do backup to avoid clearing the loss log when re-running a same launcher.
            bk_file = "{0}.bk.{1}".format(self.record_file,
                                          time.strftime('%Y_%m_%d.%H_%M_%S', time.localtime(time.time())))
            shutil.move(self.record_file, bk_file)

        torch.backends.cudnn.benchmark = True
        model.to(self.device)

        self.elements["loss_func"].to(self.device)
    def save_model(self, mod="epoch",train_lr=None,valid_loss=None):
        assert mod in ["epoch", "iter", "cycle"]
        if mod == "epoch":
            model_name = self.training_point[0]
        elif mod == "iter":
            model_name = "{}.{}".format(self.training_point[0], self.training_point[1])
        else:
            model_name = "{}_cycle".format(self.cycle_point)
        model_path = '{0}/{1}.{2}'.format(self.params["model_dir"], model_name, self.params["suffix"])

        info_log = {
            'train_lr': train_lr if train_lr else "see train.csv",
            "next_lr": self.elements["optimizer"].state_dict()['param_groups'][0]['lr'],
            'epoch': self.training_point[0],
            'iter in epoch': self.training_point[1],
            'step': self.training_point[2],
            'valid_loss':valid_loss if valid_loss else "see train.csv"
        }

        info_log_path = '{0}/{1}/{2}.{3}'.format(self.params["model_dir"], "checkpoint_info", model_name, "yaml")
        print("Save model to {0}. \n epoch/iter: {1}/{2}.  cur_step: {3}".format(model_path, self.training_point[0],
                                                                                       self.training_point[1], self.training_point[2]))
        torch.save(self.elements["model"].state_dict(), model_path)
        # torch.save(info, info_path)
        with open(info_log_path, 'w') as fout:
            data = yaml.dump(info_log)
            fout.write(data)


    def train_one_batch(self, batch):
        """A normal training core without fetching data from iterator.
        """
        model = self.elements["model"]
        optimizer = self.elements["optimizer"]
        # lr_scheduler = self.elements['lr_scheduler']
        loss_func = self.elements["loss_func"]#.to(self.device)

        # if not model.training:
        model.train()

        inputs, targets = batch
        # print("in,out:",inputs.shape,type(targets))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        optimizer.zero_grad()
        posterior, t4 = model(inputs)
        loss = loss_func(posterior, t4, targets)
        # print("posterior, t4:", posterior, t4)
        # print("loss:",loss)

        loss.backward()

        loss = loss.item()

        accuracy = self.get_accuracy(posterior.detach(), targets.detach())

        optimizer.step()

        self.training_point[2] += 1 # update step

        if self.training_point[1] % 100==0:
            self.step_lr(loss, accuracy, optimizer, self.elements["lr_scheduler"])
        self.num_batch += 1
        return loss, accuracy

    def compute_validation(self, data_loader):
        """A normal evaluation core.
        """
        model = self.elements["model"]
        loss_func = self.elements["loss_func"].to(self.device)

        # train_status = model.training  # Record status.
        model.eval()

        # progress2 = tqdm(data_loader, desc="valid", unit=" step")
        loss = 0.
        accuracy = 0.
        num_samples = 0
        with torch.no_grad():
            for idx,this_data in enumerate(data_loader):
                inputs, targets = this_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # loss_func_ppl = loss_func_ppl.to(self.device)
                num_utts = targets.size(0)
                if num_utts == 0:
                    continue
                # posterior = model(inputs)

                posterior, t4 = model(inputs)
                loss += loss_func(posterior, t4, targets).item() * len(targets)
                accuracy += self.get_accuracy(posterior.detach(), targets.detach())* len(targets)
                num_samples += len(targets)

            avg_loss = loss / num_samples
            avg_accuracy = accuracy / num_samples
            # logger.info("valid: loss={0},acc={1}".format(avg_loss, avg_accuracy))
        # if train_status:
        model.train()
        self.valid_loss = avg_loss
        self.valid_acc = avg_accuracy

        return avg_loss, avg_accuracy


    def step_lr(self,train_loss,train_acc,base_optimizer,lr_scheduler):

        valid_dataloader=self.elements["data"].valid_loader

        lr_scheduler_params = {
            "training_point": self.training_point}
        valid_loss = None
        valid_computed = False
        if lr_scheduler.name == "reduceP" and lr_scheduler.is_reduce_point(self.training_point):
            assert valid_dataloader is not None
            valid_loss, valid_acc = self.compute_validation(valid_dataloader)
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
            valid_computed = True
        if valid_computed or (valid_dataloader is not None):
            if not valid_computed:
                valid_loss, valid_acc = self.compute_validation(valid_dataloader)
                valid_computed = False
            # For ReduceLROnPlateau.
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)

        training_point = (self.training_point[0],self.training_point[1],self.training_point[2])
        self.train_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
        # self.reporter.update(snapshot,training_point,self.train_lr)
        print("\ntrain_loss={0:.4f},valid_loss={1:.4f},train_acc={2:.4f},valid_acc={3:.4f}".format(train_loss,
                                                                                 valid_loss, train_acc * 100,
                                                                                 valid_acc * 100))
        if lr_scheduler is not None:
            # It is not convenient to wrap lr_scheduler (doing).
            if isinstance(lr_scheduler, LRSchedulerWrapper):
                lr_scheduler.step(**lr_scheduler_params)
                current_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
                if lr_scheduler.name == "reduceP":
                    if current_lr < self.last_lr:
                        self.last_lr = current_lr
                        self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)
                    elif current_lr <= lr_scheduler.min_lr and lr_scheduler.is_reduce_point(self.training_point):
                        self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)

                if lr_scheduler.is_cycle_point(self.training_point):
                    self.cycle_point+=1
                    self.save_model(mod="cycle",train_lr=self.train_lr,valid_loss=valid_loss)
            else:
                # For some pytorch lr_schedulers, but it is not available for all.
                lr_scheduler.step(self.training_point[0])

        info_dict = {"train_loss": "{0:.6f}".format(train_loss),
                     "valid_loss": "{0:.6f}".format(valid_loss),
                     "train_acc": "{0:.2f}".format(train_acc * 100),
                     "valid_acc": "{0:.2f}".format(valid_acc * 100),
                     }
        self.record_value.append(info_dict)
        dataframe = pd.DataFrame(self.record_value)
        dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
        self.record_value.clear()


    def get_accuracy(self,posterior, targets):
        prediction = torch.argmax(posterior, dim=1)
        num_correct = (targets == prediction).sum()
        accuracy = num_correct.item() / len(targets)
        accuracy = accuracy

        return accuracy


    def run(self):
        """Main function to start a training process.
        """
        try:
            self.init_training()
            # self.reporter = Reporter(self)
            start_epoch = self.params["start_epoch"]
            epochs = self.params["epochs"]
            data = self.elements["data"]
            model = self.elements["model"]
            # See init_training.
            model_forward = self.elements["model_forward"]
            self.train_lr = self.elements["optimizer"].state_dict()['param_groups'][0]['lr']
            self.last_lr =  self.elements["optimizer"].state_dict()['param_groups'][0]['lr']

            print("Training will run for {0} epochs.".format(epochs))

            for this_epoch in range(start_epoch,epochs):
                self.training_point[0] += 1
                # data.train_loader.dataset.set_epoch(this_epoch)
                progress = tqdm(data.train_loader, desc="train", unit=" step")
                for batch in progress:
                    progress.set_description("epochs {0}/{1}".format(this_epoch+1, epochs))

                    if self.training_point[1] == 0:
                        self.num_sample = len(data.train_loader)

                    self.training_point[1] += 1
                    loss, acc = self.train_one_batch(batch)

                    train_acc = acc * 100

                    train_loss = loss

                    progress.set_postfix(
                        loss=f"{train_loss:.4f}",
                        acc=f"{train_acc:.4f}",
                        step=self.training_point[1],
                    )
                    progress.update()
                    info_dict = {"train_loss": "{0:.6f}".format(train_loss),
                                "valid_loss": "{0:.6f}".format(self.valid_loss),
                                "train_acc": "{0:.2f}".format(train_acc),
                                "valid_acc": "{0:.2f}".format(self.valid_acc * 100),
                                }
                    self.record_value.append(info_dict)
                    dataframe = pd.DataFrame(self.record_value)
                    if self.start_write_log:
                        dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
                    else:
                        dataframe.to_csv(self.record_file, header=True, index=False)
                        self.start_write_log = True
                    self.record_value.clear()
                progress.close()


                self.save_model()
                self.training_point[1] = 0

            # self.reporter.finish()
            final_model_name = "{}_cycle".format(self.cycle_point) if self.cycle_point else epochs
            final_model_path = os.path.join(self.params["model_dir"], 'final.params')
            if os.path.exists(final_model_path) or os.path.islink(final_model_path):
                os.remove(final_model_path)

            os.symlink('{0}/{1}.{2}'.format(self.params["model_dir"], final_model_name, self.params["suffix"]),
                       final_model_path)

        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):traceback.print_exc()
            sys.exit(1)

class SimpleTrainer_ppl2():
    def __init__(self, package):
        default_elements = {"data": None, "model": None, "loss_func":None,
                            "optimizer": None,"lr_scheduler":None}
        default_params = {"model_dir": "", "epochs": 10,"gpu_id": "","suffix": "params",
                          "compute_accuracy":True,"start_epoch":0,
                          "report_interval_iters":100,"use_tensorboard":True}

        elements, params = package
        self.elements = utils.assign_params_dict(default_elements, elements)
        self.params = utils.assign_params_dict(default_params, params, support_unknow=True)

        assert self.elements["data"] is not None
        assert self.elements["model"] is not None
        assert self.elements["optimizer"] is not None

        assert self.params["model_dir"] != ""

        self.elements["model_forward"] = self.elements["model"]

        gpu_id = self.params["gpu_id"]
        self.epochs = self.params["epochs"]
        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

        self.loss_func = self.elements["loss_func"]
        self.record_value = []
        self.start_write_log = False
        self.valid_loss = 10.0
        self.valid_acc = 0.0
        self.record_file = "{0}/log/{1}".format(self.params["model_dir"], "train.csv")

        # (epoch, iter in epoch, global step)
        self.training_point = copy.deepcopy([self.params["start_epoch"], 0, 0])
        self.cycle_point = 0  # for cycle training.
        self.batch_size = 0
        self.num_sample = 1
        self.print_sample = False
        self.num_batch = 0

        self.trans_dict = []


    def init_training(self):
        model = self.elements["model"]
        start_epoch = self.params["start_epoch"]
        model_dir = self.params["model_dir"]
        # model_blueprint = self.params["model_blueprint"]
        suffix = self.params["suffix"]

        # Recover checkpoint | Tansform learning | Initialize parametes
        if start_epoch > 0:
            # This train_stage is equal to number of completed epoch
            print("Recover training from {0} epoch.".format(start_epoch))
            model.load_state_dict(torch.load('{0}/{1}.{2}'.format(model_dir, start_epoch, suffix),
                                             map_location="cpu"))


            info_log_path = '{0}/{1}/{2}.{3}'.format(
                model_dir, "checkpoint_info", start_epoch, "yaml")
            if os.path.exists(info_log_path):
                # info = torch.load(info_log_path)
                # self.elements["optimizer"].load_state_dict(info['optimizer'])
                # for state in self.elements["optimizer"].values():
                #     for k, v in state.items():
                #         if isinstance(v, torch.Tensor):
                #             state[k] = v.cuda()
                with open(info_log_path, 'r') as fin:
                    info = yaml.load(fin, Loader=yaml.FullLoader)
                self.training_point[2] = info['step']
        else:
            # Just use the raw initial model or initialize it again by some initial functions here
            pass # Now, it means use the raw initial model
        torch.backends.cudnn.benchmark = True
        model = model.to(self.device)

    def save_model(self, mod="epoch",train_lr=None,valid_loss=None):
        assert mod in ["epoch", "iter", "cycle"]
        if mod == "epoch":
            model_name = self.training_point[0]
        elif mod == "iter":
            model_name = "{}.{}".format(self.training_point[0], self.training_point[1])
        else:
            model_name = "{}_cycle".format(self.cycle_point)
        model_path = '{0}/{1}.{2}'.format(self.params["model_dir"], model_name, self.params["suffix"])

        info_log = {
            'train_lr': train_lr if train_lr else "see train.csv",
            "next_lr": self.elements["optimizer"].state_dict()['param_groups'][0]['lr'],
            'epoch': self.training_point[0],
            'iter in epoch': self.training_point[1],
            'step': self.training_point[2],
            'valid_loss':valid_loss if valid_loss else "see train.csv"
        }

        info_log_path = '{0}/{1}/{2}.{3}'.format(self.params["model_dir"], "checkpoint_info", model_name, "yaml")
        print("Save model to {0}. \n epoch/iter: {1}/{2}.  cur_step: {3}".format(model_path, self.training_point[0],
                                                                                       self.training_point[1], self.training_point[2]))
        torch.save(self.elements["model"].state_dict(), model_path)
        # torch.save(info, info_path)
        with open(info_log_path, 'w') as fout:
            data = yaml.dump(info_log)
            fout.write(data)


    def train_one_batch(self, batch):
        """A normal training core without fetching data from iterator.
        """
        model = self.elements["model"]
        optimizer = self.elements["optimizer"]
        # lr_scheduler = self.elements['lr_scheduler']
        loss_func = self.elements["loss_func"].to(self.device)

        if not model.training:
            model.train()

        inputs, targets = batch
        # print("in,out:",inputs,inputs.shape)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        # print("model:",model)
        optimizer.zero_grad()
        # posterior, t4 = model(inputs)
        # loss = loss_func(posterior, t4, targets)
        # print("posterior, t4:", posterior, t4)
        # print("loss:",loss)
        posterior = model.forward_loss(inputs)
        loss_ppl = model.get_loss()
        loss = loss_func(posterior,targets)+0.001*loss_ppl

        loss.backward()

        loss = loss.item()

        accuracy = self.get_accuracy(posterior.detach(), targets.detach())

        optimizer.step()

        self.training_point[2] += 1 # update step

        if self.training_point[1] % 100==0:
            self.step_lr(loss, accuracy, optimizer, self.elements["lr_scheduler"])
        self.num_batch += 1
        return loss, accuracy

    def compute_validation(self, data_loader):
        """A normal evaluation core.
        """
        model = self.elements["model"]
        loss_func = self.elements["loss_func"].to(self.device)

        train_status = model.training  # Record status.
        model.eval()

        loss = 0.
        accuracy = 0.
        num_samples = 0
        with torch.no_grad():
            for idx,this_data in enumerate(data_loader):
                inputs, targets = this_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # loss_func_ppl = loss_func_ppl.to(self.device)
                num_utts = targets.size(0)
                if num_utts == 0:
                    continue
                # posterior = model(inputs)

                posterior, t4 = model(inputs)
                loss += loss_func(posterior, t4, targets).item() * len(targets)
                accuracy += self.get_accuracy(posterior.detach(), targets.detach())* len(targets)
                num_samples += len(targets)

            avg_loss = loss / num_samples
            avg_accuracy = accuracy / num_samples

        model.train()
        self.valid_loss = avg_loss
        self.valid_acc = avg_accuracy

        return avg_loss, avg_accuracy


    def step_lr(self,train_loss,train_acc,base_optimizer,lr_scheduler):

        valid_dataloader=self.elements["data"].valid_loader

        lr_scheduler_params = {
            "training_point": self.training_point}
        valid_loss = None
        valid_computed = False
        if lr_scheduler.name == "reduceP" and lr_scheduler.is_reduce_point(self.training_point):
            assert valid_dataloader is not None
            valid_loss, valid_acc = self.compute_validation(valid_dataloader)
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
            valid_computed = True
        if valid_computed or (valid_dataloader is not None):
            if not valid_computed:
                valid_loss, valid_acc = self.compute_validation(valid_dataloader)
                valid_computed = False
            # For ReduceLROnPlateau.
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)

        training_point = (self.training_point[0],self.training_point[1],self.training_point[2])
        self.train_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
        # self.reporter.update(snapshot,training_point,self.train_lr)
        print("\ntrain_loss={0:.4f},valid_loss={1:.4f},train_acc={2:.4f},valid_acc={3:.4f}".format(train_loss,
                                                                                 valid_loss, train_acc * 100,
                                                                                 valid_acc * 100))

        info_dict = {"train_loss": "{0:.6f}".format(train_loss),
                     "valid_loss": "{0:.6f}".format(valid_loss),
                     "train_acc": "{0:.2f}".format(train_acc * 100),
                     "valid_acc": "{0:.2f}".format(valid_acc * 100),
                     }
        self.record_value.append(info_dict)
        dataframe = pd.DataFrame(self.record_value)
        dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
        self.record_value.clear()

        if lr_scheduler is not None:
            # It is not convenient to wrap lr_scheduler (doing).
            if isinstance(lr_scheduler, LRSchedulerWrapper):
                lr_scheduler.step(**lr_scheduler_params)
                current_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
                if lr_scheduler.name == "reduceP":
                    if current_lr < self.last_lr:
                        self.last_lr = current_lr
                        self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)
                    elif current_lr <= lr_scheduler.min_lr and lr_scheduler.is_reduce_point(self.training_point):
                        self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)

                if lr_scheduler.is_cycle_point(self.training_point):
                    self.cycle_point+=1
                    self.save_model(mod="cycle",train_lr=self.train_lr,valid_loss=valid_loss)
            else:
                # For some pytorch lr_schedulers, but it is not available for all.
                lr_scheduler.step(self.training_point[0])


    def get_accuracy(self,posterior, targets):
        prediction = torch.argmax(posterior, dim=1)
        num_correct = (targets == prediction).sum()
        accuracy = num_correct.item() / len(targets)
        accuracy = accuracy

        return accuracy


    def run(self):
        """Main function to start a training process.
        """
        try:
            self.init_training()
            start_epoch = self.params["start_epoch"]
            epochs = self.params["epochs"]
            data = self.elements["data"]

            # See init_training.
            self.train_lr = self.elements["optimizer"].state_dict()['param_groups'][0]['lr']
            self.last_lr =  self.elements["optimizer"].state_dict()['param_groups'][0]['lr']

            print("Training will run for {0} epochs.".format(epochs))

            for this_epoch in range(start_epoch,epochs):
                self.training_point[0] += 1
                data.train_loader.dataset.set_epoch(this_epoch)
                progress = tqdm(data.train_loader, desc="train", unit=" step")
                for batch in progress:
                    progress.set_description("epochs {0}/{1}".format(this_epoch, epochs))
                    if self.training_point[1] == 0:
                        self.num_sample = data.train_loader.dataset.get_data_dur()

                    self.training_point[1] += 1
                    loss, acc = self.train_one_batch(batch)

                    train_acc = acc * 100

                    train_loss = loss#.item()

                    progress.set_postfix(
                        loss=f"{train_loss:.4f}",
                        acc=f"{train_acc:.4f}",
                        step=self.training_point[1],
                    )
                    progress.update()

                    info_dict = {"train_loss": "{0:.6f}".format(train_loss),
                                "valid_loss": "{0:.6f}".format(self.valid_loss),
                                "train_acc": "{0:.2f}".format(train_acc * 100),
                                "valid_acc": "{0:.2f}".format(self.valid_acc * 100),
                                }
                    self.record_value.append(info_dict)
                    dataframe = pd.DataFrame(self.record_value)

                    dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
                    self.record_value.clear()
                progress.close()


                self.save_model()
                self.training_point[1] = 0

            # self.reporter.finish()
            final_model_name = "{}_cycle".format(self.cycle_point) if self.cycle_point else epochs
            final_model_path = os.path.join(self.params["model_dir"], 'final.params')
            if os.path.exists(final_model_path) or os.path.islink(final_model_path):
                os.remove(final_model_path)

            os.symlink('{0}/{1}.{2}'.format(self.params["model_dir"], final_model_name, self.params["suffix"]),
                       final_model_path)

        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):traceback.print_exc()
            sys.exit(1)