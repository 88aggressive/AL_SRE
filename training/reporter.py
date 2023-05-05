# -*- coding:utf-8 -*-

import os, sys
import time
import shutil
import logging
import progressbar
import traceback
import pandas as pd

from multiprocessing import Process, Queue
sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")
sys.path.insert(0, "AL_SRE/training")
import utils

# Leo 2021-10
class Reporter_new():
    def __init__(self, trainer):
        default_params = {
            "report_interval_iters":100,
            "record_file":"train.csv",
            "use_tensorboard":False
        }
        self.trainer = trainer
        default_params = utils.assign_params_dict(default_params, self.trainer.params)

        self.report_interval_iters = default_params["report_interval_iters"]

        if default_params["use_tensorboard"]:
            from tensorboardX import SummaryWriter
            model_name = os.path.basename(self.trainer.params["model_dir"])
            time_string = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            self.board_writer = SummaryWriter("{}/log/{}-{}-tensorboard".format(self.trainer.params["model_dir"], model_name, time_string))
        else:
            self.board_writer = None

        self.epochs = self.trainer.params["epochs"]

        self.optimizer = self.trainer.elements["optimizer"]

        self.device = "[{0}]".format(utils.get_device(self.trainer.elements["model"]))

        self.record_value = []

        self.start_write_log = False
        if default_params["record_file"] != "" and default_params["record_file"] is not None:
            self.record_file = "{0}/log/{1}".format(self.trainer.params["model_dir"], default_params["record_file"])

            # The case to recover training
            if self.trainer.params["start_epoch"] > 0:
                self.start_write_log = True
            elif os.path.exists(self.record_file):
                # Do backup to avoid clearing the loss log when re-running a same launcher.
                bk_file = "{0}.bk.{1}".format(self.record_file, time.strftime('%Y_%m_%d.%H_%M_%S',time.localtime(time.time())))
                shutil.move(self.record_file, bk_file)
        else:
            self.record_file = None

        # A format to show progress
        # Do not use progressbar.Bar(marker="\x1b[32m█\x1b[39m") and progressbar.SimpleProgress(format='%(value_s)s/%(max_value_s)s') to avoid too long string.
        widgets=["Epoch:", progressbar.Variable('current_epoch', format='{formatted_value}', width=0, precision=0), "/{0}, ".format(self.epochs),
                 "Iter:", progressbar.Variable('current_iter', format='{formatted_value}', width=0, precision=0), ", ",
                 "Step:", progressbar.Variable('current_step', format='{formatted_value}', width=0, precision=0), ", ",
                 "(", progressbar.Timer(format='ELA: %(elapsed)s'), ", ",progressbar.AdaptiveTransferSpeed(format='%(scaled)5.1f %(prefix)s%(unit)-s/s',inverse_format='%(scaled)5.1f s/%(prefix)s%(unit)-s', unit='step',), ")",", ",
                 "Origin total hours:", progressbar.Variable('total_dur', format='{formatted_value}', width=0, precision='2f'), ", ",
                 progressbar.Variable('Num_sample')]

        self.bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True)

        # Use multi-process for update.
        self.queue = Queue()
        self.process = Process(target=self._update, daemon=True)
        self.process.start()

    def is_report(self, training_point):

        return (training_point[2]%self.report_interval_iters == 0 or \
                training_point[1] == 1)

    def record(self, info_dict, training_point):
        if self.record_file is not None:
            self.record_value.append(info_dict)

            if self.is_report(training_point):
                print("Device:{0}, {1}".format(self.device, utils.dict_to_params_str(info_dict, auto=False, sep=", ")))
                dataframe = pd.DataFrame(self.record_value)
                if self.start_write_log:
                    dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
                else:
                    # with open(self.record_file, "w") as f:
                    #     f.truncate()
                    dataframe.to_csv(self.record_file, header=True, index=False)
                    self.start_write_log = True
                self.record_value.clear()

    def _update(self):
        # Do not use any var which will be updated by main process, such as self.trainer.training_point.
        while True:
            try:
                res = self.queue.get()
                if res is None:
                    self.bar.finish()
                    break

                snapshot, training_point, current_lr = res
                current_epoch, current_iter, current_step = training_point
                # total_dur=snapshot.pop('total_dur')
                num_sample = format(snapshot.pop('num_sample'),',')

                self.bar.update(current_step, current_epoch=current_epoch, current_iter=current_iter,current_step=current_step,Num_sample=num_sample)
                real_snapshot = snapshot.pop("real")
                if self.board_writer is not None:
                    self.board_writer.add_scalar("epoch", float(current_epoch), current_step)
                    self.board_writer.add_scalar("lr", current_lr, current_step)

                    loss_dict = {}
                    acc_dict = {}
                    try:
                        for key in real_snapshot.keys():
                            if "loss" in key:
                                loss_dict[key] = real_snapshot[key]
                            elif "acc" in key:
                                acc_dict[key] = real_snapshot[key]
                            else:
                                self.board_writer.add_scalar(key, real_snapshot[key], current_step)

                        self.board_writer.add_scalars("scalar_acc", acc_dict, current_step)
                        self.board_writer.add_scalars("scalar_loss", loss_dict, current_step)
                    except Exception as ex:
                        logging.warning("some tensorboard porblem, pass it")
                info_dict = {"epoch":current_epoch, "iter":current_iter, "step":current_step,
                             "lr":"{0:.8f}".format(current_lr)}
                info_dict.update(snapshot)
                self.record(info_dict, training_point)
            except BaseException as e:
                self.bar.finish()
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                sys.exit(1)

    def update(self, snapshot:dict,point,current_lr):
        # One update calling and one using of self.trainer.training_point and current_lr.
        # current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.queue.put((snapshot, point, current_lr))

    def finish(self):
        self.queue.put(None)
        # Wait process completed.
        self.process.join()
        
class LRFinderReporter():
    def __init__(self, max_value, log_dir=None, comment=None):

        if log_dir is not None:
            assert isinstance(log_dir, str)
            from tensorboardX import SummaryWriter
            time_string = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            if comment is None:
                comment = ""
            else:
                comment = comment + "-"
            self.board_writer = SummaryWriter("{}/{}{}-lr-finder-tensorboard".format(log_dir, comment, time_string))
        else:
            self.board_writer = None

        widgets=[progressbar.Percentage(format='%(percentage)3.2f%%'), " | ", "Iter:",
                 progressbar.Variable('current_iter', format='{formatted_value}', width=0, precision=0), "/{0}".format(max_value), ", ",
                 progressbar.Variable('snapshot', format='{formatted_value}', width=8, precision=0),
                 " (", progressbar.Timer(format='ELA: %(elapsed)s'), ", ",progressbar.AdaptiveETA(), ")"]

        self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets, redirect_stdout=True)

        # Use multi-process for update.
        self.queue = Queue()
        self.process = Process(target=self._update, daemon=True)
        self.process.start()

    def _update(self):
        while True:
            try:
                res = self.queue.get()
                if res is None:break
                update_iters, snapshot = res
                self.bar.update(update_iters, current_iter=update_iters, snapshot=utils.dict_to_params_str(snapshot, auto=False, sep=", "))
                if self.board_writer is not None:
                    self.board_writer.add_scalars("lr_finder_scalar_group", snapshot, update_iters)
            except BaseException as e:
                self.bar.finish()
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                sys.exit(1)

    def update(self, update_iters:int, snapshot:dict):
        self.queue.put((update_iters, snapshot))

    def finish(self):
        self.queue.put(None)
        self.bar.finish()