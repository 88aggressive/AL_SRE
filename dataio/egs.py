# -*- coding:utf-8 -*-

import os
import sys
import copy
import logging
import pandas as pd
import numpy as np
import yaml
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist

# import libs.support.utils as utils
# import libs.support.kaldi_io as kaldi_io
# from libs.support.prefetch_generator import BackgroundGenerator

sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")

from prefetch_generator import BackgroundGenerator
from processor import *

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Relation: features -> chunk-egs-mapping-file -> chunk-egs -> bunch(dataloader+bunch) => trainer
class ChunkEgs(Dataset):
    def __init__(self, egs_csv, conf, io_status=True):
        """
        @egs_csv:
            eg-id:str  wav-path:str  class-lable:int

        Other option
        @io_status: if false, do not read data from disk and return zero, which is useful for saving i/o resource 
        when kipping seed index.
        """
        # assert egs_type is "chunk" or egs_type is "vector"
        assert egs_csv != "" and egs_csv is not None
        head = pd.read_csv(egs_csv, sep=" ", nrows=0).columns

        assert "eg-id" in head
        assert "wav-path" in head
        assert "class-label" in head

        self.eg_id = pd.read_csv(egs_csv, sep=" ", usecols=["eg-id"]).values.astype(np.string_)
        self.wav_path = pd.read_csv(egs_csv, sep=" ", usecols = ["wav-path"]).values.astype(np.string_)
        self.label = pd.read_csv(egs_csv, sep=" ", usecols = ["class-label"]).values

        self.io_status = io_status

        shuffle = conf.get('shuffle', True)

        self.chunk_len = conf.get('random_chunk_size', 2.015)
        # Augmentation.
        self.aug = None
        speech_aug = conf.get('speech_aug', False)  # true
        if speech_aug == True:
            print("add speech augmentation to wav")
            speech_aug_conf_file = conf.get('speech_aug_conf', '')
            assert speech_aug_conf_file != ''
            with open(speech_aug_conf_file, 'r') as fin:
                speech_aug_conf = yaml.load(
                    fin, Loader=yaml.FullLoader)

                csv_aug_folder = conf.get('csv_aug_folder', '')
                if csv_aug_folder: utils.change_csv_folder(speech_aug_conf, csv_aug_folder)
            self.aug = SpeechAug_test(**speech_aug_conf)

        ##get kaldi feature.
        feature_extraction_conf = conf.get('feature_extraction_conf', {})
        self.extract = KaldiFeature_test(**feature_extraction_conf)

    def set_io_status(self, io_status):
        self.io_status = io_status

    def __getitem__(self, index):
        if not self.io_status :
            return 0., 0.

        # Decode string from bytes after using astype(np.string_).
        wav_path = str(self.wav_path[index][0], encoding='utf-8')
        waveform, sample_rate = torchaudio.load(wav_path)

        duration_sample = waveform.shape[1]
        snt_len_sample = int(self.chunk_len * sample_rate)
        # print("chunk_len,sample_rate", chunk_len, sample['sample_rate'], type(sample['sample_rate']), type(chunk_len))

        if duration_sample > snt_len_sample:
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
            waveform = waveform[:, start:stop]
        else:
            repeat_num = math.ceil(snt_len_sample / duration_sample)
            waveform = waveform[:, :duration_sample].repeat(1, repeat_num)[:, :snt_len_sample]

        if self.aug is not None:
            waveform = self.aug.add_aug(waveform)
        feat = self.extract.get_feats(waveform, sample_rate)##[nums,80]
        feat = feat.transpose(0,1)
        # print("feat:",feat.shape)
        if (torch.any((torch.isnan(feat)))):
            logging.warning('Failed to make featrue for {}'.format(self.eg_id[index][0]))
            pass

        target = self.label[index][0]

        return feat, target

    def __len__(self):
        return len(self.wav_path)

    def get_chunk_position(self):
        ##分割长语音，整段语音全部参与训练
        pass


class BaseBunch():
    """BaseBunch:(trainset,[valid]).
    """
    def __init__(self, trainset, valid=None, use_fast_loader=False, max_prefetch=10,
                 batch_size=512, shuffle=True, num_workers=0, pin_memory=False, drop_last=True):

        num_samples = len(trainset)
        num_gpu = 1

        train_sampler = None

        if use_fast_loader:
            self.train_loader = DataLoaderFast(max_prefetch, trainset, batch_size = batch_size, shuffle=shuffle, 
                                               num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
                                               sampler=train_sampler)
        else:
            self.train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers, 
                                           pin_memory=pin_memory, drop_last=drop_last, sampler=train_sampler)

        self.num_batch_train = len(self.train_loader)

        if self.num_batch_train <= 0:
            raise ValueError("Expected num_batch of trainset > 0. There are your egs info: num_gpu={}, num_samples/gpu={}, "
                             "batch-size={}, drop_last={}.\nNote: If batch-size > num_samples/gpu and drop_last is true, then it "
                             "will get 0 batch.".format(num_gpu, len(trainset)/num_gpu, batch_size, drop_last))


        if valid is not None:
            valid_batch_size = min(batch_size, len(valid)) # To save GPU memory

            if len(valid) <= 0:
                raise ValueError("Expected num_samples of valid > 0.")

            # Do not use DataLoaderFast for valid for it increases the memory all the time when compute_valid_accuracy is True.
            # But I have not find the real reason.
            self.valid_loader = DataLoader(valid, batch_size = valid_batch_size, shuffle=False, num_workers=num_workers, 
                                           pin_memory=pin_memory, drop_last=False)

            self.num_batch_valid = len(self.valid_loader)
        else:
            self.valid_loader = None
            self.num_batch_valid = 0


    @classmethod
    def get_bunch_from_csv(self, trainset_csv: str,validset_csv: str,egs_params: dict = {}):

        train_conf = egs_params['dataset_conf']
        valid_conf = copy.deepcopy(train_conf)
        valid_conf['speech_aug'] = False
        # valid_conf['spec_aug'] = False
        valid_conf['shuffle'] = False

        trainset = ChunkEgs(trainset_csv, train_conf)
        if validset_csv != "" and validset_csv is not None:
            validset = ChunkEgs(validset_csv, valid_conf)
        else:
            validset = None

        return self(trainset, validset, **egs_params['data_loader_conf'])

    def get_train_batch_num(self):
        return self.num_batch_train

    def get_valid_batch_num(self):
        return self.num_batch_valid

    def __len__(self):
        # main: train
        return self.num_batch_train

    @classmethod
    def get_bunch_from_egsdir(self,egsdir, egs_params: dict={}):
        train_csv_name = "train.csv"
        valid_csv_name = "valid.csv"

        train_csv = egsdir + "/" + train_csv_name
        valid_csv = egsdir + "/" + valid_csv_name

        if not os.path.exists(valid_csv):
            valid_csv = None

        bunch = self.get_bunch_from_csv(train_csv,valid_csv,egs_params)

        return bunch


class DataLoaderFast(DataLoader):
    """Use prefetch_generator to fetch batch to avoid waitting.
    """
    def __init__(self, max_prefetch, *args, **kwargs):
        assert max_prefetch >= 1
        self.max_prefetch = max_prefetch
        super(DataLoaderFast, self).__init__(*args, **kwargs)

    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderFast, self).__iter__(), self.max_prefetch)
