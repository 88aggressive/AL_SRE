#!/usr/bin/python3
import torch
import torchaudio
import random,math
import yaml

import logging
import os
import sys
# import noisereduce as nr

sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")

from processor import *
import utils

def read_file(spk_file,wav_root):

    spk_utt = {}
    with open(spk_file,"r",encoding="utf-8") as spkf:
        spk_data = spkf.readlines()

        for wav_spk in spk_data:
            wav_line = wav_spk.strip().split("\t")
            wav_paths = wav_line[1:]
            # print("wav:",wav_spk,wav_line)
            spk = wav_line[0].strip()
            if spk not in spk_utt.keys():
                spk_utt[spk] = []
            for wavs in wav_paths:
                paths = os.path.join(wav_root,wavs)
                spk_utt[spk].append(paths)

    return spk_utt


def get_spk_model(save_spk_path, model,device,spk2utt,kaldi_reader):
    with torch.no_grad():
        embedding_list = []
        spk_list = []

        for spk in spk2utt:
            utts = spk2utt[spk]
        # for spk, utts in spk2utt.items():
            spk_id = []
            for utt in utts:
                # print("utt:",utt)
                embedding = get_embedding_model_kaldi(utt,model,device,kaldi_reader)
                # embedding = embedding.cpu().numpy() - mean_xvector
                spk_id.append(embedding.numpy())
            # print("spk_id:",np.shape(spk_id))
            spk_em = np.squeeze(spk_id)
            spk_em2 = np.mean(spk_em, axis=0, keepdims=True)
            # print("spk_id:", np.shape(spk_em),np.shape(spk_em2))
            embedding_list.append(spk_em2)
            spk_list.append(spk)
        embedding_list = np.squeeze(embedding_list)
        # print("spk_id:", np.shape(embedding_list), np.shape(spk_list))
    save_spk = {"emds": embedding_list, "spks": spk_list}
    np.save(save_spk_path, save_spk)
    print("save spk embedding and id")


class InputSequenceNormalization(object):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.

    Example
    -------
    >>> import torch
    >>> norm = InputSequenceNormalization()
    >>> input = torch.randn([101, 20])
    >>> feature = norm(inputs)
    """

    def __init__(
            self,
            mean_norm=True,
            std_norm=False,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.eps = 1e-10

    def __call__(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A tensor `[t,f]`.
        """
        if self.mean_norm:
            mean = torch.mean(x, dim=0).detach().data
        else:
            mean = torch.tensor([0.0], device=x.device)

        # Compute current std
        if self.std_norm:
            std = torch.std(x, dim=0).detach().data
        else:
            std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        std = torch.max(std, self.eps * torch.ones_like(std))
        x = (x - mean.data) / std.data

        return x


def get_embedding(wav_file,model,device):
    model.eval()
    norm = InputSequenceNormalization()
    wav, sample_rate = torchaudio.load(wav_file)
    if len(wav.shape) == 1:
        # add channel
        wav = wav.unsqueeze(0)

    feat = kaldi.fbank(wav,num_mel_bins=80,low_freq=40,high_freq=-200,energy_floor=0.0)
    feat = feat.detach()
    feat = feat.to(device)
    feat = norm(feat)

    with torch.no_grad():
        feat = feat.unsqueeze(0) #[1,566,80]
        feat = feat.transpose(1, 2)
        # print("feat:",feat.shape)
        embedding = model.extract_embedding(feat)
        # embedding = embedding.squeeze(2)
    embedding = embedding.cpu().detach()
    return embedding


def get_aug(wav_file,aug_reader=None,save_aug_audio=False):
    waveforms, sample_rate = torchaudio.load(wav_file)

    if aug_reader != None:
        waveforms = aug_reader.add_aug(waveforms)
        ##save aug file
        if save_aug_audio:
            wav_list = wav_file.split('/')
            spkid = wav_list[-2]
            uttpath = wav_list[-1]
            keyword_aug_path = "/mnt/x2/database/al-speaker-dataset/list/test_aug_audio_road_0_15"
            if not os.path.exists(f"{keyword_aug_path}/{spkid}"):
                os.mkdir(f"{keyword_aug_path}/{spkid}")
            test_filepath = os.path.join(keyword_aug_path,spkid,uttpath)
            torchaudio.save(test_filepath, waveforms, sample_rate)


def get_embedding_model_kaldi(wav_file,model,device,kaldi_reader,aug_reader=None,save_aug_audio=False):
    model.eval()
    waveforms, sample_rate = torchaudio.load(wav_file)

    if aug_reader != None:
        waveforms = aug_reader.add_aug(waveforms)
        ##save aug file
        # if save_aug_audio:
        #     wav_list = wav_file.split('/')
        #     spkid = wav_list[-2]
        #     uttpath = wav_list[-1]
        #     keyword_aug_path = "/mnt/x2/database/al-speaker-dataset/list/test_aug_audio_road"
        #     if not os.path.exists(f"{keyword_aug_path}/{spkid}"):
        #         os.mkdir(f"{keyword_aug_path}/{spkid}")
        #     test_filepath = os.path.join(keyword_aug_path,spkid,uttpath)
        #     torchaudio.save(test_filepath, waveforms, sample_rate)

        # print("wav2:", waveforms.shape)

    # waveforms = de_sil(waveforms, sample_rate) ##de-scilence

    feat = kaldi_reader.get_feats(waveforms, sample_rate)
    feat = feat.to(device)

    with torch.no_grad():

        embedding = model.extract_embedding(feat)
        # embedding = embedding.squeeze(2)
    embedding = embedding.cpu().detach()
    return embedding

def get_embedding_model_kaldi2(wav_file,model,device,kaldi_reader,aug_reader=None,save_aug_audio=False):
    model.eval()
    waveforms, sample_rate = torchaudio.load(wav_file)

    if aug_reader != None:
        waveforms = aug_reader.add_aug(waveforms)

        ##reduce noise
        wav_numpy = waveforms.numpy()
        reduced_noise = nr.reduce_noise(y=wav_numpy, sr=sample_rate)
        waveforms = torch.from_numpy(reduced_noise)



    feat = kaldi_reader.get_feats(waveforms, sample_rate)
    feat = feat.to(device)

    with torch.no_grad():

        embedding = model.extract_embedding(feat)
        # embedding = embedding.squeeze(2)
    embedding = embedding.cpu().detach()
    return embedding


def get_embedding_model(wav_file,model,device,kaldi_reader):
    model.eval()
    waveforms, sample_rate = torchaudio.load(wav_file)

    feat = kaldi_reader.get_feats(waveforms, sample_rate)
    feat = feat.to(device)

    with torch.no_grad():

        embedding = model.extract_embedding(feat)
        # embedding = embedding.squeeze(2)
    embedding = embedding.cpu().detach()
    return embedding

def get_embedding_chunk(wav_file,model,device,chunk_len=2.015):
    model.eval()
    # norm = InputSequenceNormalization()
    wav, sample_rate = torchaudio.load(wav_file)

    if len(wav.shape) == 1:
        # add channel
        wav = wav.unsqueeze(0)

    duration_sample = wav.shape[1]
    snt_len_sample = int(chunk_len * sample_rate)

    if duration_sample > snt_len_sample:
        start = random.randint(0, duration_sample - snt_len_sample - 1)
        stop = start + snt_len_sample
        wav = wav[:, start:stop]

    feat = kaldi.fbank(wav,num_mel_bins=80,low_freq=40,high_freq=-200,energy_floor=0.0)
    feat = feat.detach()
    feat = feat.to(device)
    # feat = norm(feat)

    with torch.no_grad():
        feat = feat.unsqueeze(0) #[1,566,80]
        feat = feat.transpose(1, 2)
        # print("feat:",feat.shape)
        embedding = model.extract_embedding(feat)
        # embedding = embedding.squeeze(2)
    embedding = embedding.cpu().detach()
    return embedding

def de_sil(data,sr,win_len=0.1,min_eng=50,retry_times=1,force_output=True):
    """
        data: Iterable[{key, wav, label, lens, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate, lens}]
    """
    waveform = data
    sr = sr
    cache_wave,cache_len = de_silence(waveform,sr=sr,win_len=win_len,min_eng=min_eng)
    while retry_times and cache_len==0:
        min_eng/=2
        cache_wave,_ = de_silence(waveform,sr=sr,win_len=win_len,min_eng=min_eng/2)
        retry_times-=1
    if force_output and cache_len==0:
        cache_wave=waveform
    return cache_wave

