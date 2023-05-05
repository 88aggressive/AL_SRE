# -*- coding:utf-8 -*-

import os
import sys
import logging
import random,math
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torchaudio
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, "AL_SRE/support")
sys.path.insert(0, "AL_SRE/dataio")

from utils import batch_pad_right,get_torchaudio_backend
from signal_processing import de_silence
from augmentation import *

torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


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


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[{eg-path,...}]): eg-path: url or local file.

        Returns:
            Iterable[{eg-path, stream}]
    """
    for sample in data:
        # print("data:",sample) #{eg-path,eg-dur,eg-num,rank,world_size,worker_id,num_workers,epoch,main_seed}
        assert 'eg-path' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['eg-path']  #exp/data/shard/train/shards_000000001.tar
        try:
            pr = urlparse(url)
            # print("pr:",pr) #ParseResult(scheme='', netloc='', path='exp/data/shard/train/shards_000000003.tar', params='', query='', fragment='')
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{eg-path, stream}]

        Returns:
            Iterable[{key, wav, label, sample_rate,lens}]
    """
    for sample in data:
        # print("data:", sample)
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        # print("stream:", stream) #<tarfile.TarFile object
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            # print("tarinfo:",tarinfo)  #<TarInfo 'id07672-kUfIhLJjoPc-00324-4.030_6.045.txt' at 0x7fac8047ab80> &.wav file
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        label = file_obj.read().decode('utf8')
                        example['label'] = int(label)
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform[:1,:]
                        example['sample_rate'] = sample_rate
                        example['lens'] = torch.ones(1)
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """
        data: Iterable[{eg-id,wav-path,class-label,...}], dict has id/wav/label

        Returns:
            Iterable[{key, wav, label, sample_rate,lens}]
    """
    for sample in data:

        assert 'eg-id' in sample
        assert 'wav-path' in sample
        assert 'class-label' in sample
        key = sample['eg-id']
        wav_file = sample['wav-path']
        label = sample['class-label']
        try:
            if 'start-position' in sample:
                assert 'end-position' in sample
                start, stop = int(sample['start-position']), int(sample['end-position'])
                waveform, sample_rate = torchaudio.load(
                    filepath=wav_file,
                    num_frames=stop - start,
                    frame_offset=start)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            waveform = waveform[:1,:]
            label = int(label)
            lens = torch.ones(1)
            example = dict(key=key,
                           label=label,
                           wav=waveform,
                           lens=lens,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(key))

def de_sil(data,win_len=0.1,min_eng=50,retry_times=1,force_output=True):
    """
        data: Iterable[{key, wav, label, lens, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate, lens}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        assert 'lens' in sample
        assert 'sample_rate' in sample
        waveform = sample['wav']
        sr = sample['sample_rate']
        duration_sample=int(sample['lens']*(sample['wav'].shape[1]))
        waveform = waveform[:,0:duration_sample]
        cache_wave,cache_len = de_silence(waveform,sr=sr,win_len=win_len,min_eng=min_eng)
        while retry_times and cache_len==0:
            min_eng/=2
            cache_wave,_ = de_silence(waveform,sr=sr,win_len=win_len,min_eng=min_eng/2)
            retry_times-=1
        if force_output and cache_len==0:
            cache_wave=waveform
        sample['lens'] = torch.ones(waveform.shape[0])
        sample['wav'] = cache_wave
        del waveform
        yield sample

def random_chunk(data,chunk_len=2.015):
    """
        data: Iterable[{key, wav, label, lens, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate, lens}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        assert 'lens' in sample
        assert 'sample_rate' in sample
        waveform = sample['wav']
        duration_sample=int(sample['lens']*(sample['wav'].shape[1]))
        snt_len_sample = int(chunk_len*sample['sample_rate'])
        # print("chunk_len,sample_rate", chunk_len, sample['sample_rate'], type(sample['sample_rate']), type(chunk_len))

        if duration_sample > snt_len_sample:
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
            sample['wav'] = waveform[:,start:stop]
        else:
            repeat_num = math.ceil(snt_len_sample/duration_sample)
            sample['wav'] = waveform[:,:duration_sample].repeat(1,repeat_num)[:,:snt_len_sample]
        sample['lens'] = torch.ones(waveform.shape[0])

        yield sample

def split_wav(data,chunk_len=2.015):
    """
        data: Iterable[{key, wav, label, lens, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate, lens}]
    """
    total_chunk_num = []
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        assert 'lens' in sample
        assert 'sample_rate' in sample
        waveform = sample['wav']
        key = sample['key']
        label = sample['label']
        sample_rate = sample['sample_rate']
        chunk_len = torch.ones(waveform.shape[0])
        # print("chunk_len,sample_rate",chunk_len,sample_rate,type(sample_rate),type(chunk_len))
        duration_sample=int(sample['lens']*(sample['wav'].shape[1]))
        snt_len_sample = int(2.015*sample['sample_rate'])
        chunk_nums = duration_sample // snt_len_sample
        total_chunk_num.append(chunk_nums)


        if chunk_nums < 1:
            del sample
            continue
        for i in range(chunk_nums):
            start = i*snt_len_sample
            stop = start + snt_len_sample
            chunk_wav = waveform[:,start:stop]
            # chunk_len = torch.ones(snt_len_sample)
            ##split_wav,add sample
            example = dict(key=key,
                           label=label,
                           wav=chunk_wav,
                           lens=chunk_len,
                           sample_rate=sample_rate)
            # print("len:",i)
            # print("lens:", chunk_len)
            yield example
    # print("total_chunk_num,len,sum:", len(total_chunk_num),sum(total_chunk_num))
    #300,1071
    # exit()

def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, lens, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, lens, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample



class SpeechAugPipline(object):
    def __init__(self, speechaug={}, tail_speechaug={}):
        super().__init__()

        self.speechaug = SpeechAug(**speechaug)
        self.tail_speechaug = SpeechAug(**tail_speechaug)
        speechaug_n_concat=self.speechaug.get_num_concat()
        tail_speechaug_n_concat=self.tail_speechaug.get_num_concat()
        # The concat number of speech augment, which is used to modify target.
        self.concat_pip= (speechaug_n_concat,tail_speechaug_n_concat)

    def __call__(self, data):
        """ speechaug.
            Args:
                data: Iterable[{key, wav, label, lens, sample_rate}]
            Returns:
                Iterable[{key, wav, label, lens, sample_rate}]
        """
        for sample in data:
            assert 'wav' in sample
            assert 'key' in sample
            assert 'lens' in sample
            assert 'label' in sample

            waveforms = sample['wav']
            lens = sample['lens']
            try:
                # print("aug:",sample)
                # exit()
                waveforms, lens = self.speechaug(waveforms, lens)

                waveforms, lens = self.tail_speechaug(waveforms, lens)
                sample['wav'] = waveforms

                sample['lens'] = lens
                yield sample
            except Exception as ex:
                logging.warning('Failed to speech aug {}'.format(sample['key']))



class KaldiFeature(object):
    """ This class extract features as kaldi's compute-mfcc-feats.

    Arguments
    ---------
    feat_type: str (fbank or mfcc).
    feature_extraction_conf: dict
        The config according to the kaldi's feature config.
    """

    def __init__(self,feature_type='mfcc',kaldi_featset={},mean_var_conf={}):
        super().__init__()
        assert feature_type in ['mfcc','fbank']
        self.feat_type=feature_type

        self.kaldi_featset=kaldi_featset
        if self.feat_type=='mfcc':
            self.extract=kaldi.mfcc
        else:
            self.extract=kaldi.fbank
        if mean_var_conf is not None:
            self.mean_var=InputSequenceNormalization(**mean_var_conf)
        else:
            self.mean_var=torch.nn.Identity()

    def __call__(self,data):
        """ make features.
            Args:
                data: Iterable[{key, wav, label, lens, sample_rate}]
            Returns:
                Iterable[{utt:str, keys:list, label, feats:list, max_len:int}]
        """
        for sample in data:
            assert 'wav' in sample
            assert 'key' in sample
            assert 'label' in sample
            assert 'lens' in sample
            assert 'sample_rate' in sample

            self.kaldi_featset['sample_frequency'] = sample['sample_rate']
            lens = sample['lens']
            waveforms = sample['wav']
            waveforms = waveforms * (1 << 15)
            feats = []
            label = sample['label']
            keys=[]
            utt = sample['key']
            try:
                with torch.no_grad():
                    lens=lens*waveforms.shape[1]

                    for i,wav in enumerate(waveforms):

                        if len(wav.shape)==1:
                            # add channel
                            wav=wav.unsqueeze(0)
                        wav= wav[:,:lens[i].long()]
                        feat=self.extract(wav,**self.kaldi_featset) ##[nums,80]
                        # print("feat:",feat.shape)
                        if(torch.any((torch.isnan(feat)))):
                            logging.warning('Failed to make featrue for {}, aug version:{}'.format(sample['key'],i))
                            pass
                        feat = feat.detach()
                        feat=self.mean_var(feat)

                        key = sample['key']+'#{}'.format(i) if i>0 else sample['key']
                        feats.append(feat)

                        keys.append(key)
                    if len(feats)==0:
                        pass

                    max_len = max([feat.size(0) for feat in feats])
                    yield dict(utt=utt,keys=keys,feats=feats,label=label,max_len=max_len)
            except Exception as ex:
                logging.warning('Failed to make featrue {}'.format(sample['key']))


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{utt:str, keys:list, label, feats:list, max_len:int}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{utt:str, keys:list, label, feats:list, max_len:int}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x

def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`
        Args:
            data: Iterable[{utt:str, keys:list, label, feats:list, max_len:int}]
            batch_size: batch size
        Returns:
            Iterable[List[{utt:str, keys:list, label, feats:list, max_len:int}]]
    """
    buf = []
    for sample in data:
        assert 'feats' in sample
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`
        Args:
            data: Iterable[{utt:str, keys:list, label, feats:list, max_len:int}]
            max_frames_in_batch: max_frames in one batch
        Returns:
            Iterable[List[{utt:str, keys:list, label, feats:list, max_len:int}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feats' in sample
        assert 'max_len' in sample
        assert 'keys' in sample
        new_max_sample_frames = sample['max_len']
        new_num = len(sample['keys'])
        longest_frames = max(longest_frames, new_max_sample_frames)
        frames_after_padding = longest_frames * (sum([len(x['keys']) for x in buf]) + new_num)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_max_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)

    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))
    


def padding(data):
    """ Padding the data into training data
        Args:
            data: Iterable[List[{utt:str, keys:list, label, feats:list, max_len:int}]]
        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:

        assert isinstance(sample, list)
        feats=[]
        labels=[]
        keys=[]
        for x in sample:
            feats.extend(x['feats'])

            labels.extend([x['label']]*len(x['feats']))
            keys.extend(x['keys'])

        labels = torch.LongTensor(labels)
        feats = [(x.T) for x in feats]
        padded_feats, lens = batch_pad_right(feats)

        yield (padded_feats, labels)

class SpeechAug_test(object):
    def __init__(self, speechaug={},speechaug_test={},tail_speechaug={}):
        super().__init__()

        self.speechaug = SpeechAug_noprint(**speechaug_test)
        self.tail_speechaug = SpeechAug_noprint(**tail_speechaug)
        speechaug_n_concat=self.speechaug.get_num_concat()
        tail_speechaug_n_concat=self.tail_speechaug.get_num_concat()
        # The concat number of speech augment, which is used to modify target.
        self.concat_pip= (speechaug_n_concat,tail_speechaug_n_concat)

    def add_aug(self, waveforms):
        """ speechaug.
            Args:
                data: Iterable[{key, wav, label, lens, sample_rate}]
            Returns:
                Iterable[{key, wav, label, lens, sample_rate}]
        """
        lens = torch.ones(1)
        try:
            # print("aug:",sample)
            # exit()
            waveforms, lens = self.speechaug(waveforms, lens)

            waveforms, lens = self.tail_speechaug(waveforms, lens)

            return waveforms
        except Exception as ex:
            logging.warning('Failed to speech aug')

class KaldiFeature_test(object):
    """ This class extract features as kaldi's compute-mfcc-feats.

    Arguments
    ---------
    feat_type: str (fbank or mfcc).
    feature_extraction_conf: dict
        The config according to the kaldi's feature config.
    """

    def __init__(self,feature_type='mfcc',kaldi_featset={},mean_var_conf={}):
        super().__init__()
        assert feature_type in ['mfcc','fbank']
        self.feat_type=feature_type

        self.kaldi_featset=kaldi_featset
        if self.feat_type=='mfcc':
            self.extract=kaldi.mfcc
        else:
            self.extract=kaldi.fbank
        if mean_var_conf is not None:
            self.mean_var=InputSequenceNormalization(**mean_var_conf)
        else:
            self.mean_var=torch.nn.Identity()

    def get_feats(self,waveforms, sample_rate):
        """ make features.
            Args:
                data: Iterable[{key, wav, label, lens, sample_rate}]
            Returns:
                Iterable[{utt:str, keys:list, label, feats:list, max_len:int}]
        """

        self.kaldi_featset['sample_frequency'] = sample_rate
        lens = torch.ones(1)

        waveforms = waveforms * (1 << 15)

        try:
            with torch.no_grad():
                lens=lens*waveforms.shape[1]

                for i,wav in enumerate(waveforms):

                    if len(wav.shape)==1:
                        # add channel
                        wav=wav.unsqueeze(0)
                    # print("wav11:", wav.shape)
                    wav= wav[:,:lens[i].long()]
                    # print("wav111:", wav.shape)
                    # print("kaldi_featset:",self.kaldi_featset)
                    feat=self.extract(wav,**self.kaldi_featset) ##[nums,80]
                    # print("feat2:", feat.shape) #[378,80]
                    feat = feat.detach()
                    feat=self.mean_var(feat)

                    return feat


        except Exception as ex:
            logging.warning('Failed to make featrue')