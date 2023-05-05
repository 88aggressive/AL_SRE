#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import tqdm
import pandas as pd
import numpy as np
import torchaudio
import random

def findAllSeqs_all(dirName,
                extension='.wav',
                speaker_level=1,
                ):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = [] #(label,path)
    print("dir_name",dirName) #/mnt/a3/cache/database/voxceleb/vox2/dev/acc/
    print(f"finding {extension}, Waiting...")

    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        # print("root, dirs, filenames:", root, dirs, filenames)
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level]) ##控制spk_id:id03439,id03439/tybcebyc
            # print("str:", speakerStr)
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            # print("speaker:", speaker,speakerStr)
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]
    for key, index in speakersTarget.items():
        outSpeakers[index] = key #[id00,id01]
    return outSequences, outSpeakers


#/home2/database/data_sre/task1/cn_1/data/id00000/singing-01-001.flac
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', help='file extension name', type=str, default="wav")
    parser.add_argument('--data_dir', help='训练数据集路径', type=str, default="data")
    parser.add_argument('--save_list_path', help='保存csv文件路径', type=str, default="data_list")
    parser.add_argument('--speaker_level', help='说话人id', type=int, default=1)
    parser.add_argument('--valid_utts', help='用于valid的语音条数', type=int, default=1024)
    args = parser.parse_args()

    outSequences, outSpeakers = findAllSeqs_all(args.data_dir,
                extension=args.extension,
                speaker_level=args.speaker_level)

    outSequences = np.array(outSequences, dtype=str)
    class_label = outSequences.T[0].astype(int)
    utt_paths = outSequences.T[1]

    eg_id = [] #
    for i in class_label:
        eg_id.append(outSpeakers[i])

    print("Spker ID   : {}".format(eg_id[0]))
    print("Utter Path : {}".format(utt_paths[0]))
    print("Spker Label  : {}".format(class_label[0]))

    chunk_samples = []
    # head = ['eg-id','wav-path', 'start-position', 'end-position', 'class-label']
    head = ['eg-id', 'wav-path', 'class-label']
    for i in range(len(utt_paths)):

        chunk = "{0} {1} {2}".format(eg_id[i],utt_paths[i],class_label[i])

        chunk_samples.append(chunk.split())

    random.shuffle(chunk_samples)
    valid_samples = chunk_samples[:args.valid_utts]
    train_samples = chunk_samples[args.valid_utts:]

    df_train = pd.DataFrame(train_samples,columns=head)
    df_valid = pd.DataFrame(valid_samples, columns=head)

    try:
        train_list_path = args.save_list_path
        df_train.to_csv(f"{train_list_path}/train.csv",sep=" ", header=True,index=False)
        print(f'Saved train data list file at {args.save_list_path}')

        valid_list_path = args.save_list_path
        df_valid.to_csv(f"{train_list_path}/valid.csv",sep=" ", header=True,index=False)
        print(f'Saved valid data list file at {args.save_list_path}')


    except OSError as err:
        print(f'Ran in an error while saving {args.save_list_path}: {err}')