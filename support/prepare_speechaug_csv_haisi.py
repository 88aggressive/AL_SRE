# -*- coding:utf-8 -*-

import os
import sys
import logging
import shutil
import glob
import argparse
import torchaudio
from tqdm.contrib import tqdm
import pandas as pd

sys.path.insert(0, 'AL_SRE/support')
from utils import get_torchaudio_backend
torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def prepare_speech_aug(haisi_folder,csv_folder='exp/aug_csv',savewav_folder='data/speech_aug2', max_noise_len=2.015):
    """Prepare the openrir and musan dataset for adding reverb and noises.

    Arguments
    ---------
    openrir_folder,musan_folder : str
        The location of the folder containing the dataset.
    csv_folder : str
        csv file save dir.
    savewav_folder : str
        The processed noise wav save dir.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    if not os.path.isdir(os.path.join(haisi_folder, "haisi_noise")):
        raise OSError("{} is not exist, please prepare it.".format(
            os.path.join(haisi_folder, "haisi_noise")))

    if not os.path.isdir(csv_folder):
        os.makedirs(csv_folder)
    if not os.path.isdir(savewav_folder):
        os.makedirs(savewav_folder)


    haisi_canteen_files = glob.glob(os.path.join(
        haisi_folder, 'haisi_noise/canteen/*.wav'))
    haisi_road_files = glob.glob(os.path.join(
        haisi_folder, 'haisi_noise/road/*.wav'))

    haisi_canteen_item = []
    haisi_road_item = []

    for file in haisi_canteen_files:
        new_filename = os.path.join(savewav_folder, '/'.join(file.split('/')[-3:]))
        haisi_canteen_item.append((file, new_filename))
    for file in haisi_road_files:
        new_filename = os.path.join(savewav_folder, '/'.join(file.split('/')[-3:]))
        haisi_road_item.append((file, new_filename))

    csv_dct={}
    bg_canteen_csv = os.path.join(csv_folder, "haisi_canteen.csv")
    csv_dct[bg_canteen_csv]=haisi_canteen_item
    road_csv = os.path.join(csv_folder, "haisi_road.csv")
    csv_dct[road_csv]=haisi_road_item

    # Prepare csv if necessary
    for csv_file,items in csv_dct.items():

        if not os.path.isfile(csv_file):
            if csv_file in [bg_canteen_csv,road_csv]:
                prepare_aug_csv(items,csv_file,max_length=None)
# ---------------------------------------------------------------------------------------
# concate csv
    combine_canteen_road_csv = os.path.join(csv_folder, "combine_canteen_road.csv")
    concat_csv(combine_canteen_road_csv,bg_canteen_csv,road_csv)

    print("Prepare the speech augment dataset Done, csv files is in {}, wavs in {}.\n".format(csv_folder,savewav_folder))




def prepare_aug_csv(items, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.

    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """

    with open(csv_file, "w") as w:
        w.write("ID duration wav sr tot_frame wav_format\n\n")
        # copy ordinary wav.
        for item in tqdm(items, dynamic_ncols=True):
            if not os.path.isdir(os.path.dirname(item[1])):
                os.makedirs(os.path.dirname(item[1]))
            shutil.copyfile(item[0], item[1])
            filename = item[1]
            # Read file for duration/channel info
            signal, rate = torchaudio.load(filename)
  
            # Ensure only one channel
            if signal.shape[0] > 1:
                signal = signal[0].unsqueeze(0)
                torchaudio.save(filename, signal, rate)

            ID, ext = os.path.basename(filename).split(".")
            duration = signal.shape[1] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)
                for i in range(int(duration / max_length)):
                    start = int(max_length * i * rate)
                    stop = int(
                        min(max_length * (i + 1), duration) * rate
                    )
                    new_filename = (
                        filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                    )
                    torchaudio.save(
                        new_filename, signal[:, start:stop], rate
                    )
                    csv_row = (
                        f"{ID}_{i}",
                        str((stop - start) / rate),
                        new_filename,
                        str(rate),
                        str(stop - start),
                        ext,
                    )
                    w.write(" ".join(csv_row)+'\n')
            else:
                w.write(
                    " ".join((ID, str(duration), filename,str(rate),str(signal.shape[1]), ext))+'\n'
                )

def concat_csv(out_file,*csv_files):
    pd_list = []
    for f in csv_files:
        pd_list.append(pd.read_csv(f, sep=" ",header=0))
    out = pd.concat(pd_list)
    out.to_csv(out_file, sep=" ", header=True, index=False)

if __name__ == '__main__':
    
    # Start
    parser = argparse.ArgumentParser(
        description=""" Prepare speech augmention csv files.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Options
    parser.add_argument("--haisi-folder", type=str, default='/tsdata/ASR',
                    help="where has openslr rir.")
    parser.add_argument("--savewav-folder", type=str, default='/work1/ldx/speech_aug_2_new',
                    help="noise clips for online speechaug, set it in SSD.")
    parser.add_argument("--max-noise-len", type=float, default=2.015,
                    help="the maximum noise length in seconds. Noises longer than this will be cut into pieces")
    parser.add_argument("csv_aug_folder", type=str, help="csv file folder.")


    # End
    print(' '.join(sys.argv))
    args = parser.parse_args()
    assert args.max_noise_len > 0.4

    prepare_speech_aug(args.haisi_folder,
                        csv_folder=args.csv_aug_folder, \
                        savewav_folder=args.savewav_folder, \
                        max_noise_len=args.max_noise_len)
