#!/bin/bash


##训练、测试数据集路径
trainset_root="/home/niumq20/data/voxceleb/vox2_wav/dev/aac"
testset_root="/home/niumq20/data/voxceleb/vox1/test/wav"
#testset_root_key="/home2/dataset/ALSpeech/keywords_audio"
testset_root_key="/mnt/a3/cache/dataset/keywords_audio"
##训练数据列表保存路径
train_csv='data/sre_vox2'

save_dir="exp/haisi/test_aug_vox2_dev_vad_snowdar_resnet_sgd_tqdm"
#save_dir="exp/ecapa_fbank80_online_e6"

epochs=10
conf_file="AL_SRE/config/conf_resnet.yaml"
##voxceleb
trials="data/veri_test2.txt"
speech_aug=false  ##测试数据是否加噪
##aug
openrir_folder="/home/niumq20/data"  # where has openslr rir. (eg. ./data/RIRS_NOISES, then, openrir_folder="./data")
musan_folder="/home/niumq20/data"    # where contains musan folder.
csv_aug_folder="exp/aug_csv"      # csv file location.
savewav_folder="exp/data/noise"  # save the noise seg into SSD.

gpu_id=0

stage=1
endstage=4

if [[ $stage -le 1 && 1 -le $endstage ]]; then
#  testset_root="/mnt/x2/database/al-speaker-dataset/dev"
#   prepare data for model training
  mkdir -p $train_csv
  echo Build $trainset_root list
  python3 AL_SRE/build_datalist_wav.py \
          --data_dir $trainset_root \
          --extension wav \
          --speaker_level 1 \
          --save_list_path $train_csv \
          --valid_utts 1000

  echo Build test list
  mkdir -p data/list
  python3 AL_SRE/create_trials_al.py \
          --testset_root $testset_root_key \
          --save_list_path data \
          --extension wav \
          --speaker_level 1 \
          --enroll_spks 100 \
          --enroll_utts 5

  echo "prepare speech augmention csv files."
  python3 AL_SRE/support/prepare_speechaug_csv.py \
    --openrir-folder=$openrir_folder \
    --musan-folder=$musan_folder \
    --savewav-folder=$savewav_folder \
    --max-noise-len=2.015 \
    $csv_aug_folder
fi

if [[ $stage -le 2 && 2 -le $endstage ]]; then
  echo "start train stage..."
  python AL_SRE/train.py \
      --conf_file $conf_file \
      --gpu_id=$gpu_id \
      --save_dir $save_dir \
      --epochs $epochs \
      --train_data $train_csv

  echo "train done"

fi

#if [[ $stage -le 3 && 3 -le $endstage ]]; then
#  python AL_SRE/test.py \
#      --conf_file $conf_file \
#      --gpu_id=$gpu_id \
#      --save_dir $save_dir \
#      --epochs $epochs \
#      --trials_type "vox" \
#      --trials $trials \
#      --wav_root $testset_root
#fi

if [[ $stage -le 4 && 4 -le $endstage ]]; then
  echo "start test stage"
  python AL_SRE/test_model_test.py \
      --conf_file $conf_file \
      --gpu_id=$gpu_id \
      --save_dir $save_dir \
      --epochs $epochs \
      --trials_type "key" \
      --wav_root $testset_root_key \
      --speech_aug $speech_aug \
      --enroll_spks 100 \
      --enroll_nums 5
fi