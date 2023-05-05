#!/bin/bash


##训练、测试数据集路径
trainset_root="/home/niumq20/data/voxceleb/vox2_wav/dev/aac"
testset_root="/home/niumq20/data/voxceleb/vox1/test/wav"
testset_root_key="/home2/dataset/ALSpeech/keywords_audio"
#train_data='data/sre_vox1_test'
train_csv='data/sre_vox2'
trainset_root="/home2/dataset/ALSpeech/keywords_audio"
trainset_root="/home2/database/voxceleb/voxceleb1/dev/wav"

#train_csv='data/vox1_haisi'

save_dir="exp/haisi/aug_vox1_haisi_resnet_sgd_random_vad_ppl_noise0_15"

epochs=60
conf_file="AL_SRE/config/conf_resnet_ppl22.yaml"

##voxceleb
#trials="data/veri_test2.txt"
speech_aug=true  ##测试数据是否加噪

##aug
haisi_noise="/home2/dataset/ALSpeech/noise"
openrir_folder="./data"  # where has openslr rir. (eg. ./data/RIRS_NOISES, then, openrir_folder="./data")
musan_folder="./data"    # where contains musan folder.
csv_aug_folder="exp/aug_csv2"      # csv file location.
savewav_folder="exp/data/noise2"  # save the noise seg into SSD.

gpu_id=1

stage=5
endstage=5

if [[ $stage -le 1 && 1 -le $endstage ]]; then
#  testset_root="/mnt/x2/database/al-speaker-dataset/dev"
#  echo "prepare data for model training"
#  mkdir -p $train_csv
#  echo Build $trainset_root list
#  python3 AL_SRE/build_datalist_wav_chunk_vox1.py \
#          --data_dir $trainset_root \
#          --extension wav \
#          --speaker_level 1 \
#          --save_list_path $train_csv \
#          --valid_utts 1000 \
#          --enroll_spks 100 \
#          --enroll_utts 5

#  echo Build test list
#  python3 AL_SRE/create_trials_al.py \
#          --testset_root $testset_root_key \
#          --save_list_path $train_csv \
#          --extension wav \
#          --speaker_level 1 \
#          --enroll_spks 100 \
#          --enroll_utts 5
#
  echo "prepare speech augmention csv files."
  python3 AL_SRE/support/prepare_speechaug_csv_haisi.py \
    --haisi-folder=$haisi_noise \
    --savewav-folder=$savewav_folder \
    --max-noise-len=2.015 \
    $csv_aug_folder

fi


if [[ $stage -le 2 && 2 -le $endstage ]]; then
  echo "start train stage..."
  python AL_SRE/train_ppl.py \
      --conf_file $conf_file \
      --gpu_id=$gpu_id \
      --save_dir $save_dir \
      --epochs $epochs \
      --train_data $train_csv
  echo "train done"
fi

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

if [[ $stage -le 5 && 5 -le $endstage ]]; then
  testset_root_key="/mnt/a3/cache/dataset/test-20-people"
#  testset_root_key="/home2/dataset/ALSpeech/haisi/test-20-people"
#  echo Build test list
#  python3 AL_SRE/create_trials_haisi2.py \
#          --testset_root $testset_root_key \
#          --extension wav \
#          --speaker_level 2 \
#          --enroll_spks 100 \
#          --enroll_utts 5

  echo "start test stage"
  python AL_SRE/test_model_test_haisi2.py \
      --conf_file $conf_file \
      --gpu_id=$gpu_id \
      --save_dir $save_dir \
      --epochs $epochs \
      --trials_type "key" \
      --wav_root $testset_root_key \
      --speech_aug true \
      --enroll_spks 100 \
      --enroll_nums 5
fi