#!/usr/bin/env python

import os
import sys
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, "AL_SRE")
sys.path.insert(0, os.getcwd())
os.getcwd()
from support.load_model import *
from dataio.dataloader import *
from support.compute_cosine import *

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
        description="""Train xvector framework with pytorch.""",
        formatter_class=argparse.RawTextHelpFormatter,
        conflict_handler='resolve')

parser.add_argument("--save_dir", type=str, default="exp/test",
                    help="save file.")

parser.add_argument("--epochs", type=int, default=20,
                    help="get model.")

parser.add_argument("--enroll_spks", type=int, default=100,
                    help="enroll_nums.")

parser.add_argument("--enroll_nums", type=int, default=5,
                    help="enroll_nums.")

parser.add_argument("--conf_file", type=str, default="config/conf_ecapa.yaml",
                    help="conf file")

parser.add_argument("--trials_type", type=str, default="vox",
                    help="trials_type")

parser.add_argument("--train_dir", type=str, default="data",
                    help="trails dir")

parser.add_argument("--wav_root", type=str, default="data",
                    help="conf file")

parser.add_argument('--speech_aug', type=str, default=False, choices=["true", "false"],
                    help='Use speech aug test')

parser.add_argument("--gpu_id", type=str, default="",
                    help="set use gpu-id.")

args = parser.parse_args()

egs_conf=args.conf_file

## Main params
gpu_id = args.gpu_id
epochs = args.epochs
save_dir = args.save_dir
suffix = "params"

trials_type = args.trials_type
enroll_nums = args.enroll_nums
wav_root = args.wav_root
# print("wav_root",wav_root)
# exit()

model_path = f"{save_dir}/{epochs}.{suffix}"
save_file = f"{save_dir}/scores_{epochs}_{trials_type}_spks{args.enroll_spks}_utts{enroll_nums}_{args.speech_aug}_clean.txt"
#### Set seed
utils.set_all_seed(1024)
# Cosine similarity initialization
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


##1.定义model，导入训练好参数
device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model,_ = get_model(save_dir,egs_conf)
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
model = model.to(device)

with open(egs_conf,'r') as fin:
    egs_params = yaml.load(fin, Loader=yaml.FullLoader)
    conf = egs_params['dataset_conf']
feature_extraction_conf = conf.get('feature_extraction_conf', {})
kaldi_reader = KaldiFeature_test(**feature_extraction_conf)

##1. 得到每个说话人注册模型和对应spk_id，save字典{"emds":,"spks":}
spk_file = f"{args.train_dir}/list3/enroll_spks{args.enroll_spks}_utts{args.enroll_nums}.lst"
save_spk_path = f"{save_dir}/emds_spks_al_{epochs}_spks{args.enroll_spks}_utts{enroll_nums}_20.npy"

trials = f"{args.train_dir}/list3/test_spks{args.enroll_spks}_utts{args.enroll_nums}_clean.lst"

if not os.path.exists(save_spk_path):
    spk2utt = read_file(spk_file,wav_root)
    get_spk_model(save_spk_path,model,device,spk2utt,kaldi_reader)

def get_dur(wav_path):
    wav, sample_rate = torchaudio.load(wav_path)
    if len(wav.shape) == 1:
        # add channel
        wav = wav.unsqueeze(0)
    wav_dur = wav.shape[1]/sample_rate
    return wav_dur

time_start = time.time()
wav_len_s = 0.0

##2. 打分得到结果
logger.info("Get embedding and score..")

emd_spk_data = np.load(save_spk_path, allow_pickle=True)
emds = emd_spk_data.item().get("emds")
spks = emd_spk_data.item().get("spks")

enroll_emds = torch.from_numpy(emds)

## if not os.path.exists(save_file):
with open(save_file,"w") as s_file:
    # Reading standard verification split
    with open(trials) as f:
        veri_test = [line.rstrip() for line in f]

    speech_aug = args.speech_aug  # True,False
    aug_reader = None
    if speech_aug == "true":
        print("add speech augmentation to wav")
        speech_aug_conf_file = conf.get('speech_aug_conf', '')
        assert speech_aug_conf_file !=''
        with open(speech_aug_conf_file, 'r') as fin:
            speech_aug_conf = yaml.load(
                fin, Loader=yaml.FullLoader)

            csv_aug_folder = conf.get('csv_aug_folder', '')
            if csv_aug_folder: utils.change_csv_folder(speech_aug_conf, csv_aug_folder)
        aug_reader = SpeechAug_test(**speech_aug_conf)

    for line in tqdm(veri_test):
        # print("line:",line)
        wav_path = line
        # Reading verification file (label enrol_file test_file)
        line = line.strip().split('/')
        spkid = line[0]
        # spkid = line[0].split('_')[0]

        test_path = os.path.join(wav_root, wav_path)
        test_e = get_embedding_model_kaldi(test_path, model, device,kaldi_reader,aug_reader=aug_reader)

        for idy, enroll_e in enumerate(enroll_emds):
            score = similarity(test_e, enroll_e)

            spk_model = spks[idy]
            if spk_model == spkid:
                lab_pair = 1
            else:
                lab_pair = 0
            # print("line,spkid,spk_model:",line, spkid,spk_model,lab_pair, score)
            # exit()
            s_file.write("%s %s %i %f\n" % (spk_model, line[1], lab_pair, score))


logger.info("Computing EER..")
y_labels, y_scores = preparedata_haisi2(save_file)

far, frr = compute_far_frr(y_labels, y_scores)  # far

eer = compute_EER(frr, far)
min_dcf = compute_minDCF2(frr * 100, far * 100)

eer = eer * 100

print("EER(%)=",eer)
print("minDCF=", min_dcf)
time_end = time.time()
time_sum = time_end - time_start
# # rt = time_sum/wav_len_s
# # print("real_time={0},wav_len_s={1},RT={2}".format(time_sum,wav_len_s,rt))
# save_eer = f"{save_dir}/eer_{epochs}_{trials_type}_spks{args.enroll_spks}_utts{args.enroll_nums}_{args.speech_aug}_clean.txt"
# with open(save_eer,"w") as eer_f:
#     eer_f.write("EER(%) \t minDCF(0.01)\n")
#     eer_f.write("%f"%eer)
#     eer_f.write("\t")
#     eer_f.write("%f"%min_dcf)
#
# # ##canteen
# save_file = f"{save_dir}/scores_{epochs}_{trials_type}_spks{args.enroll_spks}_utts{enroll_nums}_{args.speech_aug}_canteen.txt"
# trials = f"{args.train_dir}/list3/test_spks{args.enroll_spks}_utts{args.enroll_nums}_canteen.lst"
#
# with open(save_file,"w") as s_file:
#     # Reading standard verification split
#     with open(trials) as f:
#         veri_test = [line.rstrip() for line in f]
#
#     aug_reader = None
#
#     for line in tqdm(veri_test):
#         wav_path = line
#         # Reading verification file (label enrol_file test_file)
#         line = line.strip().split('/')
#         spkid = line[0]
#         # spkid = line[0].split('-')[0]
#
#         test_path = os.path.join(wav_root, wav_path)
#         test_e = get_embedding_model_kaldi(test_path, model, device,kaldi_reader,aug_reader=aug_reader)
#
#         for idy, enroll_e in enumerate(enroll_emds):
#             score = similarity(test_e, enroll_e)
#
#             spk_model = spks[idy]
#             if spk_model == spkid:
#                 lab_pair = 1
#             else:
#                 lab_pair = 0
#             # print("line,spkid,spk_model:",line, spkid,spk_model,lab_pair, score)
#             # exit()
#             s_file.write("%s %s %i %f\n" % (spk_model, line[1], lab_pair, score))
#
# logger.info("Computing EER..")
# y_labels, y_scores = preparedata_haisi2(save_file)
#
# far, frr = compute_far_frr(y_labels, y_scores)  # far
#
# eer = compute_EER(frr, far)
# min_dcf = compute_minDCF2(frr * 100, far * 100)
#
# eer = eer * 100
#
# print("EER(%)=",eer)
# print("minDCF=", min_dcf)
# time_end = time.time()
# time_sum = time_end - time_start
# # rt = time_sum/wav_len_s
# # print("real_time={0},wav_len_s={1},RT={2}".format(time_sum,wav_len_s,rt))
# save_eer = f"{save_dir}/eer_{epochs}_{trials_type}_spks{args.enroll_spks}_utts{args.enroll_nums}_{args.speech_aug}_canteen.txt"
# with open(save_eer,"w") as eer_f:
#     eer_f.write("EER(%) \t minDCF(0.01)\n")
#     eer_f.write("%f"%eer)
#     eer_f.write("\t")
#     eer_f.write("%f"%min_dcf)
#
# # ##road
# save_file = f"{save_dir}/scores_{epochs}_{trials_type}_spks{args.enroll_spks}_utts{enroll_nums}_{args.speech_aug}_road.txt"
# trials = f"{args.train_dir}/list3/test_spks{args.enroll_spks}_utts{args.enroll_nums}_road.lst"
#
# with open(save_file,"w") as s_file:
#     # Reading standard verification split
#     with open(trials) as f:
#         veri_test = [line.rstrip() for line in f]
#
#     aug_reader = None
#
#     for line in tqdm(veri_test):
#         wav_path = line
#         # Reading verification file (label enrol_file test_file)
#         line = line.strip().split('/')
#         spkid = line[0]
#         # spkid = line[0].split('-')[0]
#
#         test_path = os.path.join(wav_root, wav_path)
#
#         try:
#             test_e = get_embedding_model_kaldi(test_path, model, device,kaldi_reader,aug_reader=aug_reader)
#
#             for idy, enroll_e in enumerate(enroll_emds):
#                 score = similarity(test_e, enroll_e)
#
#                 spk_model = spks[idy]
#                 if spk_model == spkid:
#                     lab_pair = 1
#                 else:
#                     lab_pair = 0
#                 # print("line,spkid,spk_model:",line, spkid,spk_model,lab_pair, score)
#                 # exit()
#                 s_file.write("%s %s %i %f\n" % (spk_model, line[1], lab_pair, score))
#         except:
#             print("can open file ",test_path)
#
#
# logger.info("Computing EER..")
# y_labels, y_scores = preparedata_haisi2(save_file)
#
# far, frr = compute_far_frr(y_labels, y_scores)  # far
#
# eer = compute_EER(frr, far)
# min_dcf = compute_minDCF2(frr * 100, far * 100)
#
# eer = eer * 100
#
# print("EER(%)=",eer)
# print("minDCF=", min_dcf)
# time_end = time.time()
# time_sum = time_end - time_start
# # rt = time_sum/wav_len_s
# # print("real_time={0},wav_len_s={1},RT={2}".format(time_sum,wav_len_s,rt))
# save_eer = f"{save_dir}/eer_{epochs}_{trials_type}_spks{args.enroll_spks}_utts{args.enroll_nums}_{args.speech_aug}_road.txt"
# with open(save_eer,"w") as eer_f:
#     eer_f.write("EER(%) \t minDCF(0.01)\n")
#     eer_f.write("%f"%eer)
#     eer_f.write("\t")
#     eer_f.write("%f"%min_dcf)