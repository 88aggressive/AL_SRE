#!/usr/bin/env python

import os
import sys
import time
import argparse
from tqdm import tqdm
from torchsummary import summary

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

parser.add_argument("--conf_file", type=str, default="config/conf_ecapa.yaml",
                    help="conf file")

parser.add_argument("--trials_type", type=str, default="vox",
                    help="trials_type")

parser.add_argument("--trials", type=str, default="data/veri_test2.txt",
                    help="conf file")

parser.add_argument("--wav_root", type=str, default="data",
                    help="conf file")

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
trials = args.trials
wav_root = args.wav_root
#ecapa_tdnn
model_path = f"{save_dir}/{epochs}.{suffix}"
# save_dir = "exp_x3"
save_file = f"{save_dir}/scores_{epochs}_{trials_type}.txt"
#### Set seed
utils.set_all_seed(1024)
# Cosine similarity initialization
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def get_dur(wav_path):
    wav, sample_rate = torchaudio.load(wav_path)
    if len(wav.shape) == 1:
        # add channel
        wav = wav.unsqueeze(0)
    wav_dur = wav.shape[1]/sample_rate
    return wav_dur

# mean_xvector = np.load(f"{save_dir}/mean_xvector.npy", allow_pickle=True).item().get('xvector')
# print("load mean xvector")
# print("mean_xvector",mean_xvector)
# exit()
# s_file = os.open(save_file,'w')
time_start = time.time()
wav_len_s = 0.0

##1.定义model，导入训练好参数
device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model,_ = get_model(save_dir,egs_conf)
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
# model = model.to(device)
# for k,v in model.state_dict.items():
#     print(k)
#     print("1", k.par)
# for param in model.parameters():
#     print("param:",param)
total = sum([param.nelement() for param in model.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))

# print(summary(model, (80, 200), device="cpu"))
#
# exit()

with open(egs_conf,'r') as fin:
    egs_params = yaml.load(fin, Loader=yaml.FullLoader)
    conf = egs_params['dataset_conf']
feature_extraction_conf = conf.get('feature_extraction_conf', {})
kaldi_reader = KaldiFeature_test(**feature_extraction_conf)
##2.打分得到结果
# Computes positive and negative scores given the verification split.
logger.info("Get embedding and score..")

with open(save_file,"w") as s_file:

    # Reading standard verification split
    with open(trials) as f:
        veri_test = [line.rstrip() for line in f]

    for line in tqdm(veri_test):
        # Reading verification file (label enrol_file test_file)
        line = line.strip().split()
        lab_pair = int(line[0])
        enrol = line[1]
        test = line[2]
        enrol_path = os.path.join(wav_root, enrol)
        test_path = os.path.join(wav_root, test)
        ##get utt_dur
        # wav_len_s += get_dur(enrol_path)
        # wav_len_s += get_dur(test_path)

        # embed_e = get_embedding(enrol_path, model,device)
        # embed_t = get_embedding(test_path, model, device)

        embed_e = get_embedding_model(enrol_path, model,device,kaldi_reader)
        embed_t = get_embedding_model(test_path, model, device,kaldi_reader)

        ##去均值
        # enr_e_mean = embed_e.cpu().numpy() - mean_xvector
        # embed_e = torch.from_numpy(enr_e_mean)
        # test_e_mean = embed_t.cpu().numpy() - mean_xvector
        # embed_t = torch.from_numpy(test_e_mean)

        # Compute the score for the given sentence
        score = similarity(embed_e, embed_t) #[0]

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol, test, lab_pair, score))


logger.info("Computing EER..")
y_labels, y_scores = preparedata2(save_file)

far, frr = compute_far_frr(y_labels, y_scores)  # far

eer = compute_EER(frr, far)
min_dcf = compute_minDCF2(frr * 100, far * 100)

eer = eer * 100

print("EER(%)=",eer)
print("minDCF=", min_dcf)
time_end = time.time()
time_sum = time_end - time_start
# rt = time_sum/wav_len_s
# print("real_time={0},wav_len_s={1},RT={2}".format(time_sum,wav_len_s,rt))
save_eer = f"{save_dir}/eer_{epochs}_{trials_type}.txt"
with open(save_eer,"w") as eer_f:
    eer_f.write("EER(%) \t minDCF(0.01)\n")
    eer_f.write("%f"%eer)
    eer_f.write("\t")
    eer_f.write("%f"%min_dcf)

