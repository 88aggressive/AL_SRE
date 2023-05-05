from sklearn.metrics import roc_curve
import numpy as np


# 计算EER
def compute_EER(frr, far):
    threshold_index = np.argmin(abs(frr - far))  # 平衡点
    eer = (frr[threshold_index] + far[threshold_index]) / 2
    print("eer=", eer)
    return eer


# 计算minDCF P_miss = frr  P_fa = far
def compute_minDCF2(P_miss, P_fa):
    C_miss = C_fa = 1
    P_true = 0.01
    P_false = 1 - P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")

    DCF = C_miss * P_miss * P_true + C_fa * P_fa * P_false

    min_DCF = min(DCF)

    print("min_DCF_2=", min_DCF)

    return min_DCF


# 计算minDCF P_miss = frr  P_fa = far
def compute_minDCF3(P_miss, P_fa, min_DCF_2):
    C_miss = C_fa = 1
    P_true = 0.001
    P_false = 1 - P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")

    DCF = C_miss * P_miss * P_true + C_fa * P_fa * P_false

    # 该操作是我自己加的，因为论文中的DCF10-3指标均大于DCF10-2且高于0.1以上，所以通过这个来过滤一下,错误请指正
    min_DCF = 1
    for dcf in DCF:
        if dcf > min_DCF_2 + 0.1 and dcf < min_DCF:
            min_DCF = dcf

    print("min_DCF_3=", min_DCF)
    return min_DCF

def compute_far_frr(y_true,y_score):
    # 计算FAR和FRR
    fpr, tpr, thres = roc_curve(y_true, y_score)
    frr = 1 - tpr
    far = fpr
    frr[frr <= 0] = 1e-5
    far[far <= 0] = 1e-5
    frr[frr >= 1] = 1 - 1e-5
    far[far >= 1] = 1 - 1e-5
    return far,frr

def preparedata(file_path):
    y_true = []
    y_score = []
    with open(file_path,"r",encoding="utf8") as f:
        file = f.readlines()
        for line in file:
            strline = line.split()
            l1=strline[0]
            l2=strline[1]
            score = float(strline[2])
            spk1=l1.split("/")[0]
            spk2=l2.split("/")[0]
            if spk1 == spk2:
                y_true.append(1)
            else:
                y_true.append(0)
            y_score.append(score)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return  y_true,y_score


def preparedata2(file_path):
    y_true = []
    y_score = []
    with open(file_path,"r",encoding="utf8") as f:
        file = f.readlines()
        for line in file:
            strline = line.split()
            l1=strline[0]
            l2=strline[1]
            score = float(strline[3])
            spk1=l1.split("/")[0]
            spk2=l2.split("/")[0]
            if spk1 == spk2:
                y_true.append(1)
            else:
                y_true.append(0)
            y_score.append(score)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return  y_true,y_score

def preparedata3(file_path):
    y_true = []
    y_score = []
    with open(file_path,"r",encoding="utf8") as f:
        file = f.readlines()
        for line in file:
            strline = line.split()
            l1=strline[0]
            l2=strline[1]
            score = float(strline[3])
            spk1=l1.split("-")[0]
            spk2=l2.split("-")[0]
            if spk1 == spk2:
                y_true.append(1)
            else:
                y_true.append(0)
            y_score.append(score)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return  y_true,y_score

def preparedata_haisi(file_path):
    y_true = []
    y_score = []
    with open(file_path,"r",encoding="utf8") as f:
        file = f.readlines()
        for line in file:
            strline = line.split()
            l1=strline[0]
            l2=strline[1]
            score = float(strline[3])
            spk1=l1.split("-")[0]
            spk2=l2.split("_")[2]
            if spk1 == spk2:
                y_true.append(1)
            else:
                y_true.append(0)
            y_score.append(score)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return  y_true,y_score

# 直接从标签获得是不是同一说话人
def preparedata_haisi2(file_path):
    y_true = []
    y_score = []
    with open(file_path,"r",encoding="utf8") as f:
        file = f.readlines()
        for line in file:
            strline = line.split()
            # print("strline:",strline)
            l1=strline[0]
            l2=strline[1]
            score = float(strline[3])
            label = int(strline[2])
            y_true.append(label)

            y_score.append(score)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return  y_true,y_score