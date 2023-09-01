import numpy as np
from scipy.stats import multivariate_normal

file_dir = "./figures/no_transfer/"
feats_fn = "FEATS_test.csv"
preds_fn = "PREDS_test.csv"

FEATS_data = open(f'{file_dir}{feats_fn}')
FEATS = np.loadtxt(FEATS_data, delimiter=",")
#print(FEATS.shape)

PREDS_data = open(f'{file_dir}{preds_fn}')
PREDS = np.loadtxt(PREDS_data, delimiter=",")
#print(PREDS.shape)

labels = []
Y_0_FEATS = [] #93 x 2048
Y_1_FEATS = [] #107 x 2048
print(FEATS.shape)
for i in range(PREDS.shape[0]):
    if PREDS[i,:][0] < PREDS[i,:][1]:
        labels.append(0)
        Y_0_FEATS.append(FEATS[i,:])
    else:
        labels.append(1)
        Y_1_FEATS.append(FEATS[i,:])
Y_0_FEATS = np.stack(Y_0_FEATS, axis=0)
Y_1_FEATS = np.stack(Y_1_FEATS, axis=0)

Y_0_cov = np.cov(Y_0_FEATS.T, bias=True)
Y_0_mean = np.mean(Y_0_FEATS,axis=0)
Y_0_mvg = multivariate_normal(Y_0_mean, Y_0_cov, allow_singular=True)
Y_1_cov = np.cov(Y_1_FEATS.T, bias=True)
Y_1_mean = np.mean(Y_1_FEATS,axis=0)
Y_1_mvg = multivariate_normal(Y_1_mean, Y_1_cov, allow_singular=True)

def calculate_PYZ(sample, label):
    Y0 = Y_0_mvg.pdf(sample)*0.5 #priors are assumed to be 0.5
    Y1 = Y_1_mvg.pdf(sample)*0.5
    print(Y0, Y1)
    if label == 0:
        return Y0/(Y0+Y1)
    else:
        return Y1/(Y0+Y1)

PY_Z = []
for i in range(10):
    PY_Z.append(calculate_PYZ(FEATS[i,:], labels[i]))

print(PY_Z)