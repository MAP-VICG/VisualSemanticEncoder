"""
Following code shows a demo for aPY dataset to reproduce the result of the paper:

Semantic Autoencoder for Zero-shot Learning.

Elyor Kodirov, Tao Xiang, and Shaogang Gong, CVPR 2017.

Code originally written in Matlab and transformed to Python by Damares Resende
"""
import numpy as np
from scipy.io import loadmat
from utils import ZSL, Params

awa = loadmat('../../Datasets/SAE/awa_demo_data.mat')

X_tr = np.array(awa['X_tr'])
S_tr = sem_atts_train = ZSL.kill_semantic_attributes(np.array(awa['S_tr']), 1)

# with open('test.txt', 'w+') as f:
#     for att in S_tr:
#         f.write(' '.join(list(map(str, att))) + '\n')

X_te = np.array(awa['X_te'])
S_te_gt = np.array(awa['S_te_gt'])
S_te_pro = np.array(awa['S_te_pro'])

X_tr = ZSL.normalize_fea(X_tr.transpose()).transpose()

lambda_ = 500000
W = ZSL.SAE(X_tr.transpose(), S_tr.transpose(), lambda_)
param = Params(1, awa, 'awa')

# [F --> S], projecting data from feature space to semantic space
S_est = X_te.dot(ZSL.normalize_fea(W).transpose())
zsl_accuracy, _ = ZSL.zsl_el(S_est, S_te_gt, param, 'awa')

print('\n[1] AwA ZSL accuracy [V >>> S]: %.1f%%\n' % (zsl_accuracy * 100))

# [S --> F], projecting from semantic to visual space
X_te_pro = ZSL.normalize_fea(S_te_pro.transpose()).transpose().dot(ZSL.normalize_fea(W))
zsl_accuracy, _ = ZSL.zsl_el(X_te, X_te_pro, param, 'awa')

print('[2] AwA ZSL accuracy [S >>> V]: %.1f%%\n' % (zsl_accuracy * 100))
