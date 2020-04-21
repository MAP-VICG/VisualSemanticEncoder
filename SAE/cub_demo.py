"""
Following code shows a demo for aPY dataset to reproduce the result of the paper:

Semantic Autoencoder for Zero-shot Learning.

Elyor Kodirov, Tao Xiang, and Shaogang Gong, CVPR 2017.

Code originally written in Matlab and transformed to Python by Damares Resende
"""

import numpy as np
from scipy.io import loadmat
from utils import ZSL, Params
from numpy.linalg import matrix_power

cub = loadmat('../../Datasets/SAE/cub_demo_data.mat')

# Dimension reduction
X_tr = cub['X_tr']
M, N = X_tr.transpose().dot(X_tr).shape
Y = ZSL.label_matrix(cub['train_labels_cub'].transpose()).transpose()
W = matrix_power(X_tr.transpose().dot(X_tr) + 50 * np.eye(M, N), -1).dot(X_tr.transpose()).dot(Y)

X_tr = X_tr.dot(W)
X_te = cub['X_te'].dot(W)

lambda_ = .2
param = Params(1, cub, 'cub')

# Learn projection
sem_atts_train = ZSL.kill_semantic_attributes(np.array(cub['S_tr']), 1)

S_tr = ZSL.normalize_fea(sem_atts_train)
W = ZSL.SAE(X_tr.transpose(), S_tr.transpose(), lambda_).transpose()

# [F --> S], projecting data from feature space to semantic space
S_te_est = X_te.dot(W)
zsl_accuracy, _ = ZSL.zsl_el(S_te_est, cub['S_te_pro'], param, 'cub')
print('\n[1] CUB ZSL accuracy [V >>> S]: %.1f%%\n' % (zsl_accuracy * 100))

# [S --> F], projecting from semantic to visual space
X_te_pro = cub['S_te_pro'].dot(W.transpose())
zsl_accuracy, _ = ZSL.zsl_el(X_te, X_te_pro, param, 'cub')

print('[2] CUB ZSL accuracy [S >>> V]: %.1f%%\n' % (zsl_accuracy * 100))
