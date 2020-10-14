import numpy as np
from scipy.io import loadmat
from encoders.tools.src.utils import ZSL
from sklearn.preprocessing import normalize

from encoders.vse.src.encoder import Encoder
from encoders.vse.src.encoder import ModelType


def estimate_sem_data(tr_vis_data, te_vis_data, tr_sem_data, res_path):
    tr_sem_data = normalize(tr_sem_data, norm='l2', axis=1, copy=True)
    tr_vis_data = normalize(tr_vis_data, norm='l2', axis=1, copy=True)

    te_vis_data = normalize(te_vis_data, norm='l2', axis=1, copy=True)

    input_length = output_length = tr_vis_data.shape[1] + tr_sem_data.shape[1]
    ae = Encoder(input_length, tr_sem_data.shape[1], output_length, ModelType.ZSL_AE, 50, res_path)
    tr_sem, te_sem = ae.estimate_semantic_data_zsl(tr_vis_data, te_vis_data, tr_sem_data, False)

    return tr_sem, te_sem


# data = loadmat('../Datasets/SEM/cub_demo_data_resnet.mat')
# tr_est, te_est = estimate_sem_data(data['X_tr'], data['X_te'], data['S_tr'], '.')
#
# template_labels = np.array([int(x) for x in data['te_cl_id']])
# test_labels = np.array([int(x) for x in data['test_labels_cub']])

data = loadmat('../Datasets/SEM/awa_demo_data_resnet.mat')
tr_est, te_est = estimate_sem_data(data['X_tr'], data['X_te'], data['S_tr'], '.')

template_labels = np.array([int(x) for x in data['param']['testclasses_id'][0][0]])
test_labels = np.array([int(x) for x in data['param']['test_labels'][0][0]])

acc, _ = ZSL.zsl_el(te_est, data['S_te_pro'], test_labels, template_labels, 1, False)
print(acc)
