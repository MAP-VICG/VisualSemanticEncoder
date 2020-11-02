import numpy as np
from random import randrange
from scipy.io import loadmat
from sklearn.decomposition import PCA
from encoders.tools.src.utils import ZSL
from sklearn.preprocessing import normalize

from encoders.vse.src.encoder import ModelType, ModelFactory, Encoder


def get_te_sem_data(data, _type):
    if _type == "awa":
        lbs = {data['param']['testclasses_id'][0][0][j][0]: attrs for j, attrs in enumerate(data['S_te_pro'])}
        return np.array([lbs[label[0]] for label in data['param']['test_labels'][0][0]])
    elif _type == "cub":
        lbs = {data['te_cl_id'][j][0]: attributes for j, attributes in enumerate(data['S_te_pro'])}
        return np.array([lbs[label[0]] for label in data['test_labels_cub']])
    else:
        raise ValueError("Invalid data type.")


def estimate_sem_data_pca(data):
    pca = PCA(n_components=data['S_te'].shape[1])
    tr_vis_data = normalize(data['X_tr'], norm='l2', axis=1, copy=True)
    tr_sem_data = normalize(data['S_tr'], norm='l2', axis=1, copy=True)

    te_vis_data = normalize(data['X_te'], norm='l2', axis=1, copy=True)
    te_sem_data = normalize(data['S_te'], norm='l2', axis=1, copy=True)

    train_data = np.hstack((tr_vis_data, tr_sem_data))
    test_data = np.hstack((te_vis_data, te_sem_data))
    pca.fit(train_data)
    return pca.transform(train_data), pca.transform(test_data)


def estimate_sem_data_cat(data):
    tr_vis_data = normalize(data['X_tr'], norm='l2', axis=1, copy=True)
    tr_sem_data = normalize(data['S_tr'], norm='l2', axis=1, copy=True)

    te_vis_data = normalize(data['X_te'], norm='l2', axis=1, copy=True)
    te_sem_data = normalize(data['S_te'], norm='l2', axis=1, copy=True)

    train_data = np.hstack((tr_vis_data, tr_sem_data))
    test_data = np.hstack((te_vis_data, te_sem_data))
    return train_data, test_data


def estimate_sem_data_vse(data):
    tr_vis_data = normalize(data['X_tr'], norm='l2', axis=1, copy=True)
    tr_sem_data = normalize(data['S_tr'], norm='l2', axis=1, copy=True)

    te_vis_data = normalize(data['X_te'], norm='l2', axis=1, copy=True)
    te_sem_data = normalize(data['S_te'], norm='l2', axis=1, copy=True)

    train_data = np.hstack((tr_vis_data, tr_sem_data))
    test_data = np.hstack((te_vis_data, te_sem_data))

    input_length = output_length = train_data.shape[1]
    model = ModelFactory(input_length, tr_sem_data.shape[1], output_length)(ModelType.STRAIGHT_AE, run_svm=False)
    model.fit(train_data, data['train_labels'], test_data, data['test_labels'], 50, '.', save_weights=False)

    return model.predict(tr_vis_data, tr_sem_data, te_vis_data, te_sem_data)


def estimate_sem_data_sec(data):
    tr_vis_data = normalize(data['X_tr'], norm='l2', axis=1, copy=True)
    tr_sem_data = normalize(data['S_tr'], norm='l2', axis=1, copy=True)

    te_vis_data = normalize(data['X_te'], norm='l2', axis=1, copy=True)

    input_length = output_length = tr_vis_data.shape[1] + tr_sem_data.shape[1]
    ae = Encoder(input_length, tr_sem_data.shape[1], output_length, ModelType.ZSL_AE, 50, '.', run_svm=False)
    tr_sem, te_sem = ae.estimate_semantic_data_zsl(tr_vis_data, te_vis_data, tr_sem_data, False)

    return tr_sem, te_sem


def cal_acc_sec(te_est):
    acc_value, _ = ZSL.zsl_el(te_est, _data['S_te_pro'], test_labels, template_labels, 1, True)
    return acc_value


def cal_acc(te_est_or):
    te_labels = {label: [] for label in set([lb for lb in _data['test_labels']])}

    te_est_pt = te_est_or[mask, :]
    te_est_st = te_est_or[[not v for v in mask], :]
    test_labels_pt = test_labels[mask]
    test_labels_st = test_labels[[not v for v in mask]]

    for i, sample in enumerate(te_est_pt):
        te_labels[test_labels_pt[i]].append(sample)

    prototypes = []
    for lb in template_labels:
        prototypes.append(np.mean(te_labels[lb], axis=0))

    prototypes = np.array(prototypes)

    acc_value, _ = ZSL.zsl_el(te_est_st, prototypes, test_labels_st, template_labels, 1, True)
    return acc_value


_data = loadmat('../Datasets/cub_data_resnet50.mat')
_data['S_te'] = get_te_sem_data(_data, 'cub')
_data['train_labels'] = [lb[0] for lb in _data['train_labels_cub']]
_data['test_labels'] = [lb[0] for lb in _data['test_labels_cub']]
template_labels = np.array([int(x) for x in _data['te_cl_id']])
test_labels = np.array([int(x) for x in _data['test_labels_cub']])

acc = {'cat': [], 'pca': [], 'vse': [], 'sec': []}

for k in range(5):
    pt_labels = {label: [] for label in set([lb for lb in _data['test_labels']])}
    mask = [False] * len(_data['test_labels'])

    for lb in set([lb for lb in _data['test_labels']]):
        while len(pt_labels[lb]) < 3:
            i = randrange(len(_data['test_labels']))
            if _data['test_labels'][i] == lb and i not in pt_labels[lb]:
                pt_labels[lb].append(i)
                mask[i] = True

    _, te_est_sec = estimate_sem_data_sec(_data)
    _, te_est_vse = estimate_sem_data_vse(_data)
    _, te_est_pca = estimate_sem_data_pca(_data)
    _, te_est_cat = estimate_sem_data_cat(_data)

    acc['sec'].append(cal_acc_sec(te_est_sec))
    acc['vse'].append(cal_acc(te_est_vse))
    acc['cat'].append(cal_acc(te_est_cat))
    acc['pca'].append(cal_acc(te_est_pca))

for k in acc.keys():
    if not acc[k]:
        continue

    k_acc = np.array(acc[k])
    print('%s: $%.1f \\pm %.1f$' % (k, np.mean(k_acc) * 100, np.std(k_acc) * 100))
