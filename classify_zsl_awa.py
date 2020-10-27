import numpy as np
from scipy.io import loadmat
from encoders.tools.src.utils import ZSL
from sklearn.preprocessing import normalize

from encoders.vse.src.encoder import ModelType, ModelFactory


def get_te_sem_data(data, _type):
    if _type == "awa":
        lbs = {data['param']['testclasses_id'][0][0][j][0]: attrs for j, attrs in enumerate(data['S_te_pro'])}
        return np.array([lbs[label[0]] for label in data['param']['test_labels'][0][0]])
    elif _type == "cub":
        lbs = {data['te_cl_id'][j][0]: attributes for j, attributes in enumerate(data['S_te_pro'])}
        return np.array([lbs[label[0]] for label in data['test_labels_cub']])
    else:
        raise ValueError("Invalid data type.")


def estimate_sem_data(data):
    tr_vis_data = normalize(data['X_tr'], norm='l2', axis=1, copy=True)
    tr_sem_data = normalize(data['S_tr'], norm='l2', axis=1, copy=True)

    te_vis_data = normalize(data['X_te'], norm='l2', axis=1, copy=True)
    te_sem_data = normalize(data['S_te'], norm='l2', axis=1, copy=True)

    train_data = np.hstack((tr_vis_data, tr_sem_data))
    test_data = np.hstack((te_vis_data, te_sem_data))

    input_length = output_length = train_data.shape[1]
    model = ModelFactory(input_length, tr_sem_data.shape[1], output_length)(ModelType.STRAIGHT_AE, run_svm=False)
    model.fit(train_data, data['train_labels'], test_data, data['test_labels'], 5, '.', save_weights=False)

    return model.predict(tr_vis_data, tr_sem_data, te_vis_data, te_sem_data)


_data = loadmat('../Datasets/awa_data_googlenet.mat')
_data['S_te'] = get_te_sem_data(_data, 'awa')
_data['train_labels'] = [lb[0] for lb in _data['param']['train_labels'][0][0]]
_data['test_labels'] = [lb[0] for lb in _data['param']['test_labels'][0][0]]
template_labels = np.array([int(x) for x in _data['param']['testclasses_id'][0][0]])
test_labels = np.array([int(x) for x in _data['param']['test_labels'][0][0]])

tr_labels = {label: [] for label in set([lb for lb in _data['train_labels']])}
te_labels = {label: [] for label in set([lb for lb in _data['test_labels']])}

acc = []
for k in range(5):
    tr_est, te_est = estimate_sem_data(_data)

    for i, sample in enumerate(tr_est):
        tr_labels[_data['train_labels'][i]].append(sample)

    for i, sample in enumerate(te_est):
        te_labels[_data['test_labels'][i]].append(sample)

    prototypes = []
    for lb in template_labels:
        prototypes.append(np.mean(te_labels[lb], axis=0))

    prototypes = np.array(prototypes)

    acc_value, _ = ZSL.zsl_el(te_est, prototypes, test_labels, template_labels, 1, False)
    acc.append(acc_value)

acc = np.array(acc)
print('$%.1f \\pm %.1f$' % (np.mean(acc) * 100, np.std(acc) * 100))
