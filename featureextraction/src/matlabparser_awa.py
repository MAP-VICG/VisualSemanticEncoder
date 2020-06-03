import numpy as np
from scipy.io import savemat

test_labels = [25, 39, 15,  6, 42, 14, 18, 48, 34, 24]

with open('/Users/damaresresende/Projects/Datasets/AWA2/AWA2_x_sem.txt') as f:
    sem_data = [list(map(float, line.split(' '))) for line in f.readlines()]

sem_data = np.array(sem_data)

with open('/Users/damaresresende/Projects/Datasets/AWA2/AWA2_x_vis.txt') as f:
    vis_data = [list(map(float, line.split(' '))) for line in f.readlines()]

vis_data = np.array(vis_data)

with open('/Users/damaresresende/Projects/Datasets/AWA2/AWA2_y.txt') as f:
    labels = [int(line.strip()) for line in f.readlines()]

labels = np.array(labels)

tr_mask = [False if lb in test_labels else True for lb in labels]
te_mask = [True if lb in test_labels else False for lb in labels]

S_tr = sem_data[tr_mask]
S_te = sem_data[te_mask]

X_tr = vis_data[tr_mask]
X_te = vis_data[te_mask]

train_labels_awa = labels[tr_mask]
test_labels_awa = labels[te_mask]

S_te_pro = []
for lb in test_labels:
    for i, att in enumerate(sem_data):
        if labels[i] == lb:
            S_te_pro.append(att)
            break

S_te_pro = np.array(S_te_pro)

data = {'S_te_gt': S_te_pro, 'S_te_pro': S_te_pro, 'S_tr': S_tr, 'X_te': X_te, 'X_tr': X_tr,
        'param': {'testclasses_id': np.expand_dims(np.array(test_labels), axis=1),
                  'test_labels': np.expand_dims(np.array(test_labels_awa), axis=1),
                  'train_labels': np.expand_dims(np.array(train_labels_awa), axis=1)}}

savemat('../../../Datasets/SAE/awa_demo_data_resnet.mat', data)
