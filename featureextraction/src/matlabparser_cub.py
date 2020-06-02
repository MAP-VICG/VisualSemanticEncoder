import numpy as np
from scipy.io import savemat

test_labels = [1, 4, 6, 8, 9, 14, 23, 29, 31, 33, 34, 35, 36, 37, 38, 43, 49, 51, 53, 66, 72, 79,
               83, 84, 86, 91, 95, 96, 98, 101, 102, 103, 112, 114, 119, 121, 130, 135, 138, 147,
               156, 163, 165, 166, 180, 183, 185, 186, 187, 197]

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_x_train_sem.txt') as f:
    sem_data = [list(map(float, line.split(' '))) for line in f.readlines()]

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_x_test_sem.txt') as f:
    sem_data.extend([list(map(float, line.split(' '))) for line in f.readlines()])

sem_data = np.array(sem_data)

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_x_train_vis.txt') as f:
    vis_data = [list(map(float, line.split(' '))) for line in f.readlines()]

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_x_test_vis.txt') as f:
    vis_data.extend([list(map(float, line.split(' '))) for line in f.readlines()])

vis_data = np.array(vis_data)

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_y_train.txt') as f:
    labels = [int(line.strip()) for line in f.readlines()]

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_y_test.txt') as f:
    labels.extend([int(line.strip()) for line in f.readlines()])

labels = np.array(labels)

tr_mask = [False if lb in test_labels else True for lb in labels]
te_mask = [True if lb in test_labels else False for lb in labels]

S_tr = sem_data[tr_mask]
S_te = sem_data[te_mask]

X_tr = vis_data[tr_mask]
X_te = vis_data[te_mask]

train_labels_cub = labels[tr_mask]
test_labels_cub = labels[te_mask]

S_te_pro = []
for lb in test_labels:
    for i, att in enumerate(sem_data):
        if labels[i] == lb:
            S_te_pro.append(att)
            break

S_te_pro = np.array(S_te_pro)

data = {'test_labels_cub': test_labels_cub, 'train_labels_cub': train_labels_cub, 'X_tr': X_tr,
        'X_te': X_te, 'S_tr': S_tr, 'S_te': S_te, 'te_cl_id': test_labels, 'S_te_pro': S_te_pro}

savemat('../../../Datasets/SAE/cub_demo_data_resnet.mat', data)
