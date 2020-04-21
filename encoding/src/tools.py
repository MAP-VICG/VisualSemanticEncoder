import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize


def kill_semantic_attributes(data, rate):
    """
    Randomly sets to 0.1 a specific rate of the semantic attributes array

    @param data: 2D numpy array with full data (visual + semantic)
    @param rate: float number from 0 to 1 specifying the rate of values to be set to 0.1
    @return: 2D numpy array with new data set
    """
    num_sem_attrs = abs(data.shape[1] - 2048)

    new_data = np.copy(data)
    for ex in range(new_data.shape[0]):
        mask = [False] * data.shape[1]
        for idx in random.sample(range(2048, data.shape[1]), round(num_sem_attrs * rate)):
            mask[idx] = True

        new_data[ex, mask] = new_data[ex, mask] * 0
    return new_data


def zsl_eval(S_est, S_te_gt, HITK, testclasses_id, test_labels):
    S_te_gt = normalize(S_te_gt.transpose(), norm='l2', axis=1, copy=True).transpose()

    dist = 1 - (cdist(S_est, S_te_gt, metric='cosine'))
    Y_hit5 = np.zeros((dist.shape[0], HITK))

    for i in range(dist.shape[0]):
        I = np.argsort(dist[i, :])
        I = [I[x] for x in range(len(I)-1, 0, -1)]
        Y_hit5[i, :] = testclasses_id[I[0:HITK]]

    n = 0
    for i in range(dist.shape[0]):
        if test_labels[i] in Y_hit5[i, :]:
            n += 1
    zsl_accuracy = n / dist.shape[0]

    return zsl_accuracy, Y_hit5
