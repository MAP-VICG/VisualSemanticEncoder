"""
Following code shows a demo for aPY dataset to reproduce the result of the paper:

Semantic Autoencoder for Zero-shot Learning.

Elyor Kodirov, Tao Xiang, and Shaogang Gong, CVPR 2017.

Code originally written in Matlab and transformed to Python by Damares Resende
"""
import numpy as np
# from scipy.sparse import issparse
from scipy.linalg import solve_sylvester
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from scipy.stats import zscore


class Params:
    def __init__(self, HITK, data, type_):
        self.HITK = HITK
        if type_ == 'awa':
            self.testclasses_id = np.array([int(x) for x in data['param']['testclasses_id'][0][0]])
            self.test_labels = np.array([int(x) for x in data['param']['test_labels'][0][0]])
        elif type_ == 'cub':
            self.testclasses_id = np.array([int(x) for x in data['te_cl_id']])
            self.test_labels = np.array([int(x) for x in data['test_labels_cub']])
        else:
            raise ValueError('Invalid data type')


class ZSL:
    @staticmethod
    def SAE(X, S, lambda_):
        A = S.dot(S.transpose())
        B = lambda_ * X.dot(X.transpose())
        C = (1 + lambda_) * S.dot(X.transpose())
        W = solve_sylvester(A, B, C)
        return W

    # @staticmethod
    # def normalize_fea(fea):
    #     n_smp, m_fea = fea.shape
    #
    #     if issparse(fea):
    #         fea2 = fea.transpose()
    #         fea_norm = normalize(fea2, norm='l2', axis=1, copy=True)
    #
    #         for i in range(n_smp):
    #             fea2[:, i] = fea2[:, i] / max(1e-10, fea_norm[i])
    #         fea = fea2.transpose()
    #     else:
    #         fea_norm = []
    #         norm_values = list((np.sqrt(np.sum(fea * fea, axis=1))).transpose())
    #
    #         for _ in range(m_fea):
    #             fea_norm.append(norm_values)
    #
    #         fea_norm = np.array(fea_norm).transpose()
    #
    #     return fea / fea_norm

    @staticmethod
    def normalize_fea(fea):
        return normalize(fea, norm='l2', axis=1, copy=True)

    @staticmethod
    def zsl_el(S_est, S_te_gt, param, type_):
        if type_ == 'awa':
            S_te_gt = ZSL.normalize_fea(S_te_gt.transpose()).transpose()

        dist = 1 - (cdist(S_est, S_te_gt, metric='cosine'))
        Y_hit5 = np.zeros((dist.shape[0], param.HITK))

        if type_ == 'cub':
            dist = zscore(dist)

        for i in range(dist.shape[0]):
            I = np.argsort(dist[i, :])
            I = [I[x] for x in range(len(I)-1, 0, -1)]
            Y_hit5[i, :] = param.testclasses_id[I[0:param.HITK]]

        n = 0
        for i in range(dist.shape[0]):
            if param.test_labels[i] in Y_hit5[i, :]:
                n += 1
        zsl_accuracy = n / dist.shape[0]

        return zsl_accuracy, Y_hit5

    @staticmethod
    def label_matrix(label_vector):
        """
        Converts the label vector to label matrix.

        :param label_vector: 1xN, N is the number of samples.
        :return S: Nxc matrix, c is the number of classes.
        """
        Y = [v for v in label_vector[0]]
        indexes, labels = ZSL.ismember(Y)
        rows = np.array(range(len(Y)), dtype=int)

        shape = (len(Y), len(labels))
        S = np.array([0] * shape[0] * shape[1])
        S[ZSL.sub2ind(shape, rows, indexes)] = 1
        S = np.reshape(S, (shape[1], shape[0]))
        return S

    @staticmethod
    def ismember(A):
        k = -1
        X = dict()
        idx = np.zeros(len(A))

        for i, value in enumerate(A):
            if value not in X.keys():
                k += 1
                X[value] = k
            idx[i] = int(X[value])

        return np.array(idx, dtype=int), np.array(list(X.keys()), dtype=int)

    @staticmethod
    def sub2ind(array_shape, rows, cols):
        ind = cols * array_shape[0] + rows
        return [int(value) for value in ind]
