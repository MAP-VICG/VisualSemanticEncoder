"""
Contains demo for running SAE (Semantic Auto-encoder) for AwA dataset. This approach was proposed by
Elyor Kodirov, Tao Xiang, and Shaogang Gong in the paper "Semantic Autoencoder for Zero-shot Learning"
published in CVPR 2017. Code originally written in Matlab and is here transformed to Python.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 25, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import bz2
import pickle
import numpy as np
from sklearn.preprocessing import normalize

from encoders.tools.src.utils import ZSL


class AWA:
    def __init__(self, data_path_tr, data_path_te):
        """
        Defines parameters, loads data and computes weights using SAE

        :param data_path_tr: string with path pointing to pickle file with training data
        :param data_path_te: string with path pointing to pickle file with test data
        """
        self.hit_k = 1
        self.lambda_ = 500000
        self.z_score = False
        self.s_tr = None
        self.w = None

        self.data = {**pickle.load(bz2.BZ2File(data_path_tr, 'rb')), **pickle.load(bz2.BZ2File(data_path_te, 'rb'))}
        self.x_tr = self._normalize(self.data['X_tr'].transpose()).transpose()
        self.x_te = np.array(self.data['X_te'])

    def reset_weights(self):
        """
        Set w to None so it can be computed before calculating the feature space projection

        :return: None
        """
        self.w = None

    def set_semantic_data(self, sem_data=None):
        """
        Replaces the default semantic data by the given array if it has similar shape with the original one

        :param sem_data: array of shape (8855, 312)
        :return: None
        """
        if self.s_tr is None and sem_data is None:
            self.s_tr = self.data['S_tr']
        elif sem_data is not None:
            if sem_data.shape == (24295, 85):
                self.s_tr = sem_data
            else:
                raise ValueError('Data provided is invalid. It should be of shape (24295, 85)')

    def _compute_weights(self):
        """
        Computes the weights that estimates the latent space using SAE

        :return: a 2D numpy array with the matrix of weights computed
        """
        return ZSL.sae(self.x_tr.transpose(), self.s_tr.transpose(), self.lambda_).transpose()

    @staticmethod
    def _normalize(data):
        """
        Wrapper function to normalize data. Normalized data is copied into another object so values of original data
        are not changed.

        :param data: 2D numpy array with data to be normalized
        :return: normalized data
        """
        return normalize(data, norm='l2', axis=1, copy=True)

    def v2s_projection(self):
        """
        Applies zero shot learning in the estimated data, classifies each test sample with the class of the closest
        neighbor and computes the accuracy of classification comparing the estimated class with the one stated in the
        template array. The projection goes from the feature space (visual features extracted from a CNN) to the
        semantic space.

        :return: float number with the accuracy of the ZSL classification
        """
        if self.w is None:
            self.w = self._compute_weights().transpose()

        s_est = self.x_te.dot(self._normalize(self.w).transpose())
        s_te_gt = self._normalize(self.data['S_te_pro'].transpose()).transpose()
        acc, _ = ZSL.zsl_el(s_est, s_te_gt, self.data['Y_te'], self.data['S_te_pro_lb'], self.hit_k, self.z_score)
        return acc

    def s2v_projection(self):
        """
        Applies zero shot learning in the estimated data, classifies each test sample with the class of the closest
        neighbor and computes the accuracy of classification comparing the estimated class with the one stated in the
        template array. The projection goes from the semantic space to the feature space (visual features extracted
        from a CNN).

        :return: float number with the accuracy of the ZSL classification
        """
        if self.w is None:
            self.w = self._compute_weights().transpose()

        x_te_pro = self._normalize(self.data['S_te_pro'].transpose()).transpose().dot(self._normalize(self.w))
        x_te_pro = self._normalize(x_te_pro.transpose()).transpose()
        acc, _ = ZSL.zsl_el(self.x_te, x_te_pro, self.data['Y_te'], self.data['S_te_pro_lb'], self.hit_k, self.z_score)
        return acc


if __name__ == '__main__':
    data_tr = os.sep.join([os.getcwd().split('/encoders')[0], 'data', 'awa_data_inceptionV1_tr.pbz2'])
    data_te = os.sep.join([os.getcwd().split('/encoders')[0], 'data', 'awa_data_inceptionV1_te.pbz2'])

    awa = AWA(data_tr, data_te)
    awa.set_semantic_data()

    print('\n[1] AwA ZSL accuracy [V >>> S]: %.1f%%\n' % (awa.v2s_projection() * 100))
    print('[2] AwA ZSL accuracy [S >>> V]: %.1f%%\n' % (awa.s2v_projection() * 100))
