"""
Tests for module utils

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 21, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
import numpy as np
from sklearn.preprocessing import normalize

from scipy.io import loadmat
from baseline.sae.src.utils import ZSL


class ZSLTests(unittest.TestCase):
    def test_sub2ind(self):
        """
        Tests if the returned value is equal to indexed position of rows and columns combinations
        """
        row = np.array([0, 1, 2, 0])
        col = np.array([1, 1, 1, 2])
        self.assertEqual([3, 4, 5, 6], ZSL.sub2ind((3, 3), row, col))

    def test_is_member(self):
        """
        Tests if the returned value labels is equal to all available labels in the data set, and if the
        indexes correspond to the index of the latest occurrence of a label
        """
        ref = loadmat('mockfiles/indexes.mat')['indexes'][0]
        data = list(map(int, loadmat('mockfiles/cub_demo_data.mat')['train_labels_cub']))
        indexes, labels = ZSL.is_member(data)

        self.assertTrue((list(map(int, ref - 1)) == indexes).all())
        self.assertTrue((list(set(data)) == labels).all())

    def test_label_matrix(self):
        """
        Tests if the matrix of labels returned matches all the expected values
        """
        ref = loadmat('mockfiles/label_matrix.mat')['label_matrix']
        data = list(map(int, loadmat('mockfiles/cub_demo_data.mat')['train_labels_cub']))
        mat = ZSL.label_matrix(data)

        self.assertTrue((mat == ref).all())

    def test_dimension_reduction(self):
        """
        Tests if the x_tr and x_te returned matches all the expected values
        """
        data = loadmat('mockfiles/cub_demo_data.mat')
        labels = list(map(int, data['train_labels_cub']))
        x_tr, x_te = ZSL.dimension_reduction(data['X_tr'], data['X_te'], labels)
        ref = loadmat('mockfiles/dimension_reduction.mat')['dimension_reduction']

        self.assertTrue((np.around(x_tr, decimals=5) == np.around(ref['X_tr'][0][0], decimals=5)).all())
        self.assertTrue((np.around(x_te, decimals=5) == np.around(ref['X_te'][0][0], decimals=5)).all())

    def test_sae(self):
        """
        Tests if the matrix returned by the sylvester calculation matches all the expected values
        """
        data = loadmat('mockfiles/cub_demo_data.mat')
        labels = list(map(int, data['train_labels_cub']))
        s_tr = normalize(data['S_tr'], norm='l2', axis=1, copy=False)
        x_tr, _ = ZSL.dimension_reduction(data['X_tr'], data['X_te'], labels)
        ref = loadmat('mockfiles/sae.mat')['sae']
        w = ZSL.sae(x_tr.transpose(), s_tr.transpose(), .2).transpose()

        self.assertTrue((np.around(ref, decimals=5) == np.around(w, decimals=5)).all())
