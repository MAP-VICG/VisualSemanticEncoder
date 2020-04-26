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
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cls.data = loadmat('mockfiles/cub_demo_data.mat')
        cls.labels = list(map(int, loadmat('mockfiles/cub_demo_data.mat')['train_labels_cub']))

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
        indexes, labels = ZSL.is_member(self.labels)

        self.assertTrue((list(map(int, ref - 1)) == indexes).all())
        self.assertTrue((list(set(self.labels)) == labels).all())

    def test_label_matrix(self):
        """
        Tests if the matrix of labels returned matches all the expected values
        """
        ref = loadmat('mockfiles/label_matrix.mat')['label_matrix']
        mat = ZSL.label_matrix(self.labels)

        self.assertTrue((mat == ref).all())

    def test_dimension_reduction(self):
        """
        Tests if the x_tr and x_te returned matches all the expected values
        """
        x_tr, x_te = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], self.labels)
        ref = loadmat('mockfiles/dimension_reduction.mat')['dimension_reduction']

        self.assertTrue((np.around(x_tr, decimals=5) == np.around(ref['X_tr'][0][0], decimals=5)).all())
        self.assertTrue((np.around(x_te, decimals=5) == np.around(ref['X_te'][0][0], decimals=5)).all())

    def test_sae(self):
        """
        Tests if the matrix returned by the sylvester calculation matches all the expected values
        """
        s_tr = normalize(self.data['S_tr'], norm='l2', axis=1, copy=False)
        x_tr, _ = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], self.labels)
        ref = loadmat('mockfiles/sae.mat')['sae']
        w = ZSL.sae(x_tr.transpose(), s_tr.transpose(), .2).transpose()

        self.assertTrue((np.around(ref, decimals=5) == np.around(w, decimals=5)).all())

    def test_zsl_el(self):
        """
        Tests if accuracy of Zero Shot Learning computed is the same as the expected one and if
        the returned classes of 1NN is the same as the reference
        """
        s_tr = normalize(self.data['S_tr'], norm='l2', axis=1, copy=False)
        x_tr, x_te = ZSL.dimension_reduction(self.data['X_tr'], self.data['X_te'], self.labels)
        w = ZSL.sae(x_tr.transpose(), s_tr.transpose(), .2).transpose()
        temp_labels = np.array([int(x) for x in self.data['te_cl_id']])
        test_labels = np.array([int(x) for x in self.data['test_labels_cub']])
        ref = loadmat('mockfiles/y_hit.mat')['Y_hit5']

        acc, y_hit = ZSL.zsl_el(x_te.dot(w), self.data['S_te_pro'], test_labels, temp_labels, 1, z_score=True)
        self.assertEqual(0.61405, np.around(acc, decimals=5))
        self.assertTrue((ref == y_hit).all())
