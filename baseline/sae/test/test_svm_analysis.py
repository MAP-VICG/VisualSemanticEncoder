"""
Tests for module svm_analysis

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 03, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
import numpy as np
from scipy.io import loadmat

from baseline.sae.src.svm_analysis import SVMClassification


class SVMClassificationTests(unittest.TestCase):
    def test_estimate_semantic_data_awa(self):
        """
        Tests if semantic data is correctly estimated by comparing it with awa_demo results
        """
        svm = SVMClassification('awa')
        data = loadmat('sae/test/mockfiles/awa_sae_data.mat')
        sem_data = svm.estimate_semantic_data(data['x_tr'], data['x_te'], data['s_tr'])
        self.assertTrue((np.round(data['s_est'], decimals=4) == np.round(sem_data, decimals=4)).all())

    def test_estimate_semantic_data_cub(self):
        """
        Tests if semantic data is correctly estimated by comparing it with cub_demo results
        """
        svm = SVMClassification('cub')
        data = loadmat('sae/test/mockfiles/cub_sae_data.mat')
        sem_data = svm.estimate_semantic_data(data['x_tr'], data['x_te'], data['s_tr'])
        self.assertTrue((np.round(data['s_est'], decimals=4) == np.round(sem_data, decimals=4)).all())

    def test_structure_data_awa(self):
        """
        Tests if data loaded from awa_demo_data.mat file is correctly restructured
        """
        svm = SVMClassification('awa')
        sem_data, vis_data, labels = svm.structure_data('../../Datasets/SAE/awa_demo_data.mat')
        self.assertEqual((30475, 85), sem_data.shape)
        self.assertEqual((30475, 1024), vis_data.shape)
        self.assertEqual((30475,), labels.shape)

    def test_structure_data_cub(self):
        """
        Tests if data loaded from cub_demo_data.mat file is correctly restructured
        """
        svm = SVMClassification('cub')
        sem_data, vis_data, labels = svm.structure_data('../../Datasets/SAE/cub_demo_data.mat')
        self.assertEqual((11788, 312), sem_data.shape)
        self.assertEqual((11788, 1024), vis_data.shape)
        self.assertEqual((11788,), labels.shape)

    def test_classify_data(self):
        """
        Tests if classification results are in the expected shape
        """
        svm = SVMClassification('awa')
        sem_data, vis_data, labels = svm.structure_data('../../Datasets/SAE/awa_demo_data.mat')
        acc, params = svm.classify_data(sem_data, vis_data, labels, 3)

        self.assertEqual(3, len(acc))
        self.assertEqual(3, len(params))
        self.assertTrue(isinstance(acc[0], float))
        self.assertTrue(isinstance(params[0], dict))
