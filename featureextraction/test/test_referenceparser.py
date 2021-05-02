"""
Tests for module referenceparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 24, 2021

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import bz2
import pickle
import unittest
import numpy as np
from scipy.io import loadmat

from featureextraction.src.referenceparser import AWAParser, CUBParser


class AWAParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        awa_path = os.sep.join([os.getcwd().split('/VisualSemanticEncoder')[0], 'Datasets', 'awa_data_inceptionV1.mat'])
        cls.original = loadmat(awa_path)
        cls.prs = AWAParser(awa_path)

    def test_get_te_labels(self):
        """
        Tests the labels array created for the test set matches the original one
        """
        labels = self.prs._get_te_labels()
        n_instances = self.prs.new_data_te['X_te'].shape[0]
        self.assertEqual((n_instances,), labels.shape)

        for i in range(n_instances):
            self.assertEqual(labels[i], self.original['param']['test_labels'][0][0][i])

    def test_get_tr_labels(self):
        """
        Tests the labels array created for the training set matches the original one
        """
        labels = self.prs._get_tr_labels()
        n_instances = self.prs.new_data_tr['X_tr'].shape[0]
        self.assertEqual((n_instances,), labels.shape)

        for i in range(n_instances):
            self.assertEqual(labels[i], self.original['param']['train_labels'][0][0][i])

    def test_get_te_sem_data(self):
        """
        Tests the semantic data array created for the test set matches the original one
        """
        labels = self.prs._get_te_labels()
        sem_data = self.prs._get_te_sem_data()
        n_instances = self.prs.new_data_te['X_te'].shape[0]
        sem_length = self.prs.new_data_tr['S_tr'].shape[1]
        self.assertEqual((n_instances, sem_length), sem_data.shape)

        lbs = {lb[0]: i for i, lb in enumerate(self.original['param']['testclasses_id'][0][0])}
        for i, attrs in enumerate(sem_data):
            self.assertTrue((attrs == self.original['S_te_pro'][lbs[labels[i]]]).all())

    def test_training_set_data(self):
        """
        Tests the training data array created for the training set matches the original one
        """
        self.assertTrue(((self.prs.new_data_tr['X_tr'] == self.original['X_tr']).all()))

    def test_test_set_data(self):
        """
        Tests the test data array created for the test set matches the original one
        """
        self.assertTrue(((self.prs.new_data_te['X_te'] == self.original['X_te']).all()))

    def test_semantic_training_data(self):
        """
        Tests the semantic training data array created for the training set matches the original one
        """
        self.assertTrue(((self.prs.new_data_tr['S_tr'] == self.original['S_tr']).all()))

    def test_semantic_test_prototype(self):
        """
        Tests the semantic test prototype data array created for the test set matches the original one
        """
        self.assertTrue(((self.prs.new_data_te['S_te_pro'] == self.original['S_te_pro']).all()))

    def test_semantic_test_prototype_labels(self):
        """
        Tests the semantic test labels prototype array created for the test set matches the original one
        """
        labels = np.array([int(x) for x in self.original['param']['testclasses_id'][0][0]])
        self.assertTrue(((self.prs._get_te_pro_labels() == labels).all()))


class CUBParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cub_path = os.sep.join([os.getcwd().split('/VisualSemanticEncoder')[0], 'Datasets', 'cub_data_inceptionV1.mat'])
        cls.original = loadmat(cub_path)
        cls.prs = CUBParser(cub_path)

    def test_get_te_labels(self):
        """
        Tests the labels array created for the test set matches the original one
        """
        labels = self.prs._get_te_labels()
        n_instances = self.prs.new_data_te['X_te'].shape[0]
        self.assertEqual((n_instances,), labels.shape)

        for i in range(n_instances):
            self.assertEqual(labels[i], self.original['test_labels_cub'][i])

    def test_get_tr_labels(self):
        """
        Tests the labels array created for the training set matches the original one
        """
        labels = self.prs._get_tr_labels()
        n_instances = self.prs.new_data_tr['X_tr'].shape[0]
        self.assertEqual((n_instances,), labels.shape)

        for i in range(n_instances):
            self.assertEqual(labels[i], self.original['train_labels_cub'][i])

    def test_get_te_sem_data(self):
        """
        Tests the semantic data array created for the test set matches the original one
        """
        labels = self.prs._get_te_labels()
        sem_data = self.prs._get_te_sem_data()
        n_instances = self.prs.new_data_te['X_te'].shape[0]
        sem_length = self.prs.new_data_tr['S_tr'].shape[1]
        self.assertEqual((n_instances, sem_length), sem_data.shape)

        lbs = {lb[0]: i for i, lb in enumerate(self.original['te_cl_id'])}
        for i, attrs in enumerate(sem_data):
            self.assertTrue((attrs == self.original['S_te_pro'][lbs[labels[i]]]).all())

    def test_training_set_data(self):
        """
        Tests the training data array created for the training set matches the original one
        """
        self.assertTrue(((self.prs.new_data_tr['X_tr'] == self.original['X_tr']).all()))

    def test_test_set_data(self):
        """
        Tests the test data array created for the test set matches the original one
        """
        self.assertTrue(((self.prs.new_data_te['X_te'] == self.original['X_te']).all()))

    def test_semantic_training_data(self):
        """
        Tests the semantic training data array created for the training set matches the original one
        """
        self.assertTrue(((self.prs.new_data_tr['S_tr'] == self.original['S_tr']).all()))

    def test_semantic_test_prototype(self):
        """
        Tests the semantic test prototype data array created for the test set matches the original one
        """
        self.assertTrue(((self.prs.new_data_te['S_te_pro'] == self.original['S_te_pro']).all()))

    def test_semantic_test_prototype_labels(self):
        """
        Tests the semantic test labels prototype array created for the test set matches the original one
        """
        labels = np.array([int(x) for x in self.original['te_cl_id']])
        self.assertTrue(((self.prs._get_te_pro_labels() == labels).all()))


class Matlab2PickleParserParserTest(unittest.TestCase):
    def test_save_data(self):
        """
        Tests if the is saved as expected and if after is is loaded it matches the original one
        """
        data_path_tr = './mockfiles/mock_awa_data_inceptionV1_te.pbz2'
        data_path_te = './mockfiles/mock_awa_data_inceptionV1_tr.pbz2'
        awa_path = os.sep.join([os.getcwd().split('/VisualSemanticEncoder')[0], 'Datasets', 'awa_data_inceptionV1.mat'])
        prs = AWAParser(awa_path)
        prs('mockfiles/', 'mock_awa_data_inceptionV1')
        self.assertTrue(os.path.isfile(data_path_tr))
        self.assertTrue(os.path.isfile(data_path_te))

        expected_keys = ['S_te', 'S_te_pro', 'S_te_pro_lb', 'S_tr', 'X_te', 'X_tr', 'Y_te', 'Y_tr']
        self.data = {**pickle.load(bz2.BZ2File(data_path_tr, 'rb')), **pickle.load(bz2.BZ2File(data_path_te, 'rb'))}
        self.assertTrue(sorted(list(self.data.keys())) == expected_keys)

        for key in prs.new_data_te:
            self.assertTrue((prs.new_data_te[key] == self.data[key]).all())

        for key in prs.new_data_tr:
            self.assertTrue((prs.new_data_tr[key] == self.data[key]).all())

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files that were written by the tests
        """
        if os.path.isfile('./mockfiles/mock_awa_data_inceptionV1_te.pbz2'):
            os.remove('./mockfiles/mock_awa_data_inceptionV1_te.pbz2')
        if os.path.isfile('./mockfiles/mock_awa_data_inceptionV1_tr.pbz2'):
            os.remove('./mockfiles/mock_awa_data_inceptionV1_tr.pbz2')
