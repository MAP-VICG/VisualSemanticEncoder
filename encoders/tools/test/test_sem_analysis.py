"""
Tests for module sem_analysis

@author: Damares Resende
@contact: damaresresende@usp.br
@since: July 2, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import shutil
import unittest
import numpy as np
from os import path
from os import remove
from scipy.io import loadmat

from encoders.tools.src.utils import ZSL
from encoders.tools.src.sem_analysis import SemanticDegradation, CUBClassification, AwAClassification


class SemanticDegradationTests(unittest.TestCase):
    def test_kill_semantic_attributes_25(self):
        """
        Tests if semantic array is correctly destroyed at rate of 25%
        """
        rate = 0.25
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', new_value)
        sem_data = sd.data['S_tr']

        new_data = sd.kill_semantic_attributes(sem_data, rate)

        # to check if new array changes
        new_data_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == new_value:
                    new_data_count[i] += 1

        # check different values in columns
        for row in range(sem_data.shape[0]):
            self.assertTrue(len(set(new_data[row, :])) > 10)

        # check different values in rows
        for column in range(sem_data.shape[1]):
            self.assertTrue(len(set(new_data[:, column])) > 10)

        self.assertEqual(new_data.shape, sem_data.shape)
        self.assertTrue((new_data_count == round(sem_data.shape[1] * rate)).all())

    def test_kill_semantic_attributes_50(self):
        """
        Tests if semantic array is correctly destroyed at rate of 50%
        """
        rate = 0.5
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', new_value)
        sem_data = sd.data['S_tr']

        new_data = sd.kill_semantic_attributes(sem_data, rate)

        # to check if new array changes
        new_data_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == new_value:
                    new_data_count[i] += 1

        # check different values in columns
        for row in range(sem_data.shape[0]):
            self.assertTrue(len(set(new_data[row, :])) > 10)

        # check different values in rows
        for column in range(sem_data.shape[1]):
            self.assertTrue(len(set(new_data[:, column])) > 10)

        self.assertEqual(new_data.shape, sem_data.shape)
        self.assertTrue((new_data_count == round(sem_data.shape[1] * rate)).all())

    def test_kill_semantic_attributes_75(self):
        """
        Tests if semantic array is correctly destroyed at rate of 75%
        """
        rate = 0.75
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', new_value)
        sem_data = sd.data['S_tr']

        new_data = sd.kill_semantic_attributes(sem_data, rate)

        # to check if new array changes
        new_data_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == new_value:
                    new_data_count[i] += 1

        # check different values in columns
        for row in range(sem_data.shape[0]):
            self.assertTrue(len(set(new_data[row, :])) > 10)

        # check different values in rows
        for column in range(sem_data.shape[1]):
            self.assertTrue(len(set(new_data[:, column])) > 10)

        self.assertEqual(new_data.shape, sem_data.shape)
        self.assertTrue((new_data_count == round(sem_data.shape[1] * rate)).all())

    def test_estimate_semantic_data_cub(self):
        """
        Tests if semantic data is correctly estimated by comparing it with cub_demo results
        """
        input_data = loadmat('../../../../Datasets/SAE/cub_demo_data.mat')
        template = loadmat('mockfiles/cub_est_sem_data.mat')

        labels = list(map(int, input_data['train_labels_cub']))
        x_tr, x_te = ZSL.dimension_reduction(input_data['X_tr'], input_data['X_te'], labels)

        sem_data = CUBClassification.estimate_semantic_data_sae(x_tr, input_data['S_tr'], x_te)
        self.assertTrue((np.round(template['S_est'], decimals=5) == np.round(sem_data, decimals=5)).all())

    def test_estimate_semantic_data_awa(self):
        """
        Tests if semantic data is correctly estimated by comparing it with awa_demo results
        """
        input_data = loadmat('../../../../Datasets/SAE/awa_demo_data.mat')
        template = loadmat('mockfiles/awa_est_sem_data.mat')

        sem_data = AwAClassification.estimate_semantic_data_sae(input_data['X_tr'], input_data['S_tr'], input_data['X_te'])
        self.assertTrue((np.round(template['S_est'], decimals=5) == np.round(sem_data, decimals=5)).all())

    def test_structure_data_awa(self):
        """
        Tests if data loaded from awa_demo_data.mat file is correctly structured for ZSL classification
        """
        input_data = loadmat('../../../../Datasets/SAE/awa_demo_data.mat')
        temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, z_score = AwAClassification.structure_data(input_data)
        self.assertEqual((10,), temp_labels.shape)
        self.assertEqual((24295,), train_labels.shape)
        self.assertEqual((6180,), test_labels.shape)
        self.assertEqual((10, 85), s_te_pro.shape)
        self.assertEqual((6180, 85), sem_te_data.shape)
        self.assertEqual(False, z_score)
        self.assertEqual(set(), set(train_labels).intersection(set(test_labels)))

    def test_structure_data_cub(self):
        """
        Tests if data loaded from cub_demo_data.mat file is correctly structured for ZSL classification
        """
        input_data = loadmat('../../../../Datasets/SAE/cub_demo_data.mat')
        temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, z_score = CUBClassification.structure_data(input_data)
        self.assertEqual((50,), temp_labels.shape)
        self.assertEqual((8855,), train_labels.shape)
        self.assertEqual((2933,), test_labels.shape)
        self.assertEqual((50, 312), s_te_pro.shape)
        self.assertEqual((2933, 312), sem_te_data.shape)
        self.assertEqual(True, z_score)
        self.assertEqual(set(), set(train_labels).intersection(set(test_labels)))

    def test_estimate_semantic_data_sec_awa(self):
        """
        Tests if data estimated for cub_demo_data.mat file is correctly structured
        """
        data = loadmat('../../../../Datasets/SAE/awa_demo_data.mat')
        _, tr_labels, _, _, sem_te_data, _ = AwAClassification.structure_data(data)
        x_tr, x_te = data['X_tr'], data['X_te']

        w_info = {'label': 'fold_0', 'path': '.'}
        s_est, summary = CUBClassification.estimate_semantic_data_sec(x_tr, data['S_tr'], x_te, sem_te_data, tr_labels, 5, w_info)
        remove('sec_best_model_cub_v2s_fold_0.h5')

        self.assertEqual((6180, 85), s_est.shape)
        self.assertEqual(['best_accuracy', 'best_loss', 'loss', 'val_loss', 'acc'], list(summary.keys()))

    def test_estimate_semantic_data_sec_cub(self):
        """
        Tests if data estimated for cub_demo_data.mat file is correctly structured
        """
        data = loadmat('../../../../Datasets/SAE/cub_demo_data.mat')
        _, tr_labels, _, _, sem_te_data, _ = CUBClassification.structure_data(data)
        x_tr, x_te = ZSL.dimension_reduction(data['X_tr'], data['X_te'], tr_labels)

        w_info = {'label': 'fold_0', 'path': '.'}
        s_est, summary = CUBClassification.estimate_semantic_data_sec(x_tr, data['S_tr'], x_te, sem_te_data, tr_labels, 5, w_info)
        remove('sec_best_model_cub_v2s_fold_0.h5')

        self.assertEqual((2933, 312), s_est.shape)
        self.assertEqual(['best_accuracy', 'best_loss', 'loss', 'val_loss', 'acc'], list(summary.keys()))

    def test_degrade_semantic_data_zsl_awa_sae(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', rates=[0])
        acc_dict = sem.degrade_semantic_data(ae_type='sae', data_type='awa', class_type='zsl', n_folds=2, epochs=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertEqual(0.84676, np.around(acc[0], decimals=5))
        self.assertEqual(0.84676, np.around(acc[1], decimals=5))

    def test_degrade_semantic_data_zsl_cub_sae(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        acc_dict = sem.degrade_semantic_data(ae_type='sae', data_type='cub', class_type='zsl', n_folds=2, epochs=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertEqual(0.61405, np.around(acc[0], decimals=5))
        self.assertEqual(0.61405, np.around(acc[1], decimals=5))

    def test_degrade_semantic_data_zsl_awa_sec(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', rates=[0])
        acc_dict = sem.degrade_semantic_data(ae_type='sec', data_type='awa', class_type='zsl', n_folds=2, epochs=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_degrade_semantic_data_zsl_cub_sec(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        acc_dict = sem.degrade_semantic_data(ae_type='sec', data_type='cub', class_type='zsl', n_folds=2, epochs=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_get_classification_data_awa(self):
        """
        Tests if the shapes of returned data are as expected
        """
        data = loadmat('../../../../Datasets/SAE/awa_demo_data.mat')
        sem_data, vis_data, labels = AwAClassification.get_classification_data(data)
        self.assertEqual((30475, 85), sem_data.shape)
        self.assertEqual((30475, 1024), vis_data.shape)
        self.assertEqual((30475,), labels.shape)

    def test_get_classification_data_cub(self):
        """
        Tests if the shapes of returned data are as expected
        """
        data = loadmat('../../../../Datasets/SAE/cub_demo_data.mat')
        sem_data, vis_data, labels = CUBClassification.get_classification_data(data)
        self.assertEqual((11788, 312), sem_data.shape)
        self.assertEqual((11788, 1024), vis_data.shape)
        self.assertEqual((11788,), labels.shape)

    def test_set_training_params(self):
        """
        Tests if parameters values are correctly set
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sae', data_type='cub', n_folds=5, epochs=30)
        self.assertEqual(sem.ae_type, 'sae')
        self.assertEqual(sem.data_type, 'cub')
        self.assertEqual(sem.n_folds, 5)
        self.assertEqual(sem.epochs, 30)

    def test_zero_shot_learning_awa(self):
        """
        Tests if accuracy of zero short learning classification is as expected for awa
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sae', data_type='awa', n_folds=2, epochs=2)
        temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score = AwAClassification.structure_data(sem.data)
        acc = sem._zero_shot_learning(temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score, 0)
        self.assertEqual(2, len(acc))
        self.assertEqual(0.84676, np.around(acc[0], decimals=5))
        self.assertEqual(0.84676, np.around(acc[1], decimals=5))

    def test_zero_shot_learning_cub(self):
        """
        Tests if accuracy of zero short learning classification is as expected for cub
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sae', data_type='cub', n_folds=2, epochs=2)
        temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score = CUBClassification.structure_data(sem.data)
        acc = sem._zero_shot_learning(temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score, 0)
        self.assertEqual(2, len(acc))
        self.assertEqual(0.61405, np.around(acc[0], decimals=5))
        self.assertEqual(0.61405, np.around(acc[1], decimals=5))

    def test_zero_shot_learning_awa_sec(self):
        """
        Tests if accuracy of zero_shot_learning is as expected for awa using sec
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sec', data_type='awa', n_folds=2, epochs=2)
        temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score = AwAClassification.structure_data(sem.data)
        acc = sem._zero_shot_learning(temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score, 0)

        self.assertEqual(2, len(acc))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_zero_shot_learning_cub_sec(self):
        """
        Tests if accuracy of zero_shot_learning is as expected for cub using sec
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sec', data_type='cub', n_folds=2, epochs=2)
        temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score = CUBClassification.structure_data(sem.data)
        acc = sem._zero_shot_learning(temp_labels, tr_labels, te_labels, s_te_pro, s_te_data, z_score, 0)

        self.assertEqual(2, len(acc))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_svm_classification_cub(self):
        """
        Tests if accuracy of svm_classification is as expected for cub using sae
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sae', data_type='cub', n_folds=2, epochs=2)
        sem_data, vis_data, data_labels = CUBClassification.get_classification_data(sem.data)
        acc = sem._simple_classification(sem_data, vis_data, data_labels, 0)

        self.assertEqual(2, len(acc))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_svm_classification_awa(self):
        """
        Tests if accuracy of svm_classification is as expected for cub using sae
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sae', data_type='awa', n_folds=2, epochs=2)
        sem_data, vis_data, data_labels = AwAClassification.get_classification_data(sem.data)
        acc = sem._simple_classification(sem_data, vis_data, data_labels, 0)

        self.assertEqual(2, len(acc))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_svm_classification_cub_sec(self):
        """
        Tests if accuracy of svm_classification is as expected for cub using sec
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sec', data_type='cub', n_folds=2, epochs=2)
        sem_data, vis_data, data_labels = CUBClassification.get_classification_data(sem.data)
        acc = sem._simple_classification(sem_data, vis_data, data_labels, 0)

        self.assertEqual(2, len(acc))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    def test_svm_classification_awa_sec(self):
        """
        Tests if accuracy of svm_classification is as expected for cub using sec
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', rates=[0])
        sem._set_training_params(ae_type='sec', data_type='awa', n_folds=2, epochs=2)
        sem_data, vis_data, data_labels = AwAClassification.get_classification_data(sem.data)
        acc = sem._simple_classification(sem_data, vis_data, data_labels, 0)

        self.assertEqual(2, len(acc))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files created in tests
        """
        if path.isdir('0'):
            shutil.rmtree('0')
