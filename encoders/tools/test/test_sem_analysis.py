import unittest
import numpy as np
from os import remove
from scipy.io import loadmat

from encoders.tools.src.utils import ZSL
from encoders.tools.src.sem_analysis import SemanticDegradation, CUBZSL, AwAZSL


class SemanticDegradationTests(unittest.TestCase):
    def test_kill_semantic_attributes_25(self):
        """
        Tests if semantic array is correctly destroyed at rate of 25%
        """
        rate = 0.25
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value)
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
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value)
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
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value)
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

        sem_data = CUBZSL.estimate_semantic_data_sae(x_tr, input_data['S_tr'], x_te)
        self.assertTrue((np.round(template['S_est'], decimals=5) == np.round(sem_data, decimals=5)).all())

    def test_estimate_semantic_data_awa(self):
        """
        Tests if semantic data is correctly estimated by comparing it with awa_demo results
        """
        input_data = loadmat('../../../../Datasets/SAE/awa_demo_data.mat')
        template = loadmat('mockfiles/awa_est_sem_data.mat')

        sem_data = AwAZSL.estimate_semantic_data_sae(input_data['X_tr'], input_data['S_tr'], input_data['X_te'])
        self.assertTrue((np.round(template['S_est'], decimals=5) == np.round(sem_data, decimals=5)).all())

    def test_structure_data_awa(self):
        """
        Tests if data loaded from awa_demo_data.mat file is correctly structured for ZSL classification
        """
        input_data = loadmat('../../../../Datasets/SAE/awa_demo_data.mat')
        temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, z_score = AwAZSL.structure_data(input_data)
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
        temp_labels, train_labels, test_labels, s_te_pro, sem_te_data, z_score = CUBZSL.structure_data(input_data)
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
        _, tr_labels, _, _, sem_te_data, _ = AwAZSL.structure_data(data)
        x_tr, x_te = data['X_tr'], data['X_te']

        w_info = {'label': 'fold_0', 'path': '.'}
        s_est, summary = CUBZSL.estimate_semantic_data_sec(x_tr, data['S_tr'], x_te, sem_te_data, tr_labels, 5, w_info)
        remove('sec_best_model_cub_v2s_fold_0.h5')

        self.assertEqual((6180, 85), s_est.shape)
        self.assertEqual(['best_accuracy', 'best_loss', 'loss', 'val_loss', 'acc'], list(summary.keys()))

    def test_estimate_semantic_data_sec_cub(self):
        """
        Tests if data estimated for cub_demo_data.mat file is correctly structured
        """
        data = loadmat('../../../../Datasets/SAE/cub_demo_data.mat')
        _, tr_labels, _, _, sem_te_data, _ = CUBZSL.structure_data(data)
        x_tr, x_te = ZSL.dimension_reduction(data['X_tr'], data['X_te'], tr_labels)

        w_info = {'label': 'fold_0', 'path': '.'}
        s_est, summary = CUBZSL.estimate_semantic_data_sec(x_tr, data['S_tr'], x_te, sem_te_data, tr_labels, 5, w_info)
        remove('sec_best_model_cub_v2s_fold_0.h5')

        self.assertEqual((2933, 312), s_est.shape)
        self.assertEqual(['best_accuracy', 'best_loss', 'loss', 'val_loss', 'acc'], list(summary.keys()))
    def test_degrade_semantic_data_zsl_awa_sae(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa', rates=[0])
        acc_dict = sem.degrade_semantic_data(n_folds=2, ae_type='sae', epochs=2)
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
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', rates=[0])
        acc_dict = sem.degrade_semantic_data(n_folds=2, ae_type='sae', epochs=2)
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
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa', rates=[0])
        acc_dict = sem.degrade_semantic_data(n_folds=2, ae_type='sec', epochs=2)
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
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', rates=[0])
        acc_dict = sem.degrade_semantic_data(n_folds=2, ae_type='sec', epochs=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertTrue(0 <= acc[0] <= 1)
        self.assertTrue(0 <= acc[1] <= 1)
