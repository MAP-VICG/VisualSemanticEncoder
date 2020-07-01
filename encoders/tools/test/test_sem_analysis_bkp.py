"""
Tests for module sem_analysis

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 23, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import os
import shutil
import unittest
import numpy as np

from encoders.tools.src.sem_analysis_bkp import SemanticDegradation


class SemanticDegradationTests(unittest.TestCase):
    def test_kill_semantic_attributes_50(self):
        """
        Tests if semantic array is correctly destroyed at rate of 50%
        """
        rate = 0.5
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value)
        sem_data = sd.data['S_tr']

        # reference
        count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(sem_data):
            for value in example:
                if value == new_value:
                    count[i] += 1

        new_data = sd.kill_semantic_attributes(sem_data, rate)

        # to check if new array changes
        new_data_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == new_value:
                    new_data_count[i] += 1

        # to check if old array is preserved
        new_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(sem_data):
            for value in example:
                if value == new_value:
                    new_count[i] += 1

        self.assertTrue((new_count == count).all())
        self.assertEqual(new_data.shape, sem_data.shape)
        self.assertTrue((new_data_count == count + round(sem_data.shape[1] * rate)).all())

    def test_kill_semantic_attributes_75(self):
        """
        Tests if semantic array is correctly destroyed at rate of 75%
        """
        rate = 0.75
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value)
        sem_data = sd.data['S_tr']

        # reference
        count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(sem_data):
            for value in example:
                if value == new_value:
                    count[i] += 1

        new_data = sd.kill_semantic_attributes(sem_data, rate)

        # to check if new array changes
        new_data_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == new_value:
                    new_data_count[i] += 1

        # to check if old array is preserved
        new_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(sem_data):
            for value in example:
                if value == new_value:
                    new_count[i] += 1

        self.assertTrue((new_count == count).all())
        self.assertEqual(new_data.shape, sem_data.shape)
        self.assertTrue((new_data_count == count + round(sem_data.shape[1] * rate)).all())

    def test_kill_semantic_attributes_25(self):
        """
        Tests if semantic array is correctly destroyed at rate of 25%
        """
        rate = 0.25
        new_value = -9999
        sd = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value)
        sem_data = sd.data['S_tr']

        # reference
        count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(sem_data):
            for value in example:
                if value == new_value:
                    count[i] += 1

        new_data = sd.kill_semantic_attributes(sem_data, rate)

        # to check if new array changes
        new_data_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == new_value:
                    new_data_count[i] += 1

        # to check if old array is preserved
        new_count = np.zeros(sem_data.shape[0])
        for i, example in enumerate(sem_data):
            for value in example:
                if value == new_value:
                    new_count[i] += 1

        self.assertTrue((new_count == count).all())
        self.assertEqual(new_data.shape, sem_data.shape)
        self.assertTrue((new_data_count == count + round(sem_data.shape[1] * rate)).all())

    def test_estimate_semantic_data_cub(self):
        """
        Tests if semantic data is correctly estimated by comparing it with cub_demo results
        """
        svm = SemanticDegradation('mockfiles/cub_sae_data.mat', 'cub', new_value=0)
        sem_data = svm.estimate_semantic_data(svm.data['x_tr'], svm.data['s_tr'], svm.data['x_te'])
        self.assertTrue((np.round(svm.data['s_est'], decimals=5) == np.round(sem_data, decimals=5)).all())

    def test_structure_data_svm_awa(self):
        """
        Tests if data loaded from awa_demo_data.mat file is correctly restructured
        """
        svm = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa', new_value=0)
        sem_data, vis_data, labels = svm.structure_data_svm()
        self.assertEqual((30475, 85), sem_data.shape)
        self.assertEqual((30475, 1024), vis_data.shape)
        self.assertEqual((30475,), labels.shape)

    def test_structure_data_svm_cub(self):
        """
        Tests if data loaded from cub_demo_data.mat file is correctly restructured
        """
        svm = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value=0)
        sem_data, vis_data, labels = svm.structure_data_svm()
        self.assertEqual((11788, 312), sem_data.shape)
        self.assertEqual((11788, 1024), vis_data.shape)
        self.assertEqual((11788,), labels.shape)

    def test_structure_data_zsl_awa(self):
        """
        Tests if data loaded from awa_demo_data.mat file is correctly structured for ZSL classification
        """
        svm = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa', new_value=0)
        temp_labels, test_labels, s_te_pro, z_score = svm.structure_data_zsl()
        self.assertEqual((10,), temp_labels.shape)
        self.assertEqual((6180,), test_labels.shape)
        self.assertEqual((10, 85), s_te_pro.shape)
        self.assertEqual(False, z_score)

    def test_structure_data_zsl_cub(self):
        """
        Tests if data loaded from cub_demo_data.mat file is correctly structured for ZSL classification
        """
        svm = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', new_value=0)
        temp_labels, test_labels, s_te_pro, z_score = svm.structure_data_zsl()
        self.assertEqual((50,), temp_labels.shape)
        self.assertEqual((2933,), test_labels.shape)
        self.assertEqual((50, 312), s_te_pro.shape)
        self.assertEqual(True, z_score)

    def test_degrade_semantic_data_svm(self):
        """
        Tests if classification results are in the expected shape
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa', rates=[0])
        acc_dict = sem.degrade_semantic_data_svm(n_folds=2)

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))

    def test_degrade_semantic_data_zsl_awa(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/awa_demo_data.mat', 'awa', rates=[0])
        acc_dict = sem.degrade_semantic_data_zsl(n_folds=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertEqual(0.84676, np.around(acc[0], decimals=5))
        self.assertEqual(0.84676, np.around(acc[1], decimals=5))

    def test_degrade_semantic_data_zsl_cub(self):
        """
        Tests if classification results are in the expected shape and if accuracy returned is correct
        """
        sem = SemanticDegradation('../../../../Datasets/SAE/cub_demo_data.mat', 'cub', rates=[0])
        acc_dict = sem.degrade_semantic_data_zsl(n_folds=2)
        acc = list(map(float, acc_dict[0]['acc'].split(',')))

        self.assertEqual([0], list(acc_dict.keys()))
        self.assertEqual(['acc', 'mean', 'std', 'max', 'min'], list(acc_dict[0].keys()))
        self.assertEqual(2, len(acc_dict[0]['acc'].split(',')))
        self.assertEqual(0.61405, np.around(acc[0], decimals=5))
        self.assertEqual(0.61405, np.around(acc[1], decimals=5))

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files created in tests
        """
        if os.path.isdir('0'):
            shutil.rmtree('0')
