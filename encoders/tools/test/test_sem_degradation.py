import unittest
import numpy as np
from scipy.io import loadmat
from encoders.tools.src.sem_degradation import SemanticDegradation


class SemanticDegradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = loadmat('../../../../Datasets/SEM/cub_demo_data.mat')

    def test_kill_semantic_attributes_25(self):
        """
        Tests if semantic array is correctly destroyed at rate of 25%
        """
        rate = 0.25
        new_value = -9999
        sem_data = self.data['S_tr']

        new_data = SemanticDegradation.kill_semantic_attributes(sem_data, rate, new_value)

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
        sem_data = self.data['S_tr']

        new_data = SemanticDegradation.kill_semantic_attributes(sem_data, rate, new_value)

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
        sem_data = self.data['S_tr']

        new_data = SemanticDegradation.kill_semantic_attributes(sem_data, rate, new_value)

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