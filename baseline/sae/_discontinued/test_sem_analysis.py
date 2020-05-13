import unittest
import numpy as np

from baseline.sae._discontinued.sem_analysis import SemanticDegradation


class SemanticAnalysisTests(unittest.TestCase):
    def test_kill_semantic_attributes_50(self):
        """
        Tests if semantic array is correctly destroyed
        """
        rate = 0.5
        new_value = -9999
        sd = SemanticDegradation('../../Datasets/SAE/cub_demo_data.mat', 10, new_value, 'cub')
        sem_data = sd.data.data['S_tr']

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
        Tests if semantic array is correctly destroyed
        """
        rate = 0.75
        new_value = -9999
        sd = SemanticDegradation('../../Datasets/SAE/cub_demo_data.mat', 10, new_value, 'cub')
        sem_data = sd.data.data['S_tr']

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
        Tests if semantic array is correctly destroyed
        """
        rate = 0.25
        new_value = -9999
        sd = SemanticDegradation('../../Datasets/SAE/cub_demo_data.mat', 10, new_value, 'cub')
        sem_data = sd.data.data['S_tr']

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
