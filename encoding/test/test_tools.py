import unittest
import numpy as np
from os import path

from featureextraction.src.dataparsing import DataIO
from encoding.src.tools import kill_semantic_attributes


class ModelFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Defines testing data
        """
        cls.data = DataIO.get_features(path.join('mockfiles', 'CUB200_x_train.txt'))

    def test_kill_semantic_attributes(self):
        """
        Tests if semantic array is correctly destroyed
        """
        count = np.zeros(self.data.shape[0])
        for i, example in enumerate(self.data):
            for value in example:
                if value == 0.1:
                    count[i] += 1

        new_data = kill_semantic_attributes(self.data, 0.5)

        new_data_count = np.zeros(self.data.shape[0])
        for i, example in enumerate(new_data):
            for value in example:
                if value == 0.1:
                    new_data_count[i] += 1

        x_test_count = np.zeros(self.data.shape[0])
        for i, example in enumerate(self.data):
            for value in example:
                if value == 0.1:
                    x_test_count[i] += 1

        self.assertTrue((x_test_count == count).all())
        self.assertTrue((new_data_count == count + 156).all())
