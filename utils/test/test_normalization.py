"""
Unit tests for normalization module

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Nov 13, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from os import sep
import numpy as np

from utils.src.normalization import Normalization


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes model for all tests
        """
        cls.data = []
        with open(sep.join(['mockfiles', 'normdata', 'dataset.txt'])) as f:
            for line in f.readlines():
                cls.data.append(list(map(float, line.split())))
        cls.data = np.array(cls.data)
        
    def test_normalize_zero_one_global(self):
        """
        Tests if data set is normalized to values between zero and one
        """
        data = self.data.copy()
        self.assertEqual(14, round(data.max()))
        self.assertEqual(0, data.min())

        Normalization.normalize_zero_one_global(data)
        self.assertEqual(1, round(data.max()))
        self.assertEqual(0, round(data.min()))
        
    def test_normalize_zero_one_by_column(self):
        """
        Tests data set is normalized between zero and one by column
        """
        data = self.data.copy()
        Normalization.normalize_zero_one_by_column(data)

        for col in range(data.shape[1]):
            if sum(data[:, col] > 0):
                self.assertEqual(1, round(data[:, col].max()))
            self.assertEqual(0, round(data[:, col].min()))
