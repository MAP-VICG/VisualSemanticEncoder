"""
Tests for module cub_demo

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 25, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
import numpy as np

from ..src.cub_demo import CUB200


class CUB200Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cls.cub = CUB200('../../Datasets/SAE/cub_demo_data.mat')

    def test_v2s_projection(self):
        """
        Tests if the returned accuracy is the expected one
        """
        self.assertEqual(0.61405, np.around(self.cub.v2s_projection(), decimals=5))

    def test_s2v_projection(self):
        """
        Tests if the returned accuracy is the expected one
        """
        self.assertEqual(0.60893, np.around(self.cub.s2v_projection(), decimals=5))
