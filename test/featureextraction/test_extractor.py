"""
Tests for module extractor

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from pathlib import Path
from os import path, sep, remove
from src.featureextraction.extractor import BirdsData


class BirdsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cls.class_file = 'birds_test_classes.txt'
        cls.data = BirdsData(path.join(str(Path.home()), sep.join(['Projects', 'Datasets', 'Birds'])))

    def test_get_images_list(self):
        """
        Tests if images listed in the files were read in the correct format and order
        """
        self.data.get_images_list('test.txt')
        self.assertEqual(3033, len(self.data.images_list))
        self.assertEqual('001.Black_footed_Albatross/Black_footed_Albatross_0009_2408326989.jpg', self.data.images_list[0])
        self.assertEqual('101.White_Pelican/White_Pelican_0011_2973346636.jpg', self.data.images_list[1499])
        self.assertEqual('200.Common_Yellowthroat/Common_Yellowthroat_0007_2935302696.jpg', self.data.images_list[3032])

    def test_save_class_file(self):
        """
        Tests if file with list of classes is saved and if it contains the expected values in the expected order
        """
        self.data.save_class_file(self.class_file)
        self.assertTrue(path.isfile(self.class_file))

        with open(self.class_file) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    self.assertEqual('001', line.strip())
                elif i == 1499:
                    self.assertEqual('101', line.strip())
                elif i == 3032:
                    self.assertEqual('200', line.strip())

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files that were written by the tests
        """
        if path.isfile(cls.class_file):
            remove(cls.class_file)
