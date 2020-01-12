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
import numpy as np
from os import path, sep, remove
from src.featureextraction.extractor import BirdsData


class BirdsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cls.class_file = 'birds_test_classes.txt'
        cls.base_path = cls.base_path = sep.join(['..', '..', '_files', 'mockfiles', 'birds'])
        cls.data = BirdsData(cls.base_path)
        cls.train_list = cls.data.get_images_list(sep.join([cls.base_path, 'lists', 'train.txt']))
        cls.test_list = cls.data.get_images_list(sep.join([cls.base_path, 'lists', 'test.txt']))

    def test_get_images_list(self):
        """
        Tests if images listed in the files were read in the correct format and order
        """
        self.assertEqual(10, len(self.test_list))
        self.assertEqual('033.Yellow_billed_Cuckoo/Yellow_billed_Cuckoo_0013_1244195077.jpg', self.test_list[0])
        self.assertEqual('121.Grasshopper_Sparrow/Grasshopper_Sparrow_0018_2691901557.jpg', self.test_list[4])
        self.assertEqual('200.Common_Yellowthroat/Common_Yellowthroat_0002_2679007659.jpg', self.test_list[9])

    def test_save_class_file(self):
        """
        Tests if file with list of classes is saved and if it contains the expected values in the expected order
        """
        self.data.save_class_file(self.class_file, self.test_list)
        self.assertTrue(path.isfile(self.class_file))

        with open(self.class_file) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    self.assertEqual('033', line.strip())
                elif i == 4:
                    self.assertEqual('121', line.strip())
                elif i == 9:
                    self.assertEqual('200', line.strip())

    def test_map_evaluation_certainty(self):
        """
        Tests if values calculated considering the certainty factor are the expected ones
        """
        self.assertEqual(0.75, self.data.map_evaluation_certainty(1, 0))
        self.assertEqual(0.25, self.data.map_evaluation_certainty(0, 0))
        self.assertEqual(1, self.data.map_evaluation_certainty(1, 1))
        self.assertEqual(0, self.data.map_evaluation_certainty(0, 1))
        self.assertEqual(0.5, self.data.map_evaluation_certainty(1, 2))
        self.assertEqual(0.5, self.data.map_evaluation_certainty(0, 2))

    def test_get_semantic_attributes(self):
        """
        Tests if shape of returned array is correct and if all values lie in between 0 and 1
        """
        features = self.data.get_semantic_attributes(20)
        self.assertEqual((20, 288), features.shape)

        for i in range(20):
            for j in range(288):
                self.assertTrue(0 <= features[i, j] <= 1)

    def test_get_visual_attributes(self):
        """
        Tests if the shape of the numpy array returned is the expected one
        """
        self.assertEqual((10, 2048), self.data.get_visual_attributes(self.train_list).shape)

    def test_get_train_test_masks(self):
        """
        Tests if the shape of the returned masks are the expected ones and if the number of images in
        each mask is correct
        """
        train_mask, test_mask = self.data.get_train_test_masks(self.train_list)
        self.assertEqual(20, len(train_mask))
        self.assertEqual(20, len(test_mask))
        self.assertEqual(10, sum(np.ones(20)[train_mask]))
        self.assertEqual(10, sum(np.ones(20)[test_mask]))

    def test_get_images_class(self):
        """
        Tests if the id of the class returned is correct
        """
        for i, klass in enumerate(self.data.get_images_class(self.test_list)):
            if i == 0:
                self.assertEqual(33, klass)
            elif i == 4:
                self.assertEqual(121, klass)
            elif i == 9:
                self.assertEqual(200, klass)

    def test_build_birds_data(self):
        """
        Tests if shape of returned arrays and number of images per set is correct
        """
        train_fts, train_class, test_fts, test_class = self.data.build_birds_data(20)
        self.assertEqual((10, 2336), train_fts.shape)
        self.assertEqual((10, 2336), test_fts.shape)
        self.assertEqual(10, len(train_class))
        self.assertEqual(10, len(test_fts))

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files that were written by the tests
        """
        if path.isfile(cls.class_file):
            remove(cls.class_file)
