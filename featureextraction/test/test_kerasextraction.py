"""
Tests for module kerasextraction

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from os import path, sep, remove

from featureextraction.src.dataparsing import BirdsData
from featureextraction.src.kerasextraction import ResNet50FeatureExtractor


class ResNet50FeatureExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cls.base_path = path.join('mockfiles', 'birds')
        cls.data = BirdsData(cls.base_path)
        cls.images_list = cls.data.get_images_list(sep.join([cls.base_path, 'lists', 'train.txt']))
        cls.fts_file = 'birds_features.txt'
        cls.extractor = ResNet50FeatureExtractor(cls.images_list, path.join(cls.base_path, 'images'))

    def test_extract_image_features(self):
        """
        Tests if an array of 2048 features is built from the images' feature extraction and if most its values is not 0
        """
        fts = self.extractor.extract_image_features(sep.join([self.base_path, 'images', self.extractor.images_list[0]]))
        self.assertEqual((1, 2048), fts.shape)
        zeros = [1 for value in fts[0, :] if value == 0]
        self.assertTrue(sum(zeros) < 20*2048/100)

    def test_extract_images_list_features(self):
        """
        Tests if a numpy array is formed from the feature extraction based on a list. The expected array shape is
        (X, 2048) where X is the number of images listed
        """
        self.extractor.extract_images_list_features()
        self.assertEqual((10, 2048), self.extractor.features_set.shape)

    def test_save_features(self):
        """
        Tests if the features set formed from the feature extraction is correctly saved
        """
        self.extractor.save_features(self.fts_file)
        self.assertTrue(path.isfile(self.fts_file))

        with open(self.fts_file) as f:
            features = f.readlines()

        self.assertEqual(10, len(features))
        for line in features:
            self.assertEqual(2048, len(line.split()))

    @classmethod
    def tearDownClass(cls):
        """
        Deletes files that were written by the tests
        """
        if path.isfile(cls.fts_file):
            remove(cls.fts_file)
