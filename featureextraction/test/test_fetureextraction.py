"""
Tests for module featureextraction

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jan 5, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import unittest
from os import path, sep

from featureextraction.src.extractorparser import CUB200Data
from featureextraction.src.fetureextraction import ResNet50FeatureExtractor, InceptionV3FeatureExtractor
from featureextraction.src.fetureextraction import ExtractionType, ExtractorFactory


class ResNet50FeatureExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initializes variables to be used in the tests
        """
        cls.base_path = path.join('mockfiles', 'CUB200')
        cls.data = CUB200Data(cls.base_path, ExtractionType.RESNET)
        cls.images_list, _, _ = cls.data.get_images_data()
        cls.fts_file = 'birds_features.txt'

    def test_extract_image_features_resnet(self):
        """
        Tests if an array of 2048 features is built from the images' feature extraction and if most its values is not 0
        """
        extractor = ResNet50FeatureExtractor(path.join(self.base_path, 'images'))
        fts = extractor.extract_image_features(sep.join([self.base_path, 'images', self.images_list[0]]))
        self.assertEqual((1, 2048), fts.shape)
        zeros = [1 for value in fts[0, :] if value == 0]
        self.assertTrue(sum(zeros) < 0.2 * 2048)

    def test_extract_images_list_features_resnet(self):
        """
        Tests if a numpy array is formed from the feature extraction based on a list. The expected array shape is
        (X, 2048) where X is the number of images listed
        """
        extractor = ResNet50FeatureExtractor(path.join(self.base_path, 'images'))
        features_set = extractor.extract_images_list_features(self.images_list)
        self.assertEqual((30, 2048), features_set.shape)

    def test_extract_image_features_inception(self):
        """
        Tests if an array of 2048 features is built from the images' feature extraction and if most its values is not 0
        """
        extractor = InceptionV3FeatureExtractor(path.join(self.base_path, 'images'))
        fts = extractor.extract_image_features(sep.join([self.base_path, 'images', self.images_list[0]]))
        self.assertEqual((1, 2048), fts.shape)
        zeros = [1 for value in fts[0, :] if value == 0]
        self.assertTrue(sum(zeros) < 0.2 * 2048)

    def test_extract_images_list_features_inception(self):
        """
        Tests if a numpy array is formed from the feature extraction based on a list. The expected array shape is
        (X, 2048) where X is the number of images listed
        """
        extractor = InceptionV3FeatureExtractor(path.join(self.base_path, 'images'))
        features_set = extractor.extract_images_list_features(self.images_list)
        self.assertEqual((30, 2048), features_set.shape)

    def test_extractor_factory(self):
        """
        Tests if the factory works, meaning that based on a given type o extractor, the expected object is returned
        """
        self.assertTrue(isinstance(ExtractorFactory()(ExtractionType.RESNET), ResNet50FeatureExtractor))
        self.assertTrue(isinstance(ExtractorFactory()(ExtractionType.INCEPTION), InceptionV3FeatureExtractor))
