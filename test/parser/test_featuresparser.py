'''
Tests for module featuresparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import unittest
import numpy as np

from src.parser.featuresparser import FeaturesParser


class FeaturesParserTests(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):
        '''
        Initializes model for all tests
        '''
        cls.parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'), console=True)
        
    def test_get_labels(self):
        '''
        Tests if array of labels is retrieved
        '''
        labels = self.parser.get_labels()
         
        self.assertTrue(isinstance(labels, np.ndarray))
        self.assertEqual((1000,), labels.shape)
        for label in labels:
            self.assertTrue(label <= 50 and label >= 1)
             
    def test_get_viual_features(self):
        '''
        Tests if array of features is retrieved
        '''
        features = self.parser.get_visual_features()
           
        self.assertEqual((1000, 2048), features.shape)
        self.assertTrue(sum(sum(features)) != 0)
        self.assertTrue(sum(sum(features)) != 1)

    def test_get_semantic_features(self):
        '''
        Tests if semantic features are correctly retrieved
        '''
        features = self.parser.get_semantic_features()
           
        self.assertEqual((1000, 85), features.shape)
        self.assertTrue(sum(sum(features)) > 0)
        
    def test_get_semantic_features_subset(self):
        '''
        Tests if semantic features are correctly retrieved
        '''
        features = self.parser.get_semantic_features(subset=True)
           
        self.assertEqual((1000, 24), features.shape)
        self.assertTrue(sum(sum(features)) > 0)
        
    def test_concatenate_features(self):
        '''
        Tests if features are correctly concatenated
        '''
        vis_fts = self.parser.get_visual_features()
        sem_fts = self.parser.get_semantic_features()
        features = self.parser.concatenate_features(vis_fts, sem_fts)
         
        self.assertEqual((1000, 2048 + 85), features.shape)
        self.assertTrue(sum(sum(features)) > 0)