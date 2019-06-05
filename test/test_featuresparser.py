'''
Tests for module featuresparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import unittest
import numpy as np

from core.featuresparser import FeaturesParser


class AnnotationsParserTests(unittest.TestCase):
        
    def test_get_labels(self):
        '''
        Tests if array of labels is retrieved
        '''
        parser = FeaturesParser('./_mockfiles/awa2/features/ResNet101')
        labels = parser.get_labels()
         
        self.assertTrue(isinstance(labels, np.ndarray))
        self.assertEqual((1000,), labels.shape)
        for label in labels:
            self.assertTrue(label <= 50 and label >= 1)
             
    def test_get_labels_file_not_found(self):
        '''
        Tests if None is returned when file is not found
        '''
        parser = FeaturesParser('./_mockfiles/awa2/features/dummy/')
        labels = parser.get_labels()
         
        self.assertEqual(None, labels)
         
    def test_get_viual_features(self):
        '''
        Tests if array of features is retrieved
        '''
        parser = FeaturesParser('./_mockfiles/awa2/features/ResNet101')
        features = parser.get_visual_features()
          
        self.assertEqual((1000, 2048), features.shape)
        self.assertTrue(sum(sum(features)) != 0)
        self.assertTrue(sum(sum(features)) != 1)
          
    def test_get_features_file_not_found(self):
        '''
        Tests if None is returned when file is not found
        '''
        parser = FeaturesParser('./_mockfiles/awa2/features/dummy/')
        features = parser.get_visual_features()
          
        self.assertEqual(None, features)
          
    def test_get_semantic_features(self):
        '''
        Tests if semantic features are correctly retrieved
        '''
        parser = FeaturesParser('./_mockfiles/awa2/features/ResNet101')
        features = parser.get_semantic_features('./_mockfiles/awa2/base/')
          
        self.assertEqual((1000, 85), features.shape)
        self.assertTrue(sum(sum(features)) > 0)
         
    def test_concatenate_features(self):
        '''
        Tests if features are correctly concatenated
        '''
        parser = FeaturesParser('./_mockfiles/awa2/features/ResNet101')
        vis_fts = parser.get_visual_features()
        sem_fts = parser.get_semantic_features('./_mockfiles/awa2/base/')
        features = parser.concatenate_features(vis_fts, sem_fts)
         
        self.assertEqual((1000, 2048 + 85), features.shape)
        self.assertTrue(sum(sum(features)) > 0)
        