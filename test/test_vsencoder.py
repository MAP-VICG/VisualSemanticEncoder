''''
Tests for module vsencoder

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 9, 2019
@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import unittest
import numpy as np
from sklearn.model_selection import train_test_split

from core.vsencoder import SemanticEncoder
from core.vsclassifier import SVMClassifier
from core.featuresparser import FeaturesParser


class VSEncoderTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        '''
        Initializes model for all tests
        '''
        parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'), console=True)
        
        X = parser.concatenate_features(parser.get_visual_features(), parser.get_semantic_features())
        Y = parser.get_labels()
        
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        
        cls.nepochs = 5
        cls.encoder = SemanticEncoder(cls.nepochs, 128)
        
    def test_run_encoder(self):
        '''
        Tests if history for all epochs is returned and if results are saved
        '''     
        files = ['ae_loss.png', 'ae_encoding.png', 'ae_components.png', 'ae_distribution.png']
             
        for f in files:
            file_name = os.path.join(os.path.join(self.encoder.plotter.results_path, 'mock'), f)
                       
            if os.path.isfile(file_name):
                os.remove(file_name)
         
        hist = self.encoder.run_encoder(tag='mock', x_train=self.x_train, y_train=self.y_train, x_test=self.x_test, 
                                        y_test=self.y_test, simple=True, batch_norm=False)
              
        self.assertEqual(self.nepochs, len(hist))
        for res in hist:
            self.assertIsNotNone(res.get('accuracy', None))
            self.assertIsNotNone(res.get('macro avg', None))
            self.assertIsNotNone(res.get('weighted avg', None))
        for f in files:
            self.assertTrue(os.path.isfile(file_name))
                   
    def test_pick_semantic_features(self):
        '''
        Tests if features selection retrieves the correct features
        '''
        dataset = self.encoder.pick_semantic_features('SHAPE', self.x_test)
           
        self.assertEqual((200, 2053), dataset.shape)
        self.assertTrue((dataset[:, 2048] == self.x_test[:, 2048 + 17 - 1]).all())
        self.assertTrue((dataset[:, 2049] == self.x_test[:, 2048 + 45 - 1]).all())
        self.assertTrue((dataset[:, 2050] == self.x_test[:, 2048 + 46 - 1]).all())
        self.assertTrue((dataset[:, 2051] == self.x_test[:, 2048 + 24 - 1]).all())
        self.assertTrue((dataset[:, 2052] == self.x_test[:, 2048 + 25 - 1]).all())
          
    def test_pick_semantic_features_opposite(self):
        '''
        Tests if features selection retrieves the correct features with opposite option enabled
        '''
        dataset = self.encoder.pick_semantic_features('TEXTURE', self.x_test, opposite=True)
          
        self.assertEqual((200, 2066), dataset.shape)
        self.assertTrue((dataset[:, 2048] == self.x_test[:, 2048 + 1 - 1]).all())
        self.assertTrue((dataset[:, 2049] == self.x_test[:, 2048 + 2 - 1]).all())
        self.assertTrue((dataset[:, 2050] == self.x_test[:, 2048 + 3 - 1]).all())
        self.assertTrue((dataset[:, 2051] == self.x_test[:, 2048 + 4 - 1]).all())
        self.assertTrue((dataset[:, 2052] == self.x_test[:, 2048 + 5 - 1]).all())
        self.assertTrue((dataset[:, 2053] == self.x_test[:, 2048 + 6 - 1]).all())
        self.assertTrue((dataset[:, 2054] == self.x_test[:, 2048 + 7 - 1]).all())
        self.assertTrue((dataset[:, 2055] == self.x_test[:, 2048 + 8 - 1]).all())
        self.assertTrue((dataset[:, 2056] == self.x_test[:, 2048 + 17 - 1]).all())
        self.assertTrue((dataset[:, 2057] == self.x_test[:, 2048 + 45 - 1]).all())
        self.assertTrue((dataset[:, 2058] == self.x_test[:, 2048 + 46 - 1]).all())
        self.assertTrue((dataset[:, 2059] == self.x_test[:, 2048 + 24 - 1]).all())
        self.assertTrue((dataset[:, 2060] == self.x_test[:, 2048 + 25 - 1]).all())
        self.assertTrue((dataset[:, 2061] == self.x_test[:, 2048 + 19 - 1]).all())
        self.assertTrue((dataset[:, 2062] == self.x_test[:, 2048 + 20 - 1]).all())
        self.assertTrue((dataset[:, 2063] == self.x_test[:, 2048 + 23 - 1]).all())
        self.assertTrue((dataset[:, 2064] == self.x_test[:, 2048 + 26 - 1]).all())
        self.assertTrue((dataset[:, 2065] == self.x_test[:, 2048 + 31 - 1]).all())
 
    def test_save_results(self):
        '''
        Tests if results are correctly saved to XML
        '''
        results = dict()
        svm = SVMClassifier()
        results['REF'] = svm.run_svm(x_train=self.x_train[:,:2048], x_test=self.x_test[:,:2048], 
                             y_train=self.y_train, y_test=self.y_test)
         
        enc = SemanticEncoder(5, 32)
        results['ALL'] = enc.run_encoder('ALL', x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, 
                                         y_test=self.y_test, simple=True, batch_norm=False)
     
        file_name = os.path.join(svm.results_path, 'ae_results.xml')
               
        if os.path.isfile(file_name):
            os.remove(file_name)
             
        self.encoder.save_results(results)
        self.assertTrue(os.path.isfile(file_name))
        
    def test_remove_random_30(self):
        '''
        Tests if values are removed from the array based on the noise level provided
        '''
        features = np.array([[9, 2, 10, 9, 6, 2, 5, 8, 1, 2, 5, 3, 0, 5, 4, 4, 8, 3, 1, 10, 1, 7],
                             [10, 6, 9, 9, 7, 10, 0, 3, 7, 9, 2, 10, 3, 3, 6, 1, 9, 4, 9, 4, 1, 5],
                             [2, 1, 10, 7, 3, 8, 1, 1, 6, 0, 10, 7, 1, 4, 4, 9, 0, 5, 7, 10, 10, 7]])
        noisy_fts = np.array([[0, 2, 10, 0, 0, 2, 5, 0, 0, 2, 5, 3, 0, 5, 4, 4, 8, 0, 1, 10, 0, 7],
                             [0, 0, 0, 0, 7, 10, 0, 3, 7, 9, 2, 10, 3, 0, 6, 1, 9, 0, 0, 4, 1, 5],
                             [0, 1, 0, 7, 3, 8, 0, 0, 6, 0, 10, 7, 1, 0, 4, 9, 0, 5, 7, 10, 0, 7]])
        
        self.encoder.remove_random(features, 0, 21, 0.3, 42)
        self.assertTrue((features == noisy_fts).all())
        
    def test_remove_random_50(self):
        '''
        Tests if values are removed from the array based on the noise level provided
        '''
        features = np.array([[9, 2, 10, 9, 6, 2, 5, 8, 1, 2, 5, 3, 0, 5, 4, 4, 8, 3, 1, 10, 1, 7],
                             [10, 6, 9, 9, 7, 10, 0, 3, 7, 9, 2, 10, 3, 3, 6, 1, 9, 4, 9, 4, 1, 5],
                             [2, 1, 10, 7, 3, 8, 1, 1, 6, 0, 10, 7, 1, 4, 4, 9, 0, 5, 7, 10, 10, 7]])
        noisy_fts = np.array([[0, 0, 10, 0, 0, 2, 5, 0, 0, 2, 0, 0, 0, 5, 0, 4, 8, 0, 1, 10, 0, 7],
                             [0, 0, 0, 0, 7, 10, 0, 3, 0, 0, 2, 10, 3, 0, 0, 1, 9, 0, 0, 4, 0, 5],
                             [0, 1, 10, 7, 0, 8, 0, 0, 6, 0, 10, 7, 0, 0, 0, 9, 0, 0, 7, 10, 0, 0]])
        
        self.encoder.remove_random(features, 0, 21, 0.5, 42)
        self.assertTrue((features == noisy_fts).all())
        
    def test_remove_random_80(self):
        '''
        Tests if values are removed from the array based on the noise level provided
        '''
        features = np.array([[9, 2, 10, 9, 6, 2, 5, 8, 1, 2, 5, 3, 0, 5, 4, 4, 8, 3, 1, 10, 1, 7],
                             [10, 6, 9, 9, 7, 10, 0, 3, 7, 9, 2, 10, 3, 3, 6, 1, 9, 4, 9, 4, 1, 5],
                             [2, 1, 10, 7, 3, 8, 1, 1, 6, 0, 10, 7, 1, 4, 4, 9, 0, 5, 7, 10, 10, 7]])
        noisy_fts = np.array([[0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 1, 0],
                             [0, 0, 9, 0, 0, 0, 0, 3, 0, 9, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 10, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]])
        
        self.encoder.remove_random(features, 0, 21, 0.8, 17)
        self.assertTrue((features == noisy_fts).all())