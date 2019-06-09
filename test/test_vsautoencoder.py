'''
Tests for module vsautoencoder

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import unittest
from math import floor
from sklearn.model_selection import train_test_split

from core.vsautoencoder import VSAutoencoder
from core.featuresparser import FeaturesParser


class VSAutoencoderTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        '''
        Initializes model for all tests
        '''
        parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'), console=True)
        
        X = parser.concatenate_features(parser.get_visual_features(), parser.get_semantic_features())
        Y = parser.get_labels()
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        
        cls.enc_dim = 32
        cls.nexamples = x_train.shape[0]
        
        cls.ae = VSAutoencoder(cv=2, njobs=2, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        cls.svm_file = os.path.join(cls.ae.svm.results_path, 'ae_svm_prediction.txt')
        if os.path.isfile(cls.svm_file):
            os.remove(cls.svm_file)
            
        cls.history = cls.ae.run_autoencoder(cls.enc_dim, 5)
         
    def test_build_autoencoder(self):
        '''
        Tests if autoencoder is built correctly
        '''
        middle = floor(len(self.ae.autoencoder.layers) / 2)
               
        # input
        self.assertEqual((None, 2133), self.ae.autoencoder.layers[0].input_shape)
               
        # encoding
        self.assertEqual((None, self.enc_dim), self.ae.autoencoder.layers[middle].output_shape)
               
        # output
        self.assertEqual((None, 2133), self.ae.autoencoder.layers[-1].output_shape)
             
    def test_train_autoencoder(self):
        '''
        Tests if autoencoder can be trained
        '''
        self.assertEqual(self.history.params['epochs'], len(self.history.epoch))
        self.assertEqual(self.history.params['epochs'], len(self.history.history['val_loss']))
        self.assertEqual(self.history.params['epochs'], len(self.history.history['loss']))
        self.assertEqual(floor(self.nexamples * 0.8), self.history.params['samples'])
        self.assertTrue(self.history.params['do_validation'])
        
    def test_svm_results_saved(self):
        '''
        Tests if SVM results for the last epoch are saved
        '''
        self.assertTrue(os.path.isfile(self.svm_file))