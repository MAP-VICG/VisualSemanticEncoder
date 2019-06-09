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
from sklearn.model_selection import train_test_split

from core.vsencoder import SemanticEncoder
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
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        
        cls.nepochs = 5
        cls.encoder = SemanticEncoder(cls.nepochs, 128, x_train=x_train, 
                                      y_train=y_train, x_test=x_test, y_test=y_test)
        
    def test_run_encoder(self):
        '''
        Tests if history for all epochs is returned and if results are saved
        '''     
        files = ['ae_loss.png', 'ae_encoding.png', 'ae_components.png', 'ae_distribution.png']
         
        for f in files:
            file_name = os.path.join(self.encoder.plotter.results_path, f)
                   
            if os.path.isfile(file_name):
                os.remove(file_name)
     
        hist = self.encoder.run_encoder()
          
        self.assertEqual(self.nepochs, len(hist))
        for res in hist:
            self.assertIsNotNone(res.get('micro avg', None))
            self.assertIsNotNone(res.get('macro avg', None))
            self.assertIsNotNone(res.get('weighted avg', None))
        for f in files:
            self.assertTrue(os.path.isfile(file_name))
