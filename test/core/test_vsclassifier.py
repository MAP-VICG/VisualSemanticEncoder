'''
Tests for module vsautoencoder

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 27, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import unittest

from sklearn.model_selection import train_test_split

from src.core.vsclassifier import SVMClassifier
from src.core.featuresparser import FeaturesParser


class SVMClassifierTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        '''
        Initializes model for all tests
        '''
        parser = FeaturesParser(fts_dir=os.path.join('features', 'mock'), console=True)
        
        Y = parser.get_labels()
        X = parser.concatenate_features(parser.get_visual_features(), parser.get_semantic_features())
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)
        
        cls.svm = SVMClassifier()
        cls.svm.run_classifier(cls.x_train, cls.y_train, 2, 2)
        
    def test_grid_search(self):
        '''
        Tests if model was built and best parameters were found
        '''        
        self.assertFalse(self.svm.model == None)
        self.assertTrue(hasattr(self.svm.model,'best_params_'))
        self.assertTrue(hasattr(self.svm.model,'best_score_'))
        self.assertTrue(hasattr(self.svm.model,'best_index_'))
        self.assertTrue(hasattr(self.svm.model,'best_estimator_'))
    
    def test_grid_search_fail(self):
        '''
        Tests grid search method raises exception for invalid number of folds
        '''
        svm = SVMClassifier()
        self.assertRaises(ValueError, svm.run_classifier, self.x_train, self.y_train, 1)
        
        
    def test_prediction(self):
        '''
        Tests if SVM prediction returns the result in the correct format
        '''
        pred_dict, _ = self.svm.predict(self.x_test, self.y_test)
        
        for avg in pred_dict.keys():
            if avg == 'accuracy':
                self.assertIsNone(pred_dict[avg]['precision'])
                self.assertIsNone(pred_dict[avg]['recall'])
                self.assertIsNotNone(pred_dict[avg]['f1-score'])
            else:
                for key in pred_dict[avg].keys():
                    self.assertIsNotNone(pred_dict[avg][key])
        
    def test_save_results(self):
        '''
        Tests if prediction results can be saved
        '''
        result_file = os.path.join(self.svm.results_path, 'svm_prediction.txt')
            
        if os.path.isfile(result_file):
            os.remove(result_file)
        
        _, prediction = self.svm.predict(self.x_test, self.y_test)
        self.svm.save_results(prediction)
            
        self.assertTrue(os.path.isfile(result_file))
        
    def test_run_svm(self):
        '''
        Tests if SVM runs and if results are saved
        '''
        file_name = os.path.join(self.svm.results_path, 'svm_prediction.txt')
                  
        if os.path.isfile(file_name):
            os.remove(file_name)
                
        pred_dict = self.svm.run_svm(self.x_train, self.y_train, self.x_test, self.y_test)
        self.assertIsNotNone(pred_dict.get('accuracy', None))
        self.assertIsNotNone(pred_dict.get('macro avg', None))
        self.assertIsNotNone(pred_dict.get('weighted avg', None))
        
        self.assertTrue(os.path.isfile(file_name))
    
