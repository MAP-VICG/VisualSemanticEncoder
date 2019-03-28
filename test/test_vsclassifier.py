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
from keras.utils import normalize
from core.vsclassifier import SVMClassifier
from sklearn.model_selection import train_test_split
from core.featuresparser import FeaturesParser, PredicateType


class SVMClassifierTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        '''
        Initializes model for all tests
        '''
        fls_path = os.path.join(os.getcwd(), '_mockfiles/awa2')
        fts_path = os.path.join(fls_path, 'features/ResNet101')
        self.res_path = os.path.join(fls_path, 'results/')
        ann_path = os.path.join(fls_path, 'base')
        
        if not os.path.isdir(self.res_path):
            os.mkdir(self.res_path)
                    
        parser = FeaturesParser(fts_path)
        vis_fts = parser.get_visual_features()
        sem_fts = normalize(parser.get_semantic_features(ann_path, 
                                                         PredicateType.CONTINUOUS) + 1, 
                                                         order=1, axis=1)
        Y = parser.get_labels()
        X = parser.concatenate_features(vis_fts, sem_fts)
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, 
                                                                               stratify=Y, 
                                                                               test_size=0.2)
        self.svm = SVMClassifier()
        self.svm.run_classifier(self.x_train, self.y_train, 2, 2)
        
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
            for key in pred_dict[avg].keys():
                self.assertFalse(pred_dict[avg][key] == None)
        
    def test_save_results(self):
        '''
        Tests if prediction results can be saved
        '''
        result_file = os.path.join(self.res_path, 'svm_prediction.txt')
            
        if os.path.isfile(result_file):
            os.remove(result_file)
        
        _, prediction = self.svm.predict(self.x_test, self.y_test)
        self.svm.save_results(prediction, result_file)
            
        self.assertTrue(os.path.isfile(result_file))
