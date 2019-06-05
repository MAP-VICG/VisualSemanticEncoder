'''
Autoencoder for visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 27, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from utils.logwriter import Logger, MessageType


class SVMClassifier:
    
    def __init__(self):
        '''
        Setting tuning parameters
        '''
        self.model = None
        self.tuning_params = {'C': [1, 10, 100, 1000]}
        Logger().write_message('SVM tuning parameters are %s.' % str(self.tuning_params), MessageType.INF)
        
    def run_classifier(self, x_train, y_train, nfolds=5, njobs=None):
        '''
        Builds and trains classification model based on grid search
        
        @param x_train: 2D numpy array training set
        @param y_train: 1D numpy array test labels
        @param nfolds: number of folds in cross validation
        @param njobs: number of jobs to run in parallel on Grid Search
        '''
        if nfolds < 2:
            raise ValueError('Number of folds cannot be less than 2')
        
#         self.model = GridSearchCV(LinearSVC(verbose=0, max_iter=1200), 
#                                   self.tuning_params, cv=nfolds, 
#                                   iid=False, scoring='recall_macro', n_jobs=njobs)
        from sklearn.svm import SVC
        self.model = SVC(gamma='auto')
#         self.model = LinearSVC(random_state=0, tol=1e-5)
        
        self.model.fit(x_train, y_train)
    
    def predict(self, x_test, y_test):
        '''
        Tests model performance on test set and saves classification results
        
        @param x_test: 2D numpy array with test set
        @param y_test: 1D numpy array test labels
        @return tuple with dictionary with prediction results and string with 
        full prediction table
        '''
        y_true, y_pred = y_test, self.model.predict(x_test)
#         y_true, y_pred = y_test, self.model.best_estimator_.predict(x_test)
        prediction = classification_report(y_true, y_pred)

        keys = ['precision', 'recall', 'f1-score', 'support']
        pred_dict = {'micro avg': {key: None for key in keys},
                     'macro avg': {key: None for key in keys},
                     'weighted avg': {key: None for key in keys}}
        
        for row in prediction.split('\n')[-4:-1]:
            values = row.split()
            for idx, key in enumerate(keys):
                pred_dict[values[0] + ' ' + values[1]][key] = float(values[idx + 2])
               
        Logger().write_message('SVM Prediction result is %s' % str(pred_dict), MessageType.INF) 
        return pred_dict, prediction
        
    def save_results(self, prediction, results_path, appendix=None):
        '''
        Saves classification results
        
        @param prediction: string with full prediction table
        @param results_path: string with full path to save results under
        @param appendix: dictionary with extra data to save to results file
        '''
        root_path = os.sep.join(results_path.split(os.sep)[:-1])
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
        
        try:
            with open(results_path, 'w+') as f:
                f.write(prediction)
#                 f.write('\nbest parameters: %s' % str(self.model.best_params_))
                
                if appendix:
                    for key in appendix.keys():
                        f.write('\n%s: %s' % (key, str(appendix[key])))
                    
        except (IsADirectoryError, OSError):
            Logger().write_message('Could not save prediction results under %s.' % results_path, MessageType.ERR)
