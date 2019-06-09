'''
Model to encode visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 16, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import gc
import tensorflow as tf
from keras import backend as K

from core.vsclassifier import SVMClassifier
from core.vsautoencoder import VSAutoencoder
from utils.vsplotter import Plotter


class SemanticEncoder:
    def __init__(self, epochs, encoding_dim, **kwargs):
        '''
        Initializes common parameters
        
        @param kwargs: dictionary with training and testing data
        @param encoding_dim: autoencoder encoding size
        @param epochs: number of epochs
        '''
        self.epochs = epochs
        self.enc_dim = encoding_dim
        
        self.x_train = kwargs.get('x_train')
        self.y_train = kwargs.get('y_train')
        self.x_test = kwargs.get('x_test')
        self.y_test = kwargs.get('y_test')
        
        self.plotter = Plotter()
        self.svm = SVMClassifier()

    def clear_memmory(self):
        '''
        Resets Tensorflow graph, clear Keras session and calls garbage collector
        '''
        tf.reset_default_graph()
        K.clear_session()
        gc.collect()
    
    def run_encoder(self):
        '''
        Runs autoencoder and plots results. It automatically splits the data set into 
        training and test sets
        
        @return dictionary with svm results
        '''
        ae = VSAutoencoder(cv=5, njobs=-1, x_train=self.x_train, x_test=self.x_test, 
                                      y_train=self.y_train, y_test=self.y_test)
        
        history = ae.run_autoencoder(enc_dim=min(self.enc_dim, self.x_train.shape[1]), nepochs=self.epochs)
        
        encoded_fts = ae.encoder.predict(self.x_test)
        decoded_fts = ae.decoder.predict(encoded_fts)
        
        self.plotter.plot_loss(history.history)
        self.plotter.plot_encoding(self.x_test, encoded_fts, decoded_fts)
        self.plotter.plot_pca_vs_encoding(self.x_test, encoded_fts)
        self.plotter.plot_spatial_distribution(self.x_test, encoded_fts, decoded_fts, self.y_test)
        
        self.clear_memmory()
        return ae.svm_history
    
    def run_svm(self):
        '''
        Runs SVM and saves results
         
        @return dictionary with svm results
        '''
        self.svm.run_classifier(self.x_train, self.y_train, 5, -1)
         
        self.svm.model.best_estimator_.fit(self.x_train, self.y_train)
        pred_dict, prediction = self.svm.predict(self.x_test, self.y_test)
        self.svm.save_results(prediction, pred_dict)
         
        return pred_dict
 
    def __del__(self):
        '''
        Clears memory
        '''
        self.clear_memmory()
